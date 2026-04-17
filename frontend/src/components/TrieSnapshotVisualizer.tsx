import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
	ROOT_SYMBOL,
	SCROLL_TARGET_X_PX,
	SPACE_SYMBOL,
	STOP_SYMBOL,
	buildVisibleTree,
	computeLaidOutNodes,
	deepestVisibleNode,
	findTutorTargetKey,
	firstForkNode,
	relativeDepth,
	type ExpandedSnapshot,
	type VisualNode,
	type VisibleNodeTimerMap,
} from '../domain/trieLayout';
import { colorFromLetter } from '../utils/colors';

// Layout transitions over 300ms.
const TWEEN_DURATION_MS = 300;

function interpolate(start: number, end: number, progress: number): number {
	return start + (end - start) * progress;
}

type TweenKey = string;
interface TweenEntry {
	start: number;
	from: number;
	to: number;
}

function tweenKey(fullString: string, property: string): TweenKey {
	return `${property}\0${fullString}`;
}

interface TrieSnapshotVisualizerProps {
	snapshot: ExpandedSnapshot;
	timers: VisibleNodeTimerMap;
	period: number;
	expansionThreshold: number;
	scrollOffset: number;
	scrollRoot: string;
	firstForkDepth: number | null;
	showAll?: boolean;
	useVisualTutor?: boolean;
	targetPhrase?: string;
	/** When true, bright timer strokes (space, root, etc.) are shifted darker for a light canvas. */
	lightBackground?: boolean;
	/** Semi-transparent node background rectangles (default on). */
	showBoxes?: boolean;
	showDebugStats?: boolean;
}

const BOX_WIDTH = 37;
const TIMER_RADIUS = 15;
const TIMER_STROKE_WIDTH = 2;
const TIMER_FONT_SIZE = 37;
/** Pad-mode `N` / `Q` / `U` use emoji or wide glyphs; draw smaller than letter timers. */
const PAD_MODE_GLYPH_FONT_SCALE = 0.52;
const DEBUG_LEVEL_GUTTER = 56;

function timerFraction(time: number, phase: number, period: number): number {
	return ((time - phase + period) % period) / period;
}

function finalSymbol(fullString: string): string {
	if (fullString === ROOT_SYMBOL) {
		return ROOT_SYMBOL;
	}
	return fullString.at(-1) ?? ROOT_SYMBOL;
}

function displaySymbol(symbol: string): string {
	if (symbol === SPACE_SYMBOL) {
		return ' ';
	}
	// Trie pad-mode sentinels (match Rust `symbol.rs`): show pictographs in the canvas timer.
	if (symbol === 'N') {
		return '#️⃣';
	}
	if (symbol === 'Q') {
		return '🔣';
	}
	if (symbol === 'U') {
		return '⇧';
	}
	return symbol;
}

/** Trie wire sentinels — same letters as `backend/server/token_mapping.py` / Rust `symbol.rs`. */
const TRIE_NUMPAD = 'N';
const TRIE_SPECIAL_SHIFT = 'Q';
const TRIE_SHIFT = 'U';

function isAsciiDigit(c: string): boolean {
	return c >= '0' && c <= '9';
}

function isLowercaseLetter(c: string): boolean {
	return c >= 'a' && c <= 'z';
}

/**
 * Human-readable text for a trie path after the root `A` (inverse of HF → trie encoding).
 * - `S` → space
 * - `N` before a digit → digit only (omit numpad marker)
 * - `Q` before the next char → that char only (omit special-shift marker)
 * - `U` before a lowercase letter → uppercase letter (omit shift marker)
 * - `Z` (stop) omitted from the caption
 */
function trieWireToDisplayPrefix(s: string): string {
	let i = 0;
	let out = '';
	while (i < s.length) {
		const c = s[i];
		if (c === STOP_SYMBOL) {
			i += 1;
			continue;
		}
		if (c === SPACE_SYMBOL) {
			out += ' ';
			i += 1;
			continue;
		}
		if (c === TRIE_NUMPAD) {
			if (i + 1 < s.length && isAsciiDigit(s[i + 1])) {
				out += s[i + 1];
				i += 2;
			} else {
				i += 1;
			}
			continue;
		}
		if (c === TRIE_SPECIAL_SHIFT) {
			if (i + 1 < s.length) {
				out += s[i + 1];
				i += 2;
			} else {
				i += 1;
			}
			continue;
		}
		if (c === TRIE_SHIFT) {
			if (i + 1 < s.length && isLowercaseLetter(s[i + 1])) {
				out += s[i + 1].toUpperCase();
				i += 2;
			} else {
				i += 1;
			}
			continue;
		}
		out += c;
		i += 1;
	}
	return out;
}

function offscreenPrefixText(scrollRoot: string): string {
	if (scrollRoot === ROOT_SYMBOL) {
		return '';
	}
	return trieWireToDisplayPrefix(scrollRoot.slice(1));
}

/** Timer circle center and radius — fixed CSS px, not scaled with fit-to-width. */
function timerCircleGeometry(
	node: VisualNode,
	currentTime: number,
	getTweenedValue: (fullString: string, property: string, t: number) => number,
	scaleX: (layoutX: number, fullString: string) => number,
): { cx: number; cy: number; r: number } {
	const displayText = displaySymbol(node.symbol);
	let timerRadius = TIMER_RADIUS;
	if (displayText === 'm' || displayText === 'w') {
		timerRadius *= 1.15;
	}
	const lx = getTweenedValue(node.fullString, 'x', currentTime);
	const ly = getTweenedValue(node.fullString, 'y', currentTime);
	const lw = getTweenedValue(node.fullString, 'width', currentTime);
	const lh = getTweenedValue(node.fullString, 'height', currentTime);
	const cx = scaleX(lx + lw, node.fullString) - TIMER_RADIUS;
	const cy = ly + lh / 2;
	return { cx, cy, r: timerRadius };
}

function timerColor(symbol: string): [number, number, number] {
	if (symbol === SPACE_SYMBOL) {
		return colorFromLetter(' ');
	}
	if (symbol === ROOT_SYMBOL) {
		return [255, 255, 255];
	}
	return colorFromLetter(symbol);
}

/** Bézier edges from child to parent: dark mode stays saturated; light mode is softened. */
function connectorStrokeStyle(
	r: number,
	g: number,
	b: number,
	lightBackground: boolean,
): string {
	const dr = Math.floor(r / 1.8);
	const dg = Math.floor(g / 1.8);
	const db = Math.floor(b / 1.8);
	if (!lightBackground) {
		return `rgba(${dr}, ${dg}, ${db}, 1)`;
	}
	const lift = 88;
	const lr = Math.min(255, dr + lift);
	const lg = Math.min(255, dg + lift);
	const lb = Math.min(255, db + lift);
	return `rgba(${lr}, ${lg}, ${lb}, 0.4)`;
}

/** Bright colors (white space, root, pale yellow, etc.) need darkening on a light canvas. */
function timerRgbOnSurface(
	symbol: string,
	lightBackground: boolean,
): [number, number, number] {
	const [r, g, b] = timerColor(symbol);
	if (!lightBackground) {
		return [r, g, b];
	}
	const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
	if (lum < 168) {
		return [r, g, b];
	}
	const tr = 30;
	const tg = 41;
	const tb = 59;
	const t = Math.min(1, (lum - 168) / 88);
	return [
		Math.round(r + (tr - r) * t),
		Math.round(g + (tg - g) * t),
		Math.round(b + (tb - b) * t),
	];
}

function readLayoutProp(node: VisualNode | undefined, property: string): number {
	if (!node) {
		return 0;
	}
	switch (property) {
		case 'x':
			return node.x;
		case 'y':
			return node.y;
		case 'width':
			return node.width;
		case 'height':
			return node.height;
		default:
			return 0;
	}
}

function TrieSnapshotVisualizer({
	snapshot,
	timers,
	period,
	expansionThreshold,
	scrollOffset,
	scrollRoot,
	firstForkDepth,
	showAll = false,
	useVisualTutor = false,
	targetPhrase = '',
	lightBackground = false,
	showBoxes = true,
	showDebugStats = false,
}: TrieSnapshotVisualizerProps) {
	const [time, setTime] = useState(() => performance.now() / 1000);
	const containerRef = useRef<HTMLDivElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [devicePixelRatio, setDevicePixelRatio] = useState(1);
	const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });
	const tweenStartTimesRef = useRef<Record<TweenKey, TweenEntry>>({});
	const laidOutNodesRef = useRef<Record<string, VisualNode> | null>(null);

	const getTweenedValue = useCallback((fullString: string, property: string, currentTime: number): number => {
		const key = tweenKey(fullString, property);
		const laidOut = laidOutNodesRef.current?.[fullString];
		const actualRaw = readLayoutProp(laidOut, property);
		if (!tweenStartTimesRef.current[key]) {
			tweenStartTimesRef.current[key] = {
				start: currentTime,
				from: actualRaw,
				to: actualRaw,
			};
		}
		const tween = tweenStartTimesRef.current[key];
		const progress = Math.min((currentTime - tween.start) / (TWEEN_DURATION_MS / 1000), 1);
		return interpolate(tween.from, tween.to, progress);
	}, []);

	const updateTween = useCallback(
		(fullString: string, property: string, targetValue: number, currentTime: number): void => {
			const key = tweenKey(fullString, property);
			if (!tweenStartTimesRef.current[key]) {
				tweenStartTimesRef.current[key] = {
					start: currentTime,
					from: targetValue,
					to: targetValue,
				};
				return;
			}
			const current = getTweenedValue(fullString, property, currentTime);
			tweenStartTimesRef.current[key] = { start: currentTime, from: current, to: targetValue };
		},
		[getTweenedValue],
	);

	useEffect(() => {
		let frame = 0;
		const tick = () => {
			setTime(performance.now() / 1000);
			frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
		return () => cancelAnimationFrame(frame);
	}, []);

	useEffect(() => {
		const container = containerRef.current;
		if (!container) {
			return;
		}
		const updateViewportSize = () => {
			const rect = container.getBoundingClientRect();
			setDevicePixelRatio(window.devicePixelRatio || 1);
			setViewportSize({
				width: Math.round(rect.width),
				height: Math.round(rect.height),
			});
		};
		updateViewportSize();
		const observer = new ResizeObserver(updateViewportSize);
		observer.observe(container);
		window.addEventListener('resize', updateViewportSize);
		return () => {
			observer.disconnect();
			window.removeEventListener('resize', updateViewportSize);
		};
	}, []);

	const visibleTree = useMemo(
		() => buildVisibleTree(snapshot, expansionThreshold),
		[snapshot, expansionThreshold],
	);

	/** Path key for the off-screen caption: through first fork, or full deepest path if no fork. Hidden until scroll leaves root. */
	const offscreenPrefixDisplay = useMemo(() => {
		if (scrollRoot === ROOT_SYMBOL) {
			return '';
		}
		const fork = firstForkNode(visibleTree);
		const pathKey = fork
			? fork.fullString
			: (deepestVisibleNode(visibleTree)?.fullString ?? ROOT_SYMBOL);
		if (pathKey === ROOT_SYMBOL) {
			return '';
		}
		return offscreenPrefixText(pathKey);
	}, [scrollRoot, visibleTree]);

	const laidOutNodes = useMemo(() => {
		return computeLaidOutNodes(
			snapshot,
			expansionThreshold,
			scrollRoot,
			showAll,
			viewportSize.height,
		);
	}, [expansionThreshold, scrollRoot, showAll, snapshot, viewportSize.height, visibleTree]);

	const renderMetrics = useMemo(() => {
		if (!laidOutNodes) {
			return {
				scale: 1,
				offsetX: 0,
				offsetY: 0,
				contentWidth: viewportSize.width,
				levelGutter: 0,
			};
		}
		let contentWidth = 0;
		for (const node of Object.values(laidOutNodes)) {
			contentWidth = Math.max(contentWidth, node.x + node.width + TIMER_RADIUS * 6);
		}
		const levelGutter = showDebugStats ? DEBUG_LEVEL_GUTTER : 0;
		return {
			scale: 1,
			offsetX: 0,
			offsetY: 0,
			contentWidth,
			levelGutter,
		};
	}, [laidOutNodes, showDebugStats, viewportSize]);

	const tutorTargetKey = useMemo(() => {
		if (!useVisualTutor || !laidOutNodes) {
			return null;
		}
		return findTutorTargetKey(snapshot, expansionThreshold, scrollRoot, showAll, targetPhrase);
	}, [expansionThreshold, laidOutNodes, scrollRoot, showAll, snapshot, targetPhrase, useVisualTutor]);

	useLayoutEffect(() => {
		laidOutNodesRef.current = laidOutNodes;
		if (!laidOutNodes) {
			return;
		}
		const t = performance.now() / 1000;
		const active = new Set<string>();
		for (const node of Object.values(laidOutNodes)) {
			active.add(node.fullString);
			updateTween(node.fullString, 'x', node.x, t);
			updateTween(node.fullString, 'y', node.y, t);
			updateTween(node.fullString, 'width', node.width, t);
			updateTween(node.fullString, 'height', node.height, t);
		}
		for (const key of Object.keys(tweenStartTimesRef.current)) {
			const sep = key.indexOf('\0');
			if (sep === -1) {
				continue;
			}
			const fullString = key.slice(sep + 1);
			if (!active.has(fullString)) {
				delete tweenStartTimesRef.current[key];
			}
		}
	}, [laidOutNodes, updateTween]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas || !laidOutNodes) {
			return;
		}
		canvas.width = Math.max(1, Math.round(viewportSize.width * devicePixelRatio));
		canvas.height = Math.max(1, Math.round(viewportSize.height * devicePixelRatio));
		canvas.style.width = `${viewportSize.width}px`;
		canvas.style.height = `${viewportSize.height}px`;
		const ctx = canvas.getContext('2d');
		if (!ctx) {
			return;
		}

		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

		const visibleNodes = Object.values(laidOutNodes);
		const scaleX = (value: number, fullString: string) =>
			renderMetrics.offsetX +
			value * renderMetrics.scale +
			relativeDepth(fullString, scrollRoot) * renderMetrics.levelGutter;
		const scaleSize = (value: number) => value * renderMetrics.scale;
		const currentTime = performance.now() / 1000;

		ctx.beginPath();
		ctx.moveTo(SCROLL_TARGET_X_PX, 0);
		ctx.lineTo(SCROLL_TARGET_X_PX, viewportSize.height);
		ctx.strokeStyle = lightBackground ? 'rgba(15, 23, 42, 0.8)' : 'rgba(255, 255, 255, 0.75)';
		ctx.lineWidth = 4;
		ctx.stroke();
		ctx.closePath();

		if (showBoxes) {
			for (const node of visibleNodes) {
				const [r, g, b] = timerRgbOnSurface(node.symbol, lightBackground);
				const x = getTweenedValue(node.fullString, 'x', currentTime);
				const y = getTweenedValue(node.fullString, 'y', currentTime);
				const w = getTweenedValue(node.fullString, 'width', currentTime);
				const h = getTweenedValue(node.fullString, 'height', currentTime);
				const isTutorTarget = tutorTargetKey === node.fullString;
				ctx.beginPath();
				ctx.rect(
					scaleX(x + 3, node.fullString),
					y + 3,
					Math.max(0, scaleSize(w - 6)),
					Math.max(0, h - 6),
				);
				ctx.fillStyle = isTutorTarget
					? `rgba(${r}, ${g}, ${b}, 0.24)`
					: `rgba(${r}, ${g}, ${b}, 0.13)`;
				ctx.fill();
				ctx.closePath();
			}
		}

		for (const node of visibleNodes) {
			if (!node.parentKey) {
				continue;
			}
			const parent = laidOutNodes[node.parentKey];
			if (!parent) {
				continue;
			}
			const [r, g, b] = timerRgbOnSurface(node.symbol, lightBackground);
			const stroke = connectorStrokeStyle(r, g, b, lightBackground);
			const childGeom = timerCircleGeometry(node, currentTime, getTweenedValue, scaleX);
			const parentGeom = timerCircleGeometry(parent, currentTime, getTweenedValue, scaleX);
			// Parent sits left of child: leave parent circle on the right, enter child circle on the left.
			const startX = parentGeom.cx + parentGeom.r;
			const startY = parentGeom.cy;
			const endX = childGeom.cx - childGeom.r;
			const endY = childGeom.cy;

			ctx.beginPath();
			ctx.moveTo(startX, startY);
			ctx.lineTo(endX, endY);
			ctx.strokeStyle = stroke;
			ctx.lineWidth = 2;
			ctx.stroke();
			ctx.closePath();
		}

		for (const node of visibleNodes) {
			const displayText = displaySymbol(node.symbol);
			const [r, g, b] = timerRgbOnSurface(node.symbol, lightBackground);
			const timer = timers[node.fullString];
			const timerFrac = timer ? timerFraction(time, timer.phase, period) : 0;
			const isTutorTarget = tutorTargetKey === node.fullString;
			const timerFontSize = isTutorTarget ? TIMER_FONT_SIZE * 1.8 : TIMER_FONT_SIZE;
			const padModeGlyph =
				node.symbol === 'N' || node.symbol === 'Q' || node.symbol === 'U';
			const labelFontSize = padModeGlyph ? timerFontSize * PAD_MODE_GLYPH_FONT_SCALE : timerFontSize;
			const { cx, cy, r: baseTimerRadius } = timerCircleGeometry(
				node,
				currentTime,
				getTweenedValue,
				scaleX,
			);
			const timerRadius = isTutorTarget ? baseTimerRadius * 1.8 : baseTimerRadius;

			if (isTutorTarget) {
				ctx.beginPath();
				ctx.arc(cx, cy, timerRadius * 1.55, 0, 2 * Math.PI);
				ctx.fillStyle = lightBackground
					? `rgba(${r}, ${g}, ${b}, 0.12)`
					: 'rgba(255, 255, 255, 0.12)';
				ctx.fill();
				ctx.closePath();
			}

			ctx.beginPath();
			ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 1)`;
			if (node.symbol === STOP_SYMBOL) {
				const squareSize = 17;
				ctx.fillRect(cx - squareSize / 2, cy - squareSize / 2, squareSize, squareSize);
			} else if (displayText) {
				ctx.font = `${labelFontSize}px verdana, helvetica, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif`;
				ctx.textAlign = 'center';
				ctx.textBaseline = 'middle';
				ctx.fillText(displayText, cx, cy);
			}
			ctx.closePath();

			ctx.beginPath();
			ctx.arc(cx, cy, timerRadius, 0, 2 * Math.PI * timerFrac);
			ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${timerFrac * 0.9 + 0.1})`;
			ctx.lineWidth = isTutorTarget ? TIMER_STROKE_WIDTH * 2 : TIMER_STROKE_WIDTH;
			ctx.stroke();
			ctx.closePath();

			if (showDebugStats) {
				const debugFontSize = 9;
				const debugLineHeight = debugFontSize + 1;
				const debugY = cy + timerRadius + 3;
				const debugColor = lightBackground
					? 'rgba(15, 23, 42, 0.9)'
					: 'rgba(255, 255, 255, 0.9)';
				const formatStat = (label: string, value: number | null) =>
					`${label}:${value === null ? '-' : value.toFixed(2)}`;
				const lines = [
					`scroll:${scrollOffset}`,
					`fork:${firstForkDepth ?? '-'}`,
					formatStat('z', node.node.z),
					formatStat('tp', node.node.tp),
					formatStat('tp0', node.node.tp0),
					formatStat('p', node.node.p),
					formatStat('a_tl[0]', node.node.a_tl0),
				];
				ctx.font = `${debugFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
				ctx.fillStyle = debugColor;
				ctx.textAlign = 'center';
				ctx.textBaseline = 'top';
				for (const [idx, line] of lines.entries()) {
					ctx.fillText(line, cx, debugY + idx * debugLineHeight);
				}
			}
		}
	}, [
		laidOutNodes,
		timers,
		time,
		period,
		devicePixelRatio,
		renderMetrics,
		viewportSize,
		getTweenedValue,
		firstForkDepth,
		lightBackground,
		scrollOffset,
		scrollRoot,
		showAll,
		showBoxes,
		showDebugStats,
		targetPhrase,
		tutorTargetKey,
		useVisualTutor,
	]);

	return (
		<div ref={containerRef} className="relative h-full w-full overflow-hidden">
			<canvas
				ref={canvasRef}
				className="block h-full w-full bg-slate-100 dark:bg-black"
			/>
			{offscreenPrefixDisplay && (
				<div
					className="pointer-events-auto absolute bottom-[calc(50%+0.75rem)] left-4 z-10 flex max-h-[min(42vh,calc(50%-1.25rem))] w-[min(20rem,calc(100%-2rem))] flex-col gap-1.5 overflow-y-auto overscroll-contain rounded-lg border border-slate-200/90 bg-white/95 py-2.5 pl-3 pr-2.5 shadow-lg shadow-slate-900/10 ring-1 ring-slate-900/[0.04] backdrop-blur-md dark:border-white/15 dark:bg-gray-950/90 dark:shadow-black/40 dark:ring-white/[0.06]"
					role="region"
					aria-label="Text before the visible trie window"
				>
					<div className="shrink-0 select-none text-[0.65rem] font-medium uppercase tracking-wider text-slate-500 dark:text-gray-500">
						Scrolled prefix
					</div>
					<div className="min-h-0 select-text whitespace-pre-wrap break-words font-mono text-[0.8125rem] leading-relaxed text-slate-800 dark:text-gray-100">
						{offscreenPrefixDisplay}
					</div>
				</div>
			)}
		</div>
	);
}

export default TrieSnapshotVisualizer;
