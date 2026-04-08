import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { colorFromLetter } from '../utils/colors';

// Match TrieVisualizer (v2 canvas): layout transitions over 300ms.
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

export interface ExpandedSnapshotNode {
	z: number;
	p: number | null;
	tp: number | null;
	tp0: number | null;
	a_tl0: number | null;
	hash: number;
}

export type ExpandedSnapshot = Record<string, ExpandedSnapshotNode>;
export interface VisibleNodeTimer {
	phase: number;
}
export type VisibleNodeTimerMap = Record<string, VisibleNodeTimer>;

interface TrieSnapshotVisualizerProps {
	snapshot: ExpandedSnapshot;
	timers: VisibleNodeTimerMap;
	period: number;
	expansionThreshold: number;
	showAll?: boolean;
	/** When true, bright timer strokes (space, root, etc.) are shifted darker for a light canvas. */
	lightBackground?: boolean;
	/** Semi-transparent node background rectangles (default on). */
	showBoxes?: boolean;
	showDebugStats?: boolean;
}

interface VisualNode {
	fullString: string;
	node: ExpandedSnapshotNode;
	symbol: string;
	parentKey: string | null;
	children: string[];
	x: number;
	y: number;
	width: number;
	height: number;
}

const BOX_WIDTH = 37;
const BOX_WIDTH_CHILDREN_MULTIPLIER = 1.0;
const TIMER_RADIUS = 15;
const TIMER_STROKE_WIDTH = 2;
const TIMER_FONT_SIZE = 37;
const DEBUG_LEVEL_GUTTER = 56;

function timerFraction(time: number, phase: number, period: number): number {
	return ((time - phase + period) % period) / period;
}

function finalSymbol(fullString: string): string {
	if (fullString === '^') {
		return '^';
	}
	return fullString.at(-1) ?? '^';
}

function displaySymbol(symbol: string): string {
	if (symbol === '_') {
		return ' ';
	}
	if (symbol === '^') {
		return '^';
	}
	return symbol;
}

function nodeDepth(fullString: string): number {
	if (fullString === '^') {
		return 0;
	}
	return Math.max(0, fullString.length - 1);
}

function nodePassesThreshold(
	node: VisualNode,
	rootZ: number,
	expansionThreshold: number,
	showAll: boolean,
): boolean {
	return showAll || node.node.z - rootZ > expansionThreshold;
}

/** Timer circle center and radius — fixed CSS px like V2 TrieVisualizer (not scaled with fit-to-width). */
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
	if (symbol === '_') {
		return colorFromLetter(' ');
	}
	if (symbol === '^') {
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

function logaddexp(a: number, b: number): number {
	if (a === -Infinity) return b;
	if (b === -Infinity) return a;
	if (a > b) return a + Math.log(1 + Math.exp(b - a));
	return b + Math.log(1 + Math.exp(a - b));
}

function buildTree(snapshot: ExpandedSnapshot): Record<string, VisualNode> {
	const nodes: Record<string, VisualNode> = {};
	for (const [fullString, node] of Object.entries(snapshot)) {
		nodes[fullString] = {
			fullString,
			node,
			symbol: finalSymbol(fullString),
			parentKey: fullString === '^' ? null : fullString.slice(0, -1),
			children: [],
			x: 0,
			y: 0,
			width: 0,
			height: 0,
		};
	}
	for (const node of Object.values(nodes)) {
		if (node.parentKey && nodes[node.parentKey]) {
			nodes[node.parentKey].children.push(node.fullString);
		}
	}
	for (const node of Object.values(nodes)) {
		node.children.sort((a, b) => a.localeCompare(b));
	}
	return nodes;
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

function layoutTree(
	nodes: Record<string, VisualNode>,
	fullString: string,
	y: number,
	height: number,
	width: number,
): void {
	const node = nodes[fullString];
	node.x = node.parentKey === null ? 0 : nodes[node.parentKey].x + nodes[node.parentKey].width;
	node.y = y;
	node.width = width;
	node.height = height;

	if (node.children.length === 0) {
		return;
	}

	let totalChildrenZ = -Infinity;
	for (const childKey of node.children) {
		totalChildrenZ = logaddexp(totalChildrenZ, nodes[childKey].node.z);
	}

	const childWidth =
		BOX_WIDTH * (1 + BOX_WIDTH_CHILDREN_MULTIPLIER * Math.log(node.children.length));
	let childTop = y;
	for (const childKey of node.children) {
		const childZ = nodes[childKey].node.z;
		const childHeight =
			totalChildrenZ === -Infinity
				? height / node.children.length
				: height * Math.exp(childZ - totalChildrenZ);
		layoutTree(nodes, childKey, childTop, childHeight, childWidth);
		childTop += childHeight;
	}
}

function TrieSnapshotVisualizer({
	snapshot,
	timers,
	period,
	expansionThreshold,
	showAll = false,
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

	const laidOutNodes = useMemo(() => {
		if (viewportSize.height <= 0) {
			return null;
		}
		const nodes = buildTree(snapshot);
		if (!nodes['^']) {
			return null;
		}
		layoutTree(nodes, '^', 0, viewportSize.height, BOX_WIDTH * 1.5);
		return nodes;
	}, [snapshot, viewportSize.height]);

	const rootZ = snapshot['^']?.z ?? 0;

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
		let maxDepth = 0;
		for (const node of Object.values(laidOutNodes)) {
			// Extra layout slack so fixed-size circles (V2-sized) don’t crowd the viewport edge when scale < 1.
			contentWidth = Math.max(contentWidth, node.x + node.width + TIMER_RADIUS * 6);
			maxDepth = Math.max(maxDepth, nodeDepth(node.fullString));
		}
		const horizontalPadding = 12;
		const levelGutter = showDebugStats ? DEBUG_LEVEL_GUTTER : 0;
		const availableWidth = Math.max(
			1,
			viewportSize.width - horizontalPadding * 2 - maxDepth * levelGutter,
		);
		const scale = contentWidth > 0 ? Math.min(1, availableWidth / contentWidth) : 1;
		return {
			scale,
			offsetX: horizontalPadding,
			offsetY: 0,
			contentWidth,
			levelGutter,
		};
	}, [laidOutNodes, showDebugStats, viewportSize]);

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

		const orderedNodes = Object.values(laidOutNodes);
		const visibleNodes = orderedNodes.filter((node) =>
			nodePassesThreshold(node, rootZ, expansionThreshold, showAll),
		);
		const scaleX = (value: number, fullString: string) =>
			renderMetrics.offsetX +
			value * renderMetrics.scale +
			nodeDepth(fullString) * renderMetrics.levelGutter;
		const scaleSize = (value: number) => value * renderMetrics.scale;
		const currentTime = performance.now() / 1000;

		if (showBoxes) {
			for (const node of visibleNodes) {
				const [r, g, b] = timerRgbOnSurface(node.symbol, lightBackground);
				const x = getTweenedValue(node.fullString, 'x', currentTime);
				const y = getTweenedValue(node.fullString, 'y', currentTime);
				const w = getTweenedValue(node.fullString, 'width', currentTime);
				const h = getTweenedValue(node.fullString, 'height', currentTime);
				ctx.beginPath();
				ctx.rect(
					scaleX(x + 3, node.fullString),
					y + 3,
					Math.max(0, scaleSize(w - 6)),
					Math.max(0, h - 6),
				);
				ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.13)`;
				ctx.fill();
				ctx.closePath();
			}
		}

		for (const node of visibleNodes) {
			if (!node.parentKey) {
				continue;
			}
			const parent = laidOutNodes[node.parentKey];
			if (
				!parent ||
				!nodePassesThreshold(parent, rootZ, expansionThreshold, showAll)
			) {
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
			const timerFontSize = TIMER_FONT_SIZE;
			const { cx, cy, r: timerRadius } = timerCircleGeometry(
				node,
				currentTime,
				getTweenedValue,
				scaleX,
			);

			ctx.beginPath();
			ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 1)`;
			if (node.symbol === '$') {
				const squareSize = 17;
				ctx.fillRect(cx - squareSize / 2, cy - squareSize / 2, squareSize, squareSize);
			} else if (displayText) {
				ctx.font = `${timerFontSize}px verdana, helvetica, sans-serif`;
				ctx.textAlign = 'center';
				ctx.textBaseline = 'middle';
				ctx.fillText(displayText, cx, cy);
			}
			ctx.closePath();

			ctx.beginPath();
			ctx.arc(cx, cy, timerRadius, 0, 2 * Math.PI * timerFrac);
			ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${timerFrac * 0.9 + 0.1})`;
			ctx.lineWidth = TIMER_STROKE_WIDTH;
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
		expansionThreshold,
		lightBackground,
		rootZ,
		showAll,
		showBoxes,
		showDebugStats,
	]);

	return (
		<div ref={containerRef} className="relative h-full w-full overflow-hidden">
			<canvas
				ref={canvasRef}
				className="block h-full w-full bg-slate-100 dark:bg-black"
			/>
		</div>
	);
}

export default TrieSnapshotVisualizer;
