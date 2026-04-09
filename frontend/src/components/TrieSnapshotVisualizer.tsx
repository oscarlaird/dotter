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
	upper_siblings_inclusive_cum_z: number | null;
	hash: number;
}

export type ExpandedSnapshot = Record<string, ExpandedSnapshotNode>;
export interface VisibleNodeTimer {
	phase: number;
}
export type VisibleNodeTimerMap = Record<string, VisibleNodeTimer>;

export interface ScrollLayoutState {
	firstForkDepth: number;
	firstForkFullString: string | null;
	scrollOffset: number;
	scrollRoot: string;
	scrollAncestorKeys: string[];
	renderedNodeKeys: string[];
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
const ROOT_NODE_WIDTH = BOX_WIDTH * 1.5;
const SCROLL_CENTERING_WEIGHT = 1;
const SCROLL_STABILITY_WEIGHT = 1;

export const SINGLE_PARENT_NODE_WIDTH_PX = BOX_WIDTH;
export const SCROLL_TARGET_X_PX = 400;

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

function offscreenPrefixText(scrollRoot: string): string {
	if (scrollRoot === '^') {
		return '';
	}
	return scrollRoot.slice(1).replaceAll('_', ' ');
}

function nodeDepth(fullString: string): number {
	if (fullString === '^') {
		return 0;
	}
	return Math.max(0, fullString.length - 1);
}

function snapshotNodePassesThreshold(
	node: ExpandedSnapshotNode,
	rootZ: number,
	expansionThreshold: number,
): boolean {
	return node.z - rootZ > expansionThreshold;
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

function rootZOf(snapshot: ExpandedSnapshot): number {
	return snapshot['^']?.z ?? 0;
}

function ancestorKeysThroughRoot(fullString: string): string[] {
	if (fullString === '^') {
		return ['^'];
	}
	const keys = ['^'];
	for (let depth = 1; depth <= nodeDepth(fullString); depth += 1) {
		keys.push(fullString.slice(0, depth + 1));
	}
	return keys;
}

function filterNodesByKeySet(
	nodes: Record<string, VisualNode>,
	keys: Set<string>,
): Record<string, VisualNode> {
	const filtered: Record<string, VisualNode> = {};
	for (const key of keys) {
		const node = nodes[key];
		if (!node) {
			continue;
		}
		filtered[key] = {
			...node,
			children: node.children.filter((childKey) => keys.has(childKey)),
		};
	}
	return filtered;
}

export function buildVisibleTree(
	snapshot: ExpandedSnapshot,
	expansionThreshold: number,
): Record<string, VisualNode> {
	const nodes = buildTree(snapshot);
	const rootZ = rootZOf(snapshot);
	const visibleKeys = new Set<string>(['^']);
	for (const [fullString, node] of Object.entries(snapshot)) {
		if (!snapshotNodePassesThreshold(node, rootZ, expansionThreshold)) {
			continue;
		}
		for (const ancestorKey of ancestorKeysThroughRoot(fullString)) {
			visibleKeys.add(ancestorKey);
		}
	}
	return filterNodesByKeySet(nodes, visibleKeys);
}

function subtreeKeys(
	nodes: Record<string, VisualNode>,
	rootKey: string,
): Set<string> {
	const included = new Set<string>();
	const stack = [rootKey];
	while (stack.length > 0) {
		const key = stack.pop();
		if (!key || included.has(key)) {
			continue;
		}
		const node = nodes[key];
		if (!node) {
			continue;
		}
		included.add(key);
		for (const childKey of node.children) {
			stack.push(childKey);
		}
	}
	return included;
}

function firstForkNode(
	nodes: Record<string, VisualNode>,
): VisualNode | null {
	let best: VisualNode | null = null;
	for (const node of Object.values(nodes)) {
		if (node.children.length < 2) {
			continue;
		}
		if (!best || nodeDepth(node.fullString) < nodeDepth(best.fullString)) {
			best = node;
		}
	}
	return best;
}

function deepestVisibleDepth(nodes: Record<string, VisualNode>): number {
	let deepest = 0;
	for (const node of Object.values(nodes)) {
		deepest = Math.max(deepest, nodeDepth(node.fullString));
	}
	return deepest;
}

function ancestorAtDepth(fullString: string, depth: number): string {
	if (depth <= 0) {
		return '^';
	}
	return fullString.slice(0, depth + 1);
}

function scrollAncestorKeys(scrollRoot: string): string[] {
	if (scrollRoot === '^') {
		return [];
	}
	return ancestorKeysThroughRoot(scrollRoot).slice(0, -1);
}

function rootWidthFor(fullString: string): number {
	return fullString === '^' ? ROOT_NODE_WIDTH : BOX_WIDTH;
}

function relativeDepth(fullString: string, scrollRoot: string): number {
	return Math.max(0, nodeDepth(fullString) - nodeDepth(scrollRoot));
}

export function computeScrollLayoutState(
	snapshot: ExpandedSnapshot,
	expansionThreshold: number,
	previousScrollOffset: number,
): ScrollLayoutState {
	const visibleTree = buildVisibleTree(snapshot, expansionThreshold);
	const firstFork = firstForkNode(visibleTree);
	const firstForkDepth = firstFork
		? nodeDepth(firstFork.fullString)
		: deepestVisibleDepth(visibleTree);
	const unclampedOffset =
		(
			SCROLL_CENTERING_WEIGHT *
				(firstForkDepth - SCROLL_TARGET_X_PX / SINGLE_PARENT_NODE_WIDTH_PX) +
			SCROLL_STABILITY_WEIGHT * previousScrollOffset
		) /
		(SCROLL_CENTERING_WEIGHT + SCROLL_STABILITY_WEIGHT);
	const scrollOffset =
		firstForkDepth <= 0
			? 0
			: Math.max(0, Math.min(Math.floor(unclampedOffset), firstForkDepth - 1));
	const scrollRoot = firstFork
		? ancestorAtDepth(firstFork.fullString, scrollOffset)
		: '^';
	return {
		firstForkDepth,
		firstForkFullString: firstFork?.fullString ?? null,
		scrollOffset,
		scrollRoot,
		scrollAncestorKeys: scrollAncestorKeys(scrollRoot),
		renderedNodeKeys: Array.from(subtreeKeys(visibleTree, scrollRoot)),
	};
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
	node.x =
		node.parentKey === null || !nodes[node.parentKey]
			? 0
			: nodes[node.parentKey].x + nodes[node.parentKey].width;
	node.y = y;
	node.width = width;
	node.height = height;

	if (node.children.length === 0) {
		return;
	}

	const childWidth =
		BOX_WIDTH * (1 + BOX_WIDTH_CHILDREN_MULTIPLIER * Math.log(node.children.length));
	for (const childKey of node.children) {
		const child = nodes[childKey];
		const childBottomZ = child.node.upper_siblings_inclusive_cum_z;
		if (childBottomZ === null) {
			throw new Error(`Missing upper_siblings_inclusive_cum_z for ${child.fullString}`);
		}
		const childHeight = height * Math.exp(child.node.z - node.node.z);
		const childBottom = y + height * Math.exp(childBottomZ - node.node.z);
		const childTop = childBottom - childHeight;
		layoutTree(nodes, childKey, childTop, childHeight, childWidth);
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
	const offscreenText = offscreenPrefixText(scrollRoot);

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

	const laidOutNodes = useMemo(() => {
		if (viewportSize.height <= 0) {
			return null;
		}
		const baseNodes = showAll ? buildTree(snapshot) : visibleTree;
		const visibleKeys = subtreeKeys(baseNodes, scrollRoot);
		const nodes = filterNodesByKeySet(baseNodes, visibleKeys);
		if (!nodes[scrollRoot]) {
			return null;
		}
		layoutTree(nodes, scrollRoot, 0, viewportSize.height, rootWidthFor(scrollRoot));
		return nodes;
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
	]);

	return (
		<div ref={containerRef} className="relative h-full w-full overflow-hidden">
			<canvas
				ref={canvasRef}
				className="block h-full w-full bg-slate-100 dark:bg-black"
			/>
			{offscreenText && (
				<div className="pointer-events-none absolute left-4 top-[38%] w-[250px] -translate-y-1/2 rounded-md border border-slate-300/80 bg-white/90 px-3 py-2 text-sm text-slate-800 shadow-sm backdrop-blur-sm dark:border-white/20 dark:bg-black/70 dark:text-gray-100">
					<div className="whitespace-pre-wrap break-words font-mono leading-relaxed">
						{offscreenText}
					</div>
				</div>
			)}
		</div>
	);
}

export default TrieSnapshotVisualizer;
