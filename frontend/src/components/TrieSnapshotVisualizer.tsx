import { useEffect, useMemo, useRef, useState } from 'react';
import { colorFromLetter } from '../utils/colors';

export interface ExpandedSnapshotNode {
	z: number;
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
const MIN_BOX_HEIGHT = 6;
const TIMER_RADIUS = 15;
const TIMER_STROKE_WIDTH = 2;
const TIMER_FONT_SIZE = 37;

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
		return '';
	}
	return symbol;
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
				: Math.max(MIN_BOX_HEIGHT, height * Math.exp(childZ - totalChildrenZ));
		layoutTree(nodes, childKey, childTop, childHeight, childWidth);
		childTop += childHeight;
	}
}

function TrieSnapshotVisualizer({ snapshot, timers, period }: TrieSnapshotVisualizerProps) {
	const [time, setTime] = useState(() => performance.now() / 1000);
	const containerRef = useRef<HTMLDivElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [devicePixelRatio, setDevicePixelRatio] = useState(1);
	const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });

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

	const renderMetrics = useMemo(() => {
		if (!laidOutNodes) {
			return {
				scale: 1,
				offsetX: 0,
				offsetY: 0,
				contentWidth: viewportSize.width,
			};
		}
		let contentWidth = 0;
		for (const node of Object.values(laidOutNodes)) {
			contentWidth = Math.max(contentWidth, node.x + node.width + TIMER_RADIUS * 3);
		}
		const horizontalPadding = 12;
		const availableWidth = Math.max(1, viewportSize.width - horizontalPadding * 2);
		const scale = contentWidth > 0 ? Math.min(1, availableWidth / contentWidth) : 1;
		return {
			scale,
			offsetX: horizontalPadding,
			offsetY: 0,
			contentWidth,
		};
	}, [laidOutNodes, viewportSize]);

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
		const scaleX = (value: number) => renderMetrics.offsetX + value * renderMetrics.scale;
		const scaleSize = (value: number) => value * renderMetrics.scale;

		for (const node of orderedNodes) {
			const [r, g, b] = timerColor(node.symbol);
			ctx.beginPath();
			ctx.rect(
				scaleX(node.x + 3),
				node.y + 3,
				Math.max(0, scaleSize(node.width - 6)),
				Math.max(0, node.height - 6),
			);
			ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.13)`;
			ctx.fill();
			ctx.closePath();
		}

		for (const node of orderedNodes) {
			if (!node.parentKey) {
				continue;
			}
			const parent = laidOutNodes[node.parentKey];
			if (!parent) {
				continue;
			}
			const [r, g, b] = timerColor(node.symbol);
			const stroke = `rgba(${Math.floor(r / 1.8)}, ${Math.floor(g / 1.8)}, ${Math.floor(b / 1.8)}, 1)`;
			const startX = scaleX(node.x + node.width);
			const startY = node.y + node.height / 2;
			const endX = scaleX(parent.x + parent.width);
			const endY = parent.y + parent.height / 2;
			const midX = (startX + endX) / 2;

			ctx.beginPath();
			ctx.moveTo(startX, startY);
			ctx.bezierCurveTo(midX, startY, midX, endY, endX, endY);
			ctx.strokeStyle = stroke;
			ctx.lineWidth = Math.max(1, scaleSize(2));
			ctx.stroke();
			ctx.closePath();
		}

		for (const node of orderedNodes) {
			const displayText = displaySymbol(node.symbol);
			const [r, g, b] = timerColor(node.symbol);
			const timer = timers[node.fullString];
			const timerFrac = timer ? timerFraction(time, timer.phase, period) : 0;
			let timerRadius = TIMER_RADIUS;
			let timerFontSize = TIMER_FONT_SIZE;
			if (displayText === 'm' || displayText === 'w') {
				timerRadius *= 1.15;
			}
			timerRadius = Math.max(6, scaleSize(timerRadius));
			timerFontSize = Math.max(10, scaleSize(timerFontSize));
			const cx = scaleX(node.x + node.width - TIMER_RADIUS);
			const cy = node.y + node.height / 2;

			ctx.beginPath();
			ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 1)`;
			if (node.symbol === '$') {
				const squareSize = Math.max(8, scaleSize(17));
				ctx.fillRect(cx - squareSize / 2, cy - squareSize / 2, squareSize, squareSize);
			} else if (displayText) {
				ctx.font = `${timerFontSize}px verdana, helvetica, sans-serif`;
				ctx.textAlign = 'center';
				ctx.textBaseline = 'middle';
				ctx.fillText(displayText, cx, cy);
			}
			ctx.closePath();

			if (node.symbol !== '^') {
				ctx.beginPath();
				ctx.arc(cx, cy, timerRadius, 0, 2 * Math.PI * timerFrac);
				ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${timerFrac * 0.9 + 0.1})`;
				ctx.lineWidth = Math.max(1, scaleSize(TIMER_STROKE_WIDTH));
				ctx.stroke();
				ctx.closePath();
			}
		}
	}, [laidOutNodes, timers, time, period, devicePixelRatio, renderMetrics, viewportSize]);

	return (
		<div ref={containerRef} className="relative h-full w-full overflow-hidden">
			<canvas ref={canvasRef} className="block h-full w-full bg-black" />
		</div>
	);
}

export default TrieSnapshotVisualizer;
