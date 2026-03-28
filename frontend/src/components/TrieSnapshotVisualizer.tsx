import { useEffect, useMemo, useRef, useState } from 'react';
import { colorFromLetter } from '../utils/colors';

const BOX_WIDTH = 37;
const BOX_WIDTH_CHILDREN_MULTIPLIER = 1.0;
const MIN_BOX_HEIGHT = 6;
const NODE_LABEL_FONT_SIZE = 28;

type SnapshotSymbol =
	| 'A'
	| 'B'
	| 'C'
	| 'D'
	| 'E'
	| 'F'
	| 'G'
	| 'H'
	| 'I'
	| 'J'
	| 'K'
	| 'L'
	| 'M'
	| 'N'
	| 'O'
	| 'P'
	| 'Q'
	| 'R'
	| 'S'
	| 'T'
	| 'U'
	| 'V'
	| 'W'
	| 'X'
	| 'Y'
	| 'Z'
	| 'Space'
	| 'Stop'
	| 'Start';

export interface TrieSnapshotNode {
	symbol: SnapshotSymbol;
	z: number;
	likelihood: number;
	children: [SnapshotSymbol, number][];
}

export interface TrieSnapshot {
	nodes: TrieSnapshotNode[];
	root: number;
}

interface TrieSnapshotVisualizerProps {
	snapshot: TrieSnapshot;
}

interface VisualNode {
	index: number;
	symbol: SnapshotSymbol;
	label: string;
	z: number;
	x: number;
	y: number;
	width: number;
	height: number;
	parentIndex: number | null;
	children: number[];
}

function symbolToLabel(symbol: SnapshotSymbol): string {
	switch (symbol) {
		case 'Space':
			return ' ';
		case 'Stop':
			return '$';
		case 'Start':
			return '';
		default:
			return symbol.toLowerCase();
	}
}

function logaddexp(a: number, b: number): number {
	if (a === -Infinity) return b;
	if (b === -Infinity) return a;
	if (a > b) return a + Math.log(1 + Math.exp(b - a));
	return b + Math.log(1 + Math.exp(a - b));
}

function buildLayout(
	snapshot: TrieSnapshot,
	canvasHeight: number,
): Map<number, VisualNode> {
	const layout = new Map<number, VisualNode>();

	function visit(
		nodeIndex: number,
		parentIndex: number | null,
		x: number,
		y: number,
		height: number,
		width: number,
	): void {
		const node = snapshot.nodes[nodeIndex];
		const childIndices = node.children.map(([, childIndex]) => childIndex);

		layout.set(nodeIndex, {
			index: nodeIndex,
			symbol: node.symbol,
			label: symbolToLabel(node.symbol),
			z: node.z,
			x,
			y,
			width,
			height,
			parentIndex,
			children: childIndices,
		});

		if (childIndices.length === 0) {
			return;
		}

		let totalChildrenZ = -Infinity;
		for (const childIndex of childIndices) {
			totalChildrenZ = logaddexp(totalChildrenZ, snapshot.nodes[childIndex].z);
		}

		const childWidth =
			BOX_WIDTH * (1 + BOX_WIDTH_CHILDREN_MULTIPLIER * Math.log(childIndices.length));
		let childTop = y;
		for (const childIndex of childIndices) {
			const childZ = snapshot.nodes[childIndex].z;
			const childHeight =
				totalChildrenZ === -Infinity
					? height / childIndices.length
					: Math.max(MIN_BOX_HEIGHT, height * Math.exp(childZ - totalChildrenZ));
			visit(childIndex, nodeIndex, x + width, childTop, childHeight, childWidth);
			childTop += childHeight;
		}
	}

	visit(snapshot.root, null, 0, 0, canvasHeight, BOX_WIDTH * 1.5);
	return layout;
}

function drawSnapshot(
	context: CanvasRenderingContext2D,
	layout: Map<number, VisualNode>,
	canvasWidth: number,
	canvasHeight: number,
	devicePixelRatio: number,
): void {
	context.setTransform(1, 0, 0, 1, 0, 0);
	context.clearRect(0, 0, canvasWidth, canvasHeight);
	context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

	const nodes = [...layout.values()];

	for (const node of nodes) {
		const [r, g, b] = colorFromLetter(node.label);
		context.beginPath();
		context.rect(node.x + 3, node.y + 3, node.width - 6, Math.max(0, node.height - 6));
		context.fillStyle = `rgba(${r}, ${g}, ${b}, 0.13)`;
		context.fill();
		context.closePath();
	}

	for (const node of nodes) {
		if (node.parentIndex === null) {
			continue;
		}

		const parent = layout.get(node.parentIndex);
		if (!parent) {
			continue;
		}

		const [r, g, b] = colorFromLetter(node.label || '.');
		const stroke = `rgba(${Math.floor(r / 1.8)}, ${Math.floor(g / 1.8)}, ${Math.floor(b / 1.8)}, 1)`;

		const startX = node.x + node.width;
		const startY = node.y + node.height / 2;
		const endX = parent.x + parent.width;
		const endY = parent.y + parent.height / 2;
		const midX = (startX + endX) / 2;

		context.beginPath();
		context.moveTo(startX, startY);
		context.bezierCurveTo(midX, startY, midX, endY, endX, endY);
		context.strokeStyle = stroke;
		context.lineWidth = 2;
		context.stroke();
		context.closePath();
	}

	for (const node of nodes) {
		if (!node.label) {
			continue;
		}

		const [r, g, b] = colorFromLetter(node.label);
		const textColor = `rgba(${r}, ${g}, ${b}, 1)`;
		const cx = node.x + node.width - 15;
		const cy = node.y + node.height / 2;

		context.beginPath();
		context.fillStyle = textColor;
		if (node.label === '$') {
			const squareSize = 17;
			context.fillRect(cx - squareSize / 2, cy - squareSize / 2, squareSize, squareSize);
		} else {
			context.font = `${NODE_LABEL_FONT_SIZE}px verdana, helvetica, sans-serif`;
			context.textAlign = 'center';
			context.textBaseline = 'middle';
			context.fillText(node.label, cx, cy);
		}
		context.closePath();
	}
}

function TrieSnapshotVisualizer({ snapshot }: TrieSnapshotVisualizerProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [devicePixelRatio, setDevicePixelRatio] = useState(1);
	const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) {
			return;
		}

		const updateCanvasSize = () => {
			const rect = canvas.getBoundingClientRect();
			const dpr = window.devicePixelRatio || 1;
			const width = Math.round(rect.width);
			const height = Math.round(rect.height);
			const pixelWidth = width * dpr;
			const pixelHeight = height * dpr;

			setDevicePixelRatio(dpr);
			setCanvasSize((current) =>
				current.width === width && current.height === height ? current : { width, height },
			);

			if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
				canvas.width = pixelWidth;
				canvas.height = pixelHeight;
			}
			canvas.style.width = `${width}px`;
			canvas.style.height = `${height}px`;
		};

		updateCanvasSize();
		const resizeObserver = new ResizeObserver(() => updateCanvasSize());
		resizeObserver.observe(canvas);
		window.addEventListener('resize', updateCanvasSize);
		return () => {
			resizeObserver.disconnect();
			window.removeEventListener('resize', updateCanvasSize);
		};
	}, []);

	const layout = useMemo(() => {
		if (canvasSize.height <= 0) {
			return new Map<number, VisualNode>();
		}
		return buildLayout(snapshot, canvasSize.height);
	}, [snapshot, canvasSize.height]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) {
			return;
		}
		const context = canvas.getContext('2d');
		if (!context) {
			return;
		}
		drawSnapshot(context, layout, canvas.width, canvas.height, devicePixelRatio);
	}, [layout, devicePixelRatio]);

	return (
		<div className="relative h-full w-full">
			<canvas ref={canvasRef} className="h-full w-full bg-black" />
		</div>
	);
}

export default TrieSnapshotVisualizer;
