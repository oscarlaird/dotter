import { useEffect, useMemo, useState } from 'react';
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

interface SnapshotEntry {
	fullString: string;
	node: ExpandedSnapshotNode;
}

const TIMER_RADIUS = 18;
const TIMER_STROKE_WIDTH = 2;
const TIMER_FONT_SIZE = 30;

function barWidthPercent(z: number, maxZ: number): number {
	return Math.max(3, Math.exp(z - maxZ) * 100);
}

function timerFraction(time: number, phase: number, period: number): number {
	return ((time - phase + period) % period) / period;
}

function finalSymbol(fullString: string): string {
	if (fullString === '^') {
		return '^';
	}
	return fullString.at(-1) ?? '^';
}

function timerDisplayText(symbol: string): string {
	if (symbol === '_') {
		return '_';
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

function TrieSnapshotVisualizer({ snapshot, timers, period }: TrieSnapshotVisualizerProps) {
	const [time, setTime] = useState(() => performance.now() / 1000);

	useEffect(() => {
		let frame = 0;
		const tick = () => {
			setTime(performance.now() / 1000);
			frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
		return () => cancelAnimationFrame(frame);
	}, []);

	const entries = useMemo(() => {
		return Object.entries(snapshot)
			.map(([fullString, node]) => ({ fullString, node }))
			.sort((a, b) => b.node.z - a.node.z || a.fullString.localeCompare(b.fullString));
	}, [snapshot]);

	const maxZ = entries[0]?.node.z ?? 0;

	return (
		<div className="h-full overflow-auto bg-black p-4">
			<div className="grid gap-3">
				{entries.map((entry: SnapshotEntry) => (
					<TimerRow
						key={entry.fullString}
						entry={entry}
						timer={timers[entry.fullString]}
						maxZ={maxZ}
						time={time}
						period={period}
					/>
				))}
			</div>
		</div>
	);
}

interface TimerRowProps {
	entry: SnapshotEntry;
	timer: VisibleNodeTimer | undefined;
	maxZ: number;
	time: number;
	period: number;
}

function TimerRow({ entry, timer, maxZ, time, period }: TimerRowProps) {
	const symbol = finalSymbol(entry.fullString);
	const displayText = timerDisplayText(symbol);
	const [r, g, b] = timerColor(symbol);
	const timerFrac = timer ? timerFraction(time, timer.phase, period) : 0;
	const circumference = 2 * Math.PI * TIMER_RADIUS;
	const dashOffset = circumference * (1 - timerFrac);
	const circleColor = `rgba(${r}, ${g}, ${b}, 1)`;
	const arcColor = `rgba(${r}, ${g}, ${b}, ${timerFrac * 0.9 + 0.1})`;

	return (
		<div className="rounded-lg border border-white/10 bg-white/5 p-3">
			<div className="mb-2 flex items-center gap-4">
				<svg
					width={TIMER_RADIUS * 2 + 10}
					height={TIMER_RADIUS * 2 + 10}
					viewBox={`0 0 ${TIMER_RADIUS * 2 + 10} ${TIMER_RADIUS * 2 + 10}`}
					className="shrink-0 overflow-visible"
				>
					<circle
						cx={TIMER_RADIUS + 5}
						cy={TIMER_RADIUS + 5}
						r={TIMER_RADIUS}
						fill="none"
						stroke="rgba(255,255,255,0.08)"
						strokeWidth={TIMER_STROKE_WIDTH}
					/>
					<circle
						cx={TIMER_RADIUS + 5}
						cy={TIMER_RADIUS + 5}
						r={TIMER_RADIUS}
						fill="none"
						stroke={arcColor}
						strokeWidth={TIMER_STROKE_WIDTH}
						strokeDasharray={circumference}
						strokeDashoffset={dashOffset}
						strokeLinecap="round"
						transform={`rotate(-90 ${TIMER_RADIUS + 5} ${TIMER_RADIUS + 5})`}
					/>
					{displayText === '$' ? (
						<rect
							x={TIMER_RADIUS - 3}
							y={TIMER_RADIUS - 3}
							width={16}
							height={16}
							fill={circleColor}
						/>
					) : (
						<text
							x={TIMER_RADIUS + 5}
							y={TIMER_RADIUS + 5}
							fill={circleColor}
							textAnchor="middle"
							dominantBaseline="middle"
							fontSize={TIMER_FONT_SIZE}
							fontFamily="verdana, helvetica, sans-serif"
						>
							{displayText}
						</text>
					)}
				</svg>
				<div className="min-w-0 flex-1">
					<div className="mb-2 flex items-center justify-between gap-4">
						<code className="truncate text-sm text-cyan-200">{entry.fullString}</code>
						<span className="shrink-0 text-xs text-gray-400">
							z={entry.node.z.toFixed(3)}
						</span>
					</div>
					<div className="h-2 overflow-hidden rounded bg-white/10">
						<div
							className="h-full rounded"
							style={{
								width: `${barWidthPercent(entry.node.z, maxZ)}%`,
								backgroundColor: `rgba(${r}, ${g}, ${b}, 0.8)`,
							}}
						/>
					</div>
				</div>
			</div>
		</div>
	);
}

export default TrieSnapshotVisualizer;
