import { useEffect, useRef, useState, useCallback } from 'react';
import { colorFromLetter } from '../utils/colors';
import { logaddexp } from '../utils/trie_logic';
import type { TrieNode } from '../utils/trie_logic';
import type { LikelihoodModel } from './CalibrationSettings';
import type { DelayPair } from '../utils/stats';
import Eye from './Eye';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TWEEN_DURATION_MS = 300;
const BOX_WIDTH = 37;
const BOX_WIDTH_CHILDREN_MULTIPLIER = 1.0;
const TIMER_CIRCLE_RADIUS = 15;
const TIMER_CIRCLE_WIDTH = 2;
const TIMER_FONT_SIZE = 37;

// ---------------------------------------------------------------------------
// Prop types
// ---------------------------------------------------------------------------

export interface SetLikelihoodsEvent {
	new_likelihoods: Record<string, { likelihood: number; delay_pair: DelayPair }>;
	click_seq: number[];
}

interface TrieVisualizerProps {
	trie: TrieNode;
	trieUpdatedFlag: boolean;
	setTrieUpdatedFlag: (value: boolean) => void;
	useVisualTutor: boolean;
	setUseVisualTutor: (value: boolean) => void;
	targetPhrase: string;
	onSetLikelihoods: (event: SetLikelihoodsEvent) => void;
	likelihoodModel: LikelihoodModel;
	multilineMode: boolean;
	setMultilineMode: (value: boolean) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function interpolate(start: number, end: number, progress: number): number {
	return start + (end - start) * progress;
}

function normalLogpdf(x: number, mean: number, stddev: number): number {
	return (
		-0.5 * Math.pow((x - mean) / stddev, 2) - Math.log(stddev * Math.sqrt(2 * Math.PI))
	);
}

function timerLikelihood(
	time: number,
	phase: number,
	model: LikelihoodModel,
): { likelihood: number; delay_pair: DelayPair } {
	let delay = time - phase;
	delay = ((delay + model.period * 1.5) % model.period) - model.period / 2;
	const gaussianLogLikelihood = normalLogpdf(delay, model.mu_delay, model.stddev_delay);
	const uniformLogLikelihood = Math.log(1 / model.period);
	const outlierProb = Math.log(model.outliers);
	const notOutlierProb = Math.log(1 - model.outliers);
	return {
		likelihood: logaddexp(
			notOutlierProb + gaussianLogLikelihood,
			outlierProb + uniformLogLikelihood,
		),
		delay_pair: { delay, period: model.period },
	};
}

function playClickSound(): void {
	const ctx = new AudioContext();
	const osc = ctx.createOscillator();
	const gain = ctx.createGain();
	osc.frequency.value = 1000;
	gain.gain.setValueAtTime(1, ctx.currentTime);
	gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.05);
	osc.connect(gain);
	gain.connect(ctx.destination);
	osc.start(ctx.currentTime);
	osc.stop(ctx.currentTime + 0.05);
}

// ---------------------------------------------------------------------------
// Tween bookkeeping (module-level to avoid re-creation)
// ---------------------------------------------------------------------------

type TweenKey = string;
interface TweenEntry {
	start: number;
	from: number;
	to: number;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function TrieVisualizer({
	trie,
	trieUpdatedFlag,
	setTrieUpdatedFlag,
	useVisualTutor,
	setUseVisualTutor,
	targetPhrase,
	onSetLikelihoods,
	likelihoodModel,
	multilineMode,
	setMultilineMode,
}: TrieVisualizerProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [ctx, setCtx] = useState<CanvasRenderingContext2D | null>(null);
	const [firstBoxHeight, setFirstBoxHeight] = useState(0);
	const [visibleNodes, setVisibleNodes] = useState<TrieNode[]>([]);
	const [devicePixelRatio, setDevicePixelRatio] = useState(1);
	const [showBoxes, setShowBoxes] = useState(true);
	const [blinkToClick, setBlinkToClick] = useState(true);
	const [playClickSoundEnabled, setPlayClickSoundEnabled] = useState(false);
	const [developerVisualizer, setDeveloperVisualizer] = useState(false);
	const [leftOffset] = useState(0);
	const [clickSeq, setClickSeq] = useState<number[]>([]);

	const animationFrameRef = useRef<number | null>(null);
	const tweenStartTimesRef = useRef<Record<TweenKey, TweenEntry>>({});
	const oldVisibleRegistryRef = useRef<Record<string, TrieNode>>({});
	const keepPhasesRef = useRef(false);

	// ------------------------------------------------------------------
	// Tween helpers
	// ------------------------------------------------------------------

	const getTweenedValue = useCallback(
		(node: TrieNode, property: string, currentTime: number): number => {
			const key: TweenKey = `${node.val}_${property}`;
			const raw = (node as unknown as Record<string, number>)[property] ?? 0;
			if (!tweenStartTimesRef.current[key]) {
				tweenStartTimesRef.current[key] = { start: currentTime, from: raw, to: raw };
			}
			const tween = tweenStartTimesRef.current[key];
			const progress = Math.min((currentTime - tween.start) / (TWEEN_DURATION_MS / 1000), 1);
			return interpolate(tween.from, tween.to, progress);
		},
		[],
	);

	const updateTween = useCallback(
		(node: TrieNode, property: string, targetValue: number, currentTime: number): void => {
			const key: TweenKey = `${node.val}_${property}`;
			if (!tweenStartTimesRef.current[key]) {
				tweenStartTimesRef.current[key] = { start: currentTime, from: targetValue, to: targetValue };
				return;
			}
			const current = getTweenedValue(node, property, currentTime);
			tweenStartTimesRef.current[key] = { start: currentTime, from: current, to: targetValue };
		},
		[getTweenedValue],
	);

	// ------------------------------------------------------------------
	// Layout (setLocations)
	// ------------------------------------------------------------------

	const setLocations = useCallback(
		(
			node: TrieNode,
			rootCall = false,
			loc = { x: 0, y: 0 },
			sizeHeight = firstBoxHeight,
			sizeWidth = BOX_WIDTH,
			accumulatedNodes: TrieNode[] | null = null,
		): void => {
			const oldVisibleRegistry = oldVisibleRegistryRef.current;
			const keepPhases = keepPhasesRef.current;
			const nodesList: TrieNode[] = rootCall ? [] : (accumulatedNodes ?? []);

			if (rootCall) {
				loc = { x: 0, y: 0 };
				sizeWidth *= 1.5;
			}

			const currentTime = performance.now() / 1000;
			nodesList.push(node);

			node.location = { ...loc };
			node.size_height = sizeHeight;
			node.size_width = sizeWidth;

			updateTween(node, 'location_x', loc.x, currentTime);
			updateTween(node, 'location_y', loc.y, currentTime);
			updateTween(node, 'size_height', sizeHeight, currentTime);
			updateTween(node, 'size_width', sizeWidth, currentTime);

			if (node.val in oldVisibleRegistry && keepPhases) {
				node.phase = oldVisibleRegistry[node.val].phase;
			} else {
				node.phase = Math.random() * likelihoodModel.period;
			}

			const visibleChildren = node.children.filter(
				(c) => c.is_visible || (keepPhases && c.val in oldVisibleRegistry),
			);
			const visibleChildrenVals = new Set(visibleChildren.map((c) => c.val));

			if (visibleChildren.length > 0) {
				const numChildren = visibleChildren.length;
				const boxWidthMultiplier = 1 + BOX_WIDTH_CHILDREN_MULTIPLIER * Math.log(numChildren);
				let yRelativeBottom = 0;
				const totalChildrenPostZ = node.children.reduce(
					(acc, c) => logaddexp(acc, c.post_Z),
					-Infinity,
				);

				node.children.forEach((child) => {
					const childFrac = child.post_Z - totalChildrenPostZ;
					const childHeight = sizeHeight * Math.exp(childFrac);

					if (visibleChildrenVals.has(child.val)) {
						if (!(child.val in oldVisibleRegistry)) {
							const now = performance.now() / 1000;
							if (node.val === '' || !keepPhases) {
								child.go_live_time = now;
							} else if (node.go_live_time !== undefined && now < node.go_live_time) {
								child.go_live_time = node.go_live_time;
							} else {
								const phase = node.phase ?? 0;
								let timeRemaining = (phase - now) % likelihoodModel.period;
								timeRemaining = ((timeRemaining % likelihoodModel.period) + likelihoodModel.period) % likelihoodModel.period;
								const timeSinceLast = likelihoodModel.period - timeRemaining;
								const parentGoLive = node.go_live_time ?? 0;
								if (timeSinceLast < 0.25 || now < parentGoLive + 0.25) {
									child.go_live_time = now;
								} else {
									child.go_live_time =
										now + timeRemaining + likelihoodModel.mu_delay + 2.5 * likelihoodModel.stddev_delay;
								}
							}
						}
						setLocations(
							child,
							false,
							{ x: node.location!.x + node.size_width!, y: node.location!.y + yRelativeBottom },
							childHeight,
							boxWidthMultiplier * BOX_WIDTH,
							nodesList,
						);
					}
					yRelativeBottom += childHeight;
				});
			}

			if (rootCall) {
				const newRegistry: Record<string, TrieNode> = {};
				nodesList.forEach((n) => { newRegistry[n.val] = n; });
				oldVisibleRegistryRef.current = newRegistry;
				if (!keepPhases) keepPhasesRef.current = true;
				setVisibleNodes([...nodesList]);
			}
		},
		[firstBoxHeight, likelihoodModel, updateTween],
	);

	// ------------------------------------------------------------------
	// Click handler
	// ------------------------------------------------------------------

	const click = useCallback(
		(event: MouseEvent | KeyboardEvent | { timeStamp: number }) => {
			if (playClickSoundEnabled) playClickSound();

			const time =
				'timeStamp' in event && event.timeStamp
					? event.timeStamp / 1000
					: performance.now() / 1000;

			const newClickSeq = [...clickSeq, time];
			setClickSeq(newClickSeq);

			const now = performance.now() / 1000;
			const newLikelihoods: Record<string, { likelihood: number; delay_pair: DelayPair }> = {};
			visibleNodes
				.filter((node) => !(node.go_live_time && node.go_live_time > now))
				.forEach((node) => {
					newLikelihoods[node.val] = timerLikelihood(time, node.phase ?? 0, likelihoodModel);
				});

			keepPhasesRef.current = false;
			onSetLikelihoods({ new_likelihoods: newLikelihoods, click_seq: newClickSeq });
		},
		[clickSeq, visibleNodes, likelihoodModel, playClickSoundEnabled, onSetLikelihoods],
	);

	const handleBlink = useCallback(() => {
		click({ timeStamp: performance.now() });
	}, [click]);

	// ------------------------------------------------------------------
	// Canvas setup and event listeners
	// ------------------------------------------------------------------

	useEffect(() => {
		if (!canvasRef.current) return;
		const canvas = canvasRef.current;
		const context = canvas.getContext('2d');
		if (!context) return;

		setCtx(context);
		setFirstBoxHeight(canvas.clientHeight);

		const dpr = window.devicePixelRatio || 1;
		setDevicePixelRatio(dpr);
		const rect = canvas.getBoundingClientRect();
		canvas.width = rect.width * dpr;
		canvas.height = rect.height * dpr;
		context.setTransform(dpr, 0, 0, dpr, 0, 0);
		canvas.style.width = `${rect.width}px`;
		canvas.style.height = `${rect.height}px`;
		canvas.tabIndex = 0;
	}, []);

	// Attach event listeners whenever `click` changes
	useEffect(() => {
		if (!canvasRef.current) return;
		const canvas = canvasRef.current;

		const handleKeyDown = (e: KeyboardEvent) => {
			if (e.code === 'Space') {
				const tag = (document.activeElement as HTMLElement).tagName;
				const isText =
					(tag === 'INPUT' || tag === 'TEXTAREA') &&
					(document.activeElement as HTMLInputElement).type !== 'checkbox';
				if (!isText) {
					click(e);
					e.preventDefault();
				}
			}
		};

		document.addEventListener('keydown', handleKeyDown);
		canvas.addEventListener('click', click as EventListener);

		return () => {
			document.removeEventListener('keydown', handleKeyDown);
			canvas.removeEventListener('click', click as EventListener);
		};
	}, [click]);

	// Re-layout trie when it changes
	useEffect(() => {
		if (trieUpdatedFlag) {
			setLocations(trie, true);
			setTrieUpdatedFlag(false);
		}
	}, [trieUpdatedFlag, trie, setLocations, setTrieUpdatedFlag]);

	// ------------------------------------------------------------------
	// Draw loop
	// ------------------------------------------------------------------

	const draw = useCallback(() => {
		if (!ctx || !canvasRef.current) return;

		const time = performance.now() / 1000;
		const currentTime = time;
		const now = time;

		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
		ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

		const liveNodes = visibleNodes.filter(
			(n) => !(n.go_live_time && n.go_live_time > now),
		);

		// Find the longest target-prefix on screen (for visual tutor)
		let longestTargetVal = '';
		liveNodes.forEach((node) => {
			if (
				targetPhrase.startsWith(node.val) &&
				node.val.length > longestTargetVal.length
			) {
				longestTargetVal = node.val;
			}
		});

		// Draw that node last (on top)
		liveNodes.sort((a, b) => {
			if (a.val === longestTargetVal) return 1;
			if (b.val === longestTargetVal) return -1;
			return 0;
		});

		// Background boxes
		liveNodes.forEach((node) => {
			if (!showBoxes && !developerVisualizer) return;
			const color = colorFromLetter(node.letter);
			const fill = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.13)`;
			const locX = getTweenedValue(node, 'location_x', currentTime);
			const locY = getTweenedValue(node, 'location_y', currentTime);
			const sizeW = getTweenedValue(node, 'size_width', currentTime);
			const sizeH = getTweenedValue(node, 'size_height', currentTime);

			ctx.beginPath();
			ctx.rect(locX - leftOffset + 3, locY + 3, sizeW - 6, sizeH - 6);
			ctx.fillStyle = fill;
			ctx.fill();
			ctx.closePath();
		});

		// Connections
		const registry: Record<string, TrieNode> = {};
		visibleNodes.forEach((n) => { registry[n.val] = n; });

		liveNodes.forEach((node) => {
			if (node.val.length < 1) return;
			const parentNode = registry[node.val.slice(0, -1)];
			if (!parentNode) return;

			const color = colorFromLetter(node.letter);
			const darker = `rgba(${Math.floor(color[0] / 1.8)}, ${Math.floor(color[1] / 1.8)}, ${Math.floor(color[2] / 1.8)}, 1.0)`;

			const nX = getTweenedValue(node, 'location_x', currentTime);
			const nY = getTweenedValue(node, 'location_y', currentTime);
			const nW = getTweenedValue(node, 'size_width', currentTime);
			const nH = getTweenedValue(node, 'size_height', currentTime);
			const pX = getTweenedValue(parentNode, 'location_x', currentTime);
			const pY = getTweenedValue(parentNode, 'location_y', currentTime);
			const pW = getTweenedValue(parentNode, 'size_width', currentTime);
			const pH = getTweenedValue(parentNode, 'size_height', currentTime);

			const startX = nX + nW - leftOffset;
			const startY = nY + nH / 2;
			const endX = pX + pW - leftOffset;
			const endY = pY + pH / 2;
			const midX = (startX + endX) / 2;

			ctx.beginPath();
			ctx.moveTo(startX, startY);
			ctx.bezierCurveTo(midX, startY, midX, endY, endX, endY);
			ctx.strokeStyle = darker;
			ctx.lineWidth = 2;
			ctx.stroke();
			ctx.closePath();
		});

		// Timer circles + letters
		liveNodes.forEach((node) => {
			const timerFrac =
				((time - (node.phase ?? 0) + likelihoodModel.period) % likelihoodModel.period) /
				likelihoodModel.period;
			const color = colorFromLetter(node.letter);
			const colorStr = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1.0)`;
			const isTarget = useVisualTutor && node.val === longestTargetVal;

			let timerFontSize = TIMER_FONT_SIZE;
			let timerRadius = TIMER_CIRCLE_RADIUS;
			if (isTarget) {
				timerFontSize *= 2;
				timerRadius *= 2;
			}
			if (node.letter === 'm' || node.letter === 'w') timerRadius *= 1.15;

			const locX = getTweenedValue(node, 'location_x', currentTime);
			const locY = getTweenedValue(node, 'location_y', currentTime);
			const sizeW = getTweenedValue(node, 'size_width', currentTime);
			const sizeH = getTweenedValue(node, 'size_height', currentTime);

			const cx = locX + sizeW - leftOffset - TIMER_CIRCLE_RADIUS;
			const cy = locY + sizeH / 2;

			if (isTarget) {
				ctx.beginPath();
				ctx.arc(cx, cy, TIMER_CIRCLE_RADIUS * 3, 0, 2 * Math.PI);
				ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
				ctx.fill();
				ctx.closePath();
			}

			ctx.beginPath();
			ctx.fillStyle = colorStr;
			if (node.letter === '$') {
				const sq = 17;
				ctx.fillRect(cx - sq / 2, cy - sq / 2, sq, sq);
			} else {
				ctx.font = `${timerFontSize}px verdana, helvetica, sans-serif`;
				ctx.textAlign = 'center';
				ctx.textBaseline = 'middle';
				ctx.fillText(node.letter, cx, cy);
			}

			if (developerVisualizer) {
				ctx.font = `${timerFontSize / 3}px verdana, helvetica, sans-serif`;
				ctx.fillText(`l:${node.likelihood?.toFixed(2) ?? '0'}`, cx, cy + timerFontSize / 3);
				ctx.fillText(`p:${node.prior?.toFixed(2) ?? '0'}`, cx, cy + (2 * timerFontSize) / 3);
				ctx.fillText(`z:${node.post_Z?.toFixed(2) ?? '0'}`, cx, cy + timerFontSize);
			}
			ctx.closePath();

			// Timer arc
			ctx.beginPath();
			ctx.arc(cx, cy, timerRadius, 0, 2 * Math.PI * timerFrac);
			ctx.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${timerFrac * 0.9 + 0.1})`;
			ctx.lineWidth = isTarget ? TIMER_CIRCLE_WIDTH * 2 : TIMER_CIRCLE_WIDTH;
			ctx.stroke();
			ctx.closePath();
		});

		animationFrameRef.current = requestAnimationFrame(draw);
	}, [
		ctx,
		visibleNodes,
		devicePixelRatio,
		leftOffset,
		targetPhrase,
		useVisualTutor,
		developerVisualizer,
		showBoxes,
		likelihoodModel,
		getTweenedValue,
	]);

	useEffect(() => {
		if (ctx) draw();
		return () => {
			if (animationFrameRef.current !== null) {
				cancelAnimationFrame(animationFrameRef.current);
			}
		};
	}, [draw, ctx]);

	// ------------------------------------------------------------------
	// Render
	// ------------------------------------------------------------------

	return (
		<div className="flex flex-col h-full relative box-border">
			<canvas ref={canvasRef} className="h-full w-full bg-black" />

			<div className="absolute top-4 right-6 flex gap-8 text-white text-2xl">
				<CheckboxLabel
					label="Blink to click"
					checked={blinkToClick}
					onChange={setBlinkToClick}
				/>
				<CheckboxLabel
					label="Debug"
					checked={developerVisualizer}
					onChange={setDeveloperVisualizer}
				/>
				<CheckboxLabel
					label="Click sound"
					checked={playClickSoundEnabled}
					onChange={setPlayClickSoundEnabled}
				/>
				<CheckboxLabel
					label="Boxes"
					checked={showBoxes}
					onChange={setShowBoxes}
				/>
				<CheckboxLabel
					label="Tutor"
					checked={useVisualTutor}
					onChange={setUseVisualTutor}
				/>
				<CheckboxLabel
					label="Multiline"
					checked={multilineMode}
					onChange={setMultilineMode}
				/>
			</div>

			{blinkToClick && <Eye onBlink={handleBlink} />}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Small helper component
// ---------------------------------------------------------------------------

interface CheckboxLabelProps {
	label: string;
	checked: boolean;
	onChange: (value: boolean) => void;
}

function CheckboxLabel({ label, checked, onChange }: CheckboxLabelProps) {
	return (
		<label className="flex items-center gap-3">
			<input
				type="checkbox"
				checked={checked}
				onChange={(e) => onChange(e.target.checked)}
				className="w-6 h-6"
			/>
			{label}
		</label>
	);
}

export default TrieVisualizer;
