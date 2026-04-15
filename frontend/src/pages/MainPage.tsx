import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import initBayesianWasm, {
	BayesianSession,
	debugPanicTest,
	initPanicHook,
	optimizeTimerPhases,
} from '../wasm_pkg/bayesian';
import practicePhrasesText from './practice-phrases.txt?raw';
import CalibrationSettings, {
	type LikelihoodModel,
	type AutoCalibrationState,
	type VariationalParams,
	type CalibrationPair,
} from '../components/CalibrationSettings';
import Eye from '../components/Eye';
import TrieSnapshotVisualizer, {
	SCROLL_CENTERING_WEIGHT,
	SCROLL_STABILITY_WEIGHT,
	computeScrollLayoutState,
	findTutorTargetKey,
} from '../components/TrieSnapshotVisualizer';
import type {
	ExpandedSnapshot,
	VisibleNodeTimerMap,
} from '../components/TrieSnapshotVisualizer';
import { jStat } from 'jstat';

const DEFAULT_PERIOD = 1.1;
const APP_USERNAME_STORAGE_KEY = 'dotter-app-username';

function predictiveStddev(muS: number, sigmaS: number, sigmaM: number): number {
	return Math.sqrt(Math.exp(muS + (sigmaS ** 2) / 2) + sigmaM ** 2);
}

function variationalParamsToLikelihoodModel(params: VariationalParams, period: number): LikelihoodModel {
	const alpha = Math.exp(params.log_alpha);
	const beta = Math.exp(params.log_beta);
	return {
		mu_delay: params.mu_m,
		stddev_delay: predictiveStddev(params.mu_s, params.sigma_s, params.sigma_m),
		outliers: jStat.beta.inv(0.5, alpha, beta),
		period,
		intervals: {
			mu_delay: [params.mu_m - 1.96 * params.sigma_m, params.mu_m + 1.96 * params.sigma_m],
			stddev_delay: [
				predictiveStddev(params.mu_s - 1.96 * params.sigma_s, params.sigma_s, params.sigma_m),
				predictiveStddev(params.mu_s + 1.96 * params.sigma_s, params.sigma_s, params.sigma_m),
			],
			outliers: [jStat.beta.inv(0.025, alpha, beta), jStat.beta.inv(0.975, alpha, beta)],
		},
	};
}

const DEFAULT_LIKELIHOOD_MODEL: LikelihoodModel = {
	mu_delay: 0.0,
	stddev_delay: 0.064,
	outliers: 0.08,
	period: DEFAULT_PERIOD,
};
const N_SKIP_PRACTICE_PHRASES = 6;
function formatPracticePhrase(phrase: string): string {
	return ` ${phrase}$`;
}
const PRACTICE_PHRASES = practicePhrasesText
	.split('\n')
	.map((line) => line.trim())
	.filter((line) => line.length > 0)
	.map(formatPracticePhrase);

const RECENT_CONSOLE_ERROR_LIMIT = 20;
let wasmPanicConsoleCaptureInstalled = false;
let recentConsoleErrors: string[] = [];

function logaddexp(a: number, b: number): number {
	if (a === -Infinity) return b;
	if (b === -Infinity) return a;
	if (a > b) return a + Math.log(1 + Math.exp(b - a));
	return b + Math.log(1 + Math.exp(a - b));
}

function normalLogpdf(x: number, mean: number, stddev: number): number {
	return -0.5 * Math.pow((x - mean) / stddev, 2) - Math.log(stddev * Math.sqrt(2 * Math.PI));
}

function timerLikelihood(time: number, phase: number, model: LikelihoodModel): number {
	let x = time - phase;
	x = ((x % model.period) + model.period) % model.period;
	const outlierProb = Math.log(model.outliers) - Math.log(model.period);
	
	const notOutlierProb = Math.log(1 - model.outliers);
	
	const normalModes = [-1, 0, 1].map(k => {
		return normalLogpdf(x, model.mu_delay + k * model.period, model.stddev_delay);
	});
	
	let sumNormalModes = normalModes[0];
	for (let i = 1; i < normalModes.length; i++) {
		sumNormalModes = logaddexp(sumNormalModes, normalModes[i]);
	}
	
	return logaddexp(
		outlierProb,
		notOutlierProb + sumNormalModes
	);
}

// Returns effective linear-probability weight for each rendered node.
// z is in log space, so we exponentiate first, then subtract the linear
// probability of each directly-rendered child. The residual is the probability
// mass that "stops" at this node rather than passing through to a visible
// descendant.
// Returns effective linear-probability weight for each key in `keys` (all must
// be present in snapshot). Each node's weight is exp(z - rootZ) minus the sum
// of exp(z - rootZ) of its directly-rendered children, i.e. the probability
// mass that "stops" at this node rather than passing through to a visible
// descendant. Subtracting rootZ before exponentiating avoids overflow/underflow
// since z values are log-probabilities that can be large in magnitude.
function effectiveWeights(
	snapshot: ExpandedSnapshot,
	keys: readonly string[],
): number[] {
	const rootZ = snapshot['^']?.z ?? 0;
	const expZ: Record<string, number> = {};
	for (const key of keys) expZ[key] = Math.exp(snapshot[key].z - rootZ);
	const linearZ = { ...expZ };
	for (const key of keys) {
		// Find the closest rendered ancestor = longest rendered proper prefix of key
		let parent = '';
		for (const candidate of keys) {
			if (candidate !== key && key.startsWith(candidate) && candidate.length > parent.length)
				parent = candidate;
		}
		if (parent) linearZ[parent] = Math.max(0, linearZ[parent] - expZ[key]);
	}
	return keys.map(k => Math.max(0, linearZ[k]));
}

function timersForSnapshot(
	snapshot: ExpandedSnapshot,
	model: LikelihoodModel,
	existingTimers: VisibleNodeTimerMap,
	resetAll: boolean,
	renderedNodeKeys: readonly string[],
): VisibleNodeTimerMap {
	const nextTimers: VisibleNodeTimerMap = {};
	const keysInSnapshot = renderedNodeKeys.filter(k => k in snapshot);

	// Preserve timers for nodes that haven't been reset
	if (!resetAll) {
		for (const key of keysInSnapshot) {
			if (existingTimers[key]) nextTimers[key] = existingTimers[key];
		}
		// New nodes created by prior updates inherit the phase of their nearest
		// visible ancestor so expansions preserve local timing structure.
		const newKeys = keysInSnapshot
			.filter(k => !nextTimers[k])
			.sort((a, b) => a.length - b.length || a.localeCompare(b));
		for (const key of newKeys) {
			let parent = '';
			for (const candidate of keysInSnapshot) {
				if (candidate !== key && key.startsWith(candidate) && candidate.length > parent.length) {
					parent = candidate;
				}
			}
			nextTimers[key] = { phase: parent ? nextTimers[parent].phase : 0.5 * model.period };
		}
		return nextTimers;
	}

	// Full reset: compute optimized phases via timer_spacing WASM.
	// Sort lexicographically so the optimizer sees a consistent canonical ordering.
	const sortedKeys = [...keysInSnapshot].sort();
	const weights = effectiveWeights(snapshot, sortedKeys);
	const weightsJson = JSON.stringify(weights);
	const phasesJson = optimizeTimerPhases(weightsJson, model.stddev_delay, model.period);
	const phases: number[] = JSON.parse(phasesJson);

	for (let i = 0; i < sortedKeys.length; i++) {
		nextTimers[sortedKeys[i]] = { phase: phases[i] ?? (i + 0.5) * model.period / sortedKeys.length };
	}
	return nextTimers;
}

function formatStepError(step: string, err: unknown): string {
	const message = err instanceof Error ? err.message : String(err);
	console.error(`[MainPage] ${step}`, err);
	return `${step}: ${message}`;
}

function stringifyConsoleArg(arg: unknown): string {
	if (arg instanceof Error) {
		return arg.stack ? `${arg.message}\n${arg.stack}` : arg.message;
	}
	if (typeof arg === 'string') {
		return arg;
	}
	try {
		return JSON.stringify(arg);
	} catch {
		return String(arg);
	}
}

function installWasmPanicConsoleCapture(): void {
	if (wasmPanicConsoleCaptureInstalled) {
		return;
	}
	const originalConsoleError = console.error.bind(console);
	console.error = (...args: unknown[]) => {
		const text = args.map(stringifyConsoleArg).join(' ');
		recentConsoleErrors.push(text);
		if (recentConsoleErrors.length > RECENT_CONSOLE_ERROR_LIMIT) {
			recentConsoleErrors = recentConsoleErrors.slice(-RECENT_CONSOLE_ERROR_LIMIT);
		}
		originalConsoleError(...args);
	};
	wasmPanicConsoleCaptureInstalled = true;
}

function latestRustPanicConsoleMessage(): string | null {
	for (let i = recentConsoleErrors.length - 1; i >= 0; i -= 1) {
		const entry = recentConsoleErrors[i];
		if (entry.includes('panicked at') || entry.includes('console_error_panic_hook')) {
			return entry;
		}
	}
	return null;
}

function detailedWasmTrapMessage(err: unknown): string {
	const header =
		err instanceof Error && err.message
			? `BayesianSession trapped in wasm after ${err.message}`
			: `BayesianSession trapped in wasm after ${String(err)}`;
	const jsStack =
		err instanceof Error && err.stack && !err.stack.startsWith(err.message)
			? err.stack
			: null;
	const rustPanic = latestRustPanicConsoleMessage();
	const parts = [header];
	if (rustPanic) {
		parts.push(`Rust panic:\n${rustPanic}`);
	}
	if (jsStack) {
		parts.push(`JS stack:\n${jsStack}`);
	}
	parts.push('Press Reset to recover.');
	return parts.join('\n\n');
}

function isWasmTrapError(err: unknown): boolean {
	const message = err instanceof Error ? err.message : String(err);
	return message.includes('unreachable');
}

interface PredictionLogEntry {
	id: number;
	fullString: string;
	finalTokenLexindex: number;
	receivedAt: string;
}

interface SessionDebugDump {
	update_json_log: Array<{
		kind: 'likelihood' | 'prior' | string;
		json: string;
	}>;
}

const APP_THEME_STORAGE_KEY = 'dotter-app-theme';
const APP_BLINK_TO_CLICK_STORAGE_KEY = 'dotter-app-blink-to-click';

function readStoredBlinkToClick(): boolean {
	try {
		return localStorage.getItem(APP_BLINK_TO_CLICK_STORAGE_KEY) === 'true';
	} catch {
		return false;
	}
}

function readStoredUsername(): string {
	try {
		return localStorage.getItem(APP_USERNAME_STORAGE_KEY) ?? '';
	} catch {
		return '';
	}
}

function readStoredColorMode(): 'light' | 'dark' {
	try {
		const raw = localStorage.getItem(APP_THEME_STORAGE_KEY);
		if (raw === 'light' || raw === 'dark') {
			return raw;
		}
	} catch {
		// ignore
	}
	return 'dark';
}

function randomPracticePhrase(excluding?: string): string {
	const eligiblePhrases = PRACTICE_PHRASES.slice(N_SKIP_PRACTICE_PHRASES);
	const source = eligiblePhrases.length > 0 ? eligiblePhrases : PRACTICE_PHRASES;
	const candidates =
		source.length > 1 && excluding
			? source.filter((phrase) => phrase !== excluding)
			: source;
	return candidates[Math.floor(Math.random() * candidates.length)];
}

function downloadTextFile(filename: string, content: string, mimeType: string): void {
	const blob = new Blob([content], { type: mimeType });
	const url = URL.createObjectURL(blob);
	const anchor = document.createElement('a');
	anchor.href = url;
	anchor.download = filename;
	document.body.appendChild(anchor);
	anchor.click();
	anchor.remove();
	URL.revokeObjectURL(url);
}

function moduloDelay(timeSeconds: number, phase: number, period: number): number {
	let x = timeSeconds - phase;
	x = ((x % period) + period) % period;
	return x;
}

function playTutorTone(
	audioContextRef: React.MutableRefObject<AudioContext | null>,
	frequencyHz: number,
	repetitions: number,
	options?: {
		type?: OscillatorType;
		peakGain?: number;
		duration?: number;
		gap?: number;
	}
): void {
	const AudioContextCtor = window.AudioContext ?? (window as typeof window & {
		webkitAudioContext?: typeof AudioContext;
	}).webkitAudioContext;
	if (!AudioContextCtor) {
		return;
	}
	const ctx = audioContextRef.current ?? new AudioContextCtor();
	audioContextRef.current = ctx;
	void ctx.resume().catch(() => {});
	const startAt = ctx.currentTime + 0.005;
	const duration = options?.duration ?? 0.07;
	const gap = options?.gap ?? 0.05;
	const peakGain = options?.peakGain ?? 0.3;
	const oscType = options?.type ?? 'sine';
	for (let i = 0; i < repetitions; i += 1) {
		const osc = ctx.createOscillator();
		const gain = ctx.createGain();
		const t0 = startAt + i * (duration + gap);
		osc.type = oscType;
		osc.frequency.setValueAtTime(frequencyHz, t0);
		gain.gain.setValueAtTime(0.0001, t0);
		gain.gain.exponentialRampToValueAtTime(peakGain, t0 + 0.008);
		gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
		osc.connect(gain);
		gain.connect(ctx.destination);
		osc.start(t0);
		osc.stop(t0 + duration);
	}
}

function playTutorOutlierTone(
	audioContextRef: React.MutableRefObject<AudioContext | null>,
): void {
	const AudioContextCtor = window.AudioContext ?? (window as typeof window & {
		webkitAudioContext?: typeof AudioContext;
	}).webkitAudioContext;
	if (!AudioContextCtor) {
		return;
	}
	const ctx = audioContextRef.current ?? new AudioContextCtor();
	audioContextRef.current = ctx;
	void ctx.resume().catch(() => {});
	const startAt = ctx.currentTime + 0.005;
	const duration = 0.04;
	const gap = 0.03;
	for (const [idx, frequencyHz] of [520, 610].entries()) {
		const osc = ctx.createOscillator();
		const gain = ctx.createGain();
		const t0 = startAt + idx * (duration + gap);
		osc.type = 'square';
		osc.frequency.setValueAtTime(frequencyHz, t0);
		gain.gain.setValueAtTime(0.0001, t0);
		gain.gain.exponentialRampToValueAtTime(0.32, t0 + 0.006);
		gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
		osc.connect(gain);
		gain.connect(ctx.destination);
		osc.start(t0);
		osc.stop(t0 + duration);
	}
}

function MainPage() {
	const [snapshot, setSnapshot] = useState<ExpandedSnapshot | null>(null);
	const [timers, setTimers] = useState<VisibleNodeTimerMap>({});
	const [error, setError] = useState<string | null>(null);
	const [warning, setWarning] = useState<string | null>(null);
	const [predictionLog, setPredictionLog] = useState<PredictionLogEntry[]>([]);
	const [loading, setLoading] = useState(true);
	const [wsStatus, setWsStatus] = useState('Connecting...');
	const [wasmReady, setWasmReady] = useState(false);
	const [lastBatchSize, setLastBatchSize] = useState(0);
	const [likelihoodModel, setLikelihoodModel] = useState<LikelihoodModel>({
		...DEFAULT_LIKELIHOOD_MODEL,
	});
	const [useAutomaticCalibration, setUseAutomaticCalibration] = useState<AutoCalibrationState>({
		mu_delay: true,
		stddev_delay: true,
		outliers: true,
	});
	const [autoCalibrationLikelihoodModel, setAutoCalibrationLikelihoodModel] = useState<LikelihoodModel>(DEFAULT_LIKELIHOOD_MODEL);
	const [calibrationSampleCount, setCalibrationSampleCount] = useState(0);
	const [rawVariationalParams, setRawVariationalParams] = useState<VariationalParams | null>(null);
	const [recentCalibrationPairs, setRecentCalibrationPairs] = useState<CalibrationPair[]>([]);
	const [usernameInput, setUsernameInput] = useState(() => readStoredUsername());
	const [activeUsername, setActiveUsername] = useState<string | null>(null);
	const [currentViBefore, setCurrentViBefore] = useState<VariationalParams | null>(null);
	const [showPredictionLogPanel, setShowPredictionLogPanel] = useState(false);
	const [showCalibrationDebugPanel, setShowCalibrationDebugPanel] = useState(false);
	const sessionRef = useRef<BayesianSession | null>(null);
	const socketRef = useRef<WebSocket | null>(null);
	const likelihoodModelRef = useRef(likelihoodModel);
	const audioContextRef = useRef<AudioContext | null>(null);
	const bootstrapPromiseRef = useRef<Promise<void> | null>(null);
	const predictionLogIdRef = useRef(0);
	const sessionOpQueueRef = useRef<Promise<unknown>>(Promise.resolve());
	const sessionOpInFlightRef = useRef(false);
	const sessionTrappedRef = useRef(false);
	const sessionTrapMessageRef = useRef<string | null>(null);
	const expansionThresholdRef = useRef<number>(Number.NEGATIVE_INFINITY);
	const [colorMode, setColorMode] = useState<'light' | 'dark'>(readStoredColorMode);
	const [showBoxes, setShowBoxes] = useState(true);
	const [showDebugStats, setShowDebugStats] = useState(false);
	const [showAll, setShowAll] = useState(false);
	const [blinkToClick, setBlinkToClick] = useState(readStoredBlinkToClick);
	const [showPracticePhrase, setShowPracticePhrase] = useState(false);
	const [useVisualTutor, setUseVisualTutor] = useState(false);
	const [useAudioTutor, setUseAudioTutor] = useState(false);
	const [practicePhrase, setPracticePhrase] = useState(() => randomPracticePhrase());
	const [expansionThreshold, setExpansionThreshold] = useState<number>(Number.NEGATIVE_INFINITY);
	const [scrollOffset, setScrollOffset] = useState(0);
	const [scrollRoot, setScrollRoot] = useState('^');
	const [firstForkDepth, setFirstForkDepth] = useState<number | null>(null);
	const scrollOffsetRef = useRef(0);
	const scrollRootRef = useRef('^');
	const scrollAncestorKeysRef = useRef<string[]>([]);

	const tutorTargetKey = useMemo(() => {
		if (!showPracticePhrase || !snapshot) {
			return null;
		}
		return findTutorTargetKey(
			snapshot,
			expansionThreshold,
			scrollRoot,
			showAll,
			practicePhrase,
		);
	}, [expansionThreshold, practicePhrase, scrollRoot, showAll, showPracticePhrase, snapshot]);

	useEffect(() => {
		try {
			localStorage.setItem(APP_THEME_STORAGE_KEY, colorMode);
		} catch {
			// ignore
		}
	}, [colorMode]);

	useEffect(() => {
		try {
			localStorage.setItem(APP_BLINK_TO_CLICK_STORAGE_KEY, blinkToClick ? 'true' : 'false');
		} catch {
			// ignore
		}
	}, [blinkToClick]);

	useEffect(() => {
		try {
			localStorage.setItem(APP_USERNAME_STORAGE_KEY, usernameInput);
		} catch {
			// ignore
		}
	}, [usernameInput]);

	useEffect(() => {
		likelihoodModelRef.current = likelihoodModel;
	}, [likelihoodModel]);

	useEffect(() => {
		if (!showPracticePhrase) {
			if (useVisualTutor) {
				setUseVisualTutor(false);
			}
			if (useAudioTutor) {
				setUseAudioTutor(false);
			}
		}
	}, [showPracticePhrase, useAudioTutor, useVisualTutor]);

	const shufflePracticePhrase = useCallback(() => {
		setPracticePhrase((current) => randomPracticePhrase(current));
	}, []);

	const applySnapshot = useCallback((nextSnapshot: ExpandedSnapshot, resetAllTimers: boolean) => {
		const nextScrollLayout = computeScrollLayoutState(
			nextSnapshot,
			expansionThresholdRef.current,
			scrollOffsetRef.current,
		);
		scrollOffsetRef.current = nextScrollLayout.scrollOffset;
		scrollRootRef.current = nextScrollLayout.scrollRoot;
		scrollAncestorKeysRef.current = nextScrollLayout.scrollAncestorKeys;
		setScrollOffset(nextScrollLayout.scrollOffset);
		setScrollRoot(nextScrollLayout.scrollRoot);
		setFirstForkDepth(nextScrollLayout.firstForkDepth);
		setSnapshot(nextSnapshot);
		setTimers((currentTimers) =>
			timersForSnapshot(
				nextSnapshot,
				likelihoodModelRef.current,
				currentTimers,
				resetAllTimers,
				nextScrollLayout.renderedNodeKeys,
			),
		);
	}, []);

	const enqueueSessionOp = useCallback(<T,>(op: () => Promise<T> | T): Promise<T> => {
		const run = async (): Promise<T> => {
			if (sessionTrappedRef.current) {
				throw new Error(
					sessionTrapMessageRef.current ??
						'BayesianSession trapped in wasm; press Reset to recover',
				);
			}
			if (sessionOpInFlightRef.current) {
				throw new Error('BayesianSession operation re-entered before the previous one finished');
			}
			sessionOpInFlightRef.current = true;
			try {
				return await op();
			} catch (err) {
				if (isWasmTrapError(err)) {
					sessionTrappedRef.current = true;
					sessionTrapMessageRef.current = detailedWasmTrapMessage(err);
					throw new Error(sessionTrapMessageRef.current);
				}
				throw err;
			} finally {
				sessionOpInFlightRef.current = false;
			}
		};
		const result = sessionOpQueueRef.current.then(run, run) as Promise<T>;
		sessionOpQueueRef.current = result.then(
			() => undefined,
			() => undefined,
		);
		return result;
	}, []);

	const refreshSnapshot = useCallback(async (resetAllTimers = false) => {
		const snapshotJson = await enqueueSessionOp(() => {
			const session = sessionRef.current;
			if (!session) {
				throw new Error('BayesianSession is not initialized');
			}
			try {
				return session.expand_to_threshold();
			} catch (err) {
				throw new Error(formatStepError('expand_to_threshold failed', err));
			}
		});
		applySnapshot(JSON.parse(snapshotJson) as ExpandedSnapshot, resetAllTimers);
	}, [applySnapshot, enqueueSessionOp]);

	const recordPredictionLog = useCallback((contentJson: string) => {
		const payload = JSON.parse(contentJson) as {
			full_string: string;
			final_token_lexindex: number;
		};
		const nextEntry: PredictionLogEntry = {
			id: predictionLogIdRef.current++,
			fullString: payload.full_string,
			finalTokenLexindex: payload.final_token_lexindex,
			receivedAt: new Date().toLocaleTimeString(),
		};
		setPredictionLog((current) => [nextEntry, ...current]);
	}, []);

	const downloadSessionDebugDump = useCallback(() => {
		void (async () => {
			try {
				const dumpJson = await enqueueSessionOp(() => {
					const session = sessionRef.current;
					if (!session) {
						throw new Error('BayesianSession is not initialized');
					}
					try {
						return session.debug_dump_json();
					} catch (err) {
						throw new Error(formatStepError('debug_dump_json failed', err));
					}
				});
				const parsed = JSON.parse(dumpJson) as SessionDebugDump;
				const now = new Date().toISOString().replace(/:/g, '-');
				const filename = `bayesian-session-dump-${now}.json`;
				downloadTextFile(filename, JSON.stringify(parsed, null, 2), 'application/json');
				setError(null);
			} catch (err) {
				setError(err instanceof Error ? err.message : String(err));
			}
		})();
	}, [enqueueSessionOp]);

	const applyViBeforeToUi = useCallback((viBefore: VariationalParams) => {
		const nextModel = variationalParamsToLikelihoodModel(viBefore, likelihoodModelRef.current.period);
		setCurrentViBefore(viBefore);
		setCalibrationSampleCount(0);
		setRawVariationalParams(viBefore);
		setRecentCalibrationPairs([]);
		setAutoCalibrationLikelihoodModel(nextModel);
	}, []);

	const startLocalString = useCallback(async (viBefore: VariationalParams) => {
		const snapshotJson = await enqueueSessionOp(() => {
			if (sessionTrappedRef.current || !sessionRef.current) {
				try {
					sessionRef.current = new BayesianSession();
					const threshold = sessionRef.current.expansion_threshold();
					expansionThresholdRef.current = threshold;
					setExpansionThreshold(threshold);
					sessionTrappedRef.current = false;
					sessionTrapMessageRef.current = null;
				} catch (err) {
					throw new Error(formatStepError('BayesianSession constructor failed during reset', err));
				}
			}
			const session = sessionRef.current;
			if (!session) {
				throw new Error('BayesianSession is not initialized after reset');
			}
			session.reset();
			session.set_current_prior_json(JSON.stringify(viBefore));
			try {
				return session.expand_to_threshold();
			} catch (err) {
				throw new Error(formatStepError('expand_to_threshold after reset failed', err));
			}
		});
		applySnapshot(JSON.parse(snapshotJson) as ExpandedSnapshot, true);
		applyViBeforeToUi(viBefore);
	}, [applySnapshot, applyViBeforeToUi, enqueueSessionOp]);

	const startSessionOnBackend = useCallback(async (username: string) => {
		const ws = socketRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) {
			throw new Error('WebSocket must be connected before starting a session');
		}
		const trimmed = username.trim();
		if (!trimmed) {
			throw new Error('Username must be non-empty');
		}
		setActiveUsername(null);
		setPredictionLog([]);
		setCurrentViBefore(null);
		setCalibrationSampleCount(0);
		setRawVariationalParams(null);
		setRecentCalibrationPairs([]);
		ws.send(JSON.stringify({ type: 'start_session', content: { username: trimmed } }));
	}, []);

	const resetBothSides = useCallback(async () => {
		const ws = socketRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) {
			throw new Error('WebSocket must be connected before resetting');
		}
		if (!currentViBefore) {
			throw new Error('Session has no current calibration prior; start a session first');
		}
		const recalibrationJson = await enqueueSessionOp(() => {
			const session = sessionRef.current;
			if (!session) {
				throw new Error('BayesianSession is not initialized');
			}
			try {
				return session.recalibrate(JSON.stringify(currentViBefore), true);
			} catch (err) {
				throw new Error(formatStepError('recalibrate before reset failed', err));
			}
		});
		const recalibrationResult = JSON.parse(recalibrationJson) as {
			prior_params: VariationalParams;
		};
		await startLocalString(recalibrationResult.prior_params);
		ws.send(JSON.stringify({ type: 'reset' }));
	}, [currentViBefore, enqueueSessionOp, startLocalString]);

	useEffect(() => {
		let cancelled = false;

		async function loadSession() {
			try {
				setLoading(true);
				setError(null);
				if (!bootstrapPromiseRef.current) {
					try {
						bootstrapPromiseRef.current = (async () => {
							try {
								await initBayesianWasm();
								installWasmPanicConsoleCapture();
								initPanicHook();
								if (
									import.meta.env.DEV &&
									new URLSearchParams(window.location.search).get('wasmPanic') === '1'
								) {
									debugPanicTest();
								}
							} catch (err) {
								throw new Error(formatStepError('initBayesianWasm failed', err));
							}
							if (!sessionRef.current) {
								try {
									sessionRef.current = new BayesianSession();
									const threshold = sessionRef.current.expansion_threshold();
									expansionThresholdRef.current = threshold;
									setExpansionThreshold(threshold);
								} catch (err) {
									throw new Error(formatStepError('BayesianSession constructor failed', err));
								}
							} else {
								const threshold = sessionRef.current.expansion_threshold();
								expansionThresholdRef.current = threshold;
								setExpansionThreshold(threshold);
							}
							try {
								await refreshSnapshot(true);
							} catch (err) {
								throw new Error(formatStepError('initial expand_to_threshold failed', err));
							}
						})();
					} catch (err) {
						bootstrapPromiseRef.current = null;
						throw err;
					}
				}
				await bootstrapPromiseRef.current;
				if (!cancelled) {
					setWasmReady(true);
				}
			} catch (err) {
				bootstrapPromiseRef.current = null;
				if (!cancelled) {
					setError(err instanceof Error ? err.message : String(err));
				}
			} finally {
				if (!cancelled) {
					setLoading(false);
				}
			}
		}

		void loadSession();
		return () => {
			cancelled = true;
		};
	}, [refreshSnapshot]);

	useEffect(() => {
		if (!wasmReady) {
			return;
		}

		const ws = new WebSocket('ws://localhost:8000/ws');
		socketRef.current = ws;

		ws.addEventListener('open', () => {
			setWsStatus('Connected');
			setWarning(null);
			setPredictionLog([]);
			const trimmed = usernameInput.trim();
			if (trimmed) {
				ws.send(JSON.stringify({ type: 'start_session', content: { username: trimmed } }));
			}
		});

		ws.addEventListener('close', () => {
			setWsStatus('Disconnected');
			setWarning('Backend disconnected. Local likelihood updates still apply.');
		});
		ws.addEventListener('error', () => {
			setWsStatus('Error');
			setWarning('Backend connection failed. Local likelihood updates still apply.');
		});

		ws.addEventListener('message', (event) => {
			void (async () => {
				const message = JSON.parse(event.data as string) as {
					type: string;
					content_json?: string;
					content?: {
						message?: string;
						username?: string;
						variational_params?: VariationalParams;
					};
				};

				if (message.type === 'reset_ack') {
					setError(null);
					setPredictionLog([]);
					return;
				}

				if (
					message.type === 'session_started' &&
					typeof message.content?.username === 'string' &&
					message.content.variational_params
				) {
					setActiveUsername(message.content.username);
					await startLocalString(message.content.variational_params);
					ws.send(JSON.stringify({ type: 'request_next_prior' }));
					setError(null);
					setPredictionLog([]);
					return;
				}

				if (message.type === 'prior_update' && typeof message.content_json === 'string') {
					if (sessionTrappedRef.current) {
						return;
					}
					const contentJson = message.content_json;
					recordPredictionLog(contentJson);
					const snapshotJson = await enqueueSessionOp(() => {
						const session = sessionRef.current;
						if (!session) {
							throw new Error('BayesianSession is not initialized');
						}
						try {
							session.receive_prior_update(contentJson);
						} catch (err) {
							throw new Error(formatStepError('receive_prior_update failed', err));
						}
						try {
							session.apply_updates();
						} catch (err) {
							throw new Error(formatStepError('apply_updates after prior_update failed', err));
						}
						try {
							return session.expand_to_threshold();
						} catch (err) {
							throw new Error(formatStepError('expand_to_threshold after prior_update failed', err));
						}
					});
					applySnapshot(JSON.parse(snapshotJson) as ExpandedSnapshot, false);
					return;
				}

				if (message.type === 'error' && message.content?.message) {
					setError(message.content.message);
				}
			})().catch((err) => {
				setError(err instanceof Error ? err.message : String(err));
			});
		});

		return () => {
			socketRef.current = null;
			ws.close();
		};
	}, [applySnapshot, enqueueSessionOp, recordPredictionLog, startLocalString, usernameInput, wasmReady]);

	const runLikelihoodPulse = useCallback(
		(timeSeconds: number) => {
			void (async () => {
				if (!snapshot) {
					return;
				}
				if (!currentViBefore) {
					setError('Start a session before sending likelihood updates');
					return;
				}
				if (sessionTrappedRef.current) {
					setError(
						sessionTrapMessageRef.current ??
							'BayesianSession trapped in wasm; press Reset to recover',
					);
					return;
				}

				if (showPracticePhrase && useAudioTutor && tutorTargetKey) {
					const targetTimer = timers[tutorTargetKey];
					const predictiveStddev = likelihoodModelRef.current.stddev_delay;
					if (targetTimer && predictiveStddev > 0) {
						const x = moduloDelay(timeSeconds, targetTimer.phase, likelihoodModelRef.current.period);
						const offsetStddevs = (x - likelihoodModelRef.current.mu_delay) / predictiveStddev;
						if (Math.abs(offsetStddevs) > 3) {
							playTutorOutlierTone(audioContextRef);
						} else if (offsetStddevs > 2) {
							playTutorTone(audioContextRef, 1760, 2);
						} else if (offsetStddevs > 1) {
							playTutorTone(audioContextRef, 1760, 1);
						} else if (offsetStddevs < -2) {
							playTutorTone(audioContextRef, 330, 2);
						} else if (offsetStddevs < -1) {
							playTutorTone(audioContextRef, 330, 1);
						} else {
							playTutorTone(audioContextRef, 660, 1);
						}
					}
				}

				const nodes: Record<string, { l: number, phase: number }> = {};
				for (const [fullString, timer] of Object.entries(timers)) {
					if (!(fullString in snapshot)) {
						continue;
					}
					const likelihood = timerLikelihood(timeSeconds, timer.phase, likelihoodModelRef.current);
					nodes[fullString] = { l: likelihood, phase: timer.phase };
					if (fullString === scrollRootRef.current) {
						for (const ancestorKey of scrollAncestorKeysRef.current) {
							if (!(ancestorKey in snapshot)) {
								continue;
							}
							nodes[ancestorKey] = { l: likelihood, phase: timer.phase };
						}
					}
				}
				const period = likelihoodModelRef.current.period;
				const likelihoodPayload = {
					period,
					y: timeSeconds,
					nodes
				};
				const likelihoodJson = JSON.stringify(likelihoodPayload);
				const snapshotJson = await enqueueSessionOp(() => {
					const session = sessionRef.current;
					if (!session) {
						throw new Error('BayesianSession is not initialized');
					}
					try {
						session.receive_likelihood_update(likelihoodJson);
					} catch (err) {
						throw new Error(formatStepError('receive_likelihood_update failed', err));
					}
					try {
						session.apply_updates();
					} catch (err) {
						throw new Error(formatStepError('apply_updates after likelihood_update failed', err));
					}
					try {
						return session.expand_to_threshold();
					} catch (err) {
						throw new Error(formatStepError('expand_to_threshold after likelihood_update failed', err));
					}
				});
				applySnapshot(JSON.parse(snapshotJson) as ExpandedSnapshot, true);
				const ws = socketRef.current;
				if (ws && ws.readyState === WebSocket.OPEN) {
					ws.send(JSON.stringify({ type: 'likelihood_update', content_json: likelihoodJson }));
					setWarning(null);
				} else {
					setWarning('Backend disconnected. Applied likelihoods locally only.');
				}
				const metricsJson = await enqueueSessionOp(() => {
					const session = sessionRef.current;
					if (!session) {
						throw new Error('BayesianSession is not initialized');
					}
					try {
						return session.recalibrate(JSON.stringify(currentViBefore), true);
					} catch (err) {
						throw new Error(formatStepError('recalibrate failed', err));
					}
				});
				const recalibrationResult = JSON.parse(metricsJson) as {
					prior_params: VariationalParams;
					used_likelihood_updates: number;
					recent_pairs: CalibrationPair[];
				};
				const priorParams = recalibrationResult.prior_params;
				setCalibrationSampleCount(recalibrationResult.used_likelihood_updates);
				setRawVariationalParams(priorParams);
				setRecentCalibrationPairs(recalibrationResult.recent_pairs);
				setAutoCalibrationLikelihoodModel(variationalParamsToLikelihoodModel(priorParams, period));
				setLastBatchSize(Object.keys(nodes).length);
				setError(null);
			})().catch((err) => {
				setError(err instanceof Error ? err.message : String(err));
			});
		},
		[applySnapshot, currentViBefore, enqueueSessionOp, showPracticePhrase, snapshot, timers, tutorTargetKey, useAudioTutor],
	);

	useEffect(() => {
		if (!wasmReady) {
			return;
		}

		const handleKeyDown = (event: KeyboardEvent) => {
			const activeElement = document.activeElement;
			if (
				activeElement instanceof HTMLInputElement ||
				activeElement instanceof HTMLTextAreaElement ||
				activeElement instanceof HTMLSelectElement ||
				activeElement?.getAttribute('contenteditable') === 'true'
			) {
				return;
			}

			if (event.key === 'Escape') {
				void (async () => {
					try {
						await resetBothSides();
						setError(null);
					} catch (err) {
						setError(err instanceof Error ? err.message : String(err));
					}
				})();
				return;
			}

			if (event.code !== 'Space') {
				return;
			}
			event.preventDefault();

			const time =
				event.timeStamp && event.timeStamp > 0
					? event.timeStamp / 1000
					: performance.now() / 1000;
			runLikelihoodPulse(time);
		};

		window.addEventListener('keydown', handleKeyDown);
		return () => {
			window.removeEventListener('keydown', handleKeyDown);
		};
	}, [resetBothSides, runLikelihoodPulse, wasmReady]);

	return (
		<div className={colorMode === 'dark' ? 'dark' : ''}>
			<div className="h-screen min-h-0 bg-slate-100 px-3 py-2 text-slate-900 dark:bg-gray-950 dark:text-white">
				<div className="mx-auto flex h-full min-h-0 w-full max-w-[1920px] flex-col gap-2">
					<header className="flex shrink-0 items-center justify-between gap-2 border-b border-slate-200 pb-2 dark:border-white/10">
						<div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-slate-500 dark:text-gray-400">
							<span>
								WS <code className="text-slate-800 dark:text-gray-300">{wsStatus}</code>
							</span>
							<span className="text-slate-300 dark:text-white/25">·</span>
							{snapshot ? (
								<span>
									<code className="text-slate-800 dark:text-gray-300">
										{Object.keys(snapshot).length}
									</code>{' '}
									nodes
								</span>
							) : (
								<span>no snapshot</span>
							)}
							<span className="text-slate-300 dark:text-white/25">·</span>
							<span>
								last batch <code className="text-slate-800 dark:text-gray-300">{lastBatchSize}</code>
							</span>
							<span className="text-slate-300 dark:text-white/25">·</span>
							<span title="Scroll heuristic: centering weight a, stability weight b (see render-trie.tex)">
								scroll{' '}
								<code className="text-slate-800 dark:text-gray-300">a={SCROLL_CENTERING_WEIGHT}</code>
								{' '}
								<code className="text-slate-800 dark:text-gray-300">b={SCROLL_STABILITY_WEIGHT}</code>
							</span>
							<span className="text-slate-300 dark:text-white/25">·</span>
							<span>
								<code className="text-slate-800 dark:text-gray-300">Space</code> / blink likelihood ·{' '}
								<code className="text-slate-800 dark:text-gray-300">Esc</code> reset
							</span>
							{showPracticePhrase && (
								<>
									<span className="text-slate-300 dark:text-white/25">·</span>
									<div className="flex min-w-0 items-center gap-2">
										<button
											type="button"
											onClick={shufflePracticePhrase}
											className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-slate-200 bg-slate-50 text-sm text-slate-600 transition hover:bg-slate-100 hover:text-slate-800 dark:border-white/10 dark:bg-white/5 dark:text-gray-300 dark:hover:bg-white/10 dark:hover:text-white"
											aria-label="Choose another practice phrase"
											title="Choose another practice phrase"
										>
											<span aria-hidden="true">⟳</span>
										</button>
										<span className="min-w-0 max-w-[30rem] rounded-md border border-slate-200 bg-slate-50/80 px-2.5 py-1 text-slate-700 dark:border-white/10 dark:bg-white/5 dark:text-gray-200">
											<span className="block truncate">{practicePhrase}</span>
										</span>
									</div>
								</>
							)}
						</div>
						<div className="flex shrink-0 items-center gap-3">
							<div className="flex items-center gap-2 text-xs text-slate-600 dark:text-gray-300">
								<input
									type="text"
									value={usernameInput}
									onChange={(e) => setUsernameInput(e.target.value)}
									placeholder="username"
									className="w-32 rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-800 dark:border-white/20 dark:bg-white/10 dark:text-white"
								/>
								<button
									type="button"
									onClick={() => {
										void (async () => {
											try {
												await startSessionOnBackend(usernameInput);
												setError(null);
											} catch (err) {
												setError(err instanceof Error ? err.message : String(err));
											}
										})();
									}}
									className="rounded border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-800 transition hover:bg-slate-50 dark:border-white/20 dark:bg-white/10 dark:text-white dark:hover:bg-white/20"
								>
									Start session
								</button>
								<span className="text-slate-500 dark:text-gray-400">
									{activeUsername ? `active: ${activeUsername}` : 'no active session'}
								</span>
							</div>
							<label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300">
								<input
									type="checkbox"
									checked={blinkToClick}
									onChange={(e) => setBlinkToClick(e.target.checked)}
									className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
								/>
								Blink to click
							</label>
							<label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300">
								<input
									type="checkbox"
									checked={showAll}
									onChange={(e) => setShowAll(e.target.checked)}
									className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
								/>
								Show all
							</label>
							<label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300">
								<input
									type="checkbox"
									checked={showDebugStats}
									onChange={(e) => setShowDebugStats(e.target.checked)}
									className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
								/>
								Debug
							</label>
							<label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300">
								<input
									type="checkbox"
									checked={showBoxes}
									onChange={(e) => setShowBoxes(e.target.checked)}
									className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
								/>
								Boxes
							</label>
							<label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300">
								<input
									type="checkbox"
									checked={showPracticePhrase}
									onChange={(e) => setShowPracticePhrase(e.target.checked)}
									className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
								/>
								Practice
							</label>
							<label className={`flex select-none items-center gap-1.5 text-xs ${showPracticePhrase ? 'cursor-pointer text-slate-600 dark:text-gray-300' : 'cursor-not-allowed text-slate-400 dark:text-gray-500'}`}>
								<input
									type="checkbox"
									checked={useVisualTutor}
									disabled={!showPracticePhrase}
									onChange={(e) => setUseVisualTutor(e.target.checked)}
									className="h-3.5 w-3.5 accent-blue-600 disabled:cursor-not-allowed dark:accent-blue-500"
								/>
								Visual tutor
							</label>
							<label className={`flex select-none items-center gap-1.5 text-xs ${showPracticePhrase ? 'cursor-pointer text-slate-600 dark:text-gray-300' : 'cursor-not-allowed text-slate-400 dark:text-gray-500'}`}>
								<input
									type="checkbox"
									checked={useAudioTutor}
									disabled={!showPracticePhrase}
									onChange={(e) => setUseAudioTutor(e.target.checked)}
									className="h-3.5 w-3.5 accent-blue-600 disabled:cursor-not-allowed dark:accent-blue-500"
								/>
								Audio tutor
							</label>
							<button
								type="button"
								onClick={() => setColorMode((m) => (m === 'dark' ? 'light' : 'dark'))}
								className="rounded border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-800 transition hover:bg-slate-50 dark:border-white/20 dark:bg-white/10 dark:text-white dark:hover:bg-white/20"
								aria-label={colorMode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
							>
								{colorMode === 'dark' ? 'Light' : 'Dark'}
							</button>
							<button
								type="button"
								onClick={() => setShowCalibrationDebugPanel((prev) => !prev)}
								className={`rounded border px-2.5 py-1 text-xs transition ${
									showCalibrationDebugPanel
										? 'border-blue-400 bg-blue-50 text-blue-900 hover:bg-blue-100 dark:border-blue-400/60 dark:bg-blue-500/20 dark:text-blue-100 dark:hover:bg-blue-500/30'
										: 'border-slate-300 bg-white text-slate-800 hover:bg-slate-50 dark:border-white/20 dark:bg-white/10 dark:text-white dark:hover:bg-white/20'
								}`}
							>
								Calibration debug
							</button>
							<button
								type="button"
								onClick={() => setShowPredictionLogPanel((prev) => !prev)}
								className={`rounded border px-2.5 py-1 text-xs transition ${
									showPredictionLogPanel
										? 'border-blue-400 bg-blue-50 text-blue-900 hover:bg-blue-100 dark:border-blue-400/60 dark:bg-blue-500/20 dark:text-blue-100 dark:hover:bg-blue-500/30'
										: 'border-slate-300 bg-white text-slate-800 hover:bg-slate-50 dark:border-white/20 dark:bg-white/10 dark:text-white dark:hover:bg-white/20'
								}`}
							>
								Backend log
							</button>
							<button
								type="button"
								onClick={downloadSessionDebugDump}
								className="rounded border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-800 transition hover:bg-slate-50 dark:border-white/20 dark:bg-white/10 dark:text-white dark:hover:bg-white/20"
							>
								Dump logs
							</button>
							<button
								type="button"
								onClick={() => {
									void (async () => {
										try {
											await resetBothSides();
											setError(null);
										} catch (err) {
											setError(err instanceof Error ? err.message : String(err));
										}
									})();
								}}
								className="rounded border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-800 transition hover:bg-slate-50 dark:border-white/20 dark:bg-white/10 dark:text-white dark:hover:bg-white/20"
							>
								Reset
							</button>
						</div>
					</header>

					{warning && (
						<div className="shrink-0 rounded border border-amber-400/60 bg-amber-50 px-2 py-1.5 text-xs text-amber-900 dark:border-amber-500/30 dark:bg-amber-950/30 dark:text-amber-200">
							{warning}
						</div>
					)}
					{error && (
						<div className="shrink-0 whitespace-pre-wrap break-words rounded border border-red-400/70 bg-red-50 px-2 py-1.5 text-xs text-red-900 dark:border-red-500/40 dark:bg-red-950/50 dark:text-red-200">
							{error}
						</div>
					)}

					<CalibrationSettings
						useAutomaticCalibration={useAutomaticCalibration}
						setUseAutomaticCalibration={setUseAutomaticCalibration}
						likelihoodModel={likelihoodModel}
						setLikelihoodModel={setLikelihoodModel}
						autoCalibrationLikelihoodModel={autoCalibrationLikelihoodModel}
						calibrationSampleCount={calibrationSampleCount}
						rawVariationalParams={rawVariationalParams}
						recentCalibrationPairs={recentCalibrationPairs}
						showCalibrationDebug={showCalibrationDebugPanel}
					/>

					{showPredictionLogPanel && (
						<div className="shrink-0 rounded-lg border border-slate-200 bg-white/95 p-2 shadow-sm backdrop-blur-sm dark:border-white/10 dark:bg-white/5 dark:shadow-none">
							<div className="mb-1.5 flex shrink-0 items-center justify-between gap-2">
								<h2 className="text-xs font-semibold text-slate-800 dark:text-gray-100">
									Backend Prediction Log
								</h2>
								<span className="text-[0.65rem] text-slate-500 dark:text-gray-400">
									<code>{predictionLog.length}</code> entries
								</span>
							</div>
							{predictionLog.length === 0 ? (
								<p className="text-xs text-slate-500 dark:text-gray-400">
									No backend predictions received yet.
								</p>
							) : (
								<ul className="max-h-36 min-h-0 list-none space-y-1 overflow-y-auto overscroll-contain pr-1 text-xs">
									{predictionLog.map((entry) => (
										<li
											key={entry.id}
											className="flex items-baseline gap-2 border-b border-slate-100 pb-1 last:border-b-0 last:pb-0 dark:border-white/5"
										>
											<span className="min-w-0 flex-1 break-all font-mono text-slate-800 dark:text-gray-200">
												{entry.fullString}
											</span>
											<span className="shrink-0 whitespace-nowrap text-right font-mono text-[0.65rem] tabular-nums text-slate-500 dark:text-gray-400">
												[{entry.finalTokenLexindex}] ({entry.receivedAt})
											</span>
										</li>
									))}
								</ul>
							)}
						</div>
					)}

					<div className="relative min-h-0 flex-1 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm dark:border-white/10 dark:bg-black/40 dark:shadow-none">
						{loading ? (
							<div className="flex h-full min-h-[12rem] items-center justify-center text-sm text-slate-500 dark:text-gray-400">
								Loading bayesian session…
							</div>
						) : !error && snapshot ? (
							<>
								<TrieSnapshotVisualizer
									snapshot={snapshot}
									timers={timers}
									period={likelihoodModel.period}
									expansionThreshold={expansionThreshold}
									scrollOffset={scrollOffset}
									scrollRoot={scrollRoot}
									firstForkDepth={firstForkDepth}
								useVisualTutor={showPracticePhrase && useVisualTutor}
								targetPhrase={practicePhrase}
									showAll={showAll}
									lightBackground={colorMode === 'light'}
									showBoxes={showBoxes}
									showDebugStats={showDebugStats}
								/>
								{blinkToClick && wasmReady && (
									<Eye
										onBlink={() => {
											runLikelihoodPulse(performance.now() / 1000);
										}}
									/>
								)}
							</>
						) : !error ? (
							<div className="flex h-full min-h-[12rem] items-center justify-center text-sm text-slate-500 dark:text-gray-400">
								Waiting for the first visible nodes.
							</div>
						) : (
							<div className="flex h-full min-h-[12rem] items-center justify-center text-sm text-slate-500 dark:text-gray-500">
								Fix the error above to resume.
							</div>
						)}
					</div>

				</div>
			</div>
		</div>
	);
}

export default MainPage;
