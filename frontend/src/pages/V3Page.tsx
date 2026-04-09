import { useCallback, useEffect, useRef, useState } from 'react';
import initBayesianWasm, {
	BayesianSession,
	debugPanicTest,
	initPanicHook,
} from '../wasm_pkg/bayesian';
import practicePhrasesText from './v3-practice-phrases.txt?raw';
import CalibrationSettings, { type LikelihoodModel } from '../components/CalibrationSettings';
import Eye from '../components/Eye';
import TrieSnapshotVisualizer, {
	computeScrollLayoutState,
} from '../components/TrieSnapshotVisualizer';
import type {
	ExpandedSnapshot,
	VisibleNodeTimerMap,
} from '../components/TrieSnapshotVisualizer';

const DEFAULT_LIKELIHOOD_MODEL: LikelihoodModel = {
	mu_delay: 0.15,
	stddev_delay: 0.04,
	outliers: 0.03,
	period: 1.1,
};
const N_SKIP_PRACTICE_PHRASES = 6;
const PRACTICE_PHRASES = practicePhrasesText
	.split('\n')
	.map((line) => line.trim())
	.filter((line) => line.length > 0);

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
	let delay = time - phase;
	delay = ((delay + model.period * 1.5) % model.period) - model.period / 2;
	const gaussianLogLikelihood = normalLogpdf(delay, model.mu_delay, model.stddev_delay);
	const uniformLogLikelihood = Math.log(1 / model.period);
	const outlierProb = Math.log(model.outliers);
	const notOutlierProb = Math.log(1 - model.outliers);
	return logaddexp(
		notOutlierProb + gaussianLogLikelihood,
		outlierProb + uniformLogLikelihood,
	);
}

function randomTimersForSnapshot(
	snapshot: ExpandedSnapshot,
	model: LikelihoodModel,
	existingTimers: VisibleNodeTimerMap,
	resetAll: boolean,
	renderedNodeKeys: readonly string[],
): VisibleNodeTimerMap {
	const nextTimers: VisibleNodeTimerMap = {};
	for (const fullString of renderedNodeKeys) {
		if (!(fullString in snapshot)) {
			continue;
		}
		if (!resetAll && existingTimers[fullString]) {
			nextTimers[fullString] = existingTimers[fullString];
			continue;
		}
		nextTimers[fullString] = {
			phase: Math.random() * model.period,
		};
	}
	return nextTimers;
}

function formatStepError(step: string, err: unknown): string {
	const message = err instanceof Error ? err.message : String(err);
	console.error(`[V3Page] ${step}`, err);
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

const V3_THEME_STORAGE_KEY = 'dotter-v3-theme';
const V3_BLINK_TO_CLICK_STORAGE_KEY = 'dotter-v3-blink-to-click';

function readStoredBlinkToClick(): boolean {
	try {
		return localStorage.getItem(V3_BLINK_TO_CLICK_STORAGE_KEY) === 'true';
	} catch {
		return false;
	}
}

function readStoredColorMode(): 'light' | 'dark' {
	try {
		const raw = localStorage.getItem(V3_THEME_STORAGE_KEY);
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

function V3Page() {
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
	const [useAutomaticCalibration, setUseAutomaticCalibration] = useState(false);
	const sessionRef = useRef<BayesianSession | null>(null);
	const socketRef = useRef<WebSocket | null>(null);
	const likelihoodModelRef = useRef(likelihoodModel);
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
	const [practicePhrase, setPracticePhrase] = useState(() => randomPracticePhrase());
	const [expansionThreshold, setExpansionThreshold] = useState<number>(Number.NEGATIVE_INFINITY);
	const [scrollOffset, setScrollOffset] = useState(0);
	const [scrollRoot, setScrollRoot] = useState('^');
	const [firstForkDepth, setFirstForkDepth] = useState<number | null>(null);
	const scrollOffsetRef = useRef(0);
	const scrollRootRef = useRef('^');
	const scrollAncestorKeysRef = useRef<string[]>([]);

	useEffect(() => {
		try {
			localStorage.setItem(V3_THEME_STORAGE_KEY, colorMode);
		} catch {
			// ignore
		}
	}, [colorMode]);

	useEffect(() => {
		try {
			localStorage.setItem(V3_BLINK_TO_CLICK_STORAGE_KEY, blinkToClick ? 'true' : 'false');
		} catch {
			// ignore
		}
	}, [blinkToClick]);

	useEffect(() => {
		likelihoodModelRef.current = likelihoodModel;
	}, [likelihoodModel]);

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
			randomTimersForSnapshot(
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

	const resetLocalSession = useCallback(async () => {
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
			try {
				return session.expand_to_threshold();
			} catch (err) {
				throw new Error(formatStepError('expand_to_threshold after reset failed', err));
			}
		});
		applySnapshot(JSON.parse(snapshotJson) as ExpandedSnapshot, true);
	}, [applySnapshot, enqueueSessionOp]);

	const resetBothSides = useCallback(async () => {
		const ws = socketRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) {
			throw new Error('WebSocket must be connected before resetting');
		}
		await resetLocalSession();
		ws.send(JSON.stringify({ type: 'reset' }));
	}, [resetLocalSession]);

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
			void (async () => {
				try {
					await resetLocalSession();
					ws.send(JSON.stringify({ type: 'reset' }));
				} catch (err) {
					setError(err instanceof Error ? err.message : String(err));
				}
			})();
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
					content?: { message?: string };
				};

				if (message.type === 'reset_ack') {
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
	}, [applySnapshot, enqueueSessionOp, recordPredictionLog, resetLocalSession, wasmReady]);

	const runLikelihoodPulse = useCallback(
		(timeSeconds: number) => {
			void (async () => {
				if (!snapshot) {
					return;
				}
				if (sessionTrappedRef.current) {
					setError(
						sessionTrapMessageRef.current ??
							'BayesianSession trapped in wasm; press Reset to recover',
					);
					return;
				}

				const likelihoodPayload: Record<string, { l: number }> = {};
				for (const [fullString, timer] of Object.entries(timers)) {
					if (!(fullString in snapshot)) {
						continue;
					}
					const likelihood = timerLikelihood(timeSeconds, timer.phase, likelihoodModelRef.current);
					likelihoodPayload[fullString] = { l: likelihood };
					if (fullString === scrollRootRef.current) {
						for (const ancestorKey of scrollAncestorKeysRef.current) {
							if (!(ancestorKey in snapshot)) {
								continue;
							}
							likelihoodPayload[ancestorKey] = { l: likelihood };
						}
					}
				}
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
				setLastBatchSize(Object.keys(likelihoodPayload).length);
				setError(null);
			})().catch((err) => {
				setError(err instanceof Error ? err.message : String(err));
			});
		},
		[applySnapshot, enqueueSessionOp, snapshot, timers],
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

					<div className="grid shrink-0 grid-cols-1 gap-2 lg:grid-cols-[minmax(0,3fr)_minmax(16rem,1fr)]">
						<CalibrationSettings
							useAutomaticCalibration={useAutomaticCalibration}
							setUseAutomaticCalibration={setUseAutomaticCalibration}
							likelihoodModel={likelihoodModel}
							setLikelihoodModel={setLikelihoodModel}
							autoCalibrationLikelihoodModel={DEFAULT_LIKELIHOOD_MODEL}
						/>
						<div className="flex min-h-0 flex-col rounded-lg border border-slate-200 bg-white p-2 shadow-sm dark:border-white/10 dark:bg-white/5 dark:shadow-none">
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
					</div>
				</div>
			</div>
		</div>
	);
}

export default V3Page;
