import { useCallback, useEffect, useRef, useState } from 'react';
import initBayesianWasm, { BayesianSession } from '../wasm_pkg/bayesian';
import CalibrationSettings, { type LikelihoodModel } from '../components/CalibrationSettings';
import TrieSnapshotVisualizer from '../components/TrieSnapshotVisualizer';
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
): VisibleNodeTimerMap {
	const nextTimers: VisibleNodeTimerMap = {};
	for (const fullString of Object.keys(snapshot)) {
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

function V3Page() {
	const [snapshot, setSnapshot] = useState<ExpandedSnapshot | null>(null);
	const [timers, setTimers] = useState<VisibleNodeTimerMap>({});
	const [error, setError] = useState<string | null>(null);
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

	useEffect(() => {
		likelihoodModelRef.current = likelihoodModel;
	}, [likelihoodModel]);

	const applySnapshot = useCallback((nextSnapshot: ExpandedSnapshot, resetAllTimers: boolean) => {
		setSnapshot(nextSnapshot);
		setTimers((currentTimers) =>
			randomTimersForSnapshot(
				nextSnapshot,
				likelihoodModelRef.current,
				currentTimers,
				resetAllTimers,
			),
		);
	}, []);

	const refreshSnapshot = useCallback((resetAllTimers = false) => {
		const session = sessionRef.current;
		if (!session) {
			return;
		}
		const snapshotJson = session.expand_to_threshold();
		applySnapshot(JSON.parse(snapshotJson) as ExpandedSnapshot, resetAllTimers);
	}, [applySnapshot]);

	const resetLocalSession = useCallback(() => {
		const session = sessionRef.current;
		if (!session) {
			throw new Error('BayesianSession is not initialized');
		}
		session.reset();
		refreshSnapshot(true);
	}, [refreshSnapshot]);

	const resetBothSides = useCallback(() => {
		const ws = socketRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) {
			throw new Error('WebSocket must be connected before resetting');
		}
		resetLocalSession();
		ws.send(JSON.stringify({ type: 'reset' }));
	}, [resetLocalSession]);

	useEffect(() => {
		let cancelled = false;
		const formatError = (step: string, err: unknown) => {
			const message = err instanceof Error ? err.message : String(err);
			console.error(`[V3Page] ${step}`, err);
			return `${step}: ${message}`;
		};

		async function loadSession() {
			try {
				setLoading(true);
				setError(null);
				if (!bootstrapPromiseRef.current) {
					try {
						bootstrapPromiseRef.current = (async () => {
							try {
								await initBayesianWasm();
							} catch (err) {
								throw new Error(formatError('initBayesianWasm failed', err));
							}
							if (!sessionRef.current) {
								try {
									sessionRef.current = new BayesianSession();
								} catch (err) {
									throw new Error(formatError('BayesianSession constructor failed', err));
								}
							}
							try {
								refreshSnapshot(true);
							} catch (err) {
								throw new Error(formatError('initial expand_to_threshold failed', err));
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
			try {
				resetLocalSession();
				ws.send(JSON.stringify({ type: 'reset' }));
			} catch (err) {
				setError(err instanceof Error ? err.message : String(err));
			}
		});

		ws.addEventListener('close', () => setWsStatus('Disconnected'));
		ws.addEventListener('error', () => {
			setWsStatus('Error');
			setError('WebSocket connection failed');
		});

		ws.addEventListener('message', (event) => {
			try {
				const message = JSON.parse(event.data as string) as {
					type: string;
					content_json?: string;
					content?: { message?: string };
				};
				const session = sessionRef.current;
				if (!session) {
					return;
				}

				if (message.type === 'reset_ack') {
					setError(null);
					return;
				}

				if (message.type === 'prior_update' && typeof message.content_json === 'string') {
					session.receive_prior_update(message.content_json);
					session.apply_updates();
					refreshSnapshot(false);
					return;
				}

				if (message.type === 'error' && message.content?.message) {
					setError(message.content.message);
				}
			} catch (err) {
				setError(err instanceof Error ? err.message : String(err));
			}
		});

		return () => {
			socketRef.current = null;
			ws.close();
		};
	}, [refreshSnapshot, resetLocalSession, wasmReady]);

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
				try {
					resetBothSides();
					setError(null);
				} catch (err) {
					setError(err instanceof Error ? err.message : String(err));
				}
				return;
			}

			if (event.code !== 'Space') {
				return;
			}
			event.preventDefault();

			try {
				const ws = socketRef.current;
				if (!ws || ws.readyState !== WebSocket.OPEN) {
					throw new Error('WebSocket must be connected before sending likelihood updates');
				}
				const session = sessionRef.current;
				if (!session) {
					throw new Error('BayesianSession is not initialized');
				}
				if (!snapshot) {
					return;
				}

				const time =
					event.timeStamp && event.timeStamp > 0
						? event.timeStamp / 1000
						: performance.now() / 1000;
				const likelihoodPayload: Record<string, { l: number }> = {};
				for (const fullString of Object.keys(snapshot)) {
					const timer = timers[fullString];
					if (!timer) {
						continue;
					}
					likelihoodPayload[fullString] = {
						l: timerLikelihood(time, timer.phase, likelihoodModelRef.current),
					};
				}
				const likelihoodJson = JSON.stringify(likelihoodPayload);
				session.receive_likelihood_update(likelihoodJson);
				session.apply_updates();
				refreshSnapshot(true);
				ws.send(JSON.stringify({ type: 'likelihood_update', content_json: likelihoodJson }));
				setLastBatchSize(Object.keys(likelihoodPayload).length);
				setError(null);
			} catch (err) {
				setError(err instanceof Error ? err.message : String(err));
			}
		};

		window.addEventListener('keydown', handleKeyDown);
		return () => {
			window.removeEventListener('keydown', handleKeyDown);
		};
	}, [refreshSnapshot, resetBothSides, snapshot, timers, wasmReady]);

	return (
		<div className="h-screen bg-gray-950 p-6 text-white">
			<div className="mx-auto flex h-full max-w-7xl flex-col gap-4">
				<div className="flex items-center justify-between gap-4">
					<div>
						<h1 className="text-3xl font-semibold">V3 Bayesian Session</h1>
						<p className="text-sm text-gray-300">
							The frontend and backend both apply the same JSON strings to their local
							<code> BayesianSession</code>.
						</p>
					</div>
					<button
						type="button"
						onClick={() => {
							try {
								resetBothSides();
								setError(null);
							} catch (err) {
								setError(err instanceof Error ? err.message : String(err));
							}
						}}
						className="rounded border border-white/20 bg-white/10 px-3 py-2 text-sm text-white transition hover:bg-white/20"
					>
						Reset
					</button>
				</div>

				<p className="text-sm text-gray-400">
					WebSocket: <code>{wsStatus}</code>
				</p>
				{snapshot && (
					<p className="text-sm text-gray-400">
						Showing <code>{Object.keys(snapshot).length}</code> visible strings from
						<code> expand_to_threshold()</code>.
					</p>
				)}
				<p className="text-sm text-gray-400">
					Press <code>Space</code> to score all visible timers and emit one batch
					likelihood JSON string. Press <code>Escape</code> to reset. Timers are
					re-randomized after each likelihood update.
				</p>
				<p className="text-sm text-gray-400">
					Last likelihood batch size: <code>{lastBatchSize}</code>. The trie strings
					shown below use <code>^</code> for root and <code>_</code> for word
					boundaries.
				</p>
				<div className="max-w-3xl">
					<CalibrationSettings
						useAutomaticCalibration={useAutomaticCalibration}
						setUseAutomaticCalibration={setUseAutomaticCalibration}
						likelihoodModel={likelihoodModel}
						setLikelihoodModel={setLikelihoodModel}
						autoCalibrationLikelihoodModel={DEFAULT_LIKELIHOOD_MODEL}
					/>
				</div>
				{loading && <div className="text-gray-300">Loading bayesian session...</div>}
				{error && (
					<div className="rounded border border-red-500/40 bg-red-950/50 p-3 text-red-200">
						{error}
					</div>
				)}
				{!loading && !error && snapshot && (
					<div className="min-h-0 flex-1 overflow-hidden rounded-lg border border-white/10 bg-black/40">
						<TrieSnapshotVisualizer
							snapshot={snapshot}
							timers={timers}
							period={likelihoodModel.period}
						/>
					</div>
				)}
			</div>
		</div>
	);
}

export default V3Page;
