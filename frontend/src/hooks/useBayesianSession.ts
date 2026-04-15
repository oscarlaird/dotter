import { useCallback, useRef, useState } from 'react';
import initBayesianWasm, {
	BayesianSession,
	debugPanicTest,
	initPanicHook,
} from '../wasm_pkg/bayesian';
import type {
	CalibrationPair,
	VariationalParams,
} from '../domain/likelihoodModel';
import type { ExpandedSnapshot } from '../domain/trieLayout';

const RECENT_CONSOLE_ERROR_LIMIT = 20;

let wasmPanicConsoleCaptureInstalled = false;
let recentConsoleErrors: string[] = [];

export interface SessionDebugDump {
	update_json_log: Array<{
		kind: 'likelihood' | 'prior' | string;
		json: string;
	}>;
}

export interface RecalibrationResult {
	prior_params: VariationalParams;
	used_likelihood_updates: number;
	recent_pairs: CalibrationPair[];
}

function formatStepError(step: string, err: unknown): string {
	const message = err instanceof Error ? err.message : String(err);
	console.error(`[useBayesianSession] ${step}`, err);
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
		err instanceof Error && err.stack && !err.stack.startsWith(err.message) ? err.stack : null;
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

export function useBayesianSession() {
	const sessionRef = useRef<BayesianSession | null>(null);
	const bootstrapPromiseRef = useRef<Promise<void> | null>(null);
	const sessionOpQueueRef = useRef<Promise<unknown>>(Promise.resolve());
	const sessionOpInFlightRef = useRef(false);
	const sessionTrappedRef = useRef(false);
	const sessionTrapMessageRef = useRef<string | null>(null);

	const [loading, setLoading] = useState(true);
	const [wasmReady, setWasmReady] = useState(false);
	const [expansionThreshold, setExpansionThreshold] = useState<number>(Number.NEGATIVE_INFINITY);

	const withSessionThreshold = useCallback((session: BayesianSession): void => {
		const threshold = session.expansion_threshold();
		setExpansionThreshold(threshold);
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

	const refreshSnapshot = useCallback(async (): Promise<ExpandedSnapshot> => {
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
		return JSON.parse(snapshotJson) as ExpandedSnapshot;
	}, [enqueueSessionOp]);

	const initialize = useCallback(async (): Promise<ExpandedSnapshot> => {
		setLoading(true);
		try {
			if (!bootstrapPromiseRef.current) {
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
							withSessionThreshold(sessionRef.current);
						} catch (err) {
							throw new Error(formatStepError('BayesianSession constructor failed', err));
						}
					} else {
						withSessionThreshold(sessionRef.current);
					}
				})();
			}

			await bootstrapPromiseRef.current;
			const initialSnapshot = await refreshSnapshot();
			setWasmReady(true);
			return initialSnapshot;
		} catch (err) {
			bootstrapPromiseRef.current = null;
			throw err instanceof Error ? err : new Error(String(err));
		} finally {
			setLoading(false);
		}
	}, [refreshSnapshot, withSessionThreshold]);

	const startLocalString = useCallback(
		async (viBefore: VariationalParams): Promise<ExpandedSnapshot> => {
			const snapshotJson = await enqueueSessionOp(() => {
				if (sessionTrappedRef.current || !sessionRef.current) {
					try {
						sessionRef.current = new BayesianSession();
						withSessionThreshold(sessionRef.current);
						sessionTrappedRef.current = false;
						sessionTrapMessageRef.current = null;
					} catch (err) {
						throw new Error(
							formatStepError('BayesianSession constructor failed during reset', err),
						);
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
			return JSON.parse(snapshotJson) as ExpandedSnapshot;
		},
		[enqueueSessionOp, withSessionThreshold],
	);

	const receivePriorUpdate = useCallback(
		async (contentJson: string): Promise<ExpandedSnapshot> => {
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
			return JSON.parse(snapshotJson) as ExpandedSnapshot;
		},
		[enqueueSessionOp],
	);

	const applyLikelihoodUpdate = useCallback(
		async (likelihoodJson: string): Promise<ExpandedSnapshot> => {
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
					throw new Error(
						formatStepError('expand_to_threshold after likelihood_update failed', err),
					);
				}
			});
			return JSON.parse(snapshotJson) as ExpandedSnapshot;
		},
		[enqueueSessionOp],
	);

	const recalibrate = useCallback(
		async (viBefore: VariationalParams): Promise<RecalibrationResult> => {
			const metricsJson = await enqueueSessionOp(() => {
				const session = sessionRef.current;
				if (!session) {
					throw new Error('BayesianSession is not initialized');
				}
				try {
					return session.recalibrate(JSON.stringify(viBefore), true);
				} catch (err) {
					throw new Error(formatStepError('recalibrate failed', err));
				}
			});
			return JSON.parse(metricsJson) as RecalibrationResult;
		},
		[enqueueSessionOp],
	);

	const downloadSessionDebugDump = useCallback(async (): Promise<void> => {
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
	}, [enqueueSessionOp]);

	return {
		loading,
		wasmReady,
		expansionThreshold,
		initialize,
		refreshSnapshot,
		startLocalString,
		receivePriorUpdate,
		applyLikelihoodUpdate,
		recalibrate,
		downloadSessionDebugDump,
	};
}
