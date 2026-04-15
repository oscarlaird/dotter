import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import CalibrationSettings from '../components/CalibrationSettings';
import Eye from '../components/Eye';
import MainPageHeader from '../components/MainPageHeader';
import PredictionLogPanel, {
	type PredictionLogEntry,
} from '../components/PredictionLogPanel';
import TrieSnapshotVisualizer from '../components/TrieSnapshotVisualizer';
import { playTutorOutlierTone, playTutorTone } from '../domain/audioTutor';
import {
	type AutoCalibrationState,
	type LikelihoodModel,
	type VariationalParams,
	DEFAULT_LIKELIHOOD_MODEL,
	moduloDelay,
	variationalParamsToLikelihoodModel,
} from '../domain/likelihoodModel';
import { randomPracticePhrase } from '../domain/practicePhrases';
import {
	buildLikelihoodPayloadNodes,
	timersForSnapshot,
} from '../domain/snapshotTimers';
import {
	computeScrollLayoutState,
	findTutorTargetKey,
	type ExpandedSnapshot,
	type VisibleNodeTimerMap,
} from '../domain/trieLayout';
import { useBackendSocket } from '../hooks/useBackendSocket';
import { useBayesianSession } from '../hooks/useBayesianSession';

const APP_USERNAME_STORAGE_KEY = 'dotter-app-username';
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

function errorMessage(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}

function MainPage() {
	const [snapshot, setSnapshot] = useState<ExpandedSnapshot | null>(null);
	const [timers, setTimers] = useState<VisibleNodeTimerMap>({});
	const [error, setError] = useState<string | null>(null);
	const [predictionLog, setPredictionLog] = useState<PredictionLogEntry[]>([]);
	const predictionLogIdRef = useRef(0);
	const audioContextRef = useRef<AudioContext | null>(null);

	const [lastBatchSize, setLastBatchSize] = useState(0);
	const [likelihoodModel, setLikelihoodModel] = useState<LikelihoodModel>({
		...DEFAULT_LIKELIHOOD_MODEL,
	});
	const [useAutomaticCalibration, setUseAutomaticCalibration] = useState<AutoCalibrationState>({
		mu_delay: true,
		stddev_delay: true,
		outliers: true,
	});
	const [autoCalibrationLikelihoodModel, setAutoCalibrationLikelihoodModel] =
		useState<LikelihoodModel>(DEFAULT_LIKELIHOOD_MODEL);
	const [calibrationSampleCount, setCalibrationSampleCount] = useState(0);
	const [rawVariationalParams, setRawVariationalParams] = useState<VariationalParams | null>(null);
	const [recentCalibrationPairs, setRecentCalibrationPairs] = useState<
		Array<{ x: number; period: number }>
	>([]);
	const [usernameInput, setUsernameInput] = useState(() => readStoredUsername());
	const [activeUsername, setActiveUsername] = useState<string | null>(null);
	const [currentViBefore, setCurrentViBefore] = useState<VariationalParams | null>(null);

	const [showPredictionLogPanel, setShowPredictionLogPanel] = useState(false);
	const [showCalibrationDebugPanel, setShowCalibrationDebugPanel] = useState(false);
	const [colorMode, setColorMode] = useState<'light' | 'dark'>(readStoredColorMode);
	const [showBoxes, setShowBoxes] = useState(true);
	const [showDebugStats, setShowDebugStats] = useState(false);
	const [showAll, setShowAll] = useState(false);
	const [blinkToClick, setBlinkToClick] = useState(readStoredBlinkToClick);
	const [showPracticePhrase, setShowPracticePhrase] = useState(false);
	const [useVisualTutor, setUseVisualTutor] = useState(false);
	const [useAudioTutor, setUseAudioTutor] = useState(false);
	const [practicePhrase, setPracticePhrase] = useState(() => randomPracticePhrase());

	const [scrollOffset, setScrollOffset] = useState(0);
	const [scrollRoot, setScrollRoot] = useState('^');
	const [scrollAncestorKeys, setScrollAncestorKeys] = useState<string[]>([]);
	const [firstForkDepth, setFirstForkDepth] = useState<number | null>(null);
	const scrollOffsetRef = useRef(0);
	const likelihoodModelRef = useRef(likelihoodModel);

	const {
		loading,
		wasmReady,
		expansionThreshold,
		initialize,
		startLocalString,
		receivePriorUpdate,
		applyLikelihoodUpdate,
		recalibrate,
		downloadSessionDebugDump,
	} = useBayesianSession();
	const expansionThresholdRef = useRef(expansionThreshold);

	const socketActionsRef = useRef<{ requestNextPrior: () => void } | null>(null);

	const applyViBeforeToUi = useCallback(
		(viBefore: VariationalParams) => {
			const nextModel = variationalParamsToLikelihoodModel(viBefore, likelihoodModel.period);
			setCurrentViBefore(viBefore);
			setCalibrationSampleCount(0);
			setRawVariationalParams(viBefore);
			setRecentCalibrationPairs([]);
			setAutoCalibrationLikelihoodModel(nextModel);
		},
		[likelihoodModel.period],
	);

	const applySnapshot = useCallback(
		(nextSnapshot: ExpandedSnapshot, resetAllTimers: boolean) => {
			const nextScrollLayout = computeScrollLayoutState(
				nextSnapshot,
				expansionThresholdRef.current,
				scrollOffsetRef.current,
			);
			scrollOffsetRef.current = nextScrollLayout.scrollOffset;
			setScrollOffset(nextScrollLayout.scrollOffset);
			setScrollRoot(nextScrollLayout.scrollRoot);
			setScrollAncestorKeys(nextScrollLayout.scrollAncestorKeys);
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
		},
		[],
	);

	const recordPredictionLog = useCallback((contentJson: string) => {
		const payload = JSON.parse(contentJson) as {
			full_string: string;
			final_token_lexindex: number;
		};
		const nextEntry: PredictionLogEntry = {
			id: predictionLogIdRef.current += 1,
			fullString: payload.full_string,
			finalTokenLexindex: payload.final_token_lexindex,
			receivedAt: new Date().toLocaleTimeString(),
		};
		setPredictionLog((current) => [nextEntry, ...current]);
	}, []);

	const handleSocketOpen = useCallback(() => {
		setPredictionLog([]);
	}, []);

	const handleSocketResetAck = useCallback(() => {
		setError(null);
		setPredictionLog([]);
	}, []);

	const handleSocketSessionStarted = useCallback(
		async ({
			username,
			variational_params,
		}: {
			username: string;
			variational_params: VariationalParams;
		}) => {
			setActiveUsername(username);
			const nextSnapshot = await startLocalString(variational_params);
			applySnapshot(nextSnapshot, true);
			applyViBeforeToUi(variational_params);
			socketActionsRef.current?.requestNextPrior();
			setError(null);
			setPredictionLog([]);
		},
		[applySnapshot, applyViBeforeToUi, startLocalString],
	);

	const handleSocketPriorUpdate = useCallback(
		async (contentJson: string) => {
			recordPredictionLog(contentJson);
			const nextSnapshot = await receivePriorUpdate(contentJson);
			applySnapshot(nextSnapshot, false);
		},
		[applySnapshot, receivePriorUpdate, recordPredictionLog],
	);

	const {
		wsStatus,
		warning,
		startSession,
		requestNextPrior,
		reset,
		sendLikelihoodUpdate,
	} = useBackendSocket({
		enabled: wasmReady,
		autoStartUsername: usernameInput,
		onOpen: handleSocketOpen,
		onResetAck: handleSocketResetAck,
		onSessionStarted: handleSocketSessionStarted,
		onPriorUpdate: handleSocketPriorUpdate,
		onErrorMessage: setError,
	});

	useEffect(() => {
		socketActionsRef.current = { requestNextPrior };
	}, [requestNextPrior]);

	useEffect(() => {
		likelihoodModelRef.current = likelihoodModel;
	}, [likelihoodModel]);

	useEffect(() => {
		expansionThresholdRef.current = expansionThreshold;
	}, [expansionThreshold]);

	const startSessionOnBackend = useCallback(
		(username: string) => {
			setActiveUsername(null);
			setPredictionLog([]);
			setCurrentViBefore(null);
			setCalibrationSampleCount(0);
			setRawVariationalParams(null);
			setRecentCalibrationPairs([]);
			startSession(username);
		},
		[startSession],
	);

	const resetBothSides = useCallback(async () => {
		if (!currentViBefore) {
			throw new Error('Session has no current calibration prior; start a session first');
		}
		const recalibrationResult = await recalibrate(currentViBefore);
		const nextSnapshot = await startLocalString(recalibrationResult.prior_params);
		applySnapshot(nextSnapshot, true);
		applyViBeforeToUi(recalibrationResult.prior_params);
		reset();
	}, [applySnapshot, applyViBeforeToUi, currentViBefore, recalibrate, reset, startLocalString]);

	useEffect(() => {
		let cancelled = false;
		void (async () => {
			try {
				setError(null);
				const initialSnapshot = await initialize();
				if (!cancelled) {
					applySnapshot(initialSnapshot, true);
				}
			} catch (err) {
				if (!cancelled) {
					setError(errorMessage(err));
				}
			}
		})();
		return () => {
			cancelled = true;
		};
	}, [applySnapshot, initialize]);

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

				if (showPracticePhrase && useAudioTutor && tutorTargetKey) {
					const targetTimer = timers[tutorTargetKey];
					const predictiveStddev = likelihoodModel.stddev_delay;
					if (targetTimer && predictiveStddev > 0) {
						const x = moduloDelay(
							timeSeconds,
							targetTimer.phase,
							likelihoodModel.period,
						);
						const offsetStddevs =
							(x - likelihoodModel.mu_delay) / predictiveStddev;
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

				const nodes = buildLikelihoodPayloadNodes(
					snapshot,
					timers,
					timeSeconds,
					likelihoodModel,
					scrollRoot,
					scrollAncestorKeys,
				);
				const likelihoodJson = JSON.stringify({
					period: likelihoodModel.period,
					y: timeSeconds,
					nodes,
				});

				const nextSnapshot = await applyLikelihoodUpdate(likelihoodJson);
				applySnapshot(nextSnapshot, true);
				sendLikelihoodUpdate(likelihoodJson);

				const recalibrationResult = await recalibrate(currentViBefore);
				setCalibrationSampleCount(recalibrationResult.used_likelihood_updates);
				setRawVariationalParams(recalibrationResult.prior_params);
				setRecentCalibrationPairs(recalibrationResult.recent_pairs);
				setAutoCalibrationLikelihoodModel(
					variationalParamsToLikelihoodModel(
						recalibrationResult.prior_params,
						likelihoodModel.period,
					),
				);
				setLastBatchSize(Object.keys(nodes).length);
				setError(null);
			})().catch((err) => {
				setError(errorMessage(err));
			});
		},
		[
			applyLikelihoodUpdate,
			applySnapshot,
			currentViBefore,
			likelihoodModel,
			recalibrate,
			scrollAncestorKeys,
			scrollRoot,
			sendLikelihoodUpdate,
			showPracticePhrase,
			snapshot,
			timers,
			tutorTargetKey,
			useAudioTutor,
		],
	);

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
		if (!showPracticePhrase) {
			if (useVisualTutor) {
				setUseVisualTutor(false);
			}
			if (useAudioTutor) {
				setUseAudioTutor(false);
			}
		}
	}, [showPracticePhrase, useAudioTutor, useVisualTutor]);

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
						setError(errorMessage(err));
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

	const shufflePracticePhrase = useCallback(() => {
		setPracticePhrase((current) => randomPracticePhrase(current));
	}, []);

	const handleStartSession = useCallback(() => {
		void (async () => {
			try {
				startSessionOnBackend(usernameInput);
				setError(null);
			} catch (err) {
				setError(errorMessage(err));
			}
		})();
	}, [startSessionOnBackend, usernameInput]);

	const handleDownloadSessionDebugDump = useCallback(() => {
		void (async () => {
			try {
				await downloadSessionDebugDump();
				setError(null);
			} catch (err) {
				setError(errorMessage(err));
			}
		})();
	}, [downloadSessionDebugDump]);

	const handleReset = useCallback(() => {
		void (async () => {
			try {
				await resetBothSides();
				setError(null);
			} catch (err) {
				setError(errorMessage(err));
			}
		})();
	}, [resetBothSides]);

	return (
		<div className={colorMode === 'dark' ? 'dark' : ''}>
			<div className="h-screen min-h-0 bg-slate-100 px-3 py-2 text-slate-900 dark:bg-gray-950 dark:text-white">
				<div className="mx-auto flex h-full min-h-0 w-full max-w-[1920px] flex-col gap-2">
					<MainPageHeader
						wsStatus={wsStatus}
						snapshotNodeCount={snapshot ? Object.keys(snapshot).length : null}
						lastBatchSize={lastBatchSize}
						showPracticePhrase={showPracticePhrase}
						onShowPracticePhraseChange={setShowPracticePhrase}
						practicePhrase={practicePhrase}
						onShufflePracticePhrase={shufflePracticePhrase}
						usernameInput={usernameInput}
						onUsernameInputChange={setUsernameInput}
						onStartSession={handleStartSession}
						activeUsername={activeUsername}
						blinkToClick={blinkToClick}
						onBlinkToClickChange={setBlinkToClick}
						showAll={showAll}
						onShowAllChange={setShowAll}
						showDebugStats={showDebugStats}
						onShowDebugStatsChange={setShowDebugStats}
						showBoxes={showBoxes}
						onShowBoxesChange={setShowBoxes}
						useVisualTutor={useVisualTutor}
						onUseVisualTutorChange={setUseVisualTutor}
						useAudioTutor={useAudioTutor}
						onUseAudioTutorChange={setUseAudioTutor}
						colorMode={colorMode}
						onToggleColorMode={() =>
							setColorMode((mode) => (mode === 'dark' ? 'light' : 'dark'))
						}
						showCalibrationDebugPanel={showCalibrationDebugPanel}
						onToggleCalibrationDebugPanel={() =>
							setShowCalibrationDebugPanel((previous) => !previous)
						}
						showPredictionLogPanel={showPredictionLogPanel}
						onTogglePredictionLogPanel={() =>
							setShowPredictionLogPanel((previous) => !previous)
						}
						onDownloadSessionDebugDump={handleDownloadSessionDebugDump}
						onReset={handleReset}
					/>

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

					{showPredictionLogPanel && <PredictionLogPanel entries={predictionLog} />}

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
