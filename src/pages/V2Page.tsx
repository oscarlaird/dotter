import { useState, useEffect, useRef, useCallback } from 'react';
import Cookies from 'js-cookie';
import * as chartjs from 'chart.js';
import * as trieLogic from '../utils/trie_logic';
import type { TrieNode, LmMap, LmEntry } from '../utils/trie_logic';
import { autoStats } from '../utils/stats';
import prefixRangePrecompRaw from '../utils/llama_prefix_range_precomp.json';
import CalibrationSettings from '../components/CalibrationSettings';
import type { LikelihoodModel } from '../components/CalibrationSettings';
import TrieVisualizer from '../components/TrieVisualizer';
import type { SetLikelihoodsEvent } from '../components/TrieVisualizer';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type PrefixRangePrecomp = Record<string, [number, number]>;
const prefixRangePrecomp = prefixRangePrecompRaw as unknown as PrefixRangePrecomp;

interface Trial {
	delay_pairs?: { delay: number; period: number }[];
	time_elapsed: number;
	best_val: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const THRESHOLD = Math.log(0.03);
const STOP_CONFIRM_THRESHOLD = Math.log(0.9);

const DEFAULT_LIKELIHOOD_MODEL: LikelihoodModel = {
	mu_delay: 0.15,
	stddev_delay: 0.04,
	outliers: 0.03,
	period: 1.1,
};

const INITIAL_PROMPT = `my watch fell in the water
prevailing wind from the east
never too rich and never too thin
breathing is difficult
i can see the rings on saturn
`;

const N_SKIP_PHRASES = 6;
const CPS_TO_WPM = 60 / 5;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function V2Page() {
	const [trials, setTrials] = useState<Trial[]>([]);
	const [trialsCps, setTrialsCps] = useState<number[]>([]);
	const [avgWpm, setAvgWpm] = useState(0);
	const [socket, setSocket] = useState<WebSocket | null>(null);
	const lmRef = useRef<LmMap>({});
	const [trie, setTrie] = useState<TrieNode>(() => structuredClone(trieLogic.root_node));
	const trieRef = useRef<TrieNode>(trie);
	const [trieUpdatedFlag, setTrieUpdatedFlag] = useState(false);

	const [autoCalibrationModel, setAutoCalibrationModel] = useState<LikelihoodModel>(
		structuredClone(DEFAULT_LIKELIHOOD_MODEL),
	);
	const [useAutomaticCalibration, setUseAutomaticCalibration] = useState(true);
	const [awaitingFirstKeypress, setAwaitingFirstKeypress] = useState(true);
	const [wpmStartTime, setWpmStartTime] = useState<number | null>(null);
	const [confirmed, setConfirmed] = useState(false);
	const [likelihoodModel, setLikelihoodModel] = useState<LikelihoodModel>(
		structuredClone(DEFAULT_LIKELIHOOD_MODEL),
	);

	const [username, setUsername] = useState<string>(Cookies.get('username') ?? 'guest');
	const [bestVal, setBestVal] = useState<string | null>(null);
	const [useVisualTutor, setUseVisualTutor] = useState(false);
	const [multilineMode, setMultilineMode] = useState(false);
	const [webSocketStatusMsg, setWebSocketStatusMsg] = useState('Connecting...');
	const [latency, setLatency] = useState<number | null>(null);
	const [prompt, setPrompt] = useState(INITIAL_PROMPT);
	const [proposedPrompt, setProposedPrompt] = useState(INITIAL_PROMPT);
	const [time, setTime] = useState(() => performance.now() / 1000);

	const [phrases, setPhrases] = useState<string[]>([]);
	const [usedPhrases, setUsedPhrases] = useState<Set<string>>(new Set());
	const [targetPhrase, setTargetPhrase] = useState('');

	const wpmChartRef = useRef<HTMLCanvasElement>(null);
	const [cpsChart, setCpsChart] = useState<chartjs.Chart | null>(null);

	// ------------------------------------------------------------------
	// Load phrases
	// ------------------------------------------------------------------

	useEffect(() => {
		fetch('/src/utils/phrases.txt')
			.then((r) => r.text())
			.then((text) => {
				const loaded = text
					.split('\n')
					.map((l) => l.trim())
					.filter((l) => l.length > 0);
				setPhrases(loaded);
				if (loaded.length > 0) {
					const available = loaded.slice(N_SKIP_PHRASES);
					const phrase = available[Math.floor(Math.random() * available.length)];
					setTargetPhrase(phrase.toLowerCase() + '$');
				}
			})
			.catch((e) => console.error('Failed to load phrases:', e));
	}, []);

	// ------------------------------------------------------------------
	// Cookie persistence
	// ------------------------------------------------------------------

	useEffect(() => {
		Cookies.set('username', username);
	}, [username]);

	// ------------------------------------------------------------------
	// 1 ms ticker (for timer-circle animation)
	// ------------------------------------------------------------------

	useEffect(() => {
		const interval = setInterval(() => {
			setTime(performance.now() / 1000);
		}, 1);
		return () => clearInterval(interval);
	}, []);

	// suppress unused warning – `time` drives canvas animations indirectly
	void time;

	// ------------------------------------------------------------------
	// Random phrase helper
	// ------------------------------------------------------------------

	const randomPhrase = useCallback((): string => {
		let available = phrases.slice(N_SKIP_PHRASES).filter((p) => !usedPhrases.has(p));
		if (available.length === 0) {
			setUsedPhrases(new Set());
			available = phrases.slice(N_SKIP_PHRASES);
		}
		const phrase = available[Math.floor(Math.random() * available.length)];
		setUsedPhrases((prev) => new Set([...prev, phrase]));
		return phrase.toLowerCase() + '$';
	}, [phrases, usedPhrases]);

	// ------------------------------------------------------------------
	// WPM chart
	// ------------------------------------------------------------------

	const updateCpsChart = useCallback(() => {
		if (!wpmChartRef.current) return;

		if (!cpsChart) {
			const existingChart = chartjs.Chart.getChart(wpmChartRef.current);
			existingChart?.destroy();

			chartjs.Chart.register(...chartjs.registerables);
			const chart = new chartjs.Chart(wpmChartRef.current, {
				type: 'line',
				data: {
					labels: [],
					datasets: [
						{
							label: 'Speed History',
							data: [],
							borderColor: '#3b82f6',
							backgroundColor: 'rgba(59, 130, 246, 0.1)',
							tension: 0.2,
							fill: true,
							borderWidth: 2,
						},
					],
				},
				options: {
					responsive: true,
					scales: {
						y: {
							beginAtZero: true,
							title: {
								display: true,
								text: 'Words per minute',
								color: '#1e293b',
								font: { weight: 500, size: 12 },
							},
							grid: { display: false },
							ticks: { color: '#475569', font: { size: 11 } },
						},
						x: {
							title: { display: false },
							grid: { display: false },
							ticks: { color: '#475569' },
						},
					},
					layout: { padding: 0 },
					plugins: {
						title: {
							display: true,
							text:
								trialsCps.length === 0
									? 'No attempts yet'
									: `Last: ${trialsCps[trialsCps.length - 1].toFixed(1)} WPM | Average: ${avgWpm.toFixed(2)} WPM`,
							padding: 8,
							color: '#1e293b',
							font: { size: 14, weight: 600 },
						},
						legend: { display: false },
					},
					animation: { duration: 300 },
				},
			});
			setCpsChart(chart);
		} else {
			cpsChart.data.labels = trialsCps.map(() => '');
			cpsChart.data.datasets[0].data = trialsCps;
			cpsChart.update();
		}
	}, [cpsChart, trialsCps, avgWpm]);

	useEffect(() => {
		updateCpsChart();
	}, [trialsCps, avgWpm, updateCpsChart]);

	// ------------------------------------------------------------------
	// WebSocket connection
	// ------------------------------------------------------------------

	useEffect(() => {
		const ws = new WebSocket('ws://localhost:8000/ws');

		ws.addEventListener('open', () => {
			setWebSocketStatusMsg('Connected');
			ws.send(JSON.stringify({ type: 'reset', prompt, username }));

			// Latency ping
			ws.send(JSON.stringify({ type: 'ping', pingTime: performance.now() }));
			const latencyInterval = setInterval(() => {
				if (ws.readyState === WebSocket.OPEN) {
					ws.send(JSON.stringify({ type: 'ping', pingTime: performance.now() }));
				}
			}, 2000);
			(ws as WebSocket & { _latencyInterval?: ReturnType<typeof setInterval> })._latencyInterval =
				latencyInterval;
		});

		ws.addEventListener('close', () => {
			setWebSocketStatusMsg('Disconnected');
			const wsx = ws as WebSocket & { _latencyInterval?: ReturnType<typeof setInterval> };
			if (wsx._latencyInterval) clearInterval(wsx._latencyInterval);
		});

		ws.addEventListener('error', () => {
			setWebSocketStatusMsg('Error: Connection failed');
		});

		ws.addEventListener('message', (event: MessageEvent) => {
			const response = JSON.parse(event.data as string) as {
				type: string;
				pingTime?: number;
				content?: Trial[] | string;
				ftp?: string;
				cum?: number[];
				stop_prob?: number;
				prior_ill?: number;
			};

			if (response.type === 'pong' && response.pingTime !== undefined) {
				const rtt = performance.now() - response.pingTime;
				setLatency(rtt);
			}

			if (response.type === 'log_info' && Array.isArray(response.content)) {
				const content = response.content as Trial[];
				setTrials(content);

				const delayPairs = content.flatMap((t) => t.delay_pairs ?? []);
				const idealStats = autoStats(delayPairs);
				setAutoCalibrationModel({
					mu_delay: idealStats.mu_est,
					stddev_delay: idealStats.sigma_est,
					outliers: idealStats.rho_est,
					period: idealStats.ideal_period_est,
				});

				const cps = content.map(
					(t) => (t.best_val.slice(0, -1).length / t.time_elapsed) * CPS_TO_WPM,
				);
				setTrialsCps(cps);

				const totalChars = content.reduce((acc, t) => acc + t.best_val.slice(0, -1).length, 0);
				const totalTime = content.reduce((acc, t) => acc + t.time_elapsed, 0);
				setAvgWpm((totalChars / totalTime) * CPS_TO_WPM);
			}

			if (
				response.type === 'processed' &&
				response.ftp !== undefined &&
				response.cum !== undefined &&
				response.stop_prob !== undefined &&
				response.prior_ill !== undefined
			) {
				const textAfterPrompt = response.ftp.slice(prompt.length);
				const entry: LmEntry = {
					cum: response.cum,
					stop_prob: response.stop_prob,
					prior_ill: response.prior_ill,
				};
				lmRef.current = { ...lmRef.current, [textAfterPrompt]: entry };

				trieLogic.update_prior_pipeline(trieRef.current, textAfterPrompt, lmRef.current, prefixRangePrecomp);
				const pDATA = trieRef.current.post_Z;
				trieLogic.set_viztrie_new(trieRef.current, lmRef.current, THRESHOLD, prefixRangePrecomp, pDATA);
				setTrieUpdatedFlag(true);
			}
		});

		setSocket(ws);

		return () => {
			const wsx = ws as WebSocket & { _latencyInterval?: ReturnType<typeof setInterval> };
			if (wsx._latencyInterval) clearInterval(wsx._latencyInterval);
			ws.close();
		};
	// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [prompt, username]);

	// ------------------------------------------------------------------
	// setLikelihoods (called from TrieVisualizer on click/blink)
	// ------------------------------------------------------------------

	const setLikelihoods = useCallback(
		(event: SetLikelihoodsEvent) => {
			const { new_likelihoods: newLikelihoods, click_seq: clickSeq } = event;

			if (socket?.readyState === WebSocket.OPEN) {
				socket.send(JSON.stringify({ type: 'timer_likelihoods', content: newLikelihoods }));
			}

			const newTrie = structuredClone(trieRef.current) as TrieNode;
			trieLogic.pushl_recalc_post_Z_new(newTrie, newLikelihoods);

			const pDATA = newTrie.post_Z;
			trieLogic.set_viztrie_new(newTrie, lmRef.current, THRESHOLD, prefixRangePrecomp, pDATA);
			trieRef.current = newTrie;
			setTrie(newTrie);
			setTrieUpdatedFlag(true);

			if (awaitingFirstKeypress) {
				setAwaitingFirstKeypress(false);
				setWpmStartTime(performance.now() / 1000);
			}

			const bestDescendant = trieLogic.get_best_descendant(newTrie);
			setBestVal(bestDescendant.val);

			if (
				bestDescendant.letter === '$' &&
				bestDescendant.post_Z - newTrie.post_Z > STOP_CONFIRM_THRESHOLD
			) {
				if (!confirmed) {
					setConfirmed(true);
					const now = performance.now() / 1000;
					const logPayload = {
						username,
						click_seq: clickSeq,
						best_val: bestDescendant.val,
						time_elapsed: wpmStartTime !== null ? now - wpmStartTime : 0,
						use_visual_tutor: useVisualTutor,
						target_phrase: targetPhrase,
						delay_pairs: bestDescendant.delay_pairs,
						timestamp: now,
					};
					if (socket?.readyState === WebSocket.OPEN) {
						socket.send(JSON.stringify({ type: 'log', content: logPayload }));
					}

					if (multilineMode) {
						const newPrompt = proposedPrompt + bestDescendant.val.slice(0, -1) + '\n';
						setProposedPrompt(newPrompt);
						setPrompt(newPrompt);
					}

					setTimeout(
						() => resetTrie(true),
						multilineMode ? 500 : 2000,
					);
				}
			}
		},
		// resetTrie is defined below and added to deps via useCallback
		// eslint-disable-next-line react-hooks/exhaustive-deps
		[socket, awaitingFirstKeypress, wpmStartTime, confirmed, username, useVisualTutor, targetPhrase, multilineMode, proposedPrompt],
	);

	// ------------------------------------------------------------------
	// resetTrie
	// ------------------------------------------------------------------

	const resetTrie = useCallback(
		(newPhrase = false) => {
			const newTrie = structuredClone(trieLogic.root_node) as TrieNode;
			trieRef.current = newTrie;
			lmRef.current = {};
			setTrie(newTrie);
			setTrieUpdatedFlag(true);
			if (newPhrase) setTargetPhrase(randomPhrase());
			setAwaitingFirstKeypress(true);
			setConfirmed(false);
			if (socket?.readyState === WebSocket.OPEN) {
				socket.send(JSON.stringify({ type: 'reset', prompt: proposedPrompt, username }));
			}
		},
		[randomPhrase, socket, proposedPrompt, username],
	);

	// Silence unused warning for `trials` / `bestVal` (used for future display)
	void trials;
	void bestVal;

	// ------------------------------------------------------------------
	// Render
	// ------------------------------------------------------------------

	return (
		<div className="flex flex-col h-screen bg-gray-900 box-border">
			{/* ---- Top bar ---- */}
			<div className="flex flex-row gap-4 w-full p-4 h-[190px] min-h-[190px] box-border">
				{/* Phrase + controls */}
				<div className="flex flex-col min-w-[300px] max-w-[400px] h-full">
					<div className="flex justify-between items-center mb-2">
						<div className="text-sm text-white truncate">
							{webSocketStatusMsg}
							{latency !== null && (
								<span className="text-gray-400">&nbsp;|&nbsp;{latency.toFixed(1)}ms</span>
							)}
						</div>
						<input
							type="text"
							placeholder="Enter username"
							value={username}
							onChange={(e) => setUsername(e.target.value)}
							className="w-32 px-2 py-1 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm text-white placeholder-gray-400 text-sm bg-gray-700"
						/>
					</div>

					<div className="flex-grow p-6 border border-gray-400 rounded-lg bg-white shadow-lg text-xl font-semibold flex items-center justify-center text-gray-800 overflow-auto mb-3">
						{targetPhrase.slice(0, -1)}
					</div>

					<div className="flex gap-3">
						<button
							className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 active:bg-blue-800 transition-colors font-medium shadow-sm text-sm"
							onClick={() => resetTrie(true)}
						>
							Next Phrase
						</button>
						<button
							className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 active:bg-gray-800 transition-colors font-medium shadow-sm text-sm"
							onClick={() => resetTrie(false)}
						>
							Retry
						</button>
					</div>
				</div>

				{/* WPM chart */}
				<div className="h-full w-[300px] bg-white rounded-lg shadow-lg flex-shrink-0">
					<canvas ref={wpmChartRef} width={300} height={170} />
				</div>

				{/* Calibration */}
				<CalibrationSettings
					useAutomaticCalibration={useAutomaticCalibration}
					setUseAutomaticCalibration={setUseAutomaticCalibration}
					likelihoodModel={likelihoodModel}
					setLikelihoodModel={setLikelihoodModel}
					autoCalibrationLikelihoodModel={autoCalibrationModel}
				/>

				{/* Prompt editor */}
				<div className="relative flex-shrink-0 h-full">
					<textarea
						value={proposedPrompt}
						onChange={(e) => {
							const val = e.target.value;
							setProposedPrompt(val);
							setPrompt(val);
							const freshTrie = structuredClone(trieLogic.root_node) as TrieNode;
							trieRef.current = freshTrie;
							setTrie(freshTrie);
							setTrieUpdatedFlag(true);
							setAwaitingFirstKeypress(true);
							if (socket?.readyState === WebSocket.OPEN) {
								socket.send(JSON.stringify({ type: 'reset', prompt: val, username }));
							}
						}}
						className="h-full w-72 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm shadow-sm bg-white text-gray-700 placeholder-gray-400"
						placeholder="Enter your prompt here..."
					/>
					<button
						className="absolute bottom-2 right-2 p-1 bg-gray-200 hover:bg-gray-300 rounded-md text-xs text-gray-700"
						onClick={() => setProposedPrompt(INITIAL_PROMPT)}
					>
						Reset
					</button>
				</div>
			</div>

			{/* ---- Canvas area ---- */}
			<div className="flex-1 min-h-0">
				<TrieVisualizer
					trie={trie}
					trieUpdatedFlag={trieUpdatedFlag}
					setTrieUpdatedFlag={setTrieUpdatedFlag}
					useVisualTutor={useVisualTutor}
					setUseVisualTutor={setUseVisualTutor}
					targetPhrase={targetPhrase}
					onSetLikelihoods={setLikelihoods}
					likelihoodModel={likelihoodModel}
					multilineMode={multilineMode}
					setMultilineMode={setMultilineMode}
				/>
			</div>
		</div>
	);
}

export default V2Page;
