import { useEffect, useRef, useState } from 'react';
import initBayesianWasm, { BayesianSession } from '../wasm_pkg/bayesian';
import TrieSnapshotVisualizer from '../components/TrieSnapshotVisualizer';
import type { TrieSnapshot, TrieSnapshotNode } from '../components/TrieSnapshotVisualizer';

type SnapshotSymbol = TrieSnapshotNode['symbol'];
const DEFAULT_PROMPT = `my watch fell in the water
prevailing wind from the east
never too rich and never too thin
breathing is difficult
i can see the rings on saturn
`;

function keyToSnapshotSymbol(key: string): SnapshotSymbol | null {
	if (key === ' ') {
		return 'Space';
	}
	if (key === '$') {
		return 'Stop';
	}
	if (key.length === 1 && key >= 'a' && key <= 'z') {
		return key.toUpperCase() as SnapshotSymbol;
	}
	return null;
}

function withDummyLikelihoods(snapshot: TrieSnapshot, symbol: SnapshotSymbol): TrieSnapshot {
	return {
		...snapshot,
		nodes: snapshot.nodes.map((node) => ({
			...node,
			likelihood: node.symbol === symbol ? 0.0 : -2.0,
		})),
	};
}

function V3Page() {
	const [snapshot, setSnapshot] = useState<TrieSnapshot | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(true);
	const [lastKey, setLastKey] = useState<string | null>(null);
	const [wsStatus, setWsStatus] = useState('Connecting...');
	const sessionRef = useRef<BayesianSession | null>(null);
	const socketRef = useRef<WebSocket | null>(null);
	const threshold = Math.log(1 / 200);

	const refreshSnapshot = () => {
		const session = sessionRef.current;
		if (!session) {
			return;
		}
		const snapshotJson = session.snapshot_json_with_threshold(threshold);
		setSnapshot(JSON.parse(snapshotJson) as TrieSnapshot);
	};

	useEffect(() => {
		let cancelled = false;

		async function loadSnapshot() {
			try {
				setLoading(true);
				setError(null);
				await initBayesianWasm();
				if (!sessionRef.current) {
					sessionRef.current = new BayesianSession(threshold, 100_000);
				}
				const snapshotJson = sessionRef.current.snapshot_json_with_threshold(threshold);
				const parsedSnapshot = JSON.parse(snapshotJson) as TrieSnapshot;
				if (!cancelled) {
					setSnapshot(parsedSnapshot);
				}
			} catch (err) {
				if (!cancelled) {
					setError(err instanceof Error ? err.message : String(err));
				}
			} finally {
				if (!cancelled) {
					setLoading(false);
				}
			}
		}

		void loadSnapshot();
		return () => {
			cancelled = true;
		};
	}, [threshold]);

	useEffect(() => {
		const ws = new WebSocket('ws://localhost:8000/ws');
		socketRef.current = ws;

		ws.addEventListener('open', () => {
			setWsStatus('Connected');
			// Matches backend/new_lm.py (`content.prompt`, then `reset_ack`).
			ws.send(JSON.stringify({ type: 'reset', content: { prompt: DEFAULT_PROMPT } }));
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
					content?: {
						final_token?: string | null;
						full_string?: string;
						follower_logits?: number[];
						stop_logit?: number;
						message?: string;
					};
				};
				const session = sessionRef.current;
				if (!session) {
					return;
				}

				if (message.type === 'prior_update' && message.content) {
					const content = message.content;
					if (
						typeof content.full_string === 'string' &&
						Array.isArray(content.follower_logits) &&
						typeof content.stop_logit === 'number'
					) {
						session.apply_prior_update(
							content.final_token ?? null,
							content.full_string,
							new Float64Array(content.follower_logits),
							content.stop_logit,
						);
						refreshSnapshot();
					}
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
	}, []);

	useEffect(() => {
		if (!snapshot) {
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

			const symbol = keyToSnapshotSymbol(event.key.toLowerCase());
			if (!symbol) {
				return;
			}

			try {
				const nextSnapshot = withDummyLikelihoods(snapshot, symbol);
				const session = sessionRef.current;
				if (!session) {
					throw new Error('BayesianSession is not initialized');
				}
				const snapshotJson = JSON.stringify(nextSnapshot);
				session.apply_likelihood_update(snapshotJson);
				refreshSnapshot();
				const ws = socketRef.current;
				if (ws?.readyState === WebSocket.OPEN) {
					ws.send(
						JSON.stringify({
							type: 'likelihood_update',
							content: { snapshot_json: snapshotJson },
						}),
					);
				}
				setLastKey(event.key === ' ' ? 'Space' : event.key);
				setError(null);
			} catch (err) {
				setError(err instanceof Error ? err.message : String(err));
			}
		};

		window.addEventListener('keydown', handleKeyDown);
		return () => {
			window.removeEventListener('keydown', handleKeyDown);
		};
	}, [snapshot]);

	return (
		<div className="h-screen bg-gray-950 text-white p-6">
			<div className="mx-auto flex h-full max-w-7xl flex-col gap-4">
				<h1 className="text-3xl font-semibold">V3 Trie Snapshot</h1>
				<p className="text-sm text-gray-300">
					Threshold <code>ln(1/200)</code>, budget <code>100000</code>
				</p>
				<p className="text-sm text-gray-400">
					WebSocket: <code>{wsStatus}</code>
				</p>
				{snapshot && (
					<p className="text-sm text-gray-400">
						Showing <code>{snapshot.nodes.length}</code> visible nodes.
					</p>
				)}
				<p className="text-sm text-gray-400">
					Type a letter or space to apply dummy likelihoods.
					{lastKey && (
						<>
							{' '}
							Last key: <code>{lastKey}</code>
						</>
					)}
				</p>
				{loading && <div className="text-gray-300">Loading snapshot...</div>}
				{error && (
					<div className="rounded border border-red-500/40 bg-red-950/50 p-3 text-red-200">
						{error}
					</div>
				)}
				{!loading && !error && snapshot && (
					<div className="min-h-0 flex-1 overflow-hidden rounded-lg border border-white/10 bg-black/40">
						<TrieSnapshotVisualizer snapshot={snapshot} />
					</div>
				)}
			</div>
		</div>
	);
}

export default V3Page;
