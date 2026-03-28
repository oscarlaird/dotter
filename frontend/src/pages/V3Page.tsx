import { useEffect, useRef, useState } from 'react';
import initBayesianWasm, { BayesianSession } from '../wasm_pkg/bayesian';
import TrieSnapshotVisualizer from '../components/TrieSnapshotVisualizer';
import type { TrieSnapshot, TrieSnapshotNode } from '../components/TrieSnapshotVisualizer';

type SnapshotSymbol = TrieSnapshotNode['symbol'];

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
	const sessionRef = useRef<BayesianSession | null>(null);

	useEffect(() => {
		let cancelled = false;

		async function loadSnapshot() {
			try {
				setLoading(true);
				setError(null);
				await initBayesianWasm();
				if (!sessionRef.current) {
					sessionRef.current = new BayesianSession(Math.log(1 / 200), 100_000);
				}
				const snapshotJson = sessionRef.current.snapshot_json();
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
				const updatedSnapshotJson = sessionRef.current?.update_snapshot_likelihoods(
					JSON.stringify(nextSnapshot),
				);
				if (!updatedSnapshotJson) {
					throw new Error('BayesianSession is not initialized');
				}
				const updatedSnapshot = JSON.parse(updatedSnapshotJson) as TrieSnapshot;
				setSnapshot(updatedSnapshot);
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
