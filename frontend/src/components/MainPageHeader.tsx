import { SCROLL_CENTERING_WEIGHT, SCROLL_STABILITY_WEIGHT } from '../domain/trieLayout';

interface MainPageHeaderProps {
	wsStatus: string;
	snapshotNodeCount: number | null;
	lastBatchSize: number;
	showPracticePhrase: boolean;
	onShowPracticePhraseChange: (value: boolean) => void;
	practicePhrase: string;
	onShufflePracticePhrase: () => void;
	usernameInput: string;
	onUsernameInputChange: (value: string) => void;
	onStartSession: () => void;
	activeUsername: string | null;
	blinkToClick: boolean;
	onBlinkToClickChange: (value: boolean) => void;
	showAll: boolean;
	onShowAllChange: (value: boolean) => void;
	showDebugStats: boolean;
	onShowDebugStatsChange: (value: boolean) => void;
	showBoxes: boolean;
	onShowBoxesChange: (value: boolean) => void;
	showSpaceConnectors: boolean;
	onShowSpaceConnectorsChange: (value: boolean) => void;
	useVisualTutor: boolean;
	onUseVisualTutorChange: (value: boolean) => void;
	useAudioTutor: boolean;
	onUseAudioTutorChange: (value: boolean) => void;
	colorMode: 'light' | 'dark';
	onToggleColorMode: () => void;
	showCalibrationDebugPanel: boolean;
	onToggleCalibrationDebugPanel: () => void;
	showPredictionLogPanel: boolean;
	onTogglePredictionLogPanel: () => void;
	onDownloadSessionDebugDump: () => void;
	onReset: () => void;
}

function MainPageHeader({
	wsStatus,
	snapshotNodeCount,
	lastBatchSize,
	showPracticePhrase,
	onShowPracticePhraseChange,
	practicePhrase,
	onShufflePracticePhrase,
	usernameInput,
	onUsernameInputChange,
	onStartSession,
	activeUsername,
	blinkToClick,
	onBlinkToClickChange,
	showAll,
	onShowAllChange,
	showDebugStats,
	onShowDebugStatsChange,
	showBoxes,
	onShowBoxesChange,
	showSpaceConnectors,
	onShowSpaceConnectorsChange,
	useVisualTutor,
	onUseVisualTutorChange,
	useAudioTutor,
	onUseAudioTutorChange,
	colorMode,
	onToggleColorMode,
	showCalibrationDebugPanel,
	onToggleCalibrationDebugPanel,
	showPredictionLogPanel,
	onTogglePredictionLogPanel,
	onDownloadSessionDebugDump,
	onReset,
}: MainPageHeaderProps) {
	return (
		<header className="flex shrink-0 items-center justify-between gap-2 border-b border-slate-200 pb-2 dark:border-white/10">
			<div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-slate-500 dark:text-gray-400">
				<span>
					WS <code className="text-slate-800 dark:text-gray-300">{wsStatus}</code>
				</span>
				<span className="text-slate-300 dark:text-white/25">·</span>
				{snapshotNodeCount !== null ? (
					<span>
						<code className="text-slate-800 dark:text-gray-300">{snapshotNodeCount}</code> nodes
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
					scroll <code className="text-slate-800 dark:text-gray-300">a={SCROLL_CENTERING_WEIGHT}</code>{' '}
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
								onClick={onShufflePracticePhrase}
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
						onChange={(event) => onUsernameInputChange(event.target.value)}
						placeholder="username"
						className="w-32 rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-800 dark:border-white/20 dark:bg-white/10 dark:text-white"
					/>
					<button
						type="button"
						onClick={onStartSession}
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
						onChange={(event) => onBlinkToClickChange(event.target.checked)}
						className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
					/>
					Blink to click
				</label>
				<label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300">
					<input
						type="checkbox"
						checked={showAll}
						onChange={(event) => onShowAllChange(event.target.checked)}
						className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
					/>
					Show all
				</label>
				<label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300">
					<input
						type="checkbox"
						checked={showDebugStats}
						onChange={(event) => onShowDebugStatsChange(event.target.checked)}
						className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
					/>
					Debug
				</label>
				<label
					className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300"
					title="Semi-transparent rectangles behind each trie timer"
				>
					<input
						type="checkbox"
						checked={showBoxes}
						onChange={(event) => onShowBoxesChange(event.target.checked)}
						className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
					/>
					Node boxes
				</label>
				<label
					className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300"
					title="Lines from a space (word-boundary) node to its children"
				>
					<input
						type="checkbox"
						checked={showSpaceConnectors}
						onChange={(event) => onShowSpaceConnectorsChange(event.target.checked)}
						className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
					/>
					Space→child lines
				</label>
				<label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-slate-600 dark:text-gray-300">
					<input
						type="checkbox"
						checked={showPracticePhrase}
						onChange={(event) => onShowPracticePhraseChange(event.target.checked)}
						className="h-3.5 w-3.5 accent-blue-600 dark:accent-blue-500"
					/>
					Practice
				</label>
				<label
					className={`flex select-none items-center gap-1.5 text-xs ${
						showPracticePhrase
							? 'cursor-pointer text-slate-600 dark:text-gray-300'
							: 'cursor-not-allowed text-slate-400 dark:text-gray-500'
					}`}
				>
					<input
						type="checkbox"
						checked={useVisualTutor}
						disabled={!showPracticePhrase}
						onChange={(event) => onUseVisualTutorChange(event.target.checked)}
						className="h-3.5 w-3.5 accent-blue-600 disabled:cursor-not-allowed dark:accent-blue-500"
					/>
					Visual tutor
				</label>
				<label
					className={`flex select-none items-center gap-1.5 text-xs ${
						showPracticePhrase
							? 'cursor-pointer text-slate-600 dark:text-gray-300'
							: 'cursor-not-allowed text-slate-400 dark:text-gray-500'
					}`}
				>
					<input
						type="checkbox"
						checked={useAudioTutor}
						disabled={!showPracticePhrase}
						onChange={(event) => onUseAudioTutorChange(event.target.checked)}
						className="h-3.5 w-3.5 accent-blue-600 disabled:cursor-not-allowed dark:accent-blue-500"
					/>
					Audio tutor
				</label>
				<button
					type="button"
					onClick={onToggleColorMode}
					className="rounded border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-800 transition hover:bg-slate-50 dark:border-white/20 dark:bg-white/10 dark:text-white dark:hover:bg-white/20"
					aria-label={colorMode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
				>
					{colorMode === 'dark' ? 'Light' : 'Dark'}
				</button>
				<button
					type="button"
					onClick={onToggleCalibrationDebugPanel}
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
					onClick={onTogglePredictionLogPanel}
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
					onClick={onDownloadSessionDebugDump}
					className="rounded border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-800 transition hover:bg-slate-50 dark:border-white/20 dark:bg-white/10 dark:text-white dark:hover:bg-white/20"
				>
					Dump logs
				</button>
				<button
					type="button"
					onClick={onReset}
					className="rounded border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-800 transition hover:bg-slate-50 dark:border-white/20 dark:bg-white/10 dark:text-white dark:hover:bg-white/20"
				>
					Reset
				</button>
			</div>
		</header>
	);
}

export default MainPageHeader;
