export interface PredictionLogEntry {
	id: number;
	fullString: string;
	finalTokenLexindex: number;
	receivedAt: string;
}

interface PredictionLogPanelProps {
	entries: PredictionLogEntry[];
}

function PredictionLogPanel({ entries }: PredictionLogPanelProps) {
	return (
		<div className="shrink-0 rounded-lg border border-slate-200 bg-white/95 p-2 shadow-sm backdrop-blur-sm dark:border-white/10 dark:bg-white/5 dark:shadow-none">
			<div className="mb-1.5 flex shrink-0 items-center justify-between gap-2">
				<h2 className="text-xs font-semibold text-slate-800 dark:text-gray-100">
					Backend Prediction Log
				</h2>
				<span className="text-[0.65rem] text-slate-500 dark:text-gray-400">
					<code>{entries.length}</code> entries
				</span>
			</div>
			{entries.length === 0 ? (
				<p className="text-xs text-slate-500 dark:text-gray-400">
					No backend predictions received yet.
				</p>
			) : (
				<ul className="max-h-36 min-h-0 list-none space-y-1 overflow-y-auto overscroll-contain pr-1 text-xs">
					{entries.map((entry) => (
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
	);
}

export default PredictionLogPanel;
