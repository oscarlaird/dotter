import { useEffect } from 'react';

export interface LikelihoodModel {
	mu_delay: number;
	stddev_delay: number;
	outliers: number;
	period: number;
	intervals?: {
		mu_delay: [number, number];
		stddev_delay: [number, number];
		outliers: [number, number];
	};
}

export type AutoCalibrationState = {
	mu_delay: boolean;
	stddev_delay: boolean;
	outliers: boolean;
};

interface CalibrationSettingsProps {
	useAutomaticCalibration: AutoCalibrationState;
	setUseAutomaticCalibration: React.Dispatch<React.SetStateAction<AutoCalibrationState>>;
	likelihoodModel: LikelihoodModel;
	setLikelihoodModel: (updater: (prev: LikelihoodModel) => LikelihoodModel) => void;
	autoCalibrationLikelihoodModel: LikelihoodModel;
	calibrationSampleCount: number;
	rawVariationalParams: [number, number, number, number, number, number] | null;
	recentCalibrationPairs: Array<[number, number]>;
	showCalibrationDebug: boolean;
}

function CalibrationSettings({
	useAutomaticCalibration,
	setUseAutomaticCalibration,
	likelihoodModel,
	setLikelihoodModel,
	autoCalibrationLikelihoodModel,
	calibrationSampleCount,
	rawVariationalParams,
	recentCalibrationPairs,
	showCalibrationDebug,
}: CalibrationSettingsProps) {
	// Sync model from auto-calibration whenever it is enabled or updates
	useEffect(() => {
		setLikelihoodModel((prev) => {
			const next = { ...prev };
			if (useAutomaticCalibration.mu_delay) next.mu_delay = autoCalibrationLikelihoodModel.mu_delay;
			if (useAutomaticCalibration.stddev_delay) next.stddev_delay = autoCalibrationLikelihoodModel.stddev_delay;
			if (useAutomaticCalibration.outliers) next.outliers = autoCalibrationLikelihoodModel.outliers;
			return next;
		});
	}, [useAutomaticCalibration, autoCalibrationLikelihoodModel, setLikelihoodModel]);

	const handleSlider =
		(field: keyof LikelihoodModel) =>
		(e: React.ChangeEvent<HTMLInputElement>) => {
			setLikelihoodModel((prev) => ({ ...prev, [field]: parseFloat(e.target.value) }));
			if (field === 'mu_delay' || field === 'stddev_delay' || field === 'outliers') {
				setUseAutomaticCalibration((prev) => ({ ...prev, [field]: false }));
			}
		};

	return (
		<div className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-white/95 p-3 text-slate-900 shadow-sm backdrop-blur-sm dark:border-white/10 dark:bg-white/5 dark:text-white dark:shadow-none">
			<div className="flex items-center justify-between gap-3">
				<div>
					<div className="text-sm font-semibold text-slate-800 dark:text-gray-100">Calibration</div>
					<div className="text-xs text-slate-600 dark:text-gray-300">
						Calibration samples: {calibrationSampleCount}
					</div>
				</div>
				<div className="text-[0.65rem] uppercase tracking-[0.18em] text-slate-400 dark:text-gray-500">
					Auto-calibrated controls
				</div>
			</div>
			{showCalibrationDebug && rawVariationalParams && (
				<div className="rounded-md border border-slate-200 bg-slate-50 p-2 text-xs dark:border-white/8 dark:bg-black/20">
					<div className="mb-1 font-semibold text-slate-700 dark:text-gray-200">Variational params</div>
					<div className="font-mono text-slate-600 dark:text-gray-300">
						<div>{`[${rawVariationalParams.map((v) => v.toFixed(6)).join(', ')}]`}</div>
					</div>
				</div>
			)}
			{showCalibrationDebug && recentCalibrationPairs.length > 0 && (
				<div className="rounded-md border border-slate-200 bg-slate-50 p-2 text-xs dark:border-white/8 dark:bg-black/20">
					<div className="mb-1 font-semibold text-slate-700 dark:text-gray-200">Recent calibration pairs</div>
					<div className="font-mono text-slate-600 dark:text-gray-300">
						{recentCalibrationPairs.map(([x, period], idx) => (
							<div key={`${idx}-${x}-${period}`}>{`x=${x.toFixed(6)}, period=${period.toFixed(6)}`}</div>
						))}
					</div>
				</div>
			)}
			<div className="grid grid-cols-1 gap-2 md:grid-cols-4">
				<SliderRow
					label={`Mean (${(1000 * likelihoodModel.mu_delay).toFixed(0)}ms)`}
					min={-0.05}
					max={0.2}
					step={0.001}
					value={likelihoodModel.mu_delay}
					interval={autoCalibrationLikelihoodModel.intervals?.mu_delay}
					formatFn={(v) => `${(1000 * v).toFixed(0)}ms`}
					onChange={handleSlider('mu_delay')}
					autoCalibrate={{
						value: useAutomaticCalibration.mu_delay,
						onChange: (v) => setUseAutomaticCalibration((prev) => ({ ...prev, mu_delay: v }))
					}}
				/>
				<SliderRow
					label={`StdDev (${(1000 * likelihoodModel.stddev_delay).toFixed(0)}ms)`}
					min={0}
					max={0.15}
					step={0.001}
					value={likelihoodModel.stddev_delay}
					interval={autoCalibrationLikelihoodModel.intervals?.stddev_delay}
					formatFn={(v) => `${(1000 * v).toFixed(0)}ms`}
					onChange={handleSlider('stddev_delay')}
					autoCalibrate={{
						value: useAutomaticCalibration.stddev_delay,
						onChange: (v) => setUseAutomaticCalibration((prev) => ({ ...prev, stddev_delay: v }))
					}}
				/>
				<SliderRow
					label={`Outliers (${(100 * likelihoodModel.outliers).toFixed(1)}%)`}
					min={0}
					max={0.25}
					step={0.001}
					value={likelihoodModel.outliers}
					interval={autoCalibrationLikelihoodModel.intervals?.outliers}
					formatFn={(v) => `${(100 * v).toFixed(1)}%`}
					onChange={handleSlider('outliers')}
					autoCalibrate={{
						value: useAutomaticCalibration.outliers,
						onChange: (v) => setUseAutomaticCalibration((prev) => ({ ...prev, outliers: v }))
					}}
				/>
				<SliderRow
					label={`Period (${likelihoodModel.period.toFixed(2)}s)`}
					min={0.3}
					max={2.5}
					step={0.01}
					value={likelihoodModel.period}
					onChange={handleSlider('period')}
				/>
			</div>
		</div>
	);
}

interface SliderRowProps {
	label: string;
	min: number;
	max: number;
	step: number;
	value: number;
	interval?: [number, number];
	formatFn?: (val: number) => string;
	onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
	autoCalibrate?: {
		value: boolean;
		onChange: (v: boolean) => void;
	};
}

function SliderRow({ label, min, max, step, value, interval, formatFn, onChange, autoCalibrate }: SliderRowProps) {
	let intervalStyle = {};
	let intervalText = "";
	if (interval) {
		const [low, high] = interval;
		const startPct = Math.max(0, Math.min(100, ((low - min) / (max - min)) * 100));
		const endPct = Math.max(0, Math.min(100, ((high - min) / (max - min)) * 100));
		intervalStyle = {
			background: `linear-gradient(to right, transparent ${startPct}%, rgba(59, 130, 246, 0.4) ${startPct}%, rgba(59, 130, 246, 0.4) ${endPct}%, transparent ${endPct}%)`
		};
		if (formatFn) {
			intervalText = `[${formatFn(low)}, ${formatFn(high)}]`;
		}
	}

	return (
		<div className="flex flex-col gap-1 rounded-md border border-slate-200 bg-slate-50 p-2 dark:border-white/8 dark:bg-black/20">
			<div className="flex items-center justify-between">
				<label className="text-sm font-medium text-slate-700 dark:text-gray-200">{label}</label>
				<div className="flex items-center gap-2">
					{intervalText && <span className="text-xs text-slate-500 dark:text-gray-400">{intervalText}</span>}
					{autoCalibrate && (
						<input
							type="checkbox"
							checked={autoCalibrate.value}
							onChange={(e) => autoCalibrate.onChange(e.target.checked)}
							className="h-3 w-3 accent-blue-500 cursor-pointer"
							title="Auto-calibrate this parameter"
						/>
					)}
				</div>
			</div>
			<div className="relative flex items-center py-1">
				{interval && (
					<div
						className="absolute top-1/2 -translate-y-1/2 h-4 w-full rounded-sm pointer-events-none z-0"
						style={intervalStyle}
					/>
				)}
				<input
					type="range"
					min={min}
					max={max}
					step={step}
					value={value}
					onChange={onChange}
					className="relative z-10 w-full cursor-pointer accent-blue-500 opacity-90 transition-opacity hover:opacity-100 bg-transparent"
				/>
			</div>
		</div>
	);
}

export default CalibrationSettings;
