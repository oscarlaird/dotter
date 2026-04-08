import { useEffect } from 'react';

export interface LikelihoodModel {
	mu_delay: number;
	stddev_delay: number;
	outliers: number;
	period: number;
}

interface CalibrationSettingsProps {
	useAutomaticCalibration: boolean;
	setUseAutomaticCalibration: (value: boolean) => void;
	likelihoodModel: LikelihoodModel;
	setLikelihoodModel: (updater: (prev: LikelihoodModel) => LikelihoodModel) => void;
	autoCalibrationLikelihoodModel: LikelihoodModel;
}

function CalibrationSettings({
	useAutomaticCalibration,
	setUseAutomaticCalibration,
	likelihoodModel,
	setLikelihoodModel,
	autoCalibrationLikelihoodModel,
}: CalibrationSettingsProps) {
	// Sync model from auto-calibration whenever it is enabled or updates
	useEffect(() => {
		if (useAutomaticCalibration) {
			setLikelihoodModel(() => ({ ...autoCalibrationLikelihoodModel }));
		}
	}, [useAutomaticCalibration, autoCalibrationLikelihoodModel, setLikelihoodModel]);

	const handleSlider =
		(field: keyof LikelihoodModel) =>
		(e: React.ChangeEvent<HTMLInputElement>) => {
			setLikelihoodModel((prev) => ({ ...prev, [field]: parseFloat(e.target.value) }));
			setUseAutomaticCalibration(false);
		};

	return (
		<div className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-3 text-slate-900 shadow-sm dark:border-white/10 dark:bg-white/5 dark:text-white dark:shadow-none">
			<div className="flex items-center gap-2">
				<label className="text-sm font-semibold text-slate-800 dark:text-gray-100">Auto Calibration</label>
				<input
					type="checkbox"
					checked={useAutomaticCalibration}
					onChange={(e) => setUseAutomaticCalibration(e.target.checked)}
					className="h-4 w-4 accent-blue-500"
				/>
			</div>

			<div className="grid grid-cols-2 gap-2">
				<SliderRow
					label={`Mean (${(1000 * likelihoodModel.mu_delay).toFixed(0)}ms)`}
					min={-0.05}
					max={0.2}
					step={0.001}
					value={likelihoodModel.mu_delay}
					onChange={handleSlider('mu_delay')}
				/>
				<SliderRow
					label={`StdDev (${(1000 * likelihoodModel.stddev_delay).toFixed(0)}ms)`}
					min={0}
					max={0.15}
					step={0.001}
					value={likelihoodModel.stddev_delay}
					onChange={handleSlider('stddev_delay')}
				/>
				<SliderRow
					label={`Outliers (${(100 * likelihoodModel.outliers).toFixed(1)}%)`}
					min={0}
					max={0.25}
					step={0.001}
					value={likelihoodModel.outliers}
					onChange={handleSlider('outliers')}
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
	onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

function SliderRow({ label, min, max, step, value, onChange }: SliderRowProps) {
	return (
		<div className="flex flex-col gap-1 rounded-md border border-slate-200 bg-slate-50 p-2 dark:border-white/8 dark:bg-black/20">
			<label className="text-sm font-medium text-slate-700 dark:text-gray-200">{label}</label>
			<input
				type="range"
				min={min}
				max={max}
				step={step}
				value={value}
				onChange={onChange}
				className="w-full accent-blue-500"
			/>
		</div>
	);
}

export default CalibrationSettings;
