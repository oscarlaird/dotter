import { jStat } from 'jstat';

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

export interface VariationalParams {
	mu_m: number;
	sigma_m: number;
	mu_s: number;
	sigma_s: number;
	log_alpha: number;
	log_beta: number;
}

export interface CalibrationPair {
	x: number;
	period: number;
}

export const DEFAULT_PERIOD = 1.1;

export const DEFAULT_LIKELIHOOD_MODEL: LikelihoodModel = {
	mu_delay: 0.0,
	stddev_delay: 0.064,
	outliers: 0.08,
	period: DEFAULT_PERIOD,
};

export function predictiveStddev(muS: number, sigmaS: number, sigmaM: number): number {
	return Math.sqrt(Math.exp(muS + (sigmaS ** 2) / 2) + sigmaM ** 2);
}

export function variationalParamsToLikelihoodModel(
	params: VariationalParams,
	period: number,
): LikelihoodModel {
	const alpha = Math.exp(params.log_alpha);
	const beta = Math.exp(params.log_beta);
	return {
		mu_delay: params.mu_m,
		stddev_delay: predictiveStddev(params.mu_s, params.sigma_s, params.sigma_m),
		outliers: jStat.beta.inv(0.5, alpha, beta),
		period,
		intervals: {
			mu_delay: [params.mu_m - 1.96 * params.sigma_m, params.mu_m + 1.96 * params.sigma_m],
			stddev_delay: [
				predictiveStddev(params.mu_s - 1.96 * params.sigma_s, params.sigma_s, params.sigma_m),
				predictiveStddev(params.mu_s + 1.96 * params.sigma_s, params.sigma_s, params.sigma_m),
			],
			outliers: [jStat.beta.inv(0.025, alpha, beta), jStat.beta.inv(0.975, alpha, beta)],
		},
	};
}

function logaddexp(a: number, b: number): number {
	if (a === -Infinity) return b;
	if (b === -Infinity) return a;
	if (a > b) return a + Math.log(1 + Math.exp(b - a));
	return b + Math.log(1 + Math.exp(a - b));
}

function normalLogpdf(x: number, mean: number, stddev: number): number {
	return -0.5 * Math.pow((x - mean) / stddev, 2) - Math.log(stddev * Math.sqrt(2 * Math.PI));
}

export function timerLikelihood(time: number, phase: number, model: LikelihoodModel): number {
	const x = moduloDelay(time, phase, model.period);
	const outlierProb = Math.log(model.outliers) - Math.log(model.period);
	const notOutlierProb = Math.log(1 - model.outliers);
	const normalModes = [-1, 0, 1].map((k) =>
		normalLogpdf(x, model.mu_delay + k * model.period, model.stddev_delay),
	);

	let sumNormalModes = normalModes[0];
	for (let i = 1; i < normalModes.length; i += 1) {
		sumNormalModes = logaddexp(sumNormalModes, normalModes[i]);
	}

	return logaddexp(outlierProb, notOutlierProb + sumNormalModes);
}

export function moduloDelay(timeSeconds: number, phase: number, period: number): number {
	let x = timeSeconds - phase;
	x = ((x % period) + period) % period;
	return x;
}
