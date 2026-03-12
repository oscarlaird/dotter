export interface DelayPair {
	delay: number;
	period: number;
}

export interface CalibrationStats {
	mu_est: number;
	sigma_est: number;
	rho_est: number;
	ideal_period_est: number;
}

function normalProbability(x: number, mean: number, stddev: number): number {
	return Math.exp(-0.5 * Math.pow((x - mean) / stddev, 2)) / (stddev * Math.sqrt(2 * Math.PI));
}

function estimateCalibrationParameters(delayPairs: DelayPair[]): CalibrationStats {
	const delays = delayPairs.map((dp) => dp.delay);
	const periods = delayPairs.map((dp) => dp.period);
	console.log('analysing len(delays) = ', delays.length, 'delays');
	delays.sort((a, b) => a - b);

	const middleIndex = Math.floor(delays.length / 2);
	const mu_est =
		delays.length % 2 === 0
			? (delays[middleIndex - 1] + delays[middleIndex]) / 2
			: delays[middleIndex];

	let middleThreeFourths = delays.slice(
		Math.floor(delays.length / 8),
		Math.floor((delays.length * 7) / 8),
	);
	console.log('mu_est: ', mu_est);
	console.log(
		'Middle three-fourths of delays: ',
		middleThreeFourths,
		'len(middleThreeFourths) = ',
		middleThreeFourths.length,
	);
	middleThreeFourths = middleThreeFourths.map((d) => d - mu_est);

	const variance =
		middleThreeFourths.reduce((sum, d) => sum + d * d, 0) / middleThreeFourths.length;
	const middleThreeFourthsStddev = Math.sqrt(variance);
	const sigma_est = middleThreeFourthsStddev / 0.607;

	const gridSize = 1000;
	const maxRho = 0.4;
	const nGridCandidates = Math.floor(maxRho * gridSize);
	const gridSearchCandidates = new Array<number>(nGridCandidates).fill(0);

	for (let i = 0; i < delays.length; i++) {
		const gaussianPi = normalProbability(delays[i], mu_est, sigma_est);
		const uniformPi = 1 / periods[i];
		for (let j = 0; j < gridSearchCandidates.length; j++) {
			const rho = j / gridSize;
			gridSearchCandidates[j] += Math.log(rho * uniformPi + (1 - rho) * gaussianPi);
		}
	}

	const bestRhoIdx = gridSearchCandidates.indexOf(Math.max(...gridSearchCandidates));
	const rho_est = bestRhoIdx / gridSize;
	console.log('rho_est: ', rho_est);

	const t = Math.min(Math.max((sigma_est - 0.03) / (0.12 - 0.03), 0), 1);
	const ideal_period_est = 1.0 * (1 - t) + 2.4 * t;

	return { mu_est, sigma_est, rho_est, ideal_period_est };
}

// Default stats for an experienced user
const defaultStats: CalibrationStats = {
	mu_est: 0.15,
	sigma_est: 0.04,
	rho_est: 0.03,
	ideal_period_est: 1.1,
};

function autoStats(delayPairs: DelayPair[]): CalibrationStats {
	if (delayPairs.length < 40) {
		return defaultStats;
	}
	return estimateCalibrationParameters(
		delayPairs.slice(Math.max(delayPairs.length - 200, 0)),
	);
}

export { estimateCalibrationParameters, autoStats };
