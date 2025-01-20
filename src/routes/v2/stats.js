// estimating calibration parameters from an array of delays

function estimate_calibration_parameters(delay_pairs) {
    let delays = delay_pairs.map(dp => dp.delay);
    let periods = delay_pairs.map(dp => dp.period);
    console.log("analysing len(delays) = ", delays.length, "delays");
    delays.sort((a, b) => a - b);
    const middleIndex = Math.floor(delays.length / 2);
    const mu_est = delays.length % 2 === 0 ? (delays[middleIndex - 1] + delays[middleIndex]) / 2 : delays[middleIndex];
    let middleThreeFourths = delays.slice(Math.floor(delays.length / 8), Math.floor(delays.length * 7 / 8));
    console.log("mu_est: ", mu_est);
    console.log("Middle three-fourths of delays: ", middleThreeFourths, "len(middleThreeFourths) = ", middleThreeFourths.length);
    middleThreeFourths = middleThreeFourths.map(d => d - mu_est);
    // calculate sample variance
    let variance = middleThreeFourths.reduce((sum, d) => sum + d * d, 0) / (middleThreeFourths.length);
    let middle_three_fourths_stddev = Math.sqrt(variance);
    let sigma_est = middle_three_fourths_stddev / 0.607
    //
    let grid_size = 1000;
    let max_rho = 0.2;
    let n_grid_candidates = Math.floor(max_rho * grid_size);
    // estimate outliers (rho)
    let grid_search_candidates = Array(n_grid_candidates).fill(0);
    function normalProbability(x, mean, stddev) {
        return Math.exp(-0.5 * Math.pow((x - mean) / stddev, 2)) / (stddev * Math.sqrt(2 * Math.PI));
    }

    for (let i = 0; i < delays.length; i++) {
        let gaussian_p_i = normalProbability(delays[i], mu_est, sigma_est);
        let uniform_p_i = 1 / periods[i];
        for (let j = 0; j < grid_search_candidates.length; j++) {
            let rho = j / grid_size;
            grid_search_candidates[j] += Math.log(rho * uniform_p_i + (1 - rho) * gaussian_p_i);
        }
    }

    let best_rho = grid_search_candidates.indexOf(Math.max(...grid_search_candidates));
    let best_rho_value = grid_search_candidates[best_rho];
    let rho_est = best_rho / grid_size;
    console.log("rho_est: ", rho_est);
    console.log("best_rho_value: ", best_rho_value);
    // we linearly approximate the ideal period
    let t = (sigma_est - 0.030) / (0.120 - 0.030);
    t = Math.min(Math.max(t, 0), 1);
    let ideal_period_est = 1.000 * (1 - t) + 2.400 * t;

    return {mu_est, sigma_est, rho_est, ideal_period_est}
}

const default_stats = {mu_est: 0.025, sigma_est: 0.120, rho_est: 0.050, ideal_period_est: 2.400};

function auto_stats(delay_pairs) {
    if (delay_pairs.length < 40) {
        return default_stats;
    }
    // average over the last 200 delays (~10m of program usage)
    return estimate_calibration_parameters(delay_pairs.slice(Math.max(delay_pairs.length - 200, 0)));
}


export {estimate_calibration_parameters, auto_stats}