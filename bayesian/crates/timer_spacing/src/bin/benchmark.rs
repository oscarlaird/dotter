use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::time::Instant;
use timer_spacing::{constant_phases, optimize, TimerSpacingParams};

fn softmax(values: &[f64]) -> Vec<f64> {
    let max_val = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut exp_values: Vec<f64> = values.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f64 = exp_values.iter().sum();
    for v in &mut exp_values {
        *v /= sum;
    }
    exp_values
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let k = 300;
    let period = 1.0;
    let sigma = 0.020;
    let f = timer_spacing::DEFAULT_F;
    let max_iter = timer_spacing::DEFAULT_MAX_ITER;
    let trials = 10;
    let mut rng = StdRng::seed_from_u64(0);

    let warm_logits: Vec<f64> = (0..k).map(|_| StandardNormal.sample(&mut rng)).collect();
    let warm_weights = softmax(&warm_logits);
    let warm_params = TimerSpacingParams::new(warm_weights, sigma, period)
        .with_fourier_modes(f)
        .with_max_iterations(max_iter);
    let warm_init = constant_phases(k, period);
    let _ = optimize(&warm_params, &warm_init, max_iter)?;

    let mut times = Vec::with_capacity(trials);
    for _ in 0..trials {
        let logits: Vec<f64> = (0..k).map(|_| StandardNormal.sample(&mut rng)).collect();
        let weights = softmax(&logits);
        let params = TimerSpacingParams::new(weights, sigma, period)
            .with_fourier_modes(f)
            .with_max_iterations(max_iter);
        let initial_phases = constant_phases(k, period);
        let start = Instant::now();
        let _ = optimize(&params, &initial_phases, max_iter)?;
        times.push(start.elapsed().as_secs_f64());
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    println!(
        "timer_spacing benchmark: K={k} F={f} sigma={sigma} max_iter={max_iter} trials={trials} mean_s={mean:.6} median_s={median:.6}"
    );
    Ok(())
}
