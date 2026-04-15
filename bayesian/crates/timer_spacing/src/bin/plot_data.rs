use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use timer_spacing::{constant_phases, optimize, TimerSpacingParams};

#[derive(Debug, Deserialize)]
struct PlotInput {
    weights: Vec<f64>,
    sigma: f64,
    period: f64,
    f: usize,
    iter_counts: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct PlotOutput {
    initial_phases: Vec<f64>,
    optimized: Vec<OptimizedEntry>,
}

#[derive(Debug, Serialize)]
struct OptimizedEntry {
    max_iter: u32,
    phases: Vec<f64>,
    loss: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let req: PlotInput = serde_json::from_str(&input)?;

    let k = req.weights.len();
    let params = TimerSpacingParams::new(req.weights, req.sigma, req.period)
        .with_fourier_modes(req.f);
    let initial_phases = constant_phases(k, params.period);

    let mut optimized = Vec::new();
    for &max_iter in &req.iter_counts {
        let result = optimize(&params, &initial_phases, max_iter)?;
        optimized.push(OptimizedEntry {
            max_iter,
            phases: result.phases,
            loss: result.loss,
        });
    }

    let out = PlotOutput { initial_phases, optimized };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
