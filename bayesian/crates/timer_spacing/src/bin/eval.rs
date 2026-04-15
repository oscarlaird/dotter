use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use timer_spacing::{constant_phases, j, loss_and_grad, optimize, TimerSpacingParams};

#[derive(Debug, Deserialize)]
struct EvalInput {
    weights: Vec<f64>,
    sigma: f64,
    period: f64,
    f: usize,
    max_iter: u32,
    initial_phases: Option<Vec<f64>>,
}

#[derive(Debug, Serialize)]
struct EvalOutput {
    initial_loss: f64,
    final_loss: f64,
    phases: Vec<f64>,
    initial_grad: Vec<f64>,
    final_grad: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let req: EvalInput = serde_json::from_str(&input)?;

    let params = TimerSpacingParams::new(req.weights, req.sigma, req.period)
        .with_fourier_modes(req.f)
        .with_max_iterations(req.max_iter);
    let initial_phases = req
        .initial_phases
        .unwrap_or_else(|| constant_phases(params.weights.len(), params.period));

    let initial_loss = j(&initial_phases, &params);
    let (_, initial_grad) = loss_and_grad(&initial_phases, &params);
    let result = optimize(&params, &initial_phases, req.max_iter)?;
    let (_, final_grad) = loss_and_grad(&result.phases, &params);
    let out = EvalOutput {
        initial_loss,
        final_loss: result.loss,
        phases: result.phases.clone(),
        initial_grad,
        final_grad,
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
