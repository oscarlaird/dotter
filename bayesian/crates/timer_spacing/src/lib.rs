use lbfgsb::{setulb, LbfgsbParameter};

pub const DEFAULT_F: usize = 32;
pub const DEFAULT_BLOCK_SIZE: usize = 8;
pub const DEFAULT_MAX_ITER: u32 = 25;

const TASK_START: i64 = 1;
const TASK_NEW_X: i64 = 2;
const TASK_FG: i64 = 10;
const TASK_FG_END: i64 = 15;

#[derive(Debug, Clone)]
pub struct TimerSpacingParams {
    pub weights: Vec<f64>,
    pub sigma: f64,
    pub period: f64,
    pub fourier_modes: usize,
    pub block_size: usize,
    pub max_iterations: u32,
}

impl TimerSpacingParams {
    pub fn new(weights: Vec<f64>, sigma: f64, period: f64) -> Self {
        Self {
            weights,
            sigma,
            period,
            fourier_modes: DEFAULT_F,
            block_size: DEFAULT_BLOCK_SIZE,
            max_iterations: DEFAULT_MAX_ITER,
        }
    }

    pub fn with_fourier_modes(mut self, fourier_modes: usize) -> Self {
        self.fourier_modes = fourier_modes;
        self
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub phases: Vec<f64>,
    pub loss: f64,
}

#[derive(Debug, Clone)]
struct Workspace {
    weights: Vec<f64>,
    alpha: Vec<f64>,
    base_mag2: Vec<f64>,
    block_size: usize,
    theta: Vec<f64>,
    cos_theta: Vec<f64>,
    sin_theta: Vec<f64>,
    cos_base: Vec<f64>,
    sin_base: Vec<f64>,
    block_cos: Vec<f64>,
    block_sin: Vec<f64>,
    cos_batch: Vec<f64>,
    sin_batch: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    grad: Vec<f64>,
    mode_tmp1: Vec<f64>,
    mode_tmp2: Vec<f64>,
}

impl Workspace {
    fn new(params: &TimerSpacingParams, phase_count: usize) -> Self {
        let block_size = params.block_size.min(params.fourier_modes.max(1));
        let mut alpha = vec![0.0; params.fourier_modes + 1];
        let mut base_mag2 = vec![0.0; params.fourier_modes + 1];
        for n in 0..=params.fourier_modes {
            let n_f = n as f64;
            alpha[n] = 2.0 * std::f64::consts::PI * n_f / params.period;
            base_mag2[n] = (-2.0 * params.period.ln()
                - 4.0 * std::f64::consts::PI.powi(2) * n_f.powi(2) * params.sigma.powi(2)
                    / params.period.powi(2))
            .exp();
        }

        Self {
            weights: params.weights.clone(),
            alpha,
            base_mag2,
            block_size,
            theta: vec![0.0; phase_count],
            cos_theta: vec![0.0; phase_count],
            sin_theta: vec![0.0; phase_count],
            cos_base: vec![0.0; phase_count],
            sin_base: vec![0.0; phase_count],
            block_cos: vec![0.0; phase_count * block_size],
            block_sin: vec![0.0; phase_count * block_size],
            cos_batch: vec![0.0; phase_count * block_size],
            sin_batch: vec![0.0; phase_count * block_size],
            a: vec![0.0; params.fourier_modes + 1],
            b: vec![0.0; params.fourier_modes + 1],
            grad: vec![0.0; phase_count],
            mode_tmp1: vec![0.0; block_size],
            mode_tmp2: vec![0.0; block_size],
        }
    }
}

pub fn constant_phases(count: usize, period: f64) -> Vec<f64> {
    let step = period / count as f64;
    (0..count).map(|i| (i as f64 + 0.5) * step).collect()
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn prepare_block_trig(workspace: &mut Workspace, phases: &[f64], period: f64) {
    let n = phases.len();
    for i in 0..n {
        workspace.theta[i] = 2.0 * std::f64::consts::PI * phases[i] / period;
        workspace.cos_theta[i] = workspace.theta[i].cos();
        workspace.sin_theta[i] = workspace.theta[i].sin();
    }

    if workspace.block_size == 0 {
        return;
    }

    for i in 0..n {
        workspace.block_cos[i * workspace.block_size] = workspace.cos_theta[i];
        workspace.block_sin[i * workspace.block_size] = workspace.sin_theta[i];
    }

    for m in 1..workspace.block_size {
        for i in 0..n {
            let prev = i * workspace.block_size + (m - 1);
            let idx = i * workspace.block_size + m;
            let prev_cos = workspace.block_cos[prev];
            let prev_sin = workspace.block_sin[prev];
            let cos_theta = workspace.cos_theta[i];
            let sin_theta = workspace.sin_theta[i];
            workspace.block_cos[idx] = prev_cos * cos_theta - prev_sin * sin_theta;
            workspace.block_sin[idx] = prev_sin * cos_theta + prev_cos * sin_theta;
        }
    }
}

fn build_batches(workspace: &mut Workspace, width: usize) {
    let n = workspace.theta.len();
    for i in 0..n {
        let cos_base = workspace.cos_base[i];
        let sin_base = workspace.sin_base[i];
        let row = i * workspace.block_size;
        for j in 0..width {
            let cos_f = workspace.block_cos[row + j];
            let sin_f = workspace.block_sin[row + j];
            workspace.cos_batch[row + j] = cos_base * cos_f - sin_base * sin_f;
            workspace.sin_batch[row + j] = sin_base * cos_f + cos_base * sin_f;
        }
    }
}

fn loss_and_grad_for_phases_inplace(
    phases: &[f64],
    params: &TimerSpacingParams,
    workspace: &mut Workspace,
) -> f64 {
    let phase_count = phases.len();
    prepare_block_trig(workspace, phases, params.period);

    workspace.a[0] = workspace.weights.iter().sum();
    workspace.b[0] = 0.0;
    let mut loss = workspace.base_mag2[0] * workspace.a[0] * workspace.a[0];

    workspace.cos_base.fill(1.0);
    workspace.sin_base.fill(0.0);
    workspace.grad.fill(0.0);

    for start in (1..=params.fourier_modes).step_by(params.block_size) {
        let stop = (start + params.block_size).min(params.fourier_modes + 1);
        let width = stop - start;
        build_batches(workspace, width);

        for j in 0..width {
            let mut a_val = 0.0;
            let mut b_val = 0.0;
            for i in 0..phase_count {
                let idx = i * workspace.block_size + j;
                a_val += workspace.weights[i] * workspace.cos_batch[idx];
                b_val += workspace.weights[i] * workspace.sin_batch[idx];
            }
            workspace.a[start + j] = a_val;
            workspace.b[start + j] = b_val;
        }

        let mut block_loss = 0.0;
        for j in 0..width {
            let idx = start + j;
            block_loss += workspace.base_mag2[idx]
                * (workspace.a[idx] * workspace.a[idx] + workspace.b[idx] * workspace.b[idx]);
        }
        loss += 2.0 * block_loss;

        for j in 0..width {
            let idx = start + j;
            workspace.mode_tmp1[j] = 4.0 * workspace.alpha[idx] * workspace.base_mag2[idx] * workspace.b[idx];
            workspace.mode_tmp2[j] = 4.0 * workspace.alpha[idx] * workspace.base_mag2[idx] * workspace.a[idx];
        }

        for i in 0..phase_count {
            let row = i * workspace.block_size;
            let cos_vec = &workspace.cos_batch[row..row + width];
            let sin_vec = &workspace.sin_batch[row..row + width];
            workspace.grad[i] += workspace.weights[i]
                * (dot(cos_vec, &workspace.mode_tmp1[..width]) - dot(sin_vec, &workspace.mode_tmp2[..width]));
        }

        for i in 0..phase_count {
            let row = i * workspace.block_size;
            workspace.cos_base[i] = workspace.cos_batch[row + width - 1];
            workspace.sin_base[i] = workspace.sin_batch[row + width - 1];
        }
    }

    loss
}

pub fn loss_and_grad(phases: &[f64], params: &TimerSpacingParams) -> (f64, Vec<f64>) {
    let mut workspace = Workspace::new(params, phases.len());
    let loss = loss_and_grad_for_phases_inplace(phases, params, &mut workspace);
    (loss, workspace.grad.clone())
}

pub fn j(phases: &[f64], params: &TimerSpacingParams) -> f64 {
    loss_and_grad(phases, params).0
}

pub fn optimize(
    params: &TimerSpacingParams,
    initial_phases: &[f64],
    max_iter: u32,
) -> Result<OptimizationResult, String> {
    let mut x = initial_phases.to_vec();
    let mut workspace = Workspace::new(params, x.len());
    let n = x.len() as i64;
    let param = LbfgsbParameter::default();
    let m = param.m as i64;
    let mut f = 0.0_f64;
    let mut g = vec![0.0_f64; x.len()];
    let l = vec![0.0_f64; x.len()];
    let u = vec![0.0_f64; x.len()];
    let nbd = vec![0_i64; x.len()];
    let mut wa = vec![0.0_f64; 2 * param.m * x.len() + 5 * x.len() + 11 * param.m * param.m + 8 * param.m];
    let mut iwa = vec![0_i64; 3 * x.len()];
    let mut task = TASK_START;
    let mut csave = [0_i64; 60];
    let mut lsave = [0_i64; 4];
    let mut isave = [0_i64; 44];
    let mut dsave = [0.0_f64; 29];

    loop {
        unsafe {
            setulb(
                &n,
                &m,
                x.as_mut_ptr(),
                l.as_ptr(),
                u.as_ptr(),
                nbd.as_ptr(),
                &mut f,
                g.as_mut_ptr(),
                &param.factr,
                &param.pgtol,
                wa.as_mut_ptr(),
                iwa.as_mut_ptr(),
                &mut task,
                &param.iprint,
                csave.as_mut_ptr(),
                lsave.as_mut_ptr(),
                isave.as_mut_ptr(),
                dsave.as_mut_ptr(),
            );
        }

        if (TASK_FG..=TASK_FG_END).contains(&task) {
            f = loss_and_grad_for_phases_inplace(&x, params, &mut workspace);
            g.copy_from_slice(&workspace.grad);
        } else if task == TASK_NEW_X {
            if isave[29] >= max_iter as i64 {
                break;
            }
        } else {
            break;
        }
    }

    Ok(OptimizationResult {
        phases: x,
        loss: f,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimization_improves_loss() {
        let params = TimerSpacingParams::new(vec![0.5, 0.25, 0.25], 0.1, 1.0)
            .with_fourier_modes(10)
            .with_block_size(8)
            .with_max_iterations(25);
        let initial = constant_phases(3, params.period);
        let initial_loss = j(&initial, &params);
        let result = optimize(&params, &initial, params.max_iterations).expect("optimize");
        assert!(result.loss <= initial_loss);
    }
}
