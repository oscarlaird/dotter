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

// Internal workspace using f32 and column-major blocked layout.
// Buffers with "block" in the name are laid out [j * phase_count + i]
// so that the inner loops over phases are contiguous.
struct Workspace {
    weights: Vec<f32>,
    alpha: Vec<f32>,
    base_mag2: Vec<f32>,
    fourier_modes: usize,
    phase_count: usize,
    cos_theta: Vec<f32>,
    sin_theta: Vec<f32>,
    block_cos: Vec<f32>, // [j * phase_count + i]
    block_sin: Vec<f32>,
    cos_base: Vec<f32>,
    sin_base: Vec<f32>,
    cos_batch: Vec<f32>, // [j * phase_count + i]
    sin_batch: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    grad_f32: Vec<f32>,
    mode1: Vec<f32>,
    mode2: Vec<f32>,
}

impl Workspace {
    fn new(params: &TimerSpacingParams, phase_count: usize) -> Self {
        let b8 = DEFAULT_BLOCK_SIZE;
        let f = params.fourier_modes;

        let mut alpha = vec![0.0f32; f + 1];
        let mut base_mag2 = vec![0.0f32; f + 1];
        for n in 0..=f {
            let n_f = n as f64;
            alpha[n] = (2.0 * std::f64::consts::PI * n_f / params.period) as f32;
            base_mag2[n] = (-2.0 * params.period.ln()
                - 4.0 * std::f64::consts::PI.powi(2) * n_f.powi(2) * params.sigma.powi(2)
                    / params.period.powi(2))
            .exp() as f32;
        }

        Self {
            weights: params.weights.iter().map(|&x| x as f32).collect(),
            alpha,
            base_mag2,
            fourier_modes: f,
            phase_count,
            cos_theta: vec![0.0; phase_count],
            sin_theta: vec![0.0; phase_count],
            block_cos: vec![0.0; b8 * phase_count],
            block_sin: vec![0.0; b8 * phase_count],
            cos_base: vec![0.0; phase_count],
            sin_base: vec![0.0; phase_count],
            cos_batch: vec![0.0; b8 * phase_count],
            sin_batch: vec![0.0; b8 * phase_count],
            a: vec![0.0; f + 1],
            b: vec![0.0; f + 1],
            grad_f32: vec![0.0; phase_count],
            mode1: vec![0.0; b8],
            mode2: vec![0.0; b8],
        }
    }
}

pub fn constant_phases(count: usize, period: f64) -> Vec<f64> {
    let step = period / count as f64;
    (0..count).map(|i| (i as f64 + 0.5) * step).collect()
}

// Fills workspace.grad_f32 and returns the loss.
// Uses f32 internals, sin_cos, column-major blocked layout, B=8 unrolled dot.
fn eval_inplace(phases: &[f64], workspace: &mut Workspace) -> f64 {
    let k = workspace.phase_count;
    // alpha[1] = 2π / P, so we can use it as the per-phase angle scale
    let phase_scale = workspace.alpha[1];

    for i in 0..k {
        let (s, c) = (phase_scale * phases[i] as f32).sin_cos();
        workspace.cos_theta[i] = c;
        workspace.sin_theta[i] = s;
        workspace.block_cos[i] = c; // m=0 column
        workspace.block_sin[i] = s;
    }

    // Build the block trig table in column-major order
    for m in 1..DEFAULT_BLOCK_SIZE {
        let prev_off = (m - 1) * k;
        let off = m * k;
        for i in 0..k {
            let pc = workspace.block_cos[prev_off + i];
            let ps = workspace.block_sin[prev_off + i];
            workspace.block_cos[off + i] = pc * workspace.cos_theta[i] - ps * workspace.sin_theta[i];
            workspace.block_sin[off + i] = ps * workspace.cos_theta[i] + pc * workspace.sin_theta[i];
        }
    }

    workspace.a[0] = workspace.weights.iter().sum();
    workspace.b[0] = 0.0;
    let mut loss = workspace.base_mag2[0] * workspace.a[0] * workspace.a[0];

    workspace.cos_base.fill(1.0);
    workspace.sin_base.fill(0.0);
    workspace.grad_f32.fill(0.0);

    for start in (1..=workspace.fourier_modes).step_by(DEFAULT_BLOCK_SIZE) {
        // Build batches in col-major order and fuse a/b accumulation
        let stop = (start + DEFAULT_BLOCK_SIZE).min(workspace.fourier_modes + 1);
        let width_pre = stop - start;
        for j in 0..width_pre {
            let off = j * k;
            let mut av = 0.0f32;
            let mut bv = 0.0f32;
            for i in 0..k {
                let cf = workspace.block_cos[off + i];
                let sf = workspace.block_sin[off + i];
                let c = workspace.cos_base[i] * cf - workspace.sin_base[i] * sf;
                let s = workspace.sin_base[i] * cf + workspace.cos_base[i] * sf;
                workspace.cos_batch[off + i] = c;
                workspace.sin_batch[off + i] = s;
                av += workspace.weights[i] * c;
                bv += workspace.weights[i] * s;
            }
            let idxm = start + j;
            workspace.a[idxm] = av;
            workspace.b[idxm] = bv;
            loss += 2.0 * workspace.base_mag2[idxm] * (av * av + bv * bv);
            workspace.mode1[j] = 4.0 * workspace.alpha[idxm] * workspace.base_mag2[idxm] * bv;
            workspace.mode2[j] = 4.0 * workspace.alpha[idxm] * workspace.base_mag2[idxm] * av;
        }

        // Gradient accumulation: unrolled for full B=8 blocks, generic fallback otherwise
        let width = (workspace.fourier_modes + 1).saturating_sub(start).min(DEFAULT_BLOCK_SIZE);
        if width == DEFAULT_BLOCK_SIZE {
            for i in 0..k {
                let acc1 = workspace.cos_batch[i] * workspace.mode1[0]
                    + workspace.cos_batch[k + i] * workspace.mode1[1]
                    + workspace.cos_batch[2 * k + i] * workspace.mode1[2]
                    + workspace.cos_batch[3 * k + i] * workspace.mode1[3]
                    + workspace.cos_batch[4 * k + i] * workspace.mode1[4]
                    + workspace.cos_batch[5 * k + i] * workspace.mode1[5]
                    + workspace.cos_batch[6 * k + i] * workspace.mode1[6]
                    + workspace.cos_batch[7 * k + i] * workspace.mode1[7];
                let acc2 = workspace.sin_batch[i] * workspace.mode2[0]
                    + workspace.sin_batch[k + i] * workspace.mode2[1]
                    + workspace.sin_batch[2 * k + i] * workspace.mode2[2]
                    + workspace.sin_batch[3 * k + i] * workspace.mode2[3]
                    + workspace.sin_batch[4 * k + i] * workspace.mode2[4]
                    + workspace.sin_batch[5 * k + i] * workspace.mode2[5]
                    + workspace.sin_batch[6 * k + i] * workspace.mode2[6]
                    + workspace.sin_batch[7 * k + i] * workspace.mode2[7];
                workspace.grad_f32[i] += workspace.weights[i] * (acc1 - acc2);
            }
        } else {
            for i in 0..k {
                let mut acc1 = 0.0f32;
                let mut acc2 = 0.0f32;
                for j in 0..width {
                    acc1 += workspace.cos_batch[j * k + i] * workspace.mode1[j];
                    acc2 += workspace.sin_batch[j * k + i] * workspace.mode2[j];
                }
                workspace.grad_f32[i] += workspace.weights[i] * (acc1 - acc2);
            }
        }

        let last_off = (width - 1) * k;
        for i in 0..k {
            workspace.cos_base[i] = workspace.cos_batch[last_off + i];
            workspace.sin_base[i] = workspace.sin_batch[last_off + i];
        }
    }

    loss as f64
}

pub fn loss_and_grad(phases: &[f64], params: &TimerSpacingParams) -> (f64, Vec<f64>) {
    let mut workspace = Workspace::new(params, phases.len());
    let loss = eval_inplace(phases, &mut workspace);
    let grad = workspace.grad_f32.iter().map(|&x| x as f64).collect();
    (loss, grad)
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
            f = eval_inplace(&x, &mut workspace);
            for (dst, src) in g.iter_mut().zip(workspace.grad_f32.iter()) {
                *dst = *src as f64;
            }
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
