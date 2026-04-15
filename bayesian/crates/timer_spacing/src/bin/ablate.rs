use lbfgsb::{setulb, LbfgsbParameter};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use timer_spacing::{constant_phases, optimize, TimerSpacingParams, DEFAULT_BLOCK_SIZE, DEFAULT_F, DEFAULT_MAX_ITER};

const TASK_START: i64 = 1;
const TASK_NEW_X: i64 = 2;
const TASK_FG: i64 = 10;
const TASK_FG_END: i64 = 15;

const K: usize = 300;
const P: f64 = 1.0;
const SIGMA: f64 = 0.02;
const F: usize = DEFAULT_F;
const MAX_ITER: u32 = DEFAULT_MAX_ITER;
const TRIALS: usize = 5;

#[derive(Clone)]
struct Case {
    weights: Vec<f64>,
    initial_phases: Vec<f64>,
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_values: Vec<f64> = values.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f64 = exp_values.iter().sum();
    for v in &mut exp_values {
        *v /= sum;
    }
    exp_values
}

fn sample_cases() -> Vec<Case> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..TRIALS)
        .map(|_| {
            let logits: Vec<f64> = (0..K).map(|_| StandardNormal.sample(&mut rng)).collect();
            Case {
                weights: softmax(&logits),
                initial_phases: constant_phases(K, P),
            }
        })
        .collect()
}

fn optimize_with_eval<E>(initial_phases: &[f64], max_iter: u32, mut eval: E) -> (Vec<f64>, f64)
where
    E: FnMut(&[f64], &mut [f64]) -> f64,
{
    let mut x = initial_phases.to_vec();
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
            f = eval(&x, &mut g);
        } else if task == TASK_NEW_X {
            if isave[29] >= max_iter as i64 {
                break;
            }
        } else {
            break;
        }
    }
    (x, f)
}

fn precompute_base() -> (Vec<f64>, Vec<f64>) {
    let mut alpha = vec![0.0; F + 1];
    let mut base_mag2 = vec![0.0; F + 1];
    for n in 0..=F {
        let n_f = n as f64;
        alpha[n] = 2.0 * std::f64::consts::PI * n_f / P;
        base_mag2[n] = (-2.0 * P.ln() - 4.0 * std::f64::consts::PI.powi(2) * n_f.powi(2) * SIGMA.powi(2) / P.powi(2)).exp();
    }
    (alpha, base_mag2)
}

fn optimize_f32(weights: &[f64], initial_phases: &[f64]) -> (Vec<f64>, f64) {
    let (alpha64, base64) = precompute_base();
    let alpha: Vec<f32> = alpha64.iter().map(|x| *x as f32).collect();
    let base_mag2: Vec<f32> = base64.iter().map(|x| *x as f32).collect();
    let weights32: Vec<f32> = weights.iter().map(|x| *x as f32).collect();
    let block = DEFAULT_BLOCK_SIZE;
    let mut theta = vec![0f32; K];
    let mut cos_theta = vec![0f32; K];
    let mut sin_theta = vec![0f32; K];
    let mut block_cos = vec![0f32; K * block];
    let mut block_sin = vec![0f32; K * block];
    let mut cos_base = vec![0f32; K];
    let mut sin_base = vec![0f32; K];
    let mut cos_batch = vec![0f32; K * block];
    let mut sin_batch = vec![0f32; K * block];
    let mut a = vec![0f32; F + 1];
    let mut b = vec![0f32; F + 1];
    let mut grad = vec![0f32; K];
    let mut mode1 = vec![0f32; block];
    let mut mode2 = vec![0f32; block];

    optimize_with_eval(initial_phases, MAX_ITER, |x, gx| {
        for i in 0..K {
            theta[i] = (2.0f32 * std::f32::consts::PI / P as f32) * x[i] as f32;
            cos_theta[i] = theta[i].cos();
            sin_theta[i] = theta[i].sin();
        }
        for i in 0..K {
            block_cos[i * block] = cos_theta[i];
            block_sin[i * block] = sin_theta[i];
        }
        for m in 1..block {
            for i in 0..K {
                let prev = i * block + (m - 1);
                let idx = i * block + m;
                let prev_cos = block_cos[prev];
                let prev_sin = block_sin[prev];
                block_cos[idx] = prev_cos * cos_theta[i] - prev_sin * sin_theta[i];
                block_sin[idx] = prev_sin * cos_theta[i] + prev_cos * sin_theta[i];
            }
        }
        a[0] = weights32.iter().sum();
        b[0] = 0.0;
        let mut loss = base_mag2[0] * a[0] * a[0];
        cos_base.fill(1.0);
        sin_base.fill(0.0);
        grad.fill(0.0);
        for start in (1..=F).step_by(block) {
            let stop = (start + block).min(F + 1);
            let width = stop - start;
            for i in 0..K {
                let row = i * block;
                let cb = cos_base[i];
                let sb = sin_base[i];
                for j in 0..width {
                    let cf = block_cos[row + j];
                    let sf = block_sin[row + j];
                    cos_batch[row + j] = cb * cf - sb * sf;
                    sin_batch[row + j] = sb * cf + cb * sf;
                }
            }
            for j in 0..width {
                let mut av = 0.0f32;
                let mut bv = 0.0f32;
                for i in 0..K {
                    let idx = i * block + j;
                    av += weights32[i] * cos_batch[idx];
                    bv += weights32[i] * sin_batch[idx];
                }
                a[start + j] = av;
                b[start + j] = bv;
                loss += 2.0 * base_mag2[start + j] * (av * av + bv * bv);
                mode1[j] = 4.0 * alpha[start + j] * base_mag2[start + j] * b[start + j];
                mode2[j] = 4.0 * alpha[start + j] * base_mag2[start + j] * a[start + j];
            }
            for i in 0..K {
                let row = i * block;
                let mut acc1 = 0.0f32;
                let mut acc2 = 0.0f32;
                for j in 0..width {
                    acc1 += cos_batch[row + j] * mode1[j];
                    acc2 += sin_batch[row + j] * mode2[j];
                }
                grad[i] += weights32[i] * (acc1 - acc2);
                cos_base[i] = cos_batch[row + width - 1];
                sin_base[i] = sin_batch[row + width - 1];
            }
        }
        for (dst, src) in gx.iter_mut().zip(grad.iter()) {
            *dst = *src as f64;
        }
        loss as f64
    })
}

fn optimize_sincos(weights: &[f64], initial_phases: &[f64]) -> (Vec<f64>, f64) {
    let (alpha, base_mag2) = precompute_base();
    let block = DEFAULT_BLOCK_SIZE;
    let mut theta = vec![0.0; K];
    let mut cos_theta = vec![0.0; K];
    let mut sin_theta = vec![0.0; K];
    let mut block_cos = vec![0.0; K * block];
    let mut block_sin = vec![0.0; K * block];
    let mut cos_base = vec![0.0; K];
    let mut sin_base = vec![0.0; K];
    let mut cos_batch = vec![0.0; K * block];
    let mut sin_batch = vec![0.0; K * block];
    let mut a = vec![0.0; F + 1];
    let mut b = vec![0.0; F + 1];
    let mut grad = vec![0.0; K];
    let mut mode1 = vec![0.0; block];
    let mut mode2 = vec![0.0; block];

    optimize_with_eval(initial_phases, MAX_ITER, |x, gx| {
        for i in 0..K {
            theta[i] = 2.0 * std::f64::consts::PI * x[i] / P;
            let (s, c) = theta[i].sin_cos();
            cos_theta[i] = c;
            sin_theta[i] = s;
        }
        for i in 0..K {
            block_cos[i * block] = cos_theta[i];
            block_sin[i * block] = sin_theta[i];
        }
        for m in 1..block {
            for i in 0..K {
                let prev = i * block + (m - 1);
                let idx = i * block + m;
                let prev_cos = block_cos[prev];
                let prev_sin = block_sin[prev];
                block_cos[idx] = prev_cos * cos_theta[i] - prev_sin * sin_theta[i];
                block_sin[idx] = prev_sin * cos_theta[i] + prev_cos * sin_theta[i];
            }
        }
        a[0] = weights.iter().sum();
        b[0] = 0.0;
        let mut loss = base_mag2[0] * a[0] * a[0];
        cos_base.fill(1.0);
        sin_base.fill(0.0);
        grad.fill(0.0);
        for start in (1..=F).step_by(block) {
            let stop = (start + block).min(F + 1);
            let width = stop - start;
            for i in 0..K {
                let row = i * block;
                let cb = cos_base[i];
                let sb = sin_base[i];
                for j in 0..width {
                    let cf = block_cos[row + j];
                    let sf = block_sin[row + j];
                    cos_batch[row + j] = cb * cf - sb * sf;
                    sin_batch[row + j] = sb * cf + cb * sf;
                }
            }
            for j in 0..width {
                let mut av = 0.0;
                let mut bv = 0.0;
                for i in 0..K {
                    let idx = i * block + j;
                    av += weights[i] * cos_batch[idx];
                    bv += weights[i] * sin_batch[idx];
                }
                a[start + j] = av;
                b[start + j] = bv;
                loss += 2.0 * base_mag2[start + j] * (av * av + bv * bv);
                mode1[j] = 4.0 * alpha[start + j] * base_mag2[start + j] * b[start + j];
                mode2[j] = 4.0 * alpha[start + j] * base_mag2[start + j] * a[start + j];
            }
            for i in 0..K {
                let row = i * block;
                let mut acc1 = 0.0;
                let mut acc2 = 0.0;
                for j in 0..width {
                    acc1 += cos_batch[row + j] * mode1[j];
                    acc2 += sin_batch[row + j] * mode2[j];
                }
                grad[i] += weights[i] * (acc1 - acc2);
                cos_base[i] = cos_batch[row + width - 1];
                sin_base[i] = sin_batch[row + width - 1];
            }
        }
        gx.copy_from_slice(&grad);
        loss
    })
}

fn optimize_specialized_b8(weights: &[f64], initial_phases: &[f64]) -> (Vec<f64>, f64) {
    let (alpha, base_mag2) = precompute_base();
    let block = 8usize;
    let mut theta = vec![0.0; K];
    let mut cos_theta = vec![0.0; K];
    let mut sin_theta = vec![0.0; K];
    let mut block_cos = vec![0.0; K * block];
    let mut block_sin = vec![0.0; K * block];
    let mut cos_base = vec![0.0; K];
    let mut sin_base = vec![0.0; K];
    let mut cos_batch = vec![0.0; K * block];
    let mut sin_batch = vec![0.0; K * block];
    let mut a = vec![0.0; F + 1];
    let mut b = vec![0.0; F + 1];
    let mut grad = vec![0.0; K];
    let mut mode1 = [0.0; 8];
    let mut mode2 = [0.0; 8];

    optimize_with_eval(initial_phases, MAX_ITER, |x, gx| {
        for i in 0..K {
            theta[i] = 2.0 * std::f64::consts::PI * x[i] / P;
            cos_theta[i] = theta[i].cos();
            sin_theta[i] = theta[i].sin();
            block_cos[i * block] = cos_theta[i];
            block_sin[i * block] = sin_theta[i];
        }
        for m in 1..block {
            for i in 0..K {
                let prev = i * block + (m - 1);
                let idx = i * block + m;
                let prev_cos = block_cos[prev];
                let prev_sin = block_sin[prev];
                block_cos[idx] = prev_cos * cos_theta[i] - prev_sin * sin_theta[i];
                block_sin[idx] = prev_sin * cos_theta[i] + prev_cos * sin_theta[i];
            }
        }
        a[0] = weights.iter().sum();
        b[0] = 0.0;
        let mut loss = base_mag2[0] * a[0] * a[0];
        cos_base.fill(1.0);
        sin_base.fill(0.0);
        grad.fill(0.0);
        for start in (1..=F).step_by(block) {
            for i in 0..K {
                let row = i * block;
                let cb = cos_base[i];
                let sb = sin_base[i];
                for j in 0..block {
                    let cf = block_cos[row + j];
                    let sf = block_sin[row + j];
                    cos_batch[row + j] = cb * cf - sb * sf;
                    sin_batch[row + j] = sb * cf + cb * sf;
                }
            }
            for j in 0..block {
                let idxm = start + j;
                let mut av = 0.0;
                let mut bv = 0.0;
                for i in 0..K {
                    let idx = i * block + j;
                    av += weights[i] * cos_batch[idx];
                    bv += weights[i] * sin_batch[idx];
                }
                a[idxm] = av;
                b[idxm] = bv;
                loss += 2.0 * base_mag2[idxm] * (av * av + bv * bv);
                mode1[j] = 4.0 * alpha[idxm] * base_mag2[idxm] * b[idxm];
                mode2[j] = 4.0 * alpha[idxm] * base_mag2[idxm] * a[idxm];
            }
            for i in 0..K {
                let row = i * block;
                let acc1 =
                    cos_batch[row] * mode1[0] + cos_batch[row + 1] * mode1[1] + cos_batch[row + 2] * mode1[2]
                        + cos_batch[row + 3] * mode1[3] + cos_batch[row + 4] * mode1[4]
                        + cos_batch[row + 5] * mode1[5] + cos_batch[row + 6] * mode1[6]
                        + cos_batch[row + 7] * mode1[7];
                let acc2 =
                    sin_batch[row] * mode2[0] + sin_batch[row + 1] * mode2[1] + sin_batch[row + 2] * mode2[2]
                        + sin_batch[row + 3] * mode2[3] + sin_batch[row + 4] * mode2[4]
                        + sin_batch[row + 5] * mode2[5] + sin_batch[row + 6] * mode2[6]
                        + sin_batch[row + 7] * mode2[7];
                grad[i] += weights[i] * (acc1 - acc2);
                cos_base[i] = cos_batch[row + 7];
                sin_base[i] = sin_batch[row + 7];
            }
        }
        gx.copy_from_slice(&grad);
        loss
    })
}

fn optimize_col_major(weights: &[f64], initial_phases: &[f64]) -> (Vec<f64>, f64) {
    let (alpha, base_mag2) = precompute_base();
    let block = DEFAULT_BLOCK_SIZE;
    let mut theta = vec![0.0; K];
    let mut cos_theta = vec![0.0; K];
    let mut sin_theta = vec![0.0; K];
    let mut block_cos = vec![0.0; block * K];
    let mut block_sin = vec![0.0; block * K];
    let mut cos_base = vec![0.0; K];
    let mut sin_base = vec![0.0; K];
    let mut cos_batch = vec![0.0; block * K];
    let mut sin_batch = vec![0.0; block * K];
    let mut a = vec![0.0; F + 1];
    let mut b = vec![0.0; F + 1];
    let mut grad = vec![0.0; K];
    let mut mode1 = vec![0.0; block];
    let mut mode2 = vec![0.0; block];

    optimize_with_eval(initial_phases, MAX_ITER, |x, gx| {
        for i in 0..K {
            theta[i] = 2.0 * std::f64::consts::PI * x[i] / P;
            cos_theta[i] = theta[i].cos();
            sin_theta[i] = theta[i].sin();
            block_cos[i] = cos_theta[i];
            block_sin[i] = sin_theta[i];
        }
        for m in 1..block {
            let prev_off = (m - 1) * K;
            let off = m * K;
            for i in 0..K {
                let prev_cos = block_cos[prev_off + i];
                let prev_sin = block_sin[prev_off + i];
                block_cos[off + i] = prev_cos * cos_theta[i] - prev_sin * sin_theta[i];
                block_sin[off + i] = prev_sin * cos_theta[i] + prev_cos * sin_theta[i];
            }
        }
        a[0] = weights.iter().sum();
        b[0] = 0.0;
        let mut loss = base_mag2[0] * a[0] * a[0];
        cos_base.fill(1.0);
        sin_base.fill(0.0);
        grad.fill(0.0);
        for start in (1..=F).step_by(block) {
            let stop = (start + block).min(F + 1);
            let width = stop - start;
            for j in 0..width {
                let off = j * K;
                let mut av = 0.0;
                let mut bv = 0.0;
                for i in 0..K {
                    let cf = block_cos[off + i];
                    let sf = block_sin[off + i];
                    let c = cos_base[i] * cf - sin_base[i] * sf;
                    let s = sin_base[i] * cf + cos_base[i] * sf;
                    cos_batch[off + i] = c;
                    sin_batch[off + i] = s;
                    av += weights[i] * c;
                    bv += weights[i] * s;
                }
                a[start + j] = av;
                b[start + j] = bv;
                loss += 2.0 * base_mag2[start + j] * (av * av + bv * bv);
                mode1[j] = 4.0 * alpha[start + j] * base_mag2[start + j] * bv;
                mode2[j] = 4.0 * alpha[start + j] * base_mag2[start + j] * av;
            }
            for i in 0..K {
                let mut acc1 = 0.0;
                let mut acc2 = 0.0;
                for j in 0..width {
                    let off = j * K;
                    acc1 += cos_batch[off + i] * mode1[j];
                    acc2 += sin_batch[off + i] * mode2[j];
                }
                grad[i] += weights[i] * (acc1 - acc2);
                let off = (width - 1) * K;
                cos_base[i] = cos_batch[off + i];
                sin_base[i] = sin_batch[off + i];
            }
        }
        gx.copy_from_slice(&grad);
        loss
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cases = sample_cases();

    let baseline_params = TimerSpacingParams::new(cases[0].weights.clone(), SIGMA, P)
        .with_fourier_modes(F)
        .with_block_size(DEFAULT_BLOCK_SIZE)
        .with_max_iterations(MAX_ITER);
    let _ = optimize(&baseline_params, &cases[0].initial_phases, MAX_ITER)?;

    let mut baseline_times = Vec::new();
    let mut baseline_losses = Vec::new();
    for case in &cases {
        let params = TimerSpacingParams::new(case.weights.clone(), SIGMA, P)
            .with_fourier_modes(F)
            .with_block_size(DEFAULT_BLOCK_SIZE)
            .with_max_iterations(MAX_ITER);
        let start = std::time::Instant::now();
        let result = optimize(&params, &case.initial_phases, MAX_ITER)?;
        baseline_times.push(start.elapsed().as_secs_f64());
        baseline_losses.push(result.loss);
    }

    let mut fp32_times = Vec::new();
    let mut fp32_losses = Vec::new();
    for case in &cases {
        let start = std::time::Instant::now();
        let (_, loss) = optimize_f32(&case.weights, &case.initial_phases);
        fp32_times.push(start.elapsed().as_secs_f64());
        fp32_losses.push(loss);
    }

    let mut sincos_times = Vec::new();
    let mut sincos_losses = Vec::new();
    for case in &cases {
        let start = std::time::Instant::now();
        let (_, loss) = optimize_sincos(&case.weights, &case.initial_phases);
        sincos_times.push(start.elapsed().as_secs_f64());
        sincos_losses.push(loss);
    }

    let mut spec_times = Vec::new();
    let mut spec_losses = Vec::new();
    for case in &cases {
        let start = std::time::Instant::now();
        let (_, loss) = optimize_specialized_b8(&case.weights, &case.initial_phases);
        spec_times.push(start.elapsed().as_secs_f64());
        spec_losses.push(loss);
    }

    let mut col_times = Vec::new();
    let mut col_losses = Vec::new();
    for case in &cases {
        let start = std::time::Instant::now();
        let (_, loss) = optimize_col_major(&case.weights, &case.initial_phases);
        col_times.push(start.elapsed().as_secs_f64());
        col_losses.push(loss);
    }

    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    println!("ABLATION K={K} F={F} sigma={SIGMA} max_iter={MAX_ITER}");
    println!("baseline_f64_cached: {:.6}s loss={:.9}", mean(&baseline_times), mean(&baseline_losses));
    println!(
        "fp32_internal:       {:.6}s loss={:.9} gap={:.3e}",
        mean(&fp32_times),
        mean(&fp32_losses),
        mean(&fp32_losses) - mean(&baseline_losses),
    );
    println!(
        "sincos:              {:.6}s loss={:.9} gap={:.3e}",
        mean(&sincos_times),
        mean(&sincos_losses),
        mean(&sincos_losses) - mean(&baseline_losses),
    );
    println!(
        "specialized_b8:      {:.6}s loss={:.9} gap={:.3e}",
        mean(&spec_times),
        mean(&spec_losses),
        mean(&spec_losses) - mean(&baseline_losses),
    );
    println!(
        "col_major_bxk:       {:.6}s loss={:.9} gap={:.3e}",
        mean(&col_times),
        mean(&col_losses),
        mean(&col_losses) - mean(&baseline_losses),
    );
    Ok(())
}
