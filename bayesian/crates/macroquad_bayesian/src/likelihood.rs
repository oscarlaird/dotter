#[derive(Clone, Debug, PartialEq )]
pub struct LikelihoodModel {
    pub pred_mean: f32,
    pub pred_stddev: f32,
    pub pred_outliers: f32,
    pub period: f32,
}

fn logaddexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    if a > b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

fn normal_logpdf(x: f32, mean: f32, stddev: f32) -> f32 {
    -0.5 * ((x - mean) / stddev).powi(2) - (stddev * (2.0 * std::f32::consts::PI).sqrt()).ln()
}

pub fn modulo_delay(time_seconds: f32, phase: f32, period: f32) -> f32 {
    let mut x = time_seconds - phase;
    x = ((x % period) + period) % period;
    x
}

pub fn timer_likelihood(time: f32, phase: f32, model: &LikelihoodModel) -> f32 {
    let x = modulo_delay(time, phase, model.period);
    let outlier_prob = model.pred_outliers.ln() - model.period.ln();
    let not_outlier_prob = (1.0 - model.pred_outliers).ln();
    let normal_modes = [-1.0, 0.0, 1.0]
        .into_iter()
        .map(|k| normal_logpdf(x, model.pred_mean + k * model.period, model.pred_stddev))
        .collect::<Vec<_>>();
    let sum_normal_modes = normal_modes.iter().skip(1).fold(normal_modes[0], |acc, &x| logaddexp(acc, x));
    logaddexp(outlier_prob, not_outlier_prob + sum_normal_modes)
}