//! Core logic shared by the Python extension (`import bayesian`) and the dev CLI (`cargo run`).

#![allow(dead_code)]

pub use ::bpe;
pub use ::rolling_hash;
pub use ::trie;
pub use ::render_utils;

mod session;
pub use session::BayesianSession;

/// Install [`console_error_panic_hook`] so panics in the wasm32 build print to `console.error`
/// with source location (and a useful traceback when debug symbols are present). Call once
/// after wasm init (see `initPanicHook` in the wasm bindings).
#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = initPanicHook)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Dev-only: panics immediately so you can verify `initPanicHook` / `console_error_panic_hook`
/// in the browser console. Frontend calls this when `?wasmPanic=1` (Vite dev only).
#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = debugPanicTest)]
pub fn debug_panic_test() {
    panic!("debug wasm panic (intentional; remove ?wasmPanic=1 from URL)");
}

/// Compute optimally-spaced timer phases for a set of weighted nodes.
///
/// `weights_json` is a JSON array of non-negative f64 weights (one per timer node).
/// `sigma` is the timer stddev (use `likelihoodModel.stddev_delay`).
/// `period` is the timer period.
///
/// Returns a JSON array of f64 phases in [0, period), same length and order as the input weights.
#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = optimizeTimerPhases)]
pub fn optimize_timer_phases(weights_json: &str, sigma: f64, period: f64) -> String {
    let weights: Vec<f64> = serde_json::from_str(weights_json).expect("weights_json must be a JSON array of numbers");
    if weights.is_empty() {
        return "[]".to_string();
    }
    let params = timer_spacing::TimerSpacingParams::new(weights.clone(), sigma, period);
    let initial = timer_spacing::constant_phases(weights.len(), period);
    let result = timer_spacing::optimize(&params, &initial, timer_spacing::DEFAULT_MAX_ITER)
        .expect("timer_spacing::optimize failed");
    serde_json::to_string(&result.phases).expect("phases serialization failed")
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn bayesian(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BayesianSession>()?;
    Ok(())
}

// #[cfg(test)]
// mod tests;
