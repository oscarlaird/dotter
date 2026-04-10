//! Core logic shared by the Python extension (`import bayesian`) and the dev CLI (`cargo run`).

#![allow(dead_code)]

pub use ::bpe;
pub use ::rolling_hash;
pub use ::trie;

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

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn bayesian(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BayesianSession>()?;
    Ok(())
}

#[cfg(test)]
mod tests;
