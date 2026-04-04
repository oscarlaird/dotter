//! Core logic shared by the Python extension (`import bayesian`) and the dev CLI (`cargo run`).

#![allow(dead_code)]

pub mod bpe;
pub(crate) mod rolling_hash;
pub(crate) mod safe_float;
pub mod symbol;
pub mod trie;

pub use trie::BayesianSession;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn bayesian(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BayesianSession>()?;
    Ok(())
}
