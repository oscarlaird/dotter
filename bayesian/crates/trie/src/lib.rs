#![allow(dead_code)]
//! Trie engine and public session API.

pub use ::bpe as bpe;
pub use ::rolling_hash as rolling_hash;

pub mod safe_float;
pub mod symbol;

use crate::rolling_hash as rh;
use crate::safe_float::Float;
use crate::symbol::{START_SYMBOL, STOP_SYMBOL};

/// Surface string for the trie root (must match [`Symbol::Start`]).
pub const ROOT_STRING: &str = "A";

/// Rolling hash of the trie root context (only the start symbol `A`).
pub const ROOT_HASH: rh::Hash = rh::append_right(0, START_SYMBOL);
pub const STOP_HASH: rh::Hash = rh::append_right(0, STOP_SYMBOL);

/// `ln(4/100)` — expansion stop rule and snapshot child cutoff for the Bayesian trie.
pub const TRIE_EXPANSION_THRESHOLD: f64 = -3.2188758248682006;

/// Maximum node visits per `recalc_to_frontier_and_back` traversal.
pub const TRIE_MAX_VISITS: i32 = 200;

pub(crate) const MAX_TOKEN_LENGTH: usize = 20;
pub(crate) const MAX_TRUNCATION_POSSIBLE: usize = 5;

pub mod l_update;
mod p_update;
pub mod dfs;
pub mod core;
pub mod prediction;
pub use prediction::{
    ZeroOrderPredictionTimingSnapshot,
    reset_zero_order_prediction_timing,
    zero_order_prediction_timing_snapshot,
};

#[cfg(feature = "tokentrie")]
mod tokentrie;

pub fn logaddexp(a: Float, b: Float) -> Float {
    if a == Float::NEG_INFINITY {
        return b;
    }
    if b == Float::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
