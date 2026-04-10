#![allow(dead_code)]
//! Trie engine and public session API.

pub use ::bpe as bpe;
pub use ::rolling_hash as rolling_hash;

pub mod safe_float;
pub mod symbol;

use crate::rolling_hash as rh;
use crate::safe_float::Float;
use crate::symbol::Symbol;

/// Rolling hash of the trie root context (only the start symbol `^`).
pub const ROOT_HASH: rh::Hash = rh::append_right(0, Symbol::Start.to_byte());
pub const ROOT_STRING: &str = "^";

/// `ln(4/100)` — expansion stop rule and snapshot child cutoff for the Bayesian trie.
pub const TRIE_EXPANSION_THRESHOLD: f64 = -3.2188758248682006;

/// Maximum node visits per `recalc_to_frontier_and_back` traversal.
pub const TRIE_MAX_VISITS: i32 = 200;

pub(crate) const MAX_TOKEN_LENGTH: usize = 16;
pub(crate) const MAX_TRUNCATION_POSSIBLE: usize = 5;

pub mod l_update;
mod p_update;
pub mod core;
pub mod debug;
pub mod prediction;

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
