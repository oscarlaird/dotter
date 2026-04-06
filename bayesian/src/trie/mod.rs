//! Trie engine and public session API.

use crate::rolling_hash as rh;
use crate::safe_float::Float;
use crate::symbol::Symbol;

/// Rolling hash of the trie root context (only the start symbol `^`).
pub(crate) const ROOT_HASH: rh::Hash = rh::append_right(0, Symbol::Start.to_byte());
pub(crate) const ROOT_STRING: &str = "^";

/// `ln(1/200)` — expansion stop rule and snapshot child cutoff for the Bayesian trie.
pub const TRIE_EXPANSION_THRESHOLD: f64 = -5.2983173665480363;

/// Maximum node visits per `recalc_to_frontier_and_back` traversal.
pub const TRIE_MAX_VISITS: i32 = 200;

pub(crate) type TokenLexIndex = u16;
pub(crate) const INVALID_TOKEN_LEXINDEX: TokenLexIndex = u16::MAX;
pub(crate) type PrefixLexIndex = usize;
pub(crate) const MAX_TOKEN_LENGTH: usize = 16;
pub(crate) const MAX_TRUNCATION_POSSIBLE: usize = 5;

mod l_update;
mod p_update;
mod core;
mod debug;
mod session;
mod prediction;

pub use session::BayesianSession;
#[cfg(feature = "tokentrie")]
mod tokentrie;

#[cfg(test)]
mod tests;

pub(crate) fn logaddexp(a: Float, b: Float) -> Float {
    if a == Float::NEG_INFINITY {
        return b;
    }
    if b == Float::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
