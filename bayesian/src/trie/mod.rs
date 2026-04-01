//! Trie engine and public session API.

pub type NodeIndex = usize;
pub(crate) type PredictionIndex = usize;
pub(crate) type TokenLexIndex = usize;
pub(crate) type PrefixLexIndex = usize;
pub(crate) const MAX_TOKEN_LENGTH: usize = 16;

mod core;
pub(crate) mod debug;
mod prediction;
mod session;
mod snapshot;
#[cfg(feature = "tokentrie")]
mod tokentrie;

pub use session::BayesianSession;
pub use snapshot::{TrieSnapshot, TrieSnapshotNode};

pub(crate) use core::Trie;
pub(crate) use prediction::{Prediction, PredictionOrder, PredictionRegistry};
pub(crate) use snapshot::SnapshotWalker;

pub(crate) fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
