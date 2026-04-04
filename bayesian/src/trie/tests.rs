//! Trie tests that need `pub(crate)` APIs (`Trie`, `Prediction`, `SnapshotWalker`, …).
//!
//! For tests that only use the public `bayesian` surface, prefer `bayesian/tests/` at the crate root.
use crate::trie::ROOT_HASH;


#[test]
fn trie_exploratory_trace() {
    println!("ROOT_HASH: {}", ROOT_HASH);
    let mut session = crate::BayesianSession::new();
    session.expand_to_threshold();
    println!("Trie nodes: {}", session.trie.nodes.len());
    session.debug_eprint_trie("abcdefghijklmnopqrstuvwxyz_");

}
