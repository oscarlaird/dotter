//! Trie tests that need `pub(crate)` APIs (`Trie`, `Prediction`, `SnapshotWalker`, …).
//!
//! For tests that only use the public `bayesian` surface, prefer `bayesian/tests/` at the crate root.

#[test]
fn trie_expansion_threshold_is_ln_one_over_200() {
    assert_eq!(crate::TRIE_EXPANSION_THRESHOLD, (1.0_f64 / 200.0).ln());
}

#[test]
fn trie_exploratory_trace() {
    let mut session = crate::BayesianSession::new();

    let before = session.trie_snapshot_at_current();
    super::debug::eprint_trie("trie before expand", &session.trie);
    super::debug::eprint_snapshot_tree("snapshot before expand", &before);

    let alphabet_song = "abcdefghijklmnopqrstuvwxyz";
    println!("pausing..");
    session.expand_trie();

    let after = session.trie_snapshot_at_current();
    super::debug::eprint_trie("trie before expand", &session.trie);
    super::debug::eprint_snapshot_tree("snapshot after expand", &after);
}
