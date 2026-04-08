#[test]
fn repro_traverse_and_count_l_mismatch() {
    let mut session = crate::BayesianSession::new();
    
    // 1. Expand the tree so cum_likelihood has interior nodes (e.g. `^_`)
    // We send a prior update to pretend the LM ran.
    let root_prior = r#"{"full_string":"^","final_token_lexindex":17235,"follower_logits":[0.0, 0.0, 0.0]}"#; // dummy
    // But we can just use `expand_to_threshold` trick, or send real likelihoods.
    
    // Let's just create an artificial situation using the public API:
    // Expand to get some interior nodes.
    session.expand_to_threshold();
    
    // 2. Send a small likelihood update so cum_likelihood expands.
    let likelihood_json = r#"{"^_":{"likelihood": -1.0}}"#;
    session.receive_likelihood_update(likelihood_json.to_string());
    session.apply_updates(); // Now cum_likelihood contains ^_ as an interior node? Wait, apply_updates merges it.
    
    // 3. Receive another prior (this puts it in the queue)
    session.receive_prior_update(r#"{"full_string":"^_","final_token_lexindex":263,"follower_logits":[0.0, 0.0]}"#.to_string());
    
    // 4. Now, pending_likelihood is empty (just the root leaf).
    // cum_likelihood has `^_`.
    // Calling `next_requested_prior` will enqueue children of `^_`.
    // Since `^_` is in `cum_likelihood`, hit_cuml_edge is false.
    // It will call `traverse_and_count_l` on `pending_likelihood` for the children of `^_`.
    // But `pending_likelihood` only has the root! So it will panic!
    session.next_requested_prior();
}