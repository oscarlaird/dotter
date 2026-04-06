//! Trie tests that need `pub(crate)` APIs (`Trie`, `Prediction`, `SnapshotWalker`, …).
//!
//! For tests that only use the public `bayesian` surface, prefer `bayesian/tests/` at the crate root.
use crate::trie::ROOT_HASH;
use crate::trie::TokenLexIndex;
use serde::{Deserialize, Serialize};

fn expanded_hashes(snapshot_json: String) -> crate::rolling_hash::RHashSet {
    #[derive(Deserialize)]
    struct SnapshotNode {
        hash: crate::rolling_hash::Hash,
    }

    serde_json::from_str::<std::collections::HashMap<String, SnapshotNode>>(&snapshot_json)
        .unwrap()
        .into_values()
        .map(|node| node.hash)
        .collect()
}


#[test]
fn trie_exploratory_trace() {
    println!("ROOT_HASH: {}", ROOT_HASH);
    let mut session = crate::BayesianSession::new();
    //
    println!("Before expand: Trie nodes: {}", session.trie.nodes.len());
    session.debug_eprint_trie("abcdefghijklmnopqrstuvwxyz_");
    // Expand
    let expanded_hashes_0 = expanded_hashes(session.expand_to_threshold());
    println!("After expand: Trie nodes: {}", session.trie.nodes.len());
    session.debug_eprint_trie_hash_filter("abcdefghijklmnopqrstuvwxyz_", &expanded_hashes_0);
    // Receive likelihood update and expand
    session.receive_likelihood_update(r#"{"^": {"l": 0.0}, "^a": {"l": 1.0}, "^ar": {"l": 3.0}}"#.to_string());
    session.apply_updates();
    let expanded_hashes_1 = expanded_hashes(session.expand_to_threshold());
    println!("After likelihood update: Trie nodes: {}", session.trie.nodes.len());
    session.debug_eprint_trie_hash_filter("abcdefghijklmnopqrstuvwxyz_", &expanded_hashes_1);
    // Request the next prior
    let request = session.next_requested_prior();
    println!("Requested prior: {}", request);
    #[derive(Deserialize)]
    struct RequestedPrior {
        full_string: String,
        last_token_lexindex: TokenLexIndex,
    }
    let request: RequestedPrior = serde_json::from_str(&request).unwrap();
    // Receive prior update and expand
    let mut logits = vec![-10.0_f32; crate::bpe::NUM_TOKENS];
    let set_logit = |logits: &mut [f32], token: &str, logit: f32| {
        let token_hash = crate::rolling_hash::hash_string(token);
        let token_index = session.trie.tokenizer.lex_index_for_token_hash(&token_hash);
        logits[token_index] = logit;
    };
    set_logit(&mut logits, "arch", 0.0);
    set_logit(&mut logits, "arrow", -1.0);
    set_logit(&mut logits, "arab", -2.0);
    #[derive(Serialize)]
    struct Payload {
        full_string: String,
        final_token_lexindex: TokenLexIndex,
        follower_logits: Vec<f32>,
    }
    let payload = Payload {
        full_string: request.full_string,
        final_token_lexindex: request.last_token_lexindex,
        follower_logits: logits,
    };
    session.receive_prior_update(serde_json::to_string(&payload).unwrap());
    // apply updates and expand
    session.apply_updates();
    let expanded_hashes_2 = expanded_hashes(session.expand_to_threshold());
    println!("After prior update: Trie nodes: {}", session.trie.nodes.len());
    session.debug_eprint_trie_hash_filter("abcdefghijklmnopqrstuvwxyz_", &expanded_hashes_2);
    // request the next prior
    let request = session.next_requested_prior();
    println!("Requested prior: {}", request);
    // receive prior update and expand
}
