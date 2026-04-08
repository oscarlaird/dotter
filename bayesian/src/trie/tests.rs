//! Trie tests that need `pub(crate)` APIs (`Trie`, `Prediction`, `SnapshotWalker`, …).
//!
//! For tests that only use the public `bayesian` surface, prefer `bayesian/tests/` at the crate root.
#[cfg(not(feature = "wasm"))]
use crate::trie::ROOT_HASH;
use crate::trie::INVALID_TOKEN_LEXINDEX;
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

fn snapshot_strings(snapshot_json: &str) -> Vec<String> {
    #[derive(Deserialize)]
    struct SnapshotNode {
        hash: crate::rolling_hash::Hash,
    }

    let mut keys = serde_json::from_str::<std::collections::HashMap<String, SnapshotNode>>(snapshot_json)
        .unwrap()
        .into_keys()
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

#[derive(Deserialize)]
struct RequestedPrior {
    full_string: String,
    last_token_lexindex: TokenLexIndex,
}

#[derive(Serialize)]
struct PriorPayload {
    full_string: String,
    final_token_lexindex: TokenLexIndex,
    follower_logits: Vec<f32>,
}

fn zero_prior_payload(session: &crate::BayesianSession, request_json: String) -> String {
    let request: RequestedPrior = serde_json::from_str(&request_json).unwrap();
    serde_json::to_string(&PriorPayload {
        full_string: request.full_string,
        final_token_lexindex: request.last_token_lexindex,
        follower_logits: vec![0.0; session.trie.tokenizer.tokens().len()],
    })
    .unwrap()
}


#[cfg(not(feature = "wasm"))]
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

#[test]
#[ignore = "repro for root prior apply panic"]
fn repro_root_uniform_prior_apply() {
    #[derive(Deserialize)]
    struct RequestedPrior {
        full_string: String,
        last_token_lexindex: TokenLexIndex,
    }

    #[derive(Serialize)]
    struct Payload {
        full_string: String,
        final_token_lexindex: TokenLexIndex,
        follower_logits: Vec<f32>,
    }

    let mut session = crate::BayesianSession::new();
    let request: RequestedPrior = serde_json::from_str(&session.next_requested_prior()).unwrap();
    assert_eq!(request.full_string, "^");
    assert_eq!(request.last_token_lexindex, INVALID_TOKEN_LEXINDEX);

    let payload = Payload {
        full_string: request.full_string,
        final_token_lexindex: request.last_token_lexindex,
        follower_logits: vec![0.0; crate::bpe::NUM_TOKENS],
    };

    session.receive_prior_update(serde_json::to_string(&payload).unwrap());
    session.apply_updates();
}

#[test]
#[ignore = "repro for root prior apply panic with skewed logits"]
fn repro_root_skewed_prior_apply() {
    #[derive(Deserialize)]
    struct RequestedPrior {
        full_string: String,
        last_token_lexindex: TokenLexIndex,
    }

    #[derive(Serialize)]
    struct Payload {
        full_string: String,
        final_token_lexindex: TokenLexIndex,
        follower_logits: Vec<f32>,
    }

    let mut session = crate::BayesianSession::new();
    let request: RequestedPrior = serde_json::from_str(&session.next_requested_prior()).unwrap();
    assert_eq!(request.full_string, "^");
    assert_eq!(request.last_token_lexindex, INVALID_TOKEN_LEXINDEX);

    let mut logits = vec![-20.0_f32; crate::bpe::NUM_TOKENS];
    let set_logit = |logits: &mut [f32], token: &str, logit: f32| {
        let token_hash = crate::rolling_hash::hash_string(token);
        let token_index = session.trie.tokenizer.lex_index_for_token_hash(&token_hash);
        logits[token_index] = logit;
    };
    set_logit(&mut logits, "a", 0.0);
    set_logit(&mut logits, "aa", -0.5);
    set_logit(&mut logits, "_the", -1.0);
    set_logit(&mut logits, "z", -1.5);
    set_logit(&mut logits, "zz", -2.0);

    let payload = Payload {
        full_string: request.full_string,
        final_token_lexindex: request.last_token_lexindex,
        follower_logits: logits,
    };

    session.receive_prior_update(serde_json::to_string(&payload).unwrap());
    session.apply_updates();
}

#[test]
fn probe_z_to_zz_canonical_follow() {
    let tok = crate::bpe::TinyLlamaWordTokenizer::from_tokenizer_json_str(
        crate::bpe::TOKENIZER_JSON_STR,
    );
    let z = tok.lex_index("z").unwrap();
    let zz = tok.lex_index("zz").unwrap();
    assert_eq!(tok.canonical_followers("z")[zz], tok.can_canonically_follow("z", "zz"));
    println!("z -> zz canonical: {}", tok.canonical_followers("z")[zz]);
    println!("z lex index: {}", z);
    println!("zz lex index: {}", zz);
}

#[test]
fn probe_zzz_tokenization() {
    let tok = crate::bpe::TinyLlamaWordTokenizer::from_tokenizer_json_str(
        crate::bpe::TOKENIZER_JSON_STR,
    );
    let pieces = tok.tokenize_string_with_lex_indices("zzz");
    println!("zzz tokenization: {:?}", pieces);
}

#[test]
fn probe_zz_to_z_canonical_follow() {
    let tok = crate::bpe::TinyLlamaWordTokenizer::from_tokenizer_json_str(
        crate::bpe::TOKENIZER_JSON_STR,
    );
    let z = tok.lex_index("z").unwrap();
    let zz = tok.lex_index("zz").unwrap();
    println!("zz -> z canonical: {}", tok.canonical_followers("zz")[z]);
    println!("can_canonically_follow(\"zz\", \"z\"): {}", tok.can_canonically_follow("zz", "z"));
    println!("zz lex index: {}", zz);
    println!("z lex index: {}", z);
}

#[test]
fn probe_z_to_zz_can_canonically_follow() {
    let tok = crate::bpe::TinyLlamaWordTokenizer::from_tokenizer_json_str(
        crate::bpe::TOKENIZER_JSON_STR,
    );
    println!("can_canonically_follow(\"z\", \"zz\"): {}", tok.can_canonically_follow("z", "zz"));
}

#[test]
fn probe_zz_right_spine() {
    let merges = crate::bpe::BpeMerges::from_tokenizer_json_str(crate::bpe::TOKENIZER_JSON_STR);
    let right = merges.right_spine("zz").unwrap();
    let z_right = merges.right_spine("z").unwrap();
    println!("right_spine(\"zz\"): {:?}", right);
    println!("right_spine(\"z\"): {:?}", z_right);
}

#[test]
fn probe_repeated_letter_triples_canonical_direction() {
    let tok = crate::bpe::TinyLlamaWordTokenizer::from_tokenizer_json_str(
        crate::bpe::TOKENIZER_JSON_STR,
    );
    for byte in b'a'..=b'z' {
        let c = (byte as char).to_string();
        let aa = format!("{c}{c}");
        let aaa = format!("{c}{c}{c}");
        if tok.lex_index(&aa).is_none() {
            println!("{aaa}: aa missing");
            continue;
        }
        let aa_then_a = tok.can_canonically_follow(&aa, &c);
        let a_then_aa = tok.can_canonically_follow(&c, &aa);
        let classification = match (aa_then_a, a_then_aa) {
            (true, false) => "aa->a",
            (false, true) => "a->aa",
            (false, false) => "neither",
            (true, true) => "both",
        };
        println!("{aaa}: {classification}");
    }
}

#[test]
fn probe_repeated_letter_triples_table() {
    let tok = crate::bpe::TinyLlamaWordTokenizer::from_tokenizer_json_str(
        crate::bpe::TOKENIZER_JSON_STR,
    );
    println!("letter,a is token,aa is token,aaa is token,a->aa canonical (prepared),aa->a canonical (prepared)");
    for byte in b'a'..=b'z' {
        let c = (byte as char).to_string();
        let aa = format!("{c}{c}");
        let aaa = format!("{c}{c}{c}");
        let a_is_token = tok.lex_index(&c).is_some();
        let aa_is_token = tok.lex_index(&aa).is_some();
        let aaa_is_token = tok.lex_index(&aaa).is_some();
        let a_to_aa = if a_is_token && aa_is_token {
            let aa_ix = tok.lex_index(&aa).unwrap();
            tok.canonical_followers(&c)[aa_ix]
        } else {
            false
        };
        let aa_to_a = if a_is_token && aa_is_token {
            let a_ix = tok.lex_index(&c).unwrap();
            tok.canonical_followers(&aa)[a_ix]
        } else {
            false
        };
        println!("{c},{a_is_token},{aa_is_token},{aaa_is_token},{a_to_aa},{aa_to_a}");
    }
}

#[test]
#[ignore = "repro for repeated local likelihood-only apply_updates panic"]
fn repro_repeated_local_likelihood_updates() {
    let mut session = crate::BayesianSession::new();
    let mut snapshot_json = session.expand_to_threshold();

    for _ in 0..2 {
        let likelihood_json = serde_json::json!(
            snapshot_strings(&snapshot_json)
                .into_iter()
                .map(|key| (key, serde_json::json!({ "l": -0.1_f32 })))
                .collect::<std::collections::HashMap<_, _>>()
        )
        .to_string();

        session.receive_likelihood_update(likelihood_json);
        session.apply_updates();
        snapshot_json = session.expand_to_threshold();
    }
}

#[test]
#[ignore = "repro for backend-style five-prior cycle panic"]
fn repro_backend_style_five_prior_cycle() {
    let mut session = crate::BayesianSession::new();

    let initial_request_json = session.next_requested_prior();
    let initial_prior_json = zero_prior_payload(&session, initial_request_json);
    session.receive_prior_update(initial_prior_json);
    session.apply_updates();

    let snapshot_json = session.expand_to_threshold();
    let likelihood_json = serde_json::json!(
        snapshot_strings(&snapshot_json)
            .into_iter()
            .map(|key| (key, serde_json::json!({ "l": -0.1_f32 })))
            .collect::<std::collections::HashMap<_, _>>()
    )
    .to_string();

    session.receive_likelihood_update(likelihood_json);
    session.apply_updates();

    for _ in 0..5 {
        let request_json = session.next_requested_prior();
        let prior_json = zero_prior_payload(&session, request_json);
        session.receive_prior_update(prior_json);
        session.apply_updates();
    }
}

/// Replay of `testdata/frontend_prior_panic/` with the same ordering as V3: after each successful
/// `apply_updates()`, call `expand_to_threshold()` (and start with `reset` + expand like the page).
///
/// This is the canonical Rust replay of the captured WebSocket JSON. On the host target it
/// **passes**; the browser `RuntimeError: unreachable` from `apply_updates` after a prior is not
/// reproduced here yet. Use it as a regression fixture and extend it once a host panic is found.
#[test]
fn repro_captured_frontend_prior_cycle_v3_fixture_replay() {
    let mut session = crate::BayesianSession::new();
    session.reset();
    let _ = session.expand_to_threshold();

    session.receive_prior_update(
        include_str!("../../testdata/frontend_prior_panic/root_prior.json")
            .trim()
            .to_string(),
    );
    session.apply_updates();
    let _ = session.expand_to_threshold();

    session.receive_likelihood_update(
        include_str!("../../testdata/frontend_prior_panic/likelihood.json")
            .trim()
            .to_string(),
    );
    session.apply_updates();
    let _ = session.expand_to_threshold();

    session.receive_prior_update(
        include_str!("../../testdata/frontend_prior_panic/prior_1.json")
            .trim()
            .to_string(),
    );
    session.apply_updates();
    let _ = session.expand_to_threshold();

    session.receive_prior_update(
        include_str!("../../testdata/frontend_prior_panic/prior_2.json")
            .trim()
            .to_string(),
    );
    session.apply_updates();
    let _ = session.expand_to_threshold();
}

/// One root prior from `testdata/root_lm_prior.json` (no likelihood); then what `z` does
/// `expand_to_threshold` associate with `^_` and `^__`? Same numbers as `parent.c_z[slot]`
/// on the root / `^_` node for the `Space` child (`session.rs` snapshot rule).
///
/// Regenerate the fixture with `backend/export_root_prior_for_tests.py` if needed.
#[cfg(not(feature = "wasm"))]
#[test]
fn trie_explore_z_under_after_root_lm_prior_fixture() {
    use crate::rolling_hash as rh;
    use crate::safe_float::into_f32;
    use crate::symbol::Symbol;
    use crate::trie::ROOT_HASH;

    #[derive(Deserialize)]
    struct SnapshotEntry {
        z: f32,
        hash: rh::Hash,
    }

    let mut session = crate::BayesianSession::new();
    session.receive_prior_update(
        include_str!("../../testdata/root_lm_prior.json")
            .trim()
            .to_string(),
    );
    session.apply_updates();
    let snapshot_json = session.expand_to_threshold();
    let snapshot: std::collections::HashMap<String, SnapshotEntry> =
        serde_json::from_str(&snapshot_json).unwrap();

    let h_under = rh::append_right(ROOT_HASH, Symbol::Space.to_byte());
    let h_under_under = rh::append_right(h_under, Symbol::Space.to_byte());
    let root = session.trie.nodes.get(&ROOT_HASH).unwrap();
    let under = session.trie.nodes.get(&h_under).unwrap();

    println!("root.if_root_then_z = {:?}", root.if_root_then_z);
    println!(
        "{}",
        crate::trie::core::debug::format_node_slot_dump(root, "root", Symbol::Space)
    );
    println!(
        "{}",
        crate::trie::core::debug::format_node_slot_dump(under, "^_", Symbol::Space)
    );

    for (path, parent_hash, slot) in [
        ("^_", ROOT_HASH, Symbol::Space.to_slot()),
        ("^__", h_under, Symbol::Space.to_slot()),
    ] {
        let entry = snapshot
            .get(path)
            .unwrap_or_else(|| panic!("{path} missing from expand_to_threshold snapshot"));
        let expect_hash = if path == "^_" {
            h_under
        } else {
            h_under_under
        };
        if entry.hash != expect_hash {
            panic!(
                "{path}: snapshot hash {:?} != trie hash {:?}",
                entry.hash, expect_hash
            );
        }
        let parent = session
            .trie
            .nodes
            .get(&parent_hash)
            .unwrap_or_else(|| panic!("parent trie node missing for {path}"));
        let z_edge = parent.c_z[slot];
        println!(
            "{path}: snapshot z={} parent.c_z[slot] f32={} (match expand rule)",
            entry.z,
            into_f32(z_edge),
        );
    }

    println!(
        "^__ node present in trie: {}",
        session.trie.nodes.contains_key(&h_under_under)
    );
}

#[cfg(not(feature = "wasm"))]
#[test]
fn debug_z_after_first_expand_caret_space_space() {
    use crate::rolling_hash as rh;
    use crate::safe_float::{Float, into_f32};
    use crate::symbol::Symbol;
    use crate::trie::ROOT_HASH;

    let mut session = crate::BayesianSession::new();
    let _ = session.expand_to_threshold();

    // Posterior for one-step prefix ^_ : share among root children is softmax(Z_edge − Z_root_total).
    let root_node = session.trie.nodes.get(&ROOT_HASH).unwrap();
    let z_root_total = root_node.if_root_then_z;
    let z_root_to_us = root_node.c_z[Symbol::Space.to_slot()];
    let log_post = into_f32(z_root_to_us) - into_f32(z_root_total);
    let post_prob = log_post.exp();
    println!(
        "posterior ^_ : Z(^->_)={} Z_root_total={} log_ratio={} P(space|root)={}",
        into_f32(z_root_to_us),
        into_f32(z_root_total),
        log_post,
        post_prob
    );

    let h_parent = rh::append_right(ROOT_HASH, b'_');
    let h = rh::append_right(h_parent, b'_');
    let parent = session.trie.nodes.get(&h_parent).unwrap();
    let z_edge = parent.c_z[Symbol::Space.to_slot()];
    println!(
        "^_ -> ^__: c_z on parent slot Space = {:?} f32={}",
        z_edge,
        into_f32(z_edge)
    );
    println!("^__ node exists: {}", session.trie.nodes.contains_key(&h));
    if let Some(node) = session.trie.nodes.get(&h) {
        let mut sum = Float::NEG_INFINITY;
        for slot in 0..27 {
            sum = crate::trie::logaddexp(sum, node.c_z[slot]);
        }
        println!("^__ logsum(children c_z): {}", into_f32(sum));
    }
}
