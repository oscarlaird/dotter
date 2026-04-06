//! Trie tests that need `pub(crate)` APIs (`Trie`, `Prediction`, `SnapshotWalker`, …).
//!
//! For tests that only use the public `bayesian` surface, prefer `bayesian/tests/` at the crate root.
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
    println!("letter,a is token,aa is token,aaa is token,a->aa canonical,aa->a canonical");
    for byte in b'a'..=b'z' {
        let c = (byte as char).to_string();
        let aa = format!("{c}{c}");
        let aaa = format!("{c}{c}{c}");
        let a_is_token = tok.lex_index(&c).is_some();
        let aa_is_token = tok.lex_index(&aa).is_some();
        let aaa_is_token = tok.lex_index(&aaa).is_some();
        let a_to_aa = aa_is_token && tok.can_canonically_follow(&c, &aa);
        let aa_to_a = aa_is_token && tok.can_canonically_follow(&aa, &c);
        println!("{c},{a_is_token},{aa_is_token},{aaa_is_token},{a_to_aa},{aa_to_a}");
    }
}

