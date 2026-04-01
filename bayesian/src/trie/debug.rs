//! Debug helpers for native testing and tracing.
//!
//! - [`log`] / [`crate::trie_debug!`]: on **`wasm`**, writes to the browser console; on native targets,
//!   prints to stderr only when the **`trie-trace`** feature is enabled (`cargo test -p bayesian --no-default-features --features trie-trace`, etc.).
//! - [`eprint_json`], [`eprint_snapshot`], [`eprint_debug`], [`json_pretty`]: always use **`eprintln!`** / allocate a string — for ad-hoc inspection in tests or small programs.
//! - [`snapshot_tree_format`] / [`eprint_snapshot_tree`]: `tree(1)`-style view of a [`TrieSnapshot`] with `z` and likelihood per node.
//! - [`format_top_followers_log_probs`] / [`format_prediction_top_followers`]: top BPE tokens (and stop) by stored log-probability.

use std::fmt::Write;

use crate::bpe::TinyLlamaWordTokenizer;
use crate::symbol::Symbol;

use super::{NodeIndex, Prediction, TrieSnapshot, TrieSnapshotNode};

#[cfg(feature = "wasm")]
pub fn log(msg: &str) {
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(msg));
}

#[cfg(all(not(feature = "wasm"), feature = "trie-trace"))]
pub fn log(msg: &str) {
    eprintln!("{msg}");
}

#[cfg(all(not(feature = "wasm"), not(feature = "trie-trace")))]
#[inline]
pub fn log(_msg: &str) {}

/// Pretty JSON for snapshots, test fixtures, or diffing.
pub fn json_pretty(value: &impl serde::Serialize) -> String {
    serde_json::to_string_pretty(value).expect("serde_json::to_string_pretty")
}

/// `eprintln!` a labeled pretty JSON value.
pub fn eprint_json(label: &str, value: &impl serde::Serialize) {
    eprintln!("--- {label} ---\n{}", json_pretty(value));
}

/// `eprintln!` a [`TrieSnapshot`] as pretty JSON.
pub fn eprint_snapshot(label: &str, snapshot: &TrieSnapshot) {
    eprint_json(label, snapshot);
}

/// `eprintln!` a labeled `Debug` value (no serde required).
pub fn eprint_debug(label: &str, value: &impl std::fmt::Debug) {
    eprintln!("--- {label} ---\n{value:#?}");
}

// --- snapshot tree (tree(1)-style) ---

fn symbol_tree_label(s: Symbol) -> String {
    match s {
        Symbol::Space => "_".into(),
        Symbol::Stop => "$".into(),
        Symbol::Start => "^".into(),
        _ => String::from(s.to_byte() as char),
    }
}

fn fmt_f64_compact(x: f64) -> String {
    if !x.is_finite() {
        return x.to_string();
    }
    let s = format!("{:.4}", x);
    s.trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn fmt_snapshot_node_line(node: &TrieSnapshotNode) -> String {
    format!(
        "{} (z: {}, l: {})",
        symbol_tree_label(node.symbol),
        fmt_f64_compact(node.z),
        fmt_f64_compact(node.likelihood)
    )
}

fn walk_snapshot_tree(
    out: &mut String,
    snapshot: &TrieSnapshot,
    idx: NodeIndex,
    prefix: &str,
    is_last: bool,
    is_root: bool,
) {
    let node = &snapshot.nodes[idx];
    if is_root {
        let _ = writeln!(out, "{}", fmt_snapshot_node_line(node));
    } else {
        let branch = if is_last { "└── " } else { "├── " };
        let _ = writeln!(out, "{prefix}{branch}{}", fmt_snapshot_node_line(node));
    }

    let extension = if is_root {
        ""
    } else if is_last {
        "    "
    } else {
        "│   "
    };
    let child_prefix = format!("{prefix}{extension}");

    let mut ch: Vec<(Symbol, NodeIndex)> = node.children.clone();
    ch.sort_by_key(|(s, _)| s.to_slot());

    let n = ch.len();
    for (i, (_, cidx)) in ch.into_iter().enumerate() {
        walk_snapshot_tree(
            out,
            snapshot,
            cidx,
            &child_prefix,
            i + 1 == n,
            false,
        );
    }
}

/// Unicode tree lines (`├──`, `└──`, `│`) over a [`TrieSnapshot`], one node per line with symbol and `(z, likelihood)`.
pub fn snapshot_tree_format(snapshot: &TrieSnapshot) -> String {
    let mut out = String::new();
    walk_snapshot_tree(&mut out, snapshot, snapshot.root, "", true, true);
    out
}

/// `eprintln!` a [`snapshot_tree_format`] under a header line.
pub fn eprint_snapshot_tree(label: &str, snapshot: &TrieSnapshot) {
    eprintln!("--- {label} ---\n{}", snapshot_tree_format(snapshot));
}

// --- prediction top followers ---

/// Top `top_n` BPE tokens plus optional `<stop>` by **descending** `follower_log_probs` / `stop_log_prob`
/// (values are log-probabilities as stored on [`Prediction`]). Output looks like
/// `(▁the: -2, ▁him: -3.45, <stop>: -4.1)`.
pub fn format_top_followers_log_probs(
    token_strings: &[String],
    follower_log_probs: &[f64],
    stop_log_prob: f64,
    top_n: usize,
) -> String {
    assert_eq!(
        token_strings.len(),
        follower_log_probs.len(),
        "token_strings and follower_log_probs must match tokenizer length"
    );

    let mut items: Vec<(&str, f64)> = token_strings
        .iter()
        .zip(follower_log_probs.iter())
        .filter_map(|(name, &lp)| lp.is_finite().then_some((name.as_str(), lp)))
        .collect();
    if stop_log_prob.is_finite() {
        items.push(("<stop>", stop_log_prob));
    }
    items.sort_by(|a, b| b.1.total_cmp(&a.1));
    items.truncate(top_n.min(items.len()));

    if items.is_empty() {
        return "()".to_string();
    }
    let inner = items
        .iter()
        .map(|(name, lp)| format!("{}: {}", name, fmt_f64_compact(*lp)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("({inner})")
}

/// Same as [`format_top_followers_log_probs`] for an in-memory [`Prediction`] and tokenizer (crate-internal: [`Prediction`] is not public).
pub(crate) fn format_prediction_top_followers(
    prediction: &Prediction,
    tokenizer: &TinyLlamaWordTokenizer,
    top_n: usize,
) -> String {
    format_top_followers_log_probs(
        tokenizer.tokens(),
        &prediction.follower_probs,
        prediction.stop_prob,
        top_n,
    )
}

/// `eprintln!` [`format_top_followers_log_probs`] under a header.
pub fn eprint_top_followers_log_probs(
    label: &str,
    token_strings: &[String],
    follower_log_probs: &[f64],
    stop_log_prob: f64,
    top_n: usize,
) {
    eprintln!(
        "--- {label} ---\n{}",
        format_top_followers_log_probs(
            token_strings,
            follower_log_probs,
            stop_log_prob,
            top_n
        )
    );
}

/// `eprintln!` [`format_prediction_top_followers`] under a header (crate-internal).
pub(crate) fn eprint_prediction_top_followers(
    label: &str,
    prediction: &Prediction,
    tokenizer: &TinyLlamaWordTokenizer,
    top_n: usize,
) {
    eprintln!(
        "--- {label} ---\n{}",
        format_prediction_top_followers(prediction, tokenizer, top_n)
    );
}

/// `trie_debug!("nodes={}", n)` → browser console on wasm; stderr with **`trie-trace`** on native; no-op otherwise.
#[macro_export]
macro_rules! trie_debug {
    ($($t:tt)*) => {{
        $crate::trie::debug::log(&format!($($t)*));
    }};
}
