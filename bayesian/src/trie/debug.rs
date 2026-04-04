//! Pretty-print trie structure for development (linux `tree`-style lines).

use std::collections::HashSet;

use crate::rolling_hash as rh;
use crate::rolling_hash::Hash;
use crate::symbol::{Symbol, RADIX};

use super::ROOT_HASH;
use super::core::XBayes;
use super::core::XNode;

/// Parse a filter string into a set of [`Symbol`] values.
/// Bytes are interpreted like the trie alphabet: `a`–`z`, `_` → [`Symbol::Space`], `^` → [`Symbol::Start`].
/// Whitespace in `filter` is ignored. Unknown characters panic.
///
/// An **empty** filter (after trimming) means: show **every** node.
pub(crate) fn parse_symbol_filter(filter: &str) -> Option<HashSet<Symbol>> {
    let mut set = HashSet::new();
    for ch in filter.chars() {
        if ch.is_whitespace() {
            continue;
        }
        let Some(sym) = Symbol::from_byte(ch as u8) else {
            panic!("debug trie filter: unknown character {ch:?} (use a-z, _, ^)");
        };
        set.insert(sym);
    }
    if set.is_empty() {
        None
    } else {
        Some(set)
    }
}

fn collect_child_hashes(nodes: &rh::RHashMap<XNode>, parent_hash: Hash) -> Vec<(usize, Hash)> {
    let mut out = Vec::new();
    for slot in 0..RADIX {
        let child_hash = rh::append_right(parent_hash, Symbol::slot_to_byte(slot));
        if nodes.contains_key(&child_hash) {
            out.push((slot, child_hash));
        }
    }
    out
}

fn node_label(hash: Hash, node: &XNode) -> String {
    let sym = node.symbol;
    let c = sym.to_byte() as char;
    format!("{c} [{hash:#018x}]")
}

/// Print `trie` to stderr in a `tree`-like layout.
///
/// * `filter`: see [`parse_symbol_filter`]. Empty / whitespace-only ⇒ show all nodes.
/// * The **root** (`ROOT_HASH`) is always printed when it exists.
pub(crate) fn eprint_trie(trie: &XBayes, filter: &str) {
    let filter_set = parse_symbol_filter(filter);
    eprint_subtree(&trie.nodes, ROOT_HASH, "", true, &filter_set);
}

fn should_show_node(hash: Hash, node: &XNode, filter_set: &Option<HashSet<Symbol>>) -> bool {
    if filter_set.is_none() {
        return true;
    }
    if hash == ROOT_HASH {
        return true;
    }
    filter_set
        .as_ref()
        .is_some_and(|s| s.contains(&node.symbol))
}

fn eprint_subtree(
    nodes: &rh::RHashMap<XNode>,
    hash: Hash,
    prefix: &str,
    is_last: bool,
    filter_set: &Option<HashSet<Symbol>>,
) {
    let Some(node) = nodes.get(&hash) else {
        return;
    };

    let visible = should_show_node(hash, node, filter_set);

    if hash == ROOT_HASH {
        if visible {
            eprintln!("{}", node_label(hash, node));
        }
    } else if visible {
        let branch = if is_last { "└── " } else { "├── " };
        eprintln!("{prefix}{branch}{}", node_label(hash, node));
    }

    let extension = if hash == ROOT_HASH {
        ""
    } else if is_last {
        "    "
    } else {
        "│   "
    };
    let child_prefix = format!("{prefix}{extension}");

    let children = collect_child_hashes(nodes, hash);
    let n = children.len();
    for (i, (_, child_hash)) in children.into_iter().enumerate() {
        let child_is_last = i + 1 == n;
        eprint_subtree(nodes, child_hash, &child_prefix, child_is_last, filter_set);
    }
}
