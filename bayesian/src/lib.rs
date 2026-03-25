//! Core logic shared by the Python extension (`import bayesian`) and the dev CLI (`cargo run`).

#![allow(dead_code)]

use pyo3::prelude::*;

// --- byte trie (fixed alphabet, arena-allocated) -----------------------------------------------

/// Number of outgoing edges per node (size of `Node::children`).
/// Lowercase a–z, ASCII space, `!`.
const RADIX: usize = 28;
const MAX_TOKEN_LENGTH: usize = 16;

const NUM_TOKENS: usize = 17250;

/// Default trie alphabet: `a`–`z`, space, `!` (slot order matches this slice).
pub const DEFAULT_ALPHABET: [u8; RADIX] = *b"abcdefghijklmnopqrstuvwxyz !";

/// Maps any `u8` to a child slot `0..RADIX`, or `None` if that byte is not in the alphabet.
type ByteToSlot = [Option<u8>; 256];
type SlotToByte = [u8; RADIX];

#[derive(Clone, Debug)]
struct Node {
    // posterior
    z: f64,  // log unnormalized posterior
    nz: u32, // posterior likelihood time
    mz: u32, // posterior prior time
    // likelihood
    ll: f64, // log lower likelihood
    ul: f64, // log upper likelihood
    nl: u32, // likelihood tracking time
    tl: [f64; MAX_TOKEN_LENGTH], // log token branch likelihood for each ancestor i,  i.e., log "Maximum Truncation Compatible Descendant Likelihood", for each ancestor i
    ntl: u32, // token branch likelihood tracking time
    cum_likelihood_frontier: bool,
    // prior
    p: f64, // log string branch prior
    mp: u32, // prior tracking time
    fp: [f64; MAX_TOKEN_LENGTH], // token fans for each ancestor i
    tp: f64, // log token branch prior
    ntp: u32, // token branch prior tracking time
    // tokenization
    final_token_length: u32,
    // children
    /// Index in `Trie::nodes` of the first of `RADIX` consecutive child nodes (slot `s` → `nodes[base + s]`).
    /// `None` if no child block has been allocated for this node yet.
    children_start_index: Option<u32>,
}

impl Node {
    /// Default values for a new non-root node (no children allocated).
    fn fresh() -> Self {
        Self {
            z: 1.0,
            nz: 0,
            mz: 0,
            ll: 0.0,
            ul: 0.0,
            nl: 0,
            tl: [0.0; MAX_TOKEN_LENGTH],
            ntl: 0,
            cum_likelihood_frontier: false,
            p: 0.0,
            mp: 0,
            fp: [0.0; MAX_TOKEN_LENGTH],
            tp: 0.0,
            ntp: 0,
            final_token_length: 0,
            children_start_index: None,
        }
    }

    /// Root node: same defaults as [`Self::fresh`] but `z` starts at `1.0`.
    fn root() -> Self {
        Self {
            z: 1.0,
            ..Self::fresh()
        }
    }
}

#[derive(Clone, Debug)]
struct Trie {
    nodes: Vec<Node>,
    root: usize,
    byte_to_slot: ByteToSlot,
    slot_to_byte: SlotToByte,
}

impl Trie {
    fn new() -> Self {
        Self::with_alphabet(&DEFAULT_ALPHABET)
    }

    fn with_alphabet(alphabet: &[u8; RADIX]) -> Self {
        let mut byte_to_slot = [None; 256];
        let mut slot_to_byte = [0; RADIX];
        for (slot, &b) in alphabet.iter().enumerate() {
            byte_to_slot[b as usize] = Some(slot as u8);
            slot_to_byte[slot] = b;
        }
        let nodes = vec![Node::root()];

        Self {
            nodes,
            root: 0,
            byte_to_slot,
            slot_to_byte,
        }
    }

    /// Allocates a contiguous block of `RADIX` child nodes for `parent_index` if none exists yet.
    fn ensure_children(&mut self, parent_index: usize) {
        if self.nodes[parent_index].children_start_index.is_some() {
            return;
        }
        let base = self.nodes.len() as u32;
        self.nodes[parent_index].children_start_index = Some(base);
        self.nodes.extend((0..RADIX).map(|_| Node::fresh()));
    }
}

// ------------------------------------------------------------------------------------------------

/// Example entry point for Rust-side development and tests.
pub fn add(a: i64, b: i64) -> i64 {
    a + b + 20
}


// get trie
// expand trie
// queue token branches
// apply likelihood update
// apply prior update

#[pyfunction]
#[pyo3(name = "add")]
fn add_py(a: i64, b: i64) -> i64 {
    add(a, b)
}

#[pymodule]
fn bayesian(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_py, m)?)?;
    Ok(())
}
