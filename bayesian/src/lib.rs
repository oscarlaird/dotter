//! Core logic shared by the Python extension (`import bayesian`) and the dev CLI (`cargo run`).

#![allow(dead_code)]

pub mod bpe;
pub mod symbol;

use pyo3::prelude::*;
use symbol::Symbol;

// --- byte trie (fixed alphabet, arena-allocated) -----------------------------------------------
const MAX_TOKEN_LENGTH: usize = 16;

const NUM_TOKENS: usize = 17250;

#[derive(Clone, Debug)]
struct Node {
    // posterior
    z: f64,  // log unnormalized posterior
    nz: u32, // posterior likelihood time
    mz: u32, // posterior prior time
    // likelihood
    ll: f64,                     // log lower likelihood
    ul: f64,                     // log upper likelihood
    nl: u32,                     // likelihood tracking time
    tl: [f64; MAX_TOKEN_LENGTH], // log token branch likelihood for each ancestor i,  i.e., log "Maximum Truncation Compatible Descendant Likelihood", for each ancestor i
    ntl: u32,                    // token branch likelihood tracking time
    cum_likelihood_frontier: bool,
    // prior
    p: f64,                      // log string branch prior
    mp: u32,                     // prior tracking time
    fp: [f64; MAX_TOKEN_LENGTH], // token fans for each ancestor i
    tp: f64,                     // log token branch prior
    mtp: u32,                    // token branch prior tracking time
    // tokenization
    final_token_length: u8,
    n_tokens: u32,
    // children
    /// Index in `Trie::nodes` of the first of `RADIX` consecutive child nodes (slot `s` → `nodes[base + s]`).
    /// `None` if no child block has been allocated for this node yet.
    symbol: Option<Symbol>,
    depth: u32,
    children_start_index: Option<u32>,
}

#[derive(Clone, Debug)]
struct Walker {
    node: usize,
    depth: u32,
    a_symbol: [Option<Symbol>; 2 * MAX_TOKEN_LENGTH], // symbol for each ancestor i
    a_tp: [f64; MAX_TOKEN_LENGTH],                    // token branch prior for each ancestor i
    a_n_tokens: [u32; MAX_TOKEN_LENGTH],              // number of tokens for each ancestor i
    a_final_token_length: [u8; MAX_TOKEN_LENGTH],     // final token length for each ancestor i
}

impl Node {
    /// Default values for a new non-root node (no children allocated).
    fn fresh() -> Self {
        Self {
            // posterior
            z: 1.0,
            nz: 0,
            mz: 0,
            // likelihood
            ll: 0.0,
            ul: 0.0,
            nl: 0,
            tl: [0.0; MAX_TOKEN_LENGTH],
            ntl: 0,
            cum_likelihood_frontier: false,
            // prior
            p: 0.0,
            mp: 0,
            fp: [0.0; MAX_TOKEN_LENGTH],
            tp: 0.0,
            mtp: 0,
            // tokenization
            final_token_length: 0u8,
            n_tokens: 0,
            // trie
            symbol: None,
            depth: 0,
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
    nz: u32,
    mz: u32,
    nl: u32,
    ntl: u32,
    mp: u32,
    mtp: u32,
}

impl Trie {
    fn new() -> Self {
        let nodes = vec![Node::root()];

        Self {
            nodes,
            root: 0,
            nz: 0,
            mz: 0,
            nl: 0,
            ntl: 0,
            mp: 0,
            mtp: 0,
        }
    }

    fn root_walker(&self) -> Walker {
        Walker {
            node: self.root,
            depth: 0,
            a_symbol: [None; 2 * MAX_TOKEN_LENGTH],
            a_tp: [0.0; MAX_TOKEN_LENGTH],
            a_n_tokens: [0; MAX_TOKEN_LENGTH],
            a_final_token_length: [0; MAX_TOKEN_LENGTH],
        }
    }

    fn descend(&self, mut walker: Walker, symbol: Symbol) -> Walker {
        let current_node = &self.nodes[walker.node];
        let base = current_node
            .children_start_index
            .expect("Trie::descend called on a node with no initialized children");

        for i in (1..walker.a_symbol.len()).rev() {
            walker.a_symbol[i] = walker.a_symbol[i - 1];
        }
        walker.a_symbol[0] = current_node.symbol;

        for i in (1..MAX_TOKEN_LENGTH).rev() {
            walker.a_tp[i] = walker.a_tp[i - 1];
            walker.a_n_tokens[i] = walker.a_n_tokens[i - 1];
            walker.a_final_token_length[i] = walker.a_final_token_length[i - 1];
        }
        walker.a_tp[0] = current_node.tp;
        walker.a_n_tokens[0] = current_node.n_tokens;
        walker.a_final_token_length[0] = current_node.final_token_length;

        walker.node = base as usize + symbol.to_slot() as usize;
        walker.depth += 1;
        walker
    }

    /// Temporarily compile-safe wrapper around the original WIP implementation.
    fn ensure_children(&mut self, walker: &Walker) {
        // Original WIP body preserved below for continued development:
        //
        // let parent_index = walker.node;
        // if self.nodes[parent_index].children_start_index.is_some() {
        //     return;
        // }
        // let base = self.nodes.len() as u32;
        // self.nodes[parent_index].children_start_index = Some(base);
        // let parent_node = &self.nodes[parent_index];
        // let child_depth = parent_node.depth + 1;
        // for symbol in Symbol::ALL {
        //     // tokenization
        //     // lemma: ab canonical, bc canonical => abc canonical
        //     let token_ancestor: Option<u8> = None;
        //     let child_final_token_length: Option<u8> = None;
        //     let child_n_tokens: Option<u32> = None;
        //     for (i, prev_token_length) in walker.a_final_token_length.iter().enumerate() {
        //         let prev_token = walker.a_symbol[i:i+prev_token_length].rev();
        //         let new_token = walker.a_symbol[:i].rev();
        //         new_token.push(symbol);
        //         let new_token_length = i+1;
        //         if bpe::canonical_pair(prev_token, new_token) {
        //             child_final_token_length = Some(new_token_length);
        //             child_n_tokens = Some(walker.a_n_tokens[i] + 1);
        //             break;
        //         }
        //     }
        //     self.nodes.push(Node {
        //         symbol: Some(symbol),
        //         depth: child_depth,
        //         children_start_index: None,
        //         // tokenization
        //         final_token_length: child_final_token_length.expect(),
        //         n_tokens: child_n_tokens.expect(),
        //         // prior
        //         tp: -(child_n_tokens.expect() as f64) * (NUM_TOKENS as f64).ln(),
        //         mtp: 0,
        //         // likelihood
        //         ll: 0.0,
        //         ul: 0.0,
        //
        //     });
        // }
        let _ = walker;
        let _ = NUM_TOKENS;
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
