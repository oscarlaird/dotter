//! Core logic shared by the Python extension (`import bayesian`) and the dev CLI (`cargo run`).

#![allow(dead_code)]

pub mod bpe;
pub mod symbol;

use bpe::TinyLlamaWordTokenizer;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use symbol::Symbol;

// --- byte trie (fixed alphabet, arena-allocated) -----------------------------------------------
const MAX_TOKEN_LENGTH: usize = 16;

const NUM_TOKENS: usize = 17250;

type NodeIndex = usize;
type PredictionIndex = usize;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PredictionOrder {
    ZeroOrder(String),
    FirstOrder(String),
    FullOrder(Vec<String>),
}

#[derive(Clone, Debug)]
struct Prediction {
    order: PredictionOrder,
    m: u32,
    canonical_followers: [bool; NUM_TOKENS],
    cum_canonical_followers: [u32; NUM_TOKENS],
    follower_probs: [f64; NUM_TOKENS],
    cum_follower_probs: [f64; NUM_TOKENS],
}

impl Prediction {
    fn from_order_and_followers(
        order: PredictionOrder,
        canonical_followers: [bool; NUM_TOKENS],
    ) -> Self {
        let mut cum_canonical_followers = [0u32; NUM_TOKENS];
        let mut canonical_total = 0u32;
        for (idx, is_canonical) in canonical_followers.iter().copied().enumerate() {
            canonical_total += u32::from(is_canonical);
            cum_canonical_followers[idx] = canonical_total;
        }

        let mut follower_probs = [0.0; NUM_TOKENS];
        if canonical_total != 0 {
            let mass = 1.0 / canonical_total as f64;
            for (idx, is_canonical) in canonical_followers.iter().copied().enumerate() {
                if is_canonical {
                    follower_probs[idx] = mass;
                }
            }
        }

        let mut cum_follower_probs = [0.0; NUM_TOKENS];
        let mut prob_total = 0.0;
        for (idx, prob) in follower_probs.iter().copied().enumerate() {
            prob_total += prob;
            cum_follower_probs[idx] = prob_total;
        }

        Self {
            order,
            m: 0,
            canonical_followers,
            cum_canonical_followers,
            follower_probs,
            cum_follower_probs,
        }
    }
}


#[derive(Clone, Debug)]
struct PredictionRegistry {
    predictions: Vec<Prediction>,
    by_order: HashMap<PredictionOrder, PredictionIndex>,
}

impl PredictionRegistry {
    fn new() -> Self {
        Self {
            predictions: Vec::new(),
            by_order: HashMap::new(),
        }
    }

    fn alloc(&mut self, prediction: Prediction) -> PredictionIndex {
        let index = self.predictions.len();
        self.by_order.insert(prediction.order.clone(), index);
        self.predictions.push(prediction);
        index
    }

    fn get(&self, index: PredictionIndex) -> Option<&Prediction> {
        self.predictions.get(index)
    }

    fn index_for_order(&self, order: &PredictionOrder) -> Option<PredictionIndex> {
        self.by_order.get(order).copied()
    }

    fn get_mut(&mut self, index: PredictionIndex) -> Option<&mut Prediction> {
        self.predictions.get_mut(index)
    }

    fn len(&self) -> usize {
        self.predictions.len()
    }

    fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }
}

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
    prediction: Option<PredictionIndex>,
    // children
    /// Index in `Trie::nodes` of the first of `RADIX` consecutive child nodes (slot `s` → `nodes[base + s]`).
    /// `None` if no child block has been allocated for this node yet.
    symbol: Option<Symbol>,
    depth: u32,
    children_start_index: Option<NodeIndex>,
}

#[derive(Clone, Debug)]
struct Walker {
    node: NodeIndex,
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
            prediction: None,
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

impl Walker {
    fn last_token_string(&self) -> String {
        let len = self.a_final_token_length[0] as usize;
        let mut out = String::with_capacity(len);
        for symbol in self.a_symbol[..len].iter().rev().flatten() {
            out.push(symbol.to_byte() as char);
        }
        out
    }
}

#[derive(Clone, Debug)]
struct Trie {
    nodes: Vec<Node>,
    predictions: PredictionRegistry,
    tokenizer: TinyLlamaWordTokenizer,
    root: NodeIndex,
    nz: u32,
    mz: u32,
    nl: u32,
    ntl: u32,
    mp: u32,
    mtp: u32,
}

impl Trie {
    fn new(tokenizer: TinyLlamaWordTokenizer) -> Self {
        assert_eq!(
            tokenizer.tokens().len(),
            NUM_TOKENS,
            "Trie expects exactly {NUM_TOKENS} tokenizer entries"
        );
        let nodes = vec![Node::root()];

        Self {
            nodes,
            predictions: PredictionRegistry::new(),
            tokenizer,
            root: 0,
            nz: 0,
            mz: 0,
            nl: 0,
            ntl: 0,
            mp: 0,
            mtp: 0,
        }
    }

    fn from_tokenizer_json(path: impl AsRef<Path>) -> Self {
        let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json(path);
        Self::new(tokenizer)
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

        walker.node = base + symbol.to_slot() as usize;
        walker.depth += 1;
        walker
    }

    fn ensure_children(&mut self, walker: &Walker) {
        let parent_index = walker.node;
        if self.nodes[parent_index].children_start_index.is_some() {
            return;
        }
        let base = self.nodes.len();
        self.nodes[parent_index].children_start_index = Some(base);
        let child_depth = self.nodes[parent_index].depth + 1;
        if self.nodes[parent_index].prediction.is_none() {
            let last_token_string = walker.last_token_string();
            let order = PredictionOrder::ZeroOrder(last_token_string.clone());
            let prediction_index = if let Some(index) = self.predictions.index_for_order(&order) {
                index
            } else {
                self.predictions
                    .alloc(self.zero_order_prediction(last_token_string))
            };
            self.nodes[parent_index].prediction = Some(prediction_index);
        }
        // Original child-expansion body is still WIP:
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
        //         prediction: None,
        //         // prior
        //         tp: -(child_n_tokens.expect() as f64) * (NUM_TOKENS as f64).ln(),
        //         mtp: 0,
        //         // likelihood
        //         ll: 0.0,
        //         ul: 0.0,
        //     });
        // }
        let _ = child_depth;
    }

    fn zero_order_prediction(&self, last_token: String) -> Prediction {
        let canonical_followers = self.zero_order_canonical_followers(&last_token);
        Prediction::from_order_and_followers(
            PredictionOrder::ZeroOrder(last_token),
            canonical_followers,
        )
    }

    fn zero_order_canonical_followers(&self, last_token: &str) -> [bool; NUM_TOKENS] {
        assert!(
            !last_token.is_empty(),
            "zero_order_canonical_followers requires a non-empty last token"
        );
        let followers = self
            .tokenizer
            .canonical_followers(last_token)
            ;
        assert_eq!(
            followers.len(),
            NUM_TOKENS,
            "canonical_followers must return one flag per token"
        );
        followers
            .try_into()
            .expect("canonical_followers length must match NUM_TOKENS")
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
