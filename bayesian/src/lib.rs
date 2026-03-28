//! Core logic shared by the Python extension (`import bayesian`) and the dev CLI (`cargo run`).

#![allow(dead_code)]

pub mod bpe;
pub mod symbol;

use bpe::{NUM_PREFIXES, NUM_TOKENS, TinyLlamaWordTokenizer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use symbol::Symbol;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// --- byte trie (fixed alphabet, arena-allocated) -----------------------------------------------
const MAX_TOKEN_LENGTH: usize = 16;

type NodeIndex = usize;
type PredictionIndex = usize;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PredictionOrder {
    ZeroOrder(Option<String>),
    FirstOrder(String),
    FullOrder(Vec<String>),
}

#[derive(Clone, Debug)]
struct Prediction {
    order: PredictionOrder,
    m: u32,
    canonical_followers: Box<[bool]>,
    canonical_follower_for_prefix: Box<[bool]>,
    follower_probs: Box<[f64]>,
    follower_prob_for_prefix: Box<[f64]>,
    stop_prob: f64,
}

#[derive(Clone, Debug)]
struct PredictionRegistry {
    predictions: Vec<Prediction>,
    by_order: HashMap<PredictionOrder, PredictionIndex>,
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
    /// Index in `Trie::nodes` of the first child block, ordered like `Symbol::ALL` without the final `Start`.
    /// `None` if no child block has been allocated for this node yet.
    symbol: Symbol,
    children_start_index: Option<NodeIndex>,
}

#[derive(Clone, Debug)]
struct Trie {
    nodes: Vec<Node>,
    prediction_registry: PredictionRegistry,
    tokenizer: TinyLlamaWordTokenizer,
    root: NodeIndex,
    nz: u32,
    mz: u32,
    nl: u32,
    ntl: u32,
    mp: u32,
    mtp: u32,
}

#[derive(Clone, Debug)]
struct Walker {
    node: NodeIndex,
    depth: u32,
    a_symbol: [Option<Symbol>; MAX_TOKEN_LENGTH], // symbol for each ancestor i
    a_tp: [f64; MAX_TOKEN_LENGTH],                    // token branch prior for each ancestor i
    //
    a_prediction_index: [Option<PredictionIndex>; MAX_TOKEN_LENGTH], // prediction index for each ancestor i
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrieSnapshotNode {
    pub symbol: Symbol,
    pub z: f64,
    pub likelihood: f64,
    pub children: Vec<(Symbol, NodeIndex)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrieSnapshot {
    pub nodes: Vec<TrieSnapshotNode>,
    pub root: NodeIndex,
}

struct SnapshotWalker {
    node: NodeIndex,
}

impl Prediction {
    fn zero_order_for_final_token(
        final_token: Option<String>,
        tokenizer: &TinyLlamaWordTokenizer,
    ) -> Self {
        let canonical_followers_array = match final_token.as_deref() {
            Some(final_token) => tokenizer.canonical_followers(final_token),
            None => [true; NUM_TOKENS],
        };
        let canonical_counts_by_prefix =
            tokenizer.count_true_tokens_by_prefix::<NUM_PREFIXES>(&canonical_followers_array);
        let canonical_total = canonical_counts_by_prefix[tokenizer
            .prefix_lex_index("")
            .expect("empty prefix must always be present")];

        assert!(canonical_total != 0, "canonical_total must not be zero");
        let log_canonical_total = (canonical_total as f64).ln();

        let canonical_follower_for_prefix = canonical_counts_by_prefix
            .iter()
            .map(|&count| count != 0)
            .collect::<Box<[_]>>();

        let follower_probs = canonical_followers_array
            .iter()
            .map(|&is_canonical| {
                if is_canonical {
                    -log_canonical_total
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect::<Box<[_]>>();

        let follower_prob_for_prefix = canonical_counts_by_prefix
            .iter()
            .map(|&count| {
                if count != 0 {
                    (count as f64).ln() - log_canonical_total
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect::<Box<[_]>>();

        let canonical_followers = canonical_followers_array
            .into_iter()
            .collect::<Box<[_]>>();

        Self {
            order: PredictionOrder::ZeroOrder(final_token),
            m: 0,
            canonical_followers,
            canonical_follower_for_prefix,
            follower_probs,
            follower_prob_for_prefix,
            stop_prob: -f64::INFINITY,
        }
    }
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

impl Node {
    /// Default values for a new node (no children allocated).
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
            symbol: Symbol::Start,
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
impl Trie {
    fn new(tokenizer: TinyLlamaWordTokenizer) -> Self {
        assert_eq!(
            tokenizer.tokens().len(),
            NUM_TOKENS,
            "Trie expects exactly {NUM_TOKENS} tokenizer entries"
        );
        assert_eq!(
            tokenizer.prefix_count(),
            NUM_PREFIXES,
            "Trie expects exactly {NUM_PREFIXES} tokenizer prefixes"
        );
        let nodes = vec![Node::root()];

        Self {
            nodes,
            prediction_registry: PredictionRegistry::new(),
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
            a_symbol: [None; MAX_TOKEN_LENGTH],
            a_tp: [0.0; MAX_TOKEN_LENGTH],
            a_prediction_index: [None; MAX_TOKEN_LENGTH],
        }
    }

    fn expand_node_to_threshold(
        &mut self,
        threshold: f64,
        remaining_expand_budget: &mut usize,
        warned_expand_budget_exhausted: &mut bool,
        mut walker: Walker,
    ) {
        if self.nodes[walker.node].symbol == Symbol::Stop {
            return;
        }
        if *remaining_expand_budget == 0 {
            if !*warned_expand_budget_exhausted {
                eprintln!("warning: rough expand budget exhausted; stopping trie expansion early");
                *warned_expand_budget_exhausted = true;
            }
            return;
        }
        if self.nodes[walker.node].z > threshold {
            if self.nodes[walker.node].children_start_index.is_none() {
                self.ensure_prediction(&mut walker);
                self.ensure_children(&walker);
                *remaining_expand_budget -= 1;
            }
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start {
                    continue;
                }
                let child_walker = self.descend(walker.clone(), symbol);
                self.expand_node_to_threshold(
                    threshold,
                    remaining_expand_budget,
                    warned_expand_budget_exhausted,
                    child_walker,
                );
            }
        }
    }

    fn expand_trie_to_threshold(&mut self, threshold: f64, max_expand_budget: usize) {
        let walker = self.root_walker();
        let mut remaining_expand_budget = max_expand_budget;
        let mut warned_expand_budget_exhausted = false;
        self.expand_node_to_threshold(
            threshold,
            &mut remaining_expand_budget,
            &mut warned_expand_budget_exhausted,
            walker,
        );
    }

    fn snapshot_trie(&mut self, threshold: f64, max_expand_budget: usize) -> TrieSnapshot {
        self.expand_trie_to_threshold(threshold, max_expand_budget);
        self.to_snapshot(threshold)
    }

    fn descend(&self, mut walker: Walker, symbol: Symbol) -> Walker {
        let child_index = {
            let current_node = &self.nodes[walker.node];
            let base = current_node
            .children_start_index
            .expect("Trie::descend called on a node with no initialized children");
            base + symbol.to_slot() as usize
        };

        for i in (1..MAX_TOKEN_LENGTH).rev() {
            walker.a_symbol[i] = walker.a_symbol[i - 1];
            walker.a_tp[i] = walker.a_tp[i - 1];
            walker.a_prediction_index[i] = walker.a_prediction_index[i - 1];
        }
        let child_node = &self.nodes[child_index];
        walker.a_symbol[0] = Some(child_node.symbol);
        walker.a_tp[0] = child_node.tp;
        walker.a_prediction_index[0] = child_node.prediction;

        walker.node = child_index;
        walker.depth += 1;
        walker
    }

    fn push_likelihood(&mut self, walker: Walker) {
        let node_index = walker.node;
        let node_ul = self.nodes[node_index].ul;
        for symbol in Symbol::ALL {
            if symbol == Symbol::Start {
                continue;
            }
            let child_index = node_index + symbol.to_slot() as usize;
            let child_node = &mut self.nodes[child_index];
            child_node.ul += node_ul;
        }
        self.nodes[node_index].ul = 0.0;
    }

    fn sum_children_z(&self, walker: Walker) -> f64 {
        let node_index = walker.node;
        let mut sum = -f64::INFINITY;
        for symbol in Symbol::ALL {
            if symbol == Symbol::Start {
                continue;
            }
            let child_index = node_index + symbol.to_slot() as usize;
            let child_node = &self.nodes[child_index];
            sum = logaddexp(sum, child_node.z);
        }
        sum
    }

    fn ensure_prediction(&mut self, walker: &mut Walker) -> PredictionIndex {
        let node_index = walker.node;
        let prediction_index = if let Some(index) = self.nodes[node_index].prediction {
            index
        } else {
            let final_token_length = self.nodes[node_index].final_token_length;
            let final_token_string = Walker::symbol_slice_to_string(&walker.a_symbol[..final_token_length as usize]);
            let final_token = if final_token_string.is_empty() {
                None
            } else {
                Some(final_token_string)
            };
            let order = PredictionOrder::ZeroOrder(final_token.clone());
            let index = if let Some(existing) = self.prediction_registry.index_for_order(&order) {
                existing
            } else {
                self.prediction_registry
                    .alloc(Prediction::zero_order_for_final_token(
                        final_token,
                        &self.tokenizer,
                    ))
            };
            self.nodes[node_index].prediction = Some(index);
            index
        };
        walker.a_prediction_index[0] = Some(prediction_index);
        prediction_index
    }

    fn ensure_children(&mut self, walker: &Walker) {
        let parent_index = walker.node;
        if self.nodes[parent_index].children_start_index.is_some() {
            return;
        }

        assert!(
            self.nodes[parent_index].prediction.is_some(),
            "ensure_children requires parent prediction to be initialized"
        );

        let base = self.nodes.len();
        self.nodes[parent_index].children_start_index = Some(base as NodeIndex);

        let parent_ll = self.nodes[parent_index].ll;
        let parent_ul = self.nodes[parent_index].ul;
        let parent_nl = self.nodes[parent_index].nl;
        let parent_prediction_index = self.nodes[parent_index]
            .prediction
            .expect("ensure_children requires parent prediction to be initialized");
        let parent_prediction = &self.prediction_registry.predictions[parent_prediction_index];
        let available_prediction_depth = usize::min(MAX_TOKEN_LENGTH, walker.depth as usize + 1);

        for symbol in Symbol::ALL {
            if symbol == Symbol::Start {
                continue;
            }
            let mut child = Node {
                // trie
                symbol,
                children_start_index: None,
                // tokenization
                final_token_length: 0,
                n_tokens: 0,
                prediction: None,
                // prior
                p: -f64::INFINITY,
                tp: -f64::INFINITY,
                mp: 0,
                mtp: 0,
                fp: [-f64::INFINITY; MAX_TOKEN_LENGTH],
                // likelihood
                ll: parent_ll,
                ul: parent_ul,
                nl: parent_nl,
                tl: [-f64::INFINITY; MAX_TOKEN_LENGTH],
                ntl: 0,
                cum_likelihood_frontier: false,
                // posterior
                z: -f64::INFINITY,
                nz: 0,
                mz: 0,
            };

            if symbol == Symbol::Stop {
                child.final_token_length = 1;
                child.p = walker.a_tp[0] + parent_prediction.stop_prob;
                child.z = child.p + child.ll;
                self.nodes.push(child);
                continue;
            }

            for i in 0..available_prediction_depth {
                let mut new_prefix = Walker::symbol_slice_to_string(&walker.a_symbol[..i]);
                new_prefix.push(symbol.to_byte() as char);
                let new_token = new_prefix.clone();
                let prediction_index =
                    walker.a_prediction_index[i].expect("prediction index must be valid");
                let prediction = &self.prediction_registry.predictions[prediction_index];
                let maybe_new_token_lexindex = self.tokenizer.lex_index(&new_token);
                let maybe_new_prefix_lexindex = self.tokenizer.prefix_lex_index(&new_prefix);
                // is ancestor i the token ancestor of our canonical tokenization if we are the end of the string?
                if let Some(new_token_lexindex) = maybe_new_token_lexindex {
                    let canonical_pair = prediction.canonical_followers[new_token_lexindex];
                    if canonical_pair {
                        child.final_token_length = (i + 1) as u8;
                        child.tp = walker.a_tp[i] + prediction.follower_probs[new_token_lexindex];
                    }
                }
                // is it possible for ancestor i to be the closest token ancestor of the target string's canonical tokenization?
                if let Some(new_prefix_lexindex) = maybe_new_prefix_lexindex {
                    let truncation_possible = prediction.canonical_follower_for_prefix[new_prefix_lexindex];
                    if truncation_possible {
                        child.fp[i] = prediction.follower_prob_for_prefix[new_prefix_lexindex];
                        child.p = logaddexp(child.p, walker.a_tp[i] + child.fp[i]);
                        child.tl[i] = 0.0;
                    }
                }
            }
            child.z = child.p + child.ll;
            self.nodes.push(child);
        }
    }

    fn to_snapshot(&self, threshold: f64) -> TrieSnapshot {
        let mut index_map = vec![None; self.nodes.len()];
        let mut snapshot_nodes = Vec::new();
        let root = self.snapshot_subtree(
            self.root,
            threshold,
            &mut index_map,
            &mut snapshot_nodes,
        );
        TrieSnapshot {
            nodes: snapshot_nodes,
            root,
        }
    }

    fn snapshot_subtree(
        &self,
        node_index: NodeIndex,
        threshold: f64,
        index_map: &mut [Option<NodeIndex>],
        snapshot_nodes: &mut Vec<TrieSnapshotNode>,
    ) -> NodeIndex {
        if let Some(snapshot_index) = index_map[node_index] {
            return snapshot_index;
        }

        let node = &self.nodes[node_index];
        let snapshot_index = snapshot_nodes.len();
        index_map[node_index] = Some(snapshot_index);
        snapshot_nodes.push(TrieSnapshotNode {
            symbol: node.symbol,
            z: node.z,
            likelihood: 0.0,
            children: Vec::new(),
        });

        let mut children = Vec::new();
        if let Some(base) = node.children_start_index {
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start {
                    continue;
                }
                let child_index = base + symbol.to_slot() as usize;
                let child = &self.nodes[child_index];
                if child.z > threshold {
                    let child_snapshot_index =
                        self.snapshot_subtree(child_index, threshold, index_map, snapshot_nodes);
                    children.push((symbol, child_snapshot_index));
                }
            }
        }
        snapshot_nodes[snapshot_index].children = children;
        snapshot_index
    }

    fn apply_likelihood_update_subtrie(
        &mut self, snapshot: &TrieSnapshot,
        mut walker: Walker,
        mut snapshot_walker: SnapshotWalker
    ) {
        let mut new_z = -f64::INFINITY;
        let snapshot_node = &snapshot.nodes[snapshot_walker.node];
        let trie_node = &mut self.nodes[walker.node];
        trie_node.ul += snapshot_node.likelihood;
        self.push_likelihood(walker); // todo: it is expensive and probably unnecessary to push this on the frontier
        if snapshot_node.children.is_empty() {
            let p_old = trie_node.p;  // todo
            let p_new = trie_node.p;
            let l_delta = 0.0;
            new_z = trie_node.z + p_new - p_old + l_delta;
        } else {
            // compute new z by summing the z values of all children
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start {
                    continue;
                }
                let child_walker = self.descend(walker.clone(), symbol);
                // push
                let child_node = &mut self.nodes[child_walker.node];
                child_node.ul += trie_node.ul;
                // apply likelihood update to child
                let maybe_child_snapshot_walker = snapshot.descend(snapshot_walker, symbol);
                if let Some(child_snapshot_walker) = maybe_child_snapshot_walker {
                    self.apply_likelihood_update_subtrie(snapshot, child_walker, child_snapshot_walker);
                }

                new_z = logaddexp(new_z, child_node.z);
            }
            new_z = self.sum_children_z(walker);
        }
        trie_node.z = new_z;
    }
    fn apply_likelihood_update_root(&mut self, snapshot: &TrieSnapshot) {
        let mut snapshot_walker = snapshot.root_walker();
    }
}
impl Walker {
    fn symbol_slice_to_string(symbols: &[Option<Symbol>]) -> String {
        let mut out = String::with_capacity(symbols.len());
        for symbol in symbols.iter().flatten().rev() {
            out.push(symbol.to_byte() as char);
        }
        out
    }

}

impl TrieSnapshot {
    fn root_walker(&self) -> SnapshotWalker {
        SnapshotWalker {
            node: self.root,
        }
    }

    fn descend(&self, walker: SnapshotWalker, target_symbol: Symbol) -> Option<SnapshotWalker> {
        let snapshot_node = &self.nodes[walker.node];
        for (symbol, child_index) in &snapshot_node.children {
            if *symbol == target_symbol {
                return Some(SnapshotWalker {
                    node: *child_index,
                });
            }
        }
        None
    }
}



fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

// ------------------------------------------------------------------------------------------------

#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn add(a: i64, b: i64) -> i64 {
    a + b + 20
}

fn browser_tokenizer() -> TinyLlamaWordTokenizer {
    TinyLlamaWordTokenizer::from_tokenizer_json_str(bpe::TOKENIZER_JSON_STR)
}

fn browser_trie() -> Trie {
    Trie::new(browser_tokenizer())
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct BayesianSession {
    trie: Trie,
    threshold: f64,
    max_expand_budget: usize,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl BayesianSession {
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(threshold: f64, max_expand_budget: usize) -> Self {
        Self {
            trie: browser_trie(),
            threshold,
            max_expand_budget,
        }
    }

    pub fn snapshot_json(&mut self) -> String {
        let snapshot = self
            .trie
            .snapshot_trie(self.threshold, self.max_expand_budget);
        serde_json::to_string(&snapshot).expect("TrieSnapshot should serialize to JSON")
    }

    pub fn reset(&mut self) {
        self.trie = browser_trie();
    }

    pub fn update_snapshot_likelihoods(&mut self, snapshot_json: String) -> String {
        let mut snapshot: TrieSnapshot =
            serde_json::from_str(&snapshot_json).expect("snapshot_json should deserialize to TrieSnapshot");
        for node in &mut snapshot.nodes {
            node.z += node.likelihood;
        }
        serde_json::to_string(&snapshot).expect("updated TrieSnapshot should serialize to JSON")
    }
}

// get trie
// expand trie
// queue token branches
// apply likelihood update
// apply prior update

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "add")]
fn add_py(a: i64, b: i64) -> i64 {
    add(a, b)
}

#[cfg(feature = "python")]
#[pymodule]
fn bayesian(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::TOKENIZER_JSON_PATH;

    fn dump_trie_after_expand(threshold: f64, max_expand_budget: usize) {
        let mut trie = Trie::from_tokenizer_json(TOKENIZER_JSON_PATH);
        trie.expand_trie_to_threshold(threshold, max_expand_budget);

        let expanded_nodes = trie
            .nodes
            .iter()
            .filter(|node| node.children_start_index.is_some())
            .count();
        let predicted_nodes = trie
            .nodes
            .iter()
            .filter(|node| node.prediction.is_some())
            .count();
        let finite_p_nodes = trie.nodes.iter().filter(|node| node.p.is_finite()).count();
        let finite_tp_nodes = trie.nodes.iter().filter(|node| node.tp.is_finite()).count();
        let finite_z_nodes = trie.nodes.iter().filter(|node| node.z.is_finite()).count();

        eprintln!(
            "trie summary: nodes={} predictions={} expanded={} finite_p={} finite_tp={} finite_z={}",
            trie.nodes.len(),
            predicted_nodes,
            expanded_nodes,
            finite_p_nodes,
            finite_tp_nodes,
            finite_z_nodes,
        );

        let root = &trie.nodes[trie.root];
        eprintln!(
            "root: symbol={:?} prediction={:?} children_start_index={:?} p={} tp={} z={}",
            root.symbol, root.prediction, root.children_start_index, root.p, root.tp, root.z
        );

        if let Some(base) = root.children_start_index {
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start {
                    continue;
                }
                let child_index = base + symbol.to_slot() as usize;
                let child = &trie.nodes[child_index];
                eprintln!(
                    "root child {:?}: idx={} pred={:?} child_base={:?} final_token_length={} n_tokens={} p={} tp={} z={}",
                    symbol,
                    child_index,
                    child.prediction,
                    child.children_start_index,
                    child.final_token_length,
                    child.n_tokens,
                    child.p,
                    child.tp,
                    child.z,
                );
            }
        }

        let stop_nodes: Vec<(usize, &Node)> = trie
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.symbol == Symbol::Stop)
            .collect();
        eprintln!("stop nodes: count={}", stop_nodes.len());
        for (index, node) in stop_nodes.iter().take(20) {
            eprintln!(
                "stop node idx={} pred={:?} child_base={:?} final_token_length={} n_tokens={} p={} tp={} z={}",
                index,
                node.prediction,
                node.children_start_index,
                node.final_token_length,
                node.n_tokens,
                node.p,
                node.tp,
                node.z,
            );
        }

        let leaf_nodes: Vec<(usize, &Node)> = trie
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.children_start_index.is_none())
            .collect();
        eprintln!("leaf nodes: count={}", leaf_nodes.len());
        for (index, node) in leaf_nodes.iter().take(20) {
            eprintln!(
                "leaf node idx={} symbol={:?} pred={:?} final_token_length={} n_tokens={} p={} tp={} z={}",
                index,
                node.symbol,
                node.prediction,
                node.final_token_length,
                node.n_tokens,
                node.p,
                node.tp,
                node.z,
            );
        }

        let mut expanded_by_depth = std::collections::BTreeMap::<usize, usize>::new();
        for (index, node) in trie.nodes.iter().enumerate() {
            if node.children_start_index.is_none() {
                continue;
            }
            let mut depth = 0usize;
            let mut cursor = index;
            while cursor != trie.root {
                depth += 1;
                cursor = (cursor - 1) / 28;
            }
            *expanded_by_depth.entry(depth).or_insert(0) += 1;
        }
        eprintln!("expanded by depth: {:?}", expanded_by_depth);
    }

    #[test]
    fn trie_expand_threshold_smoke_test() {
        dump_trie_after_expand((1.0 / 30.0_f64).ln(), 100);
    }

    #[test]
    fn trie_expand_threshold_one_over_one_hundred() {
        dump_trie_after_expand((1.0 / 100.0_f64).ln(), 100_000);
    }

    #[test]
    fn trie_expand_threshold_one_over_two_hundred() {
        dump_trie_after_expand((1.0 / 200.0_f64).ln(), 100_000);
    }

    #[test]
    fn update_snapshot_likelihoods_adds_likelihood_into_z() {
        let snapshot = TrieSnapshot {
            root: 0,
            nodes: vec![
                TrieSnapshotNode {
                    symbol: Symbol::Start,
                    z: 1.5,
                    likelihood: -2.0,
                    children: vec![(Symbol::A, 1)],
                },
                TrieSnapshotNode {
                    symbol: Symbol::A,
                    z: -3.0,
                    likelihood: 0.0,
                    children: Vec::new(),
                },
            ],
        };

        let updated_json = BayesianSession::new(0.0, 0).update_snapshot_likelihoods(
            serde_json::to_string(&snapshot).expect("snapshot should serialize"),
        );
        let updated: TrieSnapshot =
            serde_json::from_str(&updated_json).expect("updated snapshot should deserialize");

        assert_eq!(updated.nodes[0].z, -0.5);
        assert_eq!(updated.nodes[1].z, -3.0);
        assert_eq!(updated.nodes[0].likelihood, -2.0);
        assert_eq!(updated.nodes[1].likelihood, 0.0);
    }
}
