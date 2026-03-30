use std::path::Path;

use crate::bpe::{NUM_PREFIXES, NUM_TOKENS, TinyLlamaWordTokenizer};
use crate::symbol::Symbol;

use super::{
    logaddexp, NodeIndex, Prediction, PredictionIndex, PredictionOrder, PredictionRegistry,
    SnapshotWalker, TrieSnapshot, TrieSnapshotNode, MAX_TOKEN_LENGTH,
};

#[derive(Clone, Debug)]
struct Node {
    // trie
    /// Index in `Trie::nodes` of the first child block, ordered like `Symbol::ALL` without the final `Start`.
    /// `None` if no child block has been allocated for this node yet.
    symbol: Symbol,
    children_start_index: Option<NodeIndex>,
    // tokenization
    truncation_possible: [bool; MAX_TOKEN_LENGTH],
    final_token_length: u8,
    n_tokens: u32,
    prediction: Option<PredictionIndex>,
    prediction_last_change: i32,
    new_token_lexindex: usize,
    new_prefix_lexindex: [usize; MAX_TOKEN_LENGTH],
    // prior
    p: f64,                      // log string branch prior
    p_last_change: i32,
    p_old: f64,                  // log string branch prior at the posterior prior time
    mp: i32,                     // prior tracking time
    fp: [f64; MAX_TOKEN_LENGTH], // token fans for each ancestor i
    tp: f64,                     // log token branch prior
    tp_last_change: i32,
    mtp: i32,                    // token branch prior tracking time
    // likelihood
    l: f64,                     // log likelihood
    l_old: f64,                 // log likelihood at the posterior likelihood time
    nl: i32,                    // likelihood tracking time
    ul: f64,                    // log upper likelihood
    tl: [f64; MAX_TOKEN_LENGTH], // log token branch likelihood for each ancestor i,  i.e., log "Maximum Truncation Compatible Descendant Likelihood", for each ancestor i
    ntl: i32,                    // token branch likelihood tracking time
    cum_likelihood_frontier: bool,
    // posterior
    z: f64,  // log unnormalized posterior
    nz: i32, // posterior likelihood time
    mz: i32, // posterior prior time
}

#[derive(Clone, Debug)]
pub(crate) struct Trie {
    nodes: Vec<Node>,
    prediction_registry: PredictionRegistry,
    pub(crate) tokenizer: TinyLlamaWordTokenizer,
    root: NodeIndex,
    n: i32, // determines meaning/correctness of ul
    m: i32, // determines meaning/correctness of last_changed
}

#[derive(Clone, Debug)]
struct Walker {
    node: NodeIndex,
    depth: u32,
    a_symbol: [Option<Symbol>; MAX_TOKEN_LENGTH], // symbol for each ancestor i
    a_tp: [f64; MAX_TOKEN_LENGTH],                // token branch prior for each ancestor i
    a_prediction_index: [Option<PredictionIndex>; MAX_TOKEN_LENGTH], // prediction index for each ancestor i
    a_p_last_change: [i32; MAX_TOKEN_LENGTH],
    a_tp_last_change: [i32; MAX_TOKEN_LENGTH],
    a_prediction_last_change: [i32; MAX_TOKEN_LENGTH],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DescentType {
    LikelihoodUpdate,
    PriorUpdate,
    Expansion,
}

impl Node {
    /// Default values for a new node (no children allocated).
    fn fresh() -> Self {
        Self {
            // trie
            symbol: Symbol::Start,
            children_start_index: None,
            // tokenization
            final_token_length: 0,
            n_tokens: 0,
            prediction: None,
            prediction_last_change: -1,
            truncation_possible: [false; MAX_TOKEN_LENGTH],
            new_token_lexindex: usize::MAX, // invalid
            new_prefix_lexindex: [usize::MAX; MAX_TOKEN_LENGTH], // invalid
            // prior
            p: -f64::INFINITY,
            p_last_change: -1,
            p_old: -f64::INFINITY,
            tp: -f64::INFINITY,
            tp_last_change: -1,
            mp: -1,
            mtp: -1,
            fp: [-f64::INFINITY; MAX_TOKEN_LENGTH],
            // likelihood
            ul: 0.0,
            l: 0.0,
            l_old: 0.0,
            nl: -1,
            tl: [-f64::INFINITY; MAX_TOKEN_LENGTH],
            ntl: 0,
            cum_likelihood_frontier: true,
            // posterior
            z: -f64::INFINITY,
            nz: -1,
            mz: -1,
        }
    }

    fn root() -> Self {
        Self {
            // tokenization
            final_token_length: 1,
            n_tokens: 1,
            // prior
            p: 0.0,
            p_last_change: 0,
            p_old: 0.0,
            tp: 0.0,
            tp_last_change: 0,
            mp: 0,
            mtp: 0,
            // likelihood
            nl: 0,
            ntl: 0,
            // posterior
            z: 0.0,
            nz: 0,
            mz: 0,
            ..Self::fresh()
        }
    }
}

impl Trie {
    // initialization
    pub(crate) fn new(tokenizer: TinyLlamaWordTokenizer) -> Self {
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
            n: 0,
            m: 0,
        }
    }

    fn from_tokenizer_json(path: impl AsRef<Path>) -> Self {
        let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json(path);
        Self::new(tokenizer)
    }

    // walking
    fn root_walker(&self) -> Walker {
        Walker {
            node: self.root,
            depth: 0,
            a_symbol: [None; MAX_TOKEN_LENGTH],
            a_tp: [0.0; MAX_TOKEN_LENGTH],
            a_prediction_index: [None; MAX_TOKEN_LENGTH],
            a_p_last_change: [-1; MAX_TOKEN_LENGTH],
            a_tp_last_change: [-1; MAX_TOKEN_LENGTH],
            a_prediction_last_change: [-1; MAX_TOKEN_LENGTH],
        }
    }

    fn descend(&self, walker: &Walker, symbol: Symbol) -> Walker {
        let mut walker = walker.clone();
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
            walker.a_p_last_change[i] = walker.a_p_last_change[i - 1];
            walker.a_tp_last_change[i] = walker.a_tp_last_change[i - 1];
            walker.a_prediction_last_change[i] = walker.a_prediction_last_change[i - 1];
        }

        let child_node = &self.nodes[child_index];
        walker.a_symbol[0] = Some(child_node.symbol);
        walker.a_tp[0] = child_node.tp;
        walker.a_prediction_index[0] = child_node.prediction;
        walker.a_p_last_change[0] = child_node.p_last_change;
        walker.a_tp_last_change[0] = child_node.tp_last_change;
        walker.a_prediction_last_change[0] = child_node.prediction_last_change;

        walker.node = child_index;
        walker.depth += 1;
        walker
    }

    // defaults
    fn default_visit_budget(&self) -> i32 {
        200
    }

    fn default_expand_threshold(&self) -> f64 {
        (0.01_f64).ln()
    }

    // recalc (expansion and updates)
    // Each of the three kinds corresponds to a public API
    // - apply likelihood updates
    // - apply prior updates
    // - expand the trie to a threshold
    // EXPERIMENTAL (START)
    fn recalc_to_frontier_and_back(
        &mut self,
        mut walker: Walker,
        descent_type: DescentType,
        remaining_visit_budget: &mut i32,
        // likelihood update
        likelihood_snapshot: Option<&TrieSnapshot>,
        snapshot_walker: Option<SnapshotWalker>,
        // prior update
        new_prediction_symbol_seq: Option<&[Symbol]>,
        new_prediction_index: Option<PredictionIndex>,
        comparable_to_pred_node: bool,
        // expansion
        threshold: Option<f64>,
    ) -> f64 {
        let node_index = walker.node;
        // 1. apply update
        match descent_type {
            DescentType::LikelihoodUpdate => {
                let snapshot_walker = snapshot_walker
                    .as_ref()
                    .expect("likelihood update requires snapshot walker");
                if !snapshot_walker.has_children_in_snapshot {
                    let new_likelihood = snapshot_walker.likelihood;
                    let node = &mut self.nodes[node_index];
                    node.ul += new_likelihood;
                    node.l += new_likelihood;
                    node.nl = self.n;
                } else {
                    let node = &mut self.nodes[node_index];
                    node.cum_likelihood_frontier = false;
                }
            }
            DescentType::PriorUpdate => {
                let new_prediction_symbol_seq = new_prediction_symbol_seq
                    .expect("prior update requires prediction symbol sequence");
                let is_pred_node =
                    comparable_to_pred_node && new_prediction_symbol_seq.len() == walker.depth as usize;
                if is_pred_node {
                    let node = &mut self.nodes[node_index];
                    node.prediction = new_prediction_index;
                    node.prediction_last_change = self.m;
                }
            }
            DescentType::Expansion => {}
        };
        // 2. recalc prior
        // tp
        let idx = self.nodes[node_index].final_token_length as usize;
        let new_tp_last_change =
            walker.a_tp_last_change[idx].max(walker.a_prediction_last_change[idx]);
        let tp_last_change = self.nodes[node_index].tp_last_change;
        if new_tp_last_change > tp_last_change {
            let a_pred_index = walker.a_prediction_index[idx].unwrap();
            let a_pred = self.prediction_registry.get(a_pred_index).unwrap();
            let node_symbol = self.nodes[node_index].symbol;
            let new_token_lexindex = self.nodes[node_index].new_token_lexindex;
            let final_token_prob = if node_symbol != Symbol::Stop {
                a_pred.follower_probs[new_token_lexindex]
            } else {
                // Symbol::Stop
                a_pred.stop_prob
            };
            let node = &mut self.nodes[node_index];
            node.tp = walker.a_tp[idx] + final_token_prob;
            node.tp_last_change = new_tp_last_change;
        }
        self.nodes[node_index].mtp = self.m;
        // p
        let node_symbol = self.nodes[node_index].symbol;
        let truncation_possible = self.nodes[node_index].truncation_possible;
        let new_prefix_lexindex = self.nodes[node_index].new_prefix_lexindex;
        let mut new_p_last_change = self.nodes[node_index].p_last_change;
        for i in 0..MAX_TOKEN_LENGTH {
            if truncation_possible[i] {
                new_p_last_change = new_p_last_change.max(walker.a_tp_last_change[i]);
                new_p_last_change = new_p_last_change.max(walker.a_prediction_last_change[i]);
            }
        }
        let p_last_change = self.nodes[node_index].p_last_change;
        if new_p_last_change > p_last_change {
            let mut new_p = -f64::INFINITY;
            let mut new_fp = self.nodes[node_index].fp;
            for i in 0..MAX_TOKEN_LENGTH {
                if truncation_possible[i] {
                    let a_pred_index = walker.a_prediction_index[i].unwrap();
                    let a_pred = self.prediction_registry.get(a_pred_index).unwrap();
                    new_fp[i] = if node_symbol != Symbol::Stop {
                        a_pred.follower_prob_for_prefix[new_prefix_lexindex[i]]
                    } else {
                        // Symbol::Stop
                        assert!(i == 1);
                        a_pred.stop_prob
                    };
                    new_p = logaddexp(new_p, walker.a_tp[i] + new_fp[i]);
                }
            }
            let node = &mut self.nodes[node_index];
            node.fp = new_fp;
            node.p = new_p;
            node.p_last_change = new_p_last_change;
        }
        self.nodes[node_index].mp = self.m;
        // 3. recalc z
        {
            let node = &mut self.nodes[node_index];
            if node.cum_likelihood_frontier {
                node.z = node.p + node.l;
            } else {
                node.z += (node.p - node.p_old) + (node.l - node.l_old);
            }
            node.p_old = node.p;
            node.l_old = node.l;
            node.nz = self.n;
            node.mz = self.m;
        }
        // 4. check for stop
        let node_cum_likelihood_frontier = self.nodes[node_index].cum_likelihood_frontier;
        let node_symbol = self.nodes[node_index].symbol;
        let node_z = self.nodes[node_index].z;
        let stop = match descent_type {
            DescentType::LikelihoodUpdate => {
                assert!(likelihood_snapshot.is_some());
                snapshot_walker
                    .as_ref()
                    .expect("likelihood update requires snapshot walker")
                    .has_children_in_snapshot
            }
            DescentType::PriorUpdate => {
                let new_prediction_symbol_seq = new_prediction_symbol_seq
                    .expect("prior update requires prediction symbol sequence");
                let cum_l_frontier = node_cum_likelihood_frontier;
                let extends_pred_node =
                    comparable_to_pred_node && walker.depth > new_prediction_symbol_seq.len() as u32;
                let is_space = node_symbol == Symbol::Space;
                cum_l_frontier || !comparable_to_pred_node || (extends_pred_node && is_space)
            }
            DescentType::Expansion => {
                assert!(threshold.is_some());
                node_z < threshold.unwrap()
            }
        };
        // 5. ensure prediction and children
        if !stop {
            *remaining_visit_budget -= 1;
            assert!(*remaining_visit_budget >= 0);
            self.ensure_prediction(&mut walker);
            self.ensure_children(&walker);
        }
        // 6. push likelihood
        if !stop {
            let node_ul = self.nodes[node_index].ul;
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start {
                    continue;
                }
                let child_walker = self.descend(&walker, symbol);
                let child_node = &mut self.nodes[child_walker.node];
                child_node.ul += node_ul;
                child_node.l += node_ul;
                child_node.nl = self.n;
            }
            self.nodes[node_index].ul = 0.0;
        }
        // 7. recurse
        if !stop {
            let is_update = match descent_type {
                DescentType::LikelihoodUpdate => true,
                DescentType::PriorUpdate => true,
                DescentType::Expansion => false,
            };
            let mut children_sum = -f64::INFINITY;
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start {
                    continue;
                }
                let child_walker = self.descend(&walker, symbol);
                let child_snapshot_walker = match descent_type {
                    DescentType::LikelihoodUpdate => Some(
                        likelihood_snapshot
                            .expect("likelihood update requires snapshot")
                            .descend(
                                snapshot_walker
                                    .as_ref()
                                    .expect("likelihood update requires snapshot walker"),
                                symbol,
                            ),
                    ),
                    DescentType::PriorUpdate | DescentType::Expansion => None,
                };
                let child_comparable_to_pred_node = match descent_type {
                    DescentType::PriorUpdate => {
                        let new_prediction_symbol_seq = new_prediction_symbol_seq
                            .expect("prior update requires prediction symbol sequence");
                        if walker.depth < new_prediction_symbol_seq.len() as u32 {
                            comparable_to_pred_node
                                && new_prediction_symbol_seq[walker.depth as usize] == symbol
                        } else {
                            comparable_to_pred_node
                        }
                    }
                    DescentType::LikelihoodUpdate | DescentType::Expansion => {
                        comparable_to_pred_node
                    }
                };
                let child_z = self.recalc_to_frontier_and_back(
                    child_walker,
                    descent_type,
                    remaining_visit_budget,
                    // likelihood update
                    likelihood_snapshot,
                    child_snapshot_walker,
                    // prior update
                    new_prediction_symbol_seq,
                    new_prediction_index,
                    child_comparable_to_pred_node,
                    // expansion
                    threshold,
                );
                // 6. upprop z
                if is_update {
                    children_sum = logaddexp(children_sum, child_z);
                }
            }
            if is_update {
                self.nodes[node_index].z = children_sum;
            }
        }
        self.nodes[node_index].z
    }

    pub(crate) fn apply_likelihood_update(&mut self, snapshot: &TrieSnapshot) {
        let root_walker = self.root_walker();
        let snapshot_walker = snapshot.root_walker();
        self.n += 1;
        self.recalc_to_frontier_and_back(
            root_walker,
            DescentType::LikelihoodUpdate,
            &mut self.default_visit_budget(),
            // likelihood update
            Some(snapshot),
            Some(snapshot_walker),
            // prior update
            None,
            None,
            false,
            // expansion
            None,
        );
    }

    pub(crate) fn apply_prior_update(&mut self, node_string: String, prediction: Prediction) {
        let root_walker = self.root_walker();
        let symbol_seq: Vec<Symbol> = node_string
            .chars()
            .filter_map(|c| Symbol::from_byte(c as u8))
            .collect();
        let pred_index = self.prediction_registry.alloc(prediction);
        self.recalc_to_frontier_and_back(
            root_walker,
            DescentType::PriorUpdate,
            &mut self.default_visit_budget(),
            // likelihood update
            None,
            None,
            // prior update
            Some(symbol_seq.as_slice()),
            Some(pred_index),
            true,
            // expansion
            None,
        );
    }

    fn expand_trie_to_threshold(&mut self, threshold: f64) {
        let walker = self.root_walker();
        self.recalc_to_frontier_and_back(
            walker,
            DescentType::Expansion,
            &mut self.default_visit_budget(),
            // likelihood update
            None,
            None,
            // prior update
            None,
            None,
            false,
            // expansion
            Some(threshold),
        );
    }

    // EXPERIMENTAL (END)

    // ensure
    fn ensure_prediction(&mut self, walker: &mut Walker) -> PredictionIndex {
        let node_index = walker.node;
        let prediction_index = if let Some(index) = self.nodes[node_index].prediction {
            index
        } else {
            let final_token_length = self.nodes[node_index].final_token_length;
            let final_token_string =
                Walker::symbol_slice_to_string(&walker.a_symbol[..final_token_length as usize]);
            let final_token = if final_token_string.is_empty() {
                None
            } else {
                Some(final_token_string)
            };
            let order = PredictionOrder::ZeroOrder(final_token.clone());
            let index = if let Some(existing) = self.prediction_registry.index_for_order(&order) {
                existing
            } else {
                self.prediction_registry.alloc(Prediction::create_prediction(
                    order,
                    None,
                    None,
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

        let available_prediction_depth = usize::min(MAX_TOKEN_LENGTH, walker.depth as usize + 1);

        for symbol in Symbol::ALL {
            if symbol == Symbol::Start {
                continue;
            }
            let mut child = Node::fresh();

            if symbol == Symbol::Stop {
                child.final_token_length = 1;
                self.nodes.push(child);
                continue;
            }

            for i in 0..available_prediction_depth {
                let mut new_prefix = Walker::symbol_slice_to_string(&walker.a_symbol[..i]);
                new_prefix.push(symbol.to_byte() as char);
                let new_token = new_prefix.clone();
                let prediction_index =
                    walker.a_prediction_index[i].expect("prediction index must be valid");
                let prediction = self.prediction_registry.get(prediction_index).unwrap();
                let maybe_new_token_lexindex = self.tokenizer.lex_index(&new_token);
                let maybe_new_prefix_lexindex = self.tokenizer.prefix_lex_index(&new_prefix);
                // is ancestor i the token ancestor of our canonical tokenization if we are the end of a token?
                if let Some(new_token_lexindex) = maybe_new_token_lexindex {
                    let canonical_pair = prediction.canonical_followers[new_token_lexindex];
                    if canonical_pair {
                        child.final_token_length = (i + 1) as u8;
                        child.new_token_lexindex = new_token_lexindex;
                    }
                }
                // is it possible for ancestor i to be the closest token ancestor of the target string's canonical tokenization?
                if let Some(new_prefix_lexindex) = maybe_new_prefix_lexindex {
                    let truncation_possible =
                        prediction.canonical_follower_for_prefix[new_prefix_lexindex];
                    if truncation_possible {
                        child.new_prefix_lexindex[i] = new_prefix_lexindex;
                        child.truncation_possible[i] = true;
                        child.tl[i] = 0.0;
                    }
                }
            }
            self.nodes.push(child);
        }
    }

    // snapshot
    pub(crate) fn snapshot_trie(&mut self, threshold: f64) -> TrieSnapshot {
        self.expand_trie_to_threshold(threshold);
        self.to_snapshot(threshold)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::TOKENIZER_JSON_PATH;

    fn dump_trie_after_expand(threshold: f64) {
        let mut trie = Trie::from_tokenizer_json(TOKENIZER_JSON_PATH);
        trie.expand_trie_to_threshold(threshold);

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
        dump_trie_after_expand((1.0 / 30.0_f64).ln());
    }

    #[test]
    fn trie_expand_threshold_one_over_one_hundred() {
        dump_trie_after_expand((1.0 / 100.0_f64).ln());
    }

    #[test]
    fn trie_expand_threshold_one_over_two_hundred() {
        dump_trie_after_expand((1.0 / 200.0_f64).ln());
    }
}
