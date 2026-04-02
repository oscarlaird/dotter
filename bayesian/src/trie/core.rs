use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::Path;

use crate::bpe::{NUM_PREFIXES, NUM_TOKENS, TinyLlamaWordTokenizer, TOKENIZER_JSON_STR};
use crate::symbol::{Symbol, RADIX, RadixBitmap};
use crate::trie::MAX_TRUNCATION_POSSIBLE;
use crate::trie::rolling_hash as rh;

use super::{
    MAX_TOKEN_LENGTH, NodeIndex, Prediction, PredictionIndex, PredictionOrder, PredictionRegistry,
    SnapshotWalker, TrieSnapshot, TrieSnapshotNode, TRIE_EXPANSION_THRESHOLD, TRIE_MAX_VISITS,
    logaddexp,
};

fn dump_fmt_f64(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    let s = format!("{x:.6}");
    let t = s.trim_end_matches('0').trim_end_matches('.');
    if t.is_empty() || t == "-" {
        "0".into()
    } else {
        t.to_string()
    }
}

fn dump_fmt_lex(x: usize) -> String {
    if x == usize::MAX {
        "_".into()
    } else {
        x.to_string()
    }
}



#[derive(Clone, Debug)]
pub(crate) struct XNode {
    symbol: Symbol,
    // we don't store a node's hash, because there is no way to reach the node without knowing it
    // for all matrices, children are arrayed on the major axis
    c_truncation_possible: [u16; RADIX],
    c_final_token_length: [u8; RADIX],
    // c_final_token_lexindex: [u16; RADIX],
    // c_prefix_lexindex: [[u16; MAX_TRUNCATION_POSSIBLE]; RADIX],
    c_p: [f32; RADIX],
    c_p_old: [f32; RADIX],
    c_fp: [[f32; MAX_TRUNCATION_POSSIBLE]; RADIX],
    c_tp: [f32; RADIX],
    c_tp_time: [ContextWindowSize; RADIX],
    c_old_a_tp_time: [[ContextWindowSize; MAX_TRUNCATION_POSSIBLE]; RADIX],
    c_old_fp_time: [AncestorsBitmap; RADIX],
    c_l: [f32; RADIX],
    c_has_fullorder_pred: RadixBitmap,
    // #[cfg(feature = "tokentrie")]
    c_self_tl: [f32; RADIX],
    c_a_tl: [[f32; MAX_TRUNCATION_POSSIBLE]; RADIX],
    c_l_old: [f32; RADIX],
    c_ul: [f32; RADIX],
    c_cum_likelihood_frontier: RadixBitmap,
    c_z: [f32; RADIX],
    // ROOT
    is_root_clf: bool,
}

pub(crate) struct XPrediction {
}

type Hash = u64;


pub(crate) struct XBayes {
    nodes: rh::RHashMap<XNode>,
    full_predictions: rh::RHashMap<XPrediction>,
    zero_order_predictions: rh::RHashMap<XPrediction>,
    pending_likelihood: LUpdate,
    pending_prior: PUpdate,
    tokenizer: TinyLlamaWordTokenizer,

}

pub(crate) struct LUpdate {
    likelihoods: rh::RHashMap<f32>,
    cpc_form: bool, // complete prefix code form
}

pub(crate) struct PUpdate {
    new_predictions: rh::RHashSet,
}

type ContextWindowSize = u8;
type AncestorsBitmap = u16;

#[derive(Clone, Debug)]
struct XWalker {
    // policy: don't make consecutive zero order predictions; and don't expand if we are compelled to do so
    // its a bad policy because we must go through weird finishers with weird ancestors
    // still the time on a tp can meaningfully be the number of full order ancestors i.e. the context window or better; so u8s work fine for the time
    hash: Hash,
    depth: u16,
    a_symbol: [Symbol; MAX_TOKEN_LENGTH],
    a_tp: [f32; MAX_TOKEN_LENGTH],
    a_tp_time: [ContextWindowSize; MAX_TOKEN_LENGTH],
    a_pred_time: AncestorsBitmap, // last bit is most recent
}

impl LUpdate {
    fn new() -> Self {
        Self {
            likelihoods: rh::RHashMap::new(),
            cpc_form: false,
        }
    }

    fn to_cpc_form(&mut self) {
        // Complete prefix code form ensures that no node's prefix appears as another node: every sequence represented is maximal/non-overlapping.
        // Example for alphabet {a, b, c}:
        // Starting tree: 
        //   root(4.0)
        //     |- a(3.0)
        //         |- aa(2.0)
        //     |- b(1.0)
        // Transformation yields set:
        //   { aa(2.0), ab(3.0), ac(3.0), b(1.0), c(4.0) }
        //   (each string is now maximal/non-prefix of any other in set)
        if self.cpc_form { return; }
        struct Entry {
            hash: Hash,
            likelihood: f32,
        }
        let mut new_entries: Vec<Entry> = Vec::new();
        let mut remove_entries: Vec<Hash> = Vec::new();
        for (hash, &likelihood) in self.likelihoods.iter() {
            let mut has_any_children = false;
            let mut non_preexisting_child_hashes: Vec<Hash> = Vec::new();
            for symbol in Symbol::ALL {
                let child_hash = rh::append_right(*hash, symbol.to_byte());
                if self.likelihoods.contains_key(&child_hash) {
                    has_any_children = true;
                } else {
                    non_preexisting_child_hashes.push(child_hash);
                }
            }
            if !has_any_children {
                continue;
            }
            new_entries.extend(non_preexisting_child_hashes.iter()
                .map(|child_hash| Entry { hash: *child_hash, likelihood: likelihood })
            );
            remove_entries.push(*hash);
        }
        for hash in remove_entries {
            self.likelihoods.remove(&hash);
        }
        self.likelihoods.extend(new_entries.into_iter().map(|entry| (entry.hash, entry.likelihood)));

    }

    fn merge_many(l_tries: &[&Self]) -> Self {
        struct Frame {
            hash: Hash,
            likelihood: f32,
            hit_count: u32,
        }
        // assume that all likelihood updates are already in complete prefix code form i.e. no node is the prefix of another
        for &l_trie in l_tries {
            assert!(l_trie.cpc_form, "All input tries must be in CPC form before merging");
        }
        let mut result = Self::new();
        let mut walkers = vec![Frame { hash: 0, likelihood: 0.0, hit_count: 0 }];
        let mut iters = 0;
        while let Some(Frame { hash, likelihood, hit_count }) = walkers.pop() {
            iters += 1;
            assert!(iters < 100_000, "merge likelihood update trie: too many iterations");
            //
            let (hit_delta, likelihood_delta) = l_tries.iter()
                .filter_map(|l_trie| l_trie.likelihoods.get(&hash))
                .fold((0, 0.0f32), |(hits, sum), &v| (hits + 1, sum + v));
            let new_hits = hit_count + hit_delta;
            let new_likelihood = likelihood + likelihood_delta;
            if new_hits == (l_tries.len() as u32) {
                result.likelihoods.insert(hash, new_likelihood);
                continue;
            }
            //
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start { continue; }
                let child_hash = rh::append_right(hash, symbol.to_byte());
                walkers.push(Frame { hash: child_hash, likelihood: new_likelihood, hit_count: new_hits })
            }
        }
        result.cpc_form = true;
        result

    }

    fn merge(&self, other: &Self) -> Self {
        Self::merge_many(&[self, other])
    }
}

impl PUpdate {
    fn new() -> Self {
        Self {
            new_predictions: HashSet::with_hasher(BuildHasherDefault::<IdentityHasher>::default()),
        }
    }
}

const ROOT_HASH: Hash = {
    rh::append_right(0, Symbol::Start.to_byte())
};

impl XBayes {
    fn new() -> Self {
        let nodes = rh::RHashMap::new();
        let full_predictions = rh::RHashMap::new();
        let zero_order_predictions = rh::RHashMap::new();
        let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json_str(TOKENIZER_JSON_STR);
        Self {
            nodes,
            full_predictions,
            zero_order_predictions,
            pending_likelihood: LUpdate::new(),
            pending_prior: PUpdate::new(),
            tokenizer: tokenizer,
        }
    }

    fn root_walker(&self) -> XWalker {
        let root_pred_exists = self.full_predictions.contains_key(&ROOT_HASH);
        XWalker {
            hash: ROOT_HASH,
            depth: 0,
            a_symbol: [Symbol::Start; MAX_TOKEN_LENGTH], // invalid
            a_tp: [0.0; MAX_TOKEN_LENGTH],
            a_tp_time: [0; MAX_TOKEN_LENGTH],
            a_pred_time: if root_pred_exists { 1u16 } else { 0u16 },
        }
    }

    fn descend(&self, walker: &XWalker, symbol: Symbol) -> XWalker {
        let mut walker = walker.clone();
        let node = self.nodes.get(&walker.hash).unwrap();
        let slot = symbol.to_slot() as usize;
        walker.hash = rh::append_right(walker.hash, symbol.to_byte());
        // roll the walker
        for i in (1..MAX_TOKEN_LENGTH).rev() {
            walker.a_symbol[i] = walker.a_symbol[i - 1];
            walker.a_tp[i] = walker.a_tp[i - 1];
            walker.a_tp_time[i] = walker.a_tp_time[i - 1];
        }
        walker.a_pred_time = walker.a_pred_time << 1;
        // set 0
        walker.a_symbol[0] = symbol;
        walker.a_tp[0] = node.c_tp[slot];
        walker.a_tp_time[0] = node.c_tp_time[slot];
        let child_has_pred = (node.c_has_fullorder_pred & (1 << slot)) != 0;
        if child_has_pred { walker.a_pred_time |= 1; }
        //
        walker.depth += 1;
        walker
    }

    fn set_tl_array(&self, node_hash: Hash, tl_array: &mut [f32], ul: f32) {
        struct Frame {
            hash: Hash,
            ul: f32, // ul at or above this node
        }
        let mut frames = vec![Frame {hash: node_hash, ul}];
        if !self.nodes.contains_key(&node_hash) {
            tl_array.fill(ul);
            return;
        }
        let branch_clf = {
            let is_clf = false;
            let node = self.nodes.get(&node_hash).unwrap();
            if node_hash == ROOT_HASH {
                node.is_root_clf
            } else {
                let symbol = node.symbol;
                let parent_hash = rh::pop_right(node_hash, symbol.to_byte());
                let parent_node = self.nodes.get(&parent_hash).unwrap();
                parent_node.c_cum_likelihood_frontier & (1 << symbol.to_slot()) != 0
            }
        };
        let mut iters = 0;
        while let Some(Frame { hash, ul }) = frames.pop() {
            if iters < 1000 { iters += 1; } else { panic!("set_tl_array: too many iterations"); }
            let node = self.nodes.get(&hash).unwrap();
            // handle our children
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start { continue; }
                let child_slot = symbol.to_slot() as usize;
                let child_hash = rh::append_right(hash, symbol.to_byte());
                let child_is_clf = (node.c_cum_likelihood_frontier & (1 << child_slot)) != 0;
                if child_is_clf {
                    assert!(self.tokenizer.proper_prefix_hashset.contains(&child_hash), "set_tl_array: child hash is not a proper prefix");
                    let range = self.tokenizer.token_lex_range_for_prefix_hash(&child_hash);
                    let subslice = &mut tl_array[range.0..range.1];
                    subslice.fill(ul + node.c_l[child_slot]);
                    continue;
                }
                if self.tokenizer.token_hashset.contains(&child_hash) {
                    let lexindex = self.tokenizer.lex_index_for_token_hash(&child_hash);
                    tl_array[lexindex] = ul + node.c_self_tl[child_slot];
                }
                if !self.tokenizer.proper_prefix_hashset.contains(&child_hash) {
                    continue;
                }
                assert!(self.nodes.contains_key(&child_hash), "set_tl_array: child hash is not in nodes");
                frames.push(Frame { hash: child_hash, ul: ul + node.c_ul[child_slot]});
            }
        }
    }
}

impl Trie {
    // recalc (expansion and updates)
    // Each of the three kinds corresponds to a public API
    // - apply likelihood updates
    // - apply prior updates
    // - expand the trie to a threshold
    // EXPERIMENTAL (START)
    fn recalc_to_frontier_and_back(
        &mut self,
        mut walker: Walker,
        mode: RecalcMode<'_>,
        remaining_visit_budget: &mut i32,
    ) -> f64 {
        let node_index = walker.node;
        // 1. apply update to trie
        match &mode {
            RecalcMode::LikelihoodUpdate {
                snapshot_walker, ..
            } => {
                if !snapshot_walker.has_children_in_snapshot {
                    let new_likelihood = snapshot_walker.likelihood;
                    let node = &mut self.nodes[node_index];
                    node.ul += new_likelihood;
                    node.l += new_likelihood;
                    #[cfg(feature = "tokentrie")]
                    for i in 0..MAX_TOKEN_LENGTH {
                        node.tl[i] += new_likelihood;
                    }
                    node.nl = self.n;
                } else {
                    let node = &mut self.nodes[node_index];
                    node.cum_likelihood_frontier = false;
                }
            }
            RecalcMode::PriorUpdate {
                new_prediction_symbol_seq,
                new_prediction_index,
                comparable_to_pred_node,
            } => {
                let is_pred_node = *comparable_to_pred_node
                    && new_prediction_symbol_seq.len() == walker.depth as usize;
                if is_pred_node {
                    let node = &mut self.nodes[node_index];
                    node.prediction = Some(*new_prediction_index);
                    node.prediction_last_change = self.m;
                }
            }
            RecalcMode::Expansion => {}
        };
        // 2. recalc prior
        // tp
        let idx = self.nodes[node_index].final_token_length as usize;
        assert!(
            idx < MAX_TOKEN_LENGTH,
            "walker slot OOB: node_index={node_index} final_token_length={} idx={idx} MAX_TOKEN_LENGTH={MAX_TOKEN_LENGTH}",
            self.nodes[node_index].final_token_length,
        );
        let tp_last_change = self.nodes[node_index].tp_last_change;
        let new_tp_last_change =
            if self.nodes[node_index].is_root {
                0
            } else {
                walker.a_tp_last_change[idx].max(walker.a_prediction_last_change[idx])
            };
        if new_tp_last_change > tp_last_change {
            let a_pred_index = walker.a_prediction_index[idx].unwrap();
            let a_pred = self.prediction_registry.get(a_pred_index).unwrap();
            let node_symbol = self.nodes[node_index].symbol;
            let new_token_lexindex = self.nodes[node_index].new_token_lexindex;
            let final_token_prob = if node_symbol != Symbol::Stop {
                assert!(
                    new_token_lexindex < a_pred.follower_probs.len(),
                    "follower_probs OOB: node_index={node_index} symbol={node_symbol:?} new_token_lexindex={new_token_lexindex} follower_probs_len={}",
                    a_pred.follower_probs.len(),
                );
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
        let p_last_change = self.nodes[node_index].p_last_change;
        let mut new_p_last_change = p_last_change;
        for i in 0..MAX_TOKEN_LENGTH {
            if truncation_possible[i] {
                new_p_last_change = new_p_last_change.max(walker.a_tp_last_change[i]);
                new_p_last_change = new_p_last_change.max(walker.a_prediction_last_change[i]);
            }
        }
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
        let stop = match &mode {
            RecalcMode::LikelihoodUpdate {
                snapshot_walker, ..
            } => !snapshot_walker.has_children_in_snapshot,
            RecalcMode::PriorUpdate {
                new_prediction_symbol_seq,
                comparable_to_pred_node,
                ..
            } => {
                let cum_l_frontier = node_cum_likelihood_frontier;
                let is_pred_node_prefix = *comparable_to_pred_node
                    && walker.depth <= new_prediction_symbol_seq.len() as u32;
                let is_space = node_symbol == Symbol::Space;
                !is_pred_node_prefix
                    && (
                        !*comparable_to_pred_node ||  // no change case
                    cum_l_frontier ||  // constant likelihood case
                    is_space
                        // no-reweighting case
                    )
            }
            RecalcMode::Expansion => node_z < TRIE_EXPANSION_THRESHOLD,
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
                #[cfg(feature = "tokentrie")]
                for i in 0..MAX_TOKEN_LENGTH {
                    child_node.tl[i] += node_ul;
                }
                child_node.nl = self.n;
            }
            self.nodes[node_index].ul = 0.0;
        }
        // 7. recurse
        if !stop {
            let is_update = mode.is_update();
            #[cfg(feature = "tokentrie")]
            let is_likelihood_update = matches!(mode, RecalcMode::LikelihoodUpdate { .. });
            let mut children_sum = -f64::INFINITY;
            #[cfg(feature = "tokentrie")]
            let mut mtcdl = [f64::NEG_INFINITY; MAX_TOKEN_LENGTH];
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start {
                    continue;
                }
                let child_walker = self.descend(&walker, symbol);
                let child_mode = mode.child(&walker, symbol);
                let child_z = self.recalc_to_frontier_and_back(
                    child_walker,
                    child_mode,
                    remaining_visit_budget,
                );
                // 8. upprop z
                if is_update {
                    children_sum = logaddexp(children_sum, child_z);
                }
                // 9. upprop mtcdl
                #[cfg(feature = "tokentrie")]
                if is_likelihood_update {
                    let child_mtcdl = self.nodes[child_walker.node].tl;
                    let child_final_token_length = self.nodes[child_walker.node].final_token_length;
                    for i in 0..(MAX_TOKEN_LENGTH-1) {
                        mtcdl[i] = mtcdl[i].max(child_mtcdl[i+1]);
                    }
                    let i = child_final_token_length as usize - 1;
                    mtcdl[i] = mtcdl[i].max(child_mtcdl[0]);
                }
            }
            if is_update {
                self.nodes[node_index].z = children_sum;
            }
            #[cfg(feature = "tokentrie")]
            if is_likelihood_update {
                self.nodes[node_index].tl = mtcdl;
                self.nodes[node_index].ntl = self.n;
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
            RecalcMode::LikelihoodUpdate {
                snapshot,
                snapshot_walker,
            },
            &mut Self::visit_budget(),
        );
    }

    pub(crate) fn apply_prior_update(&mut self, node_string: String, prediction: Prediction) {
        let root_walker = self.root_walker();
        let symbol_seq = Symbol::string_to_vec(&node_string);
        let pred_index = self.prediction_registry.alloc(prediction);
        self.recalc_to_frontier_and_back(
            root_walker,
            RecalcMode::PriorUpdate {
                new_prediction_symbol_seq: symbol_seq.as_slice(),
                new_prediction_index: pred_index,
                comparable_to_pred_node: true,
            },
            &mut Self::visit_budget(),
        );
    }

    pub(crate) fn expand_trie(&mut self) {
        let walker = self.root_walker();
        self.recalc_to_frontier_and_back(
            walker,
            RecalcMode::Expansion,
            &mut Self::visit_budget(),
        );
    }

    /// Current trie as a snapshot (no expansion pass). Only nodes with `z > TRIE_EXPANSION_THRESHOLD` appear as children.
    pub(crate) fn snapshot_at_current(&self) -> TrieSnapshot {
        self.to_snapshot()
    }

    /// Every [`Node`] field, every index, plus a compact listing of [`PredictionRegistry`] entries (not full follower vectors).
    pub(crate) fn full_dump_format(&self) -> String {
        self.full_dump_format_with_symbol_filter(|_| true)
    }

    /// Like [`full_dump_format`], but only emits per-node blocks when `include_symbol(node.symbol)` is true.
    pub(crate) fn full_dump_format_with_symbol_filter(
        &self,
        include_symbol: impl Fn(Symbol) -> bool,
    ) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "=== Trie full dump ===");
        let _ = writeln!(
            out,
            "summary: nodes.len()={} root_index={} n={} m={}",
            self.nodes.len(),
            self.root,
            self.n,
            self.m
        );

        for (idx, node) in self.nodes.iter().enumerate() {
            if !include_symbol(node.symbol) {
                continue;
            }
            let _ = writeln!(out);
            let _ = writeln!(out, "--- nodes[{idx}] ---");
            let _ = writeln!(out, "  symbol: {:?}", node.symbol);
            let _ = writeln!(out, "  is_root: {}", node.is_root);
            let _ = writeln!(
                out,
                "  children_start_index: {:?}",
                node.children_start_index
            );
            let _ = writeln!(
                out,
                "  truncation_possible: {:?}",
                node.truncation_possible
            );
            let _ = writeln!(out, "  final_token_length: {}", node.final_token_length);
            let pred_line = match node.prediction {
                None => "None".to_string(),
                Some(pi) => match self.prediction_registry.get(pi) {
                    None => format!("Some({pi}) (missing in registry)"),
                    Some(p) => format!("Some({pi}) order={:?}", p.order),
                },
            };
            let _ = writeln!(out, "  prediction: {pred_line}");
            let _ = writeln!(
                out,
                "  prediction_last_change: {}",
                node.prediction_last_change
            );
            let _ = writeln!(
                out,
                "  new_token_lexindex: {}",
                dump_fmt_lex(node.new_token_lexindex)
            );
            let np = node
                .new_prefix_lexindex
                .iter()
                .map(|x| dump_fmt_lex(*x))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "  new_prefix_lexindex: [{np}]");
            let _ = writeln!(out, "  p: {}", dump_fmt_f64(node.p));
            let _ = writeln!(out, "  p_last_change: {}", node.p_last_change);
            let _ = writeln!(out, "  p_old: {}", dump_fmt_f64(node.p_old));
            let _ = writeln!(out, "  mp: {}", node.mp);
            let fp_s = node
                .fp
                .iter()
                .map(|x| dump_fmt_f64(*x))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "  fp: [{fp_s}]");
            let _ = writeln!(out, "  tp: {}", dump_fmt_f64(node.tp));
            let _ = writeln!(out, "  tp_last_change: {}", node.tp_last_change);
            let _ = writeln!(out, "  mtp: {}", node.mtp);
            let _ = writeln!(out, "  l: {}", dump_fmt_f64(node.l));
            let _ = writeln!(out, "  l_old: {}", dump_fmt_f64(node.l_old));
            let _ = writeln!(out, "  nl: {}", node.nl);
            let _ = writeln!(out, "  ul: {}", dump_fmt_f64(node.ul));
            #[cfg(feature = "tokentrie")]
            {
                let tl_s = node
                    .tl
                    .iter()
                    .map(|x| dump_fmt_f64(*x))
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(out, "  tl: [{tl_s}]");
                let _ = writeln!(out, "  ntl: {}", node.ntl);
            }
            let _ = writeln!(
                out,
                "  cum_likelihood_frontier: {}",
                node.cum_likelihood_frontier
            );
            let _ = writeln!(out, "  z: {}", dump_fmt_f64(node.z));
            let _ = writeln!(out, "  nz: {}", node.nz);
            let _ = writeln!(out, "  mz: {}", node.mz);
        }

        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "=== PredictionRegistry ({} predictions) ===",
            self.prediction_registry.len()
        );
        for i in 0..self.prediction_registry.len() {
            let Some(p) = self.prediction_registry.get(i) else {
                continue;
            };
            let _ = writeln!(
                out,
                "[{i}] order={:?}  stop_prob={}  follower_probs.len={}  canonical_followers.len={}  prefix_prob_agg.len={}  children.len={}",
                p.order,
                dump_fmt_f64(p.stop_prob),
                p.follower_probs.len(),
                p.canonical_followers.len(),
                p.follower_prob_for_prefix.len(),
                p.children.len(),
            );
        }

        out
    }

    // EXPERIMENTAL (END)

    // ensure
    fn ensure_prediction(&mut self, walker: &mut Walker) -> PredictionIndex {
        let node_index = walker.node;
        assert_ne!(
            self.nodes[node_index].symbol,
            Symbol::Stop,
            "ensure_prediction must never be called on a Stop node"
        );
        let prediction_index = if let Some(index) = self.nodes[node_index].prediction {
            index
        } else {
            let final_token_length = self.nodes[node_index].final_token_length;
            let final_token_string =
                Walker::symbol_slice_to_string(&walker.a_symbol[..final_token_length as usize]);
            let final_token =
                if final_token_string.is_empty()
                    || final_token_string == "^"
                {
                None
            } else {
                Some(final_token_string)
            };
            let order = PredictionOrder::ZeroOrder(final_token.clone());
            let index = if let Some(existing) = self.prediction_registry.index_for_order(&order) {
                existing
            } else {
                self.prediction_registry
                    .alloc(Prediction::create_prediction(
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
        assert_ne!(
            self.nodes[parent_index].symbol,
            Symbol::Stop,
            "ensure_children must never be called on a Stop node"
        );
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
            child.symbol = symbol;

            if symbol == Symbol::Stop {
                child.final_token_length = 1;
                self.nodes.push(child);
                continue;
            }

            // roll the indices of the walker, but don't set0 yet
            let mut child_walker = self.roll_walker(walker);
            child_walker.a_symbol[0] = Some(symbol);
            for i in 1..=available_prediction_depth {
                let new_prefix = Walker::symbol_slice_to_string(&child_walker.a_symbol[..i]);
                let new_token = new_prefix.clone();
                let prediction_index =
                    child_walker.a_prediction_index[i].expect("prediction index must be valid");
                let prediction = self.prediction_registry.get(prediction_index).unwrap();
                let maybe_new_token_lexindex = self.tokenizer.lex_index(&new_token);
                let maybe_new_prefix_lexindex = self.tokenizer.prefix_lex_index(&new_prefix);
                // is ancestor i the token ancestor of our canonical tokenization if we are the end of a token?
                if let Some(new_token_lexindex) = maybe_new_token_lexindex {
                    let canonical_pair = prediction.canonical_followers[new_token_lexindex];
                    if canonical_pair {
                        child.final_token_length = i as u8;
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
                        #[cfg(feature = "tokentrie")]
                        {
                            child.tl[i] = 0.0;
                        }
                    }
                }
            }
            self.nodes.push(child);
        }
    }

    // snapshot
    pub(crate) fn snapshot_trie(&mut self) -> TrieSnapshot {
        self.expand_trie();
        self.to_snapshot()
    }

    fn to_snapshot(&self) -> TrieSnapshot {
        let mut index_map = vec![None; self.nodes.len()];
        let mut snapshot_nodes = Vec::new();
        let root =
            self.snapshot_subtree(self.root, &mut index_map, &mut snapshot_nodes);
        TrieSnapshot {
            nodes: snapshot_nodes,
            root,
        }
    }

    fn snapshot_subtree(
        &self,
        node_index: NodeIndex,
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
                if child.z > TRIE_EXPANSION_THRESHOLD {
                    let child_snapshot_index =
                        self.snapshot_subtree(child_index, index_map, snapshot_nodes);
                    children.push((symbol, child_snapshot_index));
                }
            }
        }
        snapshot_nodes[snapshot_index].children = children;
        snapshot_index
    }
}

impl Walker {
    fn new() -> Self {
        Self {
            node: usize::MAX,
            depth: 0,
            a_symbol: [None; MAX_TOKEN_LENGTH],
            a_tp: [0.0; MAX_TOKEN_LENGTH],
            a_prediction_index: [None; MAX_TOKEN_LENGTH],
            a_p_last_change: [-1; MAX_TOKEN_LENGTH],
            a_tp_last_change: [-1; MAX_TOKEN_LENGTH],
            a_prediction_last_change: [-1; MAX_TOKEN_LENGTH],
        }
    }
    fn symbol_slice_to_string(symbols: &[Option<Symbol>]) -> String {
        let ordered_symbols: Vec<Symbol> = symbols.iter().flatten().rev().copied().collect();
        Symbol::slice_to_string(&ordered_symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::TOKENIZER_JSON_PATH;

    fn dump_trie_after_expand() {
        let mut trie = Trie::from_tokenizer_json(TOKENIZER_JSON_PATH);
        trie.expand_trie();

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
                    "root child {:?}: idx={} pred={:?} child_base={:?} final_token_length={}  p={} tp={} z={}",
                    symbol,
                    child_index,
                    child.prediction,
                    child.children_start_index,
                    child.final_token_length,
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
                "stop node idx={} pred={:?} child_base={:?} final_token_length={}  p={} tp={} z={}",
                index,
                node.prediction,
                node.children_start_index,
                node.final_token_length,
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
                "leaf node idx={} symbol={:?} pred={:?} final_token_length={}  p={} tp={} z={}",
                index,
                node.symbol,
                node.prediction,
                node.final_token_length,
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
    fn trie_expand_smoke_test() {
        dump_trie_after_expand();
    }

    /// Mirrors V3 + `backend/new_lm.py`: expand, JSON round-trip snapshot, dummy letter
    /// likelihoods, then a full-order prior at the reset prompt.
    #[test]
    fn v3_likelihood_then_prior_like_browser() {
        use crate::bpe::NUM_TOKENS;

        let mut trie = Trie::from_tokenizer_json(TOKENIZER_JSON_PATH);
        let snapshot = trie.snapshot_trie();
        let json = serde_json::to_string(&snapshot).expect("serialize snapshot");
        let mut snap: TrieSnapshot = serde_json::from_str(&json).expect("round-trip snapshot");

        for n in &mut snap.nodes {
            if n.symbol == Symbol::A {
                n.likelihood = 0.0;
            } else {
                n.likelihood = -2.0;
            }
        }

        trie.apply_likelihood_update(&snap);

        let prompt = concat!(
            "my watch fell in the water\n",
            "prevailing wind from the east\n",
            "never too rich and never too thin\n",
            "breathing is difficult\n",
            "i can see the rings on saturn\n",
        );

        let logits = vec![0.0_f64; NUM_TOKENS].into_boxed_slice();
        let prediction = Prediction::create_prediction(
            PredictionOrder::FullOrder(None, prompt.to_string()),
            Some(logits),
            Some(0.0_f64),
            &trie.tokenizer,
        );
        trie.apply_prior_update(prompt.to_string(), prediction);
    }
}
