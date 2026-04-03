use crate::bpe::{TinyLlamaWordTokenizer, TOKENIZER_JSON_STR};
use crate::symbol::{Symbol, RADIX, RadixBitmap};
use crate::trie::MAX_TRUNCATION_POSSIBLE;
use crate::trie::rolling_hash as rh;
use crate::trie::prediction::XPrediction;

use super::{
    MAX_TOKEN_LENGTH, logaddexp,
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
    c_can_trunc: [u16; RADIX],
    c_final_token_length: [u8; RADIX],
    c_final_token_hash: [u64; RADIX],  // TODO: would it be more efficient to store the lexindex instead?
    // c_final_token_lexindex: [u16; RADIX],
    // c_prefix_lexindex: [[u16; MAX_TRUNCATION_POSSIBLE]; RADIX],
    c_p: [f32; RADIX],
    c_p_old: [f32; RADIX],
    c_fp: [[f32; MAX_TRUNCATION_POSSIBLE]; RADIX],
    c_tp: [f32; RADIX],
    c_tp0: [f32; RADIX],
    c_final_token_prob: [f32; RADIX],
    c_l: [f32; RADIX],
    // TODO: #[cfg(feature = "tokentrie")]
    c_a_tl: [[f32; MAX_TRUNCATION_POSSIBLE+1]; RADIX],
    c_accumed_l_old: [f32; RADIX],
    c_z: [f32; RADIX],
    //
    c_a_pred_changed: [AncestorsBitmap; RADIX], // ancestor predictions which have changed since we visited this child
    c_a_tp_changed: [AncestorsBitmap; RADIX], // ancestor tps which have changed since we visited this child
    // ROOT
    if_root_then_z: f32,
    if_root_then_root_pred_changed: bool,
}


type Hash = u64;


pub(crate) struct XBayes {
    nodes: rh::RHashMap<XNode>,
    full_predictions: rh::RHashMap<XPrediction>,
    zero_order_predictions: rh::RHashMap<XPrediction>,
    pending_likelihood: LUpdate,
    cum_likelihood: LUpdate,
    pending_prior: PUpdate,
    unread_predictions: PUpdate,
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
    hash: Hash,
    depth: u16,
    a_symbol: [Symbol; MAX_TOKEN_LENGTH],
    a_tp: [f32; MAX_TOKEN_LENGTH],
    a_tp0: [f32; MAX_TOKEN_LENGTH],
    a_final_token_hash: [u64; MAX_TOKEN_LENGTH], // TODO: would it be more efficient to store the lexindex instead?
    a_pred_changed: AncestorsBitmap, // ancestors which have changed since we visited this node
    a_tp_changed: AncestorsBitmap, // ancestors which have changed since we visited this node
}

impl LUpdate {
    // type Target = rh::RHashMap<f32>;

    fn new() -> Self {
        Self {
            likelihoods: rh::RHashMap::default(),
            cpc_form: false,
        }
    }

    fn deref(&self) -> &rh::RHashMap<f32> {
        &self.likelihoods
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
        // TODO: it is possible to make this faster with an in-place merge
        // where we stop descent if we haven't hit ourself and we have hit_count=len-1
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
            new_predictions: rh::RHashSet::default(),
        }
    }

    fn deref(&self) -> &rh::RHashSet {
        &self.new_predictions
    }
}

const ROOT_HASH: u64 = {
    rh::append_right(0, Symbol::Start.to_byte())
};

impl XBayes {
    fn new() -> Self {
        let nodes = rh::RHashMap::default();
        let full_predictions = rh::RHashMap::default();
        let zero_order_predictions = rh::RHashMap::default();
        let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json_str(TOKENIZER_JSON_STR);
        Self {
            nodes,
            full_predictions,
            zero_order_predictions,
            pending_likelihood: LUpdate::new(),
            cum_likelihood: LUpdate::new(),
            pending_prior: PUpdate::new(),
            unread_predictions: PUpdate::new(),
            tokenizer: tokenizer,
        }
    }

    fn root_walker(&self) -> XWalker {
        let root_node = self.nodes.get(&ROOT_HASH).unwrap();
        XWalker {
            hash: ROOT_HASH,
            depth: 0,
            a_symbol: [Symbol::Start; MAX_TOKEN_LENGTH], // invalid
            a_tp: [0.0; MAX_TOKEN_LENGTH],
            a_tp0: [0.0; MAX_TOKEN_LENGTH],
            a_final_token_hash: [u64::MAX; MAX_TOKEN_LENGTH],
            a_tp_changed: 0,
            a_pred_changed: if root_node.if_root_then_root_pred_changed { 1u16 } else { 0u16 },
        }
    }

    fn descend(&self, walker: &XWalker, symbol: Symbol) -> XWalker {
        // requires looking up the node of the input walker,
        // but does not need to look up the child node
        let mut walker = walker.clone();
        let node = self.nodes.get(&walker.hash).unwrap();
        let slot = symbol.to_slot() as usize;
        walker.hash = rh::append_right(walker.hash, symbol.to_byte());
        // roll the walker
        for i in (1..MAX_TOKEN_LENGTH).rev() {
            walker.a_symbol[i] = walker.a_symbol[i - 1];
            walker.a_tp[i] = walker.a_tp[i - 1];
            walker.a_tp0[i] = walker.a_tp0[i - 1];
            walker.a_final_token_hash[i] = walker.a_final_token_hash[i - 1];
        }
        walker.a_tp_changed = walker.a_tp_changed << 1;
        walker.a_pred_changed = walker.a_pred_changed << 1;
        // set 0
        walker.a_symbol[0] = symbol;
        walker.a_tp[0] = node.c_tp[slot];
        walker.a_tp0[0] = node.c_tp0[slot];
        walker.a_final_token_hash[0] = node.c_final_token_hash[slot];
        // if we are up to date, that does not imply that our node.a_tp_changed==0,
        // but merely that for every strict ancestor, we have a.a_tp_changed==0
        //
        // the walker should show the sum over our ancestors and ourself,
        // since we will be updating our children, and we are included in our child's strict ancestors
        //
        walker.a_tp_changed |= node.c_a_tp_changed[slot];
        walker.a_pred_changed |= node.c_a_pred_changed[slot];
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
        let branch_clf = self.cum_likelihood.deref().contains_key(&node_hash);
        let mut iters = 0;
        while let Some(Frame { hash, ul }) = frames.pop() {
            if iters < 1000 { iters += 1; } else { panic!("set_tl_array: too many iterations"); }
            let node = self.nodes.get(&hash).unwrap();
            // handle our children
            for child_slot in 0..RADIX {
                let child_hash = rh::append_right(hash, Symbol::slot_to_byte(child_slot));
                let child_is_clf = self.cum_likelihood.deref().contains_key(&child_hash);
                if child_is_clf {
                    assert!(self.tokenizer.proper_prefix_hashset.contains(&child_hash), "set_tl_array: child hash is not a proper prefix");
                    let range = self.tokenizer.token_lex_range_for_prefix_hash(&child_hash);
                    let subslice = &mut tl_array[range.0..range.1];
                    subslice.fill(ul + node.c_l[child_slot]);
                    continue;
                }
                if self.tokenizer.token_hashset.contains(&child_hash) {
                    let lexindex = self.tokenizer.lex_index_for_token_hash(&child_hash);
                    tl_array[lexindex] = ul + node.c_a_tl[child_slot][0];
                }
                if !self.tokenizer.proper_prefix_hashset.contains(&child_hash) {
                    continue;
                }
                assert!(self.nodes.contains_key(&child_hash), "set_tl_array: child hash is not in nodes");
                frames.push(Frame { hash: child_hash, ul: ul + node.c_l[child_slot]});
            }
        }
    }
    fn apply_updates(&mut self) {
        let l_update = self.pending_likelihood.likelihoods;
        let mut p_update = self.pending_prior.new_predictions;
        self.cum_likelihood = self.cum_likelihood.merge(&self.pending_likelihood);
        // mark all the nodes, for which there is a prior update ahead of us, but behind the clf
        // this indicator set is valid for nodes inside the clf, 
        // but may contain meaningless entries beyond the clf
        let p_update_proper_ancestors = {
            // remove predictions beyond the trie
            p_update.retain(|hash| self.nodes.contains_key(hash));
            let mut res = rh::RHashSet::default();
            let mut frames: Vec<Hash> = p_update
                .iter().cloned().collect::<Vec<_>>();
            while let Some(hash) = frames.pop() {
                if res.contains(&hash) { continue; }
                if self.cum_likelihood.deref().contains_key(&hash) { continue; }
                res.insert(hash);
                if hash == ROOT_HASH { continue; }
                let symbol = self.nodes.get(&hash).unwrap().symbol;
                let parent_hash = rh::pop_right(hash, symbol.to_byte());
                frames.push(parent_hash);
            }
            res
        };
        self.unread_predictions.new_predictions.extend(p_update.drain());
        struct Frame {
            walker: XWalker,
            hit_l_update: bool,
            hit_p_update: bool,
        }
        let root_frame = Frame {
            walker: self.root_walker(),
            hit_l_update: false,
            hit_p_update: false,
        };
        let mut frames: Vec<Frame> = vec![root_frame];
        let mut upprop_frames: Vec<XWalker> = vec![];
        let mut iters = 0;
        // 1. descend to valid frontier
        while let Some(Frame { walker, hit_l_update, hit_p_update }) = frames.pop() {
            {
                self.ensure_zero_order_prediction(walker.a_final_token_hash[0]);
                self.ensure_node(&walker);
            }
            if iters < 1000 { iters += 1; } else { panic!("apply_updates: too many iterations"); }
            upprop_frames.push(walker.clone());
            let node = self.nodes.get_mut(&walker.hash).unwrap();
            for slot in 0..RADIX {
                let child_symbol = Symbol::from_slot(slot);
                let child_byte = Symbol::slot_to_byte(slot);
                let child_hash = rh::append_right(walker.hash, child_byte);
                // determine children's validity
                // - valid = const_likelihood || no_reweighting
                let const_likelihood = self.cum_likelihood.deref().contains_key(&child_hash);
                // TODO: writing this code made me realize we need to add a subtlety
                // to the .tex file
                // For no-reweighting, 
                // the question is where we stand in relation to prior updates
                // - if there is a prior update behind us, we need to reach a space (to ensure no prior reweighting)
                // - if there is a prior update ahead of us, but before the likelihood frontier, we are not valid!
                // - if there is a prior update ahead of us, and after the cum likelihood frontier, we don't care
                //   - (if there is a prior update beyond the trie, we treat this the same as beyond the clf)
                let no_reweighting =
                    hit_l_update  // this update hasn't rebalanced l beyond here
                    && !p_update_proper_ancestors.contains(&child_hash) // and no p_updates are ahead of us and behind the clf
                    && (
                        !hit_p_update // no p_updates behind
                        || child_symbol == Symbol::Space // p_updates behind are determined
                        // TODO: consider a better heuristic than space
                    );
                let valid = const_likelihood || no_reweighting;
                // recalculate likelihood
                let mut l = node.c_l[slot];
                l += l_update.get(&child_hash).unwrap_or(&0.0f32);
                // recalculate prior
                let mut p = node.c_p[slot];
                // check for an unread prediction
                if self.unread_predictions.deref().remove(&child_hash) {
                    node.c_a_pred_changed[slot] |= 1; // walker will see this when it leaves this node
                    // N.B. A prediction at a node, c, doesn't change c.p or c.tp, it only affects descendants
                }
                // propagate tp changed
                let ftl = node.c_final_token_length[slot] as usize;
                let token_a_tp_changed = (walker.a_tp_changed << 1) & (1 << ftl) != 0;
                let token_a_pred_changed = (walker.a_pred_changed << 1) & (1 << ftl) != 0;
                if token_a_pred_changed {
                    let token_a_tp = walker.a_tp[ftl - 1];
                    let mut token_a_hash = walker.hash;
                    let mut final_token_hash = child_byte as u64;
                    for i in 0..(ftl-1) {
                        let s = walker.a_symbol[i].to_byte();
                        token_a_hash = rh::pop_right(token_a_hash, s);
                        final_token_hash = rh::extend_right(s as u64, final_token_hash, i+1);
                    }
                    assert!(rh::extend_right(token_a_hash, final_token_hash, ftl) == child_hash, "hash mismatch");
                    let final_token_lexindex = self.tokenizer.lex_index_for_token_hash(&final_token_hash);
                    let final_token_prob = self.full_predictions
                        .get(&token_a_hash).unwrap()
                        .follower_prob_for_prefix[final_token_lexindex];
                    node.c_final_token_prob[slot] = final_token_prob;
                    node.c_tp[slot] = token_a_tp + final_token_prob;
                    node.c_a_tp_changed[slot] |= 1;
                } else if token_a_tp_changed {
                    let token_a_tp = walker.a_tp[ftl - 1];
                    node.c_tp[slot] = token_a_tp + node.c_final_token_prob[slot];
                    node.c_a_tp_changed[slot] |= 1;
                }
                // has the prediction changed at any of our possible truncations?
                // invariant: "the predictions which have changed since we last refreshed our fp array...
                // ... are exactly those indicated by the union of our strict ancestor's a_pred_changed arrays"
                let relevant_preds_changed = node.c_can_trunc[slot] & (walker.a_pred_changed << 1);
                let fp_changed = relevant_preds_changed != 0;
                let relevant_tps_changed = node.c_can_trunc[slot] & (walker.a_tp_changed << 1);
                let tp_changed = relevant_tps_changed != 0;
                if fp_changed {
                    let mut dense_idx = 0;
                    let mut token_a_hash = walker.hash;
                    let mut final_prefix_hash = child_byte as u64;
                    for i in 1..MAX_TOKEN_LENGTH {
                        if relevant_preds_changed & (1 << i) != 0 {
                            let new_fp = self.full_predictions
                                .get(&token_a_hash).unwrap()
                                .follower_prob_for_prefix[
                                    self.tokenizer.prefix_lex_index_for_prefix_hash(&final_prefix_hash)
                                ];
                            node.c_fp[slot][dense_idx] = new_fp;
                            dense_idx += 1;
                        }
                        token_a_hash = rh::pop_right(token_a_hash, walker.a_symbol[i-1].to_byte());
                        final_prefix_hash = rh::extend_right(walker.a_symbol[i-1].to_byte() as u64, final_prefix_hash, i);
                    }
                }
                let p_changed = fp_changed || tp_changed;
                if p_changed {
                    p = f32::NEG_INFINITY;
                    let mut dense_idx = 0;
                    for i in 1..MAX_TOKEN_LENGTH {
                        let can_trunc = (node.c_can_trunc[slot] & (1 << i)) != 0;
                        if can_trunc {
                            let dp = walker.a_tp[i-1] + node.c_fp[slot][dense_idx];
                            p = logaddexp(p, dp);
                            dense_idx += 1;
                        }
                    }
                }
                node.c_p[slot] = p;
                node.c_l[slot] = l;
                // update valid children
                if valid {
                    if const_likelihood {
                        // Z = p + l
                        node.c_z[slot] = p + l;
                    } else if no_reweighting {
                        let p_old = node.c_p_old[slot];
                        let l_old = node.c_accumed_l_old[slot];
                        // Z += (p - p_old) + (l - l_old)
                        let p_delta = p - p_old;
                        let l_delta = l - l_old;
                        let z_delta = p_delta + l_delta;
                        node.c_z[slot] += z_delta;
                    }
                    // since we won't be descending from this valid child, we should set the changed arrays
                    node.c_a_pred_changed[slot] |= walker.a_pred_changed << 1;
                    node.c_a_tp_changed[slot] |= walker.a_tp_changed << 1;
                }
                node.c_p_old[slot] = p;
                node.c_accumed_l_old[slot] = l;
                // add invalid children to the stack
                if !valid {
                    let child_walker = self.descend(&walker, child_symbol);
                    // since the invalid child will be visited and all its children set to the
                    // current time, its changed arrays can be set to 0
                    node.c_a_pred_changed[slot] = 0;
                    node.c_a_tp_changed[slot] = 0;
                    // TODO: keeping the final token hash would be convenient
                    frames.push(Frame {
                        walker: child_walker,
                        hit_l_update,
                        hit_p_update,
                    });
                }
            }
        }
        // up-prop z (and mtcdl)
        while let Some(walker) = upprop_frames.pop() {
            // we are proceeding in reverse-topological order
            // hence we are guaranteed that our invalid children have already been
            // up-propagated and therefore our c_z array is correct
            let node = self.nodes.get(&walker.hash).unwrap();
            let mut z = f32::NEG_INFINITY;
            for slot in 0..RADIX {
                z = logaddexp(z, node.c_z[slot]);
            }
            // since only child values are stored, 
            // we must store our own value on our parent
            if walker.hash == ROOT_HASH {
                node.if_root_then_z = z;
            } else {
                let parent_hash = rh::pop_right(walker.hash, node.symbol.to_byte());
                let parent_node = self.nodes.get_mut(&parent_hash).unwrap();
                parent_node.c_z[node.symbol.to_slot()] = z;
            }
            // upprop mtcdl
            // todo feature=tokentrie
            let mut mtcdl = [f32::NEG_INFINITY; MAX_TOKEN_LENGTH];
            for slot in 0..RADIX {
                let c_mtcdl = node.c_a_tl[slot];
                let c_can_trunc = node.c_can_trunc[slot];
                let nz = c_can_trunc | 1;
                let expanded_c_mtcdl = dense_to_sparse16( &c_mtcdl, nz, f32::NEG_INFINITY);
                for i in 0..(MAX_TOKEN_LENGTH-1) {
                    mtcdl[i] = mtcdl[i].max(expanded_c_mtcdl[i+1]);
                }
                let c_ftl = node.c_final_token_length[slot] as usize;
                mtcdl[c_ftl - 1] = mtcdl[c_ftl - 1].max(expanded_c_mtcdl[0]);
            }
            // since only child values are stored, we must store our own value on our parent
            if walker.hash == ROOT_HASH {
                // Don't do anything because root.mtcdl is unnecessary since the root is always queried first
            } else {
                let parent_hash = rh::pop_right(walker.hash, node.symbol.to_byte());
                let parent_node = self.nodes.get_mut(&parent_hash).unwrap();
                let mut mtcdl_dense = [f32::NAN; MAX_TRUNCATION_POSSIBLE+1];
                let nz = parent_node.c_can_trunc[node.symbol.to_slot()] | 1;
                for (i, v) in sparse16_to_dense(&mtcdl, nz).iter().enumerate() {
                    mtcdl_dense[i] = *v;
                }
                parent_node.c_a_tl[node.symbol.to_slot()] = mtcdl_dense;
            }

        }
    }
    // TODO:
    // *set cum_likelihood_frontier
    // *set mtcdl
    // *is_space
    // *node.a_tp_changed (pending updates)
    // *node.a_pred_changed (pending updates)
    // *walker.a_tp_changed
    // *walker.a_pred_changed
    // Ensure children / Ensure prediction
    // *Walker roll prefix hashes (no, too expensive 8x16=128 bytes per walker)
    // Expand trie
    // update l
    // hit l update / hit p update

    fn ensure_zero_order_prediction(&mut self, final_token_hash: u64) {
        if self.zero_order_predictions.contains_key(&final_token_hash) {
            return;
        }
        let prediction = XPrediction::create_prediction(
            true,
            final_token_hash,
            None,
            None,
            &self.tokenizer,
        );
        self.zero_order_predictions.insert(final_token_hash, prediction);
    }

    fn ensure_node(&mut self, walker: &XWalker) {
        if self.nodes.contains_key(&walker.hash) {
            return;
        }
        assert!(walker.a_symbol[0] != Symbol::Stop, "ensure_node must never be called on a Stop node");
        let mut node = XNode {
            symbol: walker.a_symbol[0],
            c_can_trunc: [0; RADIX],
            c_final_token_length: [0; RADIX],
            c_final_token_hash: [u64::MAX; RADIX],
            c_p: [f32::NEG_INFINITY; RADIX],
            c_p_old: [f32::NEG_INFINITY; RADIX],
            c_fp: [[f32::NEG_INFINITY; MAX_TRUNCATION_POSSIBLE]; RADIX],
            c_tp: [f32::NEG_INFINITY; RADIX],
            c_tp0: [f32::NEG_INFINITY; RADIX],
            c_final_token_prob: [f32::NEG_INFINITY; RADIX],
            c_l: [f32::NEG_INFINITY; RADIX],
            c_a_tl: [[f32::NEG_INFINITY; MAX_TRUNCATION_POSSIBLE+1]; RADIX],
            c_accumed_l_old: [f32::NEG_INFINITY; RADIX],
            c_z: [f32::NEG_INFINITY; RADIX],
            c_a_pred_changed: [0; RADIX],
            c_a_tp_changed: [0; RADIX],
            if_root_then_z: f32::NEG_INFINITY,
            if_root_then_root_pred_changed: false,
        };
        //
        let available_prediction_depth = usize::min(MAX_TOKEN_LENGTH, walker.depth as usize + 1);
        for slot in 0..RADIX {
            let child_symbol = Symbol::from_slot(slot);
            let child_byte = Symbol::slot_to_byte(slot);
            let final_chars_hash = child_byte as u64;
            for i in 1..available_prediction_depth {
                let a_pred = self.zero_order_predictions
                    .get(&walker.a_final_token_hash[i-1])
                    .unwrap();
                // let a_pred_prob = a_pred.follower_prob_for_prefix[
                //     self.tokenizer.prefix_lex_index_for_prefix_hash(&final_chars_hash)
                // ];
                // Determine Canonical Token Ancestor
                if self.tokenizer.token_hashset.contains(&final_chars_hash) {
                    let new_token_lexindex = self.tokenizer.lex_index_for_token_hash(&final_chars_hash);
                    let canonical_pair = a_pred.canonical_followers[new_token_lexindex];
                    if canonical_pair {
                        node.c_final_token_length[slot] = i as u8;
                        node.c_final_token_hash[slot] = final_chars_hash;
                    }
                }
                // Determine Possible Truncations
                if self.tokenizer.proper_prefix_hashset.contains(&final_chars_hash)
                    || self.tokenizer.token_hashset.contains(&final_chars_hash)
                {
                    let new_prefix_lexindex = self.tokenizer.prefix_lex_index_for_prefix_hash(&final_chars_hash);
                    let can_trunc = a_pred.canonical_follower_for_prefix[new_prefix_lexindex];
                    if can_trunc {
                        node.c_can_trunc[slot] |= 1 << i;
                        node.c_a_tl[slot][i] = 0.0;
                    }
                }
            }
        }
        // at no greater price than keeping tp0 on nodes we can initialize nodes to the beginning of time
        self.nodes.insert(walker.hash, node);
    }

}

fn dense_to_sparse16(a: &[f32], nonzeros: u16, default: f32) -> [f32; 16] {
    let mut res = [default; 16];
    let mut mask_after_bit = 0;
    for i in 0..16 {
        if (nonzeros & (1 << i)) != 0 {
            res[i] = a[(mask_after_bit & nonzeros).count_ones() as usize];
        }
        mask_after_bit <<= 1;
        mask_after_bit |= 1;
    }
    res
}

fn sparse16_to_dense(a: &[f32; 16], nonzeros: u16) -> Vec<f32> {
    let mut res = Vec::new();
    for i in 0..16 {
        if (nonzeros & (1 << i)) != 0 {
            res.push(a[i]);
        }
    }
    res
}
