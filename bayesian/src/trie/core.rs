use crate::bpe::{TinyLlamaWordTokenizer, TOKENIZER_JSON_STR};
use crate::rolling_hash::Hash;
use crate::symbol::{Symbol, RADIX};
use crate::trie::MAX_TRUNCATION_POSSIBLE;
use crate::rolling_hash as rh;
use crate::trie::prediction::XPrediction;

mod sparse16;
mod y_walker;

use super::{
    MAX_TOKEN_LENGTH, ROOT_HASH, logaddexp,
};
use super::l_update::LUpdate;
use super::p_update::PUpdate;
use sparse16::{dense_to_sparse16, sparse16_to_dense};
use y_walker::{FromEnd as _, YWalker, YWalkerRow};

#[derive(Clone, Debug)]
pub(crate) struct XNode {
    pub(crate) symbol: Symbol,
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
    // TODO: #[cfg(feature = "tokentrie")]
    c_a_tl: [[f32; MAX_TRUNCATION_POSSIBLE+1]; RADIX],
    c_cond_l: [f32; RADIX],
    c_accumed_l_old: [f32; RADIX],
    pub(crate) c_z: [f32; RADIX],
    //
    c_a_pred_changed: [AncestorsBitmap; RADIX], // ancestor predictions which have changed since we visited this child
    c_a_tp_changed: [AncestorsBitmap; RADIX], // ancestor tps which have changed since we visited this child
    // ROOT
    pub(crate) if_root_then_z: f32,
}

pub(crate) struct XBayes {
    pub(crate) nodes: rh::RHashMap<XNode>,
    pub(crate) full_predictions: rh::RHashMap<XPrediction>,
    zero_order_predictions: rh::RHashMap<XPrediction>,
    pub(crate) pending_likelihood: LUpdate,
    cum_likelihood: LUpdate,
    pub(crate) pending_prior: PUpdate,
    unread_predictions: PUpdate,
    pub(crate) tokenizer: TinyLlamaWordTokenizer,

}


type ContextWindowSize = u8;
type AncestorsBitmap = u16;

pub(crate) enum RecalcType {
    Update,
    Expand { threshold: f32 },
    // N.B. you can't do both at the same time since checking threshold requires knowing the root's z which is not known until after uppropping an update
}

pub(crate) enum RecalcResult {
    Updated,
    Expanded { nodes_over_threshold: Vec<Hash> }
}

impl XBayes {
    pub(crate) fn new() -> Self {
        let mut nodes = rh::RHashMap::default();
        let full_predictions = rh::RHashMap::default();
        let mut zero_order_predictions = rh::RHashMap::default();
        let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json_str(TOKENIZER_JSON_STR);
        //
        XBayes::ensure_zero_order_prediction(&mut zero_order_predictions, &tokenizer, ROOT_HASH);
        XBayes::ensure_node(&mut nodes, &zero_order_predictions, &tokenizer, &YWalker::root(ROOT_HASH));
        //
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

    fn root_walker() -> YWalker {
        YWalker::root(ROOT_HASH)
    }

    fn descend(nodes: &rh::RHashMap<XNode>, walker: &mut YWalker, symbol: Symbol) {
        // requires looking up the node of the input walker,
        // but does not need to look up the child node
        let p_hash = *walker.a_hash().from_end(0);
        let p_node = nodes.get(&p_hash).unwrap();
        let slot = symbol.to_slot() as usize;
        let n_hash = rh::append_right(p_hash, symbol.to_byte());
        walker.push(YWalkerRow::new(
            n_hash,
            p_node.c_final_token_hash[slot],
            symbol,
            p_node.c_tp[slot],
            p_node.c_tp0[slot],
        ));
    }

    fn set_tl_array(&self, node_hash: Hash, tl_array: &mut [f32], accumed_l: f32) {
        struct Frame {
            hash: Hash,
            accumed_l: f32, // accumed_l at or above this node
        }
        let mut frames = vec![Frame {hash: node_hash, accumed_l}];
        if !self.nodes.contains_key(&node_hash) {
            tl_array.fill(accumed_l);
            return;
        }
        // TODO: the caller should be responsible for knowing if it hit the clf (it can do this when it traverses A to B for accumed l)
        unimplemented!();
        let branch_clf = self.cum_likelihood.deref().contains_key(&node_hash);
        if branch_clf {
            tl_array.fill(accumed_l);
            return;
        }
        let mut iters = 0;
        while let Some(Frame { hash, accumed_l }) = frames.pop() {
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
                    subslice.fill(accumed_l + node.c_cond_l[child_slot]);
                    continue;
                }
                if self.tokenizer.token_hashset.contains(&child_hash) {
                    let lexindex = self.tokenizer.lex_index_for_token_hash(&child_hash);
                    tl_array[lexindex] = accumed_l + node.c_a_tl[child_slot][0];
                }
                if !self.tokenizer.proper_prefix_hashset.contains(&child_hash) {
                    continue;
                }
                assert!(self.nodes.contains_key(&child_hash), "set_tl_array: child hash is not in nodes");
                frames.push(Frame { hash: child_hash, accumed_l: accumed_l + node.c_cond_l[child_slot]});
            }
        }
    }

    pub(crate) fn recalc_to_frontier(&mut self, recalc_type: RecalcType) -> RecalcResult {
        // assert that there are no pending updates if we are expanding
        match recalc_type {
            RecalcType::Update => {},
            RecalcType::Expand { .. } => {
                assert!(self.pending_prior.deref().is_empty());
                assert!(self.pending_likelihood.deref().is_empty());
            }
        }
        let p_update = self.pending_prior.deref_mut();
        self.cum_likelihood = self.cum_likelihood.merge(&self.pending_likelihood);
        // mark all the nodes, for which there is a prior update ahead of us, but behind the clf
        // this indicator set is valid for nodes inside the clf, 
        // but may contain meaningless entries beyond the clf
        let p_update_proper_ancestors = {
            // remove predictions beyond the trie
            // p_update.retain(|hash| self.nodes.contains_key(hash));  // don't do this because the trie is theoretically infinite; what matters is our relation to the clf
            let mut res = rh::RHashSet::default();
            let mut frames: Vec<Hash> = p_update
                .iter().cloned().collect::<Vec<_>>();
            while let Some(hash) = frames.pop() {
                if res.contains(&hash) { continue; }
                if self.cum_likelihood.deref().contains_key(&hash) { continue; } // ignore predictions beyond the clf
                res.insert(hash);
                if hash == ROOT_HASH { continue; }
                let symbol = self.nodes.get(&hash).unwrap().symbol;
                let parent_hash = rh::pop_right(hash, symbol.to_byte());
                frames.push(parent_hash);
            }
            res
        };
        self.unread_predictions.deref_mut().extend(p_update.drain());
        let nodes = &mut self.nodes;
        let full_predictions = &self.full_predictions;
        let zero_order_predictions = &mut self.zero_order_predictions;
        let l_update = self.pending_likelihood.deref_mut();
        let cum_likelihood = &self.cum_likelihood;
        let unread_predictions = self.unread_predictions.deref_mut();
        let tokenizer = &self.tokenizer;
        struct Frame {
            symbol: Symbol,
            depth: u16,
            target_hash: Hash,
            n_hit_l_update: bool,  // (inclusive of n)
            n_hit_p_update: bool,  // (inclusive of n)
            n_a_pred_changed: AncestorsBitmap,
            n_a_tp_changed: AncestorsBitmap,
            n_accumed_l: f32, // (inclusive of n)
        }
        let mut n_walker = Self::root_walker();
        let root_pred_changed = unread_predictions.remove(&ROOT_HASH);
        let root_z = match recalc_type {
            RecalcType::Update => None,
            RecalcType::Expand {..} => {
                let root_node = nodes.get(&ROOT_HASH).unwrap();
                Some(root_node.if_root_then_z)
            }
        };
        let root_frame = Frame {
            symbol: Symbol::Start,
            depth: 0,
            target_hash: ROOT_HASH,
            n_hit_l_update: !l_update.is_empty(),
            n_hit_p_update: root_pred_changed,
            n_a_pred_changed: if root_pred_changed { 1u16 } else { 0u16 },
            n_a_tp_changed: 0,
            n_accumed_l: 0.0f32, // WLOG we don't allow the root to have cond_l
        };
        let mut frames: Vec<Frame> = vec![root_frame];
        let mut nodes_over_threshold: Vec<Hash> = vec![];
        let mut invalid_z_hashes: Vec<Hash> = vec![];
        let mut invalid_mtcdl_hashes: Vec<Hash> = vec![];
        let mut iters = 0;
        // 1. descend to valid frontier
        while let Some(Frame {
            symbol,
            depth,
            target_hash,  // TODO: this can be removed since for now it is just for verification
            n_hit_l_update,
            n_hit_p_update,
            n_a_pred_changed,
            n_a_tp_changed,
            n_accumed_l,
        }) = frames.pop() {
            // move the walker to the frame's node
            let n_hash;
            if depth > 0 {
                n_walker.truncate(depth as usize);
                Self::descend(nodes, &mut n_walker, symbol);
                n_hash = *n_walker.a_hash().from_end(0);
                assert!(n_hash == target_hash, "hash mismatch");
            } else {
                n_hash = ROOT_HASH;
                nodes_over_threshold.push(n_hash);
                invalid_z_hashes.push(n_hash);
                invalid_mtcdl_hashes.push(n_hash);
            }
            {
                Self::ensure_zero_order_prediction(
                    zero_order_predictions,
                    tokenizer,
                    *n_walker.a_final_token_hash().from_end(0),
                );
                Self::ensure_node(nodes, zero_order_predictions, tokenizer, &n_walker);
            }
            if iters < 1000 { iters += 1; } else { panic!("apply_updates: too many iterations"); }
            let node = nodes.get_mut(&n_hash).unwrap();
            for slot in 0..RADIX {
                let child_symbol = Symbol::from_slot(slot);
                let child_byte = Symbol::slot_to_byte(slot);
                let child_hash = rh::append_right(n_hash, child_byte);
                // apply likelihood update
                let mut c_hit_l_update = n_hit_l_update;
                if let Some(l) = l_update.remove(&child_hash) {
                    node.c_cond_l[slot] += l;
                    c_hit_l_update = true;
                } 
                // apply prior update
                let mut c_hit_p_update = n_hit_p_update;
                if unread_predictions.remove(&child_hash) {
                    node.c_a_pred_changed[slot] |= 1; // walker will see this when it leaves this node
                    c_hit_p_update = true;
                    // N.B. A prediction at a node, c, doesn't change c.p or c.tp, it only affects descendants
                }
                if child_symbol == Symbol::Space {
                    c_hit_p_update = false; // hit_p_update is cleansed by space
                }
                // determine mtcdl validity
                let valid_mtcdl = c_hit_l_update; // TODO: how does not pushing likelihood affect mtcdl?
                // determine children's validity
                // - valid = const_likelihood || no_reweighting
                let const_likelihood = cum_likelihood.deref().contains_key(&child_hash);
                // TODO: writing this code made me realize we need to add a subtlety
                // to the .tex file
                // For no-reweighting, 
                // the question is where we stand in relation to prior updates
                // - if there is a prior update behind us, we need to reach a space to cleanse ourselves (to ensure no prior reweighting)
                // - if there is a prior update ahead of us, but before the likelihood frontier, we are not valid!
                // - if there is a prior update ahead of us, and after the cum likelihood frontier, we don't care
                //   - (if there is a prior update beyond the trie, this is beyond the clf and so we don't care) WRONG!: the trie is infinite, the clf is all that matters
                let no_reweighting =
                    n_hit_l_update  // this update hasn't rebalanced l beyond here
                    && !p_update_proper_ancestors.contains(&child_hash) // and no p_updates are ahead of us which are behind the clf
                    && !c_hit_p_update; // no p_updates behind (or else cleansed by space) // TODO: consider a better heuristic than space
                let valid_z = const_likelihood || no_reweighting;
                // recalculate prior
                let mut p = node.c_p[slot];
                // propagate tp changed
                let ftl = node.c_final_token_length[slot] as usize;
                let token_a_tp_changed = (n_a_tp_changed << 1) & (1 << ftl) != 0;
                let token_a_pred_changed = (n_a_pred_changed << 1) & (1 << ftl) != 0;
                if token_a_pred_changed {
                    let token_a_tp = n_walker.a_tp().from_end(ftl - 1);
                    let token_a_hash = n_walker.a_hash().from_end(ftl - 1);
                    let final_token_hash = node.c_final_token_hash[slot];
                    assert!(rh::extend_right(*token_a_hash, final_token_hash, ftl) == child_hash, "hash mismatch");
                    let final_token_lexindex = tokenizer.lex_index_for_token_hash(&final_token_hash);
                    let final_token_prob = full_predictions
                        .get(&token_a_hash).unwrap()
                        .follower_prob_for_prefix[final_token_lexindex];
                    node.c_final_token_prob[slot] = final_token_prob;
                    node.c_tp[slot] = token_a_tp + final_token_prob;
                    node.c_a_tp_changed[slot] |= 1;
                } else if token_a_tp_changed {
                    let token_a_tp = n_walker.a_tp().from_end(ftl - 1);
                    node.c_tp[slot] = token_a_tp + node.c_final_token_prob[slot];
                    node.c_a_tp_changed[slot] |= 1;
                }
                // has the prediction changed at any of our possible truncations?
                // invariant: "the predictions which have changed since we last refreshed our fp array...
                // ... are exactly those indicated by the union of our strict ancestor's a_pred_changed arrays"
                let relevant_preds_changed = node.c_can_trunc[slot] & (n_a_pred_changed << 1);
                let fp_changed = relevant_preds_changed != 0;
                let relevant_tps_changed = node.c_can_trunc[slot] & (n_a_tp_changed << 1);
                let tp_changed = relevant_tps_changed != 0;
                if fp_changed {
                    let mut dense_idx = 0;
                    // let mut token_a_hash = n_hash; // leave in case we decide to revert later
                    let mut final_prefix_hash = child_byte as u64;
                    for i in 1..MAX_TOKEN_LENGTH { // indexing is from child's perspective
                        if relevant_preds_changed & (1 << i) != 0 {
                            let new_fp = full_predictions
                                .get(
                                    n_walker.a_hash().from_end(i-1)
                                ).unwrap()
                                .follower_prob_for_prefix[
                                    tokenizer.prefix_lex_index_for_prefix_hash(&final_prefix_hash)
                                ];
                            node.c_fp[slot][dense_idx] = new_fp;
                        }
                        if node.c_can_trunc[slot] & (1 << i) != 0 {
                            dense_idx += 1;
                        }
                        // token_a_hash = rh::pop_right(token_a_hash, n_walker.a_symbol().from_end(i-1).to_byte()); // leave in case we decide to revert later
                        final_prefix_hash = rh::extend_right(n_walker.a_symbol().from_end(i-1).to_byte() as u64, final_prefix_hash, i);
                    }
                }
                let p_changed = fp_changed || tp_changed;
                if p_changed {
                    p = f32::NEG_INFINITY;
                    let mut dense_idx = 0;
                    for i in 1..MAX_TOKEN_LENGTH {
                        let can_trunc = (node.c_can_trunc[slot] & (1 << i)) != 0;
                        if can_trunc {
                            let dp = n_walker.a_tp().from_end(i-1) + node.c_fp[slot][dense_idx];
                            p = logaddexp(p, dp);
                            dense_idx += 1;
                        }
                    }
                }
                //
                node.c_p[slot] = p;
                // merge change arrays into the child
                node.c_a_pred_changed[slot] |= n_a_pred_changed << 1;
                node.c_a_tp_changed[slot] |= n_a_tp_changed << 1;
                // update valid children
                let accumed_l_new = n_accumed_l + node.c_cond_l[slot];
                if valid_z {
                    if const_likelihood {
                        // Z = p + l
                        node.c_z[slot] = p + accumed_l_new;
                    } else if no_reweighting {
                        let p_old = node.c_p_old[slot];
                        let accumed_l_old = node.c_accumed_l_old[slot];
                        // Z += (p - p_old) + (l - l_old)
                        let p_delta = p - p_old;
                        let l_delta = accumed_l_new - accumed_l_old;
                        let z_delta = p_delta + l_delta;
                        node.c_z[slot] += z_delta;
                    }
                }
                node.c_p_old[slot] = p;
                node.c_accumed_l_old[slot] = accumed_l_new;
                // push to upprop arrays and determine wether to stop descending
                let stop = match recalc_type {
                    RecalcType::Update => {
                        if !valid_z {
                            invalid_z_hashes.push(child_hash);
                        }
                        if !valid_mtcdl {
                            invalid_mtcdl_hashes.push(child_hash);
                        }
                        valid_z && valid_mtcdl
                    }
                    RecalcType::Expand { threshold } => {
                        let met_threshold = node.c_z[slot] - root_z.unwrap() >= threshold;
                        if met_threshold {
                            nodes_over_threshold.push(child_hash);
                        }
                        !met_threshold
                    }
                };
                if !stop {
                    frames.push(Frame {
                        symbol: child_symbol,
                        depth: depth + 1,
                        target_hash: child_hash,
                        n_hit_l_update: c_hit_l_update,
                        n_hit_p_update: c_hit_p_update,
                        n_a_pred_changed: node.c_a_pred_changed[slot],
                        n_a_tp_changed: node.c_a_tp_changed[slot],
                        n_accumed_l: accumed_l_new,
                    });
                }
                if !valid_z {
                    // since the invalid child will be visited and all its children set to the
                    // current time, its changed array should be zero after its children are updated
                    // we can do it now, since we won't be coming back here (and walker doesn't read this field)
                    node.c_a_pred_changed[slot] = 0;
                    node.c_a_tp_changed[slot] = 0;
                }
            }
        }
        assert!(self.pending_likelihood.deref().is_empty()); // ensure we consumed all likelihood updates
        // up-prop z 
        while let Some(n_hash) = invalid_z_hashes.pop() {
            // we are proceeding in reverse-topological order
            // hence we are guaranteed that our invalid children have already been
            // up-propagated and therefore our c_z array is correct
            let mut z = f32::NEG_INFINITY;
            let symbol;
            {
                let node = nodes.get(&n_hash).unwrap();
                for slot in 0..RADIX {
                    z = logaddexp(z, node.c_z[slot]);
                }
                symbol = node.symbol;
            }
            // since only child values are stored, 
            // we must store our own value on our parent
            if n_hash == ROOT_HASH {
                {
                    let node = nodes.get_mut(&n_hash).unwrap();
                    node.if_root_then_z = z;
                }
            } else {
                let parent_hash = rh::pop_right(n_hash, symbol.to_byte());
                let parent_node = nodes.get_mut(&parent_hash).unwrap();
                parent_node.c_z[symbol.to_slot()] = z;
            }

        }
        // up-prop mtcdl
        while let Some(n_hash) = invalid_mtcdl_hashes.pop() {
            // upprop mtcdl
            // TODO: #[cfg(feature = "tokentrie")]
            let mut mtcdl = [f32::NEG_INFINITY; MAX_TOKEN_LENGTH];
            let symbol;
            {
                let node = nodes.get(&n_hash).unwrap();
                symbol = node.symbol;
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
            }
            // since only child values are stored, we must store our own value on our parent
            if n_hash == ROOT_HASH {
                // Don't do anything because root.mtcdl is unnecessary since the root is always queried first
            } else {
                let parent_hash = rh::pop_right(n_hash, symbol.to_byte());
                let parent_node = nodes.get_mut(&parent_hash).unwrap();
                let mut mtcdl_dense = [f32::NAN; MAX_TRUNCATION_POSSIBLE+1];
                let nz = parent_node.c_can_trunc[symbol.to_slot()] | 1;
                for (i, v) in sparse16_to_dense(&mtcdl, nz).iter().enumerate() {
                    mtcdl_dense[i] = *v;
                }
                parent_node.c_a_tl[symbol.to_slot()] = mtcdl_dense;
            }
        }
        match recalc_type {
            RecalcType::Update => RecalcResult::Updated,
            RecalcType::Expand {..} => RecalcResult::Expanded { nodes_over_threshold },
        }
    }
    // NOTE: a_pred_changed
    // Logically the order of operations is:
    // - update children based on n's a_pred_changed and a_tp_changed
    // - push n's a_pred_changed and a_tp_changed to its children's changed arrays
    // - clear n's changed arrays
    // - descend to children
    // But, we can't clear n's changed array, except by going to n's parent
    // so instead we prefer:
    // - track n_a_pred_changed and n_a_tp_changed in the frame 
    // - and clear it for all children that will be visited, once we've added their frames
    // N.B. it is slightly problematic for the walker to track mutable fields

    fn ensure_zero_order_prediction(
        zero_order_predictions: &mut rh::RHashMap<XPrediction>,
        tokenizer: &TinyLlamaWordTokenizer,
        final_token_hash: u64,
    ) {
        if zero_order_predictions.contains_key(&final_token_hash) {
            return;
        }
        let prediction = XPrediction::create_prediction(
            true,
            final_token_hash,
            None,
            tokenizer,
        );
        zero_order_predictions.insert(final_token_hash, prediction);
    }

    fn ensure_node(
        nodes: &mut rh::RHashMap<XNode>,
        zero_order_predictions: &rh::RHashMap<XPrediction>,
        tokenizer: &TinyLlamaWordTokenizer,
        walker: &YWalker,
    ) {
        if nodes.contains_key(walker.a_hash().last().unwrap()) {
            return;
        }
        let mut node = XNode {
            symbol: *walker.a_symbol().from_end(0),
            c_can_trunc: [0; RADIX], // ok
            c_final_token_length: [0; RADIX], // ok
            c_final_token_hash: [u64::MAX; RADIX], // ok
            c_p: [f32::NAN; RADIX], // ok
            c_p_old: [f32::NAN; RADIX], // ok
            c_fp: [[f32::NAN; MAX_TRUNCATION_POSSIBLE]; RADIX], // ok
            c_tp0: [f32::NAN; RADIX], // ok
            c_tp: [f32::NAN; RADIX], // ok
            c_final_token_prob: [f32::NAN; RADIX], // ok
            c_cond_l: [0.0; RADIX], // ok
            c_a_tl: [[f32::NAN; MAX_TRUNCATION_POSSIBLE+1]; RADIX], // ok
            c_accumed_l_old: [0.0; RADIX], // ok
            c_z: [f32::NAN; RADIX], // ok
            c_a_pred_changed: [0; RADIX], // ok
            c_a_tp_changed: [0; RADIX], // ok
            if_root_then_z: f32::NAN, // ok
        };
        //
        let available_prediction_depth = usize::min(MAX_TOKEN_LENGTH, walker.len() + 1);
        for slot in 0..RADIX {
            let child_byte = Symbol::slot_to_byte(slot);
            let mut final_chars_hash = child_byte as u64;
            let mut can_trunc_count = 0;
            let mut p = f32::NEG_INFINITY;
            let mut found_canonical_ancestor = false;
            for i in 1..available_prediction_depth { // index is from child's perspective
                let a_pred = zero_order_predictions
                    .get(walker.a_final_token_hash().from_end(i-1))
                    .unwrap();
                // Determine Canonical Token Ancestor
                if tokenizer.token_hashset.contains(&final_chars_hash) {
                    let new_token_lexindex = tokenizer.lex_index_for_token_hash(&final_chars_hash);
                    let canonical_pair = a_pred.canonical_followers[new_token_lexindex];
                    if canonical_pair {
                        node.c_final_token_length[slot] = i as u8;
                        node.c_final_token_hash[slot] = final_chars_hash;
                        let final_token_prob = a_pred.follower_probs[new_token_lexindex];
                        let tp0 = walker.a_tp0().from_end(i-1) + final_token_prob;
                        node.c_final_token_prob[slot] = final_token_prob;
                        node.c_tp0[slot] = tp0;
                        node.c_tp[slot] = tp0;
                        found_canonical_ancestor = true;
                    }
                }
                // Determine Possible Truncations
                if tokenizer.proper_prefix_hashset.contains(&final_chars_hash)
                    || tokenizer.token_hashset.contains(&final_chars_hash)
                {
                    let new_prefix_lexindex = tokenizer.prefix_lex_index_for_prefix_hash(&final_chars_hash);
                    let can_trunc = a_pred.canonical_follower_for_prefix[new_prefix_lexindex];
                    if can_trunc {
                        let fp = a_pred.follower_prob_for_prefix[new_prefix_lexindex];
                        let a_tp0 = walker.a_tp0().from_end(i-1);
                        p = logaddexp(p, a_tp0 + fp);
                        node.c_fp[slot][can_trunc_count] = fp;
                        node.c_can_trunc[slot] |= 1 << i;
                        node.c_a_tl[slot][can_trunc_count+1] = 0.0; // intentionally different index than for c_fp
                        can_trunc_count += 1;
                    }
                }
                final_chars_hash = rh::extend_right(walker.a_symbol().from_end(i-1).to_byte() as u64, final_chars_hash, i);
            }
            assert!(found_canonical_ancestor);
            node.c_a_tl[slot][0] = 0.0;
            node.c_p[slot] = p;
            node.c_z[slot] = 0.0 + p; // accumed_l is 0.0
            node.c_p_old[slot] = p;
        }
        // at no greater price than keeping tp0 on nodes we can initialize nodes to the beginning of time
        nodes.insert(*walker.a_hash().from_end(0), node);
    }

}
