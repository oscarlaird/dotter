use std::collections::BinaryHeap;

use crate::bpe::{TinyLlamaWordTokenizer, TOKENIZER_JSON_STR};
use crate::rolling_hash::Hash;
use crate::safe_float::{Float, ZERO};
use crate::symbol::{Symbol, RADIX};
use crate::trie::{MAX_TRUNCATION_POSSIBLE, TokenLexIndex, INVALID_TOKEN_LEXINDEX};
use crate::rolling_hash as rh;
use crate::trie::prediction::XPrediction;
use crate::bpe::NUM_TOKENS;

mod sparse16;
mod y_walker;
#[cfg(test)]
pub(crate) mod debug;

use super::{
    MAX_TOKEN_LENGTH, ROOT_HASH, logaddexp,
};
use super::l_update::{merge_xl_pair, new_xlupdate, XLUpdate};
use super::p_update::PUpdate;
#[cfg(feature = "tokentrie")]
use super::tokentrie::{QueueItem};
use sparse16::{dense_to_sparse16, sparse16_to_dense};
use y_walker::{FromEnd as _, YWalker, YWalkerRow};

#[derive(Clone, Debug)]
pub(crate) struct XNode {
    pub(crate) symbol: Symbol,
    // we don't store a node's hash, because there is no way to reach the node without knowing it
    // for all matrices, children are arrayed on the major axis
    c_can_trunc: [u16; RADIX],
    c_final_token_length: [u8; RADIX],
    c_final_token_lexindex: [TokenLexIndex; RADIX],
    c_p: [Float; RADIX],
    c_p_old: [Float; RADIX],
    c_fp: [[Float; MAX_TRUNCATION_POSSIBLE]; RADIX],
    c_tp: [Float; RADIX],
    c_tp0: [Float; RADIX],
    c_final_token_prob: [Float; RADIX],
    // TODO: #[cfg(feature = "tokentrie")]
    pub(crate) c_a_tl: [[Float; MAX_TRUNCATION_POSSIBLE+1]; RADIX],
    c_cuml_l_old: [Float; RADIX],
    pub(crate) c_cuml_l_old_for_mtcdl: [Float; RADIX],
    pub(crate) c_z: [Float; RADIX],
    //
    c_a_pred_changed: [AncestorsBitmap; RADIX], // ancestor predictions which have changed since we visited this child
    c_a_tp_changed: [AncestorsBitmap; RADIX], // ancestor tps which have changed since we visited this child
    // ROOT
    pub(crate) if_root_then_z: Float,
}

pub(crate) struct XBayes {
    pub(crate) nodes: rh::RHashMap<XNode>,
    pub(crate) full_predictions: rh::RHashMap<XPrediction>,
    zero_order_predictions: Box<[Option<XPrediction>]>,
    root_zero_order_prediction: XPrediction,
    pub(crate) pending_likelihood: XLUpdate,
    pub(crate) cum_likelihood: XLUpdate,
    pub(crate) pending_prior: PUpdate,
    unread_predictions: PUpdate,
    pub(crate) tokenizer: TinyLlamaWordTokenizer,
    #[cfg(feature = "tokentrie")]
    pub(super) queue: BinaryHeap<QueueItem>,
    #[cfg(feature = "tokentrie")]
    pub(super) queue_ancestor_map: rh::RHashMap<TokenLexIndex>,
}


type ContextWindowSize = u8;
type AncestorsBitmap = u16;

pub(crate) enum RecalcType {
    Update,
    Expand { threshold: Float },
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
        let zero_order_predictions = std::iter::repeat_with(|| None)
            .take(NUM_TOKENS)
            .collect::<Box<[_]>>();
        let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json_str(TOKENIZER_JSON_STR);
        let root_zero_order_prediction = XPrediction::create_prediction(
            true,
            INVALID_TOKEN_LEXINDEX,
            None,
            &tokenizer,
        );
        XBayes::ensure_node(&mut nodes, &zero_order_predictions, &root_zero_order_prediction, &tokenizer, &YWalker::root(ROOT_HASH));
        //
        //
        Self {
            nodes,
            full_predictions,
            zero_order_predictions,
            root_zero_order_prediction,
            pending_likelihood: new_xlupdate(),
            cum_likelihood: new_xlupdate(),
            pending_prior: PUpdate::default(),
            unread_predictions: PUpdate::default(),
            tokenizer: tokenizer,
            #[cfg(feature = "tokentrie")]
            queue: Self::root_queue(),
            #[cfg(feature = "tokentrie")]
            queue_ancestor_map: rh::RHashMap::default(),
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
            p_node.c_final_token_lexindex[slot],
            symbol,
            p_node.c_tp[slot],
            p_node.c_tp0[slot],
        ));
    }

    pub(crate) fn recalc_to_frontier(&mut self, recalc_type: RecalcType) -> RecalcResult {
        // assert that there are no pending updates if we are expanding
        #[cfg(feature = "tokentrie")]
        {
            if matches!(recalc_type, RecalcType::Update)
                && !self.pending_likelihood.is_empty()
            {
                self.reset_queue();
            }
        }
        match recalc_type {
            RecalcType::Update => {},
            RecalcType::Expand { .. } => {
                assert!(self.pending_prior.is_empty());
                assert!(self.pending_likelihood.len() == 1);
            }
        }
        self.cum_likelihood = merge_xl_pair(&self.cum_likelihood, &self.pending_likelihood);
        // mark all the nodes, for which there is a prior update at or ahead of us, but strictly behind the clf
        let prior_update_ancestors = {
            let mut res = rh::RHashSet::default();
            let mut frames: Vec<Hash> = self.pending_prior
                .iter()
                .filter_map(|&hash| {
                    let is_interior_of_clf = self.cum_likelihood.get(&hash).map(|entry| !entry.is_leaf).unwrap_or(false);
                    if is_interior_of_clf { Some(hash) } else { None }
                })
                .collect::<Vec<_>>();
            while let Some(hash) = frames.pop() {
                if res.contains(&hash) { continue; }
                res.insert(hash);
                if hash == ROOT_HASH { continue; }
                let symbol = self.cum_likelihood.get(&hash).unwrap().symbol;
                let parent_hash = rh::pop_right(hash, symbol.to_byte());
                frames.push(parent_hash);
            }
            res
        };
        self.unread_predictions.extend(self.pending_prior.drain());
        let nodes = &mut self.nodes;
        let full_predictions = &self.full_predictions;
        let zero_order_predictions = &mut self.zero_order_predictions;
        let root_zero_order_prediction = &self.root_zero_order_prediction;
        let pending_likelihood = &mut self.pending_likelihood;
        let cum_likelihood = &self.cum_likelihood;
        let unread_predictions = &mut self.unread_predictions;
        let tokenizer = &self.tokenizer;
        //
        let root_const_likelihood = cum_likelihood.len() == 1;
        if root_const_likelihood && matches!(recalc_type, RecalcType::Update) {
            return RecalcResult::Updated;
        }
        //
        struct Frame {
            symbol: Symbol,
            depth: u16,
            target_hash: Hash,
            // n_hit_l_update_edge: bool,  // (inclusive of n)
            // n_lupdate_hit_edge: bool,
            // n_cuml_hit_edge: bool,
            // n_lupdate_l: Float,
            n_cuml_l: Float,
            //
            n_hit_prior_update: bool,  // (inclusive of n) (cleansed by space)
            n_a_pred_changed: AncestorsBitmap,
            n_a_tp_changed: AncestorsBitmap,
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
        // let root_lupdate = pending_likelihood.get(&ROOT_HASH).unwrap();
        let root_cuml = cum_likelihood.get(&ROOT_HASH).unwrap();
        let root_frame = Frame {
            symbol: Symbol::Start,
            depth: 0,
            target_hash: ROOT_HASH,
            // n_lupdate_hit_edge: root_lupdate.is_leaf,
            // n_cuml_hit_edge: root_cuml.is_leaf,
            // n_lupdate_l: root_lupdate.likelihood,
            n_cuml_l: root_cuml.likelihood,
            n_hit_prior_update: root_pred_changed,
            n_a_pred_changed: if root_pred_changed { 1u16 } else { 0u16 },
            n_a_tp_changed: 0,
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
            // n_lupdate_hit_edge,
            // n_cuml_hit_edge,
            // n_lupdate_l,
            n_cuml_l,
            n_hit_prior_update,
            n_a_pred_changed,
            n_a_tp_changed,
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
                if n_hash != ROOT_HASH {
                    Self::ensure_zero_order_prediction(
                        zero_order_predictions,
                        tokenizer,
                        *n_walker.a_final_token_lexindex().from_end(0),
                    );
                }
                Self::ensure_node(nodes, zero_order_predictions, root_zero_order_prediction, tokenizer, &n_walker);
            }
            if iters < 1000 { iters += 1; } else { panic!("apply_updates: too many iterations"); }
            let node = nodes.get_mut(&n_hash).unwrap();
            // let n_lupdate = pending_likelihood.get(&n_hash);
            // let n_cuml = cum_likelihood.get(&n_hash);
            for slot in 0..RADIX {
                let child_symbol = Symbol::from_slot(slot);
                let child_byte = Symbol::slot_to_byte(slot);
                let child_hash = rh::append_right(n_hash, child_byte);
                //
                let c_lupdate = pending_likelihood.get(&child_hash);
                let c_cuml = cum_likelihood.get(&child_hash);
                // extract likelihood update
                // let c_lupdate_closed_edge = c_lupdate.map(|e| e.is_leaf).unwrap_or(false);
                // let c_lupdate_open_edge = c_lupdate.is_none() && n_lupdate.map(|e| !e.is_leaf).unwrap_or(false);
                // let c_lupdate_on_edge = c_lupdate_closed_edge || c_lupdate_open_edge;
                // let c_lupdate_hit_edge = n_lupdate_hit_edge && c_lupdate_on_edge;
                let c_lupdate_hit_edge = c_lupdate.map(|e| e.is_leaf).unwrap_or(true);
                // let c_lupdate_l: Float = c_lupdate.map(|entry| entry.likelihood).unwrap_or(n_lupdate_l);
                // extract clf
                // let c_cuml_closed_edge = c_cuml.map(|entry| entry.is_leaf).unwrap_or(false);
                // let c_cuml_open_edge = c_cuml.is_none() && n_cuml.map(|e| !e.is_leaf).unwrap_or(false);
                // let c_cuml_on_edge = c_cuml_closed_edge || c_cuml_open_edge;
                // let c_cuml_hit_edge = n_cuml_hit_edge && c_cuml_on_edge;
                let c_cuml_hit_edge = c_cuml.map(|e| e.is_leaf).unwrap_or(true);
                let c_cuml_l: Float = c_cuml.map(|entry| entry.likelihood).unwrap_or(n_cuml_l);
                // apply prior update
                let mut c_hit_p_update = n_hit_prior_update;
                if unread_predictions.remove(&child_hash) {
                    node.c_a_pred_changed[slot] |= 1; // walker will see this when it leaves this node
                    c_hit_p_update = true;
                    // N.B. A prediction at a node, c, doesn't change c.p or c.tp, it only affects descendants
                }
                if child_symbol == Symbol::Space {
                    // TODO: consider a better heuristic than space
                    c_hit_p_update = false; // hit_p_update is cleansed by space
                }
                // determine children's validity
                // - valid = const_likelihood || no_reweighting
                let const_likelihood = c_cuml_hit_edge;
                // TODO: writing this code made me realize we need to add a subtlety
                // to the .tex file
                // For no-reweighting, 
                // the question is where we stand in relation to prior updates
                // - if there is a prior update behind us, we need to reach a space to cleanse ourselves (to ensure no prior reweighting)
                // - if there is a prior update ahead of us, but before the likelihood frontier, we are not valid!
                // - if there is a prior update ahead of us, and after the cum likelihood frontier, we don't care
                //   - (if there is a prior update beyond the trie, this is beyond the clf and so we don't care) WRONG!: the trie is infinite, the clf is all that matters
                let no_likelihood_reweight= c_lupdate_hit_edge;
                let valid_mtcdl = no_likelihood_reweight; // determine mtcdl validity
                let no_prior_reweight=
                    !prior_update_ancestors.contains(&child_hash) // no p_updates are at/ahead of us which are behind the clf
                    && !c_hit_p_update; // no p_updates at/behind (or else cleansed by space)
                let no_reweight = no_likelihood_reweight && no_prior_reweight;
                let valid_z = const_likelihood || no_reweight;
                // valid_z => valid_mtcdl (since both branches of valid_z contain no_likelihood_reweight)
                // RECALCULATE PRIOR
                let mut p = node.c_p[slot];
                // propagate tp changed
                let ftl = node.c_final_token_length[slot] as usize;
                let token_a_tp_changed = (n_a_tp_changed << 1) & (1 << ftl) != 0;
                let token_a_pred_changed = (n_a_pred_changed << 1) & (1 << ftl) != 0;
                if token_a_pred_changed {
                    let token_a_tp = n_walker.a_tp().from_end(ftl - 1);
                    let token_a_hash = n_walker.a_hash().from_end(ftl - 1);
                    let final_token_lexindex = node.c_final_token_lexindex[slot];
                    assert!(rh::extend_right(*token_a_hash, rh::hash_string(tokenizer.token_at(final_token_lexindex as usize)), ftl) == child_hash, "hash mismatch");
                    let final_token_prob = full_predictions
                        .get(&token_a_hash).unwrap()
                        .follower_prob_for_prefix[final_token_lexindex as usize];
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
                        if i > n_walker.len() { continue; }
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
                    p = Float::NEG_INFINITY;
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
                // RECALCULATE POSTERIOR 
                if valid_z {
                    if const_likelihood {
                        // Z = p + l
                        node.c_z[slot] = p + c_cuml_l;
                    } else if no_reweight {
                        let l_delta = c_cuml_l - node.c_cuml_l_old[slot];
                        let p_delta = if p_changed {
                            p - node.c_p_old[slot]
                        } else {
                            ZERO
                        };
                        let z_delta = p_delta + l_delta;
                        node.c_z[slot] += z_delta;
                    }
                    // recalculate mtcdl
                    assert!(valid_mtcdl);
                    let l_delta_for_mtcdl = c_cuml_l - node.c_cuml_l_old_for_mtcdl[slot];
                    for i in 0..(node.c_can_trunc[slot].count_ones() as usize + 1) {
                        node.c_a_tl[slot][i] += l_delta_for_mtcdl;
                    }
                }
                node.c_p_old[slot] = p;
                node.c_cuml_l_old[slot] = c_cuml_l;
                node.c_cuml_l_old_for_mtcdl[slot] = c_cuml_l;
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
                        // n_lupdate_hit_edge: c_lupdate_hit_edge,
                        // n_cuml_hit_edge: c_cuml_hit_edge,
                        // n_lupdate_l: c_lupdate_l,
                        n_cuml_l: c_cuml_l,
                        n_hit_prior_update: c_hit_p_update,
                        n_a_pred_changed: node.c_a_pred_changed[slot],
                        n_a_tp_changed: node.c_a_tp_changed[slot],
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
        self.pending_likelihood = new_xlupdate();
        // up-prop z 
        while let Some(n_hash) = invalid_z_hashes.pop() {
            // we are proceeding in reverse-topological order
            // hence we are guaranteed that our invalid children have already been
            // up-propagated and therefore our c_z array is correct
            let mut z = Float::NEG_INFINITY;
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
            let mut mtcdl = [Float::NEG_INFINITY; MAX_TOKEN_LENGTH];
            let symbol;
            {
                fn safe_max(a: Float, b: Float) -> Float {
                    if a == Float::NEG_INFINITY {
                        return b;
                    }
                    if b == Float::NEG_INFINITY {
                        return a;
                    }
                    return a.max(b);
                }
                let node = nodes.get(&n_hash).unwrap();
                symbol = node.symbol;
                for slot in 0..RADIX {
                    let c_mtcdl = node.c_a_tl[slot];
                    let c_can_trunc = node.c_can_trunc[slot];
                    let nz = c_can_trunc | 1;
                    let expanded_c_mtcdl = dense_to_sparse16( &c_mtcdl, nz, Float::NEG_INFINITY);
                    for i in 0..(MAX_TOKEN_LENGTH-1) {
                        mtcdl[i] = safe_max(mtcdl[i], expanded_c_mtcdl[i+1]);
                    }
                    let c_ftl = node.c_final_token_length[slot] as usize;
                    mtcdl[c_ftl - 1] = safe_max(mtcdl[c_ftl - 1], expanded_c_mtcdl[0]);
                }
            }
            // since only child values are stored, we must store our own value on our parent
            if n_hash == ROOT_HASH {
                // Don't do anything because root.mtcdl is unnecessary since the root is always queried first
            } else {
                let parent_hash = rh::pop_right(n_hash, symbol.to_byte());
                let parent_node = nodes.get_mut(&parent_hash).unwrap();
                let mut mtcdl_dense = [Float::NAN; MAX_TRUNCATION_POSSIBLE+1];
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
        zero_order_predictions: &mut [Option<XPrediction>],
        tokenizer: &TinyLlamaWordTokenizer,
        final_token_lexindex: TokenLexIndex,
    ) {
        if zero_order_predictions[final_token_lexindex as usize].is_some() {
            return;
        }
        let prediction = XPrediction::create_prediction(
            true,
            final_token_lexindex,
            None,
            tokenizer,
        );
        zero_order_predictions[final_token_lexindex as usize] = Some(prediction);
    }

    fn ensure_node(
        nodes: &mut rh::RHashMap<XNode>,
        zero_order_predictions: &[Option<XPrediction>],
        root_zero_order_prediction: &XPrediction,
        tokenizer: &TinyLlamaWordTokenizer,
        walker: &YWalker,
    ) {
        let n_hash = *walker.a_hash().last().unwrap();
        if nodes.contains_key(&n_hash) {
            return;
        }
        let mut node = XNode {
            symbol: *walker.a_symbol().from_end(0),
            c_can_trunc: [0; RADIX], // ok
            c_final_token_length: [0; RADIX], // ok
            c_final_token_lexindex: [INVALID_TOKEN_LEXINDEX; RADIX], // ok
            c_p: [Float::NAN; RADIX], // ok
            c_p_old: [Float::NAN; RADIX], // ok
            c_fp: [[Float::NAN; MAX_TRUNCATION_POSSIBLE]; RADIX], // ok
            c_tp0: [Float::NAN; RADIX], // ok
            c_tp: [Float::NAN; RADIX], // ok
            c_final_token_prob: [Float::NAN; RADIX], // ok
            c_a_tl: [[Float::NAN; MAX_TRUNCATION_POSSIBLE+1]; RADIX], // ok
            c_cuml_l_old: [ZERO; RADIX], // ok
            c_cuml_l_old_for_mtcdl: [ZERO; RADIX], // ok
            c_z: [Float::NAN; RADIX], // ok
            c_a_pred_changed: [0; RADIX], // ok
            c_a_tp_changed: [0; RADIX], // ok
            if_root_then_z: if n_hash == ROOT_HASH { ZERO } else { Float::NAN }, // ok
        };
        //
        let available_prediction_depth = usize::min(MAX_TOKEN_LENGTH, walker.len() + 1);
        for slot in 0..RADIX {
            let child_byte = Symbol::slot_to_byte(slot);
            let mut final_chars_hash = child_byte as u64;
            let mut can_trunc_count = 0;
            let mut p = Float::NEG_INFINITY;
            let mut found_canonical_ancestor = false;
            for i in 1..available_prediction_depth { // index is from child's perspective
                let a_final_token_lexindex = *walker.a_final_token_lexindex().from_end(i-1);
                let a_pred: &XPrediction = if a_final_token_lexindex == INVALID_TOKEN_LEXINDEX {
                    root_zero_order_prediction
                } else {
                    zero_order_predictions[a_final_token_lexindex as usize].as_ref().unwrap()
                };
                // Determine Canonical Token Ancestor
                if tokenizer.token_hashset.contains(&final_chars_hash) {
                    let final_token_lexindex = tokenizer.lex_index_for_token_hash(&final_chars_hash) as TokenLexIndex;
                    let canonical_pair = a_pred.canonical_followers[final_token_lexindex as usize];
                    if canonical_pair {
                        node.c_final_token_length[slot] = i as u8;
                        node.c_final_token_lexindex[slot] = final_token_lexindex;
                        let final_token_prob = a_pred.follower_probs[final_token_lexindex as usize];
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
                        node.c_a_tl[slot][can_trunc_count+1] = ZERO; // intentionally different index than for c_fp
                        can_trunc_count += 1;
                    }
                }
                final_chars_hash = rh::extend_right(walker.a_symbol().from_end(i-1).to_byte() as u64, final_chars_hash, i);
            }
            assert!(found_canonical_ancestor);
            assert!(p.is_finite());
            node.c_a_tl[slot][0] = ZERO;
            node.c_p[slot] = p;
            node.c_z[slot] = ZERO + p; // accumed_l is 0.0
            node.c_p_old[slot] = p;
        }
        // at no greater price than keeping tp0 on nodes we can initialize nodes to the beginning of time
        nodes.insert(*walker.a_hash().from_end(0), node);
    }

}
