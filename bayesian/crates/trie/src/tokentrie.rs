use std::cmp::Ordering;
use std::collections::BinaryHeap;
use crate::rolling_hash as rh;
use crate::rolling_hash::Hash;

use crate::bpe::TokenLexIndex;
use crate::symbol::{Symbol, RADIX};
use crate::safe_float::{Float, ZERO};
use crate::core::XBayes;
use crate::l_update::XLUpdate;
use crate::ROOT_HASH;
use crate::bpe::NUM_TOKENS;

const MAX_CONTINUATIONS: usize = 100;


#[derive(Clone, Copy, Debug)]
pub(crate) struct QueueItem {
    priority: Float,
    p: Float,
    kind: QueueItemKind,
}

#[derive(Clone, Copy, Debug)]
enum QueueItemKind {
    Root,
    Continuation {
        parent_hash: Hash,
        token_lexindex: TokenLexIndex,
        cuml_l: Float,
        hit_cuml_edge: bool,
    },
}

pub(crate) struct RequestedPrior {
    pub(crate) full_string: String,
    pub(crate) last_token_lexindex: TokenLexIndex,
}

impl XBayes {
    fn traverse_and_count_l(
        l_update: &XLUpdate,
        b_hash: Hash,
        suffix_chars: &[Symbol],
    ) -> (Float, bool) {
        // Returns:
        // - likelihood (Float): 
        // - on_or_beyond_edge (bool):
        assert!(suffix_chars.len() > 0);
        let on_or_beyond_edge = l_update.get(&b_hash).map(|entry| entry.is_leaf).unwrap_or(true);
        let mut cur_hash = b_hash;
        for symbol in suffix_chars.iter().rev() {
            if let Some(entry) = l_update.get(&cur_hash) {
                return (entry.likelihood, on_or_beyond_edge);
            }
            cur_hash = rh::pop_right(cur_hash, symbol.to_byte());
        }
        if let Some(entry) = l_update.get(&cur_hash) {
            return (entry.likelihood, on_or_beyond_edge);
        }
        debug_assert!({
            let a_hash = cur_hash;
            let a_on_or_beyond_edge = l_update.get(&a_hash).map(|entry| entry.is_leaf).unwrap_or(true);
            !a_on_or_beyond_edge
            }, 
            "traverse_and_count_l: should not be called when a_hash is on or beyond the edge\n  a_hash: {cur_hash:#x}\n  b_hash: {b_hash:#x}\n  suffix_chars: {:?}",
            suffix_chars.iter().map(|s| s.to_byte()).collect::<Vec<_>>()
        );
   
        panic!("traversed from b_hash to a_hash without hitting the l_update");
    }

    // API:
    // - caller gets next queue item
    // - caller sends predictions
    // After a lot of thought, I still think a tokentrie is better
    // than the character-level approach that the mtcdl computation takes
    // it is a nifty piece of math, but this way should be preffered for:
    // - clarity] it is easier to understand than the r-tree
    // - performance] it lets us progress to the next token w/o traversing its letters
    // I was initially afraid of adding 30K items to the heap on each visit,
    // but the original python implementation reminded me that this is not necessary
    // since we can restrict ourselves to the top K, say 100, continuations
    // this will be exactly correct as long as we return fewer than 100 predictions
    // before the next likelihood update
    pub(crate) fn next_requested_prior(&mut self) -> RequestedPrior {
        loop {
            let this_item = self.queue.peek().copied().expect("Queue should never be empty");
            let this_hash = match this_item.kind {
                QueueItemKind::Root => ROOT_HASH,
                QueueItemKind::Continuation { parent_hash, token_lexindex, .. } => {
                    let token_str = self.tokenizer.token_at(token_lexindex);
                    let token_hash = rh::hash_string(token_str);
                    let this_hash =rh::extend_right(parent_hash, token_hash, token_str.len());
                    self.queue_ancestor_map.insert(this_hash, token_lexindex);
                    this_hash
                }
            };
            let Some(this_pred) = self.full_predictions.get(&this_hash) else {
                // There is no prediction for this token sequence, return it to the caller
                // rebuild the string from the ancestor map
                let mut rev_str = String::new();
                let mut cur_hash: u64 = this_hash;
                let mut last_token_lexindex;
                while cur_hash != ROOT_HASH {
                    last_token_lexindex = *self.queue_ancestor_map.get(&cur_hash).unwrap();
                    let token_str = self.tokenizer.token_at(last_token_lexindex);
                    let token_hash = rh::hash_string(token_str);
                    let rev_token_str = token_str.chars().rev().collect::<String>();
                    rev_str.push_str(&rev_token_str);
                    cur_hash = rh::truncate_right(cur_hash, token_hash, token_str.len());
                }
                rev_str.push(Symbol::Start.to_byte() as char);
                let full_string = rev_str.chars().rev().collect::<String>();
                break RequestedPrior {
                    full_string,
                    last_token_lexindex: match this_item.kind {
                        QueueItemKind::Root => TokenLexIndex::INVALID,
                        QueueItemKind::Continuation { token_lexindex, .. } => token_lexindex,
                    }

                }
            };
            // visit it
            self.queue.pop();
            // PRIOR
            // determine how much likelihood was accumulated from our parent to us
            let (cuml_l, hit_cuml_edge) = match this_item.kind {
                QueueItemKind::Root => {
                    let cuml_l = self.cum_likelihood.get(&ROOT_HASH).unwrap().likelihood;
                    let hit_cuml_edge = self.cum_likelihood.get(&ROOT_HASH).unwrap().is_leaf;
                    (cuml_l, hit_cuml_edge)
                }
                QueueItemKind::Continuation { token_lexindex, cuml_l , hit_cuml_edge, .. } => {
                    if hit_cuml_edge {
                        (cuml_l, true)
                    } else {
                        let token_str = self.tokenizer.token_at(token_lexindex);
                        let token_symbols = Symbol::string_to_vec(token_str);
                        XBayes::traverse_and_count_l(
                            &self.cum_likelihood,
                            this_hash,
                            &token_symbols[..],
                        )
                    }
                }
            };
            // LIKELIHOOD (upper bound)
            let mut tl_array = vec![Float::NAN; NUM_TOKENS].into_boxed_slice();
            if hit_cuml_edge {
                tl_array.fill(cuml_l);
            } else {
                self.set_tl_array(this_hash, &mut tl_array[..]);
            }
            // POSTERIOR (upper bound)
            let mut cond_posterior_ub = vec![Float::NAN; NUM_TOKENS].into_boxed_slice();
            for i in 0..NUM_TOKENS {
                let token_lexindex = TokenLexIndex::from_usize(i);
                cond_posterior_ub[i] = if this_pred.canonical_follower(token_lexindex) {
                    tl_array[i] + this_pred.follower_prob(token_lexindex)
                } else {
                    Float::NEG_INFINITY
                };
            }
            let mut top_ix: Vec<usize> = (0..NUM_TOKENS).collect();
            top_ix.select_nth_unstable_by(MAX_CONTINUATIONS-1, |&i, &j| {
                cond_posterior_ub[j].total_cmp(&cond_posterior_ub[i])
            });
            // push to heap
            for &ix in top_ix.iter().take(MAX_CONTINUATIONS) {
                let token_lexindex = TokenLexIndex::from_usize(ix);
                self.queue.push(QueueItem {
                    priority: this_item.p + cond_posterior_ub[ix],
                    p: this_item.p + this_pred.follower_prob(token_lexindex),
                    kind: QueueItemKind::Continuation {
                        parent_hash: this_hash,
                        token_lexindex,
                        cuml_l,
                        hit_cuml_edge,
                    }
                });
            }
        }
    }

    pub(crate) fn set_tl_array(
        &self,
        node_hash: Hash,
        tl_array: &mut [Float],
    ) {
        let cuml = &self.cum_likelihood;
        let nodes = &self.nodes;
        //
        debug_assert!({
            let is_interior_of_cuml = cuml.get(&node_hash).map(|entry| !entry.is_leaf).unwrap_or(false);
            is_interior_of_cuml
        }, "set_tl_array: node_hash is not in the interior of the cum_likelihood");
        let node_cuml_l = cuml.get(&node_hash).unwrap().likelihood;
        //
        // we always need nodes to be available in the interior of the cum_likelihood
        struct Frame {
            suffix_hash: Hash,
            suffix_length: usize,
            n_cuml_l: Float,
        }
        let mut frames = vec![Frame {suffix_hash: 0u64, suffix_length: 0, n_cuml_l: node_cuml_l}];
        let mut iters = 0;
        while let Some(Frame { suffix_hash, suffix_length, n_cuml_l }) = frames.pop() {
            if iters < 1000 { iters += 1; } else { panic!("set_tl_array: too many iterations"); }
            let n_full_hash = rh::extend_right(node_hash, suffix_hash, suffix_length);
            // handle our children
            for slot in 0..RADIX {
                let c_suffix_hash = rh::append_right(suffix_hash, Symbol::slot_to_byte(slot));
                let c_suffix_length = suffix_length + 1;
                let c_full_hash = rh::extend_right(node_hash, c_suffix_hash, c_suffix_length);
                //
                if !self.tokenizer.token_hashset.contains(&c_suffix_hash) && !self.tokenizer.proper_prefix_hashset.contains(&c_suffix_hash) {
                    continue;
                }
                //
                let c_cuml_l = cuml.get(&c_full_hash).map(|entry| entry.likelihood).unwrap_or(n_cuml_l);
                let c_cuml_hit_edge = cuml.get(&c_full_hash).map(|entry| entry.is_leaf).unwrap_or(true);

                if self.tokenizer.token_hashset.contains(&c_suffix_hash) {
                    // we've hit a full token, without hitting the clf; need to use mtcdl
                    let lexindex = self.tokenizer.lex_index_for_token_hash(&c_suffix_hash);
                    let node = nodes.get(&n_full_hash).unwrap();
                    // TODO: this is a little ugly
                    let l_delta_mtcdl = c_cuml_l - node.c_cuml_l_old_for_mtcdl[slot];
                    let new_mtcdl0 = node.c_a_tl[slot][0] + l_delta_mtcdl;
                    tl_array[lexindex.as_usize()] = new_mtcdl0;
                    debug_assert!({
                        !c_cuml_hit_edge ||
                        f32::from(new_mtcdl0 - c_cuml_l).abs() < 1e-3
                    })
                }
                if !self.tokenizer.proper_prefix_hashset.contains(&c_suffix_hash) {
                    continue;
                }
                if c_cuml_hit_edge {
                    let range = self.tokenizer.token_lex_range_for_prefix_hash(&c_suffix_hash);
                    let subslice = &mut tl_array[range.0.as_usize()..range.1.as_usize()];
                    subslice.fill(c_cuml_l);
                    continue;
                }
                frames.push(Frame { suffix_hash: c_suffix_hash, suffix_length: c_suffix_length, n_cuml_l: c_cuml_l});
            }
        }
        debug_assert!(
            tl_array.iter().all(|v| !v.is_nan()),
            "set_tl_array: tl_array contains NaN"
        );
    }

    pub(crate) fn root_queue() -> BinaryHeap<QueueItem> {
        let mut queue: BinaryHeap<QueueItem> = BinaryHeap::new();
        queue.push(QueueItem {
            priority: ZERO,
            p: ZERO,
            kind: QueueItemKind::Root,
        });
        queue
    }

    pub(super) fn reset_queue(&mut self) {
        self.queue = Self::root_queue();
        self.queue_ancestor_map.clear();
    }
}


impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority.total_cmp(&other.priority) == Ordering::Equal
    }
}
impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.total_cmp(&other.priority)
    }
}
