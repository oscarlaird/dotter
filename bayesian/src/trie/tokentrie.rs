use std::cmp::Ordering;
use std::collections::BinaryHeap;
use crate::rolling_hash as rh;
use crate::rolling_hash::Hash;

use crate::symbol::{Symbol, RADIX};
use crate::safe_float::{Float, ZERO};
use crate::trie::core::XBayes;
use crate::trie::ROOT_HASH;
use crate::trie::{TokenLexIndex, INVALID_TOKEN_LEXINDEX};
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
        accumed_l: Float,
        hit_clf: bool,
    },
}

pub(crate) struct RequestedPrior {
    pub(crate) full_string: String,
    pub(crate) last_token_lexindex: TokenLexIndex,
}

impl XBayes {
    fn traverse_and_count_l(
        &self,
        a_hash: Hash,
        suffix_chars: &[Symbol],
        mut accumed_l: Float,
        mut hit_clf: bool,
    ) -> (Float, bool) {
        // traverses from a_hash to b_hash
        // while accumulating l (should include b.l, but not a.l)
        let mut cur_hash = a_hash;
        for symbol in suffix_chars {
            let Some(cur_node) = self.nodes.get(&cur_hash) else {
                break;
            };
            accumed_l += cur_node.c_cond_l[symbol.to_slot()];
            cur_hash = rh::append_right(cur_hash, symbol.to_byte());
            hit_clf = hit_clf || self.cum_likelihood.deref().contains_key(&cur_hash);
        }
        (accumed_l, hit_clf)
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
                        QueueItemKind::Root => INVALID_TOKEN_LEXINDEX,
                        QueueItemKind::Continuation { token_lexindex, .. } => token_lexindex,
                    }

                }
            };
            // visit it
            self.queue.pop();
            // PRIOR
            let cond_prior = &this_pred.follower_probs;
            // determine how much likelihood was accumulated from our parent to us
            let (accumed_l, hit_clf) = match this_item.kind {
                QueueItemKind::Root => (ZERO, false),
                QueueItemKind::Continuation { parent_hash, token_lexindex, accumed_l , hit_clf } => {
                    let token_str = self.tokenizer.token_at(token_lexindex);
                    let token_symbols = Symbol::string_to_vec(token_str);
                    self.traverse_and_count_l(parent_hash, &token_symbols[..], accumed_l, hit_clf)
                }
            };
            // LIKELIHOOD (upper bound)
            let mut tl_array = vec![Float::NAN; NUM_TOKENS].into_boxed_slice();
            if self.nodes.contains_key(&this_hash) {
                self.set_tl_array(this_hash, &mut tl_array[..], accumed_l, hit_clf);
            } else {
                tl_array.fill(accumed_l);
            }
            // POSTERIOR (upper bound)
            let mut cond_posterior_ub = vec![Float::NAN; NUM_TOKENS].into_boxed_slice();
            for i in 0..NUM_TOKENS {
                cond_posterior_ub[i] = tl_array[i] + cond_prior[i];
            }
            let mut top_ix: Vec<usize> = (0..NUM_TOKENS).collect();
            top_ix.select_nth_unstable_by(MAX_CONTINUATIONS-1, |&i, &j| {
                cond_posterior_ub[j].total_cmp(&cond_posterior_ub[i])
            });
            // push to heap
            for &ix in top_ix.iter().take(MAX_CONTINUATIONS) {
                self.queue.push(QueueItem {
                    priority: this_item.p + cond_posterior_ub[ix],
                    p: this_item.p + cond_prior[ix],
                    kind: QueueItemKind::Continuation {
                        parent_hash: this_hash,
                        token_lexindex: ix as TokenLexIndex,
                        accumed_l,
                        hit_clf,
                    }
                });
            }
        }
    }

    pub(crate) fn set_tl_array(
        &self,
        node_hash: Hash,
        tl_array: &mut [Float],
        accumed_l: Float,
        hit_clf: bool, // hit the clf at or above node_hash
    ) {
        struct Frame {
            suffix_hash: Hash,
            suffix_length: usize,
            accumed_l: Float, // accumed_l at or above this node
        }
        let mut frames = vec![Frame {suffix_hash: 0u64, suffix_length: 0, accumed_l}];
        if !self.nodes.contains_key(&node_hash) {
            tl_array.fill(accumed_l);
            return;
        }
        if hit_clf {
            tl_array.fill(accumed_l);
            return;
        }
        let mut iters = 0;
        while let Some(Frame { suffix_hash, suffix_length, accumed_l }) = frames.pop() {
            if iters < 1000 { iters += 1; } else { panic!("set_tl_array: too many iterations"); }
            let node = {
                let full_hash = rh::extend_right(node_hash, suffix_hash, suffix_length);
                self.nodes.get(&full_hash).unwrap()
            };
            // handle our children
            for child_slot in 0..RADIX {
                let child_suffix_hash = rh::append_right(suffix_hash, Symbol::slot_to_byte(child_slot));
                let child_suffix_length = suffix_length + 1;
                let child_full_hash = rh::extend_right(node_hash, child_suffix_hash, child_suffix_length);
                if self.tokenizer.token_hashset.contains(&child_suffix_hash) {
                    let lexindex = self.tokenizer.lex_index_for_token_hash(&child_suffix_hash);
                    tl_array[lexindex] = accumed_l + node.c_a_tl[child_slot][0];
                }
                if !self.tokenizer.proper_prefix_hashset.contains(&child_suffix_hash) {
                    continue;
                }
                let child_is_clf = self.cum_likelihood.deref().contains_key(&child_full_hash);
                if child_is_clf {
                    let range = self.tokenizer.token_lex_range_for_prefix_hash(&child_suffix_hash);
                    let subslice = &mut tl_array[range.0..range.1];
                    subslice.fill(accumed_l + node.c_cond_l[child_slot]);
                    continue;
                }
                frames.push(Frame { suffix_hash: child_suffix_hash, suffix_length: child_suffix_length, accumed_l: accumed_l + node.c_cond_l[child_slot]});
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

    fn reset_queue(&mut self) {
        // TODO: this needs to be called in the appropriate places
        self.queue = Self::root_queue();
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
