use std::collections::BinaryHeap;

const INVALID_TOKEN_LEXINDEX: TokenLexIndex = usize::MAX;
const INVALID_CHAR_NODE_INDEX: NodeIndex = usize::MAX;

use crate::{bpe::NUM_TOKENS, symbol::Symbol};

use super::{NodeIndex, PredictionIndex, TokenLexIndex, Trie};

#[derive(Clone, Copy, Debug)]
struct QueueItem {
    priority: f64,
    p: f64,
    parent_ul: f64,
    parent_char_node_index: Option<NodeIndex>,
    parent_prediction_index: Option<PredictionIndex>,
    token_lexindex: TokenLexIndex,
}

use std::cmp::Ordering;

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.parent_prediction_index == other.parent_prediction_index
            && self.token_lexindex == other.token_lexindex
    }
}
impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Note: BinaryHeap is a max-heap. To use f64 as priority, reverse comparison for min-heap behavior.
        other.priority.partial_cmp(&self.priority)
    }
}
impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse to get min-heap behavior if smaller priorities mean higher priority
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal)
    }
}

struct Queue {
    queue: BinaryHeap<QueueItem>,
}

const MAX_CONTINUATIONS: usize = 100;
impl Queue {
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

    fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
        }
    }

    fn receive_prediction(&mut self) {
        // Nothing to do here
        // caller should just call next() again after updating the prediction registry
    }

    fn next(&mut self, char_trie: &Trie) -> QueueItem {
        //
        // if has prediction, visit it
        // else, return it
        let pred_registry = &char_trie.prediction_registry;
        loop {
            let next_item = self
                .queue
                .peek()
                .copied()
                .expect("Queue should never be empty");
            let this_pred_idx = next_item
                .parent_prediction_index
                .and_then(|parent_prediction_index| pred_registry.get(parent_prediction_index))
                .and_then(|prediction| prediction.child_prediction_index(next_item.token_lexindex));
            // if no prediction has been made for this token sequence, return it
            if this_pred_idx.is_none() {
                break next_item;
            }
            // otherwise, visit it
            // prior
            let this_pred = pred_registry.get(this_pred_idx.unwrap()).unwrap();
            let cond_prior = &this_pred.follower_probs;
            // navigate from parent to ourself
            let final_token_str = char_trie.tokenizer.token_at(next_item.token_lexindex);
            let final_token_symbols = Symbol::string_to_vec(final_token_str);
            let (maybe_this_node_idx, ul) = char_trie.traverse_and_count_ul(
                next_item.parent_char_node_index.unwrap(),
                &final_token_symbols[..],
                next_item.parent_ul,
            );
            // populate the likelihood upper bound
            let mut tl_array = Box::new([f64::NAN; NUM_TOKENS]);
            if maybe_this_node_idx.is_none() {
                tl_array.fill(ul);
            } else {
                char_trie.set_tl_array(
                    maybe_this_node_idx.unwrap(),
                    &mut tl_array[..],
                    ul,
                );
            }
            // posterior (ub)
            let cond_posterior_ub = &mut Box::new([f64::NAN; NUM_TOKENS]);
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
                    priority: next_item.p + cond_posterior_ub[ix],
                    p: next_item.p + cond_prior[ix],
                    parent_ul: ul,
                    parent_char_node_index: maybe_this_node_idx,
                    parent_prediction_index: this_pred_idx,
                    token_lexindex: ix as TokenLexIndex,
                });
            }
        }
    }
}
