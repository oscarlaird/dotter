use std::collections::BinaryHeap;

const INVALID_TOKEN_LEXINDEX: TokenLexIndex = usize::MAX;
const INVALID_CHAR_NODE_INDEX: NodeIndex = usize::MAX;

use crate::bpe::NUM_TOKENS;

use super::{NodeIndex, PredictionIndex, PrefixLexIndex, TokenLexIndex, Trie};

#[derive(Clone, Copy, Debug)]
struct QueueItem {
    priority: f64,
    tp: f64,
    tl: f64,
    char_node_index: Option<NodeIndex>,
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

impl Queue {
    // API:
    // - caller gets next queue item
    // - caller sends predictions

    fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
        }
    }

    fn visit_fullorder(&mut self, char_trie: &Trie, queue_item: QueueItem) {
        // visit] add children to the queue
        // assert can visit
        // add children to the queue
    }

    fn receive_prediction(&mut self) {
        // mark as has_prediction
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
                return next_item;
            }
            // otherwise, visit it
            let this_pred = pred_registry.get(this_pred_idx.unwrap()).unwrap();
            let cond_prior = &this_pred.follower_probs;
            let _prior = cond_prior
                .iter()
                .map(|&p| next_item.tp + p)
                .collect::<Box<[f64]>>();
            let _likelihood_ub = [f64::NAN; NUM_TOKENS];
            // populate the likelihood upper bound
            let _tl = Box::new([f64::NAN; NUM_TOKENS]);
            todo!(
                "tokentrie needs source symbol sequence or source node entrypoint for set_tl_array"
            );
        }
    }
}
