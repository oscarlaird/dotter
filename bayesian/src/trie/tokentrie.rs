use std::collections::BinaryHeap;

const INVALID_TOKEN_LEXINDEX: TokenLexIndex = usize::MAX;
const INVALID_CHAR_NODE_INDEX: NodeIndex = usize::MAX;

use crate::{bpe::NUM_TOKENS, symbol::Symbol};

use super::{NodeIndex, PredictionIndex, TokenLexIndex, Trie};

#[derive(Clone, Copy, Debug)]
struct QueueItem {
    priority: f64,
    p: f64,
    kind: QueueItemKind,
}

#[derive(Clone, Copy, Debug)]
enum QueueItemKind {
    Root,
    Continuation {
        parent_prediction_index: PredictionIndex,
        token_lexindex: TokenLexIndex,
        parent_char_state: CharState,
    },
}

#[derive(Clone, Copy, Debug)]
enum CharState {
    Node {
        index: NodeIndex,
        accumed_ul: f64,
    },
    Missing {
        accumed_ul: f64,
    },
}

fn traverse_and_count_ul(
    char_trie: &Trie,
    suffix_chars: &[Symbol],
    char_state: CharState,
) -> CharState {
    // traverses from node_index to the target node+suffix
    // while accumulating the .ul value of each node (not including the target node)
    match char_state {
        CharState::Missing {accumed_ul: _ } => { char_state },
        CharState::Node { mut index, mut accumed_ul } => {
            for symbol in suffix_chars {
                accumed_ul += char_trie.nodes[index].ul;
                if char_trie.nodes[index].children_start_index.is_none() {
                    return CharState::Missing { accumed_ul }; // could not reach
                } else {
                    index = char_trie.child_index(index, *symbol);
                }
            }
            CharState::Node { index, accumed_ul }
        }
    }
}

use std::cmp::Ordering;

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
        let mut queue: BinaryHeap<QueueItem> = BinaryHeap::new();
        queue.push(QueueItem {
            priority: 0.0,
            p: 0.0,
            kind: QueueItemKind::Root,
        });
        Self { queue }
    }

    fn next(&mut self, char_trie: &Trie) -> QueueItem {
        //
        // if has prediction, visit it
        // else, return it
        let pred_registry = &char_trie.prediction_registry;
        loop {
            let this_item = self
                .queue
                .peek()
                .copied()
                .expect("Queue should never be empty");
            let maybe_this_pred_idx = match this_item.kind {
                QueueItemKind::Root => pred_registry.fullorder_root_prediction_index(),
                QueueItemKind::Continuation { parent_prediction_index, token_lexindex, parent_char_state: _ } => {
                    let parent_pred = pred_registry.get(parent_prediction_index);
                    parent_pred.and_then(|pred| pred.child_prediction_index(token_lexindex))
                }
            };
            if let Some(this_pred_idx) = maybe_this_pred_idx {
                // otherwise, visit it
                let this_item = self.queue.pop().unwrap();
                // prior
                let this_pred = pred_registry.get(this_pred_idx).unwrap();
                let cond_prior = &this_pred.follower_probs;
                // traverse from parent char_state to this char_state while accumulating ul
                let this_char_state = match this_item.kind {
                    QueueItemKind::Root => CharState::Node {
                        // recall that accumed ul doesn't include the target node
                        index: char_trie.root_index(), accumed_ul: 0.0
                    },
                    QueueItemKind::Continuation { parent_char_state, token_lexindex, .. } => {
                        let final_token_str = char_trie.tokenizer.token_at(token_lexindex);
                        let final_token_symbols = Symbol::string_to_vec(final_token_str);
                        traverse_and_count_ul(char_trie, &final_token_symbols[..], parent_char_state)
                    }
                };
                // populate the likelihood upper bound
                let mut tl_array = Box::new([f64::NAN; NUM_TOKENS]);
                match this_char_state {
                    CharState::Missing { accumed_ul } => {
                        tl_array.fill(accumed_ul);
                    },
                    CharState::Node { index, accumed_ul } => {
                        char_trie.set_tl_array(index, &mut tl_array[..], accumed_ul);
                    }
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
                        priority: this_item.p + cond_posterior_ub[ix],
                        p: this_item.p + cond_prior[ix],
                        kind: QueueItemKind::Continuation {
                            parent_prediction_index: this_pred_idx,
                            parent_char_state: this_char_state,
                            token_lexindex: ix as TokenLexIndex,
                        }
                    });
                }
            } else {
                // if no prediction has been made for this token sequence, return it
                break this_item;
            }
        }
    }
}
