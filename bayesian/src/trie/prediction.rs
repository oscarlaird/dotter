use std::collections::HashMap;

use crate::bpe::{NUM_PREFIXES, NUM_TOKENS, TinyLlamaWordTokenizer};

use super::{PredictionIndex, TokenLexIndex, logaddexp};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PredictionOrder {
    ZeroOrder(Option<String>),
    FirstOrder(Option<String>),
    // Full-order predictions keep both the caller-provided final token and the raw full string.
    // Canonical follower support is derived from `final_token`, so it must match the true final
    // token implied by `full_string`.
    FullOrder(Option<String>, String),
}

#[derive(Clone, Debug)]
pub(crate) struct Prediction {
    pub(crate) order: PredictionOrder,
    pub(crate) canonical_followers: Box<[bool]>,
    pub(crate) canonical_follower_for_prefix: Box<[bool]>,
    pub(crate) follower_probs: Box<[f64]>,
    pub(crate) follower_prob_for_prefix: Box<[f64]>,
    pub(crate) stop_prob: f64,
    pub(crate) children: Vec<(TokenLexIndex, PredictionIndex)>,
}

#[derive(Clone, Debug)]
pub(crate) struct PredictionRegistry {
    predictions: Vec<Prediction>,
    by_order: HashMap<PredictionOrder, PredictionIndex>,
}

impl Prediction {
    pub(crate) fn create_prediction(
        order: PredictionOrder,
        follower_logits: Option<Box<[f64]>>,
        stop_logit: Option<f64>,
        tokenizer: &TinyLlamaWordTokenizer,
    ) -> Self {
        let final_token = match &order {
            PredictionOrder::ZeroOrder(final_token) => final_token.clone(),
            PredictionOrder::FirstOrder(final_token) => final_token.clone(),
            PredictionOrder::FullOrder(final_token, _) => final_token.clone(),
        };
        let is_zero_order = matches!(order, PredictionOrder::ZeroOrder(_));
        if is_zero_order {
            assert!(
                follower_logits.is_none(),
                "zero-order predictions must not provide follower_logits"
            );
            assert!(
                stop_logit.is_none(),
                "zero-order predictions must not provide stop_logit"
            );
        } else {
            assert!(
                follower_logits.is_some(),
                "non-zero-order predictions must provide follower_logits"
            );
            assert!(
                stop_logit.is_some(),
                "non-zero-order predictions must provide stop_logit"
            );
        }

        let canonical_followers_array = match final_token.as_deref() {
            Some(final_token) => tokenizer.canonical_followers(final_token),
            None => [true; NUM_TOKENS],
        };
        let canonical_counts_by_prefix =
            tokenizer.count_true_tokens_by_prefix::<NUM_PREFIXES>(&canonical_followers_array);
        let canonical_total = canonical_counts_by_prefix[tokenizer
            .prefix_lex_index("")
            .expect("empty prefix must always be present")];

        assert!(canonical_total != 0, "canonical_total must not be zero");
        let log_canonical_total = (canonical_total as f64).ln();

        let canonical_follower_for_prefix = canonical_counts_by_prefix
            .iter()
            .map(|&count| count != 0)
            .collect::<Box<[_]>>();

        let (follower_probs, stop_prob) = if is_zero_order {
            (
                canonical_followers_array
                    .iter()
                    .map(|&is_canonical| {
                        if is_canonical {
                            -log_canonical_total
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect::<Box<[_]>>(),
                f64::NEG_INFINITY,
            )
        } else {
            // Caller-provided logits are masked to canonical support and normalized together
            // with the stop logit so the stored values are proper log-probabilities.
            let follower_logits =
                follower_logits.expect("non-zero-order predictions must provide follower_logits");
            let stop_logit =
                stop_logit.expect("non-zero-order predictions must provide stop_logit");
            assert_eq!(
                follower_logits.len(),
                NUM_TOKENS,
                "follower_logits len must match NUM_TOKENS"
            );
            let masked_follower_logits = follower_logits
                .iter()
                .zip(canonical_followers_array.iter())
                .map(|(&logit, &is_canonical)| {
                    if is_canonical {
                        logit
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect::<Box<[_]>>();
            let normalizer = masked_follower_logits
                .iter()
                .copied()
                .fold(stop_logit, logaddexp);
            assert!(
                normalizer.is_finite(),
                "prediction logits must assign finite total mass"
            );
            (
                masked_follower_logits
                    .iter()
                    .map(|&logit| {
                        if logit.is_finite() {
                            logit - normalizer
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect::<Box<[_]>>(),
                stop_logit - normalizer,
            )
        };

        let follower_prob_for_prefix = (0..NUM_PREFIXES)
            .map(|prefix_lex_index| {
                let (start, stop) = tokenizer.token_lex_range_for_prefix_index(prefix_lex_index);
                follower_probs[start..stop]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, logaddexp)
            })
            .collect::<Box<[_]>>();

        let canonical_followers = canonical_followers_array.into_iter().collect::<Box<[_]>>();

        Self {
            order,
            canonical_followers,
            canonical_follower_for_prefix,
            follower_probs,
            follower_prob_for_prefix,
            stop_prob,
            children: Vec::new(),
        }
    }

    pub(crate) fn child_prediction_index(
        &self,
        token_lexindex: TokenLexIndex,
    ) -> Option<PredictionIndex> {
        self.children
            .iter()
            .find_map(|&(child_token_lexindex, prediction_index)| {
                (child_token_lexindex == token_lexindex).then_some(prediction_index)
            })
    }
}

impl PredictionRegistry {
    pub(crate) fn new() -> Self {
        Self {
            predictions: Vec::new(),
            by_order: HashMap::new(),
        }
    }

    pub(crate) fn alloc(&mut self, prediction: Prediction) -> PredictionIndex {
        let index = self.predictions.len();
        self.by_order.insert(prediction.order.clone(), index);
        self.predictions.push(prediction);
        index
    }

    pub(crate) fn get(&self, index: PredictionIndex) -> Option<&Prediction> {
        self.predictions.get(index)
    }

    pub(crate) fn index_for_order(&self, order: &PredictionOrder) -> Option<PredictionIndex> {
        self.by_order.get(order).copied()
    }

    pub(crate) fn get_mut(&mut self, index: PredictionIndex) -> Option<&mut Prediction> {
        self.predictions.get_mut(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.predictions.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }
}
