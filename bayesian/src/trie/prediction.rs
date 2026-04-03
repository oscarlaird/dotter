use std::collections::HashMap;

use crate::bpe::{NUM_PREFIXES, NUM_TOKENS, TinyLlamaWordTokenizer};
use crate::trie::rolling_hash as rh;
use crate::symbol::Symbol;
use super::{TokenLexIndex, logaddexp};

const ROOT_HASH: u64 = {
    rh::append_right(0, Symbol::Start.to_byte())
};
pub(crate) struct XPrediction {
    pub(crate) canonical_followers: Box<[bool]>,
    pub(crate) canonical_follower_for_prefix: Box<[bool]>,
    pub(crate) follower_probs: Box<[f32]>,
    pub(crate) follower_prob_for_prefix: Box<[f32]>,
    pub(crate) stop_prob: f32,
}

impl XPrediction {
    pub(crate) fn create_prediction(
        is_zero_order: bool,
        final_token_hash: Hash,
        follower_logits: Option<Box<[f32]>>,
        stop_logit: Option<f32>,
        tokenizer: &TinyLlamaWordTokenizer,
    ) -> Self {
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

        let canonical_followers_array = if final_token_hash == ROOT_HASH {
            vec![true; tokenizer.tokens().len()]
        } else {
            tokenizer.canonical_followers_for_token_hash(final_token_hash)
        };
        let canonical_counts_by_prefix =
            tokenizer.count_true_tokens_by_prefix::<NUM_PREFIXES>(&canonical_followers_array);
        let canonical_total = canonical_counts_by_prefix[tokenizer
            .prefix_lex_index_for_prefix_hash(&0u64) // hash("")=0
            .expect("empty prefix must always be present")];

        assert!(canonical_total != 0, "canonical_total must not be zero");
        let log_canonical_total = (canonical_total as f32).ln();

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
                            f32::NEG_INFINITY
                        }
                    })
                    .collect::<Box<[_]>>(),
                f32::NEG_INFINITY,
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
                        f32::NEG_INFINITY
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
                            f32::NEG_INFINITY
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
                    .fold(f32::NEG_INFINITY, logaddexp)
            })
            .collect::<Box<[_]>>();

        let canonical_followers = canonical_followers_array.into_iter().collect::<Box<[_]>>();

        Self {
            canonical_followers,
            canonical_follower_for_prefix,
            follower_probs,
            follower_prob_for_prefix,
            stop_prob,
        }
    }
}