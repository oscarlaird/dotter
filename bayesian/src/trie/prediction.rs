use crate::bpe::{NUM_PREFIXES, NUM_TOKENS, PrefixLexIndex, TinyLlamaWordTokenizer, TokenLexIndex};
use crate::safe_float::Float;
use super::{logaddexp};

pub(crate) struct XPrediction {
    canonical_followers: Box<[bool]>,
    canonical_follower_for_prefix: Box<[bool]>,
    follower_probs: Box<[Float]>,
    follower_prob_for_prefix: Box<[Float]>,
}

impl XPrediction {
    pub(crate) fn create_prediction(
        is_zero_order: bool,
        final_token_lexindex: TokenLexIndex,
        follower_logits: Option<Box<[Float]>>,
        tokenizer: &TinyLlamaWordTokenizer,
    ) -> Self {
        if is_zero_order {
            assert!(
                follower_logits.is_none(),
                "zero-order predictions must not provide follower_logits"
            );
        } else {
            assert!(
                follower_logits.is_some(),
                "non-zero-order predictions must provide follower_logits"
            );
        }

        let canonical_followers_array = if final_token_lexindex == TokenLexIndex::INVALID {
            vec![true; tokenizer.tokens().len()]
        } else {
            tokenizer.canonical_followers_for_lex_index(final_token_lexindex)
        };
        let canonical_counts_by_prefix =
            tokenizer.count_true_tokens_by_prefix::<NUM_PREFIXES>(&canonical_followers_array);
        let canonical_total = canonical_counts_by_prefix[tokenizer
            .prefix_lex_index_for_prefix_hash(&0u64)
            .as_usize()]; // hash("")=0

        assert!(canonical_total != 0, "canonical_total must not be zero");
        let log_canonical_total = Float::from(canonical_total as f32).ln();

        let canonical_follower_for_prefix = canonical_counts_by_prefix
            .iter()
            .map(|&count| count != 0)
            .collect::<Box<[_]>>();

        let follower_probs = if is_zero_order {
            canonical_followers_array
                .iter()
                .map(|&is_canonical| {
                    if is_canonical {
                        -log_canonical_total
                    } else {
                        Float::NEG_INFINITY
                    }
                })
                .collect::<Box<[_]>>()
        } else {
            // Caller-provided logits are masked to canonical support and normalized
            // over the surviving follower set.
            let follower_logits =
                follower_logits.expect("non-zero-order predictions must provide follower_logits");
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
                        Float::NEG_INFINITY
                    }
                })
                .collect::<Box<[_]>>();
            let normalizer = masked_follower_logits
                .iter()
                .copied()
                .fold(Float::NEG_INFINITY, logaddexp);
            assert!(
                normalizer.is_finite(),
                "prediction logits must assign finite total mass"
            );
            masked_follower_logits
                .iter()
                .map(|&logit| {
                    if logit.is_finite() {
                        logit - normalizer
                    } else {
                        Float::NEG_INFINITY
                    }
                })
                .collect::<Box<[_]>>()
        };

        let follower_prob_for_prefix = (0..NUM_PREFIXES)
            .map(|prefix_lex_index| {
                let prefix_lex_index = PrefixLexIndex::from_usize(prefix_lex_index);
                let (start, stop) = tokenizer.token_lex_range_for_prefix_index(prefix_lex_index);
                follower_probs[start.as_usize()..stop.as_usize()]
                    .iter()
                    .copied()
                    .fold(Float::NEG_INFINITY, logaddexp)
            })
            .collect::<Box<[_]>>();

        let canonical_followers = canonical_followers_array.into_iter().collect::<Box<[_]>>();

        Self {
            canonical_followers,
            canonical_follower_for_prefix,
            follower_probs,
            follower_prob_for_prefix,
        }
    }

    pub(crate) fn canonical_follower(&self, token_lex_index: TokenLexIndex) -> bool {
        self.canonical_followers[token_lex_index.as_usize()]
    }

    pub(crate) fn canonical_follower_for_prefix(&self, prefix_lex_index: PrefixLexIndex) -> bool {
        self.canonical_follower_for_prefix[prefix_lex_index.as_usize()]
    }

    pub(crate) fn follower_prob(&self, token_lex_index: TokenLexIndex) -> Float {
        self.follower_probs[token_lex_index.as_usize()]
    }

    pub(crate) fn follower_prob_for_prefix(&self, prefix_lex_index: PrefixLexIndex) -> Float {
        self.follower_prob_for_prefix[prefix_lex_index.as_usize()]
    }
}