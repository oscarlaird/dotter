use crate::bpe::{NUM_PREFIXES, NUM_TOKENS, PrefixLexIndex, TinyLlamaWordTokenizer, TokenLexIndex};
use crate::safe_float::Float;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use super::{logaddexp};

#[derive(Clone, Copy, Debug, Default)]
pub struct ZeroOrderPredictionTimingSnapshot {
    pub call_count: u64,
    pub total_ns: u64,
    pub count_true_tokens_by_prefix_ns: u64,
    pub canonical_follower_for_prefix_ns: u64,
    pub follower_probs_ns: u64,
    pub follower_prob_for_prefix_ns: u64,
    pub canonical_followers_box_ns: u64,
}

#[cfg(not(target_arch = "wasm32"))]
static ZERO_ORDER_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(not(target_arch = "wasm32"))]
static ZERO_ORDER_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(not(target_arch = "wasm32"))]
static ZERO_ORDER_COUNT_TRUE_TOKENS_BY_PREFIX_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(not(target_arch = "wasm32"))]
static ZERO_ORDER_CANONICAL_FOLLOWER_FOR_PREFIX_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(not(target_arch = "wasm32"))]
static ZERO_ORDER_FOLLOWER_PROBS_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(not(target_arch = "wasm32"))]
static ZERO_ORDER_FOLLOWER_PROB_FOR_PREFIX_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(not(target_arch = "wasm32"))]
static ZERO_ORDER_CANONICAL_FOLLOWERS_BOX_NS: AtomicU64 = AtomicU64::new(0);

pub fn reset_zero_order_prediction_timing() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ZERO_ORDER_CALL_COUNT.store(0, Ordering::Relaxed);
        ZERO_ORDER_TOTAL_NS.store(0, Ordering::Relaxed);
        ZERO_ORDER_COUNT_TRUE_TOKENS_BY_PREFIX_NS.store(0, Ordering::Relaxed);
        ZERO_ORDER_CANONICAL_FOLLOWER_FOR_PREFIX_NS.store(0, Ordering::Relaxed);
        ZERO_ORDER_FOLLOWER_PROBS_NS.store(0, Ordering::Relaxed);
        ZERO_ORDER_FOLLOWER_PROB_FOR_PREFIX_NS.store(0, Ordering::Relaxed);
        ZERO_ORDER_CANONICAL_FOLLOWERS_BOX_NS.store(0, Ordering::Relaxed);
    }
}

pub fn zero_order_prediction_timing_snapshot() -> ZeroOrderPredictionTimingSnapshot {
    #[cfg(not(target_arch = "wasm32"))]
    {
        return ZeroOrderPredictionTimingSnapshot {
            call_count: ZERO_ORDER_CALL_COUNT.load(Ordering::Relaxed),
            total_ns: ZERO_ORDER_TOTAL_NS.load(Ordering::Relaxed),
            count_true_tokens_by_prefix_ns: ZERO_ORDER_COUNT_TRUE_TOKENS_BY_PREFIX_NS
                .load(Ordering::Relaxed),
            canonical_follower_for_prefix_ns: ZERO_ORDER_CANONICAL_FOLLOWER_FOR_PREFIX_NS
                .load(Ordering::Relaxed),
            follower_probs_ns: ZERO_ORDER_FOLLOWER_PROBS_NS.load(Ordering::Relaxed),
            follower_prob_for_prefix_ns: ZERO_ORDER_FOLLOWER_PROB_FOR_PREFIX_NS
                .load(Ordering::Relaxed),
            canonical_followers_box_ns: ZERO_ORDER_CANONICAL_FOLLOWERS_BOX_NS
                .load(Ordering::Relaxed),
        };
    }

    #[cfg(target_arch = "wasm32")]
    {
        ZeroOrderPredictionTimingSnapshot::default()
    }
}

pub struct XPrediction {
    canonical_followers: Box<[bool]>,
    canonical_follower_for_prefix: Box<[bool]>,
    follower_probs: Box<[Float]>,
    follower_prob_for_prefix: Box<[Float]>,
}

impl XPrediction {
    pub fn create_prediction(
        is_zero_order: bool,
        final_token_lexindex: TokenLexIndex,
        follower_logits: Option<Box<[Float]>>,
        tokenizer: &TinyLlamaWordTokenizer,
    ) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let total_started = is_zero_order.then(Instant::now);
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
        #[cfg(not(target_arch = "wasm32"))]
        let count_started = is_zero_order.then(Instant::now);
        let canonical_counts_by_prefix =
            tokenizer.count_true_tokens_by_prefix::<NUM_PREFIXES>(&canonical_followers_array);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(started) = count_started {
            ZERO_ORDER_COUNT_TRUE_TOKENS_BY_PREFIX_NS
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let canonical_total = canonical_counts_by_prefix[tokenizer
            .prefix_lex_index_for_prefix_hash(&0u64)
            .as_usize()]; // hash("")=0

        assert!(canonical_total != 0, "canonical_total must not be zero");
        let log_canonical_total = Float::from(canonical_total as f32).ln();

        #[cfg(not(target_arch = "wasm32"))]
        let canonical_prefix_started = is_zero_order.then(Instant::now);
        let canonical_follower_for_prefix = canonical_counts_by_prefix
            .iter()
            .map(|&count| count != 0)
            .collect::<Box<[_]>>();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(started) = canonical_prefix_started {
            ZERO_ORDER_CANONICAL_FOLLOWER_FOR_PREFIX_NS
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        #[cfg(not(target_arch = "wasm32"))]
        let follower_probs_started = is_zero_order.then(Instant::now);
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
            debug_assert_eq!(
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
            debug_assert!(
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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(started) = follower_probs_started {
            ZERO_ORDER_FOLLOWER_PROBS_NS
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        #[cfg(not(target_arch = "wasm32"))]
        let follower_prob_for_prefix_started = is_zero_order.then(Instant::now);
        let follower_prob_for_prefix = if is_zero_order {
            // In the zero-order case every canonical follower has the same probability mass
            // and every non-canonical follower is impossible, so a prefix's total mass depends
            // only on how many canonical followers it contains.
            canonical_counts_by_prefix
                .iter()
                .map(|&count| {
                    if count == 0 {
                        Float::NEG_INFINITY
                    } else {
                        Float::from(count as f32).ln() - log_canonical_total
                    }
                })
                .collect::<Box<[_]>>()
        } else {
            (0..NUM_PREFIXES)
                .map(|prefix_lex_index| {
                    let prefix_lex_index = PrefixLexIndex::from_usize(prefix_lex_index);
                    let (start, stop) = tokenizer.token_lex_range_for_prefix_index(prefix_lex_index);
                    follower_probs[start.as_usize()..stop.as_usize()]
                        .iter()
                        .copied()
                        .fold(Float::NEG_INFINITY, logaddexp)
                })
                .collect::<Box<[_]>>()
        };
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(started) = follower_prob_for_prefix_started {
            ZERO_ORDER_FOLLOWER_PROB_FOR_PREFIX_NS
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        #[cfg(not(target_arch = "wasm32"))]
        let canonical_followers_box_started = is_zero_order.then(Instant::now);
        let canonical_followers = canonical_followers_array.into_iter().collect::<Box<[_]>>();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(started) = canonical_followers_box_started {
            ZERO_ORDER_CANONICAL_FOLLOWERS_BOX_NS
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(started) = total_started {
            ZERO_ORDER_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
            ZERO_ORDER_TOTAL_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

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