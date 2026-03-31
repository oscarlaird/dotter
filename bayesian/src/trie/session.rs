use crate::bpe;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use super::{Prediction, PredictionOrder, Trie, TrieSnapshot};

fn browser_tokenizer() -> crate::bpe::TinyLlamaWordTokenizer {
    crate::bpe::TinyLlamaWordTokenizer::from_tokenizer_json_str(bpe::TOKENIZER_JSON_STR)
}

fn browser_trie() -> Trie {
    Trie::new(browser_tokenizer())
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct BayesianSession {
    trie: Trie,
    threshold: f64,
    max_expand_budget: usize,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl BayesianSession {
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(threshold: f64, max_expand_budget: usize) -> Self {
        Self {
            trie: browser_trie(),
            threshold,
            max_expand_budget,
        }
    }

    pub fn reset(&mut self) {
        self.trie = browser_trie();
    }

    // three kinds of descent:
    // - likelihood update
    // - prior update
    // - expansion
    pub fn apply_likelihood_update(&mut self, snapshot_json: String) {
        let snapshot: TrieSnapshot = serde_json::from_str(&snapshot_json)
            .expect("snapshot_json should deserialize to TrieSnapshot");
        self.trie.apply_likelihood_update(&snapshot);
    }

    pub fn apply_prior_update(
        &mut self,
        final_token: Option<String>,
        full_string: String,
        follower_logits: Vec<f64>,
        stop_logit: f64,
    ) {
        let order = PredictionOrder::FullOrder(final_token, full_string.clone());
        let prediction = Prediction::create_prediction(
            order,
            Some(follower_logits.into_boxed_slice()),
            Some(stop_logit),
            &self.trie.tokenizer,
        );
        self.trie.apply_prior_update(full_string, prediction);
    }

    // expand and snapshot
    pub fn snapshot_json(&mut self, threshold: f64) -> String {
        let snapshot = self.trie.snapshot_trie(threshold);
        serde_json::to_string(&snapshot).expect("TrieSnapshot should serialize to JSON")
    }
}
