use crate::bpe;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use super::{Prediction, PredictionOrder, Trie, TrieSnapshot};

fn browser_tokenizer() -> crate::bpe::TinyLlamaWordTokenizer {
    crate::bpe::TinyLlamaWordTokenizer::from_tokenizer_json_str(bpe::TOKENIZER_JSON_STR)
}

fn browser_trie() -> Trie {
    Trie::new(browser_tokenizer())
}

#[cfg_attr(feature = "python", pyclass)]
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
        crate::trie_debug!(
            "[bayesian] apply_likelihood_update json_len={}",
            snapshot_json.len()
        );
        let snapshot: TrieSnapshot = serde_json::from_str(&snapshot_json)
            .expect("snapshot_json should deserialize to TrieSnapshot");
        crate::trie_debug!(
            "[bayesian] snapshot nodes={} root={}",
            snapshot.nodes.len(),
            snapshot.root
        );
        self.trie.apply_likelihood_update(&snapshot);
        crate::trie_debug!("[bayesian] apply_likelihood_update done");
    }

    pub fn apply_prior_update(
        &mut self,
        final_token: Option<String>,
        full_string: String,
        follower_logits: Vec<f64>,
        stop_logit: f64,
    ) {
        crate::trie_debug!(
            "[bayesian] apply_prior_update final_token={:?} full_string_len={} logits_len={} stop_logit={}",
            final_token.as_deref(),
            full_string.len(),
            follower_logits.len(),
            stop_logit
        );
        let order = PredictionOrder::FullOrder(final_token, full_string.clone());
        let prediction = Prediction::create_prediction(
            order,
            Some(follower_logits.into_boxed_slice()),
            Some(stop_logit),
            &self.trie.tokenizer,
        );
        self.trie.apply_prior_update(full_string, prediction);
        crate::trie_debug!("[bayesian] apply_prior_update done");
    }

    // expand and snapshot
    pub fn snapshot_json(&mut self) -> String {
        let snapshot = self.trie.snapshot_trie(self.threshold);
        serde_json::to_string(&snapshot).expect("TrieSnapshot should serialize to JSON")
    }

    pub fn snapshot_json_with_threshold(&mut self, threshold: f64) -> String {
        let snapshot = self.trie.snapshot_trie(threshold);
        serde_json::to_string(&snapshot).expect("TrieSnapshot should serialize to JSON")
    }

    pub fn lexicographic_tokens_json(&self) -> String {
        serde_json::to_string(self.trie.tokenizer.tokens())
            .expect("token list should serialize to JSON")
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl BayesianSession {
    #[new]
    fn py_new(threshold: f64, max_expand_budget: usize) -> Self {
        Self::new(threshold, max_expand_budget)
    }

    #[pyo3(name = "reset")]
    fn py_reset(&mut self) {
        self.reset();
    }

    #[pyo3(name = "apply_likelihood_update")]
    fn py_apply_likelihood_update(&mut self, snapshot_json: String) {
        self.apply_likelihood_update(snapshot_json);
    }

    #[pyo3(name = "apply_prior_update")]
    fn py_apply_prior_update(
        &mut self,
        final_token: Option<String>,
        full_string: String,
        follower_logits: Vec<f64>,
        stop_logit: f64,
    ) {
        self.apply_prior_update(final_token, full_string, follower_logits, stop_logit);
    }

    #[pyo3(name = "snapshot_json")]
    fn py_snapshot_json(&mut self) -> String {
        self.snapshot_json()
    }

    #[pyo3(name = "snapshot_json_with_threshold")]
    fn py_snapshot_json_with_threshold(&mut self, threshold: f64) -> String {
        self.snapshot_json_with_threshold(threshold)
    }

    #[pyo3(name = "lexicographic_tokens_json")]
    fn py_lexicographic_tokens_json(&self) -> String {
        self.lexicographic_tokens_json()
    }
}
