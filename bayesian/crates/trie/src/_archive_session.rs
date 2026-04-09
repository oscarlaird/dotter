use crate::bpe;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use serde::Deserialize;

use super::{Prediction, PredictionOrder, Trie, TrieSnapshot};

/// JSON shape of websocket `prior_update` `content` (and payload passed to [`BayesianSession::apply_prior_update`]).
#[derive(Debug, Deserialize)]
struct PriorUpdatePayload {
    final_token: Option<String>,
    full_string: String,
    follower_logits: Vec<f64>,
}

fn browser_tokenizer() -> crate::bpe::TinyLlamaWordTokenizer {
    crate::bpe::TinyLlamaWordTokenizer::from_tokenizer_json_str(bpe::TOKENIZER_JSON_STR)
}

fn browser_trie() -> Trie {
    Trie::new(browser_tokenizer())
}

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct BayesianSession {
    pub(crate) trie: Trie,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl BayesianSession {
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self { trie: browser_trie() }
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

    pub fn apply_prior_update(&mut self, prior_json: String) {
        crate::trie_debug!(
            "[bayesian] apply_prior_update json_len={}",
            prior_json.len()
        );
        let payload: PriorUpdatePayload = serde_json::from_str(&prior_json)
            .expect("prior_json should deserialize to prior update content shape");
        crate::trie_debug!(
            "[bayesian] apply_prior_update final_token={:?} full_string_len={} logits_len={}",
            payload.final_token.as_deref(),
            payload.full_string.len(),
            payload.follower_logits.len()
        );
        let order = PredictionOrder::FullOrder(
            payload.final_token,
            payload.full_string.clone(),
        );
        let prediction = Prediction::create_prediction(
            order,
            Some(payload.follower_logits.into_boxed_slice()),
            &self.trie.tokenizer,
        );
        self.trie.apply_prior_update(payload.full_string, prediction);
        crate::trie_debug!("[bayesian] apply_prior_update done");
    }

    // expand and snapshot
    pub fn snapshot_json(&mut self) -> String {
        let snapshot = self.trie.snapshot_trie();
        serde_json::to_string(&snapshot).expect("TrieSnapshot should serialize to JSON")
    }

    pub fn lexicographic_tokens_json(&self) -> String {
        serde_json::to_string(self.trie.tokenizer.tokens())
            .expect("token list should serialize to JSON")
    }
}

#[cfg(test)]
impl BayesianSession {
    pub(crate) fn trie_snapshot_at_current(&self) -> TrieSnapshot {
        self.trie.snapshot_at_current()
    }

    pub(crate) fn expand_trie(&mut self) {
        self.trie.expand_trie();
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl BayesianSession {
    #[new]
    fn py_new() -> Self {
        Self::new()
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
    fn py_apply_prior_update(&mut self, prior_json: String) {
        self.apply_prior_update(prior_json);
    }

    #[pyo3(name = "snapshot_json")]
    fn py_snapshot_json(&mut self) -> String {
        self.snapshot_json()
    }

    #[pyo3(name = "lexicographic_tokens_json")]
    fn py_lexicographic_tokens_json(&self) -> String {
        self.lexicographic_tokens_json()
    }
}
