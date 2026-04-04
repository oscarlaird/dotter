use std::collections::HashMap;

use crate::trie::core::{XBayes, RecalcType, RecalcResult};
use crate::trie::l_update::LUpdate;
use crate::trie::prediction::XPrediction;
use crate::rolling_hash as rh;
use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass)]
pub struct BayesianSession {
    pub(crate) trie: XBayes,
}

const THRESHOLD: f32 = -5.2983174; // precomputed value of (1.0/200.0).ln()

impl BayesianSession {
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self { trie: XBayes::new() }
    }

    pub fn reset(&mut self) {
        self.trie = XBayes::new();
    }

    pub fn receive_likelihood_update(&mut self, likelihood_json: String) {
        #[derive(Deserialize)]
        struct NHash {
            // by default, serde will ignore extra fields
            hash: rh::Hash,
            l: f32,
        }
        let n_by_string =
            serde_json::from_str::<HashMap<String, NHash>>(&likelihood_json).unwrap();
        let l_by_hash = n_by_string.iter()
            .map(|(_string, nhash)| (nhash.hash, nhash.l))
            .collect::<rh::RHashMap<f32>>();
        let mut new_l_update = LUpdate {
            likelihoods: l_by_hash,
            cpc_form: false,
        };
        new_l_update.to_cpc_form();
        //
        self.trie.pending_likelihood = self.trie.pending_likelihood.merge(&new_l_update);
    }

    pub fn receive_prior_update(&mut self, prior_json: String) {
        #[derive(Deserialize)]
        struct Payload {
            full_string: String,
            final_token_hash: rh::Hash,
            follower_logits: Vec<f32>
        }
        let payload= serde_json::from_str::<Payload>(&prior_json).unwrap();
        // hash the full string
        let mut full_hash = 0u64;
        for byte in payload.full_string.bytes() {
            full_hash = rh::append_right(full_hash, byte)
        }
        //
        let new_prediction = XPrediction::create_prediction(
            false,
            payload.final_token_hash,
            Some(payload.follower_logits.into_boxed_slice()),
            &self.trie.tokenizer
        );
        // insert prediction into the registry
        self.trie.full_predictions.insert(
            full_hash,
            new_prediction
        );
        self.trie.pending_prior.deref_mut().insert(full_hash);
    }

    pub fn apply_updates(&mut self) {
        self.trie.recalc_to_frontier(RecalcType::Update);
    }

    pub fn expand_to_threshold(&mut self) -> String {
        let nodes_list = match self.trie.recalc_to_frontier(
            RecalcType::Expand { threshold: THRESHOLD },
        ) {
            RecalcResult::Updated => unreachable!(),
            RecalcResult::Expanded { nodes_over_threshold } => nodes_over_threshold,
        };
        // node list is in topological order
        struct NString {
            string: String,
            z: f32,
        }
        let mut snapshot_by_hash: rh::RHashMap<NString> = rh::RHashMap::default();
        for n_hash in nodes_list {
            let node = self.trie.nodes.get(&n_hash).unwrap();
            let n = if n_hash == super::ROOT_HASH {
                let s = super::ROOT_STRING.to_string();
                let z = node.if_root_then_z;
                NString { string: s, z }
            } else {
                let p_hash = rh::pop_right(n_hash, node.symbol.to_byte());
                // p_hash is guaranteed to be in the snapshot_by_hash
                // because nodes_list is in topological order
                let p_string = snapshot_by_hash.get(&p_hash).unwrap().string.clone();
                let p_node = self.trie.nodes.get(&p_hash).unwrap();
                let s = p_string + &node.symbol.to_byte().to_string();
                let z = p_node.c_z[node.symbol.to_slot()];
                NString { string: s, z }
            };
            snapshot_by_hash.insert(n_hash, n);
        }
        // swap out the string for the hash as the primary key
        #[derive(Serialize)]
        struct NHash {
            z: f32,
            hash: rh::Hash,
        }
        let snapshot_by_string = snapshot_by_hash.into_iter()
            .map(|(hash, n_string)| (n_string.string, NHash { z: n_string.z, hash }))
            .collect::<HashMap<String, NHash>>();
        let snapshot_json = serde_json::to_string(&snapshot_by_string).unwrap();
        snapshot_json
    }

    pub fn lexicographic_tokens_json(&self) -> String {
        serde_json::to_string(self.trie.tokenizer.tokens()).unwrap()
    }

    /// Print the trie to stderr (`tree`-style). `filter`: letters `a`–`z`, `_` (word boundary), `^` (start); empty shows all nodes (root is always shown when filtered).
    pub fn debug_eprint_trie(&self, filter: &str) {
        crate::trie::debug::eprint_trie(&self.trie, filter);
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl BayesianSession {
    #[new]
    fn py_new() -> Self {
        BayesianSession::new()
    }

    #[pyo3(name = "reset")]
    fn py_reset(&mut self) {
        BayesianSession::reset(self);
    }

    #[pyo3(name = "receive_likelihood_update")]
    fn py_receive_likelihood_update(&mut self, likelihood_json: String) {
        BayesianSession::receive_likelihood_update(self, likelihood_json);
    }

    #[pyo3(name = "receive_prior_update")]
    fn py_receive_prior_update(&mut self, prior_json: String) {
        BayesianSession::receive_prior_update(self, prior_json);
    }

    #[pyo3(name = "apply_updates")]
    fn py_apply_updates(&mut self) {
        BayesianSession::apply_updates(self);
    }

    #[pyo3(name = "expand_to_threshold")]
    fn py_expand_to_threshold(&mut self) -> String {
        BayesianSession::expand_to_threshold(self)
    }

    #[pyo3(name = "lexicographic_tokens_json")]
    fn py_lexicographic_tokens_json(&self) -> String {
        BayesianSession::lexicographic_tokens_json(self)
    }
}