use std::collections::HashMap;

use crate::safe_float::{Float, into_f32};
use crate::symbol::Symbol;
use crate::trie::core::{XBayes, RecalcType, RecalcResult};
use crate::trie::l_update::{merge_xl_pair, set_leaf_indicators, XLUpdate, XLUpdateEntry};
use crate::trie::prediction::XPrediction;
use crate::rolling_hash as rh;
use crate::trie::TokenLexIndex;
use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct BayesianSession {
    pub(crate) trie: XBayes,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
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
            #[serde(alias = "l")]
            likelihood: f32,
            /// Omitted in wire JSON: taken as the last character of the map key (`a`–`z`, `_`, `^`).
            #[serde(default)]
            symbol: Option<Symbol>,
        }
        let l_by_string =
            serde_json::from_str::<HashMap<String, NHash>>(&likelihood_json).unwrap();
        let mut new_l_update = l_by_string.iter()
            .map(|(s, nhash)| (rh::hash_string(&s), 
                XLUpdateEntry {
                    likelihood: Float::from(nhash.likelihood),
                    symbol: nhash.symbol.unwrap_or_else(|| {
                        let b = *s.as_bytes().last().unwrap();
                        Symbol::from_byte(b).unwrap()
                    }),
                    is_leaf: false,
                }))
            .collect::<XLUpdate>();
        set_leaf_indicators(&mut new_l_update);
        //
        self.trie.pending_likelihood =
            merge_xl_pair(&self.trie.pending_likelihood, &new_l_update);
    }

    #[cfg(feature = "tokentrie")]
    pub fn next_requested_prior(&mut self) -> String {
        let requested_prior = self.trie.next_requested_prior();
        #[derive(Serialize)]
        struct RequestedPrior {
            full_string: String,
            last_token_lexindex: TokenLexIndex,
        }
        let requested_prior = RequestedPrior {
            full_string: requested_prior.full_string,
            last_token_lexindex: requested_prior.last_token_lexindex,
        };
        serde_json::to_string(&requested_prior).unwrap()
    }

    pub fn receive_prior_update(&mut self, prior_json: String) {
        #[derive(Deserialize)]
        struct Payload {
            full_string: String,
            final_token_lexindex: TokenLexIndex,
            follower_logits: Vec<f32>
        }
        let payload= serde_json::from_str::<Payload>(&prior_json).unwrap();
        let new_prediction = XPrediction::create_prediction(
            false,
            payload.final_token_lexindex,
            Some(
                payload
                    .follower_logits
                    .into_iter()
                    .map(Float::from)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            &self.trie.tokenizer
        );
        // insert prediction into the registry
        // hash the full string
        let full_hash = rh::hash_string(&payload.full_string);
        self.trie.full_predictions.insert(
            full_hash,
            new_prediction
        );
        self.trie.pending_prior.insert(full_hash);
    }

    pub fn apply_updates(&mut self) {
        self.trie.recalc_to_frontier(RecalcType::Update);
    }

    pub fn expand_to_threshold(&mut self) -> String {
        if !self.trie.pending_prior.is_empty()
            || !self.trie.pending_likelihood.is_empty()
        {
            self.apply_updates();
        }

        let nodes_list = match self.trie.recalc_to_frontier(
            RecalcType::Expand { threshold: Float::from(super::TRIE_EXPANSION_THRESHOLD as f32) },
        ) {
            RecalcResult::Updated => {
                panic!("expand_to_threshold unexpectedly returned Updated after applying pending updates")
            }
            RecalcResult::Expanded { nodes_over_threshold } => nodes_over_threshold,
        };
        // node list is in topological order
        struct NString {
            string: String,
            z: Float,
            symbol: Symbol,
        }
        let mut snapshot_by_hash: rh::RHashMap<NString> = rh::RHashMap::default();
        for n_hash in nodes_list {
            let node = self.trie.nodes.get(&n_hash).unwrap();
            let symbol = node.symbol;
            let n = if n_hash == super::ROOT_HASH {
                let s = super::ROOT_STRING.to_string();
                let z = node.if_root_then_z;
                NString { string: s, z, symbol }
            } else {
                let p_hash = rh::pop_right(n_hash, node.symbol.to_byte());
                // p_hash is guaranteed to be in the snapshot_by_hash
                // because nodes_list is in topological order
                let p_string = snapshot_by_hash.get(&p_hash).unwrap().string.clone();
                let p_node = self.trie.nodes.get(&p_hash).unwrap();
                let s = p_string + &(node.symbol.to_byte() as char).to_string();
                let z = p_node.c_z[node.symbol.to_slot()];
                NString { string: s, z, symbol }
            };
            snapshot_by_hash.insert(n_hash, n);
        }
        // swap out the string for the hash as the primary key
        #[derive(Serialize)]
        struct NHash {
            z: f32,
            symbol: Symbol,
            hash: rh::Hash,
        }
        let snapshot_by_string = snapshot_by_hash.into_iter()
            .map(|(hash, n_string)| (n_string.string,
                NHash {
                    z: into_f32(n_string.z),
                    symbol: n_string.symbol,
                    hash
                }))
            .collect::<HashMap<String, NHash>>();
        let snapshot_json = serde_json::to_string(&snapshot_by_string).unwrap();
        snapshot_json
    }

    pub fn lexicographic_tokens_json(&self) -> String {
        serde_json::to_string(self.trie.tokenizer.tokens()).unwrap()
    }

    /// Print the trie to stderr (`tree`-style). `filter`: letters `a`–`z`, `_` (word boundary), `^` (start); empty shows all nodes (root is always shown when filtered).
    #[cfg(not(feature = "wasm"))]
    pub fn debug_eprint_trie(&self, filter: &str) {
        crate::trie::debug::eprint_trie(&self.trie, filter, None);
    }

    /// Print the trie to stderr, restricted to `hash_filter` after applying the symbol filter.
    #[cfg(not(feature = "wasm"))]
    pub fn debug_eprint_trie_hash_filter(&self, filter: &str, hash_filter: &rh::RHashSet) {
        crate::trie::debug::eprint_trie(&self.trie, filter, Some(hash_filter));
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

    #[cfg(feature = "tokentrie")]
    #[pyo3(name = "next_requested_prior")]
    fn py_next_requested_prior(&mut self) -> String {
        BayesianSession::next_requested_prior(self)
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