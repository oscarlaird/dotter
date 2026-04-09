use std::collections::HashMap;

use crate::bpe::TokenLexIndex;
use crate::safe_float::{Float, into_f32};
use crate::symbol::{Symbol, RADIX};
use crate::trie::core::{XBayes, RecalcType, RecalcResult};
use crate::trie::l_update::{merge_xl_pair, set_leaf_indicators, XLUpdate, XLUpdateEntry};
use crate::trie::prediction::XPrediction;
use crate::rolling_hash as rh;
use crate::rolling_hash::Hash;
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

    pub fn expansion_threshold(&self) -> f32 {
        super::TRIE_EXPANSION_THRESHOLD as f32
    }

    pub fn receive_likelihood_update(&mut self, likelihood_json: String) {
        #[derive(Deserialize)]
        struct NHash {
            // by default, serde will ignore extra fields
            #[serde(alias = "l")]
            likelihood: f32,
            /// Omitted in wire JSON: taken as the last character of the map key (`a`–`z`, `_`, `$`, `^`).
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
        assert!(self.trie.pending_prior.is_empty() && self.trie.pending_likelihood.len() == 1, "Tried to expand with unprocessed updates");
        let nodes_list = match self.trie.recalc_to_frontier(
            RecalcType::Expand { threshold: Float::from(super::TRIE_EXPANSION_THRESHOLD as f32) },
        ) {
            RecalcResult::Updated => {
                panic!("expand_to_threshold unexpectedly returned Updated after applying pending updates")
            }
            RecalcResult::Expanded { nodes_over_threshold } => nodes_over_threshold,
        };
        // add all siblings to that the frontend knows how to space things
        let nodes_set = nodes_list.into_iter().collect::<rh::RHashSet>();
        // node list is in topological order
        struct NString {
            string: String,
            z: Float,
            p: Option<Float>,
            tp: Option<Float>,
            tp0: Option<Float>,
            a_tl0: Option<Float>,
            symbol: Symbol,
            upper_siblings_inclusive_cum_z: Option<Float>,
        }
        let mut snapshot_by_hash: rh::RHashMap<NString> = rh::RHashMap::default();
        snapshot_by_hash.insert(super::ROOT_HASH, NString {
            string: String::from(super::ROOT_STRING),
            z: self.trie.nodes.get(&super::ROOT_HASH).unwrap().if_root_then_z,
            p: None,
            tp: None,
            tp0: None,
            a_tl0: None,
            symbol: Symbol::Start,
            upper_siblings_inclusive_cum_z: None,
        });

        struct Frame {
            n_hash: Hash,
            n_symbol: Symbol,
            n_depth: u16,
        }
        // a frame is a node that *may* have children that should be added to the snapshot
        // by the time we see a frame, it should already be in the snapshot
        // the frame list is a visit list
        let mut full_string = String::new(); // character stack
        let mut frames: Vec<Frame> = vec![Frame { n_hash: super::ROOT_HASH, n_symbol: Symbol::Start, n_depth: 0 }];
        while let Some(Frame { n_hash, n_symbol, n_depth }) = frames.pop() {
            full_string.truncate(n_depth as usize);
            full_string.push(n_symbol.to_byte() as char);
            debug_assert!(snapshot_by_hash.contains_key(&n_hash), "frame node not in snapshot");
            let c_present = (0..RADIX).map(|slot| {
                let c_byte = Symbol::slot_to_byte(slot);
                let c_hash = rh::append_right(n_hash, c_byte);
                nodes_set.contains(&c_hash)
            }).collect::<Vec<_>>();
            let is_interior = c_present.iter().cloned().any(|p| p);
            if !is_interior {
                continue;
            }
            debug_assert!(self.trie.nodes.contains_key(&n_hash), "frame node not in trie");
            let n_node = self.trie.nodes.get(&n_hash).unwrap();
            let c_z = n_node.c_z;
            // Compute the partial sums of c_z using logaddexp, i.e., log-sum-exp over upper siblings
            let c_upper_siblings_inclusive_cum_z: [_; RADIX] = {
                let mut accum = Float::NEG_INFINITY;
                let mut arr = [Float::NEG_INFINITY; RADIX];
                for i in 0..RADIX {
                    accum = crate::trie::logaddexp(accum, c_z[i]);
                    arr[i] = accum;
                }
                arr
            };
            for slot in 0..RADIX {
                if !c_present[slot] {
                    continue;
                }
                let c_byte = Symbol::slot_to_byte(slot);
                let c_symbol = Symbol::from_slot(slot);
                let c_hash = rh::append_right(n_hash, c_byte);
                let mut c_string = full_string.clone();
                c_string.push(c_symbol.to_byte() as char);
                let (z, p, tp, tp0, a_tl0) = n_node.edge_snapshot_fields(c_symbol);
                snapshot_by_hash.insert(c_hash, NString {
                    string: c_string,
                    z,
                    p: Some(p),
                    tp: Some(tp),
                    tp0: Some(tp0),
                    a_tl0: Some(a_tl0),
                    symbol: c_symbol,
                    upper_siblings_inclusive_cum_z: Some(c_upper_siblings_inclusive_cum_z[slot]),
                });
                frames.push(Frame {
                    n_hash: c_hash,
                    n_symbol: c_symbol,
                    n_depth: n_depth + 1,
                })
            }
        }
        // swap out the string for the hash as the primary key
        #[derive(Serialize)]
        struct NHash {
            z: f32,
            p: Option<f32>,
            tp: Option<f32>,
            tp0: Option<f32>,
            a_tl0: Option<f32>,
            symbol: Symbol,
            upper_siblings_inclusive_cum_z: Option<f32>,
            hash: rh::Hash,
        }
        let snapshot_by_string = snapshot_by_hash.into_iter()
            .map(|(hash, n_string)| (n_string.string,
                NHash {
                    z: into_f32(n_string.z),
                    p: n_string.p.map(into_f32),
                    tp: n_string.tp.map(into_f32),
                    tp0: n_string.tp0.map(into_f32),
                    a_tl0: n_string.a_tl0.map(into_f32),
                    symbol: n_string.symbol,
                    upper_siblings_inclusive_cum_z: n_string.upper_siblings_inclusive_cum_z.map(into_f32),
                    hash
                }))
            .collect::<HashMap<String, NHash>>();
        let snapshot_json = serde_json::to_string(&snapshot_by_string).unwrap();
        snapshot_json
    }

    pub fn lexicographic_tokens_json(&self) -> String {
        serde_json::to_string(self.trie.tokenizer.tokens()).unwrap()
    }

    /// Print the trie to stderr (`tree`-style). `filter`: letters `a`–`z`, `_` (word boundary),
    /// `$` (stop), `^` (start); empty shows all nodes (root is always shown when filtered).
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