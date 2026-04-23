use std::collections::HashMap;
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use bpe::TokenLexIndex;
use calibration::VariationalParams;
use render_utils::{ExpandedSnapshot, snapshot_by_string};
use rolling_hash as rh;
use rolling_hash::Hash;
use serde::{Deserialize, Deserializer, Serialize};
use trie::core::{RecalcResult, RecalcType, XBayes};
use trie::l_update::{XLUpdate, XLUpdateEntry, merge_xl_pair, set_leaf_indicators};
use trie::prediction::XPrediction;
use trie::safe_float::Float;
use trie::symbol::XSymbol;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
type TimingStart = f64;
#[cfg(not(target_arch = "wasm32"))]
type TimingStart = Instant;

#[cfg(target_arch = "wasm32")]
fn timing_start() -> TimingStart {
    web_sys::window().unwrap().performance().unwrap().now()
}

#[cfg(not(target_arch = "wasm32"))]
fn timing_start() -> TimingStart {
    Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn elapsed_ms_since(start: TimingStart) -> f64 {
    web_sys::window().unwrap().performance().unwrap().now() - start
}

#[cfg(not(target_arch = "wasm32"))]
fn elapsed_ms_since(start: TimingStart) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn deserialize_optional_symbol<'de, D>(deserializer: D) -> Result<Option<XSymbol>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Option::<String>::deserialize(deserializer)?;
    raw.map(|s| {
        let mut it = s.chars();
        let c = it
            .next()
            .ok_or_else(|| serde::de::Error::custom("empty symbol string"))?;
        if it.next().is_some() {
            return Err(serde::de::Error::custom(
                "symbol string must be exactly one character",
            ));
        }
        let code = u32::from(c);
        if code > 127 {
            return Err(serde::de::Error::custom("non-ASCII symbol"));
        }
        Ok(code as XSymbol)
    })
    .transpose()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionEvent {
    pub kind: String,
    duration_ms: f64,
    json_payload_ix: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionObservability {
    pub json_payloads: Vec<String>,
    pub event_log: Vec<SessionEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LikelihoodNodeInput {
    #[serde(alias = "l")]
    pub likelihood: f32,
    /// Omitted in wire JSON: taken as the last character of the map key (`a`–`z`, `S`, `Z`, `A`).
    #[serde(default, deserialize_with = "deserialize_optional_symbol")]
    pub symbol: Option<XSymbol>,
    pub phase: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LikelihoodUpdatePayload {
    pub period: f64,
    pub y: f64,
    pub nodes: HashMap<String, LikelihoodNodeInput>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CalibrationSample {
    pub x: f64,
    pub period: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RecalibrationResult {
    pub prior_params: VariationalParams,
    pub used_likelihood_updates: usize,
    pub recent_pairs: Vec<CalibrationSample>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PriorUpdatePayload {
    pub full_string: String,
    pub final_token_lexindex: TokenLexIndex,
    pub follower_logits: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RequestedPrior {
    pub full_string: String,
    pub last_token_lexindex: TokenLexIndex,
}

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct BayesianSession {
    pub(crate) trie: XBayes,
    observability: Mutex<SessionObservability>,
    likelihood_history: Vec<LikelihoodUpdatePayload>,
    current_prior: VariationalParams,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl BayesianSession {
    fn symbol_for_payload_node(node_key: &str, symbol: Option<XSymbol>) -> XSymbol {
        symbol.unwrap_or_else(|| {
            let b = *node_key
                .as_bytes()
                .last()
                .expect("likelihood update contained an empty node string");
            b as XSymbol
        })
    }

    fn certain_prefix_nodes(&mut self) -> Vec<(Hash, XSymbol)> {
        let threshold = (-0.0100503f32).into(); // ln(0.99)
        match self
            .trie
            .recalc_to_frontier(trie::core::RecalcType::Expand { threshold })
        {
            trie::core::RecalcResult::Expanded {
                nodes_over_threshold,
            } => nodes_over_threshold,
            _ => panic!("Expected Expanded"),
        }
    }

    fn push_json_payload(&self, json: String) -> usize {
        let mut observability = self.observability.lock().unwrap();
        let ix = observability.json_payloads.len();
        observability.json_payloads.push(json);
        ix
    }

    fn record_event(&self, kind: &'static str, start: TimingStart, json_payload_ix: Option<usize>) {
        self.observability
            .lock()
            .unwrap()
            .event_log
            .push(SessionEvent {
                kind: kind.to_string(),
                duration_ms: elapsed_ms_since(start),
                json_payload_ix,
            });
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        let start = timing_start();
        let session = Self {
            trie: XBayes::new(),
            observability: Mutex::new(SessionObservability::default()),
            likelihood_history: Vec::new(),
            current_prior: VariationalParams::default_calibration(),
        };
        session.record_event("new", start, None);
        session
    }

    pub fn reset(&mut self) {
        let start = timing_start();
        self.trie = XBayes::new();
        self.likelihood_history.clear();
        self.current_prior = VariationalParams::default_calibration();
        let mut observability = self.observability.lock().unwrap();
        observability.json_payloads.clear();
        observability.event_log.clear();
        observability.event_log.push(SessionEvent {
            kind: "reset".to_string(),
            duration_ms: elapsed_ms_since(start),
            json_payload_ix: None,
        });
    }

    pub fn expansion_threshold(&self) -> f32 {
        let start = timing_start();
        let threshold = trie::TRIE_EXPANSION_THRESHOLD as f32;
        self.record_event("expansion_threshold", start, None);
        threshold
    }

    pub fn receive_likelihood_update_typed(&mut self, payload: LikelihoodUpdatePayload) {
        let start = timing_start();
        assert!(
            payload.period.is_finite() && payload.period > 0.0,
            "likelihood period must be finite and positive"
        );
        assert!(payload.y.is_finite(), "likelihood y must be finite");
        assert!(
            !payload.nodes.is_empty(),
            "likelihood update contained no nodes"
        );
        for (s, nhash) in &payload.nodes {
            assert!(
                !s.is_empty(),
                "likelihood update contained an empty node string"
            );
            assert!(
                nhash.likelihood.is_finite(),
                "likelihood for node {:?} must be finite",
                s
            );
            assert!(
                nhash.phase.is_finite(),
                "phase for node {:?} must be finite",
                s
            );
            assert!(
                nhash.phase >= 0.0 && nhash.phase < payload.period,
                "phase for node {:?} must lie in [0, period); got {} with period {}",
                s,
                nhash.phase,
                payload.period,
            );
        }

        self.likelihood_history.push(payload.clone());
        let json_payload_ix = self.push_json_payload(
            serde_json::to_string(&payload).expect("likelihood serialization failed"),
        );

        let mut new_l_update = payload
            .nodes
            .iter()
            .map(|(s, nhash)| {
                (
                    rh::hash_string(&s),
                    XLUpdateEntry {
                        likelihood: Float::from(nhash.likelihood),
                        symbol: Self::symbol_for_payload_node(s, nhash.symbol),
                        is_leaf: false,
                    },
                )
            })
            .collect::<XLUpdate>();
        set_leaf_indicators(&mut new_l_update);
        //
        self.trie.pending_likelihood = merge_xl_pair(&self.trie.pending_likelihood, &new_l_update);
        self.record_event("receive_likelihood_update", start, Some(json_payload_ix));
    }

    pub fn receive_likelihood_update(&mut self, likelihood_json: String) {
        let payload = serde_json::from_str::<LikelihoodUpdatePayload>(&likelihood_json).unwrap();
        self.receive_likelihood_update_typed(payload);
    }

    pub fn recalibrate_typed(
        &mut self,
        initial_prior: VariationalParams,
        use_cross_entropy_discount: bool,
    ) -> RecalibrationResult {
        let start = timing_start();
        assert!(
            self.trie.pending_prior.is_empty() && self.trie.pending_likelihood.len() == 1,
            "recalibrate called with unprocessed trie updates"
        );

        let certain_prefix_nodes = self.certain_prefix_nodes();
        let certain_prefix_hashes = certain_prefix_nodes
            .iter()
            .map(|(hash, _)| *hash)
            .collect::<Vec<_>>();
        assert!(
            !certain_prefix_nodes.is_empty(),
            "recalibrate expected at least the root node in certain_prefix_nodes"
        );

        let mut prior = initial_prior;
        let mut used_likelihood_updates = 0usize;
        let mut recent_pairs: Vec<CalibrationSample> = Vec::new();

        if let Some(&last_certain) = certain_prefix_hashes.last() {
            for payload in &self.likelihood_history {
                let mut payload_hashes = HashMap::new();
                for (s, nhash) in &payload.nodes {
                    payload_hashes.insert(rh::hash_string(s), nhash);
                }

                if payload_hashes.contains_key(&last_certain) {
                    continue;
                }

                let mut target_node = None;
                for hash in certain_prefix_hashes.iter().rev() {
                    if let Some(&nhash) = payload_hashes.get(hash) {
                        target_node = Some(nhash);
                        break;
                    }
                }

                if let Some(target) = target_node {
                    let phase = target.phase;
                    let mut x = payload.y - phase;
                    x = ((x % payload.period) + payload.period) % payload.period;
                    prior = calibration::optimize_online(
                        x,
                        payload.period,
                        &prior,
                        use_cross_entropy_discount,
                    );
                    used_likelihood_updates += 1;
                    recent_pairs.push(CalibrationSample {
                        x,
                        period: payload.period,
                    });
                    if recent_pairs.len() > 5 {
                        recent_pairs.remove(0);
                    }
                }
            }
        }

        self.current_prior = prior;

        let metrics = RecalibrationResult {
            prior_params: self.current_prior,
            used_likelihood_updates,
            recent_pairs,
        };
        self.record_event("recalibrate", start, None);
        metrics
    }

    pub fn recalibrate(
        &mut self,
        initial_prior_json: String,
        use_cross_entropy_discount: bool,
    ) -> String {
        let initial_prior =
            serde_json::from_str(&initial_prior_json).expect("Invalid initial prior JSON");
        serde_json::to_string(&self.recalibrate_typed(initial_prior, use_cross_entropy_discount))
            .unwrap()
    }

    pub fn current_prior(&self) -> VariationalParams {
        let start = timing_start();
        let prior = self.current_prior;
        self.record_event("current_prior", start, None);
        prior
    }

    pub fn current_prior_json(&self) -> String {
        serde_json::to_string(&self.current_prior()).unwrap()
    }

    pub fn set_current_prior(&mut self, prior: VariationalParams) {
        let start = timing_start();
        self.current_prior = prior;
        self.record_event("set_current_prior", start, None);
    }

    pub fn set_current_prior_json(&mut self, prior_json: String) {
        let prior = serde_json::from_str(&prior_json).expect("Invalid variational prior JSON");
        self.set_current_prior(prior);
    }

    pub fn certain_prefix_string(&mut self) -> String {
        let certain_prefix_nodes = self.certain_prefix_nodes();
        let certain_prefix_symbols = certain_prefix_nodes
            .iter()
            .map(|(_, symbol)| *symbol)
            .collect::<Vec<_>>();
        String::from_utf8(certain_prefix_symbols).unwrap()
    }

    #[cfg(feature = "tokentrie")]
    pub fn next_requested_prior_typed(&mut self) -> RequestedPrior {
        let start = timing_start();
        let requested_prior = self.trie.next_requested_prior();
        let requested_prior = RequestedPrior {
            full_string: requested_prior.full_string,
            last_token_lexindex: requested_prior.last_token_lexindex,
        };
        self.record_event("next_requested_prior", start, None);
        requested_prior
    }

    #[cfg(feature = "tokentrie")]
    pub fn next_requested_prior(&mut self) -> String {
        serde_json::to_string(&self.next_requested_prior_typed()).unwrap()
    }

    pub fn receive_prior_update_typed(&mut self, payload: PriorUpdatePayload) {
        let start = timing_start();
        let payload_json =
            serde_json::to_string(&payload).expect("prior update serialization failed");
        let payload = payload;

        let json_payload_ix = self.push_json_payload(payload_json);
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
            &self.trie.tokenizer,
        );
        // insert prediction into the registry
        // hash the full string
        let full_hash = rh::hash_string(&payload.full_string);
        self.trie.full_predictions.insert(full_hash, new_prediction);
        self.trie.pending_prior.insert(full_hash);
        self.record_event("receive_prior_update", start, Some(json_payload_ix));
    }

    pub fn receive_prior_update(&mut self, prior_json: String) {
        let payload = serde_json::from_str::<PriorUpdatePayload>(&prior_json).unwrap_or_else(|e| {
            let truncated = if prior_json.len() > 200 {
                format!("{}...[truncated]", &prior_json[..200])
            } else {
                prior_json.clone()
            };
            panic!(
                "Deserialization of Payload failed: {} | Payload (truncated): {}",
                e, truncated
            )
        });
        self.receive_prior_update_typed(payload);
    }

    pub fn apply_updates(&mut self) {
        let start = timing_start();
        self.trie.recalc_to_frontier(RecalcType::Update);
        self.record_event("apply_updates", start, None);
    }

    pub fn expand_to_threshold_snapshot(&mut self) -> ExpandedSnapshot {
        let start = timing_start();
        assert!(
            self.trie.pending_prior.is_empty() && self.trie.pending_likelihood.len() == 1,
            "Tried to expand with unprocessed updates"
        );
        let nodes_list = match self.trie.recalc_to_frontier(RecalcType::Expand {
            threshold: Float::from(trie::TRIE_EXPANSION_THRESHOLD as f32),
        }) {
            RecalcResult::Updated => {
                panic!(
                    "expand_to_threshold unexpectedly returned Updated after applying pending updates"
                )
            }
            RecalcResult::Expanded {
                nodes_over_threshold,
            } => nodes_over_threshold,
        };

        let snapshot_by_string = snapshot_by_string(&self.trie, nodes_list);
        self.record_event("expand_to_threshold", start, None);
        snapshot_by_string
    }

    pub fn expand_to_threshold(&mut self) -> String {
        serde_json::to_string(&self.expand_to_threshold_snapshot()).unwrap()
    }

    pub fn lexicographic_tokens_json(&self) -> String {
        let start = timing_start();
        let tokens_json = serde_json::to_string(self.trie.tokenizer.tokens()).unwrap();
        self.record_event("lexicographic_tokens_json", start, None);
        tokens_json
    }

    pub fn debug_dump(&self) -> SessionObservability {
        let start = timing_start();
        let dump = self.observability.lock().unwrap().clone();
        self.record_event("debug_dump", start, None);
        dump
    }

    pub fn debug_dump_json(&self) -> String {
        serde_json::to_string(&self.debug_dump()).unwrap()
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

    #[pyo3(name = "recalibrate")]
    fn py_recalibrate(
        &mut self,
        initial_prior_json: String,
        use_cross_entropy_discount: bool,
    ) -> String {
        BayesianSession::recalibrate(self, initial_prior_json, use_cross_entropy_discount)
    }

    #[pyo3(name = "current_prior_json")]
    fn py_current_prior_json(&self) -> String {
        BayesianSession::current_prior_json(self)
    }

    #[pyo3(name = "set_current_prior_json")]
    fn py_set_current_prior_json(&mut self, prior_json: String) {
        BayesianSession::set_current_prior_json(self, prior_json);
    }

    #[pyo3(name = "certain_prefix_string")]
    fn py_certain_prefix_string(&mut self) -> String {
        BayesianSession::certain_prefix_string(self)
    }

    #[pyo3(name = "lexicographic_tokens_json")]
    fn py_lexicographic_tokens_json(&self) -> String {
        BayesianSession::lexicographic_tokens_json(self)
    }

    #[pyo3(name = "debug_dump_json")]
    fn py_debug_dump_json(&self) -> String {
        BayesianSession::debug_dump_json(self)
    }
}
