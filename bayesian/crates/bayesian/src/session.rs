use std::collections::HashMap;
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use bpe::TokenLexIndex;
use calibration::VariationalParams;
use trie::safe_float::{Float, into_f32};
use trie::symbol::{PadMode, START_SYMBOL, XSymbol};
use trie::core::{XBayes, RecalcType, RecalcResult};
use trie::l_update::{merge_xl_pair, set_leaf_indicators, XLUpdate, XLUpdateEntry};
use trie::prediction::XPrediction;
use rolling_hash as rh;
use rolling_hash::Hash;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
    web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now()
}

#[cfg(not(target_arch = "wasm32"))]
fn timing_start() -> TimingStart {
    Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn elapsed_ms_since(start: TimingStart) -> f64 {
    web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now()
        - start
}

#[cfg(not(target_arch = "wasm32"))]
fn elapsed_ms_since(start: TimingStart) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn serialize_symbol<S>(symbol: &XSymbol, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_char(*symbol as char)
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
            return Err(serde::de::Error::custom("symbol string must be exactly one character"));
        }
        let code = u32::from(c);
        if code > 127 {
            return Err(serde::de::Error::custom("non-ASCII symbol"));
        }
        Ok(code as XSymbol)
    })
    .transpose()
}

#[derive(Clone, Serialize)]
struct SessionEvent {
    kind: &'static str,
    duration_ms: f64,
    json_payload_ix: Option<usize>,
}

#[derive(Clone, Default, Serialize)]
struct SessionObservability {
    json_payloads: Vec<String>,
    event_log: Vec<SessionEvent>,
}

#[derive(Deserialize, Clone)]
struct LikelihoodUpdatePayload {
    period: f64,
    y: f64,
    nodes: HashMap<String, NHash>,
}

#[derive(Deserialize, Clone)]
struct NHash {
    // by default, serde will ignore extra fields
    #[serde(alias = "l")]
    likelihood: f32,
    /// Omitted in wire JSON: taken as the last character of the map key (`a`–`z`, `S`, `Z`, `A`).
    #[serde(default, deserialize_with = "deserialize_optional_symbol")]
    symbol: Option<XSymbol>,
    phase: f64,
}

#[derive(Serialize)]
struct CalibrationSample {
    x: f64,
    period: f64,
}

#[derive(Serialize)]
struct RecalibrationResult {
    prior_params: VariationalParams,
    used_likelihood_updates: usize,
    recent_pairs: Vec<CalibrationSample>,
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
    fn certain_prefix_nodes(&mut self) -> Vec<Hash> {
        let threshold = (-0.0100503f32).into(); // ln(0.99)
        match self.trie.recalc_to_frontier(trie::core::RecalcType::Expand { threshold }) {
            trie::core::RecalcResult::Expanded { nodes_over_threshold } => nodes_over_threshold,
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
        self.observability.lock().unwrap().event_log.push(SessionEvent {
            kind,
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
            kind: "reset",
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

    pub fn receive_likelihood_update(&mut self, likelihood_json: String) {
        let start = timing_start();
        
        let payload =
            serde_json::from_str::<LikelihoodUpdatePayload>(&likelihood_json).unwrap();
        assert!(payload.period.is_finite() && payload.period > 0.0, "likelihood period must be finite and positive");
        assert!(payload.y.is_finite(), "likelihood y must be finite");
        assert!(!payload.nodes.is_empty(), "likelihood update contained no nodes");
        for (s, nhash) in &payload.nodes {
            assert!(!s.is_empty(), "likelihood update contained an empty node string");
            assert!(nhash.likelihood.is_finite(), "likelihood for node {:?} must be finite", s);
            assert!(nhash.phase.is_finite(), "phase for node {:?} must be finite", s);
            assert!(
                nhash.phase >= 0.0 && nhash.phase < payload.period,
                "phase for node {:?} must lie in [0, period); got {} with period {}",
                s,
                nhash.phase,
                payload.period,
            );
        }
        
        self.likelihood_history.push(payload.clone());
        let json_payload_ix = self.push_json_payload(likelihood_json);
        
        let mut new_l_update = payload.nodes.iter()
            .map(|(s, nhash)| (rh::hash_string(&s), 
                XLUpdateEntry {
                    likelihood: Float::from(nhash.likelihood),
                    symbol: nhash.symbol.unwrap_or_else(|| {
                        let b = *s.as_bytes().last().unwrap();
                        b as XSymbol
                    }),
                    is_leaf: false,
                }))
            .collect::<XLUpdate>();
        set_leaf_indicators(&mut new_l_update);
        //
        self.trie.pending_likelihood =
            merge_xl_pair(&self.trie.pending_likelihood, &new_l_update);
        self.record_event("receive_likelihood_update", start, Some(json_payload_ix));
    }

    pub fn recalibrate(&mut self, initial_prior_json: String, use_cross_entropy_discount: bool) -> String {
        let start = timing_start();
        assert!(
            self.trie.pending_prior.is_empty() && self.trie.pending_likelihood.len() == 1,
            "recalibrate called with unprocessed trie updates"
        );

        let certain_prefix_nodes = self.certain_prefix_nodes();
        assert!(!certain_prefix_nodes.is_empty(), "recalibrate expected at least the root node in certain_prefix_nodes");

        let mut prior: VariationalParams =
            serde_json::from_str(&initial_prior_json).expect("Invalid initial prior JSON");
        let mut used_likelihood_updates = 0usize;
        let mut recent_pairs: Vec<CalibrationSample> = Vec::new();
        
        if let Some(&last_certain) = certain_prefix_nodes.last() {
            for payload in &self.likelihood_history {
                let mut payload_hashes = HashMap::new();
                for (s, nhash) in &payload.nodes {
                    payload_hashes.insert(rh::hash_string(s), nhash);
                }
                
                if payload_hashes.contains_key(&last_certain) {
                    continue;
                }
                
                let mut target_node = None;
                for hash in certain_prefix_nodes.iter().rev() {
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
                    recent_pairs.push(CalibrationSample { x, period: payload.period });
                    if recent_pairs.len() > 5 {
                        recent_pairs.remove(0);
                    }
                }
            }
        }

        self.current_prior = prior;

        let metrics_json = serde_json::to_string(&RecalibrationResult {
            prior_params: self.current_prior,
            used_likelihood_updates,
            recent_pairs,
        }).unwrap();
        self.record_event("recalibrate", start, None);
        metrics_json
    }

    pub fn current_prior_json(&self) -> String {
        let start = timing_start();
        let json = serde_json::to_string(&self.current_prior).unwrap();
        self.record_event("current_prior_json", start, None);
        json
    }

    pub fn set_current_prior_json(&mut self, prior_json: String) {
        let start = timing_start();
        self.current_prior = serde_json::from_str(&prior_json).expect("Invalid variational prior JSON");
        self.record_event("set_current_prior_json", start, None);
    }

    pub fn certain_prefix_string(&mut self) -> String {
        let certain_prefix_nodes = self.certain_prefix_nodes();
        let symbols = certain_prefix_nodes.iter()
            .map(|hash| self.trie.nodes.get(hash).unwrap().symbol)
            .collect::<Vec<_>>();

        String::from_utf8(symbols).unwrap()
    }

    #[cfg(feature = "tokentrie")]
    pub fn next_requested_prior(&mut self) -> String {
        let start = timing_start();
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
        let requested_prior_json = serde_json::to_string(&requested_prior).unwrap();
        self.record_event("next_requested_prior", start, None);
        requested_prior_json
    }

    pub fn receive_prior_update(&mut self, prior_json: String) {
        let start = timing_start();
        #[derive(Deserialize)]
        struct Payload {
            full_string: String,
            final_token_lexindex: TokenLexIndex,
            follower_logits: Vec<f32>
        }
        let payload = serde_json::from_str::<Payload>(&prior_json)
            .unwrap_or_else(|e| {
                let truncated = if prior_json.len() > 200 {
                    format!("{}...[truncated]", &prior_json[..200])
                } else {
                    prior_json.clone()
                };
                panic!("Deserialization of Payload failed: {} | Payload (truncated): {}", e, truncated)
            });
       
   
        let json_payload_ix = self.push_json_payload(prior_json);
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
        self.record_event("receive_prior_update", start, Some(json_payload_ix));
    }

    pub fn apply_updates(&mut self) {
        let start = timing_start();
        self.trie.recalc_to_frontier(RecalcType::Update);
        self.record_event("apply_updates", start, None);
    }

    pub fn expand_to_threshold(&mut self) -> String {
        let start = timing_start();
        assert!(self.trie.pending_prior.is_empty() && self.trie.pending_likelihood.len() == 1, "Tried to expand with unprocessed updates");
        let nodes_list = match self.trie.recalc_to_frontier(
            RecalcType::Expand { threshold: Float::from(trie::TRIE_EXPANSION_THRESHOLD as f32) },
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
            symbol: XSymbol,
            upper_siblings_inclusive_cum_z: Option<Float>,
        }
        let mut snapshot_by_hash: rh::RHashMap<NString> = rh::RHashMap::default();
        snapshot_by_hash.insert(trie::ROOT_HASH, NString {
            string: String::from(trie::ROOT_STRING),
            z: self.trie.nodes.get(&trie::ROOT_HASH).unwrap().if_root_then_z,
            p: None,
            tp: None,
            tp0: None,
            a_tl0: None,
            symbol: START_SYMBOL,
            upper_siblings_inclusive_cum_z: None,
        });

        struct Frame {
            n_hash: Hash,
            n_symbol: XSymbol,
            n_depth: u16,
        }
        // a frame is a node that *may* have children that should be added to the snapshot
        // by the time we see a frame, it should already be in the snapshot
        // the frame list is a visit list
        let mut full_string = String::new(); // character stack
        let mut frames: Vec<Frame> = vec![Frame { n_hash: trie::ROOT_HASH, n_symbol: START_SYMBOL, n_depth: 0 }];
        while let Some(Frame { n_hash, n_symbol, n_depth }) = frames.pop() {
            full_string.truncate(n_depth as usize);
            full_string.push(n_symbol as char);
            debug_assert!(snapshot_by_hash.contains_key(&n_hash), "frame node not in snapshot");
            let n_padmode = PadMode::for_xsymbol(n_symbol);
            let c_present = (0..n_padmode.radix()).map(|slot| {
                let c_byte = n_padmode.slot_to_xsymbol(slot);
                let c_hash = rh::append_right(n_hash, c_byte);
                nodes_set.contains(&c_hash)
            }).collect::<Vec<_>>();
            let is_interior = c_present.iter().any(|&p| p);
            if !is_interior {
                continue;
            }
            debug_assert!(self.trie.nodes.contains_key(&n_hash), "frame node not in trie");
            let n_node = self.trie.nodes.get(&n_hash).unwrap();
            let c_z = n_node.c_z;
            // Compute the partial sums of c_z using logaddexp, i.e., log-sum-exp over upper siblings
            let mut c_upper_siblings_inclusive_cum_z = vec![Float::NEG_INFINITY; n_padmode.radix()].into_boxed_slice();
            let mut accum = Float::NEG_INFINITY;
            for i in 0..n_padmode.radix() {
                accum = trie::logaddexp(accum, c_z[i]);
                c_upper_siblings_inclusive_cum_z[i] = accum;
            }
       
            for slot in 0..n_padmode.radix() {
                if !c_present[slot] {
                    continue;
                }
                let c_symbol = n_padmode.slot_to_xsymbol(slot);
                let c_byte = c_symbol;
                let c_hash = rh::append_right(n_hash, c_byte);
                let mut c_string = full_string.clone();
                c_string.push(c_symbol as char);
                let (z, p, tp, tp0, a_tl0) = n_node.edge_snapshot_fields(slot);
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
            #[serde(serialize_with = "serialize_symbol")]
            symbol: XSymbol,
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
        self.record_event("expand_to_threshold", start, None);
        snapshot_json
    }

    pub fn lexicographic_tokens_json(&self) -> String {
        let start = timing_start();
        let tokens_json = serde_json::to_string(self.trie.tokenizer.tokens()).unwrap();
        self.record_event("lexicographic_tokens_json", start, None);
        tokens_json
    }

    pub fn debug_dump_json(&self) -> String {
        let start = timing_start();
        let dump_json = serde_json::to_string(&*self.observability.lock().unwrap()).unwrap();
        self.record_event("debug_dump_json", start, None);
        dump_json
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
    fn py_recalibrate(&mut self, initial_prior_json: String, use_cross_entropy_discount: bool) -> String {
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