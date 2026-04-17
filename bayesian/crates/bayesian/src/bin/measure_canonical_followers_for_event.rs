use std::fs;
use std::time::Instant;

use bayesian::BayesianSession;
use bpe::{canonical_followers_timing_snapshot, reset_canonical_followers_timing};
use serde::Deserialize;
use trie::{reset_zero_order_prediction_timing, zero_order_prediction_timing_snapshot};

#[derive(Deserialize)]
struct SessionDump {
    json_payloads: Vec<String>,
    event_log: Vec<SessionEvent>,
}

#[derive(Deserialize)]
struct SessionEvent {
    kind: String,
    duration_ms: f64,
    json_payload_ix: Option<usize>,
}

fn apply_event(session: &mut BayesianSession, event: &SessionEvent, json_payloads: &[String]) {
    match event.kind.as_str() {
        "reset" => session.reset(),
        "set_current_prior_json" => {
            if let Some(payload_ix) = event.json_payload_ix {
                session.set_current_prior_json(json_payloads[payload_ix].clone());
            }
        }
        "receive_prior_update" => {
            let payload_ix = event.json_payload_ix.unwrap();
            session.receive_prior_update(json_payloads[payload_ix].clone());
        }
        "receive_likelihood_update" => {
            let payload_ix = event.json_payload_ix.unwrap();
            session.receive_likelihood_update(json_payloads[payload_ix].clone());
        }
        "apply_updates" => session.apply_updates(),
        "expand_to_threshold" => {
            let _ = session.expand_to_threshold();
        }
        "recalibrate" => {
            let initial = session.current_prior_json();
            let _ = session.recalibrate(initial, true);
        }
        other => panic!("unsupported event kind in replay: {other}"),
    }
}

fn main() {
    let dump_path = std::env::args().nth(1).expect(
        "usage: measure_canonical_followers_for_event <dump_path> <target_event_ix> [n_trials]",
    );
    let target_event_ix = std::env::args()
        .nth(2)
        .expect("missing <target_event_ix>")
        .parse::<usize>()
        .unwrap();
    let n_trials = std::env::args()
        .nth(3)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(5);

    let dump_json = fs::read_to_string(&dump_path).unwrap();
    let dump: SessionDump = serde_json::from_str(&dump_json).unwrap();
    let target_event = dump
        .event_log
        .get(target_event_ix)
        .unwrap_or_else(|| panic!("target_event_ix {} out of range", target_event_ix));

    println!(
        "target event {}: kind={} dump_duration_ms={} payload_ix={:?}",
        target_event_ix, target_event.kind, target_event.duration_ms, target_event.json_payload_ix
    );

    let mut event_durations_ms = Vec::with_capacity(n_trials);
    let mut batch_call_counts = Vec::with_capacity(n_trials);
    let mut batch_total_ms = Vec::with_capacity(n_trials);
    let mut zero_order_total_ms = Vec::with_capacity(n_trials);
    let mut zero_order_count_by_prefix_ms = Vec::with_capacity(n_trials);
    let mut zero_order_prefix_bool_ms = Vec::with_capacity(n_trials);
    let mut zero_order_follower_probs_ms = Vec::with_capacity(n_trials);
    let mut zero_order_prob_for_prefix_ms = Vec::with_capacity(n_trials);
    let mut zero_order_canonical_box_ms = Vec::with_capacity(n_trials);

    for trial in 0..n_trials {
        let mut session = BayesianSession::new();
        for event in dump.event_log.iter().take(target_event_ix) {
            apply_event(&mut session, event, &dump.json_payloads);
        }

        reset_canonical_followers_timing();
        reset_zero_order_prediction_timing();
        let started = Instant::now();
        apply_event(&mut session, target_event, &dump.json_payloads);
        let event_duration_ms = started.elapsed().as_secs_f64() * 1000.0;
        let timing = canonical_followers_timing_snapshot();
        let zero_order_timing = zero_order_prediction_timing_snapshot();
        let batch_total_ms_this_trial = timing.total_ns as f64 / 1_000_000.0;
        let zero_order_total_ms_this_trial = zero_order_timing.total_ns as f64 / 1_000_000.0;

        event_durations_ms.push(event_duration_ms);
        batch_call_counts.push(timing.call_count);
        batch_total_ms.push(batch_total_ms_this_trial);
        zero_order_total_ms.push(zero_order_total_ms_this_trial);
        zero_order_count_by_prefix_ms
            .push(zero_order_timing.count_true_tokens_by_prefix_ns as f64 / 1_000_000.0);
        zero_order_prefix_bool_ms
            .push(zero_order_timing.canonical_follower_for_prefix_ns as f64 / 1_000_000.0);
        zero_order_follower_probs_ms
            .push(zero_order_timing.follower_probs_ns as f64 / 1_000_000.0);
        zero_order_prob_for_prefix_ms
            .push(zero_order_timing.follower_prob_for_prefix_ns as f64 / 1_000_000.0);
        zero_order_canonical_box_ms
            .push(zero_order_timing.canonical_followers_box_ns as f64 / 1_000_000.0);

        let avg_batch_ms_this_trial = if timing.call_count == 0 {
            0.0
        } else {
            batch_total_ms_this_trial / timing.call_count as f64
        };

        println!(
            "trial {:>2}: event {:>8.3} ms | canonical_followers {:>8.3} ms over {:>4} calls | avg {:>8.6} ms/call",
            trial + 1,
            event_duration_ms,
            batch_total_ms_this_trial,
            timing.call_count,
            avg_batch_ms_this_trial
        );
        println!(
            "          zero_order total {:>8.3} ms | count_by_prefix {:>8.3} | prefix_bool {:>8.3} | follower_probs {:>8.3} | prob_for_prefix {:>8.3} | canonical_box {:>8.3}",
            zero_order_timing.total_ns as f64 / 1_000_000.0,
            zero_order_timing.count_true_tokens_by_prefix_ns as f64 / 1_000_000.0,
            zero_order_timing.canonical_follower_for_prefix_ns as f64 / 1_000_000.0,
            zero_order_timing.follower_probs_ns as f64 / 1_000_000.0,
            zero_order_timing.follower_prob_for_prefix_ns as f64 / 1_000_000.0,
            zero_order_timing.canonical_followers_box_ns as f64 / 1_000_000.0,
        );
    }

    let avg_event_ms = event_durations_ms.iter().sum::<f64>() / event_durations_ms.len() as f64;
    let avg_calls = batch_call_counts.iter().sum::<u64>() as f64 / batch_call_counts.len() as f64;
    let avg_batch_total_ms = batch_total_ms.iter().sum::<f64>() / batch_total_ms.len() as f64;
    let avg_batch_ms_per_call = if avg_calls == 0.0 {
        0.0
    } else {
        avg_batch_total_ms / avg_calls
    };
    let avg_zero_order_total_ms =
        zero_order_total_ms.iter().sum::<f64>() / zero_order_total_ms.len() as f64;
    let avg_zero_order_count_by_prefix_ms = zero_order_count_by_prefix_ms
        .iter()
        .sum::<f64>()
        / zero_order_count_by_prefix_ms.len() as f64;
    let avg_zero_order_prefix_bool_ms =
        zero_order_prefix_bool_ms.iter().sum::<f64>() / zero_order_prefix_bool_ms.len() as f64;
    let avg_zero_order_follower_probs_ms = zero_order_follower_probs_ms
        .iter()
        .sum::<f64>()
        / zero_order_follower_probs_ms.len() as f64;
    let avg_zero_order_prob_for_prefix_ms = zero_order_prob_for_prefix_ms
        .iter()
        .sum::<f64>()
        / zero_order_prob_for_prefix_ms.len() as f64;
    let avg_zero_order_canonical_box_ms = zero_order_canonical_box_ms
        .iter()
        .sum::<f64>()
        / zero_order_canonical_box_ms.len() as f64;

    println!(
        "average: event {:.3} ms | canonical_followers {:.3} ms over {:.1} calls | avg {:.6} ms/call",
        avg_event_ms,
        avg_batch_total_ms,
        avg_calls,
        avg_batch_ms_per_call
    );
    println!(
        "average zero_order: total {:.3} ms | count_by_prefix {:.3} | prefix_bool {:.3} | follower_probs {:.3} | prob_for_prefix {:.3} | canonical_box {:.3}",
        avg_zero_order_total_ms,
        avg_zero_order_count_by_prefix_ms,
        avg_zero_order_prefix_bool_ms,
        avg_zero_order_follower_probs_ms,
        avg_zero_order_prob_for_prefix_ms,
        avg_zero_order_canonical_box_ms,
    );
}
