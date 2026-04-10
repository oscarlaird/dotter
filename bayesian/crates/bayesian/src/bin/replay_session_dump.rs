use std::fs;
use std::time::Instant;

use bayesian::BayesianSession;
use serde::Deserialize;

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

fn main() {
    let dump_path = std::env::args().nth(1).expect("usage: replay_session_dump <dump_path> <target_event_ix> [n_trials]");
    let target_event_ix = std::env::args()
        .nth(2)
        .expect("missing <target_event_ix>")
        .parse::<usize>()
        .unwrap();
    let n_trials = std::env::args()
        .nth(3)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(1);

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

    let mut durations_ms = Vec::with_capacity(n_trials);

    for trial in 0..n_trials {
        let mut session = BayesianSession::new();
        let mut measured_ms = None;

        for (event_ix, event) in dump.event_log.iter().enumerate().take(target_event_ix + 1) {
            let started = event_ix == target_event_ix;
            let start = if started { Some(Instant::now()) } else { None };

            match event.kind.as_str() {
                "reset" => session.reset(),
                "receive_prior_update" => {
                    let payload_ix = event.json_payload_ix.unwrap();
                    session.receive_prior_update(dump.json_payloads[payload_ix].clone());
                }
                "receive_likelihood_update" => {
                    let payload_ix = event.json_payload_ix.unwrap();
                    session.receive_likelihood_update(dump.json_payloads[payload_ix].clone());
                }
                "apply_updates" => session.apply_updates(),
                "expand_to_threshold" => {
                    let _ = session.expand_to_threshold();
                }
                other => panic!("unsupported event kind in replay: {other}"),
            }

            if let Some(start) = start {
                measured_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
            }
        }

        let measured_ms = measured_ms.unwrap();
        durations_ms.push(measured_ms);
        println!(
            "trial {:>2}: measured {:>8.3} ms  dump {:>8.3} ms",
            trial + 1,
            measured_ms,
            target_event.duration_ms
        );
    }

    let average_ms = durations_ms.iter().sum::<f64>() / durations_ms.len() as f64;
    println!(
        "average: {:.3} ms over {} trials (dump recorded {:.3} ms)",
        average_ms,
        n_trials,
        target_event.duration_ms
    );
}
