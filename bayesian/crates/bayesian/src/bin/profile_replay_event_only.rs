use std::fs;
use std::thread;
use std::time::{Duration, Instant};

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
    let dump_path = std::env::args()
        .nth(1)
        .expect("usage: profile_replay_event_only <dump_path> <target_event_ix> [n_trials]");
    let target_event_ix = std::env::args()
        .nth(2)
        .expect("missing <target_event_ix>")
        .parse::<usize>()
        .unwrap();
    let n_trials = std::env::args()
        .nth(3)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(20);

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

    let mut sessions = Vec::with_capacity(n_trials);
    for _ in 0..n_trials {
        let mut session = BayesianSession::new();
        for event in dump.event_log.iter().take(target_event_ix) {
            apply_event(&mut session, event, &dump.json_payloads);
        }
        sessions.push(session);
    }

    if let Some(pidfile) = std::env::var_os("PROFILE_READY_PIDFILE") {
        fs::write(pidfile, std::process::id().to_string()).unwrap();
        thread::sleep(Duration::from_secs(2));
    }

    let mut durations_ms = Vec::with_capacity(n_trials);
    for (trial, session) in sessions.iter_mut().enumerate() {
        let start = Instant::now();
        apply_event(session, target_event, &dump.json_payloads);
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        durations_ms.push(duration_ms);
        println!(
            "trial {:>2}: measured {:>8.3} ms  dump {:>8.3} ms",
            trial + 1,
            duration_ms,
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
