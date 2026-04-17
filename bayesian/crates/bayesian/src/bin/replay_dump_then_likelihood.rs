//! Replay a full `debug_dump_json` session capture, then apply one or more extra likelihood JSON
//! payloads in order. After each payload: `apply_updates` then `expand_to_threshold` (same as the
//! live client after each likelihood gesture). The final `expand_to_threshold` snapshot is printed.
//!
//! Usage:
//!   replay_dump_then_likelihood <dump_path> <likelihood_json_path> [<likelihood_json_path> ...]
//!
//! Trie keys use the usual wire convention: root `A`, word space as `S`. Examples:
//! - `" imp"` → `ASimp`; proper prefixes: `A`, `AS`, `ASi`, `ASim`.
//! - `" implement"` → `ASimplement`; proper prefixes are every strict prefix of that string (10 nodes).
//!
//! A tail file may be a single likelihood object, or a **JSON array** of objects. An array is
//! applied as multiple `receive_likelihood_update` calls **before** one `apply_updates` +
//! `expand_to_threshold`. (A single update listing every prefix plus `ASimplement` can trip an
//! internal assertion during expansion on some captures.)

use std::fs;

use bayesian::BayesianSession;
use serde::Deserialize;

#[derive(Deserialize)]
struct SessionDump {
    json_payloads: Vec<String>,
    event_log: Vec<SessionEvent>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct SessionEvent {
    kind: String,
    duration_ms: f64,
    json_payload_ix: Option<usize>,
}

fn apply_event(session: &mut BayesianSession, event: &SessionEvent, dump: &SessionDump) {
    match event.kind.as_str() {
        "reset" => session.reset(),
        "set_current_prior_json" => {
            if let Some(payload_ix) = event.json_payload_ix {
                session.set_current_prior_json(dump.json_payloads[payload_ix].clone());
            }
        }
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
        "recalibrate" => {
            let initial = session.current_prior_json();
            let _ = session.recalibrate(initial, true);
        }
        other => panic!("unsupported event kind in replay: {other}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dump_path = args.next().expect(
        "usage: replay_dump_then_likelihood <dump_path> <likelihood_json_path> [<likelihood_json_path> ...]",
    );
    let tail_paths: Vec<String> = args.collect();
    assert!(
        !tail_paths.is_empty(),
        "usage: replay_dump_then_likelihood <dump_path> <likelihood_json_path> [<likelihood_json_path> ...]"
    );

    let dump_json = fs::read_to_string(&dump_path).unwrap();
    let dump: SessionDump = serde_json::from_str(&dump_json).unwrap();

    let mut tail_jsons: Vec<String> = Vec::new();
    for path in &tail_paths {
        let s = fs::read_to_string(path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        if v.is_array() {
            for (j, part) in v.as_array().unwrap().iter().enumerate() {
                assert!(
                    part.is_object(),
                    "{}: tail array element {} must be an object",
                    path,
                    j
                );
            }
        } else {
            assert!(v.is_object(), "{path}: tail must be a JSON object or array of objects");
        }
        tail_jsons.push(s);
    }

    let mut session = BayesianSession::new();
    for event in &dump.event_log {
        apply_event(&mut session, event, &dump);
    }

    println!(
        "replayed {} dump events, then {} tail likelihood step(s)",
        dump.event_log.len(),
        tail_paths.len()
    );
    let mut last_snapshot_json = String::new();
    for (i, (path, tail_json)) in tail_paths.iter().zip(tail_jsons.iter()).enumerate() {
        let v: serde_json::Value = serde_json::from_str(tail_json).unwrap();
        if let serde_json::Value::Array(parts) = v {
            for part in &parts {
                session.receive_likelihood_update(serde_json::to_string(part).unwrap());
            }
            println!(
                "  tail {}: {} → {}× receive_likelihood_update, then apply_updates + expand_to_threshold",
                i + 1,
                path,
                parts.len()
            );
        } else {
            session.receive_likelihood_update(tail_json.clone());
            println!(
                "  tail {}: {} → apply_updates + expand_to_threshold",
                i + 1,
                path
            );
        }
        session.apply_updates();
        last_snapshot_json = session.expand_to_threshold();
    }

    let snapshot: serde_json::Value = serde_json::from_str(&last_snapshot_json).unwrap();
    println!("--- snapshot (expand_to_threshold JSON, after last tail) ---");
    println!("{}", serde_json::to_string_pretty(&snapshot).unwrap());
}
