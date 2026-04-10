use std::time::Instant;
use std::{fs, thread, time::Duration};

use bayesian::BayesianSession;

const DEFAULT_N_TRIALS: usize = 1;

fn main() {
    let n_trials = std::env::args()
        .nth(1)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(DEFAULT_N_TRIALS);

    let mut sessions = Vec::with_capacity(n_trials);
    for _ in 0..n_trials {
        sessions.push(BayesianSession::new());
    }

    if let Some(pidfile) = std::env::var_os("PROFILE_READY_PIDFILE") {
        fs::write(pidfile, std::process::id().to_string()).unwrap();
        thread::sleep(Duration::from_secs(2));
    }

    let mut durations_ms = Vec::with_capacity(n_trials);

    for (trial, session) in sessions.iter_mut().enumerate() {
        let start = Instant::now();
        let snapshot_json = session.expand_to_threshold();
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        durations_ms.push(duration_ms);
        println!(
            "trial {:>2}: {:>8.3} ms  snapshot_bytes={}",
            trial + 1,
            duration_ms,
            snapshot_json.len()
        );
    }

    let average_ms = durations_ms.iter().sum::<f64>() / durations_ms.len() as f64;
    println!("average: {:.3} ms over {} trials", average_ms, n_trials);
}
