use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use bayesian::bpe::TinyLlamaWordTokenizer;

#[derive(Clone, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0xdead_beef_cafe_babe
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn gen_index(&mut self, len: usize) -> usize {
        (self.next_u64() % len as u64) as usize
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(tokenizer_path) = args.next() else {
        eprintln!(
            "usage: cargo run --release --bin spine_bench -- <tokenizer.json> [samples] [seed]"
        );
        return ExitCode::from(2);
    };
    let samples = match args.next() {
        Some(value) => match value.parse::<usize>() {
            Ok(samples) if samples > 0 => samples,
            _ => {
                eprintln!("samples must be a positive integer");
                return ExitCode::from(2);
            }
        },
        None => 100,
    };
    let seed = match args.next() {
        Some(value) => match value.parse::<u64>() {
            Ok(seed) => seed,
            Err(_) => {
                eprintln!("seed must be an integer");
                return ExitCode::from(2);
            }
        },
        None => 1,
    };

    let load_start = Instant::now();
    let tokenizer = match TinyLlamaWordTokenizer::from_tokenizer_json(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!("failed to load tokenizer: {err}");
            return ExitCode::from(1);
        }
    };
    let load_plus_precompute_seconds = load_start.elapsed().as_secs_f64();

    let candidate_second_ids = tokenizer.token_ids_with_left_spines().to_vec();
    let candidate_second_spines = tokenizer.packed_left_spines();

    let mut rng = XorShift64::new(seed);
    let sampled_first_ids: Vec<u32> = (0..samples)
        .map(|_| candidate_second_ids[rng.gen_index(candidate_second_ids.len())])
        .collect();

    let timed_start = Instant::now();
    let mut pair_count = 0u64;
    let mut used_first_ids = 0u64;
    let mut canonical_count = 0u64;
    for first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        used_first_ids += 1;
        for second_left_spine in candidate_second_spines {
            if black_box(
                tokenizer.canonical_pair_from_packed_spines(&first_right_spine, second_left_spine),
            ) {
                canonical_count += 1;
            }
            pair_count += 1;
        }
    }
    let timed_elapsed_seconds = timed_start.elapsed().as_secs_f64();

    println!("candidate_second_ids = {}", candidate_second_ids.len());
    println!("sampled_first_ids = {samples}");
    println!("used_first_ids = {used_first_ids}");
    println!("pair_count = {pair_count}");
    println!("canonical_count = {canonical_count}");
    println!("load_plus_precompute_seconds = {load_plus_precompute_seconds:.6}");
    println!("timed_elapsed_seconds = {timed_elapsed_seconds:.6}");
    println!(
        "micros_per_candidate = {:.3}",
        timed_elapsed_seconds * 1_000_000.0 / pair_count as f64
    );
    println!(
        "millis_per_first_token = {:.3}",
        timed_elapsed_seconds * 1_000.0 / used_first_ids as f64
    );

    ExitCode::SUCCESS
}
