use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use bayesian::bpe::prepared_dense::{
    canonical_pair_from_prepared_first_dense_left_len,
    count_mismatches_prepared_first_dense_bucket_lockstep4,
    scan_prepared_first_dense_bucket_lockstep4, PreparedSecondBuckets, PreparedSecondToken,
};
use bayesian::bpe::{
    PackedSpine, TINYLLAMA_PIECE_COUNT, TinyLlamaPreparedFirstDense, TinyLlamaWordTokenizer,
};

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

fn time_baseline(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut pair_count = 0u64;
    let mut used_first_ids = 0u64;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        used_first_ids += 1;
        for second_left_spine in candidate_second_spines {
            black_box(
                tokenizer.canonical_pair_from_packed_spines(&first_right_spine, second_left_spine),
            );
            pair_count += 1;
        }
    }

    (
        pair_count,
        used_first_ids,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn time_prepared_first_dense_split(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
    candidate_second_buckets: &PreparedSecondBuckets,
) -> Result<(u64, u64, u64, f64, f64, f64), bayesian::bpe::BpeError> {
    let total_start = Instant::now();
    let mut pair_count = 0u64;
    let mut used_first_ids = 0u64;
    let mut canonical_count = 0u64;
    let mut prepare_seconds = 0.0f64;
    let mut scan_seconds = 0.0f64;

    for &first_id in sampled_first_ids {
        let prepare_start = Instant::now();
        let Some(prepared_first) = tokenizer.prepare_canonical_pair_batch_for_token_id(first_id)?
        else {
            continue;
        };
        prepare_seconds += prepare_start.elapsed().as_secs_f64();

        used_first_ids += 1;

        let scan_start = Instant::now();
        macro_rules! scan_bucket {
            ($left_len:literal) => {
                for second_entry in &candidate_second_buckets[$left_len] {
                    if black_box(
                        canonical_pair_from_prepared_first_dense_left_len::<
                            TINYLLAMA_PIECE_COUNT,
                            $left_len,
                        >(
                            &prepared_first,
                            &second_entry.left_spine,
                        ),
                    ) {
                        canonical_count += 1;
                    }
                    pair_count += 1;
                }
            };
        }
        scan_bucket!(1);
        scan_bucket!(2);
        scan_bucket!(3);
        scan_bucket!(4);
        scan_bucket!(5);
        scan_bucket!(6);
        scan_bucket!(7);
        scan_bucket!(8);
        scan_seconds += scan_start.elapsed().as_secs_f64();
    }

    Ok((
        pair_count,
        used_first_ids,
        canonical_count,
        total_start.elapsed().as_secs_f64(),
        prepare_seconds,
        scan_seconds,
    ))
}

fn build_prepared_first_dense_batch(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
) -> Result<Vec<TinyLlamaPreparedFirstDense>, bayesian::bpe::BpeError> {
    let mut prepared = Vec::with_capacity(sampled_first_ids.len());
    for &first_id in sampled_first_ids {
        let Some(prepared_first) = tokenizer.prepare_canonical_pair_batch_for_token_id(first_id)?
        else {
            continue;
        };
        prepared.push(prepared_first);
    }
    Ok(prepared)
}

fn time_prepared_first_dense_scan_only(
    prepared_first_batch: &[TinyLlamaPreparedFirstDense],
    candidate_second_buckets: &PreparedSecondBuckets,
) -> (u64, u64, u64, f64) {
    let timed_start = Instant::now();
    let mut pair_count = 0u64;
    let used_first_ids = prepared_first_batch.len() as u64;
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_batch {
        macro_rules! scan_bucket {
            ($left_len:literal) => {
                for second_entry in &candidate_second_buckets[$left_len] {
                    if black_box(
                        canonical_pair_from_prepared_first_dense_left_len::<
                            TINYLLAMA_PIECE_COUNT,
                            $left_len,
                        >(
                            prepared_first,
                            &second_entry.left_spine,
                        ),
                    ) {
                        canonical_count += 1;
                    }
                    pair_count += 1;
                }
            };
        }
        scan_bucket!(1);
        scan_bucket!(2);
        scan_bucket!(3);
        scan_bucket!(4);
        scan_bucket!(5);
        scan_bucket!(6);
        scan_bucket!(7);
        scan_bucket!(8);
    }

    (
        pair_count,
        used_first_ids,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_mismatches_prepared_first_dense(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
    candidate_second_buckets: &PreparedSecondBuckets,
) -> Result<u64, bayesian::bpe::BpeError> {
    let mut mismatches = 0u64;

    for &first_id in sampled_first_ids {
        let Some(prepared_first) = tokenizer.prepare_canonical_pair_batch_for_token_id(first_id)?
        else {
            continue;
        };
        for bucket in candidate_second_buckets.iter().skip(1) {
            for second_entry in bucket {
                let second_left_spine = tokenizer
                    .left_packed_spine_for_token_id(second_entry.token_id)
                    .expect("prepared second token ids must resolve to packed left spines");
                let baseline = tokenizer.canonical_pair_from_packed_spines(
                    prepared_first.right_spine(),
                    second_left_spine,
                );
                let prepared = match second_entry.left_spine.len {
                    1 => canonical_pair_from_prepared_first_dense_left_len::<
                        TINYLLAMA_PIECE_COUNT,
                        1,
                    >(
                        &prepared_first,
                        &second_entry.left_spine,
                    ),
                    2 => canonical_pair_from_prepared_first_dense_left_len::<
                        TINYLLAMA_PIECE_COUNT,
                        2,
                    >(
                        &prepared_first,
                        &second_entry.left_spine,
                    ),
                    3 => canonical_pair_from_prepared_first_dense_left_len::<
                        TINYLLAMA_PIECE_COUNT,
                        3,
                    >(
                        &prepared_first,
                        &second_entry.left_spine,
                    ),
                    4 => canonical_pair_from_prepared_first_dense_left_len::<
                        TINYLLAMA_PIECE_COUNT,
                        4,
                    >(
                        &prepared_first,
                        &second_entry.left_spine,
                    ),
                    5 => canonical_pair_from_prepared_first_dense_left_len::<
                        TINYLLAMA_PIECE_COUNT,
                        5,
                    >(
                        &prepared_first,
                        &second_entry.left_spine,
                    ),
                    6 => canonical_pair_from_prepared_first_dense_left_len::<
                        TINYLLAMA_PIECE_COUNT,
                        6,
                    >(
                        &prepared_first,
                        &second_entry.left_spine,
                    ),
                    7 => canonical_pair_from_prepared_first_dense_left_len::<
                        TINYLLAMA_PIECE_COUNT,
                        7,
                    >(
                        &prepared_first,
                        &second_entry.left_spine,
                    ),
                    8 => canonical_pair_from_prepared_first_dense_left_len::<
                        TINYLLAMA_PIECE_COUNT,
                        8,
                    >(
                        &prepared_first,
                        &second_entry.left_spine,
                    ),
                    _ => unreachable!("prepared bucket entries must have packed left spines"),
                };
                if baseline != prepared {
                    mismatches += 1;
                }
            }
        }
    }

    Ok(mismatches)
}

fn print_timing(label: &str, pair_count: u64, used_first_ids: u64, elapsed_seconds: f64) {
    println!("{label}:");
    println!("  pair_count = {pair_count}");
    println!("  used_first_ids = {used_first_ids}");
    println!("  timed_elapsed_seconds = {elapsed_seconds:.6}");
    println!(
        "  micros_per_candidate = {:.3}",
        elapsed_seconds * 1_000_000.0 / pair_count as f64
    );
    println!(
        "  millis_per_first_token = {:.3}",
        elapsed_seconds * 1_000.0 / used_first_ids as f64
    );
}

fn time_bucket_scalar<const LEFT_LEN: usize>(
    prepared_first_batch: &[TinyLlamaPreparedFirstDense],
    entries: &[PreparedSecondToken],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut pair_count = 0u64;
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_batch {
        for entry in entries {
            if black_box(
                canonical_pair_from_prepared_first_dense_left_len::<
                    TINYLLAMA_PIECE_COUNT,
                    LEFT_LEN,
                >(prepared_first, &entry.left_spine),
            ) {
                canonical_count += 1;
            }
            pair_count += 1;
        }
    }

    (pair_count, canonical_count, timed_start.elapsed().as_secs_f64())
}

fn time_bucket_lockstep4<const LEFT_LEN: usize>(
    prepared_first_batch: &[TinyLlamaPreparedFirstDense],
    entries: &[PreparedSecondToken],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_batch {
        canonical_count +=
            black_box(scan_prepared_first_dense_bucket_lockstep4::<TINYLLAMA_PIECE_COUNT, LEFT_LEN>(
                prepared_first,
                entries,
            ));
    }

    (
        (prepared_first_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_lockstep4_mismatches<const LEFT_LEN: usize>(
    prepared_first_batch: &[TinyLlamaPreparedFirstDense],
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_batch {
        mismatches += count_mismatches_prepared_first_dense_bucket_lockstep4::<
            TINYLLAMA_PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, entries);
    }
    mismatches
}

fn print_bucket_timing(
    left_len: usize,
    pair_count: u64,
    scalar_canonical_count: u64,
    scalar_seconds: f64,
    lockstep_canonical_count: u64,
    lockstep_seconds: f64,
    mismatches: u64,
) {
    println!("  left_len = {left_len}");
    println!("    pair_count = {pair_count}");
    println!("    scalar_canonical_count = {scalar_canonical_count}");
    println!("    lockstep_canonical_count = {lockstep_canonical_count}");
    println!("    mismatches = {mismatches}");
    println!(
        "    scalar_micros_per_candidate = {:.3}",
        scalar_seconds * 1_000_000.0 / pair_count as f64
    );
    println!(
        "    lockstep4_micros_per_candidate = {:.3}",
        lockstep_seconds * 1_000_000.0 / pair_count as f64
    );
    println!(
        "    speedup = {:.3}x",
        scalar_seconds / lockstep_seconds
    );
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(tokenizer_path) = args.next() else {
        eprintln!(
            "usage: cargo run --release --bin hybrid_spine_bench -- <tokenizer.json> [samples] [seed]"
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
    let mode = args.next().unwrap_or_else(|| "all".to_string());
    let run_mismatch_checks = mode == "all";

    let load_start = Instant::now();
    let tokenizer = match TinyLlamaWordTokenizer::from_tokenizer_json(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!("failed to load tokenizer: {err}");
            return ExitCode::from(1);
        }
    };
    let load_plus_precompute_seconds = load_start.elapsed().as_secs_f64();

    let mut candidate_second_ids = tokenizer.token_ids_with_left_spines().to_vec();
    candidate_second_ids.sort_unstable();
    let candidate_second_spines: Vec<PackedSpine> = candidate_second_ids
        .iter()
        .map(|&token_id| {
            *tokenizer
                .left_packed_spine_for_token_id(token_id)
                .expect("sorted candidate ids must resolve to packed left spines")
        })
        .collect();
    let candidate_second_buckets = tokenizer.prepared_second_buckets();
    let mut rng = XorShift64::new(seed);
    let sampled_first_ids: Vec<u32> = (0..samples)
        .map(|_| candidate_second_ids[rng.gen_index(candidate_second_ids.len())])
        .collect();

    println!("candidate_second_ids = {}", candidate_second_ids.len());
    println!("sampled_first_ids = {samples}");
    println!("seed = {seed}");
    println!("mode = {mode}");
    println!("load_plus_precompute_seconds = {load_plus_precompute_seconds:.6}");

    if mode == "all" || mode == "baseline" {
        let (baseline_pair_count, baseline_used_first_ids, baseline_elapsed_seconds) =
            time_baseline(&tokenizer, &sampled_first_ids, &candidate_second_spines);
        print_timing(
            "baseline_hashmap",
            baseline_pair_count,
            baseline_used_first_ids,
            baseline_elapsed_seconds,
        );
    }

    if mode == "all" || mode == "preparedfirstdense" {
        let (
            pair_count,
            used_first_ids,
            canonical_count,
            elapsed_seconds,
            prepare_seconds,
            scan_seconds,
        ) = match time_prepared_first_dense_split(
            &tokenizer,
            &sampled_first_ids,
            candidate_second_buckets,
        ) {
            Ok(values) => values,
            Err(err) => {
                eprintln!("failed to time prepared-first dense mode: {err}");
                return ExitCode::from(1);
            }
        };
        let mismatches = if run_mismatch_checks {
            match count_mismatches_prepared_first_dense(
                &tokenizer,
                &sampled_first_ids,
                candidate_second_buckets,
            ) {
                Ok(mismatches) => mismatches,
                Err(err) => {
                    eprintln!("failed to compare prepared-first dense mode: {err}");
                    return ExitCode::from(1);
                }
            }
        } else {
            0
        };

        println!("prepared_first_dense:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!("  canonical_count = {canonical_count}");
        println!("  mismatches_vs_baseline = {mismatches}");
        print_timing("  timing", pair_count, used_first_ids, elapsed_seconds);
        println!("  prepare_seconds = {prepare_seconds:.6}");
        println!("  scan_seconds = {scan_seconds:.6}");
        println!(
            "  prepare_micros_per_first_token = {:.3}",
            prepare_seconds * 1_000_000.0 / used_first_ids as f64
        );
        println!(
            "  scan_micros_per_candidate = {:.3}",
            scan_seconds * 1_000_000.0 / pair_count as f64
        );
    }

    if mode == "all" || mode == "preparedfirstdense_scan" {
        let build_start = Instant::now();
        let prepared_first_batch =
            match build_prepared_first_dense_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let build_seconds = build_start.elapsed().as_secs_f64();
        let (pair_count, used_first_ids, canonical_count, elapsed_seconds) =
            time_prepared_first_dense_scan_only(&prepared_first_batch, candidate_second_buckets);

        println!("prepared_first_dense_scan:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!("  prepared_first_count = {}", prepared_first_batch.len());
        println!("  build_seconds = {build_seconds:.6}");
        println!("  canonical_count = {canonical_count}");
        print_timing("  timing", pair_count, used_first_ids, elapsed_seconds);
    }

    if mode == "preparedfirstdense_lockstep4_bybucket" {
        let prepared_first_batch =
            match build_prepared_first_dense_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!("  prepared_first_count = {}", prepared_first_batch.len());

        macro_rules! bench_bucket {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, scalar_canonical_count, scalar_seconds) =
                        time_bucket_scalar::<$left_len>(&prepared_first_batch, entries);
                    let (_lockstep_pair_count, lockstep_canonical_count, lockstep_seconds) =
                        time_bucket_lockstep4::<$left_len>(&prepared_first_batch, entries);
                    let mismatches =
                        count_bucket_lockstep4_mismatches::<$left_len>(&prepared_first_batch, entries);
                    print_bucket_timing(
                        $left_len,
                        pair_count,
                        scalar_canonical_count,
                        scalar_seconds,
                        lockstep_canonical_count,
                        lockstep_seconds,
                        mismatches,
                    );
                }
            }};
        }

        bench_bucket!(1);
        bench_bucket!(2);
        bench_bucket!(3);
        bench_bucket!(4);
        bench_bucket!(5);
        bench_bucket!(6);
        bench_bucket!(7);
        bench_bucket!(8);
    }
    ExitCode::SUCCESS
}
