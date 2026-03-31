use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use bayesian::bpe::prepared_allpairs::{
    PreparedFirstAllPairs, PreparedSecondBuckets, scan_allpairs_small_bucket,
};
use bayesian::bpe::{MAX_PACKED_SPINE_LEN, NUM_TOKENS, TinyLlamaWordTokenizer};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(tokenizer_path) = args.next() else {
        eprintln!(
            "usage: cargo run --release --bin hybrid_spine_bench -- <tokenizer.json> <mode> [iters]"
        );
        eprintln!(
            "modes: allpairs_build_scan_split | allpairs_build_scan_split_filtered_ascii | allpairs_build_cell_stats | allpairs_per_first_tail"
        );
        return ExitCode::from(2);
    };
    let Some(mode) = args.next() else {
        eprintln!("missing mode");
        return ExitCode::from(2);
    };
    let iters = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8)
        .max(1);

    let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json(&tokenizer_path);

    let prepared_second = tokenizer.prepared_second_buckets().clone();

    match mode.as_str() {
        "allpairs_build_scan_split" => {
            run_allpairs_build_scan_split(&tokenizer, &prepared_second, iters)
        }
        "allpairs_build_scan_split_filtered_ascii" => {
            run_allpairs_build_scan_split_filtered_ascii(&tokenizer, &prepared_second, iters)
        }
        "allpairs_build_cell_stats" => run_allpairs_build_cell_stats(&tokenizer),
        "allpairs_per_first_tail" => run_allpairs_per_first_tail(&tokenizer, &prepared_second),
        _ => {
            eprintln!("unknown mode: {mode}");
            return ExitCode::from(2);
        }
    }

    ExitCode::SUCCESS
}

fn token_is_ascii_lower_or_space(token: &str) -> bool {
    token.bytes().all(|b| b == b' ' || b.is_ascii_lowercase())
}

fn filter_prepared_second_ascii(
    tokenizer: &TinyLlamaWordTokenizer,
    prepared_second: &PreparedSecondBuckets,
) -> PreparedSecondBuckets {
    std::array::from_fn(|left_len| {
        prepared_second[left_len]
            .iter()
            .copied()
            .filter(|entry| {
                let token = tokenizer.token_at(entry.lex_index);
                token_is_ascii_lower_or_space(token) && !token.contains(' ')
            })
            .collect()
    })
}

fn scan_bucket_dispatch(
    prepared_first: &PreparedFirstAllPairs<NUM_TOKENS>,
    left_len: usize,
    right_len: usize,
    prepared_second: &PreparedSecondBuckets,
) -> u64 {
    macro_rules! row {
        ($l:literal) => {
            match right_len {
                1 => scan_allpairs_small_bucket::<NUM_TOKENS, $l, 1>(
                    prepared_first,
                    &prepared_second[$l],
                ),
                2 => scan_allpairs_small_bucket::<NUM_TOKENS, $l, 2>(
                    prepared_first,
                    &prepared_second[$l],
                ),
                3 => scan_allpairs_small_bucket::<NUM_TOKENS, $l, 3>(
                    prepared_first,
                    &prepared_second[$l],
                ),
                4 => scan_allpairs_small_bucket::<NUM_TOKENS, $l, 4>(
                    prepared_first,
                    &prepared_second[$l],
                ),
                5 => scan_allpairs_small_bucket::<NUM_TOKENS, $l, 5>(
                    prepared_first,
                    &prepared_second[$l],
                ),
                6 => scan_allpairs_small_bucket::<NUM_TOKENS, $l, 6>(
                    prepared_first,
                    &prepared_second[$l],
                ),
                7 => scan_allpairs_small_bucket::<NUM_TOKENS, $l, 7>(
                    prepared_first,
                    &prepared_second[$l],
                ),
                8 => scan_allpairs_small_bucket::<NUM_TOKENS, $l, 8>(
                    prepared_first,
                    &prepared_second[$l],
                ),
                _ => 0,
            }
        };
    }
    match left_len {
        1 => row!(1),
        2 => row!(2),
        3 => row!(3),
        4 => row!(4),
        5 => row!(5),
        6 => row!(6),
        7 => row!(7),
        8 => row!(8),
        _ => 0,
    }
}

fn run_allpairs_build_scan_split(
    tokenizer: &TinyLlamaWordTokenizer,
    prepared_second: &PreparedSecondBuckets,
    iters: usize,
) {
    let mut build_elapsed_ns = 0f64;
    let mut scan_elapsed_ns = 0f64;
    let mut built_first_tokens = 0u64;
    let mut scan_candidates = 0u64;
    let mut canonical_total = 0u64;

    for _ in 0..iters {
        let mut reusable = PreparedFirstAllPairs::<NUM_TOKENS>::new_reusable();
        for &first_lex_index in tokenizer.lex_indices_with_left_spines() {
            let first_token_right_spine =
                tokenizer.right_packed_spine_for_lex_index(first_lex_index);
            let right_len = first_token_right_spine.as_slice().len();

            let build_started = Instant::now();
            reusable.rebuild_in_place(first_token_right_spine, tokenizer.prepared_merge_rows());
            build_elapsed_ns += build_started.elapsed().as_secs_f64() * 1e9;
            built_first_tokens += 1;

            let scan_started = Instant::now();
            for left_len in 1..=MAX_PACKED_SPINE_LEN {
                scan_candidates += prepared_second[left_len].len() as u64;
                canonical_total +=
                    scan_bucket_dispatch(&reusable, left_len, right_len, prepared_second);
            }
            scan_elapsed_ns += scan_started.elapsed().as_secs_f64() * 1e9;
        }
    }
    canonical_total = black_box(canonical_total);

    let build_us_per_first = if built_first_tokens == 0 {
        0.0
    } else {
        (build_elapsed_ns / 1000.0) / built_first_tokens as f64
    };
    let scan_us_per_first = if built_first_tokens == 0 {
        0.0
    } else {
        (scan_elapsed_ns / 1000.0) / built_first_tokens as f64
    };
    let total_us_per_first = build_us_per_first + scan_us_per_first;
    let scan_ns_per_candidate = if scan_candidates == 0 {
        0.0
    } else {
        scan_elapsed_ns / scan_candidates as f64
    };

    println!("mode=allpairs_build_scan_split");
    println!("iters={iters}");
    println!("first_tokens_built_total={built_first_tokens}");
    println!("scan_candidates_total={scan_candidates}");
    println!("canonical_total={canonical_total}");
    println!("build_us_per_first={build_us_per_first:.6}");
    println!("scan_us_per_first={scan_us_per_first:.6}");
    println!("total_us_per_first={total_us_per_first:.6}");
    println!("scan_ns_per_candidate={scan_ns_per_candidate:.6}");
}

fn run_allpairs_build_scan_split_filtered_ascii(
    tokenizer: &TinyLlamaWordTokenizer,
    prepared_second: &PreparedSecondBuckets,
    iters: usize,
) {
    let filtered_second = filter_prepared_second_ascii(tokenizer, prepared_second);

    let mut build_elapsed_ns = 0f64;
    let mut scan_elapsed_ns = 0f64;
    let mut built_first_tokens = 0u64;
    let mut scan_candidates = 0u64;
    let mut canonical_total = 0u64;

    for _ in 0..iters {
        let mut reusable = PreparedFirstAllPairs::<NUM_TOKENS>::new_reusable();
        for &first_lex_index in tokenizer.lex_indices_with_left_spines() {
            let first_token = tokenizer.token_at(first_lex_index);
            if !token_is_ascii_lower_or_space(first_token) {
                continue;
            }

            let first_token_right_spine =
                tokenizer.right_packed_spine_for_lex_index(first_lex_index);
            let right_len = first_token_right_spine.as_slice().len();

            let build_started = Instant::now();
            reusable.rebuild_in_place(first_token_right_spine, tokenizer.prepared_merge_rows());
            build_elapsed_ns += build_started.elapsed().as_secs_f64() * 1e9;
            built_first_tokens += 1;

            let scan_started = Instant::now();
            for left_len in 1..=MAX_PACKED_SPINE_LEN {
                scan_candidates += filtered_second[left_len].len() as u64;
                canonical_total +=
                    scan_bucket_dispatch(&reusable, left_len, right_len, &filtered_second);
            }
            scan_elapsed_ns += scan_started.elapsed().as_secs_f64() * 1e9;
        }
    }
    canonical_total = black_box(canonical_total);

    let build_us_per_first = if built_first_tokens == 0 {
        0.0
    } else {
        (build_elapsed_ns / 1000.0) / built_first_tokens as f64
    };
    let scan_us_per_first = if built_first_tokens == 0 {
        0.0
    } else {
        (scan_elapsed_ns / 1000.0) / built_first_tokens as f64
    };
    let total_us_per_first = build_us_per_first + scan_us_per_first;
    let scan_ns_per_candidate = if scan_candidates == 0 {
        0.0
    } else {
        scan_elapsed_ns / scan_candidates as f64
    };

    println!("mode=allpairs_build_scan_split_filtered_ascii");
    println!("iters={iters}");
    println!("first_tokens_built_total={built_first_tokens}");
    println!("scan_candidates_total={scan_candidates}");
    println!("canonical_total={canonical_total}");
    println!("build_us_per_first={build_us_per_first:.6}");
    println!("scan_us_per_first={scan_us_per_first:.6}");
    println!("total_us_per_first={total_us_per_first:.6}");
    println!("scan_ns_per_candidate={scan_ns_per_candidate:.6}");
}

fn run_allpairs_build_cell_stats(tokenizer: &TinyLlamaWordTokenizer) {
    let mut reusable = PreparedFirstAllPairs::<NUM_TOKENS>::new_reusable();
    let mut first_tokens = 0u64;
    let mut total_cells = 0u64;
    let mut min_cells = u64::MAX;
    let mut max_cells = 0u64;

    for &first_lex_index in tokenizer.lex_indices_with_left_spines() {
        let first_token_right_spine = tokenizer.right_packed_spine_for_lex_index(first_lex_index);
        reusable.rebuild_in_place(first_token_right_spine, tokenizer.prepared_merge_rows());
        let cells = reusable
            .row_partner_bitmap()
            .iter()
            .map(|&b| b.count_ones() as u64)
            .sum::<u64>();
        total_cells += cells;
        first_tokens += 1;
        if cells < min_cells {
            min_cells = cells;
        }
        if cells > max_cells {
            max_cells = cells;
        }
    }

    let avg_cells = if first_tokens == 0 {
        0.0
    } else {
        total_cells as f64 / first_tokens as f64
    };
    if first_tokens == 0 {
        min_cells = 0;
    }

    println!("mode=allpairs_build_cell_stats");
    println!("first_tokens={first_tokens}");
    println!("avg_cells_written_per_first={avg_cells:.6}");
    println!("min_cells_written_per_first={min_cells}");
    println!("max_cells_written_per_first={max_cells}");
}

fn run_allpairs_per_first_tail(
    tokenizer: &TinyLlamaWordTokenizer,
    prepared_second: &PreparedSecondBuckets,
) {
    let mut reusable = PreparedFirstAllPairs::<NUM_TOKENS>::new_reusable();

    let mut total_us = Vec::new();
    let mut build_us = Vec::new();
    let mut scan_us = Vec::new();
    let mut canonical_total = 0u64;
    let mut first_tokens = 0u64;

    for &first_lex_index in tokenizer.lex_indices_with_left_spines() {
        let first_token_right_spine = tokenizer.right_packed_spine_for_lex_index(first_lex_index);
        let right_len = first_token_right_spine.as_slice().len();

        let t0 = Instant::now();
        reusable.rebuild_in_place(first_token_right_spine, tokenizer.prepared_merge_rows());
        let build = t0.elapsed().as_secs_f64() * 1e6;

        let t1 = Instant::now();
        let mut local = 0u64;
        for left_len in 1..=MAX_PACKED_SPINE_LEN {
            local += scan_bucket_dispatch(&reusable, left_len, right_len, prepared_second);
        }
        let scan = t1.elapsed().as_secs_f64() * 1e6;

        canonical_total += black_box(local);
        build_us.push(build);
        scan_us.push(scan);
        total_us.push(build + scan);
        first_tokens += 1;
    }

    fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((v.len() - 1) as f64 * p).round() as usize;
        v[idx]
    }

    let p50 = percentile(total_us.clone(), 0.50);
    let p90 = percentile(total_us.clone(), 0.90);
    let p95 = percentile(total_us.clone(), 0.95);
    let p99 = percentile(total_us.clone(), 0.99);
    let p999 = percentile(total_us.clone(), 0.999);
    let max = total_us.iter().copied().fold(0.0f64, f64::max);
    let mean = if total_us.is_empty() {
        0.0
    } else {
        total_us.iter().sum::<f64>() / total_us.len() as f64
    };
    let build_mean = if build_us.is_empty() {
        0.0
    } else {
        build_us.iter().sum::<f64>() / build_us.len() as f64
    };
    let scan_mean = if scan_us.is_empty() {
        0.0
    } else {
        scan_us.iter().sum::<f64>() / scan_us.len() as f64
    };

    println!("mode=allpairs_per_first_tail");
    println!("first_tokens={first_tokens}");
    println!("canonical_total={canonical_total}");
    println!("build_us_mean={build_mean:.6}");
    println!("scan_us_mean={scan_mean:.6}");
    println!("total_us_mean={mean:.6}");
    println!("total_us_p50={p50:.6}");
    println!("total_us_p90={p90:.6}");
    println!("total_us_p95={p95:.6}");
    println!("total_us_p99={p99:.6}");
    println!("total_us_p999={p999:.6}");
    println!("total_us_max={max:.6}");
}
