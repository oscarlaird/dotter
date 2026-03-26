use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use bayesian::bpe::prepared_dense::{
    PreparedFirstDense, PreparedFirstDenseContiguousSwappedTight,
    PreparedFirstDenseContiguousSwappedTightAllPairs, PreparedSecondBuckets,
    PreparedSecondBucketsAllPairs, PreparedSecondSimd8Chunk, PreparedSecondToken,
    build_prepared_second_buckets_allpairs, build_prepared_second_simd8_chunks,
    canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len,
    canonical_pair_from_prepared_first_dense_left_len,
    count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small,
    count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4,
    count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8,
};
use bayesian::bpe::{MAX_PACKED_SPINE_LEN, TINYLLAMA_PIECE_COUNT, TinyLlamaWordTokenizer};

#[derive(Clone)]
struct PreparedForFirst {
    right_len: usize,
    scalar_ptr: PreparedFirstDense<TINYLLAMA_PIECE_COUNT>,
    swapped: PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>,
    allpairs: PreparedFirstDenseContiguousSwappedTightAllPairs<TINYLLAMA_PIECE_COUNT>,
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(tokenizer_path) = args.next() else {
        eprintln!(
            "usage: cargo run --release --bin hybrid_spine_bench -- <tokenizer.json> <mode> [iters]"
        );
        eprintln!(
            "modes: scalar_pointer_bybucket | scalar_swapped_bybucket | lockstep4_bybucket | simd8_bybucket | allpairs_small_matrix | allpairs_vs_scalar_pointer_matrix | no_partner_rate"
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

    let tokenizer = match TinyLlamaWordTokenizer::from_tokenizer_json(&tokenizer_path) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("failed to load tokenizer: {err}");
            return ExitCode::from(1);
        }
    };

    let prepared_second = tokenizer.prepared_second_buckets().clone();
    let prepared_second_allpairs = build_prepared_second_buckets_allpairs(&prepared_second);
    let simd8_chunks = build_all_simd8_chunks(&prepared_second);

    let prepared_first = build_prepared_first_for_all_tokens(&tokenizer);
    if prepared_first.is_empty() {
        eprintln!("no first-token candidates with right spines");
        return ExitCode::from(1);
    }

    match mode.as_str() {
        "scalar_pointer_bybucket" => run_scalar_pointer_bybucket(&prepared_first, &prepared_second, iters),
        "scalar_swapped_bybucket" => run_scalar_swapped_bybucket(&prepared_first, &prepared_second, iters),
        "lockstep4_bybucket" => run_lockstep4_bybucket(&prepared_first, &prepared_second, iters),
        "simd8_bybucket" => run_simd8_bybucket(&prepared_first, &prepared_second, &simd8_chunks, iters),
        "allpairs_small_matrix" => run_allpairs_small_matrix(
            &prepared_first,
            &prepared_second,
            &prepared_second_allpairs,
            iters,
        ),
        "allpairs_vs_scalar_pointer_matrix" => run_allpairs_vs_scalar_pointer_matrix(
            &prepared_first,
            &prepared_second,
            &prepared_second_allpairs,
            iters,
        ),
        "no_partner_rate" => run_no_partner_rate(&prepared_first, &prepared_second),
        _ => {
            eprintln!("unknown mode: {mode}");
            return ExitCode::from(2);
        }
    }

    ExitCode::SUCCESS
}

fn build_prepared_first_for_all_tokens(tokenizer: &TinyLlamaWordTokenizer) -> Vec<PreparedForFirst> {
    let mut out = Vec::new();
    for &first_token_id in tokenizer.token_ids_with_left_spines() {
        let Ok(Some(scalar_ptr)) = tokenizer.prepare_canonical_pair_batch_for_token_id(first_token_id) else {
            continue;
        };
        let Some(right_spine) = tokenizer.right_packed_spine_for_token_id(first_token_id) else {
            continue;
        };
        let Ok(swapped) = PreparedFirstDenseContiguousSwappedTight::<TINYLLAMA_PIECE_COUNT>::build(
            right_spine,
            tokenizer.prepared_merge_rows(),
        ) else {
            continue;
        };
        let Ok(allpairs) = PreparedFirstDenseContiguousSwappedTightAllPairs::<TINYLLAMA_PIECE_COUNT>::build(
            right_spine,
            tokenizer.prepared_merge_rows(),
        ) else {
            continue;
        };
        out.push(PreparedForFirst {
            right_len: right_spine.as_slice().len(),
            scalar_ptr,
            swapped,
            allpairs,
        });
    }
    out
}

fn build_all_simd8_chunks(
    prepared_second: &PreparedSecondBuckets,
) -> [Vec<PreparedSecondSimd8Chunk>; MAX_PACKED_SPINE_LEN + 1] {
    std::array::from_fn(|left_len| match left_len {
        1 => build_prepared_second_simd8_chunks::<1>(&prepared_second[1]),
        2 => build_prepared_second_simd8_chunks::<2>(&prepared_second[2]),
        3 => build_prepared_second_simd8_chunks::<3>(&prepared_second[3]),
        4 => build_prepared_second_simd8_chunks::<4>(&prepared_second[4]),
        5 => build_prepared_second_simd8_chunks::<5>(&prepared_second[5]),
        6 => build_prepared_second_simd8_chunks::<6>(&prepared_second[6]),
        7 => build_prepared_second_simd8_chunks::<7>(&prepared_second[7]),
        8 => build_prepared_second_simd8_chunks::<8>(&prepared_second[8]),
        _ => Vec::new(),
    })
}

fn run_scalar_pointer_bybucket(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
    iters: usize,
) {
    let started = Instant::now();
    let mut total = 0u64;
    for _ in 0..iters {
        for pf in prepared_first {
            total += scan_scalar_pointer_all_buckets(&pf.scalar_ptr, prepared_second);
        }
    }
    let elapsed = started.elapsed();
    println!("mode=scalar_pointer_bybucket");
    println!("first_tokens={}", prepared_first.len());
    println!("canonical_total={total}");
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
}

fn run_scalar_swapped_bybucket(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
    iters: usize,
) {
    let started = Instant::now();
    let mut total = 0u64;
    for _ in 0..iters {
        for pf in prepared_first {
            total += scan_scalar_swapped_all_buckets(&pf.swapped, prepared_second);
        }
    }
    let elapsed = started.elapsed();
    println!("mode=scalar_swapped_bybucket");
    println!("first_tokens={}", prepared_first.len());
    println!("canonical_total={total}");
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
}

fn run_lockstep4_bybucket(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
    iters: usize,
) {
    let mismatches = run_lockstep4_mismatch_check(prepared_first, prepared_second);
    let started = Instant::now();
    let mut total = 0u64;
    for _ in 0..iters {
        for pf in prepared_first {
            total += scan_lockstep4_all_buckets(&pf.swapped, prepared_second);
        }
    }
    let elapsed = started.elapsed();
    println!("mode=lockstep4_bybucket");
    println!("first_tokens={}", prepared_first.len());
    println!("mismatches={mismatches}");
    println!("canonical_total={total}");
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
}

fn run_simd8_bybucket(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
    simd8_chunks: &[Vec<PreparedSecondSimd8Chunk>; MAX_PACKED_SPINE_LEN + 1],
    iters: usize,
) {
    let mismatches = run_simd8_mismatch_check(prepared_first, prepared_second, simd8_chunks);
    let started = Instant::now();
    let mut total = 0u64;
    for _ in 0..iters {
        for pf in prepared_first {
            total += scan_simd8_all_buckets(&pf.swapped, prepared_second, simd8_chunks);
        }
    }
    let elapsed = started.elapsed();
    println!("mode=simd8_bybucket");
    println!("first_tokens={}", prepared_first.len());
    println!("mismatches={mismatches}");
    println!("canonical_total={total}");
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
}

fn run_allpairs_small_matrix(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
    prepared_second_allpairs: &PreparedSecondBucketsAllPairs,
    iters: usize,
) {
    let mut matrix = [[0u64; 9]; 9];
    let mut mismatches = 0u64;
    for pf in prepared_first {
        if !(2..=4).contains(&pf.right_len) {
            continue;
        }
        for left_len in 2..=4 {
            let entries = &prepared_second_allpairs[left_len];
            if entries.is_empty() {
                continue;
            }
            let local_mismatches = match (left_len, pf.right_len) {
                (2, 2) => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    2,
                    2,
                >(&pf.allpairs, &pf.swapped, entries),
                (2, 3) => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    2,
                    3,
                >(&pf.allpairs, &pf.swapped, entries),
                (2, 4) => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    2,
                    4,
                >(&pf.allpairs, &pf.swapped, entries),
                (3, 2) => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    3,
                    2,
                >(&pf.allpairs, &pf.swapped, entries),
                (3, 3) => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    3,
                    3,
                >(&pf.allpairs, &pf.swapped, entries),
                (3, 4) => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    3,
                    4,
                >(&pf.allpairs, &pf.swapped, entries),
                (4, 2) => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    4,
                    2,
                >(&pf.allpairs, &pf.swapped, entries),
                (4, 3) => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    4,
                    3,
                >(&pf.allpairs, &pf.swapped, entries),
                (4, 4) => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    4,
                    4,
                >(&pf.allpairs, &pf.swapped, entries),
                _ => 0,
            };
            mismatches += local_mismatches;
        }
    }

    let started = Instant::now();
    for _ in 0..iters {
        for pf in prepared_first {
            if !(2..=4).contains(&pf.right_len) {
                continue;
            }
            for left_len in 2..=4 {
                let entries = &prepared_second_allpairs[left_len];
                let add = match (left_len, pf.right_len) {
                    (2, 2) => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                        TINYLLAMA_PIECE_COUNT,
                        2,
                        2,
                    >(&pf.allpairs, entries),
                    (2, 3) => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                        TINYLLAMA_PIECE_COUNT,
                        2,
                        3,
                    >(&pf.allpairs, entries),
                    (2, 4) => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                        TINYLLAMA_PIECE_COUNT,
                        2,
                        4,
                    >(&pf.allpairs, entries),
                    (3, 2) => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                        TINYLLAMA_PIECE_COUNT,
                        3,
                        2,
                    >(&pf.allpairs, entries),
                    (3, 3) => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                        TINYLLAMA_PIECE_COUNT,
                        3,
                        3,
                    >(&pf.allpairs, entries),
                    (3, 4) => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                        TINYLLAMA_PIECE_COUNT,
                        3,
                        4,
                    >(&pf.allpairs, entries),
                    (4, 2) => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                        TINYLLAMA_PIECE_COUNT,
                        4,
                        2,
                    >(&pf.allpairs, entries),
                    (4, 3) => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                        TINYLLAMA_PIECE_COUNT,
                        4,
                        3,
                    >(&pf.allpairs, entries),
                    (4, 4) => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                        TINYLLAMA_PIECE_COUNT,
                        4,
                        4,
                    >(&pf.allpairs, entries),
                    _ => 0,
                };
                matrix[left_len][pf.right_len] += add;
            }
        }
    }
    let elapsed = started.elapsed();
    println!("mode=allpairs_small_matrix");
    println!("first_tokens={}", prepared_first.len());
    println!("mismatches={mismatches}");
    for left_len in 2..=4 {
        for right_len in 2..=4 {
            println!(
                "canonical_total[left_len={left_len},right_len={right_len}]={}",
                matrix[left_len][right_len]
            );
        }
    }
    let scalar_ref = run_lockstep4_reference_for_small_region(prepared_first, prepared_second);
    println!("lockstep4_reference_region_total={scalar_ref}");
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
}

fn run_allpairs_vs_scalar_pointer_matrix(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
    prepared_second_allpairs: &PreparedSecondBucketsAllPairs,
    iters: usize,
) {
    let mut scalar_ns = [[0.0f64; 9]; 9];
    let mut allpairs_ns = [[0.0f64; 9]; 9];
    let mut speedup = [[0.0f64; 9]; 9];
    let mut candidates = [[0u64; 9]; 9];
    let mut mismatches = [[0u64; 9]; 9];
    let mut scalar_totals = [[0u64; 9]; 9];
    let mut allpairs_totals = [[0u64; 9]; 9];

    for left_len in 1..=MAX_PACKED_SPINE_LEN {
        for right_len in 1..=MAX_PACKED_SPINE_LEN {
            let first_for_cell: Vec<&PreparedForFirst> = prepared_first
                .iter()
                .filter(|pf| pf.right_len == right_len)
                .collect();
            let right_count = first_for_cell.len() as u64;
            let left_count = prepared_second[left_len].len() as u64;
            let cand_count = right_count * left_count * iters as u64;
            candidates[left_len][right_len] = cand_count;
            if cand_count == 0 {
                continue;
            }

            let scalar_started = Instant::now();
            let mut scalar_total = 0u64;
            for _ in 0..iters {
                for pf in &first_for_cell {
                    scalar_total += scan_scalar_pointer_bucket_dispatch(&pf.scalar_ptr, left_len, prepared_second);
                }
            }
            let scalar_elapsed_ns = scalar_started.elapsed().as_secs_f64() * 1e9;
            scalar_total = black_box(scalar_total);

            let allpairs_started = Instant::now();
            let mut allpairs_total = 0u64;
            for _ in 0..iters {
                for pf in &first_for_cell {
                    allpairs_total += scan_allpairs_bucket_dispatch(
                        &pf.allpairs,
                        left_len,
                        right_len,
                        prepared_second_allpairs,
                    );
                }
            }
            let allpairs_elapsed_ns = allpairs_started.elapsed().as_secs_f64() * 1e9;
            allpairs_total = black_box(allpairs_total);

            let mismatch = first_for_cell
                .iter()
                .map(|pf| {
                    count_allpairs_mismatches_bucket_dispatch(
                        &pf.allpairs,
                        &pf.swapped,
                        left_len,
                        right_len,
                        prepared_second_allpairs,
                    )
                })
                .sum::<u64>();

            scalar_ns[left_len][right_len] = scalar_elapsed_ns / cand_count as f64;
            allpairs_ns[left_len][right_len] = allpairs_elapsed_ns / cand_count as f64;
            speedup[left_len][right_len] = scalar_ns[left_len][right_len] / allpairs_ns[left_len][right_len];
            mismatches[left_len][right_len] = mismatch;
            scalar_totals[left_len][right_len] = scalar_total;
            allpairs_totals[left_len][right_len] = allpairs_total;
        }
    }

    println!("mode=allpairs_vs_scalar_pointer_matrix");
    println!("iters={iters}");
    for left_len in 1..=MAX_PACKED_SPINE_LEN {
        for right_len in 1..=MAX_PACKED_SPINE_LEN {
            println!(
                "cell[left_len={left_len},right_len={right_len}] candidates={} scalar_ns_per_candidate={:.4} allpairs_ns_per_candidate={:.4} speedup={:.4} mismatches={}",
                candidates[left_len][right_len],
                scalar_ns[left_len][right_len],
                allpairs_ns[left_len][right_len],
                speedup[left_len][right_len],
                mismatches[left_len][right_len]
            );
            println!(
                "cell_totals[left_len={left_len},right_len={right_len}] scalar_total={} allpairs_total={}",
                scalar_totals[left_len][right_len],
                allpairs_totals[left_len][right_len]
            );
        }
    }
}

fn scan_scalar_pointer_bucket_dispatch(
    prepared_first: &PreparedFirstDense<TINYLLAMA_PIECE_COUNT>,
    left_len: usize,
    prepared_second: &PreparedSecondBuckets,
) -> u64 {
    match left_len {
        1 => scan_scalar_pointer_bucket::<1>(prepared_first, &prepared_second[1]),
        2 => scan_scalar_pointer_bucket::<2>(prepared_first, &prepared_second[2]),
        3 => scan_scalar_pointer_bucket::<3>(prepared_first, &prepared_second[3]),
        4 => scan_scalar_pointer_bucket::<4>(prepared_first, &prepared_second[4]),
        5 => scan_scalar_pointer_bucket::<5>(prepared_first, &prepared_second[5]),
        6 => scan_scalar_pointer_bucket::<6>(prepared_first, &prepared_second[6]),
        7 => scan_scalar_pointer_bucket::<7>(prepared_first, &prepared_second[7]),
        8 => scan_scalar_pointer_bucket::<8>(prepared_first, &prepared_second[8]),
        _ => 0,
    }
}

fn scan_allpairs_bucket_dispatch(
    prepared_first: &PreparedFirstDenseContiguousSwappedTightAllPairs<TINYLLAMA_PIECE_COUNT>,
    left_len: usize,
    right_len: usize,
    prepared_second_allpairs: &PreparedSecondBucketsAllPairs,
) -> u64 {
    macro_rules! row {
        ($l:literal) => {
            match right_len {
                1 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    1,
                >(prepared_first, &prepared_second_allpairs[$l]),
                2 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    2,
                >(prepared_first, &prepared_second_allpairs[$l]),
                3 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    3,
                >(prepared_first, &prepared_second_allpairs[$l]),
                4 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    4,
                >(prepared_first, &prepared_second_allpairs[$l]),
                5 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    5,
                >(prepared_first, &prepared_second_allpairs[$l]),
                6 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    6,
                >(prepared_first, &prepared_second_allpairs[$l]),
                7 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    7,
                >(prepared_first, &prepared_second_allpairs[$l]),
                8 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    8,
                >(prepared_first, &prepared_second_allpairs[$l]),
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

fn count_allpairs_mismatches_bucket_dispatch(
    prepared_first: &PreparedFirstDenseContiguousSwappedTightAllPairs<TINYLLAMA_PIECE_COUNT>,
    prepared_first_reference: &PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>,
    left_len: usize,
    right_len: usize,
    prepared_second_allpairs: &PreparedSecondBucketsAllPairs,
) -> u64 {
    macro_rules! row {
        ($l:literal) => {
            match right_len {
                1 => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    1,
                >(prepared_first, prepared_first_reference, &prepared_second_allpairs[$l]),
                2 => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    2,
                >(prepared_first, prepared_first_reference, &prepared_second_allpairs[$l]),
                3 => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    3,
                >(prepared_first, prepared_first_reference, &prepared_second_allpairs[$l]),
                4 => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    4,
                >(prepared_first, prepared_first_reference, &prepared_second_allpairs[$l]),
                5 => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    5,
                >(prepared_first, prepared_first_reference, &prepared_second_allpairs[$l]),
                6 => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    6,
                >(prepared_first, prepared_first_reference, &prepared_second_allpairs[$l]),
                7 => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    7,
                >(prepared_first, prepared_first_reference, &prepared_second_allpairs[$l]),
                8 => count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small::<
                    TINYLLAMA_PIECE_COUNT,
                    $l,
                    8,
                >(prepared_first, prepared_first_reference, &prepared_second_allpairs[$l]),
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

fn run_no_partner_rate(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
) {
    let mut piece_instances_total = 0u64;
    let mut piece_instances_no_partner = 0u64;
    let mut first_piece_total = 0u64;
    let mut first_piece_no_partner = 0u64;

    for pf in prepared_first {
        let right_len = pf.swapped.right_len();
        if right_len == 0 {
            continue;
        }
        let dense = pf.swapped.dense_matrix();
        for left_len in 1..=MAX_PACKED_SPINE_LEN {
            for entry in &prepared_second[left_len] {
                for depth in 0..left_len {
                    let left_id = entry.left_spine.ids[depth] as usize;
                    let row_start = left_id * right_len;
                    let row = &dense[row_start..row_start + right_len];
                    let has_partner = row.iter().any(|&x| x != 0);
                    piece_instances_total += 1;
                    if !has_partner {
                        piece_instances_no_partner += 1;
                    }
                    if depth == 0 {
                        first_piece_total += 1;
                        if !has_partner {
                            first_piece_no_partner += 1;
                        }
                    }
                }
            }
        }
    }

    let piece_rate = if piece_instances_total == 0 {
        0.0
    } else {
        piece_instances_no_partner as f64 / piece_instances_total as f64
    };
    let first_piece_rate = if first_piece_total == 0 {
        0.0
    } else {
        first_piece_no_partner as f64 / first_piece_total as f64
    };

    println!("mode=no_partner_rate");
    println!("first_tokens={}", prepared_first.len());
    println!("piece_instances_total={piece_instances_total}");
    println!("piece_instances_no_partner={piece_instances_no_partner}");
    println!("piece_instances_no_partner_rate={piece_rate:.6}");
    println!("first_piece_total={first_piece_total}");
    println!("first_piece_no_partner={first_piece_no_partner}");
    println!("first_piece_no_partner_rate={first_piece_rate:.6}");
}

fn run_lockstep4_reference_for_small_region(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
) -> u64 {
    let mut total = 0u64;
    for pf in prepared_first {
        if !(2..=4).contains(&pf.right_len) {
            continue;
        }
        for left_len in 2..=4 {
            total += scan_lockstep4_bucket_dispatch(&pf.swapped, left_len, &prepared_second[left_len]);
        }
    }
    total
}

fn scan_scalar_pointer_all_buckets(
    prepared_first: &PreparedFirstDense<TINYLLAMA_PIECE_COUNT>,
    prepared_second: &PreparedSecondBuckets,
) -> u64 {
    let mut total = 0u64;
    total += scan_scalar_pointer_bucket::<1>(prepared_first, &prepared_second[1]);
    total += scan_scalar_pointer_bucket::<2>(prepared_first, &prepared_second[2]);
    total += scan_scalar_pointer_bucket::<3>(prepared_first, &prepared_second[3]);
    total += scan_scalar_pointer_bucket::<4>(prepared_first, &prepared_second[4]);
    total += scan_scalar_pointer_bucket::<5>(prepared_first, &prepared_second[5]);
    total += scan_scalar_pointer_bucket::<6>(prepared_first, &prepared_second[6]);
    total += scan_scalar_pointer_bucket::<7>(prepared_first, &prepared_second[7]);
    total += scan_scalar_pointer_bucket::<8>(prepared_first, &prepared_second[8]);
    total
}

fn scan_scalar_pointer_bucket<const LEFT_LEN: usize>(
    prepared_first: &PreparedFirstDense<TINYLLAMA_PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut total = 0u64;
    for entry in entries {
        total += canonical_pair_from_prepared_first_dense_left_len::<TINYLLAMA_PIECE_COUNT, LEFT_LEN>(
            prepared_first,
            &entry.left_spine,
        ) as u64;
    }
    total
}

fn scan_scalar_swapped_all_buckets(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>,
    prepared_second: &PreparedSecondBuckets,
) -> u64 {
    let mut total = 0u64;
    total += scan_scalar_swapped_bucket::<1>(prepared_first, &prepared_second[1]);
    total += scan_scalar_swapped_bucket::<2>(prepared_first, &prepared_second[2]);
    total += scan_scalar_swapped_bucket::<3>(prepared_first, &prepared_second[3]);
    total += scan_scalar_swapped_bucket::<4>(prepared_first, &prepared_second[4]);
    total += scan_scalar_swapped_bucket::<5>(prepared_first, &prepared_second[5]);
    total += scan_scalar_swapped_bucket::<6>(prepared_first, &prepared_second[6]);
    total += scan_scalar_swapped_bucket::<7>(prepared_first, &prepared_second[7]);
    total += scan_scalar_swapped_bucket::<8>(prepared_first, &prepared_second[8]);
    total
}

fn scan_scalar_swapped_bucket<const LEFT_LEN: usize>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut total = 0u64;
    for entry in entries {
        total += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            TINYLLAMA_PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    total
}

fn scan_lockstep4_all_buckets(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>,
    prepared_second: &PreparedSecondBuckets,
) -> u64 {
    let mut total = 0u64;
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        1,
    >(prepared_first, &prepared_second[1]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        2,
    >(prepared_first, &prepared_second[2]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        3,
    >(prepared_first, &prepared_second[3]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        4,
    >(prepared_first, &prepared_second[4]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        5,
    >(prepared_first, &prepared_second[5]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        6,
    >(prepared_first, &prepared_second[6]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        7,
    >(prepared_first, &prepared_second[7]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        8,
    >(prepared_first, &prepared_second[8]);
    total
}

fn scan_lockstep4_bucket_dispatch(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>,
    left_len: usize,
    entries: &[PreparedSecondToken],
) -> u64 {
    match left_len {
        1 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<TINYLLAMA_PIECE_COUNT, 1>(
            prepared_first, entries,
        ),
        2 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<TINYLLAMA_PIECE_COUNT, 2>(
            prepared_first, entries,
        ),
        3 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<TINYLLAMA_PIECE_COUNT, 3>(
            prepared_first, entries,
        ),
        4 => scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<TINYLLAMA_PIECE_COUNT, 4>(
            prepared_first, entries,
        ),
        _ => 0,
    }
}

fn scan_simd8_all_buckets(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>,
    prepared_second: &PreparedSecondBuckets,
    simd8_chunks: &[Vec<PreparedSecondSimd8Chunk>; MAX_PACKED_SPINE_LEN + 1],
) -> u64 {
    let mut total = 0u64;
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
        TINYLLAMA_PIECE_COUNT,
        1,
    >(prepared_first, &prepared_second[1], &simd8_chunks[1]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
        TINYLLAMA_PIECE_COUNT,
        2,
    >(prepared_first, &prepared_second[2], &simd8_chunks[2]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
        TINYLLAMA_PIECE_COUNT,
        3,
    >(prepared_first, &prepared_second[3], &simd8_chunks[3]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
        TINYLLAMA_PIECE_COUNT,
        4,
    >(prepared_first, &prepared_second[4], &simd8_chunks[4]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
        TINYLLAMA_PIECE_COUNT,
        5,
    >(prepared_first, &prepared_second[5], &simd8_chunks[5]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
        TINYLLAMA_PIECE_COUNT,
        6,
    >(prepared_first, &prepared_second[6], &simd8_chunks[6]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
        TINYLLAMA_PIECE_COUNT,
        7,
    >(prepared_first, &prepared_second[7], &simd8_chunks[7]);
    total += scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
        TINYLLAMA_PIECE_COUNT,
        8,
    >(prepared_first, &prepared_second[8], &simd8_chunks[8]);
    total
}

fn run_lockstep4_mismatch_check(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
) -> u64 {
    let mut mismatches = 0u64;
    for pf in prepared_first {
        mismatches += count_mismatch_lockstep4_all_buckets(&pf.swapped, prepared_second);
    }
    mismatches
}

fn count_mismatch_lockstep4_all_buckets(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>,
    prepared_second: &PreparedSecondBuckets,
) -> u64 {
    let mut total = 0u64;
    total += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        1,
    >(prepared_first, &prepared_second[1]);
    total += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        2,
    >(prepared_first, &prepared_second[2]);
    total += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        3,
    >(prepared_first, &prepared_second[3]);
    total += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        4,
    >(prepared_first, &prepared_second[4]);
    total += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        5,
    >(prepared_first, &prepared_second[5]);
    total += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        6,
    >(prepared_first, &prepared_second[6]);
    total += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        7,
    >(prepared_first, &prepared_second[7]);
    total += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
        TINYLLAMA_PIECE_COUNT,
        8,
    >(prepared_first, &prepared_second[8]);
    total
}

fn run_simd8_mismatch_check(
    prepared_first: &[PreparedForFirst],
    prepared_second: &PreparedSecondBuckets,
    simd8_chunks: &[Vec<PreparedSecondSimd8Chunk>; MAX_PACKED_SPINE_LEN + 1],
) -> u64 {
    let mut mismatches = 0u64;
    for pf in prepared_first {
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
            TINYLLAMA_PIECE_COUNT,
            1,
        >(&pf.swapped, &prepared_second[1], &simd8_chunks[1]);
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
            TINYLLAMA_PIECE_COUNT,
            2,
        >(&pf.swapped, &prepared_second[2], &simd8_chunks[2]);
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
            TINYLLAMA_PIECE_COUNT,
            3,
        >(&pf.swapped, &prepared_second[3], &simd8_chunks[3]);
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
            TINYLLAMA_PIECE_COUNT,
            4,
        >(&pf.swapped, &prepared_second[4], &simd8_chunks[4]);
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
            TINYLLAMA_PIECE_COUNT,
            5,
        >(&pf.swapped, &prepared_second[5], &simd8_chunks[5]);
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
            TINYLLAMA_PIECE_COUNT,
            6,
        >(&pf.swapped, &prepared_second[6], &simd8_chunks[6]);
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
            TINYLLAMA_PIECE_COUNT,
            7,
        >(&pf.swapped, &prepared_second[7], &simd8_chunks[7]);
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
            TINYLLAMA_PIECE_COUNT,
            8,
        >(&pf.swapped, &prepared_second[8], &simd8_chunks[8]);
    }
    mismatches
}
