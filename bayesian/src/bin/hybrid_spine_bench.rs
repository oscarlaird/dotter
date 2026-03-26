use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use bayesian::bpe::prepared_dense::{
    build_prefetch_left_id_chunks,
    build_prepared_second_simd4_chunks,
    build_prepared_second_simd8_chunks,
    build_prepared_second_simd16_chunks,
    canonical_pair_from_prepared_first_dense_left_len,
    count_mismatches_prepared_first_dense_bucket_lockstep4,
    count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch2,
    count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch,
    count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch,
    count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd4,
    count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8,
    count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd16,
    count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4,
    count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4,
    count_mismatches_prepared_first_dense_contiguous_bucket_lockstep4,
    scan_prepared_first_dense_bucket_lockstep4, canonical_pair_from_prepared_first_dense_contiguous_left_len,
    scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch2,
    scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch,
    scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd4,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd16,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch_prebuilt_param,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch_prebuilt,
    scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4,
    scan_prepared_first_dense_contiguous_bucket_lockstep4, PreparedFirstDenseContiguous,
    PrefetchConfig, PrefetchHint, PrefetchLeftIdChunk, PreparedSecondSimd4Chunk,
    PreparedSecondSimd8Chunk, PreparedSecondSimd16Chunk,
    PreparedFirstDenseContiguousSwapped, PreparedFirstDenseContiguousSwappedTight,
    PreparedSecondBuckets, PreparedSecondToken,
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

fn build_prepared_first_dense_contiguous_batch(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
) -> Result<Vec<PreparedFirstDenseContiguous<TINYLLAMA_PIECE_COUNT>>, bayesian::bpe::BpeError> {
    let mut prepared = Vec::with_capacity(sampled_first_ids.len());
    let merge_rows = tokenizer.prepared_merge_rows();
    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        prepared.push(
            PreparedFirstDenseContiguous::<TINYLLAMA_PIECE_COUNT>::build(first_right_spine, merge_rows)?,
        );
    }
    Ok(prepared)
}

fn build_prepared_first_dense_contiguous_batches_by_right_len(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
) -> Result<[Vec<PreparedFirstDenseContiguous<TINYLLAMA_PIECE_COUNT>>; 9], bayesian::bpe::BpeError> {
    let mut buckets: [Vec<PreparedFirstDenseContiguous<TINYLLAMA_PIECE_COUNT>>; 9] =
        std::array::from_fn(|_| Vec::new());
    let merge_rows = tokenizer.prepared_merge_rows();
    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        let right_len = first_right_spine.as_slice().len();
        if right_len > 8 {
            continue;
        }
        buckets[right_len]
            .push(PreparedFirstDenseContiguous::<TINYLLAMA_PIECE_COUNT>::build(first_right_spine, merge_rows)?);
    }
    Ok(buckets)
}

fn build_prepared_first_dense_contiguous_swapped_batch(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
) -> Result<Vec<PreparedFirstDenseContiguousSwapped<TINYLLAMA_PIECE_COUNT>>, bayesian::bpe::BpeError>
{
    let mut prepared = Vec::with_capacity(sampled_first_ids.len());
    let merge_rows = tokenizer.prepared_merge_rows();
    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        prepared.push(
            PreparedFirstDenseContiguousSwapped::<TINYLLAMA_PIECE_COUNT>::build(
                first_right_spine,
                merge_rows,
            )?,
        );
    }
    Ok(prepared)
}

fn build_prepared_first_dense_contiguous_swapped_tight_batch(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
) -> Result<Vec<PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>>, bayesian::bpe::BpeError>
{
    let mut prepared = Vec::with_capacity(sampled_first_ids.len());
    let merge_rows = tokenizer.prepared_merge_rows();
    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        prepared.push(
            PreparedFirstDenseContiguousSwappedTight::<TINYLLAMA_PIECE_COUNT>::build(
                first_right_spine,
                merge_rows,
            )?,
        );
    }
    Ok(prepared)
}

fn build_prepared_first_dense_contiguous_swapped_tight_batches_by_right_len(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
) -> Result<[Vec<PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>>; 9], bayesian::bpe::BpeError>
{
    let mut buckets: [Vec<PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>>; 9] =
        std::array::from_fn(|_| Vec::new());
    let merge_rows = tokenizer.prepared_merge_rows();
    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        let right_len = first_right_spine.as_slice().len();
        if right_len > 8 {
            continue;
        }
        buckets[right_len].push(
            PreparedFirstDenseContiguousSwappedTight::<TINYLLAMA_PIECE_COUNT>::build(
                first_right_spine,
                merge_rows,
            )?,
        );
    }
    Ok(buckets)
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

fn time_prepared_first_dense_scan_only_contiguous(
    prepared_first_batch: &[PreparedFirstDenseContiguous<TINYLLAMA_PIECE_COUNT>],
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
                    if black_box(canonical_pair_from_prepared_first_dense_contiguous_left_len::<
                        TINYLLAMA_PIECE_COUNT,
                        $left_len,
                    >(prepared_first, &second_entry.left_spine)) {
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

fn time_bucket_lockstep4_contiguous<const LEFT_LEN: usize>(
    prepared_first_contiguous_batch: &[PreparedFirstDenseContiguous<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_bucket_lockstep4::<TINYLLAMA_PIECE_COUNT, LEFT_LEN>(
                prepared_first,
                entries,
            ),
        );
    }

    (
        (prepared_first_contiguous_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_lockstep4_contiguous_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_batch: &[PreparedFirstDenseContiguous<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_batch {
        mismatches += count_mismatches_prepared_first_dense_contiguous_bucket_lockstep4::<
            TINYLLAMA_PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, entries);
    }
    mismatches
}

fn time_bucket_lockstep4_contiguous_swapped<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_batch: &[PreparedFirstDenseContiguousSwapped<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries),
        );
    }

    (
        (prepared_first_contiguous_swapped_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_lockstep4_contiguous_swapped_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_batch: &[PreparedFirstDenseContiguousSwapped<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_swapped_batch {
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4::<
            TINYLLAMA_PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, entries);
    }
    mismatches
}

fn time_bucket_lockstep4_contiguous_swapped_prefetch<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_batch: &[PreparedFirstDenseContiguousSwapped<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries),
        );
    }

    (
        (prepared_first_contiguous_swapped_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_lockstep4_contiguous_swapped_prefetch_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_batch: &[PreparedFirstDenseContiguousSwapped<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_swapped_batch {
        mismatches +=
            count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries);
    }
    mismatches
}

fn time_bucket_lockstep4_contiguous_swapped_prefetch2<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_batch: &[PreparedFirstDenseContiguousSwapped<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch2::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries),
        );
    }

    (
        (prepared_first_contiguous_swapped_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_lockstep4_contiguous_swapped_prefetch2_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_batch: &[PreparedFirstDenseContiguousSwapped<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_swapped_batch {
        mismatches +=
            count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch2::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries);
    }
    mismatches
}

fn time_bucket_lockstep4_contiguous_swapped_tight<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries),
        );
    }

    (
        (prepared_first_contiguous_swapped_tight_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_lockstep4_contiguous_swapped_tight_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        mismatches += count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4::<
            TINYLLAMA_PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, entries);
    }
    mismatches
}

fn time_bucket_lockstep4_contiguous_swapped_tight_prefetch<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries),
        );
    }

    (
        (prepared_first_contiguous_swapped_tight_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn time_bucket_lockstep4_contiguous_swapped_tight_prefetch_dedupe<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
    prefetch_chunks: &[PrefetchLeftIdChunk],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch_prebuilt::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries, prefetch_chunks),
        );
    }

    (
        (prepared_first_contiguous_swapped_tight_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn time_bucket_lockstep4_contiguous_swapped_tight_prefetch_param<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
    prefetch_chunks: &[PrefetchLeftIdChunk],
    config: PrefetchConfig,
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch_prebuilt_param::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries, prefetch_chunks, config),
        );
    }

    (
        (prepared_first_contiguous_swapped_tight_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_lockstep4_contiguous_swapped_tight_prefetch_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        mismatches +=
            count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries);
    }
    mismatches
}

fn time_bucket_simd8_contiguous_swapped_tight<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd8Chunk],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries, simd_chunks),
        );
    }

    (
        (prepared_first_contiguous_swapped_tight_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn time_bucket_simd4_contiguous_swapped_tight<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd4Chunk],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd4::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries, simd_chunks),
        );
    }

    (
        (prepared_first_contiguous_swapped_tight_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_simd8_contiguous_swapped_tight_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd8Chunk],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        mismatches +=
            count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries, simd_chunks);
    }
    mismatches
}

fn count_bucket_simd4_contiguous_swapped_tight_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd4Chunk],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        mismatches +=
            count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd4::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries, simd_chunks);
    }
    mismatches
}

fn time_bucket_simd16_contiguous_swapped_tight<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd16Chunk],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        canonical_count += black_box(
            scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd16::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries, simd_chunks),
        );
    }

    (
        (prepared_first_contiguous_swapped_tight_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_simd16_contiguous_swapped_tight_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_tight_batch: &[PreparedFirstDenseContiguousSwappedTight<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd16Chunk],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_swapped_tight_batch {
        mismatches +=
            count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd16::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries, simd_chunks);
    }
    mismatches
}

#[inline(always)]
fn use_prefetch2_for_left_len(left_len: usize) -> bool {
    matches!(left_len, 1 | 2 | 4 | 5)
}

fn time_bucket_lockstep4_contiguous_swapped_prefetch_best<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_batch: &[PreparedFirstDenseContiguousSwapped<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> (u64, u64, f64) {
    let timed_start = Instant::now();
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_contiguous_swapped_batch {
        canonical_count += if use_prefetch2_for_left_len(LEFT_LEN) {
            black_box(
                scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch2::<
                    TINYLLAMA_PIECE_COUNT,
                    LEFT_LEN,
                >(prepared_first, entries),
            )
        } else {
            black_box(
                scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch::<
                    TINYLLAMA_PIECE_COUNT,
                    LEFT_LEN,
                >(prepared_first, entries),
            )
        };
    }

    (
        (prepared_first_contiguous_swapped_batch.len() * entries.len()) as u64,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_bucket_lockstep4_contiguous_swapped_prefetch_best_mismatches<const LEFT_LEN: usize>(
    prepared_first_contiguous_swapped_batch: &[PreparedFirstDenseContiguousSwapped<TINYLLAMA_PIECE_COUNT>],
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    for prepared_first in prepared_first_contiguous_swapped_batch {
        mismatches += if use_prefetch2_for_left_len(LEFT_LEN) {
            count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch2::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries)
        } else {
            count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch::<
                TINYLLAMA_PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, entries)
        };
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

fn print_first_token_spine_len_histogram(
    tokenizer: &TinyLlamaWordTokenizer,
    sampled_first_ids: &[u32],
) {
    let mut right_hist = [0u64; 9];
    let mut left_hist = [0u64; 9];
    let mut used = 0u64;

    for &first_id in sampled_first_ids {
        let Some(right) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        let Some(left) = tokenizer.left_packed_spine_for_token_id(first_id) else {
            continue;
        };
        let right_len = right.as_slice().len();
        let left_len = left.as_slice().len();
        if right_len <= 8 {
            right_hist[right_len] += 1;
        }
        if left_len <= 8 {
            left_hist[left_len] += 1;
        }
        used += 1;
    }

    println!("first_token_spine_len_histogram:");
    println!("  used_first_ids = {used}");
    println!("  right_len:");
    for len in 0..=8 {
        println!("    len={len}: {}", right_hist[len]);
    }
    println!("  left_len:");
    for len in 0..=8 {
        println!("    len={len}: {}", left_hist[len]);
    }
}

fn print_prefetch_dedup_overlap_for_bucket<const LEFT_LEN: usize>(entries: &[PreparedSecondToken]) {
    const CHUNK_WIDTH: usize = 4;
    if entries.len() < CHUNK_WIDTH {
        println!("  left_len = {LEFT_LEN}");
        println!("    chunk_count = 0");
        return;
    }

    let chunk_count = entries.len() / CHUNK_WIDTH;
    let mut naive_prefetches = 0u64;
    let mut dedup_prefetches = 0u64;
    let mut min_unique = usize::MAX;
    let mut max_unique = 0usize;

    for chunk_idx in 0..chunk_count {
        let base = chunk_idx * CHUNK_WIDTH;
        let chunk: &[PreparedSecondToken; CHUNK_WIDTH] = entries[base..base + CHUNK_WIDTH]
            .try_into()
            .expect("chunk must contain 4 lanes");

        let mut unique_ids: [u16; 32] = [0; 32];
        let mut unique_count = 0usize;

        for entry in chunk {
            debug_assert_eq!(entry.left_spine.len as usize, LEFT_LEN);
            for depth in 0..LEFT_LEN {
                let id = entry.left_spine.ids[depth];
                let mut seen = false;
                for &existing in &unique_ids[..unique_count] {
                    if existing == id {
                        seen = true;
                        break;
                    }
                }
                if !seen {
                    unique_ids[unique_count] = id;
                    unique_count += 1;
                }
            }
        }

        let naive = CHUNK_WIDTH * LEFT_LEN;
        naive_prefetches += naive as u64;
        dedup_prefetches += unique_count as u64;
        min_unique = min_unique.min(unique_count);
        max_unique = max_unique.max(unique_count);
    }

    let reduction =
        1.0 - (dedup_prefetches as f64 / naive_prefetches as f64);
    let avg_unique = dedup_prefetches as f64 / chunk_count as f64;
    let avg_naive = naive_prefetches as f64 / chunk_count as f64;

    println!("  left_len = {LEFT_LEN}");
    println!("    chunk_count = {chunk_count}");
    println!("    avg_unique_ids_per_chunk = {avg_unique:.6}");
    println!("    naive_ids_per_chunk = {avg_naive:.6}");
    println!("    unique_min_per_chunk = {min_unique}");
    println!("    unique_max_per_chunk = {max_unique}");
    println!("    dedup_prefetch_reduction = {:.6}%", reduction * 100.0);
}

fn build_prefetch_chunks_by_left_len(
    candidate_second_buckets: &PreparedSecondBuckets,
) -> [Vec<PrefetchLeftIdChunk>; 9] {
    let mut chunks: [Vec<PrefetchLeftIdChunk>; 9] = std::array::from_fn(|_| Vec::new());
    chunks[1] = build_prefetch_left_id_chunks::<1>(&candidate_second_buckets[1]);
    chunks[2] = build_prefetch_left_id_chunks::<2>(&candidate_second_buckets[2]);
    chunks[3] = build_prefetch_left_id_chunks::<3>(&candidate_second_buckets[3]);
    chunks[4] = build_prefetch_left_id_chunks::<4>(&candidate_second_buckets[4]);
    chunks[5] = build_prefetch_left_id_chunks::<5>(&candidate_second_buckets[5]);
    chunks[6] = build_prefetch_left_id_chunks::<6>(&candidate_second_buckets[6]);
    chunks[7] = build_prefetch_left_id_chunks::<7>(&candidate_second_buckets[7]);
    chunks[8] = build_prefetch_left_id_chunks::<8>(&candidate_second_buckets[8]);
    chunks
}

fn build_simd8_chunks_by_left_len(
    candidate_second_buckets: &PreparedSecondBuckets,
) -> [Vec<PreparedSecondSimd8Chunk>; 9] {
    let mut chunks: [Vec<PreparedSecondSimd8Chunk>; 9] = std::array::from_fn(|_| Vec::new());
    chunks[1] = build_prepared_second_simd8_chunks::<1>(&candidate_second_buckets[1]);
    chunks[2] = build_prepared_second_simd8_chunks::<2>(&candidate_second_buckets[2]);
    chunks[3] = build_prepared_second_simd8_chunks::<3>(&candidate_second_buckets[3]);
    chunks[4] = build_prepared_second_simd8_chunks::<4>(&candidate_second_buckets[4]);
    chunks[5] = build_prepared_second_simd8_chunks::<5>(&candidate_second_buckets[5]);
    chunks[6] = build_prepared_second_simd8_chunks::<6>(&candidate_second_buckets[6]);
    chunks[7] = build_prepared_second_simd8_chunks::<7>(&candidate_second_buckets[7]);
    chunks[8] = build_prepared_second_simd8_chunks::<8>(&candidate_second_buckets[8]);
    chunks
}

fn build_simd4_chunks_by_left_len(
    candidate_second_buckets: &PreparedSecondBuckets,
) -> [Vec<PreparedSecondSimd4Chunk>; 9] {
    let mut chunks: [Vec<PreparedSecondSimd4Chunk>; 9] = std::array::from_fn(|_| Vec::new());
    chunks[1] = build_prepared_second_simd4_chunks::<1>(&candidate_second_buckets[1]);
    chunks[2] = build_prepared_second_simd4_chunks::<2>(&candidate_second_buckets[2]);
    chunks[3] = build_prepared_second_simd4_chunks::<3>(&candidate_second_buckets[3]);
    chunks[4] = build_prepared_second_simd4_chunks::<4>(&candidate_second_buckets[4]);
    chunks[5] = build_prepared_second_simd4_chunks::<5>(&candidate_second_buckets[5]);
    chunks[6] = build_prepared_second_simd4_chunks::<6>(&candidate_second_buckets[6]);
    chunks[7] = build_prepared_second_simd4_chunks::<7>(&candidate_second_buckets[7]);
    chunks[8] = build_prepared_second_simd4_chunks::<8>(&candidate_second_buckets[8]);
    chunks
}

fn build_simd16_chunks_by_left_len(
    candidate_second_buckets: &PreparedSecondBuckets,
) -> [Vec<PreparedSecondSimd16Chunk>; 9] {
    let mut chunks: [Vec<PreparedSecondSimd16Chunk>; 9] = std::array::from_fn(|_| Vec::new());
    chunks[1] = build_prepared_second_simd16_chunks::<1>(&candidate_second_buckets[1]);
    chunks[2] = build_prepared_second_simd16_chunks::<2>(&candidate_second_buckets[2]);
    chunks[3] = build_prepared_second_simd16_chunks::<3>(&candidate_second_buckets[3]);
    chunks[4] = build_prepared_second_simd16_chunks::<4>(&candidate_second_buckets[4]);
    chunks[5] = build_prepared_second_simd16_chunks::<5>(&candidate_second_buckets[5]);
    chunks[6] = build_prepared_second_simd16_chunks::<6>(&candidate_second_buckets[6]);
    chunks[7] = build_prepared_second_simd16_chunks::<7>(&candidate_second_buckets[7]);
    chunks[8] = build_prepared_second_simd16_chunks::<8>(&candidate_second_buckets[8]);
    chunks
}

fn prefetch_hint_name(hint: PrefetchHint) -> &'static str {
    match hint {
        PrefetchHint::T0 => "t0",
        PrefetchHint::T1 => "t1",
        PrefetchHint::T2 => "t2",
        PrefetchHint::Nta => "nta",
    }
}

fn median_f64(values: &mut [f64]) -> f64 {
    debug_assert!(!values.is_empty());
    values.sort_by(|a, b| a.partial_cmp(b).expect("timings should not be NaN"));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) * 0.5
    }
}

fn shuffle_usize(values: &mut [usize], rng: &mut XorShift64) {
    if values.len() < 2 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = rng.gen_index(i + 1);
        values.swap(i, j);
    }
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

    if mode == "firsttoken_len_hist" {
        print_first_token_spine_len_histogram(&tokenizer, &sampled_first_ids);
    }

    if mode == "firsttoken_leftlen_rightlen_matrix" {
        let mut matrix = [[0u64; 9]; 9];
        let mut used = 0u64;
        let all_ids: Vec<u32> = (0..TINYLLAMA_PIECE_COUNT as u32).collect();
        for &id in &all_ids {
            let Some(right) = tokenizer.right_packed_spine_for_token_id(id) else {
                continue;
            };
            let Some(left) = tokenizer.left_packed_spine_for_token_id(id) else {
                continue;
            };
            let r = right.as_slice().len().min(8);
            let l = left.as_slice().len().min(8);
            matrix[l][r] += 1;
            used += 1;
        }
        println!("first_token_left_len_vs_right_len_matrix:");
        println!("  total_tokens_with_both_spines = {used}");
        print!("  {:>8}", "");
        for r in 0..=8 {
            print!("  r={r:>3}");
        }
        println!();
        for l in 0..=8 {
            print!("  l={l:<5}");
            for r in 0..=8 {
                let pct = if used > 0 {
                    matrix[l][r] as f64 / used as f64 * 100.0
                } else {
                    0.0
                };
                print!("  {:>5.1}%", pct);
            }
            println!("    (count: {})", (0..=8).map(|r| matrix[l][r]).sum::<u64>());
        }
    }

    if mode == "prefetch_dedup_overlap_bybucket" {
        println!("prefetch_dedup_overlap_bybucket:");
        println!("  chunk_width = 4");
        println!("  measures dedup potential across second-token lanes");
        print_prefetch_dedup_overlap_for_bucket::<1>(&candidate_second_buckets[1]);
        print_prefetch_dedup_overlap_for_bucket::<2>(&candidate_second_buckets[2]);
        print_prefetch_dedup_overlap_for_bucket::<3>(&candidate_second_buckets[3]);
        print_prefetch_dedup_overlap_for_bucket::<4>(&candidate_second_buckets[4]);
        print_prefetch_dedup_overlap_for_bucket::<5>(&candidate_second_buckets[5]);
        print_prefetch_dedup_overlap_for_bucket::<6>(&candidate_second_buckets[6]);
        print_prefetch_dedup_overlap_for_bucket::<7>(&candidate_second_buckets[7]);
        print_prefetch_dedup_overlap_for_bucket::<8>(&candidate_second_buckets[8]);
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

    if mode == "preparedfirstdense_scan_contiguous" {
        let build_start = Instant::now();
        let prepared_first_batch =
            match build_prepared_first_dense_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild standard prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let standard_build_seconds = build_start.elapsed().as_secs_f64();
        let (standard_pair_count, standard_used_first_ids, standard_canonical_count, standard_elapsed_seconds) =
            time_prepared_first_dense_scan_only(&prepared_first_batch, candidate_second_buckets);

        let contig_build_start = Instant::now();
        let prepared_first_contiguous_batch =
            match build_prepared_first_dense_contiguous_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let contiguous_build_seconds = contig_build_start.elapsed().as_secs_f64();
        let (contig_pair_count, contig_used_first_ids, contig_canonical_count, contig_elapsed_seconds) =
            time_prepared_first_dense_scan_only_contiguous(
                &prepared_first_contiguous_batch,
                candidate_second_buckets,
            );

        println!("prepared_first_dense_scan_contiguous:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!("  standard_prepared_first_count = {}", prepared_first_batch.len());
        println!(
            "  contiguous_prepared_first_count = {}",
            prepared_first_contiguous_batch.len()
        );
        println!("  standard_build_seconds = {standard_build_seconds:.6}");
        println!("  contiguous_build_seconds = {contiguous_build_seconds:.6}");
        println!("  standard_canonical_count = {standard_canonical_count}");
        println!("  contiguous_canonical_count = {contig_canonical_count}");
        print_timing(
            "  standard_scan_timing",
            standard_pair_count,
            standard_used_first_ids,
            standard_elapsed_seconds,
        );
        print_timing(
            "  contiguous_scan_timing",
            contig_pair_count,
            contig_used_first_ids,
            contig_elapsed_seconds,
        );
        println!(
            "  contiguous_vs_standard_scan_speedup = {:.3}x",
            standard_elapsed_seconds / contig_elapsed_seconds
        );
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

    if mode == "preparedfirstdense_lockstep4_contiguous_bybucket" {
        let prepared_first_batch =
            match build_prepared_first_dense_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_batch =
            match build_prepared_first_dense_contiguous_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_contiguous_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!("  prepared_first_count = {}", prepared_first_batch.len());
        println!(
            "  contiguous_prepared_first_count = {}",
            prepared_first_contiguous_batch.len()
        );

        macro_rules! bench_bucket_contiguous {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, lockstep_canonical_count, lockstep_seconds) =
                        time_bucket_lockstep4::<$left_len>(&prepared_first_batch, entries);
                    let (_contig_pair_count, contiguous_lockstep_canonical_count, contiguous_lockstep_seconds) =
                        time_bucket_lockstep4_contiguous::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let lockstep_mismatches =
                        count_bucket_lockstep4_mismatches::<$left_len>(&prepared_first_batch, entries);
                    let contiguous_lockstep_mismatches =
                        count_bucket_lockstep4_contiguous_mismatches::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    lockstep_canonical_count = {lockstep_canonical_count}");
                    println!(
                        "    contiguous_lockstep_canonical_count = {contiguous_lockstep_canonical_count}"
                    );
                    println!("    lockstep_mismatches = {lockstep_mismatches}");
                    println!(
                        "    contiguous_lockstep_mismatches = {contiguous_lockstep_mismatches}"
                    );
                    println!(
                        "    lockstep4_micros_per_candidate = {:.3}",
                        lockstep_seconds * 1_000_000.0 / pair_count as f64
                    );
                    println!(
                        "    contiguous_lockstep4_micros_per_candidate = {:.3}",
                        contiguous_lockstep_seconds * 1_000_000.0 / pair_count as f64
                    );
                    println!(
                        "    contiguous_vs_lockstep_speedup = {:.3}x",
                        lockstep_seconds / contiguous_lockstep_seconds
                    );
                }
            }};
        }

        bench_bucket_contiguous!(1);
        bench_bucket_contiguous!(2);
        bench_bucket_contiguous!(3);
        bench_bucket_contiguous!(4);
        bench_bucket_contiguous!(5);
        bench_bucket_contiguous!(6);
        bench_bucket_contiguous!(7);
        bench_bucket_contiguous!(8);
    }

    if mode == "preparedfirstdense_lockstep4_contiguous_swapped_bybucket" {
        let prepared_first_contiguous_batch =
            match build_prepared_first_dense_contiguous_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_swapped_batch =
            match build_prepared_first_dense_contiguous_swapped_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_contiguous_swapped_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  contiguous_prepared_first_count = {}",
            prepared_first_contiguous_batch.len()
        );
        println!(
            "  swapped_contiguous_prepared_first_count = {}",
            prepared_first_contiguous_swapped_batch.len()
        );

        macro_rules! bench_bucket_swapped {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, contiguous_lockstep_canonical_count, contiguous_lockstep_seconds) =
                        time_bucket_lockstep4_contiguous::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let (
                        _swapped_pair_count,
                        swapped_lockstep_canonical_count,
                        swapped_lockstep_seconds,
                    ) = time_bucket_lockstep4_contiguous_swapped::<$left_len>(
                        &prepared_first_contiguous_swapped_batch,
                        entries,
                    );
                    let contiguous_lockstep_mismatches =
                        count_bucket_lockstep4_contiguous_mismatches::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let swapped_lockstep_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!(
                        "    contiguous_lockstep_canonical_count = {contiguous_lockstep_canonical_count}"
                    );
                    println!(
                        "    swapped_lockstep_canonical_count = {swapped_lockstep_canonical_count}"
                    );
                    println!(
                        "    contiguous_lockstep_mismatches = {contiguous_lockstep_mismatches}"
                    );
                    println!("    swapped_lockstep_mismatches = {swapped_lockstep_mismatches}");
                    println!(
                        "    contiguous_lockstep4_micros_per_candidate = {:.3}",
                        contiguous_lockstep_seconds * 1_000_000.0 / pair_count as f64
                    );
                    println!(
                        "    swapped_lockstep4_micros_per_candidate = {:.3}",
                        swapped_lockstep_seconds * 1_000_000.0 / pair_count as f64
                    );
                    println!(
                        "    swapped_vs_contiguous_speedup = {:.3}x",
                        contiguous_lockstep_seconds / swapped_lockstep_seconds
                    );
                }
            }};
        }

        bench_bucket_swapped!(1);
        bench_bucket_swapped!(2);
        bench_bucket_swapped!(3);
        bench_bucket_swapped!(4);
        bench_bucket_swapped!(5);
        bench_bucket_swapped!(6);
        bench_bucket_swapped!(7);
        bench_bucket_swapped!(8);
    }

    if mode == "preparedfirstdense_lockstep4_swapped_prefetch_bybucket" {
        let prepared_first_contiguous_swapped_batch =
            match build_prepared_first_dense_contiguous_swapped_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_swapped_prefetch_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  swapped_contiguous_prepared_first_count = {}",
            prepared_first_contiguous_swapped_batch.len()
        );

        macro_rules! bench_bucket_swapped_prefetch {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, swapped_lockstep_canonical_count, swapped_lockstep_seconds) =
                        time_bucket_lockstep4_contiguous_swapped::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let (
                        _prefetch_pair_count,
                        swapped_prefetch_canonical_count,
                        swapped_prefetch_seconds,
                    ) = time_bucket_lockstep4_contiguous_swapped_prefetch::<$left_len>(
                        &prepared_first_contiguous_swapped_batch,
                        entries,
                    );
                    let swapped_lockstep_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let swapped_prefetch_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_prefetch_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!(
                        "    swapped_lockstep_canonical_count = {swapped_lockstep_canonical_count}"
                    );
                    println!(
                        "    swapped_prefetch_canonical_count = {swapped_prefetch_canonical_count}"
                    );
                    println!(
                        "    swapped_lockstep_mismatches = {swapped_lockstep_mismatches}"
                    );
                    println!("    swapped_prefetch_mismatches = {swapped_prefetch_mismatches}");
                    println!(
                        "    swapped_lockstep4_micros_per_candidate = {:.3}",
                        swapped_lockstep_seconds * 1_000_000.0 / pair_count as f64
                    );
                    println!(
                        "    swapped_prefetch_lockstep4_micros_per_candidate = {:.3}",
                        swapped_prefetch_seconds * 1_000_000.0 / pair_count as f64
                    );
                    println!(
                        "    swapped_prefetch_vs_swapped_speedup = {:.3}x",
                        swapped_lockstep_seconds / swapped_prefetch_seconds
                    );
                }
            }};
        }

        bench_bucket_swapped_prefetch!(1);
        bench_bucket_swapped_prefetch!(2);
        bench_bucket_swapped_prefetch!(3);
        bench_bucket_swapped_prefetch!(4);
        bench_bucket_swapped_prefetch!(5);
        bench_bucket_swapped_prefetch!(6);
        bench_bucket_swapped_prefetch!(7);
        bench_bucket_swapped_prefetch!(8);
    }

    if mode == "preparedfirstdense_lockstep4_swapped_prefetch1_vs_2_bybucket" {
        let prepared_first_contiguous_swapped_batch =
            match build_prepared_first_dense_contiguous_swapped_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_swapped_prefetch1_vs_2_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  swapped_contiguous_prepared_first_count = {}",
            prepared_first_contiguous_swapped_batch.len()
        );

        macro_rules! bench_bucket_prefetch12 {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, prefetch1_canonical_count, prefetch1_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_prefetch::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let (_prefetch2_pair_count, prefetch2_canonical_count, prefetch2_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_prefetch2::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let prefetch1_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_prefetch_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let prefetch2_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_prefetch2_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    prefetch1_canonical_count = {prefetch1_canonical_count}");
                    println!("    prefetch2_canonical_count = {prefetch2_canonical_count}");
                    println!("    prefetch1_mismatches = {prefetch1_mismatches}");
                    println!("    prefetch2_mismatches = {prefetch2_mismatches}");
                    println!("    prefetch1_seconds = {prefetch1_seconds:.12}");
                    println!("    prefetch2_seconds = {prefetch2_seconds:.12}");
                    println!(
                        "    prefetch2_vs_prefetch1_speedup = {:.12}x",
                        prefetch1_seconds / prefetch2_seconds
                    );
                }
            }};
        }

        bench_bucket_prefetch12!(1);
        bench_bucket_prefetch12!(2);
        bench_bucket_prefetch12!(3);
        bench_bucket_prefetch12!(4);
        bench_bucket_prefetch12!(5);
        bench_bucket_prefetch12!(6);
        bench_bucket_prefetch12!(7);
        bench_bucket_prefetch12!(8);
    }

    if mode == "preparedfirstdense_lockstep4_swapped_prefetchbest_vs_plain_bybucket" {
        let prepared_first_contiguous_batch =
            match build_prepared_first_dense_contiguous_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_swapped_batch =
            match build_prepared_first_dense_contiguous_swapped_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_swapped_prefetchbest_vs_plain_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  contiguous_prepared_first_count = {}",
            prepared_first_contiguous_batch.len()
        );
        println!(
            "  swapped_contiguous_prepared_first_count = {}",
            prepared_first_contiguous_swapped_batch.len()
        );
        println!("  prefetch2_buckets = [1,2,4,5]");
        println!("  prefetch1_buckets = [3,6,7,8]");

        macro_rules! bench_bucket_prefetchbest {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, plain_canonical_count, plain_seconds) =
                        time_bucket_lockstep4_contiguous::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let (
                        _prefetchbest_pair_count,
                        prefetchbest_canonical_count,
                        prefetchbest_seconds,
                    ) = time_bucket_lockstep4_contiguous_swapped_prefetch_best::<$left_len>(
                        &prepared_first_contiguous_swapped_batch,
                        entries,
                    );
                    let plain_mismatches =
                        count_bucket_lockstep4_contiguous_mismatches::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let prefetchbest_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_prefetch_best_mismatches::<
                            $left_len,
                        >(&prepared_first_contiguous_swapped_batch, entries);

                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    plain_canonical_count = {plain_canonical_count}");
                    println!(
                        "    prefetchbest_canonical_count = {prefetchbest_canonical_count}"
                    );
                    println!("    plain_mismatches = {plain_mismatches}");
                    println!("    prefetchbest_mismatches = {prefetchbest_mismatches}");
                    println!("    plain_seconds = {plain_seconds:.12}");
                    println!("    prefetchbest_seconds = {prefetchbest_seconds:.12}");
                    println!(
                        "    plain_vs_prefetchbest_speedup = {:.12}x",
                        plain_seconds / prefetchbest_seconds
                    );
                }
            }};
        }

        bench_bucket_prefetchbest!(1);
        bench_bucket_prefetchbest!(2);
        bench_bucket_prefetchbest!(3);
        bench_bucket_prefetchbest!(4);
        bench_bucket_prefetchbest!(5);
        bench_bucket_prefetchbest!(6);
        bench_bucket_prefetchbest!(7);
        bench_bucket_prefetchbest!(8);
    }

    if mode == "preparedfirstdense_lockstep4_contiguous_vs_swappedprefetch_bybucket" {
        let prepared_first_contiguous_batch =
            match build_prepared_first_dense_contiguous_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_swapped_batch =
            match build_prepared_first_dense_contiguous_swapped_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_contiguous_vs_swappedprefetch_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  contiguous_prepared_first_count = {}",
            prepared_first_contiguous_batch.len()
        );
        println!(
            "  swapped_contiguous_prepared_first_count = {}",
            prepared_first_contiguous_swapped_batch.len()
        );

        macro_rules! bench_bucket_direct_compare {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, contiguous_canonical_count, contiguous_seconds) =
                        time_bucket_lockstep4_contiguous::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let (
                        _swapped_prefetch_pair_count,
                        swapped_prefetch_canonical_count,
                        swapped_prefetch_seconds,
                    ) = time_bucket_lockstep4_contiguous_swapped_prefetch::<$left_len>(
                        &prepared_first_contiguous_swapped_batch,
                        entries,
                    );

                    let contiguous_mismatches =
                        count_bucket_lockstep4_contiguous_mismatches::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let swapped_prefetch_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_prefetch_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );

                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    contiguous_canonical_count = {contiguous_canonical_count}");
                    println!(
                        "    swapped_prefetch_canonical_count = {swapped_prefetch_canonical_count}"
                    );
                    println!("    contiguous_mismatches = {contiguous_mismatches}");
                    println!(
                        "    swapped_prefetch_mismatches = {swapped_prefetch_mismatches}"
                    );
                    println!("    contiguous_seconds = {contiguous_seconds:.12}");
                    println!(
                        "    swapped_prefetch_seconds = {swapped_prefetch_seconds:.12}"
                    );
                    println!(
                        "    contiguous_vs_swappedprefetch_speedup = {:.12}x",
                        contiguous_seconds / swapped_prefetch_seconds
                    );
                }
            }};
        }

        bench_bucket_direct_compare!(1);
        bench_bucket_direct_compare!(2);
        bench_bucket_direct_compare!(3);
        bench_bucket_direct_compare!(4);
        bench_bucket_direct_compare!(5);
        bench_bucket_direct_compare!(6);
        bench_bucket_direct_compare!(7);
        bench_bucket_direct_compare!(8);
    }

    if mode == "preparedfirstdense_lockstep4_threeway_bybucket" {
        let prepared_first_contiguous_batch =
            match build_prepared_first_dense_contiguous_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_swapped_batch =
            match build_prepared_first_dense_contiguous_swapped_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_threeway_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  contiguous_prepared_first_count = {}",
            prepared_first_contiguous_batch.len()
        );
        println!(
            "  swapped_contiguous_prepared_first_count = {}",
            prepared_first_contiguous_swapped_batch.len()
        );

        macro_rules! bench_bucket_threeway {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, plain_canonical_count, plain_seconds) =
                        time_bucket_lockstep4_contiguous::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let (_swapped_pair_count, swapped_canonical_count, swapped_seconds) =
                        time_bucket_lockstep4_contiguous_swapped::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let (
                        _swapped_prefetch_pair_count,
                        swapped_prefetch_canonical_count,
                        swapped_prefetch_seconds,
                    ) = time_bucket_lockstep4_contiguous_swapped_prefetch::<$left_len>(
                        &prepared_first_contiguous_swapped_batch,
                        entries,
                    );

                    let plain_mismatches =
                        count_bucket_lockstep4_contiguous_mismatches::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let swapped_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let swapped_prefetch_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_prefetch_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );

                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    plain_canonical_count = {plain_canonical_count}");
                    println!("    swapped_canonical_count = {swapped_canonical_count}");
                    println!(
                        "    swapped_prefetch_canonical_count = {swapped_prefetch_canonical_count}"
                    );
                    println!("    plain_mismatches = {plain_mismatches}");
                    println!("    swapped_mismatches = {swapped_mismatches}");
                    println!(
                        "    swapped_prefetch_mismatches = {swapped_prefetch_mismatches}"
                    );
                    println!("    plain_seconds = {plain_seconds:.12}");
                    println!("    swapped_seconds = {swapped_seconds:.12}");
                    println!(
                        "    swapped_prefetch_seconds = {swapped_prefetch_seconds:.12}"
                    );
                    println!(
                        "    speedup_plain_ref_swapped = {:.12}x",
                        plain_seconds / swapped_seconds
                    );
                    println!(
                        "    speedup_plain_ref_swapped_prefetch = {:.12}x",
                        plain_seconds / swapped_prefetch_seconds
                    );
                }
            }};
        }

        bench_bucket_threeway!(1);
        bench_bucket_threeway!(2);
        bench_bucket_threeway!(3);
        bench_bucket_threeway!(4);
        bench_bucket_threeway!(5);
        bench_bucket_threeway!(6);
        bench_bucket_threeway!(7);
        bench_bucket_threeway!(8);
    }

    if mode == "preparedfirstdense_lockstep4_plain_vs_swappedprefetch2_bybucket" {
        let prepared_first_contiguous_batch =
            match build_prepared_first_dense_contiguous_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_swapped_batch =
            match build_prepared_first_dense_contiguous_swapped_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_plain_vs_swappedprefetch2_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  contiguous_prepared_first_count = {}",
            prepared_first_contiguous_batch.len()
        );
        println!(
            "  swapped_contiguous_prepared_first_count = {}",
            prepared_first_contiguous_swapped_batch.len()
        );

        macro_rules! bench_bucket_direct_plain_prefetch2 {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, plain_canonical_count, plain_seconds) =
                        time_bucket_lockstep4_contiguous::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let (
                        _prefetch2_pair_count,
                        swapped_prefetch2_canonical_count,
                        swapped_prefetch2_seconds,
                    ) = time_bucket_lockstep4_contiguous_swapped_prefetch2::<$left_len>(
                        &prepared_first_contiguous_swapped_batch,
                        entries,
                    );
                    let plain_mismatches =
                        count_bucket_lockstep4_contiguous_mismatches::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let swapped_prefetch2_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_prefetch2_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    plain_canonical_count = {plain_canonical_count}");
                    println!(
                        "    swapped_prefetch2_canonical_count = {swapped_prefetch2_canonical_count}"
                    );
                    println!("    plain_mismatches = {plain_mismatches}");
                    println!(
                        "    swapped_prefetch2_mismatches = {swapped_prefetch2_mismatches}"
                    );
                    println!("    plain_seconds = {plain_seconds:.12}");
                    println!("    swapped_prefetch2_seconds = {swapped_prefetch2_seconds:.12}");
                    println!(
                        "    plain_vs_swappedprefetch2_speedup = {:.12}x",
                        plain_seconds / swapped_prefetch2_seconds
                    );
                }
            }};
        }

        bench_bucket_direct_plain_prefetch2!(1);
        bench_bucket_direct_plain_prefetch2!(2);
        bench_bucket_direct_plain_prefetch2!(3);
        bench_bucket_direct_plain_prefetch2!(4);
        bench_bucket_direct_plain_prefetch2!(5);
        bench_bucket_direct_plain_prefetch2!(6);
        bench_bucket_direct_plain_prefetch2!(7);
        bench_bucket_direct_plain_prefetch2!(8);
    }

    if mode == "preparedfirstdense_lockstep4_swapped_tight_vs_swapped_bybucket" {
        let prepared_first_contiguous_swapped_batch =
            match build_prepared_first_dense_contiguous_swapped_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_swapped_tight_batch =
            match build_prepared_first_dense_contiguous_swapped_tight_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped-tight contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_swapped_tight_vs_swapped_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  swapped_contiguous_prepared_first_count = {}",
            prepared_first_contiguous_swapped_batch.len()
        );
        println!(
            "  swapped_tight_prepared_first_count = {}",
            prepared_first_contiguous_swapped_tight_batch.len()
        );

        macro_rules! bench_bucket_swapped_tight {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, swapped_canonical_count, swapped_seconds) =
                        time_bucket_lockstep4_contiguous_swapped::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let (_tight_pair_count, tight_canonical_count, tight_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    let swapped_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let tight_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_tight_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    swapped_canonical_count = {swapped_canonical_count}");
                    println!("    tight_canonical_count = {tight_canonical_count}");
                    println!("    swapped_mismatches = {swapped_mismatches}");
                    println!("    tight_mismatches = {tight_mismatches}");
                    println!("    swapped_seconds = {swapped_seconds:.12}");
                    println!("    tight_seconds = {tight_seconds:.12}");
                    println!(
                        "    tight_vs_swapped_speedup = {:.12}x",
                        swapped_seconds / tight_seconds
                    );
                }
            }};
        }

        bench_bucket_swapped_tight!(1);
        bench_bucket_swapped_tight!(2);
        bench_bucket_swapped_tight!(3);
        bench_bucket_swapped_tight!(4);
        bench_bucket_swapped_tight!(5);
        bench_bucket_swapped_tight!(6);
        bench_bucket_swapped_tight!(7);
        bench_bucket_swapped_tight!(8);
    }

    if mode == "preparedfirstdense_lockstep4_plain_vs_swappedtight_bybucket" {
        let prepared_first_contiguous_batch =
            match build_prepared_first_dense_contiguous_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_swapped_tight_batch =
            match build_prepared_first_dense_contiguous_swapped_tight_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped-tight contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_plain_vs_swappedtight_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  contiguous_prepared_first_count = {}",
            prepared_first_contiguous_batch.len()
        );
        println!(
            "  swapped_tight_prepared_first_count = {}",
            prepared_first_contiguous_swapped_tight_batch.len()
        );

        macro_rules! bench_bucket_plain_vs_tight {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, plain_canonical_count, plain_seconds) =
                        time_bucket_lockstep4_contiguous::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let (_tight_pair_count, tight_canonical_count, tight_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    let plain_mismatches =
                        count_bucket_lockstep4_contiguous_mismatches::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let tight_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_tight_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    plain_canonical_count = {plain_canonical_count}");
                    println!("    tight_canonical_count = {tight_canonical_count}");
                    println!("    plain_mismatches = {plain_mismatches}");
                    println!("    tight_mismatches = {tight_mismatches}");
                    println!("    plain_seconds = {plain_seconds:.12}");
                    println!("    tight_seconds = {tight_seconds:.12}");
                    println!(
                        "    plain_vs_tight_speedup = {:.12}x",
                        plain_seconds / tight_seconds
                    );
                }
            }};
        }

        bench_bucket_plain_vs_tight!(1);
        bench_bucket_plain_vs_tight!(2);
        bench_bucket_plain_vs_tight!(3);
        bench_bucket_plain_vs_tight!(4);
        bench_bucket_plain_vs_tight!(5);
        bench_bucket_plain_vs_tight!(6);
        bench_bucket_plain_vs_tight!(7);
        bench_bucket_plain_vs_tight!(8);
    }

    if mode == "preparedfirstdense_lockstep4_swapped_vs_swappedtightprefetch1_bybucket" {
        let prepared_first_contiguous_swapped_batch =
            match build_prepared_first_dense_contiguous_swapped_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_swapped_tight_batch =
            match build_prepared_first_dense_contiguous_swapped_tight_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped-tight contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_swapped_vs_swappedtightprefetch1_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  swapped_contiguous_prepared_first_count = {}",
            prepared_first_contiguous_swapped_batch.len()
        );
        println!(
            "  swapped_tight_prepared_first_count = {}",
            prepared_first_contiguous_swapped_tight_batch.len()
        );

        macro_rules! bench_bucket_swapped_vs_tight_prefetch {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, swapped_canonical_count, swapped_seconds) =
                        time_bucket_lockstep4_contiguous_swapped::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let (_tight_pair_count, tight_prefetch_canonical_count, tight_prefetch_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    let swapped_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_batch,
                            entries,
                        );
                    let tight_prefetch_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_tight_prefetch_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    swapped_canonical_count = {swapped_canonical_count}");
                    println!(
                        "    tight_prefetch_canonical_count = {tight_prefetch_canonical_count}"
                    );
                    println!("    swapped_mismatches = {swapped_mismatches}");
                    println!("    tight_prefetch_mismatches = {tight_prefetch_mismatches}");
                    println!("    swapped_seconds = {swapped_seconds:.12}");
                    println!("    tight_prefetch_seconds = {tight_prefetch_seconds:.12}");
                    println!(
                        "    swapped_vs_tightprefetch_speedup = {:.12}x",
                        swapped_seconds / tight_prefetch_seconds
                    );
                }
            }};
        }

        bench_bucket_swapped_vs_tight_prefetch!(1);
        bench_bucket_swapped_vs_tight_prefetch!(2);
        bench_bucket_swapped_vs_tight_prefetch!(3);
        bench_bucket_swapped_vs_tight_prefetch!(4);
        bench_bucket_swapped_vs_tight_prefetch!(5);
        bench_bucket_swapped_vs_tight_prefetch!(6);
        bench_bucket_swapped_vs_tight_prefetch!(7);
        bench_bucket_swapped_vs_tight_prefetch!(8);
    }

    if mode == "preparedfirstdense_lockstep4_swappedtight_prefetch1_dedupe_vs_prefetch1_bybucket" {
        let prepared_first_contiguous_swapped_tight_batch =
            match build_prepared_first_dense_contiguous_swapped_tight_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped-tight contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prefetch_chunks_by_left_len = build_prefetch_chunks_by_left_len(candidate_second_buckets);

        println!("prepared_first_dense_lockstep4_swappedtight_prefetch1_dedupe_vs_prefetch1_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  swapped_tight_prepared_first_count = {}",
            prepared_first_contiguous_swapped_tight_batch.len()
        );

        let mut total_pair_count = 0u64;
        let mut total_prefetch1_seconds = 0.0f64;
        let mut total_dedupe_seconds = 0.0f64;

        macro_rules! bench_bucket_tight_prefetch_vs_dedupe {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                let prefetch_chunks = &prefetch_chunks_by_left_len[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, prefetch1_canonical_count, prefetch1_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    let (_dedupe_pair_count, dedupe_canonical_count, dedupe_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch_dedupe::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            prefetch_chunks,
                        );
                    let prefetch1_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_tight_prefetch_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    prefetch1_chunk_count = {}", prefetch_chunks.len());
                    println!("    prefetch1_canonical_count = {prefetch1_canonical_count}");
                    println!("    dedupe_canonical_count = {dedupe_canonical_count}");
                    println!("    prefetch1_mismatches = {prefetch1_mismatches}");
                    println!("    prefetch1_seconds = {prefetch1_seconds:.12}");
                    println!("    dedupe_seconds = {dedupe_seconds:.12}");
                    println!(
                        "    prefetch1_vs_dedupe_speedup = {:.12}x",
                        prefetch1_seconds / dedupe_seconds
                    );
                    total_pair_count += pair_count;
                    total_prefetch1_seconds += prefetch1_seconds;
                    total_dedupe_seconds += dedupe_seconds;
                }
            }};
        }

        bench_bucket_tight_prefetch_vs_dedupe!(1);
        bench_bucket_tight_prefetch_vs_dedupe!(2);
        bench_bucket_tight_prefetch_vs_dedupe!(3);
        bench_bucket_tight_prefetch_vs_dedupe!(4);
        bench_bucket_tight_prefetch_vs_dedupe!(5);
        bench_bucket_tight_prefetch_vs_dedupe!(6);
        bench_bucket_tight_prefetch_vs_dedupe!(7);
        bench_bucket_tight_prefetch_vs_dedupe!(8);

        if total_pair_count > 0 {
            let prefetch1_ns_per_candidate =
                total_prefetch1_seconds * 1_000_000_000.0 / total_pair_count as f64;
            let dedupe_ns_per_candidate =
                total_dedupe_seconds * 1_000_000_000.0 / total_pair_count as f64;
            println!("  overall_pair_count = {total_pair_count}");
            println!("  overall_prefetch1_seconds = {total_prefetch1_seconds:.12}");
            println!("  overall_dedupe_seconds = {total_dedupe_seconds:.12}");
            println!("  overall_prefetch1_ns_per_candidate = {prefetch1_ns_per_candidate:.12}");
            println!("  overall_dedupe_ns_per_candidate = {dedupe_ns_per_candidate:.12}");
            println!(
                "  overall_prefetch1_vs_dedupe_speedup = {:.12}x",
                total_prefetch1_seconds / total_dedupe_seconds
            );
        }
    }

    if mode == "preparedfirstdense_lockstep4_swappedtight_prefetch_paramsearch_matrix" {
        let prepared_first_by_right_len =
            match build_prepared_first_dense_contiguous_swapped_tight_batches_by_right_len(
                &tokenizer,
                &sampled_first_ids,
            ) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild swapped-tight contiguous prepared-first dense batches by right_len: {err}"
                    );
                    return ExitCode::from(1);
                }
            };
        let prefetch_chunks_by_left_len = build_prefetch_chunks_by_left_len(candidate_second_buckets);

        println!("prepared_first_dense_lockstep4_swappedtight_prefetch_paramsearch_matrix:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!("  grid:");
        println!("    enabled: [false, true]");
        println!("    lookahead_chunks (when enabled): [1, 2]");
        println!("    hint (when enabled): [t0, t1, t2, nta]");
        println!("    budget (when enabled): [0, 8, 16] (0 means unlimited)");
        println!("    scope (when enabled): [left_len, min(left_len,4), min(left_len,2)]");

        let mut total_pair_count = 0u64;
        let mut total_baseline_seconds = 0.0f64;
        let mut total_noprefetch_seconds = 0.0f64;
        let mut total_best_seconds = 0.0f64;
        let mut max_speedup_nondefault_lookahead = 1.0f64;
        let mut max_speedup_nondefault_budget = 1.0f64;
        let mut max_speedup_nondefault_hint = 1.0f64;
        let mut max_speedup_nondefault_scope = 1.0f64;
        let mut max_speedup_prefetch_enable_false = 1.0f64;

        let baseline = PrefetchConfig {
            enabled: true,
            lookahead_chunks: 1,
            budget: 0,
            scope: 8, // clamped to LEFT_LEN in kernel.
            hint: PrefetchHint::T0,
        };
        let noprefetch = PrefetchConfig {
            enabled: false,
            lookahead_chunks: 1,
            budget: 0,
            scope: 8,
            hint: PrefetchHint::T0,
        };

        for left_len in 1..=8usize {
            let entries = &candidate_second_buckets[left_len];
            let prefetch_chunks = &prefetch_chunks_by_left_len[left_len];
            if entries.is_empty() {
                continue;
            }

            for right_len in 1..=8usize {
                let prepared_batch = &prepared_first_by_right_len[right_len];
                if prepared_batch.is_empty() {
                    continue;
                }

                let scope_candidates = {
                    let mut v = vec![left_len as u8, left_len.min(4) as u8, left_len.min(2) as u8];
                    v.sort_unstable();
                    v.dedup();
                    v
                };
                let mut configs = Vec::new();
                configs.push(noprefetch);
                for &lookahead_chunks in &[1usize, 2usize] {
                    for &hint in &[PrefetchHint::T0, PrefetchHint::T1, PrefetchHint::T2, PrefetchHint::Nta]
                    {
                        for &budget in &[0u8, 8u8, 16u8] {
                            for &scope in &scope_candidates {
                                configs.push(PrefetchConfig {
                                    enabled: true,
                                    lookahead_chunks,
                                    budget,
                                    scope,
                                    hint,
                                });
                            }
                        }
                    }
                }

                macro_rules! time_cfg_for_left_len {
                    ($ll:literal, $cfg:expr) => {{
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch_param::<$ll>(
                            prepared_batch,
                            entries,
                            prefetch_chunks,
                            $cfg,
                        )
                    }};
                }

                let baseline_result = match left_len {
                    1 => time_cfg_for_left_len!(1, baseline),
                    2 => time_cfg_for_left_len!(2, baseline),
                    3 => time_cfg_for_left_len!(3, baseline),
                    4 => time_cfg_for_left_len!(4, baseline),
                    5 => time_cfg_for_left_len!(5, baseline),
                    6 => time_cfg_for_left_len!(6, baseline),
                    7 => time_cfg_for_left_len!(7, baseline),
                    8 => time_cfg_for_left_len!(8, baseline),
                    _ => unreachable!("left_len buckets are in 1..=8"),
                };
                let noprefetch_result = match left_len {
                    1 => time_cfg_for_left_len!(1, noprefetch),
                    2 => time_cfg_for_left_len!(2, noprefetch),
                    3 => time_cfg_for_left_len!(3, noprefetch),
                    4 => time_cfg_for_left_len!(4, noprefetch),
                    5 => time_cfg_for_left_len!(5, noprefetch),
                    6 => time_cfg_for_left_len!(6, noprefetch),
                    7 => time_cfg_for_left_len!(7, noprefetch),
                    8 => time_cfg_for_left_len!(8, noprefetch),
                    _ => unreachable!("left_len buckets are in 1..=8"),
                };

                let mut best_cfg = baseline;
                let mut best_seconds = baseline_result.2;
                let pair_count = baseline_result.0;
                let baseline_canonical = baseline_result.1;

                for cfg in configs {
                    let result = match left_len {
                        1 => time_cfg_for_left_len!(1, cfg),
                        2 => time_cfg_for_left_len!(2, cfg),
                        3 => time_cfg_for_left_len!(3, cfg),
                        4 => time_cfg_for_left_len!(4, cfg),
                        5 => time_cfg_for_left_len!(5, cfg),
                        6 => time_cfg_for_left_len!(6, cfg),
                        7 => time_cfg_for_left_len!(7, cfg),
                        8 => time_cfg_for_left_len!(8, cfg),
                        _ => unreachable!("left_len buckets are in 1..=8"),
                    };
                    if result.1 != baseline_canonical {
                        eprintln!(
                            "canonical count mismatch for cell (left_len={left_len}, right_len={right_len})"
                        );
                        return ExitCode::from(1);
                    }
                    if result.2 < best_seconds {
                        best_seconds = result.2;
                        best_cfg = cfg;
                    }
                }

                let baseline_seconds = baseline_result.2;
                let noprefetch_seconds = noprefetch_result.2;
                let baseline_ns = baseline_seconds * 1_000_000_000.0 / pair_count as f64;
                let noprefetch_ns = noprefetch_seconds * 1_000_000_000.0 / pair_count as f64;
                let best_ns = best_seconds * 1_000_000_000.0 / pair_count as f64;
                let best_over_baseline = baseline_seconds / best_seconds;

                if !best_cfg.enabled {
                    max_speedup_prefetch_enable_false =
                        max_speedup_prefetch_enable_false.max(best_over_baseline);
                } else {
                    if best_cfg.lookahead_chunks != baseline.lookahead_chunks {
                        max_speedup_nondefault_lookahead =
                            max_speedup_nondefault_lookahead.max(best_over_baseline);
                    }
                    if best_cfg.budget != baseline.budget {
                        max_speedup_nondefault_budget =
                            max_speedup_nondefault_budget.max(best_over_baseline);
                    }
                    if best_cfg.hint != baseline.hint {
                        max_speedup_nondefault_hint =
                            max_speedup_nondefault_hint.max(best_over_baseline);
                    }
                    if usize::from(best_cfg.scope).min(left_len)
                        != usize::from(baseline.scope).min(left_len)
                    {
                        max_speedup_nondefault_scope =
                            max_speedup_nondefault_scope.max(best_over_baseline);
                    }
                }

                println!("  cell(left_len={left_len}, right_len={right_len}):");
                println!("    pair_count = {pair_count}");
                println!("    baseline_ns_per_candidate = {baseline_ns:.6}");
                println!("    noprefetch_ns_per_candidate = {noprefetch_ns:.6}");
                println!("    best_ns_per_candidate = {best_ns:.6}");
                println!(
                    "    best_over_baseline_speedup = {:.6}x",
                    best_over_baseline
                );
                println!(
                    "    best_config = enabled={}, lookahead_chunks={}, budget={}, hint={}, scope={}",
                    best_cfg.enabled,
                    best_cfg.lookahead_chunks,
                    best_cfg.budget,
                    prefetch_hint_name(best_cfg.hint),
                    best_cfg.scope
                );

                total_pair_count += pair_count;
                total_baseline_seconds += baseline_seconds;
                total_noprefetch_seconds += noprefetch_seconds;
                total_best_seconds += best_seconds;
            }
        }

        if total_pair_count > 0 {
            println!("  overall_pair_count = {total_pair_count}");
            println!(
                "  overall_baseline_ns_per_candidate = {:.12}",
                total_baseline_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_noprefetch_ns_per_candidate = {:.12}",
                total_noprefetch_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_bestcell_oracle_ns_per_candidate = {:.12}",
                total_best_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_baseline_over_bestcell_oracle_speedup = {:.12}x",
                total_baseline_seconds / total_best_seconds
            );
            println!("  worth_configuring_max_speedups:");
            println!(
                "    prefetch_enable=false_bestcell = {:.6}x",
                max_speedup_prefetch_enable_false
            );
            println!(
                "    lookahead(nondefault)_bestcell = {:.6}x",
                max_speedup_nondefault_lookahead
            );
            println!(
                "    budget(nondefault)_bestcell = {:.6}x",
                max_speedup_nondefault_budget
            );
            println!(
                "    hint(nondefault)_bestcell = {:.6}x",
                max_speedup_nondefault_hint
            );
            println!(
                "    scope(nondefault)_bestcell = {:.6}x",
                max_speedup_nondefault_scope
            );
        }
    }

    if mode == "preparedfirstdense_lockstep4_swappedtight_prefetch_paramsearch_matrix_robust" {
        let prepared_first_by_right_len =
            match build_prepared_first_dense_contiguous_swapped_tight_batches_by_right_len(
                &tokenizer,
                &sampled_first_ids,
            ) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild swapped-tight contiguous prepared-first dense batches by right_len: {err}"
                    );
                    return ExitCode::from(1);
                }
            };
        let prefetch_chunks_by_left_len = build_prefetch_chunks_by_left_len(candidate_second_buckets);

        const REPEATS: usize = 5;
        println!("prepared_first_dense_lockstep4_swappedtight_prefetch_paramsearch_matrix_robust:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!("  repeats_per_cell = {REPEATS}");
        println!("  randomized config order each repeat");
        println!("  score = median ns/candidate");
        println!("  grid:");
        println!("    enabled: [false, true]");
        println!("    lookahead_chunks (when enabled): [1, 2]");
        println!("    hint (when enabled): [t0, t1, t2, nta]");
        println!("    budget (when enabled): [0, 8, 16] (0 means unlimited)");
        println!("    scope (when enabled): [left_len, min(left_len,4), min(left_len,2)]");

        let mut total_pair_count = 0u64;
        let mut total_baseline_seconds_median = 0.0f64;
        let mut total_noprefetch_seconds_median = 0.0f64;
        let mut total_best_seconds_median = 0.0f64;
        let mut max_speedup_nondefault_lookahead = 1.0f64;
        let mut max_speedup_nondefault_budget = 1.0f64;
        let mut max_speedup_nondefault_hint = 1.0f64;
        let mut max_speedup_nondefault_scope = 1.0f64;
        let mut max_speedup_prefetch_enable_false = 1.0f64;

        let baseline = PrefetchConfig {
            enabled: true,
            lookahead_chunks: 1,
            budget: 0,
            scope: 8, // clamped to LEFT_LEN in kernel.
            hint: PrefetchHint::T0,
        };
        let noprefetch = PrefetchConfig {
            enabled: false,
            lookahead_chunks: 1,
            budget: 0,
            scope: 8,
            hint: PrefetchHint::T0,
        };

        let mut rng = XorShift64::new(seed ^ 0x9e37_79b9_7f4a_7c15);

        for left_len in 1..=8usize {
            let entries = &candidate_second_buckets[left_len];
            let prefetch_chunks = &prefetch_chunks_by_left_len[left_len];
            if entries.is_empty() {
                continue;
            }

            for right_len in 1..=8usize {
                let prepared_batch = &prepared_first_by_right_len[right_len];
                if prepared_batch.is_empty() {
                    continue;
                }

                let scope_candidates = {
                    let mut v = vec![left_len as u8, left_len.min(4) as u8, left_len.min(2) as u8];
                    v.sort_unstable();
                    v.dedup();
                    v
                };
                let mut nonbaseline_configs = Vec::new();
                for &lookahead_chunks in &[1usize, 2usize] {
                    for &hint in &[PrefetchHint::T0, PrefetchHint::T1, PrefetchHint::T2, PrefetchHint::Nta]
                    {
                        for &budget in &[0u8, 8u8, 16u8] {
                            for &scope in &scope_candidates {
                                let cfg = PrefetchConfig {
                                    enabled: true,
                                    lookahead_chunks,
                                    budget,
                                    scope,
                                    hint,
                                };
                                if cfg.enabled == baseline.enabled
                                    && cfg.lookahead_chunks == baseline.lookahead_chunks
                                    && cfg.budget == baseline.budget
                                    && cfg.scope == baseline.scope
                                    && cfg.hint == baseline.hint
                                {
                                    continue;
                                }
                                nonbaseline_configs.push(cfg);
                            }
                        }
                    }
                }
                nonbaseline_configs.push(noprefetch);

                macro_rules! time_cfg_for_left_len {
                    ($ll:literal, $cfg:expr) => {{
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch_param::<$ll>(
                            prepared_batch,
                            entries,
                            prefetch_chunks,
                            $cfg,
                        )
                    }};
                }

                let mut baseline_seconds_samples = Vec::with_capacity(REPEATS);
                let mut noprefetch_seconds_samples = Vec::with_capacity(REPEATS);
                let mut cfg_seconds_samples: Vec<Vec<f64>> =
                    vec![Vec::with_capacity(REPEATS); nonbaseline_configs.len()];
                let mut pair_count = 0u64;
                let mut baseline_canonical = 0u64;
                for repeat_idx in 0..REPEATS {
                    let baseline_result = match left_len {
                        1 => time_cfg_for_left_len!(1, baseline),
                        2 => time_cfg_for_left_len!(2, baseline),
                        3 => time_cfg_for_left_len!(3, baseline),
                        4 => time_cfg_for_left_len!(4, baseline),
                        5 => time_cfg_for_left_len!(5, baseline),
                        6 => time_cfg_for_left_len!(6, baseline),
                        7 => time_cfg_for_left_len!(7, baseline),
                        8 => time_cfg_for_left_len!(8, baseline),
                        _ => unreachable!("left_len buckets are in 1..=8"),
                    };
                    if repeat_idx == 0 {
                        pair_count = baseline_result.0;
                        baseline_canonical = baseline_result.1;
                    } else if baseline_result.1 != baseline_canonical {
                        eprintln!(
                            "baseline canonical mismatch for cell (left_len={left_len}, right_len={right_len})"
                        );
                        return ExitCode::from(1);
                    }
                    baseline_seconds_samples.push(baseline_result.2);

                    let noprefetch_result = match left_len {
                        1 => time_cfg_for_left_len!(1, noprefetch),
                        2 => time_cfg_for_left_len!(2, noprefetch),
                        3 => time_cfg_for_left_len!(3, noprefetch),
                        4 => time_cfg_for_left_len!(4, noprefetch),
                        5 => time_cfg_for_left_len!(5, noprefetch),
                        6 => time_cfg_for_left_len!(6, noprefetch),
                        7 => time_cfg_for_left_len!(7, noprefetch),
                        8 => time_cfg_for_left_len!(8, noprefetch),
                        _ => unreachable!("left_len buckets are in 1..=8"),
                    };
                    if noprefetch_result.1 != baseline_canonical {
                        eprintln!(
                            "noprefetch canonical mismatch for cell (left_len={left_len}, right_len={right_len})"
                        );
                        return ExitCode::from(1);
                    }
                    noprefetch_seconds_samples.push(noprefetch_result.2);

                    let mut order: Vec<usize> = (0..nonbaseline_configs.len()).collect();
                    shuffle_usize(&mut order, &mut rng);
                    for cfg_idx in order {
                        let cfg = nonbaseline_configs[cfg_idx];
                        let result = match left_len {
                            1 => time_cfg_for_left_len!(1, cfg),
                            2 => time_cfg_for_left_len!(2, cfg),
                            3 => time_cfg_for_left_len!(3, cfg),
                            4 => time_cfg_for_left_len!(4, cfg),
                            5 => time_cfg_for_left_len!(5, cfg),
                            6 => time_cfg_for_left_len!(6, cfg),
                            7 => time_cfg_for_left_len!(7, cfg),
                            8 => time_cfg_for_left_len!(8, cfg),
                            _ => unreachable!("left_len buckets are in 1..=8"),
                        };
                        if result.1 != baseline_canonical {
                            eprintln!(
                                "canonical mismatch for cell (left_len={left_len}, right_len={right_len})"
                            );
                            return ExitCode::from(1);
                        }
                        cfg_seconds_samples[cfg_idx].push(result.2);
                    }
                }

                let mut baseline_samples_for_median = baseline_seconds_samples.clone();
                let mut noprefetch_samples_for_median = noprefetch_seconds_samples.clone();
                let baseline_seconds = median_f64(&mut baseline_samples_for_median);
                let noprefetch_seconds = median_f64(&mut noprefetch_samples_for_median);

                let mut best_cfg = baseline;
                let mut best_seconds = baseline_seconds;
                for (cfg_idx, cfg) in nonbaseline_configs.iter().copied().enumerate() {
                    let mut samples = cfg_seconds_samples[cfg_idx].clone();
                    let seconds = median_f64(&mut samples);
                    if seconds < best_seconds {
                        best_seconds = seconds;
                        best_cfg = cfg;
                    }
                }

                let baseline_ns = baseline_seconds * 1_000_000_000.0 / pair_count as f64;
                let noprefetch_ns = noprefetch_seconds * 1_000_000_000.0 / pair_count as f64;
                let best_ns = best_seconds * 1_000_000_000.0 / pair_count as f64;
                let best_over_baseline = baseline_seconds / best_seconds;

                if !best_cfg.enabled {
                    max_speedup_prefetch_enable_false =
                        max_speedup_prefetch_enable_false.max(best_over_baseline);
                } else {
                    if best_cfg.lookahead_chunks != baseline.lookahead_chunks {
                        max_speedup_nondefault_lookahead =
                            max_speedup_nondefault_lookahead.max(best_over_baseline);
                    }
                    if best_cfg.budget != baseline.budget {
                        max_speedup_nondefault_budget =
                            max_speedup_nondefault_budget.max(best_over_baseline);
                    }
                    if best_cfg.hint != baseline.hint {
                        max_speedup_nondefault_hint =
                            max_speedup_nondefault_hint.max(best_over_baseline);
                    }
                    if usize::from(best_cfg.scope).min(left_len)
                        != usize::from(baseline.scope).min(left_len)
                    {
                        max_speedup_nondefault_scope =
                            max_speedup_nondefault_scope.max(best_over_baseline);
                    }
                }

                println!("  cell(left_len={left_len}, right_len={right_len}):");
                println!("    pair_count = {pair_count}");
                println!("    baseline_ns_per_candidate = {baseline_ns:.6}");
                println!("    noprefetch_ns_per_candidate = {noprefetch_ns:.6}");
                println!("    best_ns_per_candidate = {best_ns:.6}");
                println!(
                    "    best_over_baseline_speedup = {:.6}x",
                    best_over_baseline
                );
                println!(
                    "    best_config = enabled={}, lookahead_chunks={}, budget={}, hint={}, scope={}",
                    best_cfg.enabled,
                    best_cfg.lookahead_chunks,
                    best_cfg.budget,
                    prefetch_hint_name(best_cfg.hint),
                    best_cfg.scope
                );

                total_pair_count += pair_count;
                total_baseline_seconds_median += baseline_seconds;
                total_noprefetch_seconds_median += noprefetch_seconds;
                total_best_seconds_median += best_seconds;
            }
        }

        if total_pair_count > 0 {
            println!("  overall_pair_count = {total_pair_count}");
            println!(
                "  overall_baseline_ns_per_candidate = {:.12}",
                total_baseline_seconds_median * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_noprefetch_ns_per_candidate = {:.12}",
                total_noprefetch_seconds_median * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_bestcell_oracle_ns_per_candidate = {:.12}",
                total_best_seconds_median * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_baseline_over_bestcell_oracle_speedup = {:.12}x",
                total_baseline_seconds_median / total_best_seconds_median
            );
            println!("  worth_configuring_max_speedups:");
            println!(
                "    prefetch_enable=false_bestcell = {:.6}x",
                max_speedup_prefetch_enable_false
            );
            println!(
                "    lookahead(nondefault)_bestcell = {:.6}x",
                max_speedup_nondefault_lookahead
            );
            println!(
                "    budget(nondefault)_bestcell = {:.6}x",
                max_speedup_nondefault_budget
            );
            println!(
                "    hint(nondefault)_bestcell = {:.6}x",
                max_speedup_nondefault_hint
            );
            println!(
                "    scope(nondefault)_bestcell = {:.6}x",
                max_speedup_nondefault_scope
            );
        }
    }

    if mode == "preparedfirstdense_lockstep4_plain_vs_swappedtight_noprefetch_bybucket" {
        let prepared_first_contiguous_batch =
            match build_prepared_first_dense_contiguous_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prepared_first_contiguous_swapped_tight_batch =
            match build_prepared_first_dense_contiguous_swapped_tight_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped-tight contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prefetch_chunks_by_left_len = build_prefetch_chunks_by_left_len(candidate_second_buckets);
        let noprefetch = PrefetchConfig {
            enabled: false,
            lookahead_chunks: 1,
            budget: 0,
            scope: 8,
            hint: PrefetchHint::T0,
        };

        println!("prepared_first_dense_lockstep4_plain_vs_swappedtight_noprefetch_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  contiguous_prepared_first_count = {}",
            prepared_first_contiguous_batch.len()
        );
        println!(
            "  swapped_tight_prepared_first_count = {}",
            prepared_first_contiguous_swapped_tight_batch.len()
        );

        let mut total_pair_count = 0u64;
        let mut total_plain_seconds = 0.0f64;
        let mut total_noprefetch_seconds = 0.0f64;

        macro_rules! bench_bucket_plain_vs_noprefetch {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                let prefetch_chunks = &prefetch_chunks_by_left_len[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, plain_canonical_count, plain_seconds) =
                        time_bucket_lockstep4_contiguous::<$left_len>(
                            &prepared_first_contiguous_batch,
                            entries,
                        );
                    let (_noprefetch_pair_count, noprefetch_canonical_count, noprefetch_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch_param::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            prefetch_chunks,
                            noprefetch,
                        );
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    plain_canonical_count = {plain_canonical_count}");
                    println!("    noprefetch_canonical_count = {noprefetch_canonical_count}");
                    println!("    plain_seconds = {plain_seconds:.12}");
                    println!("    noprefetch_seconds = {noprefetch_seconds:.12}");
                    println!(
                        "    plain_vs_noprefetch_speedup = {:.12}x",
                        plain_seconds / noprefetch_seconds
                    );
                    total_pair_count += pair_count;
                    total_plain_seconds += plain_seconds;
                    total_noprefetch_seconds += noprefetch_seconds;
                }
            }};
        }

        bench_bucket_plain_vs_noprefetch!(1);
        bench_bucket_plain_vs_noprefetch!(2);
        bench_bucket_plain_vs_noprefetch!(3);
        bench_bucket_plain_vs_noprefetch!(4);
        bench_bucket_plain_vs_noprefetch!(5);
        bench_bucket_plain_vs_noprefetch!(6);
        bench_bucket_plain_vs_noprefetch!(7);
        bench_bucket_plain_vs_noprefetch!(8);

        if total_pair_count > 0 {
            let plain_ns = total_plain_seconds * 1_000_000_000.0 / total_pair_count as f64;
            let noprefetch_ns =
                total_noprefetch_seconds * 1_000_000_000.0 / total_pair_count as f64;
            println!("  overall_pair_count = {total_pair_count}");
            println!("  overall_plain_seconds = {total_plain_seconds:.12}");
            println!("  overall_noprefetch_seconds = {total_noprefetch_seconds:.12}");
            println!("  overall_plain_ns_per_candidate = {plain_ns:.12}");
            println!("  overall_noprefetch_ns_per_candidate = {noprefetch_ns:.12}");
            println!(
                "  overall_plain_vs_noprefetch_speedup = {:.12}x",
                total_plain_seconds / total_noprefetch_seconds
            );
        }
    }

    if mode == "preparedfirstdense_simd8_vs_lockstep4_swappedtight_noprefetch_bybucket" {
        let prepared_first_contiguous_swapped_tight_batch =
            match build_prepared_first_dense_contiguous_swapped_tight_batch(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild swapped-tight contiguous prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let prefetch_chunks_by_left_len = build_prefetch_chunks_by_left_len(candidate_second_buckets);
        let simd8_chunks_by_left_len = build_simd8_chunks_by_left_len(candidate_second_buckets);
        let noprefetch = PrefetchConfig {
            enabled: false,
            lookahead_chunks: 1,
            budget: 0,
            scope: 8,
            hint: PrefetchHint::T0,
        };

        println!("prepared_first_dense_simd8_vs_lockstep4_swappedtight_noprefetch_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  swapped_tight_prepared_first_count = {}",
            prepared_first_contiguous_swapped_tight_batch.len()
        );

        let mut total_pair_count = 0u64;
        let mut total_lockstep4_seconds = 0.0f64;
        let mut total_simd8_seconds = 0.0f64;

        macro_rules! bench_bucket_simd8_vs_lockstep4 {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                let prefetch_chunks = &prefetch_chunks_by_left_len[$left_len];
                let simd_chunks = &simd8_chunks_by_left_len[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, lockstep4_canonical_count, lockstep4_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch_param::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            prefetch_chunks,
                            noprefetch,
                        );
                    let (_simd_pair_count, simd8_canonical_count, simd8_seconds) =
                        time_bucket_simd8_contiguous_swapped_tight::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd_chunks,
                        );
                    let lockstep4_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_tight_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    let simd8_mismatches =
                        count_bucket_simd8_contiguous_swapped_tight_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd_chunks,
                        );

                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    lockstep4_canonical_count = {lockstep4_canonical_count}");
                    println!("    simd8_canonical_count = {simd8_canonical_count}");
                    println!("    lockstep4_mismatches = {lockstep4_mismatches}");
                    println!("    simd8_mismatches = {simd8_mismatches}");
                    println!("    lockstep4_seconds = {lockstep4_seconds:.12}");
                    println!("    simd8_seconds = {simd8_seconds:.12}");
                    println!(
                        "    lockstep4_vs_simd8_speedup = {:.12}x",
                        lockstep4_seconds / simd8_seconds
                    );

                    total_pair_count += pair_count;
                    total_lockstep4_seconds += lockstep4_seconds;
                    total_simd8_seconds += simd8_seconds;
                }
            }};
        }

        bench_bucket_simd8_vs_lockstep4!(1);
        bench_bucket_simd8_vs_lockstep4!(2);
        bench_bucket_simd8_vs_lockstep4!(3);
        bench_bucket_simd8_vs_lockstep4!(4);
        bench_bucket_simd8_vs_lockstep4!(5);
        bench_bucket_simd8_vs_lockstep4!(6);
        bench_bucket_simd8_vs_lockstep4!(7);
        bench_bucket_simd8_vs_lockstep4!(8);

        if total_pair_count > 0 {
            println!("  overall_pair_count = {total_pair_count}");
            println!(
                "  overall_lockstep4_ns_per_candidate = {:.12}",
                total_lockstep4_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_simd8_ns_per_candidate = {:.12}",
                total_simd8_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_lockstep4_vs_simd8_speedup = {:.12}x",
                total_lockstep4_seconds / total_simd8_seconds
            );
        }
    }

    if mode == "preparedfirstdense_simd8_vs_lockstep4_swappedtight_noprefetch_matrix_by_len" {
        let prepared_first_tight_by_right_len =
            match build_prepared_first_dense_contiguous_swapped_tight_batches_by_right_len(
                &tokenizer,
                &sampled_first_ids,
            ) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild swapped-tight contiguous prepared-first dense batches by right_len: {err}"
                    );
                    return ExitCode::from(1);
                }
            };
        let prefetch_chunks_by_left_len = build_prefetch_chunks_by_left_len(candidate_second_buckets);
        let simd8_chunks_by_left_len = build_simd8_chunks_by_left_len(candidate_second_buckets);
        let noprefetch = PrefetchConfig {
            enabled: false,
            lookahead_chunks: 1,
            budget: 0,
            scope: 8,
            hint: PrefetchHint::T0,
        };

        println!("prepared_first_dense_simd8_vs_lockstep4_swappedtight_noprefetch_matrix_by_len:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        for right_len in 0..=8 {
            println!(
                "  right_len = {right_len}, tight_prepared_first_count = {}",
                prepared_first_tight_by_right_len[right_len].len()
            );
        }
        println!("  speedup cell = lockstep4_seconds / simd8_seconds");

        let mut total_pair_count = 0u64;
        let mut total_lockstep4_seconds = 0.0f64;
        let mut total_simd8_seconds = 0.0f64;

        macro_rules! bench_left_len_matrix {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                let prefetch_chunks = &prefetch_chunks_by_left_len[$left_len];
                let simd_chunks = &simd8_chunks_by_left_len[$left_len];

                for right_len in 1..=8usize {
                    let prepared_batch = &prepared_first_tight_by_right_len[right_len];
                    if entries.is_empty() || prepared_batch.is_empty() {
                        println!(
                            "    cell(left_len={}, right_len={}): pair_count=0",
                            $left_len, right_len
                        );
                        continue;
                    }

                    let (pair_count, lockstep4_canonical_count, lockstep4_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch_param::<$left_len>(
                            prepared_batch,
                            entries,
                            prefetch_chunks,
                            noprefetch,
                        );
                    let (_simd_pair_count, simd8_canonical_count, simd8_seconds) =
                        time_bucket_simd8_contiguous_swapped_tight::<$left_len>(
                            prepared_batch,
                            entries,
                            simd_chunks,
                        );
                    let lockstep4_mismatches =
                        count_bucket_lockstep4_contiguous_swapped_tight_mismatches::<$left_len>(
                            prepared_batch,
                            entries,
                        );
                    let simd8_mismatches =
                        count_bucket_simd8_contiguous_swapped_tight_mismatches::<$left_len>(
                            prepared_batch,
                            entries,
                            simd_chunks,
                        );

                    println!(
                        "    cell(left_len={}, right_len={}): pair_count={pair_count}, lockstep4_canonical_count={lockstep4_canonical_count}, simd8_canonical_count={simd8_canonical_count}, lockstep4_mismatches={lockstep4_mismatches}, simd8_mismatches={simd8_mismatches}, lockstep4_seconds={lockstep4_seconds:.12}, simd8_seconds={simd8_seconds:.12}, lockstep4_vs_simd8_speedup={:.12}x",
                        $left_len,
                        right_len,
                        lockstep4_seconds / simd8_seconds
                    );

                    total_pair_count += pair_count;
                    total_lockstep4_seconds += lockstep4_seconds;
                    total_simd8_seconds += simd8_seconds;
                }
            }};
        }

        for left_len in 1..=8usize {
            println!("  left_len = {left_len}");
            match left_len {
                1 => bench_left_len_matrix!(1),
                2 => bench_left_len_matrix!(2),
                3 => bench_left_len_matrix!(3),
                4 => bench_left_len_matrix!(4),
                5 => bench_left_len_matrix!(5),
                6 => bench_left_len_matrix!(6),
                7 => bench_left_len_matrix!(7),
                8 => bench_left_len_matrix!(8),
                _ => unreachable!(),
            }
        }

        if total_pair_count > 0 {
            println!("  overall_pair_count = {total_pair_count}");
            println!(
                "  overall_lockstep4_ns_per_candidate = {:.12}",
                total_lockstep4_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_simd8_ns_per_candidate = {:.12}",
                total_simd8_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_lockstep4_vs_simd8_speedup = {:.12}x",
                total_lockstep4_seconds / total_simd8_seconds
            );
        }
    }

    if mode == "preparedfirstdense_simd4_vs_simd8_vs_lockstep4_swappedtight_noprefetch_bybucket" {
        let prepared_first_contiguous_swapped_tight_batch =
            match build_prepared_first_dense_contiguous_swapped_tight_batch(
                &tokenizer,
                &sampled_first_ids,
            ) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild swapped-tight contiguous prepared-first dense batch: {err}"
                    );
                    return ExitCode::from(1);
                }
            };
        let simd4_chunks_by_left_len = build_simd4_chunks_by_left_len(candidate_second_buckets);
        let simd8_chunks_by_left_len = build_simd8_chunks_by_left_len(candidate_second_buckets);
        println!("prepared_first_dense_simd4_vs_simd8_vs_lockstep4_swappedtight_noprefetch_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  swapped_tight_prepared_first_count = {}",
            prepared_first_contiguous_swapped_tight_batch.len()
        );

        macro_rules! bench_bucket_simd4_vs_simd8_vs_lockstep4 {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                let simd4_chunks = &simd4_chunks_by_left_len[$left_len];
                let simd8_chunks = &simd8_chunks_by_left_len[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, lockstep4_canonical_count, lockstep4_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    let (_simd4_pair_count, simd4_canonical_count, simd4_seconds) =
                        time_bucket_simd4_contiguous_swapped_tight::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd4_chunks,
                        );
                    let (_simd8_pair_count, simd8_canonical_count, simd8_seconds) =
                        time_bucket_simd8_contiguous_swapped_tight::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd8_chunks,
                        );
                    let simd4_mismatches =
                        count_bucket_simd4_contiguous_swapped_tight_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd4_chunks,
                        );
                    let simd8_mismatches =
                        count_bucket_simd8_contiguous_swapped_tight_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd8_chunks,
                        );

                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    lockstep4_canonical_count = {lockstep4_canonical_count}");
                    println!("    simd4_canonical_count = {simd4_canonical_count}");
                    println!("    simd8_canonical_count = {simd8_canonical_count}");
                    println!("    simd4_mismatches = {simd4_mismatches}");
                    println!("    simd8_mismatches = {simd8_mismatches}");
                    println!("    lockstep4_seconds = {lockstep4_seconds:.12}");
                    println!("    simd4_seconds = {simd4_seconds:.12}");
                    println!("    simd8_seconds = {simd8_seconds:.12}");
                    println!(
                        "    lockstep4_vs_simd4_speedup = {:.12}x",
                        lockstep4_seconds / simd4_seconds
                    );
                    println!(
                        "    lockstep4_vs_simd8_speedup = {:.12}x",
                        lockstep4_seconds / simd8_seconds
                    );
                    println!(
                        "    simd4_vs_simd8_speedup = {:.12}x",
                        simd4_seconds / simd8_seconds
                    );
                }
            }};
        }

        bench_bucket_simd4_vs_simd8_vs_lockstep4!(1);
        bench_bucket_simd4_vs_simd8_vs_lockstep4!(2);
        bench_bucket_simd4_vs_simd8_vs_lockstep4!(3);
        bench_bucket_simd4_vs_simd8_vs_lockstep4!(4);
        bench_bucket_simd4_vs_simd8_vs_lockstep4!(5);
        bench_bucket_simd4_vs_simd8_vs_lockstep4!(6);
        bench_bucket_simd4_vs_simd8_vs_lockstep4!(7);
        bench_bucket_simd4_vs_simd8_vs_lockstep4!(8);
    }

    if mode == "preparedfirstdense_simd16_vs_simd8_vs_lockstep4_swappedtight_noprefetch_bybucket" {
        let prepared_first_contiguous_swapped_tight_batch =
            match build_prepared_first_dense_contiguous_swapped_tight_batch(
                &tokenizer,
                &sampled_first_ids,
            ) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild swapped-tight contiguous prepared-first dense batch: {err}"
                    );
                    return ExitCode::from(1);
                }
            };
        let simd8_chunks_by_left_len = build_simd8_chunks_by_left_len(candidate_second_buckets);
        let simd16_chunks_by_left_len = build_simd16_chunks_by_left_len(candidate_second_buckets);
        println!("prepared_first_dense_simd16_vs_simd8_vs_lockstep4_swappedtight_noprefetch_bybucket:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        println!(
            "  swapped_tight_prepared_first_count = {}",
            prepared_first_contiguous_swapped_tight_batch.len()
        );

        let mut total_pair_count = 0u64;
        let mut total_lockstep4_seconds = 0.0f64;
        let mut total_simd8_seconds = 0.0f64;
        let mut total_simd16_seconds = 0.0f64;

        macro_rules! bench_bucket_simd16_vs_simd8_vs_lockstep4 {
            ($left_len:literal) => {{
                let entries = &candidate_second_buckets[$left_len];
                let simd8_chunks = &simd8_chunks_by_left_len[$left_len];
                let simd16_chunks = &simd16_chunks_by_left_len[$left_len];
                if entries.is_empty() {
                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = 0");
                } else {
                    let (pair_count, lockstep4_canonical_count, lockstep4_seconds) =
                        time_bucket_lockstep4_contiguous_swapped_tight::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                        );
                    let (_simd8_pair_count, simd8_canonical_count, simd8_seconds) =
                        time_bucket_simd8_contiguous_swapped_tight::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd8_chunks,
                        );
                    let (_simd16_pair_count, simd16_canonical_count, simd16_seconds) =
                        time_bucket_simd16_contiguous_swapped_tight::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd16_chunks,
                        );
                    let simd8_mismatches =
                        count_bucket_simd8_contiguous_swapped_tight_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd8_chunks,
                        );
                    let simd16_mismatches =
                        count_bucket_simd16_contiguous_swapped_tight_mismatches::<$left_len>(
                            &prepared_first_contiguous_swapped_tight_batch,
                            entries,
                            simd16_chunks,
                        );

                    println!("  left_len = {}", $left_len);
                    println!("    pair_count = {pair_count}");
                    println!("    lockstep4_canonical_count = {lockstep4_canonical_count}");
                    println!("    simd8_canonical_count = {simd8_canonical_count}");
                    println!("    simd16_canonical_count = {simd16_canonical_count}");
                    println!("    simd8_mismatches = {simd8_mismatches}");
                    println!("    simd16_mismatches = {simd16_mismatches}");
                    println!("    lockstep4_seconds = {lockstep4_seconds:.12}");
                    println!("    simd8_seconds = {simd8_seconds:.12}");
                    println!("    simd16_seconds = {simd16_seconds:.12}");
                    println!(
                        "    lockstep4_vs_simd8_speedup = {:.12}x",
                        lockstep4_seconds / simd8_seconds
                    );
                    println!(
                        "    lockstep4_vs_simd16_speedup = {:.12}x",
                        lockstep4_seconds / simd16_seconds
                    );
                    println!(
                        "    simd8_vs_simd16_speedup = {:.12}x",
                        simd8_seconds / simd16_seconds
                    );

                    total_pair_count += pair_count;
                    total_lockstep4_seconds += lockstep4_seconds;
                    total_simd8_seconds += simd8_seconds;
                    total_simd16_seconds += simd16_seconds;
                }
            }};
        }

        bench_bucket_simd16_vs_simd8_vs_lockstep4!(1);
        bench_bucket_simd16_vs_simd8_vs_lockstep4!(2);
        bench_bucket_simd16_vs_simd8_vs_lockstep4!(3);
        bench_bucket_simd16_vs_simd8_vs_lockstep4!(4);
        bench_bucket_simd16_vs_simd8_vs_lockstep4!(5);
        bench_bucket_simd16_vs_simd8_vs_lockstep4!(6);
        bench_bucket_simd16_vs_simd8_vs_lockstep4!(7);
        bench_bucket_simd16_vs_simd8_vs_lockstep4!(8);

        if total_pair_count > 0 {
            println!("  overall_pair_count = {total_pair_count}");
            println!(
                "  overall_lockstep4_ns_per_candidate = {:.12}",
                total_lockstep4_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_simd8_ns_per_candidate = {:.12}",
                total_simd8_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_simd16_ns_per_candidate = {:.12}",
                total_simd16_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_lockstep4_vs_simd16_speedup = {:.12}x",
                total_lockstep4_seconds / total_simd16_seconds
            );
            println!(
                "  overall_simd8_vs_simd16_speedup = {:.12}x",
                total_simd8_seconds / total_simd16_seconds
            );
        }
    }

    // Note (Mar 2026): on the current TinyLlama sample benchmark settings
    // (samples=1000, seed=11), this matrix mode showed a small overall win for
    // swappedtight_noprefetch vs plaincontiguous (~1-2% ns/candidate), while
    // per-cell winners were mixed across (left_len, right_len). Keep this mode
    // as the canonical way to re-check that tradeoff after kernel/prefetch edits.
    if mode == "preparedfirstdense_lockstep4_plain_vs_swappedtight_noprefetch_matrix_by_len" {
        let prepared_first_plain_by_right_len =
            match build_prepared_first_dense_contiguous_batches_by_right_len(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild contiguous prepared-first dense batches by right_len: {err}"
                    );
                    return ExitCode::from(1);
                }
            };
        let prepared_first_tight_by_right_len =
            match build_prepared_first_dense_contiguous_swapped_tight_batches_by_right_len(
                &tokenizer,
                &sampled_first_ids,
            ) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild swapped-tight contiguous prepared-first dense batches by right_len: {err}"
                    );
                    return ExitCode::from(1);
                }
            };
        let prefetch_chunks_by_left_len = build_prefetch_chunks_by_left_len(candidate_second_buckets);
        let noprefetch = PrefetchConfig {
            enabled: false,
            lookahead_chunks: 1,
            budget: 0,
            scope: 8,
            hint: PrefetchHint::T0,
        };

        println!("prepared_first_dense_lockstep4_plain_vs_swappedtight_noprefetch_matrix_by_len:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        for right_len in 0..=8 {
            println!(
                "  right_len = {right_len}, plain_prepared_first_count = {}, tight_prepared_first_count = {}",
                prepared_first_plain_by_right_len[right_len].len(),
                prepared_first_tight_by_right_len[right_len].len()
            );
        }
        println!("  speedup cell = plain_seconds / noprefetch_seconds");

        let mut total_pair_count = 0u64;
        let mut total_plain_seconds = 0.0f64;
        let mut total_noprefetch_seconds = 0.0f64;

        for left_len in 1..=8usize {
            let entries = &candidate_second_buckets[left_len];
            let prefetch_chunks = &prefetch_chunks_by_left_len[left_len];
            println!("  left_len = {left_len}, second_bucket_count = {}", entries.len());
            for right_len in 1..=8usize {
                let plain_batch = &prepared_first_plain_by_right_len[right_len];
                let tight_batch = &prepared_first_tight_by_right_len[right_len];
                if entries.is_empty() || plain_batch.is_empty() || tight_batch.is_empty() {
                    println!("    cell(left_len={left_len}, right_len={right_len}): pair_count=0");
                    continue;
                }

                macro_rules! time_plain {
                    ($ll:literal) => {{
                        time_bucket_lockstep4_contiguous::<$ll>(plain_batch, entries)
                    }};
                }
                macro_rules! time_noprefetch {
                    ($ll:literal) => {{
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch_param::<$ll>(
                            tight_batch,
                            entries,
                            prefetch_chunks,
                            noprefetch,
                        )
                    }};
                }

                let (pair_count, plain_canonical_count, plain_seconds) = match left_len {
                    1 => time_plain!(1),
                    2 => time_plain!(2),
                    3 => time_plain!(3),
                    4 => time_plain!(4),
                    5 => time_plain!(5),
                    6 => time_plain!(6),
                    7 => time_plain!(7),
                    8 => time_plain!(8),
                    _ => unreachable!("left_len buckets are in 1..=8"),
                };
                let (_pair_count_np, noprefetch_canonical_count, noprefetch_seconds) = match left_len {
                    1 => time_noprefetch!(1),
                    2 => time_noprefetch!(2),
                    3 => time_noprefetch!(3),
                    4 => time_noprefetch!(4),
                    5 => time_noprefetch!(5),
                    6 => time_noprefetch!(6),
                    7 => time_noprefetch!(7),
                    8 => time_noprefetch!(8),
                    _ => unreachable!("left_len buckets are in 1..=8"),
                };
                println!(
                    "    cell(left_len={left_len}, right_len={right_len}): pair_count={pair_count}, plain_canonical_count={plain_canonical_count}, noprefetch_canonical_count={noprefetch_canonical_count}, plain_seconds={plain_seconds:.12}, noprefetch_seconds={noprefetch_seconds:.12}, plain_vs_noprefetch_speedup={:.12}x",
                    plain_seconds / noprefetch_seconds
                );

                total_pair_count += pair_count;
                total_plain_seconds += plain_seconds;
                total_noprefetch_seconds += noprefetch_seconds;
            }
        }

        if total_pair_count > 0 {
            println!("  overall_pair_count = {total_pair_count}");
            println!(
                "  overall_plain_ns_per_candidate = {:.12}",
                total_plain_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_noprefetch_ns_per_candidate = {:.12}",
                total_noprefetch_seconds * 1_000_000_000.0 / total_pair_count as f64
            );
            println!(
                "  overall_plain_vs_noprefetch_speedup = {:.12}x",
                total_plain_seconds / total_noprefetch_seconds
            );
        }
    }

    if mode == "preparedfirstdense_lockstep4_swappedtightprefetch1_matrix_by_len" {
        let prepared_first_by_right_len =
            match build_prepared_first_dense_contiguous_swapped_tight_batches_by_right_len(
                &tokenizer,
                &sampled_first_ids,
            ) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild swapped-tight contiguous prepared-first dense batches by right_len: {err}"
                    );
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_swappedtightprefetch1_matrix_by_len:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        for right_len in 0..=8 {
            println!(
                "  right_len = {right_len}, prepared_first_count = {}",
                prepared_first_by_right_len[right_len].len()
            );
        }
        println!("  timing_seconds_matrix_rows = left_len (1..8), cols = right_len (1..8)");
        for left_len in 1..=8 {
            let entries = &candidate_second_buckets[left_len];
            println!("  left_len = {left_len}, second_bucket_count = {}", entries.len());
            for right_len in 1..=8 {
                let prepared_batch = &prepared_first_by_right_len[right_len];
                if entries.is_empty() || prepared_batch.is_empty() {
                    println!(
                        "    cell(left_len={left_len}, right_len={right_len}): pair_count=0"
                    );
                    continue;
                }
                macro_rules! time_cell {
                    ($ll:literal) => {{
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch::<$ll>(
                            prepared_batch,
                            entries,
                        )
                    }};
                }
                let (pair_count, canonical_count, seconds) = match left_len {
                    1 => time_cell!(1),
                    2 => time_cell!(2),
                    3 => time_cell!(3),
                    4 => time_cell!(4),
                    5 => time_cell!(5),
                    6 => time_cell!(6),
                    7 => time_cell!(7),
                    8 => time_cell!(8),
                    _ => unreachable!("left_len buckets are in 1..=8"),
                };
                let ns_per_candidate = seconds * 1_000_000_000.0 / pair_count as f64;
                println!(
                    "    cell(left_len={left_len}, right_len={right_len}): pair_count={pair_count}, canonical_count={canonical_count}, seconds={seconds:.12}, ns_per_candidate={ns_per_candidate:.12}"
                );
            }
        }
    }

    if mode == "preparedfirstdense_lockstep4_plain_vs_swappedtightprefetch1_matrix_by_len" {
        let prepared_first_plain_by_right_len =
            match build_prepared_first_dense_contiguous_batches_by_right_len(&tokenizer, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild contiguous prepared-first dense batches by right_len: {err}"
                    );
                    return ExitCode::from(1);
                }
            };
        let prepared_first_tight_by_right_len =
            match build_prepared_first_dense_contiguous_swapped_tight_batches_by_right_len(
                &tokenizer,
                &sampled_first_ids,
            ) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!(
                        "failed to prebuild swapped-tight contiguous prepared-first dense batches by right_len: {err}"
                    );
                    return ExitCode::from(1);
                }
            };

        println!("prepared_first_dense_lockstep4_plain_vs_swappedtightprefetch1_matrix_by_len:");
        println!("  piece_count = {}", TINYLLAMA_PIECE_COUNT);
        for right_len in 0..=8 {
            println!(
                "  right_len = {right_len}, plain_prepared_first_count = {}, tight_prepared_first_count = {}",
                prepared_first_plain_by_right_len[right_len].len(),
                prepared_first_tight_by_right_len[right_len].len()
            );
        }
        println!("  speedup_matrix_rows = left_len (1..8), cols = right_len (1..8)");
        println!("  speedup cell = plain_seconds / tight_prefetch1_seconds");
        for left_len in 1..=8 {
            let entries = &candidate_second_buckets[left_len];
            println!("  left_len = {left_len}, second_bucket_count = {}", entries.len());
            for right_len in 1..=8 {
                let plain_batch = &prepared_first_plain_by_right_len[right_len];
                let tight_batch = &prepared_first_tight_by_right_len[right_len];
                if entries.is_empty() || plain_batch.is_empty() || tight_batch.is_empty() {
                    println!(
                        "    cell(left_len={left_len}, right_len={right_len}): pair_count=0"
                    );
                    continue;
                }
                macro_rules! time_cell_plain {
                    ($ll:literal) => {{
                        time_bucket_lockstep4_contiguous::<$ll>(plain_batch, entries)
                    }};
                }
                macro_rules! time_cell_tight {
                    ($ll:literal) => {{
                        time_bucket_lockstep4_contiguous_swapped_tight_prefetch::<$ll>(
                            tight_batch,
                            entries,
                        )
                    }};
                }
                let (pair_count, plain_canonical_count, plain_seconds) = match left_len {
                    1 => time_cell_plain!(1),
                    2 => time_cell_plain!(2),
                    3 => time_cell_plain!(3),
                    4 => time_cell_plain!(4),
                    5 => time_cell_plain!(5),
                    6 => time_cell_plain!(6),
                    7 => time_cell_plain!(7),
                    8 => time_cell_plain!(8),
                    _ => unreachable!("left_len buckets are in 1..=8"),
                };
                let (_pair_count_tight, tight_canonical_count, tight_seconds) = match left_len {
                    1 => time_cell_tight!(1),
                    2 => time_cell_tight!(2),
                    3 => time_cell_tight!(3),
                    4 => time_cell_tight!(4),
                    5 => time_cell_tight!(5),
                    6 => time_cell_tight!(6),
                    7 => time_cell_tight!(7),
                    8 => time_cell_tight!(8),
                    _ => unreachable!("left_len buckets are in 1..=8"),
                };
                let speedup = plain_seconds / tight_seconds;
                println!(
                    "    cell(left_len={left_len}, right_len={right_len}): pair_count={pair_count}, plain_canonical_count={plain_canonical_count}, tight_prefetch_canonical_count={tight_canonical_count}, plain_seconds={plain_seconds:.12}, tight_prefetch_seconds={tight_seconds:.12}, plain_vs_tightprefetch_speedup={speedup:.12}x"
                );
            }
        }
    }
    ExitCode::SUCCESS
}
