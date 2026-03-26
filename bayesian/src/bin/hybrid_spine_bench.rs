use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use bayesian::bpe::{
    BpeMerges, PackedSpine, TinyLlamaWordTokenizer, MAX_PACKED_SPINE_LEN,
};
use serde_json::Value;

#[derive(Clone, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0xdead_beef_cafe_babe } else { seed };
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

#[derive(Clone, Copy, Debug)]
struct RowEntry {
    right: u16,
    rank: u32,
}

const TINYLLAMA_PIECE_COUNT: usize = 32_000;
static ZERO_CROSS_RANK_ROW: [u16; TINYLLAMA_PIECE_COUNT] = [0; TINYLLAMA_PIECE_COUNT];

#[derive(Debug)]
struct MergeRows {
    rows: Vec<Vec<RowEntry>>,
    piece_count: usize,
}

#[derive(Debug)]
struct PreparedFirstDense {
    right_spine: PackedSpine,
    right_len: u8,
    right_rank_plus_one: [u16; MAX_PACKED_SPINE_LEN],
    cross_rank_row_ptrs: [*const u16; MAX_PACKED_SPINE_LEN],
    _dense_rows: [Option<Box<[u16; TINYLLAMA_PIECE_COUNT]>>; MAX_PACKED_SPINE_LEN],
}

#[derive(Clone, Copy, Debug)]
struct CompactLeftSpine {
    len: u8,
    ids: [u16; MAX_PACKED_SPINE_LEN],
    rank_plus_one: [u16; MAX_PACKED_SPINE_LEN],
}

type CompactLeftSpineBuckets = [Vec<CompactLeftSpine>; MAX_PACKED_SPINE_LEN + 1];

impl MergeRows {
    fn build(tokenizer_path: &str, merges_graph: &BpeMerges) -> Result<Self, String> {
        let text = std::fs::read_to_string(tokenizer_path)
            .map_err(|err| format!("failed to read tokenizer json: {err}"))?;
        let json: Value =
            serde_json::from_str(&text).map_err(|err| format!("failed to parse tokenizer json: {err}"))?;
        let merges = json
            .get("model")
            .and_then(|model| model.get("merges"))
            .and_then(Value::as_array)
            .ok_or_else(|| "missing model.merges array".to_string())?;

        let piece_count = piece_count(merges_graph);
        let mut rows = vec![Vec::<RowEntry>::new(); piece_count];

        for (rank, item) in merges.iter().enumerate() {
            let line = item
                .as_str()
                .ok_or_else(|| "merge entry is not a string".to_string())?;
            let (left, right) = line
                .split_once(' ')
                .ok_or_else(|| format!("bad merge line: {line:?}"))?;
            let left_id = merges_graph
                .encode_piece(left)
                .ok_or_else(|| format!("unknown left piece in merges graph: {left:?}"))?;
            let right_id = merges_graph
                .encode_piece(right)
                .ok_or_else(|| format!("unknown right piece in merges graph: {right:?}"))?;
            rows[left_id as usize].push(RowEntry {
                right: u16::try_from(right_id)
                    .map_err(|_| "piece ids no longer fit in u16".to_string())?,
                rank: rank as u32,
            });
        }

        Ok(Self { rows, piece_count })
    }
}

impl PreparedFirstDense {
    fn build(
        first_right_spine: PackedSpine,
        merge_rows: &MergeRows,
    ) -> Result<Self, String> {
        if merge_rows.piece_count != TINYLLAMA_PIECE_COUNT {
            return Err(format!(
                "expected TinyLlama piece count {}, got {}",
                TINYLLAMA_PIECE_COUNT, merge_rows.piece_count
            ));
        }

        let mut right_rank_plus_one = [0u16; MAX_PACKED_SPINE_LEN];
        let zero_row_ptr = ZERO_CROSS_RANK_ROW.as_ptr();
        let mut cross_rank_row_ptrs = [zero_row_ptr; MAX_PACKED_SPINE_LEN];
        let mut dense_rows: [Option<Box<[u16; TINYLLAMA_PIECE_COUNT]>>; MAX_PACKED_SPINE_LEN] =
            std::array::from_fn(|_| None);

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            right_rank_plus_one[spine_idx] = spine_entry.rank_plus_one;
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            let mut dense_row: Box<[u16; TINYLLAMA_PIECE_COUNT]> = vec![0u16; TINYLLAMA_PIECE_COUNT]
                .into_boxed_slice()
                .try_into()
                .map_err(|_| "failed to allocate fixed TinyLlama dense row".to_string())?;
            for entry in row {
                let rank_plus_one = entry
                    .rank
                    .checked_add(1)
                    .and_then(|value| u16::try_from(value).ok())
                    .ok_or_else(|| "merge rank no longer fits in u16".to_string())?;
                dense_row[entry.right as usize] = rank_plus_one;
            }
            cross_rank_row_ptrs[spine_idx] = dense_row.as_ptr();
            dense_rows[spine_idx] = Some(dense_row);
        }

        Ok(Self {
            right_spine: first_right_spine,
            right_len: first_right_spine.as_slice().len() as u8,
            right_rank_plus_one,
            cross_rank_row_ptrs,
            _dense_rows: dense_rows,
        })
    }
}

impl CompactLeftSpine {
    fn from_packed(packed: PackedSpine) -> Self {
        let mut compact = Self {
            len: 0,
            ids: [0; MAX_PACKED_SPINE_LEN],
            rank_plus_one: [0; MAX_PACKED_SPINE_LEN],
        };
        let entries = packed.as_slice();
        compact.len = entries.len() as u8;
        for (idx, entry) in entries.iter().enumerate() {
            compact.ids[idx] = entry.id;
            compact.rank_plus_one[idx] = entry.rank_plus_one;
        }
        compact
    }
}

fn bucket_compact_left_spines(spines: &[CompactLeftSpine]) -> CompactLeftSpineBuckets {
    let mut buckets: CompactLeftSpineBuckets = std::array::from_fn(|_| Vec::new());
    for &spine in spines {
        buckets[spine.len as usize].push(spine);
    }
    buckets
}

fn piece_count(merges_graph: &BpeMerges) -> usize {
    let mut count = 0usize;
    while merges_graph.decode_piece(count as u32).is_some() {
        count += 1;
    }
    count
}

#[inline(always)]
fn canonical_pair_from_prepared_first_dense_left_len<const LEFT_LEN: usize>(
    prepared_first: &PreparedFirstDense,
    left_spine: &CompactLeftSpine,
) -> bool {
    debug_assert_eq!(left_spine.len as usize, LEFT_LEN);
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return false;
    }

    let right_spine_len = prepared_first.right_len as usize;
    let right_rank_plus_one = &prepared_first.right_rank_plus_one;
    let cross_rank_row_ptrs = &prepared_first.cross_rank_row_ptrs;
    let left_ids = &left_spine.ids;
    let left_rank_plus_one = &left_spine.rank_plus_one;
    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        debug_assert!(i < right_spine_len);
        debug_assert!(j < LEFT_LEN);

        // These indices only advance when the current node exposes a next rank.
        let right_rank_plus_one = unsafe { *right_rank_plus_one.get_unchecked(i) };
        let left_id = unsafe { *left_ids.get_unchecked(j) };
        let left_rank_plus_one = unsafe { *left_rank_plus_one.get_unchecked(j) };
        debug_assert!((left_id as usize) < TINYLLAMA_PIECE_COUNT);
        let cross_rank_plus_one =
            unsafe { *(*cross_rank_row_ptrs.get_unchecked(i)).add(left_id as usize) };

        let mut best_rank_plus_one = right_rank_plus_one;
        if left_rank_plus_one != 0
            && (best_rank_plus_one == 0 || left_rank_plus_one < best_rank_plus_one)
        {
            best_rank_plus_one = left_rank_plus_one;
        }
        if cross_rank_plus_one != 0
            && (best_rank_plus_one == 0 || cross_rank_plus_one < best_rank_plus_one)
        {
            best_rank_plus_one = cross_rank_plus_one;
        }

        if best_rank_plus_one == 0 {
            return true;
        }

        if cross_rank_plus_one == best_rank_plus_one {
            return false;
        }
        if right_rank_plus_one == best_rank_plus_one {
            i += 1;
            continue;
        }
        if left_rank_plus_one == best_rank_plus_one {
            j += 1;
            continue;
        }

        unreachable!("best rank must come from one of the three candidate events");
    }
}

fn canonical_pair_from_prepared_first_dense(
    prepared_first: &PreparedFirstDense,
    left_spine: &CompactLeftSpine,
) -> bool {
    match left_spine.len {
        0 => false,
        1 => canonical_pair_from_prepared_first_dense_left_len::<1>(prepared_first, left_spine),
        2 => canonical_pair_from_prepared_first_dense_left_len::<2>(prepared_first, left_spine),
        3 => canonical_pair_from_prepared_first_dense_left_len::<3>(prepared_first, left_spine),
        4 => canonical_pair_from_prepared_first_dense_left_len::<4>(prepared_first, left_spine),
        5 => canonical_pair_from_prepared_first_dense_left_len::<5>(prepared_first, left_spine),
        6 => canonical_pair_from_prepared_first_dense_left_len::<6>(prepared_first, left_spine),
        7 => canonical_pair_from_prepared_first_dense_left_len::<7>(prepared_first, left_spine),
        8 => canonical_pair_from_prepared_first_dense_left_len::<8>(prepared_first, left_spine),
        _ => unreachable!("packed spine len must fit MAX_PACKED_SPINE_LEN"),
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
            black_box(tokenizer.canonical_pair_from_packed_spines(&first_right_spine, second_left_spine));
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
    merge_rows: &MergeRows,
    sampled_first_ids: &[u32],
    candidate_second_buckets: &CompactLeftSpineBuckets,
) -> Result<(u64, u64, u64, f64, f64, f64), String> {
    let total_start = Instant::now();
    let mut pair_count = 0u64;
    let mut used_first_ids = 0u64;
    let mut canonical_count = 0u64;
    let mut prepare_seconds = 0.0f64;
    let mut scan_seconds = 0.0f64;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        let prepare_start = Instant::now();
        let prepared_first = PreparedFirstDense::build(first_right_spine, merge_rows)?;
        prepare_seconds += prepare_start.elapsed().as_secs_f64();

        used_first_ids += 1;

        let scan_start = Instant::now();
        macro_rules! scan_bucket {
            ($left_len:literal) => {
                for second_left_spine in &candidate_second_buckets[$left_len] {
                    if black_box(canonical_pair_from_prepared_first_dense_left_len::<$left_len>(
                        &prepared_first,
                        second_left_spine,
                    )) {
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
    merge_rows: &MergeRows,
    sampled_first_ids: &[u32],
) -> Result<Vec<PreparedFirstDense>, String> {
    let mut prepared = Vec::with_capacity(sampled_first_ids.len());
    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        prepared.push(PreparedFirstDense::build(first_right_spine, merge_rows)?);
    }
    Ok(prepared)
}

fn time_prepared_first_dense_scan_only(
    prepared_first_batch: &[PreparedFirstDense],
    candidate_second_buckets: &CompactLeftSpineBuckets,
) -> (u64, u64, u64, f64) {
    let timed_start = Instant::now();
    let mut pair_count = 0u64;
    let used_first_ids = prepared_first_batch.len() as u64;
    let mut canonical_count = 0u64;

    for prepared_first in prepared_first_batch {
        macro_rules! scan_bucket {
            ($left_len:literal) => {
                for second_left_spine in &candidate_second_buckets[$left_len] {
                    if black_box(canonical_pair_from_prepared_first_dense_left_len::<$left_len>(
                        prepared_first,
                        second_left_spine,
                    )) {
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
    merge_rows: &MergeRows,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
    candidate_second_compact_spines: &[CompactLeftSpine],
) -> Result<u64, String> {
    let mut mismatches = 0u64;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        let prepared_first = PreparedFirstDense::build(first_right_spine, merge_rows)?;
        for (second_left_spine, second_left_compact_spine) in
            candidate_second_spines.iter().zip(candidate_second_compact_spines.iter())
        {
            let baseline =
                tokenizer.canonical_pair_from_packed_spines(&prepared_first.right_spine, second_left_spine);
            let prepared = canonical_pair_from_prepared_first_dense(&prepared_first, second_left_compact_spine);
            if baseline != prepared {
                mismatches += 1;
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
    let merges_graph = match BpeMerges::from_tokenizer_json(&tokenizer_path) {
        Ok(merges_graph) => merges_graph,
        Err(err) => {
            eprintln!("failed to load merge graph: {err}");
            return ExitCode::from(1);
        }
    };
    let merge_rows = match MergeRows::build(&tokenizer_path, &merges_graph) {
        Ok(merge_rows) => merge_rows,
        Err(err) => {
            eprintln!("failed to build merge rows: {err}");
            return ExitCode::from(1);
        }
    };
    let load_plus_precompute_seconds = load_start.elapsed().as_secs_f64();

    let mut candidate_second_ids = tokenizer.token_ids_with_left_spines().to_vec();
    candidate_second_ids.sort_unstable();
    let mut candidate_second_entries: Vec<(u32, PackedSpine, CompactLeftSpine)> = candidate_second_ids
        .iter()
        .map(|&token_id| {
            let packed = tokenizer
                .left_packed_spine_for_token_id(token_id)
                .expect("sorted candidate ids must resolve to packed left spines")
                .to_owned();
            let compact = CompactLeftSpine::from_packed(packed);
            (token_id, packed, compact)
        })
        .collect();
    candidate_second_entries.sort_by(|(a_id, _a_packed, a_compact), (b_id, _b_packed, b_compact)| {
        a_compact.ids[..a_compact.len as usize]
            .cmp(&b_compact.ids[..b_compact.len as usize])
            .then_with(|| a_id.cmp(b_id))
    });
    let candidate_second_spines: Vec<PackedSpine> = candidate_second_entries
        .iter()
        .map(|(_, packed, _)| *packed)
        .collect();
    let candidate_second_compact_spines: Vec<CompactLeftSpine> = candidate_second_entries
        .iter()
        .map(|(_, _, compact)| *compact)
        .collect();
    let candidate_second_compact_buckets = bucket_compact_left_spines(&candidate_second_compact_spines);
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
        let (pair_count, used_first_ids, canonical_count, elapsed_seconds, prepare_seconds, scan_seconds) =
            match time_prepared_first_dense_split(
                &tokenizer,
                &merge_rows,
                &sampled_first_ids,
                &candidate_second_compact_buckets,
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
                &merge_rows,
                &sampled_first_ids,
                &candidate_second_spines,
                &candidate_second_compact_spines,
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
        println!("  piece_count = {}", merge_rows.piece_count);
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
            match build_prepared_first_dense_batch(&tokenizer, &merge_rows, &sampled_first_ids) {
                Ok(prepared) => prepared,
                Err(err) => {
                    eprintln!("failed to prebuild prepared-first dense batch: {err}");
                    return ExitCode::from(1);
                }
            };
        let build_seconds = build_start.elapsed().as_secs_f64();
        let (pair_count, used_first_ids, canonical_count, elapsed_seconds) =
            time_prepared_first_dense_scan_only(&prepared_first_batch, &candidate_second_compact_buckets);

        println!("prepared_first_dense_scan:");
        println!("  piece_count = {}", merge_rows.piece_count);
        println!("  prepared_first_count = {}", prepared_first_batch.len());
        println!("  build_seconds = {build_seconds:.6}");
        println!("  canonical_count = {canonical_count}");
        print_timing("  timing", pair_count, used_first_ids, elapsed_seconds);
    }
    ExitCode::SUCCESS
}
