use std::collections::HashMap;
use std::collections::BTreeMap;
use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use bayesian::bpe::{
    BpeMerges, PackedSpine, SpineEntry, TinyLlamaWordTokenizer, MAX_PACKED_SPINE_LEN,
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

const SMALL_ROW_LIMIT_15: usize = 15;
const HEAVY_ROW_SENTINEL: u8 = u8::MAX;
const EMPTY_HEAVY_SLOT: u16 = u16::MAX;
const TINYLLAMA_PIECE_COUNT: usize = 32_000;

#[derive(Clone, Copy, Debug)]
struct SmallRow15 {
    len: u8,
    rights: [u16; SMALL_ROW_LIMIT_15],
    ranks: [u32; SMALL_ROW_LIMIT_15],
}

impl SmallRow15 {
    fn empty() -> Self {
        Self {
            len: 0,
            rights: [0; SMALL_ROW_LIMIT_15],
            ranks: [0; SMALL_ROW_LIMIT_15],
        }
    }
}

#[derive(Debug)]
struct HybridLookup {
    threshold: usize,
    small_row_starts: Vec<u32>,
    small_row_lens: Vec<u16>,
    fallback_rows: Vec<bool>,
    small_rows: Vec<RowEntry>,
    fallback: HashMap<(u32, u32), u32>,
}

#[derive(Debug)]
struct SpecializedLookup15 {
    rows: Vec<SmallRow15>,
    fallback: HashMap<(u16, u16), u32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct HeavyRowDesc {
    start: u32,
    mask: u16,
    seed: u32,
}

#[derive(Debug)]
struct SpecializedLookup15HeavyHash {
    rows: Vec<SmallRow15>,
    heavy_descs: Vec<HeavyRowDesc>,
    heavy_rights: Vec<u16>,
    heavy_ranks: Vec<u32>,
}

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
    _cross_rank_tables: Box<[[u16; TINYLLAMA_PIECE_COUNT]; MAX_PACKED_SPINE_LEN]>,
}

#[derive(Clone, Copy, Debug)]
struct CompactLeftSpine {
    len: u8,
    ids: [u16; MAX_PACKED_SPINE_LEN],
    rank_plus_one: [u16; MAX_PACKED_SPINE_LEN],
}

type CompactLeftSpineBuckets = [Vec<CompactLeftSpine>; MAX_PACKED_SPINE_LEN + 1];

#[derive(Debug, Default)]
struct LookupStats32 {
    total_lookups: u64,
    fallback_lookups: u64,
    fallback_hits: u64,
    fallback_queries: Vec<(u32, u32)>,
}

#[derive(Debug, Default)]
struct LookupStats16 {
    total_lookups: u64,
    fallback_lookups: u64,
    fallback_hits: u64,
    fallback_queries: Vec<(u16, u16)>,
}

impl HybridLookup {
    fn build(
        tokenizer_path: &str,
        merges_graph: &BpeMerges,
        threshold: usize,
    ) -> Result<Self, String> {
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

        let mut small_row_starts = vec![0u32; piece_count];
        let mut small_row_lens = vec![0u16; piece_count];
        let mut fallback_rows = vec![false; piece_count];
        let mut small_rows = Vec::new();
        let mut fallback = HashMap::new();

        for (left_id, mut row) in rows.into_iter().enumerate() {
            if row.len() <= threshold {
                small_row_starts[left_id] = small_rows.len() as u32;
                small_row_lens[left_id] =
                    u16::try_from(row.len()).map_err(|_| "small row len overflow".to_string())?;
                small_rows.append(&mut row);
            } else {
                fallback_rows[left_id] = true;
                for entry in row {
                    fallback.insert((left_id as u32, entry.right as u32), entry.rank);
                }
            }
        }

        Ok(Self {
            threshold,
            small_row_starts,
            small_row_lens,
            fallback_rows,
            small_rows,
            fallback,
        })
    }

    fn lookup_rank(&self, left: u16, right: u16) -> Option<u32> {
        let left_idx = left as usize;
        if self.fallback_rows[left_idx] {
            self.fallback.get(&(left as u32, right as u32)).copied()
        } else {
            let start = self.small_row_starts[left_idx] as usize;
            let len = self.small_row_lens[left_idx] as usize;
            for entry in &self.small_rows[start..start + len] {
                if entry.right == right {
                    return Some(entry.rank);
                }
            }
            None
        }
    }
}

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

        let mut cross_rank_tables: Box<[[u16; TINYLLAMA_PIECE_COUNT]; MAX_PACKED_SPINE_LEN]> =
            vec![[0u16; TINYLLAMA_PIECE_COUNT]; MAX_PACKED_SPINE_LEN]
                .into_boxed_slice()
                .try_into()
                .map_err(|_| "failed to allocate fixed TinyLlama cross-rank tables".to_string())?;
        let mut right_rank_plus_one = [0u16; MAX_PACKED_SPINE_LEN];
        let mut cross_rank_row_ptrs = [std::ptr::null(); MAX_PACKED_SPINE_LEN];

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            let table = &mut cross_rank_tables[spine_idx];
            right_rank_plus_one[spine_idx] = spine_entry.rank_plus_one;
            cross_rank_row_ptrs[spine_idx] = table.as_ptr();
            for entry in &merge_rows.rows[spine_entry.id as usize] {
                let rank_plus_one = entry
                    .rank
                    .checked_add(1)
                    .and_then(|value| u16::try_from(value).ok())
                    .ok_or_else(|| "merge rank no longer fits in u16".to_string())?;
                table[entry.right as usize] = rank_plus_one;
            }
        }

        Ok(Self {
            right_spine: first_right_spine,
            right_len: first_right_spine.as_slice().len() as u8,
            right_rank_plus_one,
            cross_rank_row_ptrs,
            _cross_rank_tables: cross_rank_tables,
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

impl SpecializedLookup15 {
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
        let mut grouped_rows = vec![Vec::<RowEntry>::new(); piece_count];

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
            grouped_rows[left_id as usize].push(RowEntry {
                right: u16::try_from(right_id)
                    .map_err(|_| "piece ids no longer fit in u16".to_string())?,
                rank: rank as u32,
            });
        }

        let mut rows = vec![SmallRow15::empty(); piece_count];
        let mut fallback = HashMap::new();

        for (left_id, row) in grouped_rows.into_iter().enumerate() {
            if row.len() <= SMALL_ROW_LIMIT_15 {
                let mut small_row = SmallRow15::empty();
                small_row.len = row.len() as u8;
                for (slot, entry) in row.into_iter().enumerate() {
                    small_row.rights[slot] = entry.right;
                    small_row.ranks[slot] = entry.rank;
                }
                rows[left_id] = small_row;
            } else {
                rows[left_id].len = HEAVY_ROW_SENTINEL;
                for entry in row {
                    fallback.insert((left_id as u16, entry.right), entry.rank);
                }
            }
        }

        Ok(Self { rows, fallback })
    }

    #[inline(always)]
    fn lookup_rank(&self, left: u16, right: u16) -> Option<u32> {
        let row = &self.rows[left as usize];
        match row.len {
            0 => None,
            1 => {
                if row.rights[0] == right {
                    Some(row.ranks[0])
                } else {
                    None
                }
            }
            2 => {
                if row.rights[0] == right {
                    Some(row.ranks[0])
                } else if row.rights[1] == right {
                    Some(row.ranks[1])
                } else {
                    None
                }
            }
            3 => lookup_small_row(&row.rights, &row.ranks, right, 3),
            4 => lookup_small_row(&row.rights, &row.ranks, right, 4),
            5 => lookup_small_row(&row.rights, &row.ranks, right, 5),
            6 => lookup_small_row(&row.rights, &row.ranks, right, 6),
            7 => lookup_small_row(&row.rights, &row.ranks, right, 7),
            8 => lookup_small_row(&row.rights, &row.ranks, right, 8),
            9 => lookup_small_row(&row.rights, &row.ranks, right, 9),
            10 => lookup_small_row(&row.rights, &row.ranks, right, 10),
            11 => lookup_small_row(&row.rights, &row.ranks, right, 11),
            12 => lookup_small_row(&row.rights, &row.ranks, right, 12),
            13 => lookup_small_row(&row.rights, &row.ranks, right, 13),
            14 => lookup_small_row(&row.rights, &row.ranks, right, 14),
            15 => lookup_small_row(&row.rights, &row.ranks, right, 15),
            HEAVY_ROW_SENTINEL => self.fallback.get(&(left, right)).copied(),
            _ => unreachable!("row length sentinel must be valid"),
        }
    }
}

impl SpecializedLookup15HeavyHash {
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
        let mut grouped_rows = vec![Vec::<RowEntry>::new(); piece_count];

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
            grouped_rows[left_id as usize].push(RowEntry {
                right: u16::try_from(right_id)
                    .map_err(|_| "piece ids no longer fit in u16".to_string())?,
                rank: rank as u32,
            });
        }

        let mut rows = vec![SmallRow15::empty(); piece_count];
        let mut heavy_descs = vec![HeavyRowDesc::default(); piece_count];
        let mut heavy_rights = Vec::new();
        let mut heavy_ranks = Vec::new();

        for (left_id, row) in grouped_rows.into_iter().enumerate() {
            if row.len() <= SMALL_ROW_LIMIT_15 {
                let mut small_row = SmallRow15::empty();
                small_row.len = row.len() as u8;
                for (slot, entry) in row.into_iter().enumerate() {
                    small_row.rights[slot] = entry.right;
                    small_row.ranks[slot] = entry.rank;
                }
                rows[left_id] = small_row;
            } else {
                rows[left_id].len = HEAVY_ROW_SENTINEL;
                let (seed, table_rights, table_ranks) =
                    build_heavy_hash_table(left_id as u32, &row)?;
                let start = heavy_rights.len() as u32;
                let mask = u16::try_from(table_rights.len() - 1)
                    .map_err(|_| "heavy row table mask overflow".to_string())?;
                heavy_descs[left_id] = HeavyRowDesc {
                    start,
                    mask,
                    seed,
                };
                heavy_rights.extend(table_rights);
                heavy_ranks.extend(table_ranks);
            }
        }

        Ok(Self {
            rows,
            heavy_descs,
            heavy_rights,
            heavy_ranks,
        })
    }

    #[inline(always)]
    fn lookup_rank(&self, left: u16, right: u16) -> Option<u32> {
        let row = &self.rows[left as usize];
        match row.len {
            0 => None,
            1 => {
                if row.rights[0] == right {
                    Some(row.ranks[0])
                } else {
                    None
                }
            }
            2 => {
                if row.rights[0] == right {
                    Some(row.ranks[0])
                } else if row.rights[1] == right {
                    Some(row.ranks[1])
                } else {
                    None
                }
            }
            3 => lookup_small_row(&row.rights, &row.ranks, right, 3),
            4 => lookup_small_row(&row.rights, &row.ranks, right, 4),
            5 => lookup_small_row(&row.rights, &row.ranks, right, 5),
            6 => lookup_small_row(&row.rights, &row.ranks, right, 6),
            7 => lookup_small_row(&row.rights, &row.ranks, right, 7),
            8 => lookup_small_row(&row.rights, &row.ranks, right, 8),
            9 => lookup_small_row(&row.rights, &row.ranks, right, 9),
            10 => lookup_small_row(&row.rights, &row.ranks, right, 10),
            11 => lookup_small_row(&row.rights, &row.ranks, right, 11),
            12 => lookup_small_row(&row.rights, &row.ranks, right, 12),
            13 => lookup_small_row(&row.rights, &row.ranks, right, 13),
            14 => lookup_small_row(&row.rights, &row.ranks, right, 14),
            15 => lookup_small_row(&row.rights, &row.ranks, right, 15),
            HEAVY_ROW_SENTINEL => {
                let desc = self.heavy_descs[left as usize];
                let mut slot = heavy_hash_slot(desc.seed, right, desc.mask);
                loop {
                    let idx = desc.start as usize + slot as usize;
                    let stored = self.heavy_rights[idx];
                    if stored == right {
                        return Some(self.heavy_ranks[idx]);
                    }
                    if stored == EMPTY_HEAVY_SLOT {
                        return None;
                    }
                    slot = (slot + 1) & desc.mask;
                }
            }
            _ => unreachable!("row length sentinel must be valid"),
        }
    }
}

fn build_heavy_hash_table(left_id: u32, row: &[RowEntry]) -> Result<(u32, Vec<u16>, Vec<u32>), String> {
    let table_len = row
        .len()
        .checked_mul(4)
        .ok_or_else(|| "heavy row table len overflow".to_string())?
        .next_power_of_two();
    let mut table_rights = vec![EMPTY_HEAVY_SLOT; table_len];
    let mut table_ranks = vec![0u32; table_len];
    let mask =
        u16::try_from(table_len - 1).map_err(|_| "heavy row table mask overflow".to_string())?;
    let seed = left_id.wrapping_mul(0x9e37_79b9).rotate_left(7) ^ row.len() as u32;

    for entry in row {
        let mut slot = heavy_hash_slot(seed, entry.right, mask);
        loop {
            let idx = slot as usize;
            if table_rights[idx] == EMPTY_HEAVY_SLOT {
                table_rights[idx] = entry.right;
                table_ranks[idx] = entry.rank;
                break;
            }
            slot = (slot + 1) & mask;
        }
    }

    Ok((seed, table_rights, table_ranks))
}

#[inline(always)]
fn mix_heavy_seed(seed: u32, x: u16) -> u32 {
    let mut z = (x as u32) ^ seed;
    z ^= z >> 16;
    z = z.wrapping_mul(0x7feb_352d);
    z ^= z >> 15;
    z = z.wrapping_mul(0x846c_a68b);
    z ^= z >> 16;
    z
}

#[inline(always)]
fn heavy_hash_slot(seed: u32, right: u16, mask: u16) -> u16 {
    (mix_heavy_seed(seed, right) as u16) & mask
}

#[inline(always)]
fn lookup_small_row(
    rights: &[u16; SMALL_ROW_LIMIT_15],
    ranks: &[u32; SMALL_ROW_LIMIT_15],
    target: u16,
    len: usize,
) -> Option<u32> {
    let mut idx = 0usize;
    while idx < len {
        if rights[idx] == target {
            return Some(ranks[idx]);
        }
        idx += 1;
    }
    None
}

fn piece_count(merges_graph: &BpeMerges) -> usize {
    let mut count = 0usize;
    while merges_graph.decode_piece(count as u32).is_some() {
        count += 1;
    }
    count
}

fn next_rank(entry: SpineEntry) -> Option<u32> {
    if entry.rank_plus_one == 0 {
        None
    } else {
        Some((entry.rank_plus_one - 1) as u32)
    }
}

fn canonical_pair_from_spines_hybrid(
    lookup: &HybridLookup,
    right_spine: &[SpineEntry],
    left_spine: &[SpineEntry],
) -> bool {
    if right_spine.is_empty() || left_spine.is_empty() {
        return false;
    }

    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        let right_rank = next_rank(right_spine[i]);
        let left_rank = next_rank(left_spine[j]);
        let cross_rank = lookup.lookup_rank(right_spine[i].id, left_spine[j].id);

        let mut best = right_rank;
        if left_rank.is_some() && (best.is_none() || left_rank < best) {
            best = left_rank;
        }
        if cross_rank.is_some() && (best.is_none() || cross_rank < best) {
            best = cross_rank;
        }

        let Some(best_rank) = best else {
            return true;
        };

        if cross_rank == Some(best_rank) {
            return false;
        }
        if right_rank == Some(best_rank) {
            i += 1;
            continue;
        }
        if left_rank == Some(best_rank) {
            j += 1;
            continue;
        }

        unreachable!("best rank must come from one of the three candidate events");
    }
}

fn canonical_pair_from_spines_hybrid_collect_stats(
    lookup: &HybridLookup,
    right_spine: &[SpineEntry],
    left_spine: &[SpineEntry],
    stats: &mut LookupStats32,
) -> bool {
    if right_spine.is_empty() || left_spine.is_empty() {
        return false;
    }

    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        let left = right_spine[i].id;
        let right = left_spine[j].id;
        stats.total_lookups += 1;

        let right_rank = next_rank(right_spine[i]);
        let left_rank = next_rank(left_spine[j]);
        let cross_rank = if lookup.fallback_rows[left as usize] {
            stats.fallback_lookups += 1;
            let key = (left as u32, right as u32);
            stats.fallback_queries.push(key);
            let result = lookup.fallback.get(&key).copied();
            if result.is_some() {
                stats.fallback_hits += 1;
            }
            result
        } else {
            lookup.lookup_rank(left, right)
        };

        let mut best = right_rank;
        if left_rank.is_some() && (best.is_none() || left_rank < best) {
            best = left_rank;
        }
        if cross_rank.is_some() && (best.is_none() || cross_rank < best) {
            best = cross_rank;
        }

        let Some(best_rank) = best else {
            return true;
        };

        if cross_rank == Some(best_rank) {
            return false;
        }
        if right_rank == Some(best_rank) {
            i += 1;
            continue;
        }
        if left_rank == Some(best_rank) {
            j += 1;
            continue;
        }

        unreachable!("best rank must come from one of the three candidate events");
    }
}

fn canonical_pair_from_spines_specialized(
    lookup: &SpecializedLookup15,
    right_spine: &[SpineEntry],
    left_spine: &[SpineEntry],
) -> bool {
    if right_spine.is_empty() || left_spine.is_empty() {
        return false;
    }

    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        let right_rank = next_rank(right_spine[i]);
        let left_rank = next_rank(left_spine[j]);
        let cross_rank = lookup.lookup_rank(right_spine[i].id, left_spine[j].id);

        let mut best = right_rank;
        if left_rank.is_some() && (best.is_none() || left_rank < best) {
            best = left_rank;
        }
        if cross_rank.is_some() && (best.is_none() || cross_rank < best) {
            best = cross_rank;
        }

        let Some(best_rank) = best else {
            return true;
        };

        if cross_rank == Some(best_rank) {
            return false;
        }
        if right_rank == Some(best_rank) {
            i += 1;
            continue;
        }
        if left_rank == Some(best_rank) {
            j += 1;
            continue;
        }

        unreachable!("best rank must come from one of the three candidate events");
    }
}

fn canonical_pair_from_spines_specialized_heavy_hash(
    lookup: &SpecializedLookup15HeavyHash,
    right_spine: &[SpineEntry],
    left_spine: &[SpineEntry],
) -> bool {
    if right_spine.is_empty() || left_spine.is_empty() {
        return false;
    }

    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        let right_rank = next_rank(right_spine[i]);
        let left_rank = next_rank(left_spine[j]);
        let cross_rank = lookup.lookup_rank(right_spine[i].id, left_spine[j].id);

        let mut best = right_rank;
        if left_rank.is_some() && (best.is_none() || left_rank < best) {
            best = left_rank;
        }
        if cross_rank.is_some() && (best.is_none() || cross_rank < best) {
            best = cross_rank;
        }

        let Some(best_rank) = best else {
            return true;
        };

        if cross_rank == Some(best_rank) {
            return false;
        }
        if right_rank == Some(best_rank) {
            i += 1;
            continue;
        }
        if left_rank == Some(best_rank) {
            j += 1;
            continue;
        }

        unreachable!("best rank must come from one of the three candidate events");
    }
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

#[inline(always)]
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

fn canonical_pair_from_spines_specialized_collect_stats(
    lookup: &SpecializedLookup15,
    right_spine: &[SpineEntry],
    left_spine: &[SpineEntry],
    stats: &mut LookupStats16,
) -> bool {
    if right_spine.is_empty() || left_spine.is_empty() {
        return false;
    }

    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        let left = right_spine[i].id;
        let right = left_spine[j].id;
        stats.total_lookups += 1;

        let right_rank = next_rank(right_spine[i]);
        let left_rank = next_rank(left_spine[j]);
        let cross_rank = if lookup.rows[left as usize].len == HEAVY_ROW_SENTINEL {
            stats.fallback_lookups += 1;
            let key = (left, right);
            stats.fallback_queries.push(key);
            let result = lookup.fallback.get(&key).copied();
            if result.is_some() {
                stats.fallback_hits += 1;
            }
            result
        } else {
            lookup.lookup_rank(left, right)
        };

        let mut best = right_rank;
        if left_rank.is_some() && (best.is_none() || left_rank < best) {
            best = left_rank;
        }
        if cross_rank.is_some() && (best.is_none() || cross_rank < best) {
            best = cross_rank;
        }

        let Some(best_rank) = best else {
            return true;
        };

        if cross_rank == Some(best_rank) {
            return false;
        }
        if right_rank == Some(best_rank) {
            i += 1;
            continue;
        }
        if left_rank == Some(best_rank) {
            j += 1;
            continue;
        }

        unreachable!("best rank must come from one of the three candidate events");
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

fn time_hybrid(
    tokenizer: &TinyLlamaWordTokenizer,
    lookup: &HybridLookup,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
) -> (u64, u64, u64, f64) {
    let timed_start = Instant::now();
    let mut pair_count = 0u64;
    let mut used_first_ids = 0u64;
    let mut canonical_count = 0u64;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        used_first_ids += 1;
        for second_left_spine in candidate_second_spines {
            if black_box(canonical_pair_from_spines_hybrid(
                lookup,
                first_right_spine.as_slice(),
                second_left_spine.as_slice(),
            )) {
                canonical_count += 1;
            }
            pair_count += 1;
        }
    }

    (
        pair_count,
        used_first_ids,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_mismatches(
    tokenizer: &TinyLlamaWordTokenizer,
    lookup: &HybridLookup,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
) -> u64 {
    let mut mismatches = 0u64;
    let mut second_index = 0usize;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        for second_left_spine in candidate_second_spines {
            let baseline =
                tokenizer.canonical_pair_from_packed_spines(&first_right_spine, second_left_spine);
            let hybrid = canonical_pair_from_spines_hybrid(
                lookup,
                first_right_spine.as_slice(),
                second_left_spine.as_slice(),
            );
            if baseline != hybrid {
                mismatches += 1;
            }
            second_index += 1;
        }
    }

    let _ = second_index;
    mismatches
}

fn time_specialized(
    tokenizer: &TinyLlamaWordTokenizer,
    lookup: &SpecializedLookup15,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
) -> (u64, u64, u64, f64) {
    let timed_start = Instant::now();
    let mut pair_count = 0u64;
    let mut used_first_ids = 0u64;
    let mut canonical_count = 0u64;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        used_first_ids += 1;
        for second_left_spine in candidate_second_spines {
            if black_box(canonical_pair_from_spines_specialized(
                lookup,
                first_right_spine.as_slice(),
                second_left_spine.as_slice(),
            )) {
                canonical_count += 1;
            }
            pair_count += 1;
        }
    }

    (
        pair_count,
        used_first_ids,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn time_specialized_heavy_hash(
    tokenizer: &TinyLlamaWordTokenizer,
    lookup: &SpecializedLookup15HeavyHash,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
) -> (u64, u64, u64, f64) {
    let timed_start = Instant::now();
    let mut pair_count = 0u64;
    let mut used_first_ids = 0u64;
    let mut canonical_count = 0u64;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        used_first_ids += 1;
        for second_left_spine in candidate_second_spines {
            if black_box(canonical_pair_from_spines_specialized_heavy_hash(
                lookup,
                first_right_spine.as_slice(),
                second_left_spine.as_slice(),
            )) {
                canonical_count += 1;
            }
            pair_count += 1;
        }
    }

    (
        pair_count,
        used_first_ids,
        canonical_count,
        timed_start.elapsed().as_secs_f64(),
    )
}

fn count_mismatches_specialized(
    tokenizer: &TinyLlamaWordTokenizer,
    lookup: &SpecializedLookup15,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
) -> u64 {
    let mut mismatches = 0u64;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        for second_left_spine in candidate_second_spines {
            let baseline =
                tokenizer.canonical_pair_from_packed_spines(&first_right_spine, second_left_spine);
            let specialized = canonical_pair_from_spines_specialized(
                lookup,
                first_right_spine.as_slice(),
                second_left_spine.as_slice(),
            );
            if baseline != specialized {
                mismatches += 1;
            }
        }
    }

    mismatches
}

fn count_mismatches_specialized_heavy_hash(
    tokenizer: &TinyLlamaWordTokenizer,
    lookup: &SpecializedLookup15HeavyHash,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
) -> u64 {
    let mut mismatches = 0u64;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        for second_left_spine in candidate_second_spines {
            let baseline =
                tokenizer.canonical_pair_from_packed_spines(&first_right_spine, second_left_spine);
            let specialized = canonical_pair_from_spines_specialized_heavy_hash(
                lookup,
                first_right_spine.as_slice(),
                second_left_spine.as_slice(),
            );
            if baseline != specialized {
                mismatches += 1;
            }
        }
    }

    mismatches
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

fn analyze_prepared_dense_row_popcounts(
    tokenizer: &TinyLlamaWordTokenizer,
    merge_rows: &MergeRows,
    sampled_first_ids: &[u32],
) {
    let mut hist = BTreeMap::<usize, u64>::new();
    let mut total_rows = 0u64;
    let mut total_popcount = 0u64;
    let mut min_popcount = usize::MAX;
    let mut max_popcount = 0usize;

    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        for entry in first_right_spine.as_slice() {
            let popcount = merge_rows.rows[entry.id as usize].len();
            *hist.entry(popcount).or_insert(0) += 1;
            total_rows += 1;
            total_popcount += popcount as u64;
            min_popcount = min_popcount.min(popcount);
            max_popcount = max_popcount.max(popcount);
        }
    }

    println!("prepared_first_dense_row_popcount_hist:");
    println!("  sampled_first_ids = {}", sampled_first_ids.len());
    println!("  total_active_dense_rows = {total_rows}");
    if total_rows == 0 {
        return;
    }
    println!(
        "  avg_popcount = {:.3}",
        total_popcount as f64 / total_rows as f64
    );
    println!("  min_popcount = {min_popcount}");
    println!("  max_popcount = {max_popcount}");
    println!("  histogram:");
    for (popcount, count) in hist {
        println!(
            "    popcount={popcount:>4} rows={count:>8} fraction={:.6}",
            count as f64 / total_rows as f64
        );
    }
}

fn replay_fallback_32(
    fallback: &HashMap<(u32, u32), u32>,
    queries: &[(u32, u32)],
) -> (u64, f64) {
    let start = Instant::now();
    let mut hits = 0u64;
    for &key in queries {
        if black_box(fallback.get(&key).copied()).is_some() {
            hits += 1;
        }
    }
    (hits, start.elapsed().as_secs_f64())
}

fn replay_fallback_16(
    fallback: &HashMap<(u16, u16), u32>,
    queries: &[(u16, u16)],
) -> (u64, f64) {
    let start = Instant::now();
    let mut hits = 0u64;
    for &key in queries {
        if black_box(fallback.get(&key).copied()).is_some() {
            hits += 1;
        }
    }
    (hits, start.elapsed().as_secs_f64())
}

fn analyze_hybrid_fallback(
    tokenizer: &TinyLlamaWordTokenizer,
    lookup: &HybridLookup,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
    timed_elapsed_seconds: f64,
) {
    let mut stats = LookupStats32::default();
    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        for second_left_spine in candidate_second_spines {
            black_box(canonical_pair_from_spines_hybrid_collect_stats(
                lookup,
                first_right_spine.as_slice(),
                second_left_spine.as_slice(),
                &mut stats,
            ));
        }
    }

    let (replay_hits, replay_seconds) = replay_fallback_32(&lookup.fallback, &stats.fallback_queries);
    println!("  fallback_analysis:");
    println!("    total_lookups = {}", stats.total_lookups);
    println!("    fallback_lookups = {}", stats.fallback_lookups);
    println!(
        "    fallback_lookup_fraction = {:.6}",
        stats.fallback_lookups as f64 / stats.total_lookups as f64
    );
    println!("    fallback_hits = {}", stats.fallback_hits);
    println!(
        "    fallback_hit_fraction_within_fallback = {:.6}",
        stats.fallback_hits as f64 / stats.fallback_lookups as f64
    );
    println!("    fallback_replay_hits = {replay_hits}");
    println!("    fallback_replay_seconds = {replay_seconds:.6}");
    println!(
        "    fallback_replay_micros_per_lookup = {:.3}",
        replay_seconds * 1_000_000.0 / stats.fallback_lookups as f64
    );
    println!(
        "    fallback_replay_fraction_of_total_timed = {:.6}",
        replay_seconds / timed_elapsed_seconds
    );
}

fn analyze_specialized_fallback(
    tokenizer: &TinyLlamaWordTokenizer,
    lookup: &SpecializedLookup15,
    sampled_first_ids: &[u32],
    candidate_second_spines: &[PackedSpine],
    timed_elapsed_seconds: f64,
) {
    let mut stats = LookupStats16::default();
    for &first_id in sampled_first_ids {
        let Some(first_right_spine) = tokenizer.right_packed_spine_for_token_id(first_id) else {
            continue;
        };
        for second_left_spine in candidate_second_spines {
            black_box(canonical_pair_from_spines_specialized_collect_stats(
                lookup,
                first_right_spine.as_slice(),
                second_left_spine.as_slice(),
                &mut stats,
            ));
        }
    }

    let (replay_hits, replay_seconds) = replay_fallback_16(&lookup.fallback, &stats.fallback_queries);
    println!("  fallback_analysis:");
    println!("    total_lookups = {}", stats.total_lookups);
    println!("    fallback_lookups = {}", stats.fallback_lookups);
    println!(
        "    fallback_lookup_fraction = {:.6}",
        stats.fallback_lookups as f64 / stats.total_lookups as f64
    );
    println!("    fallback_hits = {}", stats.fallback_hits);
    println!(
        "    fallback_hit_fraction_within_fallback = {:.6}",
        stats.fallback_hits as f64 / stats.fallback_lookups as f64
    );
    println!("    fallback_replay_hits = {replay_hits}");
    println!("    fallback_replay_seconds = {replay_seconds:.6}");
    println!(
        "    fallback_replay_micros_per_lookup = {:.3}",
        replay_seconds * 1_000_000.0 / stats.fallback_lookups as f64
    );
    println!(
        "    fallback_replay_fraction_of_total_timed = {:.6}",
        replay_seconds / timed_elapsed_seconds
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
    let candidate_second_spines: Vec<PackedSpine> = candidate_second_ids
        .iter()
        .map(|&token_id| {
            tokenizer
                .left_packed_spine_for_token_id(token_id)
                .expect("sorted candidate ids must resolve to packed left spines")
                .to_owned()
        })
        .collect();
    let candidate_second_compact_spines: Vec<CompactLeftSpine> = candidate_second_spines
        .iter()
        .copied()
        .map(CompactLeftSpine::from_packed)
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

    for threshold in [8usize, 16, 32] {
        let mode_matches = mode == "all"
            || (threshold == 8 && mode == "hybrid8")
            || (threshold == 16 && mode == "hybrid16")
            || (threshold == 32 && mode == "hybrid32");
        if !mode_matches {
            continue;
        }
        let build_start = Instant::now();
        let lookup = match HybridLookup::build(&tokenizer_path, &merges_graph, threshold) {
            Ok(lookup) => lookup,
            Err(err) => {
                eprintln!("failed to build hybrid lookup: {err}");
                return ExitCode::from(1);
            }
        };
        let build_seconds = build_start.elapsed().as_secs_f64();
        let fallback_rows = lookup.fallback_rows.iter().filter(|&&is_fallback| is_fallback).count();
        let fallback_entries = lookup.fallback.len();
        let small_entries = lookup.small_rows.len();

        let (pair_count, used_first_ids, canonical_count, elapsed_seconds) =
            time_hybrid(&tokenizer, &lookup, &sampled_first_ids, &candidate_second_spines);
        let mismatches = if run_mismatch_checks {
            count_mismatches(&tokenizer, &lookup, &sampled_first_ids, &candidate_second_spines)
        } else {
            0
        };

        println!("hybrid_threshold_{}:", lookup.threshold);
        println!("  build_seconds = {build_seconds:.6}");
        println!("  small_entries = {small_entries}");
        println!("  fallback_rows = {fallback_rows}");
        println!("  fallback_entries = {fallback_entries}");
        println!("  canonical_count = {canonical_count}");
        println!("  mismatches_vs_baseline = {mismatches}");
        print_timing("  timing", pair_count, used_first_ids, elapsed_seconds);
        if threshold == 16 {
            analyze_hybrid_fallback(
                &tokenizer,
                &lookup,
                &sampled_first_ids,
                &candidate_second_spines,
                elapsed_seconds,
            );
        }
    }

    if mode == "all" || mode == "specialized15" {
        let build_start = Instant::now();
        let specialized_lookup = match SpecializedLookup15::build(&tokenizer_path, &merges_graph) {
            Ok(lookup) => lookup,
            Err(err) => {
                eprintln!("failed to build specialized lookup: {err}");
                return ExitCode::from(1);
            }
        };
        let build_seconds = build_start.elapsed().as_secs_f64();
        let fallback_rows = specialized_lookup
            .rows
            .iter()
            .filter(|row| row.len == HEAVY_ROW_SENTINEL)
            .count();
        let fallback_entries = specialized_lookup.fallback.len();
        let small_rows = specialized_lookup
            .rows
            .iter()
            .filter(|row| row.len != HEAVY_ROW_SENTINEL)
            .count();

        let (pair_count, used_first_ids, canonical_count, elapsed_seconds) = time_specialized(
            &tokenizer,
            &specialized_lookup,
            &sampled_first_ids,
            &candidate_second_spines,
        );
        let mismatches = if run_mismatch_checks {
            count_mismatches_specialized(
                &tokenizer,
                &specialized_lookup,
                &sampled_first_ids,
                &candidate_second_spines,
            )
        } else {
            0
        };

        println!("specialized_threshold_15:");
        println!("  build_seconds = {build_seconds:.6}");
        println!("  small_rows = {small_rows}");
        println!("  fallback_rows = {fallback_rows}");
        println!("  fallback_entries = {fallback_entries}");
        println!("  canonical_count = {canonical_count}");
        println!("  mismatches_vs_baseline = {mismatches}");
        print_timing("  timing", pair_count, used_first_ids, elapsed_seconds);
        analyze_specialized_fallback(
            &tokenizer,
            &specialized_lookup,
            &sampled_first_ids,
            &candidate_second_spines,
            elapsed_seconds,
        );
    }

    if mode == "all" || mode == "specialized15hash" {
        let build_start = Instant::now();
        let specialized_lookup = match SpecializedLookup15HeavyHash::build(&tokenizer_path, &merges_graph) {
            Ok(lookup) => lookup,
            Err(err) => {
                eprintln!("failed to build specialized open-addressed lookup: {err}");
                return ExitCode::from(1);
            }
        };
        let build_seconds = build_start.elapsed().as_secs_f64();
        let heavy_rows = specialized_lookup
            .rows
            .iter()
            .filter(|row| row.len == HEAVY_ROW_SENTINEL)
            .count();
        let heavy_slots = specialized_lookup.heavy_rights.len();
        let small_rows = specialized_lookup
            .rows
            .iter()
            .filter(|row| row.len != HEAVY_ROW_SENTINEL)
            .count();

        let (pair_count, used_first_ids, canonical_count, elapsed_seconds) =
            time_specialized_heavy_hash(
                &tokenizer,
                &specialized_lookup,
                &sampled_first_ids,
                &candidate_second_spines,
            );
        let mismatches = if run_mismatch_checks {
            count_mismatches_specialized_heavy_hash(
                &tokenizer,
                &specialized_lookup,
                &sampled_first_ids,
                &candidate_second_spines,
            )
        } else {
            0
        };

        println!("specialized_threshold_15_open4x:");
        println!("  build_seconds = {build_seconds:.6}");
        println!("  small_rows = {small_rows}");
        println!("  heavy_rows = {heavy_rows}");
        println!("  heavy_slots = {heavy_slots}");
        println!("  canonical_count = {canonical_count}");
        println!("  mismatches_vs_baseline = {mismatches}");
        print_timing("  timing", pair_count, used_first_ids, elapsed_seconds);
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

    if mode == "preparedfirstdense_rowhist" {
        analyze_prepared_dense_row_popcounts(&tokenizer, &merge_rows, &sampled_first_ids);
    }

    ExitCode::SUCCESS
}
