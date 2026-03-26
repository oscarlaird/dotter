use super::{BpeError, BpeMerges, MAX_PACKED_SPINE_LEN, PackedSpine};

pub const MAX_PREPARED_DENSE_PIECE_COUNT: usize = (u16::MAX as usize) + 1;
static ZERO_CROSS_RANK_ROW: [u16; MAX_PREPARED_DENSE_PIECE_COUNT] =
    [0; MAX_PREPARED_DENSE_PIECE_COUNT];

#[derive(Clone, Copy, Debug)]
struct RowEntry {
    right: u16,
    rank_plus_one: u16,
}

#[derive(Debug, Clone)]
pub struct MergeRows {
    rows: Vec<Vec<RowEntry>>,
    piece_count: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedFirstDense<const PIECE_COUNT: usize> {
    right_spine: PackedSpine,
    right_len: u8,
    right_rank_plus_one: [u16; MAX_PACKED_SPINE_LEN],
    cross_rank_row_ptrs: [*const u16; MAX_PACKED_SPINE_LEN],
    _dense_rows: [Option<Box<[u16; PIECE_COUNT]>>; MAX_PACKED_SPINE_LEN],
}

#[derive(Clone, Copy, Debug)]
pub struct CompactLeftSpine {
    pub len: u8,
    pub ids: [u16; MAX_PACKED_SPINE_LEN],
    pub rank_plus_one: [u16; MAX_PACKED_SPINE_LEN],
}

#[derive(Clone, Copy, Debug)]
pub struct PreparedSecondToken {
    pub token_id: u32,
    pub left_spine: CompactLeftSpine,
}

pub type PreparedSecondBuckets = [Vec<PreparedSecondToken>; MAX_PACKED_SPINE_LEN + 1];

impl MergeRows {
    pub fn from_bpe_merges(merges: &BpeMerges) -> Result<Self, BpeError> {
        let piece_count = merges.pieces.len();
        let mut rows = vec![Vec::<RowEntry>::new(); piece_count];

        for entry in merges.merges.values() {
            let right = u16::try_from(entry.right).map_err(|_| {
                BpeError::UnsupportedPreparedDense("piece ids no longer fit in u16")
            })?;
            let rank_plus_one = entry
                .rank
                .checked_add(1)
                .and_then(|value| u16::try_from(value).ok())
                .ok_or(BpeError::UnsupportedPreparedDense(
                    "merge ranks no longer fit in u16",
                ))?;
            rows[entry.left as usize].push(RowEntry {
                right,
                rank_plus_one,
            });
        }

        Ok(Self { rows, piece_count })
    }

    pub fn piece_count(&self) -> usize {
        self.piece_count
    }
}

impl<const PIECE_COUNT: usize> PreparedFirstDense<PIECE_COUNT> {
    pub fn build(first_right_spine: PackedSpine, merge_rows: &MergeRows) -> Result<Self, BpeError> {
        if PIECE_COUNT > MAX_PREPARED_DENSE_PIECE_COUNT {
            return Err(BpeError::UnsupportedPreparedDense(
                "prepared dense table exceeds the maximum supported piece-id width",
            ));
        }
        if merge_rows.piece_count > PIECE_COUNT {
            return Err(BpeError::UnsupportedPreparedDense(
                "prepared dense fast path expects piece ids to fit within the configured table",
            ));
        }

        let mut right_rank_plus_one = [0u16; MAX_PACKED_SPINE_LEN];
        let zero_row_ptr = ZERO_CROSS_RANK_ROW.as_ptr();
        let mut cross_rank_row_ptrs = [zero_row_ptr; MAX_PACKED_SPINE_LEN];
        let mut dense_rows: [Option<Box<[u16; PIECE_COUNT]>>; MAX_PACKED_SPINE_LEN] =
            std::array::from_fn(|_| None);

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            right_rank_plus_one[spine_idx] = spine_entry.rank_plus_one;
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            let mut dense_row = Box::new([0u16; PIECE_COUNT]);
            for entry in row {
                dense_row[entry.right as usize] = entry.rank_plus_one;
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

    pub fn right_spine(&self) -> &PackedSpine {
        &self.right_spine
    }
}

impl CompactLeftSpine {
    pub fn from_packed(packed: PackedSpine) -> Self {
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

pub fn sort_prepared_second_tokens(entries: &mut [PreparedSecondToken]) {
    entries.sort_by(|a, b| {
        a.left_spine.ids[..a.left_spine.len as usize]
            .cmp(&b.left_spine.ids[..b.left_spine.len as usize])
            .then_with(|| a.token_id.cmp(&b.token_id))
    });
}

pub fn bucket_prepared_second_tokens(entries: &[PreparedSecondToken]) -> PreparedSecondBuckets {
    let mut buckets: PreparedSecondBuckets = std::array::from_fn(|_| Vec::new());
    for &entry in entries {
        buckets[entry.left_spine.len as usize].push(entry);
    }
    buckets
}

#[inline(always)]
pub fn canonical_pair_from_prepared_first_dense_left_len<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDense<PIECE_COUNT>,
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

        let right_rank_plus_one = unsafe { *right_rank_plus_one.get_unchecked(i) };
        let left_id = unsafe { *left_ids.get_unchecked(j) };
        let left_rank_plus_one = unsafe { *left_rank_plus_one.get_unchecked(j) };
        debug_assert!((left_id as usize) < PIECE_COUNT);
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

pub fn canonical_pair_from_prepared_first_dense<const PIECE_COUNT: usize>(
    prepared_first: &PreparedFirstDense<PIECE_COUNT>,
    left_spine: &CompactLeftSpine,
) -> bool {
    match left_spine.len {
        0 => false,
        1 => canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, 1>(
            prepared_first,
            left_spine,
        ),
        2 => canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, 2>(
            prepared_first,
            left_spine,
        ),
        3 => canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, 3>(
            prepared_first,
            left_spine,
        ),
        4 => canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, 4>(
            prepared_first,
            left_spine,
        ),
        5 => canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, 5>(
            prepared_first,
            left_spine,
        ),
        6 => canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, 6>(
            prepared_first,
            left_spine,
        ),
        7 => canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, 7>(
            prepared_first,
            left_spine,
        ),
        8 => canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, 8>(
            prepared_first,
            left_spine,
        ),
        _ => unreachable!("packed spine len must fit MAX_PACKED_SPINE_LEN"),
    }
}

pub fn fill_canonical_pair_mask<const PIECE_COUNT: usize>(
    prepared_first: &PreparedFirstDense<PIECE_COUNT>,
    candidate_second_buckets: &PreparedSecondBuckets,
    out: &mut [bool],
) -> u64 {
    let mut canonical_count = 0u64;

    macro_rules! scan_bucket {
        ($left_len:literal) => {
            for entry in &candidate_second_buckets[$left_len] {
                let is_canonical =
                    canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, $left_len>(
                        prepared_first,
                        &entry.left_spine,
                    );
                out[entry.token_id as usize] = is_canonical;
                canonical_count += is_canonical as u64;
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

    canonical_count
}
