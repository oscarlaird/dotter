use super::{BpeError, BpeMerges, MAX_PACKED_SPINE_LEN, PackedSpine};

pub const MAX_PREPARED_DENSE_PIECE_COUNT: usize = (u16::MAX as usize) + 1;
static ZERO_CROSS_RANK_ROW: [u16; MAX_PREPARED_DENSE_PIECE_COUNT] =
    [0; MAX_PREPARED_DENSE_PIECE_COUNT];

#[derive(Clone, Copy, Debug)]
struct RowEntry {
    right: u16,
    priority_score: u16,
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
    right_priority_score: [u16; MAX_PACKED_SPINE_LEN],
    cross_rank_row_ptrs: [*const u16; MAX_PACKED_SPINE_LEN],
    _dense_rows: [Option<Box<[u16; PIECE_COUNT]>>; MAX_PACKED_SPINE_LEN],
}

#[derive(Debug, Clone)]
pub struct PreparedFirstDenseContiguous<const PIECE_COUNT: usize> {
    right_len: u8,
    right_priority_score: [u16; MAX_PACKED_SPINE_LEN],
    // Row-major matrix over right-spine index (rows) x left-piece-id (columns).
    dense_matrix: Box<[u16]>,
}

#[derive(Debug, Clone)]
pub struct PreparedFirstDenseContiguousSwapped<const PIECE_COUNT: usize> {
    right_len: u8,
    right_priority_score: [u16; MAX_PACKED_SPINE_LEN],
    // Left-id-major matrix: columns are right-spine indices.
    // Index as dense_matrix[left_id * MAX_PACKED_SPINE_LEN + right_idx].
    dense_matrix: Box<[u16]>,
}

#[derive(Debug, Clone)]
pub struct PreparedFirstDenseContiguousSwappedTight<const PIECE_COUNT: usize> {
    right_len: u8,
    right_priority_score: [u16; MAX_PACKED_SPINE_LEN],
    // Left-id-major matrix with dynamic row stride = right_len for this prepared first token.
    // Index as dense_matrix[left_id * right_len + right_idx].
    dense_matrix: Box<[u16]>,
    dense_matrix_u32: Box<[u32]>,
}

#[derive(Debug, Clone)]
pub struct PreparedFirstDenseContiguousSwappedTightAllPairs<const PIECE_COUNT: usize> {
    right_len: u8,
    // Allpairs-only convention: score that formed right piece i (i=0 => +inf sentinel).
    // Includes one padded tail slot at index right_len with score 0.
    right_piece_formed_priority_score: [u16; MAX_PACKED_SPINE_LEN + 1],
    // Same layout as swapped-tight dense matrix.
    dense_matrix: Box<[u16]>,
}

#[derive(Clone, Copy, Debug)]
pub struct CompactLeftSpine {
    pub len: u8,
    pub ids: [u16; MAX_PACKED_SPINE_LEN],
    pub priority_score: [u16; MAX_PACKED_SPINE_LEN],
}

#[derive(Clone, Copy, Debug)]
pub struct PreparedSecondToken {
    pub token_id: u32,
    pub left_spine: CompactLeftSpine,
}

pub type PreparedSecondBuckets = [Vec<PreparedSecondToken>; MAX_PACKED_SPINE_LEN + 1];

#[derive(Clone, Copy, Debug)]
pub struct CompactLeftSpineAllPairs {
    pub len: u8,
    pub ids: [u16; MAX_PACKED_SPINE_LEN],
    // Priority score of merge that formed left piece j.
    // For j=0 this is sentinel +inf (u16::MAX).
    // Includes one padded tail slot at index len with score 0.
    pub piece_formed_priority_score: [u16; MAX_PACKED_SPINE_LEN + 1],
}

#[derive(Clone, Copy, Debug)]
pub struct PreparedSecondTokenAllPairs {
    pub token_id: u32,
    pub left_spine: CompactLeftSpineAllPairs,
}

pub type PreparedSecondBucketsAllPairs = [Vec<PreparedSecondTokenAllPairs>; MAX_PACKED_SPINE_LEN + 1];

pub const PREFETCH_CHUNK_WIDTH: usize = 4;
pub const MAX_PREFETCH_LEFT_IDS_PER_CHUNK: usize = PREFETCH_CHUNK_WIDTH * MAX_PACKED_SPINE_LEN;

#[derive(Clone, Copy, Debug)]
pub struct PrefetchLeftIdChunk {
    pub count: u8,
    pub counts_by_scope: [u8; MAX_PACKED_SPINE_LEN + 1],
    pub ids: [u16; MAX_PREFETCH_LEFT_IDS_PER_CHUNK],
}

#[derive(Clone, Debug)]
pub struct PreparedSecondSimd8Chunk {
    // Per-depth vectors for 8 lanes.
    pub left_priority_scores_by_depth: [[u32; 8]; MAX_PACKED_SPINE_LEN],
    // Precomputed row bases per possible first right_len (1..=8), per depth, per lane.
    pub row_base_by_right_len_by_depth:
        [[[u32; 8]; MAX_PACKED_SPINE_LEN]; MAX_PACKED_SPINE_LEN + 1],
}

#[derive(Clone, Debug)]
pub struct PreparedSecondSimd4Chunk {
    pub left_priority_scores_by_depth: [[u32; 4]; MAX_PACKED_SPINE_LEN],
    pub row_base_by_right_len_by_depth:
        [[[u32; 4]; MAX_PACKED_SPINE_LEN]; MAX_PACKED_SPINE_LEN + 1],
}

#[derive(Clone, Debug)]
pub struct PreparedSecondSimd16Chunk {
    pub left_priority_scores_by_depth: [[u32; 16]; MAX_PACKED_SPINE_LEN],
    pub row_base_by_right_len_by_depth:
        [[[u32; 16]; MAX_PACKED_SPINE_LEN]; MAX_PACKED_SPINE_LEN + 1],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefetchHint {
    T0,
    T1,
    T2,
    Nta,
}

#[derive(Clone, Copy, Debug)]
pub struct PrefetchConfig {
    pub enabled: bool,
    pub lookahead_chunks: usize,
    pub budget: u8, // 0 means unlimited.
    pub scope: u8,  // Number of left-spine depths to prefetch (clamped to LEFT_LEN).
    pub hint: PrefetchHint,
}

impl MergeRows {
    pub fn from_bpe_merges(merges: &BpeMerges) -> Result<Self, BpeError> {
        let piece_count = merges.pieces.len();
        let mut rows = vec![Vec::<RowEntry>::new(); piece_count];

        for entry in merges.merges.values() {
            let right = u16::try_from(entry.right).map_err(|_| {
                BpeError::UnsupportedPreparedDense("piece ids no longer fit in u16")
            })?;
            let priority_score = if entry.rank < u16::MAX as u32 {
                (u16::MAX as u32 - entry.rank) as u16
            } else {
                return Err(BpeError::UnsupportedPreparedDense(
                    "merge ranks no longer fit in u16",
                ));
            };
            rows[entry.left as usize].push(RowEntry {
                right,
                priority_score,
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

        let mut right_priority_score = [0u16; MAX_PACKED_SPINE_LEN];
        let zero_row_ptr = ZERO_CROSS_RANK_ROW.as_ptr();
        let mut cross_rank_row_ptrs = [zero_row_ptr; MAX_PACKED_SPINE_LEN];
        let mut dense_rows: [Option<Box<[u16; PIECE_COUNT]>>; MAX_PACKED_SPINE_LEN] =
            std::array::from_fn(|_| None);

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            right_priority_score[spine_idx] = spine_entry.priority_score;
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            let mut dense_row = Box::new([0u16; PIECE_COUNT]);
            for entry in row {
                dense_row[entry.right as usize] = entry.priority_score;
            }
            cross_rank_row_ptrs[spine_idx] = dense_row.as_ptr();
            dense_rows[spine_idx] = Some(dense_row);
        }

        Ok(Self {
            right_spine: first_right_spine,
            right_len: first_right_spine.as_slice().len() as u8,
            right_priority_score,
            cross_rank_row_ptrs,
            _dense_rows: dense_rows,
        })
    }

    pub fn right_spine(&self) -> &PackedSpine {
        &self.right_spine
    }
}

impl<const PIECE_COUNT: usize> PreparedFirstDenseContiguous<PIECE_COUNT> {
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

        let mut right_priority_score = [0u16; MAX_PACKED_SPINE_LEN];
        let mut dense_matrix = vec![0u16; PIECE_COUNT * MAX_PACKED_SPINE_LEN].into_boxed_slice();

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            right_priority_score[spine_idx] = spine_entry.priority_score;
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            let row_base = spine_idx * PIECE_COUNT;
            for entry in row {
                dense_matrix[row_base + entry.right as usize] = entry.priority_score;
            }
        }

        Ok(Self {
            right_len: first_right_spine.as_slice().len() as u8,
            right_priority_score,
            dense_matrix,
        })
    }
}

impl<const PIECE_COUNT: usize> PreparedFirstDenseContiguousSwapped<PIECE_COUNT> {
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

        let mut right_priority_score = [0u16; MAX_PACKED_SPINE_LEN];
        let mut dense_matrix = vec![0u16; PIECE_COUNT * MAX_PACKED_SPINE_LEN].into_boxed_slice();

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            right_priority_score[spine_idx] = spine_entry.priority_score;
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            for entry in row {
                dense_matrix[entry.right as usize * MAX_PACKED_SPINE_LEN + spine_idx] =
                    entry.priority_score;
            }
        }

        Ok(Self {
            right_len: first_right_spine.as_slice().len() as u8,
            right_priority_score,
            dense_matrix,
        })
    }
}

impl<const PIECE_COUNT: usize> PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT> {
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

        let right_len = first_right_spine.as_slice().len();
        let mut right_priority_score = [0u16; MAX_PACKED_SPINE_LEN];
        let mut dense_matrix = vec![0u16; PIECE_COUNT * right_len].into_boxed_slice();
        let mut dense_matrix_u32 = vec![0u32; PIECE_COUNT * right_len].into_boxed_slice();

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            right_priority_score[spine_idx] = spine_entry.priority_score;
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            for entry in row {
                let idx = entry.right as usize * right_len + spine_idx;
                dense_matrix[idx] = entry.priority_score;
                dense_matrix_u32[idx] = entry.priority_score as u32;
            }
        }
        Ok(Self {
            right_len: right_len as u8,
            right_priority_score,
            dense_matrix,
            dense_matrix_u32,
        })
    }
}

impl<const PIECE_COUNT: usize> PreparedFirstDenseContiguousSwappedTightAllPairs<PIECE_COUNT> {
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

        let right_len = first_right_spine.as_slice().len();
        let mut right_piece_formed_priority_score = [0u16; MAX_PACKED_SPINE_LEN + 1];
        let mut dense_matrix = vec![0u16; PIECE_COUNT * right_len].into_boxed_slice();

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            for entry in row {
                let idx = entry.right as usize * right_len + spine_idx;
                dense_matrix[idx] = entry.priority_score;
            }
        }
        if right_len > 0 {
            right_piece_formed_priority_score[0] = u16::MAX;
            for idx in 1..right_len {
                right_piece_formed_priority_score[idx] = first_right_spine.as_slice()[idx - 1].priority_score;
            }
            right_piece_formed_priority_score[right_len] = 0;
        }

        Ok(Self {
            right_len: right_len as u8,
            right_piece_formed_priority_score,
            dense_matrix,
        })
    }
}

impl CompactLeftSpine {
    pub fn from_packed(packed: PackedSpine) -> Self {
        let mut compact = Self {
            len: 0,
            ids: [0; MAX_PACKED_SPINE_LEN],
            priority_score: [0; MAX_PACKED_SPINE_LEN],
        };
        let entries = packed.as_slice();
        compact.len = entries.len() as u8;
        for (idx, entry) in entries.iter().enumerate() {
            compact.ids[idx] = entry.id;
            compact.priority_score[idx] = entry.priority_score;
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

pub fn build_prepared_second_buckets_allpairs(
    buckets: &PreparedSecondBuckets,
) -> PreparedSecondBucketsAllPairs {
    let mut out: PreparedSecondBucketsAllPairs = std::array::from_fn(|_| Vec::new());
    for left_len in 1..=MAX_PACKED_SPINE_LEN {
        let src = &buckets[left_len];
        let mut dst = Vec::with_capacity(src.len());
        for entry in src {
            let mut piece_formed_priority_score = [0u16; MAX_PACKED_SPINE_LEN + 1];
            piece_formed_priority_score[0] = u16::MAX;
            for idx in 1..left_len {
                piece_formed_priority_score[idx] = entry.left_spine.priority_score[idx - 1];
            }
            piece_formed_priority_score[left_len] = 0;
            dst.push(PreparedSecondTokenAllPairs {
                token_id: entry.token_id,
                left_spine: CompactLeftSpineAllPairs {
                    len: entry.left_spine.len,
                    ids: entry.left_spine.ids,
                    piece_formed_priority_score,
                },
            });
        }
        out[left_len] = dst;
    }
    out
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
    let right_priority_score = &prepared_first.right_priority_score;
    let cross_rank_row_ptrs = &prepared_first.cross_rank_row_ptrs;
    let left_ids = &left_spine.ids;
    let left_priority_score = &left_spine.priority_score;
    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        debug_assert!(i < right_spine_len);
        debug_assert!(j < LEFT_LEN);

        let right_priority_score = unsafe { *right_priority_score.get_unchecked(i) };
        let left_id = unsafe { *left_ids.get_unchecked(j) };
        let left_priority_score = unsafe { *left_priority_score.get_unchecked(j) };
        debug_assert!((left_id as usize) < PIECE_COUNT);
        let cross_priority_score =
            unsafe { *(*cross_rank_row_ptrs.get_unchecked(i)).add(left_id as usize) };

        let mut best_priority_score = right_priority_score;
        if left_priority_score > best_priority_score
        {
            best_priority_score = left_priority_score;
        }
        if cross_priority_score > best_priority_score
        {
            best_priority_score = cross_priority_score;
        }

        if best_priority_score == 0 {
            return true;
        }

        if cross_priority_score == best_priority_score {
            return false;
        }
        if right_priority_score == best_priority_score {
            i += 1;
            continue;
        }
        if left_priority_score == best_priority_score {
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

#[inline(always)]
pub fn canonical_pair_from_prepared_first_dense_contiguous_left_len<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguous<PIECE_COUNT>,
    left_spine: &CompactLeftSpine,
) -> bool {
    debug_assert_eq!(left_spine.len as usize, LEFT_LEN);
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return false;
    }

    let right_spine_len = prepared_first.right_len as usize;
    let right_priority_score = &prepared_first.right_priority_score;
    let dense_matrix = &prepared_first.dense_matrix;
    let left_ids = &left_spine.ids;
    let left_priority_score = &left_spine.priority_score;
    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        debug_assert!(i < right_spine_len);
        debug_assert!(j < LEFT_LEN);

        let right_priority_score = unsafe { *right_priority_score.get_unchecked(i) };
        let left_id = unsafe { *left_ids.get_unchecked(j) };
        let left_priority_score = unsafe { *left_priority_score.get_unchecked(j) };
        debug_assert!((left_id as usize) < PIECE_COUNT);
        let cross_priority_score = unsafe {
            *dense_matrix.get_unchecked(i * PIECE_COUNT + left_id as usize)
        };

        let mut best_priority_score = right_priority_score;
        if left_priority_score > best_priority_score
        {
            best_priority_score = left_priority_score;
        }
        if cross_priority_score > best_priority_score
        {
            best_priority_score = cross_priority_score;
        }

        if best_priority_score == 0 {
            return true;
        }

        if cross_priority_score == best_priority_score {
            return false;
        }
        if right_priority_score == best_priority_score {
            i += 1;
            continue;
        }
        if left_priority_score == best_priority_score {
            j += 1;
            continue;
        }

        unreachable!("best rank must come from one of the three candidate events");
    }
}

#[inline(always)]
pub fn canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwapped<PIECE_COUNT>,
    left_spine: &CompactLeftSpine,
) -> bool {
    debug_assert_eq!(left_spine.len as usize, LEFT_LEN);
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return false;
    }

    let right_spine_len = prepared_first.right_len as usize;
    let right_priority_score = &prepared_first.right_priority_score;
    let dense_matrix = &prepared_first.dense_matrix;
    let left_ids = &left_spine.ids;
    let left_priority_score = &left_spine.priority_score;
    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        debug_assert!(i < right_spine_len);
        debug_assert!(j < LEFT_LEN);

        let right_priority_score = unsafe { *right_priority_score.get_unchecked(i) };
        let left_id = unsafe { *left_ids.get_unchecked(j) };
        let left_priority_score = unsafe { *left_priority_score.get_unchecked(j) };
        debug_assert!((left_id as usize) < PIECE_COUNT);
        let cross_priority_score = unsafe {
            *dense_matrix.get_unchecked(left_id as usize * MAX_PACKED_SPINE_LEN + i)
        };

        let mut best_priority_score = right_priority_score;
        if left_priority_score > best_priority_score
        {
            best_priority_score = left_priority_score;
        }
        if cross_priority_score > best_priority_score
        {
            best_priority_score = cross_priority_score;
        }

        if best_priority_score == 0 {
            return true;
        }

        if cross_priority_score == best_priority_score {
            return false;
        }
        if right_priority_score == best_priority_score {
            i += 1;
            continue;
        }
        if left_priority_score == best_priority_score {
            j += 1;
            continue;
        }

        unreachable!("best rank must come from one of the three candidate events");
    }
}

#[inline(always)]
pub fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    left_spine: &CompactLeftSpine,
) -> bool {
    debug_assert_eq!(left_spine.len as usize, LEFT_LEN);
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return false;
    }

    let right_spine_len = prepared_first.right_len as usize;
    let right_priority_score = &prepared_first.right_priority_score;
    let dense_matrix = &prepared_first.dense_matrix;
    let left_ids = &left_spine.ids;
    let left_priority_score = &left_spine.priority_score;
    let mut i = 0usize;
    let mut j = 0usize;

    loop {
        debug_assert!(i < right_spine_len);
        debug_assert!(j < LEFT_LEN);

        let right_priority_score = unsafe { *right_priority_score.get_unchecked(i) };
        let left_id = unsafe { *left_ids.get_unchecked(j) };
        let left_priority_score = unsafe { *left_priority_score.get_unchecked(j) };
        debug_assert!((left_id as usize) < PIECE_COUNT);
        let cross_priority_score =
            unsafe { *dense_matrix.get_unchecked(left_id as usize * right_spine_len + i) };

        let mut best_priority_score = right_priority_score;
        if left_priority_score > best_priority_score
        {
            best_priority_score = left_priority_score;
        }
        if cross_priority_score > best_priority_score
        {
            best_priority_score = cross_priority_score;
        }

        if best_priority_score == 0 {
            return true;
        }

        if cross_priority_score == best_priority_score {
            return false;
        }
        if right_priority_score == best_priority_score {
            i += 1;
            continue;
        }
        if left_priority_score == best_priority_score {
            j += 1;
            continue;
        }

        unreachable!("best rank must come from one of the three candidate events");
    }
}

#[inline(always)]
pub fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_right_len_allpairs_small<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
    const RIGHT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTightAllPairs<PIECE_COUNT>,
    left_spine: &CompactLeftSpineAllPairs,
) -> bool {
    debug_assert_eq!(left_spine.len as usize, LEFT_LEN);
    debug_assert_eq!(prepared_first.right_len as usize, RIGHT_LEN);
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return false;
    }

    let right_piece_formed_priority_score = &prepared_first.right_piece_formed_priority_score;
    let dense_matrix = &prepared_first.dense_matrix;
    let left_ids = &left_spine.ids;
    let left_piece_formed_priority_score = &left_spine.piece_formed_priority_score;

    for j in 0..LEFT_LEN {
        let l_next = unsafe { *left_piece_formed_priority_score.get_unchecked(j + 1) };
        let l_cur = unsafe { *left_piece_formed_priority_score.get_unchecked(j) };
        let left_id = unsafe { *left_ids.get_unchecked(j) } as usize;
        debug_assert!(left_id < PIECE_COUNT);
        let row_base = left_id * RIGHT_LEN;

        for i in 0..RIGHT_LEN {
            let r_cur = unsafe { *right_piece_formed_priority_score.get_unchecked(i) };
            let r_next = unsafe { *right_piece_formed_priority_score.get_unchecked(i + 1) };

            // Reindexed allpairs convention:
            // r_cur/l_cur = score that formed the current piece,
            // r_next/l_next = next competing internal merge score (0 at edge).
            // Boundary (i,j) exists iff alive intervals overlap:
            // max(r_next, l_next) < min(r_cur, l_cur).
            // Direct-comparison form (no max/min): r_next < l_cur && l_next < r_cur.
            let exists_ij = (r_next < l_cur) && (l_next < r_cur);
            if !exists_ij {
                continue;
            }
            let c = unsafe { *dense_matrix.get_unchecked(row_base + i) };

            // Reject when cross is a present merge and at least as eager as both competitors.
            if c != 0 && c >= r_next && c >= l_next {
                return false;
            }
        }
    }
    true
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
    const RIGHT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTightAllPairs<PIECE_COUNT>,
    entries: &[PreparedSecondTokenAllPairs],
) -> u64 {
    debug_assert_eq!(prepared_first.right_len as usize, RIGHT_LEN);
    let mut canonical_count = 0u64;
    for entry in entries {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_right_len_allpairs_small::<
            PIECE_COUNT,
            LEFT_LEN,
            RIGHT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
    const RIGHT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTightAllPairs<PIECE_COUNT>,
    prepared_first_reference: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondTokenAllPairs],
) -> u64 {
    debug_assert_eq!(prepared_first.right_len as usize, RIGHT_LEN);
    debug_assert_eq!(prepared_first_reference.right_len as usize, RIGHT_LEN);
    let mut mismatches = 0u64;
    for entry in entries {
        let allpairs = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_right_len_allpairs_small::<
            PIECE_COUNT,
            LEFT_LEN,
            RIGHT_LEN,
        >(prepared_first, &entry.left_spine);
        let mut next_priority_score = [0u16; MAX_PACKED_SPINE_LEN];
        for idx in 0..LEFT_LEN {
            next_priority_score[idx] = entry.left_spine.piece_formed_priority_score[idx + 1];
        }
        let lockstep = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first_reference, &CompactLeftSpine {
            len: entry.left_spine.len,
            ids: entry.left_spine.ids,
            priority_score: next_priority_score,
        });
        if allpairs != lockstep {
            mismatches += 1;
        }
    }
    mismatches
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

fn canonical_pair_from_prepared_first_dense_left_len_lockstep4_chunk<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDense<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 4],
) -> [bool; 4] {
    debug_assert!(entries
        .iter()
        .all(|entry| entry.left_spine.len as usize == LEFT_LEN));
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return [false; 4];
    }

    let right_spine_len = prepared_first.right_len as usize;
    let stages = right_spine_len + LEFT_LEN - 1;
    let right_priority_score = &prepared_first.right_priority_score;
    let cross_rank_row_ptrs = &prepared_first.cross_rank_row_ptrs;
    let mut i = [0usize; 4];
    let mut j = [0usize; 4];
    let mut rejected = [false; 4];

    for stage_idx in 0..stages {
        let can_step = stage_idx + 1 < stages;
        macro_rules! step_lane {
            ($lane:literal) => {{
                let active = !rejected[$lane];
                debug_assert!(i[$lane] < right_spine_len);
                debug_assert!(j[$lane] < LEFT_LEN);

                let right_priority_score = unsafe { *right_priority_score.get_unchecked(i[$lane]) };
                let left_id = unsafe { *entries[$lane].left_spine.ids.get_unchecked(j[$lane]) };
                let left_priority_score =
                    unsafe { *entries[$lane].left_spine.priority_score.get_unchecked(j[$lane]) };
                debug_assert!((left_id as usize) < PIECE_COUNT);
                let cross_priority_score =
                    unsafe { *(*cross_rank_row_ptrs.get_unchecked(i[$lane])).add(left_id as usize) };

                let reject_now = active
                    && cross_priority_score != 0
                    && cross_priority_score >= right_priority_score
                    && cross_priority_score >= left_priority_score;
                let step_now = can_step && active && !reject_now;
                let take_right = right_priority_score >= left_priority_score;

                rejected[$lane] |= reject_now;
                i[$lane] += (step_now && take_right) as usize;
                j[$lane] += (step_now && !take_right) as usize;
            }};
        }

        step_lane!(0);
        step_lane!(1);
        step_lane!(2);
        step_lane!(3);
    }

    [
        !rejected[0],
        !rejected[1],
        !rejected[2],
        !rejected[3],
    ]
}

fn canonical_pair_from_prepared_first_dense_contiguous_left_len_lockstep4_chunk<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguous<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 4],
) -> [bool; 4] {
    debug_assert!(entries
        .iter()
        .all(|entry| entry.left_spine.len as usize == LEFT_LEN));
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return [false; 4];
    }

    let right_spine_len = prepared_first.right_len as usize;
    let stages = right_spine_len + LEFT_LEN - 1;
    let right_priority_score = &prepared_first.right_priority_score;
    let dense_matrix = &prepared_first.dense_matrix;
    let mut i = [0usize; 4];
    let mut j = [0usize; 4];
    let mut rejected = [false; 4];

    for stage_idx in 0..stages {
        let can_step = stage_idx + 1 < stages;
        macro_rules! step_lane {
            ($lane:literal) => {{
                let active = !rejected[$lane];
                debug_assert!(i[$lane] < right_spine_len);
                debug_assert!(j[$lane] < LEFT_LEN);

                let right_priority_score = unsafe { *right_priority_score.get_unchecked(i[$lane]) };
                let left_id = unsafe { *entries[$lane].left_spine.ids.get_unchecked(j[$lane]) };
                let left_priority_score =
                    unsafe { *entries[$lane].left_spine.priority_score.get_unchecked(j[$lane]) };
                debug_assert!((left_id as usize) < PIECE_COUNT);
                let cross_priority_score =
                    unsafe { *dense_matrix.get_unchecked(i[$lane] * PIECE_COUNT + left_id as usize) };

                let reject_now = active
                    && cross_priority_score != 0
                    && cross_priority_score >= right_priority_score
                    && cross_priority_score >= left_priority_score;
                let step_now = can_step && active && !reject_now;
                let take_right = right_priority_score >= left_priority_score;

                rejected[$lane] |= reject_now;
                i[$lane] += (step_now && take_right) as usize;
                j[$lane] += (step_now && !take_right) as usize;
            }};
        }

        step_lane!(0);
        step_lane!(1);
        step_lane!(2);
        step_lane!(3);
    }

    [
        !rejected[0],
        !rejected[1],
        !rejected[2],
        !rejected[3],
    ]
}

pub fn scan_prepared_first_dense_bucket_lockstep4<const PIECE_COUNT: usize, const LEFT_LEN: usize>(
    prepared_first: &PreparedFirstDense<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut canonical_count = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results = canonical_pair_from_prepared_first_dense_left_len_lockstep4_chunk::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }
    for entry in chunks.remainder() {
        canonical_count += canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, LEFT_LEN>(
            prepared_first,
            &entry.left_spine,
        ) as u64;
    }
    canonical_count
}

pub fn scan_prepared_first_dense_contiguous_bucket_lockstep4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguous<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut canonical_count = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results = canonical_pair_from_prepared_first_dense_contiguous_left_len_lockstep4_chunk::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }
    for entry in chunks.remainder() {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn count_mismatches_prepared_first_dense_bucket_lockstep4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDense<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results = canonical_pair_from_prepared_first_dense_left_len_lockstep4_chunk::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, LEFT_LEN>(
                prepared_first,
                &entry.left_spine,
            );
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, LEFT_LEN>(
            prepared_first,
            &entry.left_spine,
        );
        let lockstep = canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, LEFT_LEN>(
            prepared_first,
            &entry.left_spine,
        );
        if scalar != lockstep {
            mismatches += 1;
        }
    }
    mismatches
}

pub fn count_mismatches_prepared_first_dense_contiguous_bucket_lockstep4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguous<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results = canonical_pair_from_prepared_first_dense_contiguous_left_len_lockstep4_chunk::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &entry.left_spine);
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        let lockstep = canonical_pair_from_prepared_first_dense_contiguous_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        if scalar != lockstep {
            mismatches += 1;
        }
    }
    mismatches
}

fn canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len_lockstep4_chunk<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwapped<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 4],
) -> [bool; 4] {
    debug_assert!(entries
        .iter()
        .all(|entry| entry.left_spine.len as usize == LEFT_LEN));
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return [false; 4];
    }

    let right_spine_len = prepared_first.right_len as usize;
    let stages = right_spine_len + LEFT_LEN - 1;
    let right_priority_score = &prepared_first.right_priority_score;
    let dense_matrix = &prepared_first.dense_matrix;
    let mut i = [0usize; 4];
    let mut j = [0usize; 4];
    let mut rejected = [false; 4];

    for stage_idx in 0..stages {
        let can_step = stage_idx + 1 < stages;
        macro_rules! step_lane {
            ($lane:literal) => {{
                let active = !rejected[$lane];
                debug_assert!(i[$lane] < right_spine_len);
                debug_assert!(j[$lane] < LEFT_LEN);

                let right_priority_score = unsafe { *right_priority_score.get_unchecked(i[$lane]) };
                let left_id = unsafe { *entries[$lane].left_spine.ids.get_unchecked(j[$lane]) };
                let left_priority_score =
                    unsafe { *entries[$lane].left_spine.priority_score.get_unchecked(j[$lane]) };
                debug_assert!((left_id as usize) < PIECE_COUNT);
                let cross_priority_score = unsafe {
                    *dense_matrix.get_unchecked(left_id as usize * MAX_PACKED_SPINE_LEN + i[$lane])
                };

                let reject_now = active
                    && cross_priority_score != 0
                    && cross_priority_score >= right_priority_score
                    && cross_priority_score >= left_priority_score;
                let step_now = can_step && active && !reject_now;
                let take_right = right_priority_score >= left_priority_score;

                rejected[$lane] |= reject_now;
                i[$lane] += (step_now && take_right) as usize;
                j[$lane] += (step_now && !take_right) as usize;
            }};
        }

        step_lane!(0);
        step_lane!(1);
        step_lane!(2);
        step_lane!(3);
    }

    [
        !rejected[0],
        !rejected[1],
        !rejected[2],
        !rejected[3],
    ]
}

pub fn scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwapped<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut canonical_count = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }
    for entry in chunks.remainder() {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwapped<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &entry.left_spine);
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        let lockstep = canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        if scalar != lockstep {
            mismatches += 1;
        }
    }
    mismatches
}

fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 4],
) -> [bool; 4] {
    debug_assert!(entries
        .iter()
        .all(|entry| entry.left_spine.len as usize == LEFT_LEN));
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return [false; 4];
    }

    let right_spine_len = prepared_first.right_len as usize;
    let stages = right_spine_len + LEFT_LEN - 1;
    let right_priority_score = &prepared_first.right_priority_score;
    let dense_matrix = &prepared_first.dense_matrix;
    let mut i = [0usize; 4];
    let mut j = [0usize; 4];
    let mut rejected = [false; 4];

    for stage_idx in 0..stages {
        let can_step = stage_idx + 1 < stages;
        macro_rules! step_lane {
            ($lane:literal) => {{
                let active = !rejected[$lane];
                debug_assert!(i[$lane] < right_spine_len);
                debug_assert!(j[$lane] < LEFT_LEN);

                let right_priority_score = unsafe { *right_priority_score.get_unchecked(i[$lane]) };
                let left_id = unsafe { *entries[$lane].left_spine.ids.get_unchecked(j[$lane]) };
                let left_priority_score =
                    unsafe { *entries[$lane].left_spine.priority_score.get_unchecked(j[$lane]) };
                debug_assert!((left_id as usize) < PIECE_COUNT);
                let cross_priority_score =
                    unsafe { *dense_matrix.get_unchecked(left_id as usize * right_spine_len + i[$lane]) };

                let reject_now = active
                    && cross_priority_score != 0
                    && cross_priority_score >= right_priority_score
                    && cross_priority_score >= left_priority_score;
                let step_now = can_step && active && !reject_now;
                let take_right = right_priority_score >= left_priority_score;

                rejected[$lane] |= reject_now;
                i[$lane] += (step_now && take_right) as usize;
                j[$lane] += (step_now && !take_right) as usize;
            }};
        }

        step_lane!(0);
        step_lane!(1);
        step_lane!(2);
        step_lane!(3);
    }

    [
        !rejected[0],
        !rejected[1],
        !rejected[2],
        !rejected[3],
    ]
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd8_chunk_avx2<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 8],
    simd_chunk: &PreparedSecondSimd8Chunk,
) -> [bool; 8] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256, _mm256_castsi256_ps,
        _mm256_cmpeq_epi32, _mm256_cmpgt_epi32, _mm256_i32gather_epi32, _mm256_movemask_ps,
        _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set1_epi32, _mm256_setzero_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256, _mm256_castsi256_ps,
        _mm256_cmpeq_epi32, _mm256_cmpgt_epi32, _mm256_i32gather_epi32, _mm256_movemask_ps,
        _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set1_epi32, _mm256_setzero_si256,
    };

    #[inline(always)]
    fn ge_epi32(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let gt = _mm256_cmpgt_epi32(a, b);
            let eq = _mm256_cmpeq_epi32(a, b);
            _mm256_or_si256(gt, eq)
        }
    }

    debug_assert!(entries
        .iter()
        .all(|entry| entry.left_spine.len as usize == LEFT_LEN));
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return [false; 8];
    }

    let right_spine_len = prepared_first.right_len as usize;
    let stages = right_spine_len + LEFT_LEN - 1;
    let right_priority_score_i32 = {
        let mut v = [0i32; MAX_PACKED_SPINE_LEN];
        for idx in 0..right_spine_len {
            v[idx] = prepared_first.right_priority_score[idx] as i32;
        }
        v
    };
    let right_v_table =
        unsafe { std::mem::transmute::<[i32; MAX_PACKED_SPINE_LEN], __m256i>(right_priority_score_i32) };
    let dense_u32_ptr = prepared_first.dense_matrix_u32.as_ptr() as *const i32;
    let zero_v = _mm256_setzero_si256();
    let all_ones_v = _mm256_cmpeq_epi32(zero_v, zero_v);
    let one_v = _mm256_set1_epi32(1);
    let mut i_v = zero_v;
    let mut j_v = zero_v;
    let mut rejected_v = zero_v;
    let mut left_score_by_depth_v = [zero_v; MAX_PACKED_SPINE_LEN];
    let mut row_base_by_depth_v = [zero_v; MAX_PACKED_SPINE_LEN];
    for depth in 0..LEFT_LEN {
        let mut left_arr = [0i32; 8];
        let mut row_base_arr = [0i32; 8];
        for lane in 0..8 {
            left_arr[lane] = simd_chunk.left_priority_scores_by_depth[depth][lane] as i32;
            row_base_arr[lane] =
                simd_chunk.row_base_by_right_len_by_depth[right_spine_len][depth][lane] as i32;
        }
        left_score_by_depth_v[depth] = unsafe { std::mem::transmute::<[i32; 8], __m256i>(left_arr) };
        row_base_by_depth_v[depth] =
            unsafe { std::mem::transmute::<[i32; 8], __m256i>(row_base_arr) };
    }
    let depth_constants = [
        _mm256_set1_epi32(0),
        _mm256_set1_epi32(1),
        _mm256_set1_epi32(2),
        _mm256_set1_epi32(3),
        _mm256_set1_epi32(4),
        _mm256_set1_epi32(5),
        _mm256_set1_epi32(6),
        _mm256_set1_epi32(7),
    ];

    for stage_idx in 0..stages {
        let can_step = stage_idx + 1 < stages;
        let active_v = _mm256_cmpeq_epi32(rejected_v, zero_v);
        let right_v = _mm256_permutevar8x32_epi32(right_v_table, i_v);
        let (left_v, row_base_v) = if LEFT_LEN == 1 {
            (left_score_by_depth_v[0], row_base_by_depth_v[0])
        } else if LEFT_LEN == 2 {
            let m1 = _mm256_cmpeq_epi32(j_v, depth_constants[1]);
            let m0 = _mm256_andnot_si256(m1, all_ones_v);
            (
                _mm256_or_si256(
                    _mm256_and_si256(m0, left_score_by_depth_v[0]),
                    _mm256_and_si256(m1, left_score_by_depth_v[1]),
                ),
                _mm256_or_si256(
                    _mm256_and_si256(m0, row_base_by_depth_v[0]),
                    _mm256_and_si256(m1, row_base_by_depth_v[1]),
                ),
            )
        } else if LEFT_LEN == 3 {
            let m2 = _mm256_cmpeq_epi32(j_v, depth_constants[2]);
            let m1 = _mm256_cmpeq_epi32(j_v, depth_constants[1]);
            let m0 = _mm256_andnot_si256(_mm256_or_si256(m1, m2), all_ones_v);
            (
                _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_and_si256(m0, left_score_by_depth_v[0]),
                        _mm256_and_si256(m1, left_score_by_depth_v[1]),
                    ),
                    _mm256_and_si256(m2, left_score_by_depth_v[2]),
                ),
                _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_and_si256(m0, row_base_by_depth_v[0]),
                        _mm256_and_si256(m1, row_base_by_depth_v[1]),
                    ),
                    _mm256_and_si256(m2, row_base_by_depth_v[2]),
                ),
            )
        } else if LEFT_LEN == 4 {
            let m3 = _mm256_cmpeq_epi32(j_v, depth_constants[3]);
            let m2 = _mm256_cmpeq_epi32(j_v, depth_constants[2]);
            let m1 = _mm256_cmpeq_epi32(j_v, depth_constants[1]);
            let m0 = _mm256_andnot_si256(_mm256_or_si256(_mm256_or_si256(m1, m2), m3), all_ones_v);
            (
                _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_or_si256(
                            _mm256_and_si256(m0, left_score_by_depth_v[0]),
                            _mm256_and_si256(m1, left_score_by_depth_v[1]),
                        ),
                        _mm256_and_si256(m2, left_score_by_depth_v[2]),
                    ),
                    _mm256_and_si256(m3, left_score_by_depth_v[3]),
                ),
                _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_or_si256(
                            _mm256_and_si256(m0, row_base_by_depth_v[0]),
                            _mm256_and_si256(m1, row_base_by_depth_v[1]),
                        ),
                        _mm256_and_si256(m2, row_base_by_depth_v[2]),
                    ),
                    _mm256_and_si256(m3, row_base_by_depth_v[3]),
                ),
            )
        } else {
            let mut left_v = zero_v;
            let mut row_base_v = zero_v;
            for depth in 0..LEFT_LEN {
                let depth_mask = _mm256_cmpeq_epi32(j_v, depth_constants[depth]);
                left_v = _mm256_or_si256(
                    left_v,
                    _mm256_and_si256(depth_mask, left_score_by_depth_v[depth]),
                );
                row_base_v = _mm256_or_si256(
                    row_base_v,
                    _mm256_and_si256(depth_mask, row_base_by_depth_v[depth]),
                );
            }
            (left_v, row_base_v)
        };
        let cross_index_v = _mm256_add_epi32(row_base_v, i_v);
        let cross_v = unsafe { _mm256_i32gather_epi32(dense_u32_ptr, cross_index_v, 4) };

        let cross_nonzero = _mm256_cmpgt_epi32(cross_v, zero_v);
        let cross_ge_right = ge_epi32(cross_v, right_v);
        let cross_ge_left = ge_epi32(cross_v, left_v);
        let reject_mask = {
            let m = _mm256_and_si256(active_v, cross_nonzero);
            let m = _mm256_and_si256(m, cross_ge_right);
            _mm256_and_si256(m, cross_ge_left)
        };
        let can_step_mask = if can_step { all_ones_v } else { zero_v };
        let not_reject_mask = _mm256_andnot_si256(reject_mask, all_ones_v);
        let step_mask =
            _mm256_and_si256(can_step_mask, _mm256_and_si256(active_v, not_reject_mask));
        let take_right_mask = ge_epi32(right_v, left_v);
        let i_inc_mask = _mm256_and_si256(step_mask, take_right_mask);
        let j_inc_mask = _mm256_andnot_si256(take_right_mask, step_mask);
        i_v = _mm256_add_epi32(i_v, _mm256_and_si256(i_inc_mask, one_v));
        j_v = _mm256_add_epi32(j_v, _mm256_and_si256(j_inc_mask, one_v));
        rejected_v = _mm256_or_si256(rejected_v, reject_mask);
    }

    let rejected_bits = _mm256_movemask_ps(_mm256_castsi256_ps(rejected_v)) as u32;
    let accepted_bits = (!rejected_bits) & 0xFF;
    [
        (accepted_bits & (1 << 0)) != 0,
        (accepted_bits & (1 << 1)) != 0,
        (accepted_bits & (1 << 2)) != 0,
        (accepted_bits & (1 << 3)) != 0,
        (accepted_bits & (1 << 4)) != 0,
        (accepted_bits & (1 << 5)) != 0,
        (accepted_bits & (1 << 6)) != 0,
        (accepted_bits & (1 << 7)) != 0,
    ]
}

fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd8_chunk<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 8],
    simd_chunk: &PreparedSecondSimd8Chunk,
) -> [bool; 8] {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd8_chunk_avx2::<
                    PIECE_COUNT,
                    LEFT_LEN,
                >(prepared_first, entries, simd_chunk)
            };
        }
    }

    let mut out = [false; 8];
    for lane in 0..8 {
        out[lane] = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entries[lane].left_spine);
    }
    out
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut canonical_count = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }
    for entry in chunks.remainder() {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd4_chunk_avx2<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 4],
    simd_chunk: &PreparedSecondSimd4Chunk,
) -> [bool; 4] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128i, _mm_add_epi32, _mm_and_si128, _mm_andnot_si128, _mm_castsi128_ps,
        _mm_cmpeq_epi32, _mm_cmpgt_epi32, _mm_i32gather_epi32, _mm_movemask_ps, _mm_or_si128,
        _mm_set1_epi32, _mm_setzero_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_and_si128, _mm_andnot_si128, _mm_castsi128_ps,
        _mm_cmpeq_epi32, _mm_cmpgt_epi32, _mm_i32gather_epi32, _mm_movemask_ps, _mm_or_si128,
        _mm_set1_epi32, _mm_setzero_si128,
    };

    #[inline(always)]
    fn ge_epi32(a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            let gt = _mm_cmpgt_epi32(a, b);
            let eq = _mm_cmpeq_epi32(a, b);
            _mm_or_si128(gt, eq)
        }
    }

    debug_assert!(entries
        .iter()
        .all(|entry| entry.left_spine.len as usize == LEFT_LEN));
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return [false; 4];
    }

    let right_spine_len = prepared_first.right_len as usize;
    let stages = right_spine_len + LEFT_LEN - 1;
    let right_priority_score_i32 = {
        let mut v = [0i32; MAX_PACKED_SPINE_LEN];
        for idx in 0..right_spine_len {
            v[idx] = prepared_first.right_priority_score[idx] as i32;
        }
        v
    };
    let right_ptr = right_priority_score_i32.as_ptr();
    let dense_u32_ptr = prepared_first.dense_matrix_u32.as_ptr() as *const i32;
    let zero_v = _mm_setzero_si128();
    let all_ones_v = _mm_cmpeq_epi32(zero_v, zero_v);
    let one_v = _mm_set1_epi32(1);
    let mut i_v = zero_v;
    let mut j_v = zero_v;
    let mut rejected_v = zero_v;
    let mut left_score_by_depth_v = [zero_v; MAX_PACKED_SPINE_LEN];
    let mut row_base_by_depth_v = [zero_v; MAX_PACKED_SPINE_LEN];
    for depth in 0..LEFT_LEN {
        let mut left_arr = [0i32; 4];
        let mut row_base_arr = [0i32; 4];
        for lane in 0..4 {
            left_arr[lane] = simd_chunk.left_priority_scores_by_depth[depth][lane] as i32;
            row_base_arr[lane] =
                simd_chunk.row_base_by_right_len_by_depth[right_spine_len][depth][lane] as i32;
        }
        left_score_by_depth_v[depth] =
            unsafe { std::mem::transmute::<[i32; 4], __m128i>(left_arr) };
        row_base_by_depth_v[depth] =
            unsafe { std::mem::transmute::<[i32; 4], __m128i>(row_base_arr) };
    }
    let depth_constants = [
        _mm_set1_epi32(0),
        _mm_set1_epi32(1),
        _mm_set1_epi32(2),
        _mm_set1_epi32(3),
        _mm_set1_epi32(4),
        _mm_set1_epi32(5),
        _mm_set1_epi32(6),
        _mm_set1_epi32(7),
    ];

    for stage_idx in 0..stages {
        let can_step = stage_idx + 1 < stages;
        let active_v = _mm_cmpeq_epi32(rejected_v, zero_v);
        let right_v = unsafe { _mm_i32gather_epi32(right_ptr, i_v, 4) };
        let mut left_v = zero_v;
        let mut row_base_v = zero_v;
        for depth in 0..LEFT_LEN {
            let depth_mask = _mm_cmpeq_epi32(j_v, depth_constants[depth]);
            left_v = _mm_or_si128(left_v, _mm_and_si128(depth_mask, left_score_by_depth_v[depth]));
            row_base_v = _mm_or_si128(
                row_base_v,
                _mm_and_si128(depth_mask, row_base_by_depth_v[depth]),
            );
        }
        let cross_index_v = _mm_add_epi32(row_base_v, i_v);
        let cross_v = unsafe { _mm_i32gather_epi32(dense_u32_ptr, cross_index_v, 4) };

        let cross_nonzero = _mm_cmpgt_epi32(cross_v, zero_v);
        let cross_ge_right = ge_epi32(cross_v, right_v);
        let cross_ge_left = ge_epi32(cross_v, left_v);
        let reject_mask = {
            let m = _mm_and_si128(active_v, cross_nonzero);
            let m = _mm_and_si128(m, cross_ge_right);
            _mm_and_si128(m, cross_ge_left)
        };
        let can_step_mask = if can_step { all_ones_v } else { zero_v };
        let not_reject_mask = _mm_andnot_si128(reject_mask, all_ones_v);
        let step_mask = _mm_and_si128(can_step_mask, _mm_and_si128(active_v, not_reject_mask));
        let take_right_mask = ge_epi32(right_v, left_v);
        let i_inc_mask = _mm_and_si128(step_mask, take_right_mask);
        let j_inc_mask = _mm_andnot_si128(take_right_mask, step_mask);
        i_v = _mm_add_epi32(i_v, _mm_and_si128(i_inc_mask, one_v));
        j_v = _mm_add_epi32(j_v, _mm_and_si128(j_inc_mask, one_v));
        rejected_v = _mm_or_si128(rejected_v, reject_mask);
    }

    let rejected_bits = _mm_movemask_ps(_mm_castsi128_ps(rejected_v)) as u32;
    let accepted_bits = (!rejected_bits) & 0xF;
    [
        (accepted_bits & (1 << 0)) != 0,
        (accepted_bits & (1 << 1)) != 0,
        (accepted_bits & (1 << 2)) != 0,
        (accepted_bits & (1 << 3)) != 0,
    ]
}

fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd4_chunk<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 4],
    simd_chunk: &PreparedSecondSimd4Chunk,
) -> [bool; 4] {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd4_chunk_avx2::<
                    PIECE_COUNT,
                    LEFT_LEN,
                >(prepared_first, entries, simd_chunk)
            };
        }
    }

    let mut out = [false; 4];
    for lane in 0..4 {
        out[lane] = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entries[lane].left_spine);
    }
    out
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd4Chunk],
) -> u64 {
    let mut canonical_count = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for (chunk_idx, chunk) in (&mut chunks).enumerate() {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let simd_chunk = unsafe { simd_chunks.get_unchecked(chunk_idx) };
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk, simd_chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }
    for entry in chunks.remainder() {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd4Chunk],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for (chunk_idx, chunk) in (&mut chunks).enumerate() {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let simd_chunk = unsafe { simd_chunks.get_unchecked(chunk_idx) };
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk, simd_chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &entry.left_spine);
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        let simd = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        if scalar != simd {
            mismatches += 1;
        }
    }
    mismatches
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
unsafe fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd16_chunk_avx512<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 16],
    simd_chunk: &PreparedSecondSimd16Chunk,
) -> [bool; 16] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m512i, __mmask16, _mm512_add_epi32, _mm512_cmpeq_epi32_mask, _mm512_cmpgt_epi32_mask,
        _mm512_i32gather_epi32, _mm512_mask_add_epi32, _mm512_mask_mov_epi32,
        _mm512_permutexvar_epi32, _mm512_set1_epi32, _mm512_setzero_si512,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m512i, __mmask16, _mm512_add_epi32, _mm512_cmpeq_epi32_mask, _mm512_cmpgt_epi32_mask,
        _mm512_i32gather_epi32, _mm512_mask_add_epi32, _mm512_mask_mov_epi32,
        _mm512_permutexvar_epi32, _mm512_set1_epi32, _mm512_setzero_si512,
    };

    debug_assert!(entries
        .iter()
        .all(|entry| entry.left_spine.len as usize == LEFT_LEN));
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return [false; 16];
    }

    let right_spine_len = prepared_first.right_len as usize;
    let stages = right_spine_len + LEFT_LEN - 1;
    let right_priority_score_i32 = {
        let mut v = [0i32; 16];
        for idx in 0..right_spine_len {
            v[idx] = prepared_first.right_priority_score[idx] as i32;
        }
        v
    };
    let right_v_table =
        unsafe { std::mem::transmute::<[i32; 16], __m512i>(right_priority_score_i32) };
    let dense_u32_ptr = prepared_first.dense_matrix_u32.as_ptr() as *const i32;
    let zero_v = _mm512_setzero_si512();
    let one_v = _mm512_set1_epi32(1);
    let mut i_v = zero_v;
    let mut j_v = zero_v;
    let mut rejected_mask: __mmask16 = 0;
    let mut left_score_by_depth_v = [zero_v; MAX_PACKED_SPINE_LEN];
    let mut row_base_by_depth_v = [zero_v; MAX_PACKED_SPINE_LEN];
    for depth in 0..LEFT_LEN {
        left_score_by_depth_v[depth] = unsafe {
            std::mem::transmute::<[u32; 16], __m512i>(
                simd_chunk.left_priority_scores_by_depth[depth],
            )
        };
        row_base_by_depth_v[depth] = unsafe {
            std::mem::transmute::<[u32; 16], __m512i>(
                simd_chunk.row_base_by_right_len_by_depth[right_spine_len][depth],
            )
        };
    }

    for stage_idx in 0..stages {
        let can_step = stage_idx + 1 < stages;
        let active_mask: __mmask16 = !rejected_mask;
        let right_v = _mm512_permutexvar_epi32(i_v, right_v_table);
        let (left_v, row_base_v) = if LEFT_LEN == 1 {
            (left_score_by_depth_v[0], row_base_by_depth_v[0])
        } else if LEFT_LEN == 2 {
            let m1 = _mm512_cmpeq_epi32_mask(j_v, one_v);
            (
                _mm512_mask_mov_epi32(left_score_by_depth_v[0], m1, left_score_by_depth_v[1]),
                _mm512_mask_mov_epi32(row_base_by_depth_v[0], m1, row_base_by_depth_v[1]),
            )
        } else if LEFT_LEN == 3 {
            let m1 = _mm512_cmpeq_epi32_mask(j_v, one_v);
            let m2 = _mm512_cmpeq_epi32_mask(j_v, _mm512_set1_epi32(2));
            let left_v =
                _mm512_mask_mov_epi32(left_score_by_depth_v[0], m1, left_score_by_depth_v[1]);
            let left_v = _mm512_mask_mov_epi32(left_v, m2, left_score_by_depth_v[2]);
            let rb_v =
                _mm512_mask_mov_epi32(row_base_by_depth_v[0], m1, row_base_by_depth_v[1]);
            let rb_v = _mm512_mask_mov_epi32(rb_v, m2, row_base_by_depth_v[2]);
            (left_v, rb_v)
        } else if LEFT_LEN == 4 {
            let m1 = _mm512_cmpeq_epi32_mask(j_v, one_v);
            let m2 = _mm512_cmpeq_epi32_mask(j_v, _mm512_set1_epi32(2));
            let m3 = _mm512_cmpeq_epi32_mask(j_v, _mm512_set1_epi32(3));
            let left_v =
                _mm512_mask_mov_epi32(left_score_by_depth_v[0], m1, left_score_by_depth_v[1]);
            let left_v = _mm512_mask_mov_epi32(left_v, m2, left_score_by_depth_v[2]);
            let left_v = _mm512_mask_mov_epi32(left_v, m3, left_score_by_depth_v[3]);
            let rb_v =
                _mm512_mask_mov_epi32(row_base_by_depth_v[0], m1, row_base_by_depth_v[1]);
            let rb_v = _mm512_mask_mov_epi32(rb_v, m2, row_base_by_depth_v[2]);
            let rb_v = _mm512_mask_mov_epi32(rb_v, m3, row_base_by_depth_v[3]);
            (left_v, rb_v)
        } else {
            let mut left_v = left_score_by_depth_v[0];
            let mut rb_v = row_base_by_depth_v[0];
            for depth in 1..LEFT_LEN {
                let dm = _mm512_cmpeq_epi32_mask(j_v, _mm512_set1_epi32(depth as i32));
                left_v = _mm512_mask_mov_epi32(left_v, dm, left_score_by_depth_v[depth]);
                rb_v = _mm512_mask_mov_epi32(rb_v, dm, row_base_by_depth_v[depth]);
            }
            (left_v, rb_v)
        };
        let cross_index_v = _mm512_add_epi32(row_base_v, i_v);
        let cross_v =
            unsafe { _mm512_i32gather_epi32::<4>(cross_index_v, dense_u32_ptr) };

        let cross_nonzero = _mm512_cmpgt_epi32_mask(cross_v, zero_v);
        let cross_lt_right = _mm512_cmpgt_epi32_mask(right_v, cross_v);
        let cross_lt_left = _mm512_cmpgt_epi32_mask(left_v, cross_v);
        let cross_ge_right: __mmask16 = !cross_lt_right;
        let cross_ge_left: __mmask16 = !cross_lt_left;
        let reject_now = active_mask & cross_nonzero & cross_ge_right & cross_ge_left;

        let step_mask: __mmask16 = if can_step { active_mask & !reject_now } else { 0 };
        let take_right_lt = _mm512_cmpgt_epi32_mask(left_v, right_v);
        let take_right: __mmask16 = !take_right_lt;
        let i_inc = step_mask & take_right;
        let j_inc = step_mask & !take_right;
        i_v = _mm512_mask_add_epi32(i_v, i_inc, i_v, one_v);
        j_v = _mm512_mask_add_epi32(j_v, j_inc, j_v, one_v);
        rejected_mask |= reject_now;
    }

    let accepted: u16 = !rejected_mask;
    let mut out = [false; 16];
    for lane in 0..16 {
        out[lane] = (accepted >> lane) & 1 != 0;
    }
    out
}

fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd16_chunk<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 16],
    simd_chunk: &PreparedSecondSimd16Chunk,
) -> [bool; 16] {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return unsafe {
                canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd16_chunk_avx512::<
                    PIECE_COUNT,
                    LEFT_LEN,
                >(prepared_first, entries, simd_chunk)
            };
        }
    }

    let mut out = [false; 16];
    for lane in 0..16 {
        out[lane] = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entries[lane].left_spine);
    }
    out
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd16<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd16Chunk],
) -> u64 {
    let mut canonical_count = 0u64;
    let mut chunks = entries.chunks_exact(16);
    for (chunk_idx, chunk) in (&mut chunks).enumerate() {
        let chunk: &[PreparedSecondToken; 16] = chunk
            .try_into()
            .expect("chunks_exact(16) must yield 16-lane chunks");
        let simd_chunk = unsafe { simd_chunks.get_unchecked(chunk_idx) };
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd16_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk, simd_chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }
    for entry in chunks.remainder() {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd16<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd16Chunk],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(16);
    for (chunk_idx, chunk) in (&mut chunks).enumerate() {
        let chunk: &[PreparedSecondToken; 16] = chunk
            .try_into()
            .expect("chunks_exact(16) must yield 16-lane chunks");
        let simd_chunk = unsafe { simd_chunks.get_unchecked(chunk_idx) };
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd16_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk, simd_chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &entry.left_spine);
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        let simd = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        if scalar != simd {
            mismatches += 1;
        }
    }
    mismatches
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_simd8<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd8Chunk],
) -> u64 {
    let mut canonical_count = 0u64;
    let mut chunks = entries.chunks_exact(8);
    for (chunk_idx, chunk) in (&mut chunks).enumerate() {
        let chunk: &[PreparedSecondToken; 8] = chunk
            .try_into()
            .expect("chunks_exact(8) must yield 8-lane chunks");
        let simd_chunk = unsafe { simd_chunks.get_unchecked(chunk_idx) };
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd8_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk, simd_chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }
    for entry in chunks.remainder() {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_simd8<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
    simd_chunks: &[PreparedSecondSimd8Chunk],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(8);
    for (chunk_idx, chunk) in (&mut chunks).enumerate() {
        let chunk: &[PreparedSecondToken; 8] = chunk
            .try_into()
            .expect("chunks_exact(8) must yield 8-lane chunks");
        let simd_chunk = unsafe { simd_chunks.get_unchecked(chunk_idx) };
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd8_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk, simd_chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &entry.left_spine);
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        let simd = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        if scalar != simd {
            mismatches += 1;
        }
    }
    mismatches
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut canonical_count = 0u64;
    const CHUNK_WIDTH: usize = 4;
    const PREFETCH_LOOKAHEAD_CHUNKS: usize = 1;
    let dense_matrix = &prepared_first.dense_matrix;
    let right_len = prepared_first.right_len as usize;
    let chunk_count = entries.len() / CHUNK_WIDTH;

    for chunk_idx in 0..chunk_count {
        let prefetch_chunk_idx = chunk_idx + PREFETCH_LOOKAHEAD_CHUNKS;
        if prefetch_chunk_idx < chunk_count {
            let prefetch_base = prefetch_chunk_idx * CHUNK_WIDTH;
            let prefetch_chunk: &[PreparedSecondToken; CHUNK_WIDTH] = entries
                [prefetch_base..prefetch_base + CHUNK_WIDTH]
                .try_into()
                .expect("prefetch chunk must contain 4 lanes");
            for entry in prefetch_chunk {
                debug_assert_eq!(entry.left_spine.len as usize, LEFT_LEN);
                for depth in 0..LEFT_LEN {
                    let left_id = entry.left_spine.ids[depth] as usize;
                    debug_assert!(left_id < PIECE_COUNT);
                    let ptr = unsafe { dense_matrix.as_ptr().add(left_id * right_len) };
                    prefetch_t0(ptr);
                }
            }
        }

        let base = chunk_idx * CHUNK_WIDTH;
        let chunk: &[PreparedSecondToken; CHUNK_WIDTH] = entries[base..base + CHUNK_WIDTH]
            .try_into()
            .expect("chunk must contain 4 lanes");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }

    for entry in &entries[chunk_count * CHUNK_WIDTH..] {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn build_prefetch_left_id_chunks<const LEFT_LEN: usize>(
    entries: &[PreparedSecondToken],
) -> Vec<PrefetchLeftIdChunk> {
    let chunk_count = entries.len() / PREFETCH_CHUNK_WIDTH;
    let mut chunks = Vec::with_capacity(chunk_count);
    for chunk_idx in 0..chunk_count {
        let base = chunk_idx * PREFETCH_CHUNK_WIDTH;
        let lanes: &[PreparedSecondToken; PREFETCH_CHUNK_WIDTH] = entries
            [base..base + PREFETCH_CHUNK_WIDTH]
            .try_into()
            .expect("chunk must contain 4 lanes");

        let mut ids = [0u16; MAX_PREFETCH_LEFT_IDS_PER_CHUNK];
        let mut counts_by_scope = [0u8; MAX_PACKED_SPINE_LEN + 1];
        let mut count = 0usize;
        for entry in lanes {
            debug_assert_eq!(entry.left_spine.len as usize, LEFT_LEN);
            for depth in 0..LEFT_LEN {
                let id = entry.left_spine.ids[depth];
                let mut seen = false;
                for &existing in &ids[..count] {
                    if existing == id {
                        seen = true;
                        break;
                    }
                }
                if !seen {
                    ids[count] = id;
                    count += 1;
                }
                counts_by_scope[depth + 1] = count as u8;
            }
        }

        chunks.push(PrefetchLeftIdChunk {
            count: count as u8,
            counts_by_scope,
            ids,
        });
    }
    chunks
}

pub fn build_prepared_second_simd8_chunks<const LEFT_LEN: usize>(
    entries: &[PreparedSecondToken],
) -> Vec<PreparedSecondSimd8Chunk> {
    let chunk_count = entries.len() / 8;
    let mut chunks = Vec::with_capacity(chunk_count);
    for chunk_idx in 0..chunk_count {
        let base = chunk_idx * 8;
        let lanes: &[PreparedSecondToken; 8] = entries[base..base + 8]
            .try_into()
            .expect("chunk must contain 8 lanes");
        let mut left_priority_scores_by_depth = [[0u32; 8]; MAX_PACKED_SPINE_LEN];
        let mut row_base_by_right_len_by_depth =
            [[[0u32; 8]; MAX_PACKED_SPINE_LEN]; MAX_PACKED_SPINE_LEN + 1];
        for lane in 0..8 {
            debug_assert_eq!(lanes[lane].left_spine.len as usize, LEFT_LEN);
            for depth in 0..LEFT_LEN {
                let left_id = lanes[lane].left_spine.ids[depth] as u32;
                left_priority_scores_by_depth[depth][lane] =
                    lanes[lane].left_spine.priority_score[depth] as u32;
                for right_len in 1..=MAX_PACKED_SPINE_LEN {
                    row_base_by_right_len_by_depth[right_len][depth][lane] =
                        left_id * right_len as u32;
                }
            }
        }
        chunks.push(PreparedSecondSimd8Chunk {
            left_priority_scores_by_depth,
            row_base_by_right_len_by_depth,
        });
    }
    chunks
}

pub fn build_prepared_second_simd4_chunks<const LEFT_LEN: usize>(
    entries: &[PreparedSecondToken],
) -> Vec<PreparedSecondSimd4Chunk> {
    let chunk_count = entries.len() / 4;
    let mut chunks = Vec::with_capacity(chunk_count);
    for chunk_idx in 0..chunk_count {
        let base = chunk_idx * 4;
        let lanes: &[PreparedSecondToken; 4] = entries[base..base + 4]
            .try_into()
            .expect("chunk must contain 4 lanes");
        let mut left_priority_scores_by_depth = [[0u32; 4]; MAX_PACKED_SPINE_LEN];
        let mut row_base_by_right_len_by_depth =
            [[[0u32; 4]; MAX_PACKED_SPINE_LEN]; MAX_PACKED_SPINE_LEN + 1];
        for lane in 0..4 {
            debug_assert_eq!(lanes[lane].left_spine.len as usize, LEFT_LEN);
            for depth in 0..LEFT_LEN {
                let left_id = lanes[lane].left_spine.ids[depth] as u32;
                left_priority_scores_by_depth[depth][lane] =
                    lanes[lane].left_spine.priority_score[depth] as u32;
                for right_len in 1..=MAX_PACKED_SPINE_LEN {
                    row_base_by_right_len_by_depth[right_len][depth][lane] =
                        left_id * right_len as u32;
                }
            }
        }
        chunks.push(PreparedSecondSimd4Chunk {
            left_priority_scores_by_depth,
            row_base_by_right_len_by_depth,
        });
    }
    chunks
}

pub fn build_prepared_second_simd16_chunks<const LEFT_LEN: usize>(
    entries: &[PreparedSecondToken],
) -> Vec<PreparedSecondSimd16Chunk> {
    let chunk_count = entries.len() / 16;
    let mut chunks = Vec::with_capacity(chunk_count);
    for chunk_idx in 0..chunk_count {
        let base = chunk_idx * 16;
        let lanes: &[PreparedSecondToken; 16] = entries[base..base + 16]
            .try_into()
            .expect("chunk must contain 16 lanes");
        let mut left_priority_scores_by_depth = [[0u32; 16]; MAX_PACKED_SPINE_LEN];
        let mut row_base_by_right_len_by_depth =
            [[[0u32; 16]; MAX_PACKED_SPINE_LEN]; MAX_PACKED_SPINE_LEN + 1];
        for lane in 0..16 {
            debug_assert_eq!(lanes[lane].left_spine.len as usize, LEFT_LEN);
            for depth in 0..LEFT_LEN {
                let left_id = lanes[lane].left_spine.ids[depth] as u32;
                left_priority_scores_by_depth[depth][lane] =
                    lanes[lane].left_spine.priority_score[depth] as u32;
                for right_len in 1..=MAX_PACKED_SPINE_LEN {
                    row_base_by_right_len_by_depth[right_len][depth][lane] =
                        left_id * right_len as u32;
                }
            }
        }
        chunks.push(PreparedSecondSimd16Chunk {
            left_priority_scores_by_depth,
            row_base_by_right_len_by_depth,
        });
    }
    chunks
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch_prebuilt<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
    prefetch_chunks: &[PrefetchLeftIdChunk],
) -> u64 {
    let mut canonical_count = 0u64;
    const PREFETCH_LOOKAHEAD_CHUNKS: usize = 1;
    let dense_matrix = &prepared_first.dense_matrix;
    let right_len = prepared_first.right_len as usize;
    let chunk_count = entries.len() / PREFETCH_CHUNK_WIDTH;
    debug_assert_eq!(prefetch_chunks.len(), chunk_count);

    for chunk_idx in 0..chunk_count {
        let prefetch_chunk_idx = chunk_idx + PREFETCH_LOOKAHEAD_CHUNKS;
        if prefetch_chunk_idx < chunk_count {
            let prefetch_chunk = unsafe { prefetch_chunks.get_unchecked(prefetch_chunk_idx) };
            for &left_id in &prefetch_chunk.ids[..prefetch_chunk.count as usize] {
                debug_assert!((left_id as usize) < PIECE_COUNT);
                let ptr = unsafe { dense_matrix.as_ptr().add(left_id as usize * right_len) };
                prefetch_t0(ptr);
            }
        }

        let base = chunk_idx * PREFETCH_CHUNK_WIDTH;
        let chunk: &[PreparedSecondToken; PREFETCH_CHUNK_WIDTH] = entries
            [base..base + PREFETCH_CHUNK_WIDTH]
            .try_into()
            .expect("chunk must contain 4 lanes");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }

    for entry in &entries[chunk_count * PREFETCH_CHUNK_WIDTH..] {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch_prebuilt_param<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
    prefetch_chunks: &[PrefetchLeftIdChunk],
    config: PrefetchConfig,
) -> u64 {
    let mut canonical_count = 0u64;
    let dense_matrix = &prepared_first.dense_matrix;
    let right_len = prepared_first.right_len as usize;
    let chunk_count = entries.len() / PREFETCH_CHUNK_WIDTH;
    debug_assert_eq!(prefetch_chunks.len(), chunk_count);

    let scope = usize::from(config.scope).min(LEFT_LEN);
    for chunk_idx in 0..chunk_count {
        if config.enabled && config.lookahead_chunks > 0 {
            let prefetch_chunk_idx = chunk_idx + config.lookahead_chunks;
            if prefetch_chunk_idx < chunk_count {
                let prefetch_chunk = unsafe { prefetch_chunks.get_unchecked(prefetch_chunk_idx) };
                let mut prefetch_count = prefetch_chunk.counts_by_scope[scope] as usize;
                if config.budget != 0 {
                    prefetch_count = prefetch_count.min(config.budget as usize);
                }
                for &left_id in &prefetch_chunk.ids[..prefetch_count] {
                    debug_assert!((left_id as usize) < PIECE_COUNT);
                    let ptr = unsafe { dense_matrix.as_ptr().add(left_id as usize * right_len) };
                    prefetch_with_hint(ptr, config.hint);
                }
            }
        }

        let base = chunk_idx * PREFETCH_CHUNK_WIDTH;
        let chunk: &[PreparedSecondToken; PREFETCH_CHUNK_WIDTH] = entries
            [base..base + PREFETCH_CHUNK_WIDTH]
            .try_into()
            .expect("chunk must contain 4 lanes");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }

    for entry in &entries[chunk_count * PREFETCH_CHUNK_WIDTH..] {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &entry.left_spine);
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        let lockstep = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        if scalar != lockstep {
            mismatches += 1;
        }
    }
    mismatches
}

pub fn count_mismatches_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4_prefetch<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &entry.left_spine);
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        let lockstep_prefetch = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        if scalar != lockstep_prefetch {
            mismatches += 1;
        }
    }
    mismatches
}

#[inline(always)]
fn prefetch_swapped_entry_left_ids<const PIECE_COUNT: usize, const LEFT_LEN: usize>(
    dense_matrix: &[u16],
    entry: &PreparedSecondToken,
) {
    debug_assert_eq!(entry.left_spine.len as usize, LEFT_LEN);
    for depth in 0..LEFT_LEN {
        let left_id = entry.left_spine.ids[depth] as usize;
        debug_assert!(left_id < PIECE_COUNT);
        let ptr = unsafe { dense_matrix.as_ptr().add(left_id * MAX_PACKED_SPINE_LEN) };
        prefetch_t0(ptr);
    }
}

#[inline(always)]
fn prefetch_t0(ptr: *const u16) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_MM_HINT_T0, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        let _ = ptr;
    }
}

#[inline(always)]
fn prefetch_with_hint(ptr: *const u16, hint: PrefetchHint) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _mm_prefetch};
        match hint {
            PrefetchHint::T0 => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
            PrefetchHint::T1 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
            PrefetchHint::T2 => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
            PrefetchHint::Nta => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
        };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        let _ = (ptr, hint);
    }
}

pub fn scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwapped<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut canonical_count = 0u64;
    const CHUNK_WIDTH: usize = 4;
    const PREFETCH_LOOKAHEAD_CHUNKS: usize = 1;
    let dense_matrix = &prepared_first.dense_matrix;
    let chunk_count = entries.len() / CHUNK_WIDTH;

    for chunk_idx in 0..chunk_count {
        let prefetch_chunk_idx = chunk_idx + PREFETCH_LOOKAHEAD_CHUNKS;
        if prefetch_chunk_idx < chunk_count {
            let prefetch_base = prefetch_chunk_idx * CHUNK_WIDTH;
            let prefetch_chunk: &[PreparedSecondToken; CHUNK_WIDTH] = entries
                [prefetch_base..prefetch_base + CHUNK_WIDTH]
                .try_into()
                .expect("prefetch chunk must contain 4 lanes");
            for entry in prefetch_chunk {
                prefetch_swapped_entry_left_ids::<PIECE_COUNT, LEFT_LEN>(dense_matrix, entry);
            }
        }

        let base = chunk_idx * CHUNK_WIDTH;
        let chunk: &[PreparedSecondToken; CHUNK_WIDTH] = entries[base..base + CHUNK_WIDTH]
            .try_into()
            .expect("chunk must contain 4 lanes");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }

    for entry in &entries[chunk_count * CHUNK_WIDTH..] {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn scan_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch2<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwapped<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut canonical_count = 0u64;
    const CHUNK_WIDTH: usize = 4;
    const PREFETCH_LOOKAHEAD_CHUNKS: usize = 2;
    let dense_matrix = &prepared_first.dense_matrix;
    let chunk_count = entries.len() / CHUNK_WIDTH;

    for chunk_idx in 0..chunk_count {
        let prefetch_chunk_idx = chunk_idx + PREFETCH_LOOKAHEAD_CHUNKS;
        if prefetch_chunk_idx < chunk_count {
            let prefetch_base = prefetch_chunk_idx * CHUNK_WIDTH;
            let prefetch_chunk: &[PreparedSecondToken; CHUNK_WIDTH] = entries
                [prefetch_base..prefetch_base + CHUNK_WIDTH]
                .try_into()
                .expect("prefetch chunk must contain 4 lanes");
            for entry in prefetch_chunk {
                prefetch_swapped_entry_left_ids::<PIECE_COUNT, LEFT_LEN>(dense_matrix, entry);
            }
        }

        let base = chunk_idx * CHUNK_WIDTH;
        let chunk: &[PreparedSecondToken; CHUNK_WIDTH] = entries[base..base + CHUNK_WIDTH]
            .try_into()
            .expect("chunk must contain 4 lanes");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        canonical_count += results.iter().map(|&accepted| accepted as u64).sum::<u64>();
    }

    for entry in &entries[chunk_count * CHUNK_WIDTH..] {
        canonical_count += canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical_count
}

pub fn count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwapped<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &entry.left_spine);
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        let lockstep_prefetch = canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        if scalar != lockstep_prefetch {
            mismatches += 1;
        }
    }
    mismatches
}

pub fn count_mismatches_prepared_first_dense_contiguous_swapped_bucket_lockstep4_prefetch2<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwapped<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut mismatches = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let chunk: &[PreparedSecondToken; 4] = chunk
            .try_into()
            .expect("chunks_exact(4) must yield 4-lane chunks");
        let results =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len_lockstep4_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, chunk);
        for (lane, entry) in chunk.iter().enumerate() {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &entry.left_spine);
            if scalar != results[lane] {
                mismatches += 1;
            }
        }
    }
    for entry in chunks.remainder() {
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        let lockstep_prefetch = canonical_pair_from_prepared_first_dense_contiguous_swapped_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine);
        if scalar != lockstep_prefetch {
            mismatches += 1;
        }
    }
    mismatches
}
