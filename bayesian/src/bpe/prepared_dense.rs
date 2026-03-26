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

#[derive(Debug, Clone)]
pub struct PreparedFirstDenseContiguous<const PIECE_COUNT: usize> {
    right_len: u8,
    right_rank_plus_one: [u16; MAX_PACKED_SPINE_LEN],
    // Row-major matrix over right-spine index (rows) x left-piece-id (columns).
    dense_matrix: Box<[u16]>,
}

#[derive(Debug, Clone)]
pub struct PreparedFirstDenseContiguousSwapped<const PIECE_COUNT: usize> {
    right_len: u8,
    right_rank_plus_one: [u16; MAX_PACKED_SPINE_LEN],
    // Left-id-major matrix: columns are right-spine indices.
    // Index as dense_matrix[left_id * MAX_PACKED_SPINE_LEN + right_idx].
    dense_matrix: Box<[u16]>,
}

#[derive(Debug, Clone)]
pub struct PreparedFirstDenseContiguousSwappedTight<const PIECE_COUNT: usize> {
    right_len: u8,
    right_rank_plus_one: [u16; MAX_PACKED_SPINE_LEN],
    // Left-id-major matrix with dynamic row stride = right_len for this prepared first token.
    // Index as dense_matrix[left_id * right_len + right_idx].
    dense_matrix: Box<[u16]>,
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

pub const PREFETCH_CHUNK_WIDTH: usize = 4;
pub const MAX_PREFETCH_LEFT_IDS_PER_CHUNK: usize = PREFETCH_CHUNK_WIDTH * MAX_PACKED_SPINE_LEN;

#[derive(Clone, Copy, Debug)]
pub struct PrefetchLeftIdChunk {
    pub count: u8,
    pub counts_by_scope: [u8; MAX_PACKED_SPINE_LEN + 1],
    pub ids: [u16; MAX_PREFETCH_LEFT_IDS_PER_CHUNK],
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

        let mut right_rank_plus_one = [0u16; MAX_PACKED_SPINE_LEN];
        let mut dense_matrix = vec![0u16; PIECE_COUNT * MAX_PACKED_SPINE_LEN].into_boxed_slice();

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            right_rank_plus_one[spine_idx] = spine_entry.rank_plus_one;
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            let row_base = spine_idx * PIECE_COUNT;
            for entry in row {
                dense_matrix[row_base + entry.right as usize] = entry.rank_plus_one;
            }
        }

        Ok(Self {
            right_len: first_right_spine.as_slice().len() as u8,
            right_rank_plus_one,
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

        let mut right_rank_plus_one = [0u16; MAX_PACKED_SPINE_LEN];
        let mut dense_matrix = vec![0u16; PIECE_COUNT * MAX_PACKED_SPINE_LEN].into_boxed_slice();

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            right_rank_plus_one[spine_idx] = spine_entry.rank_plus_one;
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            for entry in row {
                dense_matrix[entry.right as usize * MAX_PACKED_SPINE_LEN + spine_idx] =
                    entry.rank_plus_one;
            }
        }

        Ok(Self {
            right_len: first_right_spine.as_slice().len() as u8,
            right_rank_plus_one,
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
        let mut right_rank_plus_one = [0u16; MAX_PACKED_SPINE_LEN];
        let mut dense_matrix = vec![0u16; PIECE_COUNT * right_len].into_boxed_slice();

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            right_rank_plus_one[spine_idx] = spine_entry.rank_plus_one;
            let row = &merge_rows.rows[spine_entry.id as usize];
            if row.is_empty() {
                continue;
            }
            for entry in row {
                dense_matrix[entry.right as usize * right_len + spine_idx] = entry.rank_plus_one;
            }
        }

        Ok(Self {
            right_len: right_len as u8,
            right_rank_plus_one,
            dense_matrix,
        })
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
    let right_rank_plus_one = &prepared_first.right_rank_plus_one;
    let dense_matrix = &prepared_first.dense_matrix;
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
        let cross_rank_plus_one = unsafe {
            *dense_matrix.get_unchecked(i * PIECE_COUNT + left_id as usize)
        };

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
    let right_rank_plus_one = &prepared_first.right_rank_plus_one;
    let dense_matrix = &prepared_first.dense_matrix;
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
        let cross_rank_plus_one = unsafe {
            *dense_matrix.get_unchecked(left_id as usize * MAX_PACKED_SPINE_LEN + i)
        };

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
    let right_rank_plus_one = &prepared_first.right_rank_plus_one;
    let dense_matrix = &prepared_first.dense_matrix;
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
            unsafe { *dense_matrix.get_unchecked(left_id as usize * right_spine_len + i) };

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
    let right_rank_plus_one = &prepared_first.right_rank_plus_one;
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

                let right_rank_plus_one = unsafe { *right_rank_plus_one.get_unchecked(i[$lane]) };
                let left_id = unsafe { *entries[$lane].left_spine.ids.get_unchecked(j[$lane]) };
                let left_rank_plus_one =
                    unsafe { *entries[$lane].left_spine.rank_plus_one.get_unchecked(j[$lane]) };
                debug_assert!((left_id as usize) < PIECE_COUNT);
                let cross_rank_plus_one =
                    unsafe { *(*cross_rank_row_ptrs.get_unchecked(i[$lane])).add(left_id as usize) };

                let reject_now = active
                    && cross_rank_plus_one != 0
                    && (right_rank_plus_one == 0 || cross_rank_plus_one <= right_rank_plus_one)
                    && (left_rank_plus_one == 0 || cross_rank_plus_one <= left_rank_plus_one);
                let step_now = can_step && active && !reject_now;
                let take_right = left_rank_plus_one == 0
                    || (right_rank_plus_one != 0
                        && right_rank_plus_one <= left_rank_plus_one);

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
    let right_rank_plus_one = &prepared_first.right_rank_plus_one;
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

                let right_rank_plus_one = unsafe { *right_rank_plus_one.get_unchecked(i[$lane]) };
                let left_id = unsafe { *entries[$lane].left_spine.ids.get_unchecked(j[$lane]) };
                let left_rank_plus_one =
                    unsafe { *entries[$lane].left_spine.rank_plus_one.get_unchecked(j[$lane]) };
                debug_assert!((left_id as usize) < PIECE_COUNT);
                let cross_rank_plus_one =
                    unsafe { *dense_matrix.get_unchecked(i[$lane] * PIECE_COUNT + left_id as usize) };

                let reject_now = active
                    && cross_rank_plus_one != 0
                    && (right_rank_plus_one == 0 || cross_rank_plus_one <= right_rank_plus_one)
                    && (left_rank_plus_one == 0 || cross_rank_plus_one <= left_rank_plus_one);
                let step_now = can_step && active && !reject_now;
                let take_right = left_rank_plus_one == 0
                    || (right_rank_plus_one != 0
                        && right_rank_plus_one <= left_rank_plus_one);

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
    let right_rank_plus_one = &prepared_first.right_rank_plus_one;
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

                let right_rank_plus_one = unsafe { *right_rank_plus_one.get_unchecked(i[$lane]) };
                let left_id = unsafe { *entries[$lane].left_spine.ids.get_unchecked(j[$lane]) };
                let left_rank_plus_one =
                    unsafe { *entries[$lane].left_spine.rank_plus_one.get_unchecked(j[$lane]) };
                debug_assert!((left_id as usize) < PIECE_COUNT);
                let cross_rank_plus_one = unsafe {
                    *dense_matrix.get_unchecked(left_id as usize * MAX_PACKED_SPINE_LEN + i[$lane])
                };

                let reject_now = active
                    && cross_rank_plus_one != 0
                    && (right_rank_plus_one == 0 || cross_rank_plus_one <= right_rank_plus_one)
                    && (left_rank_plus_one == 0 || cross_rank_plus_one <= left_rank_plus_one);
                let step_now = can_step && active && !reject_now;
                let take_right = left_rank_plus_one == 0
                    || (right_rank_plus_one != 0
                        && right_rank_plus_one <= left_rank_plus_one);

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
    let right_rank_plus_one = &prepared_first.right_rank_plus_one;
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

                let right_rank_plus_one = unsafe { *right_rank_plus_one.get_unchecked(i[$lane]) };
                let left_id = unsafe { *entries[$lane].left_spine.ids.get_unchecked(j[$lane]) };
                let left_rank_plus_one =
                    unsafe { *entries[$lane].left_spine.rank_plus_one.get_unchecked(j[$lane]) };
                debug_assert!((left_id as usize) < PIECE_COUNT);
                let cross_rank_plus_one =
                    unsafe { *dense_matrix.get_unchecked(left_id as usize * right_spine_len + i[$lane]) };

                let reject_now = active
                    && cross_rank_plus_one != 0
                    && (right_rank_plus_one == 0 || cross_rank_plus_one <= right_rank_plus_one)
                    && (left_rank_plus_one == 0 || cross_rank_plus_one <= left_rank_plus_one);
                let step_now = can_step && active && !reject_now;
                let take_right = left_rank_plus_one == 0
                    || (right_rank_plus_one != 0
                        && right_rank_plus_one <= left_rank_plus_one);

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
