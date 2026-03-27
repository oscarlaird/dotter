use super::{BpeError, BpeMerges, MAX_PACKED_SPINE_LEN, PackedSpine};

pub const MAX_PREPARED_DENSE_PIECE_COUNT: usize = (u16::MAX as usize) + 1;
static ZERO_CROSS_RANK_ROW: [u16; MAX_PREPARED_DENSE_PIECE_COUNT] = [0; MAX_PREPARED_DENSE_PIECE_COUNT];

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
pub struct PreparedFirstDenseContiguousSwappedTight<const PIECE_COUNT: usize> {
    right_len: u8,
    right_priority_score: [u16; MAX_PACKED_SPINE_LEN],
    dense_matrix: Box<[u16]>,
    dense_matrix_u32: Box<[u32]>,
}

#[derive(Debug, Clone)]
pub struct PreparedFirstDenseContiguousSwappedTightAllPairs<const PIECE_COUNT: usize> {
    right_len: u8,
    right_piece_formed_priority_score: [u16; MAX_PACKED_SPINE_LEN + 1],
    dense_matrix: Vec<u16>,
    row_partner_bitmap: Vec<u8>,
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
    pub piece_formed_priority_score: [u16; MAX_PACKED_SPINE_LEN + 1],
}

#[derive(Clone, Copy, Debug)]
pub struct PreparedSecondTokenAllPairs {
    pub token_id: u32,
    pub left_spine: CompactLeftSpineAllPairs,
}

pub type PreparedSecondBucketsAllPairs = [Vec<PreparedSecondTokenAllPairs>; MAX_PACKED_SPINE_LEN + 1];

#[derive(Clone, Debug)]
pub struct PreparedSecondSimd8Chunk {
    pub left_priority_scores_by_depth: [[u32; 8]; MAX_PACKED_SPINE_LEN],
    pub row_base_by_right_len_by_depth:
        [[[u32; 8]; MAX_PACKED_SPINE_LEN]; MAX_PACKED_SPINE_LEN + 1],
}

impl MergeRows {
    pub fn from_bpe_merges(merges: &BpeMerges) -> Result<Self, BpeError> {
        let piece_count = merges.pieces.len();
        let mut rows = vec![Vec::<RowEntry>::new(); piece_count];
        for entry in merges.merges.values() {
            let right = u16::try_from(entry.right)
                .map_err(|_| BpeError::UnsupportedPreparedDense("piece ids no longer fit in u16"))?;
            let priority_score = if entry.rank < u16::MAX as u32 {
                (u16::MAX as u32 - entry.rank) as u16
            } else {
                return Err(BpeError::UnsupportedPreparedDense("merge ranks no longer fit in u16"));
            };
            rows[entry.left as usize].push(RowEntry { right, priority_score });
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

    pub fn right_len(&self) -> usize {
        self.right_len as usize
    }

    pub fn dense_matrix(&self) -> &[u16] {
        &self.dense_matrix
    }
}

impl<const PIECE_COUNT: usize> PreparedFirstDenseContiguousSwappedTightAllPairs<PIECE_COUNT> {
    pub fn new_reusable() -> Self {
        Self {
            right_len: 0,
            right_piece_formed_priority_score: [0u16; MAX_PACKED_SPINE_LEN + 1],
            dense_matrix: Vec::new(),
            row_partner_bitmap: vec![0u8; PIECE_COUNT],
        }
    }

    pub fn rebuild_in_place(
        &mut self,
        first_right_spine: PackedSpine,
        merge_rows: &MergeRows,
    ) -> Result<(), BpeError> {
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
        self.row_partner_bitmap.fill(0);
        self.right_piece_formed_priority_score = [0u16; MAX_PACKED_SPINE_LEN + 1];

        let right_len = first_right_spine.as_slice().len();
        let needed = PIECE_COUNT * right_len;
        if self.dense_matrix.len() < needed {
            self.dense_matrix.resize(needed, 0);
        }

        for (spine_idx, spine_entry) in first_right_spine.as_slice().iter().enumerate() {
            let row = &merge_rows.rows[spine_entry.id as usize];
            let r_next = if spine_idx + 1 < right_len {
                spine_entry.priority_score
            } else {
                0
            };
            for entry in row {
                if entry.priority_score < r_next {
                    continue;
                }
                let idx = entry.right as usize * right_len + spine_idx;
                unsafe {
                    *self.dense_matrix.get_unchecked_mut(idx) = entry.priority_score;
                    *self
                        .row_partner_bitmap
                        .get_unchecked_mut(entry.right as usize) |= 1u8 << spine_idx;
                }
            }
        }
        if right_len > 0 {
            self.right_piece_formed_priority_score[0] = u16::MAX;
            for idx in 1..right_len {
                self.right_piece_formed_priority_score[idx] =
                    first_right_spine.as_slice()[idx - 1].priority_score;
            }
            self.right_piece_formed_priority_score[right_len] = 0;
        }
        self.right_len = right_len as u8;
        Ok(())
    }

    pub fn build(first_right_spine: PackedSpine, merge_rows: &MergeRows) -> Result<Self, BpeError> {
        let mut out = Self::new_reusable();
        out.rebuild_in_place(first_right_spine, merge_rows)?;
        Ok(out)
    }

    pub fn row_partner_bitmap(&self) -> &[u8] {
        &self.row_partner_bitmap
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
pub fn canonical_pair_from_prepared_first_dense_left_len<const PIECE_COUNT: usize, const LEFT_LEN: usize>(
    prepared_first: &PreparedFirstDense<PIECE_COUNT>,
    left_spine: &CompactLeftSpine,
) -> bool {
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
        let right_score = unsafe { *right_priority_score.get_unchecked(i) };
        let left_id = unsafe { *left_ids.get_unchecked(j) };
        let left_score = unsafe { *left_priority_score.get_unchecked(j) };
        let cross_row = unsafe { *cross_rank_row_ptrs.get_unchecked(i) };
        let cross_score = unsafe { *cross_row.add(left_id as usize) };
        let mut best = right_score;
        if left_score > best {
            best = left_score;
        }
        if cross_score > best {
            best = cross_score;
        }
        if best == 0 {
            return true;
        }
        if cross_score == best {
            return false;
        }
        if right_score == best {
            i += 1;
            if i >= right_spine_len {
                return true;
            }
        } else {
            j += 1;
            if j >= LEFT_LEN {
                return true;
            }
        }
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
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return false;
    }
    let right_len = prepared_first.right_len as usize;
    let right_priority_score = &prepared_first.right_priority_score;
    let dense_matrix = &prepared_first.dense_matrix;
    let mut i = 0usize;
    let mut j = 0usize;
    loop {
        let right_score = unsafe { *right_priority_score.get_unchecked(i) };
        let left_id = unsafe { *left_spine.ids.get_unchecked(j) };
        let left_score = unsafe { *left_spine.priority_score.get_unchecked(j) };
        let cross = unsafe { *dense_matrix.get_unchecked(left_id as usize * right_len + i) };
        let mut best = right_score;
        if left_score > best {
            best = left_score;
        }
        if cross > best {
            best = cross;
        }
        if best == 0 {
            return true;
        }
        if cross == best {
            return false;
        }
        if right_score == best {
            i += 1;
            if i >= right_len {
                return true;
            }
        } else {
            j += 1;
            if j >= LEFT_LEN {
                return true;
            }
        }
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
    if prepared_first.right_len == 0 || LEFT_LEN == 0 {
        return false;
    }
    let r_form = &prepared_first.right_piece_formed_priority_score;
    let dense_matrix = &prepared_first.dense_matrix;
    let row_partner_bitmap = &prepared_first.row_partner_bitmap;
    let l_form = &left_spine.piece_formed_priority_score;
    macro_rules! process_j {
        ($j:expr) => {{
            let left_id = unsafe { *left_spine.ids.get_unchecked($j) } as usize;
            let partner_bitmap = unsafe { *row_partner_bitmap.get_unchecked(left_id) };
            if partner_bitmap != 0 {
                let l_cur = unsafe { *l_form.get_unchecked($j) };
                let l_next = unsafe { *l_form.get_unchecked($j + 1) };
                let row_base = left_id * RIGHT_LEN;
                for i in 0..RIGHT_LEN {
                    if (partner_bitmap & (1u8 << i)) == 0 {
                        continue;
                    }
                    let r_cur = unsafe { *r_form.get_unchecked(i) };
                    let r_next = unsafe { *r_form.get_unchecked(i + 1) };
                    let exists_ij = (r_next < l_cur) && (l_next < r_cur);
                    if !exists_ij {
                        continue;
                    }
                    let c = unsafe { *dense_matrix.get_unchecked(row_base + i) };
                    if c >= l_next {
                        return false;
                    }
                }
            }
        }};
    }

    match LEFT_LEN {
        1 => {
            process_j!(0);
        }
        2 => {
            process_j!(0);
            process_j!(1);
        }
        3 => {
            process_j!(0);
            process_j!(1);
            process_j!(2);
        }
        4 => {
            process_j!(0);
            process_j!(1);
            process_j!(2);
            process_j!(3);
        }
        5 => {
            process_j!(0);
            process_j!(1);
            process_j!(2);
            process_j!(3);
            process_j!(4);
        }
        _ => {
            for j in 0..LEFT_LEN {
                process_j!(j);
            }
        }
    }
    true
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
                let is_canonical = canonical_pair_from_prepared_first_dense_left_len::<PIECE_COUNT, $left_len>(
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

fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken; 4],
) -> [bool; 4] {
    let right_len = prepared_first.right_len as usize;
    let right = &prepared_first.right_priority_score;
    let dense = &prepared_first.dense_matrix;
    let mut i = [0usize; 4];
    let mut j = [0usize; 4];
    let mut rejected = [false; 4];
    let stages = right_len + LEFT_LEN - 1;
    for stage in 0..stages {
        let can_step = stage + 1 < stages;
        for lane in 0..4 {
            if rejected[lane] {
                continue;
            }
            let r = unsafe { *right.get_unchecked(i[lane]) };
            let lid = unsafe { *entries[lane].left_spine.ids.get_unchecked(j[lane]) };
            let l = unsafe { *entries[lane].left_spine.priority_score.get_unchecked(j[lane]) };
            let c = unsafe { *dense.get_unchecked(lid as usize * right_len + i[lane]) };
            let reject = c != 0 && c >= r && c >= l;
            if reject {
                rejected[lane] = true;
                continue;
            }
            if can_step {
                if r >= l {
                    i[lane] += 1;
                } else {
                    j[lane] += 1;
                }
            }
        }
    }
    [!rejected[0], !rejected[1], !rejected[2], !rejected[3]]
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_lockstep4<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut count = 0u64;
    let mut chunks = entries.chunks_exact(4);
    for chunk in &mut chunks {
        let lanes: &[PreparedSecondToken; 4] = chunk.try_into().expect("4-lane chunk");
        let out = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, lanes);
        count += out.iter().map(|&v| v as u64).sum::<u64>();
    }
    for entry in chunks.remainder() {
        count += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    count
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
        let lanes: &[PreparedSecondToken; 4] = chunk.try_into().expect("4-lane chunk");
        let out = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_lockstep4_chunk::<
            PIECE_COUNT,
            LEFT_LEN,
        >(prepared_first, lanes);
        for lane in 0..4 {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &lanes[lane].left_spine);
            if scalar != out[lane] {
                mismatches += 1;
            }
        }
    }
    mismatches
}

pub fn build_prepared_second_simd8_chunks<const LEFT_LEN: usize>(
    entries: &[PreparedSecondToken],
) -> Vec<PreparedSecondSimd8Chunk> {
    let chunk_count = entries.len() / 8;
    let mut chunks = Vec::with_capacity(chunk_count);
    for chunk_idx in 0..chunk_count {
        let base = chunk_idx * 8;
        let lanes: &[PreparedSecondToken; 8] = entries[base..base + 8].try_into().expect("8 lanes");
        let mut left_priority_scores_by_depth = [[0u32; 8]; MAX_PACKED_SPINE_LEN];
        let mut row_base_by_right_len_by_depth =
            [[[0u32; 8]; MAX_PACKED_SPINE_LEN]; MAX_PACKED_SPINE_LEN + 1];
        for lane in 0..8 {
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd8_chunk_avx2<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTight<PIECE_COUNT>,
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

    let right_len = prepared_first.right_len as usize;
    if right_len == 0 || LEFT_LEN == 0 {
        return [false; 8];
    }
    let stages = right_len + LEFT_LEN - 1;
    let mut right_arr = [0i32; MAX_PACKED_SPINE_LEN];
    for i in 0..right_len {
        right_arr[i] = prepared_first.right_priority_score[i] as i32;
    }
    let right_v_table =
        unsafe { std::mem::transmute::<[i32; MAX_PACKED_SPINE_LEN], __m256i>(right_arr) };
    let dense_u32_ptr = prepared_first.dense_matrix_u32.as_ptr() as *const i32;
    let zero_v = _mm256_setzero_si256();
    let ones_v = _mm256_cmpeq_epi32(zero_v, zero_v);
    let one_v = _mm256_set1_epi32(1);
    let mut i_v = zero_v;
    let mut j_v = zero_v;
    let mut rejected_v = zero_v;
    let mut left_by_depth = [zero_v; MAX_PACKED_SPINE_LEN];
    let mut row_by_depth = [zero_v; MAX_PACKED_SPINE_LEN];
    for d in 0..LEFT_LEN {
        let mut la = [0i32; 8];
        let mut ra = [0i32; 8];
        for lane in 0..8 {
            la[lane] = simd_chunk.left_priority_scores_by_depth[d][lane] as i32;
            ra[lane] = simd_chunk.row_base_by_right_len_by_depth[right_len][d][lane] as i32;
        }
        left_by_depth[d] = unsafe { std::mem::transmute::<[i32; 8], __m256i>(la) };
        row_by_depth[d] = unsafe { std::mem::transmute::<[i32; 8], __m256i>(ra) };
    }
    let depth_consts = [
        _mm256_set1_epi32(0),
        _mm256_set1_epi32(1),
        _mm256_set1_epi32(2),
        _mm256_set1_epi32(3),
        _mm256_set1_epi32(4),
        _mm256_set1_epi32(5),
        _mm256_set1_epi32(6),
        _mm256_set1_epi32(7),
    ];
    for stage in 0..stages {
        let can_step = stage + 1 < stages;
        let active_v = _mm256_cmpeq_epi32(rejected_v, zero_v);
        let right_v = _mm256_permutevar8x32_epi32(right_v_table, i_v);
        let mut left_v = zero_v;
        let mut row_v = zero_v;
        for d in 0..LEFT_LEN {
            let m = _mm256_cmpeq_epi32(j_v, depth_consts[d]);
            left_v = _mm256_or_si256(left_v, _mm256_and_si256(m, left_by_depth[d]));
            row_v = _mm256_or_si256(row_v, _mm256_and_si256(m, row_by_depth[d]));
        }
        let idx_v = _mm256_add_epi32(row_v, i_v);
        let cross_v = unsafe { _mm256_i32gather_epi32(dense_u32_ptr, idx_v, 4) };
        let cross_nonzero = _mm256_cmpgt_epi32(cross_v, zero_v);
        let cross_ge_r = ge_epi32(cross_v, right_v);
        let cross_ge_l = ge_epi32(cross_v, left_v);
        let reject_mask =
            _mm256_and_si256(_mm256_and_si256(_mm256_and_si256(active_v, cross_nonzero), cross_ge_r), cross_ge_l);
        let can_step_mask = if can_step { ones_v } else { zero_v };
        let step_mask = _mm256_and_si256(can_step_mask, _mm256_andnot_si256(reject_mask, active_v));
        let take_right_mask = ge_epi32(right_v, left_v);
        let i_inc = _mm256_and_si256(step_mask, take_right_mask);
        let j_inc = _mm256_andnot_si256(take_right_mask, step_mask);
        i_v = _mm256_add_epi32(i_v, _mm256_and_si256(i_inc, one_v));
        j_v = _mm256_add_epi32(j_v, _mm256_and_si256(j_inc, one_v));
        rejected_v = _mm256_or_si256(rejected_v, reject_mask);
    }
    let rejected_bits = _mm256_movemask_ps(_mm256_castsi256_ps(rejected_v)) as u32;
    let accepted = (!rejected_bits) & 0xFF;
    [
        (accepted & (1 << 0)) != 0,
        (accepted & (1 << 1)) != 0,
        (accepted & (1 << 2)) != 0,
        (accepted & (1 << 3)) != 0,
        (accepted & (1 << 4)) != 0,
        (accepted & (1 << 5)) != 0,
        (accepted & (1 << 6)) != 0,
        (accepted & (1 << 7)) != 0,
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
                >(prepared_first, simd_chunk)
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
        let lanes: &[PreparedSecondToken; 8] = chunk.try_into().expect("8-lane");
        let simd = unsafe { simd_chunks.get_unchecked(chunk_idx) };
        let out =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd8_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, lanes, simd);
        canonical_count += out.iter().map(|&v| v as u64).sum::<u64>();
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
        let lanes: &[PreparedSecondToken; 8] = chunk.try_into().expect("8-lane");
        let simd = unsafe { simd_chunks.get_unchecked(chunk_idx) };
        let out =
            canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_simd8_chunk::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, lanes, simd);
        for lane in 0..8 {
            let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
                PIECE_COUNT,
                LEFT_LEN,
            >(prepared_first, &lanes[lane].left_spine);
            if scalar != out[lane] {
                mismatches += 1;
            }
        }
    }
    mismatches
}

pub fn scan_prepared_first_dense_contiguous_swapped_tight_bucket_allpairs_small<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
    const RIGHT_LEN: usize,
>(
    prepared_first: &PreparedFirstDenseContiguousSwappedTightAllPairs<PIECE_COUNT>,
    entries: &[PreparedSecondTokenAllPairs],
) -> u64 {
    let mut canonical = 0u64;
    for entry in entries {
        canonical += canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_right_len_allpairs_small::<
            PIECE_COUNT,
            LEFT_LEN,
            RIGHT_LEN,
        >(prepared_first, &entry.left_spine) as u64;
    }
    canonical
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
    let mut mismatches = 0u64;
    for entry in entries {
        let allpairs = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len_right_len_allpairs_small::<
            PIECE_COUNT,
            LEFT_LEN,
            RIGHT_LEN,
        >(prepared_first, &entry.left_spine);
        let mut next = [0u16; MAX_PACKED_SPINE_LEN];
        for idx in 0..LEFT_LEN {
            next[idx] = entry.left_spine.piece_formed_priority_score[idx + 1];
        }
        let scalar = canonical_pair_from_prepared_first_dense_contiguous_swapped_tight_left_len::<
            PIECE_COUNT,
            LEFT_LEN,
        >(
            prepared_first_reference,
            &CompactLeftSpine {
                len: entry.left_spine.len,
                ids: entry.left_spine.ids,
                priority_score: next,
            },
        );
        if allpairs != scalar {
            mismatches += 1;
        }
    }
    mismatches
}
