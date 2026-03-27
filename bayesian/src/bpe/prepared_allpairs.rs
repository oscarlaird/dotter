use super::{BpeMerges, MAX_PACKED_SPINE_LEN, PackedSpine};

const MAX_PREPARED_DENSE_PIECE_COUNT: usize = (u16::MAX as usize) + 1;

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
pub struct PreparedFirstAllPairs<const PIECE_COUNT: usize> {
    right_len: u8,
    right_piece_formed_priority_score: [u16; MAX_PACKED_SPINE_LEN + 1],
    dense_matrix: Vec<u16>,
    row_partner_bitmap: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub struct LeftSpineAllPairs {
    pub len: u8,
    pub ids: [u16; MAX_PACKED_SPINE_LEN],
    pub piece_formed_priority_score: [u16; MAX_PACKED_SPINE_LEN + 1],
}

#[derive(Clone, Copy, Debug)]
pub struct PreparedSecondToken {
    pub lex_index: usize,
    pub left_spine: LeftSpineAllPairs,
}

pub type PreparedSecondBuckets = [Vec<PreparedSecondToken>; MAX_PACKED_SPINE_LEN + 1];

impl MergeRows {
    pub fn from_bpe_merges(merges: &BpeMerges) -> Self {
        let piece_count = merges.pieces.len();
        let mut rows = vec![Vec::<RowEntry>::new(); piece_count];
        for entry in merges.merges.values() {
            let right =
                u16::try_from(entry.right).expect("piece ids no longer fit in u16 for allpairs");
            let priority_score = if entry.rank < u16::MAX as u32 {
                (u16::MAX as u32 - entry.rank) as u16
            } else {
                panic!("merge ranks no longer fit in u16 for allpairs");
            };
            rows[entry.left as usize].push(RowEntry {
                right,
                priority_score,
            });
        }
        Self { rows, piece_count }
    }

    pub fn piece_count(&self) -> usize {
        self.piece_count
    }
}

impl<const PIECE_COUNT: usize> PreparedFirstAllPairs<PIECE_COUNT> {
    pub fn new_reusable() -> Self {
        Self {
            right_len: 0,
            right_piece_formed_priority_score: [0u16; MAX_PACKED_SPINE_LEN + 1],
            dense_matrix: Vec::new(),
            row_partner_bitmap: vec![0u8; PIECE_COUNT],
        }
    }

    pub fn rebuild_in_place(&mut self, first_right_spine: PackedSpine, merge_rows: &MergeRows) {
        assert!(
            PIECE_COUNT <= MAX_PREPARED_DENSE_PIECE_COUNT,
            "prepared dense table exceeds the maximum supported piece-id width"
        );
        assert!(
            merge_rows.piece_count <= PIECE_COUNT,
            "prepared dense fast path expects piece ids to fit within the configured table"
        );
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
    }

    pub fn build(first_right_spine: PackedSpine, merge_rows: &MergeRows) -> Self {
        let mut out = Self::new_reusable();
        out.rebuild_in_place(first_right_spine, merge_rows);
        out
    }

    pub fn row_partner_bitmap(&self) -> &[u8] {
        &self.row_partner_bitmap
    }
}

impl LeftSpineAllPairs {
    pub fn from_packed(packed: PackedSpine) -> Self {
        let mut compact = Self {
            len: 0,
            ids: [0; MAX_PACKED_SPINE_LEN],
            piece_formed_priority_score: [0; MAX_PACKED_SPINE_LEN + 1],
        };
        let entries = packed.as_slice();
        compact.len = entries.len() as u8;
        if !entries.is_empty() {
            compact.piece_formed_priority_score[0] = u16::MAX;
        }
        for (idx, entry) in entries.iter().enumerate() {
            compact.ids[idx] = entry.id;
            if idx + 1 < entries.len() {
                compact.piece_formed_priority_score[idx + 1] = entry.priority_score;
            }
        }
        compact.piece_formed_priority_score[entries.len()] = 0;
        compact
    }
}

pub fn sort_prepared_second_tokens(entries: &mut [PreparedSecondToken]) {
    entries.sort_by(|a, b| {
        a.left_spine.ids[..a.left_spine.len as usize]
            .cmp(&b.left_spine.ids[..b.left_spine.len as usize])
            .then_with(|| a.lex_index.cmp(&b.lex_index))
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
pub fn is_canonical_allpairs_small<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
    const RIGHT_LEN: usize,
>(
    prepared_first: &PreparedFirstAllPairs<PIECE_COUNT>,
    left_spine: &LeftSpineAllPairs,
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
                    let bit = 1u8 << i;
                    if (partner_bitmap & bit) == 0 {
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

pub fn scan_allpairs_small_bucket<
    const PIECE_COUNT: usize,
    const LEFT_LEN: usize,
    const RIGHT_LEN: usize,
>(
    prepared_first: &PreparedFirstAllPairs<PIECE_COUNT>,
    entries: &[PreparedSecondToken],
) -> u64 {
    let mut canonical = 0u64;
    for entry in entries {
        canonical += is_canonical_allpairs_small::<PIECE_COUNT, LEFT_LEN, RIGHT_LEN>(
            prepared_first,
            &entry.left_spine,
        ) as u64;
    }
    canonical
}
