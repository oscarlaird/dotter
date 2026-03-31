use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::prepared_allpairs::{
    self, MergeRows, PreparedFirstAllPairs, PreparedSecondBuckets, PreparedSecondToken,
};
use super::tokenizer_config::NUM_TOKENS;
use super::{
    BpeMerges, MAX_PACKED_SPINE_LEN, NO_PACKED_SPINE_INDEX, PackedSpine, SPACESYMBOL, SpineEntry,
};

pub type TinyLlamaPreparedFirstAllPairs = PreparedFirstAllPairs<NUM_TOKENS>;

fn token_prefixes(token: &str) -> Vec<&str> {
    let mut prefixes = Vec::with_capacity(token.chars().count() + 1);
    prefixes.push("");
    for (idx, _) in token.char_indices().skip(1) {
        prefixes.push(&token[..idx]);
    }
    if !token.is_empty() {
        prefixes.push(token);
    }
    prefixes
}

fn build_second_buckets(
    lex_indices_with_left_spines: &[usize],
    left_spine_index_by_lex_index: &[u16],
    packed_left_spines: &[PackedSpine],
    lex_tokens: &[String],
) -> PreparedSecondBuckets {
    let mut entries = Vec::with_capacity(lex_indices_with_left_spines.len());
    for &lex_index in lex_indices_with_left_spines {
        if lex_tokens[lex_index].starts_with(SPACESYMBOL) {
            continue;
        }
        let spine_index = left_spine_index_by_lex_index[lex_index] as usize;
        entries.push(PreparedSecondToken {
            lex_index,
            left_spine: prepared_allpairs::LeftSpineAllPairs::from_packed(
                packed_left_spines[spine_index],
            ),
        });
    }
    prepared_allpairs::sort_prepared_second_tokens(&mut entries);
    prepared_allpairs::bucket_prepared_second_tokens(&entries)
}

/// TinyLlama-compatible encoding for a **single word**: no whitespace in `text`.
/// Matches `transformers.AutoTokenizer.encode(text, add_special_tokens=False)` for such strings.
#[derive(Debug, Clone)]
pub struct TinyLlamaWordTokenizer {
    merges: BpeMerges,
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    piece_id_to_vocab_id: Vec<Option<u32>>,
    lex_tokens: Vec<String>,
    token_to_lex_index: HashMap<String, usize>,
    lex_prefixes: Vec<String>,
    prefix_to_lex_index: HashMap<String, usize>,
    prefix_token_starts: Vec<usize>,
    prefix_token_stops: Vec<usize>,
    lex_index_to_token_id: Vec<u32>,
    token_id_to_lex_index: Vec<Option<usize>>,
    left_spine_index_by_lex_index: Vec<u16>,
    packed_left_spines: Vec<PackedSpine>,
    lex_indices_with_left_spines: Vec<usize>,
    space_prefixed_second_token_mask: Box<[bool]>,
    prepared_merge_rows: MergeRows,
    prepared_second_buckets: PreparedSecondBuckets,
}

impl TinyLlamaWordTokenizer {
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Self {
        let text = fs::read_to_string(path.as_ref()).expect("failed to read tokenizer.json");
        Self::from_tokenizer_json_str(&text)
    }

    pub fn from_tokenizer_json_str(content: &str) -> Self {
        let merges = BpeMerges::from_tokenizer_json_str(content);
        let v: serde_json::Value = serde_json::from_str(content).expect("invalid tokenizer.json");
        let vocab_obj = v
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|m| m.as_object())
            .expect("missing model.vocab object");

        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut entries = Vec::with_capacity(vocab_obj.len());
        let mut max_id = 0usize;
        for (token, id_val) in vocab_obj {
            let id = id_val
                .as_u64()
                .or_else(|| id_val.as_i64().map(|i| i as u64))
                .expect("vocab id is not an integer");
            if id > u32::MAX as u64 {
                panic!("vocab id does not fit u32");
            }
            let id = id as u32;
            max_id = max_id.max(id as usize);
            vocab.insert(token.clone(), id);
            entries.push((token.as_str(), id));
        }

        let mut id_to_token = vec![String::new(); max_id + 1];
        let mut seen = vec![false; max_id + 1];
        for (token, id) in entries {
            let slot = &mut seen[id as usize];
            if *slot {
                panic!("duplicate vocab id");
            }
            *slot = true;
            id_to_token[id as usize] = token.to_string();
        }
        if seen.iter().any(|present| !present) {
            panic!("vocab ids must be dense from 0..max");
        }

        let mut piece_id_to_vocab_id = vec![None; merges.pieces.len()];
        for (token, &id) in &vocab {
            if let Some(piece_id) = merges.encode_piece(token) {
                piece_id_to_vocab_id[piece_id as usize] = Some(id);
            }
        }

        let mut lex_tokens = id_to_token.clone();
        lex_tokens.sort_unstable();
        let mut token_to_lex_index = HashMap::with_capacity(lex_tokens.len());
        let mut lex_index_to_token_id = Vec::with_capacity(lex_tokens.len());
        let mut token_id_to_lex_index = vec![None; id_to_token.len()];
        for (lex_index, token) in lex_tokens.iter().enumerate() {
            let token_id = *vocab
                .get(token)
                .expect("every lexicographic token must exist in the vocab");
            token_to_lex_index.insert(token.clone(), lex_index);
            lex_index_to_token_id.push(token_id);
            token_id_to_lex_index[token_id as usize] = Some(lex_index);
        }

        let mut lex_prefixes = Vec::new();
        for token in &lex_tokens {
            for prefix in token_prefixes(token) {
                lex_prefixes.push(prefix.to_string());
            }
        }
        lex_prefixes.sort_unstable();
        lex_prefixes.dedup();
        let mut prefix_to_lex_index = HashMap::with_capacity(lex_prefixes.len());
        for (lex_index, prefix) in lex_prefixes.iter().enumerate() {
            prefix_to_lex_index.insert(prefix.clone(), lex_index);
        }
        let mut prefix_token_starts = Vec::with_capacity(lex_prefixes.len());
        let mut prefix_token_stops = Vec::with_capacity(lex_prefixes.len());
        for prefix in &lex_prefixes {
            let start = lex_tokens.partition_point(|token| token.as_str() < prefix.as_str());
            let stop =
                start + lex_tokens[start..].partition_point(|token| token.starts_with(prefix));
            prefix_token_starts.push(start);
            prefix_token_stops.push(stop);
        }

        let mut left_spine_index_by_lex_index = vec![NO_PACKED_SPINE_INDEX; lex_tokens.len()];
        let mut packed_left_spines = Vec::new();
        let mut lex_indices_with_left_spines = Vec::new();
        for (token, &id) in &vocab {
            if let Some(left_spine) = merges.left_spine(token) {
                if let Some(packed_left_spine) = PackedSpine::from_entries(&left_spine) {
                    let spine_index = u16::try_from(packed_left_spines.len())
                        .expect("TinyLlama spine count fits in u16");
                    let lex_index = token_id_to_lex_index[id as usize]
                        .expect("every vocab token id must have a lex index");
                    lex_indices_with_left_spines.push(lex_index);
                    left_spine_index_by_lex_index[lex_index] = spine_index;
                    packed_left_spines.push(packed_left_spine);
                }
            }
        }
        lex_indices_with_left_spines.sort_unstable();
        let space_prefixed_second_token_mask = lex_tokens
            .iter()
            .map(|token| token.starts_with(SPACESYMBOL))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let prepared_merge_rows = MergeRows::from_bpe_merges(&merges);
        let prepared_second_buckets = build_second_buckets(
            &lex_indices_with_left_spines,
            &left_spine_index_by_lex_index,
            &packed_left_spines,
            &lex_tokens,
        );

        Self {
            merges,
            vocab,
            id_to_token,
            piece_id_to_vocab_id,
            lex_tokens,
            token_to_lex_index,
            lex_prefixes,
            prefix_to_lex_index,
            prefix_token_starts,
            prefix_token_stops,
            lex_index_to_token_id,
            token_id_to_lex_index,
            left_spine_index_by_lex_index,
            packed_left_spines,
            lex_indices_with_left_spines,
            space_prefixed_second_token_mask,
            prepared_merge_rows,
            prepared_second_buckets,
        }
    }

    fn seed_space_prefixed_second_tokens(&self, out: &mut [bool]) {
        assert_eq!(
            out.len(),
            self.space_prefixed_second_token_mask.len(),
            "psi output len must match vocab len",
        );
        out.copy_from_slice(&self.space_prefixed_second_token_mask);
    }

    fn tokenize_string_piece_ids(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        for c in text.chars() {
            assert!(
                self.merges.piece_id_char(c).is_some(),
                "piece not in vocab: {:?}",
                c.to_string()
            );
        }
        self.merges
            .tokenize_ids(text)
            .expect("string must tokenize entirely into known BPE pieces")
    }

    fn piece_ids_to_lex_indices(&self, piece_ids: &[u32]) -> Vec<usize> {
        let mut lex_indices = Vec::with_capacity(piece_ids.len());
        for &piece_id in piece_ids {
            let vocab_id = self.piece_id_to_vocab_id[piece_id as usize]
                .expect("piece must correspond to a vocab token");
            let lex_index = self.token_id_to_lex_index[vocab_id as usize]
                .expect("vocab token id must correspond to a lex index");
            lex_indices.push(lex_index);
        }
        lex_indices
    }

    fn piece_ids_with_lex_indices(&self, piece_ids: &[u32]) -> Vec<(String, usize)> {
        let mut out = Vec::with_capacity(piece_ids.len());
        for &piece_id in piece_ids {
            let piece = self
                .merges
                .decode_piece(piece_id)
                .expect("piece ids returned by tokenizer must decode");
            let vocab_id = self.piece_id_to_vocab_id[piece_id as usize]
                .expect("piece must correspond to a vocab token");
            let lex_index = self.token_id_to_lex_index[vocab_id as usize]
                .expect("vocab token id must correspond to a lex index");
            out.push((piece.to_string(), lex_index));
        }
        out
    }

    /// Tokens in ordinary lexicographic order.
    pub fn tokens(&self) -> &[String] {
        &self.lex_tokens
    }

    pub fn prefix_count(&self) -> usize {
        self.lex_prefixes.len()
    }

    pub fn prefixes(&self) -> &[String] {
        &self.lex_prefixes
    }

    /// The lexicographic index of `token`, if it is present in the tokenizer vocabulary.
    pub fn lex_index(&self, token: &str) -> Option<usize> {
        self.token_to_lex_index.get(token).copied()
    }

    pub fn prefix_lex_index(&self, prefix: &str) -> Option<usize> {
        self.prefix_to_lex_index.get(prefix).copied()
    }

    /// The token stored at `lex_index`.
    pub fn token_at(&self, lex_index: usize) -> &str {
        self.lex_tokens
            .get(lex_index)
            .map(String::as_str)
            .expect("lex index out of range")
    }

    pub fn prefix_at(&self, lex_index: usize) -> &str {
        self.lex_prefixes
            .get(lex_index)
            .map(String::as_str)
            .expect("prefix lex index out of range")
    }

    pub fn prefix_token_starts(&self) -> &[usize] {
        &self.prefix_token_starts
    }

    pub fn prefix_token_stops(&self) -> &[usize] {
        &self.prefix_token_stops
    }

    fn token_id_for_lex_index(&self, lex_index: usize) -> Option<u32> {
        self.lex_index_to_token_id.get(lex_index).copied()
    }

    fn token_for_id(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    /// True iff `prefix` is a prefix of some token in lexicographic order.
    pub fn has_token_with_prefix(&self, prefix: &str) -> bool {
        self.prefix_to_lex_index.contains_key(prefix)
    }

    /// True iff `prefix` is a strict prefix of some token in lexicographic order.
    pub fn has_token_with_strict_prefix(&self, prefix: &str) -> bool {
        let Some(prefix_lex_index) = self.prefix_lex_index(prefix) else {
            return false;
        };
        let (start, stop) = self.token_lex_range_for_prefix_index(prefix_lex_index);
        self.lex_tokens[start..stop]
            .iter()
            .any(|token| token.len() > prefix.len())
    }

    /// The half-open lexicographic token range whose tokens start with `prefix`.
    pub fn token_lex_range_for_prefix(&self, prefix: &str) -> (usize, usize) {
        let prefix_lex_index = self
            .prefix_lex_index(prefix)
            .expect("prefix must be present in tokenizer prefix set");
        self.token_lex_range_for_prefix_index(prefix_lex_index)
    }

    pub fn token_lex_range_for_prefix_index(&self, prefix_lex_index: usize) -> (usize, usize) {
        let start = *self
            .prefix_token_starts
            .get(prefix_lex_index)
            .expect("prefix lex index out of range");
        let stop = *self
            .prefix_token_stops
            .get(prefix_lex_index)
            .expect("prefix lex index out of range");
        (start, stop)
    }

    pub fn count_true_tokens_by_prefix<const PREFIX_COUNT: usize>(
        &self,
        token_flags: &[bool],
    ) -> [u32; PREFIX_COUNT] {
        assert_eq!(
            token_flags.len(),
            self.lex_tokens.len(),
            "token flag len must match vocab len"
        );
        assert_eq!(
            self.lex_prefixes.len(),
            PREFIX_COUNT,
            "prefix count must match PREFIX_COUNT"
        );
        let mut token_true_cumsum = vec![0u32; self.lex_tokens.len() + 1];
        for (idx, is_true) in token_flags.iter().copied().enumerate() {
            token_true_cumsum[idx + 1] = token_true_cumsum[idx] + u32::from(is_true);
        }
        std::array::from_fn(|prefix_lex_index| {
            let start = self.prefix_token_starts[prefix_lex_index];
            let stop = self.prefix_token_stops[prefix_lex_index];
            token_true_cumsum[stop] - token_true_cumsum[start]
        })
    }

    fn right_spine(&self, token: &str) -> Option<Vec<SpineEntry>> {
        self.merges.right_spine(token)
    }

    fn right_packed_spine(&self, token: &str) -> Option<PackedSpine> {
        PackedSpine::from_entries(&self.right_spine(token)?)
    }

    #[doc(hidden)]
    pub fn right_packed_spine_for_lex_index(&self, lex_index: usize) -> PackedSpine {
        let token = self.token_at(lex_index);
        self.right_packed_spine(token)
            .expect("token must have a packed right spine")
    }

    fn left_spine_for_lex_index(&self, lex_index: usize) -> Option<&[SpineEntry]> {
        self.left_packed_spine_for_lex_index(lex_index)
            .map(PackedSpine::as_slice)
    }

    fn left_packed_spine_for_lex_index(&self, lex_index: usize) -> Option<&PackedSpine> {
        let spine_index = *self.left_spine_index_by_lex_index.get(lex_index)?;
        if spine_index == NO_PACKED_SPINE_INDEX {
            None
        } else {
            self.packed_left_spines.get(spine_index as usize)
        }
    }

    /// Lexicographic token indices for which a left spine was successfully precomputed.
    pub fn lex_indices_with_left_spines(&self) -> &[usize] {
        &self.lex_indices_with_left_spines
    }

    #[doc(hidden)]
    pub fn prepare_canonical_pair_batch_for_lex_index(
        &self,
        first_lex_index: usize,
    ) -> TinyLlamaPreparedFirstAllPairs {
        let first_token_right_spine = self.right_packed_spine_for_lex_index(first_lex_index);
        PreparedFirstAllPairs::<NUM_TOKENS>::build(
            first_token_right_spine,
            &self.prepared_merge_rows,
        )
    }

    #[doc(hidden)]
    pub fn prepared_second_buckets(&self) -> &PreparedSecondBuckets {
        &self.prepared_second_buckets
    }

    #[doc(hidden)]
    pub fn prepared_merge_rows(&self) -> &MergeRows {
        &self.prepared_merge_rows
    }

    #[doc(hidden)]
    pub fn canonical_pair_batch_with_first_token_right_spine_into(
        &self,
        first_token_right_spine: &PackedSpine,
        out: &mut [bool],
    ) {
        assert_eq!(
            out.len(),
            self.lex_tokens.len(),
            "canonical pair output len must match vocab len"
        );
        self.seed_space_prefixed_second_tokens(out);
        let prepared_first = PreparedFirstAllPairs::<NUM_TOKENS>::build(
            *first_token_right_spine,
            &self.prepared_merge_rows,
        );
        macro_rules! fill_bucket {
            ($left_len:literal) => {
                for entry in &self.prepared_second_buckets[$left_len] {
                    let is_canonical = prepared_allpairs::is_canonical_allpairs_small::<
                        NUM_TOKENS,
                        $left_len,
                        MAX_PACKED_SPINE_LEN,
                    >(&prepared_first, &entry.left_spine);
                    out[entry.lex_index] = is_canonical;
                }
            };
        }
        // Fall back to exact right-len dispatch to preserve allpairs specialization behavior.
        let right_len = first_token_right_spine.as_slice().len();
        macro_rules! by_left {
            ($right_len:literal, $left_len:literal) => {
                for entry in &self.prepared_second_buckets[$left_len] {
                    let is_canonical = prepared_allpairs::is_canonical_allpairs_small::<
                        NUM_TOKENS,
                        $left_len,
                        $right_len,
                    >(&prepared_first, &entry.left_spine);
                    out[entry.lex_index] = is_canonical;
                }
            };
        }
        macro_rules! dispatch_right {
            ($left_len:literal) => {
                match right_len {
                    1 => by_left!(1, $left_len),
                    2 => by_left!(2, $left_len),
                    3 => by_left!(3, $left_len),
                    4 => by_left!(4, $left_len),
                    5 => by_left!(5, $left_len),
                    6 => by_left!(6, $left_len),
                    7 => by_left!(7, $left_len),
                    8 => by_left!(8, $left_len),
                    _ => fill_bucket!($left_len),
                }
            };
        }
        dispatch_right!(1);
        dispatch_right!(2);
        dispatch_right!(3);
        dispatch_right!(4);
        dispatch_right!(5);
        dispatch_right!(6);
        dispatch_right!(7);
        dispatch_right!(8);
    }

    #[doc(hidden)]
    pub fn canonical_pair_batch_with_first_token_right_spine(
        &self,
        first_token_right_spine: &PackedSpine,
    ) -> [bool; NUM_TOKENS] {
        let mut out = [false; NUM_TOKENS];
        self.canonical_pair_batch_with_first_token_right_spine_into(
            first_token_right_spine,
            &mut out,
        );
        out
    }

    pub fn canonical_followers_for_lex_index(&self, first_lex_index: usize) -> [bool; NUM_TOKENS] {
        let first_token = self.token_at(first_lex_index);
        if let Some(first_token_right_spine) = self.right_packed_spine(first_token) {
            return self
                .canonical_pair_batch_with_first_token_right_spine(&first_token_right_spine);
        }
        let mut out = [false; NUM_TOKENS];
        self.seed_space_prefixed_second_tokens(&mut out);
        for (second_lex_index, second_token) in self.lex_tokens.iter().enumerate() {
            if out[second_lex_index] {
                continue;
            }
            out[second_lex_index] = self.can_canonically_follow(first_token, second_token);
        }
        out
    }

    /// Decide canonicality for many second tokens after computing the first token's right spine
    /// once.
    fn canonical_pair_with_first_token_right_spine_and_lex_index(
        &self,
        first_token_right_spine: &[SpineEntry],
        second_lex_index: usize,
    ) -> Option<bool> {
        let second_token_left_spine = self.left_spine_for_lex_index(second_lex_index)?;
        Some(
            self.merges
                .canonical_pair_from_spines(first_token_right_spine, second_token_left_spine),
        )
    }

    fn canonical_pair_with_first_token_right_packed_spine_and_lex_index(
        &self,
        first_token_right_spine: &PackedSpine,
        second_lex_index: usize,
    ) -> Option<bool> {
        let second_token_left_spine = self.left_packed_spine_for_lex_index(second_lex_index)?;
        Some(
            self.merges.canonical_pair_from_packed_spines(
                first_token_right_spine,
                second_token_left_spine,
            ),
        )
    }

    fn canonical_pair_from_first_token_right_spine_and_second_token_left_spine(
        &self,
        first_token_right_spine: &PackedSpine,
        second_token_left_spine: &PackedSpine,
    ) -> bool {
        self.merges
            .canonical_pair_from_packed_spines(first_token_right_spine, second_token_left_spine)
    }

    fn canonical_pair_with_first_token_right_spine(
        &self,
        first_token_right_spine: &[SpineEntry],
        second_token: &str,
    ) -> Option<bool> {
        let second_lex_index = self.lex_index(second_token)?;
        self.canonical_pair_with_first_token_right_spine_and_lex_index(
            first_token_right_spine,
            second_lex_index,
        )
    }

    /// Returns true exactly when raw BPE tokenization of `a + b` is `[a, b]`.
    pub fn can_canonically_follow(&self, a: &str, b: &str) -> bool {
        let _ = self
            .lex_index(a)
            .expect("first token must be present in tokenizer vocabulary");
        let second_lex_index = self
            .lex_index(b)
            .expect("second token must be present in tokenizer vocabulary");
        let Some(first_token_right_spine) = self.right_packed_spine(a) else {
            return self.merges.canonical_pair(a, b);
        };
        self.canonical_pair_with_first_token_right_packed_spine_and_lex_index(
            &first_token_right_spine,
            second_lex_index,
        )
        .unwrap_or_else(|| self.merges.canonical_pair(a, b))
    }

    pub fn canonical_followers(&self, token: &str) -> [bool; NUM_TOKENS] {
        let first_lex_index = self
            .lex_index(token)
            .expect("token must be present in tokenizer vocabulary");
        self.canonical_followers_for_lex_index(first_lex_index)
    }

    /// Token lex indices for the exact input string.
    pub fn tokenize_string_to_lex_indices(&self, text: &str) -> Vec<usize> {
        let piece_ids = self.tokenize_string_piece_ids(text);
        self.piece_ids_to_lex_indices(&piece_ids)
    }

    /// TinyLlama pieces together with their lex indices for the exact input string.
    pub fn tokenize_string_with_lex_indices(&self, text: &str) -> Vec<(String, usize)> {
        let piece_ids = self.tokenize_string_piece_ids(text);
        self.piece_ids_with_lex_indices(&piece_ids)
    }
}
