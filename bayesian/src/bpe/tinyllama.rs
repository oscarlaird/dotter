use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::prepared_dense::{
    self, MergeRows, PreparedFirstDense, PreparedSecondBuckets, PreparedSecondToken,
};
use super::{
    BpeError, BpeMerges, NO_PACKED_SPINE_INDEX, PackedSpine, SPACESYMBOL, SpineEntry,
};

pub const TINYLLAMA_PIECE_COUNT: usize = 32_000;
pub type TinyLlamaPreparedFirstDense = PreparedFirstDense<TINYLLAMA_PIECE_COUNT>;

fn build_prepared_second_buckets(
    token_ids_with_left_spines: &[u32],
    left_spine_index_by_token_id: &[u16],
    packed_left_spines: &[PackedSpine],
) -> PreparedSecondBuckets {
    let mut entries = Vec::with_capacity(token_ids_with_left_spines.len());
    for &token_id in token_ids_with_left_spines {
        let spine_index = left_spine_index_by_token_id[token_id as usize] as usize;
        entries.push(PreparedSecondToken {
            token_id,
            left_spine: prepared_dense::CompactLeftSpine::from_packed(
                packed_left_spines[spine_index],
            ),
        });
    }
    prepared_dense::sort_prepared_second_tokens(&mut entries);
    prepared_dense::bucket_prepared_second_tokens(&entries)
}

/// TinyLlama-compatible encoding for a **single word**: no whitespace in `text`.
/// Matches `transformers.AutoTokenizer.encode(text, add_special_tokens=False)` for such strings.
#[derive(Debug, Clone)]
pub struct TinyLlamaWordTokenizer {
    merges: BpeMerges,
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    piece_id_to_vocab_id: Vec<Option<u32>>,
    left_spine_index_by_token_id: Vec<u16>,
    packed_left_spines: Vec<PackedSpine>,
    token_ids_with_left_spines: Vec<u32>,
    prepared_merge_rows: MergeRows,
    prepared_second_buckets: PreparedSecondBuckets,
}

impl TinyLlamaWordTokenizer {
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Result<Self, BpeError> {
        let text = fs::read_to_string(path.as_ref())?;
        Self::from_tokenizer_json_str(&text)
    }

    pub fn from_tokenizer_json_str(content: &str) -> Result<Self, BpeError> {
        let merges = BpeMerges::from_tokenizer_json_str(content)?;
        let v: serde_json::Value = serde_json::from_str(content)?;
        let vocab_obj = v
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|m| m.as_object())
            .ok_or(BpeError::InvalidTokenizerJson("missing model.vocab object"))?;

        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut entries = Vec::with_capacity(vocab_obj.len());
        let mut max_id = 0usize;
        for (token, id_val) in vocab_obj {
            let id = id_val
                .as_u64()
                .or_else(|| id_val.as_i64().map(|i| i as u64))
                .ok_or(BpeError::InvalidTokenizerJson("vocab id is not an integer"))?;
            if id > u32::MAX as u64 {
                return Err(BpeError::InvalidTokenizerJson("vocab id does not fit u32"));
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
                return Err(BpeError::InvalidTokenizerJson("duplicate vocab id"));
            }
            *slot = true;
            id_to_token[id as usize] = token.to_string();
        }
        if seen.iter().any(|present| !present) {
            return Err(BpeError::InvalidTokenizerJson(
                "vocab ids must be dense from 0..max",
            ));
        }

        let mut piece_id_to_vocab_id = vec![None; merges.pieces.len()];
        for (token, &id) in &vocab {
            if let Some(piece_id) = merges.encode_piece(token) {
                piece_id_to_vocab_id[piece_id as usize] = Some(id);
            }
        }

        let mut left_spine_index_by_token_id = vec![NO_PACKED_SPINE_INDEX; id_to_token.len()];
        let mut packed_left_spines = Vec::new();
        let mut token_ids_with_left_spines = Vec::new();
        for (token, &id) in &vocab {
            if let Some(left_spine) = merges.left_spine(token) {
                if let Some(packed_left_spine) = PackedSpine::from_entries(&left_spine) {
                    let spine_index = u16::try_from(packed_left_spines.len())
                        .expect("TinyLlama spine count fits in u16");
                    token_ids_with_left_spines.push(id);
                    left_spine_index_by_token_id[id as usize] = spine_index;
                    packed_left_spines.push(packed_left_spine);
                }
            }
        }
        token_ids_with_left_spines.sort_unstable();
        let prepared_merge_rows = MergeRows::from_bpe_merges(&merges)?;
        let prepared_second_buckets = build_prepared_second_buckets(
            &token_ids_with_left_spines,
            &left_spine_index_by_token_id,
            &packed_left_spines,
        );

        Ok(Self {
            merges,
            vocab,
            id_to_token,
            piece_id_to_vocab_id,
            left_spine_index_by_token_id,
            packed_left_spines,
            token_ids_with_left_spines,
            prepared_merge_rows,
            prepared_second_buckets,
        })
    }

    fn word_with_space_symbol(&self, text: &str) -> Result<String, BpeError> {
        if text.is_empty() {
            return Ok(String::new());
        }
        if text.chars().any(|c| c.is_whitespace()) {
            return Err(BpeError::WhitespaceInWord);
        }
        let mut s = String::with_capacity(text.len() + 4);
        if !text.starts_with(SPACESYMBOL) {
            s.push(SPACESYMBOL);
        }
        s.push_str(text);
        Ok(s)
    }

    /// BPE surface strings (e.g. `▁hello`, or `▁abc` + `def` for `abcdef`).
    pub fn tokenize_word(&self, text: &str) -> Result<Vec<String>, BpeError> {
        let s = self.word_with_space_symbol(text)?;
        if s.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self
            .tokenize_word_piece_ids(text)?
            .into_iter()
            .map(|piece_id| {
                self.merges
                    .decode_piece(piece_id)
                    .expect("piece ids returned by tokenize_word_piece_ids must decode")
                    .to_string()
            })
            .collect())
    }

    /// Internal BPE piece IDs for a single space-free word.
    pub fn tokenize_word_piece_ids(&self, text: &str) -> Result<Vec<u32>, BpeError> {
        let s = self.word_with_space_symbol(text)?;
        if s.is_empty() {
            return Ok(Vec::new());
        }
        for c in s.chars() {
            if self.merges.piece_id_char(c).is_none() {
                return Err(BpeError::UnknownPiece(c.to_string()));
            }
        }
        self.merges
            .tokenize_ids(&s)
            .ok_or_else(|| BpeError::UnknownPiece(s))
    }

    /// True iff `token` is present in the tokenizer vocabulary.
    pub fn is_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Encode one vocab token to its tokenizer ID.
    pub fn encode_token(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Decode one tokenizer ID to its vocab token.
    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    /// Decode tokenizer IDs to their vocab tokens.
    pub fn decode<'a>(&'a self, ids: &[u32]) -> Result<Vec<&'a str>, BpeError> {
        ids.iter()
            .map(|&id| self.decode_token(id).ok_or(BpeError::UnknownTokenId(id)))
            .collect()
    }

    /// Compute the right spine for a token surface form.
    pub fn right_spine(&self, token: &str) -> Option<Vec<SpineEntry>> {
        self.merges.right_spine(token)
    }

    /// Compute a packed right spine for a token surface form.
    pub fn right_packed_spine(&self, token: &str) -> Option<PackedSpine> {
        PackedSpine::from_entries(&self.right_spine(token)?)
    }

    /// Compute the right spine for a tokenizer vocab ID.
    pub fn right_spine_for_token_id(&self, token_id: u32) -> Option<Vec<SpineEntry>> {
        let token = self.decode_token(token_id)?;
        self.right_spine(token)
    }

    /// Compute a packed right spine for a tokenizer vocab ID.
    pub fn right_packed_spine_for_token_id(&self, token_id: u32) -> Option<PackedSpine> {
        let token = self.decode_token(token_id)?;
        self.right_packed_spine(token)
    }

    /// Returns the precomputed left spine for a tokenizer vocab ID, if that token has a BPE
    /// derivation tree.
    pub fn left_spine_for_token_id(&self, token_id: u32) -> Option<&[SpineEntry]> {
        self.left_packed_spine_for_token_id(token_id)
            .map(PackedSpine::as_slice)
    }

    pub fn left_packed_spine_for_token_id(&self, token_id: u32) -> Option<&PackedSpine> {
        let spine_index = *self.left_spine_index_by_token_id.get(token_id as usize)?;
        if spine_index == NO_PACKED_SPINE_INDEX {
            None
        } else {
            self.packed_left_spines.get(spine_index as usize)
        }
    }

    /// Tokenizer vocab IDs for which a left spine was successfully precomputed.
    pub fn token_ids_with_left_spines(&self) -> &[u32] {
        &self.token_ids_with_left_spines
    }

    pub fn packed_left_spines(&self) -> &[PackedSpine] {
        &self.packed_left_spines
    }

    #[doc(hidden)]
    pub fn prepare_canonical_pair_batch_for_token_id(
        &self,
        first_token_id: u32,
    ) -> Result<Option<TinyLlamaPreparedFirstDense>, BpeError> {
        let Some(first_right_spine) = self.right_packed_spine_for_token_id(first_token_id) else {
            return Ok(None);
        };
        PreparedFirstDense::<TINYLLAMA_PIECE_COUNT>::build(first_right_spine, &self.prepared_merge_rows)
            .map(Some)
    }

    #[doc(hidden)]
    pub fn prepared_second_buckets(&self) -> &PreparedSecondBuckets {
        &self.prepared_second_buckets
    }

    pub fn canonical_pair_batch_with_packed_right_spine_into(
        &self,
        first_right_spine: &PackedSpine,
        out: &mut [bool],
    ) -> Result<(), BpeError> {
        if out.len() != self.id_to_token.len() {
            return Err(BpeError::InvalidPreparedDenseMaskLen {
                expected: self.id_to_token.len(),
                got: out.len(),
            });
        }
        out.fill(false);
        let prepared_first = PreparedFirstDense::<TINYLLAMA_PIECE_COUNT>::build(
            *first_right_spine,
            &self.prepared_merge_rows,
        )?;
        prepared_dense::fill_canonical_pair_mask(
            &prepared_first,
            &self.prepared_second_buckets,
            out,
        );
        Ok(())
    }

    pub fn canonical_pair_batch_with_packed_right_spine(
        &self,
        first_right_spine: &PackedSpine,
    ) -> Result<Vec<bool>, BpeError> {
        let mut out = vec![false; self.id_to_token.len()];
        self.canonical_pair_batch_with_packed_right_spine_into(first_right_spine, &mut out)?;
        Ok(out)
    }

    pub fn canonical_pair_batch_for_token_id(
        &self,
        first_token_id: u32,
    ) -> Result<Option<Vec<bool>>, BpeError> {
        let Some(first_right_spine) = self.right_packed_spine_for_token_id(first_token_id) else {
            return Ok(None);
        };
        self.canonical_pair_batch_with_packed_right_spine(&first_right_spine)
            .map(Some)
    }

    /// Decide canonicality for many second tokens after computing the first token's right spine
    /// once.
    pub fn canonical_pair_with_right_spine_and_token_id(
        &self,
        first_right_spine: &[SpineEntry],
        second_token_id: u32,
    ) -> Option<bool> {
        let left_spine = self.left_spine_for_token_id(second_token_id)?;
        Some(
            self.merges
                .canonical_pair_from_spines(first_right_spine, left_spine),
        )
    }

    pub fn canonical_pair_with_packed_right_spine_and_token_id(
        &self,
        first_right_spine: &PackedSpine,
        second_token_id: u32,
    ) -> Option<bool> {
        let left_spine = self.left_packed_spine_for_token_id(second_token_id)?;
        Some(
            self.merges
                .canonical_pair_from_packed_spines(first_right_spine, left_spine),
        )
    }

    pub fn canonical_pair_from_packed_spines(
        &self,
        first_right_spine: &PackedSpine,
        second_left_spine: &PackedSpine,
    ) -> bool {
        self.merges
            .canonical_pair_from_packed_spines(first_right_spine, second_left_spine)
    }

    /// Like [`Self::canonical_pair_with_right_spine_and_token_id`], but looks up the second token
    /// by surface form first.
    pub fn canonical_pair_with_right_spine(
        &self,
        first_right_spine: &[SpineEntry],
        second_token: &str,
    ) -> Option<bool> {
        let second_token_id = self.encode_token(second_token)?;
        self.canonical_pair_with_right_spine_and_token_id(first_right_spine, second_token_id)
    }

    /// Returns true exactly when raw BPE tokenization of `a + b` is `[a, b]`.
    pub fn canonical_pair(&self, a: &str, b: &str) -> bool {
        let Some(first_right_spine) = self.right_packed_spine(a) else {
            return self.merges.canonical_pair(a, b);
        };
        let Some(second_token_id) = self.encode_token(b) else {
            return self.merges.canonical_pair(a, b);
        };
        self.canonical_pair_with_packed_right_spine_and_token_id(
            &first_right_spine,
            second_token_id,
        )
        .unwrap_or_else(|| self.merges.canonical_pair(a, b))
    }

    /// Iterate over all surface-form tokens in the vocabulary.
    pub fn vocab_tokens(&self) -> impl Iterator<Item = &str> {
        self.vocab.keys().map(String::as_str)
    }

    /// Token ids for a single space-free word (no special tokens).
    pub fn encode_word(&self, text: &str) -> Result<Vec<u32>, BpeError> {
        let piece_ids = self.tokenize_word_piece_ids(text)?;
        let mut ids = Vec::with_capacity(piece_ids.len());
        for piece_id in piece_ids {
            let piece = self
                .merges
                .decode_piece(piece_id)
                .expect("piece ids returned by tokenize_word_piece_ids must decode");
            let vocab_id = self.piece_id_to_vocab_id[piece_id as usize]
                .ok_or_else(|| BpeError::UnknownPiece(piece.to_string()))?;
            ids.push(vocab_id);
        }
        Ok(ids)
    }

    /// TinyLlama pieces together with their vocab ids for one space-free word.
    pub fn encode_word_with_pieces(&self, text: &str) -> Result<Vec<(String, u32)>, BpeError> {
        let piece_ids = self.tokenize_word_piece_ids(text)?;
        let mut out = Vec::with_capacity(piece_ids.len());
        for piece_id in piece_ids {
            let piece = self
                .merges
                .decode_piece(piece_id)
                .expect("piece ids returned by tokenize_word_piece_ids must decode");
            let vocab_id = self.piece_id_to_vocab_id[piece_id as usize]
                .ok_or_else(|| BpeError::UnknownPiece(piece.to_string()))?;
            out.push((piece.to_string(), vocab_id));
        }
        Ok(out)
    }
}
