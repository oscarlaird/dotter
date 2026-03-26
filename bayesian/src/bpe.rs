//! Byte-pair encoding: load merges and run the usual merge loop on a string.
//!
//! For **TinyLlama** (and similar HF SentencePiece+BPE models), space-free words are encoded
//! like Hugging Face when you prepend the SentencePiece whitespace marker **`▁`** (U+2581)
//! before char-level BPE. See [`TinyLlamaWordTokenizer`].

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

/// SentencePiece “space” / word-boundary marker used in TinyLlama vocab and merges.
pub const SPACESYMBOL: char = '\u{2581}';

type PieceId = u32;
pub const MAX_PACKED_SPINE_LEN: usize = 8;
const NO_PACKED_SPINE_INDEX: u16 = u16::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpineEntry {
    pub id: u16,
    /// Encodes the rank of the merge that produces the next spine node.
    /// A value of `0` means there is no next merge on this spine.
    pub rank_plus_one: u16,
}

impl SpineEntry {
    fn new(id: u32, rank: Option<u32>) -> Option<Self> {
        let id = u16::try_from(id).ok()?;
        let rank_plus_one = match rank {
            None => 0,
            Some(rank) => u16::try_from(rank.checked_add(1)?).ok()?,
        };
        Some(Self { id, rank_plus_one })
    }

    fn next_rank(self) -> Option<u32> {
        if self.rank_plus_one == 0 {
            None
        } else {
            Some((self.rank_plus_one - 1) as u32)
        }
    }

    fn id_u32(self) -> u32 {
        self.id as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedSpine {
    len: u8,
    entries: [SpineEntry; MAX_PACKED_SPINE_LEN],
}

impl PackedSpine {
    const EMPTY_ENTRY: SpineEntry = SpineEntry {
        id: 0,
        rank_plus_one: 0,
    };

    const fn empty() -> Self {
        Self {
            len: 0,
            entries: [Self::EMPTY_ENTRY; MAX_PACKED_SPINE_LEN],
        }
    }

    fn from_entries(entries: &[SpineEntry]) -> Option<Self> {
        if entries.len() > MAX_PACKED_SPINE_LEN {
            return None;
        }
        let mut packed = Self::empty();
        packed.len = entries.len() as u8;
        packed.entries[..entries.len()].copy_from_slice(entries);
        Some(packed)
    }

    pub fn as_slice(&self) -> &[SpineEntry] {
        &self.entries[..self.len as usize]
    }

    fn is_present(&self) -> bool {
        self.len != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeEntry {
    pub left: u32,
    pub right: u32,
    pub merged: u32,
    pub rank: u32,
}

/// Interned BPE pieces plus an ID-based merge table.
#[derive(Debug, Clone)]
pub struct BpeMerges {
    piece_to_id: HashMap<String, PieceId>,
    pieces: Vec<String>,
    merges: HashMap<(PieceId, PieceId), MergeEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PieceRef {
    Known(PieceId),
    Inline(String),
}

impl PieceRef {
    fn text<'a>(&'a self, pieces: &'a [String]) -> &'a str {
        match self {
            Self::Known(id) => pieces[*id as usize].as_str(),
            Self::Inline(text) => text.as_str(),
        }
    }
}

#[derive(Debug, Clone)]
struct DerivationNode {
    piece: PieceRef,
    rank: Option<u32>,
    left: Option<usize>,
    right: Option<usize>,
}

#[derive(Debug)]
pub enum BpeError {
    Io(io::Error),
    Json(serde_json::Error),
    BadMergeLine { line_no: usize, line: String },
    InvalidTokenizerJson(&'static str),
    WhitespaceInWord,
    UnknownPiece(String),
    UnknownTokenId(u32),
}

impl std::fmt::Display for BpeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BpeError::Io(e) => write!(f, "{e}"),
            BpeError::Json(e) => write!(f, "{e}"),
            BpeError::BadMergeLine { line_no, line } => {
                write!(f, "bad merge line {line_no}: {line:?}")
            }
            BpeError::InvalidTokenizerJson(msg) => write!(f, "invalid tokenizer.json: {msg}"),
            BpeError::WhitespaceInWord => write!(f, "input contains whitespace (use one word only)"),
            BpeError::UnknownPiece(s) => write!(f, "piece not in vocab: {s:?}"),
            BpeError::UnknownTokenId(id) => write!(f, "unknown token id: {id}"),
        }
    }
}

impl std::error::Error for BpeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BpeError::Io(e) => Some(e),
            BpeError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for BpeError {
    fn from(e: io::Error) -> Self {
        BpeError::Io(e)
    }
}

impl From<serde_json::Error> for BpeError {
    fn from(e: serde_json::Error) -> Self {
        BpeError::Json(e)
    }
}

impl BpeMerges {
    fn new() -> Self {
        Self {
            piece_to_id: HashMap::new(),
            pieces: Vec::new(),
            merges: HashMap::new(),
        }
    }

    fn intern_piece(&mut self, piece: &str) -> PieceId {
        if let Some(&id) = self.piece_to_id.get(piece) {
            return id;
        }
        let id = self.pieces.len() as PieceId;
        let owned = piece.to_string();
        self.piece_to_id.insert(owned.clone(), id);
        self.pieces.push(owned);
        id
    }

    fn intern_owned_piece(&mut self, piece: String) -> PieceId {
        if let Some(&id) = self.piece_to_id.get(piece.as_str()) {
            return id;
        }
        let id = self.pieces.len() as PieceId;
        self.piece_to_id.insert(piece.clone(), id);
        self.pieces.push(piece);
        id
    }

    fn piece_id_char(&self, c: char) -> Option<PieceId> {
        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        self.piece_to_id.get(s).copied()
    }

    fn char_piece(&self, c: char) -> PieceRef {
        match self.piece_id_char(c) {
            Some(id) => PieceRef::Known(id),
            None => PieceRef::Inline(c.to_string()),
        }
    }

    fn insert_merge(&mut self, left: PieceId, right: PieceId, merged: PieceId, rank: u32) {
        let entry = MergeEntry {
            left,
            right,
            merged,
            rank,
        };
        let previous = self.merges.insert((left, right), entry);
        debug_assert!(previous.is_none());
    }

    fn lookup_merge(&self, left: &PieceRef, right: &PieceRef) -> Option<MergeEntry> {
        let (PieceRef::Known(left_id), PieceRef::Known(right_id)) = (left, right) else {
            return None;
        };
        self.lookup_merge_by_pair(*left_id, *right_id)
    }

    fn best_merge(&self, symbols: &[PieceRef]) -> Option<(usize, MergeEntry)> {
        let mut best: Option<(usize, MergeEntry)> = None;
        for i in 0..symbols.len().saturating_sub(1) {
            let Some(rule) = self.lookup_merge(&symbols[i], &symbols[i + 1]) else {
                continue;
            };
            match best {
                None => best = Some((i, rule)),
                Some((best_idx, best_rule)) => {
                    if rule.rank < best_rule.rank || (rule.rank == best_rule.rank && i < best_idx) {
                        best = Some((i, rule));
                    }
                }
            }
        }
        best
    }

    fn tokenize_piece_refs(&self, text: &str) -> Vec<PieceRef> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut symbols: Vec<PieceRef> = text.chars().map(|c| self.char_piece(c)).collect();
        while let Some((idx, rule)) = self.best_merge(&symbols) {
            symbols[idx] = PieceRef::Known(rule.merged);
            symbols.remove(idx + 1);
        }
        symbols
    }

    /// Read a merges file: lines starting with `#` and empty lines are skipped.
    /// Each other line must contain two tokens separated by the **first** ASCII space (` `).
    pub fn from_merges_file(path: impl AsRef<Path>) -> Result<Self, BpeError> {
        let text = fs::read_to_string(path.as_ref())?;
        Self::from_merges_str(&text)
    }

    pub fn from_merges_str(content: &str) -> Result<Self, BpeError> {
        let mut out = Self::new();
        for (line_no, raw) in content.lines().enumerate() {
            let line_no = line_no + 1;
            let line = raw.trim_end_matches(['\r', '\n']);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (left, right) = parse_merge_line(line, line_no)?;
            let left_id = out.intern_piece(left);
            let right_id = out.intern_piece(right);
            let mut merged = String::with_capacity(left.len() + right.len());
            merged.push_str(left);
            merged.push_str(right);
            let merged_id = out.intern_owned_piece(merged);
            let rank = out.merges.len() as u32;
            out.insert_merge(left_id, right_id, merged_id, rank);
        }
        Ok(out)
    }

    /// Load BPE merge ranks from a Hugging Face `tokenizer.json` (`model.merges`).
    /// If `model.vocab` exists, its token strings are interned too so ID-based tokenization can
    /// stay in the interned representation longer.
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Result<Self, BpeError> {
        let text = fs::read_to_string(path.as_ref())?;
        Self::from_tokenizer_json_str(&text)
    }

    pub fn from_tokenizer_json_str(content: &str) -> Result<Self, BpeError> {
        let v: serde_json::Value = serde_json::from_str(content)?;
        let model = v
            .get("model")
            .ok_or(BpeError::InvalidTokenizerJson("missing model object"))?;
        let mut out = Self::new();
        if let Some(vocab) = model.get("vocab").and_then(|v| v.as_object()) {
            for token in vocab.keys() {
                out.intern_piece(token);
            }
        }
        let merges = model
            .get("merges")
            .and_then(|m| m.as_array())
            .ok_or(BpeError::InvalidTokenizerJson("missing model.merges array"))?;
        for (idx, item) in merges.iter().enumerate() {
            let line = item
                .as_str()
                .ok_or(BpeError::InvalidTokenizerJson("merge entry is not a string"))?;
            let line_no = idx + 1;
            let (left, right) = parse_merge_line(line, line_no)?;
            let left_id = out.intern_piece(left);
            let right_id = out.intern_piece(right);
            let mut merged = String::with_capacity(left.len() + right.len());
            merged.push_str(left);
            merged.push_str(right);
            let merged_id = out.intern_owned_piece(merged);
            out.insert_merge(left_id, right_id, merged_id, idx as u32);
        }
        Ok(out)
    }

    pub fn lookup_merge_by_pair(&self, left: u32, right: u32) -> Option<MergeEntry> {
        self.merges.get(&(left, right)).copied()
    }

    /// Encode an interned BPE piece to its internal ID.
    pub fn encode_piece(&self, piece: &str) -> Option<u32> {
        self.piece_to_id.get(piece).copied()
    }

    /// Decode an internal BPE piece ID back to its surface form.
    pub fn decode_piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(String::as_str)
    }

    /// Decode a slice of internal BPE piece IDs.
    pub fn decode_piece_ids<'a>(&'a self, ids: &[u32]) -> Option<Vec<&'a str>> {
        ids.iter().map(|&id| self.decode_piece(id)).collect()
    }

    /// Apply BPE merges until no adjacent pair appears in the merge table.
    /// Starts from one symbol per Unicode scalar value (`char`).
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_piece_refs(text)
            .into_iter()
            .map(|piece| piece.text(&self.pieces).to_string())
            .collect()
    }

    /// Like [`Self::tokenize`], but returns interned piece IDs when every leaf and merge result is
    /// known to the current merge graph. Returns `None` if tokenization would need an inline piece.
    pub fn tokenize_ids(&self, text: &str) -> Option<Vec<u32>> {
        self.tokenize_piece_refs(text)
            .into_iter()
            .map(|piece| match piece {
                PieceRef::Known(id) => Some(id),
                PieceRef::Inline(_) => None,
            })
            .collect()
    }

    /// Returns true exactly when raw BPE tokenization of `a + b` is `[a, b]`.
    pub fn canonical_pair(&self, a: &str, b: &str) -> bool {
        if let (Some(right_spine), Some(left_spine)) = (self.right_spine(a), self.left_spine(b)) {
            return self.canonical_pair_from_spines(&right_spine, &left_spine);
        }

        let mut text = String::with_capacity(a.len() + b.len());
        text.push_str(a);
        text.push_str(b);
        let pieces = self.tokenize_piece_refs(&text);
        pieces.len() == 2
            && pieces[0].text(&self.pieces) == a
            && pieces[1].text(&self.pieces) == b
    }

    /// Decide canonicality from the right spine of the first token and the left spine of the
    /// second token.
    ///
    /// The spine entries are assumed to be ordered leaf-to-root, with each rank denoting the
    /// merge that produces the next spine entry. The final entry must therefore carry `None`.
    ///
    /// See `math/tex/chapters/bpe-spines.tex` for the corresponding proof sketch and algorithmic
    /// description.
    pub fn canonical_pair_from_spines(
        &self,
        right_spine: &[SpineEntry],
        left_spine: &[SpineEntry],
    ) -> bool {
        if right_spine.is_empty() || left_spine.is_empty() {
            return false;
        }

        let mut i = 0usize;
        let mut j = 0usize;

        loop {
            let right_rank = right_spine[i].next_rank();
            let left_rank = left_spine[j].next_rank();
            let cross_rank = self
                .lookup_merge_by_pair(right_spine[i].id_u32(), left_spine[j].id_u32())
                .map(|entry| entry.rank);

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

    /// The leftmost root-to-leaf path in the merge tree for `token`, reported leaf-to-root.
    ///
    /// For example, if `hell` is built as `(he, ll)` with `he = (h, e)`, this returns the IDs
    /// for `[h, he, hell]`, together with the merge rank that produces the next spine node.
    pub fn left_spine(&self, token: &str) -> Option<Vec<SpineEntry>> {
        self.spine(token, true)
    }

    /// The rightmost root-to-leaf path in the merge tree for `token`, reported leaf-to-root.
    ///
    /// For example, if `hell` is built as `(he, ll)` with `ll = (l, l)`, this returns the IDs
    /// for `[l, ll, hell]`, together with the merge rank that produces the next spine node.
    pub fn right_spine(&self, token: &str) -> Option<Vec<SpineEntry>> {
        self.spine(token, false)
    }

    fn spine(&self, token: &str, take_left: bool) -> Option<Vec<SpineEntry>> {
        let (nodes, root) = self.tokenize_tree(token)?;
        let mut spine = Vec::new();
        let mut cursor = root;
        let mut rank_to_next = None;
        loop {
            let PieceRef::Known(id) = nodes[cursor].piece else {
                return None;
            };
            spine.push(SpineEntry::new(id, rank_to_next)?);
            let next = if take_left {
                nodes[cursor].left
            } else {
                nodes[cursor].right
            };
            let Some(next) = next else {
                break;
            };
            rank_to_next = nodes[cursor].rank;
            cursor = next;
        }
        spine.reverse();
        Some(spine)
    }

    pub fn canonical_pair_from_packed_spines(
        &self,
        right_spine: &PackedSpine,
        left_spine: &PackedSpine,
    ) -> bool {
        self.canonical_pair_from_spines(right_spine.as_slice(), left_spine.as_slice())
    }

    fn tokenize_tree(&self, text: &str) -> Option<(Vec<DerivationNode>, usize)> {
        if text.is_empty() {
            return None;
        }

        let mut nodes = Vec::with_capacity(text.chars().count() * 2);
        let mut symbols = Vec::new();
        for c in text.chars() {
            nodes.push(DerivationNode {
                piece: self.char_piece(c),
                rank: None,
                left: None,
                right: None,
            });
            symbols.push(nodes.len() - 1);
        }

        loop {
            let mut best: Option<(usize, MergeEntry)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let Some(rule) =
                    self.lookup_merge(&nodes[symbols[i]].piece, &nodes[symbols[i + 1]].piece)
                else {
                    continue;
                };
                match best {
                    None => best = Some((i, rule)),
                    Some((best_idx, best_rule)) => {
                        if rule.rank < best_rule.rank
                            || (rule.rank == best_rule.rank && i < best_idx)
                        {
                            best = Some((i, rule));
                        }
                    }
                }
            }

            let Some((idx, rule)) = best else {
                break;
            };

            let left = symbols[idx];
            let right = symbols[idx + 1];
            nodes.push(DerivationNode {
                piece: PieceRef::Known(rule.merged),
                rank: Some(rule.rank),
                left: Some(left),
                right: Some(right),
            });
            symbols[idx] = nodes.len() - 1;
            symbols.remove(idx + 1);
        }

        if symbols.len() == 1 && nodes[symbols[0]].piece.text(&self.pieces) == text {
            Some((nodes, symbols[0]))
        } else {
            None
        }
    }
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
            return Err(BpeError::InvalidTokenizerJson("vocab ids must be dense from 0..max"));
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
                    let spine_index =
                        u16::try_from(packed_left_spines.len()).expect("TinyLlama spine count fits in u16");
                    token_ids_with_left_spines.push(id);
                    left_spine_index_by_token_id[id as usize] = spine_index;
                    packed_left_spines.push(packed_left_spine);
                }
            }
        }

        Ok(Self {
            merges,
            vocab,
            id_to_token,
            piece_id_to_vocab_id,
            left_spine_index_by_token_id,
            packed_left_spines,
            token_ids_with_left_spines,
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
        self.canonical_pair_with_packed_right_spine_and_token_id(&first_right_spine, second_token_id)
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

fn parse_merge_line(line: &str, line_no: usize) -> Result<(&str, &str), BpeError> {
    let (left, right) = line.split_once(' ').ok_or_else(|| BpeError::BadMergeLine {
        line_no,
        line: line.to_string(),
    })?;
    if right.contains(' ') {
        return Err(BpeError::BadMergeLine {
            line_no,
            line: line.to_string(),
        });
    }
    Ok((left, right))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_order() {
        let m = BpeMerges::from_merges_str(
            "#version: 0.2\n\
             a b\n\
             ab c\n",
        )
        .unwrap();
        assert_eq!(m.tokenize("abc"), vec!["abc".to_string()]);
    }

    #[test]
    fn tokenize_ids_round_trip_to_surface_forms() {
        let m = BpeMerges::from_merges_str("a b\nab c\n").unwrap();
        let ids = m.tokenize_ids("abc").unwrap();
        assert_eq!(m.decode_piece_ids(&ids), Some(vec!["abc"]));
    }

    #[test]
    fn merge_lookup_by_pair_returns_expected_entry() {
        let m = BpeMerges::from_merges_str("a b\nab c\n").unwrap();
        let a = m.encode_piece("a").unwrap();
        let b = m.encode_piece("b").unwrap();
        let ab = m.encode_piece("ab").unwrap();

        let by_pair = m.lookup_merge_by_pair(a, b).unwrap();
        assert_eq!(by_pair.left, a);
        assert_eq!(by_pair.right, b);
        assert_eq!(by_pair.merged, ab);
        assert_eq!(by_pair.rank, 0);
    }

    #[test]
    fn leftmost_tie() {
        let m = BpeMerges::from_merges_str("a a\n").unwrap();
        assert_eq!(m.tokenize("aa"), vec!["aa".to_string()]);
    }

    #[test]
    fn canonical_pair_checks_exact_two_piece_split() {
        let m = BpeMerges::from_merges_str("a b\nab c\n").unwrap();
        assert!(!m.canonical_pair("a", "b"));
        assert!(m.canonical_pair("ab", "d"));
        assert!(!m.canonical_pair("a", "bc"));
    }

    #[test]
    fn canonical_pair_from_spines_detects_boundary_crossing() {
        let m = BpeMerges::from_merges_str("a b\nab c\n").unwrap();
        let right = m.right_spine("a").unwrap();
        let left = m.left_spine("b").unwrap();
        assert!(!m.canonical_pair_from_spines(&right, &left));
    }

    #[test]
    fn canonical_pair_from_spines_accepts_safe_boundary() {
        let m = BpeMerges::from_merges_str("a b\nab c\nx y\n").unwrap();
        let right = m.right_spine("ab").unwrap();
        let left = m.left_spine("x").unwrap();
        assert!(m.canonical_pair_from_spines(&right, &left));
        assert_eq!(m.canonical_pair_from_spines(&right, &left), m.canonical_pair("ab", "x"));
    }

    #[test]
    fn canonical_pair_from_spines_matches_string_check_on_known_pieces() {
        let m = BpeMerges::from_merges_str("a b\nb c\nab c\na bc\nx y\n").unwrap();
        for a in &m.pieces {
            let Some(right) = m.right_spine(a) else {
                continue;
            };
            for b in &m.pieces {
                let Some(left) = m.left_spine(b) else {
                    continue;
                };
                assert_eq!(
                    m.canonical_pair_from_spines(&right, &left),
                    m.canonical_pair(a, b),
                    "mismatch for pair ({a:?}, {b:?})"
                );
            }
        }
    }

    #[test]
    fn left_spine_follows_left_edge_of_merge_tree() {
        let m = BpeMerges::from_merges_str("h e\nl l\nhe ll\n").unwrap();
        let spine = m.left_spine("hell").unwrap();
        assert_eq!(
            m.decode_piece_ids(
                &spine
                    .iter()
                    .map(|entry| entry.id as u32)
                    .collect::<Vec<_>>(),
            ),
            Some(vec!["h", "he", "hell"])
        );
        assert_eq!(spine[0].next_rank(), Some(0));
        assert_eq!(spine[1].next_rank(), Some(2));
        assert_eq!(spine[2].next_rank(), None);
    }

    #[test]
    fn left_spine_uses_actual_bpe_derivation() {
        let m = BpeMerges::from_merges_str("a b\nb c\nab c\na bc\n").unwrap();
        let spine = m.left_spine("abc").unwrap();
        assert_eq!(
            m.decode_piece_ids(
                &spine
                    .iter()
                    .map(|entry| entry.id as u32)
                    .collect::<Vec<_>>(),
            ),
            Some(vec!["a", "ab", "abc"])
        );
        assert_eq!(spine[0].next_rank(), Some(0));
        assert_eq!(spine[1].next_rank(), Some(2));
        assert_eq!(spine[2].next_rank(), None);
    }

    #[test]
    fn right_spine_follows_right_edge_of_merge_tree() {
        let m = BpeMerges::from_merges_str("h e\nl l\nhe ll\n").unwrap();
        let spine = m.right_spine("hell").unwrap();
        assert_eq!(
            m.decode_piece_ids(
                &spine
                    .iter()
                    .map(|entry| entry.id as u32)
                    .collect::<Vec<_>>(),
            ),
            Some(vec!["l", "ll", "hell"])
        );
        assert_eq!(spine[0].next_rank(), Some(1));
        assert_eq!(spine[1].next_rank(), Some(2));
        assert_eq!(spine[2].next_rank(), None);
    }

    #[test]
    fn right_spine_is_none_when_string_is_not_a_single_token() {
        let m = BpeMerges::from_merges_str("h e\nl l\nhe ll\n").unwrap();
        assert_eq!(m.right_spine("hel"), None);
    }

    #[test]
    fn left_spine_is_none_when_string_is_not_a_single_token() {
        let m = BpeMerges::from_merges_str("h e\nl l\nhe ll\n").unwrap();
        assert_eq!(m.left_spine("hel"), None);
    }

    #[test]
    fn tinyllama_encode_word_if_cache_present() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let path = Path::new(&home).join(
            ".cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json",
        );
        if !path.is_file() {
            return;
        }
        let tok = TinyLlamaWordTokenizer::from_tokenizer_json(&path).unwrap();
        assert_eq!(tok.encode_word("hello").unwrap(), vec![22172]);
        assert_eq!(tok.encode_word("okay").unwrap(), vec![20759]);
        assert_eq!(tok.encode_word("abcdef").unwrap(), vec![25638, 1753]);
        assert_eq!(tok.encode_word("the").unwrap(), vec![278]);
        assert_eq!(
            tok.tokenize_word_piece_ids("abcdef").unwrap().len(),
            2
        );
        assert!(tok.is_token("▁hello"));
        assert_eq!(tok.encode_token("▁hello"), Some(22172));
        assert_eq!(tok.decode_token(22172), Some("▁hello"));
        assert_eq!(tok.decode(&[22172]).unwrap(), vec!["▁hello"]);
        let first_right_spine = tok.right_spine("▁abc").unwrap();
        let first_right_spine_by_id = tok.right_spine_for_token_id(25638).unwrap();
        assert_eq!(
            tok.canonical_pair_with_right_spine(&first_right_spine, "def"),
            Some(true)
        );
        assert_eq!(first_right_spine, first_right_spine_by_id);
        assert_eq!(
            tok.canonical_pair_with_right_spine_and_token_id(&first_right_spine, 1753),
            Some(true)
        );
        assert!(tok.left_spine_for_token_id(1753).is_some());
        assert!(tok.token_ids_with_left_spines().contains(&1753));
        assert!(!tok.is_token("not_a_real_token"));
        assert_eq!(
            tok.encode_word_with_pieces("abcdef").unwrap(),
            vec![("▁abc".to_string(), 25638), ("def".to_string(), 1753)]
        );
    }
}
