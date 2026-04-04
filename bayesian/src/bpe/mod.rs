//! Byte-pair encoding: load merges and run the usual merge loop on a string.
//!
//! For **TinyLlama** (and similar HF SentencePiece+BPE models), `tokenizer.json` uses the
//! SentencePiece whitespace marker **`▁`** (U+2581). We normalize that to ASCII **`_`** (U+005F)
//! everywhere inside this crate so BPE strings match the trie alphabet. See [`TinyLlamaWordTokenizer`].

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

#[doc(hidden)]
pub mod prepared_allpairs;

mod tokenizer_config;
mod word_tokenizer;

pub use self::tokenizer_config::{
    NUM_PREFIXES, NUM_TOKENS, TOKENIZER_JSON_PATH, TOKENIZER_JSON_STR,
};
pub use self::word_tokenizer::{TinyLlamaPreparedFirstAllPairs, TinyLlamaWordTokenizer};

/// Hugging Face / SentencePiece surface form in `tokenizer.json` only (not used after load).
pub const HF_SPACE_MARKER: char = '\u{2581}';

/// Word-boundary / “space” marker in all internally stored BPE strings (`lex_tokens`, merges, etc.).
pub const SPACESYMBOL: char = '_';

/// Map `tokenizer.json` token strings to internal form: every `HF_SPACE_MARKER` → `SPACESYMBOL`.
pub fn hf_token_to_internal(s: &str) -> String {
    if !s.contains(HF_SPACE_MARKER) {
        return s.to_string();
    }
    s.chars()
        .map(|ch| {
            if ch == HF_SPACE_MARKER {
                SPACESYMBOL
            } else {
                ch
            }
        })
        .collect()
}

type PieceId = u32;
pub const MAX_PACKED_SPINE_LEN: usize = 8;
const NO_PACKED_SPINE_INDEX: u16 = u16::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpineEntry {
    pub id: u16,
    /// Encodes the rank of the merge that produces the next spine node.
    /// A value of `0` means there is no next merge on this spine.
    pub priority_score: u16,
}

impl SpineEntry {
    fn new(id: u32, priority_score: Option<u32>) -> Option<Self> {
        let id = u16::try_from(id).ok()?;
        let priority_score = match priority_score {
            None => 0,
            Some(value) => u16::try_from(value).ok()?,
        };
        Some(Self { id, priority_score })
    }

    fn next_priority_score(self) -> u16 {
        self.priority_score
    }

    fn next_rank(self) -> Option<u32> {
        if self.priority_score == 0 {
            None
        } else {
            Some((u16::MAX as u32) - self.priority_score as u32)
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
        priority_score: 0,
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

impl MergeEntry {
    fn priority_score(self) -> u16 {
        BpeMerges::rank_to_priority_score(self.rank)
    }
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
    priority_score: Option<u32>,
    left: Option<usize>,
    right: Option<usize>,
}

#[derive(Debug)]
pub enum BpeError {
    Io(io::Error),
    Json(serde_json::Error),
    BadMergeLine { line_no: usize, line: String },
    InvalidTokenizerJson(&'static str),
    UnsupportedPreparedDense(&'static str),
    InvalidPreparedDenseMaskLen { expected: usize, got: usize },
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
            BpeError::UnsupportedPreparedDense(msg) => {
                write!(f, "unsupported prepared dense fast path: {msg}")
            }
            BpeError::InvalidPreparedDenseMaskLen { expected, got } => {
                write!(f, "prepared dense mask has len {got}, expected {expected}")
            }
            BpeError::WhitespaceInWord => {
                write!(f, "input contains whitespace (use one word only)")
            }
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
    fn rank_to_priority_score(rank: u32) -> u16 {
        debug_assert!(
            rank < u16::MAX as u32,
            "expected merge ranks to fit in u16 domain"
        );
        (u16::MAX as u32 - rank) as u16
    }

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
                    let rule_score = rule.priority_score();
                    let best_score = best_rule.priority_score();
                    let should_replace =
                        rule_score > best_score || (rule_score == best_score && i < best_idx);
                    if should_replace {
                        best = Some((i, rule));
                    };
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

    /// Load BPE merge ranks from a Hugging Face `tokenizer.json` (`model.merges`).
    /// If `model.vocab` exists, its token strings are interned too so ID-based tokenization can
    /// stay in the interned representation longer.
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Self {
        let text = fs::read_to_string(path.as_ref()).expect("failed to read tokenizer.json");
        Self::from_tokenizer_json_str(&text)
    }

    pub fn from_tokenizer_json_str(content: &str) -> Self {
        let v: serde_json::Value = serde_json::from_str(content).expect("invalid tokenizer.json");
        let model = v.get("model").expect("missing model object");
        let mut out = Self::new();
        if let Some(vocab) = model.get("vocab").and_then(|v| v.as_object()) {
            for token in vocab.keys() {
                out.intern_piece(&hf_token_to_internal(token));
            }
        }
        let merges = model
            .get("merges")
            .and_then(|m| m.as_array())
            .expect("missing model.merges array");
        for (idx, item) in merges.iter().enumerate() {
            let line = item.as_str().expect("merge entry is not a string");
            let line_no = idx + 1;
            let (left, right) = parse_merge_line(line, line_no).expect("invalid merge line");
            let left = hf_token_to_internal(left);
            let right = hf_token_to_internal(right);
            let left_id = out.intern_piece(&left);
            let right_id = out.intern_piece(&right);
            let mut merged = String::with_capacity(left.len() + right.len());
            merged.push_str(&left);
            merged.push_str(&right);
            let merged_id = out.intern_owned_piece(merged);
            out.insert_merge(left_id, right_id, merged_id, idx as u32);
        }
        out
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
        pieces.len() == 2 && pieces[0].text(&self.pieces) == a && pieces[1].text(&self.pieces) == b
    }

    /// Decide canonicality from the first token's right spine and the second token's left spine.
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
            let right_priority_score = right_spine[i].next_priority_score() as u32;
            let left_priority_score = left_spine[j].next_priority_score() as u32;
            let cross_priority_score = self
                .lookup_merge_by_pair(right_spine[i].id_u32(), left_spine[j].id_u32())
                .map_or(0, |entry| entry.priority_score() as u32);

            let mut best_priority_score = right_priority_score;
            if left_priority_score > best_priority_score {
                best_priority_score = left_priority_score;
            }
            if cross_priority_score > best_priority_score {
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

            unreachable!("best score must come from one of the three candidate events");
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
        let mut priority_score_to_next = None;
        loop {
            let PieceRef::Known(id) = nodes[cursor].piece else {
                return None;
            };
            spine.push(SpineEntry::new(id, priority_score_to_next)?);
            let next = if take_left {
                nodes[cursor].left
            } else {
                nodes[cursor].right
            };
            let Some(next) = next else {
                break;
            };
            priority_score_to_next = nodes[cursor].priority_score;
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
                priority_score: None,
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
                        let rule_score = rule.priority_score();
                        let best_score = best_rule.priority_score();
                        let should_replace =
                            rule_score > best_score || (rule_score == best_score && i < best_idx);
                        if should_replace {
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
                priority_score: Some(rule.priority_score() as u32),
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
    use std::collections::HashSet;
    use std::path::Path;

    fn tokenizer_json_with_merges(merges: &[&str]) -> String {
        let mut seen = HashSet::new();
        let mut pieces = Vec::new();
        for (line_no, merge) in merges.iter().enumerate() {
            let (left, right) = parse_merge_line(merge, line_no + 1).unwrap();
            for piece in [
                left.to_string(),
                right.to_string(),
                format!("{left}{right}"),
            ] {
                if seen.insert(piece.clone()) {
                    pieces.push(piece);
                }
            }
        }
        let vocab = pieces
            .into_iter()
            .enumerate()
            .map(|(id, piece)| (piece, serde_json::Value::from(id)))
            .collect::<serde_json::Map<String, serde_json::Value>>();
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": merges,
            }
        })
        .to_string()
    }

    fn tiny_tokenizer_json_for_tests() -> &'static str {
        r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "a": 0,
                    "b": 1,
                    "ab": 2,
                    "c": 3,
                    "abc": 4,
                    "<s>": 5
                },
                "merges": ["a b", "ab c"]
            }
        }"#
    }

    #[test]
    fn merges_order() {
        let json = tokenizer_json_with_merges(&["a b", "ab c"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
        assert_eq!(m.tokenize("abc"), vec!["abc".to_string()]);
    }

    #[test]
    fn tokenize_ids_round_trip_to_surface_forms() {
        let json = tokenizer_json_with_merges(&["a b", "ab c"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
        let ids = m.tokenize_ids("abc").unwrap();
        assert_eq!(m.decode_piece_ids(&ids), Some(vec!["abc"]));
    }

    #[test]
    fn merge_lookup_by_pair_returns_expected_entry() {
        let json = tokenizer_json_with_merges(&["a b", "ab c"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
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
        let json = tokenizer_json_with_merges(&["a a"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
        assert_eq!(m.tokenize("aa"), vec!["aa".to_string()]);
    }

    #[test]
    fn canonical_pair_checks_exact_two_piece_split() {
        let json = tokenizer_json_with_merges(&["a b", "ab c"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
        assert!(!m.canonical_pair("a", "b"));
        assert!(m.canonical_pair("ab", "d"));
        assert!(!m.canonical_pair("a", "bc"));
    }

    #[test]
    fn canonical_pair_from_spines_detects_boundary_crossing() {
        let json = tokenizer_json_with_merges(&["a b", "ab c"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
        let right = m.right_spine("a").unwrap();
        let left = m.left_spine("b").unwrap();
        assert!(!m.canonical_pair_from_spines(&right, &left));
    }

    #[test]
    fn canonical_pair_from_spines_accepts_safe_boundary() {
        let json = tokenizer_json_with_merges(&["a b", "ab c", "x y"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
        let right = m.right_spine("ab").unwrap();
        let left = m.left_spine("x").unwrap();
        assert!(m.canonical_pair_from_spines(&right, &left));
        assert_eq!(
            m.canonical_pair_from_spines(&right, &left),
            m.canonical_pair("ab", "x")
        );
    }

    #[test]
    fn canonical_pair_from_spines_matches_string_check_on_known_pieces() {
        let json = tokenizer_json_with_merges(&["a b", "b c", "ab c", "a bc", "x y"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
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
        let json = tokenizer_json_with_merges(&["h e", "l l", "he ll"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
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
        let json = tokenizer_json_with_merges(&["a b", "b c", "ab c", "a bc"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
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
        let json = tokenizer_json_with_merges(&["h e", "l l", "he ll"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
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
        let json = tokenizer_json_with_merges(&["h e", "l l", "he ll"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
        assert_eq!(m.right_spine("hel"), None);
    }

    #[test]
    fn left_spine_is_none_when_string_is_not_a_single_token() {
        let json = tokenizer_json_with_merges(&["h e", "l l", "he ll"]);
        let m = BpeMerges::from_tokenizer_json_str(&json);
        assert_eq!(m.left_spine("hel"), None);
    }

    #[test]
    fn tinyllama_encode_string_bundled_tokenizer() {
        let tok = TinyLlamaWordTokenizer::from_tokenizer_json(Path::new(TOKENIZER_JSON_PATH));
        let hello = tok.lex_index("_hello").expect("expected tokenizer token");
        let okay = tok.lex_index("_okay").expect("expected tokenizer token");
        let abc = tok.lex_index("_abc").expect("expected tokenizer token");
        let def = tok.lex_index("def").expect("expected tokenizer token");
        let the = tok.lex_index("_the").expect("expected tokenizer token");
        assert_eq!(tok.tokenize_string_to_lex_indices("_hello"), vec![hello]);
        assert_eq!(tok.tokenize_string_to_lex_indices("_okay"), vec![okay]);
        assert_eq!(
            tok.tokenize_string_to_lex_indices("_abcdef"),
            vec![abc, def]
        );
        assert_eq!(tok.tokenize_string_to_lex_indices("_the"), vec![the]);
        assert_eq!(tok.tokenize_string_to_lex_indices("def"), vec![def]);
        assert_eq!(
            tok.tokenize_string_to_lex_indices("_abcdef"),
            vec![abc, def]
        );
        assert_eq!(tok.token_at(hello), "_hello");
        assert!(tok.can_canonically_follow("_abc", "def"));
        assert!(tok.lex_indices_with_left_spines().contains(&def));
        assert_eq!(
            tok.tokenize_string_with_lex_indices("_abcdef"),
            vec![("_abc".to_string(), abc), ("def".to_string(), def)]
        );
        assert_eq!(
            tok.tokenize_string_with_lex_indices("def"),
            vec![("def".to_string(), def)]
        );
    }

    #[test]
    fn canonical_pair_batch_matches_scalar_reference_on_ordinary_token_domain() {
        let tok = TinyLlamaWordTokenizer::from_tokenizer_json_str(tiny_tokenizer_json_for_tests());

        for &first_lex_index in tok.lex_indices_with_left_spines() {
            let mask = tok.canonical_followers_for_lex_index(first_lex_index);
            assert_eq!(mask.len(), 6);
            let first_token = tok.token_at(first_lex_index);

            for &second_lex_index in tok.lex_indices_with_left_spines() {
                let second_token = tok.token_at(second_lex_index);
                let expected = tok.can_canonically_follow(first_token, second_token);
                assert_eq!(
                    mask[second_lex_index], expected,
                    "mismatch for pair ({first_lex_index}, {second_lex_index})"
                );
            }

            assert!(!mask[tok.lex_index("<s>").expect("expected tokenizer token")]);
        }
    }

    #[test]
    fn canonical_pair_batch_supports_special_first_token_via_scalar_path() {
        let tok = TinyLlamaWordTokenizer::from_tokenizer_json_str(tiny_tokenizer_json_for_tests());
        let mask = tok.canonical_followers("<s>");
        assert_eq!(mask, vec![false; tok.tokens().len()]);
    }

    #[test]
    #[should_panic]
    fn canonical_pair_batch_into_validates_output_len() {
        let tok = TinyLlamaWordTokenizer::from_tokenizer_json_str(tiny_tokenizer_json_for_tests());
        let first_token_right_spine = tok.right_packed_spine_for_lex_index(
            tok.lex_index("ab").expect("expected tokenizer token"),
        );
        tok.canonical_pair_batch_with_first_token_right_spine_into(
            &first_token_right_spine,
            &mut [false; 3],
        );
    }

    #[test]
    fn tinyllama_lex_helpers_and_prefix_ranges_work() {
        let tok = TinyLlamaWordTokenizer::from_tokenizer_json_str(tiny_tokenizer_json_for_tests());
        let tokens: Vec<&str> = tok.tokens().iter().map(String::as_str).collect();
        assert_eq!(tokens, vec!["<s>", "a", "ab", "abc", "b", "c"]);
        let prefixes: Vec<&str> = tok.prefixes().iter().map(String::as_str).collect();
        assert_eq!(
            prefixes,
            vec!["", "<", "<s", "<s>", "a", "ab", "abc", "b", "c"]
        );
        assert_eq!(tok.prefix_count(), 9);
        assert_eq!(tok.lex_index("ab"), Some(2));
        assert_eq!(tok.prefix_lex_index("ab"), Some(5));
        assert_eq!(tok.token_at(2), "ab");
        assert_eq!(tok.prefix_at(5), "ab");
        assert!(tok.has_token_with_prefix("ab"));
        assert!(tok.has_token_with_strict_prefix("ab"));
        assert!(!tok.has_token_with_strict_prefix("abc"));
        assert!(!tok.has_token_with_strict_prefix("zzz"));
        assert_eq!(tok.token_lex_range_for_prefix("ab"), (2, 4));
        assert_eq!(tok.token_lex_range_for_prefix("ab"), (2, 4));
        assert_eq!(tok.token_lex_range_for_prefix_index(5), (2, 4));
        assert!(!tok.has_token_with_prefix("zzz"));
    }

    #[test]
    fn tinyllama_counts_true_tokens_by_prefix() {
        const PREFIX_COUNT: usize = 9;
        let tok = TinyLlamaWordTokenizer::from_tokenizer_json_str(tiny_tokenizer_json_for_tests());
        let mut token_flags = vec![false; tok.tokens().len()];
        token_flags[tok.lex_index("ab").expect("expected tokenizer token")] = true;
        token_flags[tok.lex_index("abc").expect("expected tokenizer token")] = true;
        token_flags[tok.lex_index("c").expect("expected tokenizer token")] = true;

        let counts = tok.count_true_tokens_by_prefix::<PREFIX_COUNT>(&token_flags);

        assert_eq!(counts.len(), PREFIX_COUNT);
        assert_eq!(
            counts[tok.prefix_lex_index("").expect("expected tokenizer prefix")],
            3
        );
        assert_eq!(
            counts[tok
                .prefix_lex_index("a")
                .expect("expected tokenizer prefix")],
            2
        );
        assert_eq!(
            counts[tok
                .prefix_lex_index("ab")
                .expect("expected tokenizer prefix")],
            2
        );
        assert_eq!(
            counts[tok
                .prefix_lex_index("abc")
                .expect("expected tokenizer prefix")],
            1
        );
        assert_eq!(
            counts[tok
                .prefix_lex_index("b")
                .expect("expected tokenizer prefix")],
            0
        );
        assert_eq!(
            counts[tok
                .prefix_lex_index("c")
                .expect("expected tokenizer prefix")],
            1
        );
    }
}
