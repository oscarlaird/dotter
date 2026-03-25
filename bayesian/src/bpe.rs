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

/// `(left, right) -> rank` where **lower** rank means **earlier** in the merges file (higher priority).
#[derive(Debug, Clone)]
pub struct BpeMerges {
    ranks: HashMap<(String, String), u32>,
}

#[derive(Debug)]
pub enum BpeError {
    Io(io::Error),
    Json(serde_json::Error),
    BadMergeLine { line_no: usize, line: String },
    InvalidTokenizerJson(&'static str),
    WhitespaceInWord,
    UnknownPiece(String),
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
    /// Read a merges file: lines starting with `#` and empty lines are skipped.
    /// Each other line must contain two tokens separated by the **first** ASCII space (` `).
    pub fn from_merges_file(path: impl AsRef<Path>) -> Result<Self, BpeError> {
        let text = fs::read_to_string(path.as_ref())?;
        Self::from_merges_str(&text)
    }

    pub fn from_merges_str(content: &str) -> Result<Self, BpeError> {
        let mut ranks = HashMap::new();
        for (line_no, raw) in content.lines().enumerate() {
            let line_no = line_no + 1;
            let line = raw.trim_end_matches(['\r', '\n']);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (left, right) = parse_merge_line(line, line_no)?;
            let rank = ranks.len() as u32;
            ranks.insert((left.to_string(), right.to_string()), rank);
        }
        Ok(Self { ranks })
    }

    /// Load only the BPE merge ranks from a Hugging Face `tokenizer.json` (`model.merges`).
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Result<Self, BpeError> {
        let text = fs::read_to_string(path.as_ref())?;
        Self::from_tokenizer_json_str(&text)
    }

    pub fn from_tokenizer_json_str(content: &str) -> Result<Self, BpeError> {
        let v: serde_json::Value = serde_json::from_str(content)?;
        let merges = v
            .get("model")
            .and_then(|m| m.get("merges"))
            .and_then(|m| m.as_array())
            .ok_or(BpeError::InvalidTokenizerJson("missing model.merges array"))?;
        let mut ranks = HashMap::new();
        for (idx, item) in merges.iter().enumerate() {
            let line = item
                .as_str()
                .ok_or(BpeError::InvalidTokenizerJson("merge entry is not a string"))?;
            let line_no = idx + 1;
            let (left, right) = parse_merge_line(line, line_no)?;
            ranks.insert((left.to_string(), right.to_string()), idx as u32);
        }
        Ok(Self { ranks })
    }

    /// Apply BPE merges until no adjacent pair appears in the merge table.
    /// Starts from one symbol per Unicode scalar value (`char`).
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.ranks.get(&key) {
                    match best {
                        None => best = Some((i, rank)),
                        Some((bi, br)) => {
                            if rank < br || (rank == br && i < bi) {
                                best = Some((i, rank));
                            }
                        }
                    }
                }
            }
            let Some((i, _)) = best else {
                break;
            };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols[i] = merged;
            symbols.remove(i + 1);
        }
        symbols
    }

    /// Returns true exactly when raw BPE tokenization of `a + b` is `[a, b]`.
    pub fn canonical_pair(&self, a: &str, b: &str) -> bool {
        let mut text = String::with_capacity(a.len() + b.len());
        text.push_str(a);
        text.push_str(b);
        self.tokenize(&text) == [a.to_string(), b.to_string()]
    }
}

/// TinyLlama-compatible encoding for a **single word**: no whitespace in `text`.
/// Matches `transformers.AutoTokenizer.encode(text, add_special_tokens=False)` for such strings.
#[derive(Debug, Clone)]
pub struct TinyLlamaWordTokenizer {
    merges: BpeMerges,
    vocab: HashMap<String, u32>,
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
        for (token, id_val) in vocab_obj {
            let id = id_val
                .as_u64()
                .or_else(|| id_val.as_i64().map(|i| i as u64))
                .ok_or(BpeError::InvalidTokenizerJson("vocab id is not an integer"))?;
            if id > u32::MAX as u64 {
                return Err(BpeError::InvalidTokenizerJson("vocab id does not fit u32"));
            }
            vocab.insert(token.clone(), id as u32);
        }
        Ok(Self { merges, vocab })
    }

    /// BPE surface strings (e.g. `▁hello`, or `▁abc` + `def` for `abcdef`).
    pub fn tokenize_word(&self, text: &str) -> Result<Vec<String>, BpeError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        if text.chars().any(|c| c.is_whitespace()) {
            return Err(BpeError::WhitespaceInWord);
        }
        let mut s = String::with_capacity(text.len() + 4);
        if !text.starts_with(SPACESYMBOL) {
            s.push(SPACESYMBOL);
        }
        s.push_str(text);
        Ok(self.merges.tokenize(&s))
    }

    /// True iff `token` is present in the tokenizer vocabulary.
    pub fn is_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Token ids for a single space-free word (no special tokens).
    pub fn encode_word(&self, text: &str) -> Result<Vec<u32>, BpeError> {
        let pieces = self.tokenize_word(text)?;
        let mut ids = Vec::with_capacity(pieces.len());
        for p in pieces {
            let id = self
                .vocab
                .get(&p)
                .copied()
                .ok_or_else(|| BpeError::UnknownPiece(p))?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// TinyLlama pieces together with their vocab ids for one space-free word.
    pub fn encode_word_with_pieces(&self, text: &str) -> Result<Vec<(String, u32)>, BpeError> {
        let pieces = self.tokenize_word(text)?;
        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let id = self
                .vocab
                .get(&piece)
                .copied()
                .ok_or_else(|| BpeError::UnknownPiece(piece.clone()))?;
            out.push((piece, id));
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
        assert!(tok.is_token("▁hello"));
        assert!(!tok.is_token("not_a_real_token"));
        assert_eq!(
            tok.encode_word_with_pieces("abcdef").unwrap(),
            vec![("▁abc".to_string(), 25638), ("def".to_string(), 1753)]
        );
    }
}
