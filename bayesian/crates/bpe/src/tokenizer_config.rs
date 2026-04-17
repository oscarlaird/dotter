/// Lex / vocab size (`TinyLlamaWordTokenizer::tokens().len()`): LM logits, trie token arrays.
pub const NUM_TOKENS: usize = 25_471;
/// BPE piece table size (`BpeMerges.pieces.len()`, `MergeRows::token_count`): allpairs dense tables.
pub const NUM_MERGE_ROWS: usize = 25_509;
/// Distinct trie/BPE prefix strings (`TinyLlamaWordTokenizer::prefix_count()`).
pub const NUM_PREFIXES: usize = 46_506;
pub const TOKENIZER_JSON_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tokenizers/tinyllamaalpha/tokenizer.json"
);
pub const TOKENIZER_JSON_STR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tokenizers/tinyllamaalpha/tokenizer.json"
));
