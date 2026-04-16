pub const NUM_TOKENS: usize = 17_268;
pub const NUM_PREFIXES: usize = 29_959;
pub const TOKENIZER_JSON_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tokenizers/tinyllamaalpha/tokenizer.json"
);
pub const TOKENIZER_JSON_STR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tokenizers/tinyllamaalpha/tokenizer.json"
));
