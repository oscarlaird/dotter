pub const NUM_TOKENS: usize = 17_235;
pub const NUM_PREFIXES: usize = 29_915;
pub const TOKENIZER_JSON_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tokenizers/tinyllamaalpha/tokenizer.json"
);
pub const TOKENIZER_JSON_STR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tokenizers/tinyllamaalpha/tokenizer.json"
));
