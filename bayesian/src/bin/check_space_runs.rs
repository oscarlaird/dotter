use std::process::ExitCode;

use bayesian::bpe::{SPACESYMBOL, TOKENIZER_JSON_STR, TinyLlamaWordTokenizer};

fn main() -> ExitCode {
    let tok = TinyLlamaWordTokenizer::from_tokenizer_json_str(TOKENIZER_JSON_STR);

    let mut missing = Vec::new();
    for len in 1..=16 {
        let run: String = std::iter::repeat_n(SPACESYMBOL, len).collect();
        if tok.lex_index(&run).is_none() {
            missing.push(run);
        }
    }

    println!("missing_count={}", missing.len());
    for token in missing {
        println!("missing_token={token:?}");
    }

    ExitCode::SUCCESS
}
