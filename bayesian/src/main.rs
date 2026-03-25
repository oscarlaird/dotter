//! Run during development: `cargo run -- <tokenizer.json> <word>...`

use std::env;
use std::process::ExitCode;

use bayesian::bpe::TinyLlamaWordTokenizer;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(tokenizer_path) = args.next() else {
        eprintln!("usage: cargo run -- <tokenizer.json> <word>...");
        return ExitCode::from(2);
    };

    let words: Vec<String> = args.collect();
    if words.is_empty() {
        eprintln!("usage: cargo run -- <tokenizer.json> <word>...");
        return ExitCode::from(2);
    }

    let tokenizer = match TinyLlamaWordTokenizer::from_tokenizer_json(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!("failed to load tokenizer: {err}");
            return ExitCode::from(1);
        }
    };

    for word in words {
        match tokenizer.encode_word_with_pieces(&word) {
            Ok(encoded) => {
                let pieces: Vec<&str> = encoded.iter().map(|(piece, _)| piece.as_str()).collect();
                let ids: Vec<u32> = encoded.iter().map(|(_, id)| *id).collect();
                println!("{word:?}");
                println!("  pieces: {pieces:?}");
                println!("  ids:    {ids:?}");
            }
            Err(err) => {
                eprintln!("failed to encode {word:?}: {err}");
                return ExitCode::from(1);
            }
        }
    }

    ExitCode::SUCCESS
}
