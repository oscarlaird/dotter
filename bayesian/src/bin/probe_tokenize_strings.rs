use std::process::ExitCode;

use bayesian::bpe::{TOKENIZER_JSON_PATH, TinyLlamaWordTokenizer};

fn main() -> ExitCode {
    let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json(TOKENIZER_JSON_PATH);
    let strings = [
        "tikzpicturoid",
        "tikzpictur",
        "tikzpicturn",
        "tikzpictures",
        "tikzpicture",
        "atikzpicture",
    ];

    for text in strings {
        let pieces: Vec<String> = tokenizer
            .tokenize_string_with_lex_indices(text)
            .into_iter()
            .map(|(piece, _)| piece)
            .collect();
        println!("{text:?} -> {:?}", pieces);
    }

    ExitCode::SUCCESS
}
