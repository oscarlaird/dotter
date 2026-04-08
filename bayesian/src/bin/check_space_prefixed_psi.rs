use std::process::ExitCode;

use bayesian::bpe::{SPACESYMBOL, TOKENIZER_JSON_STR, TinyLlamaWordTokenizer};
use bayesian::trie::TokenLexIndex;

fn is_all_space_symbol(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|c| c == SPACESYMBOL)
}

fn main() -> ExitCode {
    let tok = TinyLlamaWordTokenizer::from_tokenizer_json_str(TOKENIZER_JSON_STR);
    let space_prefixed_second_lex_indices: Vec<usize> = tok
        .tokens()
        .iter()
        .enumerate()
        .filter_map(|(lex_index, token)| token.starts_with(SPACESYMBOL).then_some(lex_index))
        .collect();

    let mut total_counterexamples = 0usize;
    let mut all_space_symbol_counterexamples = 0usize;
    let mut non_all_space_symbol_samples = Vec::new();
    let mut all_space_symbol_samples = Vec::new();

    for first_lex_index in 0..tok.tokens().len() {
        let first_lex_index = TokenLexIndex::from_usize(first_lex_index);
        let first = tok.token_at(first_lex_index);
        let mask = tok.canonical_followers_for_lex_index(first_lex_index);
        for &second_lex_index in &space_prefixed_second_lex_indices {
            let second_lex_index = TokenLexIndex::from_usize(second_lex_index);
            let second = tok.token_at(second_lex_index);
            if !mask[second_lex_index.as_usize()] {
                total_counterexamples += 1;
                if is_all_space_symbol(first) && is_all_space_symbol(second) {
                    all_space_symbol_counterexamples += 1;
                    if all_space_symbol_samples.len() < 20 {
                        all_space_symbol_samples.push((first.to_string(), second.to_string()));
                    }
                } else if non_all_space_symbol_samples.len() < 20 {
                    non_all_space_symbol_samples.push((first.to_string(), second.to_string()));
                }
            }
        }
    }

    println!("total_counterexamples={total_counterexamples}");
    println!("all_space_symbol_counterexamples={all_space_symbol_counterexamples}");
    println!(
        "non_all_space_symbol_counterexamples={}",
        total_counterexamples - all_space_symbol_counterexamples
    );
    for (first, second) in all_space_symbol_samples {
        println!("all_space_symbol_sample first={first:?} second={second:?}");
    }
    for (first, second) in non_all_space_symbol_samples {
        println!("non_all_space_symbol_sample first={first:?} second={second:?}");
    }

    ExitCode::SUCCESS
}
