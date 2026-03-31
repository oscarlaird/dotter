use std::collections::HashMap;
use std::process::ExitCode;

use bayesian::bpe::{NUM_PREFIXES, NUM_TOKENS, TOKENIZER_JSON_PATH, TinyLlamaWordTokenizer};

const TRIE_MAX_TOKEN_LENGTH: usize = 16;

fn main() -> ExitCode {
    let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json(TOKENIZER_JSON_PATH);
    let suffix_index = build_suffix_index(&tokenizer);
    let (xi_rows, zeta_rows) = build_xi_and_zeta_rows(&tokenizer, &suffix_index);

    let token = "db";
    let prefix = "tikzpictur";
    let x = format!("{token}{prefix}");
    let char_starts = char_boundaries(&x);
    let n_chars = char_starts.len() - 1;
    let max_suffix_chars = TRIE_MAX_TOKEN_LENGTH.min(n_chars);

    println!("token={token:?}");
    println!("prefix={prefix:?}");
    println!("x={x:?}");

    for suffix_chars in 1..=max_suffix_chars {
        let suffix_start_char = n_chars - suffix_chars;
        let suffix_after_j = slice_by_char_range(&x, &char_starts, suffix_start_char, n_chars);
        let Some(prefix_lex_index) = tokenizer.prefix_lex_index(suffix_after_j) else {
            continue;
        };

        let max_h_chars = TRIE_MAX_TOKEN_LENGTH.min(suffix_start_char);
        for h_chars in 0..=max_h_chars {
            let token_suffix_before_h = slice_by_char_range(&x, &char_starts, 0, h_chars);
            let Some(&suffix_lex_index) = suffix_index.suffix_to_index.get(token_suffix_before_h)
            else {
                continue;
            };

            let middle = slice_by_char_range(&x, &char_starts, h_chars, suffix_start_char);
            if middle.is_empty() {
                continue;
            }
            let middle_token_lex_indices = tokenizer.tokenize_string_to_lex_indices(middle);
            if middle_token_lex_indices.is_empty() {
                continue;
            }

            let first_middle_token_lex_index = middle_token_lex_indices[0];
            let last_middle_token_lex_index = *middle_token_lex_indices.last().unwrap();

            if get_bit(&xi_rows[last_middle_token_lex_index], prefix_lex_index)
                && get_bit(&zeta_rows[suffix_lex_index], first_middle_token_lex_index)
            {
                let middle_tokens = middle_token_lex_indices
                    .iter()
                    .map(|&lex_index| tokenizer.token_at(lex_index).to_string())
                    .collect::<Vec<_>>();
                let xi_witness_token = find_xi_witness_token(
                    &tokenizer,
                    last_middle_token_lex_index,
                    prefix_lex_index,
                )
                .expect("xi row promised at least one token witness");
                let zeta_witness_token = find_zeta_witness_token(
                    &tokenizer,
                    token_suffix_before_h,
                    first_middle_token_lex_index,
                );
                println!(
                    "witness suffix_chars={} suffix_after_j={:?} h_chars={} token_suffix_before_h={:?} middle_tokens={:?} xi_witness_token={:?} zeta_witness_token={:?}",
                    suffix_chars,
                    suffix_after_j,
                    h_chars,
                    token_suffix_before_h,
                    middle_tokens,
                    xi_witness_token,
                    zeta_witness_token,
                );
                break;
            }
        }
    }

    ExitCode::SUCCESS
}

#[derive(Debug)]
struct SuffixIndex {
    suffix_to_index: HashMap<String, usize>,
    suffix_indices_by_token_lex_index: Vec<Vec<usize>>,
}

type BitRow = Vec<u64>;

fn build_suffix_index(tokenizer: &TinyLlamaWordTokenizer) -> SuffixIndex {
    let tokens = tokenizer.tokens();
    let mut suffixes = Vec::new();
    suffixes.push(String::new());
    for token in tokens {
        for suffix in token_suffixes(token) {
            suffixes.push(suffix.to_string());
        }
    }
    suffixes.sort_unstable();
    suffixes.dedup();

    let mut suffix_to_index = HashMap::with_capacity(suffixes.len());
    for (index, suffix) in suffixes.iter().enumerate() {
        suffix_to_index.insert(suffix.clone(), index);
    }

    let mut suffix_indices_by_token_lex_index = Vec::with_capacity(tokens.len());
    for token in tokens {
        let mut indices = Vec::new();
        for suffix in token_suffixes(token) {
            indices.push(
                *suffix_to_index
                    .get(suffix)
                    .expect("token suffix should have been indexed"),
            );
        }
        indices.push(
            *suffix_to_index
                .get("")
                .expect("empty suffix should always be indexed"),
        );
        indices.sort_unstable();
        indices.dedup();
        suffix_indices_by_token_lex_index.push(indices);
    }

    SuffixIndex {
        suffix_to_index,
        suffix_indices_by_token_lex_index,
    }
}

fn build_xi_and_zeta_rows(
    tokenizer: &TinyLlamaWordTokenizer,
    suffix_index: &SuffixIndex,
) -> (Vec<BitRow>, Vec<BitRow>) {
    let token_count = tokenizer.tokens().len();
    let suffix_count = suffix_index.suffix_to_index.len();
    let prefix_word_count = bit_words(NUM_PREFIXES);
    let token_word_count = bit_words(NUM_TOKENS);

    let mut xi_rows = Vec::with_capacity(token_count);
    let mut zeta_rows = vec![vec![0u64; token_word_count]; suffix_count];
    for first_lex_index in 0..token_count {
        let psi = tokenizer.canonical_followers_for_lex_index(first_lex_index);
        let prefix_counts = tokenizer.count_true_tokens_by_prefix::<NUM_PREFIXES>(&psi);
        let mut xi_row = vec![0u64; prefix_word_count];
        for (prefix_lex_index, &count) in prefix_counts.iter().enumerate() {
            if count != 0 {
                set_bit(&mut xi_row, prefix_lex_index);
            }
        }
        let mut psi_row = vec![0u64; token_word_count];
        for (second_lex_index, &is_canonical) in psi.iter().enumerate() {
            if is_canonical {
                set_bit(&mut psi_row, second_lex_index);
            }
        }
        for &suffix_lex_index in &suffix_index.suffix_indices_by_token_lex_index[first_lex_index] {
            or_assign_bits(&mut zeta_rows[suffix_lex_index], &psi_row);
        }
        xi_rows.push(xi_row);
    }
    (xi_rows, zeta_rows)
}

fn token_suffixes(token: &str) -> Vec<&str> {
    let mut suffixes = Vec::with_capacity(token.chars().count() + 1);
    for (idx, _) in token.char_indices() {
        suffixes.push(&token[idx..]);
    }
    if token.is_empty() {
        suffixes.push(token);
    }
    suffixes
}

fn char_boundaries(text: &str) -> Vec<usize> {
    let mut starts: Vec<usize> = text.char_indices().map(|(idx, _)| idx).collect();
    starts.push(text.len());
    starts
}

fn slice_by_char_range<'a>(
    text: &'a str,
    char_starts: &[usize],
    start_char: usize,
    end_char: usize,
) -> &'a str {
    &text[char_starts[start_char]..char_starts[end_char]]
}

fn find_xi_witness_token(
    tokenizer: &TinyLlamaWordTokenizer,
    last_middle_token_lex_index: usize,
    prefix_lex_index: usize,
) -> Option<String> {
    let psi_row = tokenizer.canonical_followers_for_lex_index(last_middle_token_lex_index);
    let (start, stop) = tokenizer.token_lex_range_for_prefix_index(prefix_lex_index);
    (start..stop)
        .find(|&second_lex_index| psi_row[second_lex_index])
        .map(|second_lex_index| tokenizer.token_at(second_lex_index).to_string())
}

fn find_zeta_witness_token(
    tokenizer: &TinyLlamaWordTokenizer,
    token_suffix_before_h: &str,
    first_middle_token_lex_index: usize,
) -> Option<String> {
    let second_token = tokenizer.token_at(first_middle_token_lex_index);
    tokenizer
        .tokens()
        .iter()
        .find(|candidate| {
            candidate.ends_with(token_suffix_before_h)
                && tokenizer.can_canonically_follow(candidate, second_token)
        })
        .cloned()
}

fn bit_words(len: usize) -> usize {
    len.div_ceil(64)
}

fn set_bit(bits: &mut [u64], index: usize) {
    bits[index / 64] |= 1u64 << (index % 64);
}

fn get_bit(bits: &[u64], index: usize) -> bool {
    ((bits[index / 64] >> (index % 64)) & 1) != 0
}

fn or_assign_bits(dst: &mut [u64], src: &[u64]) {
    for (dst_word, src_word) in dst.iter_mut().zip(src.iter()) {
        *dst_word |= *src_word;
    }
}
