//! Run during development:
//!   `cargo run -- <tokenizer.json> <word>...`
//!   `cargo run -- canonical-pair-rate <tokenizer.json> [samples] [seed] [--first-no-sp] [--first-has-sp] [--second-no-space] [--second-no-sp] [--second-has-sp]`

use std::env;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use bayesian::bpe::TinyLlamaWordTokenizer;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);

    let Some(first_arg) = args.next() else {
        print_usage();
        return ExitCode::from(2);
    };

    if first_arg == "canonical-pair-rate" {
        return canonical_pair_rate_cli(args);
    }

    let tokenizer_path = first_arg;
    let words: Vec<String> = args.collect();
    if words.is_empty() {
        print_usage();
        return ExitCode::from(2);
    }

    encode_words_cli(&tokenizer_path, &words)
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  cargo run -- <tokenizer.json> <word>...");
    eprintln!(
        "  cargo run -- canonical-pair-rate <tokenizer.json> [samples] [seed] [--first-no-sp] [--first-has-sp] [--second-no-space] [--second-no-sp] [--second-has-sp]"
    );
}

fn encode_words_cli(tokenizer_path: &str, words: &[String]) -> ExitCode {
    let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json(tokenizer_path);

    for word in words {
        let encoded = tokenizer.tokenize_word_with_lex_indices(word);
        let pieces: Vec<&str> = encoded.iter().map(|(piece, _)| piece.as_str()).collect();
        let lex_indices: Vec<usize> = encoded.iter().map(|(_, idx)| *idx).collect();
        println!("{word:?}");
        println!("  pieces: {pieces:?}");
        println!("  lex_indices: {lex_indices:?}");
    }

    ExitCode::SUCCESS
}

fn canonical_pair_rate_cli(mut args: impl Iterator<Item = String>) -> ExitCode {
    let Some(tokenizer_path) = args.next() else {
        print_usage();
        return ExitCode::from(2);
    };

    let samples = match args.next() {
        Some(value) => match value.parse::<u64>() {
            Ok(samples) if samples > 0 => samples,
            _ => {
                eprintln!("samples must be a positive integer");
                return ExitCode::from(2);
            }
        },
        None => 100_000,
    };

    let seed = match args.next() {
        Some(value) => match value.parse::<u64>() {
            Ok(seed) => seed,
            Err(_) => {
                eprintln!("seed must be an integer");
                return ExitCode::from(2);
            }
        },
        None => default_seed(),
    };

    let mut first_no_sp = false;
    let mut first_has_sp = false;
    let mut second_no_space = false;
    let mut second_no_sp = false;
    let mut second_has_sp = false;
    for arg in args {
        match arg.as_str() {
            "--first-no-sp" => first_no_sp = true,
            "--first-has-sp" => first_has_sp = true,
            "--second-no-space" => second_no_space = true,
            "--second-no-sp" => second_no_sp = true,
            "--second-has-sp" => second_has_sp = true,
            _ => {
                eprintln!("unknown option: {arg}");
                print_usage();
                return ExitCode::from(2);
            }
        }
    }

    let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json(&tokenizer_path);

    let vocab: Vec<&str> = tokenizer.tokens().iter().map(String::as_str).collect();
    if vocab.is_empty() {
        eprintln!("tokenizer vocab is empty");
        return ExitCode::from(1);
    }

    let first_filters = first_no_sp as u8 + first_has_sp as u8;
    if first_filters > 1 {
        eprintln!("choose at most one first-token filter");
        print_usage();
        return ExitCode::from(2);
    }

    let second_filters = second_no_space as u8 + second_no_sp as u8 + second_has_sp as u8;
    if second_filters > 1 {
        eprintln!("choose at most one second-token filter");
        print_usage();
        return ExitCode::from(2);
    }

    let first_vocab: Vec<&str> = if first_no_sp {
        vocab
            .iter()
            .copied()
            .filter(|token| !token.contains('▁'))
            .collect()
    } else if first_has_sp {
        vocab
            .iter()
            .copied()
            .filter(|token| token.contains('▁'))
            .collect()
    } else {
        vocab.clone()
    };
    if first_vocab.is_empty() {
        eprintln!("filtered first-token vocab is empty");
        return ExitCode::from(1);
    }

    let second_vocab: Vec<&str> = if second_no_space {
        vocab
            .iter()
            .copied()
            .filter(|token| !token.contains(' '))
            .collect()
    } else if second_no_sp {
        vocab
            .iter()
            .copied()
            .filter(|token| !token.contains('▁'))
            .collect()
    } else if second_has_sp {
        vocab
            .iter()
            .copied()
            .filter(|token| token.contains('▁'))
            .collect()
    } else {
        vocab.clone()
    };
    if second_vocab.is_empty() {
        eprintln!("filtered second-token vocab is empty");
        return ExitCode::from(1);
    }

    let mut rng = XorShift64::new(seed);
    let mut canonical = 0u64;

    for _ in 0..samples {
        let a = first_vocab[rng.gen_index(first_vocab.len())];
        let b = second_vocab[rng.gen_index(second_vocab.len())];
        if tokenizer.can_canonically_follow(a, b) {
            canonical += 1;
        }
    }

    let rate = canonical as f64 / samples as f64;
    let stderr = (rate * (1.0 - rate) / samples as f64).sqrt();

    println!("tokenizer: {tokenizer_path}");
    println!("vocab_size: {}", vocab.len());
    println!("first_vocab_size: {}", first_vocab.len());
    println!("second_vocab_size: {}", second_vocab.len());
    println!("first_no_sp: {first_no_sp}");
    println!("first_has_sp: {first_has_sp}");
    println!("second_no_space: {second_no_space}");
    println!("second_no_sp: {second_no_sp}");
    println!("second_has_sp: {second_has_sp}");
    println!("samples: {samples}");
    println!("seed: {seed}");
    println!("canonical: {canonical}");
    println!("rate: {rate:.6}");
    println!("stderr: {stderr:.6}");
    println!(
        "ci95: [{:.6}, {:.6}]",
        rate - 1.96 * stderr,
        rate + 1.96 * stderr
    );

    ExitCode::SUCCESS
}

fn default_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9_7f4a_7c15)
}

#[derive(Clone, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0xdead_beef
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn gen_index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }
}
