use std::collections::HashMap;
use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use bayesian::bpe::{
    NUM_PREFIXES, NUM_TOKENS, SPACESYMBOL, TOKENIZER_JSON_PATH, TinyLlamaWordTokenizer,
};

// Keep this binary independent of the trie module.
const TRIE_MAX_TOKEN_LENGTH: usize = 16;

#[derive(Debug, Clone)]
struct Witness {
    suffix_chars: usize,
    suffix_after_j: String,
    h_chars: usize,
    token_suffix_before_h: String,
    middle_tokens: Vec<String>,
}

#[derive(Debug, Clone)]
struct BestResult {
    count: usize,
    token: String,
    prefix: String,
    x: String,
    witnesses: Vec<Witness>,
}

#[derive(Debug, Clone)]
struct EvalSummary {
    best: BestResult,
    prefix_best_histogram: Vec<u64>,
    all_pairs_histogram: Vec<u64>,
}

#[derive(Debug, Clone)]
struct PartialScan {
    best: BestResult,
    prefix_best_counts: Vec<usize>,
    all_pairs_histogram: Vec<u64>,
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let tokenizer_path = args
        .next()
        .unwrap_or_else(|| TOKENIZER_JSON_PATH.to_string());
    let thread_count = args
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10)
        .max(1);

    let tokenizer = TinyLlamaWordTokenizer::from_tokenizer_json(&tokenizer_path);
    let started = Instant::now();

    if tokenizer.tokens().len() != NUM_TOKENS {
        eprintln!(
            "tokenizer token count mismatch: expected {}, got {}. use the tokenizer at {:?} or update tokenizer_config.rs",
            NUM_TOKENS,
            tokenizer.tokens().len(),
            TOKENIZER_JSON_PATH,
        );
        return ExitCode::from(2);
    }
    if tokenizer.prefix_count() != NUM_PREFIXES {
        eprintln!(
            "tokenizer prefix count mismatch: expected {}, got {}. use the tokenizer at {:?} or update tokenizer_config.rs",
            NUM_PREFIXES,
            tokenizer.prefix_count(),
            TOKENIZER_JSON_PATH,
        );
        return ExitCode::from(2);
    }

    eprintln!(
        "loaded tokenizer: {} tokens, {} prefixes, {} worker threads",
        tokenizer.tokens().len(),
        tokenizer.prefix_count(),
        thread_count
    );

    let suffix_index = build_suffix_index(&tokenizer);
    eprintln!(
        "built suffix index: {} suffixes",
        suffix_index.suffixes.len()
    );
    let (xi_rows, zeta_rows) = build_xi_and_zeta_rows(&tokenizer, &suffix_index, thread_count);
    let summary = evaluate_strict_upper_bound(
        &tokenizer,
        &suffix_index,
        &xi_rows,
        &zeta_rows,
        thread_count,
    );

    println!("best_count={}", summary.best.count);
    println!("token={:?}", summary.best.token);
    println!("prefix={:?}", summary.best.prefix);
    println!("x={:?}", summary.best.x);
    println!("elapsed={:.2?}", started.elapsed());
    for witness in &summary.best.witnesses {
        println!(
            "suffix_chars={} suffix_after_j={:?} h_chars={} token_suffix_before_h={:?} middle_tokens={:?}",
            witness.suffix_chars,
            witness.suffix_after_j,
            witness.h_chars,
            witness.token_suffix_before_h,
            witness.middle_tokens,
        );
    }
    for (count, freq) in summary.prefix_best_histogram.iter().enumerate() {
        if *freq != 0 {
            println!("prefix_best_histogram count={} freq={}", count, freq);
        }
    }
    for (count, freq) in summary.all_pairs_histogram.iter().enumerate() {
        if *freq != 0 {
            println!("all_pairs_histogram count={} freq={}", count, freq);
        }
    }

    ExitCode::SUCCESS
}

#[derive(Debug)]
struct SuffixIndex {
    suffixes: Vec<String>,
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
        suffixes,
        suffix_to_index,
        suffix_indices_by_token_lex_index,
    }
}

fn build_xi_and_zeta_rows(
    tokenizer: &TinyLlamaWordTokenizer,
    suffix_index: &SuffixIndex,
    thread_count: usize,
) -> (Vec<BitRow>, Vec<BitRow>) {
    let token_count = tokenizer.tokens().len();
    let suffix_count = suffix_index.suffixes.len();
    let prefix_word_count = bit_words(NUM_PREFIXES);
    let token_word_count = bit_words(NUM_TOKENS);
    let chunks = make_chunks(token_count, thread_count);
    let processed = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));
    let reporter = spawn_progress_reporter(
        "precompute xi/zeta",
        processed.clone(),
        token_count as u64,
        done.clone(),
    );

    let mut parts = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunks.len());
        for &(start, end) in &chunks {
            let processed = processed.clone();
            handles.push(scope.spawn(move || {
                let mut xi_rows = Vec::with_capacity(end - start);
                let mut zeta_rows = vec![vec![0u64; token_word_count]; suffix_count];
                for first_lex_index in start..end {
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
                    for &suffix_lex_index in
                        &suffix_index.suffix_indices_by_token_lex_index[first_lex_index]
                    {
                        or_assign_bits(&mut zeta_rows[suffix_lex_index], &psi_row);
                    }
                    xi_rows.push(xi_row);
                    processed.fetch_add(1, Ordering::Relaxed);
                }
                (start, xi_rows, zeta_rows)
            }));
        }

        let mut parts = Vec::with_capacity(handles.len());
        for handle in handles {
            parts.push(handle.join().expect("xi/zeta worker thread panicked"));
        }
        parts
    });

    done.store(true, Ordering::Relaxed);
    reporter.join().expect("progress reporter panicked");

    parts.sort_by_key(|(start, _, _)| *start);
    let mut xi_rows = Vec::with_capacity(token_count);
    let mut zeta_rows = vec![vec![0u64; token_word_count]; suffix_count];
    for (_, mut part_xi_rows, part_zeta_rows) in parts.drain(..) {
        xi_rows.append(&mut part_xi_rows);
        for (suffix_lex_index, part_row) in part_zeta_rows.into_iter().enumerate() {
            or_assign_bits(&mut zeta_rows[suffix_lex_index], &part_row);
        }
    }
    (xi_rows, zeta_rows)
}

fn evaluate_strict_upper_bound(
    tokenizer: &TinyLlamaWordTokenizer,
    suffix_index: &SuffixIndex,
    xi_rows: &[BitRow],
    zeta_rows: &[BitRow],
    thread_count: usize,
) -> EvalSummary {
    let tokens = tokenizer.tokens();
    let prefixes: Vec<&String> = tokenizer
        .prefixes()
        .iter()
        .filter(|prefix| !prefix.starts_with(SPACESYMBOL))
        .collect();
    let total_pairs = (tokens.len() as u128) * (prefixes.len() as u128);
    let chunks = make_chunks(tokens.len(), thread_count);
    let processed_pairs = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));
    let reported_best = Arc::new(Mutex::new(BestResult {
        count: 0,
        token: String::new(),
        prefix: String::new(),
        x: String::new(),
        witnesses: Vec::new(),
    }));
    let reporter = spawn_progress_reporter(
        "strict scan",
        processed_pairs.clone(),
        total_pairs as u64,
        done.clone(),
    );

    let partials = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunks.len());
        for &(start, end) in &chunks {
            let processed_pairs = processed_pairs.clone();
            let prefixes = prefixes.clone();
            let reported_best = reported_best.clone();
            handles.push(scope.spawn(move || {
                let mut partial = PartialScan {
                    best: BestResult {
                        count: 0,
                        token: String::new(),
                        prefix: String::new(),
                        x: String::new(),
                        witnesses: Vec::new(),
                    },
                    prefix_best_counts: vec![0usize; prefixes.len()],
                    all_pairs_histogram: vec![0u64; TRIE_MAX_TOKEN_LENGTH + 1],
                };

                for token in &tokens[start..end] {
                    for (prefix_idx, &prefix) in prefixes.iter().enumerate() {
                        let mut x = String::with_capacity(token.len() + prefix.len());
                        x.push_str(token);
                        x.push_str(prefix);

                        let char_starts = char_boundaries(&x);
                        let n_chars = char_starts.len() - 1;
                        let max_suffix_chars = TRIE_MAX_TOKEN_LENGTH.min(n_chars);

                        let mut count = 0usize;
                        let mut witnesses = Vec::new();

                        for suffix_chars in 1..=max_suffix_chars {
                            let suffix_start_char = n_chars - suffix_chars;
                            let suffix_after_j =
                                slice_by_char_range(&x, &char_starts, suffix_start_char, n_chars);
                            let Some(prefix_lex_index) = tokenizer.prefix_lex_index(suffix_after_j)
                            else {
                                continue;
                            };

                            let mut witness = None;
                            let max_h_chars = TRIE_MAX_TOKEN_LENGTH.min(suffix_start_char);
                            for h_chars in 0..=max_h_chars {
                                let token_suffix_before_h =
                                    slice_by_char_range(&x, &char_starts, 0, h_chars);
                                let Some(&suffix_lex_index) =
                                    suffix_index.suffix_to_index.get(token_suffix_before_h)
                                else {
                                    continue;
                                };

                                let middle = slice_by_char_range(
                                    &x,
                                    &char_starts,
                                    h_chars,
                                    suffix_start_char,
                                );
                                if middle.is_empty() {
                                    continue;
                                }
                                let middle_token_lex_indices =
                                    tokenizer.tokenize_string_to_lex_indices(middle);
                                if middle_token_lex_indices.is_empty() {
                                    continue;
                                }

                                let first_middle_token_lex_index = middle_token_lex_indices[0];
                                let last_middle_token_lex_index = *middle_token_lex_indices
                                    .last()
                                    .expect("middle tokenization should be non-empty");

                                if get_bit(&xi_rows[last_middle_token_lex_index], prefix_lex_index)
                                    && get_bit(
                                        &zeta_rows[suffix_lex_index],
                                        first_middle_token_lex_index,
                                    )
                                {
                                    let middle_tokens = middle_token_lex_indices
                                        .iter()
                                        .map(|&lex_index| tokenizer.token_at(lex_index).to_string())
                                        .collect::<Vec<_>>();
                                    witness = Some(Witness {
                                        suffix_chars,
                                        suffix_after_j: suffix_after_j.to_string(),
                                        h_chars,
                                        token_suffix_before_h: token_suffix_before_h.to_string(),
                                        middle_tokens,
                                    });
                                    break;
                                }
                            }

                            if let Some(witness) = witness {
                                count += 1;
                                witnesses.push(witness);
                            }
                        }

                        partial.all_pairs_histogram[count] += 1;
                        partial.prefix_best_counts[prefix_idx] =
                            partial.prefix_best_counts[prefix_idx].max(count);

                        if count > partial.best.count {
                            partial.best = BestResult {
                                count,
                                token: token.clone(),
                                prefix: prefix.clone(),
                                x,
                                witnesses,
                            };
                            maybe_report_new_best(&reported_best, &partial.best);
                        }

                        processed_pairs.fetch_add(1, Ordering::Relaxed);
                    }
                }

                partial
            }));
        }

        let mut partials = Vec::with_capacity(handles.len());
        for handle in handles {
            partials.push(handle.join().expect("scan worker thread panicked"));
        }
        partials
    });

    done.store(true, Ordering::Relaxed);
    reporter.join().expect("progress reporter panicked");

    let mut best = BestResult {
        count: 0,
        token: String::new(),
        prefix: String::new(),
        x: String::new(),
        witnesses: Vec::new(),
    };
    let mut prefix_best_counts = vec![0usize; prefixes.len()];
    let mut all_pairs_histogram = vec![0u64; TRIE_MAX_TOKEN_LENGTH + 1];

    for partial in partials {
        if partial.best.count > best.count {
            best = partial.best;
        }
        for (global, local) in prefix_best_counts
            .iter_mut()
            .zip(partial.prefix_best_counts.into_iter())
        {
            *global = (*global).max(local);
        }
        for (global, local) in all_pairs_histogram
            .iter_mut()
            .zip(partial.all_pairs_histogram.into_iter())
        {
            *global += local;
        }
    }

    let mut prefix_best_histogram = vec![0u64; TRIE_MAX_TOKEN_LENGTH + 1];
    for count in prefix_best_counts {
        prefix_best_histogram[count] += 1;
    }

    EvalSummary {
        best,
        prefix_best_histogram,
        all_pairs_histogram,
    }
}

fn maybe_report_new_best(reported_best: &Mutex<BestResult>, candidate: &BestResult) {
    let mut guard = reported_best
        .lock()
        .expect("best-result mutex should not be poisoned");
    if candidate.count <= guard.count {
        return;
    }

    *guard = candidate.clone();
    eprintln!(
        "strict scan new best: count={} token={:?} prefix={:?} x={:?} witnesses={}",
        candidate.count,
        candidate.token,
        candidate.prefix,
        candidate.x,
        format_witnesses(&candidate.witnesses),
    );
}

fn format_witnesses(witnesses: &[Witness]) -> String {
    witnesses
        .iter()
        .take(4)
        .map(|witness| {
            format!(
                "{{suffix_chars={}, suffix_after_j={:?}, h_chars={}, token_suffix_before_h={:?}, middle_tokens={:?}}}",
                witness.suffix_chars,
                witness.suffix_after_j,
                witness.h_chars,
                witness.token_suffix_before_h,
                witness.middle_tokens,
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
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

fn make_chunks(len: usize, thread_count: usize) -> Vec<(usize, usize)> {
    let worker_count = thread_count.min(len.max(1));
    let chunk_size = len.div_ceil(worker_count);
    let mut chunks = Vec::with_capacity(worker_count);
    let mut start = 0usize;
    while start < len {
        let end = (start + chunk_size).min(len);
        chunks.push((start, end));
        start = end;
    }
    if chunks.is_empty() {
        chunks.push((0, 0));
    }
    chunks
}

fn spawn_progress_reporter(
    label: &'static str,
    processed: Arc<AtomicU64>,
    total: u64,
    done: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let started = Instant::now();
        let mut last_reported = 0u64;
        loop {
            thread::sleep(Duration::from_secs(1));
            let done_now = done.load(Ordering::Relaxed);
            let processed_now = processed.load(Ordering::Relaxed);
            if processed_now != last_reported || done_now {
                let pct = if total == 0 {
                    100.0
                } else {
                    100.0 * processed_now as f64 / total as f64
                };
                eprintln!(
                    "{}: {}/{} ({:.3}%) elapsed={:.2?}",
                    label,
                    processed_now,
                    total,
                    pct,
                    started.elapsed()
                );
                last_reported = processed_now;
            }
            if done_now {
                break;
            }
        }
    })
}
