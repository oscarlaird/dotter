use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::process::ExitCode;

use bayesian::bpe::{BpeMerges, TinyLlamaWordTokenizer};
use serde_json::Value;

#[derive(Clone, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0xdead_beef_cafe_babe
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

    fn gen_index(&mut self, len: usize) -> usize {
        (self.next_u64() % len as u64) as usize
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(tokenizer_path) = args.next() else {
        eprintln!(
            "usage: cargo run --release --bin lookup_row_hist -- <tokenizer.json> [samples] [seed]"
        );
        return ExitCode::from(2);
    };
    let samples = match args.next() {
        Some(value) => match value.parse::<usize>() {
            Ok(samples) if samples > 0 => samples,
            _ => {
                eprintln!("samples must be a positive integer");
                return ExitCode::from(2);
            }
        },
        None => 100,
    };
    let seed = match args.next() {
        Some(value) => match value.parse::<u64>() {
            Ok(seed) => seed,
            Err(_) => {
                eprintln!("seed must be an integer");
                return ExitCode::from(2);
            }
        },
        None => 1,
    };

    let tokenizer = match TinyLlamaWordTokenizer::from_tokenizer_json(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!("failed to load tokenizer: {err}");
            return ExitCode::from(1);
        }
    };
    let merges = match BpeMerges::from_tokenizer_json(&tokenizer_path) {
        Ok(merges) => merges,
        Err(err) => {
            eprintln!("failed to load merges: {err}");
            return ExitCode::from(1);
        }
    };

    let (id_to_piece, out_degree, in_degree) = match load_degrees(&tokenizer_path, &merges) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let candidate_second_ids = tokenizer.token_ids_with_left_spines().to_vec();
    let mut rng = XorShift64::new(seed);
    let sampled_first_ids: Vec<u32> = (0..samples)
        .map(|_| candidate_second_ids[rng.gen_index(candidate_second_ids.len())])
        .collect();

    let mut first_hist = BTreeMap::<u32, u64>::new();
    let mut second_hist = BTreeMap::<u32, u64>::new();
    let mut first_piece_hits = vec![0u64; id_to_piece.len()];
    let mut second_piece_hits = vec![0u64; id_to_piece.len()];
    let mut lookup_events = 0u64;

    for first_id in sampled_first_ids {
        let Some(right_spine) = tokenizer.right_spine_for_token_id(first_id) else {
            continue;
        };
        for &second_id in &candidate_second_ids {
            let left_spine = tokenizer
                .left_spine_for_token_id(second_id)
                .expect("candidate second ids must have left spines");

            let mut i = 0usize;
            let mut j = 0usize;
            loop {
                let right_id = right_spine[i].id as usize;
                let left_id = left_spine[j].id as usize;

                let first_row = out_degree[right_id];
                let second_row = in_degree[left_id];
                *first_hist.entry(first_row).or_default() += 1;
                *second_hist.entry(second_row).or_default() += 1;
                first_piece_hits[right_id] += 1;
                second_piece_hits[left_id] += 1;
                lookup_events += 1;

                let right_priority_score = right_spine[i].priority_score as u32;
                let left_priority_score = left_spine[j].priority_score as u32;
                let cross_priority_score = merges
                    .lookup_merge_by_pair(right_spine[i].id as u32, left_spine[j].id as u32)
                    .map_or(0, |entry| u32::MAX - entry.rank);
                let mut best_priority_score = right_priority_score;
                if left_priority_score > best_priority_score {
                    best_priority_score = left_priority_score;
                }
                if cross_priority_score > best_priority_score {
                    best_priority_score = cross_priority_score;
                }

                if best_priority_score == 0 {
                    break;
                }

                if cross_priority_score == best_priority_score {
                    break;
                }
                if right_priority_score == best_priority_score {
                    i += 1;
                    continue;
                }
                if left_priority_score == best_priority_score {
                    j += 1;
                    continue;
                }

                unreachable!("best score must come from one of the spine scores");
            }
        }
    }

    println!("lookup_events = {lookup_events}");
    print_weighted_side(
        "first_keyed_rows_seen_at_lookup_time",
        &first_hist,
        &first_piece_hits,
        &out_degree,
        &id_to_piece,
        lookup_events,
    );
    print_weighted_side(
        "second_keyed_rows_seen_at_lookup_time",
        &second_hist,
        &second_piece_hits,
        &in_degree,
        &id_to_piece,
        lookup_events,
    );

    ExitCode::SUCCESS
}

fn load_degrees(
    path: &str,
    merges_graph: &BpeMerges,
) -> Result<(Vec<String>, Vec<u32>, Vec<u32>), String> {
    let text =
        fs::read_to_string(path).map_err(|err| format!("failed to read tokenizer json: {err}"))?;
    let json: Value = serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse tokenizer json: {err}"))?;

    let merges = json
        .get("model")
        .and_then(|model| model.get("merges"))
        .and_then(Value::as_array)
        .ok_or_else(|| "missing model.merges array".to_string())?;

    let mut out_degree_by_piece = HashMap::<String, u32>::new();
    let mut in_degree_by_piece = HashMap::<String, u32>::new();

    for item in merges {
        let Some(line) = item.as_str() else {
            return Err("non-string merge entry".to_string());
        };
        let Some((left, right)) = line.split_once(' ') else {
            return Err(format!("bad merge line: {line:?}"));
        };
        *out_degree_by_piece.entry(left.to_string()).or_default() += 1;
        *in_degree_by_piece.entry(right.to_string()).or_default() += 1;
    }

    let mut id_to_piece = Vec::new();
    let mut out_degree = Vec::new();
    let mut in_degree = Vec::new();
    let mut piece_id = 0u32;

    while let Some(piece) = merges_graph.decode_piece(piece_id) {
        id_to_piece.push(piece.to_string());
        out_degree.push(*out_degree_by_piece.get(piece).unwrap_or(&0));
        in_degree.push(*in_degree_by_piece.get(piece).unwrap_or(&0));
        piece_id += 1;
    }

    Ok((id_to_piece, out_degree, in_degree))
}

fn print_weighted_side(
    label: &str,
    hist: &BTreeMap<u32, u64>,
    piece_hits: &[u64],
    degrees: &[u32],
    id_to_piece: &[String],
    total_hits: u64,
) {
    let weighted_sum: u64 = hist
        .iter()
        .map(|(row_size, count)| *row_size as u64 * *count)
        .sum();
    println!("{label}:");
    println!(
        "  avg_row_size_seen = {:.6}",
        weighted_sum as f64 / total_hits as f64
    );

    for threshold in [8u32, 16, 32, 64, 128, 256, 512, 1024] {
        let hits_at_or_above: u64 = hist
            .iter()
            .filter(|(row_size, _)| **row_size >= threshold)
            .map(|(_, count)| *count)
            .sum();
        println!(
            "  fraction_with_row_size_ge_{threshold} = {:.6}",
            hits_at_or_above as f64 / total_hits as f64
        );
    }

    println!("  histogram:");
    for (row_size, count) in hist {
        println!("    {row_size}: {count}");
    }

    let mut ranked: Vec<(usize, u64)> = piece_hits.iter().copied().enumerate().collect();
    ranked.sort_by_key(|&(_, hits)| std::cmp::Reverse(hits));
    println!("  top10_pieces_by_lookup_frequency:");
    for (piece_id, hits) in ranked.into_iter().take(10) {
        if hits == 0 {
            break;
        }
        println!(
            "    hits={hits} row_size={} piece={:?}",
            degrees[piece_id], id_to_piece[piece_id]
        );
    }
}
