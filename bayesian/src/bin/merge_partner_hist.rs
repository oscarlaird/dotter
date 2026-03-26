use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::process::ExitCode;

use bayesian::bpe::BpeMerges;
use serde_json::Value;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(tokenizer_path) = args.next() else {
        eprintln!("usage: cargo run --bin merge_partner_hist -- <tokenizer.json>");
        return ExitCode::from(2);
    };

    let text = match fs::read_to_string(&tokenizer_path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("failed to read tokenizer json: {err}");
            return ExitCode::from(1);
        }
    };

    let json: Value = match serde_json::from_str(&text) {
        Ok(json) => json,
        Err(err) => {
            eprintln!("failed to parse tokenizer json: {err}");
            return ExitCode::from(1);
        }
    };

    let Some(merges) = json
        .get("model")
        .and_then(|model| model.get("merges"))
        .and_then(Value::as_array)
    else {
        eprintln!("missing model.merges array");
        return ExitCode::from(1);
    };

    let merges_graph = match BpeMerges::from_tokenizer_json(&tokenizer_path) {
        Ok(merges_graph) => merges_graph,
        Err(err) => {
            eprintln!("failed to load merges graph: {err}");
            return ExitCode::from(1);
        }
    };

    let mut out_degree_by_piece = HashMap::<String, u32>::new();
    let mut in_degree_by_piece = HashMap::<String, u32>::new();

    for item in merges {
        let Some(line) = item.as_str() else {
            eprintln!("non-string merge entry");
            return ExitCode::from(1);
        };
        let Some((left, right)) = line.split_once(' ') else {
            eprintln!("bad merge line: {line:?}");
            return ExitCode::from(1);
        };
        *out_degree_by_piece.entry(left.to_string()).or_default() += 1;
        *in_degree_by_piece.entry(right.to_string()).or_default() += 1;
    }

    let (id_to_piece, out_degree, in_degree) =
        build_internal_degree_vectors(&merges_graph, &out_degree_by_piece, &in_degree_by_piece);

    print_side("first_piece_rows", &out_degree, &id_to_piece);
    print_side("second_piece_rows", &in_degree, &id_to_piece);

    ExitCode::SUCCESS
}

fn build_internal_degree_vectors(
    merges_graph: &BpeMerges,
    out_degree_by_piece: &HashMap<String, u32>,
    in_degree_by_piece: &HashMap<String, u32>,
) -> (Vec<String>, Vec<u32>, Vec<u32>) {
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

    (id_to_piece, out_degree, in_degree)
}

fn print_side(label: &str, degrees: &[u32], id_to_piece: &[String]) {
    let mut hist = BTreeMap::<u32, u32>::new();
    let mut nonzero = 0u32;
    let mut sum = 0u64;
    let mut max_degree = 0u32;
    let mut max_piece_id = 0usize;

    for (piece_id, &degree) in degrees.iter().enumerate() {
        *hist.entry(degree).or_default() += 1;
        if degree != 0 {
            nonzero += 1;
            sum += degree as u64;
            if degree > max_degree {
                max_degree = degree;
                max_piece_id = piece_id;
            }
        }
    }

    println!("{label}:");
    println!("  total_pieces = {}", degrees.len());
    println!("  pieces_with_at_least_one_partner = {nonzero}");
    println!(
        "  avg_partners_over_all_pieces = {:.6}",
        sum as f64 / degrees.len() as f64
    );
    println!(
        "  avg_partners_over_nonzero_pieces = {:.6}",
        sum as f64 / nonzero as f64
    );
    println!(
        "  max_partners = {} ({:?})",
        max_degree, id_to_piece[max_piece_id]
    );

    println!("  histogram:");
    for (degree, count) in hist {
        println!("    {degree}: {count}");
    }

    let mut ranked: Vec<(usize, u32)> = degrees.iter().copied().enumerate().collect();
    ranked.sort_by_key(|&(_, degree)| std::cmp::Reverse(degree));
    println!("  top10:");
    for (piece_id, degree) in ranked.into_iter().take(10) {
        println!("    {degree}: {:?}", id_to_piece[piece_id]);
    }
}
