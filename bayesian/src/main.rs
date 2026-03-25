//! Run during development: `cargo run` from the `bayesian/` crate root.

use bayesian::add;

fn main() {
    println!("bayesian dev CLI: 2 + 3 = {}", add(2, 4));
}
