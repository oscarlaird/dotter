// We desire a hash function that we can compute incrementally as we descend the trie.
// We desire that we can increment a string by a character or by a token, and
// compute the new hash from the old hash.
// A "Rolling Polynomial Hash" satisfies these requirements beautifully.

// It even gives consecutive strings, consecutive hashes, which may help with locality.

// H(s) = \sum_{i<|s|} s_i * p^(|s| - 1 - i)

const MOD: u64 = (1 << 61) - 1;
#[cfg(not(any(test, debug_assertions)))]
const BASE: u64 = 257; // guarantees no collisions from a common prefix unto the seventh generation
#[cfg(not(any(test, debug_assertions)))]
const INV_BASE: u64 = 1_327_878_464_449_909_357;

#[cfg(any(test, debug_assertions))]
const BASE: u64 = 10000;
#[cfg(any(test, debug_assertions))]
const INV_BASE: u64 = 1_487_038_156_641_911_229;

const MAX_APPEND_LENGTH: usize = 32; // TODO: this should come from the tokenizer

pub type Hash = u64;

const POWERS: [u64; MAX_APPEND_LENGTH + 1] = {
    let mut powers = [0u64; MAX_APPEND_LENGTH + 1];
    let mut cur_power: u128 = 1; // need to use u128 to avoid overflow
    let mut i = 0;
    while i <= MAX_APPEND_LENGTH {
        powers[i] = cur_power as u64;
        cur_power = fast_mod(cur_power * (BASE as u128)) as u128;
        i += 1;
    }
    powers
};
const INV_POWERS: [u64; MAX_APPEND_LENGTH + 1] = {
    let mut inv_powers = [0u64; MAX_APPEND_LENGTH + 1];
    let mut cur_power: u128 = 1; // need to use u128 to avoid overflow
    let mut i = 0;
    while i <= MAX_APPEND_LENGTH {
        inv_powers[i] = cur_power as u64;
        cur_power = fast_mod(cur_power * (INV_BASE as u128)) as u128;
        i += 1;
    }
    inv_powers
};

const fn fast_mod(x: u128) -> u64 {
    debug_assert!((x >> 122) == 0, "We require x < MOD**2");
    let res = ((x & MOD as u128) + (x >> 61)) as u64;
    if res >= MOD {
        res - MOD
    } else {
        res
    }
}

pub const fn extend_right(hash: Hash, right_hash: Hash, right_length: usize) -> Hash {
    let mut result = hash as u128;
    let power_shift = POWERS[right_length] as u128;
    result = (result * power_shift) + (right_hash as u128);
    fast_mod(result)
}

pub const fn append_right(hash: Hash, right_char: u8) -> Hash {
    extend_right(hash, right_char as Hash, 1)
}

pub fn truncate_right(hash: Hash, right_hash: Hash, right_length: usize) -> Hash {
    let sub = if hash < right_hash {
        MOD + hash - right_hash
    } else {
        hash - right_hash
    };
    let invpower_shift = INV_POWERS[right_length] as u128;
    fast_mod((sub as u128) * invpower_shift)
}

pub fn pop_right(hash: Hash, right_char: u8) -> Hash {
    truncate_right(hash, right_char as Hash, 1)
}

pub fn hash_string(s: &str) -> Hash {
    let mut hash = 0;
    for c in s.as_bytes() {
        hash = append_right(hash, *c);
    }
    hash
}

// Identity Hashser for rust

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_assoc() {
        let s = "hello world";
        let mut hash1 = 0;
        for c in s.as_bytes() {
            hash1 = append_right(hash1, *c);
        }
        let mut hash2_hello = 0;
        for c in "hello ".as_bytes() {
            hash2_hello = append_right(hash2_hello, *c);
        }
        let mut hash2_world = 0;
        for c in "world".as_bytes() {
            hash2_world = append_right(hash2_world, *c);
        }
        let hash2 = extend_right(hash2_hello, hash2_world, 5);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_extend_truncate_right_round_trip() {
        let left = hash_string("hello ");
        let right = hash_string("world");
        let combined = extend_right(left, right, "world".len());
        assert_eq!(truncate_right(combined, right, "world".len()), left);
    }
}

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};


/// Keys are raw rolling-hash `u64` values; skip SipHash on lookup.
///
/// Must be `pub` so `RHashMap` / `RHashSet` can appear in `pub` fields and APIs
/// without violating Rust’s “public type uses private type” rule.
#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        panic!("IdentityHasher: key type must hash only via write_u64 (use u64 keys)");
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

pub type HashSymbolPair = (Hash, u8);
#[derive(Default)]
pub struct HashSymbolPairIdentityHasher(HashSymbolPair);
impl Hasher for HashSymbolPairIdentityHasher {
    fn finish(&self) -> u64 {
        self.0.0
    }

    fn write(&mut self, _: &[u8]) {
        panic!("HashSymbolPairIdentityHasher: key type must hash only via write_u64 (use u64 keys)");
    }

    fn write_u64(&mut self, i: u64) {
        self.0.0 = i;
    }
}
// Generic type definitions for rolling hash collections

pub type RHashSet = HashSet<Hash, BuildHasherDefault<IdentityHasher>>;
pub type RHashMap<V> = HashMap<Hash, V, BuildHasherDefault<IdentityHasher>>;
