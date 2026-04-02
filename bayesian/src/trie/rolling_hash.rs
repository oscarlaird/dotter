// We desire a hash function that we can compute incrementally as we descend the trie.
// We desire that we can increment a string by a character or by a token, and 
// compute the new hash from the old hash.
// A "Rolling Polynomial Hash" satisfies these requirements beautifully.

// It even gives consecutive strings, consecutive hashes, which may help with locality.

// H(s) = \sum_{i<|s|} s_i * p^(|s| - 1 - i)

const MOD: u64 = (1 << 61) - 1;
const BASE: u64 = 257; // guarantees no collisions from a common prefix unto the seventh generation
const MAX_APPEND_LENGTH: usize = 16; // TODO: this should come from the tokenizer
const POWERS: [u64; MAX_APPEND_LENGTH + 1] = {
    let mut powers = [0u64; MAX_APPEND_LENGTH + 1];
    let mut cur_power: u128 = 1;  // need to use u128 to avoid overflow
    let mut i = 0;
    while i <= MAX_APPEND_LENGTH {
        powers[i] = cur_power as u64;
        cur_power = fast_mod(cur_power * (BASE as u128)) as u128;
        i += 1;
    }
    powers
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

pub(crate) fn append_right_hash(hash: u64, right_hash: u64, right_length: usize) -> u64 {
    let mut result = hash as u128;
    let power_shift = POWERS[right_length] as u128;
    result = (result * power_shift) + (right_hash as u128);
    fast_mod(result)
}

pub(crate) fn append_right_char(hash: u64, right_char: u8) -> u64 {
    append_right_hash(hash, right_char as u64, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_assoc() {
        let s = "hello world";
        let mut hash1 = 0;
        for c in s.as_bytes() {
            hash1 = append_right_char(hash1, *c);
        }
        let mut hash2_hello = 0;
        for c in "hello ".as_bytes() {
            hash2_hello = append_right_char(hash2_hello, *c);
        }
        let mut hash2_world = 0;
        for c in "world".as_bytes() {
            hash2_world = append_right_char(hash2_world, *c);
        }
        let hash2 = append_right_hash(hash2_hello, hash2_world, 5);
        assert_eq!(hash1, hash2);
    }
}