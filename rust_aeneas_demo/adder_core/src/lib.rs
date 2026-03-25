#![no_std]

pub fn add_u32(x: u32, y: u32) -> u32 {
    x.wrapping_add(y)
}

pub fn is_even(x: u32) -> bool {
    (x % 2) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_numbers() {
        assert_eq!(add_u32(20, 22), 42);
    }

    #[test]
    fn wrapping_add_preserves_evenness_for_even_inputs() {
        let x = 10u32;
        let y = 14u32;
        assert!(is_even(x));
        assert!(is_even(y));
        assert!(is_even(add_u32(x, y)));
    }
}
