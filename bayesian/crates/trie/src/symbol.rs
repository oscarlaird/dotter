/// Number of symbols in the fixed trie alphabet, excluding the start marker.
// pub const RADIX: usize = 31;
// pub const N_SYMBOLS: usize = RADIX + 1;
pub type RadixBitmap = u64;

macro_rules! concat_u8_arrays {
    ($($array:expr),*) => {{
        const N: usize = 0 $(+ $array.len())*;
        let mut res = [b' '; N];
        let mut i = 0;
        $(let mut j = 0; while j < $array.len() { res[i] = $array[j]; i += 1; j += 1; })*
        res
    }};
}

pub type XSymbol = u8;

pub const START_SYMBOL: XSymbol = b'A';

pub const SPACE_SYMBOL: XSymbol = b'S';
pub const STOP_SYMBOL: XSymbol = b'Z';

pub const LETTERS: &[XSymbol; 26] = b"abcdefghijklmnopqrstuvwxyz";
pub const PUNCTUATION: &[XSymbol; 4] = b"',.?";

pub const TRIE_CONTROL_CHARS: &[XSymbol; 3] = b"NUQ";

pub const NUMBERS: &[XSymbol; 10] = b"0123456789";
pub const SPECIAL: &[XSymbol; 29] = b"!:\"$%&()*+-/:;<=>@[\\]^_`{|}~#";

pub const DEFAULT_PAD_SYMBOLS: [XSymbol; 35] = concat_u8_arrays!(LETTERS, PUNCTUATION, TRIE_CONTROL_CHARS, [SPACE_SYMBOL, STOP_SYMBOL]);

pub const MAX_RADIX: usize = DEFAULT_PAD_SYMBOLS.len();

#[derive(Clone, Copy)]
pub enum PadMode {
    Default,
    Numpad,
    Shiftpad,
    Specialpad
}

impl PadMode {
    pub fn radix(self) -> usize {
        match self {
            // space and stop are included
            PadMode::Default => DEFAULT_PAD_SYMBOLS.len(),
            PadMode::Numpad => NUMBERS.len(),
            PadMode::Shiftpad => LETTERS.len(),
            PadMode::Specialpad => SPECIAL.len(),
        }
    }

    pub fn for_xsymbol(xsymbol: XSymbol) -> Self {
        match xsymbol {
            b'N' => PadMode::Numpad,
            b'U' => PadMode::Shiftpad,
            b'Q' => PadMode::Specialpad,
            _ => PadMode::Default,
        }
    }

    pub fn slot_to_xsymbol(&self, slot: usize) -> XSymbol {
        match self {
            Self::Default => { DEFAULT_PAD_SYMBOLS[slot] },
            Self::Numpad => { NUMBERS[slot] },
            Self::Shiftpad => { LETTERS[slot] },
            Self::Specialpad => { SPECIAL[slot] },
        }
    }

}

