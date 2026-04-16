/// Number of symbols in the fixed trie alphabet, excluding the start marker.
pub const RADIX: usize = 31;
pub const N_SYMBOLS: usize = RADIX + 1;
pub type RadixBitmap = u32;

/// Default trie alphabet: `a`–`z`, `,`, `.`, `'`, `_` (word boundary), `$` (stop), `^` (start).
pub const DEFAULT_ALPHABET: [u8; N_SYMBOLS] = *b"abcdefghijklmnopqrstuvwxyz,.'_$^";

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Symbol {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
    I = 8,
    J = 9,
    K = 10,
    L = 11,
    M = 12,
    N = 13,
    O = 14,
    P = 15,
    Q = 16,
    R = 17,
    S = 18,
    T = 19,
    U = 20,
    V = 21,
    W = 22,
    X = 23,
    Y = 24,
    Z = 25,
    Comma = 26,
    Period = 27,
    Apostrophe = 28,
    Space = 29,
    Stop = 30,
    Start = 31,
}

impl Symbol {
    pub const ALL: [Self; N_SYMBOLS] = [
        Self::A,
        Self::B,
        Self::C,
        Self::D,
        Self::E,
        Self::F,
        Self::G,
        Self::H,
        Self::I,
        Self::J,
        Self::K,
        Self::L,
        Self::M,
        Self::N,
        Self::O,
        Self::P,
        Self::Q,
        Self::R,
        Self::S,
        Self::T,
        Self::U,
        Self::V,
        Self::W,
        Self::X,
        Self::Y,
        Self::Z,
        Self::Comma,
        Self::Period,
        Self::Apostrophe,
        Self::Space,
        Self::Stop,
        Self::Start,
    ];

    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            b'a' => Some(Self::A),
            b'b' => Some(Self::B),
            b'c' => Some(Self::C),
            b'd' => Some(Self::D),
            b'e' => Some(Self::E),
            b'f' => Some(Self::F),
            b'g' => Some(Self::G),
            b'h' => Some(Self::H),
            b'i' => Some(Self::I),
            b'j' => Some(Self::J),
            b'k' => Some(Self::K),
            b'l' => Some(Self::L),
            b'm' => Some(Self::M),
            b'n' => Some(Self::N),
            b'o' => Some(Self::O),
            b'p' => Some(Self::P),
            b'q' => Some(Self::Q),
            b'r' => Some(Self::R),
            b's' => Some(Self::S),
            b't' => Some(Self::T),
            b'u' => Some(Self::U),
            b'v' => Some(Self::V),
            b'w' => Some(Self::W),
            b'x' => Some(Self::X),
            b'y' => Some(Self::Y),
            b'z' => Some(Self::Z),
            b',' => Some(Self::Comma),
            b'.' => Some(Self::Period),
            b'\'' => Some(Self::Apostrophe),
            b'_' => Some(Self::Space),
            b'$' => Some(Self::Stop),
            b'^' => Some(Self::Start),
            _ => None,
        }
    }

    pub const fn to_byte(self) -> u8 {
        match self {
            Self::A => b'a',
            Self::B => b'b',
            Self::C => b'c',
            Self::D => b'd',
            Self::E => b'e',
            Self::F => b'f',
            Self::G => b'g',
            Self::H => b'h',
            Self::I => b'i',
            Self::J => b'j',
            Self::K => b'k',
            Self::L => b'l',
            Self::M => b'm',
            Self::N => b'n',
            Self::O => b'o',
            Self::P => b'p',
            Self::Q => b'q',
            Self::R => b'r',
            Self::S => b's',
            Self::T => b't',
            Self::U => b'u',
            Self::V => b'v',
            Self::W => b'w',
            Self::X => b'x',
            Self::Y => b'y',
            Self::Z => b'z',
            Self::Comma => b',',
            Self::Period => b'.',
            Self::Apostrophe => b'\'',
            Self::Space => b'_',
            Self::Stop => b'$',
            Self::Start => b'^',
        }
    }

    pub const fn from_slot(slot: usize) -> Self {
        match slot {
            0 => Self::A,
            1 => Self::B,
            2 => Self::C,
            3 => Self::D,
            4 => Self::E,
            5 => Self::F,
            6 => Self::G,
            7 => Self::H,
            8 => Self::I,
            9 => Self::J,
            10 => Self::K,
            11 => Self::L,
            12 => Self::M,
            13 => Self::N,
            14 => Self::O,
            15 => Self::P,
            16 => Self::Q,
            17 => Self::R,
            18 => Self::S,
            19 => Self::T,
            20 => Self::U,
            21 => Self::V,
            22 => Self::W,
            23 => Self::X,
            24 => Self::Y,
            25 => Self::Z,
            26 => Self::Comma,
            27 => Self::Period,
            28 => Self::Apostrophe,
            29 => Self::Space,
            30 => Self::Stop,
            31 => Self::Start,
            _ => panic!("Symbol::from_slot: invalid slot"),
        }
    }

    pub const fn to_slot(self) -> usize {
        self as usize
    }

    pub fn slot_to_byte(slot: usize) -> u8 {
        match slot {
            0 => b'a',
            1 => b'b',
            2 => b'c',
            3 => b'd',
            4 => b'e',
            5 => b'f',
            6 => b'g',
            7 => b'h',
            8 => b'i',
            9 => b'j',
            10 => b'k',
            11 => b'l',
            12 => b'm',
            13 => b'n',
            14 => b'o',
            15 => b'p',
            16 => b'q',
            17 => b'r',
            18 => b's',
            19 => b't',
            20 => b'u',
            21 => b'v',
            22 => b'w',
            23 => b'x',
            24 => b'y',
            25 => b'z',
            26 => b',',
            27 => b'.',
            28 => b'\'',
            29 => b'_',
            30 => b'$',
            31 => b'^',
            _ => panic!("invalid slot: {}", slot),
        }
    }

    pub fn byte_to_slot(byte: u8) -> usize {
        match byte {
            b'a' => 0,
            b'b' => 1,
            b'c' => 2,
            b'd' => 3,
            b'e' => 4,
            b'f' => 5,
            b'g' => 6,
            b'h' => 7,
            b'i' => 8,
            b'j' => 9,
            b'k' => 10,
            b'l' => 11,
            b'm' => 12,
            b'n' => 13,
            b'o' => 14,
            b'p' => 15,
            b'q' => 16,
            b'r' => 17,
            b's' => 18,
            b't' => 19,
            b'u' => 20,
            b'v' => 21,
            b'w' => 22,
            b'x' => 23,
            b'y' => 24,
            b'z' => 25,
            b',' => 26,
            b'.' => 27,
            b'\'' => 28,
            b'_' => 29,
            b'$' => 30,
            b'^' => 31,
            _ => panic!("invalid byte: {}", byte),
        }
    }

    pub fn slice_to_string(symbols: &[Self]) -> String {
        let mut out = String::with_capacity(symbols.len());
        for &symbol in symbols {
            out.push(symbol.to_byte() as char);
        }
        out
    }

    pub fn string_to_vec(text: &str) -> Vec<Self> {
        text.bytes().filter_map(Self::from_byte).collect()
    }
}

