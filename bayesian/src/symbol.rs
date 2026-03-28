use serde::{Deserialize, Serialize};

/// Number of symbols in the fixed trie alphabet.
pub const RADIX: usize = 29;

/// Default trie alphabet: `a`–`z`, space, `$`, `^` (slot order matches this slice).
pub const DEFAULT_ALPHABET: [u8; RADIX] = *b"abcdefghijklmnopqrstuvwxyz $^";

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
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
    Space = 26,
    Stop = 27,
    Start = 28,
}

impl Symbol {
    pub const ALL: [Self; RADIX] = [
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
            b' ' => Some(Self::Space),
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
            Self::Space => b' ',
            Self::Stop => b'$',
            Self::Start => b'^',
        }
    }

    pub const fn from_slot(slot: u8) -> Option<Self> {
        match slot {
            0 => Some(Self::A),
            1 => Some(Self::B),
            2 => Some(Self::C),
            3 => Some(Self::D),
            4 => Some(Self::E),
            5 => Some(Self::F),
            6 => Some(Self::G),
            7 => Some(Self::H),
            8 => Some(Self::I),
            9 => Some(Self::J),
            10 => Some(Self::K),
            11 => Some(Self::L),
            12 => Some(Self::M),
            13 => Some(Self::N),
            14 => Some(Self::O),
            15 => Some(Self::P),
            16 => Some(Self::Q),
            17 => Some(Self::R),
            18 => Some(Self::S),
            19 => Some(Self::T),
            20 => Some(Self::U),
            21 => Some(Self::V),
            22 => Some(Self::W),
            23 => Some(Self::X),
            24 => Some(Self::Y),
            25 => Some(Self::Z),
            26 => Some(Self::Space),
            27 => Some(Self::Stop),
            28 => Some(Self::Start),
            _ => None,
        }
    }

    pub const fn to_slot(self) -> u8 {
        self as u8
    }
}
