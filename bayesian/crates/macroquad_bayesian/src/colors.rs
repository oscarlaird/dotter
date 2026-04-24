use macroquad::prelude::{Color};
use macroquad::color_u8;

const GREEN: Color = color_u8!(34, 177, 76, 255);
const RED: Color = color_u8!(255, 77, 77, 255);
const BLUE: Color = color_u8!(77, 121, 255, 255);
const YELLOW: Color = color_u8!(255, 225, 77, 255);
const PURPLE: Color = color_u8!(153, 77, 255, 255);
const ORANGE: Color = color_u8!(255, 153, 51, 255);
const TAN: Color = color_u8!(204, 153, 102, 255);
const GRAY: Color = color_u8!(160, 160, 160, 255);
const WHITE: Color = color_u8!(255, 255, 255, 255);
const BLACK: Color = color_u8!(0, 0, 0, 255);
// Trie pad-mode sentinels (Rust `N` / `Q` / `U`): distinct from letter keys.
const NUMPAD: Color = color_u8!(0, 168, 150, 255);
const SPECIALPAD: Color = color_u8!(139, 92, 246, 255);
const SHIFTPAD: Color = color_u8!(245, 158, 11, 255);
const UNDEFINED: Color = color_u8!(0, 0, 0, 255);

pub fn color_from_symbol(symbol: u8) -> Color {
    match symbol as char {
        'a' | '$' | '.' | ',' | '\'' | 'Z' | 'r' | 'j' => RED,
        'b' | 'n' | 'w' => BLUE,
        'c' | 'i' | 'p' | 'v' => PURPLE,
        'd' | 'k' | 't' => TAN,
        'e' | 'm' | 'u' => GRAY,
        'f' | 'l' | 'q' | 'y' => YELLOW,
        'g' | 's' | 'z' => GREEN,
        'h' | 'o' | 'x' => ORANGE,
        // ' ' => WHITE,
        // Trie pad-mode sentinels (Rust `N` / `Q` / `U`): distinct from letter keys.
        'N' => NUMPAD,
        'Q' => SPECIALPAD,
        'U' => SHIFTPAD,
        _ => UNDEFINED,
    }
}
