// use macroquad::prelude::*;
// use super::*;

// pub(crate) enum Mode {
//     Straddle,
//     Outline,
//     Inline
// }

// pub(crate) fn f(mut bbox: BBox, mode: Mode, thickness: f32, color: Color) {
//     bbox = match mode {
//         Mode::Straddle => bbox,
//         Mode::Inline => bbox.pad(thickness / 2.),
//         Mode::Outline => bbox.pad(-thickness / 2.)
//     };
//     draw_rectangle_lines(bbox.ul.0, bbox.ul.1, bbox.width, bbox.height, thickness, color);
// }