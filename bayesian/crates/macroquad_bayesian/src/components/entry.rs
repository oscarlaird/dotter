use macroquad::prelude::*;
use super::*;

pub(crate) struct EntryState {
    cursor_idx: u32
}


pub(crate) fn f<'a>(b: BBox, val: &'a mut String, font: Option<&Font>, id: WidgetId, persistent: &'a Persistent, cur_time: f32, callback_canvas: &mut CallbackCanvas<'a>) {
    let current_cursor_idx = {
        let mut persistent_mutref = persistent.borrow_mut();
        let any_state = persistent_mutref.entry(id).or_insert_with(|| Box::new(EntryState {
            cursor_idx: 2 // default
        }));
        let Some(EntryState{ cursor_idx }) = any_state.downcast_ref::<EntryState>() else {
            unreachable!()
        };
        *cursor_idx
    };
    //
    let h = 20.0;
    let font_size = 18;
    draw_rectangle_lines(b.ul.0, b.y_center() - h / 2.0, b.width, h, 2., RED);
    draw_text_ex(
        val.as_str(),
        b.ul.0, b.y_center() + 8., TextParams { font, font_size, color: RED, ..TextParams::default()}
    );
    let cursor_pos = (current_cursor_idx as usize).min(val.len());
    let cursor_x = measure_text(&val[..cursor_pos], font, font_size, 1.0).width;
    let blink = ((cur_time + 1.0) % 1.0) > 0.5;
    if blink {
        draw_line(
            cursor_x, b.y_center() + 8. - (font_size as f32),
            cursor_x, b.y_center() + 8.,
            1.0, RED
        );
    }
    // let cursor_idx = *cursor_idx;
    let callback = move |x: &HashSet<KeyCode>| {
        for key in x.iter() {
            match key {
                KeyCode::Backspace => {
                    if !val.is_empty() && current_cursor_idx > 0 {
                        let rem_idx = (current_cursor_idx as usize - 1).min(val.len() - 1);
                        val.remove(rem_idx);
                        persistent.borrow_mut().get_mut(&id).unwrap().downcast_mut::<EntryState>().unwrap().cursor_idx = current_cursor_idx - 1;
                    }
                }
                KeyCode::Left => {
                    if current_cursor_idx > 0 {
                        persistent.borrow_mut().get_mut(&id).unwrap().downcast_mut::<EntryState>().unwrap().cursor_idx = current_cursor_idx - 1;
                    }
                },
                KeyCode::Right => {
                    if (current_cursor_idx as usize) < val.len() {
                        persistent.borrow_mut().get_mut(&id).unwrap().downcast_mut::<EntryState>().unwrap().cursor_idx = current_cursor_idx + 1;
                    }
                }
                _ => {}
            }
            // Use the KeyCode discriminant value for A..Z to deduce character (assuming KeyCode::{A,B,...,Z} are ordered consecutively)
            let c = match key {
                KeyCode::A | KeyCode::B | KeyCode::C | KeyCode::D | KeyCode::E | KeyCode::F | KeyCode::G | KeyCode::H | KeyCode::I
                | KeyCode::J | KeyCode::K | KeyCode::L | KeyCode::M | KeyCode::N | KeyCode::O | KeyCode::P | KeyCode::Q | KeyCode::R
                | KeyCode::S | KeyCode::T | KeyCode::U | KeyCode::V | KeyCode::W | KeyCode::X | KeyCode::Y
                | KeyCode::Z => {
                    let base = KeyCode::A as u8;
                    let off = (*key as u8) - base;
                    ('A' as u8 + off) as char
                }
                _ => '\0',
            };
            match c {
                'A'..'Z' => {
                    val.insert((current_cursor_idx as usize).min(val.len()), c.to_ascii_lowercase());
                    persistent.borrow_mut().get_mut(&id).unwrap().downcast_mut::<EntryState>().unwrap().cursor_idx = current_cursor_idx + 1;
                },
                _ => { }
            }
        }
    };
    callback_canvas.push(
        CallbackRegion {
            bbox: b,
            click_callback: None,
            keys_callback: Some(Box::new(callback)),
            id: Some(id)
    });
}