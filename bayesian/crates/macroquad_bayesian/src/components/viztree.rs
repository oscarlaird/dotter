use macroquad::prelude::*;

use super::*;

pub(crate) enum PassMode<'a> {
    Draw { timers_map: &'a RHashMap<TimersData> },
    Measure
}

pub(crate) enum ViztreePassResult {
    DrawResult,
    MeasureResult(RHashMap<ExclZData>)
}

pub(crate) fn viztree(bbox: BBox, trie: &XBayes, font: &Font, cur_time: f32, period: f32, pass_mode: PassMode, persistent: &Persistent) -> ViztreePassResult {
    //
    struct Frame {
        hash: Hash,
        ul: Point,
        height: f32,
        z: f32,
    }
    let root_z = trie.nodes.get(&ROOT_HASH).unwrap().if_root_then_z;
    let root_frame = Frame {
        hash: ROOT_HASH,
        ul: bbox.ul,
        height: bbox.height,
        z: root_z
    };
    let mut unvisited_stack = vec![root_frame];
    //
    let mut excl_z_map: Option<RHashMap<ExclZData>> = None;
    if let PassMode::Measure = pass_mode {
        excl_z_map = Some(RHashMap::default());
    };
    while let Some(frame) = unvisited_stack.pop() {
        let n_node = &trie.nodes[&frame.hash];
        let n_symbol = n_node.symbol;
        let n_color = colors::color_from_symbol(n_symbol);
        // let n_color
        // draw box
        let frame_box = BBox {
            ul: frame.ul,
            height: frame.height,
            width: 30.0
        };
        if let PassMode::Draw {..} = pass_mode {
            draw_rectangle(
                frame_box.ul.0,
                frame_box.ul.1,
                frame_box.width,
                frame_box.height,
                Color{a: 0.20, ..n_color}
            );
        }
        // draw symbol
        let s = (n_symbol as char).to_string();
        let measurements = measure_text(&s, Some(font), 20, 1.0);
        if let PassMode::Draw {..} = pass_mode {
            draw_text_ex(
                &(n_symbol as char).to_string()[..],
                frame_box.x_center() - measurements.width / 2.0,
                frame_box.y_center() + measurements.height / 2.0,
                TextParams {
                    font: Some(font),
                    font_size: 20,
                    color: n_color,
                    ..Default::default()
                },
            );
        };
        // draw timer circle
        if let PassMode::Draw { timers_map} = pass_mode {
            if timers_map.contains_key(&frame.hash) {
                let frac_elapsed = (cur_time - timers_map[&frame.hash].phase + period) % period;
                let target_angle = 360. * frac_elapsed;
                let alpha = 1.00 * frac_elapsed + 0.40 * (1. - frac_elapsed);
                draw_arc(
                    frame_box.x_center(),
                    frame_box.y_center(),
                    24,
                    12.,
                    0.,
                    2.0,
                    target_angle,
                    Color{a: alpha, ..n_color}
                )
            };
        }
        // push children to unvisited_stack
        let n_padmode = PadMode::for_xsymbol(n_symbol);
        let mut cum_conditional = f32::NEG_INFINITY; // log conditional prob strictly above child
        let mut cum_visible_c_z = f32::NEG_INFINITY;
        for slot in 0..n_padmode.radix() {
            let c_symbol = n_padmode.slot_to_xsymbol(slot);
            let c_hash = append_right(frame.hash, c_symbol);
            let c_z = n_node.c_z[slot];
            let c_conditional = c_z - frame.z;
            if (c_z - root_z) > TRIE_EXPANSION_THRESHOLD {
                let c_y_offset = frame.height * cum_conditional.exp();
                let c_height = frame.height * c_conditional.exp();
                unvisited_stack.push(Frame {
                    hash: c_hash,
                    ul: Point(
                        frame.ul.0 + frame_box.width,
                        frame.ul.1 + c_y_offset,
                    ),
                    height: c_height,
                    z: c_z
                });
                cum_visible_c_z = logaddexp(cum_visible_c_z, c_z);
            }
            cum_conditional = logaddexp(cum_conditional, c_conditional);
        }
        if let PassMode::Measure = pass_mode {
            let n_excl_z = (1. - (cum_visible_c_z - frame.z).exp()).ln() + frame.z;
            let normalized_n_excl_z = n_excl_z - root_z;
            excl_z_map.as_mut().unwrap().insert(frame.hash, ExclZData { symbol: n_symbol, excl_z: normalized_n_excl_z });
        }
    }
    match pass_mode {
        PassMode::Draw {..} => ViztreePassResult::DrawResult,
        PassMode::Measure => ViztreePassResult::MeasureResult(excl_z_map.unwrap())
    }
}
