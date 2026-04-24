use ewebsock;

use std::collections::{HashMap, HashSet};
use std::any::Any;
use std::hash;
use std::cell::RefCell;

use macroquad::{prelude::*, ui::widgets::Slider};
const SANS_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/DejaVuSans.ttf");
const SERIF_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/DejaVuSerif.ttf");
const FONT_BYTES: &[u8] = SERIF_FONT_BYTES;

mod colors;
mod likelihood;
mod components;
mod websocket;

use crate::components as c;

const WINDOW_WIDTH: i32 = 800;
const WINDOW_HEIGHT: i32 = 600;

use bayesian::{
    BayesianSession,
    rolling_hash::{Hash, RHashMap, append_right},
    trie::{
        ROOT_HASH, TRIE_EXPANSION_THRESHOLD, core::XBayes, l_update::{XLUpdateEntry, XLUpdate, merge_xl_pair, set_leaf_indicators}, logaddexp, symbol::{PadMode, XSymbol}
    },
};
use timer_spacing::{self, TimerSpacingParams, constant_phases, optimize};

fn window_conf() -> Conf {
    Conf {
        window_title: "macroquad tutor".to_owned(),
        window_width: WINDOW_WIDTH, window_height: WINDOW_HEIGHT, high_dpi: true, sample_count: 4,
        ..Default::default()
    }
}

#[derive(Copy, Clone, Debug)]
struct Point(f32, f32);

#[derive(Copy, Clone)]
struct BBox {
    ul: Point,
    width: f32,
    height: f32,
}
impl BBox {
    fn x_center(&self) -> f32 {
        self.ul.0 + self.width / 2.0
    }
    fn y_center(&self) -> f32 {
        self.ul.1 + self.height / 2.0
    }
    fn contains(&self, point: Point) -> bool {
        self.ul.0 <= point.0 &&
        point.0 < self.ul.0 + self.width &&
        self.ul.1 <= point.1 &&
        point.1 < self.ul.1 + self.height
    }
    fn x_split(&self, n: u32) -> Vec<Self> {
        let child_width = self.width / (n as f32);
        (0..n).map(|k| Self {
            ul: Point(self.ul.0 + child_width * (k as f32), self.ul.1),
            width: child_width,
            height: self.height,
        }).collect()
    }
    fn x_pad(&self, p: f32) -> Self {
        Self {
            ul: Point(self.ul.0 + p, self.ul.1),
            width: self.width - 2. * p,
            height: self.height
        }
    }
    fn y_pad(&self, p: f32) -> Self {
        Self {
            ul: Point(self.ul.0, self.ul.1 + p),
            width: self.width,
            height: self.height - 2. * p,
        }
    }
    fn pad(&self, p: f32) -> Self {
        self.x_pad(p).y_pad(p)
    }
}

struct CallbackRegion<'a> {
    bbox: BBox,
    click_callback: Option<Box<dyn FnMut(Point) -> () + 'a>>,
    keys_callback: Option<Box<dyn FnMut(&HashSet<KeyCode>) -> () + 'a>>,
    id: Option<WidgetId>,
}
// frame-specific map of callbacks
type CallbackCanvas<'a> = Vec<CallbackRegion<'a>>;


// widget-specific persistent data e.g. a cursor; undesirable, but sometimes unavoidable
type Persistent = RefCell<HashMap::<WidgetId, Box<dyn Any>>>;

struct ExclZData {
    symbol: XSymbol,
    excl_z: f32
}

struct TimersData {
    symbol: XSymbol,
    phase: f32,
}

#[derive(PartialEq, Clone, Copy, Eq, hash::Hash)]
enum SliderId {
    Mean,
    Stddev,
    Outliers,
    Period
}

#[derive(PartialEq, Clone, Copy, Eq, hash::Hash)]
enum WidgetId {
    Slider(SliderId),
    Username
}

#[macroquad::main(window_conf)]
async fn main() {
    let font = load_ttf_font_from_bytes(FONT_BYTES).unwrap();
    // let layout_font =
    //     fontdue::Font::from_bytes(FONT_BYTES, fontdue::FontSettings::default()).unwrap();
    let mut my_model = vec![0.132f32, 0.050f32, 0.10f32, 1.00f32];
    let model_names = vec!["mean", "stddev", "outliers", "period"];
    let model_ranges = vec![(-0.1, 0.2), (0.01, 0.25), (0.005, 0.20), (0.30, 2.00)];
    let model_widget_ids = vec![WidgetId::Slider(SliderId::Mean), WidgetId::Slider(SliderId::Stddev), WidgetId::Slider(SliderId::Outliers), WidgetId::Slider(SliderId::Period)];
    let mut session = BayesianSession::new();
    session.expand_to_threshold();
    let mut needs_timer_recalc = true;
    let mut timers_map: Option<RHashMap<TimersData>> = None;
    //
    let mut username = String::from("hello");
    let mut persistent = Persistent::default();
    let mut focused_id: Option<WidgetId> = None;
    //
    //
    let options = ewebsock::Options::default();
    let (mut sender, receiver) = ewebsock::connect("ws://127.0.0.1:8000/ws",ewebsock::Options::default()).unwrap();
    //

    loop {
        let cur_time = get_time() as f32;
        clear_background(BLACK);
        //
        let model = likelihood::LikelihoodModel {
            pred_mean: my_model[0],
            pred_stddev: my_model[1],
            pred_outliers: my_model[2],
            period: my_model[3]
        };
        // listen for prior updates
        while let Some(event) = receiver.try_recv() {
            println!("Received {:?}", event);
        }
        // check for signals
        if is_key_pressed(KeyCode::Space) {
            assert!(timers_map.is_some());
            let press_time = get_time() as f32;
            let mut new_l_update: XLUpdate = timers_map.as_ref().unwrap().iter()
                .map(|(hash, TimersData{symbol, phase})|
                    (*hash, XLUpdateEntry {
                        symbol: *symbol,
                        likelihood: likelihood::timer_likelihood(press_time, *phase, &model),
                        is_leaf: false
                    }))
                .collect();
            set_leaf_indicators(&mut new_l_update);
            session.trie.pending_likelihood =
                merge_xl_pair(&session.trie.pending_likelihood, &new_l_update);
            session.apply_updates();
            session.expand_to_threshold();
            // pass
            needs_timer_recalc = true;
        }
        // help! miscellaneous
        let viztree_box = BBox { ul: Point(200.0,200.0), width: 600.0, height: 600.0 };
        // layout pass
        if needs_timer_recalc {
            // calculate layout and timers
            let c::viztree::ViztreePassResult::MeasureResult(excl_z_map) =
                c::viztree::viztree(viztree_box, &session.trie, &font, cur_time, model.period, c::viztree::PassMode::Measure, &persistent) else { unreachable!() };

            let timer_spacing_params = TimerSpacingParams::new(
                excl_z_map.values().map(|x| x.excl_z as f64).collect(),
                model.pred_stddev as f64,
                model.period as f64
            );
            let initial_phases = constant_phases(excl_z_map.len(), model.period as f64);
            let optimal_phases = optimize(&timer_spacing_params, &initial_phases[..], 25).unwrap().phases;
            timers_map = Some(excl_z_map.iter().zip(optimal_phases.iter())
                .map(|((hash, ExclZData { symbol, excl_z: _ }), phase)|
                    (*hash, TimersData{symbol: *symbol, phase: *phase as f32}))
                .collect());

        }
        // draw pass
        let mut callback_canvas = CallbackCanvas::default();
        let sliders_box = BBox { ul: Point(0.0, 0.0), width: WINDOW_WIDTH as f32, height: 100.0 };
        for ((((mut child_box, var), range), name), widget_id) in sliders_box.x_split(4).iter()
            .zip(my_model.iter_mut())
            .zip(model_ranges.iter())
            .zip(model_names.iter().cloned())
            .zip(model_widget_ids.iter())
            {
            let b = (*child_box).pad(4.);
            draw_rectangle_lines(b.ul.0, b.ul.1, b.width, b.height, 2., RED);
            draw_text_ex(
                &format!("{}: {}", name, *var), b.ul.0, b.ul.1 + 20., TextParams { font: Some(&font), font_size: 14, color: RED, ..TextParams::default() }
            );
            // c::border::f()
            c::slider::slider(b.pad(2.), *range, var, &mut callback_canvas, Some(*widget_id));
        }
        let ubox = BBox { ul: Point(0.0, 100.0), width: 200., height: 100.};
        c::entry::f(ubox, &mut username, Some(&font), WidgetId::Username, &persistent, cur_time, &mut callback_canvas);
        draw_rectangle(viztree_box.ul.0, viztree_box.ul.1, viztree_box.width, viztree_box.height, WHITE);
        c::viztree::viztree(viztree_box, &session.trie, &font, cur_time, model.period, c::viztree::PassMode::Draw { timers_map: timers_map.as_ref().unwrap() }, &persistent);
        //
        next_frame().await;
        // handle callbacks for this frame; focus widget and send click
        let (x, y) = mouse_position();
        let click_point = Point(x, y);
        let keys = get_keys_pressed();
        for region in callback_canvas.iter_mut().rev() {
            if is_mouse_button_pressed(MouseButton::Left) {
                if region.bbox.contains(click_point) {
                    focused_id = region.id;
                    if let Some(cb) = &mut region.click_callback {
                        cb(click_point);
                    }
                }
            }
            // click should be assigned by bbox
            // hover should be assigned by bbox and by focus
            // all other events (mouseUp, keyPressed), should go to the focused element
            if focused_id == region.id && focused_id.is_some() && keys.len() > 0 {
                if let Some(cb) = &mut region.keys_callback {
                    cb(&keys);
                }
            }
        }
        // keystrokes are sent to focused widget
        // if let(id) = focused_id {
        //     for region in callback_canvas.iter_mut().rev() {
        //         (region.callback)(Point(0.0, 0.0))
        //     }
        // };
        needs_timer_recalc = false;
    }
}
