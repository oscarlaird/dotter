use bayesian::calibration::VariationalParams;
use bayesian::render_utils::ExpandedSnapshot;
use bayesian::CalibrationSample;
use dioxus::prelude::*;
use layout::{
    build_visible_tree, compute_laid_out_nodes, deepest_visible_node, find_tutor_target_key,
    first_fork_node, relative_depth, LikelihoodModel, VisibleNodeTimerMap, SCROLL_TARGET_X_PX,
    SPACE_SYMBOL, STOP_SYMBOL,
};

use crate::domain::{
    color_from_letter, display_text, offscreen_prefix_text, AutoCalibrationState,
    CalibratedLikelihoodModel,
};

#[derive(Clone, Debug, PartialEq)]
pub struct PredictionLogEntry {
    pub id: usize,
    pub full_string: String,
    pub final_token_lexindex: String,
    pub received_at: String,
}

#[component]
pub fn CalibrationSettingsPanel(
    mut use_automatic_calibration: Signal<AutoCalibrationState>,
    mut likelihood_model: Signal<LikelihoodModel>,
    auto_calibration_likelihood_model: CalibratedLikelihoodModel,
    calibration_sample_count: usize,
    raw_variational_params: Option<VariationalParams>,
    recent_calibration_pairs: Vec<CalibrationSample>,
    show_calibration_debug: bool,
) -> Element {
    let auto_model = auto_calibration_likelihood_model.model.clone();
    use_effect(move || {
        let auto_flags = use_automatic_calibration();
        let mut next_model = likelihood_model();
        if auto_flags.mu_delay {
            next_model.mu_delay = auto_model.mu_delay;
        }
        if auto_flags.stddev_delay {
            next_model.stddev_delay = auto_model.stddev_delay;
        }
        if auto_flags.outliers {
            next_model.outliers = auto_model.outliers;
        }
        if next_model != likelihood_model() {
            likelihood_model.set(next_model);
        }
    });

    rsx! {
        div { class: "panel calibration-panel",
            div { class: "panel-header",
                div {
                    div { class: "panel-title", "Calibration" }
                    div { class: "panel-subtitle", "Calibration samples: {calibration_sample_count}" }
                }
                div { class: "panel-badge", "Auto-calibrated controls" }
            }
            if show_calibration_debug {
                if let Some(params) = raw_variational_params {
                    div { class: "debug-card",
                        div { class: "debug-title", "Variational params" }
                        div { class: "debug-mono",
                            div { "mu_m={params.mu_m:.6}" }
                            div { "sigma_m={params.sigma_m:.6}" }
                            div { "mu_s={params.mu_s:.6}" }
                            div { "sigma_s={params.sigma_s:.6}" }
                            div { "log_alpha={params.log_alpha:.6}" }
                            div { "log_beta={params.log_beta:.6}" }
                        }
                    }
                }
                if !recent_calibration_pairs.is_empty() {
                    div { class: "debug-card",
                        div { class: "debug-title", "Recent calibration pairs" }
                        div { class: "debug-mono",
                            for pair in recent_calibration_pairs {
                                div { "x={pair.x:.6}, period={pair.period:.6}" }
                            }
                        }
                    }
                }
            }
            div { class: "slider-grid",
                SliderRow {
                    label: format!("Mean ({:.0}ms)", 1000.0 * likelihood_model().mu_delay),
                    min: -0.05,
                    max: 0.2,
                    step: 0.001,
                    value: likelihood_model().mu_delay,
                    interval: auto_calibration_likelihood_model.intervals.as_ref().map(|intervals| intervals.mu_delay),
                    interval_text: auto_calibration_likelihood_model
                        .intervals
                        .as_ref()
                        .map(|intervals| format!("[{:.0}ms, {:.0}ms]", intervals.mu_delay.0 * 1000.0, intervals.mu_delay.1 * 1000.0)),
                    on_change: move |next_value| {
                        let mut next_model = likelihood_model();
                        next_model.mu_delay = next_value;
                        likelihood_model.set(next_model);
                        let mut next_flags = use_automatic_calibration();
                        next_flags.mu_delay = false;
                        use_automatic_calibration.set(next_flags);
                    },
                    show_auto_toggle: true,
                    auto_calibrate_value: use_automatic_calibration().mu_delay,
                    on_auto_calibrate_change: move |enabled| {
                        let mut next_flags = use_automatic_calibration();
                        next_flags.mu_delay = enabled;
                        use_automatic_calibration.set(next_flags);
                    },
                }
                SliderRow {
                    label: format!("StdDev ({:.0}ms)", 1000.0 * likelihood_model().stddev_delay),
                    min: 0.0,
                    max: 0.15,
                    step: 0.001,
                    value: likelihood_model().stddev_delay,
                    interval: auto_calibration_likelihood_model.intervals.as_ref().map(|intervals| intervals.stddev_delay),
                    interval_text: auto_calibration_likelihood_model
                        .intervals
                        .as_ref()
                        .map(|intervals| format!("[{:.0}ms, {:.0}ms]", intervals.stddev_delay.0 * 1000.0, intervals.stddev_delay.1 * 1000.0)),
                    on_change: move |next_value| {
                        let mut next_model = likelihood_model();
                        next_model.stddev_delay = next_value;
                        likelihood_model.set(next_model);
                        let mut next_flags = use_automatic_calibration();
                        next_flags.stddev_delay = false;
                        use_automatic_calibration.set(next_flags);
                    },
                    show_auto_toggle: true,
                    auto_calibrate_value: use_automatic_calibration().stddev_delay,
                    on_auto_calibrate_change: move |enabled| {
                        let mut next_flags = use_automatic_calibration();
                        next_flags.stddev_delay = enabled;
                        use_automatic_calibration.set(next_flags);
                    },
                }
                SliderRow {
                    label: format!("Outliers ({:.1}%)", 100.0 * likelihood_model().outliers),
                    min: 0.0,
                    max: 0.25,
                    step: 0.001,
                    value: likelihood_model().outliers,
                    interval: auto_calibration_likelihood_model.intervals.as_ref().map(|intervals| intervals.outliers),
                    interval_text: auto_calibration_likelihood_model
                        .intervals
                        .as_ref()
                        .map(|intervals| format!("[{:.1}%, {:.1}%]", intervals.outliers.0 * 100.0, intervals.outliers.1 * 100.0)),
                    on_change: move |next_value| {
                        let mut next_model = likelihood_model();
                        next_model.outliers = next_value;
                        likelihood_model.set(next_model);
                        let mut next_flags = use_automatic_calibration();
                        next_flags.outliers = false;
                        use_automatic_calibration.set(next_flags);
                    },
                    show_auto_toggle: true,
                    auto_calibrate_value: use_automatic_calibration().outliers,
                    on_auto_calibrate_change: move |enabled| {
                        let mut next_flags = use_automatic_calibration();
                        next_flags.outliers = enabled;
                        use_automatic_calibration.set(next_flags);
                    },
                }
                SliderRow {
                    label: format!("Period ({:.2}s)", likelihood_model().period),
                    min: 0.3,
                    max: 2.5,
                    step: 0.01,
                    value: likelihood_model().period,
                    interval: None,
                    interval_text: None,
                    on_change: move |next_value| {
                        let mut next_model = likelihood_model();
                        next_model.period = next_value;
                        likelihood_model.set(next_model);
                    },
                    show_auto_toggle: false,
                    auto_calibrate_value: false,
                    on_auto_calibrate_change: move |_| {},
                }
            }
        }
    }
}

#[component]
fn SliderRow(
    label: String,
    min: f64,
    max: f64,
    step: f64,
    value: f64,
    interval: Option<(f64, f64)>,
    interval_text: Option<String>,
    mut on_change: EventHandler<f64>,
    show_auto_toggle: bool,
    auto_calibrate_value: bool,
    mut on_auto_calibrate_change: EventHandler<bool>,
) -> Element {
    let interval_style = interval.map(|(low, high)| {
        let start_pct = ((low - min) / (max - min)).clamp(0.0, 1.0) * 100.0;
        let end_pct = ((high - min) / (max - min)).clamp(0.0, 1.0) * 100.0;
        format!(
            "background: linear-gradient(to right, transparent {start_pct:.2}%, rgba(59, 130, 246, 0.4) {start_pct:.2}%, rgba(59, 130, 246, 0.4) {end_pct:.2}%, transparent {end_pct:.2}%);"
        )
    });

    rsx! {
        div { class: "slider-card",
            div { class: "slider-header",
                label { class: "slider-label", "{label}" }
                div { class: "slider-header-right",
                    if let Some(interval_text) = interval_text {
                        span { class: "slider-interval", "{interval_text}" }
                    }
                    if show_auto_toggle {
                        input {
                            r#type: "checkbox",
                            checked: auto_calibrate_value,
                            title: "Auto-calibrate this parameter",
                            oninput: move |event| {
                                on_auto_calibrate_change.call(event.checked());
                            }
                        }
                    }
                }
            }
            div { class: "slider-track-wrap",
                if let Some(interval_style) = interval_style {
                    div { class: "slider-interval-fill", style: "{interval_style}" }
                }
                input {
                    class: "slider-input",
                    r#type: "range",
                    min: "{min}",
                    max: "{max}",
                    step: "{step}",
                    value: "{value}",
                    oninput: move |event| {
                        if let Ok(parsed) = event.value().parse::<f64>() {
                            on_change.call(parsed);
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn PredictionLogPanel(entries: Vec<PredictionLogEntry>) -> Element {
    rsx! {
        div { class: "panel prediction-log-panel",
            div { class: "panel-header",
                h2 { class: "panel-title", "Backend Prediction Log" }
                span { class: "panel-subtitle", "{entries.len()} entries" }
            }
            if entries.is_empty() {
                p { class: "panel-subtitle", "No backend predictions received yet." }
            } else {
                ul { class: "prediction-log-list",
                    for entry in entries {
                        li { class: "prediction-log-item",
                            span { class: "prediction-log-string", "{entry.full_string}" }
                            span { class: "prediction-log-meta", "[{entry.final_token_lexindex}] ({entry.received_at})" }
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn TrieSvgVisualizer(
    snapshot: ExpandedSnapshot,
    timers: VisibleNodeTimerMap,
    period: f64,
    expansion_threshold: f32,
    scroll_offset: usize,
    scroll_root: String,
    first_fork_depth: Option<usize>,
    show_all: bool,
    use_visual_tutor: bool,
    target_phrase: String,
    light_background: bool,
    show_boxes: bool,
    show_space_connectors: bool,
    show_debug_stats: bool,
    viewport_height: f64,
) -> Element {
    let visible_tree = build_visible_tree(&snapshot, expansion_threshold);
    let offscreen_prefix_display = if scroll_root == "A" {
        String::new()
    } else {
        let path_key = first_fork_node(&visible_tree)
            .map(|node| node.full_string)
            .or_else(|| deepest_visible_node(&visible_tree).map(|node| node.full_string))
            .unwrap_or_else(|| "A".to_string());
        if path_key == "A" {
            String::new()
        } else {
            offscreen_prefix_text(&path_key)
        }
    };
    let laid_out_nodes = compute_laid_out_nodes(
        &snapshot,
        expansion_threshold,
        &scroll_root,
        show_all,
        viewport_height,
    );
    let Some(laid_out_nodes) = laid_out_nodes else {
        return rsx! {
            div { class: "trie-empty", "Waiting for layout…" }
        };
    };

    let mut visible_nodes = laid_out_nodes.values().cloned().collect::<Vec<_>>();
    visible_nodes.sort_by(|a, b| a.full_string.cmp(&b.full_string));
    let content_width = visible_nodes
        .iter()
        .fold(0.0f64, |acc, node| acc.max(node.x + node.width + 90.0));
    let level_gutter = if show_debug_stats { 56.0 } else { 0.0 };
    let tutor_target_key = if use_visual_tutor {
        find_tutor_target_key(
            &snapshot,
            expansion_threshold,
            &scroll_root,
            show_all,
            &target_phrase,
        )
    } else {
        None
    };

    rsx! {
        div { class: "trie-shell",
            svg {
                class: if light_background { "trie-svg light" } else { "trie-svg dark" },
                width: "100%",
                height: "100%",
                view_box: "0 0 {content_width} {viewport_height}",
                preserve_aspect_ratio: "xMinYMin meet",
                line {
                    class: "scroll-guide",
                    x1: "{SCROLL_TARGET_X_PX}",
                    y1: "0",
                    x2: "{SCROLL_TARGET_X_PX}",
                    y2: "{viewport_height}",
                }
                if show_boxes {
                    for node in visible_nodes.iter() {
                        {
                            let (r, g, b) = timer_rgb_on_surface(node.symbol, light_background);
                            let x = scale_x(node.x + 3.0, &node.full_string, &scroll_root, level_gutter);
                            let width = (node.width - 6.0).max(0.0);
                            let tutor = tutor_target_key.as_ref() == Some(&node.full_string);
                            let fill = format!(
                                "rgba({r}, {g}, {b}, {})",
                                if tutor { 0.24 } else { 0.13 }
                            );
                            rsx! {
                                rect {
                                    class: "node-box",
                                    x: "{x}",
                                    y: "{node.y + 3.0}",
                                    width: "{width}",
                                    height: "{(node.height - 6.0).max(0.0)}",
                                    rx: "5",
                                    fill: "{fill}",
                                }
                            }
                        }
                    }
                }
                for node in visible_nodes.iter() {
                    if let Some(parent_key) = &node.parent_key {
                        if let Some(parent) = laid_out_nodes.get(parent_key) {
                            if show_space_connectors || parent.symbol != SPACE_SYMBOL {
                                {
                                    let (r, g, b) = timer_rgb_on_surface(node.symbol, light_background);
                                    let stroke = connector_stroke_style(r, g, b, light_background);
                                    let parent_geom = timer_circle_geometry(parent, &scroll_root, level_gutter);
                                    let child_geom = timer_circle_geometry(node, &scroll_root, level_gutter);
                                    rsx! {
                                        line {
                                            class: "connector-line",
                                            x1: "{parent_geom.0 + parent_geom.2}",
                                            y1: "{parent_geom.1}",
                                            x2: "{child_geom.0 - child_geom.2}",
                                            y2: "{child_geom.1}",
                                            stroke: "{stroke}",
                                            stroke_width: "2",
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for node in visible_nodes.iter() {
                    {
                        let display_text = display_text(node.symbol);
                        let (r, g, b) = timer_rgb_on_surface(node.symbol, light_background);
                        let timer = timers.get(&node.full_string);
                        let timer_frac = timer
                            .map(|timer| timer_fraction(current_time_seconds(), timer.phase, period))
                            .unwrap_or(0.0);
                        let is_tutor_target = tutor_target_key.as_ref() == Some(&node.full_string);
                        let (cx, cy, base_radius) = timer_circle_geometry(node, &scroll_root, level_gutter);
                        let timer_radius = if is_tutor_target { base_radius * 1.8 } else { base_radius };
                        let animation_delay = format!("-{:.6}s", timer_frac * period);
                        let debug_lines = vec![
                            format!("scroll:{scroll_offset}"),
                            format!("fork:{}", first_fork_depth.map(|depth| depth.to_string()).unwrap_or_else(|| "-".to_string())),
                            format_stat("z", Some(node.node.z as f64)),
                            format_stat("tp", node.node.tp.map(|value| value as f64)),
                            format_stat("tp0", node.node.tp0.map(|value| value as f64)),
                            format_stat("p", node.node.p.map(|value| value as f64)),
                            format_stat("a_tl[0]", node.node.a_tl0.map(|value| value as f64)),
                        ];
                        rsx! {
                            if is_tutor_target {
                                circle {
                                    class: "tutor-halo",
                                    cx: "{cx}",
                                    cy: "{cy}",
                                    r: "{timer_radius * 1.55}",
                                    fill: if light_background {
                                        format!("rgba({r}, {g}, {b}, 0.12)")
                                    } else {
                                        "rgba(255, 255, 255, 0.12)".to_string()
                                    },
                                }
                            }
                            if node.symbol == STOP_SYMBOL {
                                rect {
                                    class: "node-glyph",
                                    x: "{cx - 8.5}",
                                    y: "{cy - 8.5}",
                                    width: "17",
                                    height: "17",
                                    fill: "rgba({r}, {g}, {b}, 1)",
                                }
                            } else {
                                text {
                                    class: "node-label",
                                    x: "{cx}",
                                    y: "{cy}",
                                    text_anchor: "middle",
                                    dominant_baseline: "middle",
                                    fill: "rgba({r}, {g}, {b}, 1)",
                                    font_size: if is_tutor_target { "66" } else { "37" },
                                    "{display_text}"
                                }
                            }
                            if timer.is_some() {
                                circle {
                                    class: "timer-arc timer-arc-animated",
                                    cx: "{cx}",
                                    cy: "{cy}",
                                    r: "{timer_radius}",
                                    fill: "none",
                                    stroke: "rgba({r}, {g}, {b}, 1)",
                                    stroke_width: if is_tutor_target { "4" } else { "2" },
                                    "pathLength": "1",
                                    "stroke-dasharray": "0 1",
                                    style: "animation-duration: {period}s; animation-delay: {animation_delay};",
                                }
                            }
                            if show_debug_stats {
                                for (index , line) in debug_lines.iter().enumerate() {
                                    text {
                                        class: "debug-line",
                                        x: "{cx}",
                                        y: "{cy + timer_radius + 14.0 + index as f64 * 10.0}",
                                        text_anchor: "middle",
                                        fill: if light_background { "rgba(15, 23, 42, 0.9)" } else { "rgba(255, 255, 255, 0.9)" },
                                        "{line}"
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !offscreen_prefix_display.is_empty() {
                div { class: "offscreen-prefix",
                    div { class: "offscreen-prefix-title", "Scrolled prefix" }
                    div { class: "offscreen-prefix-body", "{offscreen_prefix_display}" }
                }
            }
        }
    }
}

fn scale_x(value: f64, full_string: &str, scroll_root: &str, level_gutter: f64) -> f64 {
    value + relative_depth(full_string, scroll_root) as f64 * level_gutter
}

fn timer_fraction(time: f64, phase: f64, period: f64) -> f64 {
    ((time - phase + period) % period) / period
}

fn timer_circle_geometry(
    node: &layout::VisualNode,
    scroll_root: &str,
    level_gutter: f64,
) -> (f64, f64, f64) {
    let mut timer_radius = 15.0;
    if matches!(node.symbol, 'm' | 'w') {
        timer_radius *= 1.15;
    }
    let cx = scale_x(
        node.x + node.width,
        &node.full_string,
        scroll_root,
        level_gutter,
    ) - 15.0;
    let cy = node.y + node.height / 2.0;
    (cx, cy, timer_radius)
}

fn connector_stroke_style(r: u8, g: u8, b: u8, light_background: bool) -> String {
    let dr = (r as f64 / 1.8).floor() as u8;
    let dg = (g as f64 / 1.8).floor() as u8;
    let db = (b as f64 / 1.8).floor() as u8;
    if !light_background {
        format!("rgba({dr}, {dg}, {db}, 1)")
    } else {
        let lift = 88u8;
        let lr = dr.saturating_add(lift);
        let lg = dg.saturating_add(lift);
        let lb = db.saturating_add(lift);
        format!("rgba({lr}, {lg}, {lb}, 0.4)")
    }
}

fn timer_rgb_on_surface(symbol: char, light_background: bool) -> (u8, u8, u8) {
    let (r, g, b) = if symbol == SPACE_SYMBOL {
        color_from_letter(' ')
    } else if symbol == 'A' {
        (255, 255, 255)
    } else {
        color_from_letter(symbol)
    };
    if !light_background {
        return (r, g, b);
    }
    let lum = 0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64;
    if lum < 168.0 {
        return (r, g, b);
    }
    let t = ((lum - 168.0) / 88.0).clamp(0.0, 1.0);
    (
        (r as f64 + (30.0 - r as f64) * t).round() as u8,
        (g as f64 + (41.0 - g as f64) * t).round() as u8,
        (b as f64 + (59.0 - b as f64) * t).round() as u8,
    )
}

fn format_stat(label: &str, value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{label}:{value:.2}"),
        None => format!("{label}:-"),
    }
}

fn current_time_seconds() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time after epoch")
        .as_secs_f64()
}
