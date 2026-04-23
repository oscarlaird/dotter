use std::fs;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bayesian::render_utils::ExpandedSnapshot;
use bayesian::{BayesianSession, LikelihoodNodeInput, LikelihoodUpdatePayload, PriorUpdatePayload};
use dioxus::prelude::*;
use layout::{
    build_likelihood_payload_nodes, compute_scroll_layout_state, find_tutor_target_key,
    modulo_delay, timers_for_snapshot, VisibleNodeTimerMap, ROOT_SYMBOL, SCROLL_CENTERING_WEIGHT,
    SCROLL_STABILITY_WEIGHT,
};

use crate::backend::{BackendCommand, BackendEvent, BackendHandle};
use crate::components::{
    CalibrationSettingsPanel, PredictionLogEntry, PredictionLogPanel, TrieSvgVisualizer,
};
use crate::domain::{
    default_likelihood_model, random_practice_phrase, variational_params_to_likelihood_model,
    AutoCalibrationState, CalibratedLikelihoodModel, ColorMode,
};

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");
const VIEWPORT_HEIGHT: f64 = 900.0;

#[component]
pub fn App() -> Element {
    let mut session = use_signal(BayesianSession::new);
    let backend = use_signal(|| BackendHandle::new("ws://localhost:8000/ws"));
    let mut head_assets_injected = use_signal(|| false);

    let mut loading = use_signal(|| true);
    let mut snapshot = use_signal(|| None::<ExpandedSnapshot>);
    let mut timers = use_signal(VisibleNodeTimerMap::new);
    let mut error = use_signal(|| None::<String>);
    let mut warning = use_signal(|| None::<String>);
    let mut prediction_log = use_signal(Vec::<PredictionLogEntry>::new);
    let last_batch_size = use_signal(|| 0usize);

    let likelihood_model = use_signal(default_likelihood_model);
    let use_automatic_calibration = use_signal(AutoCalibrationState::default);
    let mut auto_calibration_likelihood_model = use_signal(|| CalibratedLikelihoodModel {
        model: default_likelihood_model(),
        intervals: None,
    });
    let mut calibration_sample_count = use_signal(|| 0usize);
    let mut raw_variational_params = use_signal(|| None);
    let mut recent_calibration_pairs = use_signal(Vec::new);
    let mut username_input = use_signal(String::new);
    let mut active_username = use_signal(|| None::<String>);
    let mut current_vi_before = use_signal(|| None);

    let mut show_prediction_log_panel = use_signal(|| false);
    let mut show_calibration_debug_panel = use_signal(|| false);
    let mut color_mode = use_signal(|| ColorMode::Dark);
    let mut show_boxes = use_signal(|| true);
    let mut show_space_connectors = use_signal(|| true);
    let mut show_debug_stats = use_signal(|| false);
    let mut show_all = use_signal(|| false);
    let mut show_practice_phrase = use_signal(|| false);
    let mut use_visual_tutor = use_signal(|| false);
    let mut use_audio_tutor = use_signal(|| false);
    let mut practice_phrase = use_signal(|| random_practice_phrase(None));

    let mut scroll_offset = use_signal(|| 0usize);
    let mut scroll_root = use_signal(|| ROOT_SYMBOL.to_string());
    let mut scroll_ancestor_keys = use_signal(Vec::new);
    let mut first_fork_depth = use_signal(|| None::<usize>);
    let mut expansion_threshold = use_signal(|| f32::NEG_INFINITY);
    let mut ws_status = use_signal(|| "Connecting…".to_string());
    let mut now_seconds = use_signal(current_time_seconds);

    let mut apply_vi_before_to_ui = move |vi_before: bayesian::calibration::VariationalParams| {
        current_vi_before.set(Some(vi_before));
        calibration_sample_count.set(0);
        raw_variational_params.set(Some(vi_before));
        recent_calibration_pairs.set(Vec::new());
        auto_calibration_likelihood_model.set(variational_params_to_likelihood_model(
            vi_before,
            likelihood_model().period,
        ));
    };

    let mut apply_snapshot = move |next_snapshot: ExpandedSnapshot, reset_all_timers: bool| {
        let next_scroll_layout =
            compute_scroll_layout_state(&next_snapshot, expansion_threshold(), scroll_offset());
        let next_timers = timers_for_snapshot(
            &next_snapshot,
            &likelihood_model(),
            &timers(),
            reset_all_timers,
            &next_scroll_layout.rendered_node_keys,
        );
        scroll_offset.set(next_scroll_layout.scroll_offset);
        scroll_root.set(next_scroll_layout.scroll_root.clone());
        scroll_ancestor_keys.set(next_scroll_layout.scroll_ancestor_keys.clone());
        first_fork_depth.set(next_scroll_layout.first_fork_depth);
        snapshot.set(Some(next_snapshot));
        timers.set(next_timers);
    };

    use_effect(move || {
        let threshold = session.read().expansion_threshold();
        expansion_threshold.set(threshold);
        let initial_snapshot = session.write().expand_to_threshold_snapshot();
        apply_snapshot(initial_snapshot, true);
        loading.set(false);
    });

    use_effect(move || {
        if head_assets_injected() {
            return;
        }
        head_assets_injected.set(true);
        let mut error = error;
        spawn(async move {
            let favicon = FAVICON.to_string();
            let main_css = MAIN_CSS.to_string();
            let tailwind_css = TAILWIND_CSS.to_string();
            let script = format!(
                r#"
                (() => {{
                    const ensureLink = (selector, rel, href) => {{
                        let existing = document.head.querySelector(selector);
                        if (existing) {{
                            existing.setAttribute('href', href);
                            existing.setAttribute('rel', rel);
                            return;
                        }}
                        const link = document.createElement('link');
                        link.setAttribute('rel', rel);
                        link.setAttribute('href', href);
                        document.head.appendChild(link);
                    }};

                    ensureLink('link[data-dotter-favicon=\"true\"]', 'icon', {favicon:?});
                    let favicon = document.head.querySelector('link[data-dotter-favicon=\"true\"]');
                    if (!favicon) {{
                        favicon = document.head.querySelector('link[rel=\"icon\"][href={favicon:?}]');
                    }}
                    if (favicon) {{
                        favicon.setAttribute('data-dotter-favicon', 'true');
                    }}

                    ensureLink('link[data-dotter-main-css=\"true\"]', 'stylesheet', {main_css:?});
                    let mainCss = document.head.querySelector('link[data-dotter-main-css=\"true\"]');
                    if (!mainCss) {{
                        mainCss = document.head.querySelector('link[rel=\"stylesheet\"][href={main_css:?}]');
                    }}
                    if (mainCss) {{
                        mainCss.setAttribute('data-dotter-main-css', 'true');
                    }}

                    ensureLink('link[data-dotter-tailwind-css=\"true\"]', 'stylesheet', {tailwind_css:?});
                    let tailwindCss = document.head.querySelector('link[data-dotter-tailwind-css=\"true\"]');
                    if (!tailwindCss) {{
                        tailwindCss = document.head.querySelector('link[rel=\"stylesheet\"][href={tailwind_css:?}]');
                    }}
                    if (tailwindCss) {{
                        tailwindCss.setAttribute('data-dotter-tailwind-css', 'true');
                    }}

                    return true;
                }})()
            "#,
                favicon = favicon,
                main_css = main_css,
                tailwind_css = tailwind_css
            );
            match document::eval(&script).await {
                Ok(_) => {}
                Err(err) => {
                    error.set(Some(format!("head asset injection failed: {err}")));
                }
            }
        });
    });

    use_effect(move || {
        spawn(async move {
            loop {
                now_seconds.set(current_time_seconds());
                tokio::time::sleep(Duration::from_millis(33)).await;
            }
        });
    });

    use_effect(move || {
        spawn(async move {
            loop {
                for event in backend().drain_events() {
                    match event {
                        BackendEvent::Connected => {
                            ws_status.set("Connected".to_string());
                            warning.set(None);
                            let trimmed = username_input().trim().to_string();
                            if !trimmed.is_empty() {
                                backend().send(BackendCommand::StartSession { username: trimmed });
                            }
                        }
                        BackendEvent::Disconnected(reason) => {
                            ws_status.set("Disconnected".to_string());
                            warning.set(Some(format!(
                                "Backend disconnected ({reason}). Local likelihood updates still apply."
                            )));
                        }
                        BackendEvent::SessionStarted {
                            username,
                            variational_params,
                        } => {
                            active_username.set(Some(username));
                            prediction_log.set(Vec::new());
                            error.set(None);
                            let next_snapshot = {
                                let mut session = session.write();
                                session.reset();
                                session.set_current_prior(variational_params);
                                session.expand_to_threshold_snapshot()
                            };
                            apply_snapshot(next_snapshot, true);
                            apply_vi_before_to_ui(variational_params);
                            backend().send(BackendCommand::RequestNextPrior);
                        }
                        BackendEvent::PriorUpdate { content_json } => {
                            if let Ok(payload) =
                                serde_json::from_str::<PriorUpdatePayload>(&content_json)
                            {
                                let mut next_log = prediction_log();
                                next_log.insert(
                                    0,
                                    PredictionLogEntry {
                                        id: next_log.len() + 1,
                                        full_string: payload.full_string.clone(),
                                        final_token_lexindex: format!(
                                            "{:?}",
                                            payload.final_token_lexindex
                                        ),
                                        received_at: human_time_label(),
                                    },
                                );
                                prediction_log.set(next_log);
                            }
                            let next_snapshot = {
                                let mut session = session.write();
                                session.receive_prior_update(content_json);
                                session.apply_updates();
                                session.expand_to_threshold_snapshot()
                            };
                            apply_snapshot(next_snapshot, false);
                        }
                        BackendEvent::ResetAck => {
                            error.set(None);
                            prediction_log.set(Vec::new());
                        }
                        BackendEvent::Error(message) => {
                            error.set(Some(message));
                        }
                    }
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });
    });

    use_effect(move || {
        if !show_practice_phrase() {
            if use_visual_tutor() {
                use_visual_tutor.set(false);
            }
            if use_audio_tutor() {
                use_audio_tutor.set(false);
            }
        }
    });

    let tutor_target_key = snapshot().and_then(|snapshot| {
        if !show_practice_phrase() {
            None
        } else {
            find_tutor_target_key(
                &snapshot,
                expansion_threshold(),
                &scroll_root(),
                show_all(),
                &practice_phrase(),
            )
        }
    });

    let make_run_likelihood_pulse = move || {
        let snapshot = snapshot;
        let current_vi_before = current_vi_before;
        let mut error = error;
        let show_practice_phrase = show_practice_phrase;
        let use_audio_tutor = use_audio_tutor;
        let tutor_target_key = tutor_target_key.clone();
        let timers = timers;
        let likelihood_model = likelihood_model;
        let mut warning = warning;
        let scroll_root = scroll_root;
        let scroll_ancestor_keys = scroll_ancestor_keys;
        let mut session = session;
        let mut apply_snapshot = apply_snapshot;
        let backend = backend;
        let mut calibration_sample_count = calibration_sample_count;
        let mut raw_variational_params = raw_variational_params;
        let mut recent_calibration_pairs = recent_calibration_pairs;
        let mut auto_calibration_likelihood_model = auto_calibration_likelihood_model;
        let mut last_batch_size = last_batch_size;

        move |time_seconds: f64| {
            let Some(snapshot_value) = snapshot() else {
                return;
            };
            let Some(vi_before) = current_vi_before() else {
                error.set(Some(
                    "Start a session before sending likelihood updates".to_string(),
                ));
                return;
            };

            if show_practice_phrase() && use_audio_tutor() {
                if let Some(tutor_target_key) = tutor_target_key.clone() {
                    if let Some(target_timer) = timers().get(&tutor_target_key) {
                        let predictive_stddev = likelihood_model().stddev_delay;
                        if predictive_stddev > 0.0 {
                            let x = modulo_delay(
                                time_seconds,
                                target_timer.phase,
                                likelihood_model().period,
                            );
                            let offset_stddevs =
                                (x - likelihood_model().mu_delay) / predictive_stddev;
                            warning.set(Some(format!(
                                "Audio tutor not wired on native desktop yet (offset {offset_stddevs:.2} stddevs)."
                            )));
                        }
                    }
                }
            }

            let likelihood_nodes = build_likelihood_payload_nodes(
                &snapshot_value,
                &timers(),
                time_seconds,
                &likelihood_model(),
                &scroll_root(),
                &scroll_ancestor_keys(),
            );
            let payload = LikelihoodUpdatePayload {
                period: likelihood_model().period,
                y: time_seconds,
                nodes: likelihood_nodes
                    .iter()
                    .map(|(full_string, node)| {
                        (
                            full_string.clone(),
                            LikelihoodNodeInput {
                                likelihood: node.likelihood as f32,
                                symbol: None,
                                phase: node.phase,
                            },
                        )
                    })
                    .collect(),
            };
            let likelihood_json =
                serde_json::to_string(&payload).expect("likelihood payload serialization");
            let (next_snapshot, recalibration_result) = {
                let mut session = session.write();
                session.receive_likelihood_update_typed(payload);
                session.apply_updates();
                let next_snapshot = session.expand_to_threshold_snapshot();
                let recalibration = session.recalibrate_typed(vi_before, true);
                (next_snapshot, recalibration)
            };
            apply_snapshot(next_snapshot, true);
            backend().send(BackendCommand::LikelihoodUpdate {
                content_json: likelihood_json,
            });
            calibration_sample_count.set(recalibration_result.used_likelihood_updates);
            raw_variational_params.set(Some(recalibration_result.prior_params));
            recent_calibration_pairs.set(recalibration_result.recent_pairs.clone());
            auto_calibration_likelihood_model.set(variational_params_to_likelihood_model(
                recalibration_result.prior_params,
                likelihood_model().period,
            ));
            last_batch_size.set(likelihood_nodes.len());
            error.set(None);
        }
    };

    let mut run_likelihood_pulse = make_run_likelihood_pulse();
    let mut run_likelihood_pulse_button = make_run_likelihood_pulse();

    let make_reset_both_sides = move || {
        let current_vi_before = current_vi_before;
        let mut error = error;
        let mut session = session;
        let mut apply_snapshot = apply_snapshot;
        let mut apply_vi_before_to_ui = apply_vi_before_to_ui;
        let backend = backend;

        move || {
            let Some(vi_before) = current_vi_before() else {
                error.set(Some(
                    "Session has no current calibration prior; start a session first".to_string(),
                ));
                return;
            };
            let recalibration_result = session.write().recalibrate_typed(vi_before, true);
            let next_snapshot = {
                let mut session = session.write();
                session.reset();
                session.set_current_prior(recalibration_result.prior_params);
                session.expand_to_threshold_snapshot()
            };
            apply_snapshot(next_snapshot, true);
            apply_vi_before_to_ui(recalibration_result.prior_params);
            backend().send(BackendCommand::Reset);
            error.set(None);
        }
    };

    let mut reset_both_sides = make_reset_both_sides();
    let mut reset_both_sides_button = make_reset_both_sides();

    let mut handle_download_session_debug_dump = move || {
        let dump = session.read().debug_dump();
        let json = serde_json::to_string_pretty(&dump).expect("debug dump serialization");
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current time after epoch")
            .as_secs();
        let filename = format!("bayesian-session-dump-{timestamp}.json");
        match fs::write(&filename, json) {
            Ok(()) => warning.set(Some(format!("Wrote session dump to {filename}"))),
            Err(err) => error.set(Some(format!("failed to write session dump: {err}"))),
        }
    };

    rsx! {
        div {
            class: if color_mode().is_light() { "app-shell light" } else { "app-shell dark" },
            tabindex: "0",
            onkeydown: move |event| {
                let key = format!("{:?}", event.key());
                if key == "Escape" {
                    reset_both_sides();
                    return;
                }
                if key == "Character(\" \")" || key == "Space" {
                    event.prevent_default();
                    run_likelihood_pulse(current_time_seconds());
                }
            },
            div { class: "app-frame",
                header { class: "header-bar",
                    div { class: "header-left",
                        span { "WS ", code { "{ws_status}" } }
                        span { class: "dot-separator", "·" }
                        span {
                            if let Some(snapshot) = snapshot() {
                                code { "{snapshot.len()}" }
                                " nodes"
                            } else {
                                "no snapshot"
                            }
                        }
                        span { class: "dot-separator", "·" }
                        span { "last batch ", code { "{last_batch_size}" } }
                        span { class: "dot-separator", "·" }
                        span {
                            title: "Scroll heuristic weights from the React app",
                            "scroll ",
                            code { "a={SCROLL_CENTERING_WEIGHT}" },
                            " ",
                            code { "b={SCROLL_STABILITY_WEIGHT}" }
                        }
                        span { class: "dot-separator", "·" }
                        span { code { "Space" } " pulse ", code { "Esc" } " reset" }
                        if show_practice_phrase() {
                            span { class: "dot-separator", "·" }
                            button {
                                class: "icon-button",
                                onclick: move |_| practice_phrase.set(random_practice_phrase(Some(&practice_phrase()))),
                                "Shuffle"
                            }
                            span { class: "practice-phrase", "{practice_phrase}" }
                        }
                    }
                    div { class: "header-right",
                        div { class: "session-controls",
                            input {
                                class: "username-input",
                                value: "{username_input}",
                                placeholder: "username",
                                oninput: move |event| username_input.set(event.value()),
                            }
                            button {
                                class: "action-button",
                                onclick: move |_| {
                                    let trimmed = username_input().trim().to_string();
                                    if trimmed.is_empty() {
                                        error.set(Some("Username must be non-empty".to_string()));
                                    } else {
                                        backend().send(BackendCommand::StartSession { username: trimmed });
                                        error.set(None);
                                    }
                                },
                                "Start session"
                            }
                            span {
                                class: "session-status",
                                if let Some(active_username) = active_username() {
                                    "active: {active_username}"
                                } else {
                                    "no active session"
                                }
                            }
                        }
                        Toggle {
                            label: "Blink to click (deferred)",
                            checked: false,
                            disabled: true,
                            on_toggle: move |_| {}
                        }
                        Toggle {
                            label: "Show all",
                            checked: show_all(),
                            disabled: false,
                            on_toggle: move |enabled| show_all.set(enabled)
                        }
                        Toggle {
                            label: "Debug",
                            checked: show_debug_stats(),
                            disabled: false,
                            on_toggle: move |enabled| show_debug_stats.set(enabled)
                        }
                        Toggle {
                            label: "Node boxes",
                            checked: show_boxes(),
                            disabled: false,
                            on_toggle: move |enabled| show_boxes.set(enabled)
                        }
                        Toggle {
                            label: "Space→child lines",
                            checked: show_space_connectors(),
                            disabled: false,
                            on_toggle: move |enabled| show_space_connectors.set(enabled)
                        }
                        Toggle {
                            label: "Practice",
                            checked: show_practice_phrase(),
                            disabled: false,
                            on_toggle: move |enabled| show_practice_phrase.set(enabled)
                        }
                        Toggle {
                            label: "Visual tutor",
                            checked: use_visual_tutor(),
                            disabled: !show_practice_phrase(),
                            on_toggle: move |enabled| use_visual_tutor.set(enabled)
                        }
                        Toggle {
                            label: "Audio tutor",
                            checked: use_audio_tutor(),
                            disabled: !show_practice_phrase(),
                            on_toggle: move |enabled| use_audio_tutor.set(enabled)
                        }
                        button {
                            class: "action-button",
                            onclick: move |_| color_mode.set(color_mode().toggle()),
                            if color_mode() == ColorMode::Dark { "Light" } else { "Dark" }
                        }
                        button {
                            class: if show_calibration_debug_panel() { "action-button active" } else { "action-button" },
                            onclick: move |_| show_calibration_debug_panel.set(!show_calibration_debug_panel()),
                            "Calibration debug"
                        }
                        button {
                            class: if show_prediction_log_panel() { "action-button active" } else { "action-button" },
                            onclick: move |_| show_prediction_log_panel.set(!show_prediction_log_panel()),
                            "Backend log"
                        }
                        button {
                            class: "action-button",
                            onclick: move |_| handle_download_session_debug_dump(),
                            "Dump logs"
                        }
                        button {
                            class: "action-button",
                            onclick: move |_| run_likelihood_pulse_button(current_time_seconds()),
                            "Pulse"
                        }
                        button {
                            class: "action-button",
                            onclick: move |_| reset_both_sides_button(),
                            "Reset"
                        }
                    }
                }

                if let Some(warning_text) = warning() {
                    div { class: "banner warning-banner", "{warning_text}" }
                }
                if let Some(error_text) = error() {
                    div { class: "banner error-banner", "{error_text}" }
                }

                CalibrationSettingsPanel {
                    use_automatic_calibration,
                    likelihood_model,
                    auto_calibration_likelihood_model: auto_calibration_likelihood_model(),
                    calibration_sample_count: calibration_sample_count(),
                    raw_variational_params: raw_variational_params(),
                    recent_calibration_pairs: recent_calibration_pairs(),
                    show_calibration_debug: show_calibration_debug_panel(),
                }

                if show_prediction_log_panel() {
                    PredictionLogPanel { entries: prediction_log() }
                }

                div { class: "trie-frame",
                    if loading() {
                        div { class: "trie-empty", "Loading bayesian session…" }
                    } else if let Some(snapshot_value) = snapshot() {
                        TrieSvgVisualizer {
                            snapshot: snapshot_value,
                            timers: timers(),
                            period: likelihood_model().period,
                            expansion_threshold: expansion_threshold(),
                            scroll_offset: scroll_offset(),
                            scroll_root: scroll_root(),
                            first_fork_depth: first_fork_depth(),
                            use_visual_tutor: show_practice_phrase() && use_visual_tutor(),
                            target_phrase: practice_phrase(),
                            show_all: show_all(),
                            light_background: color_mode().is_light(),
                            show_boxes: show_boxes(),
                            show_space_connectors: show_space_connectors(),
                            show_debug_stats: show_debug_stats(),
                            viewport_height: VIEWPORT_HEIGHT,
                            now_seconds: now_seconds(),
                        }
                    } else {
                        div { class: "trie-empty", "Waiting for the first visible nodes." }
                    }
                }
            }
        }
    }
}

#[component]
fn Toggle(
    label: &'static str,
    checked: bool,
    disabled: bool,
    mut on_toggle: EventHandler<bool>,
) -> Element {
    rsx! {
        label { class: if disabled { "toggle disabled" } else { "toggle" },
            input {
                r#type: "checkbox",
                checked: checked,
                disabled: disabled,
                oninput: move |event| on_toggle.call(event.checked()),
            }
            span { "{label}" }
        }
    }
}

fn current_time_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time after epoch")
        .as_secs_f64()
}

fn human_time_label() -> String {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time after epoch")
        .as_secs();
    let seconds = elapsed % 60;
    let minutes = (elapsed / 60) % 60;
    let hours = (elapsed / 3600) % 24;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}
