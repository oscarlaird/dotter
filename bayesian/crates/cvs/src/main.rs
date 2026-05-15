use dioxus::prelude::*;

mod canvas;
mod opfs;

use serde::{Deserialize, Serialize};

use crate::canvas::CanvasComponent;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");
const SAVE_FILE_NAME: &str = "cvs-state.json";

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct AppState {
    number: i32,
}

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let mut number = use_signal(|| 0);
    let mut storage_status = use_signal(|| "OPFS state not loaded".to_string());

    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS } document::Link { rel: "stylesheet", href: TAILWIND_CSS }
        "Hello world"
        div { "The number is {number}" }
        button { onclick: move |_| {*number.write() += 1}, "increment" }
        button {
            onclick: move |_| {
                let state = AppState { number: number() };
                spawn(async move {
                    match opfs::save_json(SAVE_FILE_NAME, &state).await {
                        Ok(()) => storage_status.set(format!("Saved {SAVE_FILE_NAME} to OPFS")),
                        Err(err) => storage_status.set(format!("Save failed: {err}")),
                    }
                });
            },
            "save"
        }
        button {
            onclick: move |_| {
                spawn(async move {
                    match opfs::load_json::<AppState>(SAVE_FILE_NAME).await {
                        Ok(state) => {
                            number.set(state.number);
                            storage_status.set(format!("Loaded {SAVE_FILE_NAME} from OPFS"));
                        }
                        Err(err) => storage_status.set(format!("Load failed: {err}")),
                    }
                });
            },
            "load"
        }
        div { "{storage_status}" }
        CanvasComponent {}
    }
}
