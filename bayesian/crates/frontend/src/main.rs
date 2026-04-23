mod app;
mod backend;
mod components;
mod domain;

fn main() {
    dioxus::launch(app::App);
}
