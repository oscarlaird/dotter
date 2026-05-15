use dioxus::prelude::*;

use wasm_bindgen::{closure::Closure, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

#[component]
pub fn CanvasComponent() -> Element {
    let mut canvas = use_signal(|| None::<HtmlCanvasElement>);
    let mut speed = use_signal(|| 1.0_f64);

    use_effect(move || {
        let Some(canvas) = canvas() else {
            return;
        };

        spawn(async move {
            loop {
                let frame_time_ms = next_animation_frame().await;
                draw_frame(&canvas, frame_time_ms, speed());
            }
        });
    });

    rsx! {
        div {
            style: "display: grid; gap: 12px; width: 400px;",
            label {
                "Speed: {speed():.2}"
                input {
                    r#type: "range",
                    min: "0.25",
                    max: "4.0",
                    step: "0.05",
                    value: "{speed()}",
                    oninput: move |event| speed.set(event.value().parse().unwrap()),
                }
            }
            canvas {
                onmounted: move |mounted| {
                    let mounted = mounted.data();
                    let element = mounted.downcast::<web_sys::Element>().unwrap();
                    let canvas_element: HtmlCanvasElement = element.clone().dyn_into().unwrap();
                    canvas.set(Some(canvas_element));
                },
                width: "400",
                height: "400",
                style: "border: 1px solid black;"
            }
        }
    }
}

async fn next_animation_frame() -> f64 {
    let (sender, receiver) = futures_channel::oneshot::channel();
    let callback = Closure::once_into_js(move |timestamp: f64| {
        sender.send(timestamp).unwrap();
    });
    web_sys::window()
        .unwrap()
        .request_animation_frame(callback.unchecked_ref())
        .unwrap();
    receiver.await.unwrap()
}

fn draw_frame(canvas: &HtmlCanvasElement, time_ms: f64, speed: f64) {
    let ctx = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();
    let width = canvas.width() as f64;
    let height = canvas.height() as f64;
    let size = 40.0;
    let x = ((time_ms / 1_000.0) * 120.0 * speed).rem_euclid(width - size);
    let y = height / 2.0 - size / 2.0;

    ctx.clear_rect(0.0, 0.0, width, height);
    ctx.set_fill_style_str("red");
    ctx.fill_rect(x, y, size, size);
}
