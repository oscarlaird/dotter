use macroquad::prelude::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "macroquad tutor".to_owned(),
        window_width: 800, window_height: 600, high_dpi: true, sample_count: 4,
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
    fn y_center(&self) -> f32 {
        self.ul.1 + self.height / 2.0
    }
}

#[derive(Default)]
struct ClickCanvas<'a> {
    bboxes: Vec<BBox>,
    callbacks: Vec<Box<dyn FnMut(Point) -> () + 'a>>
}

impl<'a> ClickCanvas<'a> {
    fn draw_callback(&mut self, bbox: BBox, callback: Box<dyn FnMut(Point) -> () + 'a>) {
        self.bboxes.push(bbox);
        self.callbacks.push(callback);
    }
}

fn slider<'a>(bbox: BBox, range: (f32, f32), val: &'a mut f32, click_canvas: &mut ClickCanvas<'a>) {
    // slider bar
    let slider_height = (12.0f32).min(bbox.height);
    let slider_ul = Point(
        bbox.ul.0,
        bbox.y_center() - slider_height/2.0
    );
    draw_rectangle(
        slider_ul.0+2.0, slider_ul.1+2.0, bbox.width, slider_height, 
        GREEN
    );
    draw_rectangle(
        slider_ul.0, slider_ul.1, bbox.width, slider_height, 
        RED
    );
    // slider circle
    let bar_frac = *val / (range.1 - range.0);
    draw_circle(
        slider_ul.0+3.0 + bar_frac * bbox.width,
        bbox.y_center()+2.0,
        16.0,
        RED,
    );
    draw_circle(
        slider_ul.0 + bar_frac * bbox.width,
        bbox.y_center(),
        16.0,
        GREEN,
    );
    let callback = move |click_point: Point| {
        println!("callback called for point {:?}", click_point);
        let click_perc = (click_point.0 - bbox.ul.0) / bbox.width;
        *val = range.0 + click_perc * (range.1 - range.0);
        println!("New value: {}", *val);
    };
    click_canvas.draw_callback(
        bbox,
        Box::new(callback)
    );

}

#[macroquad::main(window_conf)]
async fn main() {
    let mean = &mut 32.0;
    loop {
        clear_background(BLACK);
        let slider_box = BBox { ul: Point(10.0,10.0), width: 200.0, height: 100.0 };
        let mut click_canvas = ClickCanvas::default();
        // draw
        draw_text(
            &format!("Welcome to Macroquad! Mean = {}", *mean),
            100.0,
            100.0,
            32.0,
            WHITE
        );
        slider(slider_box, (0.0, 100.0), mean, &mut click_canvas);
        next_frame().await;
        // wait for input
        if is_mouse_button_pressed(MouseButton::Left) {
            let (x, y) = mouse_position();
            let click_point = Point(x, y);
            for mut boxed_callback in click_canvas.callbacks {
                boxed_callback(click_point);
            }
        }
    }
}
