use macroquad::prelude::*;


use super::*;

pub(crate) fn slider<'a>(bbox: BBox, range: (f32, f32), val: &'a mut f32, callback_canvas: &mut CallbackCanvas<'a>, id: Option<WidgetId>) {
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
    let bar_frac = (*val - range.0) / (range.1 - range.0);
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
    callback_canvas.push(CallbackRegion {
        bbox,
        click_callback: Some(Box::new(callback)),
        keys_callback: None,
        id
    });

}