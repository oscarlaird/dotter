//! Debug logging to the browser console when the crate is built with `feature = "wasm"`.
//! On other targets the [`log`] function is a no-op.

#[cfg(feature = "wasm")]
pub fn log(msg: &str) {
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(msg));
}

#[cfg(not(feature = "wasm"))]
#[inline]
pub fn log(_msg: &str) {}

/// `trie_debug!("nodes={}", n)` → `console.log` on wasm; no-op elsewhere.
#[macro_export]
macro_rules! trie_debug {
    ($($t:tt)*) => {{
        $crate::trie::debug::log(&format!($($t)*));
    }};
}
