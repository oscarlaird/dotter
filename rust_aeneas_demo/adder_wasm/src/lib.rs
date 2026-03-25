#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub extern "C" fn add_u32_wasm(x: u32, y: u32) -> u32 {
    adder_core::add_u32(x, y)
}
