const wasm_url = "http://localhost:5173/root.wasm";

const logger = {
    buffer: "",
    write_char(c) {
        // c is a u8 (0-255), convert to string using TextDecoder
        const bytes = new Uint8Array([c]);
        const decoder = new TextDecoder('utf-8');
        this.buffer = this.buffer + decoder.decode(bytes);
    },
    flush() {
        console.log(this.buffer);
        this.buffer = "";
    }
}
async function initWasm() {
    try {
        const response = await fetch(wasm_url);
        const wasmBuffer = await response.arrayBuffer();
        const importObject = {
            env: {
                memory: new WebAssembly.Memory({initial: 1024, maximum: 1024}),
                write_char: logger.write_char.bind(logger),
                flush: logger.flush.bind(logger)
            }
        };
        const wasmModule = await WebAssembly.instantiate(wasmBuffer, importObject);
        console.log(wasmModule.instance.exports.add(1, 2));
        let arr_ptr = wasmModule.instance.exports.setup();
        console.log("arr_ptr: ", arr_ptr);
        wasmModule.instance.exports.read_and_print_arr_to_js(arr_ptr);

    } catch (error) {
        console.error("Error loading WASM module:", error);
    }
}

initWasm();
