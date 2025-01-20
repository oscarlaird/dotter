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

let wasmModule_exports = null;
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
        let wasmModule = await WebAssembly.instantiate(wasmBuffer, importObject);
        wasmModule_exports = wasmModule.instance.exports;
        // // TRIE
        // let root = wasmModule_exports.setup_trie();    
        // // WORD 1
        // let word = "hello";
        // let word_len = word.length;
        // let word_buffer = wasmModule_exports.allocate_word_buffer(word_len);
        // let word_buffer_ptr = new Uint8Array(wasmModule_exports.memory.buffer, word_buffer, word_len);
        // word_buffer_ptr.set([...word].map(c => c.charCodeAt(0)));
        // wasmModule_exports.add_word_to_trie(root, word_buffer, word_len, 0.1, false);
        // // WORD 2
        // let word2 = "helpful";
        // let word2_len = word2.length;
        // let word2_buffer = wasmModule_exports.allocate_word_buffer(word2_len);
        // let word2_buffer_ptr = new Uint8Array(wasmModule_exports.memory.buffer, word2_buffer, word2_len);
        // word2_buffer_ptr.set([...word2].map(c => c.charCodeAt(0)));
        // wasmModule_exports.add_word_to_trie(root, word2_buffer, word2_len, 0.1, false);
        // // let prob = wasmModule.instance.exports.add_word_to_trie(root, word_buffer, word_len, 0.1, true);
        // let prob = wasmModule_exports.add_word_to_trie(root, word2_buffer, word2_len, 0.2, true);
        // console.log("prole:", prob);
        // let json_string = wasmModule_exports.jsonifyTrie(root);
        // let json_string_buffer = new Uint8Array(wasmModule_exports.memory.buffer, json_string, 2000);
        // const decoder = new TextDecoder('utf-8');
        // const jsonStr = decoder.decode(json_string_buffer.slice(0, 2000));
        // console.log("json_trie:", jsonStr);
    } catch (error) {
        console.error("Error loading WASM module:", error);
    }

    onmessage = (event) => {
        let data = event.data;
        if (data.type === 'test') {
            let content = data.content;
            console.log("worker 3 received test message:", content);
            console.log("probs_length:", content.probs.length);
            let lm_result_ptr = wasmModule_exports.allocate_lm_result();
            let probs_ptr = wasmModule_exports.get_probs_ptr(lm_result_ptr);
            let cum_ptr = wasmModule_exports.get_cum_ptr(lm_result_ptr);
            wasmModule_exports.set_stop_prob(lm_result_ptr, content.stop_prob);
            let probsWasmMemory = new Float32Array(wasmModule_exports.memory.buffer, probs_ptr, content.probs.length);
            let cumWasmMemory = new Float32Array(wasmModule_exports.memory.buffer, cum_ptr, content.cum.length);
            probsWasmMemory.set(new Float32Array(content.probs));
            cumWasmMemory.set(new Float32Array(content.cum));
            console.log("probsWasmMemory:", [...probsWasmMemory]);
            console.log("cumWasmMemory:", [...cumWasmMemory]);
            let token = 8578;
            let prob = wasmModule_exports.token_prob_lookup(lm_result_ptr, token);
            console.log(`prob of token ${token}:`, prob);
            // create a trie
            let trie = wasmModule_exports.setup_trie();
            wasmModule_exports.set_result(trie, lm_result_ptr);
            // get the beginning token trie
            let beginning_token_trie = wasmModule_exports.get_beginning_token_trie();
            console.log("beginning_token_trie:", beginning_token_trie);
            // expand the cm root
            wasmModule_exports.expand_root(trie, beginning_token_trie, 0, -6);
            // log the trie
            let json_string = wasmModule_exports.jsonifyTrie(trie);
            let json_string_buffer = new Uint8Array(wasmModule_exports.memory.buffer, json_string, 10000);
            const decoder = new TextDecoder('utf-8');
            const jsonStr = decoder.decode(json_string_buffer.slice(0, 10000));
            console.log("json_trie after expansion:", jsonStr);
        }
    }
}

initWasm();

