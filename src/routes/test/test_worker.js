// worker.js
import * as wasm from 'http://localhost:5173/trie_worker_rust/pkg/trie_worker_rust.js';

let tokenizer = null;
let sharedBuffer = null;

async function initWasm() {
    await wasm.default(
        "http://localhost:5173/trie_worker_rust/pkg/trie_worker_rust_bg.wasm"
    );
    self.postMessage("I am the worker; I have been initialized!");
    tokenizer = new wasm.WrappedTokenizer();
    console.log("WASM tokenizer initialized");
    let result = await tokenizer.encode("Hello, world!");
    console.log("WASM tokenizer encoded:", result);
}


initWasm();

self.onmessage = async (event) => {
    console.log("I am the worker; I received a message:", event.data);
    // const result = wasm.process_query(event.data); // Assuming you have such a function
    // self.postMessage(result);
    let type = event.data.type;
    if (type === "encode") {
        if (!tokenizer) {
            console.log("the tokenization worker has not been initialized!")
        } else {
            let text = event.data.text;
            let result = await tokenizer.encode(text);
            // increment the first byte of the shared buffer
            let view = new Uint8Array(sharedBuffer);
            view[0] = view[0] + 1;
            console.log("sharedBuffer incremented:", view[0]);
            self.postMessage(result);
        }
    }
    if (type === "sharedBuffer") {
        sharedBuffer = event.data.sharedBuffer;
        console.log("sharedBuffer initialized:", sharedBuffer);
    }
};