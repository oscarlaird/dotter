import { readFile } from "node:fs/promises";

type WasmExports = {
  add_u32_wasm(a: number, b: number): number;
};

let wasmExportsPromise: Promise<WasmExports> | undefined;

async function loadExports(): Promise<WasmExports> {
  if (!wasmExportsPromise) {
    wasmExportsPromise = (async () => {
      const wasmUrl = new URL("./adder_wasm.wasm", import.meta.url);
      const fileBytes = await readFile(wasmUrl);
      const wasmBytes = new Uint8Array(fileBytes.byteLength);
      wasmBytes.set(fileBytes);
      const wasmModule = await WebAssembly.compile(wasmBytes);
      const instance = await WebAssembly.instantiate(wasmModule, {});
      return instance.exports as unknown as WasmExports;
    })();
  }

  return wasmExportsPromise;
}

export async function add(a: number, b: number): Promise<number> {
  if (!Number.isInteger(a) || !Number.isInteger(b)) {
    throw new RangeError("add() expects integer inputs.");
  }

  const wasm = await loadExports();
  return wasm.add_u32_wasm(a, b) >>> 0;
}
