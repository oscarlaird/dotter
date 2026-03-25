import { readFile } from "node:fs/promises";

type WasmExports = {
  add_u64(a: bigint, b: bigint): bigint;
};

let wasmExportsPromise: Promise<WasmExports> | undefined;

async function loadExports(): Promise<WasmExports> {
  if (!wasmExportsPromise) {
    wasmExportsPromise = (async () => {
      const wasmUrl = new URL("./add_u64.wasm", import.meta.url);
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

export async function hello(): Promise<string> {
  return "Hello from Lean via WASM!";
}

export async function addBigInt(a: bigint, b: bigint): Promise<bigint> {
  const wasm = await loadExports();
  return wasm.add_u64(a, b);
}

export async function add(a: number, b: number): Promise<number> {
  if (!Number.isSafeInteger(a) || !Number.isSafeInteger(b)) {
    throw new RangeError("add() expects safe integers. Use addBigInt() for larger values.");
  }

  const result = await addBigInt(BigInt(a), BigInt(b));
  return Number(result);
}
