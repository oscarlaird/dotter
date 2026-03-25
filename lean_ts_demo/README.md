# lean-hello-ts

This is a tiny end-to-end demo showing how to call Lean from TypeScript through
WebAssembly.

## How it works

1. `Add.lean` defines a tiny exported Lean function:
   `@[export add_u64] def addU64 (x y : UInt64) : UInt64 := x + y`
2. The build script runs `lean -c` to generate C from that Lean file.
3. `clang` compiles the generated C into a tiny `wasm32-wasi` module.
4. `src/index.ts` loads the `.wasm` file with `WebAssembly.instantiate`.
5. The TypeScript wrapper exposes:
   - `hello(): Promise<string>`
   - `add(a: number, b: number): Promise<number>`
   - `addBigInt(a: bigint, b: bigint): Promise<bigint>`

## Why this version works

For this tiny example, the Lean export uses an unboxed `UInt64` ABI, so the
generated C contains a plain exported function:

`uint64_t add_u64(uint64_t, uint64_t);`

That makes it possible to compile a very small standalone WASM module without
packaging the full Lean runtime.

## Commands

```bash
cd lean_ts_demo
npm install
npm run build
npm run test:run
```

## Practical recommendation

If the long-term goal is a trustworthy production artifact, a good workflow is:

1. Write the mathematical spec in Lean.
2. Prove the important properties in Lean.
3. Write the executable kernel in Rust in a verification-friendly subset.
4. Use Lean-based post-hoc verification where possible, or at least keep the
   Rust code very close to the Lean spec.
5. Export Rust to wasm for the TypeScript boundary.

This keeps the final shipped artifact simple, avoids depending on the Lean
runtime in production, and preserves a path toward proving properties of code
that more closely matches what actually runs.
