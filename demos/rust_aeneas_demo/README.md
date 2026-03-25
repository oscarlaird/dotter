# rust_aeneas_demo

This demo uses:

- `adder_core/` for the Rust function verified through Aeneas
- `adder_wasm/` for the tiny exported wasm wrapper
- `lean_verify/` for the Aeneas-generated Lean definitions and proof
- `ts_demo/` for the TypeScript wrapper that calls the wasm artifact

## Rust function

`adder_core/src/lib.rs` defines:

- `add_u32(x, y) = x.wrapping_add(y)`
- `is_even(x) = x % 2 == 0`

## Correctness goal

The Lean proof in `lean_verify/RustAdderProof.lean` proves that if two `U32`
inputs are even, then the result of the extracted Rust adder is also even.

In particular:

```lean
theorem add_u32_preserves_evenness (x y : Std.U32)
    (hx : x.val % 2 = 0)
    (hy : y.val % 2 = 0) :
    ∃ z, RustAdder.add_u32 x y = ok z ∧ z.val % 2 = 0
```

## Toolchain used

- Rust via `rustup`
- Charon and Aeneas built from source
- Lean package depending on the local Aeneas Lean backend

## Useful commands

```bash
cd demos/rust_aeneas_demo

# Rust tests
cargo test -p adder_core

# Build the tiny wasm artifact
cargo build --release --target wasm32-unknown-unknown -p adder_wasm

# Extract Rust to LLBC (adjust CHARON/AENEAS paths to your install)
cd adder_core
charon cargo --preset=aeneas --dest-file "$PWD/../charon_out/adder_core.llbc"

# Translate LLBC to Lean
aeneas -backend lean -split-files -gen-lib-entry -namespace RustAdder -dest "$PWD/../lean_verify/generated" "$PWD/../charon_out/adder_core.llbc"

# Check the Lean proof
cd ../lean_verify
lake -KmaxJobs=1 build

# Build and run the TS wrapper
cd ../ts_demo
npm install
npm run build
npm run test:run
```

## Result

The final wasm artifact at
`target/wasm32-unknown-unknown/release/adder_wasm.wasm` is only `118` bytes,
and the TypeScript wrapper successfully calls it.
