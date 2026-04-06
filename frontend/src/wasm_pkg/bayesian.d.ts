/* tslint:disable */
/* eslint-disable */

export class BayesianSession {
    free(): void;
    [Symbol.dispose](): void;
    apply_updates(): void;
    expand_to_threshold(): string;
    lexicographic_tokens_json(): string;
    constructor();
    next_requested_prior(): string;
    receive_likelihood_update(likelihood_json: string): void;
    receive_prior_update(prior_json: string): void;
    reset(): void;
}

/**
 * Dev-only: panics immediately so you can verify `initPanicHook` / `console_error_panic_hook`
 * in the browser console. Frontend calls this when `?wasmPanic=1` (Vite dev only).
 */
export function debugPanicTest(): void;

/**
 * Install [`console_error_panic_hook`] so panics in the wasm32 build print to `console.error`
 * with source location (and a useful traceback when debug symbols are present). Call once
 * after wasm init (see `initPanicHook` in the wasm bindings).
 */
export function initPanicHook(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_bayesiansession_free: (a: number, b: number) => void;
    readonly bayesiansession_apply_updates: (a: number) => void;
    readonly bayesiansession_expand_to_threshold: (a: number) => [number, number];
    readonly bayesiansession_lexicographic_tokens_json: (a: number) => [number, number];
    readonly bayesiansession_new: () => number;
    readonly bayesiansession_next_requested_prior: (a: number) => [number, number];
    readonly bayesiansession_receive_likelihood_update: (a: number, b: number, c: number) => void;
    readonly bayesiansession_receive_prior_update: (a: number, b: number, c: number) => void;
    readonly bayesiansession_reset: (a: number) => void;
    readonly initPanicHook: () => void;
    readonly debugPanicTest: () => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
