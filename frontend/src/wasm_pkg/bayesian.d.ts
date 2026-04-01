/* tslint:disable */
/* eslint-disable */

export class BayesianSession {
    free(): void;
    [Symbol.dispose](): void;
    apply_likelihood_update(snapshot_json: string): void;
    apply_prior_update(prior_json: string): void;
    lexicographic_tokens_json(): string;
    constructor();
    reset(): void;
    snapshot_json(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly main: (a: number, b: number) => number;
    readonly __wbg_bayesiansession_free: (a: number, b: number) => void;
    readonly bayesiansession_apply_likelihood_update: (a: number, b: number, c: number) => void;
    readonly bayesiansession_apply_prior_update: (a: number, b: number, c: number) => void;
    readonly bayesiansession_lexicographic_tokens_json: (a: number) => [number, number];
    readonly bayesiansession_new: () => number;
    readonly bayesiansession_reset: (a: number) => void;
    readonly bayesiansession_snapshot_json: (a: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
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
