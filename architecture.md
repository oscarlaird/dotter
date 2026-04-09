# Architecture

For a more formal, mathematics-oriented description of the system, see
`math/tex/chapters/architecture.tex`. This document is the code-oriented
architecture reference for the intended target system.

## Where to work (active code vs archive)

**Server.** Ongoing backend work belongs under **`backend/`**. The canonical
runtime entrypoint is **`backend/lm.py`** (FastAPI + WebSocket, Rust
`bayesian` session, prior model, `prior_update` / `likelihood_update` protocol).
The large historical server at **`ARCHIVE_language_server/lm.py`** is kept in
the tree as an **archive** only; do not extend it for new features and do not
treat it as the process to run for current development.

**Client.** Ongoing UI work targets **`frontend/` route `/v3`** (`V3Page` and
related components). **`frontend/` route `/v2`** (`V2Page` and the legacy trie
visualizer stack) is an **archive** for reference; new product behavior should
land in v3.

## Overview

Dotter runs in two execution environments:

- a client-side TypeScript runtime
- a server-side Python runtime

Both sides embed the Rust `bayesian` library. The Rust library provides the
core Bayesian data structures and update machinery, most importantly the
ability to create and operate on a **Bayesian session**. A Bayesian session
contains a persistent Bayesian trie together with the additional session state
required to apply updates efficiently and incrementally.

The client and server each maintain their own local Bayesian session. These
sessions are intended to converge by exchanging update events over a websocket.

The cross-language API for `bayesian` is intentionally minimal: the frontend
and backend call Rust methods whose inputs and outputs are JSON encoded as
plain strings. This is a good FFI boundary here because the websocket protocol
can forward those same strings unchanged, so the exact payload applied locally
is also the payload applied remotely.

## Runtime Responsibilities

### Client Side

The client-side code is written in TypeScript. Its primary responsibility is to
run the likelihood cycle by eliciting gestures from the user and converting
those gestures into likelihood updates.

For each likelihood update produced by the client:

1. the client applies the update to its own local Bayesian session
2. the client sends the update as an event over the websocket to the server

When the client receives an update event over the websocket, it applies that
update to its local Bayesian session.

The client embeds the `bayesian` crate as WebAssembly. After `initBayesianWasm()`
the app calls `initPanicHook()` so Rust panics are forwarded to `console.error`
via `console_error_panic_hook` (message and location; best results with a debug
wasm build). From `frontend/`, rebuild the wasm package with `npm run
build:wasm:dev` (debug symbols, unoptimized) or `npm run build:wasm` (optimized
release). In Chrome DevTools, use **Sources** and pause on uncaught exceptions
to inspect the **WASM** stack when a panic surfaces as a JS exception.

### Server Side

The server-side code is written in Python. Its primary responsibility is to run
the prior cycle by querying a language model for the highest-priority token
sequences and converting those results into prior updates.

The server must also maintain a key-value cache trie so that repeated language
model queries can be served efficiently. This cache trie is a core server-side
performance structure and is necessary for the prior cycle to scale.

For each prior update produced by the server:

1. the server applies the update to its own local Bayesian session
2. the server sends the update as an event over the websocket to the client

When the server receives an update event over the websocket, it applies that
update to its local Bayesian session.

## Shared Bayesian Core

The Rust `bayesian` library is embedded in both runtimes because both runtimes
need the same trie semantics, update logic, and persistence behavior. The Rust
implementation is therefore the shared source of truth for Bayesian session
behavior.

Architecturally, this means:

- the TypeScript client does not implement an independent trie algorithm
- the Python server does not implement an independent trie algorithm
- both sides rely on the same Rust data model and update semantics

This arrangement reduces divergence between client and server behavior and
ensures that likelihood updates and prior updates are interpreted identically on
both sides.

## Update Propagation Model

The websocket connection carries update events in both directions.

There are two update classes:

- **likelihood updates**, produced on the client from user gestures
- **prior updates**, produced on the server from language-model queries

The propagation rule is symmetric:

1. produce an update locally
2. apply it locally to the local Bayesian session
3. transmit it over the websocket
4. when a remote update is received, apply it to the local Bayesian session

The API boundary between client and server is intentionally thin: websocket
payloads are expected to carry the same JSON strings that are passed directly
into the Rust `bayesian` session methods. In practice, compatibility between
environments depends on those Rust update contracts and semantics rather than
independently designed frontend/backend DTOs.

Concretely, the flow is:

1. produce an update locally as a JSON string
2. pass that string into the local Rust `BayesianSession`
3. forward that same string over the websocket
4. when the remote side receives it, pass that same string into its own local
   Rust `BayesianSession`

This is a good design choice because remote updates are applied in exactly the
same way as local updates. The frontend and backend are not translating between
separate transport objects and FFI objects; they are forwarding the same
serialized update payloads.

This design ensures that each side can continue making progress independently
while still converging toward the same Bayesian state.

## Architectural Invariants

The target architecture depends on the following invariants:

- both client and server always maintain a local Bayesian session
- every locally produced update is applied locally before or as it is emitted
- every emitted update is serialized as a websocket event carrying the same
  JSON string used at the local Rust FFI boundary
- every received websocket update is applied to the local Bayesian session
- the result of applying updates must be order-independent with respect to
  client/server delivery order

The last invariant is particularly important. Because websocket delivery and
local computation are concurrent, the client and server must be able to receive
the same set of updates in different orders without diverging. The update
algorithms provided by the Rust Bayesian library are therefore required to be
correct under reordering of independently generated prior and likelihood
updates.
