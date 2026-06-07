# HVAT

Hyperspectral Visualization and Annotation Tool.

HVAT is a GPU-accelerated system for viewing and annotating hyperspectral and RGB images, with a Leptos/WebGPU frontend and an Axum backend.

## Demo

[Live demo](https://fjodborg.github.io/hvat/)

## Workspace

- `hvat_leptos`: frontend (Leptos CSR + Thaw + WebGPU)
- `hvat_backend`: backend (Axum + `/api/ws` multiplexed protocol)
- `hvat_common`: shared protocol/error/data types
- `hvat_gpu`: rendering layer shared by frontend/tests
- `hvat_visual_tests`: visual and E2E tests

## Quick Start

```bash
# Format and build workspace
cargo fmt --all
cargo build
```

### Frontend (WASM)

```bash
cd hvat_leptos && trunk build                              # debug (54 MB, for dev)
cd hvat_leptos && trunk build --cargo-profile release-dev # optimised (8.8 MB, fast compile)
cd hvat_leptos && trunk build --release                   # release (7 MB)
cd hvat_leptos && trunk serve
```

#### Optional: wasm-opt

Installing [wasm-opt](https://github.com/WebAssembly/binaryen) reduces the release WASM by ~430 KB (5.9%). Trunk picks it up automatically when it is on `PATH`.

```bash
# openSUSE / SUSE
sudo zypper install binaryen

# Ubuntu / Debian
sudo apt install binaryen

# macOS
brew install binaryen

# Cargo (cross-platform, slower)
cargo install wasm-opt
```

**Measured impact** (release build, before vs. after wasm-opt):

| File | Before | After | Saved |
|---|---|---|---|
| `hvat_leptos_bg.wasm` | 7.3 MB | 6.9 MB | 430 KB (5.9%) |
| `image-decoder-worker_bg.wasm` | 1.0 MB | 961 KB | 49 KB (4.8%) |
| Gzipped (main) | 2.09 MB | 2.08 MB | 14 KB (0.7%) |

The gain is modest because `opt-level = 3` + `lto = "thin"` in the release profile already does most of the work. The biggest size win is using `--release` instead of the debug default (54 MB → 7 MB).

### Backend (from repo root)

Cargo aliases are defined in `.cargo/config.toml`.

```bash
cargo serve
cargo serve-sam
cargo serve-sam3
cargo serve-debug
```

### Visual Tests

```bash
cargo test -p hvat_visual_tests
HEADLESS=false cargo test -p hvat_visual_tests -- --nocapture
UPDATE_BASELINES=true cargo test -p hvat_visual_tests
```

### Test Event Build (frontend)

```bash
cd hvat_leptos && TRUNK_BUILD_FEATURES="test-events" trunk build
```

## Building a Custom Backend

**`openapi.yaml`** is the single reference for implementing a compatible backend in another language. It covers all endpoints, request/response schemas, and the CORS requirement for the binary band payload.

Paste it into [editor.swagger.io](https://editor.swagger.io) or any OpenAPI viewer to browse it interactively.

The minimum viable backend is four endpoints: `GET /api/info`, `GET /api/images`, `GET /api/images/{id}/bands`, and `GET /api/capabilities`. WebSocket is a legacy compatibility path and not required.

`PROTOCOL.md` is the normative internal contract (kept in sync with the codebase, referenced by `AGENTS.md`).

## Capability-First Backend Contract

- Frontend/backend interoperability is capability-driven, not backend-name driven.
- Backends are replaceable as long as they advertise compatible capability schemas.
- Frontend must use capability-declared input/output field names and types, never hardcoded model IDs.
- Universal behavior agreement is expressed through:
  - server-level feature flags (for example, streaming, project state, downloads),
  - model input/output type combinations (for example, point-to-mask, bbox-to-mask, point+bbox-to-mask).
- Protocol/docs are a development guideline during implementation and should be aligned before release candidates.

## Active Planning Docs

- `SAM3_BACKEND_PLAN.md`: active SAM3 backend roadmap and execution plan
- `docs/sam3_code_change_map.md`: file-level SAM3 implementation checklist
- `streaming_plan.md`: current streaming/navigation orchestration plan

## Notes

- WebSocket communication is multiplexed on `/api/ws`.
- `hvat_leptos` stays on Rust 2021 (wasm-bindgen constraint); other crates use Rust 2024.
