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
cd hvat_leptos && trunk build
cd hvat_leptos && trunk serve
```

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

## Protocol and Integration Docs

- `PROTOCOL.md`: canonical WebSocket protocol spec
- `CUSTOM_BACKEND_BOUNDARY.md`: minimum compatibility contract for custom backends
- `CUSTOM_BACKEND_API_REFERENCE.md`: HTTP + WebSocket API reference

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
