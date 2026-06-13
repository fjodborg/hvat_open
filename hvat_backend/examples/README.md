# hvat_backend Examples

> Note: These are protocol/examples for reusable backend helpers. The active `full_sam` product runtime is REST-first and does not rely on `/api/ws` for image/SAM paths.

These examples keep all backend-related code inside `hvat_backend` and reuse the
internal `helper` module for common behavior.

## `minimal_streaming_backend`

A small streaming-only backend:

- HTTP: `/api/info`, `/api/images`
- WebSocket: `/api/ws`
- Features: streaming only (`project_state=false`, `downloads=false`, `thumbnails=false`, `inference=false`, `sam=false`)

Run:

```bash
cargo run -p hvat_backend --example minimal_streaming_backend -- ./data
```

## `custom_protocol_actions`

Shows how to keep standard HVAT protocol support while adding custom backend actions.

- Standard HVAT WebSocket: `/api/ws`
- Custom WebSocket endpoint: `/api/custom-ws`
- Custom actions: `echo`, `sum`, `capabilities`
- Features: streaming only (`project_state=false`, `downloads=false`, `thumbnails=false`, `inference=false`, `sam=false`)

Run:

```bash
cargo run -p hvat_backend --example custom_protocol_actions -- ./data
```

Note: SAM inference remains in the existing full backend implementation and is
not implemented in these lightweight examples.

## `streaming_state_downloads_backend`

Shows the profile with streaming + project-state + downloads, without inference:

- HTTP: `/api/info`, `/api/images`, `/api/project-state`, `/api/images/download`
- WebSocket: `/api/ws`
- Features: `streaming=true`, `project_state=true`, `downloads=true`, `thumbnails=true`, `inference=false`, `sam=false`

Run:

```bash
cargo run -p hvat_backend --example streaming_state_downloads_backend -- ./data
```
