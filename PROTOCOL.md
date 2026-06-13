# HVAT Backend Interop Contract

**Version:** 4  
**Status:** Active (REST-first)  
**Last Updated:** 2026-06-01

This document is the backend-facing contract for integrating with HVAT clients without reading frontend source code.

## 1. Scope

Product runtime is REST-first.

- Image listing/loading: REST
- SAM inference: REST
- Capability discovery: REST
- Project-state persistence: REST
- Download flow: REST

WebSocket transport remains a compatibility path only (see section 9).

## 2. Conformance Language

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## 3. Base Rules

- Base path: `/api`
- JSON endpoints: `Content-Type: application/json`
- Binary image payload endpoint: `Content-Type: application/octet-stream`
- Image IDs are opaque strings to clients.
- Clients URL-encode `{id}` when calling path endpoints.
- Backend MUST treat unknown query parameters as non-fatal unless explicitly defined.

## 4. Compliance Profiles

### 4.1 Core Profile (required)

A backend claiming HVAT REST compatibility MUST implement:

- `GET /api/info`
- `GET /api/images`
- `GET /api/images/{id}/bands`
- `GET /api/capabilities`

### 4.2 Optional Profiles

Implement each profile only if you advertise it in `/api/capabilities`:

- Project state: `GET/PUT /api/project-state` (`features.project_state=true`)
- Thumbnails: `GET /api/images/{id}/thumbnail` (`features.thumbnails=true`)
- Downloads:
  - `download_mode=single_zip`: `GET /api/images/download`
  - `download_mode=chunked`: `GET /api/images/download/plan`, `GET /api/images/download/part/{part_index}`
- Uploads: `POST /api/images/upload`
- SAM: `POST /api/sam/infer` (+ optional `POST /api/sam/warm`)
- Metrics: `GET /api/metrics`

## 5. Core Endpoints

### 5.1 `GET /api/info`

Response:

```json
{
  "name": "project-or-server-name",
  "image_count": 42,
  "version": "0.1.0"
}
```

### 5.2 `GET /api/images`

Response: sorted array of images.

```json
[
  {
    "id": "id:6e65737465642f696d6167652e706e67",
    "name": "image.png",
    "path": "nested/image.png",
    "format": "PNG"
  }
]
```

Rules:

- `id` MUST be stable for the same image path.
- `path` MUST be project-relative (no absolute paths).
- `name` is display filename.
- `format` is informational.

### 5.3 `GET /api/images/{id}/bands`

Returns one binary payload containing **all source bands** packed into RGBA layers.

This endpoint is intentionally full-band. Do not restrict to 3 channels or request-time RGB selection.

Required headers:

- `X-Hvat-Width: <u32>`
- `X-Hvat-Height: <u32>`
- `X-Hvat-Num-Bands: <u32>`
- `X-Hvat-Num-Layers: <u32>`
- `X-Hvat-Payload: rgba_layers_u8`
- `X-Hvat-Layout: layer-major`
- `X-Hvat-Payload-Version: 1`

Body layout:

- Layer size = `width * height * 4` bytes (RGBA u8)
- Layers are concatenated in ascending layer index order
- Total bytes MUST equal `width * height * 4 * num_layers`

Packing rules:

- Bands are packed in groups of 4 per layer:
  - layer `L`, channel `0..3` maps to band index `L*4 + channel`
- `num_layers = max(ceil(num_bands / 4), 2)`
- Values are clamped to `[0.0, 1.0]` then scaled to `[0, 255]`
- If a layer has no alpha band (`band L*4+3` missing), alpha channel MUST be `255`

Example for 7-band image:

- `X-Hvat-Num-Bands: 7`
- `X-Hvat-Num-Layers: 2`

### 5.4 `GET /api/capabilities`

Returns `ServerCapabilities`:

```json
{
  "protocol_version": 2,
  "server": { "name": "hvat-axum", "version": "0.1.0" },
  "limits": {
    "max_image_size": 4294967296,
    "max_pyramid_levels": 8,
    "max_concurrent_streams": 4,
    "max_concurrent_inferences": 2
  },
  "features": {
    "streaming": true,
    "project_state": true,
    "downloads": true,
    "thumbnails": true,
    "inference": true,
    "sam": true,
    "progressive_streaming": true
  },
  "download_mode": "chunked",
  "models": []
}
```

Semantics:

- `features.project_state=true` means `/api/project-state` is implemented.
- `features.thumbnails=true` means `/api/images/{id}/thumbnail` is implemented.
- `download_mode` contract:
  - `single_zip` => `/api/images/download`
  - `chunked` => `/api/images/download/plan` and `/api/images/download/part/{part_index}`
  - `unknown` => compatibility probing by clients
- `features.inference` and `features.sam` MUST match model availability in `models`.
- `features.streaming` MAY remain true for compatibility signaling; REST remains product path.
- For SAM UI compatibility, advertise at least one model that:
  - has segmentation semantics (`type=segmentation`)
  - has input type `point_list`
  - has output type `polygon_list`
  - optionally has input type `bbox` (enables point+box prompting)

## 6. SAM Profile

### 6.1 `POST /api/sam/infer`

Request:

```json
{
  "model_id": "sam3",
  "image_id": "id:...",
  "bands": { "r": 0, "g": 1, "b": 2 },
  "inputs": {
    "points": [{ "x": 120.5, "y": 340.0, "label": 1 }],
    "box": [100.0, 100.0, 300.0, 400.0]
  }
}
```

Accepted `inputs.points` shapes:

- `[{"x":...,"y":...,"label":...}, ...]`
- `[[x,y], ...]` with parallel `inputs.labels`

Response:

```json
{
  "masks": [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]],
  "scores": [0.93]
}
```

Rules:

- `masks` MUST be a list of flattened polygon vertex arrays (`[x1,y1,x2,y2,...]`)
- `scores` MAY be omitted

### 6.2 `POST /api/sam/warm`

Request fields: `model_id`, `image_id`, optional `bands`.

Response: `202 Accepted` when warming request is accepted/completed.

### 6.3 SAM Error Envelope

On non-2xx, return:

```json
{
  "error": "human-readable message",
  "code": "snake_case_code"
}
```

Recommended status mapping:

- `400`: invalid request/model inputs
- `404`: image not found
- `503`: model unavailable
- `500`: backend execution failure

## 7. Project/Asset Optional Endpoints

### 7.1 `GET /api/project-state`

- `200 OK` + JSON bytes when state exists
- `404 Not Found` when state does not exist

### 7.2 `PUT /api/project-state`

- Request body: native project bundle JSON
- Current reference limit: 64 MiB max payload
- Response: `204 No Content` on success

### 7.3 `POST /api/images/upload`

Multipart fields:

- `target_root` (optional text, MUST appear before `files` fields)
- `files` (one or more file parts)

Response:

```json
{
  "uploaded": 2,
  "files": ["nested/a.png", "nested/b.npy"]
}
```

### 7.4 `GET /api/images/{id}/thumbnail`

- Response content-type: `image/png`

### 7.5 `GET /api/images/download`

- Response content-type: `application/zip`

### 7.6 `GET /api/images/download/plan?part_size_mb=<u64>`

Response:

```json
{
  "total_files": 12,
  "total_bytes": 123456789,
  "part_size_bytes": 268435456,
  "part_count": 2,
  "parts": [
    {
      "index": 0,
      "filename": "project_images_part001.zip",
      "file_count": 8,
      "total_bytes": 77777777
    }
  ]
}
```

### 7.7 `GET /api/images/download/part/{part_index}?part_size_mb=<u64>`

- Response content-type: `application/zip`
- `part_index` MUST correspond to an index returned by `/download/plan`

### 7.8 `GET /api/metrics`

Response:

```json
{
  "image_bands": { "requests": 0, "errors": 0, "avg_latency_ms": 0, "max_latency_ms": 0 },
  "sam_infer": { "requests": 0, "errors": 0, "avg_latency_ms": 0, "max_latency_ms": 0 },
  "sam_warm": { "requests": 0, "errors": 0, "avg_latency_ms": 0, "max_latency_ms": 0 }
}
```

### 7.9 `GET /api/images/{id}/meta`

Response:

```json
{
  "id": "id:...",
  "name": "hyper_7band.npy",
  "format": "NPY",
  "width": 1024,
  "height": 1024,
  "bands": 7,
  "pyramid": {
    "status": "ready",
    "levels": [
      { "level": 0, "width": 1024, "height": 1024 },
      { "level": 1, "width": 512, "height": 512 }
    ]
  }
}
```

`pyramid.status` values: `ready`, `building`, `pending`, `failed`.

### 7.10 `GET /api/images/{id}/raw`

- Returns original file bytes for the image
- `Content-Type: application/octet-stream`
- SHOULD include `Content-Disposition: attachment; filename=\"...\"`

## 8. Error Behavior (non-SAM endpoints)

For non-SAM endpoints, plain-text error responses are acceptable.
Use consistent HTTP status codes and deterministic messages.

## 9. WebSocket Compatibility Path (not product default)

A compatibility backend may expose multiplexed WebSocket at `/api/ws`.

- This path is for compatibility/harness scenarios.
- REST contract in this document is the product interoperability target.

## 10. Change Policy

Any REST contract change MUST include:

1. `PROTOCOL.md` update in the same change.
2. Route/integration test updates (see `hvat_backend/tests/full_sam_rest_routes.rs`).
3. Capability semantics update if endpoint behavior/availability changed.
