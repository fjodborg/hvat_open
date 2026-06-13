# hvat_backend Components

## What This Crate Does

`hvat_backend` is the backend API server for HVAT. It serves image/project metadata and image-band payloads over HTTP, and runs model inference (including SAM) through REST contracts.

## Runtime Flow

1. `main.rs` builds `ServerConfig` from CLI/env and creates `AppState`.
2. `startup::pregenerate_pyramids()` eagerly builds cached pyramid levels where missing.
3. `routes::api_router()` mounts:
   - `GET /api/info`
   - `GET /api/images`
   - `GET /api/images/:id/meta`
   - `GET /api/images/:id/bands`
   - `GET /api/images/:id/thumbnail`
   - `GET /api/capabilities`
   - `GET /api/metrics`
   - `POST /api/sam/infer`
   - `POST /api/sam/warm`
4. `routes/sam.rs` handles model resolve, embedding prepare/cache, and infer response mapping.

## Module Map

| Module | What It Does | Key Entry Points | Requires | Used By |
|---|---|---|---|---|
| `config.rs` | CLI/env config parsing and validation | `ServerConfig::from_cli`, `CliArgs` | Valid paths and numeric limits | `main.rs`, `state.rs` |
| `state.rs` | Shared app wiring (loaders, pyramid storage, SAM, model registry, connection limits) | `AppState::new`, `try_acquire_connection` | Valid `ServerConfig`, optional SAM models | All routes and startup |
| `loaders/` | Image format abstraction (`PNG/JPEG/...`, `NPY`) | `ImageLoader` trait, `ImageLoaderRegistry` | Supported file extensions and readable files | `routes/images.rs`, `startup.rs` |
| `pyramid/` | Pyramid generation and cache IO | `PyramidBuilder`, `PyramidStorage`, `FilesystemStorage` | Writable cache dir | `startup.rs`, `routes/images.rs` |
| `packer.rs` | RGBA layer packing + downsampling | `pack_bands_to_rgba_layers`, `downsample_bands` | Normalized `BandData` | `routes/images.rs`, `pyramid/builder.rs` |
| `routes/projects.rs` | Project/image listing endpoints | `router()` | `data_dir` walk + loaders | Client image tree bootstrap |
| `routes/images.rs` | Metadata and thumbnail endpoints | `router()` | Loader metadata + pyramid metadata/level reads | Client metadata + thumbnail fetch |
| `routes/sam.rs` | REST SAM infer/warm orchestration | `router()`, `post_infer()`, `post_warm()` | Model registry + image resolve + band mapping | `routes/mod.rs` |
| `routes/sam_service.rs` | Shared SAM image/band helper functions | `resolve_sam_bands()`, `load_image_rgb_with_bands()` | Valid image ID + band inputs | `routes/sam.rs` |
| `routes/metrics.rs` | REST telemetry snapshot endpoint | `get_metrics()` | Runtime request counters | `routes/mod.rs` |
| `inference/` | Model-agnostic backend trait + registry + SAM adapter | `InferenceBackend`, `ModelRegistry`, `SamInferenceAdapter` | Registered backend(s) | `state.rs`, `routes/sam.rs` |
| `sam/` | SAM backend contract, ONNX engine, model download/checking, embedding cache | `SamBackend`, `OnnxSamEngine`, `EmbeddingCache`, `SamVariant` | ONNX model files + runtime support | `state.rs`, inference adapter |

## How To Interact With It

### HTTP

- `GET /api/info`: server/project name + image count.
- `GET /api/images`: recursive image list (URL-safe IDs + paths).
- `GET /api/images/:id/meta`: dimensions, band count, pyramid status/levels.
- `GET /api/images/:id/bands`: full-band payload packed as `rgba_layers_u8` layers.
- `GET /api/images/:id/thumbnail`: smallest cached pyramid level as PNG.
- `GET /api/capabilities`: server model/features contract.
- `GET /api/metrics`: REST route telemetry snapshot.
- `POST /api/sam/infer`: one-shot SAM infer request.
- `POST /api/sam/warm`: embedding warm request.

## What It Can Do

- Serve image band payloads for direct client rendering.
- Build missing pyramids asynchronously for metadata/thumbnail workflows.
- Run model-agnostic inference through `ModelRegistry`.
- Run SAM infer/warm flows when SAM is enabled.
- Expose capability and telemetry contracts for client gating and dashboards.

## Requirements and Assumptions

- `data_dir` must be readable and contain files supported by registered loaders.
- `cache_dir` must be writable for pyramid metadata/levels/status files.
- For SAM:
  - `--sam-enabled` must be set.
  - Missing SAM2/compatibility ONNX artifacts are downloaded into
    `sam_model_dir` before runtime initialization.
  - Native SAM3 checkpoint and config artifacts must be supplied explicitly.
- If SAM is not enabled:
  - Server still supports image/project endpoints.
  - `models` is empty and `features.sam=false`.

## Non-Simple Areas To Read First

- `hvat_backend/src/full_sam/routes/sam.rs`: REST infer/warm orchestration + error mapping.
- `hvat_backend/src/full_sam/sam/engine.rs`: ONNX encode/decode flow and mask post-processing.
- `hvat_backend/src/full_sam/pyramid/storage.rs`: on-disk cache format and status/progress tracking.
