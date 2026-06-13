# hvat_common Components

## What This Crate Does

`hvat_common` is the shared contract and utilities crate for the server and
compatible clients.

It provides:

- protocol schema and shared wire-level types
- binary parsing/encoding helpers
- structured protocol errors
- hyperspectral band packing helpers
- interpolation and pyramid-level safety types

## Module Map

| Module | What It Does | Key Types / APIs | Requires | Used By |
|---|---|---|---|---|
| `protocol.rs` | Versioned protocol types and model capability schema | `ClientMessage`, `ServerMessageType`, `ServerCapabilities`, `ServerFeatures`, `ErrorCode` | Protocol v2 compatibility | Backend and compatible clients |
| `error.rs` | Structured protocol error payloads + context | `ProtocolError`, `ErrorContext` | Error codes + severity | Backend and compatible clients |
| `binary.rs` | Safe cursor-based parsing | `BinaryReader` | Correct buffer layout | Protocol decoders and tests |
| `packing.rs` | Band packing into RGBA texture-array layers | `pack_bands_to_rgba_layers`, `MIN_TEXTURE_LAYERS` | Bands normalized to `0.0..1.0` | Backend and rendering clients |
| `interpolation.rs` | Bilinear sampling | `bilinear_sample` | Flat row-major input buffers | Backend downsampling/SAM post-processing |
| `pyramid.rs` | Pyramid level validation and helpers | `PyramidLevel`, `MAX_PYRAMID_LEVEL` | Valid level bounds | Backend streaming and clients |
| `lib.rs` helpers | Overflow-safe pixel count helpers and re-exports | `pixel_count`, `pixel_count_u32` | dimensions fit checked multiply | all crates |

## What It Can Do

- Keep client/server message contracts in one place.
- Encode/decode robust protocol errors with retry semantics and context payloads.
- Provide shared packing behavior so server and client layer counts stay consistent.
- Prevent common parsing bugs with bounds-checked binary reads.
- Prevent invalid pyramid-level values at compile/runtime boundaries.

## Requirements and Assumptions

- `PROTOCOL_VERSION` must match across client and server implementations.
- `ServerCapabilities.features` must stay backward-compatible; omitted fields should deserialize to defaults.
- Any code writing binary frames must keep byte layout aligned with `BinaryReader` usage.
- Packing assumes per-band buffers represent exactly `width * height` samples.
- Packed channel values are clamped to `0.0..1.0` before conversion to `u8`.

## How To Interact With It

- If adding a protocol message/field:
  1. Update `protocol.rs` types.
  2. Update server encoders/decoders and compatible client parsers.
  3. Keep `PROTOCOL.md` in sync.
- If adding a new error:
  1. Add code to `ErrorCode`.
  2. Ensure category/retry behavior is correct.
  3. Use `ProtocolError::{error,retryable,fatal}` for emission.
- If adding a new packing/downsampling path:
  1. Reuse `pack_bands_to_rgba_layers` and `bilinear_sample`.
  2. Avoid local re-implementations to keep behavior identical.
