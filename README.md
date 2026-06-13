# HVAT Open Backend

Public backend services and wire contracts for HVAT, the Hyperspectral
Visualization and Annotation Tool.

The browser frontend and GPU renderer are maintained separately and are not. But the binary will be deployed/released as part of this repository.

## Work in progress
This is in the process of being migrated from a monolithic binary to a more modular approach. I expect to finish the migration during the next few days and then i'll continue on adding features. 

Before using SAM3 backend please read the license: [sam3 license](https://github.com/facebookresearch/sam3?tab=License-1-ov-file#readme).

License is not fully decided yet, but for i just put AGPL on it, i might make it MIT later.

## Repository Contents

- `hvat_backend`: Axum REST server, image loaders, project persistence,
  pyramid generation, SAM inference, and executable entry points.
- `hvat_common`: Protocol types, structured errors, annotation exchange
  formats, binary helpers, and shared image data utilities.
- `openapi.yaml`: Machine-readable REST API specification.
- `PROTOCOL.md`: Normative protocol behavior and compatibility requirements.

## Binaries

- `hvat_backend`: Primary REST backend with optional SAM2 inference.
- `hvat_backend_sam3`: SAM3 backend entry point. Its default compatibility
  runtime uses the SAM2 ONNX artifacts.
- `simple`: Compatibility backend enabled with the `legacy-ws` feature.
- `hvat_backend_mock_sam`: Deterministic development and integration-test
  backend. It is not included in public release archives.

## Build

```bash
cargo build --release -p hvat_backend --bin hvat_backend
cargo build --release -p hvat_backend --bin hvat_backend_sam3
```

The workspace uses the Rust toolchain declared in `rust-toolchain.toml`.

## Run

Serve images without SAM:

```bash
cargo serve -- --data-dir /path/to/images
```

Enable SAM2:

```bash
cargo serve -- \
  --data-dir /path/to/images \
  --sam-enabled \
  --sam-variant tiny
```

When SAM is enabled, missing ONNX artifacts are downloaded from the model
source declared in `hvat_backend/src/full_sam/sam/models.rs` and cached in
`.cache/models`. Override that location with `--sam-model-dir` or
`HVAT_SAM_MODEL_DIR`.

Model files are not stored in this repository and are not included in release
archives. Review the upstream model license before downloading or
redistributing model artifacts.

Native SAM3 mode requires explicit `--sam3-checkpoint` and `--sam3-config`
paths. Those artifacts are not downloaded automatically.

## API Compatibility

REST implementations and clients should use `openapi.yaml` as the
machine-readable contract and `PROTOCOL.md` for normative behavior that is not
fully expressed by OpenAPI.

Any REST route or schema change must update both documents in the same change.

## Releases

Version tags build Linux x86-64 archives for `hvat_backend` and
`hvat_backend_sam3`. Release archives contain executables, documentation, and
checksums only. They never contain model artifacts.

## License

`hvat_backend` and `hvat_common` are licensed under the GNU Affero General
Public License v3.0. See `LICENSE`.
