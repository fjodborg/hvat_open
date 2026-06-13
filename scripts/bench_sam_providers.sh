#!/usr/bin/env bash
#
# Benchmark SAM2 inference across CPU, WebGPU, and CUDA on the real models.
#
# WebGPU and CUDA ship as mutually-exclusive prebuilt ONNX Runtime binaries, so a
# single build can measure CPU plus at most one GPU provider. This script runs the
# benchmark twice — once with `sam-webgpu` and once with `sam-cuda` — so the
# combined output covers all three providers.
#
# CUDA is optional: ORT's CUDA EP hard-requires libcudnn.so.9. If a copy is found
# anywhere on the system it is added to LD_LIBRARY_PATH for the CUDA run; otherwise
# the CUDA row is reported as UNAVAILABLE (not a failure).
#
# Requires the SAM Tiny ONNX models in .cache/models (see the model download step).
#
# Usage:
#   scripts/bench_sam_providers.sh
set -uo pipefail

cd "$(dirname "$0")/.."

PROFILE="release-dev"
# Substring filter that matches both the SAM2 (`benchmark_sam_providers`) and
# SAM3 (`benchmark_sam3_providers`) benchmark tests.
TEST_FILTER="benchmark_sam"

run_bench() {
  # $1 = cargo feature
  cargo test -p hvat_backend --profile "$PROFILE" --features "$1" --lib \
    "$TEST_FILTER" -- --ignored --nocapture
}

echo "############################################################"
echo "# SAM provider benchmark"
echo "############################################################"

# --- Run 1: WebGPU build (CPU + WebGPU; CUDA reported UNAVAILABLE) ------------
echo
echo ">>> Build: --features sam-webgpu (CPU + WebGPU)"
run_bench sam-webgpu

# --- Run 2: CUDA build (CPU + CUDA; WebGPU reported UNAVAILABLE) --------------
echo
echo ">>> Build: --features sam-cuda (CPU + CUDA)"

# ORT's CUDA EP needs libcudnn.so.9. Locate one if it isn't already on the path.
CUDNN_LIB="$(find /usr /home -name 'libcudnn.so.9' 2>/dev/null | head -1)"
if [ -n "${CUDNN_LIB}" ]; then
  CUDNN_DIR="$(dirname "${CUDNN_LIB}")"
  echo "    using cuDNN: ${CUDNN_LIB}"
  export LD_LIBRARY_PATH="${CUDNN_DIR}:/usr/local/cuda-12/lib64:${LD_LIBRARY_PATH:-}"
else
  echo "    no libcudnn.so.9 found — CUDA will report UNAVAILABLE"
fi

run_bench sam-cuda

echo
echo "Done."
