#!/usr/bin/env bash
# Source after activating a compatible SGLang/CUDA environment.
DSPARK_BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DSPARK_SOURCE="$(cd -- "$DSPARK_BENCH_DIR/../.." && pwd)"
export DSPARK_PY="${DSPARK_PY:-$(command -v python)}"
: "${DSPARK_MODEL_PATH:?Set DSPARK_MODEL_PATH to the downloaded model snapshot directory}"
export DSPARK_MODEL_PATH
export DSPARK_MANIFEST="$DSPARK_BENCH_DIR/manifest-12.json"
export PYTHONPATH="$DSPARK_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export DSPARK_EXPERIMENT_LOADER_WORKERS=1 SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_APPEND_FLAGS="-I$CUDA_HOME/include/cccl"
