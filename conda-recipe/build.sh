#!/usr/bin/env bash
set -euo pipefail

# Expose the conda build environment's include directory so nvcc can find
# the cuFFTDx headers installed as a host dependency.
export EXTRA_INCLUDE_DIRS="${PREFIX}/include${EXTRA_INCLUDE_DIRS:+:${EXTRA_INCLUDE_DIRS}}"

# Auto-detect CUDA_HOME if not set
if [ -z "${CUDA_HOME:-}" ]; then
    NVCC_PATH=$(which nvcc)
    export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
    echo "Auto-detected CUDA_HOME: $CUDA_HOME"
fi

# If CUDA_ARCHITECTURES is not set, set it to a default
if [ -z "${CUDA_ARCHITECTURES:-}" ]; then
    echo "CUDA_ARCHITECTURES not set. Defaulting to a broad multi-arch fat binary."
    export CUDA_ARCHITECTURES="8.0,8.6,8.9,9.0,12.0"
else
    echo "Using specified CUDA_ARCHITECTURES: $CUDA_ARCHITECTURES"
fi

# If ENABLED_EXTENSIONS is not set, set it to a default
if [ -z "${ENABLED_EXTENSIONS:-}" ]; then
    echo "ENABLED_EXTENSIONS not set. Defaulting to 'padded_rconv2d'."
    export ENABLED_EXTENSIONS="padded_rconv2d"
else
    echo "Using specified ENABLED_EXTENSIONS: $ENABLED_EXTENSIONS"
fi

pip install --no-deps --no-build-isolation -vv .
