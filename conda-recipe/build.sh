#!/usr/bin/env bash
set -euo pipefail

# Expose the conda build environment's include directory so nvcc can find
# the cuFFTDx headers installed as a host dependency.
export EXTRA_INCLUDE_DIRS="${PREFIX}/include${EXTRA_INCLUDE_DIRS:+:${EXTRA_INCLUDE_DIRS}}"

# Default to a broad multi-arch fat binary when the caller hasn't specified a
# target. Users with a known GPU should set CUDA_ARCHITECTURES before running
# conda build to reduce compile time, e.g.:
#   CUDA_ARCHITECTURES=8.9 conda build conda-recipe/ -c nvidia -c pytorch
: "${CUDA_ARCHITECTURES:=8.0,8.6,8.9,9.0,12.0}"
export CUDA_ARCHITECTURES

: "${ENABLED_EXTENSIONS:=padded_rconv2d}"
export ENABLED_EXTENSIONS

pip install --no-deps --no-build-isolation -vv .
