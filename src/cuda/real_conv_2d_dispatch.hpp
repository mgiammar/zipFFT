// Declaration of the per-config dispatch entry point for the padded real (R2C/C2R) 2D
// convolution/cross-correlation path.
//
// This header intentionally does NOT include real_conv_2d.cuh (and therefore never pulls in
// cuFFTDx). real_conv_2d_binding.cu includes only this declaration: it needs the function's
// *type* to build its compile-time dispatch table (taking the address of each
// dispatch_padded_real_conv<...> specialization), but never calls or instantiates the body, so
// it never compiles a single cuFFTDx kernel itself.
//
// The definition lives in real_conv_2d_dispatch_impl.cuh, which IS included by cuFFTDx and is
// in turn included only by the generated per-family shard .cu files under src/cuda/generated/
// (see setup.py's generate_real_conv_2d_shards()). Each shard provides the explicit
// instantiations for its slice of SUPPORTED_CONV_CONFIGS; the linker resolves the binding TU's
// function-pointer references against those symbols.
#pragma once

#include <cuda_runtime.h>

template <unsigned int SignalLengthX, unsigned int SignalLengthY, unsigned int FFTSizeX,
          unsigned int FFTSizeY, unsigned int BatchSize, bool CrossCorrelate,
          bool UseTiledSwizzledIO, unsigned int FFTsPerBlockY>
void dispatch_padded_real_conv(float* input_data, float2* fft_workspace, const float2* conv_data,
                               float* output_data, int device_index, cudaStream_t stream);
