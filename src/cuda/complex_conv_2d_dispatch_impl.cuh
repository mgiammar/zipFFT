// Definition of dispatch_padded_complex_conv (declared in complex_conv_2d_dispatch.hpp). Kept out
// of complex_conv_2d_binding.cu so that only the generated per-family shard .cu files -- which
// include this header -- ever compile the cuFFTDx kernel chain (via complex_conv_2d.cuh); the
// binding TU includes only the declaration. See complex_conv_2d_dispatch.hpp for the full
// rationale.
#pragma once

#include "./complex_conv_2d.cuh"
#include "./complex_conv_2d_dispatch.hpp"

template <unsigned int SignalLengthX, unsigned int SignalLengthY, unsigned int FFTSizeX,
          unsigned int FFTSizeY, unsigned int BatchSize, bool CrossCorrelate,
          bool UseTiledSwizzledIO, unsigned int FFTsPerBlockY>
void dispatch_padded_complex_conv(float2* input_data, float2* fft_workspace,
                                  const float2* conv_data, float2* output_data, int device_index,
                                  cudaStream_t stream) {
    // NOTE: elements_per_thread and ffts_per_block_x are left at their cuFFTDx-recommended
    // defaults (0); only ffts_per_block_y is ever overridden, and only because
    // UseTiledSwizzledIO requires it (see real_conv_2d_io.hpp).
    padded_block_complex_conv_2d<float2, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY,
                                 BatchSize, CrossCorrelate, 0, 0, 0, FFTsPerBlockY,
                                 UseTiledSwizzledIO>(input_data, fft_workspace, conv_data,
                                                     output_data, device_index, stream);
}
