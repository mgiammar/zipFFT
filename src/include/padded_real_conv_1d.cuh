#ifndef PADDED_REAL_CONV_1D_CUH
#define PADDED_REAL_CONV_1D_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

template<typename ScalarType, typename ComplexType, unsigned int SignalLength, unsigned int FFTSize, unsigned int elements_per_thread, unsigned int FFTs_per_block>
int padded_block_conv_real_1d(
    ScalarType* input_data,
    ScalarType* output_data,
    ComplexType* filter_data
);

#endif // PADDED_REAL_CONV_1D_CUH
