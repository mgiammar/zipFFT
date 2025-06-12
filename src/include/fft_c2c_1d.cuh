#ifndef FFT_C2C_1D_CUH
#define FFT_C2C_1D_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

template <typename T, unsigned int DataSize>
int block_fft_c2c_1d(T* data);

#endif // FFT_C2C_1D_CUH