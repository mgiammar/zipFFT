#ifndef COMPLEX_FFT_1D_CUH
#define COMPLEX_FFT_1D_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

template <typename T, unsigned int FFTSize>
int block_fft_c2c_1d(T* data);

template <typename T, unsigned int FFTSize>
int block_ifft_c2c_1d(T* data);

#endif // COMPLEX_FFT_1D_CUH