#ifndef COMPLEX_FFT_1D_CUH
#define COMPLEX_FFT_1D_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

template <typename T, unsigned int FFTSize, bool IsForwardFFT, unsigned int elements_per_thread, unsigned int FFTs_per_block>
int block_complex_fft_1d(T* data);

#endif // COMPLEX_FFT_1D_CUH