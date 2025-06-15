#ifndef FFT_R2C_1D_CUH
#define FFT_R2C_1D_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

template<typename Input_T, typename Output_T, unsigned int FFTSize, bool IsForwardFFT>
int block_real_fft_1d(Input_T* input_data, Output_T* output_data);

#endif // FFT_R2C_1D_CUH