#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftdx.hpp>

template<class FFT>
static inline __device__ void load_strided_smem(const float2* input,
                                            float2*          thread_data,
                                            float2*       shared_memory,
                                            unsigned int           local_fft_id,
                                            unsigned int stride_len) {
    const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
    const unsigned int tidx         = tid / blockDim.y;
    const unsigned int tidy         = tid % blockDim.y;

    const unsigned int batch_offset = tidy + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;

    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (tidx * stride_len);
    unsigned int       smem_index   = tidx + tidy * blockDim.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
            shared_memory[smem_index] = input[index];
            index += stride;
            smem_index += (blockDim.x * blockDim.y);
        }
    }
    __syncthreads();
    smem_index = threadIdx.x + threadIdx.y * blockDim.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            thread_data[i] = shared_memory[smem_index];
            smem_index += (blockDim.x * blockDim.y);
        }
    }
}

template<class FFT>
static inline __device__ void store_strided_smem(const float2* thread_data,
                                            float2*    shared_memory,
                                            float2*    output,
                                            unsigned int        local_fft_id,
                                            unsigned int stride_len,
                                            bool disable_transpose) {
    unsigned int smem_index = threadIdx.x + threadIdx.y * blockDim.x;

    if (!disable_transpose) {
        __syncthreads();
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                shared_memory[smem_index] = thread_data[i];
                smem_index += (blockDim.x * blockDim.y);
            }
        }
        __syncthreads();
    }

    const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
    const unsigned int tidx         = tid / blockDim.y;
    const unsigned int tidy         = tid % blockDim.y;
    const unsigned int batch_offset = tidy + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value; //batch_offset_strided(tidy, stride_len);
    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (tidx * stride_len);
    smem_index                      = tidx + tidy * blockDim.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
            if (disable_transpose)
                output[index] = thread_data[i];
            else
                output[index] = shared_memory[smem_index];
            
                index += stride;
            smem_index += (blockDim.x * blockDim.y);
        }
    }
}

