#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftdx.hpp>

template<class FFT>
static inline __device__ void load_strided_smem(const float2* input,
                                            float2*          thread_data,
                                            float2*       shared_memory,
                                            unsigned int stride_len) {
    const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
    const unsigned int tidx         = tid / blockDim.y;
    const unsigned int tidy         = tid % blockDim.y;

    const unsigned int batch_offset = tidy + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;

    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (tidx * stride_len);
    unsigned int       smem_index   = tidx + tidy * blockDim.x;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
            shared_memory[smem_index] = input[index];
            
            index += stride;
            smem_index += (blockDim.x * blockDim.y);
        }
    }
    __syncthreads();
    smem_index = threadIdx.x + threadIdx.y * blockDim.x;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            thread_data[i] = shared_memory[smem_index];
            smem_index += (blockDim.x * blockDim.y);
        }
    }
}

template<class FFT>
static inline __device__ void load_strided(const float2* input,
                                            float2*          thread_data,
                                            unsigned int stride_len) {

    const unsigned int batch_offset = threadIdx.y + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;
    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (threadIdx.x * stride_len);

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            thread_data[i] = input[index];
            index += stride;
        }
    }
}

template<class FFT>
static inline __device__ void store_strided_smem(const float2* thread_data,
                                            float2*    shared_memory,
                                            float2*    output,
                                            unsigned int stride_len) {
    unsigned int smem_index = threadIdx.x + threadIdx.y * blockDim.x;

    __syncthreads();
    
    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            shared_memory[smem_index] = thread_data[i];
            smem_index += (blockDim.x * blockDim.y);
        }
    }
    __syncthreads();

    const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
    const unsigned int tidx         = tid / blockDim.y;
    const unsigned int tidy         = tid % blockDim.y;
    const unsigned int batch_offset = tidy + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;
    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (tidx * stride_len);
    smem_index                      = tidx + tidy * blockDim.x;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
            output[index] = shared_memory[smem_index];

            index += stride;
            smem_index += (blockDim.x * blockDim.y);
        }
    }
}

template<class FFT>
static inline __device__ void store_strided(const float2* thread_data,
                                            float2*    output,
                                            unsigned int stride_len) {

    const unsigned int batch_offset = threadIdx.y + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;
    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (threadIdx.x * stride_len);

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
            output[index] = thread_data[i];
            index += stride;
        }
    }
}

template<class FFT, int padding_ratio>
static inline __device__ void load_strided_padded_smem(const float2* input,
                                            float2*          thread_data,
                                            float2*       shared_memory,
                                            unsigned int stride_len) {
    const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
    const unsigned int tidx         = tid / blockDim.y;
    const unsigned int tidy         = tid % blockDim.y;
    
    constexpr unsigned int signal_len = cufftdx::size_of<FFT>::value / padding_ratio;
    const unsigned int batch_offset = tidy + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;

    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (tidx * stride_len);
    unsigned int       smem_index   = tidx + tidy * blockDim.x;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        unsigned int fft_index = i * FFT::stride + tidx;

        if (fft_index < signal_len) {
            shared_memory[smem_index] = input[index];

            index += stride;
            smem_index += (blockDim.x * blockDim.y);
        } else if (fft_index < cufftdx::size_of<FFT>::value) {
            //shared_memory[smem_index] = float2{0.0f, 0.0f};
            
            smem_index += (blockDim.x * blockDim.y);
        }
    }

    __syncthreads();
    smem_index = threadIdx.x + threadIdx.y * blockDim.x;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        unsigned int fft_index = i * FFT::stride + threadIdx.x;

        if (fft_index < signal_len) {
            thread_data[i] = shared_memory[smem_index];
            smem_index += (blockDim.x * blockDim.y);
        } else if (fft_index < cufftdx::size_of<FFT>::value) {
            thread_data[i] = float2{0.0f, 0.0f};
            smem_index += (blockDim.x * blockDim.y);
        }
    }
}

template<class FFT>
static inline __device__ void load_transposed_kernel(const float2* input, float2* thread_data) {
    // Calculate global offset of FFT batch
    const size_t stride       = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    size_t       index        = threadIdx.x + blockDim.x * threadIdx.y;
    index += (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i] = input[index];
        index += stride;
    }
}

template<class FFT>
static inline __device__ void store_transposed_kernel(const float2* thread_data, float2* output) {
    const size_t stride       = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    size_t       index        = threadIdx.x + blockDim.x * threadIdx.y;
    index += (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        output[index] = thread_data[i];
        index += stride;
    }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void kernel_transpose_kernel(
        float2* data,
        float2* kernel,
        unsigned int inner_batch_count) {

    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    
    // Local array for thread
    float2 thread_data[FFT::storage_size];

    load_strided_smem<FFT>(
        data,
        thread_data,
        shared_mem,
        inner_batch_count * FFT::ffts_per_block
    );

    store_transposed_kernel<FFT>(thread_data, kernel);

    //const size_t kernel_stride       = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    //size_t       kernel_index        = threadIdx.x + blockDim.x * threadIdx.y;
    //kernel_index += (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y;
    //for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    //    kernel[kernel_index] = thread_data[i];
    //    kernel_index += kernel_stride;
    //}
}