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

template<class FFT, bool smem_transpose>
static inline __device__ void load_strided(const float2* input,
                                            float2*          thread_data,
                                            float2*       shared_memory,
                                            unsigned int stride_len) {
    
    if constexpr (smem_transpose) {
        load_strided_smem<FFT>(input, thread_data, shared_memory, stride_len);
        return;
    }

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

template<class FFT, bool smem_transpose>
static inline __device__ void store_strided(const float2* thread_data,
                                            float2*    shared_memory,
                                            float2*    output,
                                            unsigned int stride_len) {
    if constexpr (smem_transpose) {
        store_strided_smem<FFT>(thread_data, shared_memory, output, stride_len);
        return;
    }

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

template<class FFT, bool smem_transpose, int padding_ratio>
static inline __device__ void load_strided_padded(const float2* input,
                                            float2*          thread_data,
                                            float2*       shared_memory,
                                            unsigned int stride_len) {
    
    if constexpr (smem_transpose) {
        load_strided_padded_smem<FFT, padding_ratio>(input, thread_data, shared_memory, stride_len);
        return;
    }

    constexpr unsigned int signal_len = cufftdx::size_of<FFT>::value / padding_ratio;
    const unsigned int batch_offset = threadIdx.y + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;
    const unsigned int stride       = stride_len * FFT::stride;
    unsigned int       index        = batch_offset + (threadIdx.x * stride_len);

    #pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        if ((i * FFT::stride + threadIdx.x) < signal_len) {
            thread_data[i] = input[index];
            index += stride;
        } else {
            thread_data[i] = float2{0.0f, 0.0f};
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

template<class FFT>
static inline unsigned int get_transposed_kernel_size() {
    return cufftdx::size_of<FFT>::value;
}

template<class FFT, bool smem_transpose, bool read_kernel_transposed> 
__device__ void apply_kernel(float2* kernel, float2* thread_data, float2* shared_mem, unsigned int inner_batch_count) {
    if constexpr (read_kernel_transposed) {
        const size_t kernel_stride       = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
        size_t       kernel_index        = threadIdx.x + blockDim.x * threadIdx.y;
        kernel_index += (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y;

        // complex multiplication in the frequency domain
        for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
            float2 kernel_thread_data = kernel[kernel_index];
            kernel_index += kernel_stride;

            float2 a;
            a.x = thread_data[i].x;
            a.y = thread_data[i].y;

            float2 b;
            b.x = kernel_thread_data.x;
            b.y = kernel_thread_data.y;
            
            float2 c;
            c.x = a.x * b.x - a.y * b.y;
            c.y = a.x * b.y + a.y * b.x;

            thread_data[i].x = c.x;
            thread_data[i].y = c.y;
        }
    } else {
        // Local array for thread
        float2 kernel_thread_data[FFT::storage_size];

        if constexpr (smem_transpose)
            __syncthreads();

        load_strided<FFT, smem_transpose>(kernel, kernel_thread_data, shared_mem, inner_batch_count * FFT::ffts_per_block);

        if constexpr (smem_transpose)
            __syncthreads();

        // complex multiplication in the frequency domain
        for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
            float2 a;
            a.x = thread_data[i].x;
            a.y = thread_data[i].y;

            float2 b;
            b.x = kernel_thread_data[i].x;
            b.y = kernel_thread_data[i].y;
            
            float2 c;
            c.x = a.x * b.x - a.y * b.y;
            c.y = a.x * b.y + a.y * b.x;

            thread_data[i].x = c.x;
            thread_data[i].y = c.y;
        }
    }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void kernel_transpose_kernel(
        float2* data,
        float2* kernel,
        unsigned int inner_batch_count) {
    
    // Local array for thread
    float2 thread_data[FFT::storage_size];

    load_strided<FFT, false>(
        data,
        thread_data,
        static_cast<float2*>(nullptr),
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