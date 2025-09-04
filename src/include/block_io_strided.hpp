#ifndef CUFFTDX_EXAMPLE_BLOCK_IO_STRIDED_HPP
#define CUFFTDX_EXAMPLE_BLOCK_IO_STRIDED_HPP

#include "block_io.hpp"
#include "mixed_io.hpp"

namespace example {
    template<class FFT>
    struct io_strided: public io<FFT> {
        using base_type = io<FFT>;

        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        static inline __device__ unsigned int batch_id(unsigned int local_fft_id) {
            //unsigned int global_fft_id = blockIdx.x * FFT::ffts_per_block + local_fft_id;
            unsigned int global_fft_id = blockIdx.y; //x * FFT::ffts_per_block + local_fft_id;
            return global_fft_id;
        }

        static inline __device__ unsigned int batch_offset_strided(unsigned int local_fft_id, unsigned int stride_len) {
            //return batch_id(local_fft_id);

            unsigned int global_fft_offset = local_fft_id + blockIdx.y * FFT::ffts_per_block + blockIdx.x * stride_len * cufftdx::size_of<FFT>::value;
            return global_fft_offset;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        // Note: loads / stores for 2d FFTs, based on stride and batches for specific dimension (inner most)
        template<typename InputOutputType>
        static inline __device__ void load_strided(const InputOutputType* input,
                                                   complex_type*          thread_data,
                                                   unsigned int           local_fft_id,
                                                   unsigned int stride_len) {
            // Calculate global offset of FFT batch
            const unsigned int batch_offset = batch_offset_strided(local_fft_id, stride_len);
            const unsigned int bid          = batch_id(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride       = stride_len * FFT::stride;
            unsigned int       index        = batch_offset + (threadIdx.x * stride_len);
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    thread_data[i] = convert<complex_type>(input[index]);
                    index += stride;
                }
            }
        }

        template<typename InputOutputType>
        static inline __device__ void store_strided(const complex_type* thread_data,
                                                    InputOutputType*    output,
                                                    unsigned int        local_fft_id,
                                                    unsigned int stride_len) {
            const unsigned int batch_offset = batch_offset_strided(local_fft_id, stride_len);
            const unsigned int bid          = batch_id(local_fft_id);
            const unsigned int stride       = stride_len * FFT::stride;
            unsigned int       index        = batch_offset + (threadIdx.x * stride_len);
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    output[index] = convert<InputOutputType>(thread_data[i]);
                    index += stride;
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Note: loads and stores for 2d FFTs with shared memory used, based on stride and batches for specific dimension (inner most)
        template<typename InputOutputType>
        static inline __device__ void load_strided_smem(const InputOutputType* input,
                                                   complex_type*          thread_data,
                                                   InputOutputType*       shared_memory,
                                                   unsigned int           local_fft_id,
                                                   unsigned int stride_len) {
            const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int tidx         = tid / blockDim.y;
            const unsigned int tidy         = tid % blockDim.y;
            // Calculate global offset of FFT batch
            const unsigned int batch_offset = batch_offset_strided(tidy, stride_len);
            // Get stride, this shows how elements from batch should be split between threads
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
                    thread_data[i] = convert<complex_type>(shared_memory[smem_index]);
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        // Note: loads / stores for 2d FFTs, based on stride and batches for specific dimension (inner most)
        template<typename InputOutputType>
        static inline __device__ void load_transposed_kernel(const InputOutputType* input,
                                                   complex_type*          thread_data) {
            // Calculate global offset of FFT batch
            const size_t stride       = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
            size_t       index        = threadIdx.x + blockDim.x * threadIdx.y;
            index += (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                thread_data[i] = convert<complex_type>(input[index]);
                index += stride;
            }
        }

        template<typename InputOutputType>
        static inline __device__ void store_transposed_kernel(const complex_type* thread_data,
                                                    InputOutputType*    output) {
            const size_t stride       = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
            size_t       index        = threadIdx.x + blockDim.x * threadIdx.y;
            index += (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                output[index] = convert<InputOutputType>(thread_data[i]);
                index += stride;
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Note: loads and stores for 2d FFTs with shared memory used, based on stride and batches for specific dimension (inner most)
        template<typename InputOutputType>
        static inline __device__ void load_strided_padded_smem(const InputOutputType* input,
                                                   complex_type*          thread_data,
                                                   InputOutputType*       shared_memory,
                                                   unsigned int           local_fft_id,
                                                   unsigned int stride_len,
                                                   int s) {
            const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int tidx         = tid / blockDim.y;
            const unsigned int tidy         = tid % blockDim.y;
            // Calculate global offset of FFT batch
            const unsigned int batch_offset = batch_offset_strided(tidy, stride_len);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride       = stride_len * FFT::stride;
            unsigned int       index        = batch_offset + (tidx * stride_len);
            unsigned int       smem_index   = tidx + tidy * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                unsigned int fft_index = i * FFT::stride + tidx;

                if (fft_index < s) {
                    shared_memory[smem_index] = input[index];
                    index += stride;
                    smem_index += (blockDim.x * blockDim.y);
                } else if (fft_index < cufftdx::size_of<FFT>::value) {
                    shared_memory[smem_index] = get_zero<InputOutputType>();
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
            __syncthreads();
            smem_index = threadIdx.x + threadIdx.y * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                unsigned int fft_index = i * FFT::stride + threadIdx.x;

                if (fft_index < s) {
                    thread_data[i] = convert<complex_type>(shared_memory[smem_index]);
                    smem_index += (blockDim.x * blockDim.y);
                } else if (fft_index < cufftdx::size_of<FFT>::value) {
                    thread_data[i] = get_zero<complex_type>();
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
        }

        template<typename InputOutputType>
        static inline __device__ void store_strided_smem(const complex_type* thread_data,
                                                    InputOutputType*    shared_memory,
                                                    InputOutputType*    output,
                                                    unsigned int        local_fft_id,
                                                    unsigned int stride_len) {
            __syncthreads();
            unsigned int smem_index = threadIdx.x + threadIdx.y * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    shared_memory[smem_index] = convert<InputOutputType>(thread_data[i]);
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
            __syncthreads();
            const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int tidx         = tid / blockDim.y;
            const unsigned int tidy         = tid % blockDim.y;
            const unsigned int batch_offset = batch_offset_strided(tidy, stride_len);
            const unsigned int stride       = stride_len * FFT::stride;
            unsigned int       index        = batch_offset + (tidx * stride_len);
            smem_index                      = tidx + tidy * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
                    output[index] = shared_memory[smem_index];
                    index += stride;
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
        }
    };
} // namespace example

#endif // CUFFTDX_EXAMPLE_BLOCK_IO_STRIDED_HPP
