#ifndef ZIPFFT_IO_STRIDED_HPP_
#define ZIPFFT_IO_STRIDED_HPP_

#include <type_traits>

#include "zipfft_block_io.hpp"
#include "zipfft_common.hpp"

namespace zipfft {
// I/O functionality for block-based FFT execution with cuFFTDx where data
// on a block level is accessed in a strided pattern (e.g., non-contigious
// dimension for a )
template <class FFT, unsigned int Stride>
struct io_strided : public io<FFT> {
    using io<FFT>::apparent_ffts_per_block;

    // Starting array offset for this batch within global memory
    static inline __device__ unsigned int input_batch_offset(unsigned int local_fft_id) {
        unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        return global_fft_id;  // corresponds to column in 2D array
    }

    // Starting array offset for this batch within global memory
    static inline __device__ unsigned int output_batch_offset(unsigned int local_fft_id) {
        unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        return global_fft_id;  // corresponds to column in 2D array
    }

    static inline __device__ unsigned int input_batch_id(unsigned int local_fft_id) {
        return input_batch_offset(local_fft_id);
    }

    static inline __device__ unsigned int output_batch_id(unsigned int local_fft_id) {
        return output_batch_offset(local_fft_id);
    }

    // Load for multi-dimensional FFTs across non-contigious (not
    // innermost) dimension using stride to access values. Templated structure
    // parameter 'Stride' defines this and can be used for multi-dimensional
    // arrays even above 2D.
    // NOTE: templated Batches parameter is used to prevent out-of-bounds
    // memory access when there are fewer than 'Stride' elements to read in
    template <unsigned int Batches = Stride, typename RegisterType, typename IOType>
    static inline __device__ void load(const IOType* input, RegisterType* thread_data,
                                       unsigned int local_fft_id) {
        using input_t = typename FFT::input_type;

        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);
        const unsigned int batch_id = input_batch_id(local_fft_id);

        const unsigned int batch_offset = input_batch_offset(local_fft_id);
        const unsigned int stride = Stride * FFT::stride;
        unsigned int index = batch_offset + (threadIdx.x * Stride * inner_loop_limit);

        // Loop over all elements doing appropriate memory reads
        for (unsigned int i = 0; i < FFT::input_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    if (batch_id < Batches) {
                        thread_data[i * inner_loop_limit + j] =
                            convert<RegisterType>(input[index + j]);
                    }
                }
                index += inner_loop_limit * stride;
            }
        }
    }

    // Store for multi-dimensional FFTs across non-contigious (not innermost)
    // dimension using stride to access values.
    template <unsigned int Batches = Stride, typename RegisterType, typename IOType>
    static inline __device__ void store(const RegisterType* thread_data, IOType* output,
                                        unsigned int local_fft_id) {
        using output_t = typename FFT::output_type;

        constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(IOType);
        const unsigned int batch_id = output_batch_id(local_fft_id);

        const unsigned int batch_offset = output_batch_offset(local_fft_id);
        const unsigned int stride = Stride * FFT::stride;
        unsigned int index = batch_offset + (threadIdx.x * Stride * inner_loop_limit);

        // Loop over all elements doing appropriate memory writes
        for (unsigned int i = 0; i < FFT::output_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    if (batch_id < Batches) {
                        output[index + j] = convert<IOType>(thread_data[i * inner_loop_limit + j]);
                    }
                }
                index += inner_loop_limit * stride;
            }
        }
    }

    // TODO: implement the shared memory version of the above functions
    //       (and profile the relative performance impact).
};  // struct io_strided
}  // namespace zipfft

#endif  // ZIPFFT_IO_STRIDED_HPP_