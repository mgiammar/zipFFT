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

    static inline __device__ unsigned int this_global_fft_id(unsigned int local_fft_id) {
        return blockIdx.x * apparent_ffts_per_block + local_fft_id;
    }

    // For an input of (outer_batch, FFT::input_length, Stride), determine
    // which index [0, outer_batch-1] we are in. Used to adjust the starting
    // read index of the load method(s).
    // NOTE: Assumes kernel grid launch dimensions are set to encompass outer_batch.
    static inline __device__ unsigned int input_outer_batch_index(unsigned int local_fft_id) {
        return this_global_fft_id(local_fft_id) / Stride;
    }

    static inline __device__ unsigned int output_outer_batch_index(unsigned int local_fft_id) {
        return this_global_fft_id(local_fft_id) / Stride;
    }

    // For an input of (outer_batch, FFT::input_length, Stride), determine
    // which index [0, Stride-1] we are in *within* an assumed 2D array. This
    // is effectively the column index within the 2D array.
    static inline __device__ unsigned int input_inner_batch_index(unsigned int local_fft_id) {
        return this_global_fft_id(local_fft_id) % Stride;
    }

    static inline __device__ unsigned int output_inner_batch_index(unsigned int local_fft_id) {
        return this_global_fft_id(local_fft_id) % Stride;
    }

    // For an input of (outer_batch, FFT::input_length, Stride), determine
    // the starting read index within global memory in the range of
    // [0, outer_batch * FFT::input_length * Stride). Is effectively the
    // starting index of the column within the 2D array.
    static inline __device__ unsigned int input_batch_offset(unsigned int local_fft_id) {
        unsigned int outer_batch_index = input_outer_batch_index(local_fft_id);
        unsigned int inner_batch_index = input_inner_batch_index(local_fft_id);
        return (outer_batch_index * FFT::input_length * Stride) + inner_batch_index;
    }

    static inline __device__ unsigned int output_batch_offset(unsigned int local_fft_id) {
        unsigned int outer_batch_index = output_outer_batch_index(local_fft_id);
        unsigned int inner_batch_index = output_inner_batch_index(local_fft_id);
        return (outer_batch_index * FFT::output_length * Stride) + inner_batch_index;
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
        const unsigned int batch_id = input_inner_batch_index(local_fft_id);

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
        const unsigned int batch_id = output_inner_batch_index(local_fft_id);

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