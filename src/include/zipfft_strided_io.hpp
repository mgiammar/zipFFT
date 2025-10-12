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
struct io_strided {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    static constexpr bool is_supported_type() {
        // Check if this structure is currently supported
        // NOTE: I'm including this to restrict to float/float2 computation
        //       only for now since I've not delved
        //       into __half/__half2 support for mixed precision.
        return std::is_same_v<scalar_type, float>;
    }

    static inline __device__ unsigned int batch_id(unsigned int local_fft_id) {
        unsigned int global_fft_id = blockIdx.x * FFT::ffts_per_block + local_fft_id;
        return global_fft_id;
    }

    static inline __device__ unsigned int batch_offset_strided(unsigned int local_fft_id) {
        // Function is extremely similar to the base 'input_batch_offset' func
        // from the block io structure, but does not include implicit batching
        // for __half precision. May become combined in the future...
        return batch_id(local_fft_id);
    }

    // Load for multi-dimensional FFTs across non-contigious (not
    // innermost) dimension using stride to access values. Templated structure
    // parameter 'Stride' defines this and can be used for multi-dimensional
    // arrays even above 2D.
    // NOTE: templated Batches parameter is used to prevent out-of-bounds
    // memory access when there are fewer than 'Stride' elements to read in
    template <unsigned int Batches = Stride, typename IOType>
    static inline __device__ void load_strided(const IOType* input, complex_type* thread_data,
                                               unsigned int local_fft_id) {
        // Calc variables for loop
        const unsigned int batch_offset = batch_offset_strided(local_fft_id);
        const unsigned int bid = batch_id(local_fft_id);
        const unsigned int stride = Stride * FFT::stride;
        unsigned int index = batch_offset + (threadIdx.x * Stride);

        // Loop over all element doing appropriate memory reads
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                if (bid < Batches) {
                    thread_data[i] = convert<IOType>(input[index]);
                }
            }
            index += stride;
        }
    }

    // Store for multi-dimensional FFTs across non-contigious (not innermost)
    // dimension using stride to access values.
    template <unsigned int Batches = Stride, typename IOType>
    static inline __device__ void store_strided(const complex_type* thread_data, IOType* output,
                                                unsigned int local_fft_id) {
        // Calc variables for loop
        const unsigned int batch_offset = batch_offset_strided(local_fft_id);
        const unsigned int bid = batch_id(local_fft_id);
        const unsigned int stride = Stride * FFT::stride;
        unsigned int index = batch_offset + (threadIdx.x * Stride);

        // Loop over all element doing appropriate memory reads
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                if (bid < Batches) {
                    output[index] = convert<IOType>(thread_data[i]);
                }
            }
            index += stride;
        }
    }
};  // struct io_strided
}  // namespace zipfft

#endif  // ZIPFFT_IO_STRIDED_HPP_