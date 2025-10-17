#ifndef ZIPFFT_STRIDED_PADDED_IO_HPP_
#define ZIPFFT_STRIDED_PADDED_IO_HPP_

#include <type_traits>

#include "zipfft_block_io.hpp"
#include "zipfft_common.hpp"

namespace zipfft {
// I/O functionality for block-based FFT execution with cuFFTDx where data
// on a block level is accessed in a strided pattern AND where the signal
// is zero-padded up to a certain FFT length
template <class FFT, unsigned int Stride, unsigned int SignalLength>
struct io_strided_padded : public io<FFT> {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    static inline __device__ unsigned int batch_id(unsigned int local_fft_id) {
        // Implicit batching is currently mandatory for __half precision, and it forces two
        // batches of data to be put together into a single complex __half2 value. This makes
        // it so a "single" batch of complex __half2 values in reality contains 2 batches of
        // complex __half values. Full reference can be found in documentation:
        // https://docs.nvidia.com/cuda/cufftdx/api/methods.html#half-precision-implicit-batching
        unsigned int global_fft_id =
            blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
        return global_fft_id;
    }

    static inline __device__ unsigned int batch_offset_strided(unsigned int local_fft_id) {
        return batch_id(local_fft_id);
    }

    // Do a zero-padded load of the data into the registers while accessing the
    // data in a strided (non-contigious) pattern. Structure templated
    // SignalLength parameter determines how many elements should be read from
    // memory (e.g. 32 non-zero values), and the function places zeros in all
    // other register positions where the FFT length surpasses the signal
    // length. The Stride parameter determines how many elements are skipped
    // between each read, allowing for multi-dimensional arrays to be read
    // across non-contigious dimensions.
    // NOTE: templated Batches parameter is used to prevent out-of-bounds
    // memory access when there are fewer than 'Stride' elements to read in
    template <unsigned int Batches = Stride, typename RegisterType, typename IOType>
    static inline __device__ void load(const IOType* input, RegisterType* thread_data,
                                       unsigned int local_fft_id) {
        using input_t = typename FFT::input_type;
        using complex_type = typename FFT::value_type;

        // Inner loop limit is being used for scalar <--> vector type conversion
        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);

        // Calculate global offset of FFT batch
        // NOTE: defining 'signal_length_limit' because we are interested in
        // comparing the *strided* index within the loop
        const unsigned int batch_offset = batch_offset_strided(local_fft_id);
        const unsigned int bid = batch_id(local_fft_id);
        const unsigned int stride = Stride * FFT::stride;
        const unsigned int signal_length_limit = SignalLength * Stride;
        unsigned int index = batch_offset + threadIdx.x * (inner_loop_limit * Stride);

        for (unsigned int i = 0; i < FFT::input_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // Check if the read index is within the signal length
                if ((i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit) <
                    signal_length_limit) {
                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        reinterpret_cast<const IOType*>(input)[index + j];
                } else {
                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        get_zero<IOType>();
                }
                index += inner_loop_limit * stride;
            }
        }
    }

    // Store data based on a stride length. Any values in the registers which exceed
    // stride length will be skipped on their storage. The Stride parameter
    // determines how many elements are skipped between each write, allowing for
    // multi-dimensional arrays to be written across non-contigious dimensions.
    template <unsigned int Batches = Stride, typename RegisterType, typename IOType>
    static inline __device__ void store(const RegisterType* thread_data, IOType* output,
                                        unsigned int local_fft_id) {
        using output_t = typename FFT::output_type;

        // Inner loop limit is being used for scalar <--> vector type conversion
        constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(IOType);

        // Calculate global offset of FFT batch
        // NOTE: defining 'signal_length_limit' because we are interested in
        // comparing the *strided* index within the loop
        const unsigned int batch_offset = batch_offset_strided(local_fft_id);
        const unsigned int bid = batch_id(local_fft_id);
        const unsigned int stride = Stride * FFT::stride;
        const unsigned int signal_length_limit = SignalLength * Stride;
        unsigned int index = batch_offset + threadIdx.x * (inner_loop_limit * Stride);

        // // DEBUGGING: print block/thread indices together with calculated values
        // printf("(store) blockIdx.x: %d, threadIdx.x: %d, threadIdx.y: %d, local_fft_id: %u, "
        //        "batch_offset: %u, bid: %u, stride: %u, index: %u, inner_loop_limit: %u\n",
        //        blockIdx.x, threadIdx.x, threadIdx.y, local_fft_id, batch_offset, bid, stride,
        //        index, inner_loop_limit);

        for (unsigned int i = 0; i < FFT::output_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                if (i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit <
                    signal_length_limit) {
                    reinterpret_cast<IOType*>(output)[index + j] =
                        reinterpret_cast<const IOType*>(thread_data)[i * inner_loop_limit + j];
                }
            }
            index += inner_loop_limit * stride;
        }
    }
};  // struct io_strided_padded
}  // namespace zipfft

#endif  // ZIPFFT_STRIDED_PADDED_IO_HPP_