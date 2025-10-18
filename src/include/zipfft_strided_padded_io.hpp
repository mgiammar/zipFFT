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

        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);
        const unsigned int batch_id = input_batch_id(local_fft_id);

        const unsigned int batch_offset = input_batch_offset(local_fft_id);
        const unsigned int stride = Stride * FFT::stride;
        unsigned int index = batch_offset + (threadIdx.x * Stride * inner_loop_limit);

        const unsigned int signal_length_limit = SignalLength * Stride;

        unsigned int read_idx;

        // Loop over all elements doing appropriate memory reads
        for (unsigned int i = 0; i < FFT::input_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // // Check batch ID against Batches to prevent out-of-bounds access
                // if (batch_id < Batches) {
                read_idx = (i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit);
                if (read_idx < signal_length_limit) {
                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        reinterpret_cast<const IOType*>(input)[index + j];
                } else {
                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        get_zero<IOType>();
                }
                index += inner_loop_limit * stride;
                // }
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
        using input_t = typename FFT::input_type;

        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);
        const unsigned int batch_id = output_batch_id(local_fft_id);

        const unsigned int batch_offset = output_batch_offset(local_fft_id);
        const unsigned int stride = Stride * FFT::stride;
        unsigned int index = batch_offset + (threadIdx.x * Stride * inner_loop_limit);

        const unsigned int signal_length_limit = SignalLength * Stride;

        unsigned int write_idx;

        // Loop over all elements doing appropriate memory writes
        for (unsigned int i = 0; i < FFT::output_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // // Check batch ID against Batches to prevent out-of-bounds access
                // if (batch_id < Batches) {
                write_idx = i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit;
                if (write_idx < signal_length_limit) {
                    reinterpret_cast<IOType*>(output)[index + j] =
                        reinterpret_cast<const IOType*>(thread_data)[i * inner_loop_limit + j];
                }
                // }
                index += inner_loop_limit * stride;
            }
        }
    }
};  // struct io_strided_padded
}  // namespace zipfft

#endif  // ZIPFFT_STRIDED_PADDED_IO_HPP_