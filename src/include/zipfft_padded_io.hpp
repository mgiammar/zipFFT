#ifndef ZIPFFT_IO_PADDED_HPP_
#define ZIPFFT_IO_PADDED_HPP_

#include <cufft.h>

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "zipfft_block_io.hpp"
#include "zipfft_common.hpp"

namespace zipfft {

template <typename FFT, unsigned int SignalLength>
struct io_padded : public io<FFT> {
    using io<FFT>::apparent_ffts_per_block;

    // Starting array offset for this batch within global memory
    // NOTE: SignalLength < = FFT::input_length and SignalLength corresponds to
    // the size of data in global memory.
    static inline __device__ unsigned int input_batch_offset(unsigned int local_fft_id) {
        unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        return SignalLength * global_fft_id;
    }

    // Starting array offset for this batch within global memory
    // NOTE: SignalLength < = FFT::output_length and SignalLength corresponds to
    // the size of data in global memory.
    static inline __device__ unsigned int output_batch_offset(unsigned int local_fft_id) {
        unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        return SignalLength * global_fft_id;
    }

    // Do a zero-padded load of the data into the register. Structure templated
    // SignalLength parameter determines how many elements should be read from
    // memory (e.g. 32 non-zero values), and the function places zeros in all
    // other register positions where the FFT length surpasses the signal
    // length
    template <typename RegisterType, typename IOType>
    static inline __device__ void load(const IOType* input, RegisterType* thread_data,
                                       unsigned int local_fft_id) {
        using input_t = typename FFT::input_type;

        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);

        const unsigned int batch_offset = input_batch_offset(local_fft_id);
        const unsigned int stride = FFT::stride;
        unsigned int index = batch_offset + threadIdx.x * inner_loop_limit;

        unsigned int read_idx;

        for (unsigned int i = 0; i < FFT::input_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // Check if the read index (of 1D array) is within the signal length
                read_idx = (i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit);
                if (read_idx < SignalLength) {
                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        reinterpret_cast<const IOType*>(input)[index + j];
                } else {
                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        get_zero<IOType>();
                }
            }
            index += inner_loop_limit * stride;
        }
    }

    // Store data based on a stride length. Any values in the registers which exceed
    // stride length will be skipped on their storage.
    template <typename RegisterType, typename IOType>
    static inline __device__ void store(const RegisterType* thread_data, IOType* output,
                                        unsigned int local_fft_id) {
        using output_t = typename FFT::output_type;

        constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(IOType);

        const unsigned int batch_offset = output_batch_offset(local_fft_id);
        const unsigned int stride = FFT::stride;
        unsigned int index = batch_offset + threadIdx.x * inner_loop_limit;

        unsigned int read_idx;

        for (unsigned int i = 0; i < FFT::output_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // Check if the read index (of 1D array) is within the signal length
                read_idx = i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit;
                if (read_idx < SignalLength) {
                    reinterpret_cast<IOType*>(output)[index + j] =
                        reinterpret_cast<const IOType*>(thread_data)[i * inner_loop_limit + j];
                }
            }
            index += inner_loop_limit * stride;
        }
    }

    // TODO: store strided.

};  // namespace io_padded

}  // namespace zipfft

#endif  // ZIPFFT_IO_PADDED_HPP_