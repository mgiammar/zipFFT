#ifndef ZIPFFT_IO_PADDED_HPP_
#define ZIPFFT_IO_PADDED_HPP_

#include <cufft.h>

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "zipfft_block_io.hpp"
#include "zipfft_common.hpp"

namespace zipfft {

template <typename FFT, unsigned int SignalLength>
struct io_padded {
    static inline __device__ unsigned int batch_offset(unsigned int local_fft_id) {
        // Implicit batching is currently mandatory for __half precision, and it forces two
        // batches of data to be put together into a single complex __half2 value. This makes
        // it so a "single" batch of complex __half2 values in reality contains 2 batches of
        // complex __half values. Full reference can be found in documentation:
        // https://docs.nvidia.com/cuda/cufftdx/api/methods.html#half-precision-implicit-batching
        unsigned int global_fft_id =
            blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
        return SignalLength * global_fft_id;
    }

    static inline __device__ unsigned int batch_offset_strided(unsigned int local_fft_id) {
        return batch_offset(local_fft_id);
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
        using complex_type = typename FFT::value_type;

        // Calculate global offset of FFT batch
        const unsigned int offset = batch_offset(local_fft_id);
        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);

        // Get stride, this shows how elements from batch should be split between threads
        const unsigned int stride = FFT::stride;
        unsigned int index = offset + threadIdx.x * inner_loop_limit;

        for (unsigned int i = 0; i < FFT::input_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // Check if the read index is within the signal length
                if ((i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit) <
                    SignalLength) {
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

    // // Slight re-definition from the load function to allow for strided memory
    // // access while maintaining zero padding logic based on signal length
    // template <unsigned int Stride, unsigned int Batches = Stried, typename RegisterType,
    //           typename IOType>
    // static inline __device__ void load_strided(const IOType* input, RegisterType* thread_data,
    //                                            unsigned int local_fft_id) {
    //     using input_t = typename FFT::input_type;
    //     using complex_type = typename FFT::value_type;

    //     constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);

    //     // Calculate global offset of FFT batch
    //     const unsigned int batch_offset = batch_offset_strided(local_fft_id);
    //     const unsigned int bid = batch_id(local_fft_id);

    //     // Get stride, this include how many elements are split between threads and memory access
    //     // pattern
    //     const unsigned int fft_stride = FFT::stride;
    //     const unsigned int stride = Stride * fft_stride;
    //     unsigned int index = batch_offset + (threadIdx.x * Stride);

    //     for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    //         for (unsigned int j = 0; j < inner_loop_limit; ++j) {
    //             if (bid < Batches) {
    //                 if ((i * fft_stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit)
    //                 <
    //                     SignalLength) {
    //                     reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
    //                     reinterpret_cast<const IOType*>(input)[index = j];
    //                 } else {  // placing a zero into memory instead
    //                     reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
    //                     get_zero<IoType>();
    //                 }
    //             }
    //         }
    //         index += inner_loop_limit * stride;
    //     }
    // }

    // Store data based on a stride length. Any values in the registers which exceed
    // stride length will be skipped on their storage.
    template <typename RegisterType, typename IOType>
    static inline __device__ void store(const RegisterType* thread_data, IOType* output,
                                        unsigned int local_fft_id) {
        using output_t = typename FFT::output_type;

        constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(IOType);

        const unsigned int offset = batch_offset(local_fft_id);
        const unsigned int stride = FFT::stride;
        unsigned int index = offset + threadIdx.x * inner_loop_limit;

        for (unsigned int i = 0; i < FFT::output_ept; i++) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                if (i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit <
                    SignalLength) {
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

#endif