#ifndef ZIPFFT_IO_PADDED_HPP_
#define ZIPFFT_IO_PADDED_HPP_

#include <cufft.h>

#include <type_traits>

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "zipfft_block_io.hpp"
#include "zipfft_common.hpp"
#include "zipfft_index_mapper.hpp"

namespace zipfft {

// Enhanced I/O functionality for padded memory access with explicit index mapper
// SignalLength: actual data length in memory (< FFT size, rest padded with zeros)
template <class FFT, class InputIndexMapper, class OutputIndexMapper, unsigned int SignalLength>
struct io_padded_with_layout {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    static constexpr unsigned int apparent_ffts_per_block =
        FFT::ffts_per_block / FFT::implicit_type_batching;

    // Detect dimensionality of input and output mappers
    static constexpr size_t input_mapper_dims = __io::index_mapper_dims<InputIndexMapper>::value;
    static constexpr size_t output_mapper_dims = __io::index_mapper_dims<OutputIndexMapper>::value;

    // Check if the register type and FFT structure are type compatible
    template <typename RegisterType, typename MemoryType>
    static constexpr bool is_type_compatible() {
        return !CUFFTDX_STD::is_void_v<RegisterType> &&
               (sizeof(RegisterType) == sizeof(complex_type)) &&
               (alignof(RegisterType) == alignof(complex_type));
    }

    // Zero-padded load function using InputIndexMapper for addressing
    // Reads SignalLength elements from memory, pads rest with zeros
    template <bool InputInRRIILayout = false, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    load(const IOType* input, RegisterType* thread_data, unsigned int local_fft_id) {
        static constexpr bool needs_half2_format_conversion =
            cufftdx::type_of<FFT>::value != cufftdx::fft_type::r2c &&
            std::is_same_v<IOType, cufftdx::detail::complex<__half2>>;

        using input_t = typename FFT::input_type;

        // Instantiate the input index mapper
        InputIndexMapper input_layout;

        // Compute global FFT ID
        const unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        const unsigned int stride = FFT::stride;

        // Decompose global_fft_id into fft_id and batch_id
        const unsigned int fft_id = global_fft_id % dim_size_v<1, InputIndexMapper>;
        const unsigned int batch_id = global_fft_id / dim_size_v<1, InputIndexMapper>;

        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);

        // Load data with zero-padding using index mapper for memory offset calculation
        for (unsigned int i = 0; i < FFT::input_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;

            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // Calculate flat element index
                const unsigned int flat_elem = elem_id * inner_loop_limit + j;

                if (flat_elem < SignalLength) {
                    // Read from memory using index mapper
                    const size_t offset = input_layout(flat_elem, fft_id, batch_id);

                    if constexpr (needs_half2_format_conversion) {
                        reinterpret_cast<input_t*>(thread_data)[i * inner_loop_limit + j] =
                            __io::convert_to_rrii<InputInRRIILayout>(
                                reinterpret_cast<const input_t*>(input)[offset]);
                    } else {
                        reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                            reinterpret_cast<const IOType*>(input)[offset];
                    }
                } else {
                    // Pad with zeros for elements beyond SignalLength
                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        get_zero<IOType>();
                }
            }
        }
    }

    // Truncated store function using OutputIndexMapper for addressing
    // Only stores first SignalLength elements, discards rest
    template <bool OutputInRRIILayout = false, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    store(const RegisterType* thread_data, IOType* output, unsigned int local_fft_id) {
        static constexpr bool needs_half2_format_conversion =
            cufftdx::type_of<FFT>::value != cufftdx::fft_type::c2r &&
            std::is_same_v<IOType, cufftdx::detail::complex<__half2>>;

        using output_t = typename FFT::output_type;

        // Instantiate the output index mapper
        OutputIndexMapper output_layout;

        // Compute global FFT ID
        const unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        const unsigned int stride = FFT::stride;

        // Decompose global_fft_id into fft_id and batch_id
        const unsigned int fft_id = global_fft_id % dim_size_v<1, OutputIndexMapper>;
        const unsigned int batch_id = global_fft_id / dim_size_v<1, OutputIndexMapper>;

        constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(IOType);

        // Store data with truncation using index mapper for memory offset calculation
        for (unsigned int i = 0; i < FFT::output_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;

            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // Calculate flat element index
                const unsigned int flat_elem = elem_id * inner_loop_limit + j;

                if (flat_elem < SignalLength) {
                    // Write to memory using index mapper
                    const size_t offset = output_layout(flat_elem, fft_id, batch_id);

                    if constexpr (needs_half2_format_conversion) {
                        reinterpret_cast<output_t*>(output)[offset] =
                            __io::convert_to_riri<OutputInRRIILayout>(
                                reinterpret_cast<const output_t*>(
                                    thread_data)[i * inner_loop_limit + j]);
                    } else {
                        reinterpret_cast<IOType*>(output)[offset] =
                            reinterpret_cast<const IOType*>(thread_data)[i * inner_loop_limit + j];
                    }
                }
                // Elements beyond SignalLength are simply not stored
            }
        }
    }
};

// Backward-compatible io_padded struct that uses default contiguous layout
template <class FFT, unsigned int SignalLength>
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

    // If InputInRRIILayout is false, then function assumes that values in input are in RIRI
    // layout, and before loading them to thread_data they are converted to RRII layout.
    // Otherwise, if InputInRRIILayout is true, then function assumes values in input are in RRII
    // layout, and don't need to be converted before loading to thread_data.
    template <bool InputInRRIILayout = false, typename RegisterType, typename IOType,
              typename LoadOp = zipfft::identity>
    static inline __device__ void load(const IOType* input, RegisterType* thread_data,
                                       unsigned int local_fft_id, LoadOp op = {}) {
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
                if ((i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit) <
                    SignalLength) {
                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        op(reinterpret_cast<const IOType*>(input)[index + j]);
                } else {
                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        get_zero<typename complex_type::value_type>();
                }
            }
            index += inner_loop_limit * stride;
        }
    }

    // Function assumes that values in thread_data are in RRII layout.
    // If OutputInRRIILayout is false, values are saved into output in RIRI layout; otherwise - in
    // RRII.
    template <bool OutputInRRIILayout = false, typename RegisterType, typename IOType,
              typename StoreOp = zipfft::identity>
    static inline __device__ void store(const RegisterType* thread_data, IOType* output,
                                        unsigned int local_fft_id, StoreOp op = {}) {
        using output_t = typename FFT::output_type;
        constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(IOType);

        const unsigned int offset = batch_offset(local_fft_id);
        const unsigned int stride = FFT::stride;
        unsigned int index = offset + threadIdx.x * inner_loop_limit;

        for (int i = 0; i < FFT::output_ept; ++i) {
            for (int j = 0; j < inner_loop_limit; ++j) {
                if (i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit <
                    SignalLength) {
                    reinterpret_cast<IOType*>(output)[index + j] =
                        op(reinterpret_cast<const IOType*>(thread_data)[i * inner_loop_limit + j]);
                }
            }
            index += inner_loop_limit * stride;
        }
    }
};

}  // namespace zipfft

#endif  // ZIPFFT_IO_PADDED_HPP_