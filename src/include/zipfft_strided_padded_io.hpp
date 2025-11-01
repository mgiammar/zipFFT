#ifndef ZIPFFT_STRIDED_PADDED_IO_HPP_
#define ZIPFFT_STRIDED_PADDED_IO_HPP_

#include <type_traits>

#include "zipfft_block_io.hpp"
#include "zipfft_common.hpp"
#include "zipfft_index_mapper.hpp"

namespace zipfft {

// Enhanced I/O functionality for strided + padded memory access with explicit index mappers
// Supports both zero-padding on load and truncation on store
template <class FFT, class InputIndexMapper, class OutputIndexMapper, unsigned int Stride,
          unsigned int SignalLength>
struct io_strided_padded_with_layout {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    static constexpr unsigned int apparent_ffts_per_block =
        FFT::ffts_per_block / FFT::implicit_type_batching;

    // Detect dimensionality of input and output mappers
    static constexpr size_t input_mapper_dims = __io::index_mapper_dims<InputIndexMapper>::value;
    static constexpr size_t output_mapper_dims = __io::index_mapper_dims<OutputIndexMapper>::value;

    // Type compatibility check
    template <typename RegisterType, typename MemoryType>
    static constexpr bool is_type_compatible() {
        return !CUFFTDX_STD::is_void_v<RegisterType> &&
               (sizeof(RegisterType) == sizeof(complex_type)) &&
               (alignof(RegisterType) == alignof(complex_type));
    }

    // Zero-padded strided load function using InputIndexMapper
    // Reads SignalLength elements from strided memory, pads rest with zeros
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

        // Decompose global_fft_id based on dimensionality
        // For strided access: global_fft_id represents which column we're processing
        const unsigned int column_id = global_fft_id % Stride;
        const unsigned int batch_id = global_fft_id / Stride;

        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);

        // Load data with zero-padding using strided access pattern
        for (unsigned int i = 0; i < FFT::input_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;

            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // Calculate flat element index (row index for strided column access)
                const unsigned int flat_elem = elem_id * inner_loop_limit + j;

                if (flat_elem < SignalLength) {
                    // Read from memory using index mapper
                    // Mapper takes: (row_in_column, column_id, batch_id)
                    const size_t offset = input_layout(flat_elem, column_id, batch_id);

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

    // Truncated strided store function using OutputIndexMapper
    // Only stores first SignalLength elements to strided memory, discards rest
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

        // Decompose global_fft_id for strided access
        const unsigned int column_id = global_fft_id % dim_size_v<1, OutputIndexMapper>;
        const unsigned int batch_id = global_fft_id / dim_size_v<1, OutputIndexMapper>;

        constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(IOType);

        // Store data with truncation using strided access pattern
        for (unsigned int i = 0; i < FFT::output_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;

            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                // Calculate flat element index (row index for strided column access)
                const unsigned int flat_elem = elem_id * inner_loop_limit + j;

                if (flat_elem < SignalLength) {
                    // Write to memory using index mapper
                    // Mapper takes: (row_in_column, column_id, batch_id)
                    const size_t offset = output_layout(flat_elem, column_id, batch_id);

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
            }
        }
    }
};

// Backward-compatible io_strided_padded struct that uses default layouts
template <class FFT, unsigned int Stride, unsigned int SignalLength>
struct io_strided_padded : public io<FFT> {
    using io<FFT>::apparent_ffts_per_block;
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Define default strided + padded memory layouts
    // Shape: (batch, SignalLength_rows, Stride_columns)
    // Access pattern: column-major with stride, reading/writing only first SignalLength rows
    using default_input_layout =
        index_mapper<int_pair<SignalLength, Stride>,       // rows strided by Stride
                     int_pair<Stride, 1>,                  // columns contiguous
                     int_pair<1, SignalLength * Stride>>;  // batches

    using default_output_layout = default_input_layout;

    // Delegate to io_strided_padded_with_layout with default layouts
    using io_impl = io_strided_padded_with_layout<FFT, default_input_layout, default_output_layout,
                                                  Stride, SignalLength>;

    // Forward type compatibility check
    template <typename RegisterType, typename MemoryType>
    static constexpr bool is_type_compatible() {
        return io_impl::template is_type_compatible<RegisterType, MemoryType>();
    }

    // Forward load/store to implementation with default layouts
    template <bool InputInRRIILayout = false, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    load(const IOType* input, RegisterType* thread_data, unsigned int local_fft_id) {
        io_impl::template load<InputInRRIILayout>(input, thread_data, local_fft_id);
    }

    template <bool OutputInRRIILayout = false, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    store(const RegisterType* thread_data, IOType* output, unsigned int local_fft_id) {
        io_impl::template store<OutputInRRIILayout>(thread_data, output, local_fft_id);
    }

    // Legacy helper functions for backward compatibility
    static inline __device__ unsigned int this_global_fft_id(unsigned int local_fft_id) {
        return blockIdx.x * apparent_ffts_per_block + local_fft_id;
    }

    static inline __device__ unsigned int input_outer_batch_index(unsigned int local_fft_id) {
        return this_global_fft_id(local_fft_id) / Stride;
    }

    static inline __device__ unsigned int output_outer_batch_index(unsigned int local_fft_id) {
        return this_global_fft_id(local_fft_id) / Stride;
    }

    static inline __device__ unsigned int input_inner_batch_index(unsigned int local_fft_id) {
        return this_global_fft_id(local_fft_id) % Stride;
    }

    static inline __device__ unsigned int output_inner_batch_index(unsigned int local_fft_id) {
        return this_global_fft_id(local_fft_id) % Stride;
    }

    static inline __device__ unsigned int input_batch_offset(unsigned int local_fft_id) {
        return input_inner_batch_index(local_fft_id);
    }

    static inline __device__ unsigned int output_batch_offset(unsigned int local_fft_id) {
        return output_inner_batch_index(local_fft_id);
    }
};

}  // namespace zipfft

#endif  // ZIPFFT_STRIDED_PADDED_IO_HPP_