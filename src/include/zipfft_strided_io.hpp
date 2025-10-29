#ifndef ZIPFFT_IO_STRIDED_HPP_
#define ZIPFFT_IO_STRIDED_HPP_

#include <type_traits>

#include "zipfft_block_io.hpp"
#include "zipfft_common.hpp"
#include "zipfft_index_mapper.hpp"

namespace zipfft {

// Enhanced I/O functionality for strided memory access with explicit index mapper
template <class FFT, class InputIndexMapper, class OutputIndexMapper, unsigned int Stride = 1>
struct io_strided_with_layout : public io<FFT> {
    using io<FFT>::apparent_ffts_per_block;
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Type compatibility check
    template <typename RegisterType, typename MemoryType>
    static constexpr bool is_type_compatible() {
        return !CUFFTDX_STD::is_void_v<RegisterType> &&
               (sizeof(RegisterType) == sizeof(complex_type)) &&
               (alignof(RegisterType) == alignof(complex_type));
    }

    // Helper to compute offset based on layout dimensionality
    template <typename Layout, typename... Coords>
    static inline __device__ size_t compute_offset(Layout& layout, Coords... coords) {
        return layout(coords...);
    }

    // Load function using InputIndexMapper for strided addressing
    template <unsigned int Batches = Stride, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    load(const IOType* input, RegisterType* thread_data, unsigned int local_fft_id) {
        using input_t = typename FFT::input_type;

        // Instantiate the input index mapper
        InputIndexMapper input_layout;

        // Compute global FFT ID
        const unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        const unsigned int stride = FFT::stride;

        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);

        // For strided access, decompose global_fft_id into:
        // - outer_batch: which batch we're in (global_fft_id / Stride)
        // - inner_batch: which strided element within batch (global_fft_id % Stride)
        const unsigned int outer_batch = global_fft_id / Stride;
        const unsigned int inner_batch = global_fft_id % Stride;

        size_t offset;

        // Load data using index mapper for memory offset calculation
        for (unsigned int i = 0; i < FFT::input_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;

            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                if (elem_id < FFT::input_length && inner_batch < Batches) {
                    // Calculate flat element index
                    const unsigned int flat_elem = elem_id * inner_loop_limit + j;

                    offset = input_layout(inner_batch, flat_elem, outer_batch);

                    reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] =
                        reinterpret_cast<const IOType*>(input)[offset];
                }
            }
        }
    }

    // Store function using OutputIndexMapper for strided addressing
    template <unsigned int Batches = Stride, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    store(const RegisterType* thread_data, IOType* output, unsigned int local_fft_id) {
        using output_t = typename FFT::output_type;

        // Instantiate the output index mapper
        OutputIndexMapper output_layout;

        // Compute global FFT ID
        const unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        const unsigned int stride = FFT::stride;

        constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(IOType);

        // Decompose global_fft_id
        const unsigned int outer_batch = global_fft_id / Stride;
        const unsigned int inner_batch = global_fft_id % Stride;

        size_t offset;

        // Store data using index mapper for memory offset calculation
        for (unsigned int i = 0; i < FFT::output_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;

            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                if (elem_id < FFT::output_length && inner_batch < Batches) {
                    // Calculate flat element index
                    const unsigned int flat_elem = elem_id * inner_loop_limit + j;

                    offset = output_layout(inner_batch, flat_elem, outer_batch);

                    reinterpret_cast<IOType*>(output)[offset] =
                        reinterpret_cast<const IOType*>(thread_data)[i * inner_loop_limit + j];
                }
            }
        }
    }
};

// Backward-compatible io_strided struct that generates appropriate index mappers
template <class FFT, unsigned int Stride>
struct io_strided : public io<FFT> {
    using io<FFT>::apparent_ffts_per_block;

    // Define strided memory layouts using 3-level index_mapper
    // For input of shape (outer_batch, FFT::input_length, Stride):
    // Level 1: Stride elements with stride 1 (innermost, contiguous columns)
    // Level 2: FFT::input_length elements strided by Stride (rows)
    // Level 3: outer_batch strided by FFT::input_length * Stride

    using input_layout =
        index_mapper<int_pair<Stride, 1>,                     // columns
                     int_pair<FFT::input_length, Stride>,     // rows
                     int_pair<1, FFT::input_length * Stride>  // batches (placeholder size=1)
                     >;

    using output_layout = index_mapper<int_pair<Stride, 1>, int_pair<FFT::output_length, Stride>,
                                       int_pair<1, FFT::output_length * Stride> >;

    // Delegate to io_strided_with_layout with Stride template parameter
    using io_impl = io_strided_with_layout<FFT, input_layout, output_layout, Stride>;

    // Forward type compatibility check
    template <typename RegisterType, typename MemoryType>
    static constexpr bool is_type_compatible() {
        return io_impl::template is_type_compatible<RegisterType, MemoryType>();
    }

    // Forward load/store to implementation with strided layouts
    template <unsigned int Batches = Stride, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    load(const IOType* input, RegisterType* thread_data, unsigned int local_fft_id) {
        io_impl::template load<Batches>(input, thread_data, local_fft_id);
    }

    template <unsigned int Batches = Stride, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    store(const RegisterType* thread_data, IOType* output, unsigned int local_fft_id) {
        io_impl::template store<Batches>(thread_data, output, local_fft_id);
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
        unsigned int outer_batch_index = input_outer_batch_index(local_fft_id);
        unsigned int inner_batch_index = input_inner_batch_index(local_fft_id);
        return (outer_batch_index * FFT::input_length * Stride) + inner_batch_index;
    }

    static inline __device__ unsigned int output_batch_offset(unsigned int local_fft_id) {
        unsigned int outer_batch_index = output_outer_batch_index(local_fft_id);
        unsigned int inner_batch_index = output_inner_batch_index(local_fft_id);
        return (outer_batch_index * FFT::output_length * Stride) + inner_batch_index;
    }
};

}  // namespace zipfft

#endif  // ZIPFFT_IO_STRIDED_HPP_