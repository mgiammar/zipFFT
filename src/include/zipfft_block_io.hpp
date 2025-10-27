#ifndef ZIPFFT_BLOCK_IO_HPP_
#define ZIPFFT_BLOCK_IO_HPP_

#include <type_traits>

#include "fp16_common.hpp"
#include "zipfft_common.hpp"
#include "zipfft_index_mapper.hpp"

namespace zipfft {
// Changes layout of complex<__half2> value from ((Real, Imag), (Real, Imag)) layout to
// ((Real, Real), (Imag, Imag)) layout.
__device__ __host__ __forceinline__ cufftdx::complex<__half2> to_rrii(
    cufftdx::complex<__half2> riri) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    cufftdx::complex<__half2> rrii(__lows2half2(riri.x, riri.y), __highs2half2(riri.x, riri.y));
#else
    cufftdx::complex<__half2> rrii(__half2{riri.x.x, riri.y.x}, __half2{riri.x.y, riri.y.y});
#endif
    return rrii;
}

// Converts to __half complex values to complex<__half2> in ((Real, Real), (Imag, Imag)) layout.
__device__ __host__ __forceinline__ cufftdx::complex<__half2> to_rrii(__half2 ri1, __half2 ri2) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    cufftdx::complex<__half2> rrii(__lows2half2(ri1, ri2), __highs2half2(ri1, ri2));
#else
    cufftdx::complex<__half2> rrii(__half2{ri1.x, ri2.x}, __half2{ri1.y, ri2.y});
#endif
    return rrii;
}

// Changes layout of complex<__half2> value from ((Real, Real), (Imag, Imag)) layout to
// ((Real, Imag), (Real, Imag)) layout.
__device__ __host__ __forceinline__ cufftdx::complex<__half2> to_riri(
    cufftdx::complex<__half2> rrii) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    cufftdx::complex<__half2> riri(__lows2half2(rrii.x, rrii.y), __highs2half2(rrii.x, rrii.y));
#else
    cufftdx::complex<__half2> riri(__half2{rrii.x.x, rrii.y.x}, __half2{rrii.x.y, rrii.y.y});
#endif
    return riri;
}

// Return the first half complex number (as __half2) from complex<__half2> value with
// ((Real, Real), (Imag, Imag)) layout.
// Example: for rrii equal to ((1,2), (3,4)), it return __half2 (1, 3).
__device__ __host__ __forceinline__ __half2 to_ri1(cufftdx::complex<__half2> rrii) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    return __lows2half2(rrii.x, rrii.y);
#else
    return __half2{rrii.x.x, rrii.y.x};
#endif
}

// Return the second half complex number (as __half2) from complex<__half2> value with
// ((Real, Real), (Imag, Imag)) layout.
// Example: for rrii equal to ((1,2), (3,4)), it return __half2 (2, 4).
__device__ __host__ __forceinline__ __half2 to_ri2(cufftdx::complex<__half2> rrii) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    return __highs2half2(rrii.x, rrii.y);
#else
    return __half2{rrii.x.y, rrii.y.y};
#endif
}

// Helper functions to conditionally convert between RIRI and RRII layouts based
// on template parameter (for mixed precision with __half2 types).
namespace __io {
template <bool InRRIILayout = false>
inline __device__ cufftdx::complex<__half2> convert_to_rrii(
    const cufftdx::complex<__half2>& value) {
    return to_rrii(value);
}
template <>
inline __device__ cufftdx::complex<__half2> convert_to_rrii<true>(
    const cufftdx::complex<__half2>& value) {
    return value;
}
template <bool InRIRILayout = false>
inline __device__ cufftdx::complex<__half2> convert_to_riri(
    const cufftdx::complex<__half2>& value) {
    return to_riri(value);
}
template <>
inline __device__ cufftdx::complex<__half2> convert_to_riri<true>(
    const cufftdx::complex<__half2>& value) {
    return value;
}
}  // namespace __io

// Enhanced I/O functionality with index mapper template parameters
// This allows FFT implementation to define custom memory layouts
template <class FFT, class InputIndexMapper, class OutputIndexMapper>
struct io_with_layout {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Implicit batching offset calculation for input and output
    // Implicit batching is currently mandatory for __half precision, and it forces two
    // batches of data to be put together into a single complex __half2 value. This makes
    // it so a "single" batch of complex __half2 values in reality contains 2 batches of
    // complex __half values. Full reference can be found in documentation:
    // https://docs.nvidia.com/cuda/cufftdx/api/methods.html#half-precision-implicit-batching
    //
    // FFT::implicit_type_batching is 2 for __half and 1 for other types
    // and effectively scales the number of FFTs per block accordingly. Common
    // between all types I/O operations in zipFFT.
    static constexpr unsigned int apparent_ffts_per_block =
        FFT::ffts_per_block / FFT::implicit_type_batching;

    // Check if the register type and FFT structure are type compatible
    template <typename RegisterType, typename MemoryType>
    static constexpr bool is_type_compatible() {
        // TODO: Check why using 'complex_type' rather than 'MemoryType' here?
        return !CUFFTDX_STD::is_void_v<RegisterType> &&
               (sizeof(RegisterType) == sizeof(complex_type)) &&
               (alignof(RegisterType) == alignof(complex_type));
    }

    // Load function using InputIndexMapper for addressing
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

        // Load data using index mapper for memory offset calculation
        for (unsigned int i = 0; i < FFT::input_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;
            if (elem_id < FFT::input_length) {
                // Use index mapper: (element_index, batch_index)
                const size_t offset = input_layout(elem_id, global_fft_id);
                
                if constexpr (needs_half2_format_conversion) {
                    reinterpret_cast<input_t*>(thread_data)[i] =
                        __io::convert_to_rrii<InputInRRIILayout>(
                            reinterpret_cast<const input_t*>(input)[offset]);
                } else {
                    reinterpret_cast<input_t*>(thread_data)[i] =
                        reinterpret_cast<const input_t*>(input)[offset];
                }
            }
        }
    }

    // Store function using OutputIndexMapper for addressing
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

        // Store data using index mapper for memory offset calculation
        for (int i = 0; i < FFT::output_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;
            if (elem_id < FFT::output_length) {
                // Use index mapper: (element_index, batch_index)
                const size_t offset = output_layout(elem_id, global_fft_id);
                
                if constexpr (needs_half2_format_conversion) {
                    reinterpret_cast<output_t*>(output)[offset] =
                        __io::convert_to_riri<OutputInRRIILayout>(
                            reinterpret_cast<const output_t*>(thread_data)[i]);
                } else {
                    reinterpret_cast<output_t*>(output)[offset] =
                        reinterpret_cast<const output_t*>(thread_data)[i];
                }
            }
        }
    }
};

// Backward-compatible io struct that uses default contiguous layout
template <class FFT>
struct io {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    static constexpr unsigned int apparent_ffts_per_block =
        FFT::ffts_per_block / FFT::implicit_type_batching;

    // Define default contiguous memory layouts
    // Layout: (element_index, batch_index) for input
    using default_input_layout = index_mapper<
        int_pair<FFT::input_length, 1>,  // elements contiguous
        int_pair<1, FFT::input_length>    // batches strided by input_length (placeholder size=1)
    >;

    // Layout: (element_index, batch_index) for output
    using default_output_layout = index_mapper<
        int_pair<FFT::output_length, 1>,  // elements contiguous
        int_pair<1, FFT::output_length>    // batches strided by output_length (placeholder size=1)
    >;

    // Delegate to io_with_layout with default layouts
    using io_impl = io_with_layout<FFT, default_input_layout, default_output_layout>;

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
    
    // Legacy offset functions for compatibility
    static inline __device__ unsigned int input_batch_offset(unsigned int local_fft_id) {
        unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        return FFT::input_length * global_fft_id;
    }

    static inline __device__ unsigned int output_batch_offset(unsigned int local_fft_id) {
        unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
        return FFT::output_length * global_fft_id;
    }
};

}  // namespace zipfft

#endif  // ZIPFFT_BLOCK_IO_HPP_