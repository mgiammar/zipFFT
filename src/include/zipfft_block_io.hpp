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

// Helper trait to detect the number of dimensions in an index_mapper
template <typename T>
struct index_mapper_dims;

template <typename ThisDim, typename... NextDims>
struct index_mapper_dims<index_mapper<ThisDim, NextDims...>> {
    static constexpr size_t value = 1 + sizeof...(NextDims);
};

}  // namespace __io

// Enhanced I/O functionality with index mapper template parameters
// This allows FFT implementation to define custom memory layouts
// Automatically handles 2D (element, fft) or 3D (element, fft, batch) layouts
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

    // Detect dimensionality of input and output mappers
    static constexpr size_t input_mapper_dims = __io::index_mapper_dims<InputIndexMapper>::value;
    static constexpr size_t output_mapper_dims = __io::index_mapper_dims<OutputIndexMapper>::value;

    // Check if the register type and FFT structure are type compatible
    template <typename RegisterType, typename MemoryType>
    static constexpr bool is_type_compatible() {
        // TODO: Check why using 'complex_type' rather than 'MemoryType' here?
        return !CUFFTDX_STD::is_void_v<RegisterType> &&
               (sizeof(RegisterType) == sizeof(complex_type)) &&
               (alignof(RegisterType) == alignof(complex_type));
    }

    // Load function using InputIndexMapper for addressing
    // Supports both 2D (element, fft) and 3D (element, fft, batch) mappers
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

        // For 3D layout (element, fft, batch):
        // The index_mapper expects (element_id, fft_id, batch_id)
        // We need to decompose global_fft_id into fft_id and batch_id

        // Dimension 1 is the FFT dimension (number of FFTs per batch)
        // Dimension 2 is the batch dimension
        const unsigned int fft_id = global_fft_id % dim_size_v<1, InputIndexMapper>;
        const unsigned int batch_id = global_fft_id / dim_size_v<1, InputIndexMapper>;

        // Load data using index mapper for memory offset calculation
        for (unsigned int i = 0; i < FFT::input_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;
            if (elem_id < FFT::input_length) {
                const size_t offset = input_layout(elem_id, fft_id, batch_id);

                // /// DEBUGGING: Print the computed offset and values being loaded
                // const int offset_int = static_cast<int>(offset);
                // if (blockIdx.x == 0 && threadIdx.y == 0) {
                //     float2 val = reinterpret_cast<const float2*>(input)[offset];
                //     printf("Load Offset Calculation - global_fft_id: %u, fft_id: %u, "
                //            "batch_id: %u, offset: %u\n",
                //            global_fft_id, fft_id, batch_id, offset_int);
                //     printf("Loaded Value at offset %u: (%f, %f)\n", offset_int, val.x, val.y);
                // }

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
    // Supports both 2D (element, fft) and 3D (element, fft, batch) mappers
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

        // For 3D layout (element, fft, batch):
        // The index_mapper expects (element_id, fft_id, batch_id)
        // We need to decompose global_fft_id into fft_id and batch_id

        // Dimension 1 is the FFT dimension (number of FFTs per batch)
        // Dimension 2 is the batch dimension
        const unsigned int fft_id = global_fft_id % dim_size_v<1, OutputIndexMapper>;
        const unsigned int batch_id = global_fft_id / dim_size_v<1, OutputIndexMapper>;

        // Store data using index mapper for memory offset calculation
        for (int i = 0; i < FFT::output_ept; ++i) {
            const unsigned int elem_id = threadIdx.x + i * stride;
            if (elem_id < FFT::output_length) {
                const size_t offset = output_layout(elem_id, fft_id, batch_id);

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

    static constexpr bool this_fft_is_folded =
        cufftdx::real_fft_mode_of<FFT>::value == cufftdx::real_mode::folded;

    template <typename RegType, typename MemType>
    static constexpr bool is_type_compatible() {
        return !CUFFTDX_STD::is_void_v<RegType> && (sizeof(RegType) == sizeof(complex_type)) &&
               (alignof(RegType) == alignof(complex_type));
    }

    static inline __device__ unsigned int input_batch_offset(unsigned int local_fft_id) {
        // Implicit batching is currently mandatory for __half precision, and it forces two
        // batches of data to be put together into a single complex __half2 value. This makes
        // it so a "single" batch of complex __half2 values in reality contains 2 batches of
        // complex __half values. Full reference can be found in documentation:
        // https://docs.nvidia.com/cuda/cufftdx/api/methods.html#half-precision-implicit-batching
        unsigned int global_fft_id =
            blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
        return FFT::input_length * global_fft_id;
    }

    static inline __device__ unsigned int output_batch_offset(unsigned int local_fft_id) {
        // See note regarding implicit batching in input_batch_offset
        unsigned int global_fft_id =
            blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
        // If Fold Optimization is enabled, the real values are packed together
        // into complex values which decreases the effective size twofold
        return FFT::output_length * global_fft_id;
    }

    template <unsigned int EPT, typename DataType>
    static inline __device__ void copy(const DataType* source, DataType* target, unsigned int n) {
        unsigned int stride = blockDim.x * blockDim.y;
        unsigned int index = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = 0; i < EPT; i++) {
            if (index < n) {
                target[index] = source[index];
            }
            index += stride;
        }
    }

    template <class DataType>
    static inline __device__ void load_to_smem(const DataType* global, unsigned char* shared) {
        using input_t = typename FFT::input_type;
        copy<FFT::input_ept>(reinterpret_cast<const input_t*>(global),
                             reinterpret_cast<input_t*>(shared), blockDim.y * FFT::input_length);
        __syncthreads();
    }

    template <class DataType>
    static inline __device__ void store_from_smem(const unsigned char* shared, DataType* global) {
        __syncthreads();
        using output_t = typename FFT::output_type;
        copy<FFT::output_ept>(reinterpret_cast<const output_t*>(shared),
                              reinterpret_cast<output_t*>(global), blockDim.y * FFT::output_length);
    }

    // If InputInRRIILayout is false, then function assumes that values in input are in RIRI
    // layout, and before loading them to thread_data they are converted to RRII layout.
    // Otherwise, if InputInRRIILayout is true, then function assumes values in input are in RRII
    // layout, and don't need to be converted before loading to thread_data.
    template <bool InputInRRIILayout = false, typename RegisterType, typename IOType,
              class LoadOp = zipfft::identity>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    load(const IOType* input, RegisterType* thread_data, unsigned int local_fft_id,
         LoadOp op = {}) {
        static constexpr bool needs_half2_format_conversion =
            cufftdx::type_of<FFT>::value != cufftdx::fft_type::r2c &&
            std::is_same_v<IOType, cufftdx::detail::complex<__half2>>;
        using input_t = typename FFT::input_type;

        // Calculate global offset of FFT batch
        const unsigned int offset = input_batch_offset(local_fft_id);
        // Get stride, this shows how elements from batch should be split between threads
        const unsigned int stride = FFT::stride;
        unsigned int index = offset + threadIdx.x;
        for (unsigned int i = 0; i < FFT::input_ept; i++) {
            if ((i * stride + threadIdx.x) < FFT::input_length) {
                if constexpr (needs_half2_format_conversion) {
                    reinterpret_cast<input_t*>(thread_data)[i] =
                        op(__io::convert_to_rrii<InputInRRIILayout>(
                            reinterpret_cast<const input_t*>(input)[index]));
                } else {
                    reinterpret_cast<input_t*>(thread_data)[i] =
                        op(reinterpret_cast<const input_t*>(input)[index]);
                }
                index += stride;
            }
        }
    }

    // Function assumes that values in thread_data are in RRII layout.
    // If OutputInRRIILayout is false, values are saved into output in RIRI layout; otherwise - in
    // RRII.
    template <bool OutputInRRIILayout = false, typename RegisterType, typename IOType,
              class StoreOp = zipfft::identity>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    store(const RegisterType* thread_data, IOType* output, unsigned int local_fft_id,
          StoreOp op = {}) {
        static constexpr bool needs_half2_format_conversion =
            cufftdx::type_of<FFT>::value != cufftdx::fft_type::c2r &&
            std::is_same_v<IOType, cufftdx::detail::complex<__half2>>;
        using output_t = typename FFT::output_type;

        const unsigned int offset = output_batch_offset(local_fft_id);
        const unsigned int stride = FFT::stride;
        unsigned int index = offset + threadIdx.x;

        for (int i = 0; i < FFT::output_ept; ++i) {
            if ((i * stride + threadIdx.x) < FFT::output_length) {
                if constexpr (needs_half2_format_conversion) {
                    reinterpret_cast<output_t*>(output)[index] =
                        op(__io::convert_to_riri<OutputInRRIILayout>(
                            reinterpret_cast<const output_t*>(thread_data)[i]));
                } else {
                    reinterpret_cast<output_t*>(output)[index] =
                        op(reinterpret_cast<const output_t*>(thread_data)[i]);
                }
                index += stride;
            }
        }
    }
};

}  // namespace zipfft

#endif  // ZIPFFT_BLOCK_IO_HPP_