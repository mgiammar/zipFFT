#ifndef ZIPFFT_BLOCK_IO_HPP_
#define ZIPFFT_BLOCK_IO_HPP_

#include <type_traits>

#include "fp16_common.hpp"
#include "zipfft_common.hpp"

namespace zipfft {
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

// Basic I/O functionality for load/store operations on contigious memory for
// cuFFTDx execution. Handles real-to-complex and complex-to-real load/store,
// but requires contigious memory (e.g., values {r1, r2} <--> c1 require the
// scalar components to reside next to each other in memory and the complex type
// is half the size of the real type).
template <class FFT>
struct io {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Check if the register type and FFT structure are type compatible
    template <typename RegisterType, typename MemoryType>
    static constexpr bool is_type_compatible() {
        return !CUFFTDX_STD::is_void_v<RegisterType> &&
               (sizeof(RegisterType) == sizeof(complex_type)) &&
               (alignof(RegisterType) == alignof(complex_type));
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

    // Removed: CUDALibrarySample functions for load/store to/from shared memory

    // If InputInRRIILayout is false (and 'input' is of __half2 type), then function assumes that
    // values in 'input' are in RIRI layout and need converted to RRII layout before loading them
    // into 'thread_data'. Otherwise, if InputInRRIILayout is true, then function assumes values in
    // 'input' are in RRII layout and don't need to be converted before loading to 'thread_data'.
    //
    // NOTE: I've excluded the templated LoadOp from this function for simplicity.
    template <bool InputInRRIILayout = false, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    load(const IOType* input, RegisterType* thread_data, unsigned int local_fft_id) {
        static constexpr bool needs_half2_format_conversion =
            cufftdx::type_of<FFT>::value != cufftdx::fft_type::r2c &&
            std::is_same_v<IOType, cufftdx::detail::complex<__half2>>;

        using input_t = typename FFT::input_type;

        // Calculate the global memory offset of the FFT batch (this includes
        // which FFT is being executed through blockIdx.x and the possibly
        // batched executing within the block by threadIdx.y / local_fft_id).
        const unsigned int offset = input_batch_offset(local_fft_id);

        // Stride and other loop constants defining how data should be accessed
        // within the loop
        const unsigned int stride = FFT::stride;
        unsigned int index = offset + threadIdx.x;

        // Stride loop for coalesced memory access with a conditional statement
        // to do __half2 conversion, if necessary.
        for (unsigned int i = 0; i < FFT::input_ept; ++i) {
            if ((i * stride + threadIdx.x) < FFT::input_length) {
                if constexpr (needs_half2_format_conversion) {
                    reinterpret_cast<input_t*>(thread_data)[i] =
                        __io::convert_to_rrii<InputInRRIILayout>(
                            reinterpret_cast<const input_t*>(input)[index]);
                } else {
                    reinterpret_cast<input_t*>(thread_data)[i] =
                        reinterpret_cast<const input_t*>(input)[index];
                }
                index += stride;
            }
        }
    }

    // If OutputInRRIILayout is false (and 'input' is of __half2 type), then function assumes that
    // values in 'input' are in RIRI layout and need converted to RRII layout before loading them
    // into 'thread_data'. Otherwise, if OutputInRRIILayout is true, then function assumes values in
    // 'input' are in RRII layout and don't need to be converted before loading to 'thread_data'.
    //
    // NOTE: I've excluded the templated StoreOp from this function for simplicity.
    template <bool OutputInRRIILayout = false, typename RegisterType, typename IOType>
    static inline __device__ CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
    store(const RegisterType* thread_data, IOType* output, unsigned int local_fft_id) {
        static constexpr bool needs_half2_format_conversion =
            cufftdx::type_of<FFT>::value != cufftdx::fft_type::c2r &&
            std::is_same_v<IOType, cufftdx::detail::complex<__half2>>;

        using output_t = typename FFT::output_type;

        // Calculate the global memory offset of the FFT batch (this includes
        // which FFT is being executed through blockIdx.x and the possibly
        // batched executing within the block by threadIdx.y / local_fft_id).
        const unsigned int offset = output_batch_offset(local_fft_id);

        // Stride and other loop constants defining how data should be accessed
        // within the loop
        const unsigned int stride = FFT::stride;
        unsigned int index = offset + threadIdx.x;

        for (int i = 0; i < FFT::output_ept; ++i) {
            if ((i * stride + threadIdx.x) < FFT::output_length) {
                if constexpr (needs_half2_format_conversion) {
                    reinterpret_cast<output_t*>(output)[index] =
                        __io::convert_to_riri<OutputInRRIILayout>(
                            reinterpret_cast<const output_t*>(thread_data)[i]);
                } else {
                    reinterpret_cast<output_t*>(output)[index] =
                        reinterpret_cast<const output_t*>(thread_data)[i];
                }
                index += stride;
            }
        }
    }  // struct io

}  // namespace zipfft

#endif  // ZIPFFT_BLOCK_IO_HPP_