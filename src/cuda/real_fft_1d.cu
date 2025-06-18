#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "../include/block_io.hpp"
#include "../include/common.hpp"
#include "../include/dispatcher.hpp"


// --- r2c & c2r Kernel Definitions ---
/**
 * @brief Execute a 1-dimensional real-to-complex FFT using cuFFTDx.
 * 
 * @tparam FFT The cuFFTDx FFT struct
 * @tparam ComplexType The complex-valued data type for the FFT
 * @tparam ScalarType The real-valued (scalar component) data type for the FFT
 */
template<
    class    FFT,
    typename ComplexType = typename FFT::value_type,
    typename ScalarType  = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_r2c_1d_kernel(ScalarType* input_data, ComplexType* output_data) {
    using complex_type = ComplexType;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    // Then load data from global memory to registers
    const unsigned int local_fft_id = threadIdx.y;
    example::io<FFT>::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, output_data, local_fft_id);
}

/**
 * @brief Execute a 1-dimensional complex-to-real FFT using cuFFTDx.
 * 
 * @tparam FFT The cuFFTDx FFT struct
 * @tparam ComplexType The complex-valued data type for the FFT
 * @tparam ScalarType The real-valued (scalar component) data type for the FFT
 */
template<
    class    FFT,
    typename ComplexType = typename FFT::value_type,
    typename ScalarType  = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_c2r_1d_kernel(ComplexType* input_data, ScalarType* output_data) {
    using complex_type = ComplexType;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    // Then load data from global memory to registers
    const unsigned int local_fft_id = threadIdx.y;
    example::io<FFT>::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, output_data, local_fft_id);
}

// --- Unified Launcher Definition (for both r2c & c2r) ---
template<
    unsigned int Arch,
    typename     Input_T,
    typename     Output_T,
    unsigned int FFTSize,
    bool         IsForwardFFT>
inline void block_real_fft_1d_launcher(Input_T* input_data, Output_T* output_data) {
    using namespace cufftdx;

    // R2C and C2R specific data layout property
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using scalar_precision_type = std::conditional_t<IsForwardFFT, Input_T, Output_T>;

    // Conditional statements are used to determine the FFT traits
    // about direction and precision
    constexpr auto fft_direction = IsForwardFFT ? fft_direction::forward : fft_direction::inverse;
    constexpr auto fft_type = IsForwardFFT ? fft_type::r2c : fft_type::c2r;

    using FFT = decltype(
        Block() +
        Size<FFTSize>() +
        Type<fft_type>() +
        real_fft_options() +
        Direction<fft_direction>() +
        Precision<scalar_precision_type>() +
        ElementsPerThread<8>() +
        FFTsPerBlock<2>() +
        SM<Arch>()
    );

    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    // Compile-time branching to determine which FFT kernel to use
    if constexpr (IsForwardFFT) {
        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            block_fft_r2c_1d_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            FFT::shared_memory_size
        ));

        // Cast input data to cuFFTDx types
        scalar_type* input_data_t  = reinterpret_cast<scalar_type*>(input_data);
        complex_type* output_data_t = reinterpret_cast<complex_type*>(output_data);

        // Launch the kernel
        block_fft_r2c_1d_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(input_data_t, output_data_t);
    } else {
        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            block_fft_c2r_1d_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            FFT::shared_memory_size
        ));

        // Cast input data to cuFFTDx types
        complex_type* input_data_t  = reinterpret_cast<complex_type*>(input_data);
        scalar_type* output_data_t = reinterpret_cast<scalar_type*>(output_data);

        // Launch the kernel
        block_fft_c2r_1d_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(input_data_t, output_data_t);
    }

    // Ensure no errors afterwards
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Functor for Dispatcher ---
// This is necessary to grab the correct CUDA architecture on any system during
// runtime. Functions for all architectures in the dispatcher namespace are
// compiled.
template<
    unsigned int Arch,
    typename     Input_T_functor,
    typename     Output_T_functor,
    unsigned int FFTSize_functor,
    bool         IsForwardFFT_functor>
struct fft_dispatch_functor {
    void operator()(Input_T_functor* input_data, Output_T_functor* output_data) {
        block_real_fft_1d_launcher<Arch, Input_T_functor, Output_T_functor, FFTSize_functor, IsForwardFFT_functor>(input_data, output_data);
    }
};

// --- Public API Function Template Definition ---
template<typename Input_T, typename Output_T, unsigned int FFTSize, bool IsForwardFFT>
int block_real_fft_1d(Input_T* input_data, Output_T* output_data) {
    // Static assertions to ensure the correct types and sizes are used
    if constexpr (IsForwardFFT) {
#include "../generated/forward_fft_r2c_1d_asserts.inc"
    } else {
#include "../generated/inverse_fft_c2r_1d_asserts.inc"
    }

    // Call the modified dispatcher which determined the architecture
    int result = dispatcher::sm_runner_out_of_place<fft_dispatch_functor, Input_T, Output_T, FFTSize, IsForwardFFT>(input_data, output_data);

    // Runtime assertion that the dispatcher returned successfully
    if (result != 0) {
        std::cerr << "block_fft_r2c_1d: Error in dispatcher, returned code: " << result << std::endl;
        std::exit(result);
    }

    return result;
}


// --- Template Instantiations ---
// real-to-complex
#include "../generated/forward_fft_r2c_1d_impl.inc"

// complex-to-real
#include "../generated/inverse_fft_c2r_1d_impl.inc"
