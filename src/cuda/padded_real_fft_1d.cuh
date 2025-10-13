#include <cufftdx.hpp>

#include "../include/block_io.hpp"
#include "../include/common.hpp"
#include "../include/padded_io.hpp"

template <int SignalLength, class FFT,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_r2c_1d_kernel(
        ScalarType* input_data, ComplexType* output_data,
        typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Input is padded, use padded I/O utilities. Output is not padded
    using input_utils = example::io_padded<FFT, SignalLength>;
    using output_utils = example::io<FFT>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    // then load data from global memory to registers
    const unsigned int local_fft_id = threadIdx.y;
    input_utils::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    output_utils::store(thread_data, output_data, local_fft_id);
}

// template<
//     int      SignalLength,  // How many elements are actually in the
//     input_data class    FFT, typename ComplexType = typename FFT::value_type,
//     typename ScalarType  = typename ComplexType::value_type>
// __launch_bounds__(FFT::max_threads_per_block) __global__
// void padded_block_fft_c2r_1d_kernel(ComplexType* input_data, ScalarType*
// output_data) {
//     using complex_type = typename FFT::value_type;
//     using scalar_type  = typename complex_type::value_type;

//     // Determine weather padding is necessary based on the SignalLength
//     // and FFT size, then deciding which I/O namespace to use
//     // NOTE: This does not handle the case where SignalLength is larger!
//     constexpr bool needs_padding = (SignalLength !=
//     cufftdx::size_of<FFT>::value); using input_utils  =
//     std::conditional_t<needs_padding, example::io_padded<FFT, SignalLength>,
//     example::io<FFT>>; using output_utils = std::conditional_t<needs_padding,
//     example::io_padded<FFT, SignalLength>, example::io<FFT>>;

//     // Local array for thread
//     complex_type thread_data[FFT::storage_size];

//     // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
//     // then load data from global memory to registers
//     const unsigned int local_fft_id = threadIdx.y;
//     input_utils::load(input_data, thread_data, local_fft_id);

//     // Execute FFT
//     extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
//     FFT().execute(thread_data, shared_mem);

//     // Save results
//     output_utils::store(thread_data, output_data, local_fft_id);
// }

// --- Unified Launcher Definition (for both padded r2c & c2r) ---
template <unsigned int Arch, typename Input_T, typename Output_T,
          unsigned int SignalLength, unsigned int FFTSize, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
inline void padded_block_real_fft_1d_launcher(Input_T* input_data,
                                              Output_T* output_data) {
    using namespace cufftdx;

    // R2C and C2R specific data layout property
    using real_fft_options =
        RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using scalar_precision_type =
        std::conditional_t<IsForwardFFT, Input_T, Output_T>;

    // Conditional statements are used to determine the FFT traits
    // about direction and precision
    constexpr auto fft_direction =
        IsForwardFFT ? fft_direction::forward : fft_direction::inverse;
    constexpr auto fft_type = IsForwardFFT ? fft_type::r2c : fft_type::c2r;

    using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type>() +
                         real_fft_options() + Direction<fft_direction>() +
                         Precision<scalar_precision_type>() +
                         ElementsPerThread<elements_per_thread>() +
                         FFTsPerBlock<FFTs_per_block>() + SM<Arch>());

    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Compile-time branching to determine which FFT kernel to use
    if constexpr (IsForwardFFT) {
        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            padded_block_fft_r2c_1d_kernel<SignalLength, FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            FFT::shared_memory_size));

        // Cast input data to cuFFTDx types
        scalar_type* input_data_t =
        reinterpret_cast<scalar_type*>(input_data); complex_type*
        output_data_t =
            reinterpret_cast<complex_type*>(output_data);

        // create workspaces for FFT
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code);
        CUDA_CHECK_AND_EXIT(error_code);

        // Launch the kernel
        padded_block_fft_r2c_1d_kernel<SignalLength, FFT>
            <<<1, FFT::block_dim, FFT::shared_memory_size>>>(
                input_data_t, output_data_t, workspace);
    } else {
        // // Increase shared memory size, if needed
        // CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        //     padded_block_fft_c2r_1d_kernel<SignalLength, FFT>,
        //     cudaFuncAttributeMaxDynamicSharedMemorySize,
        //     FFT::shared_memory_size
        // ));

        // // Cast input data to cuFFTDx types
        // complex_type* input_data_t  =
        // reinterpret_cast<complex_type*>(input_data); scalar_type*
        // output_data_t = reinterpret_cast<scalar_type*>(output_data);

        // // Launch the kernel
        // padded_block_fft_c2r_1d_kernel<SignalLength, FFT><<<1,
        // FFT::block_dim, FFT::shared_memory_size>>>(input_data_t,
        // output_data_t);
    }

    // Ensure no errors afterwards
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
template <typename Input_T, typename Output_T, unsigned int SignalLength,
          unsigned int FFTSize, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
int padded_block_real_fft_1d(Input_T* input_data, Output_T* output_data) {
    // Throw if backward FFTs since they are not implemented yet
    static_assert(IsForwardFFT,
                  "Backward padded real FFTs (c2r) are not implemented yet.");

    auto arch = example::get_cuda_device_arch();

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: padded_block_real_fft_1d_launcher<800, Input_T, Output_T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: padded_block_real_fft_1d_launcher<860, Input_T, Output_T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: padded_block_real_fft_1d_launcher<870, Input_T, Output_T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: padded_block_real_fft_1d_launcher<890, Input_T, Output_T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: padded_block_real_fft_1d_launcher<900, Input_T, Output_T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        // Fallback: Architecture 1200 uses the 900 template as cuFFTDx does not yet
        // provide specific optimizations for newer architectures like Hopper/Blackwell.
        case 1200: padded_block_real_fft_1d_launcher<900, Input_T, Output_T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
        default:
            std::cerr << "Unsupported CUDA architecture: " << arch
                    << ". Supported architectures are 800, 860, 870, 890, "
                        "900, and 1200."
                    << std::endl;
            return -1;  // Error code for unsupported architecture
    }
    /* clang-format on */

    return 0;  // Success
}
