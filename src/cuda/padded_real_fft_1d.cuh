#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_index_mapper.hpp"
#include "../include/zipfft_padded_io.hpp"

// --- Forward R2C Kernel with Index Mappers for Padded Input ---
template <class FFT, class InputLayout, class OutputLayout, unsigned int SignalLength,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_r2c_1d_kernel_with_layout(ScalarType* input_data, 
                                                     ComplexType* output_data,
                                                     typename FFT::workspace_type workspace) {
    using complex_type = ComplexType;
    using scalar_type = ScalarType;

    // Input is padded with custom layout, output uses standard layout
    using input_io = zipfft::io_padded_with_layout<FFT, InputLayout, InputLayout, SignalLength>;
    using output_io = zipfft::io_with_layout<FFT, OutputLayout, OutputLayout>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using padded input layout (zero-pads to FFT size)
    input_io::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Store using output layout (full FFT output)
    output_io::store(thread_data, output_data, local_fft_id);
}

// --- Inverse C2R Kernel with Index Mappers for Padded Output ---
template <class FFT, class InputLayout, class OutputLayout, unsigned int SignalLength,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_c2r_1d_kernel_with_layout(ComplexType* input_data, 
                                                     ScalarType* output_data) {
    using complex_type = ComplexType;
    using scalar_type = ScalarType;

    // Input uses standard layout, output is truncated with custom layout
    using input_io = zipfft::io_with_layout<FFT, InputLayout, InputLayout>;
    using output_io = zipfft::io_padded_with_layout<FFT, OutputLayout, OutputLayout, SignalLength>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using input layout (full complex input)
    input_io::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Store using padded output layout (truncates to SignalLength)
    output_io::store(thread_data, output_data, local_fft_id);
}

// --- Unified Launcher Definition (for both padded r2c & c2r) ---
template <unsigned int Arch, typename Input_T, typename Output_T, unsigned int SignalLength,
          unsigned int FFTSize, bool IsForwardFFT, unsigned int elements_per_thread,
          unsigned int FFTs_per_block>
inline void padded_block_real_fft_1d_launcher(Input_T* input_data, Output_T* output_data) {
    using namespace cufftdx;

    // R2C and C2R specific data layout property
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using scalar_precision_type = std::conditional_t<IsForwardFFT, Input_T, Output_T>;

    // Conditional statements are used to determine the FFT traits
    constexpr auto fft_direction = IsForwardFFT ? fft_direction::forward : fft_direction::inverse;
    constexpr auto fft_type = IsForwardFFT ? fft_type::r2c : fft_type::c2r;

    using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type>() + real_fft_options() +
                         Direction<fft_direction>() + Precision<scalar_precision_type>() +
                         ElementsPerThread<elements_per_thread>() + FFTsPerBlock<FFTs_per_block>() +
                         SM<Arch>());

    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Compile-time branching to determine which FFT kernel to use
    if constexpr (IsForwardFFT) {
        // Forward R2C: padded real input -> full complex output
        // Input layout: (batch, SignalLength) where SignalLength < FFT::input_length
        using InputLayout = zipfft::index_mapper<
            zipfft::int_pair<SignalLength, 1>,     // elements contiguous
            zipfft::int_pair<1, SignalLength>,     // batches strided by SignalLength
            zipfft::int_pair<0, 0>                 // dummy dimension
            >;

        // Output layout: (batch, FFT::output_length) - full complex output
        using OutputLayout = zipfft::index_mapper<
            zipfft::int_pair<FFT::output_length, 1>,   // elements contiguous
            zipfft::int_pair<1, FFT::output_length>,   // batches strided by output_length
            zipfft::int_pair<0, 0>                     // dummy dimension
            >;

        auto kernel_ptr = padded_block_fft_r2c_1d_kernel_with_layout<FFT, InputLayout, OutputLayout, SignalLength>;

        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

        // Cast input data to cuFFTDx types
        scalar_type* input_data_t = reinterpret_cast<scalar_type*>(input_data);
        complex_type* output_data_t = reinterpret_cast<complex_type*>(output_data);

        // Create workspace for FFT
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code);
        CUDA_CHECK_AND_EXIT(error_code);

        // Launch the kernel
        kernel_ptr<<<1, FFT::block_dim, FFT::shared_memory_size>>>(input_data_t, output_data_t, workspace);
    } else {
        // Inverse C2R: full complex input -> padded real output
        // Input layout: (batch, FFT::input_length) - full complex input
        using InputLayout = zipfft::index_mapper<
            zipfft::int_pair<FFT::input_length, 1>,    // elements contiguous
            zipfft::int_pair<1, FFT::input_length>,    // batches strided by input_length
            zipfft::int_pair<0, 0>                     // dummy dimension
            >;

        // Output layout: (batch, SignalLength) where SignalLength < FFT::output_length
        using OutputLayout = zipfft::index_mapper<
            zipfft::int_pair<SignalLength, 1>,     // elements contiguous
            zipfft::int_pair<1, SignalLength>,     // batches strided by SignalLength
            zipfft::int_pair<0, 0>                 // dummy dimension
            >;

        auto kernel_ptr = padded_block_fft_c2r_1d_kernel_with_layout<FFT, InputLayout, OutputLayout, SignalLength>;

        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

        // Cast input data to cuFFTDx types
        complex_type* input_data_t = reinterpret_cast<complex_type*>(input_data);
        scalar_type* output_data_t = reinterpret_cast<scalar_type*>(output_data);

        // Launch the kernel
        kernel_ptr<<<1, FFT::block_dim, FFT::shared_memory_size>>>(input_data_t, output_data_t);
    }

    // Ensure no errors afterwards
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
/**
 * @brief Performs a 1D real-to-complex or complex-to-real FFT with padding using cuFFTDx.
 *
 * For R2C (forward): Reads SignalLength real values, zero-pads to FFTSize, performs FFT.
 * For C2R (inverse): Performs FFT, truncates output to SignalLength real values.
 *
 * @tparam Input_T Input data type (float for R2C, float2 for C2R).
 * @tparam Output_T Output data type (float2 for R2C, float for C2R).
 * @tparam SignalLength Actual signal length in memory (< FFTSize).
 * @tparam FFTSize Size of FFT to perform (must be >= SignalLength).
 * @tparam IsForwardFFT true for R2C, false for C2R.
 * @tparam elements_per_thread Elements processed per thread.
 * @tparam FFTs_per_block FFTs computed per thread block (effective batch size).
 *
 * @param input_data Pointer to device input array.
 * @param output_data Pointer to device output array.
 *
 * @return int Returns 0 on success, or error code on failure.
 */
template <typename Input_T, typename Output_T, unsigned int SignalLength, unsigned int FFTSize,
          bool IsForwardFFT, unsigned int elements_per_thread, unsigned int FFTs_per_block>
int padded_block_real_fft_1d(Input_T* input_data, Output_T* output_data) {
    auto arch = zipfft::get_cuda_device_arch();

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
