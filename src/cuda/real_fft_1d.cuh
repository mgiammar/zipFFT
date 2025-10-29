#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_index_mapper.hpp"

// --- r2c Kernel with Explicit Index Mappers ---
template <class FFT, class InputLayout, class OutputLayout,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_fft_r2c_1d_kernel_with_layout(ScalarType* input_data, ComplexType* output_data) {
    using complex_type = ComplexType;
    using io_type = zipfft::io_with_layout<FFT, InputLayout, OutputLayout>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using custom input layout
    io_type::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Store using custom output layout
    io_type::store(thread_data, output_data, local_fft_id);
}

// --- c2r Kernel with Explicit Index Mappers ---
template <class FFT, class InputLayout, class OutputLayout,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_fft_c2r_1d_kernel_with_layout(ComplexType* input_data, ScalarType* output_data) {
    using complex_type = ComplexType;
    using io_type = zipfft::io_with_layout<FFT, InputLayout, OutputLayout>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using custom input layout
    io_type::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Store using custom output layout
    io_type::store(thread_data, output_data, local_fft_id);
}

// --- Unified Launcher Definition (for both r2c & c2r) ---
template <unsigned int Arch, typename Input_T, typename Output_T, unsigned int FFTSize,
          bool IsForwardFFT, unsigned int elements_per_thread, unsigned int FFTs_per_block>
inline void block_real_fft_1d_launcher(Input_T* input_data, Output_T* output_data) {
    using namespace cufftdx;

    // R2C and C2R specific data layout property
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using scalar_precision_type = std::conditional_t<IsForwardFFT, Input_T, Output_T>;

    constexpr auto fft_direction = IsForwardFFT ? fft_direction::forward : fft_direction::inverse;
    constexpr auto fft_type = IsForwardFFT ? fft_type::r2c : fft_type::c2r;

    using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type>() + real_fft_options() +
                         Direction<fft_direction>() + Precision<scalar_precision_type>() +
                         ElementsPerThread<elements_per_thread>() + FFTsPerBlock<FFTs_per_block>() +
                         SM<Arch>());

    // Define explicit index mappers
    // NOTE: Batch size here is the number of FFTs per block, used only for
    // testing purposes when indexing non-singleton FFTs per block.
    using InputLayout = zipfft::index_mapper<
        zipfft::int_pair<FFT::input_length, 1>,               // elements contiguous (columns)
        zipfft::int_pair<FFTs_per_block, FFT::input_length>,  // batches strided by input_length
                                                              // (rows)
        zipfft::int_pair<0, 0>  // dummy batch dimension for compatibility
        >;

    using OutputLayout = zipfft::index_mapper<
        zipfft::int_pair<FFT::output_length, 1>,               // elements contiguous (columns)
        zipfft::int_pair<FFTs_per_block, FFT::output_length>,  // batches strided by output_length
                                                               // (rows)
        zipfft::int_pair<0, 0>  // dummy batch dimension for compatibility
        >;

    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Compile-time branching to determine which FFT kernel to use
    if constexpr (IsForwardFFT) {
        auto kernel_ptr = block_fft_r2c_1d_kernel_with_layout<FFT, InputLayout, OutputLayout>;

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

        scalar_type* input_data_t = reinterpret_cast<scalar_type*>(input_data);
        complex_type* output_data_t = reinterpret_cast<complex_type*>(output_data);

        kernel_ptr<<<1, FFT::block_dim, FFT::shared_memory_size>>>(input_data_t, output_data_t);
    } else {
        auto kernel_ptr = block_fft_c2r_1d_kernel_with_layout<FFT, InputLayout, OutputLayout>;

        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

        complex_type* input_data_t = reinterpret_cast<complex_type*>(input_data);
        scalar_type* output_data_t = reinterpret_cast<scalar_type*>(output_data);

        kernel_ptr<<<1, FFT::block_dim, FFT::shared_memory_size>>>(input_data_t, output_data_t);
    }

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
template <typename Input_T, typename Output_T, unsigned int FFTSize, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
int block_real_fft_1d(Input_T* input_data, Output_T* output_data) {
    auto arch = zipfft::get_cuda_device_arch();

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: block_real_fft_1d_launcher<800, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: block_real_fft_1d_launcher<860, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: block_real_fft_1d_launcher<870, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: block_real_fft_1d_launcher<890, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: block_real_fft_1d_launcher<900, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
        // Fallback: Architecture 1200 uses the 900 template as cuFFTDx does not yet
        // provide specific optimizations for newer architectures like Hopper/Blackwell.
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        case 1200: block_real_fft_1d_launcher<900, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data); break;
#endif
        default:
            std::cerr << "Unsupported CUDA architecture: " << arch
                      << ". Supported architectures are 800, 860, 870, 890, "
                         "900, and 1200."
                      << std::endl;
            return -1;
    }
    /* clang-format on */

    return 0;
}
