#include <cufftdx.hpp>

#include "../include/zipfft_common.hpp"
#include "../include/zipfft_strided_io.hpp"

// --- Kernel Definition ---
template <class FFT, unsigned int Stride>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void strided_block_fft_c2c_1d_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;
    using io_strided = zipfft::io_strided<FFT, Stride>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;
    io_strided::load_strided(data, thread_data, local_fft_id);

    // Execute the FFT within shared memory
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results back to global memory
    io_strided::store_strided(thread_data, data, local_fft_id);
}

// --- Launcher Definition ---
template <unsigned int Arch, typename T, unsigned int FFTSize, unsigned int Stride,
          bool IsForwardFFT, unsigned int elements_per_thread, unsigned int FFTs_per_block>
inline void strided_block_fft_c2c_1d_launcher(T* data, unsigned int batch_size) {
    using namespace cufftdx;

    // Since complex input to FFT, convert vector into its scalar type
    using scalar_precision_type = zipfft::get_scalar_component_t<T>;
    constexpr auto fft_direction = IsForwardFFT ? fft_direction::forward : fft_direction::inverse;

    using FFT =
        decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() + Direction<fft_direction>() +
                 Precision<scalar_precision_type>() + ElementsPerThread<elements_per_thread>() +
                 FFTsPerBlock<FFTs_per_block>() + SM<Arch>());

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(strided_block_fft_c2c_1d_kernel<FFT, Stride>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             FFT::shared_memory_size));

    // Cast to cuFFTDx complex type from the FFT struct
    using complex_type = typename FFT::value_type;
    complex_type* data_t = reinterpret_cast<complex_type*>(data);

    // Figure out the grid dimensions (how many collective kernels to launch
    // which satisfies the FFT batch size).
    const unsigned int grid_size = (batch_size + FFT::ffts_per_block - 1) / FFT::ffts_per_block;

    // Launch the kernel and ensure no errors afterwards
    strided_block_fft_c2c_1d_kernel<FFT, Stride>
        <<<grid_size, FFT::block_dim, FFT::shared_memory_size>>>(data_t);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition
template <typename T, unsigned int FFTSize, unsigned int Stride, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
int strided_block_complex_fft_1d(T* data, unsigned int batch_size) {
    auto arch = zipfft::get_cuda_device_arch();

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: strided_block_fft_c2c_1d_launcher<800, T, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: strided_block_fft_c2c_1d_launcher<860, T, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: strided_block_fft_c2c_1d_launcher<870, T, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: strided_block_fft_c2c_1d_launcher<890, T, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: strided_block_fft_c2c_1d_launcher<900, T, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, batch_size); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        case 1200: strided_block_fft_c2c_1d_launcher<900, T, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, batch_size); break;
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
