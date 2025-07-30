#include <cuda_runtime.h>

#include <cufftdx.hpp>
#include <type_traits>

#include "../include/block_io.hpp"
#include "../include/common.hpp"
// #include "../include/dispatcher.hpp"

// --- Kernel Definition ---
template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_fft_c2c_1d_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;
    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;
    example::io<FFT>::load(data, thread_data, local_fft_id);

    // Execute the FFT with shared memory
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results back to global memory
    example::io<FFT>::store(thread_data, data, local_fft_id);
}

// --- Launcher Definition ---
template <unsigned int Arch, typename T, unsigned int FFTSize,
          bool IsForwardFFT, unsigned int elements_per_thread,
          unsigned int FFTs_per_block>
inline void block_fft_c2c_1d_launcher(T* data) {
    using namespace cufftdx;

    // Since complex input to FFT, convert vector into its scalar type
    using scalar_precision_type = example::get_scalar_component_t<T>;
    constexpr auto fft_direction =
        IsForwardFFT ? fft_direction::forward : fft_direction::inverse;

    using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                         Direction<fft_direction>() +
                         Precision<scalar_precision_type>() +
                         ElementsPerThread<elements_per_thread>() +
                         FFTsPerBlock<FFTs_per_block>() + SM<Arch>());

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_c2c_1d_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    // Cast to cuFFTDx complex type form the FFT struct
    using complex_type = typename FFT::value_type;
    complex_type* data_t = reinterpret_cast<complex_type*>(data);

    // Launch the kernel and ensure no errors afterwards
    block_fft_c2c_1d_kernel<FFT>
        <<<1, FFT::block_dim, FFT::shared_memory_size>>>(data_t);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
/**
 * @brief Performs a 1D complex-to-complex FFT using cuFFTDx, automatically
 * selecting the CUDA architecture.
 *
 * @tparam T Data type (e.g., float2).
 * @tparam FFTSize Number of elements in the FFT.
 * @tparam IsForwardFFT true for forward FFT, false for inverse.
 * @tparam elements_per_thread Elements processed per thread.
 * @tparam FFTs_per_block FFTs computed per thread block (effective batch size).
 *
 * @param data Pointer to device array of complex numbers.
 *
 * @return int Returns 0 on success, or error code on failure.
 */
template <typename T, unsigned int FFTSize, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
int block_complex_fft_1d(T* data) {
    auto arch = example::get_cuda_device_arch();

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
        case 800: block_fft_c2c_1d_launcher<800, T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
        case 860: block_fft_c2c_1d_launcher<860, T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
        case 870: block_fft_c2c_1d_launcher<870, T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
        case 890: block_fft_c2c_1d_launcher<890, T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
        case 900: block_fft_c2c_1d_launcher<900, T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
        case 1200: block_fft_c2c_1d_launcher<900, T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break; 
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
