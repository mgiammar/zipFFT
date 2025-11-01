#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_index_mapper.hpp"
#include "../include/zipfft_padded_io.hpp"

// --- Kernel Definition with Index Mappers for Padded C2C ---
template <class FFT, class InputLayout, class OutputLayout, unsigned int SignalLength>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_c2c_1d_kernel_with_layout(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;
    using io_type = zipfft::io_padded_with_layout<FFT, InputLayout, OutputLayout, SignalLength>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using custom padded layout (zero-pads if needed)
    io_type::load(data, thread_data, local_fft_id);

    // Execute the FFT with shared memory
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Store using custom padded layout (truncates if needed)
    io_type::store(thread_data, data, local_fft_id);
}

// --- Launcher Definition ---
template <unsigned int Arch, typename T, int SignalLength, unsigned int FFTSize, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
inline void padded_block_fft_c2c_1d_launcher(T* data) {
    using namespace cufftdx;

    // Since complex input to FFT, convert vector into its scalar type
    using scalar_precision_type = zipfft::get_scalar_component_t<T>;
    constexpr auto fft_direction = IsForwardFFT ? fft_direction::forward : fft_direction::inverse;

    using FFT =
        decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() + Direction<fft_direction>() +
                 Precision<scalar_precision_type>() + ElementsPerThread<elements_per_thread>() +
                 FFTsPerBlock<FFTs_per_block>() + SM<Arch>());

    // Define index mappers for padded contiguous memory layout
    // Input layout: (element_index, batch_index)
    // Shape is (batch, SignalLength) where SignalLength < FFT::input_length
    using InputLayout = zipfft::index_mapper<
        zipfft::int_pair<SignalLength, 1>,      // elements contiguous (stride 1)
        zipfft::int_pair<1, SignalLength>,      // batches strided by SignalLength
        zipfft::int_pair<0, 0>                  // dummy batch dimension for compatibility
        >;

    // Output layout: same as input for in-place padded C2C
    // Shape is (batch, SignalLength) where SignalLength < FFT::output_length
    using OutputLayout = zipfft::index_mapper<
        zipfft::int_pair<SignalLength, 1>,      // elements contiguous (stride 1)
        zipfft::int_pair<1, SignalLength>,      // batches strided by SignalLength
        zipfft::int_pair<0, 0>                  // dummy batch dimension for compatibility
        >;

    // Increase shared memory size, if needed
    auto kernel_ptr = padded_block_fft_c2c_1d_kernel_with_layout<FFT, InputLayout, OutputLayout, SignalLength>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    // Cast to cuFFTDx complex type from the FFT struct
    using complex_type = typename FFT::value_type;
    complex_type* data_t = reinterpret_cast<complex_type*>(data);

    // Launch the kernel and ensure no errors afterwards
    kernel_ptr<<<1, FFT::block_dim, FFT::shared_memory_size>>>(data_t);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
/**
 * @brief Performs a 1D complex-to-complex FFT with zero-padding/truncation using cuFFTDx.
 *
 * @tparam T Data type (e.g., float2).
 * @tparam SignalLength Actual signal length in memory (< FFTSize).
 * @tparam FFTSize Size of FFT to perform (must be >= SignalLength).
 * @tparam IsForwardFFT true for forward FFT, false for inverse.
 * @tparam elements_per_thread Elements processed per thread.
 * @tparam FFTs_per_block FFTs computed per thread block (effective batch size).
 *
 * @param data Pointer to device array of complex numbers with shape (batch, SignalLength).
 *
 * @return int Returns 0 on success, or error code on failure.
 */
template <typename T, int SignalLength, unsigned int FFTSize, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
int padded_block_complex_fft_1d(T* data) {
    auto arch = zipfft::get_cuda_device_arch();

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: padded_block_fft_c2c_1d_launcher<800, T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: padded_block_fft_c2c_1d_launcher<860, T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: padded_block_fft_c2c_1d_launcher<870, T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: padded_block_fft_c2c_1d_launcher<890, T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: padded_block_fft_c2c_1d_launcher<900, T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        case 1200: padded_block_fft_c2c_1d_launcher<900, T, SignalLength, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data); break;
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