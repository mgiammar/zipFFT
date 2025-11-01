#include <cufftdx.hpp>

#include "../include/zipfft_common.hpp"
#include "../include/zipfft_strided_io.hpp"
#include "../include/zipfft_strided_padded_io.hpp"

// --- Kernel Definition ---
template <int SignalLength, class FFT, unsigned int Stride, bool IsForwardFFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void strided_padded_block_fft_c2c_1d_kernel(typename FFT::value_type* input_data,
                                                typename FFT::value_type* output_data) {
    using complex_type = typename FFT::value_type;

    // If forward, input is padded and output is strided only
    // If inverse, input is strided only and output is padded.
    using strided_io = zipfft::io_strided<FFT, Stride>;
    using io_strided_padded = zipfft::io_strided_padded<FFT, Stride, SignalLength>;
    using input_utils = std::conditional_t<IsForwardFFT, io_strided_padded, strided_io>;
    using output_utils = std::conditional_t<IsForwardFFT, strided_io, io_strided_padded>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;
    input_utils::load(input_data, thread_data, local_fft_id);

        // Execute the FFT within shared memory
        extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
        FFT().execute(thread_data, shared_mem);

    // Save results back to global memory
    output_utils::store(thread_data, output_data, local_fft_id);
}

// --- Launcher Definition ---
template <unsigned int Arch, typename T, int SignalLength, unsigned int FFTSize,
          unsigned int Stride, bool IsForwardFFT, unsigned int elements_per_thread,
          unsigned int FFTs_per_block>
inline void strided_padded_block_fft_c2c_1d_launcher(T* input_data, T* output_data,
                                                     unsigned int batch_size) {
    using namespace cufftdx;

    // Since complex input to FFT, convert vector into its scalar type
    using scalar_precision_type = zipfft::get_scalar_component_t<T>;
    constexpr auto fft_direction = IsForwardFFT ? fft_direction::forward : fft_direction::inverse;

    using FFT =
        decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() + Direction<fft_direction>() +
                 Precision<scalar_precision_type>() + ElementsPerThread<elements_per_thread>() +
                 FFTsPerBlock<FFTs_per_block>() + SM<Arch>());

    // Increase shared memory size, if needed
    auto kernel_ptr =
        strided_padded_block_fft_c2c_1d_kernel<SignalLength, FFT, Stride, IsForwardFFT>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    // Cast to cuFFTDx complex type from the FFT struct
    using complex_type = typename FFT::value_type;
    complex_type* input_data_t = reinterpret_cast<complex_type*>(input_data);
    complex_type* output_data_t = reinterpret_cast<complex_type*>(output_data);

    // Figure out the grid dimensions (how many collective kernels to launch
    // which satisfies the FFT batch size).
    const unsigned int grid_size = (batch_size + FFT::ffts_per_block - 1) / FFT::ffts_per_block;

    // Launch the kernel and ensure no errors afterwards
    kernel_ptr<<<grid_size, FFT::block_dim, FFT::shared_memory_size>>>(input_data_t, output_data_t);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
template <typename T, unsigned int SignalLength, unsigned int FFTSize, unsigned int Stride,
          bool IsForwardFFT, unsigned int elements_per_thread, unsigned int FFTs_per_block>
int strided_padded_block_complex_fft_1d(T* input_data, T* output_data, unsigned int batch_size) {
    auto arch = zipfft::get_cuda_device_arch();

    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: strided_padded_block_fft_c2c_1d_launcher<800, T, SignalLength, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: strided_padded_block_fft_c2c_1d_launcher<860, T, SignalLength, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: strided_padded_block_fft_c2c_1d_launcher<870, T, SignalLength, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: strided_padded_block_fft_c2c_1d_launcher<890, T, SignalLength, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: strided_padded_block_fft_c2c_1d_launcher<900, T, SignalLength, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, batch_size); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        // Fallback: Architecture 1200 uses the 900 template as cuFFTDx does not yet
        // provide specific optimizations for newer architectures like Hopper/Blackwell.
        case 1200: strided_padded_block_fft_c2c_1d_launcher<900, T, SignalLength, FFTSize, Stride, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, batch_size); break;
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