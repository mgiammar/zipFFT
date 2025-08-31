#include <cufftdx.hpp>

#include "../include/block_io_strided.hpp"
#include "../include/common.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>

// --- Kernel Definition ---
template <class FFT, class FFT_inv>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_convolution_strided_kernel(typename FFT::value_type* data, typename FFT::value_type* kernel, unsigned int inner_batch_count) {
    using complex_type = typename FFT::value_type;

    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    
    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;
    example::io_strided<FFT>::load_strided_smem(data, thread_data, shared_mem, local_fft_id, inner_batch_count * FFT::ffts_per_block);
    
    __syncthreads();

    // Execute the FFT with shared memory
    FFT().execute(thread_data, shared_mem);
    
    __syncthreads();

    complex_type kernel_thread_data[FFT::storage_size];
    example::io_strided<FFT>::load_strided_smem(kernel, kernel_thread_data, shared_mem, local_fft_id, inner_batch_count * FFT::ffts_per_block);

    __syncthreads();

    // complex multiplication in the frequency domain
    for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
        complex_type a;
        a.x = thread_data[i].x;
        a.y = thread_data[i].y;

        complex_type b;
        b.x = kernel_thread_data[i].x;
        b.y = kernel_thread_data[i].y;
        
        complex_type c;
        c.x = a.x * b.x - a.y * b.y;
        c.y = a.x * b.y + a.y * b.x;
        
        thread_data[i].x = c.x;
        thread_data[i].y = c.y;
    }

    __syncthreads();

    FFT_inv().execute(thread_data, shared_mem);

    __syncthreads();

    // Save results back to global memory
    example::io_strided<FFT>::store_strided_smem(thread_data, shared_mem, data, local_fft_id, inner_batch_count * FFT::ffts_per_block);
}

// --- Launcher Definition ---
template <unsigned int Arch, typename T, unsigned int FFTSize,
          unsigned int elements_per_thread,
          unsigned int FFTs_per_block>
inline void block_convolution_strided_launcher(T* data, T* kernel, unsigned int inner_batch_count, unsigned int outer_batch_count) {
    using namespace cufftdx;

    // Since complex input to FFT, convert vector into its scalar type
    using scalar_precision_type = example::get_scalar_component_t<T>;

    using FFT_Base = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                                 Precision<scalar_precision_type>() +
                                 ElementsPerThread<elements_per_thread>() +
                                 FFTsPerBlock<FFTs_per_block>() + SM<Arch>());

    using FFT = decltype(FFT_Base() + Direction<fft_direction::forward>());
    using FFT_inv = decltype(FFT_Base() + Direction<fft_direction::inverse>());

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_convolution_strided_kernel<FFT, FFT_inv>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    // Cast to cuFFTDx complex type form the FFT struct
    using complex_type = typename FFT::value_type;
    complex_type* data_t = reinterpret_cast<complex_type*>(data);
    complex_type* kernel_t = reinterpret_cast<complex_type*>(kernel);

    // use the pytorch cuda stream to allow for graph capture
    cudaStream_t strm = at::cuda::getCurrentCUDAStream().stream();

    dim3 grid_dims(outer_batch_count, inner_batch_count);

    block_convolution_strided_kernel<FFT, FFT_inv>
        <<<grid_dims, FFT::block_dim, FFT::shared_memory_size, strm>>>(data_t, kernel_t, inner_batch_count);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
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
template <typename T, unsigned int FFTSize,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
int block_convolution_strided(T* data, T* kernel, unsigned int inner_batch_count,  unsigned int outer_batch_count) {
    auto arch = example::get_cuda_device_arch();

    //printf("Running on CUDA architecture: %u\n", arch);
    //printf("Outer batch count: %u\n", outer_batch_count);

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
        case 800: block_convolution_strided_launcher<800, T, FFTSize, elements_per_thread, FFTs_per_block>(data, kernel, inner_batch_count, outer_batch_count); break;
        case 860: block_convolution_strided_launcher<860, T, FFTSize, elements_per_thread, FFTs_per_block>(data, kernel, inner_batch_count, outer_batch_count); break;
        case 870: block_convolution_strided_launcher<870, T, FFTSize, elements_per_thread, FFTs_per_block>(data, kernel, inner_batch_count, outer_batch_count); break;
        case 890: block_convolution_strided_launcher<890, T, FFTSize, elements_per_thread, FFTs_per_block>(data, kernel, inner_batch_count, outer_batch_count); break;
        case 900: block_convolution_strided_launcher<900, T, FFTSize, elements_per_thread, FFTs_per_block>(data, kernel, inner_batch_count, outer_batch_count); break;
        case 1200: block_convolution_strided_launcher<900, T, FFTSize, elements_per_thread, FFTs_per_block>(data, kernel, inner_batch_count, outer_batch_count); break; 
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
