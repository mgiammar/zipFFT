#include <cufftdx.hpp>

#include "../include/block_io.hpp"
#include "../include/common.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// --- r2c & c2r Kernel Definitions ---
template <class FFT, typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_fft_r2c_1d_kernel(ScalarType* input_data,
                                 ComplexType* output_data) {
    using complex_type = ComplexType;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    // Then load data from global memory to local memory
    const unsigned int local_fft_id = threadIdx.y;
    example::io<FFT>::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, output_data, local_fft_id);
}

template <class FFT, typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_fft_c2r_1d_kernel(ComplexType* input_data,
                                 ScalarType* output_data) {
    using complex_type = ComplexType;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    // Then load data from global memory to local memory
    const unsigned int local_fft_id = threadIdx.y;
    example::io<FFT>::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, output_data, local_fft_id);
}

// --- Unified Launcher Definition (for both r2c & c2r) ---
template <unsigned int Arch, typename Input_T, typename Output_T,
          unsigned int FFTSize, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
inline void block_real_fft_1d_launcher(Input_T* input_data,
                                       Output_T* output_data,
                                       unsigned int outer_batch_count) {
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

    // use the pytorch cuda stream to allow for graph capture
    cudaStream_t strm = at::cuda::getCurrentCUDAStream().stream();

    // Compile-time branching to determine which FFT kernel to use
    if constexpr (IsForwardFFT) {
        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(
            cudaFuncSetAttribute(block_fft_r2c_1d_kernel<FFT>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FFT::shared_memory_size));

        // Cast input data to cuFFTDx types
        scalar_type* input_data_t = reinterpret_cast<scalar_type*>(input_data);
        complex_type* output_data_t =
            reinterpret_cast<complex_type*>(output_data);

        // Launch the kernel
        block_fft_r2c_1d_kernel<FFT>
            <<<outer_batch_count, FFT::block_dim, FFT::shared_memory_size, strm>>>(input_data_t,
                                                             output_data_t);
    } else {
        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(
            cudaFuncSetAttribute(block_fft_c2r_1d_kernel<FFT>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FFT::shared_memory_size));

        // Cast input data to cuFFTDx types
        complex_type* input_data_t =
            reinterpret_cast<complex_type*>(input_data);
        scalar_type* output_data_t =
            reinterpret_cast<scalar_type*>(output_data);

        // Launch the kernel
        block_fft_c2r_1d_kernel<FFT>
            <<<outer_batch_count, FFT::block_dim, FFT::shared_memory_size, strm>>>(input_data_t,
                                                             output_data_t);
    }

    // Ensure no errors afterwards
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
template <typename Input_T, typename Output_T, unsigned int FFTSize,
          bool IsForwardFFT, unsigned int elements_per_thread,
          unsigned int FFTs_per_block>
int block_real_fft_1d(Input_T* input_data, Output_T* output_data, unsigned int outer_batch_count) {
    auto arch = example::get_cuda_device_arch();

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
        case 800: block_real_fft_1d_launcher<800, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, outer_batch_count); break;
        case 860: block_real_fft_1d_launcher<860, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, outer_batch_count); break;
        case 870: block_real_fft_1d_launcher<870, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, outer_batch_count); break;
        case 890: block_real_fft_1d_launcher<890, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, outer_batch_count); break;
        case 900: block_real_fft_1d_launcher<900, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, outer_batch_count); break;
        // Fallback: Architecture 1200 uses the 900 template as cuFFTDx does not yet
        // provide specific optimizations for newer architectures like Hopper/Blackwell.
        case 1200: block_real_fft_1d_launcher<900, Input_T, Output_T, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(input_data, output_data, outer_batch_count); break; 
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
