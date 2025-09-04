#include <cufftdx.hpp>

#include "../include/block_io.hpp"
#include "../include/common.hpp"
#include "../include/padded_io.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_c2c_1d_kernel(
        typename FFT::value_type* data,
        typename FFT::workspace_type workspace,
        unsigned int signal_length,
        unsigned int active_layers,
        unsigned int extra_layers) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    // then load data from global memory to registers
    const unsigned int local_fft_id = threadIdx.y;
    example::io<FFT>::load_padded_layered(data, thread_data, local_fft_id, signal_length, active_layers, extra_layers);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    example::io<FFT>::store_layered(thread_data, data, local_fft_id, active_layers, extra_layers);
}

// --- Unified Launcher Definition (for both padded r2c & c2r) ---
template <unsigned int Arch, typename InputType,
          unsigned int FFTSize, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
inline void padded_block_complex_fft_1d_launcher(InputType* data,
                                              unsigned int signal_length,
                                              unsigned int outer_batch_count,
                                              unsigned int active_layers,
                                              unsigned int extra_layers) {
    using namespace cufftdx;

    using scalar_precision_type = example::get_scalar_component_t<InputType>;
    constexpr auto fft_direction =
        IsForwardFFT ? fft_direction::forward : fft_direction::inverse;

    using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                         Direction<fft_direction>() +
                         Precision<scalar_precision_type>() +
                         ElementsPerThread<elements_per_thread>() +
                         FFTsPerBlock<FFTs_per_block>() + SM<Arch>());

    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    cudaStream_t strm = at::cuda::getCurrentCUDAStream().stream();

    // Compile-time branching to determine which FFT kernel to use
    if constexpr (IsForwardFFT) {
        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            padded_block_fft_c2c_1d_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            FFT::shared_memory_size));

        // Cast input data to cuFFTDx types
        //scalar_type* input_data_t = reinterpret_cast<scalar_type*>(input_data);
        complex_type* data_t = reinterpret_cast<complex_type*>(data);

        // create workspaces for FFT
        cudaError_t error_code = cudaSuccess;
        auto workspace = make_workspace<FFT>(error_code);
        CUDA_CHECK_AND_EXIT(error_code);

        // Launch the kernel
        padded_block_fft_c2c_1d_kernel<FFT>
            <<<outer_batch_count, FFT::block_dim, FFT::shared_memory_size, strm>>>(
                data_t, workspace, signal_length, active_layers, extra_layers);
    } else {

    }

    // Ensure no errors afterwards
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    //CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
template <typename InputType,
          unsigned int FFTSize, bool IsForwardFFT,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
int padded_block_real_fft_1d(InputType* data, unsigned int signal_length, unsigned int outer_batch_count, unsigned int active_layers, unsigned int extra_layers) {
    // Throw if backward FFTs since they are not implemented yet
    static_assert(IsForwardFFT,
                  "Backward padded real FFTs (c2r) are not implemented yet.");

    auto arch = example::get_cuda_device_arch();

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
        case 800: padded_block_complex_fft_1d_launcher<800, InputType, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, signal_length, outer_batch_count, active_layers, extra_layers); break;
        case 860: padded_block_complex_fft_1d_launcher<860, InputType, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, signal_length, outer_batch_count, active_layers, extra_layers); break;
        case 870: padded_block_complex_fft_1d_launcher<870, InputType, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, signal_length, outer_batch_count, active_layers, extra_layers); break;
        case 890: padded_block_complex_fft_1d_launcher<890, InputType, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, signal_length, outer_batch_count, active_layers, extra_layers); break;
        case 900: padded_block_complex_fft_1d_launcher<900, InputType, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, signal_length, outer_batch_count, active_layers, extra_layers); break;
        // Fallback: Architecture 1200 uses the 900 template as cuFFTDx does not yet
        // provide specific optimizations for newer architectures like Hopper/Blackwell.
        case 1200: padded_block_complex_fft_1d_launcher<900, InputType, FFTSize, IsForwardFFT, elements_per_thread, FFTs_per_block>(data, signal_length, outer_batch_count, active_layers, extra_layers); break;
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
