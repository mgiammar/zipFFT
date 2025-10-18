#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_padded_io.hpp"

/**
 * @brief Kernel which performs the cross-correlation of an implicitly padded
 * input filter with an input signal using the FFT-based convolution.
 *
 * The signal being cross-correlated against must be already RFFT-ed, and its
 * real-space length is assumed to be the same size as the padding length (e.g.
 * if the input filter is padded to 128, the signal length must be 128).
 * Taking this approach is good when the input signal does not change (and its
 * RFFT can be computed once and reused) and the only thing changing is the
 * filter.
 *
 * Alternate constructs which take in an arbitrary real-space signal and filter
 * and transform them together are possible and may be implemented elsewhere.
 *
 * NOTE: Signal length here corresponds to the un-padded length of the input
 * filter in real-space.
 *
 * @tparam FilterLength - Integer length of the un-padded filter in real-space.
 * @tparam ForwardFFT - cuFFTDx forward FFT structure.
 * @tparam InverseFFT - cuFFTDx inverse FFT structure.
 * @tparam ComplexType - The cuFFTDx complex type used in the FFTs.
 * @tparam ScalarType - The scalar type used in the input and output data.
 *
 * @param input_data - Pointer to the filter data in rea-space (un-padded).
 * @param output_data - Pointer to the ouput cross-correlogram in real-space.
 * @param signal_data - Pointer to the signal data, already RFFT-ed.
 * @param workspace_fwd - Workspace for the forward FFT.
 * @param workspace_inv - Workspace for the inverse FFT.
 */
template <int FilterLength, class ForwardFFT, class InverseFFT,
          typename ComplexType = typename ForwardFFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(ForwardFFT::max_threads_per_block) __global__
    void padded_block_conv_real_1d_kernel(
        ScalarType* input_data, ScalarType* output_data,
        ComplexType* signal_data, typename ForwardFFT::workspace_type workspace_fwd,
        typename InverseFFT::workspace_type workspace_inv) {
    using complex_type = typename ForwardFFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Input is padded (the data to be FFT-ed), output is not padded
    using input_utils_padded = zipfft::io_padded<ForwardFFT, FilterLength>;
    // using input_utils = zipfft::io<ForwardFFT>;
    using output_utils = zipfft::io<InverseFFT>;

    // ID of FFT in CUDA block, in range [0, FFT::ffts_per_block)
    // Data here are loaded from global memory to local thread data based on
    // a strided access pattern where roughly
    // thread_data[i] = input_data[threadIdx.x + i * FFT:input_ept]
    const unsigned int local_fft_id = threadIdx.y;
    complex_type thread_data[ForwardFFT::storage_size];
    input_utils_padded::load(input_data, thread_data, local_fft_id);

    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    ForwardFFT().execute(thread_data, shared_mem, workspace_fwd);

    // Data is now transformed to complex type in the Fourier domain. Each
    // thread stores 'FFT::output_ept' complex numbers and has a similar strided
    // access pattern.
    //
    // thread_data[i] is the (threadIdx.x + i * FFT::output_ept)-th complex
    // number in the sequence of [0, FFT:output_length - 1]
    //
    // NOTE: The last element 'thread_data[FFT::storage_size - 1]' is only valid
    // for the first thread in the block (threadIdx.x == 0) based on the
    // definition of the R2C FFT. Other threads last element contain unknown
    // data and should not be accessed for convolution.

    for (unsigned int i = 0; i < ForwardFFT::output_ept - 1; i++) {
        thread_data[i] *= signal_data[threadIdx.x + i * ForwardFFT::output_ept];
    }

    // Conditional multiplication for the first thread
    if (threadIdx.x == 0) {
        thread_data[ForwardFFT::output_ept - 1] *=
            signal_data[ForwardFFT::output_length - 1];
    }

    // Execute inverse FFT to get the convolution result
    InverseFFT().execute(thread_data, shared_mem, workspace_inv);

    // Save results
    output_utils::store(thread_data, output_data, local_fft_id);
}

template <unsigned int Arch, typename ScalarType, typename ComplexType,
          unsigned int FilterLength, unsigned int FFTSize,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
inline void padded_real_conv_1d_launcher(ScalarType* input_data,
                                         ScalarType* output_data,
                                         ComplexType* signal_data) {
    using namespace cufftdx;

    // Real FFT specific data layout properties
    using real_fft_options =
        RealFFTOptions<complex_layout::natural, real_mode::folded>;

    using fft_base = decltype(Block() + Size<FFTSize>() + real_fft_options() +
                              Precision<ScalarType>() +
                              ElementsPerThread<elements_per_thread>() +
                              FFTsPerBlock<FFTs_per_block>() + SM<Arch>());
    using fft = decltype(fft_base() + Type<fft_type::r2c>() +
                         Direction<fft_direction::forward>());
    using ifft = decltype(fft_base() + Type<fft_type::c2r>() +
                          Direction<fft_direction::inverse>());

    // Increase max shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        padded_block_conv_real_1d_kernel<FilterLength, fft, ifft>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, fft::shared_memory_size));

    // Type casing
    using complex_type = typename fft::value_type;
    using scalar_type = typename complex_type::value_type;
    scalar_type* input_data_t = reinterpret_cast<scalar_type*>(input_data);
    scalar_type* output_data_t = reinterpret_cast<scalar_type*>(output_data);
    complex_type* signal_data_t = reinterpret_cast<complex_type*>(signal_data);

    // Allocate workspace for FFT
    cudaError_t error_code = cudaSuccess;
    auto workspace_fwd = make_workspace<fft>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_inv = make_workspace<ifft>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Launch the kernel
    padded_block_conv_real_1d_kernel<FilterLength, fft, ifft>
        <<<1, fft::block_dim, fft::shared_memory_size>>>(
            input_data_t, output_data_t, signal_data_t, workspace_fwd,
            workspace_inv);

    // Check for errors in kernel launch
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

template <typename ScalarType, typename ComplexType, unsigned int FilterLength,
          unsigned int FFTSize, unsigned int elements_per_thread,
          unsigned int FFTs_per_block>
int padded_block_conv_real_1d(ScalarType* input_data, ScalarType* output_data,
                              ComplexType* signal_data) {
    auto arch = zipfft::get_cuda_device_arch();

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
        case 800: padded_real_conv_1d_launcher<800, ScalarType, ComplexType, FilterLength, FFTSize, elements_per_thread, FFTs_per_block>(input_data, output_data, signal_data); break;
        case 860: padded_real_conv_1d_launcher<860, ScalarType, ComplexType, FilterLength, FFTSize, elements_per_thread, FFTs_per_block>(input_data, output_data, signal_data); break;
        case 870: padded_real_conv_1d_launcher<870, ScalarType, ComplexType, FilterLength, FFTSize, elements_per_thread, FFTs_per_block>(input_data, output_data, signal_data); break;
        case 890: padded_real_conv_1d_launcher<890, ScalarType, ComplexType, FilterLength, FFTSize, elements_per_thread, FFTs_per_block>(input_data, output_data, signal_data); break;
        case 900: padded_real_conv_1d_launcher<900, ScalarType, ComplexType, FilterLength, FFTSize, elements_per_thread, FFTs_per_block>(input_data, output_data, signal_data); break;
        // Fallback: Architecture 1200 uses the 900 template as cuFFTDx does not yet
        // provide specific optimizations for newer architectures like Hopper/Blackwell.
        case 1200: padded_real_conv_1d_launcher<900, ScalarType, ComplexType, FilterLength, FFTSize, elements_per_thread, FFTs_per_block>(input_data, output_data, signal_data); break;
        default:
            std::cerr << "Unsupported CUDA architecture: " << arch
                    << ". Supported architectures are 800, 860, 870, 890, "
                        "900, and 1200."
                    << std::endl;
            return -1;  // Error code for unsupported architecture
    }
    /* clang-format on */

    return 0;
}


// Debugging for compilation
template int padded_block_conv_real_1d<float, float2, 512, 4096, 32u, 1u>(float*, float*, float2*);