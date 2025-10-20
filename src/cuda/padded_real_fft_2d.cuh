#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_padded_io.hpp"
#include "../include/zipfft_strided_io.hpp"
#include "../include/zipfft_strided_padded_io.hpp"

// --- Kernel Definitions ---
template <int SignalLength, class FFT, typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_r2c_2d_kernel(ScalarType* input_data, ComplexType* output_data,
                                        typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Input is padded, use padded I/O utilities. Output is not padded
    using input_utils = zipfft::io_padded<FFT, SignalLength>;
    using output_utils = zipfft::io<FFT>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // Grid dimension (blockIdx.x) corresponds to the row/col in the input
    // signal and handled automatically by included padded_io.hpp functions.
    // Local FFT index used for multiple FFTs per block (generally just 1)
    const unsigned int local_fft_id = threadIdx.y;
    input_utils::load(input_data, thread_data, local_fft_id);

    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    output_utils::store(thread_data, output_data, local_fft_id);
}

template <int SignalLength, class FFT, typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_c2r_2d_kernel(ComplexType* input_data, ScalarType* output_data,
                                        typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Input is not padded, output is padded (truncated)
    using input_utils = zipfft::io<FFT>;
    using output_utils = zipfft::io_padded<FFT, SignalLength>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    // then load data from global memory to registers
    const unsigned int local_fft_id = threadIdx.y;
    input_utils::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    output_utils::store(thread_data, output_data, local_fft_id);
}

template <unsigned int SignalLength, class FFT, unsigned int Stride, bool IsForwardFFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_c2c_2d_kernel_x(typename FFT::value_type* data,
                                          typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    // If forward, input is padded and output is strided only
    // If inverse, input is strided only and output is padded.
    using stride_io = zipfft::io_strided<FFT, Stride>;
    using io_strided_padded = zipfft::io_strided_padded<FFT, Stride, SignalLength>;
    using input_utils = std::conditional_t<IsForwardFFT, io_strided_padded, stride_io>;
    using output_utils = std::conditional_t<IsForwardFFT, stride_io, io_strided_padded>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;
    input_utils::load(data, thread_data, local_fft_id);

    // Execute FFT within shared memory
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    output_utils::store(thread_data, data, local_fft_id);
}

// --- Launcher Definitions ---
/**
 * @brief Launcher function for the padded 2D FFT execution
 *
 * If IsForwardFFT is true, then the data are assumed to have the following shape
 * - input_data: (batch_size, SignalLengthY, SignalLengthX)
 * - output_data: (batch_size, FFTSizeY, FFTSizeX // 2 + 1)
 *
 * If IsForwardFFT is false (inverse FFT), then the data are assumed to have the following shape
 * - input_data: (batch_size, FFTSizeY, FFTSizeX // 2 + 1)
 * - output_data: (batch_size, SignalLengthY, SignalLengthX)
 *
 * @tparam Arch - CUDA architecture
 * @tparam Input_T - Pointer type of input data. If forward, then some real type. If inverse (not
 * forward), then some complex type.
 * @tparam Output_T - Pointer type of the output data. If forward, then some complex type. If
 * inverse (not forward), then some real type.
 * @tparam SignalLengthX - Number of elements in the X dimension (fastest dimension). Corresponds to
 * the number of columns in the 2D array.
 * @tparam SignalLengthY - Number of elements in the Y dimension (second fastest dimension).
 * Corresponds to the number of rows in the 2D array.
 * @tparam FFTSizeX - How large the FFT is in the X dimension. (FFTSizeX - SignalLengthX) gives the
 * amount of padding in the X dimension.
 * @tparam FFTSizeY - How large the FFT is in the Y dimension. (FFTSizeY - SignalLengthY) gives the
 * amount of padding in the Y dimension.
 * @tparam IsForwardFFT - Whether the FFT is forward (true) or inverse (false).
 * @tparam elements_per_thread_x - Number of elements processed by each thread in the X dimension.
 * @tparam elements_per_thread_y - Number of elements processed by each thread in the Y dimension.
 * @tparam FFTs_per_block_x - Number of FFTs processed by each block in the X dimension.
 * @tparam FFTs_per_block_y - Number of FFTs processed by each block in the Y dimension.
 *
 * @param input_data - Pointer to the input data.
 * @param output_data - Pointer to the output data.
 * @param batch_size - batch size of the data to process
 */
template <unsigned int Arch, typename Input_T, typename Output_T, unsigned int SignalLengthX,
          unsigned int SignalLengthY, unsigned int FFTSizeX, unsigned int FFTSizeY,
          bool IsForwardFFT, unsigned int elements_per_thread_x, unsigned int elements_per_thread_y,
          unsigned int FFTs_per_block_x, unsigned int FFTs_per_block_y>
inline void padded_block_real_ft_2d_launcher(Input_T* input_data, Output_T* output_data,
                                             unsigned int batch_size) {
    using namespace cufftdx;

    // Conditional statements for the FFT declarations
    using real_ftt_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using scalar_precision_type = std::conditional_t<IsForwardFFT, Input_T, Output_T>;
    constexpr auto fft_direction = IsForwardFFT ? fft_direction::forward : fft_direction::inverse;
    constexpr auto fft_type = IsForwardFFT ? fft_type::r2c : fft_type::c2r;

    // FFTX is the r2c or c2r transform
    using FFTX = decltype(Block() + Size<FFTSizeX>() + Type<fft_type>() + real_ftt_options() +
                          Direction<fft_direction>() + Precision<scalar_precision_type>() +
                          ElementsPerThread<elements_per_thread_x>() +
                          FFTsPerBlock<FFTs_per_block_x>() + SM<Arch>());

    // FFTY is the c2c transform
    using FFTY =
        decltype(Block() + Size<FFTSizeY>() + Type<fft_type::c2c>() + Direction<fft_direction>() +
                 Precision<scalar_precision_type>() + ElementsPerThread<elements_per_thread_y>() +
                 FFTsPerBlock<FFTs_per_block_y>() + SM<Arch>());

    using complex_type = typename FFTX::value_type;
    using scalar_type = typename complex_type::value_type;
    constexpr auto stride_y = FFTSizeX / 2 + 1;

    // Create workspaces for FFTs
    cudaError_t error_code = cudaSuccess;
    auto workspace_x = make_workspace<FFTX>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_y = make_workspace<FFTY>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Compile-time branching for kernel selection and execution
    if constexpr (IsForwardFFT) {
        auto kernel_ptr_x = padded_block_fft_r2c_2d_kernel<SignalLengthX, FFTX>;
        auto kernel_ptr_y =
            padded_block_fft_c2c_2d_kernel_x<SignalLengthY, FFTY, stride_y, IsForwardFFT>;

        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr_x, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTX::shared_memory_size));
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr_y, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY::shared_memory_size));

        // Prepare data pointers
        scalar_type* input_data_t = reinterpret_cast<scalar_type*>(input_data);
        complex_type* output_data_t = reinterpret_cast<complex_type*>(output_data);

        // Launch transform along X dimension
        // Only need to launch the first SignalLengthY blocks since all blocks
        // past that are zeros.
        const unsigned int grid_size_x =
            ((batch_size * SignalLengthY) + FFTX::ffts_per_block - 1) / FFTX::ffts_per_block;
        kernel_ptr_x<<<grid_size_x, FFTX::block_dim, FFTX::shared_memory_size>>>(
            input_data_t, output_data_t, workspace_x);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

        // Launch transform along Y dimension
        // All blocks are needed since after X transform, all columns hold
        // some non-zero data.
        const unsigned int grid_size_y =
            ((batch_size * stride_y) + FFTY::ffts_per_block - 1) / FFTY::ffts_per_block;
        kernel_ptr_y<<<grid_size_y, FFTY::block_dim, FFTY::shared_memory_size>>>(
            output_data_t, workspace_y);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    } else {
        auto kernel_ptr_x = padded_block_fft_c2r_2d_kernel<SignalLengthX, FFTX>;
        auto kernel_ptr_y =
            padded_block_fft_c2c_2d_kernel_x<SignalLengthY, FFTY, stride_y, IsForwardFFT>;

        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr_x, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTX::shared_memory_size));
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr_y, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY::shared_memory_size));

        // Prepare data pointers
        complex_type* input_data_t = reinterpret_cast<complex_type*>(input_data);
        scalar_type* output_data_t = reinterpret_cast<scalar_type*>(output_data);

        // Launch transform along Y dimension
        const unsigned int grid_size_y =
            ((batch_size * stride_y) + FFTY::ffts_per_block - 1) / FFTY::ffts_per_block;
        kernel_ptr_y<<<grid_size_y, FFTY::block_dim, FFTY::shared_memory_size>>>(
            input_data_t, workspace_y);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

        // Launch transform along X dimension
        // Only need to launch the first SignalLengthY blocks since all blocks
        // past that will be excluded on the cropped output.
        const unsigned int grid_size_x =
            ((batch_size * SignalLengthY) + FFTX::ffts_per_block - 1) / FFTX::ffts_per_block;
        kernel_ptr_x<<<grid_size_x, FFTX::block_dim, FFTX::shared_memory_size>>>(
            input_data_t, output_data_t, workspace_x);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }

    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
template <typename Input_T, typename Output_T, unsigned int SignalLengthX, unsigned int SignalLengthY,
          unsigned int FFTSizeX, unsigned int FFTSizeY, bool IsForwardFFT,
          unsigned int elements_per_thread_x, unsigned int elements_per_thread_y,
          unsigned int FFTs_per_block_x, unsigned int FFTs_per_block_y>
int padded_block_real_fft_2d(Input_T* input_data, Output_T* output_data, unsigned int batch_size) {
    auto arch = zipfft::get_cuda_device_arch();

    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: padded_block_real_ft_2d_launcher<800, Input_T, Output_T, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: padded_block_real_ft_2d_launcher<860, Input_T, Output_T, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: padded_block_real_ft_2d_launcher<870, Input_T, Output_T, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: padded_block_real_ft_2d_launcher<890, Input_T, Output_T, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data, batch_size); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: padded_block_real_ft_2d_launcher<900, Input_T, Output_T, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data, batch_size); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        // Fallback: use the 900 specialization for newer architectures
        case 1200: padded_block_real_ft_2d_launcher<900, Input_T, Output_T, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data, batch_size); break;
#endif
        default:
            std::cerr << "Unsupported CUDA architecture: " << arch
                      << ". Supported architectures are 800, 860, 870, 890, 900, and 1200."
                      << std::endl;
            return -1;
    }
    /* clang-format on */

    return 0;
}