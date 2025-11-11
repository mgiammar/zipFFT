#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_index_mapper.hpp"
#include "../include/zipfft_padded_io.hpp"
#include "../include/zipfft_strided_io.hpp"
#include "../include/zipfft_strided_padded_io.hpp"

// --- Forward r2c Kernel Definition with Index Mappers ---
// This kernel just does the real transform, no convolution
template <class FFT, class InputLayout, class OutputLayout, unsigned int SignalLength,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_r2c_1d_kernel_with_layout(ScalarType* input_data,
                                                    ComplexType* output_data,
                                                    typename FFT::workspace_type workspace) {
    using complex_type = ComplexType;
    using scalar_type = ScalarType;

    // Input is padded (zero-pad from SignalLength to FFT size)
    // Output uses standard layout
    // InputLayout (second template arg) only matters for the input_io
    // OutputLayout (third template arg) only matters for the output_io
    using input_io = zipfft::io_padded_with_layout<FFT, InputLayout, InputLayout, SignalLength>;
    using output_io = zipfft::io_with_layout<FFT, OutputLayout, OutputLayout>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using padded input layout (zero-pads to FFT size)
    input_io::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Store using output layout (full FFT output)
    output_io::store(thread_data, output_data, local_fft_id);
}

// --- Inverse c2r Kernel Definition with Index Mappers ---
// This kernel just does the real transform, no convolution
template <class FFT, class InputLayout, class OutputLayout, unsigned int SignalLength,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_c2r_2d_kernel_with_layout(ComplexType* input_data,
                                                    ScalarType* output_data,
                                                    typename FFT::workspace_type workspace) {
    using complex_type = ComplexType;
    using scalar_type = ScalarType;

    // Input uses standard layout
    // Output is padded (truncates from FFT size to SignalLength)
    using input_io = zipfft::io_with_layout<FFT, InputLayout, InputLayout>;
    using output_io = zipfft::io_padded_with_layout<FFT, OutputLayout, OutputLayout, SignalLength>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using input layout (full complex input)
    input_io::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Store using padded output layout (truncates to SignalLength)
    output_io::store(thread_data, output_data, local_fft_id);
}

// --- Strided C2C convolution (or cross-correlation) Kernel with Padding ---
template <class FFT_fwd, class FFT_inv, class InputLayout, class OutputLayout, class ConvDataLayout,
          unsigned int Stride, unsigned int InputSignalLength, unsigned int OutputSignalLength,
          bool CrossCorrelate = false>
__launch_bounds__(FFT_fwd::max_threads_per_block) __global__
    void strided_padded_block_conv_c2c_2d_kernel_with_layout(
        typename FFT_fwd::value_type* data, const typename FFT_fwd::value_type* conv_data,
        typename FFT_fwd::workspace_type workspace_fwd,
        typename FFT_inv::workspace_type workspace_inv) {
    using complex_type = typename FFT_fwd::value_type;

    // TODO: Static assertions to ensure FFT_fwd and FFT_inv are the same
    // except for their direction.

    /* clang-format off */
    using input_io =  zipfft::io_strided_padded_with_layout<FFT_fwd, InputLayout,    InputLayout,   Stride,  InputSignalLength>;
    using output_io = zipfft::io_strided_padded_with_layout<FFT_inv, OutputLayout,   OutputLayout,   Stride, OutputSignalLength>;
    // using conv_io =   zipfft::io_strided_with_layout<       FFT_fwd, ConvDataLayout, ConvDataLayout, Stride>;
    /* clang-format on */
    ConvDataLayout conv_index_mapper;

    const unsigned int local_fft_id = threadIdx.y;
    const unsigned int apparent_ffts_per_block = input_io::apparent_ffts_per_block;
    const unsigned int global_fft_id = blockIdx.x * apparent_ffts_per_block + local_fft_id;
    const unsigned int column_id = global_fft_id % Stride;

    // Local array for FFT thread execution
    complex_type thread_data[FFT_fwd::storage_size];
    input_io::load(data, thread_data, local_fft_id);

    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT_fwd().execute(thread_data, shared_mem, workspace_fwd);

    // Point-wise multiply in the frequency domain

#pragma unroll
    for (unsigned int i = 0; i < FFT_fwd::output_ept; ++i) {
        const unsigned int elem_id = threadIdx.x + i * FFT_fwd::stride;
        const unsigned int conv_index = conv_index_mapper(column_id, elem_id, 0);

        const float2 a = reinterpret_cast<float2*>(thread_data)[i];
        const float2 b = __ldg(&reinterpret_cast<const float2*>(conv_data)[conv_index]);
        float2 c;

        // computing c = a * b       (convolution)
        // or        c = conj(a) * b (cross-correlation)
        if (CrossCorrelate) {
            c.x = a.x * b.x + a.y * b.y;
            c.y = -a.y * b.x + a.x * b.y;
        } else {
            c.x = a.x * b.x - a.y * b.y;
            c.y = a.x * b.y + a.y * b.x;
        }
        reinterpret_cast<float2*>(thread_data)[i] = c;
    }

    FFT_inv().execute(thread_data, shared_mem, workspace_inv);

    output_io::store(thread_data, data, local_fft_id);
}

// --- Convolution/Cross-correlation 2D FFT Launcher ---
/**
 * @brief Launcher function for the 3-kernel padded real 2D convolution/cross-correlation
 *
 * @tparam Arch - CUDA Architecture specifier for cuFFTDx
 * @tparam FFTSizeX - FFT size in the X dimension
 * @tparam FFTSizeY - FFT size in the Y dimension
 * @tparam Batch - Number of batches
 * @tparam SignalLengthX - Length of the input signal in the X dimension
 * @tparam SignalLengthY - Length of the input signal in the Y dimension
 * @tparam elements_per_thread_x - Number of elements processed per thread in the X dimension. If 0
 * then use the cuFFTDx recommended elements per thread value.
 * @tparam elements_per_thread_y - Number of elements processed per thread in the Y dimension. If 0
 * then use the cuFFTDx recommended elements per thread value.
 * @tparam FFTs_per_block_x - Number of FFTs processed per block in the X dimension. If 0 then use
 * the cuFFTDx recommended FFTs per block value.
 * @tparam FFTs_per_block_y - Number of FFTs processed per block in the Y dimension. If 0 then use
 * the cuFFTDx recommended FFTs per block value.
 * @tparam CrossCorrelate - Whether to perform cross-correlation (true) or convolution (false)
 * @param input_data - Pointer to the input data. Assumed to be in row-major order with shape
 * (Batch, SignalLengthY, SignalLengthX)
 * @param fft_workspace - Pointer to the FFT workspace (complex type). Assumed to be in row-major
 * order with shape (Batch, FFTSizeY, FFTSizeX / 2 + 1)
 * @param conv_data - Pointer to the convolution data. Assumed to be in row-major order with shape
 * (1, FFTSizeY, FFTSizeX / 2 + 1)
 * @param output_data - Pointer to the output data. Assumed to be in row-major order with shape
 * (Batch, FFTSizeY - SignalLengthY + 1, FFTSizeX - SignalLengthX + 1)
 */
template <unsigned int Arch, unsigned int FFTSizeX, unsigned int FFTSizeY, unsigned int Batch,
          unsigned int SignalLengthX, unsigned int SignalLengthY,
          unsigned int elements_per_thread_x = 0, unsigned int elements_per_thread_y = 0,
          unsigned int FFTs_per_block_x = 0, unsigned int FFTs_per_block_y = 0,
          bool CrossCorrelate = false>
inline void padded_block_real_conv_2d_launcher(float* input_data, float2* fft_workspace,
                                               const float2* conv_data, float* output_data) {
    using namespace cufftdx;

    // 1. FFT Structure Definitions
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using FFT_minimal = decltype(Block() + Precision<float>() + SM<Arch>());
    using FFTX_base = decltype(FFT_minimal() + Size<FFTSizeX>());
    using FFTY_base = decltype(FFT_minimal() + Size<FFTSizeY>() + Type<fft_type::c2c>());

    using FFTX_fwd_base = decltype(FFTX_base() + Type<fft_type::r2c>() +
                                   Direction<fft_direction::forward>() + real_fft_options());
    using FFTX_inv_base = decltype(FFTX_base() + Type<fft_type::c2r>() +
                                   Direction<fft_direction::inverse>() + real_fft_options());

    using FFTY_fwd_base = decltype(FFTY_base() + Direction<fft_direction::forward>());
    using FFTY_inv_base = decltype(FFTY_base() + Direction<fft_direction::inverse>());

    // Apply optional elements per thread
    using FFTX_fwd_ept =
        std::conditional_t<elements_per_thread_x != 0,
                           decltype(FFTX_fwd_base() + ElementsPerThread<elements_per_thread_x>()),
                           FFTX_fwd_base>;
    using FFTX_inv_ept =
        std::conditional_t<elements_per_thread_x != 0,
                           decltype(FFTX_inv_base() + ElementsPerThread<elements_per_thread_x>()),
                           FFTX_inv_base>;
    using FFTY_fwd_ept =
        std::conditional_t<elements_per_thread_y != 0,
                           decltype(FFTY_fwd_base() + ElementsPerThread<elements_per_thread_y>()),
                           FFTY_fwd_base>;
    using FFTY_inv_ept =
        std::conditional_t<elements_per_thread_y != 0,
                           decltype(FFTY_inv_base() + ElementsPerThread<elements_per_thread_y>()),
                           FFTY_inv_base>;

    // Apply optional FFTs per block
    using FFTX_fwd = std::conditional_t<FFTs_per_block_x != 0,
                                        decltype(FFTX_fwd_ept() + FFTsPerBlock<FFTs_per_block_x>()),
                                        FFTX_fwd_ept>;
    using FFTX_inv = std::conditional_t<FFTs_per_block_x != 0,
                                        decltype(FFTX_inv_ept() + FFTsPerBlock<FFTs_per_block_x>()),
                                        FFTX_inv_ept>;
    using FFTY_fwd = std::conditional_t<FFTs_per_block_y != 0,
                                        decltype(FFTY_fwd_ept() + FFTsPerBlock<FFTs_per_block_y>()),
                                        FFTY_fwd_ept>;
    using FFTY_inv = std::conditional_t<FFTs_per_block_y != 0,
                                        decltype(FFTY_inv_ept() + FFTsPerBlock<FFTs_per_block_y>()),
                                        FFTY_inv_ept>;

    using complex_type = typename FFTX_fwd::value_type;
    using scalar_type = typename complex_type::value_type;

    // 3. Data layouts used for accessing data throughout the transformation
    // NOTE: We are doing a batched set of filter transformations (multiple image kernels)
    //       which are being cross-correlated with a single image (conv_data).
    //       Using batch dimension to effectively broadcast conv_data across all
    //       FFT batch dimensions.
    // NOTE: Output data are always being cropped to valid convolution/cross-correlation
    //       shape where input signal 'n' is being zero-padded to length 'N'. Therefore
    //       valid region is 'N - n + 1' (for output_data)
    const unsigned int StrideY = FFTX_fwd::output_length;
    const unsigned int ValidLengthX = FFTSizeX - SignalLengthX + 1;
    const unsigned int ValidLengthY = FFTSizeY - SignalLengthY + 1;
    using FwdInputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<SignalLengthX, 1>,
                             zipfft::int_pair<SignalLengthY, SignalLengthX>,
                             zipfft::int_pair<Batch, SignalLengthY * SignalLengthX>>;
    using FwdOutputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<StrideY, 1>, zipfft::int_pair<SignalLengthY, StrideY>,
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>;

    using FwdInputLayoutY =
        zipfft::index_mapper<zipfft::int_pair<FFTSizeY, StrideY>, zipfft::int_pair<StrideY, 1>,
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>;
    using FwdOutputLayoutY =
        zipfft::index_mapper<zipfft::int_pair<FFTSizeY, StrideY>, zipfft::int_pair<StrideY, 1>,
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>;

    using ConvDataLayout =
        zipfft::index_mapper<zipfft::int_pair<StrideY, FFTSizeY>, zipfft::int_pair<FFTSizeY, 1>,
                             zipfft::int_pair<Batch, 0>>;  // Broadcast across batches

    using InvInputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<StrideY, 1>, zipfft::int_pair<ValidLengthY, StrideY>,
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>;
    using InvOutputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<ValidLengthX, 1>,
                             zipfft::int_pair<ValidLengthY, ValidLengthX>,
                             zipfft::int_pair<Batch, ValidLengthY * ValidLengthX>>;

    // 4. Construct the kernel pointers and associated attributes
    auto kernel_r2c_x = padded_block_fft_r2c_1d_kernel_with_layout<FFTX_fwd, FwdInputLayoutX,
                                                                   FwdOutputLayoutX, SignalLengthX>;
    auto kernel_c2c_y = strided_padded_block_conv_c2c_2d_kernel_with_layout<
        FFTY_fwd, FFTY_inv, FwdInputLayoutY, FwdOutputLayoutY, ConvDataLayout, StrideY,
        SignalLengthY, ValidLengthY, CrossCorrelate>;
    auto kernel_c2r_x = padded_block_fft_c2r_2d_kernel_with_layout<FFTX_inv, InvInputLayoutX,
                                                                   InvOutputLayoutX, ValidLengthX>;

    // Increase shared memory size to maximum of the three kernels
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_r2c_x, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTX_fwd::shared_memory_size));
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_c2c_y, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY_fwd::shared_memory_size));
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_c2r_x, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTX_inv::shared_memory_size));

    // Create workspace for FFTs
    cudaError_t workspace_error = cudaSuccess;
    auto workspace_fwd_x = make_workspace<FFTX_fwd>(workspace_error);
    CUDA_CHECK_AND_EXIT(workspace_error);
    auto workspace_fwd_y = make_workspace<FFTY_fwd>(workspace_error);
    CUDA_CHECK_AND_EXIT(workspace_error);
    auto workspace_inv_y = make_workspace<FFTY_inv>(workspace_error);
    CUDA_CHECK_AND_EXIT(workspace_error);
    auto workspace_inv_x = make_workspace<FFTX_inv>(workspace_error);
    CUDA_CHECK_AND_EXIT(workspace_error);

    // 5. Launch the three kernels in sequence
    const unsigned int grid_size_fwd_x =
        ((Batch * SignalLengthY) + FFTX_fwd::ffts_per_block - 1) / FFTX_fwd::ffts_per_block;
    const unsigned int grid_size_fwd_y =
        ((Batch * StrideY) + FFTY_fwd::ffts_per_block - 1) / FFTY_fwd::ffts_per_block;
    const unsigned int grid_size_inv_x =
        ((Batch * ValidLengthY) + FFTX_inv::ffts_per_block - 1) / FFTX_inv::ffts_per_block;

    // Cast the input data into cuFFTDx types
    scalar_type* input_data_cast = reinterpret_cast<scalar_type*>(input_data);
    complex_type* fft_workspace_cast = reinterpret_cast<complex_type*>(fft_workspace);
    const complex_type* conv_data_cast = reinterpret_cast<const complex_type*>(conv_data);
    scalar_type* output_data_cast = reinterpret_cast<scalar_type*>(output_data);

    kernel_r2c_x<<<grid_size_fwd_x, FFTX_fwd::block_dim, FFTX_fwd::shared_memory_size>>>(
        input_data_cast, fft_workspace_cast, workspace_fwd_x);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    kernel_c2c_y<<<grid_size_fwd_y, FFTY_fwd::block_dim, FFTY_fwd::shared_memory_size>>>(
        fft_workspace_cast, conv_data_cast, workspace_fwd_y, workspace_inv_y);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    kernel_c2r_x<<<grid_size_inv_x, FFTX_inv::block_dim, FFTX_inv::shared_memory_size>>>(
        fft_workspace_cast, output_data_cast, workspace_inv_x);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    // // TODO: Remove explicit sync
    // CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
template <typename ScalarType, typename ComplexType, unsigned int SignalLengthX,
          unsigned int SignalLengthY, unsigned int FFTSizeX, unsigned int FFTSizeY,
          unsigned int Batch, bool CrossCorrelate, unsigned int elements_per_thread_x = 0,
          unsigned int elements_per_thread_y = 0, unsigned int FFTs_per_block_x = 0,
          unsigned int FFTs_per_block_y = 0>
int padded_block_real_conv_2d(ScalarType* input_data, ComplexType* fft_workspace,
                              const ComplexType* conv_data, ScalarType* output_data) {
    auto arch = zipfft::get_cuda_device_arch();

    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: padded_block_real_conv_2d_launcher<800, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: padded_block_real_conv_2d_launcher<860, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: padded_block_real_conv_2d_launcher<870, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: padded_block_real_conv_2d_launcher<890, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: padded_block_real_conv_2d_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        case 1200: padded_block_real_conv_2d_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate>(input_data, fft_workspace, conv_data, output_data); break;
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
