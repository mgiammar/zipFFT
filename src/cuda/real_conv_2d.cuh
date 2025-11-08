#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_index_mapper.hpp"
#include "../include/zipfft_padded_io.hpp"
#include "../include/zipfft_strided_io.hpp"
#include "../include/zipfft_strided_padded_io.hpp"

// --- Forward r2c Kernel Definition with Index Mappers ---
/**
 * @brief Forward real-to-complex 1D FFT kernel where input is zero-padded from SignalLength to FFT
 * size. Supports custom input and output layouts via index mappers as well as transposition of
 * input and output data.
 *
 * @tparam FFT - cuFFTDx FFT type
 * @tparam InputLayout - Layout of input data as IndexMapper struct
 * @tparam OutputLayout - Layout of output data as IndexMapper struct
 * @tparam SignalLength - Length of the real input signal (before padding)
 * @tparam StoreTransposed - Wether to store the output data in a transposed layout (potential
 * memory coalescing optimization) using 'StoreStride'. Default is false.
 * @tparam StoreStride - Stride used when storing transposed output data. Default is 1, and is only
 * used if 'StoreTransposed' is true.
 * @tparam FFT::value_type - Complex type used by the FFT
 * @tparam ComplexType::value_type - Scalar type used by the FFT
 * @param input_data - Pointer to the real input data
 * @param output_data - Pointer to the complex output data
 * @param workspace - cuFFTDx workspace for the FFT
 */
template <class FFT, class InputLayout, class OutputLayout, unsigned int SignalLength,
          bool StoreTransposed = false, unsigned int StoreStride = 1,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_r2c_1d_kernel_with_layout(ScalarType* input_data,
                                                    ComplexType* output_data,
                                                    typename FFT::workspace_type workspace) {
    using complex_type = ComplexType;
    using scalar_type = ScalarType;

    // Input is padded (zero-pads to FFT size)
    // Output uses standard layout
    // InputLayout (second template arg) only matters for the input_io
    // OutputLayout (third template arg) only matters for the output_io
    using input_io = zipfft::io_padded_with_layout<FFT, InputLayout, InputLayout, SignalLength>;

    // /// DEBUGGING: Print the StoreStride value
    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("StoreStride = %u\n", StoreStride);
    // }

    // If `StoreTransposed` is true, then store data into 'output_data' using a transposed layout
    using output_io = std::conditional_t<
        StoreTransposed,
        zipfft::io_strided_with_layout<FFT, OutputLayout, OutputLayout, StoreStride>,
        zipfft::io_with_layout<FFT, OutputLayout, OutputLayout>>;

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
/**
 * @brief Inverse complex-to-real 2D FFT kernel where output is truncated from FFT size to
 * SignalLength. Supports custom input and output layouts via index mappers.
 *
 * @tparam FFT - cuFFTDx FFT type
 * @tparam InputLayout - Layout of input data as IndexMapper struct
 * @tparam OutputLayout - Layout of output data as IndexMapper struct
 * @tparam SignalLength - Length of the real output signal (after truncation)
 * @tparam LoadTransposed - Wether to load the input data in a transposed layout (potential memory
 * coalescing optimization) using 'LoadStride'. Default is false.
 * @tparam LoadStride - Stride used when loading input data in a transposed layout. Default is 1.
 * @tparam FFT::value_type - Complex type used by the FFT
 * @tparam ComplexType::value_type - Scalar type used by the FFT
 */
template <class FFT, class InputLayout, class OutputLayout, unsigned int SignalLength,
          bool LoadTransposed = false, unsigned int LoadStride = 1,
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
    using input_io = std::conditional_t<
        LoadTransposed,
        zipfft::io_strided_with_layout<FFT, InputLayout, InputLayout, LoadStride>,  // Transposed
        zipfft::io_with_layout<FFT, InputLayout, InputLayout>>;
    using output_io = zipfft::io_padded_with_layout<FFT, OutputLayout, OutputLayout, SignalLength>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using input layout (full complex input)
    input_io::load(input_data, thread_data, local_fft_id);

    // /// DEBUGGING: Print the contents of thread_data for each thread
    // for (unsigned int i = 0; i < FFT::storage_size; ++i) {
    //     if (blockIdx.x == 0) {
    //         printf("Block: (%u, %u, %u) Thread (%u, %u) thread_data[%u] = (%f, %f)\n",
    //         blockIdx.x,
    //                blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, i, thread_data[i].x,
    //                thread_data[i].y);
    //     }
    // }

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Store using padded output layout (truncates to SignalLength)
    output_io::store(thread_data, output_data, local_fft_id);
}

// --- Strided C2C convolution (or cross-correlation) Kernel with Padding ---
/**
 * @brief Fused forward C2C, point-wise multiply, and inverse C2C kernel for 2D
 * convolution/cross-correlation. Supports custom input, output, and convolution data layouts
 * via index mappers as well as strided data access and padding.
 *
 * @tparam FFT_fwd - cuFFTDx FFT type for forward transform
 * @tparam FFT_inv - cuFFTDx FFT type for inverse transform
 * @tparam InputLayout - Layout of input data as IndexMapper struct
 * @tparam OutputLayout - Layout of output data as IndexMapper struct
 * @tparam ConvDataLayout - Layout of convolution data as IndexMapper struct
 * @tparam Stride - Stride used for data access
 * @tparam InputSignalLength - Length of the input signal
 * @tparam OutputSignalLength - Length of the output signal
 * @tparam CrossCorrelate - Whether to perform cross-correlation (true) or convolution (false)
 * @tparam WorkspaceIsTransposed - Whether the FFT workspace is transposed. If true, then the
 * data can be read/stored in a contiguous manner for the C2C FFTs.
 * @tparam ConvDataIsTransposed - Whether the convolution data has been pre-transposed. If true,
 * then the convolution data will be accessed in a contiguous manner.
 * @param data - Pointer to the input/output data (in-place operation)
 * @param conv_data - Pointer to the convolution data
 * @param workspace_fwd - cuFFTDx workspace for the forward FFT
 * @param workspace_inv - cuFFTDx workspace for the inverse FFT
 */
template <class FFT_fwd, class FFT_inv, class InputLayout, class OutputLayout, class ConvDataLayout,
          unsigned int Stride, unsigned int InputSignalLength, unsigned int OutputSignalLength,
          bool CrossCorrelate = false, bool WorkspaceIsTransposed = false,
          bool ConvDataIsTransposed = false>
__launch_bounds__(FFT_fwd::max_threads_per_block) __global__
    void strided_padded_block_conv_c2c_2d_kernel_with_layout(
        typename FFT_fwd::value_type* data, const typename FFT_fwd::value_type* conv_data,
        typename FFT_fwd::workspace_type workspace_fwd,
        typename FFT_inv::workspace_type workspace_inv) {
    using complex_type = typename FFT_fwd::value_type;

    // TODO: Static assertions to ensure FFT_fwd and FFT_inv are the same
    // except for their direction.

    /* clang-format off */
    using input_io = std::conditional_t<
        WorkspaceIsTransposed,
        zipfft::io_padded_with_layout<FFT_fwd, InputLayout, InputLayout, InputSignalLength>,
        zipfft::io_strided_padded_with_layout<FFT_fwd, InputLayout, InputLayout, Stride, InputSignalLength>
    >;
    using output_io = std::conditional_t<
        WorkspaceIsTransposed,
        zipfft::io_padded_with_layout<FFT_inv, OutputLayout, OutputLayout, OutputSignalLength>,
        zipfft::io_strided_padded_with_layout<FFT_inv, OutputLayout, OutputLayout, Stride, OutputSignalLength>
    >;
    /* clang-format on */
    ConvDataLayout conv_index_mapper;

    // Constants for accessing convolution data
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
        const unsigned int conv_index = conv_index_mapper(elem_id, column_id, 0);

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
 * @tparam elements_per_thread_x - Number of elements processed per thread in the X dimension.
 * If 0 then use the cuFFTDx recommended elements per thread value.
 * @tparam elements_per_thread_y - Number of elements processed per thread in the Y dimension.
 * If 0 then use the cuFFTDx recommended elements per thread value.
 * @tparam FFTs_per_block_x - Number of FFTs processed per block in the X dimension. If 0 then
 * use the cuFFTDx recommended FFTs per block value.
 * @tparam FFTs_per_block_y - Number of FFTs processed per block in the Y dimension. If 0 then
 * use the cuFFTDx recommended FFTs per block value.
 * @tparam CrossCorrelate - Whether to perform cross-correlation (true) or convolution (false)
 * @param input_data - Pointer to the input data. Assumed to be in row-major order with shape
 * (Batch, SignalLengthY, SignalLengthX)
 * @param fft_workspace - Pointer to the FFT workspace (complex type). Assumed to be in
 * row-major order with shape (Batch, FFTSizeY, FFTSizeX / 2 + 1)
 * @param conv_data - Pointer to the convolution data. Assumed to be in row-major order with
 * shape (1, FFTSizeY, FFTSizeX / 2 + 1)
 * @param output_data - Pointer to the output data. Assumed to be in row-major order with shape
 * (Batch, FFTSizeY - SignalLengthY + 1, FFTSizeX - SignalLengthX + 1)
 */
template <unsigned int Arch, unsigned int FFTSizeX, unsigned int FFTSizeY, unsigned int Batch,
          unsigned int SignalLengthX, unsigned int SignalLengthY,
          unsigned int elements_per_thread_x = 0, unsigned int elements_per_thread_y = 0,
          unsigned int FFTs_per_block_x = 0, unsigned int FFTs_per_block_y = 0,
          bool CrossCorrelate = false, bool TransposeAxes = false,
          bool ConvDataIsTransposed = false>
inline void padded_block_real_conv_2d_launcher(float* input_data, float2* fft_workspace,
                                               const float2* conv_data, float* output_data) {
    using namespace cufftdx;

    // 1. FFT Structures for transforms operating along the X dimension
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using FFT_minimal = decltype(Block() + Precision<float>() + SM<Arch>());
    using FFTX_base = decltype(FFT_minimal() + Size<FFTSizeX>());
    using FFTY_base = decltype(FFT_minimal() + Size<FFTSizeY>() + Type<fft_type::c2c>());

    using FFTX_fwd = decltype(FFTX_base() + Type<fft_type::r2c>() +
                              Direction<fft_direction::forward>() + real_fft_options());
    using FFTX_inv = decltype(FFTX_base() + Type<fft_type::c2r>() +
                              Direction<fft_direction::inverse>() + real_fft_options());

    using FFTY_fwd = decltype(FFTY_base() + Direction<fft_direction::forward>());
    using FFTY_inv = decltype(FFTY_base() + Direction<fft_direction::inverse>());

    if constexpr (elements_per_thread_x != 0) {
        using FFTX_fwd = decltype(FFTX_fwd() + ElementsPerThread<elements_per_thread_x>());
        using FFTX_inv = decltype(FFTX_inv() + ElementsPerThread<elements_per_thread_x>());
    }
    if constexpr (elements_per_thread_y != 0) {
        using FFTY_fwd = decltype(FFTY_fwd() + ElementsPerThread<elements_per_thread_y>());
        using FFTY_inv = decltype(FFTY_inv() + ElementsPerThread<elements_per_thread_y>());
    }

    if constexpr (FFTs_per_block_x != 0) {
        using FFTX_fwd = decltype(FFTX_fwd() + FFTsPerBlock<FFTs_per_block_x>());
        using FFTX_inv = decltype(FFTX_inv() + FFTsPerBlock<FFTs_per_block_x>());
    }
    if constexpr (FFTs_per_block_y != 0) {
        using FFTY_fwd = decltype(FFTY_fwd() + FFTsPerBlock<FFTs_per_block_y>());
        using FFTY_inv = decltype(FFTY_inv() + FFTsPerBlock<FFTs_per_block_y>());
    }

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

    // Compile-time decision if to operate on a transposed set of axes for the
    // FFT. If true, then the workspace data is assumed to be transposed and the
    // R2C/C2R FFTs along the X dimension will load/store the data in a
    // transposed layout. The C2C FFT along the Y dimension then assumes a
    // contigious access pattern. When true, then the following happen in order
    // 1. R2C FFT along X dimension with: contiguous input, strided output
    // 2. C2C FFT along Y dimension with: contigious input, contigious output
    // 3. C2R FFT along X dimension with: strided input, contiguous output
    //
    // If false, then the following happen in order
    // 1. R2C FFT along X dimension with: contiguous input, contiguous output
    // 2. C2C FFT along Y dimension with: strided input, strided output
    // 3. C2R FFT along X dimension with: contiguous input, contiguous output
    constexpr bool StoreTransposed = TransposeAxes;  // Store for R2C kernel
    constexpr bool LoadTransposed = TransposeAxes;   // Load for C2R kernel
    // constexpr unsigned int StoreStride = StrideY;
    // constexpr unsigned int LoadStride = StrideY;
    constexpr unsigned int StoreStride = SignalLengthY;  // For transposed storage on R2C
    constexpr unsigned int LoadStride = ValidLengthY;    // For transposed loading on C2R

    using FwdInputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<SignalLengthX, 1>,
                             zipfft::int_pair<SignalLengthY, SignalLengthX>,
                             zipfft::int_pair<Batch, SignalLengthY * SignalLengthX>>;

    // Compile-time decision to transpose the storage layout for the R2C kernel
    using FwdOutputLayoutX = std::conditional_t<
        TransposeAxes,
        zipfft::index_mapper<zipfft::int_pair<StrideY, 1>,
                             zipfft::int_pair<SignalLengthY, FFTSizeY>,  // This one is correct
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>,
        zipfft::index_mapper<zipfft::int_pair<StrideY, 1>,
                             zipfft::int_pair<SignalLengthY, StrideY>,  // Non-transposed
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>>;

    // Compile-time decision to transpose the layout for the workspace (input/output)
    // of C2C kernel along the Y dimension
    using FwdInputLayoutY =
        std::conditional_t<TransposeAxes,
                           zipfft::index_mapper<zipfft::int_pair<FFTSizeY, 1>,
                                                zipfft::int_pair<StrideY, FFTSizeY>,  // read rows
                                                zipfft::int_pair<Batch, FFTSizeY * StrideY>>,
                           zipfft::index_mapper<zipfft::int_pair<FFTSizeY, StrideY>,
                                                zipfft::int_pair<StrideY, 1>,  // read cols
                                                zipfft::int_pair<Batch, FFTSizeY * StrideY>>>;

    using FwdOutputLayoutY = FwdInputLayoutY;

    // Compile-time decision to transpose the loading layout for the C2R kernel
    using InvInputLayoutX = std::conditional_t<
        TransposeAxes,
        zipfft::index_mapper<zipfft::int_pair<StrideY, 1>,
                             zipfft::int_pair<ValidLengthY, FFTSizeY>,  // Transposed
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>,
        zipfft::index_mapper<zipfft::int_pair<StrideY, 1>,
                             zipfft::int_pair<ValidLengthY, StrideY>,  // Non-transposed
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>>;
    using InvOutputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<ValidLengthX, 1>,
                             zipfft::int_pair<ValidLengthY, ValidLengthX>,
                             zipfft::int_pair<Batch, ValidLengthY * ValidLengthX>>;

    // NOTE: Unsure if the ConvDataLayout needs to be transposed or if the fused C2C kernel
    // is handling this automatically...
    using ConvDataLayout = std::conditional_t<
        TransposeAxes,
        // zipfft::index_mapper<zipfft::int_pair<FFTSizeY, 1>, zipfft::int_pair<StrideY, FFTSizeY>,
        zipfft::index_mapper<zipfft::int_pair<FFTSizeY, 1>,  // If pre-transposed, access contig
                             zipfft::int_pair<StrideY, FFTSizeY>,  // layout of rows here
                             zipfft::int_pair<Batch, 0>>,
        zipfft::index_mapper<zipfft::int_pair<FFTSizeY, 1>,        // layout of cols here
                             zipfft::int_pair<StrideY, FFTSizeY>,  // If not, stride across cols
                             zipfft::int_pair<Batch, 0>>>;

    // 4. Construct the kernel pointers and associated attributes
    auto kernel_r2c_x =
        padded_block_fft_r2c_1d_kernel_with_layout<FFTX_fwd, FwdInputLayoutX, FwdOutputLayoutX,
                                                   SignalLengthX, StoreTransposed, StoreStride>;
    auto kernel_c2c_y = strided_padded_block_conv_c2c_2d_kernel_with_layout<
        FFTY_fwd, FFTY_inv, FwdInputLayoutY, FwdOutputLayoutY, ConvDataLayout, StrideY,
        SignalLengthY, ValidLengthY, CrossCorrelate, TransposeAxes, ConvDataIsTransposed>;
    auto kernel_c2r_x =
        padded_block_fft_c2r_2d_kernel_with_layout<FFTX_inv, InvInputLayoutX, InvOutputLayoutX,
                                                   ValidLengthX, LoadTransposed, LoadStride>;

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
          unsigned int FFTs_per_block_y = 0, bool TransposeAxes = false,
          bool ConvDataIsTransposed = false>
int padded_block_real_conv_2d(ScalarType* input_data, ComplexType* fft_workspace,
                              const ComplexType* conv_data, ScalarType* output_data) {
    auto arch = zipfft::get_cuda_device_arch();

    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: padded_block_real_conv_2d_launcher<800, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, TransposeAxes, ConvDataIsTransposed>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: padded_block_real_conv_2d_launcher<860, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, TransposeAxes, ConvDataIsTransposed>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: padded_block_real_conv_2d_launcher<870, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, TransposeAxes, ConvDataIsTransposed>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: padded_block_real_conv_2d_launcher<890, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, TransposeAxes, ConvDataIsTransposed>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: padded_block_real_conv_2d_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, TransposeAxes, ConvDataIsTransposed>(input_data, fft_workspace, conv_data, output_data); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        case 1200: padded_block_real_conv_2d_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, TransposeAxes, ConvDataIsTransposed>(input_data, fft_workspace, conv_data, output_data); break;
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
