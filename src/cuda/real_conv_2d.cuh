/**
 * @file real_conv_2d.cuh
 * @author Matthew D. Giammar (mdgiammar@gmail.com)
 * @brief Real 2D convolution/cross-correlation using cuFFTDx with IO handlers optimized for the
 * particular problem at hand (zero-padding, memory coalescing along strided dimensions, etc.).
 * @version 0.1
 * @date 2025-11-09
 * @copyright Copyright (c) 2025
 *
 * The implementation in this file seeks to optimize the computational problem of cross-correlating
 * a varying input filter (with large dimensions) with an unchanging image (conv_data). This boils
 * down to IRFFT2D(RFFT2D(image) * conj(Padded_RFFT2D(filter))). The following are the main
 * optimizations made by this implementation over just a cuFFT implementation:
 * 1. In-register zero-padding: We know the FFT size and signal length at compile time which means
 *    we also know which elements need to be zero-padded during loads. Rather than reading these
 *    zeros from global memory, we use optimized IO functions to conditionally read values from
 *    memory placing zeros during the load if necessary. See 'strided_io_smem_test.hpp' for more
 *    details on optimized IO.
 * 2. Fused C2C transform + conjugate multiply kernel: Cross-correlation (or convolution) can be
 *    performed within the same kernel as the forward and inverse C2C transforms which saves on
 *    global memory writes. Data remains in thread registers, and convolution data is read in and
 *    multiplied directly within this thread data.
 * 3. Coalesced convolution data access: Since the image data is unchanging, we can pre-transpose it
 *    from shape (H, W // 2 + 1) to (W // 2 + 1, H) such that memory access becomes contiguous.
 * 4. Shared-memory transposition between R2C & C2C as well as C2C & C2R kernels: Since cuFFTDx
 *    is optimized for single row/col execution per block (in the regime of FFT size 4096), we want
 *    to make all memory reads/writes for the FFT kernels contiguous. The restrictions to single
 *    row/col per block means it's impossible to easily split the problem which operate over tiles
 *    of the input data from global memory. Instead, we introduce an intermediary shared-memory
 *    transposition kernel using memory swizzling to achieve highly coalesced memory access (see
 *    file 'src/include/zipfft_smem_transpose.hpp' for more details).
 */

#ifndef ZIPFFT_REAL_CONV_2D_CUH
#define ZIPFFT_REAL_CONV_2D_CUH

#include <cufftdx.hpp>

#include "../include/zipfft_common.hpp"
#include "../include/zipfft_smem_transpose.hpp"
#include "strided_io_smem_test.hpp"

// --- Forward r2c Kernel Definition with Optimized IO ---
/**
 * @brief Forward real-to-complex 1D FFT kernel using optimized IO handler.
 *
 * @tparam FFT - cuFFTDx R2C FFT type
 * @tparam IO_Handler - Optimized IO handler for X dimension
 * @tparam ComplexType - Complex type used by the FFT
 * @tparam ScalarType - Scalar type used by the FFT
 * @param input_data - Pointer to the real input data
 * @param output_data - Pointer to the complex output data
 * @param workspace - cuFFTDx workspace for the FFT
 */
template <class FFT, class IO_Handler, typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void optimized_r2c_kernel(ScalarType* input_data, ComplexType* output_data,
                              typename FFT::workspace_type workspace) {
    using complex_type = ComplexType;
    using scalar_type = ScalarType;

    // Allocate shared and register memory
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    complex_type thread_data[FFT::storage_size];

    // Create IO handler instance
    IO_Handler io_handler;

    // Load: gmem -> (smem) -> rmem
    io_handler.load_gmem_to_rmem(input_data, shared_mem, thread_data);

    // Execute FFT
    FFT().execute(thread_data, shared_mem, workspace);

    // Store: rmem -> (smem) -> gmem
    io_handler.store_rmem_to_gmem(thread_data, shared_mem, output_data);
}

// --- Inverse c2r Kernel Definition with Optimized IO ---
/**
 * @brief Inverse complex-to-real 2D FFT kernel using optimized IO handler.
 *
 * @tparam FFT - cuFFTDx C2R FFT type
 * @tparam IO_Handler - Optimized IO handler for X dimension
 * @tparam ComplexType - Complex type used by the FFT
 * @tparam ScalarType - Scalar type used by the FFT
 * @param input_data - Pointer to the complex input data
 * @param output_data - Pointer to the real output data
 * @param workspace - cuFFTDx workspace for the FFT
 */
template <class FFT, class IO_Handler, typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void optimized_c2r_kernel(ComplexType* input_data, ScalarType* output_data,
                              typename FFT::workspace_type workspace) {
    using complex_type = ComplexType;
    using scalar_type = ScalarType;

    // Allocate shared and register memory
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    complex_type thread_data[FFT::storage_size];

    // Create IO handler instance
    IO_Handler io_handler;

    // Load: gmem -> (smem) -> rmem
    io_handler.load_gmem_to_rmem(input_data, shared_mem, thread_data);

    // Execute FFT
    FFT().execute(thread_data, shared_mem, workspace);

    // Store: rmem -> (smem) -> gmem
    io_handler.store_rmem_to_gmem(thread_data, shared_mem, output_data);
}

// --- C2C Convolution Kernel with Optimized IO ---
/**
 * @brief Fused forward C2C, point-wise multiply, and inverse C2C kernel for 2D
 * convolution/cross-correlation using optimized IO handler.
 *
 * @tparam FFT_fwd - cuFFTDx FFT type for forward transform
 * @tparam FFT_inv - cuFFTDx FFT type for inverse transform
 * @tparam IO_Handler - Optimized IO handler for Y dimension
 * @tparam ConvDataLayout - Layout of convolution data as IndexMapper struct
 * @tparam StrideY - Stride for Y dimension (FFT output length in X)
 * @tparam CrossCorrelate - Whether to perform cross-correlation (true) or convolution (false)
 * @tparam ConvDataIsTransposed - Whether the convolution data has been pre-transposed
 * @param input_data - Pointer to the input data
 * @param output_data - Pointer to the output data
 * @param conv_data - Pointer to the convolution data
 * @param workspace_fwd - cuFFTDx workspace for the forward FFT
 * @param workspace_inv - cuFFTDx workspace for the inverse FFT
 */
template <class FFT_fwd, class FFT_inv, class IO_Handler_fwd, class IO_Handler_inv,
          class ConvDataLayout, unsigned int StrideY, bool CrossCorrelate = false,
          bool ConvDataIsTransposed = false>
__launch_bounds__(FFT_fwd::max_threads_per_block) __global__
    void optimized_c2c_conv_kernel(typename FFT_fwd::value_type* input_data,
                                   typename FFT_fwd::value_type* output_data,
                                   typename FFT_fwd::value_type* conv_data,
                                   typename FFT_fwd::workspace_type workspace_fwd,
                                   typename FFT_inv::workspace_type workspace_inv) {
    using complex_type = typename FFT_fwd::value_type;

    // Allocate shared and register memory
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    complex_type thread_data[FFT_fwd::storage_size];

    // Create IO handler and convolution layout instances
    IO_Handler_fwd io_handler_fwd;
    IO_Handler_inv io_handler_inv;
    ConvDataLayout conv_index_mapper;

    // Constants for accessing convolution data
    const unsigned int local_fft_id = threadIdx.y;
    const unsigned int global_fft_id = blockIdx.x * FFT_fwd::ffts_per_block + local_fft_id;
    const unsigned int column_id = global_fft_id % StrideY;

    // Load: gmem -> smem -> rmem
    io_handler_fwd.load_gmem_to_rmem(input_data, shared_mem, thread_data);

    // Forward FFT
    FFT_fwd().execute(thread_data, shared_mem, workspace_fwd);

// Point-wise multiply in the frequency domain
#pragma unroll
    for (unsigned int i = 0; i < FFT_fwd::elements_per_thread; ++i) {
        const unsigned int elem_id = threadIdx.x + i * FFT_fwd::stride;
        const unsigned int conv_index = conv_index_mapper(elem_id, column_id, 0);

        const float2 a = reinterpret_cast<float2*>(thread_data)[i];
        const float2 b = __ldg(&reinterpret_cast<const float2*>(conv_data)[conv_index]);
        float2 c;

        // Computing c = a * b       (convolution)
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

    // Inverse FFT
    FFT_inv().execute(thread_data, shared_mem, workspace_inv);

    // Store: rmem -> smem -> gmem (now to output_data)
    io_handler_inv.store_rmem_to_gmem(thread_data, shared_mem, output_data);
}

// --- Convolution/Cross-correlation 2D FFT Launcher ---
/**
 * @brief Launcher function for the 5-kernel padded real 2D convolution/cross-correlation
 * using optimized IO handlers with shared memory transposes.
 *
 * @tparam Arch - CUDA Architecture specifier for cuFFTDx
 * @tparam FFTSizeX - FFT size in the X dimension
 * @tparam FFTSizeY - FFT size in the Y dimension
 * @tparam Batch - Number of batches
 * @tparam SignalLengthX - Length of the input signal in the X dimension
 * @tparam SignalLengthY - Length of the input signal in the Y dimension
 * @tparam elements_per_thread_x - Number of elements processed per thread in the X dimension
 * @tparam elements_per_thread_y - Number of elements processed per thread in the Y dimension
 * @tparam FFTs_per_block_x - Number of FFTs processed per block in the X dimension
 * @tparam FFTs_per_block_y - Number of FFTs processed per block in the Y dimension
 * @tparam CrossCorrelate - Whether to perform cross-correlation (true) or convolution (false)
 * @tparam ConvDataIsTransposed - Whether the convolution data is pre-transposed
 * @param input_data - Pointer to the input data (Batch, SignalLengthY, SignalLengthX)
 * @param fft_workspace_r2c - Pointer to R2C output workspace (Batch, SignalLengthY, FFTSizeX/2+1)
 * @param fft_workspace_r2c_transposed - Transposed R2C workspace (Batch, FFTSizeX/2+1,
 * SignalLengthY)
 * @param fft_workspace_c2c_transposed - Transposed C2C workspace (Batch, FFTSizeX/2+1, FFTSizeY)
 * @param fft_workspace_c2r - C2R input workspace (Batch, ValidLengthY, FFTSizeX/2+1)
 * @param conv_data - Pointer to the convolution data (1, FFTSizeY, FFTSizeX/2+1)
 * @param output_data - Pointer to the output data (Batch, ValidLengthY, ValidLengthX)
 * @param stream - CUDA stream for asynchronous execution
 */
template <unsigned int Arch, unsigned int FFTSizeX, unsigned int FFTSizeY, unsigned int Batch,
          unsigned int SignalLengthX, unsigned int SignalLengthY,
          unsigned int elements_per_thread_x = 0, unsigned int elements_per_thread_y = 0,
          unsigned int FFTs_per_block_x = 0, unsigned int FFTs_per_block_y = 0,
          bool CrossCorrelate = false, bool ConvDataIsTransposed = false>
inline void padded_block_real_conv_2d_launcher(float* input_data, float2* fft_workspace_r2c,
                                               float2* fft_workspace_r2c_transposed,
                                               float2* fft_workspace_c2c_transposed,
                                               float2* fft_workspace_c2r, float2* conv_data,
                                               float* output_data, cudaStream_t stream = 0) {
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

    // 2. Constants
    const unsigned int StrideY = FFTX_fwd::output_length;
    const unsigned int ValidLengthX = FFTSizeX - SignalLengthX + 1;
    const unsigned int ValidLengthY = FFTSizeY - SignalLengthY + 1;

    // 3. Create Optimized IO Handlers
    using IO_X_fwd = zipfft::io_conv_smem<zipfft::dimension::x, true, Batch, FFTX_fwd, FFTX_inv,
                                          FFTY_fwd, FFTY_inv, SignalLengthX, SignalLengthY>;
    using IO_Y_fwd = zipfft::io_conv_smem<zipfft::dimension::y, true, Batch, FFTX_fwd, FFTX_inv,
                                          FFTY_fwd, FFTY_inv, SignalLengthX, SignalLengthY>;
    using IO_Y_inv = zipfft::io_conv_smem<zipfft::dimension::y, false, Batch, FFTX_fwd, FFTX_inv,
                                          FFTY_fwd, FFTY_inv, SignalLengthX, SignalLengthY>;
    using IO_X_inv = zipfft::io_conv_smem<zipfft::dimension::x, false, Batch, FFTX_fwd, FFTX_inv,
                                          FFTY_fwd, FFTY_inv, SignalLengthX, SignalLengthY>;

    // 4. Convolution Data Layout
    using ConvDataLayout = std::conditional_t<
        ConvDataIsTransposed,
        zipfft::index_mapper<zipfft::int_pair<FFTSizeY, 1>, zipfft::int_pair<StrideY, FFTSizeY>,
                             zipfft::int_pair<Batch, 0>>,
        zipfft::index_mapper<zipfft::int_pair<FFTSizeY, StrideY>, zipfft::int_pair<StrideY, 1>,
                             zipfft::int_pair<Batch, 0>>>;

    // 5. Instantiate transpose handlers
    // Transpose 1: (Batch, SignalLengthY, StrideY) -> (Batch, StrideY, SignalLengthY)
    using Transpose1 = zipfft::BatchedTranspose<64, 32, 32, 8>;

    // Transpose 2: (Batch, FFTSizeY, StrideY) -> (Batch, StrideY, FFTSizeY)
    using Transpose2 = zipfft::BatchedTranspose<64, 32, 32, 8>;

    // 6. Construct kernel pointers
    auto kernel_r2c_x = optimized_r2c_kernel<FFTX_fwd, IO_X_fwd>;
    auto kernel_c2c_y = optimized_c2c_conv_kernel<FFTY_fwd, FFTY_inv, IO_Y_fwd, IO_Y_inv, ConvDataLayout,
                                                  StrideY, CrossCorrelate, ConvDataIsTransposed>;
    auto kernel_c2r_x = optimized_c2r_kernel<FFTX_inv, IO_X_inv>;

    // 7. Set shared memory sizes
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_r2c_x, cudaFuncAttributeMaxDynamicSharedMemorySize, IO_X_fwd::get_shared_bytes()));
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_c2c_y, cudaFuncAttributeMaxDynamicSharedMemorySize, IO_Y_fwd::get_shared_bytes()));
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_c2r_x, cudaFuncAttributeMaxDynamicSharedMemorySize, IO_X_inv::get_shared_bytes()));

    // 8. Create workspaces for FFTs
    cudaError_t workspace_error = cudaSuccess;
    auto workspace_fwd_x = make_workspace<FFTX_fwd>(workspace_error);
    CUDA_CHECK_AND_EXIT(workspace_error);
    auto workspace_fwd_y = make_workspace<FFTY_fwd>(workspace_error);
    CUDA_CHECK_AND_EXIT(workspace_error);
    auto workspace_inv_y = make_workspace<FFTY_inv>(workspace_error);
    CUDA_CHECK_AND_EXIT(workspace_error);
    auto workspace_inv_x = make_workspace<FFTX_inv>(workspace_error);
    CUDA_CHECK_AND_EXIT(workspace_error);

    // 9. Calculate grid sizes
    const unsigned int grid_size_fwd_x =
        (SignalLengthY + FFTX_fwd::ffts_per_block - 1) / FFTX_fwd::ffts_per_block;
    const unsigned int grid_size_fwd_y =
        (StrideY + FFTY_fwd::ffts_per_block - 1) / FFTY_fwd::ffts_per_block;
    const unsigned int grid_size_inv_x =
        (ValidLengthY + FFTX_inv::ffts_per_block - 1) / FFTX_inv::ffts_per_block;

    // 10. Cast pointers to cuFFTDx types
    scalar_type* input_data_cast = reinterpret_cast<scalar_type*>(input_data);
    complex_type* fft_workspace_r2c_cast = reinterpret_cast<complex_type*>(fft_workspace_r2c);
    complex_type* fft_workspace_r2c_transposed_cast =
        reinterpret_cast<complex_type*>(fft_workspace_r2c_transposed);
    complex_type* fft_workspace_c2c_transposed_cast =
        reinterpret_cast<complex_type*>(fft_workspace_c2c_transposed);
    complex_type* fft_workspace_c2r_cast = reinterpret_cast<complex_type*>(fft_workspace_c2r);
    complex_type* conv_data_cast = reinterpret_cast<complex_type*>(conv_data);
    scalar_type* output_data_cast = reinterpret_cast<scalar_type*>(output_data);

    // 11. Launch the five kernels in sequence with transposes

    // Step 1: R2C FFT along X dimension
    // Input: (Batch, SignalLengthY, SignalLengthX)
    // Output: (Batch, SignalLengthY, StrideY) where StrideY = FFTSizeX/2+1
    kernel_r2c_x<<<dim3{grid_size_fwd_x, Batch}, FFTX_fwd::block_dim, IO_X_fwd::get_shared_bytes(),
                   stream>>>(input_data_cast, fft_workspace_r2c_cast, workspace_fwd_x);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    // Step 2: Transpose (Batch, SignalLengthY, StrideY) -> (Batch, StrideY, SignalLengthY)
    cudaError_t transpose1_err = Transpose1::template launch<complex_type>(
        fft_workspace_r2c_cast, fft_workspace_r2c_transposed_cast, Batch, SignalLengthY, StrideY,
        stream);
    CUDA_CHECK_AND_EXIT(transpose1_err);

    // Step 3: C2C FFT along Y dimension with convolution
    // Input: (Batch, StrideY, SignalLengthY)
    // Output: (Batch, StrideY, FFTSizeY) [transposed view]
    kernel_c2c_y<<<dim3{grid_size_fwd_y, Batch}, FFTY_fwd::block_dim, IO_Y_fwd::get_shared_bytes(),
                   stream>>>(fft_workspace_r2c_transposed_cast, fft_workspace_c2c_transposed_cast,
                             conv_data_cast, workspace_fwd_y, workspace_inv_y);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    // Step 4: Transpose (Batch, StrideY, FFTSizeY) -> (Batch, ValidLengthY, StrideY)
    // Note: ValidLengthY = FFTSizeY - SignalLengthY + 1
    cudaError_t transpose2_err = Transpose2::template launch<complex_type>(
        fft_workspace_c2c_transposed_cast, fft_workspace_c2r_cast, Batch, StrideY, ValidLengthY,
        stream);
    CUDA_CHECK_AND_EXIT(transpose2_err);

    // Step 5: C2R FFT along X dimension
    // Input: (Batch, ValidLengthY, StrideY)
    // Output: (Batch, ValidLengthY, ValidLengthX)
    kernel_c2r_x<<<dim3{grid_size_inv_x, Batch}, FFTX_inv::block_dim, IO_X_inv::get_shared_bytes(),
                   stream>>>(fft_workspace_c2r_cast, output_data_cast, workspace_inv_x);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
}

// --- Public API Function Template Definition ---
/**
 * @brief Public API for 2D real convolution using optimized IO handlers with shared memory
 * transposes.
 */
template <typename ScalarType, typename ComplexType, unsigned int SignalLengthX,
          unsigned int SignalLengthY, unsigned int FFTSizeX, unsigned int FFTSizeY,
          unsigned int Batch, bool CrossCorrelate, unsigned int elements_per_thread_x = 0,
          unsigned int elements_per_thread_y = 0, unsigned int FFTs_per_block_x = 0,
          unsigned int FFTs_per_block_y = 0, bool ConvDataIsTransposed = false>
int padded_block_real_conv_2d(ScalarType* input_data, ComplexType* fft_workspace_r2c,
                              ComplexType* fft_workspace_r2c_transposed,
                              ComplexType* fft_workspace_c2c_transposed,
                              ComplexType* fft_workspace_c2r, ComplexType* conv_data,
                              ScalarType* output_data, cudaStream_t stream = 0) {
    auto arch = zipfft::get_cuda_device_arch();

    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: padded_block_real_conv_2d_launcher<800, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, ConvDataIsTransposed>(input_data, fft_workspace_r2c, fft_workspace_r2c_transposed, fft_workspace_c2c_transposed, fft_workspace_c2r, conv_data, output_data, stream); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: padded_block_real_conv_2d_launcher<860, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, ConvDataIsTransposed>(input_data, fft_workspace_r2c, fft_workspace_r2c_transposed, fft_workspace_c2c_transposed, fft_workspace_c2r, conv_data, output_data, stream); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: padded_block_real_conv_2d_launcher<870, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, ConvDataIsTransposed>(input_data, fft_workspace_r2c, fft_workspace_r2c_transposed, fft_workspace_c2c_transposed, fft_workspace_c2r, conv_data, output_data, stream); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: padded_block_real_conv_2d_launcher<890, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, ConvDataIsTransposed>(input_data, fft_workspace_r2c, fft_workspace_r2c_transposed, fft_workspace_c2c_transposed, fft_workspace_c2r, conv_data, output_data, stream); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: padded_block_real_conv_2d_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, ConvDataIsTransposed>(input_data, fft_workspace_r2c, fft_workspace_r2c_transposed, fft_workspace_c2c_transposed, fft_workspace_c2r, conv_data, output_data, stream); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        case 1200: padded_block_real_conv_2d_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y, CrossCorrelate, ConvDataIsTransposed>(input_data, fft_workspace_r2c, fft_workspace_r2c_transposed, fft_workspace_c2c_transposed, fft_workspace_c2r, conv_data, output_data, stream); break;
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

#endif  // ZIPFFT_REAL_CONV_2D_CUH
