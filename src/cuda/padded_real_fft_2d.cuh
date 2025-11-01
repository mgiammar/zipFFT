#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_index_mapper.hpp"
#include "../include/zipfft_padded_io.hpp"
#include "../include/zipfft_strided_io.hpp"
#include "../include/zipfft_strided_padded_io.hpp"

// --- Forward R2C Kernel with Index Mappers (Step 1) ---
template <class FFT, class InputLayout, class OutputLayout, unsigned int SignalLength,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_r2c_2d_kernel_with_layout(ScalarType* input_data,
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

// --- Inverse C2R Kernel with Index Mappers (Step 3) ---
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

// --- Strided C2C Kernel with Padding (Steps 2a & 2c) ---
template <class FFT, class InputLayout, class OutputLayout, unsigned int Stride,
          unsigned int InputSignalLength, unsigned int OutputSignalLength>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void strided_padded_block_fft_c2c_2d_kernel_with_layout(
        typename FFT::value_type* data, typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    // Use strided + padded I/O for both input and output
    using io_type = zipfft::io_strided_padded_with_layout<FFT, InputLayout, OutputLayout, Stride,
                                                          InputSignalLength>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using strided padded layout (zero-pads if InputSignalLength < FFT size)
    io_type::load(data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Store using strided padded layout (truncates if OutputSignalLength < FFT size)
    // Note: For forward, OutputSignalLength = FFT::output_length (no truncation)
    //       For inverse, OutputSignalLength < FFT::output_length (truncation)
    using output_io_type = zipfft::io_strided_padded_with_layout<FFT, InputLayout, OutputLayout,
                                                                 Stride, OutputSignalLength>;
    output_io_type::store(thread_data, data, local_fft_id);
}

// --- Forward R2C 2D FFT Launcher ---
template <unsigned int Arch, unsigned int FFTSizeX, unsigned int FFTSizeY, unsigned int Batch,
          unsigned int SignalLengthX, unsigned int SignalLengthY,
          unsigned int elements_per_thread_x, unsigned int elements_per_thread_y,
          unsigned int FFTs_per_block_x, unsigned int FFTs_per_block_y>
inline void padded_block_real_fft_2d_r2c_launcher(float* input_data, float2* output_data) {
    using namespace cufftdx;

    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;

    // FFTX is the r2c transform (contiguous dimension, Step 1)
    using FFTX = decltype(Block() + Size<FFTSizeX>() + Type<fft_type::r2c>() + real_fft_options() +
                          Direction<fft_direction::forward>() + Precision<float>() +
                          ElementsPerThread<elements_per_thread_x>() +
                          FFTsPerBlock<FFTs_per_block_x>() + SM<Arch>());

    // FFTY is the c2c transform along strided dimension (Step 2a)
    using FFTY = decltype(Block() + Size<FFTSizeY>() + Type<fft_type::c2c>() +
                          Direction<fft_direction::forward>() + Precision<float>() +
                          ElementsPerThread<elements_per_thread_y>() +
                          FFTsPerBlock<FFTs_per_block_y>() + SM<Arch>());

    using complex_type = typename FFTY::value_type;
    using scalar_type = typename complex_type::value_type;

    // StrideY is the output length from the r2c transform (FFTSizeX / 2 + 1)
    // const unsigned int InputLengthX_complex = SignalLengthX / 2;
    const unsigned int StrideY = FFTX::output_length;

    // --- Step 1: Forward R2C along X dimension ---
    // Input: (Batch, SignalLengthY, FFTX::input_length) - padded real input
    using InputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<SignalLengthX, 1>,
                             zipfft::int_pair<SignalLengthY, SignalLengthX>,
                             zipfft::int_pair<Batch, SignalLengthY * SignalLengthX>>;

    // Output: (Batch, SignalLengthY, StrideY) - only first SignalLengthY rows written
    using OutputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<StrideY, 1>, zipfft::int_pair<SignalLengthY, StrideY>,
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>;

    // --- Step 2a: Forward C2C along Y dimension (strided) ---
    // Input: (Batch, SignalLengthY, StrideY) - read only first SignalLengthY rows per column
    using InputLayoutY =
        zipfft::index_mapper<zipfft::int_pair<SignalLengthY, StrideY>,  // rows strided
                             zipfft::int_pair<StrideY, 1>,              // columns contiguous
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>;

    using OutputLayoutY = zipfft::index_mapper<zipfft::int_pair<FFTSizeY, StrideY>,  // rows strided
                                               zipfft::int_pair<StrideY, 1>,  // columns contiguous
                                               zipfft::int_pair<Batch, FFTSizeY * StrideY>>;

    // Get kernel pointers
    auto kernel_ptr_x = padded_block_fft_r2c_2d_kernel_with_layout<FFTX, InputLayoutX,
                                                                   OutputLayoutX, SignalLengthX>;
    auto kernel_ptr_y =
        strided_padded_block_fft_c2c_2d_kernel_with_layout<FFTY, InputLayoutY, OutputLayoutY,
                                                           StrideY, SignalLengthY, FFTSizeY>;

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_ptr_x, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTX::shared_memory_size));
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_ptr_y, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY::shared_memory_size));

    // Prepare data pointers
    scalar_type* input_data_t = reinterpret_cast<scalar_type*>(input_data);
    complex_type* output_data_t = reinterpret_cast<complex_type*>(output_data);

    // Create workspaces
    cudaError_t error_code = cudaSuccess;
    auto workspace_x = make_workspace<FFTX>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_y = make_workspace<FFTY>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Launch Step 1: R2C transform along X dimension (only SignalLengthY rows)
    const unsigned int grid_size_x =
        ((Batch * SignalLengthY) + FFTX::ffts_per_block - 1) / FFTX::ffts_per_block;
    kernel_ptr_x<<<grid_size_x, FFTX::block_dim, FFTX::shared_memory_size>>>(
        input_data_t, output_data_t, workspace_x);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    // Launch Step 2a: C2C transform along Y dimension (all StrideY columns)
    const unsigned int grid_size_y =
        ((Batch * StrideY) + FFTY::ffts_per_block - 1) / FFTY::ffts_per_block;
    kernel_ptr_y<<<grid_size_y, FFTY::block_dim, FFTY::shared_memory_size>>>(output_data_t,
                                                                             workspace_y);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Inverse C2R 2D FFT Launcher ---
template <unsigned int Arch, unsigned int FFTSizeX, unsigned int FFTSizeY, unsigned int Batch,
          unsigned int SignalLengthX, unsigned int SignalLengthY,
          unsigned int elements_per_thread_x, unsigned int elements_per_thread_y,
          unsigned int FFTs_per_block_x, unsigned int FFTs_per_block_y>
inline void padded_block_real_fft_2d_c2r_launcher(float2* input_data, float* output_data) {
    using namespace cufftdx;

    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;

    // FFTX is the c2r transform (contiguous dimension, Step 3)
    using FFTX = decltype(Block() + Size<FFTSizeX>() + Type<fft_type::c2r>() + real_fft_options() +
                          Direction<fft_direction::inverse>() + Precision<float>() +
                          ElementsPerThread<elements_per_thread_x>() +
                          FFTsPerBlock<FFTs_per_block_x>() + SM<Arch>());

    // FFTY is the c2c transform along strided dimension (Step 2c)
    using FFTY = decltype(Block() + Size<FFTSizeY>() + Type<fft_type::c2c>() +
                          Direction<fft_direction::inverse>() + Precision<float>() +
                          ElementsPerThread<elements_per_thread_y>() +
                          FFTsPerBlock<FFTs_per_block_y>() + SM<Arch>());

    using complex_type = typename FFTY::value_type;
    using scalar_type = typename complex_type::value_type;

    // StrideY is the input length to the c2r transform (FFTSizeX / 2 + 1)
    const unsigned int StrideY = FFTX::input_length;

    // --- Step 2c: Inverse C2C along Y dimension (strided) ---
    // Input: (Batch, FFTSizeY, StrideY) - read full FFTSizeY rows per column
    using InputLayoutY = zipfft::index_mapper<zipfft::int_pair<FFTSizeY, StrideY>,  // rows strided
                                              zipfft::int_pair<StrideY, 1>,  // columns contiguous
                                              zipfft::int_pair<Batch, FFTSizeY * StrideY>>;

    // Output: (Batch, SignalLengthY, StrideY) - write only first SignalLengthY rows per column
    using OutputLayoutY =
        zipfft::index_mapper<zipfft::int_pair<SignalLengthY, StrideY>,  // rows strided
                             zipfft::int_pair<StrideY, 1>,              // columns contiguous
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>;

    // --- Step 3: Inverse C2R along X dimension ---
    // Input: (Batch, SignalLengthY, StrideY) - read only first SignalLengthY rows
    using InputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<StrideY, 1>, zipfft::int_pair<SignalLengthY, StrideY>,
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>;

    // Output: (Batch, SignalLengthY, SignalLengthX) - truncated real output
    using OutputLayoutX =
        zipfft::index_mapper<zipfft::int_pair<SignalLengthX, 1>,
                             zipfft::int_pair<SignalLengthY, SignalLengthX>,
                             zipfft::int_pair<Batch, SignalLengthY * SignalLengthX>>;

    // Get kernel pointers
    auto kernel_ptr_x = padded_block_fft_c2r_2d_kernel_with_layout<FFTX, InputLayoutX,
                                                                   OutputLayoutX, SignalLengthX>;
    auto kernel_ptr_y =
        strided_padded_block_fft_c2c_2d_kernel_with_layout<FFTY, InputLayoutY, OutputLayoutY,
                                                           StrideY, FFTSizeY, SignalLengthY>;

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_ptr_x, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTX::shared_memory_size));
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_ptr_y, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY::shared_memory_size));

    // Prepare data pointers
    complex_type* input_data_t = reinterpret_cast<complex_type*>(input_data);
    scalar_type* output_data_t = reinterpret_cast<scalar_type*>(output_data);

    // Create workspaces
    cudaError_t error_code = cudaSuccess;
    auto workspace_x = make_workspace<FFTX>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_y = make_workspace<FFTY>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Launch Step 2c: C2C transform along Y dimension (all StrideY columns)
    const unsigned int grid_size_y =
        ((Batch * StrideY) + FFTY::ffts_per_block - 1) / FFTY::ffts_per_block;
    kernel_ptr_y<<<grid_size_y, FFTY::block_dim, FFTY::shared_memory_size>>>(input_data_t,
                                                                             workspace_y);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    // Launch Step 3: C2R transform along X dimension (only SignalLengthY rows)
    const unsigned int grid_size_x =
        ((Batch * SignalLengthY) + FFTX::ffts_per_block - 1) / FFTX::ffts_per_block;
    kernel_ptr_x<<<grid_size_x, FFTX::block_dim, FFTX::shared_memory_size>>>(
        input_data_t, output_data_t, workspace_x);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
template <typename Input_T, typename Output_T, unsigned int SignalLengthX,
          unsigned int SignalLengthY, unsigned int FFTSizeX, unsigned int FFTSizeY,
          unsigned int Batch, bool IsForwardFFT, unsigned int elements_per_thread_x,
          unsigned int elements_per_thread_y, unsigned int FFTs_per_block_x,
          unsigned int FFTs_per_block_y>
int padded_block_real_fft_2d(Input_T* input_data, Output_T* output_data) {
    auto arch = zipfft::get_cuda_device_arch();

    /* clang-format off */
    if constexpr (IsForwardFFT) {
        switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
            case 800: padded_block_real_fft_2d_r2c_launcher<800, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
            case 860: padded_block_real_fft_2d_r2c_launcher<860, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
            case 870: padded_block_real_fft_2d_r2c_launcher<870, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
            case 890: padded_block_real_fft_2d_r2c_launcher<890, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
            case 900: padded_block_real_fft_2d_r2c_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
            case 1200: padded_block_real_fft_2d_r2c_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
            default:
                std::cerr << "Unsupported CUDA architecture: " << arch
                          << ". Supported architectures are 800, 860, 870, 890, 900, and 1200."
                          << std::endl;
                return -1;
        }
    } else {
        switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
            case 800: padded_block_real_fft_2d_c2r_launcher<800, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
            case 860: padded_block_real_fft_2d_c2r_launcher<860, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
            case 870: padded_block_real_fft_2d_c2r_launcher<870, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
            case 890: padded_block_real_fft_2d_c2r_launcher<890, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
            case 900: padded_block_real_fft_2d_c2r_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
            case 1200: padded_block_real_fft_2d_c2r_launcher<900, FFTSizeX, FFTSizeY, Batch, SignalLengthX, SignalLengthY, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
            default:
                std::cerr << "Unsupported CUDA architecture: " << arch
                          << ". Supported architectures are 800, 860, 870, 890, 900, and 1200."
                          << std::endl;
                return -1;
        }
    }
    /* clang-format on */

    return 0;
}