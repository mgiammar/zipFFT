#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_index_mapper.hpp"
#include "../include/zipfft_strided_io.hpp"

// --- Forward r2c Kernel Definition with Index Mappers ---
template <class FFT, class InputLayout, class OutputLayout,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_fft_r2c_1d_kernel_with_layout(ScalarType* input_data, ComplexType* output_data) {
    using complex_type = ComplexType;
    using io_type = zipfft::io_with_layout<FFT, InputLayout, OutputLayout>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using custom input layout
    io_type::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Store using custom output layout
    io_type::store(thread_data, output_data, local_fft_id);
}

// --- Inverse c2r Kernel Definition with Index Mappers ---
template <class FFT, class InputLayout, class OutputLayout,
          typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_fft_c2r_1d_kernel_with_layout(ComplexType* input_data, ScalarType* output_data) {
    using complex_type = ComplexType;
    using io_type = zipfft::io_with_layout<FFT, InputLayout, OutputLayout>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using custom input layout
    io_type::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Store using custom output layout
    io_type::store(thread_data, output_data, local_fft_id);
}

// --- Forward & Inverse C2C Kernel with Index Mappers and Stride ---
template <class FFT, class InputLayout, class OutputLayout, unsigned int Stride>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void strided_block_fft_c2c_1d_kernel_with_layout(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;
    using io_type = zipfft::io_strided_with_layout<FFT, InputLayout, OutputLayout, Stride>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    // Load using custom strided layout
    io_type::load(data, thread_data, local_fft_id);

    // Execute the FFT with shared memory
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Store using custom strided layout
    io_type::store(thread_data, data, local_fft_id);
}

// --- Launcher for 2D Real FFTs ---
template <unsigned int Arch, typename Input_T, typename Output_T, unsigned int FFTSizeX,
          unsigned int FFTSizeY, unsigned int Batch, bool IsForwardFFT,
          unsigned int elements_per_thread_x, unsigned int elements_per_thread_y,
          unsigned int FFTs_per_block_x, unsigned int FFTs_per_block_y>
inline void block_real_fft_2d_launcher(Input_T* input_data, Output_T* output_data) {
    using namespace cufftdx;

    // Conditional statements for the FFT declarations
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using scalar_precision_type = std::conditional_t<IsForwardFFT, Input_T, Output_T>;
    constexpr auto fft_direction = IsForwardFFT ? fft_direction::forward : fft_direction::inverse;
    constexpr auto fft_type_real = IsForwardFFT ? fft_type::r2c : fft_type::c2r;

    // FFTX is the r2c or c2r transform (contiguous dimension)
    using FFTX = decltype(Block() + Size<FFTSizeX>() + Type<fft_type_real>() + real_fft_options() +
                          Direction<fft_direction>() + Precision<scalar_precision_type>() +
                          ElementsPerThread<elements_per_thread_x>() +
                          FFTsPerBlock<FFTs_per_block_x>() + SM<Arch>());

    // FFTY is the c2c transform along strided dimension
    using FFTY =
        decltype(Block() + Size<FFTSizeY>() + Type<fft_type::c2c>() + Direction<fft_direction>() +
                 Precision<scalar_precision_type>() + ElementsPerThread<elements_per_thread_y>() +
                 FFTsPerBlock<FFTs_per_block_y>() + SM<Arch>());

    using complex_type = typename FFTY::value_type;
    using scalar_type = typename complex_type::value_type;

    // Define index mappers for Y dimension (strided by StrideY)
    // Both InputLayoutY and OutputLayoutY assume complex-valued data with shape
    // (Batch, FFTSizeY, StrideY)
    // where StrideY is FFTX::output_length if IsForwardFFT is true
    // otherwise StrideY is FFTX::input_length if IsForwardFFT is false
    const unsigned int StrideY = IsForwardFFT ? FFTX::output_length : FFTX::input_length;
    using InputLayoutY =
        zipfft::index_mapper<zipfft::int_pair<StrideY, 1>, zipfft::int_pair<FFTSizeY, StrideY>,
                             zipfft::int_pair<Batch, FFTSizeY * StrideY>>;

    // Same as InputLayoutY
    using OutputLayoutY = InputLayoutY;

    // Compile-time branching for kernel selection and execution
    if constexpr (IsForwardFFT) {
        // Define index mappers for X dimension (contiguous)
        // InputLayoutX assumes real-valued input data with shape (Batch, FFTSizeY, FFTSizeX)
        //
        // NOTE: Instead of using templated parameter FFTSizeX, we use FFTX::input_length
        // since under real_mode::folded, the data input is considered complex-valued
        // with half the length of FFTSizeX. Taking parameters directly from the FFT structures
        // ensures consistency and proper strided data access.
        //
        // NOTE: the order of the dimensions appears in reverse to what's normal with an array
        // library. First pair is the (Size, Stride) of the columns (X dimension), second pair is
        // for the rows (Y dimension), and third pair is for the batches.
        using InputLayoutX =
            zipfft::index_mapper<zipfft::int_pair<FFTX::input_length, 1>,
                                 zipfft::int_pair<FFTSizeY, FFTX::input_length>,
                                 zipfft::int_pair<Batch, FFTSizeY * FFTX::input_length>>;

        // OutputLayoutX assumes complex-valued output data with shape (Batch, FFTSizeY,
        // FFTX::output_length) where FFTX::output_length = FFTSizeX / 2 + 1 = StrideY
        using OutputLayoutX =
            zipfft::index_mapper<zipfft::int_pair<FFTX::output_length, 1>,
                                 zipfft::int_pair<FFTSizeY, FFTX::output_length>,
                                 zipfft::int_pair<Batch, FFTSizeY * FFTX::output_length>>;

        // Get the kernel pointers
        auto kernel_ptr_x = block_fft_r2c_1d_kernel_with_layout<FFTX, InputLayoutX, OutputLayoutX>;
        auto kernel_ptr_y =
            strided_block_fft_c2c_1d_kernel_with_layout<FFTY, InputLayoutY, OutputLayoutY, StrideY>;

        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr_x, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTX::shared_memory_size));
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr_y, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY::shared_memory_size));

        // Prepare data pointers
        scalar_type* input_data_t = reinterpret_cast<scalar_type*>(input_data);
        complex_type* output_data_t = reinterpret_cast<complex_type*>(output_data);

        // Launch transform along X dimension (contiguous)
        const unsigned int grid_size_x =
            ((Batch * FFTSizeY) + FFTX::ffts_per_block - 1) / FFTX::ffts_per_block;
        kernel_ptr_x<<<grid_size_x, FFTX::block_dim, FFTX::shared_memory_size>>>(input_data_t,
                                                                                 output_data_t);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

        // Launch transform along Y dimension (strided by StrideY)
        const unsigned int grid_size_y =
            ((Batch * StrideY) + FFTY::ffts_per_block - 1) / FFTY::ffts_per_block;
        kernel_ptr_y<<<grid_size_y, FFTY::block_dim, FFTY::shared_memory_size>>>(output_data_t);

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    } else {
        // Define index mappers for X dimension (contiguous)
        // InputLayoutX assumes real-valued input data with shape (Batch, FFTSizeY, FFTSizeX)
        //
        // NOTE: Instead of using templated parameter FFTSizeX, we use FFTX::input_length
        // since under real_mode::folded, the data input is considered complex-valued
        // with half the length of FFTSizeX. Taking parameters directly from the FFT structures
        // ensures consistency and proper strided data access.
        //
        // NOTE: the order of the dimensions appears in reverse to what's normal with an array
        // library. First pair is the (Size, Stride) of the columns (X dimension), second pair is
        // for the rows (Y dimension), and third pair is for the batches.
        using InputLayoutX =
            zipfft::index_mapper<zipfft::int_pair<FFTX::input_length, 1>,
                                 zipfft::int_pair<FFTSizeY, FFTX::input_length>,
                                 zipfft::int_pair<Batch, FFTSizeY*FFTX::input_length>>;

        // OutputLayoutX assumes complex-valued output data with shape (Batch, FFTSizeY,
        // FFTX::output_length) where FFTX::output_length = FFTSizeX / 2 + 1 = StrideY
        using OutputLayoutX =
            zipfft::index_mapper<zipfft::int_pair<FFTX::output_length, 1>,
                                 zipfft::int_pair<FFTSizeY, FFTX::output_length>,
                                 zipfft::int_pair<Batch, FFTSizeY * FFTX::output_length>>;

        // Get the kernel pointers
        auto kernel_ptr_x = block_fft_c2r_1d_kernel_with_layout<FFTX, InputLayoutX, OutputLayoutX>;
        auto kernel_ptr_y =
            strided_block_fft_c2c_1d_kernel_with_layout<FFTY, InputLayoutY, OutputLayoutY, StrideY>;

        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr_x, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTX::shared_memory_size));
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel_ptr_y, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY::shared_memory_size));

        // Prepare data pointers
        complex_type* input_data_t = reinterpret_cast<complex_type*>(input_data);
        scalar_type* output_data_t = reinterpret_cast<scalar_type*>(output_data);

        // Launch transform along Y dimension (strided by StrideY)
        const unsigned int grid_size_y =
            ((Batch * StrideY) + FFTY::ffts_per_block - 1) / FFTY::ffts_per_block;
        kernel_ptr_y<<<grid_size_y, FFTY::block_dim, FFTY::shared_memory_size>>>(input_data_t);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

        // Launch transform along X dimension (contiguous)
        const unsigned int grid_size_x =
            ((Batch * FFTSizeY) + FFTX::ffts_per_block - 1) / FFTX::ffts_per_block;
        kernel_ptr_x<<<grid_size_x, FFTX::block_dim, FFTX::shared_memory_size>>>(input_data_t,
                                                                                 output_data_t);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    }

    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Public API Function Template Definition ---
template <typename Input_T, typename Output_T, unsigned int FFTSizeX, unsigned int FFTSizeY,
          unsigned int Batch, bool IsForwardFFT, unsigned int elements_per_thread_x,
          unsigned int elements_per_thread_y, unsigned int FFTs_per_block_x,
          unsigned int FFTs_per_block_y>
int block_real_fft_2d(Input_T* input_data, Output_T* output_data) {
    auto arch = zipfft::get_cuda_device_arch();

    /* clang-format off */
    switch (arch) {
#ifdef ENABLE_CUDA_ARCH_800
        case 800: block_real_fft_2d_launcher<800, Input_T, Output_T, FFTSizeX, FFTSizeY, Batch, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_860
        case 860: block_real_fft_2d_launcher<860, Input_T, Output_T, FFTSizeX, FFTSizeY, Batch, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_870
        case 870: block_real_fft_2d_launcher<870, Input_T, Output_T, FFTSizeX, FFTSizeY, Batch, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_890
        case 890: block_real_fft_2d_launcher<890, Input_T, Output_T, FFTSizeX, FFTSizeY, Batch, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#ifdef ENABLE_CUDA_ARCH_900
        case 900: block_real_fft_2d_launcher<900, Input_T, Output_T, FFTSizeX, FFTSizeY, Batch, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
#endif
#if defined(ENABLE_CUDA_ARCH_1200) || defined(ENABLE_CUDA_ARCH_120)
        // Fallback: use the 900 specialization for newer architectures
        case 1200: block_real_fft_2d_launcher<900, Input_T, Output_T, FFTSizeX, FFTSizeY, Batch, IsForwardFFT, elements_per_thread_x, elements_per_thread_y, FFTs_per_block_x, FFTs_per_block_y>(input_data, output_data); break;
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
