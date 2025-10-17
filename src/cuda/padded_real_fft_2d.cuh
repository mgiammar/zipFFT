#include <cufftdx.hpp>

#include "../include/block_io.hpp"
#include "../include/common.hpp"
#include "../include/padded_io.hpp"

/**
 * @brief Execute the first real-to-complex FFT along the Y dimension
 * (contigious dimension) of a zero-padded 2D input.
 *
 * Each separate row/column in the 2D input is processed in its own CUDA block.
 *
 * @tparam SignalLength - Length of non-zero signal in the X dimension
 * @tparam FFT - FFT structure from cuFFTDx
 * @tparam ScalarType - Real type of the input data
 * @tparam ComplexType - Complex type of the FFT output
 *
 * @param input_data - Input data (real type)
 * @param output_data - Output data (complex type)
 * @param workspace - Workspace for the FFT execution, possibly empty
 */
template <int SignalLength, class FFT, typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_r2c_2d_kernel_y(ScalarType* input_data, ComplexType* output_data,
                                          typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    // Input is padded, use padded I/O utilities. Output is not padded
    using input_utils = example::io_padded<FFT, SignalLengthX>;
    using output_utils = example::io<FFT>;

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

/**
 * @brief Execute the third complex-to-real FFT along the Y dimension (no longer
 * padded, contigious data access) of a 2D input.
 *
 * @tparam FFT - FFT structure from cuFFTDx
 * @tparam ComplexType - Complex type of the input data
 * @tparam ScalarType - Real type of the output data
 *
 * @param input_data - Input data (complex type)
 * @param output_data - Output data (real type)
 * @param workspace - Workspace for the FFT execution, possibly empty
 */
template <class FFT, typename ComplexType = typename FFT::value_type,
          typename ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void block_fft_c2r_2d_kernel_y(ComplexType* input_data, ScalarType* output_data,
                                   typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;
    using scalar_type = typename complex_type::value_type;

    using io_utils = example::io<FFT>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // Grid dimension (blockIdx.x) for row/col handled automatically.
    const unsigned int local_fft_id = threadIdx.y;
    io_utils::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    io_utils::store(thread_data, output_data, local_fft_id);
}

/**
 * @brief Execute the forward 2D FFT along the X dimension (non-contigious)
 * with zero-padding. Note that this function operates in-place.
 *
 * @tparam FFT - FFT structure from cuFFTDx
 * @tparam Stride - Stride of the input data (size of the Y dimension after
 *                  transform and padding)
 * @tparam SignalLength - Length of the non-zero signal in the X dimension
 * @tparam ComplexType - Complex type of the data
 */
template <class FFT, unsigned int Stride, unsigned int SignalLength,
          class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void padded_block_fft_c2c_2d_kernel_x(ComplexType* data,
                                          typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    // Input is padded, but output is not padded. Use two different i/o.
    using input_utils = example::io_padded<FFT, SignalLength>;
    using output_utils = example::io_strided<FFT>;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // Grid dimension (blockIdx.x) for row/col handled automatically.
    const unsigned int local_fft_id = threadIdx.y;
    input_utils::load<Stride>(data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    output_utils::store<Stride, FFT::output_length>(thread_data, data, local_fft_id);
}

// template<class FFTF,
//          class FFTI,
//          unsigned int Stride,
//          unsigned int SizeY,
//          bool         UseSharedMemoryStridedIO,
//          class ComplexType = typename FFTF::value_type>
// __launch_bounds__(FFTF::max_threads_per_block) __global__
//     void fft_2d_kernel_x(const ComplexType*            input,
//                          ComplexType*                  output,
//                          typename FFTF::workspace_type workspacef,
//                          typename FFTI::workspace_type workspacei) {
//     using complex_type = typename FFTF::value_type;

//     extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];

//     // Local array for thread
//     complex_type thread_data[FFTF::storage_size];

//     // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
//     const unsigned int local_fft_id = threadIdx.y;
//     // Load data from global memory to registers
//     if constexpr (UseSharedMemoryStridedIO) {
//         example::io_strided<FFTF>::load<Stride, SizeY>(input,
//         thread_data, shared_mem, local_fft_id);
//     } else {
//         example::io_strided<FFTF>::load<Stride, SizeY>(input,
//         thread_data, local_fft_id);
//     }

//     // Execute FFT (part of the 2D R2C FFT)
//     FFTF().execute(thread_data, shared_mem, workspacef);

//     // Note: You can do any point-wise operation in here.

//     // Execute FFT (part of the 2D C2R FFT)
//     FFTI().execute(thread_data, shared_mem, workspacei);

//     // Save results
//     if constexpr (UseSharedMemoryStridedIO) {
//         example::io_strided<FFTI>::store<Stride, SizeY>(thread_data,
//         shared_mem, output, local_fft_id);
//     } else {
//         example::io_strided<FFTI>::store<Stride, SizeY>(thread_data,
//         output, local_fft_id);
//     }
// }

/**
 * @brief Executes a forward 2D real-to-complex FFT with zero-padding along both
 * the X and Y dimensions.
 *
 * @tparam Arch - CUDA architecture version
 * @tparam InputType - Input type (real data)
 * @tparam OutputType - Output type (complex data)
 * @tparam SignalLengthX - Length of the non-zero signal in the X dimension.
 * Should be less than or equal to FFTSizeX.
 * @tparam FFTSizeX - Size of the FFT in the X dimension.
 * @tparam SignalLengthY - Length of the non-zero signal in the Y dimension.
 * Should be less than or equal to FFTSizeY.
 * @tparam FFTSizeY - Size of the FFT in the Y dimension.
 * @tparam elements_per_thread - Number of elements processed by each thread.
 * @tparam FFTs_per_block - Number of independent FFTs processed per CUDA block.
 *
 * @param input_data - Input data (real type) with shape (SignalLengthX,
 * SignalLengthY) and stored in row-major order.
 * @param output_data - Output data (complex type) with shape (FFTSizeX,
 * FFTSizeY / 2 + 1) and stored in row-major order. Note the Y dimension is
 * halved due to the RFFT.
 */
template <unsigned int Arch, typename InputType, typename OutputType, unsigned int SignalLengthX,
          unsigned int FFTSizeX, unsigned int SignalLengthY, unsigned int FFTSizeY,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
inline void padded_block_real_fft_2d_launcher(InputType* input_data, OutputType* output_data) {
    using namespace cufftdx;

    // Define the R2C FFT structure
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::folded>;
    using scalar_precision_type = float;  // TODO: Implement other precision types

    using ForwardFFTY =
        decltype(Block() + Size<FFTSizeY>() + Type<fft_type::r2c>() + real_fft_options() +
                 Direction<fft_direction::forward>() + Precision<scalar_precision_type>() +
                 ElementsPerThread<elements_per_thread>() + FFTsPerBlock<FFTs_per_block>() +
                 SM<Arch>());

    // Define the C2C FFT structure
    using ForwardFFTX =
        decltype(Block() + Size<FFTSizeX>() + Type<fft_type::c2c>() +
                 Direction<fft_direction::forward>() + Precision<scalar_precision_type>() +
                 ElementsPerThread<elements_per_thread>() + FFTsPerBlock<FFTs_per_block>() +
                 SM<Arch>());

    using complex_type = typename ForwardFFTY::value_type;
    using scalar_type = typename complex_type::value_type;

    // Set the shared memory sizes checking for any errors
    // TODO

    // Allocate FFT workspaces
    // TODO

    // Launch the first kernel for the Y dimension (R2C)
    // TODO

    // Launch the second kernel for the X dimension (C2C)
    // TODO

    // Ensure no errors afterwards
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

template <typename InputType, typename OutputType, unsigned int SignalLengthX,
          unsigned int FFTSizeX, unsigned int SignalLengthY, unsigned int FFTSizeY,
          unsigned int elements_per_thread, unsigned int FFTs_per_block>
inline void padded_block_real_fft_2d(InputType* input_data, OutputType* output_data) {
    // Get the CUDA architecture version
    auto arch = example::get_cuda_device_arch();

    // Switch statement to select appropriate architecture template param
    // NOTE: Using fallback to 900 for newer hopper/blackwell architectures
    /* clang-format off */
    switch (arch) {
        // TODO
    }
    /* clang-format on */

    return 0;
}
