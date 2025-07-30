#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "../include/block_io.hpp"
#include "../include/padded_io.hpp"
#include "../include/common.hpp"
#include "../include/dispatcher.hpp"


template<
    int      SignalLength,  // How many elements are actually in the input_data
    class    FFT,
    class    IFFT,
    typename ComplexType = typename FFT::value_type,
    typename ScalarType  = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void padded_block_conv_real_1d_kernel(
    ScalarType*  input_data,
    ScalarType*  output_data,
    ComplexType* filter_data,
    typename FFT::workspace_type  workspace_fwd,
    typename IFFT::workspace_type workspace_inv
) {
    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    // Input is padded (the data to be FFT-ed), output is not padded
    using input_utils_padded  = example::io_padded<FFT, SignalLength>;
    using input_utils         = example::io<FFT>;
    using output_utils        = example::io<IFFT>;

    // Local arrays for FFT and filter data
    complex_type thread_data[FFT::storage_size];
    complex_type filter_data_local[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    input_utils_padded::load(input_data, thread_data, local_fft_id);
    input_utils::load(filter_data, filter_data_local, local_fft_id);

    // Execute FFT on input data
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace_fwd);

    // // Apply point-wise multiplication for convolution
    // for (unsigned int i = 0; i < FFT::storage_size; ++i) {
    //     thread_data[i] *= filter_data_local[i];
    // }

    // Execute inverse FFT to get the convolution result
    IFFT().execute(thread_data, shared_mem, workspace_inv);

    // Save results
    output_utils::store(thread_data, output_data, local_fft_id);
}
    

template<
    unsigned int Arch,
    typename     ScalarType,
    typename     ComplexType,
    unsigned int SignalLength,
    unsigned int FFTSize,
    unsigned int elements_per_thread,
    unsigned int FFTs_per_block>
inline void padded_real_conv_1d_launcher( ScalarType* input_data, ScalarType* output_data, ComplexType* filter_data) {
    using namespace cufftdx;

    // Real FFT specific data layout properties
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::normal>;

    using fft_base = decltype(
        Block() +
        Size<FFTSize>() +
        real_fft_options() +
        Precision<ScalarType>() +
        ElementsPerThread<elements_per_thread>() +
        FFTsPerBlock<FFTs_per_block>() +
        SM<Arch>()
    );
    using fft  = decltype(fft_base() + Type<fft_type::r2c>() + Direction<fft_direction::forward>());
    using ifft = decltype(fft_base() + Type<fft_type::c2r>() + Direction<fft_direction::inverse>());

    // Increase max shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        padded_block_conv_real_1d_kernel<SignalLength, fft, ifft>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fft::shared_memory_size
    ));

    // Type casing
    using complex_type = typename fft::value_type;
    using scalar_type  = typename complex_type::value_type;
    scalar_type*  input_data_t  = reinterpret_cast<scalar_type*>(input_data);
    scalar_type*  output_data_t = reinterpret_cast<scalar_type*>(output_data);
    complex_type* filter_data_t = reinterpret_cast<complex_type*>(filter_data);

    // Allocate workspace for FFT
    cudaError_t error_code = cudaSuccess;
    auto        workspace_fwd = make_workspace<fft>(error_code);
    auto        workspace_inv = make_workspace<ifft>(error_code);

    // DEBUG: Print some of the fft attributes
    printf("FFT block size: %u\n", fft::block_dim);
    printf("FFT shared memory size: %u\n", fft::shared_memory_size);
    printf("FFT storage size: %u\n", fft::storage_size);

    // Launch the kernel
    padded_block_conv_real_1d_kernel<SignalLength, fft, ifft><<<1, fft::block_dim, fft::shared_memory_size>>>(
        input_data_t,
        output_data_t,
        filter_data_t,
        workspace_fwd,
        workspace_inv
    );

    // Check for errors in kernel launch
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}


template<
    unsigned int Arch,
    typename     ScalarType_functor,
    typename     ComplexType_functor,
    unsigned int SignalLength_functor,
    unsigned int FFTSize_functor,
    unsigned int elements_per_thread_functor,
    unsigned int FFTs_per_block_functor>
struct padded_real_conv_1d_functor {
    void operator()(
        ScalarType_functor* input_data,
        ScalarType_functor* output_data,
        ComplexType_functor* filter_data
    ) const {
        padded_real_conv_1d_launcher<
            Arch,
            ScalarType_functor,
            ComplexType_functor,
            SignalLength_functor,
            FFTSize_functor,
            elements_per_thread_functor,
            FFTs_per_block_functor>(input_data, output_data, filter_data);
    }
};


template<
    typename ScalarType,
    typename ComplexType,
    unsigned int SignalLength,
    unsigned int FFTSize,
    unsigned int elements_per_thread,
    unsigned int FFTs_per_block>
int padded_block_conv_real_1d(
    ScalarType* input_data,
    ScalarType* output_data,
    ComplexType* filter_data
) {
    // TODO: Static assertions

    // Call the modified dispatcher which determines the CUDA architecture
    int result = dispatcher::sm_runner_padded_conv<
        padded_real_conv_1d_functor,
        ScalarType,
        ComplexType,
        SignalLength,
        FFTSize,
        elements_per_thread,
        FFTs_per_block>(input_data, output_data, filter_data);

    // Runtime assertion that the dispatcher returned successfully
    if (result != 0) {
        std::cerr << "padded_block_conv_real_1d: Error in dispatcher, result code: " << result << std::endl;
        std::exit(result);
    }

    return result;
}


// --- Template INstantiations ---
template int padded_block_conv_real_1d<float, float2, 64, 128, 8u, 2u>(
    float* input_data,
    float* output_data,
    float2* filter_data
);
template int padded_block_conv_real_1d<float, float2, 64, 256, 8u, 2u>(
    float* input_data,
    float* output_data,
    float2* filter_data
);
template int padded_block_conv_real_1d<float, float2, 64, 512, 8u, 2u>(
    float* input_data,
    float* output_data,
    float2* filter_data
);