#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "../include/block_io.hpp"
#include "../include/common.hpp"
#include "../include/dispatcher.hpp"


// --- Kernel Definition ---
template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_r2c_1d_kernel(ScalarType* input_data, ComplexType* output_data) {
    using complex_type = ComplexType;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(input_data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, output_data, local_fft_id);
}

// --- Launcher Definition ---
template<unsigned int Arch, typename Input_T, typename Output_T, unsigned int FFTSize>
inline void block_fft_r2c_1d_launcher(Input_T* input_data, Output_T* output_data) {
    using namespace cufftdx;

    // R2C and C2R specific properties about the data layout
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::normal>;

    using FFT = decltype(
        Block() +
        Size<FFTSize>() +
        Type<fft_type::r2c>() +
        real_fft_options() +
        Direction<fft_direction::forward>() +
        Precision<Input_T>() +
        ElementsPerThread<8>() +
        FFTsPerBlock<2>() +
        SM<Arch>()
    );

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_r2c_1d_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size
    ));

    // Cast to cuFFTDx complex type form the FFT struct
    using complex_type = typename FFT::value_type;
    complex_type* input_data_t  = reinterpret_cast<complex_type*>(input_data);
    complex_type* output_data_t = reinterpret_cast<complex_type*>(output_data);

    // Launch the kernel and ensure no errors afterwards
    block_fft_r2c_1d_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(input_data_t, output_data_t);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Functor for Dispatcher ---
template<unsigned int Arch, typename Input_T_functor, typename Output_T_functor, unsigned int FFTSize_functor>
struct fft_dispatch_functor {
    void operator()(Input_T_functor* input_data, Output_T_functor* output_data) {
        block_fft_r2c_1d_launcher<Arch, Input_T_functor, Output_T_functor, FFTSize_functor>(input_data, output_data);
    }
};

// --- Public API Function Template Definition ---
template<typename Input_T, typename Output_T, unsigned int FFTSize>
int block_fft_r2c_1d(Input_T* input_data, Output_T* output_data) {
    static_assert(std::is_same_v<Input_T, float>, "block_fft_r2c_1d: Only float input is currently supported.");
    static_assert(std::is_same_v<Output_T, float2>, "block_fft_r2c_1d: Only float2 output is currently supported.");
    static_assert(FFTSize == 16 || FFTSize == 32 || FFTSize == 64 || FFTSize == 128 || FFTSize == 256 || FFTSize == 512 || FFTSize == 1024,
                  "block_fft_r2c_1d: Only FFT sizes of 16, 32, 64, 128, 256, 512, or 1024 are currently supported.");

    // Call the modified dispatcher which determined the architecture
    int result = dispatcher::sm_runner_out_of_place<fft_dispatch_functor, Input_T, Output_T, FFTSize>(input_data, output_data);

    // Runtime assertion that the dispatcher returned successfully
    if (result != 0) {
        std::cerr << "block_fft_r2c_1d: Error in dispatcher, returned code: " << result << std::endl;
        std::exit(result);
    }

    return result;
}


// --- Explicit Template Instantiations ---
template int block_fft_r2c_1d<float, float2, 16>(float*, float2*);
template int block_fft_r2c_1d<float, float2, 32>(float*, float2*);
template int block_fft_r2c_1d<float, float2, 64>(float*, float2*);
template int block_fft_r2c_1d<float, float2, 128>(float*, float2*);
template int block_fft_r2c_1d<float, float2, 256>(float*, float2*);
template int block_fft_r2c_1d<float, float2, 512>(float*, float2*);
template int block_fft_r2c_1d<float, float2, 1024>(float*, float2*);