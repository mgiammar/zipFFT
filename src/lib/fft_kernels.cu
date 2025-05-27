#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>


// Headers from cuFFTDx examples in CUDALibrarySamples repo
#include "../common/block_io.hpp"
#include "../common/common.hpp"



// Simple 1-dimensional FFT real-to-complex kernel using cuFFTDx
//
// Template:
//  - FFT: The cuFFTDx FFT descriptor type with precision, arch, etc. Assumed
//    to be of fft_type::r2c.
//  - ComplexType: The complex type used by the FFT, e.g., float2
//  - ScalarType: The scalar type used for input data, e.g., float
//
// Arguments:
//  - input_data: Pointer to the real-valued input data in global memory.
//  - output_data: Pointer to the complex-valued output data in global memory.
template<class FFT, class ComplexType = typename FFT::value_type, class ScalarType = typename ComplexType::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_1D_r2c_kernel(ScalarType* input_data, ComplexType* output_data) {
    using complex_type = ComplexType;

    // Local array for the thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0, FFT:ffts_per_block)
    // Then load data from global memory into registers (using supplied example::io)
    const unsigned int local_fft_id = threadIdx.y;
    example::io<FFT>::load(input_data, thread_data, local_fft_id);

    // Execute the FFT (assumes FFT requires no workspace memory)
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Store the results into output_data
    example::io<FFT>::store(thread_data, output_data, local_fft_id);
}


template<unsigned int Arch>
void simple_block_fft_1D_r2c() {
    using namespace cufftdx;

    // Options for the real-to-complex (and complex-to-real) data layout and
    // execution modes for the requested transform.
    using real_fft_options = RealFFTOptions<complex_layout::natural, real_mode::normal>;
    
    // Block-based FFT descriptor
    // TODO: Remove requirement for hardcoded values, possibly make a "lookup"
    // function based on the the size and shape of the FFT problem.
    using FFT = decltype(
        Block() +
        Direction<fft_direction::forward>() +
        Type<fft_type::r2c>() +
        real_fft_options() +
        Precision<float>() +
        Size<128>() +
        SM<Arch>() +
        ElementsPerThread<8>() +
        FFTsPerBlock<4>());
    using complex_type = typename FFT::value_type;
    using real_type    = typename complex_type::value_type;

    // Allocate managed memory for input/output
    // TODO: Make these input data passable arguments. Will allow linkage with
    // PyTorch managed memory in tensors as:
    // PyTorch --> C++ linker --> cuFFTDx kernel --> C++ linker --> PyTorch
    real_type* input_data;
    auto       input_size       = FFT::ffts_per_block * FFT::input_length;
    auto       input_size_bytes = input_size * sizeof(real_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&input_data, input_size_bytes));

    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = float(i);
    }

    complex_type* output_data;
    auto       output_size       = FFT::ffts_per_block * FFT::output_length;
    auto       output_size_bytes = output_size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output_data, output_size_bytes));

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_1D_r2c_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size
    ));

    // Launch the kernel
    block_fft_1D_r2c_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(input_data, output_data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < FFT::output_length; i++) {
        std::cout << output_data[i].x << " " << output_data[i].y << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(input_data));
    CUDA_CHECK_AND_EXIT(cudaFree(output_data));
    std::cout << "Success" << std::endl;
}

template<unsigned int Arch>
struct simple_block_fft_1D_r2c_functor {
    void operator()() { return simple_block_fft_1D_r2c<Arch>(); }
};

int simple_block_fft_1D_r2c_launcher() {
    return example::sm_runner<simple_block_fft_1D_r2c_functor>();
}
