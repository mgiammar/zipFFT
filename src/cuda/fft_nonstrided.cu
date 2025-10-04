/* Python bindings for 1-dimensional complex-to-complex FFT operations
 * written using the cuFFTDx library.
 *
 * This file was adapted from the zipfft package to allow for the testing
 * of the performance of non-strided complex FFTs as implemented with cufftdx.
 *
 * Author:  Shahar Sandhaus
 * E-mail:  shahar.sandhaus@gmail.com
 * License: MIT License
 * Date:    3 Oct 2025
 */

#include <pybind11/pybind11.h>
#include <stdio.h>

#include "../include/dispatch_table_utils.cuh"
#include "../include/memory_nonstrided_utils.cuh"

struct FFTParams {
    float2* data;
    unsigned int outer_batch_count;
    bool direction;
};

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void fft_kernel(float2* data) {

    float2 thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    load_nonstrided<FFT>(data, thread_data, local_fft_id);

    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    FFT().execute(thread_data, shared_mem);

    store_nonstrided<FFT>(thread_data, data, local_fft_id);
}

template <unsigned int FFTSize, unsigned int BatchSize, unsigned int Arch>
void dispatch_function(void* params, cudaStream_t strm) {
    struct FFTParams* fft_params = static_cast<FFTParams*>(params);

    using namespace cufftdx;

    if(fft_params->direction) {
        using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                    Direction<fft_direction::inverse>() +
                    Precision<float>() +
                    ElementsPerThread<8u>() +
                    FFTsPerBlock<BatchSize>() + SM<Arch>());

        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            fft_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

        fft_kernel<FFT><<<fft_params->outer_batch_count, FFT::block_dim, FFT::shared_memory_size, strm>>>(fft_params->data);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    } else {
        using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                         Direction<fft_direction::forward>() +
                         Precision<float>() +
                         ElementsPerThread<8u>() +
                         FFTsPerBlock<BatchSize>() + SM<Arch>());

        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            fft_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

        fft_kernel<FFT><<<fft_params->outer_batch_count, FFT::block_dim, FFT::shared_memory_size, strm>>>(fft_params->data);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    }
    
}

// Common implementation function
void fft_impl(torch::Tensor input, bool direction) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");

    unsigned int fft_size, batch_size, outer_batch_count;

    c10::cuda::CUDAGuard guard(input.device()); 

    // Doing dimension checks for fft size and batch dimension
    if (input.dim() == 1) {
        fft_size = input.size(0);
        batch_size = 1;
        outer_batch_count = 1;
    } else if (input.dim() == 2) {
        fft_size = input.size(1);
        auto batch_size_pair = get_supported_batches_runtime(fft_size, input.size(0), 0);
        outer_batch_count = batch_size_pair.first;
        batch_size = batch_size_pair.second;
    } else {
        TORCH_CHECK(false, "Input tensor must be 1D or 2D. Got ", input.dim(),
                    "D.");
    }

    float2* data_ptr =
        reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    auto fft_func = get_function_from_table(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    struct FFTParams params = {data_ptr, outer_batch_count, direction};
    fft_func(&params);
}

PYBIND11_MODULE(fft_nonstrided, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT operations using cuFFTDx";
    m.def("fft", &fft_impl, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
}