/* Python bindings for 1-dimensional complex-to-complex convolution operations
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

#include <array>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

#include "../include/dispatch_table_utils.cuh"
#include "../include/memory_nonstrided_utils.cuh"

struct FFTParams {
    float2* data;
    float scale;
    unsigned int outer_batch_count;
};

template <class FFT, class FFT_inv>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void conv_scale_nonstrided_kernel(
        float2* data,
        float scale,
        typename FFT::workspace_type workspace,
        typename FFT::workspace_type inverse_workspace) {

    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    
    // Local array for thread
    float2 thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;
    load_nonstrided<FFT>(data, thread_data, local_fft_id);

    FFT().execute(thread_data, shared_mem, workspace);

    // complex multiplication in the frequency domain
    for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
        thread_data[i].x *= scale;
        thread_data[i].y *= scale;
    }

    FFT_inv().execute(thread_data, shared_mem, inverse_workspace);

    store_nonstrided<FFT>(thread_data, data, local_fft_id);
}

template <unsigned int FFTSize, unsigned int BatchSize, unsigned int Arch>
void dispatch_function(void* params, cudaStream_t strm) {
    struct FFTParams* fft_params = static_cast<FFTParams*>(params);

    using namespace cufftdx;

    using FFT_Base = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                                 Precision<float>() +
                                 FFTsPerBlock<BatchSize>() + SM<Arch>());

    using FFT = decltype(FFT_Base() + Direction<fft_direction::forward>());
    using FFT_inv = decltype(FFT_Base() + Direction<fft_direction::inverse>());

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        conv_scale_nonstrided_kernel<FFT, FFT_inv>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code, strm);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_inverse = make_workspace<FFT_inv>(error_code, strm);
    CUDA_CHECK_AND_EXIT(error_code);

    dim3 grid_dims(fft_params->outer_batch_count, 1);

    conv_scale_nonstrided_kernel<FFT, FFT_inv>
        <<<grid_dims, FFT::block_dim, FFT::shared_memory_size, strm>>>(
            fft_params->data,
            fft_params->scale,
            workspace,
            workspace_inverse
        );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
}

// Common implementation function
void conv_strided_impl(torch::Tensor input, float scale) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");

    TORCH_CHECK(input.dim() == 3, 
                "Input tensor must be 3D (batch_size_0, batch_size_1, fft_size). Got ",
                input.dim(), "D.");

    c10::cuda::CUDAGuard guard(input.device());

    unsigned int fft_size = input.size(2);
    auto batch_size_pair = get_supported_batches_runtime(fft_size, input.size(0) * input.size(1), 0);
    unsigned int outer_batch_count = batch_size_pair.first;
    unsigned int batch_size = batch_size_pair.second;

    float2* data_ptr =
        reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    auto fft_func = get_function_from_table(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    struct FFTParams fft_params;
    fft_params.data = data_ptr;
    fft_params.scale = scale;
    fft_params.outer_batch_count = outer_batch_count;

    fft_func(&fft_params);
}

PYBIND11_MODULE(conv_nonstrided, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT convolution using cuFFTDx";
    m.def("conv", &conv_strided_impl, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
}