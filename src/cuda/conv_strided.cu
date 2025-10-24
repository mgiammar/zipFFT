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
#include "../include/memory_strided_utils.cuh"

struct FFTParams {
    float2* data;
    float2* kernel;
    unsigned int inner_batch_count;
    unsigned int outer_batch_count;
    bool kernel_transpose;
};

template <class FFT, class FFT_inv>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void conv_strided_kernel(
        float2* data,
        float2* kernel,
        unsigned int inner_batch_count,
        typename FFT::workspace_type workspace,
        typename FFT::workspace_type inverse_workspace) {

    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    
    // Local array for thread
    float2 thread_data[FFT::storage_size];
    
    load_strided_smem<FFT>(data, thread_data, shared_mem, inner_batch_count * FFT::ffts_per_block);

    FFT().execute(thread_data, shared_mem, workspace);

    const size_t kernel_stride       = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    size_t       kernel_index        = threadIdx.x + blockDim.x * threadIdx.y;
    kernel_index += (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y;

    // complex multiplication in the frequency domain
    for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
        float2 kernel_thread_data = kernel[kernel_index];
        kernel_index += kernel_stride;

        float2 a;
        a.x = thread_data[i].x;
        a.y = thread_data[i].y;

        float2 b;
        b.x = kernel_thread_data.x;
        b.y = kernel_thread_data.y;
        
        float2 c;
        c.x = a.x * b.x - a.y * b.y;
        c.y = a.x * b.y + a.y * b.x;

        thread_data[i].x = c.x;
        thread_data[i].y = c.y;
    }

    FFT_inv().execute(thread_data, shared_mem, inverse_workspace);

    store_strided_smem<FFT>(thread_data, shared_mem, data, inner_batch_count * FFT::ffts_per_block);
}

template <class FFT>
void do_kernel_transpose(struct FFTParams* fft_params, cudaStream_t strm) {
    using namespace cufftdx;

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel_transpose_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    dim3 grid_dims(fft_params->outer_batch_count, fft_params->inner_batch_count);

    kernel_transpose_kernel<FFT>
        <<<grid_dims, FFT::block_dim, FFT::shared_memory_size, strm>>>(
            fft_params->data,
            fft_params->kernel,
            fft_params->inner_batch_count
        );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
}

template <class FFT, class FFT_inv>
void do_convolution(struct FFTParams* fft_params, cudaStream_t strm) {
    using namespace cufftdx;

    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code, strm);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_inverse = make_workspace<FFT_inv>(error_code, strm);
    CUDA_CHECK_AND_EXIT(error_code);

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        conv_strided_kernel<FFT, FFT_inv>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    dim3 grid_dims(fft_params->outer_batch_count, fft_params->inner_batch_count);

    conv_strided_kernel<FFT, FFT_inv>
        <<<grid_dims, FFT::block_dim, FFT::shared_memory_size, strm>>>(
            fft_params->data,
            fft_params->kernel,
            fft_params->inner_batch_count,
            workspace,
            workspace_inverse
        );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
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
    
    if(fft_params->kernel_transpose) {
        do_kernel_transpose<FFT>(fft_params, strm);
        return;
    }

    do_convolution<FFT, FFT_inv>(fft_params, strm);
}

// Common implementation function
void conv_strided_impl(torch::Tensor input, torch::Tensor kernel, bool kernel_transpose) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");
    TORCH_CHECK(kernel.device().is_cuda(),
                "Kernel tensor must be on CUDA device");
    TORCH_CHECK(kernel.dtype() == torch::kComplexFloat,
                "Kernel tensor must be of type torch.complex64");

    unsigned int fft_size, batch_size, outer_batch_count, inner_batch_count;

    c10::cuda::CUDAGuard guard(input.device()); 
    c10::cuda::CUDAGuard guard_kernel(kernel.device());

    // Doing dimension checks for fft size and batch dimension
    if (input.dim() == 2) {
        inner_batch_count = input.size(1);
        fft_size = input.size(0);
        batch_size = 1;
        outer_batch_count = 1;
    } else if (input.dim() == 3) {
        fft_size = input.size(1);
        auto batch_size_pair = get_supported_batches_runtime(fft_size, input.size(2), 0);
        inner_batch_count = batch_size_pair.first;
        batch_size = batch_size_pair.second;
        outer_batch_count = input.size(0);
    } else {
        TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", input.dim(),
                    "D.");
    }

    float2* data_ptr =
        reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    float2* kernel_ptr =
        reinterpret_cast<float2*>(kernel.data_ptr<c10::complex<float>>());


    auto fft_func = get_function_from_table(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    struct FFTParams fft_params;
    fft_params.data = data_ptr;
    fft_params.kernel = kernel_ptr;
    fft_params.inner_batch_count = inner_batch_count;
    fft_params.outer_batch_count = outer_batch_count;

    fft_func(&fft_params);
}

void conv_strided_func(torch::Tensor input, torch::Tensor kernel) {
    conv_strided_impl(input, kernel, false);  // Normal convolution
}

void kernel_transpose_impl(torch::Tensor input, torch::Tensor kernel) {
    conv_strided_impl(input, kernel, true);  // Kernel transpose operation
}

PYBIND11_MODULE(conv_strided, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT convolution using cuFFTDx";
    m.def("conv", &conv_strided_func, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("kernel_transpose", &kernel_transpose_impl, "In-place kernel transpose using cuFFTDx.");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
}