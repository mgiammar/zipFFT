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
    bool read_kernel_transposed;
    bool smem_transpose;

    bool get_size;
    unsigned int kernel_size;
};

template <class FFT, class FFT_inv, bool smem_transpose, bool read_kernel_transposed>
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
    
    load_strided<FFT, smem_transpose>(data, thread_data, shared_mem, inner_batch_count * FFT::ffts_per_block);

    FFT().execute(thread_data, shared_mem, workspace);

    apply_kernel<FFT, smem_transpose, read_kernel_transposed>(kernel, thread_data, shared_mem, inner_batch_count);    

    FFT_inv().execute(thread_data, shared_mem, inverse_workspace);

    store_strided<FFT, smem_transpose>(thread_data, shared_mem, data, inner_batch_count * FFT::ffts_per_block);
}

template <class FFT>
void do_kernel_transpose(struct FFTParams* fft_params, cudaStream_t strm) {
    using namespace cufftdx;

    dim3 grid_dims(fft_params->outer_batch_count, fft_params->inner_batch_count);

    kernel_transpose_kernel<FFT>
        <<<grid_dims, FFT::block_dim, 0, strm>>>(
            fft_params->data,
            fft_params->kernel,
            fft_params->inner_batch_count
        );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
}

template <class FFT, class FFT_inv, bool smem_transpose, bool read_kernel_transposed>
void do_convolution(struct FFTParams* fft_params, cudaStream_t strm) {
    using namespace cufftdx;

    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code, strm);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_inverse = make_workspace<FFT_inv>(error_code, strm);
    CUDA_CHECK_AND_EXIT(error_code);

    unsigned int needed_shared_mem_size = cufftdx::size_of<FFT>::value * FFT::ffts_per_block * sizeof(float2);
    unsigned int my_shared_mem_size = (smem_transpose && needed_shared_mem_size >= FFT::shared_memory_size)
                                            ? needed_shared_mem_size : FFT::shared_memory_size;
    

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        conv_strided_kernel<FFT, FFT_inv, smem_transpose, read_kernel_transposed>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, my_shared_mem_size));

    dim3 grid_dims(fft_params->outer_batch_count, fft_params->inner_batch_count);

    conv_strided_kernel<FFT, FFT_inv, smem_transpose, read_kernel_transposed>
        <<<grid_dims, FFT::block_dim, my_shared_mem_size, strm>>>(
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
                                 ElementsPerThread<8u>() +
                                 Precision<float>() +
                                 FFTsPerBlock<BatchSize>() + SM<Arch>());

    using FFT = decltype(FFT_Base() + Direction<fft_direction::forward>());
    using FFT_inv = decltype(FFT_Base() + Direction<fft_direction::inverse>());

    if(fft_params->get_size) {
        fft_params->kernel_size = FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.z *
            fft_params->outer_batch_count * fft_params->inner_batch_count * FFT::elements_per_thread;
        return;
    }
    
    if(fft_params->kernel_transpose) {
        do_kernel_transpose<FFT>(fft_params, strm);
        return;
    }
    
    if(fft_params->read_kernel_transposed) {
        if(fft_params->smem_transpose) {
            do_convolution<FFT, FFT_inv, true, true>(fft_params, strm);
        } else {
            do_convolution<FFT, FFT_inv, false, true>(fft_params, strm);
        }
    } else {
        if(fft_params->smem_transpose) {
            do_convolution<FFT, FFT_inv, true, false>(fft_params, strm);
        } else {
            do_convolution<FFT, FFT_inv, false, false>(fft_params, strm);
        }
    }
    
}

// Common implementation function
unsigned int conv_strided_impl(torch::Tensor input,
                        torch::Tensor kernel,
                        bool kernel_transpose,
                        bool read_kernel_transposed,
                        bool smem_transpose,
                        bool get_size) {
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
    //c10::cuda::CUDAGuard guard_kernel(kernel.device());

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
    fft_params.smem_transpose = smem_transpose;
    fft_params.kernel_transpose = kernel_transpose;
    fft_params.read_kernel_transposed = read_kernel_transposed;
    fft_params.get_size = get_size;
    fft_params.kernel_size = 0;

    fft_func(&fft_params);

    return fft_params.kernel_size;
}

void conv_strided_func(torch::Tensor input, torch::Tensor kernel, bool read_kernel_transposed, bool smem_transpose) {
    conv_strided_impl(input, kernel, false, read_kernel_transposed, smem_transpose, false);
}

void kernel_transpose_impl(torch::Tensor input, torch::Tensor kernel) {
    conv_strided_impl(input, kernel, true, false, false, false);
}

unsigned int kernel_size_impl(torch::Tensor input) {
    return conv_strided_impl(input, input, false, false, false, true);
}

PYBIND11_MODULE(conv_strided, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT convolution using cuFFTDx";
    m.def("conv", &conv_strided_func, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("kernel_transpose", &kernel_transpose_impl, "In-place kernel transpose using cuFFTDx.");
    m.def("kernel_size", &kernel_size_impl, "Get required kernel size for given input tensor.");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
}