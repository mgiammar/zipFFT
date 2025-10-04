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
};

template <class FFT, class FFT_inv>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void conv_strided_kernel(float2* data, float2* kernel, unsigned int inner_batch_count) {

    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    
    // Local array for thread
    float2 thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;
    
    //example::io_strided<FFT>::load_strided_smem(data, thread_data, shared_mem, local_fft_id, inner_batch_count * FFT::ffts_per_block);
    load_strided_smem<FFT>(data, thread_data, shared_mem, local_fft_id, inner_batch_count * FFT::ffts_per_block);

    // Execute the FFT with shared memory
    FFT().execute(thread_data, shared_mem);

    __syncthreads();

    float2 kernel_thread_data[FFT::storage_size];
    //example::io_strided<FFT>::load_strided_smem(kernel, kernel_thread_data, shared_mem, local_fft_id, inner_batch_count * FFT::ffts_per_block);

    load_strided_smem<FFT>(kernel, kernel_thread_data, shared_mem, local_fft_id, inner_batch_count * FFT::ffts_per_block);

    // complex multiplication in the frequency domain
    for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
        float2 a;
        a.x = thread_data[i].x;
        a.y = thread_data[i].y;

        float2 b;
        b.x = kernel_thread_data[i].x;
        b.y = kernel_thread_data[i].y;
        
        float2 c;
        c.x = a.x * b.x - a.y * b.y;
        c.y = a.x * b.y + a.y * b.x;

        thread_data[i].x = c.x;
        thread_data[i].y = c.y;
    }

    FFT_inv().execute(thread_data, shared_mem);

    // Save results back to global memory
    //example::io_strided<FFT>::store_strided_smem<FFT::value_type, false>(thread_data, shared_mem, data, local_fft_id, inner_batch_count * FFT::ffts_per_block);

    store_strided_smem<FFT>(thread_data, shared_mem, data, local_fft_id, inner_batch_count * FFT::ffts_per_block, false);
}

template <unsigned int FFTSize, unsigned int BatchSize, unsigned int Arch>
void dispatch_function(void* params, cudaStream_t strm) {
    struct FFTParams* fft_params = static_cast<FFTParams*>(params);

    using namespace cufftdx;

    using FFT_Base = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                                 Precision<float>() +
                                 ElementsPerThread<8u>() +
                                 FFTsPerBlock<BatchSize>() + SM<Arch>());

    using FFT = decltype(FFT_Base() + Direction<fft_direction::forward>());
    using FFT_inv = decltype(FFT_Base() + Direction<fft_direction::inverse>());

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        conv_strided_kernel<FFT, FFT_inv>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    dim3 grid_dims(fft_params->outer_batch_count, fft_params->inner_batch_count);

    conv_strided_kernel<FFT, FFT_inv>
        <<<grid_dims, FFT::block_dim, FFT::shared_memory_size, strm>>>(fft_params->data, fft_params->kernel, fft_params->inner_batch_count);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
}

// Common implementation function
void conv_strided_impl(torch::Tensor input, torch::Tensor kernel) {
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

PYBIND11_MODULE(conv_strided, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT convolution using cuFFTDx";
    m.def("conv", &conv_strided_impl, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
}