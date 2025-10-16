/* Python bindings for 1-dimensional complex-to-complex strided FFT operations
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
    unsigned int inner_batch_count;
    unsigned int outer_batch_count;
    bool direction;
};

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void fft_strided_kernel(float2* data, unsigned int inner_batch_count, bool disable_compute) {

    float2 thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    
    load_strided_smem<FFT>(data, thread_data, shared_mem, inner_batch_count * FFT::ffts_per_block);

    if (!disable_compute) {
        FFT().execute(thread_data, shared_mem);
    }

    store_strided_smem<FFT>(thread_data, shared_mem, data, inner_batch_count * FFT::ffts_per_block);
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
            fft_strided_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));
        
        dim3 grid_dims(fft_params->outer_batch_count, fft_params->inner_batch_count);

        fft_strided_kernel<FFT><<<grid_dims, FFT::block_dim, FFT::shared_memory_size, strm>>>(
            fft_params->data,
            fft_params->inner_batch_count,
            get_disable_compute()
        );
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    } else {
        using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                         Direction<fft_direction::forward>() +
                         Precision<float>() +
                         ElementsPerThread<8u>() +
                         FFTsPerBlock<BatchSize>() + SM<Arch>());

        // Increase shared memory size, if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            fft_strided_kernel<FFT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

        dim3 grid_dims(fft_params->outer_batch_count, fft_params->inner_batch_count);

        fft_strided_kernel<FFT><<<grid_dims, FFT::block_dim, FFT::shared_memory_size, strm>>>(
            fft_params->data,
            fft_params->inner_batch_count,
            get_disable_compute()
        );
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    }
    
}

// Common implementation function
void fft_strided_impl(torch::Tensor input, bool is_forward) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");

    unsigned int fft_size, batch_size, outer_batch_count, inner_batch_count;

    c10::cuda::CUDAGuard guard(input.device()); 

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

    // Use the dispatch table instead to figure out the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func = get_function_from_table(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    struct FFTParams fft_params;
    fft_params.data = data_ptr;
    fft_params.inner_batch_count = inner_batch_count;
    fft_params.outer_batch_count = outer_batch_count;
    fft_params.direction = !is_forward;

    fft_func(&fft_params);
}

void fft_strided(torch::Tensor input) {
    fft_strided_impl(input, true);  // Forward FFT
}

void ifft_strided(torch::Tensor input) {
    fft_strided_impl(input, false);  // Inverse FFT
}

PYBIND11_MODULE(fft_strided, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT operations using cuFFTDx";
    m.def("fft", &fft_strided, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("ifft", &ifft_strided, "In-place 1D C2C IFFT using cuFFTDx.");
    m.def("set_disable_compute", &set_disable_compute_impl, "Enable/disable the use of custom FFT computations");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
}