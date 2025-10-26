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
    bool inverse;
    bool smem_transpose;
};

template <class FFT, bool smem_traspose>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void fft_strided_kernel(float2* data, unsigned int inner_batch_count, typename FFT::workspace_type workspace) {

    float2 thread_data[FFT::storage_size];
    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    unsigned int stride_len = inner_batch_count * FFT::ffts_per_block;
    
    load_strided<FFT, smem_traspose>(data, thread_data, shared_mem, stride_len);

    FFT().execute(thread_data, shared_mem, workspace);

    store_strided<FFT, smem_traspose>(thread_data, shared_mem, data, stride_len);
}

template<unsigned int FFTSize, unsigned int BatchSize, unsigned int Arch, bool inverse, bool smem_traspose>
void exec_fft_function(struct FFTParams* fft_params, cudaStream_t strm) {
    using namespace cufftdx;

    using FFT_Base = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                    ElementsPerThread<8u>() +
                    Precision<float>() +
                    FFTsPerBlock<BatchSize>() + SM<Arch>());

    using FFT = std::conditional_t<
        inverse,
        decltype(FFT_Base() + Direction<fft_direction::inverse>()),
        decltype(FFT_Base() + Direction<fft_direction::forward>())
    >;

    unsigned int needed_shared_mem_size = cufftdx::size_of<FFT>::value * FFT::ffts_per_block * sizeof(float2);
    unsigned int my_shared_mem_size = (smem_traspose && needed_shared_mem_size >= FFT::shared_memory_size)
                                            ? needed_shared_mem_size : FFT::shared_memory_size;
    
    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        fft_strided_kernel<FFT, smem_traspose>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, my_shared_mem_size));

    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code, strm);
    CUDA_CHECK_AND_EXIT(error_code);

    dim3 grid_dims(fft_params->outer_batch_count, fft_params->inner_batch_count);
    
    fft_strided_kernel<FFT, smem_traspose><<<grid_dims, FFT::block_dim, my_shared_mem_size, strm>>>(
        fft_params->data,
        fft_params->inner_batch_count,
        workspace
    );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
}

template <unsigned int FFTSize, unsigned int BatchSize, unsigned int Arch>
void dispatch_function(void* params, cudaStream_t strm) {
    struct FFTParams* fft_params = static_cast<FFTParams*>(params);

    using namespace cufftdx;

    if(fft_params->smem_transpose) {
        if(fft_params->inverse) {
            exec_fft_function<FFTSize, BatchSize, Arch, true, true>(fft_params, strm);
        } else {
            exec_fft_function<FFTSize, BatchSize, Arch, false, true>(fft_params, strm);
        }
    } else {
        if(fft_params->inverse) {
            exec_fft_function<FFTSize, BatchSize, Arch, true, false>(fft_params, strm);
        } else {
            exec_fft_function<FFTSize, BatchSize, Arch, false, false>(fft_params, strm);
        }
    }    
}

// Common implementation function
void fft_strided_impl(torch::Tensor input, bool inverse, bool smem_transpose) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");

    TORCH_CHECK(input.dim() == 3, 
                "Input tensor must be 3D for strided FFTs. Got ", input.dim(),
                "D.");

    c10::cuda::CUDAGuard guard(input.device()); 

    unsigned int fft_size = input.size(1);
    auto batch_size_pair = get_supported_batches_runtime(fft_size, input.size(2), 0);
    unsigned int inner_batch_count = batch_size_pair.first;
    unsigned int batch_size = batch_size_pair.second;
    unsigned int outer_batch_count = input.size(0);

    float2* data_ptr =
        reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    auto fft_func = get_function_from_table(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    struct FFTParams fft_params;
    fft_params.data = data_ptr;
    fft_params.inner_batch_count = inner_batch_count;
    fft_params.outer_batch_count = outer_batch_count;
    fft_params.inverse = inverse;
    fft_params.smem_transpose = smem_transpose;

    fft_func(&fft_params);
}

void fft_strided(torch::Tensor input, bool smem_transpose) {
    fft_strided_impl(input, false, smem_transpose);  // Forward FFT
}

void ifft_strided(torch::Tensor input, bool smem_transpose) {
    fft_strided_impl(input, true, smem_transpose);  // Inverse FFT
}

PYBIND11_MODULE(fft_strided, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT operations using cuFFTDx";
    m.def("fft", &fft_strided, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("ifft", &ifft_strided, "In-place 1D C2C IFFT using cuFFTDx.");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
}