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
__global__ void fft_kernel(float2* data, typename FFT::workspace_type workspace) {

    float2 thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;

    load_nonstrided<FFT>(data, thread_data, local_fft_id);

    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    store_nonstrided<FFT>(thread_data, data, local_fft_id);
}

template <class FFT>
void exec_fft_function(struct FFTParams* fft_params, cudaStream_t strm) {
    using namespace cufftdx;

    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code, strm);
    CUDA_CHECK_AND_EXIT(error_code);

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    fft_kernel<FFT><<<fft_params->outer_batch_count, FFT::block_dim, FFT::shared_memory_size, strm>>>(
        fft_params->data, workspace
    );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
}

template <unsigned int FFTSize, unsigned int BatchSize, unsigned int Arch>
void dispatch_function(void* params, cudaStream_t strm) {
    struct FFTParams* fft_params = static_cast<FFTParams*>(params);

    using namespace cufftdx;

    if(fft_params->direction) {
        using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                    Direction<fft_direction::inverse>() +
                    Precision<float>() +
                    FFTsPerBlock<BatchSize>() + SM<Arch>());

        exec_fft_function<FFT>(fft_params, strm);
    } else {
        using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                         Direction<fft_direction::forward>() +
                         Precision<float>() +
                         FFTsPerBlock<BatchSize>() + SM<Arch>());

        exec_fft_function<FFT>(fft_params, strm);
    }
}

// Common implementation function
void fft_impl(torch::Tensor input, bool direction) {
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

    struct FFTParams params = {data_ptr, outer_batch_count, direction};
    fft_func(&params);
}

void fft_forward(torch::Tensor input) {
    fft_impl(input, false);
}

void fft_inverse(torch::Tensor input) {
    fft_impl(input, true);
}   

PYBIND11_MODULE(fft_nonstrided, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT operations using cuFFTDx";
    m.def("fft", &fft_forward, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("ifft", &fft_inverse, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
}