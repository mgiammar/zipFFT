/* Python bindings for 1-dimensional complex-to-complex padded convolution operations
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
    int s;
    bool get_params;
    bool print_params;
    bool kernel_transpose;
    size_t size_result;
};

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void kernel_transpose_kernel(
        float2* data,
        float2* kernel,
        unsigned int inner_batch_count) {

    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    
    // Local array for thread
    float2 thread_data[FFT::storage_size];
    load_strided_padded_smem<FFT>(
        data,
        thread_data,
        shared_mem,
        inner_batch_count * FFT::ffts_per_block,
        cufftdx::size_of<FFT>::value
    );

    const size_t kernel_stride       = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    size_t       kernel_index        = threadIdx.x + blockDim.x * threadIdx.y;
    kernel_index += (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
        kernel[kernel_index] = thread_data[i];
        kernel_index += kernel_stride;
    }
}

template <
unsigned int FFTSize,
unsigned int BatchSize,
unsigned int Arch>
void do_kernel_transpose(struct FFTParams* fft_params, cudaStream_t strm) {
    using namespace cufftdx;


    using FFT_Base = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                                 Precision<float>() +
                                 ElementsPerThread<8u>() +
                                 FFTsPerBlock<BatchSize>() + SM<Arch>());

    using FFT = decltype(FFT_Base() + Direction<fft_direction::forward>());

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
__launch_bounds__(FFT::max_threads_per_block) __global__
    void conv_strided_padded_kernel(
        float2* data,
        float2* kernel,
        unsigned int inner_batch_count,
        int s,
        bool disable_compute) {

    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    
    // Local array for thread
    float2 thread_data[FFT::storage_size];
    load_strided_padded_smem<FFT>(data, thread_data, shared_mem, inner_batch_count * FFT::ffts_per_block, s);

    // Execute the FFT with shared memory
    if (!disable_compute)
        FFT().execute(thread_data, shared_mem);

    __syncthreads();

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
    
    if (!disable_compute)
        FFT_inv().execute(thread_data, shared_mem);

    store_strided_smem<FFT>(thread_data, shared_mem, data, inner_batch_count * FFT::ffts_per_block);
}

template <
unsigned int FFTSize,
unsigned int BatchSize,
unsigned int Arch>
void do_padded_conv(struct FFTParams* fft_params, cudaStream_t strm) {
    using namespace cufftdx;


    using FFT_Base = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                                 Precision<float>() +
                                 ElementsPerThread<8u>() +
                                 FFTsPerBlock<BatchSize>() + SM<Arch>());

    using FFT = decltype(FFT_Base() + Direction<fft_direction::forward>());
    using FFT_inv = decltype(FFT_Base() + Direction<fft_direction::inverse>());

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        conv_strided_padded_kernel<FFT, FFT_inv>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

    dim3 grid_dims(fft_params->outer_batch_count, fft_params->inner_batch_count);

    conv_strided_padded_kernel<FFT, FFT_inv>
        <<<grid_dims, FFT::block_dim, FFT::shared_memory_size, strm>>>(
            fft_params->data,
            fft_params->kernel,
            fft_params->inner_batch_count,
            fft_params->s,
            get_disable_compute()
        );
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
}



template <unsigned int FFTSize, unsigned int BatchSize, unsigned int Arch>
void dispatch_function(void* params, cudaStream_t strm) {
    struct FFTParams* fft_params = static_cast<FFTParams*>(params);

    if(fft_params->get_params) {
        using namespace cufftdx;

        using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                                 Direction<fft_direction::forward>() + 
                                 Precision<float>() +
                                 ElementsPerThread<8u>() +
                                 FFTsPerBlock<BatchSize>() + SM<Arch>());

        if(fft_params->print_params) {
            std::cout << "FFT Size: " << FFTSize << ", Batch Size: " << BatchSize << ", Arch: " << Arch << std::endl;
            std::cout << "Block Dim: (" << FFT::block_dim.x << ", " << FFT::block_dim.y << ", " << FFT::block_dim.z << ")" << std::endl;
            std::cout << "Max Threads per Block: " << FFT::max_threads_per_block << std::endl;
            std::cout << "Elements per Thread: " << FFT::elements_per_thread << std::endl;
            std::cout << "Outer Batch Count: " << fft_params->outer_batch_count << std::endl;
            std::cout << "Inner Batch Count: " << fft_params->inner_batch_count << std::endl;
            std::cout << "FFTs per Block: " << FFT::ffts_per_block << std::endl;
            std::cout << "Shared Memory Size: " << FFT::shared_memory_size << " bytes" << std::endl;
            std::cout << "Storage Size per Thread: " << FFT::storage_size * sizeof(float2) << " bytes" << std::endl;
        }

        fft_params->size_result = FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.z *
            fft_params->outer_batch_count * fft_params->inner_batch_count * FFT::elements_per_thread;

        return;
    }
    
    if(fft_params->kernel_transpose) {
        do_kernel_transpose<FFTSize, BatchSize, Arch>(fft_params, strm);
        return;
    }

    do_padded_conv<FFTSize, BatchSize, Arch>(fft_params, strm);
}

// Common implementation function
void conv_strided_padded_impl(torch::Tensor input, torch::Tensor kernel, int s) {
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

    // Use the dispatch table instead to figure out the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func = get_function_from_table(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    struct FFTParams fft_params;
    fft_params.data = data_ptr;
    fft_params.kernel = kernel_ptr;
    fft_params.inner_batch_count = inner_batch_count;
    fft_params.outer_batch_count = outer_batch_count;
    fft_params.s = s;
    fft_params.get_params = false;
    fft_params.kernel_transpose = false;

    fft_func(&fft_params);
}

void conv_kernel_transpose_impl(torch::Tensor kernel, torch::Tensor kernel_transpose) {
    TORCH_CHECK(kernel.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(kernel.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");

    TORCH_CHECK(kernel_transpose.device().is_cuda(),
                "Kernel tensor must be on CUDA device");
    TORCH_CHECK(kernel_transpose.dtype() == torch::kComplexFloat,
                "Kernel tensor must be of type torch.complex64");

    unsigned int fft_size, batch_size, outer_batch_count, inner_batch_count;

    c10::cuda::CUDAGuard guard(kernel.device()); 

    // Doing dimension checks for fft size and batch dimension
    if (kernel.dim() == 2) {
        inner_batch_count = kernel.size(1);
        fft_size = kernel.size(0);
        batch_size = 1;
        outer_batch_count = 1;
    } else if (kernel.dim() == 3) {
        fft_size = kernel.size(1);
        auto batch_size_pair = get_supported_batches_runtime(fft_size, kernel.size(2), 0);
        inner_batch_count = batch_size_pair.first;
        batch_size = batch_size_pair.second;
        outer_batch_count = kernel.size(0);
    } else {
        TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", kernel.dim(),
                    "D.");
    }

    float2* kernel_ptr =
        reinterpret_cast<float2*>(kernel.data_ptr<c10::complex<float>>());

    float2* kernel_transpose_ptr =
        reinterpret_cast<float2*>(kernel_transpose.data_ptr<c10::complex<float>>());

    // Use the dispatch table instead to figure out the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func = get_function_from_table(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    struct FFTParams fft_params;
    fft_params.data = kernel_ptr;
    fft_params.kernel = kernel_transpose_ptr;
    fft_params.inner_batch_count = inner_batch_count;
    fft_params.outer_batch_count = outer_batch_count;
    fft_params.s = fft_size;
    fft_params.get_params = false;
    fft_params.kernel_transpose = true;

    fft_func(&fft_params);
}


// Common implementation function
size_t conv_kernel_size_impl(torch::Tensor input, bool print_params) {
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
    fft_params.get_params = true;
    fft_params.print_params = print_params;
    fft_params.s = 0;

    fft_func(&fft_params);

    return fft_params.size_result;

    //return fft_func(data_ptr, data_ptr, inner_batch_count, outer_batch_count, 0, true, false);
}

PYBIND11_MODULE(conv_strided_padded, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT convolution using cuFFTDx";
    m.def("conv", &conv_strided_padded_impl, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("conv_kernel_transpose", &conv_kernel_transpose_impl, "Transpose kernel for 1D C2C FFT convolution using cuFFTDx.");
    m.def("conv_kernel_size", &conv_kernel_size_impl, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("set_disable_compute", &set_disable_compute_impl, "Enable/disable the use of custom FFT computations");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
}