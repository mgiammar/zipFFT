/* Python bindings for an implicitly zero-padded 1-dimensional real-to-complex
 * FFT operation written using the cuFFTDx library. These values are assumed to
 * be padded on the right-hand side of the signal.
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
};

template <class FFT, int padding_ratio>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void fft_padded_kernel(
        float2* data,
        typename FFT::workspace_type workspace) {

    float2 thread_data[FFT::storage_size];

    const unsigned int local_fft_id = threadIdx.y;
    load_padded_layered<FFT, padding_ratio>(data, thread_data, local_fft_id);

    extern __shared__ __align__(alignof(float4)) float2 shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    store_layered<FFT, padding_ratio>(thread_data, data, local_fft_id);
}

template <unsigned int FFTSize, unsigned int BatchSize, unsigned int Arch>
void dispatch_function(void* params, cudaStream_t strm) {
    struct FFTParams* fft_params = static_cast<FFTParams*>(params);

    using namespace cufftdx;

    using FFT = decltype(Block() + Size<FFTSize>() + Type<fft_type::c2c>() +
                        Direction<fft_direction::forward>() +
                        Precision<float>() +
                        FFTsPerBlock<BatchSize>() + SM<Arch>());    

    constexpr int padding_ratio = get_padding_ratio();

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        fft_padded_kernel<FFT, padding_ratio>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    // create workspaces for FFT
    cudaError_t error_code = cudaSuccess;
    auto workspace = make_workspace<FFT>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Launch the kernel
    fft_padded_kernel<FFT, padding_ratio>
        <<<fft_params->outer_batch_count, FFT::block_dim, FFT::shared_memory_size, strm>>>(
            fft_params->data,
            workspace
        );
}

void fft_layered_impl(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");
    TORCH_CHECK(input.dim() == 3,
                "Input tensor must be 3D");

    TORCH_CHECK(input.size(1) == input.size(2), 
            "Input tensor's 2nd and 3rd dimensions must be equal (padded FFT). Got ",
            input.size(1), " and ", input.size(2), ".");

    unsigned int fft_size = input.size(2);
    unsigned int outer_batch_count = input.size(0) * input.size(1) / get_padding_ratio();
    auto batch_size_pair = get_supported_batches_runtime(fft_size, outer_batch_count, 0);
    outer_batch_count = batch_size_pair.first;
    unsigned int batch_size = batch_size_pair.second;

    // Cast input and output tensors to raw pointers
    float2* input_data = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    auto fft_func =
        get_function_from_table(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);
    
    struct FFTParams fft_params;
    fft_params.data = input_data;
    fft_params.outer_batch_count = outer_batch_count;
    //fft_params.signal_length = signal_length;
    //fft_params.active_layers = layer_count;
    //fft_params.extra_layers = input.size(1) - layer_count;

    fft_func(&fft_params);
}

PYBIND11_MODULE(fft_nonstrided_padded, m) {
    m.doc() = "Implicitly zero-padded 1D real-to-complex FFT using cuFFTDx";
    m.def("fft", &fft_layered_impl, "Perform a padded complex-to-complex FFT on a 1D input tensor");
    m.def("get_supported_sizes", &get_supported_sizes, "Get list of supported FFT sizes");
    m.def("get_supported_padding_ratio", &get_padding_ratio, "Get the supported padding ratio");
}