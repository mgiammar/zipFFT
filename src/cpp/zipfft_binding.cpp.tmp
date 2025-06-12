// C++ file for binding function calls to Python
#include <stdio.h>
#include <iostream>
#include <c10/util/complex.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cufftdx.hpp>

// #include "../common/common.hpp"
// #include "fft_kernels.cuh"
// #include "fft_1d.cuh"
#include "../include/fft_c2c_1d.cuh"


// // Binding for the FFT launcher
// int run_fft() {
//     return simple_block_fft_1D_r2c_launcher();
// }

// int run_fft_1D_c2c(torch::Tensor input) {
//     TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
//     TORCH_CHECK(input.dtype() == torch::kComplexFloat, "Input tensor must be of type torch.complex64");
//     TORCH_CHECK(input.dim() == 1 && input.size(0) == 128, "Input tensor must be 1D with size 128");

//     float2* data_ptr = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
//     // auto* data_ptr = reinterpret_cast<float2*>(input.data_ptr<std::complex<float>>());

//     return simple_block_fft_1D_c2c_launcher(data_ptr);
// }

void fft_c2c_1d(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat, "Input tensor must be of type torch.complex64");
    TORCH_CHECK(input.dim() == 1, "Input tensor must be 1D.");

    float2* data_ptr = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    unsigned int fft_size = input.size(0);

    // Using a switch statement to handle the pre-defined FFT sizes
    switch (fft_size) {
        case 128:
            block_fft_c2c_1d<float2, 128>(data_ptr);
            break;
        case 256:
            block_fft_c2c_1d<float2, 256>(data_ptr);
            break;
        case 512:
            block_fft_c2c_1d<float2, 512>(data_ptr);
            break;
        case 1024:
            block_fft_c2c_1d<float2, 1024>(data_ptr);
            break;
        default:
            TORCH_CHECK(false, "Unsupported FFT size: " + std::to_string(fft_size) + ". Supported sizes are: 128, 256, 512, 1024");
    }
}

PYBIND11_MODULE(binding_cuda, m) {
    m.doc() = "pybind11 binding example";
    // m.def("run_fft", &run_fft, "Run the simple_block_fft_1D_r2c_launcher FFT kernel");
    m.def("fft_c2c_1d", &fft_c2c_1d, "Run in-place 1D C2C FFT using cuFFTDx.");
}


// int main() {
//     // Example usage of the FFT launcher
//     const size_t fft_size = 128;
//     float2* data;
//     size_t data_size = fft_size * sizeof(float2);
    
//     // Allocate managed memory for input/output
//     CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, data_size));

//     // Initialize input data (for demonstration purposes)
//     for (size_t i = 0; i < fft_size; i++) {
//         data[i].x = static_cast<float>(i);
//         data[i].y = 0.0f; // Imaginary part initialized to zero
//     }

//     // Launch the FFT
//     simple_block_fft_1D_c2c_launcher(data);

//     // Print output (for demonstration purposes)
//     std::cout << "Output after FFT:" << std::endl;
//     for (size_t i = 0; i < fft_size; i++) {
//         std::cout << "data[" << i << "] = (" << data[i].x << ", " << data[i].y << ")" << std::endl;
//     }

//     // Free allocated memory
//     CUDA_CHECK_AND_EXIT(cudaFree(data));
    
//     return 0;
// }