// C++ file for binding function calls to Python
#include <stdio.h>
#include <iostream>
#include <c10/util/complex.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cufftdx.hpp>

#include "../include/fft_c2c_1d.cuh"


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

void ifft_c2c_1d(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat, "Input tensor must be of type torch.complex64");
    TORCH_CHECK(input.dim() == 1, "Input tensor must be 1D.");

    float2* data_ptr = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    unsigned int fft_size = input.size(0);

    // Using a switch statement to handle the pre-defined FFT sizes
    switch (fft_size) {
        case 128:
            block_ifft_c2c_1d<float2, 128>(data_ptr);
            break;
        case 256:
            block_ifft_c2c_1d<float2, 256>(data_ptr);
            break;
        case 512:
            block_ifft_c2c_1d<float2, 512>(data_ptr);
            break;
        case 1024:
            block_ifft_c2c_1d<float2, 1024>(data_ptr);
            break;
        default:
            TORCH_CHECK(false, "Unsupported IFFT size: " + std::to_string(fft_size) + ". Supported sizes are: 128, 256, 512, 1024");
    }
}

PYBIND11_MODULE(binding_cuda, m) {
    m.doc() = "pybind11 binding example";
    m.def("fft_c2c_1d", &fft_c2c_1d, "Run in-place 1D C2C FFT using cuFFTDx.");
    m.def("ifft_c2c_1d", &ifft_c2c_1d, "Run in-place 1D C2C IFFT using cuFFTDx.");
}
