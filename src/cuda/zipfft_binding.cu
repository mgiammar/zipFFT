 // C++ file for binding function calls to Python
#include <stdio.h>
#include <iostream>
#include <c10/util/complex.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cufftdx.hpp>

#include "../include/complex_fft_1d.cuh"
#include "../include/real_fft_1d.cuh"
#include "../include/padded_real_fft_1d.cuh"


void fft_c2c_1d(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat, "Input tensor must be of type torch.complex64");
    TORCH_CHECK(input.dim() == 1, "Input tensor must be 1D.");

    float2* data_ptr = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    unsigned int fft_size = input.size(0);

    // Using a switch statement to handle the pre-defined FFT sizes
    switch (fft_size) {
        #include "../generated/fwd_fft_c2c_1d_binding_cases.inc"
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
        #include "../generated/inv_fft_c2c_1d_binding_cases.inc"
    }
}

void fft_r2c_1d(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat, "Input tensor must be of type torch.float32");
    TORCH_CHECK(input.dim() == 1, "Input tensor must be 1D.");

    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(output.dtype() == torch::kComplexFloat, "Output tensor must be of type torch.complex64");
    TORCH_CHECK(output.dim() == 1, "Output tensor must be 1D.");

    TORCH_CHECK(input.size(0) / 2 + 1 == output.size(0), "Output tensor size must be (input_size / 2 + 1)");

    float*       input_ptr  = input.data_ptr<float>();
    float2*      output_ptr = reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>());
    unsigned int fft_size   = input.size(0);

    // Using a switch statement to handle the pre-defined FFT sizes
    switch (fft_size) {
        #include "../generated/fwd_fft_r2c_1d_binding_cases.inc"
    }
}

void ifft_c2r_1d(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat, "Output tensor must be of type torch.complex64");
    TORCH_CHECK(input.dim() == 1, "Input tensor must be 1D.");

    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(output.dtype() == torch::kFloat, "Input tensor must be of type torch.float32");
    TORCH_CHECK(output.dim() == 1, "Output tensor must be 1D.");

    TORCH_CHECK(output.size(0) / 2 + 1 == input.size(0), "Output tensor size must be (output_size / 2 + 1)");

    float2*      input_ptr  = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    float*       output_ptr = output.data_ptr<float>();
    unsigned int fft_size   = output.size(0);

    // Using a switch statement to handle the pre-defined FFT sizes
    switch (fft_size) {
        #include "../generated/inv_fft_c2r_1d_binding_cases.inc"
    }
}

void padded_fft_r2c_1d(torch::Tensor input, torch::Tensor output, unsigned int s) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    // TORCH_CHECK(input.dtype() == torch::kFloat, "Input tensor must be of type torch.float32");
    TORCH_CHECK(input.dim() == 1, "Input tensor must be 1D.");

    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");
    // TORCH_CHECK(output.dtype() == torch::kComplexFloat, "Output tensor must be of type torch.complex64");
    TORCH_CHECK(output.dim() == 1, "Output tensor must be 1D.");

    // TORCH_CHECK(input.size(0) / 2 + 1 == output.size(0), "Output tensor size must be (input_size / 2 + 1)");

    float*       input_ptr  = input.data_ptr<float>();
    float2*      output_ptr = reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>());

    unsigned int fft_size      = s;
    unsigned int signal_length = input.size(0);

    // // Using a switch statement to handle the pre-defined FFT sizes
    // switch (fft_size) {
    //     #include "../generated/forward_fft_r2c_padded_1d_cases.inc"
    // }
  // Nested switch to handle all valid (signal_length, fft_size) combinations
  switch (signal_length) {
    case 128:
        switch (fft_size) {
            case 256:
                padded_block_real_fft_1d<float, float2, 128, 256, true>(input_ptr, output_ptr);
                break;
            case 512:
                padded_block_real_fft_1d<float, float2, 128, 512, true>(input_ptr, output_ptr);
                break;
            case 1024:
                padded_block_real_fft_1d<float, float2, 128, 1024, true>(input_ptr, output_ptr);
                break;
            default:
                TORCH_CHECK(false, "Unsupported FFT size ", fft_size, " for signal length 128");
        }
        break;
    case 256:
        switch (fft_size) {
            case 512:
                padded_block_real_fft_1d<float, float2, 256, 512, true>(input_ptr, output_ptr);
                break;
            case 1024:
                padded_block_real_fft_1d<float, float2, 256, 1024, true>(input_ptr, output_ptr);
                break;
            default:
                TORCH_CHECK(false, "Unsupported FFT size ", fft_size, " for signal length 256");
        }
        break;
    case 512:
        switch (fft_size) {
            case 1024:
                padded_block_real_fft_1d<float, float2, 512, 1024, true>(input_ptr, output_ptr);
                break;
            default:
                TORCH_CHECK(false, "Unsupported FFT size ", fft_size, " for signal length 512");
        }
        break;
    default:
        TORCH_CHECK(false, "Unsupported signal length: ", signal_length);
}
}

PYBIND11_MODULE(zipfft_binding, m) {
    m.doc() = "pybind11 binding example";
    m.def("fft_c2c_1d",  &fft_c2c_1d,  "Run in-place 1D C2C FFT using cuFFTDx.");
    m.def("ifft_c2c_1d", &ifft_c2c_1d, "Run in-place 1D C2C IFFT using cuFFTDx.");
    m.def("fft_r2c_1d",  &fft_r2c_1d,  "Run out-of-place 1D R2C FFT using cuFFTDx.");
    m.def("ifft_c2r_1d", &ifft_c2r_1d, "Run out-of-place 1D C2R IFFT using cuFFTDx.");
    m.def("padded_fft_r2c_1d", &padded_fft_r2c_1d, "Run padded out-of-place 1D R2C FFT using cuFFTDx.");
}
