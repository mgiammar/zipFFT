/* Python bindings for 1-dimensional complex-to-complex FFT operations with
 * strided load/store using the cuFFTDx library.
 *
 * Author:  Matthew Giammar
 * E-mail:  mdgiammar@gmail.com
 * License: MIT License
 * Date:    11 October 2025
 */

#include <c10/util/complex.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <torch/extension.h>

#include <array>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

#include "strided_complex_fft_1d.cuh"

// FFT configuration structure
struct StridedComplexFFTConfig1D {
    unsigned int fft_size;
    unsigned int stride;
    unsigned int batch_size;
    bool is_forward;

    bool operator==(const StridedComplexFFTConfig1D& other) const {
        return fft_size == other.fft_size && stride == other.stride &&
               batch_size == other.batch_size && is_forward == other.is_forward;
    }
};

// Define supported FFT configurations at the top of the file for easy
// modification Format: (fft_size, stride, batch_size, is_forward)
static constexpr std::array<std::tuple<unsigned int, unsigned int, unsigned int, bool>, 24>
    SUPPORTED_FFT_CONFIGS = {{                      // Forward FFT configurations
                              {64, 32, 1, true},    // 2D input (64, 32)
                              {64, 32, 4, true},    // 2D input (64, 32), 4 batches
                              {64, 64, 1, true},    // 2D input (64, 64)
                              {64, 64, 4, true},    // 2D input (64, 64), 4 batches
                              {64, 128, 1, true},   // 2D input (64, 128)
                              {64, 128, 4, true},   // 2D input (64, 128), 4 batches
                              {128, 64, 1, true},   // 2D input (128, 64)
                              {128, 64, 4, true},   // 2D input (128, 64), 4 batches
                              {128, 128, 1, true},  // 2D input (128, 128)
                              {128, 128, 4, true},  // 2D input (128, 128), 4 batches
                              {128, 256, 1, true},  // 2D input (128, 256)
                              {128, 256, 4, true},  // 2D input (128, 256), 4 batches
                                                    // Inverse FFT configurations
                              {64, 32, 1, false},
                              {64, 32, 4, false},
                              {64, 64, 1, false},
                              {64, 64, 4, false},
                              {64, 128, 1, false},
                              {64, 128, 4, false},
                              {128, 64, 1, false},
                              {128, 64, 4, false},
                              {128, 128, 1, false},
                              {128, 128, 4, false},
                              {128, 256, 1, false},
                              {128, 256, 4, false}}};

// Template dispatch functions for each supported configuration
template <unsigned int FFTSize, unsigned int Stride, unsigned int BatchSize, bool IsForwardFFT>
void dispatch_fft(float2* data) {
    strided_block_complex_fft_1d<float2, FFTSize, Stride, BatchSize, IsForwardFFT, 8u, 1u>(data);
}

// Helper template to create dispatch table entries at compile time
template <std::size_t... Is>
constexpr auto make_dispatch_table(std::index_sequence<Is...>) {
    return std::array<
        std::pair<StridedComplexFFTConfig1D, std::function<void(float2*)>>,
        sizeof...(Is)>{
        {{StridedComplexFFTConfig1D{
              std::get<0>(SUPPORTED_FFT_CONFIGS[Is]), std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<2>(SUPPORTED_FFT_CONFIGS[Is]), std::get<3>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft<std::get<0>(config), std::get<1>(config), std::get<2>(config),
                                  std::get<3>(config)>;
          }()}...}};
}

// Create the dispatch table automatically
static const auto dispatch_table =
    make_dispatch_table(std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

// Create lookup function with compile-time dispatch table
std::function<void(float2*)> get_fft_function(unsigned int fft_size,
                                              unsigned int stride,
                                              unsigned int batch_size,
                                              bool is_forward) {
    // Find matching configuration
    for (const auto& entry : dispatch_table) {
        if (entry.first == StridedComplexFFTConfig1D{fft_size, stride, batch_size, is_forward}) {
            return entry.second;
        }
    }

    // If no match found return nullptr
    return nullptr;
}

// Function to expose supported configurations to Python
std::vector<std::tuple<int, int, int, bool>> get_supported_fft_configs() {
    std::vector<std::tuple<int, int, int, bool>> configs;
    configs.reserve(SUPPORTED_FFT_CONFIGS.size());

    for (const auto& config : SUPPORTED_FFT_CONFIGS) {
        configs.emplace_back(std::get<0>(config), std::get<1>(config), std::get<2>(config),
                             std::get<3>(config));
    }

    return configs;
}

// Common implementation function
void strided_fft_c2c_1d_impl(torch::Tensor input, bool is_forward) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64 (float2)");

    unsigned int fft_size, batch_size, stride_size;

    // Doing dimension checks for fft size and batch dimensions
    if (input.dim() == 2) {  // shape (fft_size, stride)
        fft_size = input.size(0);
        stride_size = input.size(1);
        batch_size = 1;
    } else if (input.dim() == 3) {  // shape (batch, fft_size, stride)
        batch_size = input.size(0);
        fft_size = input.size(1);
        stride_size = input.size(2);
    } else {
        TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", input.dim(), "D.");
    }

    float2* data_ptr = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    // Use the dispatch table to get the appropriate function
    auto fft_func = get_fft_function(fft_size, stride_size, batch_size, is_forward);
    TORCH_CHECK(fft_func != nullptr, "Unsupported FFT configuration: size=", fft_size,
                ", stride=", stride_size, ", batch=", batch_size, ", is_forward=", is_forward);

    // Execute the FFT function (batch_size is now baked into the template)
    fft_func(data_ptr);
}

void strided_fft_c2c_1d(torch::Tensor input) {
    strided_fft_c2c_1d_impl(input, true);  // Forward FFT
}

void strided_ifft_c2c_1d(torch::Tensor input) {
    strided_fft_c2c_1d_impl(input, false);  // Inverse FFT
}

PYBIND11_MODULE(strided_cfft1d, m) {  // First arg needs to match name in setup.py
    m.doc() = "1D strided complex-to-complex FFT using cuFFTDx";
    m.def("fft", &strided_fft_c2c_1d, "1D strided complex-to-complex FFT (C2C)");
    m.def("ifft", &strided_ifft_c2c_1d, "1D strided complex-to-complex inverse FFT (C2C)");
    m.def("get_supported_fft_configs", &get_supported_fft_configs,
          "Get list of supported (fft_size, stride, batch_size, is_forward) configurations");
}