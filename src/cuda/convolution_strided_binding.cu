/* Python bindings for 1-dimensional complex-to-complex FFT operations
 * written using the cuFFTDx library.
 *
 * This file, while part of the zipFFT package, is not intended for highly
 * efficient FFT operations, and it is instead designed to provide a simplified
 * interface for testing FFT operations written with cuFFTDx and their bindings
 * to Python.
 *
 * Author:  Matthew Giammar
 * E-mail:  mdgiammar@gmail.com
 * License: MIT License
 * Date:    28 July 2025
 */


#include <c10/util/complex.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

#include <stdio.h>

#include "convolution_strided.cuh"

// FFT configuration structure
struct ComplexFFTConfig1D {
    unsigned int fft_size;    // Signal size for FFT
    unsigned int batch_size;  // Number of FFTs (maps to FFTs per block)

    bool operator==(const ComplexFFTConfig1D& other) const {
        return fft_size == other.fft_size && batch_size == other.batch_size;
    }
};


std::pair<unsigned int, unsigned int> factorizePowerOfTwo_bitwise(int n) {
    if (n == 0) {
        return {0, 0};
    }

    int k = 1;
    // Continue as long as the last bit is 0 (i.e., the number is even)
    while ((n & 1) == 0) {
        n >>= 1; // Right shift is equivalent to dividing by 2
        k = k << 1; // Multiply k by 2
    }
    return {k, n};
}

std::pair<unsigned int, unsigned int> get_supported_batches_runtime(unsigned int FFTSize, unsigned int inner_batches) {
    auto factors = factorizePowerOfTwo_bitwise(inner_batches);
    unsigned int batch_size = 1;

    if (FFTSize <= 256 && factors.first >= 32) {
        factors.first = factors.first / 32;
        batch_size = 32;
    } else if (FFTSize <= 512 && factors.first >= 16) {
        factors.first = factors.first / 16;
        batch_size = 16;
    } else if (FFTSize <= 1024 && factors.first >= 8) {
        factors.first = factors.first / 8;
        batch_size = 8;
    } else if (FFTSize <= 2048 && factors.first >= 4) {
        factors.first = factors.first / 4;
        batch_size = 4;
    } else if (FFTSize <= 4096 && factors.first >= 2) {
        factors.first = factors.first / 2;
        batch_size = 2;
    }

    return {factors.first * factors.second, batch_size};
}



// --- END OF CORRECTED CODE ---

// Define supported FFT configurations at the top of the file for easy
// modification Format: (fft_size, batch_size, is_forward)
static constexpr std::array<std::tuple<unsigned int, unsigned int>, 32>
    SUPPORTED_FFT_CONFIGS = {{// Forward FFT configurations
                              {64, 1},
                              {64, 2},
                              {64, 4},
                              {64, 8},
                              {64, 16},
                              {64, 32},

                              {128, 1},
                              {128, 2},
                              {128, 4},
                              {128, 8},
                              {128, 16},
                              {128, 32},

                              {256, 1},
                              {256, 2},
                              {256, 4},
                              {256, 8},
                              {256, 16},
                              {256, 32},

                              {512, 1},
                              {512, 2},
                              {512, 4},
                              {512, 8},
                              {512, 16},

                              {1024, 1},
                              {1024, 2},
                              {1024, 4},
                              {1024, 8},

                              {2048, 1},
                              {2048, 2},
                              {2048, 4},

                              {4096, 1},
                              {4096, 2},
                            }};

// Template dispatch functions for each supported configuration
template <unsigned int FFTSize, unsigned int BatchSize>
void dispatch_fft(float2* data, float2* kernel, unsigned int inner_batch_count, unsigned int outer_batch_count) {
    block_convolution_strided<float2, FFTSize, 8u, BatchSize>(data, kernel, inner_batch_count, outer_batch_count);
}



// Helper template to create dispatch table entries at compile time
template <std::size_t... Is>
constexpr auto make_dispatch_table(std::index_sequence<Is...>) {
    return std::array<  
        std::pair<ComplexFFTConfig1D, std::function<void(float2*, float2*, unsigned int, unsigned int)>>,
        sizeof...(Is)>{
        {{ComplexFFTConfig1D{std::get<0>(SUPPORTED_FFT_CONFIGS[Is]),
                             std::get<1>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft<std::get<0>(config), std::get<1>(config)>;
          }()}...}};
}

// Create the dispatch table automatically
static const auto dispatch_table = make_dispatch_table(
    std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

// Create lookup function with compile-time dispatch table
std::function<void(float2*, float2*, unsigned int, unsigned int)> get_fft_function(unsigned int fft_size,
                                              unsigned int batch_size) {
    // Find matching configuration
    for (const auto& entry : dispatch_table) {
        if (entry.first.fft_size == fft_size &&
            entry.first.batch_size == batch_size) {
            return entry.second;
        }
    }

    // Return nullptr if configuration not found
    return nullptr;
}

// Function to expose supported configurations to Python
std::vector<std::tuple<int, int>> get_supported_conv_configs() {
    std::vector<std::tuple<int, int>> configs;
    configs.reserve(SUPPORTED_FFT_CONFIGS.size());

    for (const auto& config : SUPPORTED_FFT_CONFIGS) {
        configs.emplace_back(static_cast<int>(std::get<0>(config)),
                             static_cast<int>(std::get<1>(config)));
    }

    return configs;
}

// Common implementation function
void conv_c2c_1d_strided_impl(torch::Tensor input, torch::Tensor kernel) {
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
        auto batch_size_pair = get_supported_batches_runtime(fft_size, input.size(2));
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
    auto fft_func = get_fft_function(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    fft_func(data_ptr, kernel_ptr, inner_batch_count, outer_batch_count);
}

void conv_c2c_1d_strided(torch::Tensor input, torch::Tensor kernel) {
    conv_c2c_1d_strided_impl(input, kernel);  // Forward FFT
}

PYBIND11_MODULE(conv1d_strided, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT convolution using cuFFTDx";
    m.def("conv", &conv_c2c_1d_strided, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("get_supported_configs", &get_supported_conv_configs,
          "Get list of supported (fft_size, batch_size, is_forward) "
          "configurations");
}