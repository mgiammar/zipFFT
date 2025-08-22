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

#include "complex_fft_1d.cuh"

// FFT configuration structure
struct ComplexFFTConfig1D {
    unsigned int fft_size;    // Signal size for FFT
    unsigned int batch_size;  // Number of FFTs (maps to FFTs per block)
    bool is_forward;          // True for forward FFT, false for inverse

    bool operator==(const ComplexFFTConfig1D& other) const {
        return fft_size == other.fft_size && batch_size == other.batch_size &&
               is_forward == other.is_forward;
    }
};

// Define supported FFT configurations at the top of the file for easy
// modification Format: (fft_size, batch_size, is_forward)
static constexpr std::array<std::tuple<unsigned int, unsigned int, bool>, 28>
    SUPPORTED_FFT_CONFIGS = {{// Forward FFT configurations
                              {64, 1, true},
                              {64, 2, true},
                              {128, 1, true},
                              {128, 2, true},
                              {256, 1, true},
                              {256, 2, true},
                              {512, 1, true},
                              {512, 2, true},
                              {1024, 1, true},
                              {1024, 2, true},
                              {2048, 1, true},
                              {2048, 2, true},
                              {4096, 1, true},
                              {4096, 2, true},
                              // Inverse FFT configurations
                              {64, 1, false},
                              {64, 2, false},
                              {128, 1, false},
                              {128, 2, false},
                              {256, 1, false},
                              {256, 2, false},
                              {512, 1, false},
                              {512, 2, false},
                              {1024, 1, false},
                              {1024, 2, false},
                              {2048, 1, false},
                              {2048, 2, false},
                              {4096, 1, false},
                              {4096, 2, false}}};

// Template dispatch functions for each supported configuration
template <unsigned int FFTSize, unsigned int BatchSize, bool IsForwardFFT>
void dispatch_fft(float2* data, unsigned int outer_batch_count) {
    block_complex_fft_1d<float2, FFTSize, IsForwardFFT, 8u, BatchSize>(data, outer_batch_count);
}

// Helper template to create dispatch table entries at compile time
template <std::size_t... Is>
constexpr auto make_dispatch_table(std::index_sequence<Is...>) {
    return std::array<
        std::pair<ComplexFFTConfig1D, std::function<void(float2*, unsigned int)>>,
        sizeof...(Is)>{
        {{ComplexFFTConfig1D{std::get<0>(SUPPORTED_FFT_CONFIGS[Is]),
                             std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
                             std::get<2>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft<std::get<0>(config), std::get<1>(config),
                                  std::get<2>(config)>;
          }()}...}};
}

// Create the dispatch table automatically
static const auto dispatch_table = make_dispatch_table(
    std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

// Create lookup function with compile-time dispatch table
std::function<void(float2*, unsigned int)> get_fft_function(unsigned int fft_size,
                                              unsigned int batch_size,
                                              bool is_forward) {
    // Find matching configuration
    for (const auto& entry : dispatch_table) {
        if (entry.first.fft_size == fft_size &&
            entry.first.batch_size == batch_size &&
            entry.first.is_forward == is_forward) {
            return entry.second;
        }
    }

    // Return nullptr if configuration not found
    return nullptr;
}

// Function to expose supported configurations to Python
std::vector<std::tuple<int, int, bool>> get_supported_fft_configs() {
    std::vector<std::tuple<int, int, bool>> configs;
    configs.reserve(SUPPORTED_FFT_CONFIGS.size());

    for (const auto& config : SUPPORTED_FFT_CONFIGS) {
        configs.emplace_back(static_cast<int>(std::get<0>(config)),
                             static_cast<int>(std::get<1>(config)),
                             std::get<2>(config));
    }

    return configs;
}

// Common implementation function
void fft_c2c_1d_impl(torch::Tensor input, bool is_forward) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");

    unsigned int fft_size, batch_size, outer_batch_count;

    c10::cuda::CUDAGuard guard(input.device()); 

    // Doing dimension checks for fft size and batch dimension
    if (input.dim() == 1) {
        fft_size = input.size(0);
        batch_size = 1;
        outer_batch_count = 1;
    } else if (input.dim() == 2) {
        fft_size = input.size(1);
        batch_size = 1;
        outer_batch_count = input.size(0);
        if(outer_batch_count % 2 == 0) {
            batch_size = 2;
            outer_batch_count /= 2;
        }

        //printf("Batch size set to %u based on outer batch count of %u\n", batch_size, outer_batch_count);
        
    } else {
        TORCH_CHECK(false, "Input tensor must be 1D or 2D. Got ", input.dim(),
                    "D.");
    }

    float2* data_ptr =
        reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    // Use the dispatch table instead to figure out the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func = get_fft_function(fft_size, batch_size, is_forward);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size, ", is_forward=", is_forward);

    fft_func(data_ptr, outer_batch_count);
}

void fft_c2c_1d(torch::Tensor input) {
    fft_c2c_1d_impl(input, true);  // Forward FFT
}

void ifft_c2c_1d(torch::Tensor input) {
    fft_c2c_1d_impl(input, false);  // Inverse FFT
}

PYBIND11_MODULE(cfft1d, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT operations using cuFFTDx";
    m.def("fft", &fft_c2c_1d, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("ifft", &ifft_c2c_1d, "In-place 1D C2C IFFT using cuFFTDx.");
    m.def("get_supported_configs", &get_supported_fft_configs,
          "Get list of supported (fft_size, batch_size, is_forward) "
          "configurations");
}