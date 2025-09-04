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

#include "convolution_strided_padded.cuh"

// FFT configuration structure
struct ComplexFFTConfig1D {
    unsigned int fft_size;    // Signal size for FFT
    unsigned int batch_size;  // Number of FFTs (maps to FFTs per block)
    bool kernel_transpose; // Whether to load/store the kernel in transposed format

    bool operator==(const ComplexFFTConfig1D& other) const {
        return fft_size == other.fft_size && batch_size == other.batch_size && kernel_transpose == other.kernel_transpose;
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
static constexpr std::array<std::tuple<unsigned int, unsigned int, bool>, 64>
    SUPPORTED_FFT_CONFIGS = {{// Forward FFT configurations
                              {64, 1, true},
                              {64, 1, false},
                              {64, 2, true},
                              {64, 2, false},
                              {64, 4, true},
                              {64, 4, false},
                              {64, 8, true},
                              {64, 8, false},
                              {64, 16, true},
                              {64, 16, false},
                              {64, 32, true},
                              {64, 32, false},

                              {128, 1, true},
                              {128, 1, false},
                              {128, 2, true},
                              {128, 2, false},
                              {128, 4, true},
                              {128, 4, false},
                              {128, 8, true},
                              {128, 8, false},
                              {128, 16, true},
                              {128, 16, false},
                              {128, 32, true},
                              {128, 32, false},

                              {256, 1, true},
                              {256, 1, false},
                              {256, 2, true},
                              {256, 2, false},
                              {256, 4, true},
                              {256, 4, false},
                              {256, 8, true},
                              {256, 8, false},
                              {256, 16, true},
                              {256, 16, false},
                              {256, 32, true},
                              {256, 32, false},

                              {512, 1, true},
                              {512, 1, false},
                              {512, 2, true},
                              {512, 2, false},
                              {512, 4, true},
                              {512, 4, false},
                              {512, 8, true},
                              {512, 8, false},
                              {512, 16, true},
                              {512, 16, false},

                              {1024, 1, true},
                              {1024, 1, false},
                              {1024, 2, true},
                              {1024, 2, false},
                              {1024, 4, true},
                              {1024, 4, false},
                              {1024, 8, true},
                              {1024, 8, false},

                              {2048, 1, true},
                              {2048, 1, false},
                              {2048, 2, true},
                              {2048, 2, false},
                              {2048, 4, true},
                              {2048, 4, false},

                              {4096, 1, true},
                              {4096, 1, false},
                              {4096, 2, true},
                              {4096, 2, false}
                            }};

// Template dispatch functions for each supported configuration
template <unsigned int FFTSize, unsigned int BatchSize, bool kernel_transpose>
size_t dispatch_fft(float2* data, float2* kernel, unsigned int inner_batch_count, unsigned int outer_batch_count, int s, bool get_params) {
    if (get_params) {
        return block_convolution_transposed_kernel_size<float2, FFTSize, 8u, BatchSize, kernel_transpose>(inner_batch_count, outer_batch_count);
    }

    block_convolution_strided_padded<float2, FFTSize, 8u, BatchSize, kernel_transpose>(data, kernel, inner_batch_count, outer_batch_count, s);

    return 0; // Dummy return to satisfy compiler
}

// Helper template to create dispatch table entries at compile time
template <std::size_t... Is>
constexpr auto make_dispatch_table(std::index_sequence<Is...>) {
    return std::array<  
        std::pair<ComplexFFTConfig1D, std::function<size_t(float2*, float2*, unsigned int, unsigned int, int, bool)>>,
        sizeof...(Is)>{
        {{ComplexFFTConfig1D{std::get<0>(SUPPORTED_FFT_CONFIGS[Is]),
                             std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
                             std::get<2>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft<std::get<0>(config), std::get<1>(config), std::get<2>(config)>;
          }()}...}};
}

// Create the dispatch table automatically
static const auto dispatch_table = make_dispatch_table(
    std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

// Create lookup function with compile-time dispatch table
std::function<size_t(float2*, float2*, unsigned int, unsigned int, int, bool)> get_fft_function(unsigned int fft_size, unsigned int batch_size, bool kernel_transpose) {
    // Find matching configuration
    for (const auto& entry : dispatch_table) {
        if (entry.first.fft_size == fft_size &&
            entry.first.batch_size == batch_size &&
            entry.first.kernel_transpose == kernel_transpose) {
            return entry.second;
        }
    }

    // Return nullptr if configuration not found
    return nullptr;
}

// Function to expose supported configurations to Python
std::vector<std::tuple<int, int, bool>> get_supported_conv_configs() {
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
void conv_c2c_1d_strided_padded(torch::Tensor input, torch::Tensor kernel, int s) {
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
    auto fft_func = get_fft_function(fft_size, batch_size, false);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    fft_func(data_ptr, kernel_ptr, inner_batch_count, outer_batch_count, s, false);
}

void conv_c2c_1d_strided_padded_kernel_transpose(torch::Tensor kernel, torch::Tensor kernel_transpose) {
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
        auto batch_size_pair = get_supported_batches_runtime(fft_size, kernel.size(2));
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
    auto fft_func = get_fft_function(fft_size, batch_size, true);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    fft_func(kernel_ptr, kernel_transpose_ptr, inner_batch_count, outer_batch_count, fft_size, false);
}


// Common implementation function
size_t conv_c2c_1d_strided_padded_kernel_size(torch::Tensor input) {
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

    // Use the dispatch table instead to figure out the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func = get_fft_function(fft_size, batch_size, false);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    return fft_func(data_ptr, data_ptr, inner_batch_count, outer_batch_count, 0, true);
}

PYBIND11_MODULE(conv1d_strided_padded, m) {  // First arg needs to match name in setup.py
    m.doc() = "Complex-to-complex 1D FFT convolution using cuFFTDx";
    m.def("conv", &conv_c2c_1d_strided_padded, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("conv_kernel_transpose", &conv_c2c_1d_strided_padded_kernel_transpose, "Transpose kernel for 1D C2C FFT convolution using cuFFTDx.");
    m.def("conv_kernel_size", &conv_c2c_1d_strided_padded_kernel_size, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("get_supported_configs", &get_supported_conv_configs,
          "Get list of supported (fft_size, batch_size, is_forward) "
          "configurations");
}