/* Python bindings for an implicitly zero-padded 1-dimensional complex-to-complex
 * FFT operation written using the cuFFTDx library. These values are assumed to
 * be padded on the right-hand side of the signal.
 *
 * This file, while part of the zipFFT package, is not intended for highly
 * efficient FFT operations, and it is instead designed to provide a simplified
 * interface for testing FFT operations written with cuFFTDx and their bindings
 * to Python.
 *
 * Author:  Matthew Giammar
 * E-mail:  mdgiammar@gmail.com
 * License: MIT License
 * Date:    13 October 2025
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

#include "padded_complex_fft_1d.cuh"

// FFT configuration structure
struct PaddedComplexFFTConfig1D {
    unsigned int fft_size;       // Total size of the FFT (Signal + 0s)
    unsigned int signal_length;  // Length of input signals, not padded
    unsigned int batch_size;     // Number of FFTs to compute in parallel
    bool is_forward;             // True for forward FFT, false for inverse
};

// Pre-defined array of supported padded complex FFT configurations in the form of
// (fft_size, signal_length, batch_size, is_forward).
static constexpr std::array<std::tuple<unsigned int, unsigned int, unsigned int, bool>, 52>
    SUPPORTED_FFT_CONFIGS = {{// fft_size of 64 - Forward
                              {64, 16, 1, true},
                              {64, 32, 1, true},
                              // fft_size of 128 - Forward
                              {128, 16, 1, true},
                              {128, 32, 1, true},
                              {128, 64, 1, true},
                              // fft_size of 256 - Forward
                              {256, 16, 1, true},
                              {256, 32, 1, true},
                              {256, 64, 1, true},
                              {256, 128, 1, true},
                              // fft_size of 512 - Forward
                              {512, 16, 1, true},
                              {512, 32, 1, true},
                              {512, 64, 1, true},
                              {512, 128, 1, true},
                              {512, 256, 1, true},
                              // fft_size of 1024 - Forward
                              {1024, 128, 1, true},
                              {1024, 256, 1, true},
                              {1024, 512, 1, true},
                              // fft_size of 2048 - Forward
                              {2048, 128, 1, true},
                              {2048, 256, 1, true},
                              {2048, 512, 1, true},
                              {2048, 1024, 1, true},
                              // fft_size of 4096 - Forward
                              {4096, 128, 1, true},
                              {4096, 256, 1, true},
                              {4096, 512, 1, true},
                              {4096, 1024, 1, true},
                              {4096, 2048, 1, true},
                              // fft_size of 64 - Inverse
                              {64, 16, 1, false},
                              {64, 32, 1, false},
                              // fft_size of 128 - Inverse
                              {128, 16, 1, false},
                              {128, 32, 1, false},
                              {128, 64, 1, false},
                              // fft_size of 256 - Inverse
                              {256, 16, 1, false},
                              {256, 32, 1, false},
                              {256, 64, 1, false},
                              {256, 128, 1, false},
                              // fft_size of 512 - Inverse
                              {512, 16, 1, false},
                              {512, 32, 1, false},
                              {512, 64, 1, false},
                              {512, 128, 1, false},
                              {512, 256, 1, false},
                              // fft_size of 1024 - Inverse
                              {1024, 128, 1, false},
                              {1024, 256, 1, false},
                              {1024, 512, 1, false},
                              // fft_size of 2048 - Inverse
                              {2048, 128, 1, false},
                              {2048, 256, 1, false},
                              {2048, 512, 1, false},
                              {2048, 1024, 1, false},
                              // fft_size of 4096 - Inverse
                              {4096, 128, 1, false},
                              {4096, 256, 1, false},
                              {4096, 512, 1, false},
                              {4096, 1024, 1, false},
                              {4096, 2048, 1, false}}};

// NOTE: Elements-per-thread (8u) current fixed for now, but could be changed
// in the future...
template <unsigned int FFTSize, unsigned int SignalLength, unsigned int BatchSize>
void dispatch_fft_forward(float2* data) {
    // clang-format off
    padded_block_complex_fft_1d<float2, SignalLength, FFTSize, true, 8u, BatchSize>(data);
    // clang-format on
}

template <unsigned int FFTSize, unsigned int SignalLength, unsigned int BatchSize>
void dispatch_fft_inverse(float2* data) {
    // clang-format off
    padded_block_complex_fft_1d<float2, SignalLength, FFTSize, false, 8u, BatchSize>(data);
    // clang-format on
}

template <std::size_t... Is>
constexpr auto make_forward_dispatch_table(std::index_sequence<Is...>) {
    return std::array<std::pair<PaddedComplexFFTConfig1D, std::function<void(float2*)>>,
                      sizeof...(Is)>{
        {{PaddedComplexFFTConfig1D{
              std::get<0>(SUPPORTED_FFT_CONFIGS[Is]), std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<2>(SUPPORTED_FFT_CONFIGS[Is]), std::get<3>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft_forward<std::get<0>(config), std::get<1>(config),
                                          std::get<2>(config)>;
          }()}...}};
}

template <std::size_t... Is>
constexpr auto make_inverse_dispatch_table(std::index_sequence<Is...>) {
    return std::array<std::pair<PaddedComplexFFTConfig1D, std::function<void(float2*)>>,
                      sizeof...(Is)>{
        {{PaddedComplexFFTConfig1D{
              std::get<0>(SUPPORTED_FFT_CONFIGS[Is]), std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<2>(SUPPORTED_FFT_CONFIGS[Is]), std::get<3>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft_inverse<std::get<0>(config), std::get<1>(config),
                                          std::get<2>(config)>;
          }()}...}};
}

// Create the dispatch table for the supported FFT configurations
static const auto forward_dispatch_table =
    make_forward_dispatch_table(std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

static const auto inverse_dispatch_table =
    make_inverse_dispatch_table(std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

std::function<void(float2*)> get_forward_fft_function(unsigned int fft_size,
                                                      unsigned int signal_length,
                                                      unsigned int batch_size) {
    for (const auto& entry : forward_dispatch_table) {
        const auto& config = entry.first;
        if (config.fft_size == fft_size && config.signal_length == signal_length &&
            config.batch_size == batch_size && config.is_forward == true) {
            return entry.second;
        }
    }

    // Return nullptr if configuration not found
    return nullptr;
}

std::function<void(float2*)> get_inverse_fft_function(unsigned int fft_size,
                                                      unsigned int signal_length,
                                                      unsigned int batch_size) {
    for (const auto& entry : inverse_dispatch_table) {
        const auto& config = entry.first;
        if (config.fft_size == fft_size && config.signal_length == signal_length &&
            config.batch_size == batch_size && config.is_forward == false) {
            return entry.second;
        }
    }

    // Return nullptr if configuration not found
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

/**
 * @brief Function (exposed to Python) to perform a padded complex-to-complex forward FFT
 *
 * @param data Complex torch tensor of shape (batch_size, signal_length) (in-place operation)
 * @param s Total size of the FFT, up to the padded length
 */
void padded_fft_c2c_1d(torch::Tensor data, int s) {
    TORCH_CHECK(data.device().is_cuda(), "Data tensor must be on CUDA device");
    TORCH_CHECK(data.dtype() == torch::kComplexFloat,
                "Data tensor must be of type torch.complex64");

    unsigned int fft_size, signal_length, batch_size;
    fft_size = s;

    // Size and shape extractions with necessary checks
    if (data.dim() == 1) {
        signal_length = data.size(0);
        batch_size = 1;
    } else if (data.dim() == 2) {
        signal_length = data.size(1);
        batch_size = data.size(0);
    } else {
        TORCH_CHECK(false, "Data tensor must be 1D or 2D");
    }

    TORCH_CHECK(signal_length <= fft_size, "Signal length must be less than or equal to FFT size");

    // Cast tensor to raw pointer
    float2* data_ptr = reinterpret_cast<float2*>(data.data_ptr<c10::complex<float>>());

    // Use the dispatch function to get the appropriate FFT function
    auto fft_func = get_forward_fft_function(fft_size, signal_length, batch_size);
    TORCH_CHECK(fft_func != nullptr, "Unsupported FFT configuration: fft_size=", fft_size,
                ", signal_length=", signal_length, ", batch_size=", batch_size);

    fft_func(data_ptr);
}

/**
 * @brief Function (exposed to Python) to perform a padded complex-to-complex inverse FFT
 *
 * @param data Complex torch tensor of shape (batch_size, signal_length) (in-place operation)
 * @param s Total size of the FFT, up to the padded length
 */
void padded_ifft_c2c_1d(torch::Tensor data, int s) {
    TORCH_CHECK(data.device().is_cuda(), "Data tensor must be on CUDA device");
    TORCH_CHECK(data.dtype() == torch::kComplexFloat,
                "Data tensor must be of type torch.complex64");

    unsigned int fft_size, signal_length, batch_size;
    fft_size = s;

    // Size and shape extractions with necessary checks
    if (data.dim() == 1) {
        signal_length = data.size(0);
        batch_size = 1;
    } else if (data.dim() == 2) {
        signal_length = data.size(1);
        batch_size = data.size(0);
    } else {
        TORCH_CHECK(false, "Data tensor must be 1D or 2D");
    }

    TORCH_CHECK(signal_length <= fft_size, "Signal length must be less than or equal to FFT size");

    // Cast tensor to raw pointer
    float2* data_ptr = reinterpret_cast<float2*>(data.data_ptr<c10::complex<float>>());

    // Use the dispatch function to get the appropriate FFT function
    auto fft_func = get_inverse_fft_function(fft_size, signal_length, batch_size);
    TORCH_CHECK(fft_func != nullptr, "Unsupported inverse FFT configuration: fft_size=", fft_size,
                ", signal_length=", signal_length, ", batch_size=", batch_size);

    fft_func(data_ptr);
}

PYBIND11_MODULE(padded_cfft1d, m) {
    m.doc() = "Implicitly zero-padded 1D complex-to-complex FFT using cuFFTDx";
    m.def("pcfft", &padded_fft_c2c_1d,
          "Perform a padded complex-to-complex FFT on a 1D input tensor");
    m.def("picfft", &padded_ifft_c2c_1d,
          "Perform a padded complex-to-complex inverse FFT on a 1D input tensor");
    m.def("get_supported_configs", &get_supported_fft_configs,
          "Get the list of supported padded complex FFT configurations");
}