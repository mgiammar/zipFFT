/* Python bindings for an implicitly zero-padded 1-dimensional real-to-complex
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
 * Date:    30 July 2025
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

#include "padded_real_fft_1d.cuh"

// FFT configuration structure
struct PaddedRealFFTConfig1D {
    unsigned int fft_size;       // Total size of the FFT (Signal + 0s)
    unsigned int signal_length;  // Length of input signals, not padded
    unsigned int batch_size;     // Number of FFTs to compute in parallel
    bool is_forward;             // True for forward FFT, false for inverse
};

// Pre-defined array of supported padded real FFT configurations in the form of
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
void dispatch_fft_forward(float* input_data, float2* output_data) {
    // clang-format off
    padded_block_real_fft_1d<float, float2, SignalLength, FFTSize, true, 8u, BatchSize>(input_data, output_data);
    // clang-format on
}

template <unsigned int FFTSize, unsigned int SignalLength, unsigned int BatchSize>
void dispatch_fft_inverse(float2* input_data, float* output_data) {
    // clang-format off
    padded_block_real_fft_1d<float2, float, SignalLength, FFTSize, false, 8u, BatchSize>(input_data, output_data);
    // clang-format on
}

template <std::size_t... Is>
constexpr auto make_forward_dispatch_table(std::index_sequence<Is...>) {
    return std::array<std::pair<PaddedRealFFTConfig1D, std::function<void(float*, float2*)>>,
                      sizeof...(Is)>{
        {{PaddedRealFFTConfig1D{
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
    return std::array<std::pair<PaddedRealFFTConfig1D, std::function<void(float2*, float*)>>,
                      sizeof...(Is)>{
        {{PaddedRealFFTConfig1D{
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

std::function<void(float*, float2*)> get_forward_fft_function(unsigned int fft_size,
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

std::function<void(float2*, float*)> get_inverse_fft_function(unsigned int fft_size,
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
 * @brief Function (exposed to Python) to perform a padded real-to-complex FFT
 *
 * @param input Real torch tensor of shape (batch_size, signal_length)
 * @param output Complex torch tensor of shape (batch_size, FFTSize/2 + 1)
 * @param s Total size of the FFT, up to the padded length
 */
void padded_fft_r2c_1d(torch::Tensor input, torch::Tensor output, int s) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat, "Input tensor must be of type torch.float32");

    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(output.dtype() == torch::kComplexFloat,
                "Output tensor must be of type torch.complex64");

    TORCH_CHECK(input.dim() == output.dim(),
                "Input and output tensors must have the same number of dimensions");

    unsigned int fft_size, signal_length, batch_size;
    fft_size = s;

    // Size and shape extractions with necessary checks
    if (input.dim() == 1) {
        signal_length = input.size(0);
        batch_size = 1;
        TORCH_CHECK(output.size(0) == fft_size / 2 + 1,
                    "Output tensor size must be "
                    "equal to (s / 2 + 1) for a forward padded real-to-complex FFT.");
    } else if (input.dim() == 2) {
        signal_length = input.size(1);
        batch_size = input.size(0);
        TORCH_CHECK(output.size(0) == input.size(0),
                    "Output tensor batch size must match input tensor batch size");
        TORCH_CHECK(output.size(1) == fft_size / 2 + 1,
                    "Output tensor size must be "
                    "equal to (s / 2 + 1) for a forward padded real-to-complex FFT.");
    } else {
        TORCH_CHECK(false, "Input tensor must be 1D or 2D");
    }

    // Cast input and output tensors to raw pointers
    float* input_data = input.data_ptr<float>();
    float2* output_data = reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>());

    // Use the dispatch function to get the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func = get_forward_fft_function(fft_size, signal_length, batch_size);
    TORCH_CHECK(fft_func != nullptr, "Unsupported FFT configuration: fft_size=", fft_size,
                ", signal_length=", signal_length, ", batch_size=", batch_size);

    fft_func(input_data, output_data);
}

/**
 * @brief Function (exposed to Python) to perform a padded complex-to-real inverse FFT
 *
 * @param input Complex torch tensor of shape (batch_size, FFTSize/2 + 1)
 * @param output Real torch tensor of shape (batch_size, signal_length)
 * @param s Total size of the FFT, up to the padded length
 */
void padded_fft_c2r_1d(torch::Tensor input, torch::Tensor output, int s) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");

    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(output.dtype() == torch::kFloat, "Output tensor must be of type torch.float32");

    TORCH_CHECK(input.dim() == output.dim(),
                "Input and output tensors must have the same number of dimensions");

    unsigned int fft_size, signal_length, batch_size;
    fft_size = s;

    // Size and shape extractions with necessary checks
    if (input.dim() == 1) {
        signal_length = output.size(0);
        batch_size = 1;
        TORCH_CHECK(input.size(0) == fft_size / 2 + 1,
                    "Input tensor size must be equal to (s / 2 + 1) for an inverse padded "
                    "complex-to-real FFT.");
    } else if (input.dim() == 2) {
        signal_length = output.size(1);
        batch_size = input.size(0);
        TORCH_CHECK(output.size(0) == input.size(0),
                    "Output tensor batch size must match input tensor batch size");
        TORCH_CHECK(input.size(1) == fft_size / 2 + 1,
                    "Input tensor size must be equal to (s / 2 + 1) for an inverse padded "
                    "complex-to-real FFT.");
    } else {
        TORCH_CHECK(false, "Input tensor must be 1D or 2D");
    }

    // Cast input and output tensors to raw pointers
    float2* input_data = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    float* output_data = output.data_ptr<float>();

    // Use the dispatch function to get the appropriate FFT function
    auto fft_func = get_inverse_fft_function(fft_size, signal_length, batch_size);
    TORCH_CHECK(fft_func != nullptr, "Unsupported inverse FFT configuration: fft_size=", fft_size,
                ", signal_length=", signal_length, ", batch_size=", batch_size);

    fft_func(input_data, output_data);
}

PYBIND11_MODULE(padded_rfft1d, m) {
    m.doc() = "Implicitly zero-padded 1D real-to-complex FFT using cuFFTDx";
    m.def("prfft", &padded_fft_r2c_1d, "Perform a padded real-to-complex FFT on a 1D input tensor");
    m.def("pirfft", &padded_fft_c2r_1d,
          "Perform a padded complex-to-real inverse FFT on a 1D input tensor");
    m.def("get_supported_configs", &get_supported_fft_configs,
          "Get the list of supported padded real FFT configurations");
}