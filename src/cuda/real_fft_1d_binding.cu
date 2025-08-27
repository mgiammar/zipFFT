/* Python bindings for 1-dimensional real-to-complex (and complex-to-real) FFT
 * operations written using the cuFFTDx library.
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

#include "real_fft_1d.cuh"

// FFT configuration structure
struct RealFFTConfig1D {
    unsigned int fft_size;    // Signal size for FFT
    unsigned int batch_size;  // Number of FFTs (maps to FFTs per block)
    bool is_forward;          // True for forward FFT, false for inverse

    bool operator==(const RealFFTConfig1D& other) const {
        return fft_size == other.fft_size && batch_size == other.batch_size &&
               is_forward == other.is_forward;
    }
};

// Pre-defined array of supported real FFT configurations in the form of
// (fft_size, batch_size). This is used to create the dispatch table and
// expose a list of supported configurations to Python.
static constexpr std::array<std::tuple<unsigned int, unsigned int>, 14>
    SUPPORTED_FFT_CONFIGS = {{{64, 1},
                              {64, 2},
                              {128, 1},
                              {128, 2},
                              {256, 1},
                              {256, 2},
                              {512, 1},
                              {512, 2},
                              {1024, 1},
                              {1024, 2},
                              {2048, 1},
                              {2048, 2},
                              {4096, 1},
                              {4096, 2}}};

// NOTE: Elements-per-thread (8u) currently fixed for now, but could be changed
// in the future...
template <unsigned int FFTSize, unsigned int BatchSize>
void dispatch_fft_forward(float* input_data, float2* output_data, unsigned int outer_batch_count) {
    block_real_fft_1d<float, float2, FFTSize, true, 8u, BatchSize>(input_data,
                                                                   output_data, outer_batch_count);
}

// NOTE: Elements-per-thread (8u) currently fixed for now, but could be changed
// in the future...
template <unsigned int FFTSize, unsigned int BatchSize>
void dispatch_fft_inverse(float2* input_data, float* output_data, unsigned int outer_batch_count) {
    block_real_fft_1d<float2, float, FFTSize, false, 8u, BatchSize>(
        input_data, output_data, outer_batch_count);
}

template <std::size_t... Is>
constexpr auto make_forward_dispatch_table(std::index_sequence<Is...>) {
    return std::array<
        std::pair<RealFFTConfig1D, std::function<void(float*, float2*, unsigned int)>>,
        sizeof...(Is)>{
        {{RealFFTConfig1D{std::get<0>(SUPPORTED_FFT_CONFIGS[Is]),
                          std::get<1>(SUPPORTED_FFT_CONFIGS[Is]), true},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft_forward<std::get<0>(config),
                                          std::get<1>(config)>;
          }()}...}};
}

template <std::size_t... Is>
constexpr auto make_inverse_dispatch_table(std::index_sequence<Is...>) {
    return std::array<
        std::pair<RealFFTConfig1D, std::function<void(float2*, float*, unsigned int)>>,
        sizeof...(Is)>{
        {{RealFFTConfig1D{std::get<0>(SUPPORTED_FFT_CONFIGS[Is]),
                          std::get<1>(SUPPORTED_FFT_CONFIGS[Is]), false},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft_inverse<std::get<0>(config),
                                          std::get<1>(config)>;
          }()}...}};
}

// Create the dispatch tables automatically
static const auto forward_dispatch_table = make_forward_dispatch_table(
    std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

static const auto inverse_dispatch_table = make_inverse_dispatch_table(
    std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

std::function<void(float*, float2*, unsigned int)> get_forward_fft_function(
    unsigned int fft_size, unsigned int batch_size, bool is_forward) {
    // Find matching configuration
    for (const auto& entry : forward_dispatch_table) {
        if (entry.first.fft_size == fft_size &&
            entry.first.batch_size == batch_size &&
            entry.first.is_forward == is_forward) {
            return entry.second;
        }
    }

    // Return nullptr if configuration not found
    return nullptr;
}

std::function<void(float2*, float*, unsigned int)> get_inverse_fft_function(
    unsigned int fft_size, unsigned int batch_size, bool is_forward) {
    // Find matching configuration
    for (const auto& entry : inverse_dispatch_table) {
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
std::vector<std::tuple<int, int>> get_supported_fft_configs() {
    std::vector<std::tuple<int, int>> configs;
    configs.reserve(SUPPORTED_FFT_CONFIGS.size());

    for (const auto& config : SUPPORTED_FFT_CONFIGS) {
        configs.emplace_back(static_cast<int>(std::get<0>(config)),
                             static_cast<int>(std::get<1>(config)));
    }

    return configs;
}

void fft_r2c_1d(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat,
                "Input tensor must be of type torch.float32");

    TORCH_CHECK(output.device().is_cuda(),
                "Output tensor must be on CUDA device");
    TORCH_CHECK(output.dtype() == torch::kComplexFloat,
                "Output tensor must be of type torch.complex64");

    TORCH_CHECK(
        output.dim() == input.dim(),
        "Input and output tensors must have the same number of dimensions");

    unsigned int fft_size, batch_size, outer_batch_count;

    // Size and shape extractions with necessary checks for correctness
    if (input.dim() == 1) {
        fft_size = input.size(0);
        batch_size = 1;
        outer_batch_count = 1;
        TORCH_CHECK(output.size(0) == (fft_size / 2 + 1),
                    "Output tensor size must be (input_size / 2 + 1)");
    } else if (input.dim() == 2) {
        fft_size = input.size(1);
        //batch_size = input.size(0);
        batch_size = 1;
        outer_batch_count = input.size(0);
        if(outer_batch_count % 2 == 0) {
            batch_size = 2;
            outer_batch_count /= 2;
        }
        TORCH_CHECK(
            output.size(0) == input.size(0),
            "Output tensor batch size must match input tensor batch size");
        TORCH_CHECK(output.size(1) == (fft_size / 2 + 1),
                    "Output tensor size along the transformed dimension must "
                    "be (input_size / 2 + 1)");
    } else {
        TORCH_CHECK(false, "Input tensor must be 1D or 2D.");
    }

    // Cast input and output tensors to raw pointers
    float* input_data = input.data_ptr<float>();
    float2* output_data =
        reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>());

    // Use the dispatch function to get the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func = get_forward_fft_function(fft_size, batch_size, true);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    fft_func(input_data, output_data, outer_batch_count);
}

void fft_c2r_1d(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");

    TORCH_CHECK(output.device().is_cuda(),
                "Output tensor must be on CUDA device");
    TORCH_CHECK(output.dtype() == torch::kFloat,
                "Output tensor must be of type torch.float32");

    TORCH_CHECK(
        output.dim() == input.dim(),
        "Input and output tensors must have the same number of dimensions");

    unsigned int fft_size, batch_size, outer_batch_count;

    // Size and shape extractions with necessary checks for correctness
    if (output.dim() == 1) {
        fft_size = output.size(0);
        batch_size = 1;
        outer_batch_count = 1;
        TORCH_CHECK(input.size(0) == fft_size / 2 + 1,
                    "Output tensor size must be (input_size * 2 - 2)");
    } else if (output.dim() == 2) {
        fft_size = output.size(1);
        //batch_size = output.size(0);
        batch_size = 1;
        outer_batch_count = input.size(0);
        if(outer_batch_count % 2 == 0) {
            batch_size = 2;
            outer_batch_count /= 2;
        }
        TORCH_CHECK(
            output.size(0) == input.size(0),
            "Output tensor batch size must match input tensor batch size");
        TORCH_CHECK(input.size(1) == fft_size / 2 + 1,
                    "Output tensor size along the transformed dimension must "
                    "match input size");
    } else {
        TORCH_CHECK(false, "Input tensor must be 1D or 2D.");
    }

    // Cast input and output tensors to raw pointers
    float2* input_data =
        reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    float* output_data = output.data_ptr<float>();

    // Use the dispatch function to get the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func = get_inverse_fft_function(fft_size, batch_size, false);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    fft_func(input_data, output_data, outer_batch_count);
}

PYBIND11_MODULE(rfft1d, m) {
    m.doc() = "Real-to-complex 1D FFT operations using cuFFTDx";
    m.def("rfft", &fft_r2c_1d, "Perform 1D real-to-complex forward FFT");
    m.def("irfft", &fft_c2r_1d, "Perform 1D complex-to-real inverse FFT");
    m.def("get_supported_configs", &get_supported_fft_configs,
          "Get list of supported (fft_size, batch_size) configurations");
}