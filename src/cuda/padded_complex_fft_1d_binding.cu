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

#include "padded_complex_fft_1d.cuh"

// FFT configuration structure
struct PaddedRealFFTConfig1D {
    unsigned int fft_size;       // Total size of the FFT (Signal + 0s)
    //unsigned int signal_length;  // Length of input signals, not padded
    unsigned int batch_size;     // Number of FFTs to compute in parallel
    bool is_forward;             // True for forward FFT, false for inverse
};



// Pre-defined array of supported padded real FFT configurations in the form of
// (fft_size, signal_length, batch_size). Note that backwards padded FFTs are
// not currently implemented, but could be added in the future.
// static constexpr std::array<
//     std::tuple<unsigned int, unsigned int, unsigned int>, 26>
//     SUPPORTED_FFT_CONFIGS = {{// fft_size of 64
//                               {64, 16, 1},
//                               {64, 32, 1},
//                               // fft_size of 128
//                               {128, 16, 1},
//                               {128, 32, 1},
//                               {128, 64, 1},
//                               // fft_size of 256
//                               {256, 16, 1},
//                               {256, 32, 1},
//                               {256, 64, 1},
//                               {256, 128, 1},
//                               // fft_size of 512
//                               {512, 16, 1},
//                               {512, 32, 1},
//                               {512, 64, 1},
//                               {512, 128, 1},
//                               {512, 256, 1},
//                               // fft_size of 1024
//                               {1024, 128, 1},
//                               {1024, 256, 1},
//                               {1024, 512, 1},
//                               // fft_size of 2048
//                               {2048, 128, 1},
//                               {2048, 256, 1},
//                               {2048, 512, 1},
//                               {2048, 1024, 1},
//                               // fft_size of 4096
//                               {4096, 128, 1},
//                               {4096, 256, 1},
//                               {4096, 512, 1},
//                               {4096, 1024, 1},
//                               {4096, 2048, 1}}};

static constexpr std::array<
    std::tuple<unsigned int, unsigned int>, 7>
    SUPPORTED_FFT_CONFIGS = {{
                              {64, 1},
                              {128, 1},
                              {256, 1},
                              {512, 1},
                              {1024, 1},
                              {2048, 1},
                              {4096, 1}}};

// NOTE: Elements-per-thread (8u) current fixed for now, but could be changed
// in the future...
template <unsigned int FFTSize, unsigned int BatchSize>
void dispatch_fft_forward(float2* data, unsigned int signal_length, unsigned int outer_batch_count, unsigned int active_layers, unsigned int extra_layers) {
    // clang-format off
    padded_block_real_fft_1d<float2, FFTSize, true, 8u, BatchSize>(data, signal_length, outer_batch_count, active_layers, extra_layers);
    // clang-format on
}

template <std::size_t... Is>
constexpr auto make_dispatch_table(std::index_sequence<Is...>) {
    return std::array<
        std::pair<PaddedRealFFTConfig1D, std::function<void(float2*, unsigned int , unsigned int, unsigned int , unsigned int)>>,
        sizeof...(Is)>{
        {{PaddedRealFFTConfig1D{std::get<0>(SUPPORTED_FFT_CONFIGS[Is]),
                                std::get<1>(SUPPORTED_FFT_CONFIGS[Is]), true},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft_forward<std::get<0>(config),
                                          std::get<1>(config)>;
          }()}...}};
}


// Create the dispatch table for the supported FFT configurations
static const auto forward_dispatch_table = make_dispatch_table(
    std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

std::function<void(float2*, unsigned int , unsigned int, unsigned int , unsigned int)> get_forward_fft_function(unsigned int fft_size, unsigned int batch_size) {
    for (const auto& entry : forward_dispatch_table) {
        const auto& config = entry.first;
        if (config.fft_size == fft_size &&
            //config.signal_length == signal_length &&
            config.batch_size == batch_size && config.is_forward == true) {
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
        configs.emplace_back(std::get<0>(config), std::get<1>(config));
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
void padded_fft_c2c_1d(torch::Tensor input, int signal_length) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");


    unsigned int fft_size, batch_size, outer_batch_count;

    // Size and shape extractions with necessary checks
    if (input.dim() == 1) {
        fft_size = input.size(0);
        batch_size = 1;
        outer_batch_count = 1;
    } else if (input.dim() == 2) {
        fft_size = input.size(1);
        batch_size = 1;
        outer_batch_count = input.size(0);
    } else {
        TORCH_CHECK(false, "Input tensor must be 1D or 2D");
    }

    // Cast input and output tensors to raw pointers
    float2* input_data = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    // Use the dispatch function to get the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func =
        get_forward_fft_function(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", signal_length=", signal_length, ", batch_size=", batch_size);

    fft_func(input_data, signal_length, outer_batch_count, 1, 0);
}

/**
 * @brief Function (exposed to Python) to perform a padded real-to-complex FFT
 *
 * @param input Real torch tensor of shape (batch_size, signal_length)
 * @param output Complex torch tensor of shape (batch_size, FFTSize/2 + 1)
 * @param s Total size of the FFT, up to the padded length
 */
void padded_fft_c2c_layered(torch::Tensor input, int signal_length, int layer_count) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");
    TORCH_CHECK(input.dim() == 3,
                "Input tensor must be 3D");


    unsigned int fft_size, batch_size, outer_batch_count;

    fft_size = input.size(2);
    batch_size = 1;
    outer_batch_count = input.size(0) * layer_count;


    // Cast input and output tensors to raw pointers
    float2* input_data = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    // Use the dispatch function to get the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func =
        get_forward_fft_function(fft_size, batch_size);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", signal_length=", signal_length, ", batch_size=", batch_size);

    fft_func(input_data, signal_length, outer_batch_count, layer_count, input.size(1) - layer_count);
}



PYBIND11_MODULE(padded_fft1d, m) {
    m.doc() = "Implicitly zero-padded 1D real-to-complex FFT using cuFFTDx";
    m.def("pfft", &padded_fft_c2c_1d,
          "Perform a padded real-to-complex FFT on a 1D input tensor");
    m.def("pfft_layered", &padded_fft_c2c_layered,
          "Perform a padded real-to-complex FFT on a 1D input tensor");
    m.def("get_supported_configs", &get_supported_fft_configs,
          "Get the list of supported padded real FFT configurations");
}