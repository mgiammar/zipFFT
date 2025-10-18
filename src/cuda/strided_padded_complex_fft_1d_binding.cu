/* Python bindings for an implicitly zero-padded 1-dimensional complex-to-complex
 * FFT operation with strided load/store using the cuFFTDx library. These values
 * are assumed to be padded on the right-hand side of the signal.
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

#include "strided_padded_complex_fft_1d.cuh"

// FFT configuration structure
struct PaddedStridedComplexFFTConfig1D {
    unsigned int fft_size;       // Total size of the FFT (Signal + 0s)
    unsigned int signal_length;  // Length of input signals, not padded
    unsigned int stride;         // Stride for strided memory access
    unsigned int batch_size;     // Number of FFTs to compute in parallel
    bool is_forward;             // True for forward FFT, false for inverse

    bool operator==(const PaddedStridedComplexFFTConfig1D& other) const {
        return fft_size == other.fft_size && signal_length == other.signal_length &&
               stride == other.stride && batch_size == other.batch_size &&
               is_forward == other.is_forward;
    }
};

// Pre-defined array of supported padded strided complex FFT configurations in the form of
// (fft_size, signal_length, stride, batch_size, is_forward).
static constexpr std::array<
    std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, bool>, 24>
    SUPPORTED_FFT_CONFIGS = {
        {                           // Forward FFT configurations
         {64, 32, 32, 1, true},     // 2D input (64, 32)
         {64, 32, 64, 1, true},     // 2D input (64, 64) with signal_length=32
         {64, 48, 64, 1, true},     // 2D input (64, 64) with signal_length=48
         {64, 32, 128, 1, true},    // 2D input (64, 128) with signal_length=32
         {64, 64, 128, 1, true},    // 2D input (64, 128) with signal_length=64
         {128, 64, 64, 1, true},    // 2D input (128, 64)
         {128, 64, 128, 1, true},   // 2D input (128, 128) with signal_length=64
         {128, 96, 128, 1, true},   // 2D input (128, 128) with signal_length=96
         {128, 128, 256, 1, true},  // 2D input (128, 256) with signal_length=128
         {256, 128, 128, 1, true},  // 2D input (256, 128)
         {256, 192, 256, 1, true},  // 2D input (256, 256) with signal_length=192
         {256, 256, 512, 1, true},  // 2D input (256, 512) with signal_length=256

         // Inverse FFT configurations
         {64, 32, 32, 1, false},     // 2D input (64, 32) with signal_length=32
         {64, 32, 64, 1, false},     // 2D input (64, 64) with signal_length=32
         {64, 48, 64, 1, false},     // 2D input (64, 64) with signal_length=48
         {64, 32, 128, 1, false},    // 2D input (64, 128) with signal_length=32
         {64, 64, 128, 1, false},    // 2D input (64, 128) with signal_length=64
         {128, 64, 64, 1, false},    // 2D input (128, 64)
         {128, 64, 128, 1, false},   // 2D input (128, 128) with signal_length=64
         {128, 96, 128, 1, false},   // 2D input (128, 128) with signal_length=96
         {128, 128, 256, 1, false},  // 2D input (128, 256) with signal_length=128
         {256, 128, 128, 1, false},  // 2D input (256, 128)
         {256, 192, 256, 1, false},  // 2D input (256, 256) with signal_length=192
         {256, 256, 512, 1, false}}};

// NOTE: Elements-per-thread (8u) current fixed for now, but could be changed
// in the future...
template <unsigned int FFTSize, unsigned int SignalLength, unsigned int Stride,
          unsigned int BatchSize>
void dispatch_fft_forward(float2* input_data, float2* output_data, unsigned int batch_size) {
    // clang-format off
    strided_padded_block_complex_fft_1d<float2, SignalLength, FFTSize, Stride, true, 8u, BatchSize>(input_data, output_data, batch_size);
    // clang-format on
}

template <unsigned int FFTSize, unsigned int SignalLength, unsigned int Stride,
          unsigned int BatchSize>
void dispatch_fft_inverse(float2* input_data, float2* output_data, unsigned int batch_size) {
    // clang-format off
    strided_padded_block_complex_fft_1d<float2, SignalLength, FFTSize, Stride, false, 8u, BatchSize>(input_data, output_data, batch_size);
    // clang-format on
}

template <std::size_t... Is>
constexpr auto make_forward_dispatch_table(std::index_sequence<Is...>) {
    return std::array<std::pair<PaddedStridedComplexFFTConfig1D,
                                std::function<void(float2*, float2*, unsigned int)>>,
                      sizeof...(Is)>{
        {{PaddedStridedComplexFFTConfig1D{
              std::get<0>(SUPPORTED_FFT_CONFIGS[Is]), std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<2>(SUPPORTED_FFT_CONFIGS[Is]), std::get<3>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<4>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft_forward<std::get<0>(config), std::get<1>(config),
                                          std::get<2>(config), std::get<3>(config)>;
          }()}...}};
}

template <std::size_t... Is>
constexpr auto make_inverse_dispatch_table(std::index_sequence<Is...>) {
    return std::array<std::pair<PaddedStridedComplexFFTConfig1D,
                                std::function<void(float2*, float2*, unsigned int)>>,
                      sizeof...(Is)>{
        {{PaddedStridedComplexFFTConfig1D{
              std::get<0>(SUPPORTED_FFT_CONFIGS[Is]), std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<2>(SUPPORTED_FFT_CONFIGS[Is]), std::get<3>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<4>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              return dispatch_fft_inverse<std::get<0>(config), std::get<1>(config),
                                          std::get<2>(config), std::get<3>(config)>;
          }()}...}};
}

// Create the dispatch table for the supported FFT configurations
static const auto forward_dispatch_table =
    make_forward_dispatch_table(std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

static const auto inverse_dispatch_table =
    make_inverse_dispatch_table(std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

std::function<void(float2*, float2*, unsigned int)> get_forward_fft_function(
    unsigned int fft_size, unsigned int signal_length, unsigned int stride,
    unsigned int batch_size) {
    for (const auto& entry : forward_dispatch_table) {
        const auto& config = entry.first;
        if (config.fft_size == fft_size && config.signal_length == signal_length &&
            config.stride == stride && config.batch_size == batch_size &&
            config.is_forward == true) {
            return entry.second;
        }
    }

    // Return nullptr if configuration not found
    return nullptr;
}

std::function<void(float2*, float2*, unsigned int)> get_inverse_fft_function(
    unsigned int fft_size, unsigned int signal_length, unsigned int stride,
    unsigned int batch_size) {
    for (const auto& entry : inverse_dispatch_table) {
        const auto& config = entry.first;
        if (config.fft_size == fft_size && config.signal_length == signal_length &&
            config.stride == stride && config.batch_size == batch_size &&
            config.is_forward == false) {
            return entry.second;
        }
    }

    // Return nullptr if configuration not found
    return nullptr;
}

// Function to expose supported configurations to Python
std::vector<std::tuple<int, int, int, int, bool>> get_supported_fft_configs() {
    std::vector<std::tuple<int, int, int, int, bool>> configs;
    configs.reserve(SUPPORTED_FFT_CONFIGS.size());

    for (const auto& config : SUPPORTED_FFT_CONFIGS) {
        configs.emplace_back(std::get<0>(config), std::get<1>(config), std::get<2>(config),
                             std::get<3>(config), std::get<4>(config));
    }

    return configs;
}

/**
 * @brief Common implementation (fwd/inv) for padded strided complex-to-complex 1D FFT.
 *
 * The strided padded FFT is defined using the following cases:
 * - Forward FFT: Input tensor is transformed along a non-contiguous dimension (strided) and is
 * padded with zeros up to some FFT length. This is equivalent to `output = torch.fft.fft(input,
 * n=fft_size, dim=-2)` where `input` has shape `(signal_length, stride)` or `(batch_size,
 * signal_length, stride)`, and `signal_length < fft_size`.
 * - Inverse FFT: Input tensor is transformed along a non-contiguous dimension (strided), but *no
 * zero-padding* is applied to the input at this stage. The output tensor is cropped, in memory, to
 * the desired signal length. This is equivalent to `output = torch.fft.ifft(input, dim=-2)[...,
 * :signal_length, :]` where `input` has shape `(fft_size, stride)` or `(batch_size, fft_size,
 * stride)`, and `signal_length < fft_size`, except that the whole inverse FFT is never written to
 * memory.
 *
 * @param input : Input tensor to transform. If forward FFT, then this tensor either has shape
 * (signal_length, stride) or (batch_size, signal_length, stride). If inverse FFT, then this tensor
 * either has shape (fft_size, stride) or (batch_size, fft_size, stride).
 * @param output : Tensor to store the output results. If forward FFT, then this tensor either has
 * shape (fft_size, stride) or (batch_size, fft_size, stride). If inverse FFT, then this tensor
 * either has shape (signal_length, stride) or (batch_size, signal_length, stride).
 * @param n : The size of the FFT.
 * @param is_forward : Boolean indicating whether the FFT is forward (true) or inverse (false).
 */
void strided_padded_fft_c2c_1d_impl(torch::Tensor input, torch::Tensor output, int n,
                                    bool is_forward) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(output.dtype() == torch::kComplexFloat,
                "Output tensor must be of type torch.complex64");
    TORCH_CHECK(input.dim() == output.dim(),
                "Input and output tensors must have the same number of dimensions");

    unsigned int fft_size, signal_length, batch_size, stride_size;

    // --- Dimension checks for the input tensors ---
    // Last dimension (strided) must be same shape for input and output
    TORCH_CHECK(input.size(-1) == output.size(-1),
                "Input and output tensors must have the same size in the last dimension");
    stride_size = input.size(-1);

    // If batched (dim=3), first dimension (batch) must be same shape for input and output
    if (input.dim() == 3) {
        TORCH_CHECK(input.size(0) == output.size(0),
                    "Input and output tensors must have the same size in the first dimension");
        batch_size = input.size(0);
    } else {
        batch_size = 1;
    }

    // Checks based on forward or inverse FFT definitions
    if (is_forward) {
        // For forward FFT:
        // - input has shape (signal_length, stride) or (batch, signal_length, stride)
        // - output has shape (fft_size, stride) or (batch, fft_size, stride)

        fft_size = n;

        if (input.dim() == 2) {
            signal_length = input.size(0);
            // Check output has correct fft_size dimension
            TORCH_CHECK(output.size(0) == fft_size,
                        "Output tensor first dimension must match FFT size parameter n=", fft_size);
        } else if (input.dim() == 3) {
            signal_length = input.size(1);
            // Check output has correct fft_size dimension
            TORCH_CHECK(
                output.size(1) == fft_size,
                "Output tensor second dimension must match FFT size parameter n=", fft_size);
        } else {
            TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", input.dim(), "D.");
        }
    } else {
        // For inverse FFT:
        // - input has shape (fft_size, stride) or (batch, fft_size, stride)
        // - output has shape (signal_length, stride) or (batch, signal_length, stride)

        fft_size = input.dim() == 2 ? input.size(0) : input.size(1);
        signal_length = output.dim() == 2 ? output.size(0) : output.size(1);

        // Check that data is either 2D or 3D
        if (input.dim() != 2 && input.dim() != 3) {
            TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", input.dim(), "D.");
        }
    }

    float2* input_data_ptr = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    float2* output_data_ptr = reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>());

    // Use the dispatch table to get the appropriate function
    auto fft_func =
        is_forward ? get_forward_fft_function(fft_size, signal_length, stride_size, batch_size)
                   : get_inverse_fft_function(fft_size, signal_length, stride_size, batch_size);

    TORCH_CHECK(fft_func != nullptr,
                "Unsupported padded strided FFT configuration: fft_size=", fft_size,
                ", signal_length=", signal_length, ", stride=", stride_size, ", batch=", batch_size,
                ", is_forward=", is_forward);

    fft_func(input_data_ptr, output_data_ptr, batch_size * stride_size);
}

/**
 * @brief Function (exposed to Python) to perform a padded strided complex-to-complex forward FFT
 *
 * @param input Complex torch tensor of shape (batch_size, signal_length, stride) or (signal_length,
 * stride) (in-place operation)
 * @param s Total size of the FFT, up to the padded length
 */
void strided_padded_fft_c2c_1d(torch::Tensor input, torch::Tensor output, int s) {
    strided_padded_fft_c2c_1d_impl(input, output, s, true);  // Forward FFT
}

/**
 * @brief Function (exposed to Python) to perform a padded strided complex-to-complex inverse FFT
 *
 * @param input Complex torch tensor of shape (batch_size, signal_length, stride) or (signal_length,
 * stride) (in-place operation)
 * @param s Total size of the FFT, up to the padded length
 */
void strided_padded_ifft_c2c_1d(torch::Tensor input, torch::Tensor output, int s) {
    strided_padded_fft_c2c_1d_impl(input, output, s, false);  // Inverse FFT
}

PYBIND11_MODULE(strided_padded_cfft1d, m) {
    m.doc() = "Implicitly zero-padded 1D strided complex-to-complex FFT using cuFFTDx";
    m.def("psfft", &strided_padded_fft_c2c_1d,
          "Perform a padded strided complex-to-complex FFT on a 2D/3D input tensor");
    m.def("psifft", &strided_padded_ifft_c2c_1d,
          "Perform a padded strided complex-to-complex inverse FFT on a 2D/3D input tensor");
    m.def("get_supported_configs", &get_supported_fft_configs,
          "Get the list of supported padded strided complex FFT configurations");
}