/* Python bindings for 2-dimensional real convolution/cross-correlation operations
 * with implicit zero-padding using the cuFFTDx library.
 *
 * This implements convolution/cross-correlation where the input signal is smaller
 * than the FFT size, with automatic zero-padding.
 *
 * Author:  Matthew Giammar
 * E-mail:  mdgiammar@gmail.com
 * License: MIT License
 * Date:    01 November 2025
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

#include "real_conv_2d.cuh"

// Convolution configuration structure for padded 2D real convolution
struct PaddedRealConvConfig2D {
    // Signal dimensions (actual data size)
    unsigned int signal_length_y;
    unsigned int signal_length_x;
    // FFT dimensions (padded size)
    unsigned int fft_size_y;
    unsigned int fft_size_x;
    // Batch size for parallel processing
    unsigned int batch_size;
    // Cross-correlation flag
    bool cross_correlate;
    // Transpose axes flag - whether to transpose the FFT workspace internally
    bool transpose_axes;
    // Conv data is transposed flag - whether the convolution data is pre-transposed
    bool conv_data_is_transposed;

    bool operator==(const PaddedRealConvConfig2D& other) const {
        return signal_length_x == other.signal_length_x &&
               signal_length_y == other.signal_length_y && fft_size_x == other.fft_size_x &&
               fft_size_y == other.fft_size_y && batch_size == other.batch_size &&
               cross_correlate == other.cross_correlate && transpose_axes == other.transpose_axes &&
               conv_data_is_transposed == other.conv_data_is_transposed;
    }
};

// Define supported convolution configurations
// Format: (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch_size, cross_correlate,
// transpose_axes, conv_data_is_transposed)
static constexpr std::array<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                                       unsigned int, bool, bool, bool>,
                            17 * 8>
    SUPPORTED_CONV_CONFIGS = {{
        // Convolution configurations (transpose_axes=false, conv_data_is_transposed=false) (17
        // total)
        {48, 48, 64, 64, 1, false, false, false},      // (48, 48) -> (64, 64) FFT, batch 1
        {48, 48, 64, 64, 8, false, false, false},      // (48, 48) -> (64, 64) FFT, batch 8
        {96, 96, 128, 128, 1, false, false, false},    // (96, 96) -> (128, 128) FFT, batch 1
        {96, 96, 128, 128, 8, false, false, false},    // (96, 96) -> (128, 128) FFT, batch 8
        {192, 192, 256, 256, 1, false, false, false},  // (192, 192) -> (256, 256) FFT, batch 1
        {192, 192, 256, 256, 4, false, false, false},  // (192, 192) -> (256, 256) FFT, batch 4
        {384, 384, 512, 512, 1, false, false, false},  // (384, 384) -> (512, 512) FFT, batch 1
        {384, 384, 512, 512, 4, false, false, false},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, false, false, false},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, false, false, false},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, false, false, false},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, false, false, false},  // (192, 384) -> (256, 512) FFT, batch 4
        {384, 384, 512, 512, 4, false, false, false},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, false, false, false},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, false, false, false},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, false, false, false},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, false, false, false},  // (192, 384) -> (256, 512) FFT, batch 4

        // Cross-correlation configurations (transpose_axes=false, conv_data_is_transposed=false)
        {48, 48, 64, 64, 1, true, false, false},      // (48, 48) -> (64, 64) FFT, batch 1
        {48, 48, 64, 64, 8, true, false, false},      // (48, 48) -> (64, 64) FFT, batch 8
        {96, 96, 128, 128, 1, true, false, false},    // (96, 96) -> (128, 128) FFT, batch 1
        {96, 96, 128, 128, 8, true, false, false},    // (96, 96) -> (128, 128) FFT, batch 8
        {192, 192, 256, 256, 1, true, false, false},  // (192, 192) -> (256, 256) FFT, batch 1
        {192, 192, 256, 256, 4, true, false, false},  // (192, 192) -> (256, 256) FFT, batch 4
        {384, 384, 512, 512, 1, true, false, false},  // (384, 384) -> (512, 512) FFT, batch 1
        {384, 384, 512, 512, 4, true, false, false},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, true, false, false},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, true, false, false},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, true, false, false},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, true, false, false},  // (192, 384) -> (256, 512) FFT, batch 4
        {384, 384, 512, 512, 4, true, false, false},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, true, false, false},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, true, false, false},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, true, false, false},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, true, false, false},  // (192, 384) -> (256, 512) FFT, batch 4

        // Convolution configurations (transpose_axes=true, conv_data_is_transposed=false)
        {48, 48, 64, 64, 1, false, true, false},      // (48, 48) -> (64, 64) FFT, batch 1
        {48, 48, 64, 64, 8, false, true, false},      // (48, 48) -> (64, 64) FFT, batch 8
        {96, 96, 128, 128, 1, false, true, false},    // (96, 96) -> (128, 128) FFT, batch 1
        {96, 96, 128, 128, 8, false, true, false},    // (96, 96) -> (128, 128) FFT, batch 8
        {192, 192, 256, 256, 1, false, true, false},  // (192, 192) -> (256, 256) FFT, batch 1
        {192, 192, 256, 256, 4, false, true, false},  // (192, 192) -> (256, 256) FFT, batch 4
        {384, 384, 512, 512, 1, false, true, false},  // (384, 384) -> (512, 512) FFT, batch 1
        {384, 384, 512, 512, 4, false, true, false},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, false, true, false},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, false, true, false},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, false, true, false},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, false, true, false},  // (192, 384) -> (256, 512) FFT, batch 4
        {384, 384, 512, 512, 4, false, true, false},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, false, true, false},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, false, true, false},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, false, true, false},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, false, true, false},  // (192, 384) -> (256, 512) FFT, batch 4

        // Cross-correlation configurations (transpose_axes=true, conv_data_is_transposed=false)
        {48, 48, 64, 64, 1, true, true, false},      // (48, 48) -> (64, 64) FFT, batch 1
        {48, 48, 64, 64, 8, true, true, false},      // (48, 48) -> (64, 64) FFT, batch 8
        {96, 96, 128, 128, 1, true, true, false},    // (96, 96) -> (128, 128) FFT, batch 1
        {96, 96, 128, 128, 8, true, true, false},    // (96, 96) -> (128, 128) FFT, batch 8
        {192, 192, 256, 256, 1, true, true, false},  // (192, 192) -> (256, 256) FFT, batch 1
        {192, 192, 256, 256, 4, true, true, false},  // (192, 192) -> (256, 256) FFT, batch 4
        {384, 384, 512, 512, 1, true, true, false},  // (384, 384) -> (512, 512) FFT, batch 1
        {384, 384, 512, 512, 4, true, true, false},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, true, true, false},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, true, true, false},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, true, true, false},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, true, true, false},  // (192, 384) -> (256, 512) FFT, batch 4
        {384, 384, 512, 512, 4, true, true, false},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, true, true, false},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, true, true, false},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, true, true, false},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, true, true, false},  // (192, 384) -> (256, 512) FFT, batch 4

        // Convolution configurations (transpose_axes=false, conv_data_is_transposed=true)
        {48, 48, 64, 64, 1, false, false, true},      // (48, 48) -> (64, 64) FFT, batch 1
        {48, 48, 64, 64, 8, false, false, true},      // (48, 48) -> (64, 64) FFT, batch 8
        {96, 96, 128, 128, 1, false, false, true},    // (96, 96) -> (128, 128) FFT, batch 1
        {96, 96, 128, 128, 8, false, false, true},    // (96, 96) -> (128, 128) FFT, batch 8
        {192, 192, 256, 256, 1, false, false, true},  // (192, 192) -> (256, 256) FFT, batch 1
        {192, 192, 256, 256, 4, false, false, true},  // (192, 192) -> (256, 256) FFT, batch 4
        {384, 384, 512, 512, 1, false, false, true},  // (384, 384) -> (512, 512) FFT, batch 1
        {384, 384, 512, 512, 4, false, false, true},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, false, false, true},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, false, false, true},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, false, false, true},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, false, false, true},  // (192, 384) -> (256, 512) FFT, batch 4
        {384, 384, 512, 512, 4, false, false, true},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, false, false, true},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, false, false, true},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, false, false, true},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, false, false, true},  // (192, 384) -> (256, 512) FFT, batch 4

        // Cross-correlation configurations (transpose_axes=false, conv_data_is_transposed=true)
        {48, 48, 64, 64, 1, true, false, true},      // (48, 48) -> (64, 64) FFT, batch 1
        {48, 48, 64, 64, 8, true, false, true},      // (48, 48) -> (64, 64) FFT, batch 8
        {96, 96, 128, 128, 1, true, false, true},    // (96, 96) -> (128, 128) FFT, batch 1
        {96, 96, 128, 128, 8, true, false, true},    // (96, 96) -> (128, 128) FFT, batch 8
        {192, 192, 256, 256, 1, true, false, true},  // (192, 192) -> (256, 256) FFT, batch 1
        {192, 192, 256, 256, 4, true, false, true},  // (192, 192) -> (256, 256) FFT, batch 4
        {384, 384, 512, 512, 1, true, false, true},  // (384, 384) -> (512, 512) FFT, batch 1
        {384, 384, 512, 512, 4, true, false, true},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, true, false, true},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, true, false, true},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, true, false, true},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, true, false, true},  // (192, 384) -> (256, 512) FFT, batch 4
        {384, 384, 512, 512, 4, true, false, true},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, true, false, true},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, true, false, true},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, true, false, true},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, true, false, true},  // (192, 384) -> (256, 512) FFT, batch 4

        // Convolution configurations (transpose_axes=true, conv_data_is_transposed=true)
        {48, 48, 64, 64, 1, false, true, true},      // (48, 48) -> (64, 64) FFT, batch 1
        {48, 48, 64, 64, 8, false, true, true},      // (48, 48) -> (64, 64) FFT, batch 8
        {96, 96, 128, 128, 1, false, true, true},    // (96, 96) -> (128, 128) FFT, batch 1
        {96, 96, 128, 128, 8, false, true, true},    // (96, 96) -> (128, 128) FFT, batch 8
        {192, 192, 256, 256, 1, false, true, true},  // (192, 192) -> (256, 256) FFT, batch 1
        {192, 192, 256, 256, 4, false, true, true},  // (192, 192) -> (256, 256) FFT, batch 4
        {384, 384, 512, 512, 1, false, true, true},  // (384, 384) -> (512, 512) FFT, batch 1
        {384, 384, 512, 512, 4, false, true, true},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, false, true, true},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, false, true, true},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, false, true, true},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, false, true, true},  // (192, 384) -> (256, 512) FFT, batch 4
        {384, 384, 512, 512, 4, false, true, true},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, false, true, true},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, false, true, true},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, false, true, true},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, false, true, true},  // (192, 384) -> (256, 512) FFT, batch 4

        // Cross-correlation configurations (transpose_axes=true, conv_data_is_transposed=true)
        {48, 48, 64, 64, 1, true, true, true},      // (48, 48) -> (64, 64) FFT, batch 1
        {48, 48, 64, 64, 8, true, true, true},      // (48, 48) -> (64, 64) FFT, batch 8
        {96, 96, 128, 128, 1, true, true, true},    // (96, 96) -> (128, 128) FFT, batch 1
        {96, 96, 128, 128, 8, true, true, true},    // (96, 96) -> (128, 128) FFT, batch 8
        {192, 192, 256, 256, 1, true, true, true},  // (192, 192) -> (256, 256) FFT, batch 1
        {192, 192, 256, 256, 4, true, true, true},  // (192, 192) -> (256, 256) FFT, batch 4
        {384, 384, 512, 512, 1, true, true, true},  // (384, 384) -> (512, 512) FFT, batch 1
        {384, 384, 512, 512, 4, true, true, true},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, true, true, true},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, true, true, true},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, true, true, true},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, true, true, true},  // (192, 384) -> (256, 512) FFT, batch 4
        {384, 384, 512, 512, 4, true, true, true},  // (384, 384) -> (512, 512) FFT, batch 4
        {384, 192, 512, 256, 1, true, true, true},  // (384, 192) -> (512, 256) FFT, batch 1
        {384, 192, 512, 256, 4, true, true, true},  // (384, 192) -> (512, 256) FFT, batch 4
        {192, 384, 256, 512, 1, true, true, true},  // (192, 384) -> (256, 512) FFT, batch 1
        {192, 384, 256, 512, 4, true, true, true},  // (192, 384) -> (256, 512) FFT, batch 4

    }};

// Template dispatch functions for each supported configuration
template <unsigned int SignalLengthX, unsigned int SignalLengthY, unsigned int FFTSizeX,
          unsigned int FFTSizeY, unsigned int BatchSize, bool CrossCorrelate, bool TransposeAxes,
          bool ConvDataIsTransposed>
void dispatch_padded_real_conv(float* input_data, float2* fft_workspace, const float2* conv_data,
                               float* output_data) {
    // NOTE: <..., 0, 0, 0, 0, ...> corresponds to the EPT and FTB for x and y where zero is to use
    // cuFFTDx recommended defaults
    padded_block_real_conv_2d<float, float2, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY,
                              BatchSize, CrossCorrelate, 0, 0, 0, 0, TransposeAxes,
                              ConvDataIsTransposed>(input_data, fft_workspace, conv_data,
                                                    output_data);
}

// Helper template to create dispatch table entries at compile time
template <std::size_t... Is>
constexpr auto make_padded_conv_dispatch_table(std::index_sequence<Is...>) {
    return std::array<std::pair<PaddedRealConvConfig2D,
                                std::function<void(float*, float2*, const float2*, float*)>>,
                      sizeof...(Is)>{
        {{PaddedRealConvConfig2D{
              std::get<0>(SUPPORTED_CONV_CONFIGS[Is]), std::get<1>(SUPPORTED_CONV_CONFIGS[Is]),
              std::get<2>(SUPPORTED_CONV_CONFIGS[Is]), std::get<3>(SUPPORTED_CONV_CONFIGS[Is]),
              std::get<4>(SUPPORTED_CONV_CONFIGS[Is]), std::get<5>(SUPPORTED_CONV_CONFIGS[Is]),
              std::get<6>(SUPPORTED_CONV_CONFIGS[Is]), std::get<7>(SUPPORTED_CONV_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_CONV_CONFIGS[Is];
              return dispatch_padded_real_conv<std::get<1>(config), std::get<0>(config),
                                               std::get<3>(config), std::get<2>(config),
                                               std::get<4>(config), std::get<5>(config),
                                               std::get<6>(config), std::get<7>(config)>;
          }()}...}};
}

// Create the dispatch table automatically
static const auto padded_conv_dispatch_table =
    make_padded_conv_dispatch_table(std::make_index_sequence<SUPPORTED_CONV_CONFIGS.size()>{});

// Create lookup function with compile-time dispatch table
std::function<void(float*, float2*, const float2*, float*)> get_padded_conv_function(
    unsigned int signal_length_y, unsigned int signal_length_x, unsigned int fft_size_y,
    unsigned int fft_size_x, unsigned int batch_size, bool cross_correlate, bool transpose_axes,
    bool conv_data_is_transposed) {
    // Find matching configuration
    for (const auto& entry : padded_conv_dispatch_table) {
        if (entry.first == PaddedRealConvConfig2D{signal_length_y, signal_length_x, fft_size_y,
                                                  fft_size_x, batch_size, cross_correlate,
                                                  transpose_axes, conv_data_is_transposed}) {
            return entry.second;
        }
    }

    // If no match found return nullptr
    return nullptr;
}

// Function to expose supported configurations to Python
std::vector<std::tuple<int, int, int, int, int, bool, bool, bool>>
get_supported_padded_conv_configs() {
    std::vector<std::tuple<int, int, int, int, int, bool, bool, bool>> configs;
    configs.reserve(SUPPORTED_CONV_CONFIGS.size());

    for (const auto& config : SUPPORTED_CONV_CONFIGS) {
        configs.emplace_back(std::get<0>(config), std::get<1>(config), std::get<2>(config),
                             std::get<3>(config), std::get<4>(config), std::get<5>(config),
                             std::get<6>(config), std::get<7>(config));
    }

    return configs;
}

/**
 * @brief Common implementation function for padded real convolution/cross-correlation
 *
 * @param input - Input filter tensor of shape (h, w) or (batch, h, w)
 * @param fft_workspace - FFT workspace tensor for intermediate storage between X and Y dimensions.
 * Has shape (batch, H, W // 2 + 1) when transpose_axes is false, otherwise (batch, W, H // 2 + 1)
 * @param conv_data - Pre-computed convolution data (RFFT(image)) with shape (H, W // 2 + 1) when
 * conv_data_is_transposed is false, otherwise (W, H // 2 + 1).
 * @param output - Output tensor of shape (H_valid, W_valid) or (batch, H_valid, W_valid) where
 * H_valid = fft_size_y - signal_length_y + 1 and W_valid = fft_size_x - signal_length_x + 1
 * @param fft_size_y - FFT size in the Y dimension
 * @param fft_size_x - FFT size in the X dimension
 * @param cross_correlate - Whether to perform cross-correlation (true) or convolution (false)
 * @param transpose_axes - Whether to transpose the input and output axes
 * @param conv_data_is_transposed - Whether the convolution data is already transposed
 */
void padded_real_conv_2d_impl(torch::Tensor input, torch::Tensor fft_workspace,
                              torch::Tensor conv_data, torch::Tensor output, int fft_size_y,
                              int fft_size_x, bool cross_correlate, bool transpose_axes,
                              bool conv_data_is_transposed) {
    // --- Device and data type checks ---
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(fft_workspace.is_cuda(), "FFT workspace tensor must be on CUDA device");
    TORCH_CHECK(conv_data.is_cuda(), "Convolution data tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be of type torch.float32");
    TORCH_CHECK(fft_workspace.dtype() == torch::kComplexFloat,
                "FFT workspace tensor must be of type torch.complex64 (float2)");
    TORCH_CHECK(conv_data.dtype() == torch::kComplexFloat,
                "Convolution data tensor must be of type torch.complex64 (float2)");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output tensor must be of type torch.float32");

    // --- Decoding signal length and batch size from input tensor ---
    unsigned int signal_length_x, signal_length_y, batch_size;
    if (input.dim() == 2) {  // input shape (h, w)
        batch_size = 1;
        signal_length_y = input.size(0);  // h - number of rows
        signal_length_x = input.size(1);  // w - number of columns
    } else if (input.dim() == 3) {        // input shape (batch, h, w)
        batch_size = input.size(0);       // batch size
        signal_length_y = input.size(1);  // h - number of rows
        signal_length_x = input.size(2);  // w - number of columns
    } else {
        TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", input.dim(), "D.");
    }

    // --- Tensor shape validations (fft-workspace) ---
    TORCH_CHECK(input.dim() == fft_workspace.dim(),
                "Input and workspace tensors must have the same number of dimensions. Input dim: ",
                input.dim(), ", workspace dim: ", fft_workspace.dim());
    if (fft_workspace.dim() == 3) {
        TORCH_CHECK(batch_size == fft_workspace.size(0), "Workspace batch size does not match.");
        if (transpose_axes) {
            TORCH_CHECK(fft_workspace.size(1) == (fft_size_x / 2 + 1),
                        "Transposed workspace height (", fft_workspace.size(1),
                        ") != fft_size_x/2 + 1 (", (fft_size_x / 2 + 1), ").");
            TORCH_CHECK(fft_workspace.size(2) == fft_size_y, "Transposed workspace width (",
                        fft_workspace.size(2), ") != fft_size_y (", fft_size_y, ").");
        } else {
            TORCH_CHECK(fft_workspace.size(1) == fft_size_y, "Workspace height (",
                        fft_workspace.size(1), ") != fft_size_y (", fft_size_y, ").");
            TORCH_CHECK(fft_workspace.size(2) == (fft_size_x / 2 + 1), "Workspace width (",
                        fft_workspace.size(2), ") != fft_size_x/2 + 1 (", (fft_size_x / 2 + 1),
                        ").");
        }
    } else {
        if (transpose_axes) {
            TORCH_CHECK(fft_workspace.size(0) == (fft_size_x / 2 + 1),
                        "Transposed workspace height (", fft_workspace.size(0),
                        ") != fft_size_x/2 + 1 (", (fft_size_x / 2 + 1), ").");
            TORCH_CHECK(fft_workspace.size(1) == fft_size_y, "Transposed workspace width (",
                        fft_workspace.size(1), ") != fft_size_y (", fft_size_y, ").");
        } else {
            TORCH_CHECK(fft_workspace.size(0) == fft_size_y, "Workspace height (",
                        fft_workspace.size(0), ") != fft_size_y (", fft_size_y, ").");
            TORCH_CHECK(fft_workspace.size(1) == (fft_size_x / 2 + 1), "Workspace width (",
                        fft_workspace.size(1), ") != fft_size_x/2 + 1 (", (fft_size_x / 2 + 1),
                        ").");
        }
    }

    // --- Tensor shape validations (conv_data) ---
    TORCH_CHECK(conv_data.dim() == 2, "Convolution data tensor must be 2D. Got ", conv_data.dim(),
                "D.");
    if (conv_data_is_transposed) {
        TORCH_CHECK(conv_data.size(0) == (fft_size_x / 2 + 1), "Transposed conv data height (",
                    conv_data.size(0), ") != fft_size_x/2 + 1 (", (fft_size_x / 2 + 1), ").");
        TORCH_CHECK(conv_data.size(1) == fft_size_y, "Transposed conv data width (",
                    conv_data.size(1), ") != fft_size_y (", fft_size_y, ").");
    } else {
        TORCH_CHECK(conv_data.size(0) == fft_size_y, "Conv data height (", conv_data.size(0),
                    ") != fft_size_y (", fft_size_y, ").");
        TORCH_CHECK(conv_data.size(1) == (fft_size_x / 2 + 1), "Conv data width (",
                    conv_data.size(1), ") != fft_size_x/2 + 1 (", (fft_size_x / 2 + 1), ").");
    }

    // --- Tensor shape validations (output) ---
    unsigned int valid_length_x = fft_size_x - signal_length_x + 1;
    unsigned int valid_length_y = fft_size_y - signal_length_y + 1;
    if (output.dim() == 3) {  // output shape (batch, H_valid, W_valid)
        TORCH_CHECK(batch_size == output.size(0), "Output batch size does not match.");
        TORCH_CHECK(output.size(1) == valid_length_y, "Output height (", output.size(1),
                    ") != valid_length_y (", valid_length_y, ").");
        TORCH_CHECK(output.size(2) == valid_length_x, "Output width (", output.size(2),
                    ") != valid_length_x (", valid_length_x, ").");
    } else {
        TORCH_CHECK(output.size(0) == valid_length_y, "Output height (", output.size(0),
                    ") != valid_length_y (", valid_length_y, ").");
        TORCH_CHECK(output.size(1) == valid_length_x, "Output width (", output.size(1),
                    ") != valid_length_x (", valid_length_x, ").");
    }

    // Ensure signal length is smaller than or equal to FFT size
    TORCH_CHECK(signal_length_y <= fft_size_y,
                "Signal length in Y dimension cannot exceed FFT size");
    TORCH_CHECK(signal_length_x <= fft_size_x,
                "Signal length in X dimension cannot exceed FFT size");

    float* input_ptr = input.data_ptr<float>();
    float2* workspace_ptr =
        reinterpret_cast<float2*>(fft_workspace.data_ptr<c10::complex<float>>());
    const float2* conv_ptr =
        reinterpret_cast<const float2*>(conv_data.data_ptr<c10::complex<float>>());
    float* output_ptr = output.data_ptr<float>();

    // Use the dispatch table to get the appropriate function
    auto conv_func = get_padded_conv_function(signal_length_y, signal_length_x, fft_size_y,
                                              fft_size_x, batch_size, cross_correlate,
                                              transpose_axes, conv_data_is_transposed);
    TORCH_CHECK(conv_func != nullptr,
                "Unsupported padded convolution configuration: signal_y=", signal_length_y,
                ", signal_x=", signal_length_x, ", fft_y=", fft_size_y, ", fft_x=", fft_size_x,
                ", batch=", batch_size, ", cross_correlate=", cross_correlate,
                ", transpose_axes=", transpose_axes,
                ", conv_data_is_transposed=", conv_data_is_transposed);

    conv_func(input_ptr, workspace_ptr, conv_ptr, output_ptr);
}

// Function to expose to Python - Convolution
void padded_real_conv_2d(torch::Tensor input, torch::Tensor fft_workspace, torch::Tensor conv_data,
                         torch::Tensor output, int fft_size_y, int fft_size_x,
                         bool transpose_axes = false, bool conv_data_is_transposed = false) {
    padded_real_conv_2d_impl(input, fft_workspace, conv_data, output, fft_size_y, fft_size_x, false,
                             transpose_axes, conv_data_is_transposed);  // Convolution
}

// Function to expose to Python - Cross-correlation
void padded_real_corr_2d(torch::Tensor input, torch::Tensor fft_workspace, torch::Tensor conv_data,
                         torch::Tensor output, int fft_size_y, int fft_size_x,
                         bool transpose_axes = false, bool conv_data_is_transposed = false) {
    padded_real_conv_2d_impl(input, fft_workspace, conv_data, output, fft_size_y, fft_size_x, true,
                             transpose_axes, conv_data_is_transposed);  // Cross-correlation
}

PYBIND11_MODULE(padded_rconv2d, m) {
    m.doc() = "2D padded real convolution/cross-correlation using cuFFTDx";
    m.def("conv", &padded_real_conv_2d, "2D padded real convolution", pybind11::arg("input"),
          pybind11::arg("fft_workspace"), pybind11::arg("conv_data"), pybind11::arg("output"),
          pybind11::arg("fft_size_y"), pybind11::arg("fft_size_x"),
          pybind11::arg("transpose_axes") = false,
          pybind11::arg("conv_data_is_transposed") = false);
    m.def("corr", &padded_real_corr_2d, "2D padded real cross-correlation", pybind11::arg("input"),
          pybind11::arg("fft_workspace"), pybind11::arg("conv_data"), pybind11::arg("output"),
          pybind11::arg("fft_size_y"), pybind11::arg("fft_size_x"),
          pybind11::arg("transpose_axes") = false,
          pybind11::arg("conv_data_is_transposed") = false);
    m.def("get_supported_conv_configs", &get_supported_padded_conv_configs,
          "Get list of supported (signal_y, signal_x, fft_y, fft_x, batch_size, cross_correlate, "
          "transpose_axes, conv_data_is_transposed) configurations");
}