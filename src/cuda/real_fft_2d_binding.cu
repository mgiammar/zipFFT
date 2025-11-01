/* Python bindings for 2-dimensional real-to-complex and complex-to-real FFT operations
 * using the cuFFTDx library.
 *
 * Author:  Matthew Giammar
 * E-mail:  mdgiammar@gmail.com
 * License: MIT License
 * Date:    18 October 2025
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

#include "real_fft_2d.cuh"

// FFT configuration structure
struct RealFFTConfig2D {
    // H dimension (rows)
    unsigned int fft_size_y;
    // W dimension (columns)
    unsigned int fft_size_x;
    unsigned int batch_size;
    bool is_forward;

    bool operator==(const RealFFTConfig2D& other) const {
        return fft_size_x == other.fft_size_x && fft_size_y == other.fft_size_y &&
               batch_size == other.batch_size && is_forward == other.is_forward;
    }
};

// Define supported FFT configurations
// Format: (fft_size_y, fft_size_x, batch_size, is_forward)
// Where: fft_size_y = H (rows), fft_size_x = W (columns)
// Note: stride_y is calculated automatically as fft_size_x // 2 + 1
static constexpr std::array<std::tuple<unsigned int, unsigned int, unsigned int, bool>, 36>
    SUPPORTED_FFT_CONFIGS = {{
        // Forward FFT configurations - input shape (H, W) -> output shape (H, W//2+1)
        {64, 64, 1, true},   // 2D input (64, 64), 1 batch
        {64, 64, 5, true},   // 2D input (64, 64), 5 batches
        {64, 64, 10, true},  // 2D input (64, 64), 10 batches

        {128, 64, 1, true},   // 2D input (128, 64), 1 batch
        {128, 64, 5, true},   // 2D input (128, 64), 5 batches
        {128, 64, 10, true},  // 2D input (128, 64), 10 batches

        {64, 128, 1, true},   // 2D input (64, 128), 1 batch
        {64, 128, 5, true},   // 2D input (64, 128), 5 batches
        {64, 128, 10, true},  // 2D input (64, 128), 10 batches

        {128, 128, 1, true},   // 2D input (128, 128), 1 batch
        {128, 128, 5, true},   // 2D input (128, 128), 5 batches
        {128, 128, 10, true},  // 2D input (128, 128), 10 batches

        {256, 256, 1, true},   // 2D input (256, 256), 1 batch
        {256, 256, 5, true},   // 2D input (256, 256), 5 batches
        {256, 256, 10, true},  // 2D input (256, 256), 10 batches

        {512, 512, 1, true},   // 2D input (512, 512), 1 batch
        {512, 512, 5, true},   // 2D input (512, 512), 5 batches
        {512, 512, 10, true},  // 2D input (512, 512), 10 batches

        // Inverse FFT configurations - input shape (H, W//2+1) -> output shape (H, W)
        {64, 64, 1, false},   // 2D output (64, 64), 1 batch
        {64, 64, 5, false},   // 2D output (64, 64), 5 batches
        {64, 64, 10, false},  // 2D output (64, 64), 10 batches

        {128, 64, 1, false},   // 2D output (128, 64), 1 batch
        {128, 64, 5, false},   // 2D output (128, 64), 5 batches
        {128, 64, 10, false},  // 2D output (128, 64), 10 batches

        {64, 128, 1, false},   // 2D output (64, 128), 1 batch
        {64, 128, 5, false},   // 2D output (64, 128), 5 batches
        {64, 128, 10, false},  // 2D output (64, 128), 10 batches

        {128, 128, 1, false},   // 2D output (128, 128), 1 batch
        {128, 128, 5, false},   // 2D output (128, 128), 5 batches
        {128, 128, 10, false},  // 2D output (128, 128), 10 batches

        {256, 256, 1, false},   // 2D output (256, 256), 1 batch
        {256, 256, 5, false},   // 2D output (256, 256), 5 batches
        {256, 256, 10, false},  // 2D output (256, 256), 10 batches

        {512, 512, 1, false},   // 2D output (512, 512), 1 batch
        {512, 512, 5, false},   // 2D output (512, 512), 5 batches
        {512, 512, 10, false},  // 2D output (512, 512), 10 batches
    }};

// Template dispatch functions for each supported configuration
// For forward FFT (real-to-complex)
template <unsigned int FFTSizeY, unsigned int FFTSizeX, unsigned int BatchSize, bool IsForwardFFT>
void dispatch_r2c_fft(float* input_data, float2* output_data) {
    // Using optimal elements per thread and ffts per block based on FFT sizes
    constexpr unsigned int elements_per_thread_x = FFTSizeX <= 128 ? 8 : 16;
    constexpr unsigned int elements_per_thread_y = FFTSizeY <= 128 ? 8 : 16;
    constexpr unsigned int ffts_per_block_x = 1;
    constexpr unsigned int ffts_per_block_y = 1;

    block_real_fft_2d<float, float2, FFTSizeX, FFTSizeY, BatchSize, true, elements_per_thread_x,
                      elements_per_thread_y, ffts_per_block_x, ffts_per_block_y>(input_data,
                                                                                 output_data);
}

// For inverse FFT (complex-to-real)
template <unsigned int FFTSizeY, unsigned int FFTSizeX, unsigned int BatchSize, bool IsForwardFFT>
void dispatch_c2r_fft(float2* input_data, float* output_data) {
    // Using optimal elements per thread and ffts per block based on FFT sizes
    constexpr unsigned int elements_per_thread_x = FFTSizeX <= 128 ? 8 : 16;
    constexpr unsigned int elements_per_thread_y = FFTSizeY <= 128 ? 8 : 16;
    constexpr unsigned int ffts_per_block_x = 1;
    constexpr unsigned int ffts_per_block_y = 1;

    block_real_fft_2d<float2, float, FFTSizeX, FFTSizeY, BatchSize, false, elements_per_thread_x,
                      elements_per_thread_y, ffts_per_block_x, ffts_per_block_y>(input_data,
                                                                                 output_data);
}

// Helper template to create dispatch table entries at compile time for r2c
template <std::size_t... Is>
constexpr auto make_r2c_dispatch_table(std::index_sequence<Is...>) {
    return std::array<std::pair<RealFFTConfig2D, std::function<void(float*, float2*)>>,
                      sizeof...(Is)>{
        {{RealFFTConfig2D{
              std::get<0>(SUPPORTED_FFT_CONFIGS[Is]), std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<2>(SUPPORTED_FFT_CONFIGS[Is]), std::get<3>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              if constexpr (std::get<3>(config) == true) {  // Only include forward transforms
                  return dispatch_r2c_fft<std::get<0>(config), std::get<1>(config),
                                          std::get<2>(config), true>;
              } else {
                  return static_cast<void (*)(float*, float2*)>(nullptr);
              }
          }()}...}};
}

// Helper template to create dispatch table entries at compile time for c2r
template <std::size_t... Is>
constexpr auto make_c2r_dispatch_table(std::index_sequence<Is...>) {
    return std::array<std::pair<RealFFTConfig2D, std::function<void(float2*, float*)>>,
                      sizeof...(Is)>{
        {{RealFFTConfig2D{
              std::get<0>(SUPPORTED_FFT_CONFIGS[Is]), std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<2>(SUPPORTED_FFT_CONFIGS[Is]), std::get<3>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              if constexpr (std::get<3>(config) == false) {  // Only include inverse transforms
                  return dispatch_c2r_fft<std::get<0>(config), std::get<1>(config),
                                          std::get<2>(config), false>;
              } else {
                  return static_cast<void (*)(float2*, float*)>(nullptr);
              }
          }()}...}};
}

// Create the dispatch tables automatically
static const auto r2c_dispatch_table =
    make_r2c_dispatch_table(std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});
static const auto c2r_dispatch_table =
    make_c2r_dispatch_table(std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

// Create lookup function for r2c with compile-time dispatch table
std::function<void(float*, float2*)> get_r2c_fft_function(unsigned int fft_size_y,
                                                          unsigned int fft_size_x,
                                                          unsigned int batch_size,
                                                          bool is_forward) {
    // Find matching configuration
    for (const auto& entry : r2c_dispatch_table) {
        if (entry.first == RealFFTConfig2D{fft_size_y, fft_size_x, batch_size, is_forward} &&
            entry.second != nullptr) {
            return entry.second;
        }
    }

    // If no match found return nullptr
    return nullptr;
}

// Create lookup function for c2r with compile-time dispatch table
std::function<void(float2*, float*)> get_c2r_fft_function(unsigned int fft_size_y,
                                                          unsigned int fft_size_x,
                                                          unsigned int batch_size,
                                                          bool is_forward) {
    // Find matching configuration
    for (const auto& entry : c2r_dispatch_table) {
        if (entry.first == RealFFTConfig2D{fft_size_y, fft_size_x, batch_size, is_forward} &&
            entry.second != nullptr) {
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

// Common implementation function for r2c FFT
void real_fft_r2c_2d_impl(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be of type torch.float32");
    TORCH_CHECK(output.dtype() == torch::kComplexFloat,
                "Output tensor must be of type torch.complex64 (float2)");

    unsigned int fft_size_x, fft_size_y, batch_size;

    // Doing dimension checks for fft size and batch dimensions
    if (input.dim() == 2) {          // input shape (H, W)
        fft_size_y = input.size(0);  // H - number of rows
        fft_size_x = input.size(1);  // W - number of columns
        batch_size = 1;
    } else if (input.dim() == 3) {   // input shape (batch, H, W)
        batch_size = input.size(0);  // batch size
        fft_size_y = input.size(1);  // H - number of rows
        fft_size_x = input.size(2);  // W - number of columns
    } else {
        TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", input.dim(), "D.");
    }

    float* input_ptr = input.data_ptr<float>();
    float2* output_ptr = reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>());

    // Use the dispatch table to get the appropriate function
    auto fft_func = get_r2c_fft_function(fft_size_y, fft_size_x, batch_size, true);
    TORCH_CHECK(fft_func != nullptr, "Unsupported FFT configuration: size_y=", fft_size_y,
                ", size_x=", fft_size_x, ", batch=", batch_size, ", is_forward=true");

    // Execute the FFT function (batch_size is now baked into the template)
    fft_func(input_ptr, output_ptr);
}

// Common implementation function for c2r FFT
void real_fft_c2r_2d_impl(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64 (float2)");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output tensor must be of type torch.float32");

    unsigned int fft_size_x, fft_size_y, batch_size;

    // Doing dimension checks for fft size and batch dimensions
    if (input.dim() == 2) {                     // input shape (H, W//2+1)
        fft_size_y = input.size(0);             // H - number of rows
        unsigned int stride_y = input.size(1);  // W//2+1 - number of complex columns
        fft_size_x = (stride_y - 1) * 2;        // W - original size of real data
        batch_size = 1;
    } else if (input.dim() == 3) {              // input shape (batch, H, W//2+1)
        batch_size = input.size(0);             // batch size
        fft_size_y = input.size(1);             // H - number of rows
        unsigned int stride_y = input.size(2);  // W//2+1 - number of complex columns
        fft_size_x = (stride_y - 1) * 2;        // W - original size of real data
    } else {
        TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", input.dim(), "D.");
    }

    float2* input_ptr = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    float* output_ptr = output.data_ptr<float>();

    // Use the dispatch table to get the appropriate function
    auto fft_func = get_c2r_fft_function(fft_size_y, fft_size_x, batch_size, false);
    TORCH_CHECK(fft_func != nullptr, "Unsupported FFT configuration: size_y=", fft_size_y,
                ", size_x=", fft_size_x, ", batch=", batch_size, ", is_forward=false");

    // Execute the FFT function (batch_size is now baked into the template)
    fft_func(input_ptr, output_ptr);
}

void real_fft_r2c_2d(torch::Tensor input, torch::Tensor output) {
    real_fft_r2c_2d_impl(input, output);  // Forward FFT
}

void real_fft_c2r_2d(torch::Tensor input, torch::Tensor output) {
    real_fft_c2r_2d_impl(input, output);  // Inverse FFT
}

PYBIND11_MODULE(rfft2d, m) {  // First arg needs to match name in setup.py
    m.doc() = "2D real-to-complex and complex-to-real FFT using cuFFTDx";
    m.def("fft", &real_fft_r2c_2d, "2D real-to-complex FFT (R2C)");
    m.def("ifft", &real_fft_c2r_2d, "2D complex-to-real inverse FFT (C2R)");
    m.def("get_supported_fft_configs", &get_supported_fft_configs,
          "Get list of supported (fft_size_y, fft_size_x, batch_size, is_forward) "
          "configurations");
}