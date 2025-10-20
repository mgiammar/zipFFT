/* Python bindings for 2-dimensional real-to-complex and complex-to-real FFT operations
 * with implicit zero-padding using the cuFFTDx library.
 *
 * This implements FFTs where the input signal is smaller than the FFT size,
 * with automatic zero-padding.
 *
 * Author:  Matthew Giammar
 * E-mail:  mdgiammar@gmail.com
 * License: MIT License
 * Date:    20 October 2025
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

#include "padded_real_fft_2d.cuh"

// FFT configuration structure for padded 2D real FFT
struct PaddedRealFFTConfig2D {
    // Signal dimensions (actual data size)
    unsigned int signal_length_y;
    unsigned int signal_length_x;
    // FFT dimensions (padded size)
    unsigned int fft_size_y;
    unsigned int fft_size_x;
    // Batch size for parallel processing
    unsigned int batch_size;
    // Direction flag
    bool is_forward;

    bool operator==(const PaddedRealFFTConfig2D& other) const {
        return signal_length_x == other.signal_length_x &&
               signal_length_y == other.signal_length_y && fft_size_x == other.fft_size_x &&
               fft_size_y == other.fft_size_y && batch_size == other.batch_size &&
               is_forward == other.is_forward;
    }
};

// Define supported FFT configurations
// Format: (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch_size, is_forward)
static constexpr std::array<
    std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool>, 24>
    SUPPORTED_FFT_CONFIGS = {{
        // Forward FFT configurations - padded real to complex
        {48, 48, 64, 64, 1, true},      // (48, 48) -> (64, 64), batch=1
        {48, 48, 64, 64, 8, true},      // (48, 48) -> (64, 64), batch=8
        {96, 96, 128, 128, 1, true},    // (96, 96) -> (128, 128), batch=1
        {96, 96, 128, 128, 8, true},    // (96, 96) -> (128, 128), batch=8
        {192, 192, 256, 256, 1, true},  // (192, 192) -> (256, 256), batch=1
        {192, 192, 256, 256, 4, true},  // (192, 192) -> (256, 256), batch=4
        {384, 384, 512, 512, 1, true},  // (384, 384) -> (512, 512), batch=1
        {384, 384, 512, 512, 4, true},  // (384, 384) -> (512, 512), batch=4
        {384, 192, 512, 256, 1, true},  // (384, 192) -> (512, 256), batch=1
        {384, 192, 512, 256, 4, true},  // (384, 192) -> (512, 256), batch=4
        {192, 384, 256, 512, 1, true},  // (192, 384) -> (256, 512), batch=1
        {192, 384, 256, 512, 4, true},  // (192, 384) -> (256, 512), batch=4

        // Inverse FFT configurations - complex to padded real
        {48, 48, 64, 64, 1, false},      // (64, 33) -> (48, 48), batch=1
        {48, 48, 64, 64, 8, false},      // (64, 33) -> (48, 48), batch=8
        {96, 96, 128, 128, 1, false},    // (128, 65) -> (96, 96), batch=1
        {96, 96, 128, 128, 8, false},    // (128, 65) -> (96, 96), batch=8
        {192, 192, 256, 256, 1, false},  // (256, 129) -> (192, 192), batch=1
        {192, 192, 256, 256, 4, false},  // (256, 129) -> (192, 192), batch=4
        {384, 384, 512, 512, 1, false},  // (512, 257) -> (384, 384), batch=1
        {384, 384, 512, 512, 4, false},  // (512, 257) -> (384, 384), batch=4
        {384, 192, 512, 256, 1, false},  // (512, 129) -> (384, 192), batch=1
        {384, 192, 512, 256, 4, false},  // (512, 129) -> (384, 192), batch=4
        {192, 384, 256, 512, 1, false},  // (256, 257) -> (192, 384), batch=1
        {192, 384, 256, 512, 4, false},  // (256, 257) -> (192, 384), batch=4
    }};

// Template dispatch functions for each supported configuration
// For forward FFT (real-to-complex)
template <unsigned int SignalLengthX, unsigned int SignalLengthY, unsigned int FFTSizeX,
          unsigned int FFTSizeY, unsigned int BatchSize, bool IsForwardFFT>
void dispatch_padded_r2c_fft(float* input_data, float2* output_data, unsigned int batch_size) {
    // Using optimal elements per thread and ffts per block based on FFT sizes
    constexpr unsigned int elements_per_thread_x = FFTSizeX <= 128 ? 8 : 16;
    constexpr unsigned int elements_per_thread_y = FFTSizeY <= 128 ? 8 : 16;
    constexpr unsigned int ffts_per_block_x = 1;
    constexpr unsigned int ffts_per_block_y = 1;

    padded_block_real_fft_2d<float, float2, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY, true,
                             elements_per_thread_x, elements_per_thread_y, ffts_per_block_x,
                             ffts_per_block_y>(input_data, output_data, batch_size);
}

// For inverse FFT (complex-to-real)
template <unsigned int SignalLengthX, unsigned int SignalLengthY, unsigned int FFTSizeX,
          unsigned int FFTSizeY, unsigned int BatchSize, bool IsForwardFFT>
void dispatch_padded_c2r_fft(float2* input_data, float* output_data, unsigned int batch_size) {
    // Using optimal elements per thread and ffts per block based on FFT sizes
    constexpr unsigned int elements_per_thread_x = FFTSizeX <= 128 ? 8 : 16;
    constexpr unsigned int elements_per_thread_y = FFTSizeY <= 128 ? 8 : 16;
    constexpr unsigned int ffts_per_block_x = 1;
    constexpr unsigned int ffts_per_block_y = 1;

    padded_block_real_fft_2d<float2, float, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY, false,
                             elements_per_thread_x, elements_per_thread_y, ffts_per_block_x,
                             ffts_per_block_y>(input_data, output_data, batch_size);
}

// Helper template to create dispatch table entries at compile time for r2c
template <std::size_t... Is>
constexpr auto make_padded_r2c_dispatch_table(std::index_sequence<Is...>) {
    return std::array<
        std::pair<PaddedRealFFTConfig2D, std::function<void(float*, float2*, unsigned int)>>,
        sizeof...(Is)>{
        {{PaddedRealFFTConfig2D{
              std::get<0>(SUPPORTED_FFT_CONFIGS[Is]), std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<2>(SUPPORTED_FFT_CONFIGS[Is]), std::get<3>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<4>(SUPPORTED_FFT_CONFIGS[Is]), std::get<5>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              if constexpr (std::get<5>(config) == true) {  // Only include forward transforms
                  return dispatch_padded_r2c_fft<std::get<1>(config), std::get<0>(config),
                                                 std::get<3>(config), std::get<2>(config),
                                                 std::get<4>(config), true>;
              } else {
                  return static_cast<void (*)(float*, float2*, unsigned int)>(nullptr);
              }
          }()}...}};
}

// Helper template to create dispatch table entries at compile time for c2r
template <std::size_t... Is>
constexpr auto make_padded_c2r_dispatch_table(std::index_sequence<Is...>) {
    return std::array<
        std::pair<PaddedRealFFTConfig2D, std::function<void(float2*, float*, unsigned int)>>,
        sizeof...(Is)>{
        {{PaddedRealFFTConfig2D{
              std::get<0>(SUPPORTED_FFT_CONFIGS[Is]), std::get<1>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<2>(SUPPORTED_FFT_CONFIGS[Is]), std::get<3>(SUPPORTED_FFT_CONFIGS[Is]),
              std::get<4>(SUPPORTED_FFT_CONFIGS[Is]), std::get<5>(SUPPORTED_FFT_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_FFT_CONFIGS[Is];
              if constexpr (std::get<5>(config) == false) {  // Only include inverse transforms
                  return dispatch_padded_c2r_fft<std::get<1>(config), std::get<0>(config),
                                                 std::get<3>(config), std::get<2>(config),
                                                 std::get<4>(config), false>;
              } else {
                  return static_cast<void (*)(float2*, float*, unsigned int)>(nullptr);
              }
          }()}...}};
}

// Create the dispatch tables automatically
static const auto padded_r2c_dispatch_table =
    make_padded_r2c_dispatch_table(std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});
static const auto padded_c2r_dispatch_table =
    make_padded_c2r_dispatch_table(std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{});

// Create lookup function for r2c with compile-time dispatch table
std::function<void(float*, float2*, unsigned int)> get_padded_r2c_fft_function(
    unsigned int signal_length_y, unsigned int signal_length_x, unsigned int fft_size_y,
    unsigned int fft_size_x, unsigned int batch_size, bool is_forward) {
    // Find matching configuration
    for (const auto& entry : padded_r2c_dispatch_table) {
        if (entry.first == PaddedRealFFTConfig2D{signal_length_y, signal_length_x, fft_size_y,
                                                 fft_size_x, batch_size, is_forward} &&
            entry.second != nullptr) {
            return entry.second;
        }
    }

    // If no match found return nullptr
    return nullptr;
}

// Create lookup function for c2r with compile-time dispatch table
std::function<void(float2*, float*, unsigned int)> get_padded_c2r_fft_function(
    unsigned int signal_length_y, unsigned int signal_length_x, unsigned int fft_size_y,
    unsigned int fft_size_x, unsigned int batch_size, bool is_forward) {
    // Find matching configuration
    for (const auto& entry : padded_c2r_dispatch_table) {
        if (entry.first == PaddedRealFFTConfig2D{signal_length_y, signal_length_x, fft_size_y,
                                                 fft_size_x, batch_size, is_forward} &&
            entry.second != nullptr) {
            return entry.second;
        }
    }

    // If no match found return nullptr
    return nullptr;
}

// Function to expose supported configurations to Python
std::vector<std::tuple<int, int, int, int, int, bool>> get_supported_padded_fft_configs() {
    std::vector<std::tuple<int, int, int, int, int, bool>> configs;
    configs.reserve(SUPPORTED_FFT_CONFIGS.size());

    for (const auto& config : SUPPORTED_FFT_CONFIGS) {
        configs.emplace_back(std::get<0>(config), std::get<1>(config), std::get<2>(config),
                             std::get<3>(config), std::get<4>(config), std::get<5>(config));
    }

    return configs;
}

// Common implementation function for padded r2c FFT
void padded_real_fft_r2c_2d_impl(torch::Tensor input, torch::Tensor output, int fft_size_y, int fft_size_x) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be of type torch.float32");
    TORCH_CHECK(output.dtype() == torch::kComplexFloat,
                "Output tensor must be of type torch.complex64 (float2)");

    unsigned int signal_length_x, signal_length_y, batch_size;

    // Doing dimension checks for signal/fft sizes and batch dimensions
    if (input.dim() == 2) {  // input shape (H, W)
        batch_size = 1;
        signal_length_y = input.size(0);  // H - number of rows
        signal_length_x = input.size(1);  // W - number of columns
    } else if (input.dim() == 3) {        // input shape (batch, H, W)
        batch_size = input.size(0);       // batch size
        signal_length_y = input.size(1);  // H - number of rows
        signal_length_x = input.size(2);  // W - number of columns
    } else {
        TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", input.dim(), "D.");
    }

    // Validate output dimensions match expected FFT output size
    unsigned int expected_stride = fft_size_x / 2 + 1;
    if (output.dim() == 2) {          // output shape (H', W'//2+1)
        TORCH_CHECK(output.size(0) == fft_size_y, 
                    "Output tensor first dimension must match fft_size_y");
        TORCH_CHECK(output.size(1) == expected_stride,
                    "Output tensor second dimension must be fft_size_x/2+1");
    } else if (output.dim() == 3) {             // output shape (batch, H', W'//2+1)
        TORCH_CHECK(batch_size == output.size(0), "Input and output batch sizes must match");
        TORCH_CHECK(output.size(1) == fft_size_y, 
                    "Output tensor second dimension must match fft_size_y");
        TORCH_CHECK(output.size(2) == expected_stride,
                    "Output tensor third dimension must be fft_size_x/2+1");
    } else {
        TORCH_CHECK(false, "Output tensor must be 2D or 3D. Got ", output.dim(), "D.");
    }

    // Ensure signal length is smaller than or equal to FFT size
    TORCH_CHECK(signal_length_y <= fft_size_y,
                "Signal length in Y dimension cannot exceed FFT size");
    TORCH_CHECK(signal_length_x <= fft_size_x,
                "Signal length in X dimension cannot exceed FFT size");

    float* input_ptr = input.data_ptr<float>();
    float2* output_ptr = reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>());

    // Use the dispatch table to get the appropriate function
    auto fft_func = get_padded_r2c_fft_function(signal_length_y, signal_length_x, fft_size_y,
                                                fft_size_x, batch_size, true);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported padded FFT configuration: signal_y=", signal_length_y,
                ", signal_x=", signal_length_x, ", fft_y=", fft_size_y, ", fft_x=", fft_size_x,
                ", batch=", batch_size, ", is_forward=true");

    fft_func(input_ptr, output_ptr, batch_size);
}

// Common implementation function for padded c2r FFT
void padded_real_fft_c2r_2d_impl(torch::Tensor input, torch::Tensor output, int fft_size_y, int fft_size_x) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64 (float2)");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output tensor must be of type torch.float32");

    unsigned int signal_length_x, signal_length_y, batch_size;

    // Validate input dimensions match expected FFT input size
    unsigned int expected_stride = fft_size_x / 2 + 1;
    if (input.dim() == 2) {  // input shape (H', W'//2+1)
        batch_size = 1;
        TORCH_CHECK(input.size(0) == fft_size_y, 
                    "Input tensor first dimension must match fft_size_y");
        TORCH_CHECK(input.size(1) == expected_stride,
                    "Input tensor second dimension must be fft_size_x/2+1");
    } else if (input.dim() == 3) {             // input shape (batch, H', W'//2+1)
        batch_size = input.size(0);            // batch size
        TORCH_CHECK(input.size(1) == fft_size_y, 
                    "Input tensor second dimension must match fft_size_y");
        TORCH_CHECK(input.size(2) == expected_stride,
                    "Input tensor third dimension must be fft_size_x/2+1");
    } else {
        TORCH_CHECK(false, "Input tensor must be 2D or 3D. Got ", input.dim(), "D.");
    }

    // Output dimensions determine signal size
    if (output.dim() == 2) {               // output shape (H, W)
        signal_length_y = output.size(0);  // H - number of rows
        signal_length_x = output.size(1);  // W - number of columns
    } else if (output.dim() == 3) {        // output shape (batch, H, W)
        TORCH_CHECK(batch_size == output.size(0), "Input and output batch sizes must match");
        signal_length_y = output.size(1);  // H - number of rows
        signal_length_x = output.size(2);  // W - number of columns
    } else {
        TORCH_CHECK(false, "Output tensor must be 2D or 3D. Got ", output.dim(), "D.");
    }

    // Ensure signal length is smaller than or equal to FFT size
    TORCH_CHECK(signal_length_y <= fft_size_y,
                "Signal length in Y dimension cannot exceed FFT size");
    TORCH_CHECK(signal_length_x <= fft_size_x,
                "Signal length in X dimension cannot exceed FFT size");

    float2* input_ptr = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    float* output_ptr = output.data_ptr<float>();

    // Use the dispatch table to get the appropriate function
    auto fft_func = get_padded_c2r_fft_function(signal_length_y, signal_length_x, fft_size_y,
                                                fft_size_x, batch_size, false);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported padded FFT configuration: signal_y=", signal_length_y,
                ", signal_x=", signal_length_x, ", fft_y=", fft_size_y, ", fft_x=", fft_size_x,
                ", batch=", batch_size, ", is_forward=false");

    fft_func(input_ptr, output_ptr, batch_size);
}

// Function to expose to Python - Forward FFT (r2c)
void padded_real_fft_r2c_2d(torch::Tensor input, torch::Tensor output, int fft_size_y, int fft_size_x) {
    padded_real_fft_r2c_2d_impl(input, output, fft_size_y, fft_size_x);  // Forward FFT
}

// Function to expose to Python - Inverse FFT (c2r)
void padded_real_fft_c2r_2d(torch::Tensor input, torch::Tensor output, int fft_size_y, int fft_size_x) {
    padded_real_fft_c2r_2d_impl(input, output, fft_size_y, fft_size_x);  // Inverse FFT
}

PYBIND11_MODULE(padded_rfft2d, m) {  // Name should match in setup.py
    m.doc() = "2D padded real-to-complex and complex-to-real FFT using cuFFTDx";
    m.def("fft", &padded_real_fft_r2c_2d, "2D padded real-to-complex FFT (R2C)");
    m.def("ifft", &padded_real_fft_c2r_2d, "2D padded complex-to-real inverse FFT (C2R)");
    m.def("get_supported_fft_configs", &get_supported_padded_fft_configs,
          "Get list of supported (signal_y, signal_x, fft_y, fft_x, batch_size, is_forward) "
          "configurations");
}