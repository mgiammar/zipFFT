/* Python bindings for 2-dimensional complex-to-complex convolution/cross-
 * correlation operations with implicit zero-padding using the cuFFTDx library.
 *
 * This implements convolution/cross-correlation where the input signal is
 * smaller than the FFT size, with automatic zero-padding. Unlike the real-
 * valued path, both the input signal and the precomputed FFT of the reference
 * image are complex-valued, so the X and Y dimension transforms are both
 * complex-to-complex (C2C) FFTs.
 *
 * Author:  Matthew Giammar
 * E-mail:  mdgiammar@gmail.com
 * License: MIT License
 * Date:    25 July 2026
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <torch/extension.h>

#include <array>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

#include "complex_conv_2d.cuh"

// Convolution configuration structure for padded 2D complex-to-complex convolution
struct PaddedComplexConvConfig2D {
    unsigned int signal_length_y;
    unsigned int signal_length_x;
    unsigned int fft_size_y;
    unsigned int fft_size_x;
    unsigned int batch_size;
    bool cross_correlate;

    bool operator==(const PaddedComplexConvConfig2D& other) const {
        return signal_length_x == other.signal_length_x &&
               signal_length_y == other.signal_length_y && fft_size_x == other.fft_size_x &&
               fft_size_y == other.fft_size_y && batch_size == other.batch_size &&
               cross_correlate == other.cross_correlate;
    }
};

// Define supported convolution configurations
// (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch_size, cross_correlate)
static constexpr std::array<
    std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool>, 34>
    SUPPORTED_C2C_CONV_CONFIGS = {{
        // Convolution configurations (TEST CONFIGURATIONS)
        {48, 48, 64, 64, 1, false},      // (48, 48) -> (64, 64), batch=1
        {48, 48, 64, 64, 8, false},      // (48, 48) -> (64, 64), batch=8
        {96, 96, 128, 128, 1, false},    // (96, 96) -> (128, 128), batch=1
        {96, 96, 128, 128, 4, false},    // (96, 96) -> (128, 128), batch=4
        {192, 192, 256, 256, 1, false},  // (192, 192) -> (256, 256), batch=1
        {192, 192, 256, 256, 4, false},  // (192, 192) -> (256, 256), batch=4
        {384, 192, 512, 256, 1, false},  // (384, 192) -> (512, 256), batch=1
        {192, 384, 256, 512, 1, false},  // (192, 384) -> (256, 512), batch=1

        // Cross-correlation configurations (TEST CONFIGURATIONS)
        {48, 48, 64, 64, 1, true},      // (48, 48) -> (64, 64), batch=1
        {48, 48, 64, 64, 8, true},      // (48, 48) -> (64, 64), batch=8
        {96, 96, 128, 128, 1, true},    // (96, 96) -> (128, 128), batch=1
        {96, 96, 128, 128, 4, true},    // (96, 96) -> (128, 128), batch=4
        {192, 192, 256, 256, 1, true},  // (192, 192) -> (256, 256), batch=1
        {192, 192, 256, 256, 4, true},  // (192, 192) -> (256, 256), batch=4
        {384, 192, 512, 256, 1, true},  // (384, 192) -> (512, 256), batch=1
        {192, 384, 256, 512, 1, true},  // (192, 384) -> (256, 512), batch=1
        {16, 16, 64, 64, 1, true},      // (16, 16) -> (64, 64), batch=1
        {32, 32, 128, 128, 1, true},    // (32, 32) -> (128, 128), batch=1

        // Cross-correlation configurations (DEMO: signal_length x image_size sweep)
        {128, 128, 1024, 1024, 1, true},  // (128, 128) -> (1024, 1024), batch=1
        {128, 128, 2048, 2048, 1, true},  // (128, 128) -> (2048, 2048), batch=1
        {128, 128, 4096, 4096, 1, true},  // (128, 128) -> (4096, 4096), batch=1
        {128, 128, 8192, 8192, 1, true},  // (128, 128) -> (8192, 8192), batch=1
        {256, 256, 1024, 1024, 1, true},  // (256, 256) -> (1024, 1024), batch=1
        {256, 256, 2048, 2048, 1, true},  // (256, 256) -> (2048, 2048), batch=1
        {256, 256, 4096, 4096, 1, true},  // (256, 256) -> (4096, 4096), batch=1
        {256, 256, 8192, 8192, 1, true},  // (256, 256) -> (8192, 8192), batch=1
        {384, 384, 1024, 1024, 1, true},  // (384, 384) -> (1024, 1024), batch=1
        {384, 384, 2048, 2048, 1, true},  // (384, 384) -> (2048, 2048), batch=1
        {384, 384, 4096, 4096, 1, true},  // (384, 384) -> (4096, 4096), batch=1
        {384, 384, 8192, 8192, 1, true},  // (384, 384) -> (8192, 8192), batch=1
        {512, 512, 1024, 1024, 1, true},  // (512, 512) -> (1024, 1024), batch=1
        {512, 512, 2048, 2048, 1, true},  // (512, 512) -> (2048, 2048), batch=1
        {512, 512, 4096, 4096, 1, true},  // (512, 512) -> (4096, 4096), batch=1
        {512, 512, 8192, 8192, 1, true},  // (512, 512) -> (8192, 8192), batch=1
    }};

// Template dispatch functions for each supported configuration
template <unsigned int SignalLengthX, unsigned int SignalLengthY, unsigned int FFTSizeX,
          unsigned int FFTSizeY, unsigned int BatchSize, bool CrossCorrelate>
void dispatch_padded_complex_conv(float2* input_data, float2* fft_workspace,
                                  const float2* conv_data, float2* output_data, int device_index,
                                  cudaStream_t stream) {
    // NOTE: Removing the elements_per_thread and ffts_per_block template parameters to use defaults
    padded_block_complex_conv_2d<float2, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY,
                                 BatchSize, CrossCorrelate>(input_data, fft_workspace, conv_data,
                                                            output_data, device_index, stream);
}

// Helper template to create dispatch table entries at compile time
template <std::size_t... Is>
constexpr auto make_padded_complex_conv_dispatch_table(std::index_sequence<Is...>) {
    return std::array<
        std::pair<PaddedComplexConvConfig2D,
                  std::function<void(float2*, float2*, const float2*, float2*, int, cudaStream_t)>>,
        sizeof...(Is)>{
        {{PaddedComplexConvConfig2D{std::get<0>(SUPPORTED_C2C_CONV_CONFIGS[Is]),
                                    std::get<1>(SUPPORTED_C2C_CONV_CONFIGS[Is]),
                                    std::get<2>(SUPPORTED_C2C_CONV_CONFIGS[Is]),
                                    std::get<3>(SUPPORTED_C2C_CONV_CONFIGS[Is]),
                                    std::get<4>(SUPPORTED_C2C_CONV_CONFIGS[Is]),
                                    std::get<5>(SUPPORTED_C2C_CONV_CONFIGS[Is])},
          []() {
              constexpr auto config = SUPPORTED_C2C_CONV_CONFIGS[Is];
              return dispatch_padded_complex_conv<std::get<1>(config), std::get<0>(config),
                                                  std::get<3>(config), std::get<2>(config),
                                                  std::get<4>(config), std::get<5>(config)>;
          }()}...}};
}

// Create the dispatch table automatically
static const auto padded_complex_conv_dispatch_table = make_padded_complex_conv_dispatch_table(
    std::make_index_sequence<SUPPORTED_C2C_CONV_CONFIGS.size()>{});

// Create lookup function with compile-time dispatch table
std::function<void(float2*, float2*, const float2*, float2*, int, cudaStream_t)>
get_padded_complex_conv_function(unsigned int signal_length_y, unsigned int signal_length_x,
                                 unsigned int fft_size_y, unsigned int fft_size_x,
                                 unsigned int batch_size, bool cross_correlate) {
    // Find matching configuration
    for (const auto& entry : padded_complex_conv_dispatch_table) {
        if (entry.first == PaddedComplexConvConfig2D{signal_length_y, signal_length_x, fft_size_y,
                                                     fft_size_x, batch_size, cross_correlate}) {
            return entry.second;
        }
    }

    // If no match found return nullptr
    return nullptr;
}

// Function to expose supported configurations to Python
std::vector<std::tuple<int, int, int, int, int, bool>> get_supported_padded_complex_conv_configs() {
    std::vector<std::tuple<int, int, int, int, int, bool>> configs;
    configs.reserve(SUPPORTED_C2C_CONV_CONFIGS.size());

    for (const auto& config : SUPPORTED_C2C_CONV_CONFIGS) {
        configs.emplace_back(std::get<0>(config), std::get<1>(config), std::get<2>(config),
                             std::get<3>(config), std::get<4>(config), std::get<5>(config));
    }

    return configs;
}

/**
 * @brief Common implementation function for the padded complex-to-complex 2D
 * convolution/cross-correlation. Function performs input shape/size/type validation on the input
 * tensors before dispatching to the appropriate templated implementation function based on the
 * input parameters.
 *
 * @param input - The complex input projection tensor with shape (h, w) if non-batched or
 * (batch, h, w) if batched.
 * @param fft_workspace - Workspace tensor for intermediate FFT calculations between the two
 * dimensions. Must have shape (H, W) if non-batched or (batch, H, W) if batched.
 * @param conv_data - The precomputed complex FFT of the input image for convolution/
 * cross-correlation. Must have shape (W, H) (pre-transposed for coalesced access).
 * @param output - Complex output cross-correlogram / convolution tensor with valid
 * cross-correlation shape of (H - h + 1, W - w + 1) if non-batched or
 * (batch, H - h + 1, W - w + 1) if batched.
 * @param fft_size_y - The FFT size in the Y dimension a.k.a. 'H' the number of rows.
 * @param fft_size_x - The FFT size in the X dimension a.k.a. 'W' the number of columns.
 * @param cross_correlate - Whether to perform cross-correlation (true) or convolution (false).
 */
void padded_complex_conv_2d_impl(torch::Tensor input, torch::Tensor fft_workspace,
                                 torch::Tensor conv_data, torch::Tensor output, int fft_size_y,
                                 int fft_size_x, bool cross_correlate) {
    auto reference_device = input.device();
    int device_index = reference_device.index();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();

    // --- Type and device checks ---
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(fft_workspace.device().is_cuda(), "FFT workspace tensor must be on CUDA device");
    TORCH_CHECK(conv_data.device().is_cuda(), "Convolution data tensor must be on CUDA device");
    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");

    TORCH_CHECK(fft_workspace.device() == reference_device,
                "FFT workspace tensor must be on the same device as the input tensor");
    TORCH_CHECK(conv_data.device() == reference_device,
                "Convolution data tensor must be on the same device as the input tensor");
    TORCH_CHECK(output.device() == reference_device,
                "Output tensor must be on the same device as the input tensor");

    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64 (float2)");
    TORCH_CHECK(fft_workspace.dtype() == torch::kComplexFloat,
                "FFT workspace tensor must be of type torch.complex64 (float2)");
    TORCH_CHECK(conv_data.dtype() == torch::kComplexFloat,
                "Convolution data tensor must be of type torch.complex64 (float2)");
    TORCH_CHECK(output.dtype() == torch::kComplexFloat,
                "Output tensor must be of type torch.complex64 (float2)");

    // --- Helpful values for dimension checks ---
    unsigned int signal_length_x, signal_length_y, batch_size;
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

    unsigned int valid_length_x = fft_size_x - signal_length_x + 1;
    unsigned int valid_length_y = fft_size_y - signal_length_y + 1;

    // NOTE: Unlike the real-valued path, the complex-to-complex FFT along X does not use
    // Hermitian symmetry, so the workspace stride is the full FFT size (not fft_size_x/2 + 1).
    unsigned int expected_stride = fft_size_x;

    // --- Dimension checks for the workspace tensor ---
    TORCH_CHECK(fft_workspace.dim() == input.dim(),
                "FFT workspace tensor must have the same number of dimensions as the input tensor");
    TORCH_CHECK(fft_workspace.size(-1) == expected_stride,
                "FFT workspace tensor last dimension must be fft_size_x");
    TORCH_CHECK(fft_workspace.size(-2) == fft_size_y,
                "FFT workspace tensor second-to-last dimension must be fft_size_y");
    if (batch_size > 1) {
        TORCH_CHECK(fft_workspace.size(0) == batch_size,
                    "FFT workspace tensor first dimension (batch) must match input batch size");
    }

    // --- Dimension checks for the convolution data tensor ---
    /// NOTE: The convolution data tensor is assumed to be pre-transposed into
    /// column-major order for coalesced memory access on the point-wise multiply.
    TORCH_CHECK(conv_data.dim() == 2, "Convolution data tensor must be 2D. Got ", conv_data.dim(),
                "D.");
    TORCH_CHECK(conv_data.size(0) == expected_stride,
                "Convolution data tensor first dimension must match fft_size_x");
    TORCH_CHECK(conv_data.size(1) == fft_size_y,  // contig dimension
                "Convolution data tensor second dimension must be fft_size_y");

    // --- Dimension checks for the output tensor ---
    TORCH_CHECK(output.dim() == input.dim(),
                "Output tensor must have the same number of dimensions as the input tensor");
    TORCH_CHECK(output.size(-1) == valid_length_x,
                "Output tensor last dimension must be fft_size_x - signal_length_x + 1");
    TORCH_CHECK(output.size(-2) == valid_length_y,
                "Output tensor second-to-last dimension must be fft_size_y - signal_length_y + 1");

    // --- CUDA guard ---
    c10::cuda::CUDAGuard guard(reference_device);

    // --- Raw pointers for each of the tensor data ---
    float2* input_ptr = reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());
    float2* workspace_ptr =
        reinterpret_cast<float2*>(fft_workspace.data_ptr<c10::complex<float>>());
    const float2* conv_ptr =
        reinterpret_cast<const float2*>(conv_data.data_ptr<c10::complex<float>>());
    float2* output_ptr = reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>());

    // --- Get function from dispatch table and execute ---
    auto conv_func = get_padded_complex_conv_function(signal_length_y, signal_length_x, fft_size_y,
                                                      fft_size_x, batch_size, cross_correlate);
    TORCH_CHECK(conv_func != nullptr,
                "Unsupported padded complex convolution configuration: signal_y=", signal_length_y,
                ", signal_x=", signal_length_x, ", fft_y=", fft_size_y, ", fft_x=", fft_size_x,
                ", batch=", batch_size, ", cross_correlate=", cross_correlate);

    conv_func(input_ptr, workspace_ptr, conv_ptr, output_ptr, device_index, stream);
}

// Function to expose to Python - Convolution
void padded_complex_conv_2d(torch::Tensor input, torch::Tensor fft_workspace,
                            torch::Tensor conv_data, torch::Tensor output, int fft_size_y,
                            int fft_size_x) {
    padded_complex_conv_2d_impl(input, fft_workspace, conv_data, output, fft_size_y, fft_size_x,
                                false);  // Convolution
}

// Function to expose to Python - Cross-correlation
void padded_complex_corr_2d(torch::Tensor input, torch::Tensor fft_workspace,
                            torch::Tensor conv_data, torch::Tensor output, int fft_size_y,
                            int fft_size_x) {
    padded_complex_conv_2d_impl(input, fft_workspace, conv_data, output, fft_size_y, fft_size_x,
                                true);  // Cross-correlation
}

PYBIND11_MODULE(padded_cconv2d, m) {  // Name should match in setup.py
    m.doc() = "2D padded complex-to-complex convolution/cross-correlation using cuFFTDx";
    m.def("conv", &padded_complex_conv_2d, "2D padded complex-to-complex convolution");
    m.def("corr", &padded_complex_corr_2d, "2D padded complex-to-complex cross-correlation");
    m.def("get_supported_conv_configs", &get_supported_padded_complex_conv_configs,
          "Get list of supported (signal_y, signal_x, fft_y, fft_x, batch_size, cross_correlate) "
          "configurations");
}
