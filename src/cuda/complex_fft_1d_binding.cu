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

#include <array>
#include <functional>
#include <iostream>

#include "complex_fft_1d.cuh"

// FFT configuration structure
struct ComplexFFTConfig1D {
    unsigned int fft_size;    // Signal size for FFT
    unsigned int batch_size;  // Number of FFTs (maps to FFTs per block)
    bool is_forward;          // True for forward FFT, false for inverse

    bool operator==(const ComplexFFTConfig1D& other) const {
        return fft_size == other.fft_size && batch_size == other.batch_size &&
               is_forward == other.is_forward;
    }
};

// Template dispatch functions for each supported configuration
template <unsigned int FFTSize, unsigned int BatchSize, bool IsForwardFFT>
void dispatch_fft(float2* data) {
    block_complex_fft_1d<float2, FFTSize, IsForwardFFT, 8u, BatchSize>(data);
}

// Create lookup function with compile-time dispatch table
std::function<void(float2*)> get_fft_function(unsigned int fft_size,
                                              unsigned int batch_size,
                                              bool is_forward) {
    // Define supported configurations as a compile-time table
    // NOTE: To add another configuration, all you need to do is add a new line
    // in the array below and update the array's size.
    // Elements in pair correspond to:
    // 1. The size of the FFT (fft_size)
    // 2. The batch size (batch_size)
    // 3. Whether it is a forward FFT (is_forward)
    static const std::array<
        std::pair<ComplexFFTConfig1D, std::function<void(float2*)>>, 28>
        dispatch_table = {{// Forward FFT configurations
                           {{64, 1, true}, dispatch_fft<64, 1, true>},
                           {{64, 2, true}, dispatch_fft<64, 2, true>},
                           {{128, 1, true}, dispatch_fft<128, 1, true>},
                           {{128, 2, true}, dispatch_fft<128, 2, true>},
                           {{256, 1, true}, dispatch_fft<256, 1, true>},
                           {{256, 2, true}, dispatch_fft<256, 2, true>},
                           {{512, 1, true}, dispatch_fft<512, 1, true>},
                           {{512, 2, true}, dispatch_fft<512, 2, true>},
                           {{1024, 1, true}, dispatch_fft<1024, 1, true>},
                           {{1024, 2, true}, dispatch_fft<1024, 2, true>},
                           {{2048, 1, true}, dispatch_fft<2048, 1, true>},
                           {{2048, 2, true}, dispatch_fft<2048, 2, true>},
                           {{4096, 1, true}, dispatch_fft<4096, 1, true>},
                           {{4096, 2, true}, dispatch_fft<4096, 2, true>},
                           // Inverse FFT configurations
                           {{64, 1, false}, dispatch_fft<64, 1, false>},
                           {{64, 2, false}, dispatch_fft<64, 2, false>},
                           {{128, 1, false}, dispatch_fft<128, 1, false>},
                           {{128, 2, false}, dispatch_fft<128, 2, false>},
                           {{256, 1, false}, dispatch_fft<256, 1, false>},
                           {{256, 2, false}, dispatch_fft<256, 2, false>},
                           {{512, 1, false}, dispatch_fft<512, 1, false>},
                           {{512, 2, false}, dispatch_fft<512, 2, false>},
                           {{1024, 1, false}, dispatch_fft<1024, 1, false>},
                           {{1024, 2, false}, dispatch_fft<1024, 2, false>},
                           {{2048, 1, false}, dispatch_fft<2048, 1, false>},
                           {{2048, 2, false}, dispatch_fft<2048, 2, false>},
                           {{4096, 1, false}, dispatch_fft<4096, 1, false>},
                           {{4096, 2, false}, dispatch_fft<4096, 2, false>}}};

    // Find matching configuration
    for (const auto& entry : dispatch_table) {
        if (entry.first.fft_size == fft_size &&
            entry.first.batch_size == batch_size &&
            entry.first.is_forward == is_forward) {
            return entry.second;
        }
    }

    // Return nullptr if configuration not found
    return nullptr;
}

// Common implementation function
void fft_c2c_1d_impl(torch::Tensor input, bool is_forward) {
    TORCH_CHECK(input.device().is_cuda(),
                "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat,
                "Input tensor must be of type torch.complex64");

    unsigned int fft_size, batch_size;

    // Doing dimension checks for fft size and batch dimension
    if (input.dim() == 1) {
        fft_size = input.size(0);
        batch_size = 1;
    } else if (input.dim() == 2) {
        fft_size = input.size(1);
        batch_size = input.size(0);
    } else {
        TORCH_CHECK(false, "Input tensor must be 1D or 2D. Got ", input.dim(),
                    "D.");
    }

    float2* data_ptr =
        reinterpret_cast<float2*>(input.data_ptr<c10::complex<float>>());

    // Use the dispatch table instead to figure out the appropriate FFT function
    // TODO: Better error handling, namely giving info on the supported FFT
    // configurations.
    auto fft_func = get_fft_function(fft_size, batch_size, is_forward);
    TORCH_CHECK(fft_func != nullptr,
                "Unsupported FFT configuration: fft_size=", fft_size,
                ", batch_size=", batch_size);

    fft_func(data_ptr);
}

void fft_c2c_1d(torch::Tensor input) {
    fft_c2c_1d_impl(input, true);  // Forward FFT
}

void ifft_c2c_1d(torch::Tensor input) {
    fft_c2c_1d_impl(input, false);  // Inverse FFT
}

PYBIND11_MODULE(zipfft_binding, m) {
    m.doc() = "pybind11 binding example";
    m.def("fft_c2c_1d", &fft_c2c_1d, "In-place 1D C2C FFT using cuFFTDx.");
    m.def("ifft_c2c_1d", &ifft_c2c_1d, "In-place 1D C2C IFFT using cuFFTDx.");
}