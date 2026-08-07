// Auto-generated from configs.yaml by setup.py -- do not edit directly.
// Add/remove shapes in configs.yaml and rebuild instead.
#pragma once

#include <array>
#include <tuple>

// (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch_size, cross_correlate)
static constexpr std::array<
    std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool>,
    27>
    SUPPORTED_C2C_CONV_CONFIGS = {{
        {48, 48, 64, 64, 1, false},
        {48, 48, 64, 64, 8, false},
        {96, 96, 128, 128, 1, false},
        {96, 96, 128, 128, 4, false},
        {192, 192, 256, 256, 1, false},
        {192, 192, 256, 256, 4, false},
        {384, 192, 512, 256, 1, false},
        {192, 384, 256, 512, 1, false},
        {48, 48, 64, 64, 1, true},
        {48, 48, 64, 64, 8, true},
        {96, 96, 128, 128, 1, true},
        {96, 96, 128, 128, 4, true},
        {192, 192, 256, 256, 1, true},
        {192, 192, 256, 256, 4, true},
        {384, 192, 512, 256, 1, true},
        {192, 384, 256, 512, 1, true},
        {16, 16, 64, 64, 1, true},
        {32, 32, 128, 128, 1, true},
        {512, 512, 4096, 4096, 1, true},
        {512, 512, 4096, 4096, 4, true},
        {512, 512, 4096, 4096, 8, true},
        {512, 512, 4096, 4096, 12, true},
        {512, 512, 4096, 4096, 16, true},
        {512, 512, 4096, 4096, 20, true},
        {512, 512, 4096, 4096, 24, true},
        {512, 512, 4096, 4096, 28, true},
        {512, 512, 4096, 4096, 32, true},
    }};
