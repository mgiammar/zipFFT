// Auto-generated from configs.yaml by setup.py -- do not edit directly.
// Add/remove shapes in configs.yaml and rebuild instead.
#pragma once

#include <array>
#include <tuple>

// (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch_size, cross_correlate,
//  use_tiled_swizzled_io, ffts_per_block_y)
static constexpr std::array<
    std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool,
               bool, unsigned int>,
    27>
    SUPPORTED_C2C_CONV_CONFIGS = {{
        {48, 48, 64, 64, 1, false, false, 0},
        {48, 48, 64, 64, 8, false, false, 0},
        {96, 96, 128, 128, 1, false, false, 0},
        {96, 96, 128, 128, 4, false, false, 0},
        {192, 192, 256, 256, 1, false, false, 0},
        {192, 192, 256, 256, 4, false, false, 0},
        {384, 192, 512, 256, 1, false, true, 4},
        {192, 384, 256, 512, 1, false, false, 0},
        {48, 48, 64, 64, 1, true, false, 0},
        {48, 48, 64, 64, 8, true, false, 0},
        {96, 96, 128, 128, 1, true, false, 0},
        {96, 96, 128, 128, 4, true, false, 0},
        {192, 192, 256, 256, 1, true, false, 0},
        {192, 192, 256, 256, 4, true, false, 0},
        {384, 192, 512, 256, 1, true, true, 4},
        {192, 384, 256, 512, 1, true, false, 0},
        {16, 16, 64, 64, 1, true, false, 0},
        {32, 32, 128, 128, 1, true, false, 0},
        {512, 512, 4096, 4096, 1, true, true, 2},
        {512, 512, 4096, 4096, 4, true, true, 2},
        {512, 512, 4096, 4096, 8, true, true, 2},
        {512, 512, 4096, 4096, 12, true, true, 2},
        {512, 512, 4096, 4096, 16, true, true, 2},
        {512, 512, 4096, 4096, 20, true, true, 2},
        {512, 512, 4096, 4096, 24, true, true, 2},
        {512, 512, 4096, 4096, 28, true, true, 2},
        {512, 512, 4096, 4096, 32, true, true, 2},
    }};
