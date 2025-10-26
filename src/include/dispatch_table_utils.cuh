#include <array>
#include <vector>
#include <functional>

#include <c10/util/complex.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftdx.hpp>

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT

std::pair<unsigned int, unsigned int> factorizePowerOfTwo_bitwise(int n) {
    if (n == 0) {
        return {0, 0};
    }

    int k = 1;
    // Continue as long as the last bit is 0 (i.e., the number is even)
    while ((n & 1) == 0) {
        n >>= 1; // Right shift is equivalent to dividing by 2
        k = k << 1; // Multiply k by 2
    }
    return {k, n};
}

std::pair<unsigned int, unsigned int> get_supported_batches_runtime(
    unsigned int FFTSize,
    unsigned int total_batches,
    unsigned int max_fused_batches) {

    auto factors = factorizePowerOfTwo_bitwise(total_batches);
    unsigned int batch_size = 1;

    if (FFTSize <= 256 && factors.first >= 32) {
        factors.first = factors.first / 32;
        batch_size = 32;
    } else if (FFTSize <= 512 && factors.first >= 16) {
        factors.first = factors.first / 16;
        batch_size = 16;
    } else if (FFTSize <= 1024 && factors.first >= 8) {
        factors.first = factors.first / 8;
        batch_size = 8;
    } else if (FFTSize <= 2048 && factors.first >= 4) {
        factors.first = factors.first / 4;
        batch_size = 4;
    } else if (FFTSize <= 4096 && factors.first >= 2) {
        factors.first = factors.first / 2;
        batch_size = 2;
    }

    // if (FFTSize <= 1024 && factors.first >= 4) {
    //     factors.first = factors.first / 4;
    //     batch_size = 4;
    // } else if (FFTSize <= 2048 && factors.first >= 2) {
    //     factors.first = factors.first / 2;
    //     batch_size = 2;
    // }

    return {factors.first * factors.second, batch_size};
}

static constexpr std::array<std::pair<unsigned int, unsigned int>, 32> // 18
    SUPPORTED_FFT_CONFIGS = {{
                              {64, 1},
                              {64, 2},
                              {64, 4},
                              {64, 8},
                              {64, 16},
                              {64, 32},

                              {128, 1},
                              {128, 2},
                              {128, 4},
                              {128, 8},
                              {128, 16},
                              {128, 32},

                              {256, 1},
                              {256, 2},
                              {256, 4},
                              {256, 8},
                              {256, 16},
                              {256, 32},

                              {512, 1},
                              {512, 2},
                              {512, 4},
                              {512, 8},
                              {512, 16},

                              {1024, 1},
                              {1024, 2},
                              {1024, 4},
                              {1024, 8},

                              {2048, 1},
                              {2048, 2},
                              {2048, 4},

                              {4096, 1},
                              {4096, 2},
                            }};

static constexpr std::array<unsigned int, 7> SUPPORTED_FFT_SIZES = {64, 128, 256, 512, 1024, 2048, 4096};

template <unsigned int FFTSize, unsigned int BatchSize, unsigned int Arch>
void dispatch_function(void* params, cudaStream_t strm);

inline unsigned int get_cuda_device_arch() {
    int device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));

    int major = 0;
    int minor = 0;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

    return static_cast<unsigned>(major) * 100 + static_cast<unsigned>(minor) * 10;
}

template <unsigned int FFTSize, unsigned int BatchSize>
int select_arch_and_dispatch(void* params) {
    auto arch = get_cuda_device_arch();
    cudaStream_t strm = at::cuda::getCurrentCUDAStream().stream();

    switch (arch) {
        case 800: dispatch_function<FFTSize, BatchSize, 800>(params, strm); break;
        case 860: dispatch_function<FFTSize, BatchSize, 860>(params, strm); break;
        case 870: dispatch_function<FFTSize, BatchSize, 870>(params, strm); break;
        case 890: dispatch_function<FFTSize, BatchSize, 890>(params, strm); break;
        case 900: dispatch_function<FFTSize, BatchSize, 900>(params, strm); break;
        case 1200: dispatch_function<FFTSize, BatchSize, 1200>(params, strm); break;
        default:
            std::cerr << "Unsupported CUDA architecture: " << arch
                      << ". Supported architectures are 800, 860, 870, 890, "
                         "900, and 1200."
                      << std::endl;
            return -1;  // Error code for unsupported architecture
    }

    return 0;
}

template <std::size_t... Is>
constexpr auto make_dispatch_table(std::index_sequence<Is...>) {
    return std::array<
        std::pair<std::pair<unsigned int, unsigned int>, std::function<int(void*)>>,
        sizeof...(Is)>{{{SUPPORTED_FFT_CONFIGS[Is], []() {
            return select_arch_and_dispatch<SUPPORTED_FFT_CONFIGS[Is].first, SUPPORTED_FFT_CONFIGS[Is].second>;
        }()}...}};
}

static const auto dispatch_table = make_dispatch_table(
    std::make_index_sequence<SUPPORTED_FFT_CONFIGS.size()>{}
);

std::function<int(void*)> get_function_from_table(unsigned int fft_size, unsigned int batch_size) {
    for (const auto& entry : dispatch_table) {
        if (entry.first.first == fft_size &&
            entry.first.second == batch_size) {
            return entry.second;
        }
    }

    return nullptr;
}

std::vector<int> get_supported_sizes() {
    std::vector<int> sizes;
    sizes.reserve(SUPPORTED_FFT_SIZES.size());

    for (const auto& fft_size : SUPPORTED_FFT_SIZES) {
        sizes.emplace_back(static_cast<int>(fft_size));
    }

    return sizes;
}

constexpr int get_padding_ratio() {
    return 8;
}