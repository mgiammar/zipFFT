// Microbenchmark for zipFFT's real-valued (R2C/C2R) 2D padded convolution/cross-correlation.
//
// Shape and backend-IO-path are compile-time set via -D macros with the defaults below.
//
// Compile (defaults -- 4096x4096 FFT, 512x512 signal, batch 16, cross-correlation, plain IO):
//   nvcc -o bench_real_conv2d benchmark/bench_real_conv2d.cu \
//     -I<conda-env>/include -I<conda-env>/include/cufftdx \
//     -Isrc/include -Isrc/cuda \
//     -arch=sm_89 -O3 -DENABLE_CUDA_ARCH_890
//
// Override shape/IO path at compile time, e.g.:
//   nvcc ... -DFFT_SIZE_X=512 -DFFT_SIZE_Y=512 -DSIGNAL_X=384 -DSIGNAL_Y=384 -DBATCH_SIZE=4 \
//             -DUSE_TILED_SWIZZLED_IO=1 -DFFTS_PER_BLOCK_Y=4
//
// (benchmark/build.py wraps this invocation with a friendlier CLI and unique output paths.)
//
// Run:
//   ./bench_real_conv2d --warmup=20 --iters=100 --json=result.json

#ifndef FFT_SIZE_X
#define FFT_SIZE_X 4096
#endif
#ifndef FFT_SIZE_Y
#define FFT_SIZE_Y 4096
#endif
#ifndef SIGNAL_X
#define SIGNAL_X 512
#endif
#ifndef SIGNAL_Y
#define SIGNAL_Y 512
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 16
#endif
#ifndef CROSS_CORRELATE
#define CROSS_CORRELATE 1
#endif
#ifndef USE_TILED_SWIZZLED_IO
#define USE_TILED_SWIZZLED_IO 0
#endif
#ifndef FFTS_PER_BLOCK_X
#define FFTS_PER_BLOCK_X 0
#endif
#ifndef FFTS_PER_BLOCK_Y
#define FFTS_PER_BLOCK_Y 0
#endif
#ifndef ELEMENTS_PER_THREAD_X
#define ELEMENTS_PER_THREAD_X 0
#endif
#ifndef ELEMENTS_PER_THREAD_Y
#define ELEMENTS_PER_THREAD_Y 0
#endif

#include <cufftdx.hpp>

#include "../src/cuda/real_conv_2d.cuh"
#include "../src/include/zipfft_common.hpp"
#include "common/bench_common.cuh"

using namespace zipfft_bench;

constexpr unsigned int FFTSizeX = FFT_SIZE_X;
constexpr unsigned int FFTSizeY = FFT_SIZE_Y;
constexpr unsigned int SignalLengthX = SIGNAL_X;
constexpr unsigned int SignalLengthY = SIGNAL_Y;
constexpr unsigned int Batch = BATCH_SIZE;
constexpr bool CrossCorrelate = CROSS_CORRELATE != 0;
constexpr bool UseTiledSwizzledIO = USE_TILED_SWIZZLED_IO != 0;
constexpr unsigned int FFTsPerBlockX = FFTS_PER_BLOCK_X;
constexpr unsigned int FFTsPerBlockY = FFTS_PER_BLOCK_Y;
constexpr unsigned int ElementsPerThreadX = ELEMENTS_PER_THREAD_X;
constexpr unsigned int ElementsPerThreadY = ELEMENTS_PER_THREAD_Y;

int main(int argc, char** argv) {
    CliArgs args(argc, argv);

    if (args.has("help")) {
        std::cout << "Usage: bench_real_conv2d [--warmup=N] [--iters=N] [--repeats-per-sample=K] "
                     "[--device=N] [--json=path] [--peak-flops-tflops=X] "
                     "[--peak-bandwidth-gbs=X]\n"
                     "Shape/IO path are compile-time; see file header for -D overrides.\n";
        return 0;
    }

    const int device = args.get_int("device", 0);
    const int num_warmup = args.get_int("warmup", 20);
    const int num_iters = args.get_int("iters", 100);
    const int repeats_per_sample = std::max(1, args.get_int("repeats-per-sample", 1));
    const std::string json_path = args.get_string("json", "");

    CUDA_CHECK_AND_EXIT(cudaSetDevice(device));
    GpuSpec gpu = query_gpu_spec(device, args.get_optional_double("peak-flops-tflops"),
                                 args.get_optional_double("peak-bandwidth-gbs"));

    std::cout << "========================================\n";
    std::cout << "zipFFT Real (R2C/C2R) Conv2D Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "  Filter size: " << SignalLengthY << " x " << SignalLengthX << "\n";
    std::cout << "  FFT size:    " << FFTSizeY << " x " << FFTSizeX << "\n";
    std::cout << "  Output size: " << (FFTSizeY - SignalLengthY + 1) << " x "
              << (FFTSizeX - SignalLengthX + 1) << "\n";
    std::cout << "  Batch size:  " << Batch << "\n";
    std::cout << "  Operation:   " << (CrossCorrelate ? "Cross-correlation" : "Convolution")
              << "\n";
    std::cout << "  Tiled+swizzled Y IO: " << (UseTiledSwizzledIO ? "ON" : "OFF")
              << " (ffts_per_block_y=" << FFTsPerBlockY << ")\n";
    std::cout << "  Warmup/timed iters:  " << num_warmup << " / " << num_iters << " (x"
              << repeats_per_sample << " repeats/sample)\n";
    std::cout << "========================================\n\n";

    print_gpu_spec(gpu);

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    const unsigned int StrideY = FFTSizeX / 2 + 1;
    const unsigned int ValidLengthX = FFTSizeX - SignalLengthX + 1;
    const unsigned int ValidLengthY = FFTSizeY - SignalLengthY + 1;

    const size_t input_size = size_t(Batch) * SignalLengthY * SignalLengthX;
    const size_t workspace_size = size_t(Batch) * FFTSizeY * StrideY;
    const size_t conv_size = size_t(FFTSizeY) * StrideY;
    const size_t output_size = size_t(Batch) * ValidLengthY * ValidLengthX;

    MemoryFootprintMb mem{};
    mem.input = input_size * sizeof(float) / (1024.0 * 1024.0);
    mem.workspace = workspace_size * sizeof(float2) / (1024.0 * 1024.0);
    mem.conv_data = conv_size * sizeof(float2) / (1024.0 * 1024.0);
    mem.output = output_size * sizeof(float) / (1024.0 * 1024.0);
    mem.total = mem.input + mem.workspace + mem.conv_data + mem.output;

    std::cout << "Memory allocation (MB): input=" << mem.input << " workspace=" << mem.workspace
              << " conv=" << mem.conv_data << " output=" << mem.output << " total=" << mem.total
              << "\n\n";

    float* d_input = nullptr;
    float2* d_workspace = nullptr;
    float2* d_conv = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_workspace, workspace_size * sizeof(float2)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_conv, conv_size * sizeof(float2)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_output, output_size * sizeof(float)));

    const int threads = 256;
    init_random_float<<<(input_size + threads - 1) / threads, threads>>>(d_input, input_size,
                                                                         12345ULL);
    init_random_complex<<<(conv_size + threads - 1) / threads, threads>>>(d_conv, conv_size,
                                                                          67890ULL);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    auto launch = [&]() {
        padded_block_real_conv_2d<float, float2, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY,
                                  Batch, CrossCorrelate, ElementsPerThreadX, ElementsPerThreadY,
                                  FFTsPerBlockX, FFTsPerBlockY, UseTiledSwizzledIO>(
            d_input, d_workspace, d_conv, d_output, device, stream);
    };

    for (int i = 0; i < num_warmup; ++i)
        launch();
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    std::vector<float> times_ms;
    times_ms.reserve(num_iters);
    CudaTimer timer;
    for (int i = 0; i < num_iters; ++i) {
        timer.start(stream);
        for (int r = 0; r < repeats_per_sample; ++r)
            launch();
        times_ms.push_back(timer.stop(stream));
    }

    WorkTrafficPair work_traffic = real_path::estimate_work_and_traffic(
        Batch, FFTSizeX, FFTSizeY, SignalLengthX, SignalLengthY);
    BenchmarkStats stats = compute_stats(times_ms, work_traffic, repeats_per_sample, gpu);

    std::cout << "========================================\n";
    std::cout << "Results\n";
    std::cout << "========================================\n";
    print_stats(stats);
    std::cout << "========================================\n";

    BenchConfig cfg{"real",
                    CrossCorrelate ? "corr" : "conv",
                    FFTSizeX,
                    FFTSizeY,
                    SignalLengthX,
                    SignalLengthY,
                    Batch,
                    UseTiledSwizzledIO,
                    FFTsPerBlockX,
                    FFTsPerBlockY,
                    ElementsPerThreadX,
                    ElementsPerThreadY};
    emit_json(build_result_json(cfg, gpu, stats, mem), json_path);

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    cudaFree(d_input);
    cudaFree(d_workspace);
    cudaFree(d_conv);
    cudaFree(d_output);

    return 0;
}
