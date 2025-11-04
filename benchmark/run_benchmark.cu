// Benchmark script for 2D real convolution performance testing
// Tests (4096, 4096) image with (512, 512) filter
//
// Compile with: nvcc -I/home/mgiammar/miniconda3/envs/zipfft/include/
// -I/home/mgiammar/miniconda3/envs/zipfft-dev/include/cufftdx/
// -I/home/mgiammar/git_repositories/zipFFT/src/include
// -I/home/mgiammar/git_repositories/zipFFT/src/cuda -o run_benchmark
// /home/mgiammar/git_repositories/zipFFT/benchmark/run_benchmark.cu -arch=sm_89 -O3
// -DENABLE_CUDA_ARCH_890 Run with: ./run_benchmark
//
// GPU (RTX 6000 Ada) has following specs:
//  * 48 GB GDDR6 ECC
//  * 91.1 TFLOPS (FP32)
//  * 960 GB/s memory bandwidth

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <cmath>
#include <cufftdx.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../src/cuda/real_conv_2d.cuh"
#include "../src/include/zipfft_common.hpp"

// GPU specifications for roofline analysis
constexpr double GPU_PEAK_FLOPS = 91.1e12;      // 91.1 TFLOPS
constexpr double GPU_PEAK_BANDWIDTH = 960.0e9;  // 960 GB/s

// Benchmark configuration
// For Falcon 4i images
constexpr unsigned int FFTSizeX = 4096;
constexpr unsigned int FFTSizeY = 4096;

// // For k3 images
// constexpr unsigned int FFTSizeX = 4096;
// constexpr unsigned int FFTSizeY = 6561;  // 5760, closest

// constexpr unsigned int FFTSizeX = 8192;
// constexpr unsigned int FFTSizeY = 8192;

// constexpr unsigned int SignalLengthX = 256;
// constexpr unsigned int SignalLengthY = 256;
constexpr unsigned int SignalLengthX = 512;
constexpr unsigned int SignalLengthY = 512;

constexpr unsigned int BATCH_SIZE = 1;
constexpr bool CROSS_CORRELATE = true;

// Benchmark parameters
constexpr int NUM_WARMUP_RUNS = 20;
constexpr int NUM_TIMING_RUNS = 100;

// CUDA kernel for random data initialization
__global__ void init_random_float(float* data, size_t size, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_uniform(&state) * 2.0f - 1.0f;  // Range [-1, 1]
    }
}

__global__ void init_random_complex(float2* data, size_t size, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx].x = curand_uniform(&state) * 2.0f - 1.0f;
        data[idx].y = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

// Timing helper class using CUDA events
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;

public:
    CudaTimer() {
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&start_event));
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&stop_event));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        CUDA_CHECK_AND_EXIT(cudaEventRecord(start_event));
    }

    float stop() {
        CUDA_CHECK_AND_EXIT(cudaEventRecord(stop_event));
        CUDA_CHECK_AND_EXIT(cudaEventSynchronize(stop_event));
        float milliseconds = 0;
        CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }
};

// Benchmark statistics
struct BenchmarkStats {
    float min_time_ms;
    float max_time_ms;
    float mean_time_ms;
    float std_dev_ms;
    double throughput_gflops;
    double achieved_bandwidth_gbs;
    double arithmetic_intensity;
    double roofline_predicted_gflops;
    double efficiency_percent;
    size_t memory_used_mb;
};

// Calculate work (FLOPs) for R2C FFT with zero-padding
double work_r2c_zip(unsigned int n, unsigned int N) {
    return 2.5 * N * std::log2(N);
}

// Calculate traffic (bytes) for R2C FFT with zero-padding
double traffic_r2c_zip(unsigned int n, unsigned int N) {
    double real_traffic = n;
    double complex_traffic = N / 2 + 1;
    return 4 * real_traffic + 8 * complex_traffic;
}

// Calculate arithmetic intensity for R2C FFT with zero-padding
double intensity_r2c_zip(unsigned int n, unsigned int N) {
    return work_r2c_zip(n, N) / traffic_r2c_zip(n, N);
}

// Calculate work (FLOPs) for fused C2C convolution with zero-padding
double work_fused_c2c_conv_zip(unsigned int n, unsigned int N) {
    double work_forward = 5.0 * N * std::log2(N);
    double work_multiply = 6.0 * N;
    double work_inverse = 5.0 * N * std::log2(N);
    return work_forward + work_multiply + work_inverse;
}

// Calculate traffic (bytes) for fused C2C convolution with zero-padding
double traffic_fused_c2c_conv_zip(unsigned int n, unsigned int N) {
    double input_traffic = n;
    double output_traffic = N - n + 1;
    double conv_traffic = N;
    return 8.0 * (input_traffic + output_traffic + conv_traffic);  // all complex float2
}

// Calculate arithmetic intensity for fused C2C convolution with zero-padding
double intensity_fused_c2c_conv_zip(unsigned int n, unsigned int N) {
    return work_fused_c2c_conv_zip(n, N) / traffic_fused_c2c_conv_zip(n, N);
}

// Calculate work (FLOPs) for C2R FFT with zero-padding
double work_c2r_zip(unsigned int n, unsigned int N) {
    return 2.5 * N * std::log2(N);
}

// Calculate traffic (bytes) for C2R FFT with zero-padding
double traffic_c2r_zip(unsigned int n, unsigned int N) {
    double complex_traffic = N / 2 + 1;
    double real_traffic = N - n + 1;
    return 8.0 * complex_traffic + 4.0 * real_traffic;
}

// Calculate arithmetic intensity for C2R FFT with zero-padding
double intensity_c2r_zip(unsigned int n, unsigned int N) {
    return work_c2r_zip(n, N) / traffic_c2r_zip(n, N);
}

// Estimate total FLOPs and traffic for 2D real convolution
struct WorkTrafficPair {
    double work;     // FLOPs
    double traffic;  // Bytes
};

WorkTrafficPair estimate_work_and_traffic(unsigned int batch, unsigned int fft_x,
                                          unsigned int fft_y, unsigned int signal_x,
                                          unsigned int signal_y) {
    WorkTrafficPair result = {0.0, 0.0};
    unsigned int valid_y = fft_y - signal_y + 1;

    // Stage 1: R2C forward FFT along X dimension
    // batch * signal_y FFTs of size fft_x, padding from signal_x to fft_x
    double r2c_work = batch * signal_y * work_r2c_zip(signal_x, fft_x);
    double r2c_traffic = batch * signal_y * traffic_r2c_zip(signal_x, fft_x);

    // Stage 2: Fused C2C convolution along Y dimension
    // batch * (fft_x/2+1) FFTs of size fft_y, padding from signal_y to fft_y
    double c2c_work = batch * (fft_x / 2 + 1) * work_fused_c2c_conv_zip(signal_y, fft_y);
    double c2c_traffic = batch * (fft_x / 2 + 1) * traffic_fused_c2c_conv_zip(signal_y, fft_y);

    // Stage 3: C2R inverse FFT along X dimension
    // batch * valid_y FFTs of size fft_x, truncating to valid_x = fft_x - signal_x + 1
    unsigned int valid_x = fft_x - signal_x + 1;
    double c2r_work = batch * valid_y * work_c2r_zip(signal_x, fft_x);
    double c2r_traffic = batch * valid_y * traffic_c2r_zip(signal_x, fft_x);

    result.work = r2c_work + c2c_work + c2r_work;
    result.traffic = r2c_traffic + c2c_traffic + c2r_traffic;

    return result;
}

BenchmarkStats compute_stats(const std::vector<float>& times, const WorkTrafficPair& work_traffic,
                             size_t memory_bytes) {
    BenchmarkStats stats;

    stats.min_time_ms = *std::min_element(times.begin(), times.end());
    stats.max_time_ms = *std::max_element(times.begin(), times.end());

    float sum = 0.0f;
    for (float t : times)
        sum += t;
    stats.mean_time_ms = sum / times.size();

    float sq_sum = 0.0f;
    for (float t : times) {
        float diff = t - stats.mean_time_ms;
        sq_sum += diff * diff;
    }
    stats.std_dev_ms = std::sqrt(sq_sum / times.size());

    // Convert time to seconds for throughput calculations
    double min_time_s = stats.min_time_ms / 1000.0;

    // Throughput in GFLOPS (using best time)
    stats.throughput_gflops = (work_traffic.work / 1e9) / min_time_s;

    // Achieved bandwidth in GB/s (using best time)
    stats.achieved_bandwidth_gbs = (work_traffic.traffic / 1e9) / min_time_s;

    // Arithmetic intensity (FLOPs per byte)
    stats.arithmetic_intensity = work_traffic.work / work_traffic.traffic;

    // Roofline predicted performance
    // Performance is limited by either compute or memory bandwidth
    double compute_bound_gflops = GPU_PEAK_FLOPS / 1e9;
    double memory_bound_gflops = (GPU_PEAK_BANDWIDTH / 1e9) * stats.arithmetic_intensity;
    stats.roofline_predicted_gflops = std::min(compute_bound_gflops, memory_bound_gflops);

    // Efficiency as percentage of roofline prediction
    stats.efficiency_percent = 100.0 * stats.throughput_gflops / stats.roofline_predicted_gflops;

    stats.memory_used_mb = memory_bytes / (1024 * 1024);

    return stats;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "2D Real Convolution Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Filter size: " << SignalLengthY << " x " << SignalLengthX << "\n";
    std::cout << "  FFT size:    " << FFTSizeY << " x " << FFTSizeX << "\n";
    std::cout << "  Output size: " << (FFTSizeY - SignalLengthY + 1) << " x "
              << (FFTSizeX - SignalLengthX + 1) << "\n";
    std::cout << "  Batch size:  " << BATCH_SIZE << "\n";
    std::cout << "  Operation:   " << (CROSS_CORRELATE ? "Cross-correlation" : "Convolution")
              << "\n";
    std::cout << "  Warmup runs: " << NUM_WARMUP_RUNS << "\n";
    std::cout << "  Timing runs: " << NUM_TIMING_RUNS << "\n";
    std::cout << "========================================\n\n";

    // Get GPU properties
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
    CUDA_CHECK_AND_EXIT(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Total Memory: " << (prop.totalGlobalMem / (1024 * 1024 * 1024)) << " GB\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Peak FP32: " << (GPU_PEAK_FLOPS / 1e12) << " TFLOPS\n";
    std::cout << "Peak Bandwidth: " << (GPU_PEAK_BANDWIDTH / 1e9) << " GB/s\n\n";

    const unsigned int StrideY = FFTSizeX / 2 + 1;
    const unsigned int ValidLengthX = FFTSizeX - SignalLengthX + 1;
    const unsigned int ValidLengthY = FFTSizeY - SignalLengthY + 1;

    // Calculate buffer sizes
    const size_t input_size = size_t(BATCH_SIZE) * SignalLengthY * SignalLengthX;
    const size_t workspace_size = size_t(BATCH_SIZE) * FFTSizeY * StrideY;
    const size_t conv_size = size_t(FFTSizeY) * StrideY;
    const size_t output_size = size_t(BATCH_SIZE) * ValidLengthY * ValidLengthX;

    const size_t total_memory = input_size * sizeof(float) + workspace_size * sizeof(float2) +
                                conv_size * sizeof(float2) + output_size * sizeof(float);

    std::cout << "Memory allocation:\n";
    std::cout << "  Input:     " << (input_size * sizeof(float) / (1024 * 1024)) << " MB\n";
    std::cout << "  Workspace: " << (workspace_size * sizeof(float2) / (1024 * 1024)) << " MB\n";
    std::cout << "  Conv data: " << (conv_size * sizeof(float2) / (1024 * 1024)) << " MB\n";
    std::cout << "  Output:    " << (output_size * sizeof(float) / (1024 * 1024)) << " MB\n";
    std::cout << "  Total:     " << (total_memory / (1024 * 1024)) << " MB\n\n";

    // Calculate theoretical work and traffic
    WorkTrafficPair work_traffic =
        estimate_work_and_traffic(BATCH_SIZE, FFTSizeX, FFTSizeY, SignalLengthX, SignalLengthY);

    std::cout << "Theoretical analysis:\n";
    std::cout << "  Total FLOPs:  " << std::scientific << std::setprecision(3) << work_traffic.work
              << " (" << (work_traffic.work / 1e9) << " GFLOPs)\n";
    std::cout << "  Total traffic: " << work_traffic.traffic << " bytes ("
              << (work_traffic.traffic / (1024 * 1024)) << " MB)\n";
    std::cout << "  Intensity (theoretic): " << std::fixed << std::setprecision(2)
              << (work_traffic.work / work_traffic.traffic) << " FLOPs/byte\n\n";

    // Allocate device memory
    std::cout << "Allocating GPU memory... " << std::flush;
    float* d_input = nullptr;
    float2* d_workspace = nullptr;
    float2* d_conv = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_workspace, workspace_size * sizeof(float2)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_conv, conv_size * sizeof(float2)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_output, output_size * sizeof(float)));
    std::cout << "Done!\n";

    // Initialize with random data
    std::cout << "Initializing random data... " << std::flush;
    const int threads = 256;
    {
        int blocks = (input_size + threads - 1) / threads;
        init_random_float<<<blocks, threads>>>(d_input, input_size, 12345ULL);
    }
    {
        int blocks = (conv_size + threads - 1) / threads;
        init_random_complex<<<blocks, threads>>>(d_conv, conv_size, 67890ULL);
    }
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    std::cout << "Done!\n\n";

    // Warmup runs
    std::cout << "Running " << NUM_WARMUP_RUNS << " warmup iterations... " << std::flush;
    for (int i = 0; i < NUM_WARMUP_RUNS; ++i) {
        padded_block_real_conv_2d<float, float2, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY,
                                  BATCH_SIZE, CROSS_CORRELATE>(d_input, d_workspace, d_conv,
                                                               d_output);
    }
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    std::cout << "Done!\n\n";

    // Timing runs - batch all iterations together to amortize overhead
    std::cout << "Running " << NUM_TIMING_RUNS << " timed iterations (batched)...\n";
    CudaTimer timer;

    // Queue all kernel launches before starting timer to maximize GPU utilization
    timer.start();
    for (int i = 0; i < NUM_TIMING_RUNS; ++i) {
        padded_block_real_conv_2d<float, float2, SignalLengthX, SignalLengthY, FFTSizeX, FFTSizeY,
                                  BATCH_SIZE, CROSS_CORRELATE>(d_input, d_workspace, d_conv,
                                                               d_output);
    }
    float total_elapsed = timer.stop();
    std::cout << "Done!\n\n";

    // Calculate average time per iteration
    float avg_time_ms = total_elapsed / NUM_TIMING_RUNS;
    std::vector<float> times(NUM_TIMING_RUNS, avg_time_ms);

    // Calculate statistics
    BenchmarkStats stats = compute_stats(times, work_traffic, total_memory);

    // Print results
    std::cout << "========================================\n";
    std::cout << "Benchmark Results\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Timing (ms):\n";
    std::cout << "  Minimum:  " << stats.min_time_ms << "\n";
    std::cout << "  Maximum:  " << stats.max_time_ms << "\n";
    std::cout << "  Mean:     " << stats.mean_time_ms << "\n";
    std::cout << "  Std Dev:  " << stats.std_dev_ms << "\n\n";

    std::cout << std::setprecision(2);
    std::cout << "Performance:\n";
    std::cout << "  Achieved throughput:    " << stats.throughput_gflops << " GFLOPS\n";
    std::cout << "  Roofline prediction:    " << stats.roofline_predicted_gflops << " GFLOPS ("
              << (100.0 * stats.throughput_gflops / stats.roofline_predicted_gflops) << "%)\n";
    std::cout << "  Achieved bandwidth:     " << stats.achieved_bandwidth_gbs << " GB/s\n";
    std::cout << "  Theo/Pct bandwidth:     " << GPU_PEAK_BANDWIDTH / 1e9 << " GB/s ("
              << (100.0 * stats.achieved_bandwidth_gbs / (GPU_PEAK_BANDWIDTH / 1e9)) << "%)\n";
    std::cout << "  Intensity (theoretic):  " << stats.arithmetic_intensity << " FLOPs/byte\n";
    std::cout << "  Time per batch:         " << (1000.0 * stats.min_time_ms / BATCH_SIZE) << " ns\n";
    std::cout << "  Batches/sec:            " << (1000.0 / (stats.min_time_ms / BATCH_SIZE)) << "\n";
    std::cout << "========================================\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_workspace);
    cudaFree(d_conv);
    cudaFree(d_output);

    return 0;
}