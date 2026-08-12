#ifndef ZIPFFT_BENCHMARK_COMMON_CUH
#define ZIPFFT_BENCHMARK_COMMON_CUH

/**
 * @brief Shared infrastructure for the zipFFT CUDA microbenchmarks (bench_real_conv2d.cu,
 * bench_complex_conv2d.cu). Shape (FFT size, batch, IO path) is intentionally NOT configured
 * here; cuFFTDx requires those as compile-time template parameters, so each driver takes them
 * as preprocessor macros (see the driver files). Everything in this header is genuinely runtime:
 * timing, GPU spec discovery, work/traffic estimation, and JSON output.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/include/zipfft_common.hpp"

namespace zipfft_bench {

// ---------------------------------------------------------------------------------------------
// Random data initialization kernels
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// Per-iteration CUDA event timer
// ---------------------------------------------------------------------------------------------

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

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK_AND_EXIT(cudaEventRecord(start_event, stream));
    }

    // Blocks until the stream has finished; returns elapsed milliseconds since start().
    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK_AND_EXIT(cudaEventRecord(stop_event, stream));
        CUDA_CHECK_AND_EXIT(cudaEventSynchronize(stop_event));
        float milliseconds = 0;
        CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }
};

// ---------------------------------------------------------------------------------------------
// Best-effort GPU spec discovery for roofline calculations. Some are always exact through the
// CUDA runtime API (SM count, clock, memory bandwidth); others are best-effort estimates based on
// compute capability lookup tables (peak FP32 FLOPs) which may be overriden.
// ---------------------------------------------------------------------------------------------

enum class SpecSource { kQueried, kEstimated, kUserSupplied, kUnavailable };

inline const char* to_string(SpecSource s) {
    switch (s) {
        case SpecSource::kQueried:
            return "queried";
        case SpecSource::kEstimated:
            return "estimated";
        case SpecSource::kUserSupplied:
            return "user_supplied";
        default:
            return "unavailable";
    }
}

// Best-effort FP32 CUDA-cores-per-SM by compute capability (major*10 + minor).
inline std::optional<int> fp32_cores_per_sm(int major, int minor) {
    const int cc = major * 10 + minor;
    switch (cc) {
        case 80:
            return 64;  // Ampere GA100 (A100)
        case 86:
        case 87:
            return 128;  // Ampere GA10x
        case 89:
            return 128;  // Ada Lovelace
        case 90:
            return 128;  // Hopper
        case 120:
            return 128;  // Blackwell (best-effort; confirm with --peak-flops-tflops if uncertain)
        default:
            return std::nullopt;
    }
}

struct GpuSpec {
    std::string name;
    int device_id = 0;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    double clock_hz = 0.0;
    size_t total_mem_bytes = 0;

    double bandwidth_gbs = 0.0;
    SpecSource bandwidth_source = SpecSource::kUnavailable;

    std::optional<double> peak_flops_tflops;
    SpecSource flops_source = SpecSource::kUnavailable;
};

// Queries device properties and derives bandwidth (always) and peak FP32 FLOPs (best-effort).
// `user_peak_flops_tflops` / `user_bandwidth_gbs` (if provided, > 0) take precedence over
// anything queried/estimated.
inline GpuSpec query_gpu_spec(int device,
                              std::optional<double> user_peak_flops_tflops = std::nullopt,
                              std::optional<double> user_bandwidth_gbs = std::nullopt) {
    cudaDeviceProp prop;
    CUDA_CHECK_AND_EXIT(cudaGetDeviceProperties(&prop, device));

    int clock_khz = 0, mem_clock_khz = 0, mem_bus_width_bits = 0;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device));
    CUDA_CHECK_AND_EXIT(
        cudaDeviceGetAttribute(&mem_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device));

    GpuSpec spec;
    spec.name = prop.name;
    spec.device_id = device;
    spec.cc_major = prop.major;
    spec.cc_minor = prop.minor;
    spec.sm_count = prop.multiProcessorCount;
    spec.clock_hz = static_cast<double>(clock_khz) * 1000.0;
    spec.total_mem_bytes = prop.totalGlobalMem;

    if (user_bandwidth_gbs.has_value()) {
        spec.bandwidth_gbs = *user_bandwidth_gbs;
        spec.bandwidth_source = SpecSource::kUserSupplied;
    } else {
        // Peak bandwidth (GB/s) = 2 (DDR) * memClock(Hz) * busWidth(bytes)
        const double mem_clock_hz = static_cast<double>(mem_clock_khz) * 1000.0;
        const double bus_width_bytes = static_cast<double>(mem_bus_width_bits) / 8.0;
        if (mem_clock_hz > 0.0 && bus_width_bytes > 0.0) {
            spec.bandwidth_gbs = 2.0 * mem_clock_hz * bus_width_bytes / 1e9;
            spec.bandwidth_source = SpecSource::kQueried;
        } else {
            spec.bandwidth_gbs = 0.0;
            spec.bandwidth_source = SpecSource::kUnavailable;
        }
    }

    if (user_peak_flops_tflops.has_value()) {
        spec.peak_flops_tflops = *user_peak_flops_tflops;
        spec.flops_source = SpecSource::kUserSupplied;
    } else {
        auto cores = fp32_cores_per_sm(prop.major, prop.minor);
        if (cores.has_value()) {
            const double flops = 2.0 * spec.sm_count * (*cores) * spec.clock_hz;  // 2 FLOPs/FMA
            spec.peak_flops_tflops = flops / 1e12;
            spec.flops_source = SpecSource::kEstimated;
        } else {
            spec.peak_flops_tflops = std::nullopt;
            spec.flops_source = SpecSource::kUnavailable;
        }
    }

    return spec;
}

// ---------------------------------------------------------------------------------------------
// Timing statistics
// ---------------------------------------------------------------------------------------------

struct BenchmarkStats {
    float min_time_ms = 0.0f;
    float max_time_ms = 0.0f;
    float mean_time_ms = 0.0f;
    float std_dev_ms = 0.0f;
    int num_samples = 0;
    int repeats_per_sample = 1;

    double throughput_gflops = 0.0;
    double achieved_bandwidth_gbs = 0.0;
    double arithmetic_intensity = 0.0;

    // Roofline numbers are only meaningful when the GPU spec they're derived
    // from isn't 'Unavailable'. See GpuSpec::flops_source.
    std::optional<double> roofline_gflops;
    std::optional<double> roofline_efficiency_percent;
    std::optional<double> bandwidth_efficiency_percent;
};

struct WorkTrafficPair {
    double work = 0.0;     // FLOPs
    double traffic = 0.0;  // Bytes
};

inline BenchmarkStats compute_stats(const std::vector<float>& times_ms,
                                    const WorkTrafficPair& work_traffic, int repeats_per_sample,
                                    const GpuSpec& gpu_spec) {
    BenchmarkStats stats;
    stats.num_samples = static_cast<int>(times_ms.size());
    stats.repeats_per_sample = repeats_per_sample;

    stats.min_time_ms = *std::min_element(times_ms.begin(), times_ms.end());
    stats.max_time_ms = *std::max_element(times_ms.begin(), times_ms.end());

    float sum = 0.0f;
    for (float t : times_ms)
        sum += t;
    stats.mean_time_ms = sum / times_ms.size();

    float sq_sum = 0.0f;
    for (float t : times_ms) {
        float diff = t - stats.mean_time_ms;
        sq_sum += diff * diff;
    }
    stats.std_dev_ms = std::sqrt(sq_sum / times_ms.size());

    // Best-case single-launch time (a "sample" may bundle repeats_per_sample launches).
    const double best_single_launch_ms =
        static_cast<double>(stats.min_time_ms) / repeats_per_sample;
    const double best_time_s = best_single_launch_ms / 1000.0;

    stats.throughput_gflops = (work_traffic.work / 1e9) / best_time_s;
    stats.achieved_bandwidth_gbs = (work_traffic.traffic / 1e9) / best_time_s;
    stats.arithmetic_intensity = work_traffic.work / work_traffic.traffic;

    if (gpu_spec.flops_source != SpecSource::kUnavailable && gpu_spec.peak_flops_tflops) {
        const double compute_bound_gflops = *gpu_spec.peak_flops_tflops * 1000.0;
        const double memory_bound_gflops = gpu_spec.bandwidth_gbs * stats.arithmetic_intensity;
        const double roofline = std::min(compute_bound_gflops, memory_bound_gflops);
        stats.roofline_gflops = roofline;
        stats.roofline_efficiency_percent = 100.0 * stats.throughput_gflops / roofline;
    }
    if (gpu_spec.bandwidth_source != SpecSource::kUnavailable && gpu_spec.bandwidth_gbs > 0.0) {
        stats.bandwidth_efficiency_percent =
            100.0 * stats.achieved_bandwidth_gbs / gpu_spec.bandwidth_gbs;
    }

    return stats;
}

// ---------------------------------------------------------------------------------------------
// Work (FLOPs) / traffic (bytes) estimators for the 3-kernel padded pipeline, used for the
// roofline calculation. Two variants: real-valued path (R2C forward X / fused C2C Y / C2R
// inverse X, with Hermitian-symmetry-reduced X traffic) and complex-valued path (C2C/C2C/C2C,
// full-width X traffic, no Hermitian reduction).
// ---------------------------------------------------------------------------------------------

namespace real_path {

inline double work_r2c(unsigned int N) {
    return 2.5 * N * std::log2(static_cast<double>(N));
}
inline double traffic_r2c(unsigned int n, unsigned int N) {
    return 4.0 * n + 8.0 * (N / 2 + 1);
}

inline double work_fused_c2c_conv(unsigned int N) {
    return 10.0 * N * std::log2(static_cast<double>(N)) + 6.0 * N;
}
inline double traffic_fused_c2c_conv(unsigned int n, unsigned int N) {
    return 8.0 * (n + (N - n + 1) + N);
}

inline double work_c2r(unsigned int N) {
    return 2.5 * N * std::log2(static_cast<double>(N));
}
inline double traffic_c2r(unsigned int n, unsigned int N) {
    return 8.0 * (N / 2 + 1) + 4.0 * (N - n + 1);
}

inline WorkTrafficPair estimate_work_and_traffic(unsigned int batch, unsigned int fft_x,
                                                 unsigned int fft_y, unsigned int signal_x,
                                                 unsigned int signal_y) {
    const unsigned int valid_y = fft_y - signal_y + 1;

    // Stage 1: R2C forward FFT along X. batch * signal_y FFTs of size fft_x.
    const double r2c_work = static_cast<double>(batch) * signal_y * work_r2c(fft_x);
    const double r2c_traffic = static_cast<double>(batch) * signal_y * traffic_r2c(signal_x, fft_x);

    // Stage 2: fused forward-Y / multiply / inverse-Y. batch * (fft_x/2+1) FFTs of size fft_y.
    const double c2c_work =
        static_cast<double>(batch) * (fft_x / 2 + 1) * work_fused_c2c_conv(fft_y);
    const double c2c_traffic =
        static_cast<double>(batch) * (fft_x / 2 + 1) * traffic_fused_c2c_conv(signal_y, fft_y);

    // Stage 3: C2R inverse FFT along X. batch * valid_y FFTs of size fft_x.
    const double c2r_work = static_cast<double>(batch) * valid_y * work_c2r(fft_x);
    const double c2r_traffic = static_cast<double>(batch) * valid_y * traffic_c2r(signal_x, fft_x);

    return {r2c_work + c2c_work + c2r_work, r2c_traffic + c2c_traffic + c2r_traffic};
}

}  // namespace real_path

namespace complex_path {

inline double work_c2c(unsigned int N) {
    return 5.0 * N * std::log2(static_cast<double>(N));
}
inline double traffic_c2c_fwd(unsigned int n, unsigned int N) {
    return 8.0 * (n + N);
}
inline double traffic_c2c_inv(unsigned int n, unsigned int N) {
    return 8.0 * (N + (N - n + 1));
}

inline double work_fused_c2c_conv(unsigned int N) {
    return 2.0 * work_c2c(N) + 6.0 * N;  // forward + multiply + inverse
}
inline double traffic_fused_c2c_conv(unsigned int n, unsigned int N) {
    return 8.0 * (n + (N - n + 1) + N);
}

inline WorkTrafficPair estimate_work_and_traffic(unsigned int batch, unsigned int fft_x,
                                                 unsigned int fft_y, unsigned int signal_x,
                                                 unsigned int signal_y) {
    const unsigned int valid_y = fft_y - signal_y + 1;

    // Stage 1: forward C2C FFT along X. batch * signal_y FFTs of size fft_x.
    const double x_fwd_work = static_cast<double>(batch) * signal_y * work_c2c(fft_x);
    const double x_fwd_traffic =
        static_cast<double>(batch) * signal_y * traffic_c2c_fwd(signal_x, fft_x);

    // Stage 2: fused forward-Y / multiply / inverse-Y. batch * fft_x FFTs of size fft_y (no
    // Hermitian reduction -- full complex spectrum in both directions).
    const double y_work = static_cast<double>(batch) * fft_x * work_fused_c2c_conv(fft_y);
    const double y_traffic =
        static_cast<double>(batch) * fft_x * traffic_fused_c2c_conv(signal_y, fft_y);

    // Stage 3: inverse C2C FFT along X. batch * valid_y FFTs of size fft_x.
    const double x_inv_work = static_cast<double>(batch) * valid_y * work_c2c(fft_x);
    const double x_inv_traffic =
        static_cast<double>(batch) * valid_y * traffic_c2c_inv(signal_x, fft_x);

    return {x_fwd_work + y_work + x_inv_work, x_fwd_traffic + y_traffic + x_inv_traffic};
}

}  // namespace complex_path

// ---------------------------------------------------------------------------------------------
// Minimal CLI parsing for the runtime-only flags (--warmup, --iters, --json, ...). Shape/IO-path
// parameters are compile-time macros (see driver files), not parsed here.
// ---------------------------------------------------------------------------------------------

class CliArgs {
public:
    CliArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--", 0) != 0)
                continue;
            auto eq = arg.find('=');
            if (eq == std::string::npos) {
                flags_[arg.substr(2)] = "1";
            } else {
                values_[arg.substr(2, eq - 2)] = arg.substr(eq + 1);
            }
        }
    }

    bool has(const std::string& key) const {
        return flags_.count(key) > 0 || values_.count(key) > 0;
    }

    int get_int(const std::string& key, int default_value) const {
        auto it = values_.find(key);
        return it == values_.end() ? default_value : std::atoi(it->second.c_str());
    }

    double get_double(const std::string& key, double default_value) const {
        auto it = values_.find(key);
        return it == values_.end() ? default_value : std::atof(it->second.c_str());
    }

    std::optional<double> get_optional_double(const std::string& key) const {
        auto it = values_.find(key);
        if (it == values_.end())
            return std::nullopt;
        return std::atof(it->second.c_str());
    }

    std::string get_string(const std::string& key, const std::string& default_value) const {
        auto it = values_.find(key);
        return it == values_.end() ? default_value : it->second;
    }

private:
    std::map<std::string, std::string> flags_;
    std::map<std::string, std::string> values_;
};

// ---------------------------------------------------------------------------------------------
// JSON output -- hand-rolled since the schema is small and fixed; avoids pulling in a JSON
// dependency for a benchmark utility. `sweep.py` parses this with Python's stdlib json module.
// ---------------------------------------------------------------------------------------------

struct BenchConfig {
    std::string path;       // "real" or "complex"
    std::string operation;  // "corr" or "conv"
    unsigned int fft_x, fft_y, signal_x, signal_y, batch;
    bool use_tiled_swizzled_io;
    unsigned int ffts_per_block_x, ffts_per_block_y;
    unsigned int elements_per_thread_x, elements_per_thread_y;
};

struct MemoryFootprintMb {
    double input, workspace, conv_data, output, total;
};

inline std::string optional_to_json(const std::optional<double>& v) {
    if (!v.has_value())
        return "null";
    std::ostringstream oss;
    oss << std::setprecision(10) << *v;
    return oss.str();
}

inline std::string build_result_json(const BenchConfig& cfg, const GpuSpec& gpu,
                                     const BenchmarkStats& stats, const MemoryFootprintMb& mem) {
    std::ostringstream j;
    j << std::setprecision(10);
    j << "{";
    j << "\"tool\":\"zipfft_cuda_bench\",\"schema_version\":1,";
    j << "\"path\":\"" << cfg.path << "\",\"operation\":\"" << cfg.operation << "\",";
    j << "\"shape\":{"
      << "\"fft_x\":" << cfg.fft_x << ",\"fft_y\":" << cfg.fft_y << ",\"signal_x\":" << cfg.signal_x
      << ",\"signal_y\":" << cfg.signal_y << ",\"batch\":" << cfg.batch << "},";
    j << "\"io\":{"
      << "\"use_tiled_swizzled_io\":" << (cfg.use_tiled_swizzled_io ? "true" : "false")
      << ",\"ffts_per_block_x\":" << cfg.ffts_per_block_x
      << ",\"ffts_per_block_y\":" << cfg.ffts_per_block_y
      << ",\"elements_per_thread_x\":" << cfg.elements_per_thread_x
      << ",\"elements_per_thread_y\":" << cfg.elements_per_thread_y << "},";
    j << "\"timing_ms\":{"
      << "\"min\":" << stats.min_time_ms << ",\"max\":" << stats.max_time_ms
      << ",\"mean\":" << stats.mean_time_ms << ",\"std_dev\":" << stats.std_dev_ms
      << ",\"num_samples\":" << stats.num_samples
      << ",\"repeats_per_sample\":" << stats.repeats_per_sample << "},";
    j << "\"performance\":{"
      << "\"throughput_gflops\":" << stats.throughput_gflops
      << ",\"achieved_bandwidth_gbs\":" << stats.achieved_bandwidth_gbs
      << ",\"arithmetic_intensity\":" << stats.arithmetic_intensity
      << ",\"roofline_gflops\":" << optional_to_json(stats.roofline_gflops)
      << ",\"roofline_efficiency_percent\":" << optional_to_json(stats.roofline_efficiency_percent)
      << ",\"bandwidth_efficiency_percent\":"
      << optional_to_json(stats.bandwidth_efficiency_percent) << "},";
    j << "\"gpu\":{"
      << "\"name\":\"" << gpu.name << "\",\"device_id\":" << gpu.device_id
      << ",\"compute_capability\":\"" << gpu.cc_major << "." << gpu.cc_minor << "\""
      << ",\"sm_count\":" << gpu.sm_count << ",\"clock_hz\":" << gpu.clock_hz
      << ",\"bandwidth_gbs\":" << gpu.bandwidth_gbs << ",\"bandwidth_source\":\""
      << to_string(gpu.bandwidth_source) << "\""
      << ",\"peak_flops_tflops\":" << optional_to_json(gpu.peak_flops_tflops)
      << ",\"flops_source\":\"" << to_string(gpu.flops_source) << "\"},";
    j << "\"memory_mb\":{"
      << "\"input\":" << mem.input << ",\"workspace\":" << mem.workspace
      << ",\"conv_data\":" << mem.conv_data << ",\"output\":" << mem.output
      << ",\"total\":" << mem.total << "}";
    j << "}";
    return j.str();
}

inline void emit_json(const std::string& json, const std::string& path) {
    if (path.empty()) {
        std::cout << json << std::endl;
        return;
    }
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Warning: could not open --json output path '" << path << "'\n";
        return;
    }
    out << json << std::endl;
}

// ---------------------------------------------------------------------------------------------
// Human-readable console report shared by both drivers.
// ---------------------------------------------------------------------------------------------

inline void print_gpu_spec(const GpuSpec& gpu) {
    std::cout << "GPU: " << gpu.name << " (device " << gpu.device_id << ")\n";
    std::cout << "Compute Capability: " << gpu.cc_major << "." << gpu.cc_minor
              << "  SMs: " << gpu.sm_count << "\n";
    std::cout << "Peak Bandwidth: " << std::fixed << std::setprecision(1) << gpu.bandwidth_gbs
              << " GB/s (" << to_string(gpu.bandwidth_source) << ")\n";
    if (gpu.peak_flops_tflops.has_value()) {
        std::cout << "Peak FP32: " << *gpu.peak_flops_tflops << " TFLOPS ("
                  << to_string(gpu.flops_source) << ")\n";
    } else {
        std::cout << "Peak FP32: unavailable -- pass --peak-flops-tflops=X to enable roofline %\n";
    }
    std::cout << "\n";
}

inline void print_stats(const BenchmarkStats& stats) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Timing (ms, " << stats.num_samples << " samples x " << stats.repeats_per_sample
              << " repeats/sample):\n";
    std::cout << "  Minimum:  " << stats.min_time_ms << "\n";
    std::cout << "  Maximum:  " << stats.max_time_ms << "\n";
    std::cout << "  Mean:     " << stats.mean_time_ms << "\n";
    std::cout << "  Std Dev:  " << stats.std_dev_ms << "\n\n";

    std::cout << std::setprecision(2);
    std::cout << "Performance:\n";
    std::cout << "  Achieved throughput:  " << stats.throughput_gflops << " GFLOPS\n";
    if (stats.roofline_gflops.has_value()) {
        std::cout << "  Roofline prediction:  " << *stats.roofline_gflops << " GFLOPS ("
                  << *stats.roofline_efficiency_percent << "%)\n";
    } else {
        std::cout << "  Roofline prediction:  N/A (peak FLOPs unavailable)\n";
    }
    std::cout << "  Achieved bandwidth:   " << stats.achieved_bandwidth_gbs << " GB/s";
    if (stats.bandwidth_efficiency_percent.has_value()) {
        std::cout << " (" << *stats.bandwidth_efficiency_percent << "% of peak)";
    }
    std::cout << "\n";
    std::cout << "  Arithmetic intensity: " << stats.arithmetic_intensity << " FLOPs/byte\n";
}

}  // namespace zipfft_bench

#endif  // ZIPFFT_BENCHMARK_COMMON_CUH
