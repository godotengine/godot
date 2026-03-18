/*
 * Performance Benchmarking Framework for Gaussian Splatting
 * Provides automated performance tracking and regression detection.
 */

#ifndef PERFORMANCE_BENCHMARK_H
#define PERFORMANCE_BENCHMARK_H

#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "core/io/json.h"
#include "core/templates/local_vector.h"
#include "../core/gaussian_data.h"
#include <vector>
#include <map>
#include <chrono>

class RenderingDevice;

// Performance metrics for a single benchmark run
struct BenchmarkMetrics {
    // Timing metrics (in milliseconds)
    float avg_frame_time_ms = 0.0f;
    float min_frame_time_ms = 0.0f;
    float max_frame_time_ms = 0.0f;
    float p50_frame_time_ms = 0.0f;  // Median
    float p95_frame_time_ms = 0.0f;  // 95th percentile
    float p99_frame_time_ms = 0.0f;  // 99th percentile
    float std_deviation_ms = 0.0f;

    // Component timings
    float avg_upload_time_ms = 0.0f;
    float avg_sort_time_ms = 0.0f;
    float avg_render_time_ms = 0.0f;
    float avg_cull_time_ms = 0.0f;
    float avg_painterly_time_ms = 0.0f;

    // Memory metrics
    float peak_gpu_memory_mb = 0.0f;
    float avg_gpu_memory_mb = 0.0f;
    float peak_cpu_memory_mb = 0.0f;
    float avg_cpu_memory_mb = 0.0f;

    // Throughput metrics
    uint32_t splat_count = 0;
    float splats_per_second = 0.0f;
    float triangles_per_second = 0.0f;  // For comparison
    float bandwidth_gbps = 0.0f;        // GPU bandwidth utilization

    // Quality metrics
    float avg_fps = 0.0f;
    float fps_stability = 0.0f;  // 1.0 = perfectly stable
    uint32_t dropped_frames = 0;
    uint32_t frame_spikes = 0;   // Frames > 2x average time

    Dictionary to_dict() const;
    void from_dict(const Dictionary &dict);
    String to_json() const;
};

// Configuration for a benchmark run
struct BenchmarkConfig {
    uint32_t splat_count = 100000;
    uint32_t frame_count = 100;
    uint32_t warmup_frames = 10;
    bool enable_sorting = true;
    bool enable_culling = true;
    bool enable_lod = false;
    bool use_async_compute = true;
    String data_pattern = "uniform";  // uniform, clustered, worst_case
    String sort_method = "radix";   // radix only
    uint32_t viewport_width = 1920;
    uint32_t viewport_height = 1080;

    Dictionary to_dict() const;
    void from_dict(const Dictionary &dict);
};

// Benchmark result including config and metrics
struct BenchmarkResult {
    String name;
    String timestamp;
    String git_hash;
    String platform;
    String gpu_name;
    String driver_version;
    BenchmarkConfig config;
    BenchmarkMetrics metrics;

    Dictionary to_dict() const;
    void from_dict(const Dictionary &dict);
    void save_to_file(const String &filepath) const;
    static BenchmarkResult load_from_file(const String &filepath);
};

// Main performance benchmarking class
class PerformanceBenchmark : public RefCounted {
    GDCLASS(PerformanceBenchmark, RefCounted);

protected:
    static void _bind_methods();

private:
    RenderingDevice *rd = nullptr;

    // Timing storage for statistical analysis
    std::vector<float> frame_times;
    std::vector<float> upload_times;
    std::vector<float> sort_times;
    std::vector<float> render_times;
    std::vector<float> cull_times;
    std::vector<float> painterly_times;
    std::vector<float> memory_samples;

    // System info
    String platform_name;
    String gpu_name;
    String driver_version;

    // Helper methods
    void collect_system_info();
    float calculate_percentile(std::vector<float> &values, float percentile);
    float calculate_std_deviation(const std::vector<float> &values, float mean) const;
    void clear_metrics();

public:
    PerformanceBenchmark();
    ~PerformanceBenchmark();

    // Initialize with rendering device
    Error initialize(RenderingDevice *p_rd);

    // Run a single benchmark
    BenchmarkResult run_benchmark(const BenchmarkConfig &config);

    // Run standard benchmark suite
    void run_benchmark_suite();

    // Specific benchmark scenarios
    BenchmarkResult benchmark_scaling(uint32_t min_splats, uint32_t max_splats, uint32_t steps);
    BenchmarkResult benchmark_memory_pressure();
    BenchmarkResult benchmark_async_overlap();
    BenchmarkResult benchmark_worst_case_sorting();
    BenchmarkResult benchmark_culling_efficiency();
    BenchmarkResult benchmark_lod_transitions();

    // Profiling helpers
    void start_frame_timing();
    void end_frame_timing();
    void record_upload_time(float ms);
    void record_sort_time(float ms);
    void record_render_time(float ms);
    void record_cull_time(float ms);
    void record_painterly_time(float ms);
    void sample_memory_usage();

    // Analysis and reporting
    BenchmarkMetrics calculate_metrics() const;
    void generate_report(const String &output_file) const;
    void generate_csv_report(const String &output_file) const;
    void generate_html_report(const String &output_file) const;

    // Regression detection
    bool check_regression(const BenchmarkResult &baseline, const BenchmarkResult &current, float threshold = 0.1f);
    String generate_regression_report(const BenchmarkResult &baseline, const BenchmarkResult &current) const;

    // Baseline management
    void save_baseline(const BenchmarkResult &result, const String &name);
    BenchmarkResult load_baseline(const String &name) const;
    bool has_baseline(const String &name) const;

    // Continuous benchmarking
    void run_continuous_benchmark(uint32_t duration_seconds);
    void run_stress_test(uint32_t duration_seconds);

    // Platform-specific optimizations testing
    void benchmark_platform_features();
    void benchmark_gpu_vendor_optimizations();

    float simulate_painterly_pass(const LocalVector<Gaussian> &splats, int width, int height, bool dense_scene);

    // Getters
    String get_platform_name() const { return platform_name; }
    String get_gpu_name() const { return gpu_name; }
    String get_driver_version() const { return driver_version; }
};

// Automated benchmark runner
class BenchmarkRunner {
private:
    Ref<PerformanceBenchmark> benchmark;
    LocalVector<BenchmarkConfig> test_configs;
    LocalVector<BenchmarkResult> results;

public:
    BenchmarkRunner();

    // Setup test configurations
    void add_test_config(const BenchmarkConfig &config);
    void add_standard_configs();
    void add_scaling_configs(uint32_t min_splats, uint32_t max_splats);
    void add_stress_configs();

    // Run benchmarks
    void run_all();
    void run_with_config(const BenchmarkConfig &config);

    // Analysis
    void analyze_results();
    void detect_regressions(const String &baseline_dir);
    void generate_summary_report(const String &output_file);

    // Export results
    void export_json(const String &filepath) const;
    void export_csv(const String &filepath) const;
    void export_html(const String &filepath) const;

    // CI/CD integration
    bool check_performance_gates();
    String generate_github_comment() const;
    void upload_to_dashboard(const String &url) const;
};

// Performance profiler for detailed analysis
class PerformanceProfiler {
private:
    struct ProfileBlock {
        String name;
        uint64_t start_time;
        uint64_t end_time;
        uint32_t depth;
    };

    LocalVector<ProfileBlock> blocks;
    uint32_t current_depth = 0;
    bool enabled = true;

public:
    class ScopedTimer {
    private:
        PerformanceProfiler *profiler;
        String name;
        uint64_t start_time;

    public:
        ScopedTimer(PerformanceProfiler *p_profiler, const String &p_name);
        ~ScopedTimer();
    };

    void begin_block(const String &name);
    void end_block();
    void clear();
    void set_enabled(bool p_enabled) { enabled = p_enabled; }

    void generate_flame_graph(const String &output_file) const;
    void generate_timeline(const String &output_file) const;
    Dictionary get_summary() const;
};

// Macros for easy profiling
#define BENCHMARK_SCOPE(profiler, name) \
    PerformanceProfiler::ScopedTimer _timer(profiler, name)

#define BENCHMARK_START(name) \
    uint64_t _bench_start_##name = OS::get_singleton()->get_ticks_usec()

#define BENCHMARK_END(name) \
    uint64_t _bench_end_##name = OS::get_singleton()->get_ticks_usec(); \
    print_line(vformat("[Benchmark] %s: %.2f ms", #name, (_bench_end_##name - _bench_start_##name) / 1000.0f))

#endif // PERFORMANCE_BENCHMARK_H
