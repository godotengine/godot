/*
 * Performance Benchmarking Framework Implementation
 */

#include "performance_benchmark.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/gpu_memory_stream.h"
#include "../renderer/gpu_sorter.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "../core/gaussian_data.h"
#include "servers/rendering/rendering_device.h"
#include "core/os/os.h"
#include "core/io/file_access.h"
#include "core/math/random_number_generator.h"
#include "core/math/math_funcs.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

// BenchmarkMetrics implementation
Dictionary BenchmarkMetrics::to_dict() const {
    Dictionary dict;
    dict["avg_frame_time_ms"] = avg_frame_time_ms;
    dict["min_frame_time_ms"] = min_frame_time_ms;
    dict["max_frame_time_ms"] = max_frame_time_ms;
    dict["p50_frame_time_ms"] = p50_frame_time_ms;
    dict["p95_frame_time_ms"] = p95_frame_time_ms;
    dict["p99_frame_time_ms"] = p99_frame_time_ms;
    dict["std_deviation_ms"] = std_deviation_ms;
    dict["avg_upload_time_ms"] = avg_upload_time_ms;
    dict["avg_sort_time_ms"] = avg_sort_time_ms;
    dict["avg_render_time_ms"] = avg_render_time_ms;
    dict["avg_cull_time_ms"] = avg_cull_time_ms;
    dict["avg_painterly_time_ms"] = avg_painterly_time_ms;
    dict["peak_gpu_memory_mb"] = peak_gpu_memory_mb;
    dict["avg_gpu_memory_mb"] = avg_gpu_memory_mb;
    dict["peak_cpu_memory_mb"] = peak_cpu_memory_mb;
    dict["avg_cpu_memory_mb"] = avg_cpu_memory_mb;
    dict["splat_count"] = splat_count;
    dict["splats_per_second"] = splats_per_second;
    dict["triangles_per_second"] = triangles_per_second;
    dict["bandwidth_gbps"] = bandwidth_gbps;
    dict["avg_fps"] = avg_fps;
    dict["fps_stability"] = fps_stability;
    dict["dropped_frames"] = dropped_frames;
    dict["frame_spikes"] = frame_spikes;
    return dict;
}

void BenchmarkMetrics::from_dict(const Dictionary &dict) {
    if (dict.has("avg_frame_time_ms")) avg_frame_time_ms = dict["avg_frame_time_ms"];
    if (dict.has("min_frame_time_ms")) min_frame_time_ms = dict["min_frame_time_ms"];
    if (dict.has("max_frame_time_ms")) max_frame_time_ms = dict["max_frame_time_ms"];
    if (dict.has("p50_frame_time_ms")) p50_frame_time_ms = dict["p50_frame_time_ms"];
    if (dict.has("p95_frame_time_ms")) p95_frame_time_ms = dict["p95_frame_time_ms"];
    if (dict.has("p99_frame_time_ms")) p99_frame_time_ms = dict["p99_frame_time_ms"];
    if (dict.has("std_deviation_ms")) std_deviation_ms = dict["std_deviation_ms"];
    if (dict.has("avg_upload_time_ms")) avg_upload_time_ms = dict["avg_upload_time_ms"];
    if (dict.has("avg_sort_time_ms")) avg_sort_time_ms = dict["avg_sort_time_ms"];
    if (dict.has("avg_render_time_ms")) avg_render_time_ms = dict["avg_render_time_ms"];
    if (dict.has("avg_cull_time_ms")) avg_cull_time_ms = dict["avg_cull_time_ms"];
    if (dict.has("avg_painterly_time_ms")) avg_painterly_time_ms = dict["avg_painterly_time_ms"];
    if (dict.has("peak_gpu_memory_mb")) peak_gpu_memory_mb = dict["peak_gpu_memory_mb"];
    if (dict.has("avg_gpu_memory_mb")) avg_gpu_memory_mb = dict["avg_gpu_memory_mb"];
    if (dict.has("peak_cpu_memory_mb")) peak_cpu_memory_mb = dict["peak_cpu_memory_mb"];
    if (dict.has("avg_cpu_memory_mb")) avg_cpu_memory_mb = dict["avg_cpu_memory_mb"];
    if (dict.has("splat_count")) splat_count = dict["splat_count"];
    if (dict.has("splats_per_second")) splats_per_second = dict["splats_per_second"];
    if (dict.has("triangles_per_second")) triangles_per_second = dict["triangles_per_second"];
    if (dict.has("bandwidth_gbps")) bandwidth_gbps = dict["bandwidth_gbps"];
    if (dict.has("avg_fps")) avg_fps = dict["avg_fps"];
    if (dict.has("fps_stability")) fps_stability = dict["fps_stability"];
    if (dict.has("dropped_frames")) dropped_frames = dict["dropped_frames"];
    if (dict.has("frame_spikes")) frame_spikes = dict["frame_spikes"];
}

String BenchmarkMetrics::to_json() const {
    JSON json;
    return json.stringify(to_dict(), "\t");
}

// BenchmarkConfig implementation
Dictionary BenchmarkConfig::to_dict() const {
    Dictionary dict;
    dict["splat_count"] = splat_count;
    dict["frame_count"] = frame_count;
    dict["warmup_frames"] = warmup_frames;
    dict["enable_sorting"] = enable_sorting;
    dict["enable_culling"] = enable_culling;
    dict["enable_lod"] = enable_lod;
    dict["use_async_compute"] = use_async_compute;
    dict["data_pattern"] = data_pattern;
    dict["sort_method"] = sort_method;
    dict["viewport_width"] = viewport_width;
    dict["viewport_height"] = viewport_height;
    return dict;
}

void BenchmarkConfig::from_dict(const Dictionary &dict) {
    if (dict.has("splat_count")) splat_count = dict["splat_count"];
    if (dict.has("frame_count")) frame_count = dict["frame_count"];
    if (dict.has("warmup_frames")) warmup_frames = dict["warmup_frames"];
    if (dict.has("enable_sorting")) enable_sorting = dict["enable_sorting"];
    if (dict.has("enable_culling")) enable_culling = dict["enable_culling"];
    if (dict.has("enable_lod")) enable_lod = dict["enable_lod"];
    if (dict.has("use_async_compute")) use_async_compute = dict["use_async_compute"];
    if (dict.has("data_pattern")) data_pattern = dict["data_pattern"];
    if (dict.has("sort_method")) sort_method = dict["sort_method"];
    if (dict.has("viewport_width")) viewport_width = dict["viewport_width"];
    if (dict.has("viewport_height")) viewport_height = dict["viewport_height"];
}

// BenchmarkResult implementation
Dictionary BenchmarkResult::to_dict() const {
    Dictionary dict;
    dict["name"] = name;
    dict["timestamp"] = timestamp;
    dict["git_hash"] = git_hash;
    dict["platform"] = platform;
    dict["gpu_name"] = gpu_name;
    dict["driver_version"] = driver_version;
    dict["config"] = config.to_dict();
    dict["metrics"] = metrics.to_dict();
    return dict;
}

void BenchmarkResult::from_dict(const Dictionary &dict) {
    if (dict.has("name")) name = dict["name"];
    if (dict.has("timestamp")) timestamp = dict["timestamp"];
    if (dict.has("git_hash")) git_hash = dict["git_hash"];
    if (dict.has("platform")) platform = dict["platform"];
    if (dict.has("gpu_name")) gpu_name = dict["gpu_name"];
    if (dict.has("driver_version")) driver_version = dict["driver_version"];
    if (dict.has("config")) config.from_dict(dict["config"]);
    if (dict.has("metrics")) metrics.from_dict(dict["metrics"]);
}

void BenchmarkResult::save_to_file(const String &filepath) const {
    Ref<FileAccess> file = FileAccess::open(filepath, FileAccess::WRITE);
    if (file.is_valid()) {
        JSON json;
        file->store_string(json.stringify(to_dict(), "\t"));
    }
}

BenchmarkResult BenchmarkResult::load_from_file(const String &filepath) {
    BenchmarkResult result;
    Ref<FileAccess> file = FileAccess::open(filepath, FileAccess::READ);
    if (file.is_valid()) {
        String json_string = file->get_as_text();
        JSON json;
        Error err = json.parse(json_string);
        if (err == OK) {
            result.from_dict(json.get_data());
        }
    }
    return result;
}

// PerformanceBenchmark implementation
void PerformanceBenchmark::_bind_methods() {
    // Bind methods for GDScript access if needed
}

PerformanceBenchmark::PerformanceBenchmark() {
    collect_system_info();
}

PerformanceBenchmark::~PerformanceBenchmark() {
}

Error PerformanceBenchmark::initialize(RenderingDevice *p_rd) {
    if (!p_rd) {
        return ERR_INVALID_PARAMETER;
    }
    rd = p_rd;
    return OK;
}

void PerformanceBenchmark::collect_system_info() {
    OS *os = OS::get_singleton();

    // Platform info
    platform_name = os->get_name();

    // GPU info
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs) {
        // Get GPU info from rendering server
        gpu_name = rs->get_video_adapter_name();
        driver_version = rs->get_video_adapter_api_version();
    }
}

float PerformanceBenchmark::calculate_percentile(std::vector<float> &values, float percentile) {
    if (values.empty()) return 0.0f;

    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>((percentile / 100.0f) * (values.size() - 1));
    return values[index];
}

float PerformanceBenchmark::simulate_painterly_pass(const LocalVector<Gaussian> &splats, int width, int height, bool dense_scene) {
    const int clamped_width = width > 32 ? width : 32;
    const int clamped_height = height > 32 ? height : 32;
    const int step = dense_scene ? 1 : 2;
    const int sample_splats = MIN((int)splats.size(), dense_scene ? 512 : 256);

    uint64_t start = OS::get_singleton()->get_ticks_usec();
    float accumulator = 0.0f;

    for (int y = 0; y < clamped_height; y += step) {
        float normalized_y = ((float)y / (float)clamped_height) * 2.0f - 1.0f;
        for (int x = 0; x < clamped_width; x += step) {
            float normalized_x = ((float)x / (float)clamped_width) * 2.0f - 1.0f;
            for (int i = 0; i < sample_splats; i++) {
                const Gaussian &g = splats[i];
                float radius = MAX(g.scale.x, MAX(g.scale.y, g.scale.z)) + 0.01f;
                float dx = normalized_x - g.position.x * 0.05f;
                float dy = normalized_y - g.position.y * 0.05f;
                float dist_sq = dx * dx + dy * dy;
                accumulator += Math::exp(-dist_sq / MAX(radius, 0.001f));
            }
        }
    }

    uint64_t end = OS::get_singleton()->get_ticks_usec();
    return (end - start) / 1000.0f;
}

float PerformanceBenchmark::calculate_std_deviation(const std::vector<float> &values, float mean) const {
    if (values.size() <= 1) return 0.0f;

    float sum_squared_diff = 0.0f;
    for (float value : values) {
        float diff = value - mean;
        sum_squared_diff += diff * diff;
    }

    return std::sqrt(sum_squared_diff / (values.size() - 1));
}

void PerformanceBenchmark::clear_metrics() {
    frame_times.clear();
    upload_times.clear();
    sort_times.clear();
    render_times.clear();
    cull_times.clear();
    painterly_times.clear();
    memory_samples.clear();
}

void PerformanceBenchmark::record_painterly_time(float ms) {
    painterly_times.push_back(ms);
}

BenchmarkResult PerformanceBenchmark::run_benchmark(const BenchmarkConfig &config) {
    BenchmarkResult result;
    result.name = vformat("Benchmark_%d_splats_%s", config.splat_count, config.data_pattern);
    // Get current timestamp using OS time functions
    OS::DateTime datetime = OS::get_singleton()->get_datetime();
    result.timestamp = vformat("%04d-%02d-%02d_%02d:%02d:%02d",
        datetime.year, datetime.month, datetime.day,
        datetime.hour, datetime.minute, datetime.second);
    result.platform = platform_name;
    result.gpu_name = gpu_name;
    result.driver_version = driver_version;
    result.config = config;

    clear_metrics();

    // Reserve space for metrics
    frame_times.reserve(config.frame_count);
    upload_times.reserve(config.frame_count);
    sort_times.reserve(config.frame_count);
    render_times.reserve(config.frame_count);
    cull_times.reserve(config.frame_count);

    // Create test components
    Ref<::GaussianData> data;
    data.instantiate();

    Ref<GaussianMemoryStream> memory_stream;
    memory_stream.instantiate();
    memory_stream->initialize(rd, config.splat_count, 256);

    Ref<BitonicSort> sorter;
    if (config.enable_sorting) {
        sorter.instantiate();
        sorter->initialize(rd, config.splat_count);
    }

    // Generate test data based on pattern
    LocalVector<Gaussian> test_splats;
    test_splats.resize(config.splat_count);

    RandomNumberGenerator rng;
    rng.set_seed(42); // Deterministic

    if (config.data_pattern == "uniform") {
        for (uint32_t i = 0; i < config.splat_count; i++) {
            Gaussian &g = test_splats[i];
            g.position = Vector3(
                rng.randf_range(-10.0f, 10.0f),
                rng.randf_range(-10.0f, 10.0f),
                rng.randf_range(-10.0f, 10.0f)
            );
            float scale = rng.randf_range(0.1f, 1.0f);
            g.scale = Vector3(scale, scale, scale);
            g.rotation = Quaternion();
            g.opacity = rng.randf_range(0.3f, 1.0f);
            g.sh_dc = Color(rng.randf(), rng.randf(), rng.randf(), g.opacity);
            g.normal = Vector3(0, 1, 0);
            g.area = scale * scale * static_cast<float>(Math::PI);
        }
    } else if (config.data_pattern == "clustered") {
        // Create clusters
        const int num_clusters = 10;
        LocalVector<Vector3> cluster_centers;
        for (int i = 0; i < num_clusters; i++) {
            cluster_centers.push_back(Vector3(
                rng.randf_range(-20.0f, 20.0f),
                rng.randf_range(-20.0f, 20.0f),
                rng.randf_range(-20.0f, 20.0f)
            ));
        }

        for (uint32_t i = 0; i < config.splat_count; i++) {
            Gaussian &g = test_splats[i];
            Vector3 center = cluster_centers[i % num_clusters];
            g.position = center + Vector3(
                rng.randf_range(-2.0f, 2.0f),
                rng.randf_range(-2.0f, 2.0f),
                rng.randf_range(-2.0f, 2.0f)
            );
            float scale = rng.randf_range(0.05f, 0.5f);
            g.scale = Vector3(scale, scale, scale);
            g.rotation = Quaternion();
            g.opacity = rng.randf_range(0.5f, 1.0f);
            g.sh_dc = Color(rng.randf(), rng.randf(), rng.randf(), g.opacity);
            g.normal = Vector3(0, 1, 0);
            g.area = scale * scale * static_cast<float>(Math::PI);
        }
    } else if (config.data_pattern == "worst_case") {
        // All at same depth - worst case for sorting
        for (uint32_t i = 0; i < config.splat_count; i++) {
            Gaussian &g = test_splats[i];
            g.position = Vector3(
                rng.randf_range(-10.0f, 10.0f),
                rng.randf_range(-10.0f, 10.0f),
                5.0f // Same Z depth
            );
            float scale = rng.randf_range(0.1f, 0.5f);
            g.scale = Vector3(scale, scale, scale);
            g.rotation = Quaternion();
            g.opacity = rng.randf_range(0.5f, 1.0f);
            g.sh_dc = Color(rng.randf(), rng.randf(), rng.randf(), g.opacity);
            g.normal = Vector3(0, 0, -1);
            g.area = scale * scale * static_cast<float>(Math::PI);
        }
    }

    data->set_gaussians(test_splats);

    // Warmup frames
    for (uint32_t i = 0; i < config.warmup_frames; i++) {
        memory_stream->begin_frame(i);
        memory_stream->stream_gaussians_immediate(test_splats);
        memory_stream->end_frame();

        if (config.enable_sorting && sorter.is_valid()) {
            RID keys_buffer = memory_stream->get_current_gpu_buffer();
            RID values_buffer = memory_stream->get_sort_keys_buffer();
            sorter->sort(keys_buffer, values_buffer, test_splats.size());
        }
    }

    // Benchmark frames
    uint64_t total_start = OS::get_singleton()->get_ticks_usec();

    for (uint32_t frame = 0; frame < config.frame_count; frame++) {
        uint64_t frame_start = OS::get_singleton()->get_ticks_usec();

        // Upload phase
        uint64_t upload_start = OS::get_singleton()->get_ticks_usec();
        memory_stream->begin_frame(config.warmup_frames + frame);
        memory_stream->stream_gaussians_immediate(test_splats);
        memory_stream->end_frame();
        uint64_t upload_end = OS::get_singleton()->get_ticks_usec();
        upload_times.push_back((upload_end - upload_start) / 1000.0f);

        // Sort phase
        if (config.enable_sorting && sorter.is_valid()) {
            uint64_t sort_start = OS::get_singleton()->get_ticks_usec();
            RID keys_buffer = memory_stream->get_current_gpu_buffer();
            RID values_buffer = memory_stream->get_sort_keys_buffer();
            sorter->sort(keys_buffer, values_buffer, test_splats.size());
            uint64_t sort_end = OS::get_singleton()->get_ticks_usec();
            sort_times.push_back((sort_end - sort_start) / 1000.0f);
        } else {
            sort_times.push_back(0.0f);
        }

        // Culling phase (simulated for now)
        if (config.enable_culling) {
            uint64_t cull_start = OS::get_singleton()->get_ticks_usec();
            // Culling would happen here
            uint64_t cull_end = OS::get_singleton()->get_ticks_usec();
            cull_times.push_back((cull_end - cull_start) / 1000.0f);
        } else {
            cull_times.push_back(0.0f);
        }

        // Render phase (simulated for now)
        uint64_t render_start = OS::get_singleton()->get_ticks_usec();
        // Rendering would happen here
        uint64_t render_end = OS::get_singleton()->get_ticks_usec();
        render_times.push_back((render_end - render_start) / 1000.0f);

        bool dense_scene = config.data_pattern != "worst_case";
        int painterly_width = MAX(64, (int)config.viewport_width / 8);
        int painterly_height = MAX(64, (int)config.viewport_height / 8);
        float painterly_time = simulate_painterly_pass(test_splats, painterly_width, painterly_height, dense_scene);
        record_painterly_time(painterly_time);

        // Total frame time
        uint64_t frame_end = OS::get_singleton()->get_ticks_usec();
        frame_times.push_back((frame_end - frame_start) / 1000.0f);

        // Sample memory usage
        if (rd) {
            float gpu_memory_mb = rd->get_memory_usage(RenderingDevice::MEMORY_TOTAL) / (1024.0f * 1024.0f);
            memory_samples.push_back(gpu_memory_mb);
        }
    }

    uint64_t total_end = OS::get_singleton()->get_ticks_usec();
    float total_time_ms = (total_end - total_start) / 1000.0f;

    // Calculate metrics
    result.metrics = calculate_metrics();
    result.metrics.splat_count = config.splat_count;

    // Calculate throughput
    float total_frames = static_cast<float>(config.frame_count);
    result.metrics.splats_per_second = (config.splat_count * total_frames) / (total_time_ms / 1000.0f);
    result.metrics.bandwidth_gbps = (sizeof(PackedGaussian) * config.splat_count * total_frames) /
                                    (total_time_ms / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);

    return result;
}

BenchmarkMetrics PerformanceBenchmark::calculate_metrics() const {
    BenchmarkMetrics metrics;

    if (frame_times.empty()) {
        return metrics;
    }

    // Frame time statistics
    metrics.avg_frame_time_ms = std::accumulate(frame_times.begin(), frame_times.end(), 0.0f) / frame_times.size();
    metrics.min_frame_time_ms = *std::min_element(frame_times.begin(), frame_times.end());
    metrics.max_frame_time_ms = *std::max_element(frame_times.begin(), frame_times.end());

    // Percentiles
    std::vector<float> sorted_times = frame_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    metrics.p50_frame_time_ms = sorted_times[sorted_times.size() / 2];
    metrics.p95_frame_time_ms = sorted_times[static_cast<size_t>(sorted_times.size() * 0.95)];
    metrics.p99_frame_time_ms = sorted_times[static_cast<size_t>(sorted_times.size() * 0.99)];

    // Standard deviation
    metrics.std_deviation_ms = calculate_std_deviation(frame_times, metrics.avg_frame_time_ms);

    // Component timings
    if (!upload_times.empty()) {
        metrics.avg_upload_time_ms = std::accumulate(upload_times.begin(), upload_times.end(), 0.0f) / upload_times.size();
    }
    if (!sort_times.empty()) {
        metrics.avg_sort_time_ms = std::accumulate(sort_times.begin(), sort_times.end(), 0.0f) / sort_times.size();
    }
    if (!render_times.empty()) {
        metrics.avg_render_time_ms = std::accumulate(render_times.begin(), render_times.end(), 0.0f) / render_times.size();
    }
    if (!cull_times.empty()) {
        metrics.avg_cull_time_ms = std::accumulate(cull_times.begin(), cull_times.end(), 0.0f) / cull_times.size();
    }
    if (!painterly_times.empty()) {
        metrics.avg_painterly_time_ms = std::accumulate(painterly_times.begin(), painterly_times.end(), 0.0f) / painterly_times.size();
    }

    // Memory metrics
    if (!memory_samples.empty()) {
        metrics.avg_gpu_memory_mb = std::accumulate(memory_samples.begin(), memory_samples.end(), 0.0f) / memory_samples.size();
        metrics.peak_gpu_memory_mb = *std::max_element(memory_samples.begin(), memory_samples.end());
    }

    // FPS and stability
    metrics.avg_fps = 1000.0f / metrics.avg_frame_time_ms;
    metrics.fps_stability = 1.0f - (metrics.std_deviation_ms / metrics.avg_frame_time_ms);
    metrics.fps_stability = CLAMP(metrics.fps_stability, 0.0f, 1.0f);

    // Count dropped frames and spikes
    const float target_frame_time = 16.67f; // 60 FPS
    for (float frame_time : frame_times) {
        if (frame_time > target_frame_time) {
            metrics.dropped_frames++;
        }
        if (frame_time > metrics.avg_frame_time_ms * 2.0f) {
            metrics.frame_spikes++;
        }
    }

    return metrics;
}

void PerformanceBenchmark::generate_report(const String &output_file) const {
    std::stringstream report;

    report << "=== Gaussian Splatting Performance Report ===\n\n";
    report << "System Information:\n";
    report << "  Platform: " << platform_name.utf8().get_data() << "\n";
    report << "  GPU: " << gpu_name.utf8().get_data() << "\n";
    report << "  Driver: " << driver_version.utf8().get_data() << "\n\n";

    report << "Benchmark Results:\n";
    report << "  Frame Count: " << frame_times.size() << "\n";

    if (!frame_times.empty()) {
        BenchmarkMetrics metrics = calculate_metrics();

        report << "\nFrame Time Statistics:\n";
        report << std::fixed << std::setprecision(2);
        report << "  Average: " << metrics.avg_frame_time_ms << " ms (" << metrics.avg_fps << " FPS)\n";
        report << "  Min: " << metrics.min_frame_time_ms << " ms\n";
        report << "  Max: " << metrics.max_frame_time_ms << " ms\n";
        report << "  P50: " << metrics.p50_frame_time_ms << " ms\n";
        report << "  P95: " << metrics.p95_frame_time_ms << " ms\n";
        report << "  P99: " << metrics.p99_frame_time_ms << " ms\n";
        report << "  Std Dev: " << metrics.std_deviation_ms << " ms\n";
        report << "  FPS Stability: " << (metrics.fps_stability * 100.0f) << "%\n";
        report << "  Dropped Frames: " << metrics.dropped_frames << "\n";
        report << "  Frame Spikes: " << metrics.frame_spikes << "\n\n";

        report << "Component Timings:\n";
        report << "  Upload: " << metrics.avg_upload_time_ms << " ms\n";
        report << "  Sort: " << metrics.avg_sort_time_ms << " ms\n";
        report << "  Cull: " << metrics.avg_cull_time_ms << " ms\n";
        report << "  Render: " << metrics.avg_render_time_ms << " ms\n";
        report << "  Painterly Resolve: " << metrics.avg_painterly_time_ms << " ms\n\n";

        report << "Memory Usage:\n";
        report << "  Average GPU: " << metrics.avg_gpu_memory_mb << " MB\n";
        report << "  Peak GPU: " << metrics.peak_gpu_memory_mb << " MB\n";
    }

    // Write to file
    Ref<FileAccess> file = FileAccess::open(output_file, FileAccess::WRITE);
    if (file.is_valid()) {
        file->store_string(String(report.str().c_str()));
        print_line(vformat("Performance report saved to: %s", output_file));
    }
}

bool PerformanceBenchmark::check_regression(const BenchmarkResult &baseline,
                                           const BenchmarkResult &current,
                                           float threshold) {
    // Check if performance has regressed by more than threshold
    float baseline_fps = baseline.metrics.avg_fps;
    float current_fps = current.metrics.avg_fps;

    float regression = (baseline_fps - current_fps) / baseline_fps;
    return regression > threshold;
}

String PerformanceBenchmark::generate_regression_report(const BenchmarkResult &baseline,
                                                       const BenchmarkResult &current) const {
    std::stringstream report;

    report << "=== Performance Regression Report ===\n\n";

    // Calculate differences
    float fps_diff = current.metrics.avg_fps - baseline.metrics.avg_fps;
    float fps_pct = (fps_diff / baseline.metrics.avg_fps) * 100.0f;

    float frame_time_diff = current.metrics.avg_frame_time_ms - baseline.metrics.avg_frame_time_ms;
    float frame_time_pct = (frame_time_diff / baseline.metrics.avg_frame_time_ms) * 100.0f;

    report << "Configuration: " << current.config.splat_count << " splats\n";
    report << "Pattern: " << current.config.data_pattern.utf8().get_data() << "\n\n";

    report << "Performance Changes:\n";
    report << std::fixed << std::setprecision(2);

    // FPS comparison
    report << "  FPS: " << baseline.metrics.avg_fps << " -> " << current.metrics.avg_fps;
    report << " (" << (fps_pct >= 0 ? "+" : "") << fps_pct << "%)\n";

    // Frame time comparison
    report << "  Frame Time: " << baseline.metrics.avg_frame_time_ms << " ms -> ";
    report << current.metrics.avg_frame_time_ms << " ms";
    report << " (" << (frame_time_pct >= 0 ? "+" : "") << frame_time_pct << "%)\n";

    // Component timings
    float upload_diff = current.metrics.avg_upload_time_ms - baseline.metrics.avg_upload_time_ms;
    float sort_diff = current.metrics.avg_sort_time_ms - baseline.metrics.avg_sort_time_ms;

    report << "\nComponent Changes:\n";
    report << "  Upload: " << (upload_diff >= 0 ? "+" : "") << upload_diff << " ms\n";
    report << "  Sort: " << (sort_diff >= 0 ? "+" : "") << sort_diff << " ms\n";

    // Memory changes
    float memory_diff = current.metrics.peak_gpu_memory_mb - baseline.metrics.peak_gpu_memory_mb;
    report << "\nMemory Changes:\n";
    report << "  Peak GPU Memory: " << (memory_diff >= 0 ? "+" : "") << memory_diff << " MB\n";

    // Overall assessment
    report << "\nAssessment: ";
    if (fps_pct < -10.0f) {
        report << "SIGNIFICANT REGRESSION\n";
    } else if (fps_pct < -5.0f) {
        report << "Minor regression\n";
    } else if (fps_pct > 10.0f) {
        report << "SIGNIFICANT IMPROVEMENT\n";
    } else if (fps_pct > 5.0f) {
        report << "Minor improvement\n";
    } else {
        report << "No significant change\n";
    }

    return String(report.str().c_str());
}

// BenchmarkRunner implementation
BenchmarkRunner::BenchmarkRunner() {
    benchmark.instantiate();
}

void BenchmarkRunner::add_standard_configs() {
    // Standard test configurations
    BenchmarkConfig config;

    // Small dataset
    config.splat_count = 10000;
    config.frame_count = 100;
    config.data_pattern = "uniform";
    test_configs.push_back(config);

    // Medium dataset
    config.splat_count = 100000;
    config.frame_count = 100;
    config.data_pattern = "uniform";
    test_configs.push_back(config);

    // Large dataset
    config.splat_count = 500000;
    config.frame_count = 50;
    config.data_pattern = "uniform";
    test_configs.push_back(config);

    // Clustered data
    config.splat_count = 100000;
    config.frame_count = 100;
    config.data_pattern = "clustered";
    test_configs.push_back(config);

    // Worst case sorting
    config.splat_count = 100000;
    config.frame_count = 50;
    config.data_pattern = "worst_case";
    test_configs.push_back(config);
}

void BenchmarkRunner::run_all() {
    print_line(vformat("Running %d benchmark configurations...", test_configs.size()));

    for (const BenchmarkConfig &config : test_configs) {
        print_line(vformat("Running benchmark: %d splats, %s pattern",
                          config.splat_count, config.data_pattern));

        BenchmarkResult result = benchmark->run_benchmark(config);
        results.push_back(result);

        print_line(vformat("  Result: %.1f FPS (%.2f ms)",
                          result.metrics.avg_fps,
                          result.metrics.avg_frame_time_ms));
    }
}

void BenchmarkRunner::generate_summary_report(const String &output_file) {
    std::stringstream report;

    report << "=== Benchmark Summary Report ===\n\n";
    report << "Total Benchmarks: " << results.size() << "\n\n";

    report << std::left << std::setw(20) << "Configuration";
    report << std::right << std::setw(10) << "Splats";
    report << std::setw(15) << "Avg FPS";
    report << std::setw(15) << "Frame Time";
    report << std::setw(15) << "GPU Memory\n";
    report << std::string(75, '-') << "\n";

    for (const BenchmarkResult &result : results) {
        report << std::left << std::setw(20) << result.config.data_pattern.utf8().get_data();
        report << std::right << std::setw(10) << result.config.splat_count;
        report << std::setw(15) << std::fixed << std::setprecision(1) << result.metrics.avg_fps;
        report << std::setw(15) << std::fixed << std::setprecision(2) << result.metrics.avg_frame_time_ms;
        report << std::setw(15) << std::fixed << std::setprecision(1) << result.metrics.peak_gpu_memory_mb;
        report << "\n";
    }

    // Write to file
    Ref<FileAccess> file = FileAccess::open(output_file, FileAccess::WRITE);
    if (file.is_valid()) {
        file->store_string(String(report.str().c_str()));
        print_line(vformat("Summary report saved to: %s", output_file));
    }
}

bool BenchmarkRunner::check_performance_gates() {
    // Check if all benchmarks meet performance requirements
    bool all_passed = true;

    for (const BenchmarkResult &result : results) {
        // Check 60 FPS target for 100K splats
        if (result.config.splat_count <= 100000) {
            if (result.metrics.avg_fps < 60.0f) {
                print_line(vformat("FAILED: %d splats only achieved %.1f FPS (target: 60 FPS)",
                                  result.config.splat_count, result.metrics.avg_fps));
                all_passed = false;
            }
        }

        // Check memory budget
        if (result.metrics.peak_gpu_memory_mb > 500.0f) {
            print_line(vformat("FAILED: Peak memory %.1f MB exceeds 500 MB budget",
                              result.metrics.peak_gpu_memory_mb));
            all_passed = false;
        }
    }

    return all_passed;
}

// PerformanceProfiler implementation
PerformanceProfiler::ScopedTimer::ScopedTimer(PerformanceProfiler *p_profiler, const String &p_name)
    : profiler(p_profiler), name(p_name) {
    if (profiler && profiler->enabled) {
        start_time = OS::get_singleton()->get_ticks_usec();
        profiler->begin_block(name);
    }
}

PerformanceProfiler::ScopedTimer::~ScopedTimer() {
    if (profiler && profiler->enabled) {
        profiler->end_block();
    }
}

void PerformanceProfiler::begin_block(const String &name) {
    if (!enabled) return;

    ProfileBlock block;
    block.name = name;
    block.start_time = OS::get_singleton()->get_ticks_usec();
    block.depth = current_depth++;
    blocks.push_back(block);
}

void PerformanceProfiler::end_block() {
    if (!enabled || blocks.is_empty()) return;

    blocks[blocks.size() - 1].end_time = OS::get_singleton()->get_ticks_usec();
    current_depth--;
}

void PerformanceProfiler::clear() {
    blocks.clear();
    current_depth = 0;
}

Dictionary PerformanceProfiler::get_summary() const {
    Dictionary summary;

    std::map<String, float> total_times;
    std::map<String, uint32_t> call_counts;

    for (const ProfileBlock &block : blocks) {
        float duration_ms = (block.end_time - block.start_time) / 1000.0f;
        total_times[block.name] += duration_ms;
        call_counts[block.name]++;
    }

    for (const auto &pair : total_times) {
        Dictionary block_info;
        block_info["total_ms"] = pair.second;
        block_info["calls"] = call_counts[pair.first];
        block_info["avg_ms"] = pair.second / call_counts[pair.first];
        summary[pair.first] = block_info;
    }

    return summary;
}