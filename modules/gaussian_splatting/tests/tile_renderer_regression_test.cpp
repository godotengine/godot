#include "core/math/vector2i.h"
#include "core/math/vector3.h"
#include "core/object/ref_counted.h"
#include "core/templates/vector.h"
#include "core/os/os.h"
#include "servers/rendering/rendering_device.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <utility>

#include "../renderer/tile_renderer.h"
#include "../core/gaussian_data.h"

namespace {

static TileRenderer::RenderParams make_render_params(RID p_gaussian_buffer, RID p_sorted_indices, uint32_t p_splat_count,
        int p_viewport_width, int p_viewport_height, int p_tile_size) {
    TileRenderer::RenderParams params;
    params.gaussian_buffer = p_gaussian_buffer;
    params.sorted_indices = p_sorted_indices;
    params.splat_count = p_splat_count;
    params.total_gaussians = p_splat_count;
    params.viewport_size = Vector2i(p_viewport_width, p_viewport_height);
    params.world_to_camera_transform = Transform3D();
    const int safe_height = (p_viewport_height > 0) ? p_viewport_height : 1;
    params.projection.set_perspective(60.0f, float(p_viewport_width) / float(safe_height), 0.1f, 100.0f);
    params.render_projection = params.projection;
    params.tile_size = p_tile_size;
    params.debug_show_performance_hud = true; // Enables async tile stats collection in test mode.
    return params;
}

static bool read_texture_pixels(RenderingDevice *p_rd, RID p_texture, Vector<uint8_t> &r_pixels) {
    r_pixels.clear();
    if (p_rd == nullptr || !p_texture.is_valid()) {
        return false;
    }
    r_pixels = p_rd->texture_get_data(p_texture, 0);
    return !r_pixels.is_empty() && (r_pixels.size() % 4) == 0;
}

struct TextureMetrics {
    float average_luma = 0.0f;
    float average_alpha = 0.0f;
    uint32_t non_zero_pixels = 0;
};

static TextureMetrics compute_texture_metrics(const Vector<uint8_t> &p_pixels) {
    TextureMetrics metrics;
    if (p_pixels.is_empty() || (p_pixels.size() % 4) != 0) {
        return metrics;
    }

    const uint8_t *read = p_pixels.ptr();
    const int pixel_count = p_pixels.size() / 4;
    double luma_sum = 0.0;
    double alpha_sum = 0.0;

    for (int i = 0; i < pixel_count; i++) {
        const uint8_t r = read[i * 4 + 0];
        const uint8_t g = read[i * 4 + 1];
        const uint8_t b = read[i * 4 + 2];
        const uint8_t a = read[i * 4 + 3];
        const uint8_t intensity = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
        if (intensity > 0 || a > 0) {
            metrics.non_zero_pixels++;
        }

        luma_sum += (0.2126 * double(r) + 0.7152 * double(g) + 0.0722 * double(b)) / 255.0;
        alpha_sum += double(a) / 255.0;
    }

    metrics.average_luma = float(luma_sum / double(pixel_count));
    metrics.average_alpha = float(alpha_sum / double(pixel_count));
    return metrics;
}

} // namespace

#ifndef TILE_RENDERER_REGRESSION_TEST_H
#define TILE_RENDERER_REGRESSION_TEST_H

/**
 * Tile Renderer Regression Test Suite
 *
 * Tests for Issue #127 - Tile Rasterization Fortification
 * Validates overflow protection, error detection, and dense scene rendering
 */
class TileRendererRegressionTest : public RefCounted {
    GDCLASS(TileRendererRegressionTest, RefCounted);

public:
    struct TestResult {
        bool passed = false;
        String error_message;
        float execution_time_ms = 0.0f;
        TileRenderer::RenderStats stats;
    };

    TileRendererRegressionTest();
    ~TileRendererRegressionTest();

    // Main test suite entry point
    bool run_all_tests(RenderingDevice *p_rd);

    // Individual test cases
    TestResult test_overflow_protection(RenderingDevice *p_rd);
    TestResult test_dense_scene_rendering(RenderingDevice *p_rd);
    TestResult test_validation_and_error_detection(RenderingDevice *p_rd);
    TestResult test_compute_fragment_clamp_parity(RenderingDevice *p_rd);
    TestResult test_distance_cull_sort_order_stability(RenderingDevice *p_rd);
    TestResult test_compute_format_fallback(RenderingDevice *p_rd);
    TestResult test_alpha_compositing_accuracy(RenderingDevice *p_rd);
    TestResult test_performance_regression(RenderingDevice *p_rd);
    TestResult test_renderer_lifecycle_leak_detection(RenderingDevice *p_rd);
    TestResult test_zero_work_frame_resets_raster_timing(RenderingDevice *p_rd);

    // Test utilities
    Vector<Gaussian> generate_test_gaussians(uint32_t count, bool valid = true);
    RID create_test_gaussian_buffer(RenderingDevice *p_rd, const Vector<Gaussian> &gaussians);
    RID create_test_sorted_indices(RenderingDevice *p_rd, uint32_t count);
    bool compare_render_output(RID texture1, RID texture2, float tolerance = 0.01f);

    // Reference data generation
    bool generate_reference_captures(RenderingDevice *p_rd);
    bool validate_against_reference(RID output_texture, const String &reference_name);

private:
    Ref<TileRenderer> tile_renderer;
    Vector<TestResult> test_results;

    // Test configuration
    static constexpr int TEST_VIEWPORT_WIDTH = 512;
    static constexpr int TEST_VIEWPORT_HEIGHT = 512;
    static constexpr int TEST_TILE_SIZE = 16;
    static constexpr uint32_t DENSE_SCENE_SPLAT_COUNT = 50000;
    static constexpr uint32_t OVERFLOW_TEST_SPLAT_COUNT = 100000;

    void _log_test_result(const String &test_name, const TestResult &result);
    Gaussian _create_test_gaussian(const Vector3 &position, const Vector3 &scale = Vector3(1, 1, 1), float opacity = 1.0f);
    bool _validate_tile_overflow_handling(const TileRenderer::RenderStats &stats);
};

#endif // TILE_RENDERER_REGRESSION_TEST_H

// Implementation

TileRendererRegressionTest::TileRendererRegressionTest() {
    tile_renderer.instantiate();
}

TileRendererRegressionTest::~TileRendererRegressionTest() {
    if (tile_renderer.is_valid()) {
        tile_renderer->cleanup();
    }
}

bool TileRendererRegressionTest::run_all_tests(RenderingDevice *p_rd) {
    ERR_FAIL_NULL_V(p_rd, false);

    print_line("[TileRendererRegressionTest] Starting tile rasterization regression tests...");

    test_results.clear();
    bool all_passed = true;

    // Initialize tile renderer
    Error err = tile_renderer->initialize(p_rd, Vector2i(TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT), TEST_TILE_SIZE);
    if (err != OK) {
        ERR_PRINT("[TileRendererRegressionTest] Failed to initialize tile renderer");
        return false;
    }

    // Run individual test cases
    Vector<std::pair<String, std::function<TestResult()>>> tests = {
        {"overflow_protection", [this, p_rd]() { return test_overflow_protection(p_rd); }},
        {"dense_scene_rendering", [this, p_rd]() { return test_dense_scene_rendering(p_rd); }},
        {"validation_and_error_detection", [this, p_rd]() { return test_validation_and_error_detection(p_rd); }},
        {"compute_fragment_clamp_parity", [this, p_rd]() { return test_compute_fragment_clamp_parity(p_rd); }},
        {"distance_cull_sort_order_stability", [this, p_rd]() { return test_distance_cull_sort_order_stability(p_rd); }},
        {"compute_format_fallback", [this, p_rd]() { return test_compute_format_fallback(p_rd); }},
        {"alpha_compositing_accuracy", [this, p_rd]() { return test_alpha_compositing_accuracy(p_rd); }},
        {"performance_regression", [this, p_rd]() { return test_performance_regression(p_rd); }},
        {"renderer_lifecycle_leak_detection", [this, p_rd]() { return test_renderer_lifecycle_leak_detection(p_rd); }},
        {"zero_work_frame_resets_raster_timing", [this, p_rd]() { return test_zero_work_frame_resets_raster_timing(p_rd); }}
    };

    for (const auto &test : tests) {
        uint64_t start_time = OS::get_singleton()->get_ticks_usec();
        TestResult result = test.second();
        result.execution_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;

        test_results.push_back(result);
        _log_test_result(test.first, result);

        if (!result.passed) {
            all_passed = false;
        }
    }

    print_line(vformat("[TileRendererRegressionTest] Test suite completed. %s",
               all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED"));

    return all_passed;
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_overflow_protection(RenderingDevice *p_rd) {
    TestResult result;

    tile_renderer->set_debug_binning_counters_enabled(true);

    Vector<Gaussian> gaussians = generate_test_gaussians(OVERFLOW_TEST_SPLAT_COUNT);
    RID gaussian_buffer = create_test_gaussian_buffer(p_rd, gaussians);
    RID sorted_indices = create_test_sorted_indices(p_rd, OVERFLOW_TEST_SPLAT_COUNT);
    auto cleanup = [&]() {
        if (gaussian_buffer.is_valid()) {
            p_rd->free(gaussian_buffer);
            gaussian_buffer = RID();
        }
        if (sorted_indices.is_valid()) {
            p_rd->free(sorted_indices);
            sorted_indices = RID();
        }
    };

    if (!gaussian_buffer.is_valid() || !sorted_indices.is_valid()) {
        cleanup();
        result.error_message = "Failed to create overflow test buffers";
        return result;
    }

    TileRenderer::RenderParams params = make_render_params(gaussian_buffer, sorted_indices, OVERFLOW_TEST_SPLAT_COUNT,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);

    TileRenderer::RenderParams baseline_params = params;
    baseline_params.splat_count = MIN<uint32_t>(2048u, OVERFLOW_TEST_SPLAT_COUNT);
    uint32_t baseline_overlap_records = 0;
    for (int frame = 0; frame < 6; frame++) {
        RID baseline_output = tile_renderer->render(p_rd, baseline_params);
        if (!baseline_output.is_valid()) {
            cleanup();
            result.error_message = "Baseline render failed before overflow workload spike";
            return result;
        }
        baseline_overlap_records = MAX<uint32_t>(baseline_overlap_records, tile_renderer->get_last_render_stats().overlap_records);
    }

    TileRenderer::OverflowStatsSnapshot overflow_stats;
    bool counters_ready = false;
    uint32_t peak_overlap_records = 0;
    for (int frame = 0; frame < 10; frame++) {
        RID output = tile_renderer->render(p_rd, params);
        if (!output.is_valid()) {
            cleanup();
            result.error_message = "Render failed under overflow workload";
            return result;
        }

        peak_overlap_records = MAX<uint32_t>(peak_overlap_records, tile_renderer->get_last_render_stats().overlap_records);
        overflow_stats = tile_renderer->get_overflow_stats();
        if (overflow_stats.raster_splats_iterated > 0) {
            counters_ready = true;
        }
    }

    result.stats = tile_renderer->get_last_render_stats();

    if (!counters_ready) {
        cleanup();
        result.error_message = "Overflow counters did not become available";
        return result;
    }

    const bool overflow_detected = overflow_stats.overflow_tile_count > 0 ||
            overflow_stats.overflow_splats_clamped > 0 ||
            overflow_stats.overflow_splats_aggregated > 0;
    if (!overflow_detected) {
        cleanup();
        result.error_message = "Expected overflow workload to trigger overflow counters";
        return result;
    }
    if (peak_overlap_records == 0) {
        cleanup();
        result.error_message = "Expected non-zero overlap records after overflow workload spike";
        return result;
    }
    if (peak_overlap_records < baseline_overlap_records) {
        cleanup();
        result.error_message = vformat("Overlap records did not increase after workload spike (baseline=%u peak=%u)",
                baseline_overlap_records, peak_overlap_records);
        return result;
    }

    if (overflow_stats.raster_reject_gaussian_idx_oob != 0 || overflow_stats.raster_reject_sorted_idx_oob != 0) {
        cleanup();
        result.error_message = vformat("Valid overflow workload should not hit OOB indices (gaussian=%d sorted=%d)",
                overflow_stats.raster_reject_gaussian_idx_oob, overflow_stats.raster_reject_sorted_idx_oob);
        return result;
    }

    cleanup();
    result.passed = true;

    return result;
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_dense_scene_rendering(RenderingDevice *p_rd) {
    TestResult result;

    tile_renderer->set_debug_binning_counters_enabled(true);

    Vector<Gaussian> gaussians = generate_test_gaussians(DENSE_SCENE_SPLAT_COUNT);
    RID gaussian_buffer = create_test_gaussian_buffer(p_rd, gaussians);
    RID sorted_indices = create_test_sorted_indices(p_rd, DENSE_SCENE_SPLAT_COUNT);
    auto cleanup = [&]() {
        if (gaussian_buffer.is_valid()) {
            p_rd->free(gaussian_buffer);
            gaussian_buffer = RID();
        }
        if (sorted_indices.is_valid()) {
            p_rd->free(sorted_indices);
            sorted_indices = RID();
        }
    };

    if (!gaussian_buffer.is_valid() || !sorted_indices.is_valid()) {
        cleanup();
        result.error_message = "Failed to create dense-scene test buffers";
        return result;
    }

    TileRenderer::RenderParams params = make_render_params(gaussian_buffer, sorted_indices, DENSE_SCENE_SPLAT_COUNT,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);

    bool counters_ready = false;
    float worst_total_ms = 0.0f;
    for (int frame = 0; frame < 6; frame++) {
        RID output = tile_renderer->render(p_rd, params);
        if (!output.is_valid()) {
            cleanup();
            result.error_message = "Failed to render dense scene workload";
            return result;
        }

        const TileRenderer::DebugCounterSnapshot counters = tile_renderer->get_debug_counters();
        if (counters.success_count > 0) {
            counters_ready = true;
        }

        const float total_ms = tile_renderer->get_tile_assignment_time() + tile_renderer->get_rasterization_time();
        if (!std::isfinite(total_ms) || total_ms < 0.0f) {
            cleanup();
            result.error_message = vformat("Invalid timing metric for dense scene: %.3f ms", total_ms);
            return result;
        }
        if (total_ms > worst_total_ms) {
            worst_total_ms = total_ms;
        }
    }

    const TileRenderer::OverflowStatsSnapshot overflow_stats = tile_renderer->get_overflow_stats();
    if (overflow_stats.raster_reject_gaussian_idx_oob != 0 || overflow_stats.raster_reject_sorted_idx_oob != 0) {
        cleanup();
        result.error_message = vformat("Dense scene hit OOB rejects (gaussian=%d sorted=%d)",
                overflow_stats.raster_reject_gaussian_idx_oob, overflow_stats.raster_reject_sorted_idx_oob);
        return result;
    }
    if (!counters_ready) {
        cleanup();
        result.error_message = "Dense scene did not report any visible splats";
        return result;
    }
    if (worst_total_ms > 5000.0f) {
        cleanup();
        result.error_message = vformat("Dense scene render exceeded sanity budget: %.3f ms", worst_total_ms);
        return result;
    }

    result.stats = tile_renderer->get_last_render_stats();
    cleanup();
    result.passed = true;

    return result;
}

Vector<Gaussian> TileRendererRegressionTest::generate_test_gaussians(uint32_t count, bool valid) {
    Vector<Gaussian> gaussians;
    gaussians.resize(count);

    for (uint32_t i = 0; i < count; i++) {
        Vector3 position(0.0f, 0.0f, -5.0f);
        Vector3 scale(1.0f, 1.0f, 1.0f);
        float opacity = 0.8f;

        if (valid) {
            // Generate valid Gaussians distributed across the viewport
            position = Vector3(
                (float(i % 32) / 32.0f - 0.5f) * 10.0f,
                (float((i / 32) % 32) / 32.0f - 0.5f) * 10.0f,
                -5.0f - float(i / 1024) * 0.1f // Depth variation
            );
            scale = Vector3(0.1f, 0.1f, 0.1f);
            opacity = 0.8f;
        } else {
            // Generate invalid Gaussians for error testing
            if (i % 4 == 0) {
                position = Vector3(NAN, 0, 0); // NaN position
            } else if (i % 4 == 1) {
                position = Vector3(0, 0, 1000000); // Extreme position
            } else if (i % 4 == 2) {
                scale = Vector3(-1, 1, 1); // Invalid scale
            } else {
                opacity = -1.0f; // Invalid opacity
            }
        }

        gaussians.write[i] = _create_test_gaussian(position, scale, opacity);
    }

    return gaussians;
}

Gaussian TileRendererRegressionTest::_create_test_gaussian(const Vector3 &position, const Vector3 &scale, float opacity) {
    Gaussian gaussian = {};  // Zero-initialize all fields

    gaussian.position = position;
    gaussian.scale = scale;
    gaussian.opacity = opacity;

    // Identity rotation (quaternion)
    gaussian.rotation = Quaternion(0, 0, 0, 1);

    // Simple color (red for testing)
    gaussian.sh_dc = Color(1.0f, 0.5f, 0.2f, 1.0f);

    // First-order SH coefficients (zeroed by default initialization above)
    // gaussian.sh_1[0..2] are already zero

    // Initialize other fields
    gaussian.normal = Vector3(0, 0, 1);
    gaussian.area = scale.x * scale.y;
    gaussian.brush_axes = Vector2(1, 1);
    gaussian.stroke_age = 0.0f;
    gaussian.painterly_meta = 0;

    return gaussian;
}

bool TileRendererRegressionTest::_validate_tile_overflow_handling(const TileRenderer::RenderStats &stats) {
    // Check that overflow was detected and handled
    if (stats.tiles_with_overflow == 0) {
        return false; // Should have detected overflow with this many splats
    }

    // Check that max splats per tile doesn't exceed the hard limit by too much
    const float tile_capacity = tile_renderer->get_tile_splat_capacity();
    if (stats.max_splats_in_tile > tile_capacity * 1.1f) {
        return false; // Overflow protection failed
    }

    // Check that error flag is set appropriately
    if (stats.tiles_with_overflow > 0 && !stats.has_rendering_errors) {
        return false; // Should have flagged errors
    }

    return true;
}

void TileRendererRegressionTest::_log_test_result(const String &test_name, const TestResult &result) {
    String status = result.passed ? "PASS" : "FAIL";
    String message = result.passed ? "" : vformat(" - %s", result.error_message);

    print_line(vformat("[%s] %s (%.2f ms)%s", status, test_name, result.execution_time_ms, message));

    if (result.passed && result.stats.total_tiles > 0) {
        print_verbose(vformat("  Stats: %d tiles, %d overflow, %.1f avg splats/tile",
                              result.stats.total_tiles, result.stats.tiles_with_overflow, result.stats.average_splats_per_tile));
    }
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_validation_and_error_detection(RenderingDevice *p_rd) {
    TestResult result;

    tile_renderer->set_debug_binning_counters_enabled(true);

    const uint32_t splat_count = 512;
    Vector<Gaussian> gaussians = generate_test_gaussians(splat_count);
    RID gaussian_buffer = create_test_gaussian_buffer(p_rd, gaussians);
    RID sorted_indices = create_test_sorted_indices(p_rd, splat_count);
    RID invalid_sorted_indices;
    auto cleanup = [&]() {
        if (gaussian_buffer.is_valid()) {
            p_rd->free(gaussian_buffer);
            gaussian_buffer = RID();
        }
        if (sorted_indices.is_valid()) {
            p_rd->free(sorted_indices);
            sorted_indices = RID();
        }
        if (invalid_sorted_indices.is_valid()) {
            p_rd->free(invalid_sorted_indices);
            invalid_sorted_indices = RID();
        }
    };

    if (!gaussian_buffer.is_valid() || !sorted_indices.is_valid()) {
        cleanup();
        result.error_message = "Failed to create validation buffers";
        return result;
    }

    TileRenderer::RenderParams params = make_render_params(gaussian_buffer, sorted_indices, splat_count,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);

    RID valid_output = tile_renderer->render(p_rd, params);
    if (!valid_output.is_valid()) {
        cleanup();
        result.error_message = "Control render failed before validation checks";
        return result;
    }

    TileRenderer::RenderParams missing_sorted = params;
    missing_sorted.sorted_indices = RID();
    RID missing_sorted_output = tile_renderer->render(p_rd, missing_sorted);
    if (missing_sorted_output.is_valid()) {
        cleanup();
        result.error_message = "Render should fail with missing sorted index buffer";
        return result;
    }

    TileRenderer::RenderParams missing_gaussian = params;
    missing_gaussian.gaussian_buffer = RID();
    RID missing_gaussian_output = tile_renderer->render(p_rd, missing_gaussian);
    if (missing_gaussian_output.is_valid()) {
        cleanup();
        result.error_message = "Render should fail with missing gaussian buffer";
        return result;
    }

    Vector<uint32_t> bad_indices;
    bad_indices.resize(splat_count);
    for (uint32_t i = 0; i < splat_count; i++) {
        bad_indices.write[i] = (i % 3 == 0) ? (splat_count + i + 17) : i;
    }
    Vector<uint8_t> bad_index_bytes;
    bad_index_bytes.resize(bad_indices.size() * sizeof(uint32_t));
    memcpy(bad_index_bytes.ptrw(), bad_indices.ptr(), bad_index_bytes.size());
    invalid_sorted_indices = p_rd->storage_buffer_create(bad_index_bytes.size(), bad_index_bytes);
    if (!invalid_sorted_indices.is_valid()) {
        cleanup();
        result.error_message = "Failed to create invalid sorted index buffer";
        return result;
    }
    p_rd->set_resource_name(invalid_sorted_indices, "GS_Test_Regression_InvalidSortedIndices");

    TileRenderer::RenderParams bad_index_params = params;
    bad_index_params.sorted_indices = invalid_sorted_indices;

    bool saw_oob_reject = false;
    for (int frame = 0; frame < 10; frame++) {
        RID output = tile_renderer->render(p_rd, bad_index_params);
        if (!output.is_valid()) {
            cleanup();
            result.error_message = "Render failed unexpectedly with invalid sorted indices workload";
            return result;
        }

        const TileRenderer::OverflowStatsSnapshot overflow_stats = tile_renderer->get_overflow_stats();
        if (overflow_stats.raster_reject_gaussian_idx_oob > 0 || overflow_stats.raster_reject_sorted_idx_oob > 0) {
            saw_oob_reject = true;
            break;
        }
    }
    if (!saw_oob_reject) {
        cleanup();
        result.error_message = "Expected OOB rejection counters for invalid sorted indices";
        return result;
    }

    result.stats = tile_renderer->get_last_render_stats();
    cleanup();
    result.passed = true;
    return result;
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_compute_fragment_clamp_parity(RenderingDevice *p_rd) {
    TestResult result;

    tile_renderer->set_debug_binning_counters_enabled(true);

    const uint32_t splat_count = 4096;
    Vector<Gaussian> gaussians;
    gaussians.resize(splat_count);
    for (uint32_t i = 0; i < splat_count; i++) {
        gaussians.write[i] = _create_test_gaussian(Vector3(0.0f, 0.0f, -3.0f), Vector3(2.5f, 2.5f, 2.5f), 0.9f);
    }

    RID gaussian_buffer = create_test_gaussian_buffer(p_rd, gaussians);
    RID sorted_indices = create_test_sorted_indices(p_rd, splat_count);
    auto cleanup = [&]() {
        if (gaussian_buffer.is_valid()) {
            p_rd->free(gaussian_buffer);
            gaussian_buffer = RID();
        }
        if (sorted_indices.is_valid()) {
            p_rd->free(sorted_indices);
            sorted_indices = RID();
        }
    };

    if (!gaussian_buffer.is_valid() || !sorted_indices.is_valid()) {
        cleanup();
        result.error_message = "Failed to create clamp parity buffers";
        return result;
    }

    TileRenderer::RenderParams params = make_render_params(gaussian_buffer, sorted_indices, splat_count,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);

    TileRenderer::OverflowStatsSnapshot fragment_overflow;
    TileRenderer::OverflowStatsSnapshot compute_overflow;
    TextureMetrics fragment_metrics;
    TextureMetrics compute_metrics;
    bool fragment_ready = false;
    bool compute_ready = false;
    bool compute_path_observed = false;
    bool fragment_metrics_ready = false;
    bool compute_metrics_ready = false;

    params.compute_raster_policy = GaussianSplatting::ComputeRasterPolicy::ForceOff;
    for (int frame = 0; frame < 12; frame++) {
        RID output = tile_renderer->render(p_rd, params);
        if (!output.is_valid()) {
            cleanup();
            result.error_message = "Fragment reference render failed in clamp parity test";
            return result;
        }
        const TileRenderer::RenderStats stats = tile_renderer->get_last_render_stats();
        if (stats.last_raster_used_compute) {
            cleanup();
            result.error_message = "ForceOff compute policy unexpectedly used compute raster";
            return result;
        }
        fragment_overflow = tile_renderer->get_overflow_stats();
        if (fragment_overflow.raster_sample_count > 0) {
            fragment_ready = true;
        }
    }

    if (!fragment_ready) {
        cleanup();
        result.error_message = "Fragment clamp parity phase did not produce readable overflow stats";
        return result;
    }
    if (fragment_overflow.overflow_splats_clamped == 0) {
        cleanup();
        result.error_message = "Clamp parity workload did not trigger any clamped splats in fragment mode";
        return result;
    }
    {
        RID fragment_output = tile_renderer->render(p_rd, params);
        Vector<uint8_t> fragment_pixels;
        if (fragment_output.is_valid() && read_texture_pixels(p_rd, fragment_output, fragment_pixels)) {
            fragment_metrics = compute_texture_metrics(fragment_pixels);
            fragment_metrics_ready = fragment_metrics.non_zero_pixels > 0;
        }
    }

    params.compute_raster_policy = GaussianSplatting::ComputeRasterPolicy::ForceOn;
    for (int frame = 0; frame < 12; frame++) {
        RID output = tile_renderer->render(p_rd, params);
        if (!output.is_valid()) {
            cleanup();
            result.error_message = "Compute phase render failed in clamp parity test";
            return result;
        }
        const TileRenderer::RenderStats stats = tile_renderer->get_last_render_stats();
        compute_path_observed = compute_path_observed || stats.last_raster_used_compute;
        compute_overflow = tile_renderer->get_overflow_stats();
        if (compute_overflow.raster_sample_count > 0) {
            compute_ready = true;
        }
    }

    if (!compute_ready) {
        cleanup();
        result.error_message = "Compute clamp parity phase did not produce readable overflow stats";
        return result;
    }
    {
        RID compute_output = tile_renderer->render(p_rd, params);
        Vector<uint8_t> compute_pixels;
        if (compute_output.is_valid() && read_texture_pixels(p_rd, compute_output, compute_pixels)) {
            compute_metrics = compute_texture_metrics(compute_pixels);
            compute_metrics_ready = compute_metrics.non_zero_pixels > 0;
        }
    }

    if (compute_path_observed) {
        if (compute_overflow.overflow_splats_clamped != fragment_overflow.overflow_splats_clamped) {
            cleanup();
            result.error_message = vformat("Clamp parity mismatch between fragment (%u) and compute (%u)",
                    fragment_overflow.overflow_splats_clamped, compute_overflow.overflow_splats_clamped);
            return result;
        }
        if (!fragment_metrics_ready || !compute_metrics_ready) {
            cleanup();
            result.error_message = "Failed to collect render metrics for compute/fragment parity comparison";
            return result;
        }
        const float luma_delta = std::abs(fragment_metrics.average_luma - compute_metrics.average_luma);
        const float alpha_delta = std::abs(fragment_metrics.average_alpha - compute_metrics.average_alpha);
        if (luma_delta > 0.03f || alpha_delta > 0.03f) {
            cleanup();
            result.error_message = vformat("Compute/fragment visual parity drift exceeds tolerance (luma=%.4f alpha=%.4f)",
                    luma_delta, alpha_delta);
            return result;
        }
    }

    result.stats = tile_renderer->get_last_render_stats();
    cleanup();
    result.passed = true;
    return result;
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_distance_cull_sort_order_stability(RenderingDevice *p_rd) {
    TestResult result;

    tile_renderer->set_debug_binning_counters_enabled(true);

    const uint32_t grid_x = 48;
    const uint32_t grid_y = 32;
    const uint32_t splat_count = grid_x * grid_y;
    Vector<Gaussian> gaussians;
    gaussians.resize(splat_count);
    for (uint32_t y = 0; y < grid_y; y++) {
        for (uint32_t x = 0; x < grid_x; x++) {
            const uint32_t idx = y * grid_x + x;
            const float fx = (float(x) + 0.5f) / float(grid_x);
            const float fy = (float(y) + 0.5f) / float(grid_y);
            const float wx = (fx - 0.5f) * 8.5f;
            const float wy = (fy - 0.5f) * 5.5f;
            const float wz = -8.0f - 0.002f * float(idx % 31);

            Gaussian gaussian = _create_test_gaussian(Vector3(wx, wy, wz), Vector3(0.09f, 0.09f, 0.09f), 0.92f);
            const float color_a = float((idx * 17u) % 251u) / 250.0f;
            const float color_b = float((idx * 29u) % 241u) / 240.0f;
            const float color_c = float((idx * 43u) % 239u) / 238.0f;
            gaussian.sh_dc = Color(0.15f + 0.85f * color_a, 0.10f + 0.90f * color_b, 0.12f + 0.88f * color_c, 1.0f);
            gaussians.write[idx] = gaussian;
        }
    }

    RID gaussian_buffer = create_test_gaussian_buffer(p_rd, gaussians);
    RID sorted_indices_identity = create_test_sorted_indices(p_rd, splat_count);
    RID sorted_indices_reversed;
    auto cleanup = [&]() {
        if (gaussian_buffer.is_valid()) {
            p_rd->free(gaussian_buffer);
            gaussian_buffer = RID();
        }
        if (sorted_indices_identity.is_valid()) {
            p_rd->free(sorted_indices_identity);
            sorted_indices_identity = RID();
        }
        if (sorted_indices_reversed.is_valid()) {
            p_rd->free(sorted_indices_reversed);
            sorted_indices_reversed = RID();
        }
    };

    if (!gaussian_buffer.is_valid() || !sorted_indices_identity.is_valid()) {
        cleanup();
        result.error_message = "Failed to create distance-cull sort-order stability buffers";
        return result;
    }

    Vector<uint32_t> reversed_indices;
    reversed_indices.resize(splat_count);
    for (uint32_t i = 0; i < splat_count; i++) {
        reversed_indices.write[i] = splat_count - 1u - i;
    }
    Vector<uint8_t> reversed_bytes;
    reversed_bytes.resize(reversed_indices.size() * sizeof(uint32_t));
    memcpy(reversed_bytes.ptrw(), reversed_indices.ptr(), reversed_bytes.size());
    sorted_indices_reversed = p_rd->storage_buffer_create(reversed_bytes.size(), reversed_bytes);
    if (!sorted_indices_reversed.is_valid()) {
        cleanup();
        result.error_message = "Failed to create reversed sorted index buffer";
        return result;
    }
    p_rd->set_resource_name(sorted_indices_reversed, "GS_Test_Regression_SortedIndicesReversed");

    TileRenderer::RenderParams params = make_render_params(gaussian_buffer, sorted_indices_identity, splat_count,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);
    params.compute_raster_policy = GaussianSplatting::ComputeRasterPolicy::ForceOff;
    params.distance_cull_enabled = true;
    params.distance_cull_start = 1.0f;
    params.distance_cull_max_rate = 0.8f;
    params.opacity_aware_culling = false;
    params.tiny_splat_screen_radius = 0.0f;
    params.lod_blend_enabled = false;
    params.enable_direct_lighting = false;

    RID identity_output;
    RID reversed_output;
    Vector<uint8_t> identity_pixels;
    Vector<uint8_t> reversed_pixels;
    bool saw_identity_cull = false;
    bool saw_reversed_cull = false;
    for (int frame = 0; frame < 6; frame++) {
        params.sorted_indices = sorted_indices_identity;
        identity_output = tile_renderer->render(p_rd, params);
        if (!identity_output.is_valid()) {
            cleanup();
            result.error_message = "Identity-order render failed in distance-cull stability test";
            return result;
        }
        const TileRenderer::DebugCounterSnapshot counters = tile_renderer->get_debug_counters();
        saw_identity_cull = saw_identity_cull || counters.distance_cull_reject > 0;
    }
    if (!read_texture_pixels(p_rd, identity_output, identity_pixels)) {
        cleanup();
        result.error_message = "Failed to read identity-order output in distance-cull stability test";
        return result;
    }

    for (int frame = 0; frame < 6; frame++) {
        params.sorted_indices = sorted_indices_reversed;
        reversed_output = tile_renderer->render(p_rd, params);
        if (!reversed_output.is_valid()) {
            cleanup();
            result.error_message = "Reversed-order render failed in distance-cull stability test";
            return result;
        }
        const TileRenderer::DebugCounterSnapshot counters = tile_renderer->get_debug_counters();
        saw_reversed_cull = saw_reversed_cull || counters.distance_cull_reject > 0;
    }
    if (!read_texture_pixels(p_rd, reversed_output, reversed_pixels)) {
        cleanup();
        result.error_message = "Failed to read reversed-order output in distance-cull stability test";
        return result;
    }

    if (!saw_identity_cull || !saw_reversed_cull) {
        cleanup();
        result.error_message = "Distance-cull stability workload did not trigger distance_cull_reject counters";
        return result;
    }
    if (identity_pixels.size() != reversed_pixels.size() || identity_pixels.is_empty()) {
        cleanup();
        result.error_message = "Distance-cull stability output size mismatch";
        return result;
    }

    const TextureMetrics identity_metrics = compute_texture_metrics(identity_pixels);
    const TextureMetrics reversed_metrics = compute_texture_metrics(reversed_pixels);
    if (identity_metrics.non_zero_pixels == 0 || reversed_metrics.non_zero_pixels == 0) {
        cleanup();
        result.error_message = "Distance-cull stability outputs are unexpectedly empty";
        return result;
    }

    double normalized_error = 0.0;
    const uint8_t *identity_read = identity_pixels.ptr();
    const uint8_t *reversed_read = reversed_pixels.ptr();
    for (int i = 0; i < identity_pixels.size(); i++) {
        int diff = int(identity_read[i]) - int(reversed_read[i]);
        if (diff < 0) {
            diff = -diff;
        }
        normalized_error += double(diff) / 255.0;
    }
    normalized_error /= double(identity_pixels.size());

    const float luma_delta = std::abs(identity_metrics.average_luma - reversed_metrics.average_luma);
    const float alpha_delta = std::abs(identity_metrics.average_alpha - reversed_metrics.average_alpha);
    if (normalized_error > 0.015 || luma_delta > 0.015f || alpha_delta > 0.015f) {
        cleanup();
        result.error_message = vformat(
                "Distance-cull output drift under sorted-order churn exceeds tolerance (error=%.4f luma=%.4f alpha=%.4f)",
                normalized_error, luma_delta, alpha_delta);
        return result;
    }

    result.stats = tile_renderer->get_last_render_stats();
    cleanup();
    result.passed = true;
    return result;
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_compute_format_fallback(RenderingDevice *p_rd) {
    TestResult result;

    const uint32_t splat_count = 2048;
    Vector<Gaussian> gaussians;
    gaussians.resize(splat_count);
    for (uint32_t i = 0; i < splat_count; i++) {
        const float x = (float(i % 64) / 63.0f - 0.5f) * 0.6f;
        const float y = (float((i / 64) % 32) / 31.0f - 0.5f) * 0.6f;
        gaussians.write[i] = _create_test_gaussian(Vector3(x, y, -3.0f), Vector3(0.6f, 0.6f, 0.6f), 0.85f);
    }

    RID gaussian_buffer = create_test_gaussian_buffer(p_rd, gaussians);
    RID sorted_indices = create_test_sorted_indices(p_rd, splat_count);
    auto cleanup = [&]() {
        tile_renderer->set_output_format(RD::DATA_FORMAT_R8G8B8A8_UNORM);
        if (gaussian_buffer.is_valid()) {
            p_rd->free(gaussian_buffer);
            gaussian_buffer = RID();
        }
        if (sorted_indices.is_valid()) {
            p_rd->free(sorted_indices);
            sorted_indices = RID();
        }
    };

    if (!gaussian_buffer.is_valid() || !sorted_indices.is_valid()) {
        cleanup();
        result.error_message = "Failed to create compute format fallback buffers";
        return result;
    }

    TileRenderer::RenderParams params = make_render_params(gaussian_buffer, sorted_indices, splat_count,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);
    params.compute_raster_policy = GaussianSplatting::ComputeRasterPolicy::ForceOn;

    tile_renderer->set_output_format(RD::DATA_FORMAT_R8G8B8A8_UNORM);
    bool baseline_compute_supported = false;
    for (int frame = 0; frame < 4; frame++) {
        RID output = tile_renderer->render(p_rd, params);
        if (!output.is_valid()) {
            cleanup();
            result.error_message = "Baseline render failed in compute format fallback test";
            return result;
        }
        baseline_compute_supported = baseline_compute_supported || tile_renderer->get_last_render_stats().last_raster_used_compute;
    }

    tile_renderer->set_output_format(RD::DATA_FORMAT_R16G16B16A16_SFLOAT);
    if (tile_renderer->get_output_format() != RD::DATA_FORMAT_R16G16B16A16_SFLOAT) {
        cleanup();
        result.stats = tile_renderer->get_last_render_stats();
        result.passed = true;
        return result;
    }

    bool saw_fragment_fallback = false;
    bool saw_compute = false;
    bool saw_valid_output = false;
    for (int frame = 0; frame < 8; frame++) {
        RID output = tile_renderer->render(p_rd, params);
        if (!output.is_valid()) {
            cleanup();
            result.error_message = "Render failed in compute format fallback test";
            return result;
        }
        saw_valid_output = true;
        const TileRenderer::RenderStats stats = tile_renderer->get_last_render_stats();
        saw_compute = saw_compute || stats.last_raster_used_compute;
        saw_fragment_fallback = saw_fragment_fallback || !stats.last_raster_used_compute;
    }

    if (!saw_valid_output) {
        cleanup();
        result.error_message = "Compute format fallback test did not produce output";
        return result;
    }
    if (baseline_compute_supported && (saw_compute || !saw_fragment_fallback)) {
        cleanup();
        result.error_message = "Expected non-RGBA8 output format to force compute->fragment fallback";
        return result;
    }

    result.stats = tile_renderer->get_last_render_stats();
    cleanup();
    result.passed = true;
    return result;
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_alpha_compositing_accuracy(RenderingDevice *p_rd) {
    TestResult result;

    Vector<Gaussian> low_opacity_gaussians;
    Vector<Gaussian> high_opacity_gaussians;
    low_opacity_gaussians.resize(4);
    high_opacity_gaussians.resize(4);

    const Vector3 positions[4] = {
        Vector3(-0.15f, 0.0f, -3.0f),
        Vector3(0.15f, 0.0f, -3.0f),
        Vector3(0.0f, -0.15f, -3.0f),
        Vector3(0.0f, 0.15f, -3.0f),
    };
    for (int i = 0; i < 4; i++) {
        low_opacity_gaussians.write[i] = _create_test_gaussian(positions[i], Vector3(0.45f, 0.45f, 0.45f), 0.2f);
        high_opacity_gaussians.write[i] = _create_test_gaussian(positions[i], Vector3(0.45f, 0.45f, 0.45f), 0.9f);
    }

    RID low_gaussian_buffer = create_test_gaussian_buffer(p_rd, low_opacity_gaussians);
    RID high_gaussian_buffer = create_test_gaussian_buffer(p_rd, high_opacity_gaussians);
    RID sorted_indices = create_test_sorted_indices(p_rd, 4);
    auto cleanup = [&]() {
        if (low_gaussian_buffer.is_valid()) {
            p_rd->free(low_gaussian_buffer);
            low_gaussian_buffer = RID();
        }
        if (high_gaussian_buffer.is_valid()) {
            p_rd->free(high_gaussian_buffer);
            high_gaussian_buffer = RID();
        }
        if (sorted_indices.is_valid()) {
            p_rd->free(sorted_indices);
            sorted_indices = RID();
        }
    };

    if (!low_gaussian_buffer.is_valid() || !high_gaussian_buffer.is_valid() || !sorted_indices.is_valid()) {
        cleanup();
        result.error_message = "Failed to create alpha compositing test buffers";
        return result;
    }

    TileRenderer::RenderParams low_params = make_render_params(low_gaussian_buffer, sorted_indices, 4,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);
    TileRenderer::RenderParams high_params = make_render_params(high_gaussian_buffer, sorted_indices, 4,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);

    RID low_output = tile_renderer->render(p_rd, low_params);
    low_output = tile_renderer->render(p_rd, low_params); // Use second frame after pipeline warmup.
    RID high_output = tile_renderer->render(p_rd, high_params);
    high_output = tile_renderer->render(p_rd, high_params);

    if (!low_output.is_valid() || !high_output.is_valid()) {
        cleanup();
        result.error_message = "Render failed during alpha compositing comparison";
        return result;
    }

    Vector<uint8_t> low_pixels;
    Vector<uint8_t> high_pixels;
    if (!read_texture_pixels(p_rd, low_output, low_pixels) || !read_texture_pixels(p_rd, high_output, high_pixels)) {
        cleanup();
        result.error_message = "Failed to read back output textures for alpha comparison";
        return result;
    }

    if (low_pixels.size() != high_pixels.size()) {
        cleanup();
        result.error_message = "Alpha comparison texture size mismatch";
        return result;
    }

    const TextureMetrics low_metrics = compute_texture_metrics(low_pixels);
    const TextureMetrics high_metrics = compute_texture_metrics(high_pixels);
    if (low_metrics.non_zero_pixels == 0 || high_metrics.non_zero_pixels == 0) {
        cleanup();
        result.error_message = "Expected non-zero rendered pixels for alpha compositing validation";
        return result;
    }

    const int center_x = TEST_VIEWPORT_WIDTH / 2;
    const int center_y = TEST_VIEWPORT_HEIGHT / 2;
    const int center_idx = (center_y * TEST_VIEWPORT_WIDTH + center_x) * 4;
    const int low_center_rgb = int(low_pixels[center_idx + 0]) + int(low_pixels[center_idx + 1]) + int(low_pixels[center_idx + 2]);
    const int high_center_rgb = int(high_pixels[center_idx + 0]) + int(high_pixels[center_idx + 1]) + int(high_pixels[center_idx + 2]);

    if (high_center_rgb <= low_center_rgb) {
        cleanup();
        result.error_message = vformat("Expected higher opacity to increase center intensity (low=%d high=%d)",
                low_center_rgb, high_center_rgb);
        return result;
    }
    if (high_metrics.average_luma <= low_metrics.average_luma + 0.001f) {
        cleanup();
        result.error_message = vformat("Expected higher opacity luma (low=%.4f high=%.4f)",
                low_metrics.average_luma, high_metrics.average_luma);
        return result;
    }

    result.stats = tile_renderer->get_last_render_stats();
    cleanup();
    result.passed = true;
    return result;
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_performance_regression(RenderingDevice *p_rd) {
    TestResult result;

    const uint32_t splat_count = 8192;
    Vector<Gaussian> gaussians = generate_test_gaussians(splat_count);
    RID gaussian_buffer = create_test_gaussian_buffer(p_rd, gaussians);
    RID sorted_indices = create_test_sorted_indices(p_rd, splat_count);
    auto cleanup = [&]() {
        if (gaussian_buffer.is_valid()) {
            p_rd->free(gaussian_buffer);
            gaussian_buffer = RID();
        }
        if (sorted_indices.is_valid()) {
            p_rd->free(sorted_indices);
            sorted_indices = RID();
        }
    };

    if (!gaussian_buffer.is_valid() || !sorted_indices.is_valid()) {
        cleanup();
        result.error_message = "Failed to create performance regression buffers";
        return result;
    }

    TileRenderer::RenderParams params = make_render_params(gaussian_buffer, sorted_indices, splat_count,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);

    float worst_ms = 0.0f;
    bool saw_non_zero_timing = false;
    for (int frame = 0; frame < 6; frame++) {
        RID output = tile_renderer->render(p_rd, params);
        if (!output.is_valid()) {
            cleanup();
            result.error_message = "Performance regression workload render failed";
            return result;
        }

        const float total_ms = tile_renderer->get_tile_assignment_time() + tile_renderer->get_rasterization_time();
        if (!std::isfinite(total_ms) || total_ms < 0.0f) {
            cleanup();
            result.error_message = vformat("Invalid performance timing value: %.4f ms", total_ms);
            return result;
        }
        saw_non_zero_timing = saw_non_zero_timing || (total_ms > 0.0f);
        if (total_ms > worst_ms) {
            worst_ms = total_ms;
        }
    }

    if (!saw_non_zero_timing) {
        cleanup();
        result.error_message = "Performance timing metrics stayed at zero for all frames";
        return result;
    }
    if (worst_ms > 5000.0f) {
        cleanup();
        result.error_message = vformat("Performance sanity budget exceeded: %.3f ms", worst_ms);
        return result;
    }

    result.stats = tile_renderer->get_last_render_stats();
    cleanup();
    result.passed = true;
    return result;
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_renderer_lifecycle_leak_detection(RenderingDevice *p_rd) {
    TestResult result;

    static constexpr int LIFECYCLE_ITERATIONS = 3;
    static constexpr uint32_t TEST_SPLAT_COUNT = 2048;
    static constexpr uint64_t ALLOCATION_SLACK = 32;
    static constexpr uint64_t MEMORY_SLACK_BYTES = 16ull * 1024ull * 1024ull;

    uint64_t cleanup_baseline_allocations = 0;
    uint64_t cleanup_baseline_memory = 0;
    bool cleanup_baseline_set = false;

    for (int iteration = 0; iteration < LIFECYCLE_ITERATIONS; iteration++) {
        Vector<Gaussian> gaussians = generate_test_gaussians(TEST_SPLAT_COUNT);
        RID gaussian_buffer = create_test_gaussian_buffer(p_rd, gaussians);
        RID sorted_indices = create_test_sorted_indices(p_rd, TEST_SPLAT_COUNT);
        auto free_cycle_buffers = [&]() {
            if (gaussian_buffer.is_valid()) {
                p_rd->free(gaussian_buffer);
                gaussian_buffer = RID();
            }
            if (sorted_indices.is_valid()) {
                p_rd->free(sorted_indices);
                sorted_indices = RID();
            }
        };

        if (!gaussian_buffer.is_valid() || !sorted_indices.is_valid()) {
            free_cycle_buffers();
            result.error_message = "Failed to create lifecycle leak detection buffers";
            return result;
        }

        TileRenderer::RenderParams params = make_render_params(gaussian_buffer, sorted_indices, TEST_SPLAT_COUNT,
                TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);
        for (int frame = 0; frame < 3; frame++) {
            RID output = tile_renderer->render(p_rd, params);
            if (!output.is_valid()) {
                free_cycle_buffers();
                result.error_message = vformat("Renderer lifecycle iteration %d failed to render", iteration);
                return result;
            }
            result.stats = tile_renderer->get_last_render_stats();
        }

        free_cycle_buffers();
        tile_renderer->cleanup();

        // Drain queued GPU work so allocation counters reflect post-cleanup state.
        p_rd->submit();
        p_rd->sync();

        const uint64_t allocations_after_cleanup = p_rd->get_device_allocation_count();
        const uint64_t memory_after_cleanup = p_rd->get_device_total_memory();
        if (!cleanup_baseline_set) {
            cleanup_baseline_set = true;
            cleanup_baseline_allocations = allocations_after_cleanup;
            cleanup_baseline_memory = memory_after_cleanup;
        } else {
            if (allocations_after_cleanup > cleanup_baseline_allocations + ALLOCATION_SLACK) {
                result.error_message = vformat(
                        "Allocation count grew across renderer lifecycle cleanup (baseline=%s current=%s slack=%s)",
                        String::num_uint64(cleanup_baseline_allocations),
                        String::num_uint64(allocations_after_cleanup),
                        String::num_uint64(ALLOCATION_SLACK));
                return result;
            }
            if (memory_after_cleanup > cleanup_baseline_memory + MEMORY_SLACK_BYTES) {
                result.error_message = vformat(
                        "Device memory grew across renderer lifecycle cleanup (baseline=%s current=%s slack=%s bytes)",
                        String::num_uint64(cleanup_baseline_memory),
                        String::num_uint64(memory_after_cleanup),
                        String::num_uint64(MEMORY_SLACK_BYTES));
                return result;
            }
        }

        if (iteration + 1 < LIFECYCLE_ITERATIONS) {
            Error err = tile_renderer->initialize(p_rd, Vector2i(TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT), TEST_TILE_SIZE);
            if (err != OK) {
                result.error_message = vformat("Failed to reinitialize tile renderer in lifecycle iteration %d", iteration);
                return result;
            }
        }
    }

    result.passed = true;
    return result;
}

TileRendererRegressionTest::TestResult TileRendererRegressionTest::test_zero_work_frame_resets_raster_timing(RenderingDevice *p_rd) {
    TestResult result;

    static constexpr uint32_t TEST_SPLAT_COUNT = 2048;

    Vector<Gaussian> gaussians = generate_test_gaussians(TEST_SPLAT_COUNT);
    RID gaussian_buffer = create_test_gaussian_buffer(p_rd, gaussians);
    RID sorted_indices = create_test_sorted_indices(p_rd, TEST_SPLAT_COUNT);
    auto cleanup = [&]() {
        if (gaussian_buffer.is_valid()) {
            p_rd->free(gaussian_buffer);
            gaussian_buffer = RID();
        }
        if (sorted_indices.is_valid()) {
            p_rd->free(sorted_indices);
            sorted_indices = RID();
        }
    };

    if (!gaussian_buffer.is_valid() || !sorted_indices.is_valid()) {
        cleanup();
        result.error_message = "Failed to create buffers for zero-work raster timing test";
        return result;
    }

    TileRenderer::RenderParams active_params = make_render_params(gaussian_buffer, sorted_indices, TEST_SPLAT_COUNT,
            TEST_VIEWPORT_WIDTH, TEST_VIEWPORT_HEIGHT, TEST_TILE_SIZE);
    RID active_output = tile_renderer->render(p_rd, active_params);
    if (!active_output.is_valid()) {
        cleanup();
        result.error_message = "Failed to render active frame for zero-work timing test";
        return result;
    }

    TileRenderer::RenderParams idle_params = active_params;
    idle_params.splat_count = 0;
    idle_params.total_gaussians = 0;
    RID idle_output = tile_renderer->render(p_rd, idle_params);
    if (!idle_output.is_valid()) {
        cleanup();
        result.error_message = "Failed to render zero-work frame for timing reset test";
        return result;
    }

    const float idle_raster_ms = tile_renderer->get_rasterization_time();
    if (idle_raster_ms != 0.0f) {
        cleanup();
        result.error_message = vformat("Expected rasterization_ms to reset to 0 on zero-work frame (got %.6f)", idle_raster_ms);
        return result;
    }

    cleanup();
    result.passed = true;
    return result;
}

RID TileRendererRegressionTest::create_test_gaussian_buffer(RenderingDevice *p_rd, const Vector<Gaussian> &gaussians) {
    if (gaussians.size() == 0) {
        return RID();
    }

    Vector<uint8_t> buffer_data;
    buffer_data.resize(gaussians.size() * sizeof(Gaussian));
    memcpy(buffer_data.ptrw(), gaussians.ptr(), buffer_data.size());

    RID buffer = p_rd->storage_buffer_create(buffer_data.size(), buffer_data);
    p_rd->set_resource_name(buffer, "GS_Test_Regression_GaussianBuffer");
    return buffer;
}

RID TileRendererRegressionTest::create_test_sorted_indices(RenderingDevice *p_rd, uint32_t count) {
    Vector<uint32_t> indices;
    indices.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        indices.write[i] = i;
    }

    Vector<uint8_t> buffer_data;
    buffer_data.resize(indices.size() * sizeof(uint32_t));
    memcpy(buffer_data.ptrw(), indices.ptr(), buffer_data.size());

    RID buffer = p_rd->storage_buffer_create(buffer_data.size(), buffer_data);
    p_rd->set_resource_name(buffer, "GS_Test_Regression_SortedIndices");
    return buffer;
}

bool TileRendererRegressionTest::compare_render_output(RID texture1, RID texture2, float tolerance) {
    if (!texture1.is_valid() || !texture2.is_valid()) {
        return false;
    }

    RenderingDevice *rd = tile_renderer->get_output_texture_owner();
    if (!rd) {
        return false;
    }

    Vector<uint8_t> pixels_a;
    Vector<uint8_t> pixels_b;
    if (!read_texture_pixels(rd, texture1, pixels_a) || !read_texture_pixels(rd, texture2, pixels_b)) {
        return false;
    }
    if (pixels_a.size() != pixels_b.size()) {
        return false;
    }

    const uint8_t *a = pixels_a.ptr();
    const uint8_t *b = pixels_b.ptr();
    double normalized_error = 0.0;
    for (int i = 0; i < pixels_a.size(); i++) {
        const int diff = int(a[i]) - int(b[i]);
        const int abs_diff = (diff >= 0) ? diff : -diff;
        normalized_error += double(abs_diff) / 255.0;
    }
    normalized_error /= double(pixels_a.size());
    return normalized_error <= double(tolerance);
}

bool TileRendererRegressionTest::generate_reference_captures(RenderingDevice *p_rd) {
    // Stub implementation
    return true;
}

bool TileRendererRegressionTest::validate_against_reference(RID output_texture, const String &reference_name) {
    // Stub implementation
    return true;
}

// TEST_CASE wrapper to integrate with doctest framework
#include "test_macros.h"

TEST_CASE("[TileRenderer] Range pipeline regression test") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("[TileRenderer] RenderingServer not available, skipping regression tests");
        return;
    }

    RenderingDevice *rd = rs->create_local_rendering_device();
    if (!rd) {
        MESSAGE("[TileRenderer] Could not create local rendering device, skipping regression tests");
        return;
    }

    Ref<TileRendererRegressionTest> regression_test;
    regression_test.instantiate();

    bool all_passed = regression_test->run_all_tests(rd);

    memdelete(rd);

    CHECK(all_passed);
}
