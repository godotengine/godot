#include "test_lod_system.h"
#include "../lod/hierarchical_splat_structure.h"
#include "../lod/adaptive_lod_system.h"
#include "../core/gaussian_data.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../core/gaussian_splat_manager.h"
#include "core/os/os.h"
#include "core/math/random_number_generator.h"
#include "core/math/projection.h"
#include "core/templates/hash_set.h"
#include "tests/test_macros.h"

namespace GaussianSplatting {
namespace Tests {

// Test fixture
class LODSystemTest {
public:
    Vector<GaussianData> generate_test_splats(uint32_t count, float spread = 100.0f) {
        Vector<GaussianData> splats;
        splats.resize(count);

        RandomNumberGenerator rng;
        rng.set_seed(12345);  // Fixed seed for reproducibility

        for (uint32_t i = 0; i < count; i++) {
            GaussianData& splat = splats.write[i];
            splat.position = Vector3(
                rng.randf_range(-spread, spread),
                rng.randf_range(-spread, spread),
                rng.randf_range(-spread, spread)
            );
            splat.color = Color(
                rng.randf(),
                rng.randf(),
                rng.randf(),
                rng.randf_range(0.5f, 1.0f)  // Alpha
            );
            splat.rotation = Quaternion();  // Identity
            splat.scale = Vector3(
                rng.randf_range(0.5f, 2.0f),
                rng.randf_range(0.5f, 2.0f),
                rng.randf_range(0.5f, 2.0f)
            );
            splat.index = i;
        }

        return splats;
    }

    Camera3D* create_test_camera(const Vector3& position = Vector3(0, 0, 10)) {
        Camera3D* camera = memnew(Camera3D);
        Transform3D transform;
        transform.origin = position;
        transform.basis = Basis();  // Looking down -Z
        camera->set_global_transform(transform);
        camera->set_fov(60.0f);
        camera->set_near(0.1f);
        camera->set_far(1000.0f);
        return camera;
    }
};

static float compute_visibility_churn_ratio(
    const LocalVector<uint32_t> &previous_visible,
    const LocalVector<uint32_t> &current_visible) {

    HashSet<uint32_t> previous_set;
    for (uint32_t idx : previous_visible) {
        previous_set.insert(idx);
    }

    uint32_t retained = 0;
    for (uint32_t idx : current_visible) {
        if (previous_set.has(idx)) {
            retained++;
            previous_set.erase(idx);
        }
    }

    const uint32_t removed = previous_set.size();
    const uint32_t added = current_visible.size() - retained;
    const uint32_t denominator = MAX((uint32_t)1, MAX(previous_visible.size(), current_visible.size()));

    return float(added + removed) / float(denominator);
}

// Test hierarchical splat structure
bool test_hierarchical_structure_build() {
    print_line("Testing: Hierarchical Structure Build");

    LODSystemTest test;
    auto splats = test.generate_test_splats(10000);

    HierarchicalSplatStructure structure;
    HierarchicalSplatStructure::BuildParams params;
    params.max_depth = 6;
    params.min_splats_per_node = 16;

    uint64_t start = OS::get_singleton()->get_ticks_usec();
    structure.build_hierarchy(splats, params);
    uint64_t build_time = OS::get_singleton()->get_ticks_usec() - start;

    // Validate structure
    auto stats = structure.get_statistics();
    bool success = true;

    if (stats.total_nodes == 0) {
        ERR_PRINT("Failed: No nodes created");
        success = false;
    }

    if (stats.leaf_nodes == 0) {
        ERR_PRINT("Failed: No leaf nodes created");
        success = false;
    }

    if (build_time > 100000) {  // 100ms for 10K splats
        WARN_PRINT(vformat("Warning: Build time too high: %.2f ms", build_time / 1000.0f));
    }

    print_line(vformat("  Build time: %.2f ms", build_time / 1000.0f));
    print_line(vformat("  Total nodes: %d", stats.total_nodes));
    print_line(vformat("  Leaf nodes: %d", stats.leaf_nodes));
    print_line(vformat("  Max depth: %d", stats.max_depth_reached));
    print_line(vformat("  Avg splats/leaf: %.1f", stats.avg_splats_per_leaf));
    print_line(vformat("  Memory usage: %d KB", stats.memory_usage / 1024));

    return success;
}

// Test frustum culling
bool test_frustum_culling() {
    print_line("Testing: Frustum Culling");

    LODSystemTest test;
    auto splats = test.generate_test_splats(100000);
    auto camera = test.create_test_camera(Vector3(0, 0, 50));

    HierarchicalSplatStructure structure;
    structure.build_hierarchy(splats);

    Frustum frustum = camera->get_frustum();

    uint64_t start = OS::get_singleton()->get_ticks_usec();
    auto result = structure.query_visible_splats(
        frustum,
        camera->get_global_transform().origin,
        1.0f,  // LOD bias
        50000  // Max splats
    );
    uint64_t cull_time = OS::get_singleton()->get_ticks_usec() - start;

    bool success = true;

    if (result.visible_indices.size() == 0) {
        ERR_PRINT("Failed: No visible splats found");
        success = false;
    }

    if (result.visible_indices.size() == splats.size()) {
        WARN_PRINT("Warning: No culling occurred");
    }

    if (cull_time > 1000) {  // 1ms target
        WARN_PRINT(vformat("Warning: Culling time too high: %.2f ms", cull_time / 1000.0f));
    }

    print_line(vformat("  Cull time: %.2f ms", cull_time / 1000.0f));
    print_line(vformat("  Visible splats: %d / %d", result.visible_indices.size(), splats.size()));
    print_line(vformat("  Culled: %.1f%%", result.culled_percentage));
    print_line(vformat("  LOD0: %d, LOD1: %d, LOD2: %d, LOD3: %d",
                      result.lod_stats.lod0_count,
                      result.lod_stats.lod1_count,
                      result.lod_stats.lod2_count,
                      result.lod_stats.lod3_count));

    memdelete(camera);
    return success;
}

// Test adaptive LOD selection
bool test_adaptive_lod() {
    print_line("Testing: Adaptive LOD Selection");

    LODSystemTest test;
    auto splats = test.generate_test_splats(50000);
    auto camera = test.create_test_camera();

    GaussianSplatting::AdaptiveLODSystem lod_system;
    GaussianSplatting::AdaptiveLODSystem::LODConfig config;
    config.max_splats_per_frame = 10000;
    config.lod_bias = 1.0f;
    lod_system.initialize(config);

    // Test different strategies
    const char* strategies[] = {
        "Distance-based",
        "Importance-based",
        "Budget-based",
        "Hybrid"
    };

    bool success = true;

    for (int i = 0; i < 4; i++) {
        auto strategy = static_cast<AdaptiveLODSystem::LODStrategy>(i);

        uint64_t start = OS::get_singleton()->get_ticks_usec();
        auto selection = lod_system.select_lod_splats(
            splats,
            camera,
            nullptr,  // No spatial structure for this test
            strategy
        );
        uint64_t select_time = OS::get_singleton()->get_ticks_usec() - start;

        print_line(vformat("  %s:", strategies[i]));
        print_line(vformat("    Selection time: %.2f ms", select_time / 1000.0f));
        print_line(vformat("    Selected: %d / %d", selection.visible_indices.size(), splats.size()));
        print_line(vformat("    Frustum culled: %d", selection.stats.frustum_culled));
        print_line(vformat("    Distance culled: %d", selection.stats.distance_culled));
        print_line(vformat("    Size culled: %d", selection.stats.size_culled));
        print_line(vformat("    Budget culled: %d", selection.stats.budget_culled));

        if (selection.visible_indices.size() == 0) {
            ERR_PRINT(vformat("Failed: No splats selected for strategy %s", strategies[i]));
            success = false;
        }

        if (strategy == AdaptiveLODSystem::BUDGET_BASED &&
            selection.visible_indices.size() > config.max_splats_per_frame) {
            ERR_PRINT("Failed: Budget constraint violated");
            success = false;
        }
    }

    memdelete(camera);
    return success;
}

// Test LOD transition smoothness
bool test_lod_transitions() {
    print_line("Testing: LOD Transitions");

    LODSystemTest test;
    auto splats = test.generate_test_splats(10000);
    auto camera = test.create_test_camera();

    GaussianSplatting::AdaptiveLODSystem lod_system;
    GaussianSplatting::AdaptiveLODSystem::LODConfig config;
    config.smooth_transitions = true;
    config.transition_time = 0.25f;
    lod_system.initialize(config);

    bool success = true;
    float prev_weight_sum = 0.0f;

    // Simulate smooth camera movement
    for (float z = 10.0f; z <= 100.0f; z += 5.0f) {
        Transform3D transform = camera->get_global_transform();
        transform.origin = Vector3(0, 0, z);
        camera->set_global_transform(transform);

        auto selection = lod_system.select_lod_splats(
            splats,
            camera,
            nullptr,
            AdaptiveLODSystem::DISTANCE_BASED
        );

        // Check weight smoothness
        float weight_sum = 0.0f;
        for (float w : selection.lod_weights) {
            weight_sum += w;
        }

        if (prev_weight_sum > 0.0f) {
            float weight_change = abs(weight_sum - prev_weight_sum) / prev_weight_sum;
            if (weight_change > 0.2f) {  // More than 20% change
                WARN_PRINT(vformat("Warning: Large weight change at Z=%.0f: %.1f%%",
                                  z, weight_change * 100.0f));
            }
        }

        prev_weight_sum = weight_sum;
    }

    print_line("  LOD transitions validated");

    memdelete(camera);
    return success;
}

bool test_node_quality_presets() {
    print_line("Testing: Node Quality Presets");

    bool success = true;

    GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
    const float test_distance = 600.0f;
    node->set_max_render_distance(test_distance);

    struct PresetExpectation {
        GaussianSplatNode3D::QualityPreset preset;
        float expected_bias;
        uint32_t expected_budget;
        float load_factor;
        int gpu_memory_mb;
        bool expect_temporal_coherence;
        bool expect_adaptive_quality;
    };

    PresetExpectation preset_expectations[] = {
        { GaussianSplatNode3D::QUALITY_PERFORMANCE, 2.0f, 200000, 0.15f, 256, false, false },
        { GaussianSplatNode3D::QUALITY_BALANCED, 1.0f, 500000, 0.25f, 512, true, true },
        { GaussianSplatNode3D::QUALITY_QUALITY, 0.75f, 1000000, 0.35f, 768, true, true }
    };

    for (const PresetExpectation &expectation : preset_expectations) {
        node->set_quality_preset(expectation.preset);

        const GaussianSplatting::GaussianSplatLODConfig &lod_config = node->get_lod_config();
        const GaussianSplatting::GaussianSplatStreamingConfig &stream_config = node->get_streaming_config();

        if (!Math::is_equal_approx(lod_config.lod_bias, expectation.expected_bias, 0.001f)) {
            ERR_PRINT(vformat("Failed: Preset %d expected bias %.2f but got %.2f", (int)expectation.preset, expectation.expected_bias, lod_config.lod_bias));
            success = false;
        }

        if (lod_config.max_splats_per_frame != expectation.expected_budget) {
            ERR_PRINT(vformat("Failed: Preset %d expected budget %d but got %d", (int)expectation.preset, expectation.expected_budget, lod_config.max_splats_per_frame));
            success = false;
        }

        if (lod_config.enable_temporal_coherence != expectation.expect_temporal_coherence) {
            ERR_PRINT(vformat("Failed: Preset %d temporal coherence mismatch", (int)expectation.preset));
            success = false;
        }

        float expected_load_ahead = test_distance * expectation.load_factor;
        if (!Math::is_equal_approx(stream_config.load_ahead_distance, expected_load_ahead, MAX(1.0f, expected_load_ahead * 0.05f))) {
            ERR_PRINT(vformat("Failed: Preset %d expected load ahead %.2f but got %.2f", (int)expectation.preset, expected_load_ahead, stream_config.load_ahead_distance));
            success = false;
        }

        uint64_t expected_gpu_bytes = (uint64_t)expectation.gpu_memory_mb * 1024 * 1024;
        if (stream_config.max_gpu_memory != expected_gpu_bytes) {
            ERR_PRINT(vformat("Failed: Preset %d expected GPU budget %d MB but got %d MB", (int)expectation.preset, expectation.gpu_memory_mb, (int)(stream_config.max_gpu_memory / (1024 * 1024))));
            success = false;
        }

        if (stream_config.enable_adaptive_quality != expectation.expect_adaptive_quality) {
            ERR_PRINT(vformat("Failed: Preset %d adaptive quality mismatch", (int)expectation.preset));
            success = false;
        }
    }

    // Custom preset should honor manual overrides while keeping balanced defaults for derived parameters.
    node->set_quality_preset(GaussianSplatNode3D::QUALITY_CUSTOM);
    node->set_lod_bias(1.3f);
    node->set_max_splat_count(750000);

    const GaussianSplatting::GaussianSplatLODConfig &custom_lod = node->get_lod_config();
    const GaussianSplatting::GaussianSplatStreamingConfig &custom_stream = node->get_streaming_config();

    if (!Math::is_equal_approx(custom_lod.lod_bias, 1.3f, 0.001f)) {
        ERR_PRINT("Failed: Custom preset did not preserve manual LOD bias override.");
        success = false;
    }

    if (custom_lod.max_splats_per_frame != 750000) {
        ERR_PRINT("Failed: Custom preset did not preserve max splat override.");
        success = false;
    }

    float expected_custom_load = test_distance * 0.25f; // Custom inherits balanced defaults.
    if (!Math::is_equal_approx(custom_stream.load_ahead_distance, expected_custom_load, MAX(1.0f, expected_custom_load * 0.05f))) {
        ERR_PRINT("Failed: Custom preset load ahead distance did not match balanced default.");
        success = false;
    }

    if (custom_stream.max_gpu_memory != (uint64_t)512 * 1024 * 1024) {
        ERR_PRINT("Failed: Custom preset GPU budget did not fall back to balanced default (512 MB).");
        success = false;
    }

    memdelete(node);
    return success;
}

// Test scalability with increasing splat counts
bool test_scalability() {
    print_line("Testing: Scalability");

    LODSystemTest test;
    const uint32_t splat_counts[] = {1000, 10000, 100000, 1000000};
    const float targets[] = {0.1f, 0.5f, 5.0f, 50.0f};  // Target times in ms

    bool success = true;

    for (int i = 0; i < 4; i++) {
        uint32_t count = splat_counts[i];
        float target = targets[i];

        auto splats = test.generate_test_splats(count);

        HierarchicalSplatStructure structure;
        HierarchicalSplatStructure::BuildParams params;
        params.max_depth = 8;
        params.parallel_build = (count > 10000);

        uint64_t start = OS::get_singleton()->get_ticks_usec();
        structure.build_hierarchy(splats, params);
        float build_time = (OS::get_singleton()->get_ticks_usec() - start) / 1000.0f;

        // Test query performance
        auto camera = test.create_test_camera();
        Frustum frustum = camera->get_frustum();

        start = OS::get_singleton()->get_ticks_usec();
        auto result = structure.query_visible_splats(
            frustum,
            camera->get_global_transform().origin,
            1.0f,
            count / 2
        );
        float query_time = (OS::get_singleton()->get_ticks_usec() - start) / 1000.0f;

        print_line(vformat("  %d splats:", count));
        print_line(vformat("    Build: %.2f ms (target: < %.1f ms)", build_time, target * 100));
        print_line(vformat("    Query: %.2f ms (target: < %.1f ms)", query_time, target));

        if (query_time > target) {
            WARN_PRINT(vformat("    Performance target missed: %.2f ms > %.1f ms",
                             query_time, target));
        }

        memdelete(camera);
    }

    return success;
}

// Main test runner
void run_lod_system_tests() {
    print_line("\n========== LOD System Tests ==========");

    struct TestCase {
        const char* name;
        bool (*test_func)();
    };

    TestCase tests[] = {
        {"Hierarchical Structure Build", test_hierarchical_structure_build},
        {"Frustum Culling", test_frustum_culling},
        {"Adaptive LOD Selection", test_adaptive_lod},
        {"LOD Transitions", test_lod_transitions},
        {"Node Quality Presets", test_node_quality_presets},
        {"Scalability", test_scalability}
    };

    int passed = 0;
    int failed = 0;

    for (const auto& test : tests) {
        print_line(vformat("\n[TEST] %s", test.name));
        bool result = test.test_func();

        if (result) {
            print_line(vformat("[PASS] %s\n", test.name));
            passed++;
        } else {
            ERR_PRINT(vformat("[FAIL] %s\n", test.name));
            failed++;
        }
    }

    print_line("\n========== Test Summary ==========");
    print_line(vformat("Passed: %d", passed));
    print_line(vformat("Failed: %d", failed));
    print_line(vformat("Total:  %d", passed + failed));

    if (failed == 0) {
        print_line("\nAll LOD system tests passed!");
    } else {
        ERR_PRINT(vformat("\n%d tests failed!", failed));
    }
}

} // namespace Tests
} // namespace GaussianSplatting

TEST_CASE("[GaussianSplatting] Renderer LOD bias and distance affect culling") {
    GaussianSplatManager *manager = memnew(GaussianSplatManager);
    CHECK(manager != nullptr);
    if (manager == nullptr) {
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(manager->get_primary_rendering_device());

    Vector<Vector3> positions;
    positions.push_back(Vector3(0.0f, 0.0f, -10.0f));
    positions.push_back(Vector3(0.0f, 0.0f, -100.0f));
    positions.push_back(Vector3(1000.0f, 0.0f, -10.0f));

    Vector<Vector3> scales;
    scales.push_back(Vector3(1.0f, 1.0f, 1.0f));
    scales.push_back(Vector3(1.0f, 1.0f, 1.0f));
    scales.push_back(Vector3(1.0f, 1.0f, 1.0f));

    renderer->test_set_test_splats(positions, scales);
    renderer->set_lod_enabled(true);
    renderer->set_lod_min_screen_size(0.0f);
    renderer->set_lod_max_distance(200.0f);
    renderer->set_lod_bias(1.0f);
    renderer->set_frustum_culling(true);

    Transform3D cam_transform;
    cam_transform.origin = Vector3(0.0f, 0.0f, 0.0f);

    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 500.0f);

    Size2i viewport(1280, 720);

    int visible = renderer->test_cull_visible_count(cam_transform, projection, viewport);
    Dictionary stats = renderer->get_render_stats();
    CHECK(visible == 2);
    CHECK(int(stats["culled_by_frustum"]) >= 1);
    CHECK(int(stats["visible_after_culling"]) == visible);

    renderer->set_lod_max_distance(50.0f);
    visible = renderer->test_cull_visible_count(cam_transform, projection, viewport);
    stats = renderer->get_render_stats();
    CHECK(visible == 1);
    CHECK(int(stats["culled_by_distance"]) >= 1);

    renderer->set_lod_max_distance(200.0f);
    renderer->set_lod_bias(1.0f);
    visible = renderer->test_cull_visible_count(cam_transform, projection, viewport);
    stats = renderer->get_render_stats();
    CHECK(visible == 2);
    CHECK(int(stats["culled_by_frustum"]) >= 1);

    renderer->set_lod_bias(4.0f);
    visible = renderer->test_cull_visible_count(cam_transform, projection, viewport);
    CHECK_MESSAGE(visible == 1, "Increased LOD bias should tighten max distance culling");
    CHECK(Math::is_equal_approx(renderer->get_lod_bias(), 4.0f));

    renderer->set_frustum_culling(false);
    renderer->set_lod_bias(1.0f);
    renderer->set_lod_max_distance(200.0f);
    visible = renderer->test_cull_visible_count(cam_transform, projection, viewport);
    stats = renderer->get_render_stats();
    CHECK(visible == 3);
    CHECK(int(stats["culled_by_frustum"]) == 0);
    CHECK(int(stats["culling_candidate_count"]) == 3);

    CHECK(Math::is_equal_approx(renderer->get_lod_bias(), 1.0f));
    CHECK(Math::is_equal_approx(renderer->get_lod_max_distance(), 200.0f));

    renderer.unref();
    memdelete(manager);
}

TEST_CASE("[GaussianSplatting] Hierarchical LOD query keeps index and weight cardinality") {
    GaussianSplatting::Tests::LODSystemTest fixture;
    Vector<GaussianSplatting::GaussianData> splats = fixture.generate_test_splats(4096, 40.0f);

    GaussianSplatting::HierarchicalSplatStructure structure;
    GaussianSplatting::HierarchicalSplatStructure::BuildParams params;
    params.max_depth = 7;
    params.min_splats_per_node = 8;
    structure.build_hierarchy(splats, params);

    Camera3D *camera = fixture.create_test_camera(Vector3(0.0f, 0.0f, 280.0f));
    CHECK(camera != nullptr);
    if (camera == nullptr) {
        return;
    }

    const GaussianSplatting::Frustum frustum = camera->get_frustum();
    const Vector3 camera_pos = camera->get_global_transform().origin;

    auto uncapped = structure.query_visible_splats(
        frustum,
        camera_pos,
        1.0f,
        splats.size()
    );
    CHECK(uncapped.visible_indices.size() > 0);
    CHECK(uncapped.visible_indices.size() == uncapped.lod_weights.size());

    auto capped = structure.query_visible_splats(
        frustum,
        camera_pos,
        1.0f,
        256
    );
    CHECK(capped.visible_indices.size() <= 256);
    CHECK(capped.visible_indices.size() == capped.lod_weights.size());

    memdelete(camera);
}

TEST_CASE("[GaussianSplatting] Hierarchical parallel_build fallback still subdivides the tree") {
    GaussianSplatting::Tests::LODSystemTest fixture;
    Vector<GaussianSplatting::GaussianData> splats = fixture.generate_test_splats(20000, 75.0f);

    GaussianSplatting::HierarchicalSplatStructure structure;
    GaussianSplatting::HierarchicalSplatStructure::BuildParams params;
    params.max_depth = 7;
    params.min_splats_per_node = 16;
    params.parallel_build = true;
    structure.build_hierarchy(splats, params);

    const GaussianSplatting::HierarchicalSplatStructure::TreeStats stats = structure.get_statistics();
    const GaussianSplatting::HierarchicalSplatStructure::OctreeNode *root = structure.get_root();

    REQUIRE(root != nullptr);
    CHECK(stats.total_nodes > 1);
    CHECK(stats.leaf_nodes > 1);
    CHECK(stats.max_depth_reached > 0);
    CHECK_FALSE(root->is_leaf());
}

TEST_CASE("[GaussianSplatting] Hybrid LOD selection enforces cardinality invariants") {
    GaussianSplatting::Tests::LODSystemTest fixture;
    Vector<GaussianSplatting::GaussianData> splats = fixture.generate_test_splats(6000, 60.0f);

    GaussianSplatting::HierarchicalSplatStructure structure;
    GaussianSplatting::HierarchicalSplatStructure::BuildParams params;
    params.max_depth = 7;
    params.min_splats_per_node = 8;
    structure.build_hierarchy(splats, params);

    Camera3D *camera = fixture.create_test_camera(Vector3(0.0f, 0.0f, 320.0f));
    CHECK(camera != nullptr);
    if (camera == nullptr) {
        return;
    }

    GaussianSplatting::AdaptiveLODSystem lod_system;
    GaussianSplatting::AdaptiveLODSystem::LODConfig config;
    config.max_splats_per_frame = 2048;
    lod_system.initialize(config);

    auto selection = lod_system.select_lod_splats(
        splats,
        camera,
        &structure,
        GaussianSplatting::AdaptiveLODSystem::HYBRID
    );

    CHECK(selection.visible_indices.size() > 0);
    CHECK(selection.visible_indices.size() == selection.lod_weights.size());
    CHECK(selection.visible_indices.size() == selection.lod_levels.size());

    memdelete(camera);
}

TEST_CASE("[GaussianSplatting] Hybrid LOD can use hierarchy without aggregated splat vector") {
    GaussianSplatting::Tests::LODSystemTest fixture;
    Vector<GaussianSplatting::GaussianData> splats = fixture.generate_test_splats(2048, 6.0f);

    GaussianSplatting::HierarchicalSplatStructure structure;
    structure.build_hierarchy(splats);

    Camera3D *camera = fixture.create_test_camera(Vector3(0.0f, 0.0f, 8.0f));
    CHECK(camera != nullptr);
    if (camera == nullptr) {
        return;
    }

    GaussianSplatting::AdaptiveLODSystem lod_system;
    GaussianSplatting::AdaptiveLODSystem::LODConfig config;
    config.max_splats_per_frame = 4096;
    lod_system.initialize(config);

    Vector<GaussianSplatting::GaussianData> no_aggregated_splats;
    auto selection = lod_system.select_lod_splats(
        no_aggregated_splats,
        camera,
        &structure,
        GaussianSplatting::AdaptiveLODSystem::HYBRID
    );

    CHECK(selection.visible_indices.size() > 0);

    memdelete(camera);
}

TEST_CASE("[GaussianSplatting] Distance LOD far sampling stays deterministic and spatially stable") {
    GaussianSplatting::Tests::LODSystemTest fixture;
    Vector<GaussianSplatting::GaussianData> splats = fixture.generate_test_splats(20000, 15.0f);
    Camera3D *camera = fixture.create_test_camera(Vector3(0.0f, 0.0f, 220.0f));
    CHECK(camera != nullptr);
    if (camera == nullptr) {
        return;
    }

    GaussianSplatting::AdaptiveLODSystem lod_system;
    GaussianSplatting::AdaptiveLODSystem::LODConfig config;
    config.enable_temporal_coherence = false;
    config.lod0_distance = 5.0f;
    config.lod1_distance = 10.0f;
    config.lod2_distance = 20.0f;
    config.lod3_distance = 30.0f;
    config.cull_distance = 400.0f;
    config.far_lod_keep_ratio = 0.15f;
    config.max_splats_per_frame = 20000;
    lod_system.initialize(config);

    auto frame_a = lod_system.select_lod_splats(
        splats,
        camera,
        nullptr,
        GaussianSplatting::AdaptiveLODSystem::DISTANCE_BASED
    );
    auto frame_b = lod_system.select_lod_splats(
        splats,
        camera,
        nullptr,
        GaussianSplatting::AdaptiveLODSystem::DISTANCE_BASED
    );

    CHECK(frame_a.visible_indices.size() == frame_b.visible_indices.size());
    for (uint32_t i = 0; i < frame_a.visible_indices.size(); i++) {
        CHECK(frame_a.visible_indices[i] == frame_b.visible_indices[i]);
    }

    Transform3D jittered_transform = camera->get_global_transform();
    jittered_transform.origin += Vector3(1.0f, 0.0f, 1.0f);
    camera->set_global_transform(jittered_transform);
    auto frame_c = lod_system.select_lod_splats(
        splats,
        camera,
        nullptr,
        GaussianSplatting::AdaptiveLODSystem::DISTANCE_BASED
    );

    const float churn_ratio = GaussianSplatting::Tests::compute_visibility_churn_ratio(
        frame_a.visible_indices,
        frame_c.visible_indices);
    CHECK(churn_ratio < 0.05f);

    const float observed_keep_ratio = float(frame_a.visible_indices.size()) / float(splats.size());
    CHECK(observed_keep_ratio > config.far_lod_keep_ratio * 0.5f);
    CHECK(observed_keep_ratio < MIN(1.0f, config.far_lod_keep_ratio * 1.5f));

    memdelete(camera);
}

TEST_CASE("[GaussianSplatting] Temporal churn metrics scale with camera motion") {
    GaussianSplatting::Tests::LODSystemTest fixture;
    Vector<GaussianSplatting::GaussianData> splats = fixture.generate_test_splats(18000, 80.0f);
    Camera3D *camera = fixture.create_test_camera(Vector3(0.0f, 0.0f, 35.0f));
    CHECK(camera != nullptr);
    if (camera == nullptr) {
        return;
    }

    GaussianSplatting::AdaptiveLODSystem lod_system;
    GaussianSplatting::AdaptiveLODSystem::LODConfig config;
    config.enable_temporal_coherence = true;
    config.smooth_transitions = true;
    config.lod0_distance = 20.0f;
    config.lod1_distance = 45.0f;
    config.lod2_distance = 90.0f;
    config.lod3_distance = 140.0f;
    config.cull_distance = 260.0f;
    config.max_splats_per_frame = 18000;
    lod_system.initialize(config);

    auto frame_a = lod_system.select_lod_splats(
        splats,
        camera,
        nullptr,
        GaussianSplatting::AdaptiveLODSystem::DISTANCE_BASED
    );
    CHECK(Math::is_equal_approx(frame_a.stats.temporal_visibility_churn_ratio, 0.0f, 0.0001f));

    Transform3D small_move = camera->get_global_transform();
    small_move.origin = Vector3(0.0f, 0.0f, 38.0f);
    camera->set_global_transform(small_move);
    auto frame_b = lod_system.select_lod_splats(
        splats,
        camera,
        nullptr,
        GaussianSplatting::AdaptiveLODSystem::DISTANCE_BASED
    );
    CHECK(frame_b.stats.temporal_visibility_churn_ratio < 0.5f);

    Transform3D large_move = camera->get_global_transform();
    large_move.origin = Vector3(0.0f, 0.0f, 190.0f);
    camera->set_global_transform(large_move);
    auto frame_c = lod_system.select_lod_splats(
        splats,
        camera,
        nullptr,
        GaussianSplatting::AdaptiveLODSystem::DISTANCE_BASED
    );

    CHECK(frame_c.stats.temporal_visibility_churn_ratio > frame_b.stats.temporal_visibility_churn_ratio);
    CHECK((frame_c.stats.temporal_added + frame_c.stats.temporal_removed) >=
            (frame_b.stats.temporal_added + frame_b.stats.temporal_removed));

    memdelete(camera);
}
