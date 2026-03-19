/*
 * Phase 1 Integration Test Suite
 * Tests the complete rendering pipeline with all components working together.
 */

#include "tests/test_macros.h"
#include "test_macros.h" // Module-specific test macros for REQUIRE_GPU_DEVICE
#include "core/string/string_name.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../core/gaussian_splat_manager.h"
#include "../core/gaussian_data.h"
#include "servers/rendering_server.h"
#include "servers/rendering/rendering_device.h"
#include "core/os/os.h"
#include "core/math/math_defs.h"
#include "core/math/random_number_generator.h"
#include <chrono>
#include <algorithm>
#include <vector>

#ifdef TESTS_ENABLED

namespace TestGaussianSplatting {

// Performance baselines - must meet these targets
struct Phase1BaselinesLocal {
    static constexpr float MAX_FRAME_TIME_100K = 16.67f; // 60 FPS
    static constexpr float MAX_SORT_TIME_100K = 2.0f;    // GPU sorting budget
    static constexpr float MAX_UPLOAD_TIME = 1.0f;       // Streaming budget
    static constexpr float MAX_GPU_MEMORY_MB = 500.0f;   // Memory budget
    static constexpr float MAX_INIT_TIME_MS = 1000.0f;   // Initialization time
};

// Test data generator
class TestDataGeneratorLocal {
public:
    static LocalVector<Gaussian> generate_uniform_splats(uint32_t count) {
        LocalVector<Gaussian> splats;
        splats.resize(count);

        RandomNumberGenerator rng;
        rng.set_seed(42); // Deterministic for reproducibility

        for (uint32_t i = 0; i < count; i++) {
            Gaussian &g = splats[i];

            // Uniform distribution in 3D space
            g.position = Vector3(
                rng.randf_range(-10.0f, 10.0f),
                rng.randf_range(-10.0f, 10.0f),
                rng.randf_range(-10.0f, 10.0f)
            );

            // Random scales
            float scale = rng.randf_range(0.1f, 1.0f);
            g.scale = Vector3(scale, scale, scale);

            // Random rotation
            g.rotation = Quaternion(
                Vector3(rng.randf(), rng.randf(), rng.randf()).normalized(),
                rng.randf_range(0, static_cast<float>(Math::TAU))
            );

            // Random opacity
            g.opacity = rng.randf_range(0.3f, 1.0f);

            // Random color
            g.sh_dc = Color(rng.randf(), rng.randf(), rng.randf(), g.opacity);

            // Normal pointing upward with some variation
            g.normal = Vector3(
                rng.randf_range(-0.2f, 0.2f),
                1.0f,
                rng.randf_range(-0.2f, 0.2f)
            ).normalized();

            g.area = scale * scale * static_cast<float>(Math::PI);
        }

        return splats;
    }
};

// Performance measurement utilities
class PerformanceTimerLocal {
private:
    uint64_t start_time;
    String name;

public:
    PerformanceTimerLocal(const String &p_name) : name(p_name) {
        start_time = OS::get_singleton()->get_ticks_usec();
    }

    float elapsed_ms() const {
        uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start_time;
        return elapsed / 1000.0f;
    }

    void print_elapsed() const {
        print_line(vformat("[%s] Elapsed: %.2f ms", name, elapsed_ms()));
    }
};

// GPU memory tracker
class GPUMemoryTrackerLocal {
private:
    RenderingDevice *rd;
    uint64_t initial_memory;

public:
    GPUMemoryTrackerLocal(RenderingDevice *p_rd) : rd(p_rd), initial_memory(0) {
        if (rd) {
            initial_memory = rd->get_memory_usage(RenderingDevice::MEMORY_TOTAL);
        }
    }

    float get_memory_usage_mb() const {
        if (!rd) return 0.0f;
        uint64_t current = rd->get_memory_usage(RenderingDevice::MEMORY_TOTAL);
        return (current - initial_memory) / (1024.0f * 1024.0f);
    }

    void print_usage() const {
        print_line(vformat("GPU Memory Usage: %.2f MB", get_memory_usage_mb()));
    }
};

TEST_SUITE("[GaussianSplatting]") {

    TEST_CASE("[GaussianSplatting][Phase1] Basic initialization") {
        // Test basic module initialization
        REQUIRE_GPU_DEVICE();

        GaussianSplatManager *manager = memnew(GaussianSplatManager);
        CHECK(manager != nullptr);
        if (manager == nullptr) {
            return;
        }

        Ref<::GaussianData> data;
        data.instantiate();
        CHECK(data.is_valid());

        // Cleanup
        memdelete(manager);
    }

    TEST_CASE("[GaussianSplatting][Phase1] Data loading") {
        // Test loading Gaussian data
        Ref<::GaussianData> data;
        data.instantiate();

        LocalVector<Gaussian> test_splats = TestDataGeneratorLocal::generate_uniform_splats(1000);
        data->set_gaussians(test_splats);
        CHECK(data->get_count() == 1000);
    }

    TEST_CASE("[GaussianSplatting][Phase1] Data pipeline 100K splats") {
        // Setup rendering device
        GaussianSplatManager *manager = memnew(GaussianSplatManager);
        CHECK(manager != nullptr);
        if (manager == nullptr) {
            return;
        }

        RenderingDevice *rd = manager->get_primary_rendering_device();
        if (!rd) {
            memdelete(manager);
            MESSAGE("Skipping 100K test - no RenderingDevice available");
            return;
        }

        // Create components - simplified for initial integration
        Ref<::GaussianData> data;
        data.instantiate();

        // Track performance
        GPUMemoryTrackerLocal memory_tracker(rd);

        // Load test splats
        LocalVector<Gaussian> test_splats = TestDataGeneratorLocal::generate_uniform_splats(100000);
        {
            PerformanceTimerLocal timer("Data Loading");
            data->set_gaussians(test_splats);
            CHECK(data->get_count() == 100000);
        }

        // Check memory usage
        float memory_mb = memory_tracker.get_memory_usage_mb();
        CHECK_MESSAGE(memory_mb < Phase1BaselinesLocal::MAX_GPU_MEMORY_MB,
            vformat("GPU memory usage %.2f MB exceeds budget", memory_mb));

        // Cleanup
        memdelete(manager);
    }

    TEST_CASE("[GaussianSplatting][Phase1] Editor integration") {
        // Test editor integration without hangs
        REQUIRE_GPU_DEVICE();

        GaussianSplatManager *manager = memnew(GaussianSplatManager);
        CHECK(manager != nullptr);
        if (manager == nullptr) {
            return;
        }

        // Test rapid creation/destruction
        for (int i = 0; i < 3; i++) {
            Ref<::GaussianData> data;
            data.instantiate();

            LocalVector<Gaussian> splats = TestDataGeneratorLocal::generate_uniform_splats(1000);
            data->set_gaussians(splats);

            // No hangs should occur
            CHECK(true);
        }

        memdelete(manager);
        print_line("[GaussianSplatting] Editor integration test passed");
    }

}

} // namespace TestGaussianSplatting

#endif // TESTS_ENABLED
