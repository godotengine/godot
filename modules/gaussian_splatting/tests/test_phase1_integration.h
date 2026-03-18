/**************************************************************************/
/*  test_phase1_integration.h                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/gpu_memory_stream.h"
#include "../renderer/gpu_sorter.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "servers/rendering_server.h"
#include "servers/rendering/rendering_device.h"
#include "core/os/os.h"
#include "core/math/math_defs.h"
#include "core/math/random_number_generator.h"
#include "tests/test_macros.h"

namespace TestGaussianSplatting {

// Performance baselines for Phase 1
struct Phase1Baselines {
	static constexpr float MAX_FRAME_TIME_100K = 16.67f; // 60 FPS
	static constexpr float MAX_SORT_TIME_100K = 2.0f;    // GPU sorting budget
	static constexpr float MAX_UPLOAD_TIME = 1.0f;       // Streaming budget
	static constexpr float MAX_GPU_MEMORY_MB = 500.0f;   // Memory budget
	static constexpr float MAX_INIT_TIME_MS = 1000.0f;   // Initialization time
};

// Test data generator helper
class TestDataGenerator {
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

	static LocalVector<Gaussian> generate_clustered_splats(uint32_t count, uint32_t num_clusters = 10) {
		LocalVector<Gaussian> splats;
		splats.resize(count);

		RandomNumberGenerator rng;
		rng.set_seed(42);

		// Generate cluster centers
		LocalVector<Vector3> centers;
		for (uint32_t i = 0; i < num_clusters; i++) {
			centers.push_back(Vector3(
				rng.randf_range(-20.0f, 20.0f),
				rng.randf_range(-20.0f, 20.0f),
				rng.randf_range(-20.0f, 20.0f)
			));
		}

		// Generate splats around clusters
		for (uint32_t i = 0; i < count; i++) {
			Gaussian &g = splats[i];

			// Pick a cluster
			uint32_t cluster_idx = i % num_clusters;
			Vector3 center = centers[cluster_idx];

			// Position near cluster center
			g.position = center + Vector3(
				rng.randf_range(-2.0f, 2.0f),
				rng.randf_range(-2.0f, 2.0f),
				rng.randf_range(-2.0f, 2.0f)
			);

			float scale = rng.randf_range(0.05f, 0.5f);
			g.scale = Vector3(scale, scale, scale);
			g.rotation = Quaternion();
			g.opacity = rng.randf_range(0.5f, 1.0f);
			g.sh_dc = Color(
				0.5f + 0.5f * sin(cluster_idx),
				0.5f + 0.5f * cos(cluster_idx),
				0.5f + 0.5f * sin(cluster_idx * 2),
				g.opacity
			);
			g.normal = Vector3(0, 1, 0);
			g.area = scale * scale * static_cast<float>(Math::PI);
		}

		return splats;
	}
};

// NOTE: create_storage_buffer template is defined in test_gpu_sorting.h
// Do not duplicate it here to avoid ODR violations

TEST_CASE("[GaussianSplatting] Phase 1 Integration - Basic Components") {
	// Get or create rendering device
	RenderingDevice *rd = nullptr;
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		rd = manager->get_primary_rendering_device();
	}
	if (!rd) {
		RenderingServer *rs = RenderingServer::get_singleton();
		if (rs) {
			rd = rs->create_local_rendering_device();
		}
	}

	if (!rd) {
		WARN("Skipping Phase 1 integration tests - no RenderingDevice available");
		return;
	}

	SUBCASE("GaussianData basic operations") {
		Ref<::GaussianData> data;
		data.instantiate();

		LocalVector<Gaussian> test_splats = TestDataGenerator::generate_uniform_splats(1000);
		data->set_gaussians(test_splats);
		CHECK(data->get_count() == 1000);

		AABB aabb = data->compute_aabb();
		CHECK(aabb.size.length() > 0); // Non-empty bounding box
	}

	SUBCASE("Memory stream initialization") {
		Ref<GaussianMemoryStream> memory_stream;
		memory_stream.instantiate();

		Error err = memory_stream->initialize(rd, 10000, 256);
		CHECK(err == OK);
		CHECK(memory_stream->get_max_gaussians() == 10000);
	}

	SUBCASE("GPU sorter initialization") {
		Ref<BitonicSort> sorter;
		sorter.instantiate();

		Error err = sorter->initialize(rd, 10000);
		CHECK(err == OK);
		CHECK(sorter->get_max_elements() == 10000);
	}

	SUBCASE("Memory streaming with sorting") {
		Ref<GaussianMemoryStream> memory_stream;
		memory_stream.instantiate();

		Ref<BitonicSort> sorter;
		sorter.instantiate();

		// Initialize
		CHECK(memory_stream->initialize(rd, 10000, 256) == OK);
		CHECK(sorter->initialize(rd, 10000) == OK);

		// Generate test data
		LocalVector<Gaussian> splats = TestDataGenerator::generate_uniform_splats(1000);

		// Test streaming
		memory_stream->begin_frame(0);
		Error err = memory_stream->stream_gaussians_immediate(splats);
		CHECK(err == OK);
		memory_stream->end_frame();
		CHECK(memory_stream->is_upload_complete());

		// Verify we can get the buffer
		RID buffer = memory_stream->get_current_gpu_buffer();
		CHECK(buffer.is_valid());
	}

	SUBCASE("Performance scaling test") {
		const uint32_t test_counts[] = {1000, 5000, 10000};

		for (uint32_t count : test_counts) {
			Ref<GaussianMemoryStream> memory_stream;
			memory_stream.instantiate();

			CHECK(memory_stream->initialize(rd, count, 256) == OK);

			// Generate test data
			LocalVector<Gaussian> splats = TestDataGenerator::generate_uniform_splats(count);

			// Measure upload time
			uint64_t upload_start = OS::get_singleton()->get_ticks_usec();
			memory_stream->begin_frame(0);
			Error err = memory_stream->stream_gaussians_immediate(splats);
			CHECK(err == OK);
			memory_stream->end_frame();
			float upload_time = (OS::get_singleton()->get_ticks_usec() - upload_start) / 1000.0f;

			// Verify reasonable performance (very conservative limits)
			CHECK_MESSAGE(upload_time < 100.0f,
				vformat("Upload of %d splats took %.2fms, expected < 100ms", count, upload_time));
		}
	}
}

} // namespace TestGaussianSplatting
