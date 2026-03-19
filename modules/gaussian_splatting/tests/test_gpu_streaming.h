/**************************************************************************/
/*  test_gpu_streaming.h                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "../renderer/gpu_memory_stream.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "core/math/math_defs.h"
#include "core/math/random_number_generator.h"
#include "core/os/os.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"
#include "tests/test_macros.h"
#include <atomic>

namespace TestGaussianSplatting {

struct ConcurrentStreamingAssertionsContext {
	Ref<GaussianMemoryStream> stream;
	uint32_t uploads_per_thread = 0;
	uint32_t gaussians_per_upload = 0;
	std::atomic<int> successful_uploads{0};
	std::atomic<int> failed_uploads{0};
	Semaphore begin;
	Semaphore done;
};

static void _concurrent_streaming_assertions_worker(void *p_userdata) {
	ConcurrentStreamingAssertionsContext *ctx = static_cast<ConcurrentStreamingAssertionsContext *>(p_userdata);
	if (!ctx || !ctx->stream.is_valid()) {
		return;
	}

	ctx->begin.wait();
	for (uint32_t i = 0; i < ctx->uploads_per_thread; i++) {
		LocalVector<Gaussian> gaussians;
		gaussians.resize(ctx->gaussians_per_upload);
		for (uint32_t j = 0; j < ctx->gaussians_per_upload; j++) {
			Gaussian &g = gaussians[j];
			g.position = Vector3(float(j) * 0.01f, float(i), 0.0f);
			g.scale = Vector3(0.5f, 0.5f, 0.5f);
			g.rotation = Quaternion();
			g.opacity = 1.0f;
			g.sh_dc = Color(1.0f, 0.6f, 0.3f, 1.0f);
			g.normal = Vector3(0.0f, 1.0f, 0.0f);
			g.area = 0.25f;
		}

		const Error upload_err = ctx->stream->stream_gaussians_async(gaussians);
		if (upload_err == OK) {
			ctx->successful_uploads.fetch_add(1, std::memory_order_relaxed);
		} else {
			ctx->failed_uploads.fetch_add(1, std::memory_order_relaxed);
		}
	}
	ctx->done.post();
}

TEST_CASE("[GaussianSplatting] GPU Memory Streaming") {
	// Get or create rendering device
	RenderingDevice *rd = RenderingDevice::get_singleton();
	if (!rd) {
		RenderingServer *rs = RenderingServer::get_singleton();
		if (rs) {
			rd = rs->create_local_rendering_device();
		}
	}

	// Skip GPU tests if no rendering device available
	if (!rd) {
		MESSAGE("Skipping GPU streaming tests - no RenderingDevice available");
		return;
	}

	SUBCASE("Initialize memory stream") {
		Ref<GaussianMemoryStream> stream;
		stream.instantiate();

		Error err = stream->initialize(rd, 10000, 256);
		CHECK(err == OK);
		CHECK(stream->get_max_gaussians() == 10000);
	}

	SUBCASE("Upload gaussians to GPU") {
		Ref<GaussianMemoryStream> stream;
		stream.instantiate();

		Error err = stream->initialize(rd, 1000, 256);
		CHECK(err == OK);
		if (err != OK) {
			return;
		}

		// Create test data
		LocalVector<Gaussian> splats;
		splats.resize(100);

		RandomNumberGenerator rng;
		rng.set_seed(42);

		for (int i = 0; i < 100; i++) {
			Gaussian &g = splats[i];
			g.position = Vector3(
				rng.randf_range(-5.0f, 5.0f),
				rng.randf_range(-5.0f, 5.0f),
				rng.randf_range(-5.0f, 5.0f)
			);
			g.scale = Vector3(0.5f, 0.5f, 0.5f);
			g.rotation = Quaternion();
			g.opacity = 1.0f;
			g.sh_dc = Color(1, 0, 0, 1);
			g.normal = Vector3(0, 1, 0);
			g.area = 0.785f;
		}

		// Test upload
		stream->begin_frame(0);

		err = stream->stream_gaussians_immediate(splats);
		CHECK(err == OK);

		stream->end_frame();
		CHECK(stream->is_upload_complete());
	}

	SUBCASE("Triple buffering") {
		Ref<GaussianMemoryStream> stream;
		stream.instantiate();

		Error err = stream->initialize(rd, 500, 256);
		CHECK(err == OK);
		if (err != OK) {
			return;
		}

		// Create test data for multiple frames
		LocalVector<Gaussian> frame_data[3];
		for (int frame = 0; frame < 3; frame++) {
			frame_data[frame].resize(100);
			for (int i = 0; i < 100; i++) {
				Gaussian &g = frame_data[frame][i];
				g.position = Vector3(frame, i, 0);
				g.scale = Vector3(1, 1, 1);
				g.rotation = Quaternion();
				g.opacity = 1.0f;
				g.sh_dc = Color(1, 1, 1, 1);
				g.normal = Vector3(0, 1, 0);
				g.area = static_cast<float>(Math::PI);
			}
		}

		// Test triple buffering
		for (int frame = 0; frame < 6; frame++) {
			stream->begin_frame(frame);

			int data_idx = frame % 3;
			err = stream->stream_gaussians_immediate(frame_data[data_idx]);
			CHECK(err == OK);

			stream->end_frame();
			CHECK(stream->is_upload_complete());

			// Verify we get valid buffer for each frame
			RID current = stream->get_current_gpu_buffer();
			CHECK(current.is_valid());
		}
	}

	SUBCASE("Invalid initialization") {
		Ref<GaussianMemoryStream> stream;
		stream.instantiate();

		// Test with null device
		Error err = stream->initialize(nullptr, 1000, 256);
		CHECK(err == ERR_INVALID_PARAMETER);
		CHECK(stream->get_max_gaussians() == 1000000); // Default value, not changed

		// Test with zero capacity
		err = stream->initialize(rd, 0, 256);
		CHECK(err == ERR_INVALID_PARAMETER);
	}

	SUBCASE("Concurrent async uploads keep deterministic accounting") {
#ifndef THREADS_ENABLED
		MESSAGE("Skipping concurrent upload assertions - THREADS_ENABLED is not enabled");
		return;
#endif

		Ref<GaussianMemoryStream> stream;
		stream.instantiate();

		const uint32_t gaussians_per_upload = 256;
		const uint32_t uploads_per_thread = 8;
		Error err = stream->initialize(rd, gaussians_per_upload * uploads_per_thread * 2, 64);
		CHECK(err == OK);
		if (err != OK) {
			return;
		}

		ConcurrentStreamingAssertionsContext ctx;
		ctx.stream = stream;
		ctx.uploads_per_thread = uploads_per_thread;
		ctx.gaussians_per_upload = gaussians_per_upload;

		Thread worker_a;
		Thread worker_b;
		worker_a.start(_concurrent_streaming_assertions_worker, &ctx);
		worker_b.start(_concurrent_streaming_assertions_worker, &ctx);
		const bool worker_a_started = worker_a.is_started();
		const bool worker_b_started = worker_b.is_started();
		CHECK(worker_a_started);
		CHECK(worker_b_started);
		if (!worker_a_started || !worker_b_started) {
			if (worker_a_started) {
				ctx.begin.post();
				worker_a.wait_to_finish();
			}
			if (worker_b_started) {
				ctx.begin.post();
				worker_b.wait_to_finish();
			}
			stream->shutdown();
			return;
		}

		ctx.begin.post(2);
		ctx.done.wait();
		ctx.done.wait();
		worker_a.wait_to_finish();
		worker_b.wait_to_finish();

		stream->wait_for_all_uploads();
		CHECK(stream->is_upload_complete());

		const int successful_uploads = ctx.successful_uploads.load(std::memory_order_relaxed);
		const int failed_uploads = ctx.failed_uploads.load(std::memory_order_relaxed);
		const int total_uploads = int(uploads_per_thread * 2);
		CHECK(successful_uploads + failed_uploads == total_uploads);
		CHECK(successful_uploads > 0);

		const StreamingStats stats = stream->get_stats();
		CHECK(int(stats.buffer_switches) == successful_uploads);
		const uint64_t expected_bytes_uploaded = uint64_t(successful_uploads) *
				uint64_t(gaussians_per_upload) * uint64_t(sizeof(PackedGaussian));
		CHECK(stats.total_bytes_uploaded == expected_bytes_uploaded);
	}
}

TEST_CASE("[GaussianSplatting] GPU Memory Streaming Performance") {
	RenderingDevice *rd = RenderingDevice::get_singleton();
	if (!rd) {
		RenderingServer *rs = RenderingServer::get_singleton();
		if (rs) {
			rd = rs->create_local_rendering_device();
		}
	}

	if (!rd) {
		MESSAGE("Skipping GPU performance tests - no RenderingDevice available");
		return;
	}

	SUBCASE("Upload performance scaling") {
		const uint32_t test_sizes[] = {100, 1000, 10000, 50000};

		for (uint32_t size : test_sizes) {
			Ref<GaussianMemoryStream> stream;
			stream.instantiate();

			Error err = stream->initialize(rd, size, 256);
			CHECK(err == OK);
			if (err != OK) {
				return;
			}

			// Create test data
			LocalVector<Gaussian> splats;
			splats.resize(size);

			for (uint32_t i = 0; i < size; i++) {
				Gaussian &g = splats[i];
				g.position = Vector3(i, 0, 0);
				g.scale = Vector3(1, 1, 1);
				g.rotation = Quaternion();
				g.opacity = 1.0f;
				g.sh_dc = Color(1, 1, 1, 1);
				g.normal = Vector3(0, 1, 0);
				g.area = static_cast<float>(Math::PI);
			}

			// Measure upload time
			uint64_t start = OS::get_singleton()->get_ticks_usec();

			stream->begin_frame(0);

			err = stream->stream_gaussians_immediate(splats);
			CHECK(err == OK);

			stream->end_frame();

			uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start;
			float ms = elapsed / 1000.0f;

			// Performance targets
			if (size <= 10000) {
				CHECK_MESSAGE(ms < 10.0f,
					vformat("Upload of %d splats took %.2fms, expected < 10ms", size, ms));
			} else if (size <= 50000) {
				CHECK_MESSAGE(ms < 50.0f,
					vformat("Upload of %d splats took %.2fms, expected < 50ms", size, ms));
			}
		}
	}
}

TEST_CASE("[GaussianSplatting] Stage-B instance depth culling toggles") {
	GaussianSplatManager *manager_owner = nullptr;
	GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
	if (!manager) {
		manager_owner = memnew(GaussianSplatManager);
		manager = manager_owner;
	}

	CHECK(manager != nullptr);
	if (manager == nullptr) {
		return;
	}

	RenderingDevice *primary_device = manager->get_primary_rendering_device();
	if (primary_device == nullptr) {
		MESSAGE("Skipping Stage-B culling test - primary RenderingDevice unavailable");
		if (manager_owner) {
			memdelete(manager_owner);
		}
		return;
	}

	Ref<GaussianSplatRenderer> renderer;
	renderer.instantiate(primary_device);
	CHECK(renderer.is_valid());
	if (!renderer.is_valid()) {
		if (manager_owner) {
			memdelete(manager_owner);
		}
		return;
	}

	const uint32_t total_gaussians = 4096;
	LocalVector<Gaussian> gaussians;
	gaussians.resize(total_gaussians);
	const uint32_t near_band_count = total_gaussians / 2;
	for (uint32_t i = 0; i < total_gaussians; i++) {
		Gaussian &g = gaussians[i];
		g = Gaussian{};
		const bool in_near_band = i < near_band_count;
		const float band_x = in_near_band ? -0.8f : 28.0f;
		const float local_x = float(i % 64) * 0.02f;
		const float local_y = (float((i / 64) % 64) - 32.0f) * 0.03f;
		g.position = Vector3(band_x + local_x, local_y, -6.0f);
		g.scale = Vector3(0.03f, 0.03f, 0.03f);
		g.rotation = Quaternion();
		g.opacity = 1.0f;
		g.sh_dc = Color(1.0f, 1.0f, 1.0f, 1.0f);
		g.normal = Vector3(0.0f, 1.0f, 0.0f);
		g.area = 0.01f;
	}

	Ref<::GaussianData> data;
	data.instantiate();
	data->set_gaussians(gaussians);

	Error set_data_err = renderer->set_gaussian_data(data);
	CHECK(set_data_err == OK);
	if (set_data_err != OK) {
		renderer.unref();
		if (manager_owner) {
			memdelete(manager_owner);
		}
		return;
	}

	renderer->set_lod_enabled(true);
	renderer->set_lod_bias(1.0f);
	renderer->set_lod_min_screen_size(0.0f);
	renderer->set_lod_max_distance(0.0f);
	renderer->set_tiny_splat_screen_radius(0.0f);
	renderer->set_frustum_culling(false);

	Transform3D cam_transform;
	Projection projection;
	projection.set_perspective(60.0f, 1.0f, 0.1f, 200.0f);

	auto render_sample = [&](int p_frames) {
		uint32_t visible = 0;
		for (int i = 0; i < p_frames; i++) {
			const bool rendered = renderer->render_for_view(cam_transform, projection, RID(), Size2i(512, 512));
			CHECK(rendered);
			if (!rendered) {
				break;
			}
			visible = renderer->get_visible_splat_count();
			OS::get_singleton()->delay_usec(500);
		}
		return visible;
	};

	uint32_t baseline_visible = 0;
	for (int i = 0; i < 180; i++) {
		const uint32_t visible = render_sample(1);
		if (renderer->has_instance_pipeline_buffers() && renderer->has_rendered_content() && visible > 0) {
			baseline_visible = visible;
			break;
		}
	}

	if (baseline_visible == 0) {
		MESSAGE("Skipping Stage-B culling test - instance pipeline did not become ready");
		renderer.unref();
		if (manager_owner) {
			memdelete(manager_owner);
		}
		return;
	}

	renderer->set_frustum_culling(true);
	const uint32_t frustum_visible = render_sample(6);

	renderer->set_frustum_culling(false);
	renderer->set_tiny_splat_screen_radius(64.0f);
	const uint32_t screen_visible = render_sample(6);

	renderer->set_tiny_splat_screen_radius(0.0f);
	renderer->set_lod_max_distance(8.0f);
	const uint32_t distance_visible = render_sample(6);

	CHECK(frustum_visible < baseline_visible);
	CHECK(screen_visible < baseline_visible);
	CHECK(distance_visible < baseline_visible);

	renderer.unref();
	if (manager_owner) {
		memdelete(manager_owner);
	}
}

} // namespace TestGaussianSplatting
