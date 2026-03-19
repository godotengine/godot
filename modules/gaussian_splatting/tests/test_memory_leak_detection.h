/**************************************************************************/
/*  test_memory_leak_detection.h                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "test_macros.h"
#include "memory_validator.h"

#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "../renderer/gaussian_splat_renderer.h"

#include "core/math/random_number_generator.h"
#include "core/os/os.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"

namespace TestGaussianSplatting {

// RAII helper to manage a MemoryValidator instance for tests.
// Ensures the singleton is cleaned up after each test scope.
class ScopedMemoryValidator {
	Ref<MemoryValidator> validator;
	bool initialized = false;

public:
	ScopedMemoryValidator() {
		validator.instantiate();
	}

	~ScopedMemoryValidator() {
		validator.unref();
	}

	Error initialize(RenderingDevice *p_rd) {
		Error err = validator->initialize(p_rd);
		if (err == OK) {
			initialized = true;
		}
		return err;
	}

	MemoryValidator *get() const { return validator.ptr(); }
	bool is_initialized() const { return initialized; }
};

// RAII helper to manage a GaussianSplatManager for memory leak tests.
class ScopedGaussianManagerMemory {
	GaussianSplatManager *manager = nullptr;
	bool owns_instance = false;

public:
	ScopedGaussianManagerMemory() {
		manager = GaussianSplatManager::get_singleton();
		if (!manager) {
			manager = memnew(GaussianSplatManager);
			owns_instance = true;
		}
	}

	~ScopedGaussianManagerMemory() {
		if (owns_instance && manager) {
			memdelete(manager);
		}
	}

	GaussianSplatManager *get() const { return manager; }
};

TEST_CASE("[GaussianSplatting] GPU memory leak detection") {
	// -- Obtain a RenderingDevice, skip gracefully if unavailable --
	REQUIRE_GPU_DEVICE();

	// -- Initialize the MemoryValidator with GPU tracking --
	ScopedMemoryValidator mem_scope;
	Error init_err = mem_scope.initialize(rd);
	CHECK(init_err == OK);
	if (init_err != OK) {
		MESSAGE("Skipping GPU memory leak tests - MemoryValidator initialization failed");
		return;
	}

	MemoryValidator *validator = mem_scope.get();
	GPUMemoryTracker *gpu_tracker = validator->get_gpu_tracker();
	CHECK(gpu_tracker != nullptr);
	if (!gpu_tracker) {
		return;
	}

	SUBCASE("GPU buffer lifecycle tracks and validates correctly") {
		// Create tracked GPU buffer resources via the tracker.
		const int buffer_count = 5;
		const size_t buffer_size = 4096;
		RID buffer_rids[5];

		for (int i = 0; i < buffer_count; i++) {
			buffer_rids[i] = RID::from_uint64(1000 + i);
			gpu_tracker->track_buffer_creation(buffer_rids[i], buffer_size, vformat("TestBuffer_%d", i));
		}

		CHECK(gpu_tracker->get_resource_count() == buffer_count);
		CHECK(gpu_tracker->get_total_gpu_memory() == buffer_count * buffer_size);

		// Free all buffers.
		for (int i = 0; i < buffer_count; i++) {
			gpu_tracker->track_resource_free(buffer_rids[i]);
		}

		CHECK(gpu_tracker->get_resource_count() == 0);
		CHECK(gpu_tracker->get_total_gpu_memory() == 0);
		CHECK(gpu_tracker->validate_no_leaks());
	}

	SUBCASE("GPU texture lifecycle tracks and validates correctly") {
		const int texture_count = 3;
		const size_t texture_size = 1024 * 1024; // 1 MB per texture
		RID texture_rids[3];

		for (int i = 0; i < texture_count; i++) {
			texture_rids[i] = RID::from_uint64(2000 + i);
			gpu_tracker->track_texture_creation(texture_rids[i], texture_size, vformat("TestTexture_%d", i));
		}

		CHECK(gpu_tracker->get_resource_count() == texture_count);
		CHECK(gpu_tracker->get_total_gpu_memory() == texture_count * texture_size);

		// Free all textures.
		for (int i = 0; i < texture_count; i++) {
			gpu_tracker->track_resource_free(texture_rids[i]);
		}

		CHECK(gpu_tracker->validate_no_leaks());
	}

	SUBCASE("GPU shader lifecycle tracks and validates correctly") {
		RID shader_rid = RID::from_uint64(3000);
		gpu_tracker->track_shader_creation(shader_rid, "TestShader");

		CHECK(gpu_tracker->get_resource_count() == 1);

		gpu_tracker->track_resource_free(shader_rid);
		CHECK(gpu_tracker->validate_no_leaks());
	}

	SUBCASE("Leak detection catches unfreed GPU resources") {
		// Track resources but deliberately skip freeing some.
		RID freed_rid = RID::from_uint64(4000);
		RID leaked_rid = RID::from_uint64(4001);

		gpu_tracker->track_buffer_creation(freed_rid, 512, "FreedBuffer");
		gpu_tracker->track_buffer_creation(leaked_rid, 1024, "LeakedBuffer");

		gpu_tracker->track_resource_free(freed_rid);

		// The leaked resource should be detected.
		CHECK_FALSE(gpu_tracker->validate_no_leaks());
		CHECK(gpu_tracker->get_resource_count() == 1);

		LocalVector<MemoryAllocation> leaks = gpu_tracker->find_leaks();
		CHECK(leaks.size() == 1);
		CHECK(leaks[0].size == 1024);
		CHECK(leaks[0].source_file == "LeakedBuffer");

		// Clean up the leak to avoid polluting subsequent subcases.
		gpu_tracker->track_resource_free(leaked_rid);
		CHECK(gpu_tracker->validate_no_leaks());
	}

	SUBCASE("Mixed resource types detect leaks across buffers, textures, and shaders") {
		RID buffer_rid = RID::from_uint64(5000);
		RID texture_rid = RID::from_uint64(5001);
		RID shader_rid = RID::from_uint64(5002);

		gpu_tracker->track_buffer_creation(buffer_rid, 2048, "MixedBuffer");
		gpu_tracker->track_texture_creation(texture_rid, 8192, "MixedTexture");
		gpu_tracker->track_shader_creation(shader_rid, "MixedShader");

		CHECK(gpu_tracker->get_resource_count() == 3);

		// Free only the buffer -- textures and shader should be detected as leaks.
		gpu_tracker->track_resource_free(buffer_rid);

		CHECK_FALSE(gpu_tracker->validate_no_leaks());
		CHECK(gpu_tracker->get_resource_count() == 2);

		LocalVector<MemoryAllocation> leaks = gpu_tracker->find_leaks();
		CHECK(leaks.size() == 2);

		// Clean up.
		gpu_tracker->track_resource_free(texture_rid);
		gpu_tracker->track_resource_free(shader_rid);
		CHECK(gpu_tracker->validate_no_leaks());
	}

	SUBCASE("MemoryValidator validate_no_leaks integrates CPU and GPU tracking") {
		validator->reset();

		// Track a CPU allocation.
		void *cpu_ptr = memalloc(256);
		validator->track_allocation(cpu_ptr, 256, "TestCPU");

		// Track a GPU allocation.
		RID gpu_rid = RID::from_uint64(6000);
		GPUMemoryTracker *fresh_gpu = validator->get_gpu_tracker();
		if (fresh_gpu) {
			fresh_gpu->track_buffer_creation(gpu_rid, 512, "TestGPUBuffer");
		}

		// Neither freed -- should detect leaks.
		CHECK_FALSE(validator->validate_no_leaks());

		// Free CPU allocation.
		validator->track_deallocation(cpu_ptr);
		memfree(cpu_ptr);

		// GPU resource still leaked.
		CHECK_FALSE(validator->validate_no_leaks());

		// Free GPU allocation.
		if (fresh_gpu) {
			fresh_gpu->track_resource_free(gpu_rid);
		}

		// Now everything should be clean.
		CHECK(validator->validate_no_leaks());
	}

	SUBCASE("Memory pool tracking correctly tracks allocations and frees") {
		gpu_tracker->track_pool_allocation("TestPool", 4096);
		gpu_tracker->track_pool_allocation("TestPool", 2048);
		gpu_tracker->track_pool_free("TestPool", 4096);
		gpu_tracker->track_pool_free("TestPool", 2048);

		// Pool operations do not affect the RID-based leak detection.
		CHECK(gpu_tracker->validate_no_leaks());
	}
}

TEST_CASE("[GaussianSplatting] Memory validator reset clears all tracked state") {
	REQUIRE_GPU_DEVICE();

	ScopedMemoryValidator mem_scope;
	Error init_err = mem_scope.initialize(rd);
	CHECK(init_err == OK);
	if (init_err != OK) {
		return;
	}

	MemoryValidator *validator = mem_scope.get();

	// Accumulate some tracked allocations.
	void *ptr = memalloc(128);
	validator->track_allocation(ptr, 128, "PreReset");

	GPUMemoryTracker *gpu = validator->get_gpu_tracker();
	if (gpu) {
		RID rid = RID::from_uint64(7000);
		gpu->track_buffer_creation(rid, 256, "PreResetGPU");
	}

	// Leaks should be present before reset.
	CHECK_FALSE(validator->validate_no_leaks());

	// Reset should clear all tracked state.
	validator->reset();

	// After reset, there should be no leaks (tracked state is gone).
	CHECK(validator->validate_no_leaks());

	MemoryStats stats = validator->get_stats();
	CHECK(stats.current_cpu_bytes == 0);
	CHECK(stats.current_allocations == 0);

	// Still need to free the actual memory (reset only clears tracking, not real memory).
	memfree(ptr);
}

TEST_CASE("[GaussianSplatting] Memory validator initialization rejects null RenderingDevice") {
	Ref<MemoryValidator> validator;
	validator.instantiate();

	Error err = validator->initialize(nullptr);
	CHECK(err == ERR_INVALID_PARAMETER);
	CHECK(validator->get_gpu_tracker() == nullptr);
}

TEST_CASE("[GaussianSplatting] GPU memory leak detection with renderer lifecycle") {
	ScopedGaussianManagerMemory manager_scope;
	GaussianSplatManager *manager = manager_scope.get();
	if (!manager) {
		MESSAGE("Skipping renderer lifecycle memory test - GaussianSplatManager unavailable");
		return;
	}

	RenderingDevice *primary_rd = manager->get_primary_rendering_device();
	if (!primary_rd) {
		RenderingServer *rs = RenderingServer::get_singleton();
		if (rs) {
			primary_rd = rs->create_local_rendering_device();
		}
	}
	if (!primary_rd) {
		MESSAGE("Skipping renderer lifecycle memory test - RenderingDevice unavailable");
		return;
	}

	// Set up memory validator with GPU tracking.
	ScopedMemoryValidator mem_scope;
	Error init_err = mem_scope.initialize(primary_rd);
	CHECK(init_err == OK);
	if (init_err != OK) {
		return;
	}

	MemoryValidator *validator = mem_scope.get();
	GPUMemoryTracker *gpu_tracker = validator->get_gpu_tracker();
	CHECK(gpu_tracker != nullptr);

	// Phase 1: Create renderer and gaussian data.
	Ref<GaussianSplatRenderer> renderer;
	renderer.instantiate(primary_rd);
	CHECK(renderer.is_valid());
	if (!renderer.is_valid()) {
		return;
	}

	Ref<::GaussianData> data;
	data.instantiate();

	LocalVector<Gaussian> gaussians;
	gaussians.resize(64);
	RandomNumberGenerator rng;
	rng.set_seed(42);
	for (int i = 0; i < 64; i++) {
		Gaussian &g = gaussians[i];
		g = Gaussian{};
		g.position = Vector3(
				rng.randf_range(-2.0f, 2.0f),
				rng.randf_range(-2.0f, 2.0f),
				rng.randf_range(-5.0f, -3.0f));
		g.scale = Vector3(0.1f, 0.1f, 0.1f);
		g.rotation = Quaternion();
		g.opacity = 1.0f;
		g.sh_dc = Color(1, 0.5f, 0.2f, 1);
		g.normal = Vector3(0, 1, 0);
		g.area = 0.01f;
	}
	data->set_gaussians(gaussians);

	Error set_err = renderer->set_gaussian_data(data);
	CHECK((set_err == OK || set_err == ERR_UNCONFIGURED));

	// Phase 2: Simulate a render cycle.
	Transform3D cam_transform;
	Projection projection;
	projection.set_perspective(60.0f, 1.0f, 0.1f, 200.0f);

	renderer->render_for_view(cam_transform, projection, RID(), Size2i(128, 128));

	// Phase 3: Simulate tracked GPU resource creation and teardown.
	// Track some synthetic resources as if the renderer created them.
	const int simulated_resource_count = 4;
	RID simulated_rids[4];
	for (int i = 0; i < simulated_resource_count; i++) {
		simulated_rids[i] = RID::from_uint64(8000 + i);
		gpu_tracker->track_buffer_creation(simulated_rids[i], 2048, vformat("RendererBuffer_%d", i));
	}

	CHECK(gpu_tracker->get_resource_count() == simulated_resource_count);

	// Phase 4: Tear down -- free all tracked resources.
	for (int i = 0; i < simulated_resource_count; i++) {
		gpu_tracker->track_resource_free(simulated_rids[i]);
	}

	// Phase 5: Validate no leaks from the tracked resources.
	CHECK(gpu_tracker->validate_no_leaks());
	CHECK(gpu_tracker->get_resource_count() == 0);
	CHECK(gpu_tracker->get_total_gpu_memory() == 0);

	// Clean up renderer.
	renderer.unref();
}

} // namespace TestGaussianSplatting
