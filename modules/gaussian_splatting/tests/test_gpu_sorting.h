/**************************************************************************/
/*  test_gpu_sorting.h                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "../renderer/gpu_sorter.h"
#include "../core/gaussian_data.h"
#include "core/math/math_funcs.h"
#include "core/math/random_number_generator.h"
#include "core/os/os.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"
#include "tests/test_macros.h"
#include <algorithm>

namespace TestGaussianSplatting {

// Helper to create storage buffer from typed data
template<typename T>
static RID create_storage_buffer(RenderingDevice *rd, const LocalVector<T> &data) {
	Vector<uint8_t> bytes;
	bytes.resize(data.size() * sizeof(T));
	memcpy(bytes.ptrw(), data.ptr(), bytes.size());
	return rd->storage_buffer_create(bytes.size(), bytes);
}

TEST_CASE("[GaussianSplatting][RequiresGPU] GPU Bitonic Sorting") {
	RenderingDevice *rd = RenderingDevice::get_singleton();
	if (!rd) {
		RenderingServer *rs = RenderingServer::get_singleton();
		if (rs) {
			rd = rs->create_local_rendering_device();
		}
	}

	if (!rd) {
		MESSAGE("Skipping GPU sorting tests - no RenderingDevice available");
		return;
	}

	SUBCASE("Initialize GPU sorter") {
		Ref<BitonicSort> sorter;
		sorter.instantiate();

		Error err = sorter->initialize(rd, 10000);
		CHECK(err == OK);
		CHECK(sorter->get_max_elements() == 10000);
	}

	SUBCASE("Sort small dataset") {
		Ref<BitonicSort> sorter;
		sorter.instantiate();

		const uint32_t count = 256;
		Error err = sorter->initialize(rd, count);
		CHECK(err == OK);
		if (err != OK) {
			return;
		}

		// Create test data with random depths
		LocalVector<float> depths;
		LocalVector<uint32_t> indices;
		depths.resize(count);
		indices.resize(count);

		RandomNumberGenerator rng;
		rng.set_seed(42);

		for (uint32_t i = 0; i < count; i++) {
			depths[i] = rng.randf_range(0.0f, 100.0f);
			indices[i] = i;
		}

		// Create GPU buffers
		RID keys_buffer = create_storage_buffer(rd, depths);
		RID values_buffer = create_storage_buffer(rd, indices);

		// Sort on GPU
		err = sorter->sort(keys_buffer, values_buffer, count);
		CHECK(err == OK);

		// Read back results
		Vector<uint8_t> sorted_depths = rd->buffer_get_data(keys_buffer);

		// Verify sorting order
		const float *depth_ptr = (const float *)sorted_depths.ptr();
		for (uint32_t i = 1; i < count; i++) {
			CHECK(depth_ptr[i] >= depth_ptr[i - 1]);
		}

		// Clean up
		rd->free(keys_buffer);
		rd->free(values_buffer);
	}

	SUBCASE("Sort power-of-two sizes") {
		const uint32_t test_sizes[] = {128, 256, 512, 1024, 2048};

		for (uint32_t size : test_sizes) {
			Ref<BitonicSort> sorter;
			sorter.instantiate();

			Error err = sorter->initialize(rd, size);
			CHECK(err == OK);
			if (err != OK) {
				continue;
			}

			// Create reverse-sorted data (worst case)
			LocalVector<float> depths;
			LocalVector<uint32_t> indices;
			depths.resize(size);
			indices.resize(size);

			for (uint32_t i = 0; i < size; i++) {
				depths[i] = float(size - i); // Reverse order
				indices[i] = i;
			}

			// Create GPU buffers
			RID keys_buffer = create_storage_buffer(rd, depths);
			RID values_buffer = create_storage_buffer(rd, indices);

			// Sort
			err = sorter->sort(keys_buffer, values_buffer, size);
			CHECK(err == OK);

			// Verify
			Vector<uint8_t> sorted_depths = rd->buffer_get_data(keys_buffer);
			const float *depth_ptr = (const float *)sorted_depths.ptr();

			for (uint32_t i = 0; i < size; i++) {
				CHECK(Math::is_equal_approx(depth_ptr[i], float(i + 1)));
			}

			// Clean up
			rd->free(keys_buffer);
			rd->free(values_buffer);
		}
	}

	SUBCASE("Handle non-power-of-two sizes") {
		Ref<BitonicSort> sorter;
		sorter.instantiate();

		const uint32_t count = 1000; // Not a power of 2
		Error err = sorter->initialize(rd, count);
		CHECK(err == OK);
		if (err != OK) {
			return;
		}

		// BitonicSort handles non-power-of-two internally
		CHECK(sorter->supports_non_power_of_two());

		// Create test data
		LocalVector<float> depths;
		LocalVector<uint32_t> indices;
		depths.resize(count);
		indices.resize(count);

		RandomNumberGenerator rng;
		rng.set_seed(42);

		for (uint32_t i = 0; i < count; i++) {
			depths[i] = rng.randf_range(0.0f, 100.0f);
			indices[i] = i;
		}

		// Create GPU buffers
		RID keys_buffer = create_storage_buffer(rd, depths);
		RID values_buffer = create_storage_buffer(rd, indices);

		// Sort
		err = sorter->sort(keys_buffer, values_buffer, count);
		CHECK(err == OK);

		// Verify sorting
		Vector<uint8_t> sorted_depths = rd->buffer_get_data(keys_buffer);
		const float *depth_ptr = (const float *)sorted_depths.ptr();

		for (uint32_t i = 1; i < count; i++) {
			CHECK(depth_ptr[i] >= depth_ptr[i - 1]);
		}

		// Clean up
		rd->free(keys_buffer);
		rd->free(values_buffer);
	}
}

TEST_CASE("[GaussianSplatting][RequiresGPU] GPU Sorting Performance") {
	RenderingDevice *rd = RenderingDevice::get_singleton();
	if (!rd) {
		RenderingServer *rs = RenderingServer::get_singleton();
		if (rs) {
			rd = rs->create_local_rendering_device();
		}
	}

	if (!rd) {
		MESSAGE("Skipping GPU sorting performance tests - no RenderingDevice available");
		return;
	}

	SUBCASE("Bitonic sort performance scaling") {
		const uint32_t test_sizes[] = {1024, 4096, 16384, 65536};

		for (uint32_t size : test_sizes) {
			Ref<BitonicSort> sorter;
			sorter.instantiate();

			Error err = sorter->initialize(rd, size);
			CHECK(err == OK);
			if (err != OK) {
				continue;
			}

			// Create random test data
			LocalVector<float> depths;
			LocalVector<uint32_t> indices;
			depths.resize(size);
			indices.resize(size);

			RandomNumberGenerator rng;
			rng.set_seed(42);

			for (uint32_t i = 0; i < size; i++) {
				depths[i] = rng.randf_range(0.0f, 1000.0f);
				indices[i] = i;
			}

			// Create GPU buffers
			RID keys_buffer = create_storage_buffer(rd, depths);
			RID values_buffer = create_storage_buffer(rd, indices);

			// Measure sort time
			uint64_t start = OS::get_singleton()->get_ticks_usec();

			err = sorter->sort(keys_buffer, values_buffer, size);
			CHECK(err == OK);

			uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start;
			float ms = elapsed / 1000.0f;

			// Performance expectations (conservative for CI)
			if (size <= 4096) {
				CHECK_MESSAGE(ms < 50.0f,
					vformat("Sorting %d elements took %.2fms, expected < 50ms", size, ms));
			} else if (size <= 16384) {
				CHECK_MESSAGE(ms < 100.0f,
					vformat("Sorting %d elements took %.2fms, expected < 100ms", size, ms));
			} else if (size <= 65536) {
				CHECK_MESSAGE(ms < 200.0f,
					vformat("Sorting %d elements took %.2fms, expected < 200ms", size, ms));
			}

			// Clean up
			rd->free(keys_buffer);
			rd->free(values_buffer);
		}
	}

	SUBCASE("Compare with CPU sorting") {
		const uint32_t count = 10000;

		// Generate test data
		LocalVector<float> depths;
		LocalVector<uint32_t> indices;
		depths.resize(count);
		indices.resize(count);

		RandomNumberGenerator rng;
		rng.set_seed(42);

		for (uint32_t i = 0; i < count; i++) {
			depths[i] = rng.randf_range(0.0f, 1000.0f);
			indices[i] = i;
		}

		// CPU sort timing
		LocalVector<float> cpu_depths = depths;
		LocalVector<uint32_t> cpu_indices = indices;

		uint64_t cpu_start = OS::get_singleton()->get_ticks_usec();

		// Sort indices by depth
		std::sort(cpu_indices.ptr(), cpu_indices.ptr() + count,
			[&cpu_depths](uint32_t a, uint32_t b) {
				return cpu_depths[a] < cpu_depths[b];
			});

		uint64_t cpu_elapsed = OS::get_singleton()->get_ticks_usec() - cpu_start;
		float cpu_ms = cpu_elapsed / 1000.0f;

		// GPU sort timing
		Ref<BitonicSort> sorter;
		sorter.instantiate();
		Error err = sorter->initialize(rd, count);
		CHECK(err == OK);
		if (err != OK) {
			return;
		}

		RID keys_buffer = create_storage_buffer(rd, depths);
		RID values_buffer = create_storage_buffer(rd, indices);

		uint64_t gpu_start = OS::get_singleton()->get_ticks_usec();

		err = sorter->sort(keys_buffer, values_buffer, count);
		CHECK(err == OK);

		uint64_t gpu_elapsed = OS::get_singleton()->get_ticks_usec() - gpu_start;
		float gpu_ms = gpu_elapsed / 1000.0f;

		// GPU should be reasonable compared to CPU (not always faster due to transfer overhead)
		CHECK_MESSAGE(gpu_ms < cpu_ms * 10.0f,
			vformat("GPU sort (%.2fms) should be within 10x of CPU sort (%.2fms)", gpu_ms, cpu_ms));

		// Clean up
		rd->free(keys_buffer);
		rd->free(values_buffer);
	}
}

} // namespace TestGaussianSplatting
