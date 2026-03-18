/**************************************************************************/
/*  test_gaussian_data.h                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "../core/gaussian_data.h"
#include "core/io/dir_access.h"
#include "core/math/math_defs.h"
#include "core/math/random_number_generator.h"
#include "core/os/os.h"
#include "tests/test_macros.h"
#include <limits>

namespace TestGaussianSplatting {

static Gaussian _make_test_gaussian(const Vector3 &p_position, const Color &p_color = Color(1, 1, 1, 1)) {
	Gaussian g;
	g.position = p_position;
	g.scale = Vector3(1.0f, 1.0f, 1.0f);
	g.rotation = Quaternion();
	g.opacity = 1.0f;
	g.sh_dc = p_color;
	g.normal = Vector3(0, 1, 0);
	g.area = static_cast<float>(Math::PI);
	g.brush_axes = Vector2(1.0f, 1.0f);
	g.stroke_age = 0.0f;
	return g;
}

static String _make_gaussian_data_fixture_path(const String &p_prefix, const String &p_suffix = ".ply") {
	const uint64_t ticks = OS::get_singleton() ? OS::get_singleton()->get_ticks_usec() : 0;
	const String base_temp = OS::get_singleton() ? OS::get_singleton()->get_temp_path() : ".";
	return base_temp.path_join("godotgs_gaussian_data_" + p_prefix + "_" + itos(ticks) + p_suffix);
}

TEST_CASE("[GaussianSplatting] GaussianData basic operations") {
	SUBCASE("Create and resize GaussianData") {
		Ref<::GaussianData> data;
		data.instantiate();
		
		CHECK(data.is_valid());
		CHECK(data->get_count() == 0);
		
		// Create test splats
		LocalVector<Gaussian> splats;
		splats.resize(100);
		
		RandomNumberGenerator rng;
		rng.set_seed(42);

		for (int i = 0; i < 100; i++) {
			Gaussian &g = splats[i];
			g.position = Vector3(
				rng.randf_range(-10.0f, 10.0f),
				rng.randf_range(-10.0f, 10.0f),
				rng.randf_range(-10.0f, 10.0f)
			);
			g.scale = Vector3(0.5f, 0.5f, 0.5f);
			g.rotation = Quaternion();
			g.opacity = 1.0f;
			g.sh_dc = Color(1, 0, 0, 1);
			g.normal = Vector3(0, 1, 0);
			g.area = 0.785f; // PI * 0.5^2
		}
		
		data->set_gaussians(splats);
		CHECK(data->get_count() == 100);
	}
	
	SUBCASE("GaussianData AABB calculation") {
		Ref<::GaussianData> data;
		data.instantiate();
		
		LocalVector<Gaussian> splats;
		splats.resize(3);
		
		// Create splats at known positions
		splats[0].position = Vector3(-5, -5, -5);
		splats[0].scale = Vector3(1, 1, 1);
		
		splats[1].position = Vector3(5, 5, 5);
		splats[1].scale = Vector3(1, 1, 1);
		
		splats[2].position = Vector3(0, 0, 0);
		splats[2].scale = Vector3(1, 1, 1);
		
		for (int i = 0; i < 3; i++) {
			splats[i].rotation = Quaternion();
			splats[i].opacity = 1.0f;
			splats[i].sh_dc = Color(1, 1, 1, 1);
			splats[i].normal = Vector3(0, 1, 0);
			splats[i].area = static_cast<float>(Math::PI);
		}
		
		data->set_gaussians(splats);
		
		AABB aabb = data->compute_aabb();
		// Account for scale * 3.0f extent in bounds (Gaussian 3-sigma extent)
		// Positions: (-5,-5,-5) to (5,5,5), scale (1,1,1) * 3 = extent of 3
		// So min = (-5-3) = -8, max = (5+3) = 8, size = 16
		CHECK(aabb.position.is_equal_approx(Vector3(-8, -8, -8)));
		CHECK(aabb.size.is_equal_approx(Vector3(16, 16, 16)));
	}
	
	SUBCASE("GaussianData empty state") {
		Ref<::GaussianData> data;
		data.instantiate();
		
		LocalVector<Gaussian> empty_splats;
		data->set_gaussians(empty_splats);
		
		CHECK(data->get_count() == 0);
		AABB aabb = data->compute_aabb();
		CHECK(aabb == AABB()); // Should be empty AABB
	}
	
	SUBCASE("GaussianData single splat") {
		Ref<::GaussianData> data;
		data.instantiate();
		
		LocalVector<Gaussian> single_splat;
		single_splat.resize(1);
		
		single_splat[0].position = Vector3(10, 20, 30);
		single_splat[0].scale = Vector3(2, 2, 2);
		single_splat[0].rotation = Quaternion();
		single_splat[0].opacity = 0.5f;
		single_splat[0].sh_dc = Color(0, 1, 0, 0.5f);
		single_splat[0].normal = Vector3(0, 0, 1);
		single_splat[0].area = 4 * static_cast<float>(Math::PI);
		
		data->set_gaussians(single_splat);
		
		CHECK(data->get_count() == 1);
		const Gaussian *g = data->get_gaussians();
		CHECK(g != nullptr);
		CHECK(g[0].position.is_equal_approx(Vector3(10, 20, 30)));
		CHECK(Math::is_equal_approx(g[0].opacity, 0.5f));
	}

	SUBCASE("GaussianData runtime range edits and 2D mode toggle") {
		Ref<::GaussianData> data;
		data.instantiate();

		LocalVector<Gaussian> splats;
		splats.resize(3);
		splats[0] = _make_test_gaussian(Vector3(0, 0, 0), Color(1, 0, 0, 1));
		splats[1] = _make_test_gaussian(Vector3(1, 0, 0), Color(0, 1, 0, 1));
		splats[2] = _make_test_gaussian(Vector3(2, 0, 0), Color(0, 0, 1, 1));
		data->set_gaussians(splats);

		CHECK_FALSE(data->get_2d_mode());
		data->set_2d_mode(true);
		CHECK(data->get_2d_mode());
		data->set_2d_mode(false);
		CHECK_FALSE(data->get_2d_mode());

		const Color range_color = Color(0.2f, 0.4f, 0.6f, 1.0f);
		data->apply_color_range(1, 2, range_color);
		CHECK(data->has_modifications());
		data->commit_runtime_changes();
		CHECK_FALSE(data->has_modifications());

		const Gaussian *after_color_apply = data->get_gaussians();
		CHECK(after_color_apply != nullptr);
		CHECK(after_color_apply[0].sh_dc.is_equal_approx(Color(1, 0, 0, 1)));
		CHECK(after_color_apply[1].sh_dc.is_equal_approx(range_color));
		CHECK(after_color_apply[2].sh_dc.is_equal_approx(range_color));

		const Color unchanged_color = after_color_apply[0].sh_dc;
		data->mark_range_dirty(0, 1);
		CHECK(data->has_modifications());
		data->commit_runtime_changes();
		CHECK_FALSE(data->has_modifications());

		const Gaussian *after_mark_dirty = data->get_gaussians();
		CHECK(after_mark_dirty != nullptr);
		CHECK(after_mark_dirty[0].sh_dc.is_equal_approx(unchanged_color));
	}
}

TEST_CASE("[GaussianSplatting] Gaussian structure memory layout") {
	// Verify the Gaussian structure is properly aligned for GPU usage
	CHECK(sizeof(Gaussian) % 16 == 0); // Should be 16-byte aligned
	
	// Check field offsets for GPU compatibility
	Gaussian g;
	ptrdiff_t pos_offset = (char*)&g.position - (char*)&g;
	ptrdiff_t scale_offset = (char*)&g.scale - (char*)&g;
	ptrdiff_t rot_offset = (char*)&g.rotation - (char*)&g;
	
	CHECK(pos_offset == 0); // Position should be at start
	CHECK(scale_offset >= sizeof(Vector3)); // Scale after position
	CHECK(rot_offset >= scale_offset + sizeof(Vector3)); // Rotation after scale
}

TEST_CASE("[GaussianSplatting] GaussianData load_from_file invalidates storage-derived caches") {
	Ref<::GaussianData> source_a;
	source_a.instantiate();
	Ref<::GaussianData> source_b;
	source_b.instantiate();
	Ref<::GaussianData> data;
	data.instantiate();

	LocalVector<Gaussian> gaussians_a;
	gaussians_a.resize(1);
	gaussians_a[0] = _make_test_gaussian(Vector3(0, 0, 0), Color(1, 0, 0, 1));
	source_a->set_gaussians(gaussians_a);

	LocalVector<Gaussian> gaussians_b;
	gaussians_b.resize(1);
	gaussians_b[0] = _make_test_gaussian(Vector3(10, 10, 10), Color(0, 1, 0, 1));
	source_b->set_gaussians(gaussians_b);

	const String path_a = _make_gaussian_data_fixture_path("load_invalidation_a");
	const String path_b = _make_gaussian_data_fixture_path("load_invalidation_b");

	Error save_a_err = source_a->save_to_file(path_a);
	CHECK(save_a_err == OK);
	if (save_a_err != OK) {
		return;
	}
	Error save_b_err = source_b->save_to_file(path_b);
	CHECK(save_b_err == OK);
	if (save_b_err != OK) {
		return;
	}

	Error load_a_err = data->load_from_file(path_a);
	CHECK(load_a_err == OK);
	if (load_a_err != OK) {
		return;
	}

	data->build_octree();
	TypedArray<int> first_hits = data->query_octree(AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)));
	CHECK(first_hits.size() == 1);

	Array stroke_history;
	Dictionary stroke;
	stroke["center"] = Vector3(0, 0, 0);
	stroke["radius"] = 1.0f;
	stroke["color"] = Color(1, 1, 1, 1);
	stroke["opacity"] = 1.0f;
	stroke["hardness"] = 1.0f;
	stroke_history.push_back(stroke);
	data->set_brush_strokes(stroke_history);
	CHECK(data->get_brush_strokes().size() == 1);

	Error load_b_err = data->load_from_file(path_b);
	CHECK(load_b_err == OK);
	if (load_b_err != OK) {
		return;
	}

	// Regression check: stale octree queries from dataset A must not leak after loading dataset B.
	TypedArray<int> stale_hits = data->query_octree(AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)));
	CHECK(stale_hits.size() == 0);
	CHECK(data->get_brush_strokes().size() == 0);

	data->build_octree();
	TypedArray<int> second_hits = data->query_octree(AABB(Vector3(9, 9, 9), Vector3(2, 2, 2)));
	CHECK(second_hits.size() == 1);

	DirAccess::remove_absolute(path_a);
	DirAccess::remove_absolute(path_b);
}

TEST_CASE("[GaussianSplatting] GaussianData GPU payload validation rejects non-finite and out-of-range values") {
	Ref<::GaussianData> data;
	data.instantiate();

	LocalVector<Gaussian> gaussians;
	gaussians.resize(1);
	gaussians[0] = _make_test_gaussian(Vector3(1, 2, 3), Color(0.5f, 0.2f, 0.8f, 1.0f));
	data->set_gaussians(gaussians);
	CHECK(data->validate_gpu_payload() == OK);

	gaussians[0].opacity = std::numeric_limits<float>::infinity();
	data->set_gaussians(gaussians);
	String inf_error;
	CHECK(data->validate_gpu_payload(&inf_error) == ERR_INVALID_DATA);
	CHECK_FALSE(inf_error.is_empty());

	gaussians[0] = _make_test_gaussian(Vector3(1, 2, 3), Color(1, 1, 1, 1));
	gaussians[0].scale.x = -1.0f;
	data->set_gaussians(gaussians);
	String scale_error;
	CHECK(data->validate_gpu_payload(&scale_error) == ERR_INVALID_DATA);
	CHECK_FALSE(scale_error.is_empty());
}

} // namespace TestGaussianSplatting
