/**
 * @file test_node_bootstrap.h
 * @brief Regression test: GaussianSplatNode3D must register with the
 *        shared renderer without force-populating primary gaussian_data
 *        for removed resident/bootstrap fallback behavior.
 */

#pragma once

#include "tests/test_macros.h"

#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_asset.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace {

/// Minimal single-splat asset used only by the node registration regression test.
static Ref<GaussianSplatAsset> _make_bootstrap_test_asset() {
	Ref<GaussianSplatAsset> asset;
	asset.instantiate();
	asset->set_splat_count(1);

	PackedFloat32Array positions;
	positions.resize(3);
	{
		float *ptr = positions.ptrw();
		ptr[0] = 0.0f;
		ptr[1] = 0.0f;
		ptr[2] = 0.0f;
	}
	asset->set_positions(positions);

	PackedFloat32Array scales;
	scales.resize(3);
	{
		float *ptr = scales.ptrw();
		ptr[0] = 1.0f;
		ptr[1] = 1.0f;
		ptr[2] = 1.0f;
	}
	asset->set_scales(scales);

	PackedFloat32Array rotations;
	rotations.resize(4);
	{
		float *ptr = rotations.ptrw();
		ptr[0] = 1.0f; // w
		ptr[1] = 0.0f;
		ptr[2] = 0.0f;
		ptr[3] = 0.0f;
	}
	asset->set_rotations(rotations);

	PackedFloat32Array sh_dc;
	sh_dc.resize(3);
	{
		float *ptr = sh_dc.ptrw();
		ptr[0] = 1.0f;
		ptr[1] = 1.0f;
		ptr[2] = 1.0f;
	}
	asset->set_sh_dc_coefficients(sh_dc);

	PackedFloat32Array opacity_logits;
	opacity_logits.resize(1);
	opacity_logits.set(0, 10.0f);
	asset->set_opacity_logits(opacity_logits);

	return asset;
}

} // anonymous namespace

// [SceneTree] tag ensures the test harness creates a SceneTree before running.
TEST_CASE("[GaussianSplatting][SceneTree] Asset-based node registers without bootstrapping renderer gaussian_data") {
	// After #177, node registration should still populate SceneDirector
	// instance data, but it should not seed renderer scene_state.gaussian_data
	// just to keep a resident/bootstrap fallback path alive.

	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE_MESSAGE(tree != nullptr, "SceneTree must exist (provided by [SceneTree] tag)");

	Window *root = tree->get_root();
	REQUIRE_MESSAGE(root != nullptr, "SceneTree root window must exist");

	Ref<GaussianSplatAsset> asset = _make_bootstrap_test_asset();
	CHECK(asset.is_valid());
	CHECK(asset->get_splat_count() > 0);

	GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
	node->set_splat_asset(asset);

	root->add_child(node);
	tree->process(0.0);

	Ref<GaussianSplatRenderer> renderer = node->get_renderer();

		// In headless mode (no RenderingServer), the renderer won't be created.
		// The registration contract is only exercisable with a GPU device.
		if (!renderer.is_valid()) {
			MESSAGE("Renderer unavailable (headless mode) — skipping renderer state checks");
		}
		if (renderer.is_valid()) {
			const auto &scene_state = renderer->get_scene_state();
			CHECK_MESSAGE(scene_state.gaussian_data.is_null(),
					"Renderer scene_state.gaussian_data should remain unset after node registration without bootstrap fallback");

			GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
			if (director) {
			LocalVector<InstanceDataGPU> instance_buffer;
			director->build_instance_buffer_for_renderer(renderer.ptr(), instance_buffer);
			CHECK(instance_buffer.size() >= 1);
		}
	}

	root->remove_child(node);
	memdelete(node);
}
