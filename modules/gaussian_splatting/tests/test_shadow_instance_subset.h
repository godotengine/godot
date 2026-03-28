#pragma once

#include "tests/test_macros.h"

#include "../core/gaussian_splat_asset.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace {

static Ref<GaussianSplatAsset> _make_shadow_subset_test_asset(float p_x_offset) {
	Ref<GaussianSplatAsset> asset;
	asset.instantiate();
	asset->set_splat_count(1);

	PackedFloat32Array positions;
	positions.resize(3);
	{
		float *ptr = positions.ptrw();
		ptr[0] = p_x_offset;
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
		ptr[0] = 1.0f;
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

} // namespace

TEST_CASE("[GaussianSplatting][SceneTree] Shared renderer shadow subset follows per-instance cast_shadow state") {
	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE_MESSAGE(tree != nullptr, "SceneTree must exist (provided by [SceneTree] tag)");

	Window *root = tree->get_root();
	REQUIRE_MESSAGE(root != nullptr, "SceneTree root window must exist");

	GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
	GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
	node_a->set_splat_asset(_make_shadow_subset_test_asset(0.0f));
	node_b->set_splat_asset(_make_shadow_subset_test_asset(10.0f));
	node_b->set_cast_shadow(true);

	root->add_child(node_a);
	root->add_child(node_b);
	tree->process(0.0);

	Ref<GaussianSplatRenderer> renderer_a = node_a->get_renderer();
	Ref<GaussianSplatRenderer> renderer_b = node_b->get_renderer();
	if (!renderer_a.is_valid() || !renderer_b.is_valid()) {
		MESSAGE("Shared renderer unavailable (headless/no RenderingDevice) — skipping shadow subset checks");
		root->remove_child(node_b);
		root->remove_child(node_a);
		memdelete(node_b);
		memdelete(node_a);
		return;
	}
	REQUIRE(renderer_a == renderer_b);

	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	REQUIRE(director != nullptr);

	LocalVector<InstanceDataGPU> all_instances;
	LocalVector<InstanceDataGPU> shadow_instances;
	LocalVector<InstanceAssetRegistration> all_assets;
	LocalVector<InstanceAssetRegistration> shadow_assets;

	director->build_instance_buffer_for_renderer(renderer_a.ptr(), all_instances);
	director->build_instance_buffer_for_renderer(renderer_a.ptr(), shadow_instances, true);
	director->collect_instance_assets_for_renderer(renderer_a.ptr(), all_assets);
	director->collect_instance_assets_for_renderer(renderer_a.ptr(), shadow_assets, true);

	CHECK_EQ(all_instances.size(), 2);
	CHECK_EQ(shadow_instances.size(), 1);
	CHECK_EQ(all_assets.size(), 2);
	CHECK_EQ(shadow_assets.size(), 1);

	node_b->set_cast_shadow(false);
	tree->process(0.0);
	director->build_instance_buffer_for_renderer(renderer_a.ptr(), shadow_instances, true);
	director->collect_instance_assets_for_renderer(renderer_a.ptr(), shadow_assets, true);
	CHECK_EQ(shadow_instances.size(), 0);
	CHECK_EQ(shadow_assets.size(), 0);

	node_a->set_cast_shadow(true);
	tree->process(0.0);
	director->build_instance_buffer_for_renderer(renderer_a.ptr(), shadow_instances, true);
	director->collect_instance_assets_for_renderer(renderer_a.ptr(), shadow_assets, true);
	CHECK_EQ(shadow_instances.size(), 1);
	CHECK_EQ(shadow_assets.size(), 1);

	root->remove_child(node_b);
	root->remove_child(node_a);
	memdelete(node_b);
	memdelete(node_a);
}
