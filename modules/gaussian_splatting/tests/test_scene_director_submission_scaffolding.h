#pragma once

#include "test_macros.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_asset.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

#if defined(TESTS_ENABLED) || defined(TOOLS_ENABLED)

namespace {

Ref<GaussianData> stage1a_make_submission_test_data(int p_count, float p_x_offset = 0.0f) {
	Ref<GaussianData> data;
	data.instantiate();
	data->resize(p_count);
	for (int i = 0; i < p_count; i++) {
		Gaussian g;
		g.position = Vector3(p_x_offset + float(i), 0.0f, 0.0f);
		g.scale = Vector3(1.0f, 1.0f, 1.0f);
		g.rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
		g.opacity = 1.0f;
		g.sh_dc = Color(1.0f, 1.0f, 1.0f, 1.0f);
		data->set_gaussian(i, g);
	}
	return data;
}

Ref<GaussianSplatAsset> stage1a_make_submission_test_asset(float p_x_offset = 0.0f) {
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

StaticChunk stage1a_make_submission_test_chunk(uint32_t p_index) {
	StaticChunk chunk;
	chunk.bounds = AABB(Vector3(float(p_index), 0.0f, 0.0f), Vector3(1.0f, 1.0f, 1.0f));
	chunk.center = chunk.bounds.get_center();
	chunk.radius = 1.0f;
	chunk.indices.push_back(p_index);
	return chunk;
}

} // namespace

TEST_CASE("[GaussianSplatting][SceneDirector][SceneTree] World submission scaffolding preserves one active source per scenario") {
	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	const bool owns_director = (director == nullptr);
	if (!director) {
		director = memnew(GaussianSplatSceneDirector);
	}
	REQUIRE(director != nullptr);
	const GaussianSplatSceneDirector::SubmissionCounts baseline_counts = director->get_submission_counts();

	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

	Window *root = tree->get_root();
	REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

	Ref<World3D> world = root->get_world_3d();
	REQUIRE(world.is_valid());
	const RID scenario = world->get_scenario();
	REQUIRE(scenario.is_valid());

	GaussianSplatSceneDirector::WorldSubmission submission_a;
	submission_a.owner_id = ObjectID(uint64_t(101));
	submission_a.scenario = scenario;
	submission_a.gaussian_data = stage1a_make_submission_test_data(3, 0.0f);
	submission_a.static_chunks.push_back(stage1a_make_submission_test_chunk(0));
	submission_a.bounds = AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(2.0f, 2.0f, 2.0f));
	submission_a.metadata[StringName("label")] = String("world_a");
	submission_a.desired_residency_hint = 7;
	submission_a.desired_renderer_overrides[StringName("max_splats")] = int64_t(4096);

	CHECK(director->upsert_world_submission(submission_a));

	GaussianSplatSceneDirector::WorldSubmission queried_submission;
	CHECK(director->get_world_submission(submission_a.owner_id, &queried_submission));
	CHECK(queried_submission.scenario == scenario);
	CHECK(queried_submission.gaussian_data == submission_a.gaussian_data);
	CHECK(queried_submission.static_chunks.size() == 1);
	CHECK(queried_submission.metadata[StringName("label")] == String("world_a"));
	CHECK(queried_submission.desired_residency_hint == 7);
	CHECK(int64_t(queried_submission.desired_renderer_overrides[StringName("max_splats")]) == 4096);

	GaussianSplatSceneDirector::SubmissionCounts counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions + 1);
	CHECK(counts.preview_submissions == baseline_counts.preview_submissions);

	GaussianSplatSceneDirector::WorldSubmission submission_b;
	submission_b.owner_id = ObjectID(uint64_t(202));
	submission_b.scenario = scenario;
	submission_b.gaussian_data = stage1a_make_submission_test_data(2, 10.0f);
	submission_b.static_chunks.push_back(stage1a_make_submission_test_chunk(1));
	submission_b.bounds = AABB(Vector3(10.0f, 0.0f, 0.0f), Vector3(3.0f, 3.0f, 3.0f));
	submission_b.metadata[StringName("label")] = String("world_b");

	CHECK(director->upsert_world_submission(submission_b));
	CHECK_FALSE(director->get_world_submission(submission_a.owner_id, &queried_submission));
	CHECK(director->get_world_submission(submission_b.owner_id, &queried_submission));
	CHECK(queried_submission.owner_id == submission_b.owner_id);
	CHECK(queried_submission.gaussian_data == submission_b.gaussian_data);

	GaussianSplatSceneDirector::WorldSubmission by_scenario;
	CHECK(director->get_world_submission_for_scenario(scenario, &by_scenario));
	CHECK(by_scenario.owner_id == submission_b.owner_id);

	director->unregister_world_submission(submission_b.owner_id);
	CHECK_FALSE(director->get_world_submission(submission_b.owner_id, &queried_submission));
	counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions);
	CHECK(counts.preview_submissions == baseline_counts.preview_submissions);

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][SceneDirector] Preview submission scaffolding round-trips and unregisters cleanly") {
	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	const bool owns_director = (director == nullptr);
	if (!director) {
		director = memnew(GaussianSplatSceneDirector);
	}
	REQUIRE(director != nullptr);
	const GaussianSplatSceneDirector::SubmissionCounts baseline_counts = director->get_submission_counts();

	GaussianSplatSceneDirector::PreviewSubmission preview_submission;
	preview_submission.owner_id = ObjectID(uint64_t(303));
	preview_submission.gaussian_data = stage1a_make_submission_test_data(4, 20.0f);
	preview_submission.metadata[StringName("label")] = String("preview");
	preview_submission.source_label = "editor_preview";

	CHECK(director->upsert_preview_submission(preview_submission));

	GaussianSplatSceneDirector::PreviewSubmission queried_preview;
	CHECK(director->get_preview_submission(preview_submission.owner_id, &queried_preview));
	CHECK(queried_preview.owner_id == preview_submission.owner_id);
	CHECK(queried_preview.gaussian_data == preview_submission.gaussian_data);
	CHECK(queried_preview.metadata[StringName("label")] == String("preview"));
	CHECK(queried_preview.source_label == String("editor_preview"));

	GaussianSplatSceneDirector::SubmissionCounts counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions);
	CHECK(counts.preview_submissions == baseline_counts.preview_submissions + 1);

	director->unregister_preview_submission(preview_submission.owner_id);
	CHECK_FALSE(director->get_preview_submission(preview_submission.owner_id, &queried_preview));
	counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions);
	CHECK(counts.preview_submissions == baseline_counts.preview_submissions);

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][SceneDirector][SceneTree] Instance submission query mirrors live node registration") {
	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

	Window *root = tree->get_root();
	REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	const bool owns_director = (director == nullptr);
	if (!director) {
		director = memnew(GaussianSplatSceneDirector);
	}
	REQUIRE(director != nullptr);
	const GaussianSplatSceneDirector::SubmissionCounts baseline_counts = director->get_submission_counts();

	GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
	REQUIRE(node != nullptr);
	node->set_splat_asset(stage1a_make_submission_test_asset(42.0f));
	node->set_opacity(0.5f);
	node->set_quality_preset(GaussianSplatNode3D::QUALITY_CUSTOM);
	node->set_lod_bias(1.25f);
	node->set_cast_shadow(true);

	root->add_child(node);
	tree->process(0.0);

	GaussianSplatSceneDirector::InstanceSubmission submission;
	CHECK(director->get_instance_submission(node->get_instance_id(), &submission));
	CHECK(submission.node_id == node->get_instance_id());
	CHECK(submission.asset.is_valid());
	CHECK(submission.opacity == doctest::Approx(0.5f));
	CHECK(submission.lod_bias == doctest::Approx(1.25f));
	CHECK(submission.casts_shadow);
	CHECK(submission.visible);

	GaussianSplatSceneDirector::SubmissionCounts counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions + 1);
	CHECK(counts.world_submissions == baseline_counts.world_submissions);
	CHECK(counts.preview_submissions == baseline_counts.preview_submissions);

	root->remove_child(node);
	memdelete(node);
	tree->process(0.0);

	CHECK_FALSE(director->get_instance_submission(submission.node_id, &submission));
	counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions);
	CHECK(counts.preview_submissions == baseline_counts.preview_submissions);

	if (owns_director) {
		memdelete(director);
	}
}

#endif // TESTS_ENABLED || TOOLS_ENABLED
