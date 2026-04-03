#pragma once

#include "test_macros.h"
#include "gs_test_setting_guard.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_asset.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../core/gs_project_settings.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../nodes/gaussian_splat_world_3d.h"
#include "../renderer/quantization_config.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "servers/rendering_server.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/storage/render_scene_buffers.h"

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
	submission_a.has_desired_residency_hint = true;
	submission_a.desired_residency_hint = 7;
	submission_a.desired_renderer_overrides[StringName("max_splats")] = int64_t(4096);

	CHECK(director->upsert_world_submission(submission_a));

	GaussianSplatSceneDirector::WorldSubmission queried_submission;
	CHECK(director->get_world_submission(submission_a.owner_id, &queried_submission));
	CHECK(queried_submission.scenario == scenario);
	CHECK(queried_submission.gaussian_data == submission_a.gaussian_data);
	CHECK(queried_submission.static_chunks.size() == 1);
	CHECK(queried_submission.metadata[StringName("label")] == String("world_a"));
	CHECK(queried_submission.has_desired_residency_hint);
	CHECK(queried_submission.desired_residency_hint == 7);
	CHECK(int64_t(queried_submission.desired_renderer_overrides[StringName("max_splats")]) == 4096);

	GaussianSplatSceneDirector::SubmissionCounts counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions + 1);

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

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][SceneDirector][SceneTree] World submission entrypoints arbitrate ownership and release cleanly") {
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

	Node *owner_a = memnew(Node);
	Node *owner_b = memnew(Node);
	REQUIRE(owner_a != nullptr);
	REQUIRE(owner_b != nullptr);
	root->add_child(owner_a);
	root->add_child(owner_b);
	tree->process(0.0);

	GaussianSplatSceneDirector::WorldSubmission submission_a;
	submission_a.owner_id = owner_a->get_instance_id();
	submission_a.scenario = scenario;
	submission_a.gaussian_data = stage1a_make_submission_test_data(3, 0.0f);
	submission_a.static_chunks.push_back(stage1a_make_submission_test_chunk(0));
	submission_a.metadata[StringName("label")] = String("owner_a");
	submission_a.desired_renderer_overrides[StringName("max_splats")] = int64_t(2048);

	GaussianSplatSceneDirector::WorldSubmission submission_b;
	submission_b.owner_id = owner_b->get_instance_id();
	submission_b.scenario = scenario;
	submission_b.gaussian_data = stage1a_make_submission_test_data(2, 20.0f);
	submission_b.static_chunks.push_back(stage1a_make_submission_test_chunk(1));
	submission_b.metadata[StringName("label")] = String("owner_b");
	submission_b.desired_renderer_overrides[StringName("max_splats")] = int64_t(1024);

	CHECK(director->submit_world_submission(submission_a));

	GaussianSplatSceneDirector::WorldSubmission queried_submission;
	CHECK(director->get_world_submission_for_scenario(scenario, &queried_submission));
	CHECK(queried_submission.owner_id == submission_a.owner_id);
	CHECK(queried_submission.gaussian_data == submission_a.gaussian_data);

	CHECK_FALSE(director->submit_world_submission(submission_b));
	CHECK(director->get_world_submission_for_scenario(scenario, &queried_submission));
	CHECK(queried_submission.owner_id == submission_a.owner_id);

	GaussianSplatSceneDirector::SubmissionCounts counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions + 1);

	director->release_world_submission(submission_a.owner_id);
	CHECK_FALSE(director->get_world_submission(submission_a.owner_id, &queried_submission));
	CHECK_FALSE(director->get_world_submission_for_scenario(scenario, &queried_submission));

	CHECK(director->submit_world_submission(submission_b));
	CHECK(director->get_world_submission_for_scenario(scenario, &queried_submission));
	CHECK(queried_submission.owner_id == submission_b.owner_id);

	director->release_world_submission(submission_b.owner_id);
	CHECK_FALSE(director->get_world_submission(submission_b.owner_id, &queried_submission));
	counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions);

	root->remove_child(owner_b);
	root->remove_child(owner_a);
	memdelete(owner_b);
	memdelete(owner_a);
	tree->process(0.0);

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][World][SceneTree] World node forwards desired overrides through director submission") {
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

	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	REQUIRE(project_settings != nullptr);
	ProjectSettingGuard tier_preset_guard(project_settings, "rendering/gaussian_splatting/quality/tier_preset");
	ProjectSettingGuard tier_apply_guard(project_settings, "rendering/gaussian_splatting/quality/tier_apply_streaming_budgets");
	ProjectSettingGuard predictive_guard(project_settings, "rendering/gaussian_splatting/streaming/predictive_prefetch_enabled");
	ProjectSettingGuard prefetch_guard(project_settings, "rendering/gaussian_splatting/streaming/prefetch_lookahead_distance");
	project_settings->set_setting("rendering/gaussian_splatting/quality/tier_preset", "low");
	project_settings->set_setting("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", true);
	project_settings->set_setting("rendering/gaussian_splatting/streaming/predictive_prefetch_enabled", false);
	project_settings->set_setting("rendering/gaussian_splatting/streaming/prefetch_lookahead_distance", 6.0f);

	Ref<GaussianSplatWorld> world_resource;
	world_resource.instantiate();
	Ref<GaussianData> data = stage1a_make_submission_test_data(5, 5.0f);
	world_resource->set_gaussian_data(data);
	Vector<GaussianSplatRenderer::StaticChunk> chunks;
	chunks.push_back(stage1a_make_submission_test_chunk(0));
	world_resource->set_static_chunks(chunks);
	Dictionary metadata;
	metadata[StringName("label")] = String("stage1b_world");
	world_resource->set_metadata(metadata);
	world_resource->set_path("res://stage1b_world.gsplatworld");

	GaussianSplatWorld3D *node = memnew(GaussianSplatWorld3D);
	REQUIRE(node != nullptr);
	node->set_auto_apply_on_ready(false);
	node->set_world(world_resource);
	node->set_lod_enabled(false);
	node->set_lod_bias(1.75f);
	node->set_max_render_distance(80.0f);
	node->set_max_splat_count(900000);
	node->set_use_frustum_culling(false);
	node->set_async_upload_enabled(false);
	node->set_opacity(0.35f);

	root->add_child(node);
	tree->process(0.0);
	node->apply_world();

	GaussianSplatSceneDirector::WorldSubmission submission;
	CHECK(director->get_world_submission(node->get_instance_id(), &submission));
	CHECK(submission.owner_id == node->get_instance_id());
	CHECK(submission.gaussian_data == data);
	CHECK(submission.static_chunks.size() == 1);
	CHECK(submission.metadata[StringName("label")] == String("stage1b_world"));
	CHECK(submission.metadata[StringName("world_path")] == String("res://stage1b_world.gsplatworld"));
	CHECK(submission.has_desired_residency_hint);
	CHECK(submission.desired_residency_hint == GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_RESIDENT);
	CHECK_FALSE((bool)submission.desired_renderer_overrides[StringName("lod_enabled")]);
	CHECK(float(submission.desired_renderer_overrides[StringName("lod_bias")]) == doctest::Approx(1.75f));
	CHECK(float(submission.desired_renderer_overrides[StringName("lod_max_distance")]) == doctest::Approx(80.0f));
	CHECK(int64_t(submission.desired_renderer_overrides[StringName("max_splats")]) == 300000);
	CHECK_FALSE((bool)submission.desired_renderer_overrides[StringName("frustum_culling")]);
	CHECK_FALSE((bool)submission.desired_renderer_overrides[StringName("async_upload_enabled")]);
	CHECK(float(submission.desired_renderer_overrides[StringName("opacity_multiplier")]) == doctest::Approx(0.35f));

	const Dictionary streaming_overrides = submission.desired_renderer_overrides[StringName("streaming")];
	CHECK((bool)streaming_overrides[StringName("override_prefetch")]);
	CHECK_FALSE((bool)streaming_overrides[StringName("predictive_prefetch_enabled")]);
	CHECK(float(streaming_overrides[StringName("prefetch_lookahead_distance")]) == doctest::Approx(12.0f));
	CHECK((bool)streaming_overrides[StringName("override_vram_budget")]);
	CHECK(int64_t(streaming_overrides[StringName("vram_budget_mb")]) == 256);
	CHECK(int64_t(streaming_overrides[StringName("vram_min_chunks")]) == 2);
	CHECK(int64_t(streaming_overrides[StringName("vram_max_chunks")]) == 32);
	CHECK((bool)streaming_overrides[StringName("override_io_source")]);
	CHECK(streaming_overrides[StringName("io_source_path")] == String("res://stage1b_world.gsplatworld"));

	Ref<GaussianSplatRenderer> renderer = node->get_renderer();
	if (renderer.is_valid()) {
		int32_t residency_hint = GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_STREAMING;
		String residency_source;
		CHECK(director->get_submission_residency_hint_for_renderer(renderer.ptr(), &residency_hint, &residency_source));
		CHECK(residency_hint == GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_RESIDENT);
		CHECK(residency_source == String("world_submission"));
		String backend_reason;
		CHECK(renderer->should_prefer_resident_backend(gs::settings::GS_ROUTE_STREAMING, &backend_reason));
		CHECK(backend_reason == String("submission_hint_resident:world_submission"));
		CHECK(renderer->get_gaussian_data() == data);
		CHECK(renderer->get_static_chunks().size() == 1);
		CHECK_FALSE(renderer->get_lod_enabled());
		CHECK(renderer->get_lod_bias() == doctest::Approx(1.75f));
		CHECK(renderer->get_lod_max_distance() == doctest::Approx(80.0f));
		CHECK(renderer->get_max_splats() == 5);
		CHECK_FALSE(renderer->get_frustum_culling());
		CHECK_FALSE(renderer->get_async_upload_enabled());
		CHECK(renderer->get_opacity_multiplier() == doctest::Approx(0.35f));
	}

	node->clear_world();
	CHECK_FALSE(director->get_world_submission(node->get_instance_id(), &submission));

	root->remove_child(node);
	memdelete(node);
	tree->process(0.0);

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][World][SceneTree][RequiresGPU] World submission renders through the resident instanced route without a streaming system") {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs == nullptr) {
		MESSAGE("Skipping test - Rendering server unavailable");
		return;
	}

	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

	Window *root = tree->get_root();
	REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	REQUIRE(project_settings != nullptr);
	ProjectSettingGuard route_guard(project_settings, "rendering/gaussian_splatting/streaming/route_policy");
	ProjectSettingGuard instance_guard(project_settings, "rendering/gaussian_splatting/instance_pipeline/enabled");
	project_settings->set_setting("rendering/gaussian_splatting/streaming/route_policy",
			int64_t(gs::settings::GS_ROUTE_STREAMING));
	project_settings->set_setting("rendering/gaussian_splatting/instance_pipeline/enabled", true);
	project_settings->emit_signal("settings_changed");

	Ref<GaussianSplatWorld> world_resource;
	world_resource.instantiate();
	Ref<GaussianData> data = stage1a_make_submission_test_data(32, 15.0f);
	world_resource->set_gaussian_data(data);
	Vector<GaussianSplatRenderer::StaticChunk> chunks;
	chunks.push_back(stage1a_make_submission_test_chunk(0));
	world_resource->set_static_chunks(chunks);

	GaussianSplatWorld3D *node = memnew(GaussianSplatWorld3D);
	REQUIRE(node != nullptr);
	node->set_auto_apply_on_ready(false);
	node->set_world(world_resource);
	root->add_child(node);
	tree->process(0.0);
	node->apply_world();

	Ref<GaussianSplatRenderer> renderer = node->get_renderer();
	if (!renderer.is_valid()) {
		MESSAGE("Skipping test - renderer unavailable");
		root->remove_child(node);
		memdelete(node);
		tree->process(0.0);
		return;
	}

	renderer->get_debug_state().show_performance_hud = true;
	renderer->test_release_current_streaming_system();
	CHECK_FALSE(renderer->test_has_current_streaming_system());

	RenderSceneDataRD scene_data;
	scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
	scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

	RenderDataRD render_data;
	render_data.scene_data = &scene_data;
	render_data.render_buffers = Ref<RenderSceneBuffersRD>();

	renderer->render_scene_instance(&render_data);

	CHECK_FALSE(renderer->test_has_current_streaming_system());
	CHECK(renderer->has_instance_pipeline_buffers());
	CHECK(renderer->has_instance_asset_remap());

	const Dictionary stats = renderer->get_render_stats();
	CHECK(stats.get("route_uid", String()) == String("INSTANCE.RESIDENT"));
	CHECK(stats.get("requested_route_policy", String()) == String("streaming"));
	CHECK(stats.get("instance_backend_policy", String()) == String("resident"));
	CHECK(stats.get("backend_selection_reason", String()) == String("submission_hint_resident:world_submission"));
	CHECK(bool(stats.get("instance_contract_ready", false)));
	CHECK(stats.get("data_source", String()) == String("ResidentInstanceAtlas"));

	root->remove_child(node);
	memdelete(node);
	tree->process(0.0);
}

TEST_CASE("[GaussianSplatting][World][SceneTree][RequiresGPU] Resident rejection preserves chained streaming fallback diagnostics") {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs == nullptr) {
		MESSAGE("Skipping test - Rendering server unavailable");
		return;
	}

	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

	Window *root = tree->get_root();
	REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	REQUIRE(project_settings != nullptr);
	ProjectSettingGuard route_guard(project_settings, "rendering/gaussian_splatting/streaming/route_policy");
	ProjectSettingGuard instance_guard(project_settings, "rendering/gaussian_splatting/instance_pipeline/enabled");
	project_settings->set_setting("rendering/gaussian_splatting/streaming/route_policy",
			int64_t(gs::settings::GS_ROUTE_STREAMING));
	project_settings->set_setting("rendering/gaussian_splatting/instance_pipeline/enabled", true);
	project_settings->emit_signal("settings_changed");

	const QuantizationConfig saved_quantization_config = g_quantization_config;
	g_quantization_config.per_chunk_quantization = true;
	g_quantization_config.position_bits = 16;
	g_quantization_config.scale_bits = 12;
	g_quantization_config.quantize_scales = false;

	Ref<GaussianSplatWorld> world_resource;
	world_resource.instantiate();
	Ref<GaussianData> data = stage1a_make_submission_test_data(32, 20.0f);
	world_resource->set_gaussian_data(data);
	Vector<GaussianSplatRenderer::StaticChunk> chunks;
	chunks.push_back(stage1a_make_submission_test_chunk(0));
	world_resource->set_static_chunks(chunks);

	GaussianSplatWorld3D *node = memnew(GaussianSplatWorld3D);
	REQUIRE(node != nullptr);
	node->set_auto_apply_on_ready(false);
	node->set_world(world_resource);
	root->add_child(node);
	tree->process(0.0);
	node->apply_world();

	Ref<GaussianSplatRenderer> renderer = node->get_renderer();
	if (!renderer.is_valid()) {
		MESSAGE("Skipping test - renderer unavailable");
		root->remove_child(node);
		memdelete(node);
		tree->process(0.0);
		g_quantization_config = saved_quantization_config;
		return;
	}

	renderer->get_debug_state().show_performance_hud = true;
	renderer->test_release_current_streaming_system();
	CHECK_FALSE(renderer->test_has_current_streaming_system());

	RenderSceneDataRD scene_data;
	scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
	scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

	RenderDataRD render_data;
	render_data.scene_data = &scene_data;
	render_data.render_buffers = Ref<RenderSceneBuffersRD>();

	renderer->render_scene_instance(&render_data);

	CHECK(renderer->test_has_current_streaming_system());
	const Dictionary stats = renderer->get_render_stats();
	CHECK(stats.get("route_uid", String()) == String("INSTANCE.STREAMING"));
	CHECK(stats.get("requested_route_policy", String()) == String("streaming"));
	CHECK(stats.get("instance_backend_policy", String()) == String("streaming"));
	CHECK(stats.get("backend_selection_reason", String()) ==
			String("submission_hint_resident:world_submission_not_feasible:resident_quantization_unsupported -> streaming_contract_published"));
	CHECK(String(stats.get("backend_selection_reason_label", String())).find(
			"Resident was requested by the world submission") != -1);
	CHECK(String(stats.get("backend_selection_reason_label", String())).find(
			"quantized resident data cannot publish the resident instance contract") != -1);
	CHECK(String(stats.get("backend_selection_reason_label", String())).find(
			"published the streaming instance contract") != -1);

	root->remove_child(node);
	memdelete(node);
	tree->process(0.0);
	g_quantization_config = saved_quantization_config;
}

TEST_CASE("[GaussianSplatting][World][SceneTree][RequiresGPU] Explicit resident quantization rejection falls back to the legacy resident path") {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs == nullptr) {
		MESSAGE("Skipping test - Rendering server unavailable");
		return;
	}

	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

	Window *root = tree->get_root();
	REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	REQUIRE(project_settings != nullptr);
	ProjectSettingGuard route_guard(project_settings, "rendering/gaussian_splatting/streaming/route_policy");
	ProjectSettingGuard instance_guard(project_settings, "rendering/gaussian_splatting/instance_pipeline/enabled");
	project_settings->set_setting("rendering/gaussian_splatting/streaming/route_policy",
			int64_t(gs::settings::GS_ROUTE_RESIDENT));
	project_settings->set_setting("rendering/gaussian_splatting/instance_pipeline/enabled", true);
	project_settings->emit_signal("settings_changed");

	const QuantizationConfig saved_quantization_config = g_quantization_config;
	g_quantization_config.per_chunk_quantization = true;
	g_quantization_config.position_bits = 16;
	g_quantization_config.scale_bits = 12;
	g_quantization_config.quantize_scales = false;

	Ref<GaussianSplatWorld> world_resource;
	world_resource.instantiate();
	Ref<GaussianData> data = stage1a_make_submission_test_data(32, 20.0f);
	world_resource->set_gaussian_data(data);
	Vector<GaussianSplatRenderer::StaticChunk> chunks;
	chunks.push_back(stage1a_make_submission_test_chunk(0));
	world_resource->set_static_chunks(chunks);

	GaussianSplatWorld3D *node = memnew(GaussianSplatWorld3D);
	REQUIRE(node != nullptr);
	node->set_auto_apply_on_ready(false);
	node->set_world(world_resource);
	root->add_child(node);
	tree->process(0.0);
	node->apply_world();

	Ref<GaussianSplatRenderer> renderer = node->get_renderer();
	if (!renderer.is_valid()) {
		MESSAGE("Skipping test - renderer unavailable");
		root->remove_child(node);
		memdelete(node);
		tree->process(0.0);
		g_quantization_config = saved_quantization_config;
		return;
	}

	renderer->get_debug_state().show_performance_hud = true;
	renderer->test_release_current_streaming_system();
	CHECK_FALSE(renderer->test_has_current_streaming_system());

	RenderSceneDataRD scene_data;
	scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
	scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

	RenderDataRD render_data;
	render_data.scene_data = &scene_data;
	render_data.render_buffers = Ref<RenderSceneBuffersRD>();

	renderer->render_scene_instance(&render_data);

	CHECK_FALSE(renderer->test_has_current_streaming_system());
	CHECK_FALSE(renderer->has_instance_pipeline_buffers());
	CHECK_FALSE(renderer->has_instance_asset_remap());
	CHECK(renderer->get_instance_backend_policy() == GaussianRenderPipeline::InstanceBackendPolicy::RESIDENT);
	CHECK_FALSE(renderer->is_instance_contract_ready());
	CHECK(renderer->has_rendered_content());
	CHECK(renderer->get_visible_splat_count() > 0);

	const Dictionary stats = renderer->get_render_stats();
	CHECK(stats.get("requested_route_policy", String()) == String("resident"));
	CHECK(stats.get("instance_backend_policy", String()) == String("resident"));
	CHECK(stats.get("backend_selection_reason", String()) == String("requested_resident_policy"));
	CHECK(stats.get("backend_selection_reason_label", String()) == String("Resident was requested by the route policy"));
	CHECK_FALSE(bool(stats.get("instance_contract_ready", true)));
	CHECK(stats.get("route_uid", String()) != String("INSTANCE.RESIDENT"));
	CHECK(stats.get("route_uid", String()) != String("INSTANCE.STREAMING"));

	// Direct assertions stop at "resident was requested, no streaming system was used, and no
	// resident instance contract/remap survived publication." The current renderer diagnostics do
	// not expose a dedicated legacy-resident route token, so the final legacy-resident path is
	// proven indirectly by the successful render under those conditions.

	root->remove_child(node);
	memdelete(node);
	tree->process(0.0);
	g_quantization_config = saved_quantization_config;
}

TEST_CASE("[GaussianSplatting][World][SceneTree] World node preserves prior renderer streaming overrides when tier budgets are disabled") {
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

	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	REQUIRE(project_settings != nullptr);
	ProjectSettingGuard tier_preset_guard(project_settings, "rendering/gaussian_splatting/quality/tier_preset");
	ProjectSettingGuard tier_apply_guard(project_settings, "rendering/gaussian_splatting/quality/tier_apply_streaming_budgets");
	ProjectSettingGuard predictive_guard(project_settings, "rendering/gaussian_splatting/streaming/predictive_prefetch_enabled");
	ProjectSettingGuard prefetch_guard(project_settings, "rendering/gaussian_splatting/streaming/prefetch_lookahead_distance");
	project_settings->set_setting("rendering/gaussian_splatting/quality/tier_preset", "low");
	project_settings->set_setting("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", true);
	project_settings->set_setting("rendering/gaussian_splatting/streaming/predictive_prefetch_enabled", false);
	project_settings->set_setting("rendering/gaussian_splatting/streaming/prefetch_lookahead_distance", 6.0f);

	Ref<GaussianSplatWorld> world_resource;
	world_resource.instantiate();
	world_resource->set_gaussian_data(stage1a_make_submission_test_data(5, 50.0f));
	Vector<GaussianSplatRenderer::StaticChunk> chunks;
	chunks.push_back(stage1a_make_submission_test_chunk(0));
	world_resource->set_static_chunks(chunks);

	GaussianSplatWorld3D *node = memnew(GaussianSplatWorld3D);
	REQUIRE(node != nullptr);
	node->set_auto_apply_on_ready(false);
	node->set_world(world_resource);
	node->set_max_render_distance(80.0f);
	root->add_child(node);
	tree->process(0.0);
	node->apply_world();

	Ref<GaussianSplatRenderer> renderer = node->get_renderer();
	if (!renderer.is_valid()) {
		MESSAGE("Skipping test - renderer unavailable");
		root->remove_child(node);
		memdelete(node);
		if (owns_director) {
			memdelete(director);
		}
		return;
	}

	const GaussianStreamingSystem::ConfigOverrides before_overrides = renderer->get_streaming_config_overrides();
	CHECK(before_overrides.override_prefetch);
	CHECK_FALSE(before_overrides.predictive_prefetch_enabled);
	CHECK(before_overrides.prefetch_lookahead_distance == doctest::Approx(12.0f));
	CHECK(before_overrides.override_vram_budget);

	project_settings->set_setting("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", false);
	node->set_max_render_distance(120.0f);

	GaussianSplatSceneDirector::WorldSubmission submission;
	CHECK(director->get_world_submission(node->get_instance_id(), &submission));
	CHECK_FALSE(submission.desired_renderer_overrides.has(StringName("streaming")));

	const GaussianStreamingSystem::ConfigOverrides after_overrides = renderer->get_streaming_config_overrides();
	CHECK(after_overrides.override_prefetch == before_overrides.override_prefetch);
	CHECK(after_overrides.predictive_prefetch_enabled == before_overrides.predictive_prefetch_enabled);
	CHECK(after_overrides.prefetch_lookahead_distance == doctest::Approx(before_overrides.prefetch_lookahead_distance));
	CHECK(after_overrides.override_vram_budget == before_overrides.override_vram_budget);
	CHECK(after_overrides.vram_budget_config.budget_mb == before_overrides.vram_budget_config.budget_mb);
	CHECK(after_overrides.vram_budget_config.min_chunks == before_overrides.vram_budget_config.min_chunks);
	CHECK(after_overrides.vram_budget_config.max_chunks == before_overrides.vram_budget_config.max_chunks);

	root->remove_child(node);
	memdelete(node);
	tree->process(0.0);

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

	root->remove_child(node);
	memdelete(node);
	tree->process(0.0);

	CHECK_FALSE(director->get_instance_submission(submission.node_id, &submission));
	counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions);

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][SceneDirector][SceneTree] Explicit instance submission entrypoints round-trip") {
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
	root->add_child(node);
	tree->process(0.0);

	Ref<GaussianSplatAsset> asset = stage1a_make_submission_test_asset(8.0f);
	const Transform3D initial_transform(Basis(), Vector3(1.0f, 2.0f, 3.0f));
	const Transform3D updated_transform(Basis(), Vector3(4.0f, 5.0f, 6.0f));
	const Vector3 updated_wind_direction(0.0f, 1.0f, 0.0f);

	director->register_instance_submission(node->get_instance_id(), asset, initial_transform,
			0.25f, 1.5f, 0u, true, 0.8f,
			GaussianSplatSceneDirector::INSTANCE_WIND_FORCE_ENABLED,
			Vector3(1.0f, 0.0f, 0.0f), 2.0f, true,
			true, GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_STREAMING);

	GaussianSplatSceneDirector::InstanceSubmission submission;
	CHECK(director->get_instance_submission(node->get_instance_id(), &submission));
	CHECK(submission.node_id == node->get_instance_id());
	CHECK(submission.asset == asset);
	CHECK(submission.transform.origin.is_equal_approx(initial_transform.origin));
	CHECK(submission.opacity == doctest::Approx(0.25f));
	CHECK(submission.lod_bias == doctest::Approx(1.5f));
	CHECK(submission.casts_shadow);
	CHECK(submission.visible);
	CHECK(submission.wind_intensity == doctest::Approx(0.8f));
	CHECK(submission.wind_mode == GaussianSplatSceneDirector::INSTANCE_WIND_FORCE_ENABLED);
	CHECK(submission.wind_direction.is_equal_approx(Vector3(1.0f, 0.0f, 0.0f)));
	CHECK(submission.wind_frequency == doctest::Approx(2.0f));
	CHECK(submission.has_desired_residency_hint);
	CHECK(submission.desired_residency_hint == GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_STREAMING);
	int32_t renderer_hint = GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_RESIDENT;
	String renderer_hint_source;
	if (submission.renderer.is_valid()) {
		CHECK(director->get_submission_residency_hint_for_renderer(submission.renderer.ptr(), &renderer_hint, &renderer_hint_source));
		CHECK(renderer_hint == GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_STREAMING);
		CHECK(renderer_hint_source == String("instance_submission"));
	}

	director->update_instance_submission_transform(node->get_instance_id(), updated_transform);
	director->update_instance_submission_params(node->get_instance_id(), 0.6f, 0.9f, 0u, false, 1.2f,
			GaussianSplatSceneDirector::INSTANCE_WIND_FORCE_DISABLED,
			updated_wind_direction, 3.5f, false,
			true, GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_RESIDENT);

	CHECK(director->get_instance_submission(node->get_instance_id(), &submission));
	CHECK(submission.transform.origin.is_equal_approx(updated_transform.origin));
	CHECK(submission.opacity == doctest::Approx(0.6f));
	CHECK(submission.lod_bias == doctest::Approx(0.9f));
	CHECK_FALSE(submission.casts_shadow);
	CHECK_FALSE(submission.visible);
	CHECK(submission.wind_intensity == doctest::Approx(1.2f));
	CHECK(submission.wind_mode == GaussianSplatSceneDirector::INSTANCE_WIND_FORCE_DISABLED);
	CHECK(submission.wind_direction.is_equal_approx(updated_wind_direction));
	CHECK(submission.wind_frequency == doctest::Approx(3.5f));
	CHECK(submission.has_desired_residency_hint);
	CHECK(submission.desired_residency_hint == GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_RESIDENT);
	if (submission.renderer.is_valid()) {
		CHECK(director->get_submission_residency_hint_for_renderer(submission.renderer.ptr(), &renderer_hint, &renderer_hint_source));
		CHECK(renderer_hint == GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_RESIDENT);
		CHECK(renderer_hint_source == String("instance_submission"));
	}

	GaussianSplatSceneDirector::SubmissionCounts counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions + 1);
	CHECK(counts.world_submissions == baseline_counts.world_submissions);

	director->unregister_instance_submission(node->get_instance_id());
	CHECK_FALSE(director->get_instance_submission(node->get_instance_id(), &submission));

	root->remove_child(node);
	memdelete(node);
	tree->process(0.0);

	counts = director->get_submission_counts();
	CHECK(counts.instance_submissions == baseline_counts.instance_submissions);
	CHECK(counts.world_submissions == baseline_counts.world_submissions);

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][SceneDirector][SceneTree] Mixed instance residency hints collapse to no effective renderer hint") {
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

	GaussianSplatNode3D *node_a = memnew(GaussianSplatNode3D);
	GaussianSplatNode3D *node_b = memnew(GaussianSplatNode3D);
	REQUIRE(node_a != nullptr);
	REQUIRE(node_b != nullptr);
	root->add_child(node_a);
	root->add_child(node_b);
	tree->process(0.0);

	director->register_instance_submission(node_a->get_instance_id(), stage1a_make_submission_test_asset(2.0f),
			Transform3D(), 1.0f, 0.0f, 0u, false, 1.0f,
			GaussianSplatSceneDirector::INSTANCE_WIND_INHERIT, Vector3(), 1.0f, true,
			true, GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_RESIDENT);
	director->register_instance_submission(node_b->get_instance_id(), stage1a_make_submission_test_asset(12.0f),
			Transform3D(), 1.0f, 0.0f, 0u, false, 1.0f,
			GaussianSplatSceneDirector::INSTANCE_WIND_INHERIT, Vector3(), 1.0f, true,
			true, GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_STREAMING);

	GaussianSplatSceneDirector::InstanceSubmission submission_a;
	GaussianSplatSceneDirector::InstanceSubmission submission_b;
	CHECK(director->get_instance_submission(node_a->get_instance_id(), &submission_a));
	CHECK(director->get_instance_submission(node_b->get_instance_id(), &submission_b));
	if (!submission_a.renderer.is_valid() || !submission_b.renderer.is_valid()) {
		MESSAGE("Skipping renderer-hint checks — shared renderer unavailable (headless)");
		director->unregister_instance_submission(node_a->get_instance_id());
		director->unregister_instance_submission(node_b->get_instance_id());
		root->remove_child(node_a);
		root->remove_child(node_b);
		memdelete(node_a);
		memdelete(node_b);
		tree->process(0.0);
		if (owns_director) {
			memdelete(director);
		}
		return;
	}
	CHECK(submission_a.renderer == submission_b.renderer);
	CHECK_FALSE(director->has_world_submission_for_renderer(submission_a.renderer.ptr()));

	int32_t renderer_hint = GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_RESIDENT;
	String renderer_hint_source;
	CHECK_FALSE(director->get_submission_residency_hint_for_renderer(submission_a.renderer.ptr(),
			&renderer_hint, &renderer_hint_source));
	CHECK(renderer_hint_source == String("mixed_instance_submissions"));

	director->unregister_instance_submission(node_a->get_instance_id());
	director->unregister_instance_submission(node_b->get_instance_id());

	root->remove_child(node_a);
	root->remove_child(node_b);
	memdelete(node_a);
	memdelete(node_b);
	tree->process(0.0);

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][SceneDirector][SceneTree] Active world residency hint takes precedence over instance hints") {
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

	Ref<GaussianSplatWorld> world_resource;
	world_resource.instantiate();
	world_resource->set_gaussian_data(stage1a_make_submission_test_data(8, 4.0f));
	Vector<GaussianSplatRenderer::StaticChunk> chunks;
	chunks.push_back(stage1a_make_submission_test_chunk(0));
	world_resource->set_static_chunks(chunks);

	GaussianSplatWorld3D *world_node = memnew(GaussianSplatWorld3D);
	GaussianSplatNode3D *instance_node = memnew(GaussianSplatNode3D);
	REQUIRE(world_node != nullptr);
	REQUIRE(instance_node != nullptr);
	world_node->set_auto_apply_on_ready(false);
	world_node->set_world(world_resource);
	root->add_child(world_node);
	root->add_child(instance_node);
	tree->process(0.0);
	world_node->apply_world();

	Ref<GaussianSplatRenderer> renderer = world_node->get_renderer();
	if (!renderer.is_valid()) {
		MESSAGE("Skipping test - renderer unavailable");
		root->remove_child(world_node);
		root->remove_child(instance_node);
		memdelete(world_node);
		memdelete(instance_node);
		tree->process(0.0);
		if (owns_director) {
			memdelete(director);
		}
		return;
	}

	director->register_instance_submission(instance_node->get_instance_id(), stage1a_make_submission_test_asset(18.0f),
			Transform3D(), 1.0f, 0.0f, 0u, false, 1.0f,
			GaussianSplatSceneDirector::INSTANCE_WIND_INHERIT, Vector3(), 1.0f, true,
			true, GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_STREAMING);

	int32_t renderer_hint = GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_STREAMING;
	String renderer_hint_source;
	CHECK(director->get_submission_residency_hint_for_renderer(renderer.ptr(), &renderer_hint, &renderer_hint_source));
	CHECK(renderer_hint == GaussianSplatSceneDirector::SUBMISSION_RESIDENCY_HINT_RESIDENT);
	CHECK(renderer_hint_source == String("world_submission"));

	director->unregister_instance_submission(instance_node->get_instance_id());
	world_node->clear_world();

	root->remove_child(world_node);
	root->remove_child(instance_node);
	memdelete(world_node);
	memdelete(instance_node);
	tree->process(0.0);

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][SceneDirector][SceneTree] Shared renderer survives temporary last-instance unregister") {
	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

	Window *root = tree->get_root();
	REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

	Ref<World3D> world = root->get_world_3d();
	REQUIRE(world.is_valid());

	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	const bool owns_director = (director == nullptr);
	if (!director) {
		director = memnew(GaussianSplatSceneDirector);
	}
	REQUIRE(director != nullptr);
	const GaussianSplatSceneDirector::SubmissionCounts baseline_counts = director->get_submission_counts();

	GaussianSplatNode3D *node = memnew(GaussianSplatNode3D);
	REQUIRE(node != nullptr);
	node->set_splat_asset(stage1a_make_submission_test_asset(6.0f));
	root->add_child(node);
	tree->process(0.0);

	Ref<GaussianSplatRenderer> retained_renderer = node->get_renderer();
	if (!retained_renderer.is_valid()) {
		MESSAGE("Skipping renderer-retention test - shared renderer unavailable");
		root->remove_child(node);
		memdelete(node);
		tree->process(0.0);
		if (owns_director) {
			memdelete(director);
		}
		return;
	}

	CHECK(director->get_submission_counts().instance_submissions == baseline_counts.instance_submissions + 1);

	root->remove_child(node);
	tree->process(0.0);

	CHECK(director->get_submission_counts().instance_submissions == baseline_counts.instance_submissions);

	Ref<GaussianSplatRenderer> shared_renderer = director->get_shared_renderer(world.ptr());
	CHECK(shared_renderer == retained_renderer);

	root->add_child(node);
	tree->process(0.0);

	CHECK(node->get_renderer() == retained_renderer);
	CHECK(director->get_submission_counts().instance_submissions == baseline_counts.instance_submissions + 1);

	root->remove_child(node);
	memdelete(node);
	tree->process(0.0);

	if (owns_director) {
		memdelete(director);
	}
}

TEST_CASE("[GaussianSplatting][SceneDirector][SceneTree] Active world submission survives last-instance unregister") {
	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE_MESSAGE(tree != nullptr, "SceneTree singleton required");

	Window *root = tree->get_root();
	REQUIRE_MESSAGE(root != nullptr, "SceneTree root window required");

	Ref<World3D> world = root->get_world_3d();
	REQUIRE(world.is_valid());

	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	const bool owns_director = (director == nullptr);
	if (!director) {
		director = memnew(GaussianSplatSceneDirector);
	}
	REQUIRE(director != nullptr);

	Ref<GaussianSplatWorld> world_resource;
	world_resource.instantiate();
	world_resource->set_gaussian_data(stage1a_make_submission_test_data(8, 4.0f));
	Vector<GaussianSplatRenderer::StaticChunk> chunks;
	chunks.push_back(stage1a_make_submission_test_chunk(0));
	world_resource->set_static_chunks(chunks);

	GaussianSplatWorld3D *world_node = memnew(GaussianSplatWorld3D);
	GaussianSplatNode3D *instance_node = memnew(GaussianSplatNode3D);
	REQUIRE(world_node != nullptr);
	REQUIRE(instance_node != nullptr);
	world_node->set_auto_apply_on_ready(false);
	world_node->set_world(world_resource);
	instance_node->set_splat_asset(stage1a_make_submission_test_asset(18.0f));
	root->add_child(world_node);
	root->add_child(instance_node);
	tree->process(0.0);
	world_node->apply_world();

	Ref<GaussianSplatRenderer> renderer = world_node->get_renderer();
	if (!renderer.is_valid()) {
		MESSAGE("Skipping active-world retention test - renderer unavailable");
		root->remove_child(world_node);
		root->remove_child(instance_node);
		memdelete(world_node);
		memdelete(instance_node);
		tree->process(0.0);
		if (owns_director) {
			memdelete(director);
		}
		return;
	}

	GaussianSplatSceneDirector::WorldSubmission queried_submission;
	CHECK(director->has_world_submission_for_renderer(renderer.ptr()));
	CHECK(director->get_world_submission_for_scenario(world->get_scenario(), &queried_submission));

	root->remove_child(instance_node);
	tree->process(0.0);

	CHECK(director->has_world_submission_for_renderer(renderer.ptr()));
	CHECK(director->get_world_submission_for_scenario(world->get_scenario(), &queried_submission));
	CHECK(director->get_shared_renderer(world.ptr()) == renderer);

	world_node->clear_world();
	root->remove_child(world_node);
	memdelete(world_node);
	memdelete(instance_node);
	tree->process(0.0);

	if (owns_director) {
		memdelete(director);
	}
}

#endif // TESTS_ENABLED || TOOLS_ENABLED
