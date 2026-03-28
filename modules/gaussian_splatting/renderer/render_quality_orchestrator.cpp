#include "render_quality_orchestrator.h"

#include "../logger/gs_logger.h"
#include "../core/gaussian_splat_scene_director.h"
#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/object/callable_method_pointer.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "servers/rendering_server.h"

namespace {

struct FidelityOverrideSnapshot {
	bool lod_enabled = true;
	bool opacity_aware_culling = true;
	float visibility_threshold = 0.01f;
	bool distance_cull_enabled = true;
	float distance_cull_start = 30.0f;
	float distance_cull_max_rate = 0.5f;
	float importance_threshold = 0.0f;
	float importance_baseline = 0.0f;
	float tiny_radius = 0.0f;
	float tiny_baseline = 0.0f;
	bool valid = false;
};

static HashMap<uint64_t, FidelityOverrideSnapshot> g_fidelity_override_snapshots;

static int _metadata_int(const Dictionary &p_metadata, const StringName &p_key, int p_default) {
	if (!p_metadata.has(p_key)) {
		return p_default;
	}
	const Variant value = p_metadata[p_key];
	if (value.get_type() == Variant::FLOAT) {
		return int((double)value);
	}
	return int(value);
}

static double _metadata_double(const Dictionary &p_metadata, const StringName &p_key, double p_default) {
	if (!p_metadata.has(p_key)) {
		return p_default;
	}
	const Variant value = p_metadata[p_key];
	if (value.get_type() == Variant::INT) {
		return double(int64_t(value));
	}
	return (double)value;
}

static bool _asset_requests_full_fidelity_runtime(const Ref<GaussianSplatAsset> &p_asset) {
	if (p_asset.is_null()) {
		return false;
	}
	const Dictionary import_metadata = p_asset->get_import_metadata();
	const int import_max_splats = _metadata_int(import_metadata, StringName("max_splats"), -1);
	const double density_multiplier = _metadata_double(import_metadata, StringName("density_multiplier"), 1.0);
	return import_max_splats == 0 && density_multiplier >= 0.999;
}

static bool _renderer_requests_conservative_full_fidelity_runtime(const GaussianSplatRenderer *p_renderer,
		const Ref<GaussianSplatAsset> &p_active_asset) {
	if (_asset_requests_full_fidelity_runtime(p_active_asset)) {
		return true;
	}
	if (!p_renderer) {
		return false;
	}
	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	if (!director) {
		return false;
	}

	LocalVector<InstanceAssetRegistration> instance_assets;
	director->collect_instance_assets_for_renderer(p_renderer, instance_assets,
			p_renderer->is_shadow_instance_filter_enabled());
	if (instance_assets.is_empty()) {
		return false;
	}
	for (const InstanceAssetRegistration &entry : instance_assets) {
		if (entry.requests_full_fidelity_runtime) {
			return true;
		}
	}
	if (p_active_asset.is_null()) {
		return true;
	}
	const Ref<GaussianData> active_data = p_active_asset->get_gaussian_data();
	if (active_data.is_null()) {
		return true;
	}
	for (const InstanceAssetRegistration &entry : instance_assets) {
		if (entry.data.is_null()) {
			continue;
		}
		if (entry.data == active_data) {
			return false;
		}
	}
	return true;
}

static void _apply_source_fidelity_overrides(uint64_t p_renderer_key, GPUCuller *p_gpu_culler, bool p_enable) {
	if (!p_gpu_culler) {
		return;
	}

	GPUCuller::CullingConfig &config = p_gpu_culler->get_config();
	GPUCuller::CullingState &state = p_gpu_culler->get_state();

	if (p_enable) {
		FidelityOverrideSnapshot *existing = g_fidelity_override_snapshots.getptr(p_renderer_key);
		if (!existing) {
			FidelityOverrideSnapshot snapshot;
			snapshot.lod_enabled = config.lod_enabled;
			snapshot.opacity_aware_culling = config.opacity_aware_culling;
			snapshot.visibility_threshold = config.visibility_threshold;
			snapshot.distance_cull_enabled = config.distance_cull_enabled;
			snapshot.distance_cull_start = config.distance_cull_start;
			snapshot.distance_cull_max_rate = config.distance_cull_max_rate;
			snapshot.importance_threshold = config.importance_cull_threshold;
			snapshot.importance_baseline = config.importance_cull_baseline;
			snapshot.tiny_radius = state.tiny_splat_screen_radius_px;
			snapshot.tiny_baseline = state.tiny_splat_screen_radius_baseline;
			snapshot.valid = true;
			g_fidelity_override_snapshots.insert(p_renderer_key, snapshot);
		}

		config.lod_enabled = false;
		config.lod_cache_dirty = true;
		config.importance_cull_threshold = 0.0f;
		config.importance_cull_baseline = 0.0f;
		config.opacity_aware_culling = false;
		config.visibility_threshold = 0.0f;
		config.distance_cull_enabled = false;
		config.distance_cull_start = 0.0f;
		config.distance_cull_max_rate = 0.0f;
		config.cull_params_dirty = true;
		state.tiny_splat_screen_radius_px = 0.0f;
		state.tiny_splat_screen_radius_baseline = 0.0f;
		return;
	}

	FidelityOverrideSnapshot *snapshot = g_fidelity_override_snapshots.getptr(p_renderer_key);
	if (!snapshot || !snapshot->valid) {
		return;
	}

	config.lod_enabled = snapshot->lod_enabled;
	config.lod_cache_dirty = true;
	config.opacity_aware_culling = snapshot->opacity_aware_culling;
	config.visibility_threshold = snapshot->visibility_threshold;
	config.distance_cull_enabled = snapshot->distance_cull_enabled;
	config.distance_cull_start = snapshot->distance_cull_start;
	config.distance_cull_max_rate = snapshot->distance_cull_max_rate;
	config.importance_cull_threshold = snapshot->importance_threshold;
	config.importance_cull_baseline = snapshot->importance_baseline;
	config.cull_params_dirty = true;
	state.tiny_splat_screen_radius_px = snapshot->tiny_radius;
	state.tiny_splat_screen_radius_baseline = snapshot->tiny_baseline;
	g_fidelity_override_snapshots.erase(p_renderer_key);
}

} // namespace

RenderQualityOrchestrator::RenderQualityOrchestrator(const Dependencies &p_dependencies) :
		renderer(p_dependencies.renderer),
		gpu_culler(p_dependencies.gpu_culler),
		test_data_state(p_dependencies.test_data_state),
		runtime_ports(p_dependencies.runtime_ports) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(gpu_culler);
	ERR_FAIL_NULL(test_data_state);
	ERR_FAIL_COND_MSG(!runtime_ports.refresh_gpu_sorter, "RenderQualityOrchestrator requires a GPU sorter refresh callback.");
	ERR_FAIL_COND_MSG(!runtime_ports.track_resource_owner, "RenderQualityOrchestrator requires a resource-owner tracking callback.");
	ERR_FAIL_COND_MSG(!runtime_ports.get_streaming_state_mut, "RenderQualityOrchestrator requires mutable streaming-state access.");
	ERR_FAIL_COND_MSG(!runtime_ports.get_streaming_state_view, "RenderQualityOrchestrator requires const streaming-state access.");
}

void RenderQualityOrchestrator::set_lod_enabled(bool p_enabled) {
	if (gpu_culler->get_config().lod_enabled == p_enabled) {
		return;
	}
	gpu_culler->get_config().lod_enabled = p_enabled;
	gpu_culler->invalidate_lod_cache();
}

void RenderQualityOrchestrator::set_lod_bias(float p_bias) {
	float clamped = CLAMP(p_bias, 0.01f, 8.0f);
	if (!Math::is_equal_approx(gpu_culler->get_config().lod_bias, clamped)) {
		gpu_culler->get_config().lod_bias = clamped;
	}
	gpu_culler->get_config().lod_bias_override = true;
	gpu_culler->invalidate_lod_cache();
}

void RenderQualityOrchestrator::set_lod_min_screen_size(float p_pixels) {
	float clamped = CLAMP(p_pixels, 0.0f, 128.0f);
	if (!Math::is_equal_approx(gpu_culler->get_config().lod_min_screen_size, clamped)) {
		gpu_culler->get_config().lod_min_screen_size = clamped;
	}
	gpu_culler->get_config().lod_min_screen_size_override = true;
	gpu_culler->invalidate_lod_cache();
}

void RenderQualityOrchestrator::set_lod_max_distance(float p_distance) {
	float clamped = CLAMP(p_distance, 0.0f, 10000.0f);
	if (!Math::is_equal_approx(gpu_culler->get_config().lod_max_distance, clamped)) {
		gpu_culler->get_config().lod_max_distance = clamped;
	}
	gpu_culler->get_config().lod_max_distance_override = true;
	gpu_culler->invalidate_lod_cache();
}

void RenderQualityOrchestrator::set_importance_cull_threshold(float p_threshold) {
	float clamped = MAX(0.0f, p_threshold);
	if (!Math::is_equal_approx(gpu_culler->get_config().importance_cull_threshold, clamped)) {
		gpu_culler->get_config().importance_cull_threshold = clamped;
		gpu_culler->get_config().importance_cull_baseline = clamped;
	}
	gpu_culler->get_config().importance_cull_override = true;
	gpu_culler->get_config().cull_params_dirty = true;
}

void RenderQualityOrchestrator::set_cull_radius_multiplier(float p_multiplier) {
	float clamped = CLAMP(p_multiplier, 0.5f, 16.0f);
	if (!Math::is_equal_approx(gpu_culler->get_config().cull_radius_multiplier, clamped)) {
		gpu_culler->get_config().cull_radius_multiplier = clamped;
	}
	gpu_culler->get_config().cull_params_dirty = true;

	GaussianSplatRenderer::StreamingState &streaming_state = (renderer->*runtime_ports.get_streaming_state_mut)();
	if (streaming_state.current_streaming_system.is_valid()) {
		streaming_state.current_streaming_system->set_chunk_radius_multiplier(
				p_multiplier * gpu_culler->get_config().cull_frustum_plane_slack);
	}
}

void RenderQualityOrchestrator::set_cull_frustum_plane_slack(float p_slack) {
	float clamped = CLAMP(p_slack, 1.0f, 8.0f);
	if (!Math::is_equal_approx(gpu_culler->get_config().cull_frustum_plane_slack, clamped)) {
		gpu_culler->get_config().cull_frustum_plane_slack = clamped;
	}
	gpu_culler->get_config().cull_params_dirty = true;

	GaussianSplatRenderer::StreamingState &streaming_state = (renderer->*runtime_ports.get_streaming_state_mut)();
	if (streaming_state.current_streaming_system.is_valid()) {
		streaming_state.current_streaming_system->set_chunk_radius_multiplier(
				gpu_culler->get_config().cull_radius_multiplier * p_slack);
	}
}

void RenderQualityOrchestrator::set_cull_near_tolerance(float p_tolerance) {
	float clamped = CLAMP(p_tolerance, 0.0f, 1.0f);
	if (!Math::is_equal_approx(gpu_culler->get_config().cull_near_tolerance, clamped)) {
		gpu_culler->get_config().cull_near_tolerance = clamped;
	}
	gpu_culler->get_config().cull_params_dirty = true;
}

void RenderQualityOrchestrator::set_cull_far_tolerance(float p_tolerance) {
	float clamped = CLAMP(p_tolerance, 0.0f, 1.0f);
	if (!Math::is_equal_approx(gpu_culler->get_config().cull_far_tolerance, clamped)) {
		gpu_culler->get_config().cull_far_tolerance = clamped;
	}
	gpu_culler->get_config().cull_params_dirty = true;
}

void RenderQualityOrchestrator::set_tiny_splat_screen_radius(float p_pixels) {
	float clamped = MAX(0.0f, p_pixels);
	if (!Math::is_equal_approx(gpu_culler->get_state().tiny_splat_screen_radius_px, clamped)) {
		gpu_culler->get_state().tiny_splat_screen_radius_px = clamped;
		gpu_culler->get_state().tiny_splat_screen_radius_baseline = clamped;
	}
	gpu_culler->get_config().cull_params_dirty = true;
}

void RenderQualityOrchestrator::set_opacity_aware_culling(bool p_enabled) {
	if (gpu_culler->get_config().opacity_aware_culling == p_enabled) {
		return;
	}
	gpu_culler->get_config().opacity_aware_culling = p_enabled;
	gpu_culler->get_config().cull_params_dirty = true;
}

void RenderQualityOrchestrator::set_visibility_threshold(float p_threshold) {
	float clamped = CLAMP(p_threshold, 0.001f, 0.1f);
	if (!Math::is_equal_approx(gpu_culler->get_config().visibility_threshold, clamped)) {
		gpu_culler->get_config().visibility_threshold = clamped;
		gpu_culler->get_config().cull_params_dirty = true;
	}
}

void RenderQualityOrchestrator::set_distance_cull_enabled(bool p_enabled) {
	if (gpu_culler->get_config().distance_cull_enabled == p_enabled) {
		return;
	}
	gpu_culler->get_config().distance_cull_enabled = p_enabled;
}

void RenderQualityOrchestrator::set_distance_cull_start(float p_distance) {
	float clamped = MAX(p_distance, 0.0f);
	if (Math::is_equal_approx(gpu_culler->get_config().distance_cull_start, clamped)) {
		return;
	}
	gpu_culler->get_config().distance_cull_start = clamped;
}

void RenderQualityOrchestrator::set_distance_cull_max_rate(float p_rate) {
	float clamped = CLAMP(p_rate, 0.0f, 1.0f);
	if (Math::is_equal_approx(gpu_culler->get_config().distance_cull_max_rate, clamped)) {
		return;
	}
	gpu_culler->get_config().distance_cull_max_rate = clamped;
}

void RenderQualityOrchestrator::set_overflow_autotune_enabled(bool p_enabled) {
	if (gpu_culler->get_state().overflow_autotune_enabled == p_enabled) {
		return;
	}
	gpu_culler->get_state().overflow_autotune_enabled = p_enabled;
	if (!gpu_culler->get_state().overflow_autotune_enabled) {
		// Restore user baselines when disabling auto-tune.
		gpu_culler->get_config().importance_cull_threshold =
				gpu_culler->get_config().importance_cull_baseline;
		gpu_culler->get_state().tiny_splat_screen_radius_px =
				gpu_culler->get_state().tiny_splat_screen_radius_baseline;
	}
	gpu_culler->get_config().cull_params_dirty = true;
}

void RenderQualityOrchestrator::set_max_splats(int p_count) {
	ERR_FAIL_COND(p_count < 1000);
	if (p_count == performance_settings.max_splats) {
		return;
	}
	performance_settings.max_splats = p_count;
	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	GaussianSplatRenderer::IFrameMutationAccess &state_mut = state_provider;
	state_mut.get_sorting_state_mut().sorter_needs_rebuild = true;
	(renderer->*runtime_ports.refresh_gpu_sorter)("set_max_splats");
}

void RenderQualityOrchestrator::set_frustum_culling(bool p_enabled) {
	gpu_culler->get_config().frustum_culling = p_enabled;
}

void RenderQualityOrchestrator::set_async_upload_enabled(bool p_enabled) {
	GaussianSplatRenderer::StreamingState &streaming_state = (renderer->*runtime_ports.get_streaming_state_mut)();
	if (streaming_state.memory_stream.is_valid()) {
		streaming_state.memory_stream->set_async_upload(p_enabled);
	}
}

void RenderQualityOrchestrator::set_quality_preset(const String &p_preset) {
	// Apply quality presets
	String preset_lower = p_preset.to_lower();

	if (preset_lower == "quality" || preset_lower == "high" || preset_lower == "ultra") {
		performance_settings.max_splats = 1000000;
		set_lod_enabled(true);
		set_lod_bias(0.8f);
		gpu_culler->get_config().frustum_culling = true;
		gpu_culler->get_config().temporal_coherence = true;
	} else if (preset_lower == "balanced" || preset_lower == "medium") {
		performance_settings.max_splats = 500000;
		set_lod_enabled(true);
		set_lod_bias(1.0f);
		gpu_culler->get_config().frustum_culling = true;
		gpu_culler->get_config().temporal_coherence = true;
	} else if (preset_lower == "performance" || preset_lower == "low") {
		performance_settings.max_splats = 250000;
		set_lod_enabled(true);
		set_lod_bias(1.5f);
		gpu_culler->get_config().frustum_culling = true;
		gpu_culler->get_config().temporal_coherence = false;
	}

	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	GaussianSplatRenderer::IFrameMutationAccess &state_mut = state_provider;
	state_mut.get_sorting_state_mut().sorter_needs_rebuild = true;
}

String RenderQualityOrchestrator::get_quality_preset() const {
	// Return current quality level based on settings
	if (performance_settings.max_splats >= 10000000 &&
			!gpu_culler->get_config().lod_enabled) {
		return "ultra";
	} else if (performance_settings.max_splats >= 5000000) {
		return "high";
	} else if (performance_settings.max_splats >= 2000000) {
		return "medium";
	} else {
		return "low";
	}
}

bool RenderQualityOrchestrator::get_async_upload_enabled() const {
	const GaussianSplatRenderer::StreamingState &streaming_state = (renderer->*runtime_ports.get_streaming_state_view)();
	if (streaming_state.memory_stream.is_valid()) {
		return streaming_state.memory_stream->get_async_upload();
	}
	return true; // Default to enabled
}

// GPU culling pass (absorbed from RenderCullingOrchestrator, ISSUE-016).
GaussianRenderState::CullStageOutput RenderQualityOrchestrator::cull_for_view(const Transform3D &p_world_to_camera_transform,
		const Projection &p_projection, const Size2i &p_viewport_size) {
	GaussianRenderState::CullStageOutput output;
	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	GaussianSplatRenderer::IFrameMutationAccess &state_mut = state_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	GaussianSplatRenderer::FrameState &frame_state = state_mut.get_frame_state_mut();
	GaussianSplatRenderer::PerformanceState &performance_state = state_mut.get_performance_state_mut();
	auto &metrics = performance_state.metrics;
	const GaussianSplatRenderer::SceneState &scene_state = state_view.get_scene_state();
	if (!gpu_culler) {
		frame_state.visible_splat_count.store(0, std::memory_order_release);
		metrics.visible_after_culling = 0;
		metrics.culling_candidate_count = 0;
		metrics.culled_frustum_count = 0;
		metrics.culled_distance_count = 0;
		metrics.culled_screen_count = 0;
		metrics.culled_importance_count = 0;
		metrics.used_hierarchical_culling = false;
		metrics.culling_time_ms = 0.0f;
		return output;
	}

	const bool preserve_source_fidelity = _renderer_requests_conservative_full_fidelity_runtime(renderer, scene_state.active_asset);
	const uint64_t renderer_key = renderer ? uint64_t(renderer->get_instance_id()) : 0u;
	_apply_source_fidelity_overrides(renderer_key, gpu_culler, preserve_source_fidelity);

	GPUCuller::CullingInputs inputs;
	inputs.gaussian_data = scene_state.gaussian_data;
	inputs.test_positions = &test_data_state->positions;
	inputs.test_scales = &test_data_state->scales;
	const int source_splat_count = scene_state.gaussian_data.is_valid() ? scene_state.gaussian_data->get_count() : 0;
	const int max_splats = preserve_source_fidelity && source_splat_count > 0
			? source_splat_count
			: performance_settings.max_splats;
	inputs.max_splats = max_splats > 0 ? static_cast<uint32_t>(max_splats) : 0;
	// Runtime sorting path is GPU/cached only; avoid readbacks that were only needed for CPU sort fallback.
	inputs.readback_distances = false;
	inputs.readback_importance = false;

	const GaussianSplatRenderer::StreamingState &streaming_state = (renderer->*runtime_ports.get_streaming_state_view)();
	const bool can_attempt_legacy_gpu_cull =
			!streaming_state.current_streaming_system.is_valid() &&
			streaming_state.registered_gaussian_buffer.is_valid();
	if (can_attempt_legacy_gpu_cull) {
		RenderingDevice *buffer_device = renderer->get_resource_owner(streaming_state.registered_gaussian_buffer,
				renderer->get_device_state().rd);
		if (!buffer_device) {
			buffer_device = renderer->get_device_state().rd;
		}
		if (buffer_device) {
			inputs.gpu_cull_attempted = true;
			inputs.gpu_input.gaussian_buffer = streaming_state.registered_gaussian_buffer;
			inputs.gpu_input.buffer_device = buffer_device;
			inputs.gpu_input.total_splats = source_splat_count > 0 ? static_cast<uint32_t>(source_splat_count) : 0u;
		}
	}

	if (inputs.gpu_cull_attempted && inputs.gpu_input.gaussian_buffer.is_valid() && inputs.gpu_input.buffer_device) {
		(renderer->*runtime_ports.track_resource_owner)(
				inputs.gpu_input.gaussian_buffer, inputs.gpu_input.buffer_device, false, "legacy_gpu_cull_gaussian_buffer");
	}

	GPUCuller::CullingSummary summary =
			gpu_culler->cull_for_view(p_world_to_camera_transform, p_projection, p_viewport_size, inputs);
	const auto &cull_state = gpu_culler->get_state();
	if (cull_state.gpu_visible_indices_buffer.is_valid() && cull_state.gpu_visible_indices_device) {
		(renderer->*runtime_ports.track_resource_owner)(
				cull_state.gpu_visible_indices_buffer, cull_state.gpu_visible_indices_device,
				false, "gpu_cull_visible_indices");
	}

	frame_state.visible_splat_count.store(summary.visible_after_culling, std::memory_order_release);
	metrics.visible_after_culling = summary.visible_after_culling;
	metrics.culling_candidate_count = summary.culling_candidate_count;
	metrics.culled_frustum_count = summary.culled_frustum_count;
	metrics.culled_distance_count = summary.culled_distance_count;
	metrics.culled_screen_count = summary.culled_screen_count;
	metrics.culled_importance_count = summary.culled_importance_count;
	metrics.used_hierarchical_culling = summary.used_hierarchical_culling;
	metrics.culling_time_ms = summary.culling_time_ms;

	output.visible_count = summary.visible_after_culling;
	output.candidate_count = summary.culling_candidate_count;
	output.cull_time_ms = summary.culling_time_ms;
	output.has_visible = summary.visible_after_culling > 0;
	output.visible_domain = summary.used_instance_pipeline
			? GaussianRenderState::IndexDomain::CHUNK_REF
			: GaussianRenderState::IndexDomain::GAUSSIAN_GLOBAL;
	return output;
}

void GaussianSplatRenderer::set_lod_enabled(bool p_enabled) {
	quality_orchestrator->set_lod_enabled(p_enabled);
}

void GaussianSplatRenderer::set_lod_bias(float p_bias) {
	quality_orchestrator->set_lod_bias(p_bias);
}

void GaussianSplatRenderer::set_lod_min_screen_size(float p_pixels) {
	quality_orchestrator->set_lod_min_screen_size(p_pixels);
}

void GaussianSplatRenderer::set_lod_max_distance(float p_distance) {
	quality_orchestrator->set_lod_max_distance(p_distance);
}

void GaussianSplatRenderer::set_importance_cull_threshold(float p_threshold) {
	quality_orchestrator->set_importance_cull_threshold(p_threshold);
}

void GaussianSplatRenderer::set_cull_radius_multiplier(float p_multiplier) {
	quality_orchestrator->set_cull_radius_multiplier(p_multiplier);
}

void GaussianSplatRenderer::set_cull_frustum_plane_slack(float p_slack) {
	quality_orchestrator->set_cull_frustum_plane_slack(p_slack);
}

void GaussianSplatRenderer::set_cull_near_tolerance(float p_tolerance) {
	quality_orchestrator->set_cull_near_tolerance(p_tolerance);
}

void GaussianSplatRenderer::set_cull_far_tolerance(float p_tolerance) {
	quality_orchestrator->set_cull_far_tolerance(p_tolerance);
}

void GaussianSplatRenderer::set_tiny_splat_screen_radius(float p_pixels) {
	quality_orchestrator->set_tiny_splat_screen_radius(p_pixels);
}

void GaussianSplatRenderer::set_opacity_aware_culling(bool p_enabled) {
	quality_orchestrator->set_opacity_aware_culling(p_enabled);
}

void GaussianSplatRenderer::set_visibility_threshold(float p_threshold) {
	quality_orchestrator->set_visibility_threshold(p_threshold);
}

void GaussianSplatRenderer::set_distance_cull_enabled(bool p_enabled) {
	quality_orchestrator->set_distance_cull_enabled(p_enabled);
}

void GaussianSplatRenderer::set_distance_cull_start(float p_distance) {
	quality_orchestrator->set_distance_cull_start(p_distance);
}

void GaussianSplatRenderer::set_distance_cull_max_rate(float p_rate) {
	quality_orchestrator->set_distance_cull_max_rate(p_rate);
}

void GaussianSplatRenderer::set_overflow_autotune_enabled(bool p_enabled) {
	quality_orchestrator->set_overflow_autotune_enabled(p_enabled);
}

void GaussianSplatRenderer::set_max_splats(int p_count) {
	RenderingServer *rs = RenderingServer::get_singleton();
	bool dispatch_submitted = false;
	if (rs && !rs->is_on_render_thread()) {
		if (_dispatch_call_on_render_thread_blocking(
					callable_mp(this, &GaussianSplatRenderer::_set_max_splats_on_render_thread).bind(p_count),
					&dispatch_submitted)) {
			return;
		}
		if (dispatch_submitted) {
			GS_LOG_RENDERER_WARN("[GaussianSplatRenderer] set_max_splats dispatch timed out after submit; skipping unsafe local fallback");
			return;
		}
	}
	quality_orchestrator->set_max_splats(p_count);
}

void GaussianSplatRenderer::_set_max_splats_on_render_thread(int p_count, uint64_t p_request_id) {
	quality_orchestrator->set_max_splats(p_count);
	_notify_render_thread_dispatch_completed(p_request_id);
}

void GaussianSplatRenderer::set_frustum_culling(bool p_enabled) {
	quality_orchestrator->set_frustum_culling(p_enabled);
}

void GaussianSplatRenderer::set_async_upload_enabled(bool p_enabled) {
	quality_orchestrator->set_async_upload_enabled(p_enabled);
}

bool GaussianSplatRenderer::get_async_upload_enabled() const {
	return quality_orchestrator->get_async_upload_enabled();
}

void GaussianSplatRenderer::set_quality_preset(const String &p_preset) {
	quality_orchestrator->set_quality_preset(p_preset);
}

String GaussianSplatRenderer::get_quality_preset() const {
	return quality_orchestrator->get_quality_preset();
}
