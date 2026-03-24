#include "render_quality_orchestrator.h"

#include "../logger/gs_logger.h"
#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/object/callable_method_pointer.h"
#include "core/string/ustring.h"
#include "servers/rendering_server.h"

RenderQualityOrchestrator::RenderQualityOrchestrator(const Dependencies &p_dependencies) :
		renderer(p_dependencies.renderer),
		gpu_culler(p_dependencies.gpu_culler),
		runtime_ports(p_dependencies.runtime_ports) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(gpu_culler);
	ERR_FAIL_COND_MSG(!runtime_ports.refresh_gpu_sorter, "RenderQualityOrchestrator requires a GPU sorter refresh callback.");
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
}

void RenderQualityOrchestrator::set_cull_frustum_plane_slack(float p_slack) {
	float clamped = CLAMP(p_slack, 1.0f, 8.0f);
	if (!Math::is_equal_approx(gpu_culler->get_config().cull_frustum_plane_slack, clamped)) {
		gpu_culler->get_config().cull_frustum_plane_slack = clamped;
	}
	gpu_culler->get_config().cull_params_dirty = true;
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

	GPUCuller::CullingInputs inputs;
	inputs.gaussian_data = scene_state.gaussian_data;
	inputs.test_positions = &renderer->get_test_data_state().positions;
	inputs.test_scales = &renderer->get_test_data_state().scales;
	const int max_splats = performance_settings.max_splats;
	inputs.max_splats = max_splats > 0 ? static_cast<uint32_t>(max_splats) : 0;
	// Runtime sorting path is GPU/cached only; avoid readbacks that were only needed for CPU sort fallback.
	inputs.readback_distances = false;
	inputs.readback_importance = false;

	if (inputs.gpu_cull_attempted && inputs.gpu_input.gaussian_buffer.is_valid() && inputs.gpu_input.buffer_device) {
		renderer->track_resource_owner(inputs.gpu_input.gaussian_buffer, inputs.gpu_input.buffer_device, true, nullptr);
	}

	GPUCuller::CullingSummary summary =
			gpu_culler->cull_for_view(p_world_to_camera_transform, p_projection, p_viewport_size, inputs);
	const auto &cull_state = gpu_culler->get_state();
	if (cull_state.gpu_visible_indices_buffer.is_valid() && cull_state.gpu_visible_indices_device) {
		renderer->track_resource_owner(cull_state.gpu_visible_indices_buffer, cull_state.gpu_visible_indices_device,
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
	FrameStateProvider state_provider(this);
	const IFrameStateView &state_view = state_provider;
	if (state_view.get_streaming_state().current_streaming_system.is_valid()) {
		state_view.get_streaming_state().current_streaming_system->set_chunk_radius_multiplier(
				p_multiplier * get_cull_frustum_plane_slack());
	}
}

void GaussianSplatRenderer::set_cull_frustum_plane_slack(float p_slack) {
	quality_orchestrator->set_cull_frustum_plane_slack(p_slack);
	FrameStateProvider state_provider(this);
	const IFrameStateView &state_view = state_provider;
	if (state_view.get_streaming_state().current_streaming_system.is_valid()) {
		state_view.get_streaming_state().current_streaming_system->set_chunk_radius_multiplier(
				get_cull_radius_multiplier() * p_slack);
	}
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
	FrameStateProvider state_provider(this);
	const IFrameStateView &state_view = state_provider;
	if (state_view.get_streaming_state().memory_stream.is_valid()) {
		state_view.get_streaming_state().memory_stream->set_async_upload(p_enabled);
	}
}

bool GaussianSplatRenderer::get_async_upload_enabled() const {
	if (get_streaming_state().memory_stream.is_valid()) {
		return get_streaming_state().memory_stream->get_async_upload();
	}
	return true; // Default to enabled
}

void GaussianSplatRenderer::set_quality_preset(const String &p_preset) {
	quality_orchestrator->set_quality_preset(p_preset);
}

String GaussianSplatRenderer::get_quality_preset() const {
	return quality_orchestrator->get_quality_preset();
}
