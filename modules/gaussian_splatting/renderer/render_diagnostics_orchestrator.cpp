#include "render_diagnostics_orchestrator.h"

#include "../core/gs_project_settings.h"
#include "render_debug_state_orchestrator.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/math/vector2i.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "rendering_diagnostics.h"
#include "sorting_config.h"
#include "gpu_sorter.h"
#include "../interfaces/debug_overlay_system.h"
#include "../interfaces/gpu_sorting_pipeline.h"
#include "../interfaces/rasterizer_interfaces.h"
#include "../logger/gs_logger.h"

namespace {
static String _stage_status_to_string(GaussianSplatRenderer::StageResult::StageStatus p_status) {
	switch (p_status) {
		case GaussianSplatRenderer::StageResult::StageStatus::SUCCESS:
			return "success";
		case GaussianSplatRenderer::StageResult::StageStatus::SKIPPED:
			return "skipped";
		case GaussianSplatRenderer::StageResult::StageStatus::FALLBACK:
			return "fallback";
		case GaussianSplatRenderer::StageResult::StageStatus::FAILED:
			return "failed";
	}
	return "unknown";
}

static String _normalize_route_uid_for_stats(const String &p_route_uid) {
	if (p_route_uid.is_empty()) {
		return RenderRouteUID::COMMON_UNKNOWN_ROUTE;
	}
	return p_route_uid;
}

static String _normalize_sort_route_uid_for_stats(const String &p_sort_route_uid) {
	if (p_sort_route_uid.is_empty()) {
		return RenderRouteUID::COMMON_UNKNOWN_SORT_ROUTE;
	}
	return p_sort_route_uid;
}

struct ProductionMetricsConfig {
	bool validate_metrics = true;
	uint32_t summary_interval_frames = 600;
	uint32_t summary_history_size = 60;
	bool perf_gate_enabled = false;
	uint32_t perf_gate_splat_threshold = 100000;
	float perf_gate_budget_ms = 16.0f;
};

// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).
static uint32_t _get_uint_setting(ProjectSettings *p_settings, const StringName &p_name, uint32_t p_fallback) {
	return gs::settings::get_uint(p_settings, p_name, p_fallback);
}

static float _get_float_setting(ProjectSettings *p_settings, const StringName &p_name, float p_fallback) {
	return gs::settings::get_float(p_settings, p_name, p_fallback);
}

static bool _get_bool_setting(ProjectSettings *p_settings, const StringName &p_name, bool p_fallback) {
	return gs::settings::get_bool(p_settings, p_name, p_fallback);
}

static ProductionMetricsConfig _load_production_metrics_config() {
	ProductionMetricsConfig config;
	ProjectSettings *settings = ProjectSettings::get_singleton();
	if (!settings) {
		return config;
	}
	config.validate_metrics = _get_bool_setting(settings,
			"rendering/gaussian_splatting/diagnostics/validate_production_metrics",
			config.validate_metrics);
	config.summary_interval_frames = _get_uint_setting(settings,
			"rendering/gaussian_splatting/diagnostics/summary_interval_frames",
			config.summary_interval_frames);
	config.summary_history_size = _get_uint_setting(settings,
			"rendering/gaussian_splatting/diagnostics/summary_history_size",
			config.summary_history_size);
	config.perf_gate_enabled = _get_bool_setting(settings,
			"rendering/gaussian_splatting/diagnostics/perf_gate_enabled",
			config.perf_gate_enabled);
	config.perf_gate_splat_threshold = _get_uint_setting(settings,
			"rendering/gaussian_splatting/diagnostics/perf_gate_splat_threshold",
			config.perf_gate_splat_threshold);
	config.perf_gate_budget_ms = _get_float_setting(settings,
			"rendering/gaussian_splatting/diagnostics/perf_gate_budget_ms",
			config.perf_gate_budget_ms);
	if (config.summary_interval_frames == 0) {
		config.summary_interval_frames = 1;
	}
	if (config.summary_history_size == 0) {
		config.summary_history_size = 1;
	}
	if (config.perf_gate_budget_ms < 0.0f) {
		config.perf_gate_budget_ms = 0.0f;
	}
	return config;
}

static Array _production_metrics_contract() {
	Array keys;
	keys.push_back("frame");
	keys.push_back("visible_splats");
	keys.push_back("total_splats");
	keys.push_back("cull_ms");
	keys.push_back("sort_ms");
	keys.push_back("raster_ms");
	keys.push_back("composite_ms");
	keys.push_back("stage_total_ms");
	keys.push_back("render_ms");
	keys.push_back("frame_time_ms");
	keys.push_back("gpu_frame_ms");
	keys.push_back("gpu_binning_ms");
	keys.push_back("gpu_prefix_ms");
	keys.push_back("gpu_raster_ms");
	keys.push_back("gpu_resolve_ms");
	keys.push_back("gpu_timing_frame_serial");
	keys.push_back("gpu_timing_frames_behind");
	keys.push_back("gpu_pass_breakdown_available");
	keys.push_back("data_source");
	keys.push_back("data_source_error");
	keys.push_back("raster_path");
	keys.push_back("render_mode");
	keys.push_back("stage_metrics_valid");
	keys.push_back("stage_cull_status");
	keys.push_back("stage_sort_status");
	keys.push_back("stage_raster_status");
	keys.push_back("stage_composite_status");
	keys.push_back("route_uid");
	keys.push_back("sort_route_uid");
	return keys;
}

static void _merge_dictionary(Dictionary &p_target, const Dictionary &p_source) {
	Array keys = p_source.keys();
	for (int i = 0; i < keys.size(); i++) {
		const Variant &key = keys[i];
		p_target[key] = p_source[key];
	}
}

static bool _is_finite(float p_value) {
	return !(Math::is_nan(p_value) || Math::is_inf(p_value));
}

static Dictionary _build_production_metrics_snapshot(GaussianSplatRenderer &p_renderer,
		const GaussianSplatRenderer::StageMetrics &p_stage_metrics, bool p_stage_valid, float p_frame_time_ms) {
	Dictionary metrics;
	const auto &frame_state = p_renderer.get_frame_state();
	const auto &perf = p_renderer.get_performance_state().metrics;
	const auto &debug_state = p_renderer.get_debug_state();
	uint32_t visible_splats = frame_state.visible_splat_count.load(std::memory_order_acquire);
	const auto &scene_state = p_renderer.get_scene_state();
	uint32_t total_splats = scene_state.gaussian_data.is_valid()
			? scene_state.gaussian_data->get_count()
			: 0;
	float cull_ms = p_stage_valid ? p_stage_metrics.cull.cull_time_ms : perf.culling_time_ms;
	float sort_ms = p_stage_valid ? p_stage_metrics.sort.sort_time_ms : frame_state.sort_time_ms;
	float raster_ms = p_stage_valid ? p_stage_metrics.raster.render_time_ms : frame_state.render_time_ms;
	float composite_ms = p_stage_valid ? p_stage_metrics.composite_time_ms : 0.0f;
	float stage_total_ms = cull_ms + sort_ms + raster_ms;

	metrics["frame"] = static_cast<int64_t>(frame_state.frame_counter);
	metrics["visible_splats"] = static_cast<int64_t>(visible_splats);
	metrics["total_splats"] = static_cast<int64_t>(total_splats);
	metrics["cull_ms"] = cull_ms;
	metrics["sort_ms"] = sort_ms;
	metrics["raster_ms"] = raster_ms;
	metrics["composite_ms"] = composite_ms;
	metrics["stage_total_ms"] = stage_total_ms;
	metrics["render_ms"] = frame_state.render_time_ms;
	metrics["frame_time_ms"] = p_frame_time_ms;
	metrics["gpu_frame_ms"] = perf.gpu_frame_time_ms;
	metrics["gpu_binning_ms"] = perf.gpu_tile_binning_time_ms;
	metrics["gpu_prefix_ms"] = perf.gpu_tile_prefix_time_ms;
	metrics["gpu_raster_ms"] = perf.gpu_tile_raster_time_ms;
	metrics["gpu_resolve_ms"] = perf.gpu_tile_resolve_time_ms;
	metrics["gpu_timing_frame_serial"] = static_cast<int64_t>(perf.gpu_timing_frame_serial);
	metrics["gpu_timing_frames_behind"] = static_cast<int64_t>(perf.gpu_timing_frames_behind);
	metrics["gpu_pass_breakdown_available"] = perf.gpu_tile_binning_time_ms > 0.0f ||
			perf.gpu_tile_prefix_time_ms > 0.0f ||
			perf.gpu_tile_raster_time_ms > 0.0f ||
			perf.gpu_tile_resolve_time_ms > 0.0f;
	metrics["data_source"] = perf.data_source;
	metrics["data_source_error"] = perf.data_source_error;
	String raster_path = perf.raster_path;
	if (p_stage_valid && !p_stage_metrics.raster.raster_path.is_empty()) {
		raster_path = p_stage_metrics.raster.raster_path;
	}
	if (raster_path.is_empty()) {
		raster_path = "unknown";
	}
	metrics["raster_path"] = raster_path;
	const auto &render_config = p_renderer.get_render_config();
	metrics["render_mode"] = static_cast<int64_t>(render_config.render_mode);
	metrics["stage_metrics_valid"] = p_stage_valid;
	metrics["stage_cull_status"] = p_stage_valid ? _stage_status_to_string(p_stage_metrics.cull_result.status) : String("unknown");
	metrics["stage_sort_status"] = p_stage_valid ? _stage_status_to_string(p_stage_metrics.sort_result.status) : String("unknown");
	metrics["stage_raster_status"] = p_stage_valid ? _stage_status_to_string(p_stage_metrics.raster_result.status) : String("unknown");
	metrics["stage_composite_status"] = p_stage_valid ? _stage_status_to_string(p_stage_metrics.composite_result.status) : String("unknown");
	metrics["route_uid"] = debug_state.route_uid;
	metrics["sort_route_uid"] = debug_state.sort_route_uid;

	return metrics;
}

static void _append_telemetry_extras(GaussianSplatRenderer &p_renderer,
		const GaussianSplatRenderer::StageMetrics &p_stage_metrics, bool p_stage_valid,
		float p_frame_time_ms, Dictionary &r_metrics) {
	const auto &frame_state = p_renderer.get_frame_state();
	const auto &perf = p_renderer.get_performance_state().metrics;
	const auto &sorting_state = p_renderer.get_sorting_state();
	const auto &debug_state = p_renderer.get_debug_state();
	const auto &subsystem_state = p_renderer.get_subsystem_state();

	r_metrics["frame_count"] = static_cast<int64_t>(frame_state.frame_counter);
	r_metrics["sorted_splats"] = static_cast<int64_t>(sorting_state.sorted_splat_count);
	r_metrics["render_time_ms"] = frame_state.render_time_ms;
	r_metrics["sort_time_ms"] = frame_state.sort_time_ms;
	r_metrics["frame_time_ms"] = p_frame_time_ms;
	r_metrics["total_frames_rendered"] = static_cast<int64_t>(perf.total_frames_rendered);
	r_metrics["avg_frame_time_ms"] = perf.avg_frame_time_ms;
	r_metrics["avg_frame_to_frame_ms"] = perf.avg_frame_to_frame_ms;
	r_metrics["peak_frame_time_ms"] = perf.peak_frame_time_ms;
	r_metrics["cull_projection_contract_mismatches"] = static_cast<int64_t>(perf.cull_projection_contract_mismatch_count);
	r_metrics["buffer_upload_time_ms"] = perf.buffer_upload_time_ms;
	r_metrics["culling_time_ms"] = perf.culling_time_ms;
	r_metrics["gpu_memory_usage_mb"] = perf.gpu_memory_usage_mb;
	r_metrics["uploaded_splat_count"] = static_cast<int64_t>(perf.uploaded_splat_count);
	r_metrics["rendered_splat_count"] = static_cast<int64_t>(perf.rendered_splat_count);
	r_metrics["using_real_data"] = perf.using_real_data;
	r_metrics["sort_submission_time_ms"] = perf.sort_submission_time_ms;
	r_metrics["sort_wait_time_ms"] = perf.sort_wait_time_ms;
	r_metrics["sort_input_build_time_ms"] = perf.sort_input_build_time_ms;
	r_metrics["instance_sort_sync_fallback_count"] = static_cast<int64_t>(perf.instance_sort_sync_fallback_count);
	r_metrics["tile_sort_sync_fallback_count"] = static_cast<int64_t>(perf.tile_sort_sync_fallback_count);
	r_metrics["sort_sync_fallback_count"] = static_cast<int64_t>(
			perf.instance_sort_sync_fallback_count + perf.tile_sort_sync_fallback_count);
	r_metrics["sort_cached_fallback_count"] = static_cast<int64_t>(perf.sort_cached_fallback_count);
	r_metrics["sort_identity_fallback_count"] = static_cast<int64_t>(perf.sort_identity_fallback_count);
	r_metrics["sort_cull_order_fallback_count"] = static_cast<int64_t>(perf.sort_cull_order_fallback_count);
	r_metrics["sort_total_route_fallback_count"] = static_cast<int64_t>(
			perf.sort_cached_fallback_count + perf.sort_identity_fallback_count + perf.sort_cull_order_fallback_count);
	r_metrics["sort_active_algorithm"] = sorting_state.active_sort_algorithm;
	r_metrics["sort_switch_reason"] = sorting_state.sort_switch_reason;
	r_metrics["sort_override_force_cpu"] = sorting_state.override_force_cpu;
	r_metrics["sort_override_force_algorithm"] = sorting_state.override_force_algorithm;
	r_metrics["sort_override_forced_algorithm"] = sorting_state.override_forced_algorithm;
	r_metrics["async_sort_used"] = perf.async_sort_used;
	r_metrics["async_sort_waited"] = perf.async_sort_waited;
	r_metrics["async_overlap_efficiency"] = perf.async_overlap_efficiency;
	r_metrics["culled_by_frustum"] = perf.culled_frustum_count;
	r_metrics["culled_by_distance"] = perf.culled_distance_count;
	r_metrics["culled_by_screen"] = perf.culled_screen_count;
	r_metrics["culled_by_importance"] = perf.culled_importance_count;
	r_metrics["culling_candidate_count"] = perf.culling_candidate_count;
	r_metrics["visible_after_culling"] = perf.visible_after_culling;
	r_metrics["used_hierarchical_culling"] = perf.used_hierarchical_culling;
	r_metrics["sort_cache_hits"] = static_cast<int64_t>(perf.sort_cache_hits);
	r_metrics["sort_cache_misses"] = static_cast<int64_t>(perf.sort_cache_misses);
	r_metrics["gpu_utilization_percent"] = perf.gpu_utilization;
	r_metrics["gpu_frame_time_ms"] = perf.gpu_frame_time_ms;
	r_metrics["gpu_tile_binning_time_ms"] = perf.gpu_tile_binning_time_ms;
	r_metrics["gpu_tile_raster_time_ms"] = perf.gpu_tile_raster_time_ms;
	r_metrics["gpu_tile_prefix_time_ms"] = perf.gpu_tile_prefix_time_ms;
	r_metrics["gpu_tile_resolve_time_ms"] = perf.gpu_tile_resolve_time_ms;
	r_metrics["gpu_timing_frame_serial"] = static_cast<int64_t>(perf.gpu_timing_frame_serial);
	r_metrics["gpu_timing_frames_behind"] = static_cast<int64_t>(perf.gpu_timing_frames_behind);
	r_metrics["gpu_timeline_inflight_frames"] = static_cast<int64_t>(perf.gpu_timeline_inflight_frames);
	r_metrics["gpu_timeline_completed_frames"] = static_cast<int64_t>(perf.gpu_timeline_completed_frames);
	r_metrics["gpu_timeline_stall_count"] = static_cast<int64_t>(perf.gpu_timeline_stall_count);
	r_metrics["gpu_timeline_stall_ms"] = perf.gpu_timeline_stall_ms;
	r_metrics["gpu_timeline_last_value"] = static_cast<int64_t>(perf.gpu_timeline_last_value);
	r_metrics["tile_assignment_ms"] = debug_state.last_tile_assignment_ms;
	r_metrics["tile_rasterization_ms"] = debug_state.last_tile_rasterization_ms;
	r_metrics["debug_last_render_time_ms"] = debug_state.last_render_time_ms;
	r_metrics["debug_last_sort_time_ms"] = debug_state.last_sort_time_ms;
	r_metrics["stage_metrics_valid"] = p_stage_valid;
	r_metrics["stage_cull_has_visible"] = p_stage_metrics.cull.has_visible;
	r_metrics["stage_cull_visible_count"] = p_stage_metrics.cull.visible_count;
	r_metrics["stage_cull_candidate_count"] = p_stage_metrics.cull.candidate_count;
	r_metrics["stage_cull_time_ms"] = p_stage_metrics.cull.cull_time_ms;
	r_metrics["stage_cull_visible_domain"] = GaussianRenderState::index_domain_to_string(p_stage_metrics.cull.visible_domain);
	r_metrics["stage_sort_executed"] = p_stage_metrics.sort.did_sort;
	r_metrics["stage_sort_input_count"] = p_stage_metrics.sort.input_count;
	r_metrics["stage_sort_sorted_count"] = p_stage_metrics.sort.sorted_count;
	r_metrics["stage_sort_time_ms"] = p_stage_metrics.sort.sort_time_ms;
	r_metrics["stage_sort_input_domain"] = GaussianRenderState::index_domain_to_string(p_stage_metrics.sort.input_domain);
	r_metrics["stage_sort_output_domain"] = GaussianRenderState::index_domain_to_string(p_stage_metrics.sort.output_domain);
	r_metrics["stage_raster_time_ms"] = p_stage_metrics.raster.render_time_ms;
	r_metrics["stage_raster_cached"] = p_stage_metrics.raster.reused_cached_render;
	r_metrics["stage_raster_painterly"] = p_stage_metrics.raster.painterly_active;
	r_metrics["stage_composite_executed"] = p_stage_metrics.composite_executed;
	r_metrics["stage_composite_time_ms"] = p_stage_metrics.composite_time_ms;
	r_metrics["stage_cull_status"] = p_stage_valid ? _stage_status_to_string(p_stage_metrics.cull_result.status) : String("unknown");
	r_metrics["stage_cull_reason"] = p_stage_metrics.cull_result.reason;
	r_metrics["stage_cull_is_error"] = p_stage_metrics.cull_result.is_error;
	r_metrics["stage_sort_status"] = p_stage_valid ? _stage_status_to_string(p_stage_metrics.sort_result.status) : String("unknown");
	r_metrics["stage_sort_reason"] = p_stage_metrics.sort_result.reason;
	r_metrics["stage_sort_is_error"] = p_stage_metrics.sort_result.is_error;
	r_metrics["stage_raster_status"] = p_stage_valid ? _stage_status_to_string(p_stage_metrics.raster_result.status) : String("unknown");
	r_metrics["stage_raster_reason"] = p_stage_metrics.raster_result.reason;
	r_metrics["stage_raster_is_error"] = p_stage_metrics.raster_result.is_error;
	r_metrics["stage_composite_status"] = p_stage_valid ? _stage_status_to_string(p_stage_metrics.composite_result.status) : String("unknown");
	r_metrics["stage_composite_reason"] = p_stage_metrics.composite_result.reason;
	r_metrics["stage_composite_is_error"] = p_stage_metrics.composite_result.is_error;
	r_metrics["route_uid"] = debug_state.route_uid;
	r_metrics["sort_route_uid"] = debug_state.sort_route_uid;

	if (!perf.streaming_state.is_empty()) {
		const Dictionary streaming_state = perf.streaming_state;
		r_metrics["streaming_state"] = streaming_state;
		Dictionary streaming_diagnostics;
		if (streaming_state.has("diagnostics")) {
			streaming_diagnostics = streaming_state["diagnostics"];
			r_metrics["streaming_diagnostics"] = streaming_diagnostics;
		}
		r_metrics["streaming_diagnostics_category"] = streaming_state.get("diagnostics_category", String("ok"));
		r_metrics["streaming_diagnostics_fingerprint"] = streaming_state.get("diagnostics_fingerprint", String("ok"));
		r_metrics["streaming_diagnostics_has_failure"] = streaming_state.get("diagnostics_has_failure", false);
		r_metrics["streaming_cap_tier_preset"] = streaming_state.get("cap_tier_preset", String("custom"));
		r_metrics["streaming_cap_tier_active"] = streaming_state.get("cap_tier_active", false);
		r_metrics["streaming_effective_upload_cap_mb_per_frame"] = streaming_state.get("effective_upload_cap_mb_per_frame", int64_t(0));
		r_metrics["streaming_effective_upload_cap_mb_per_slice"] = streaming_state.get("effective_upload_cap_mb_per_slice", int64_t(0));
		r_metrics["streaming_effective_upload_cap_mb_per_second"] = streaming_state.get("effective_upload_cap_mb_per_second", int64_t(0));
		r_metrics["streaming_effective_vram_budget_mb"] = streaming_state.get("effective_vram_budget_mb", int64_t(0));
		r_metrics["streaming_effective_vram_min_chunks"] = streaming_state.get("effective_vram_min_chunks", int64_t(0));
		r_metrics["streaming_effective_vram_max_chunks"] = streaming_state.get("effective_vram_max_chunks", int64_t(0));
		r_metrics["streaming_cap_source_upload_mb_per_frame"] = streaming_state.get("cap_source_upload_mb_per_frame", String("project_default"));
		r_metrics["streaming_cap_source_upload_mb_per_slice"] = streaming_state.get("cap_source_upload_mb_per_slice", String("project_default"));
		r_metrics["streaming_cap_source_upload_mb_per_second"] = streaming_state.get("cap_source_upload_mb_per_second", String("project_default"));
		r_metrics["streaming_cap_source_vram_budget_mb"] = streaming_state.get("cap_source_vram_budget_mb", String("project_default"));
		r_metrics["streaming_cap_source_vram_min_chunks"] = streaming_state.get("cap_source_vram_min_chunks", String("project_default"));
		r_metrics["streaming_cap_source_vram_max_chunks"] = streaming_state.get("cap_source_vram_max_chunks", String("project_default"));
		r_metrics["streaming_upload_frame_cap_hit"] = streaming_state.get("upload_frame_cap_hit", false);
		r_metrics["streaming_upload_slice_cap_hit"] = streaming_state.get("upload_slice_cap_hit", false);
		r_metrics["streaming_upload_bandwidth_cap_hit"] = streaming_state.get("upload_bandwidth_cap_hit", false);
		r_metrics["streaming_chunk_load_cap_hit"] = streaming_state.get("chunk_load_cap_hit", false);
		r_metrics["streaming_vram_chunk_cap_hit"] = streaming_state.get("vram_chunk_cap_hit", false);
		r_metrics["streaming_queue_pressure_active"] = streaming_state.get("queue_pressure_active", false);
		r_metrics["streaming_queue_pressure_frames"] = streaming_diagnostics.get("queue_pressure_frames", int64_t(0));
		r_metrics["streaming_vram_cap_hit_frames"] = streaming_diagnostics.get("vram_cap_hit_frames", int64_t(0));
	} else {
		r_metrics["streaming_diagnostics_category"] = String("unknown");
		r_metrics["streaming_diagnostics_fingerprint"] = String("unavailable");
		r_metrics["streaming_diagnostics_has_failure"] = false;
		r_metrics["streaming_cap_tier_preset"] = String("custom");
		r_metrics["streaming_cap_tier_active"] = false;
		r_metrics["streaming_effective_upload_cap_mb_per_frame"] = static_cast<int64_t>(0);
		r_metrics["streaming_effective_upload_cap_mb_per_slice"] = static_cast<int64_t>(0);
		r_metrics["streaming_effective_upload_cap_mb_per_second"] = static_cast<int64_t>(0);
		r_metrics["streaming_effective_vram_budget_mb"] = static_cast<int64_t>(0);
		r_metrics["streaming_effective_vram_min_chunks"] = static_cast<int64_t>(0);
		r_metrics["streaming_effective_vram_max_chunks"] = static_cast<int64_t>(0);
		r_metrics["streaming_cap_source_upload_mb_per_frame"] = String("project_default");
		r_metrics["streaming_cap_source_upload_mb_per_slice"] = String("project_default");
		r_metrics["streaming_cap_source_upload_mb_per_second"] = String("project_default");
		r_metrics["streaming_cap_source_vram_budget_mb"] = String("project_default");
		r_metrics["streaming_cap_source_vram_min_chunks"] = String("project_default");
		r_metrics["streaming_cap_source_vram_max_chunks"] = String("project_default");
		r_metrics["streaming_upload_frame_cap_hit"] = false;
		r_metrics["streaming_upload_slice_cap_hit"] = false;
		r_metrics["streaming_upload_bandwidth_cap_hit"] = false;
		r_metrics["streaming_chunk_load_cap_hit"] = false;
		r_metrics["streaming_vram_chunk_cap_hit"] = false;
		r_metrics["streaming_queue_pressure_active"] = false;
		r_metrics["streaming_queue_pressure_frames"] = static_cast<int64_t>(0);
		r_metrics["streaming_vram_cap_hit_frames"] = static_cast<int64_t>(0);
	}

	if (subsystem_state.rasterizer.is_valid()) {
		Vector2i tile_grid = subsystem_state.rasterizer->get_tile_grid_size();
		r_metrics["tile_grid_size"] = tile_grid;
		r_metrics["tile_size"] = subsystem_state.rasterizer->get_tile_size();
		RasterStats raster_stats = subsystem_state.rasterizer->get_render_stats();
		r_metrics["overlap_records"] = static_cast<int64_t>(raster_stats.overlap_records);
		r_metrics["overlap_record_budget"] = static_cast<int64_t>(raster_stats.overlap_record_budget);
		r_metrics["overlap_record_budget_effective"] = static_cast<int64_t>(raster_stats.overlap_record_budget_effective);
		r_metrics["overlap_record_budget_configured"] = static_cast<int64_t>(raster_stats.overlap_record_budget_configured);
		r_metrics["overlap_thinning_keep_ratio"] = raster_stats.overlap_thinning_keep_ratio;
		r_metrics["sorted_indices_blend_fallback_active"] = raster_stats.sorted_indices_blend_fallback_active;
		r_metrics["sorted_indices_blend_fallback_reason"] = raster_stats.sorted_indices_blend_fallback_reason;
	} else {
		r_metrics["tile_grid_size"] = Vector2i(0, 0);
		r_metrics["tile_size"] = 0;
		r_metrics["overlap_records"] = static_cast<int64_t>(0);
		r_metrics["overlap_record_budget"] = static_cast<int64_t>(0);
		r_metrics["overlap_record_budget_effective"] = static_cast<int64_t>(0);
		r_metrics["overlap_record_budget_configured"] = static_cast<int64_t>(0);
		r_metrics["overlap_thinning_keep_ratio"] = 1.0f;
		r_metrics["sorted_indices_blend_fallback_active"] = false;
		r_metrics["sorted_indices_blend_fallback_reason"] = String();
	}
}

static Dictionary _validate_production_metrics(const Dictionary &p_metrics) {
	Array issues;
	Array contract = _production_metrics_contract();
	for (int i = 0; i < contract.size(); i++) {
		const String key = contract[i];
		if (!p_metrics.has(key)) {
			issues.push_back(vformat("missing:%s", key));
		}
	}

	const int64_t visible_splats = p_metrics.get("visible_splats", int64_t(-1));
	const int64_t total_splats = p_metrics.get("total_splats", int64_t(-1));
	if (visible_splats < 0) {
		issues.push_back("visible_splats_negative");
	}
	if (total_splats < 0) {
		issues.push_back("total_splats_negative");
	}
	// NOTE: visible_splats CAN exceed total_splats with instance pipeline because:
	// - visible_splats is aggregated across all instances
	// - total_splats is from a single GaussianData asset
	// - overlap rendering duplicates splats across tile boundaries
	// This is expected behavior, not a contract violation.

	const float cull_ms = static_cast<float>(p_metrics.get("cull_ms", -1.0f));
	const float sort_ms = static_cast<float>(p_metrics.get("sort_ms", -1.0f));
	const float raster_ms = static_cast<float>(p_metrics.get("raster_ms", -1.0f));
	const float composite_ms = static_cast<float>(p_metrics.get("composite_ms", -1.0f));
	const float stage_total_ms = static_cast<float>(p_metrics.get("stage_total_ms", -1.0f));
	const float render_ms = static_cast<float>(p_metrics.get("render_ms", -1.0f));
	const float frame_time_ms = static_cast<float>(p_metrics.get("frame_time_ms", -1.0f));
	const float gpu_frame_ms = static_cast<float>(p_metrics.get("gpu_frame_ms", -1.0f));
	const float gpu_binning_ms = static_cast<float>(p_metrics.get("gpu_binning_ms", -1.0f));
	const float gpu_prefix_ms = static_cast<float>(p_metrics.get("gpu_prefix_ms", -1.0f));
	const float gpu_raster_ms = static_cast<float>(p_metrics.get("gpu_raster_ms", -1.0f));
	const float gpu_resolve_ms = static_cast<float>(p_metrics.get("gpu_resolve_ms", -1.0f));
	const int64_t gpu_timing_frame_serial = p_metrics.get("gpu_timing_frame_serial", int64_t(-1));
	const int64_t gpu_timing_frames_behind = p_metrics.get("gpu_timing_frames_behind", int64_t(-1));

	if (cull_ms < 0.0f || !_is_finite(cull_ms)) {
		issues.push_back("cull_ms_invalid");
	}
	if (sort_ms < 0.0f || !_is_finite(sort_ms)) {
		issues.push_back("sort_ms_invalid");
	}
	if (raster_ms < 0.0f || !_is_finite(raster_ms)) {
		issues.push_back("raster_ms_invalid");
	}
	if (composite_ms < 0.0f || !_is_finite(composite_ms)) {
		issues.push_back("composite_ms_invalid");
	}
	if (stage_total_ms < 0.0f || !_is_finite(stage_total_ms)) {
		issues.push_back("stage_total_ms_invalid");
	}
	if (render_ms < 0.0f || !_is_finite(render_ms)) {
		issues.push_back("render_ms_invalid");
	}
	if (frame_time_ms < 0.0f || !_is_finite(frame_time_ms)) {
		issues.push_back("frame_time_ms_invalid");
	}
	if (gpu_frame_ms < 0.0f || !_is_finite(gpu_frame_ms)) {
		issues.push_back("gpu_frame_ms_invalid");
	}
	if (gpu_binning_ms < 0.0f || !_is_finite(gpu_binning_ms)) {
		issues.push_back("gpu_binning_ms_invalid");
	}
	if (gpu_prefix_ms < 0.0f || !_is_finite(gpu_prefix_ms)) {
		issues.push_back("gpu_prefix_ms_invalid");
	}
	if (gpu_raster_ms < 0.0f || !_is_finite(gpu_raster_ms)) {
		issues.push_back("gpu_raster_ms_invalid");
	}
	if (gpu_resolve_ms < 0.0f || !_is_finite(gpu_resolve_ms)) {
		issues.push_back("gpu_resolve_ms_invalid");
	}
	if (gpu_timing_frame_serial < 0) {
		issues.push_back("gpu_timing_frame_serial_invalid");
	}
	if (gpu_timing_frames_behind < 0) {
		issues.push_back("gpu_timing_frames_behind_invalid");
	}

	const bool stage_valid = bool(p_metrics.get("stage_metrics_valid", false));
	if (!stage_valid) {
		issues.push_back("stage_metrics_invalid");
	}
	const String route_uid = p_metrics.get("route_uid", String());
	const String sort_route_uid = p_metrics.get("sort_route_uid", String());
	const bool route_no_device = route_uid == String(RenderRouteUID::COMMON_FAIL_NO_DEVICE);
	if (stage_valid && route_uid.is_empty()) {
		issues.push_back("route_uid_empty");
	}
	// No-device fallback can legitimately skip sort-route assignment.
	if (stage_valid && !route_no_device && sort_route_uid.is_empty()) {
		issues.push_back("sort_route_uid_empty");
	}
	const String stage_cull_status = p_metrics.get("stage_cull_status", String("unknown"));
	const String stage_sort_status = p_metrics.get("stage_sort_status", String("unknown"));
	const String stage_raster_status = p_metrics.get("stage_raster_status", String("unknown"));
	if (!route_uid.is_empty() && !sort_route_uid.is_empty()) {
		const bool sort_route_no_device = sort_route_uid == String(RenderRouteUID::COMMON_FAIL_NO_DEVICE);
		if (route_no_device != sort_route_no_device) {
			issues.push_back("route_sort_route_device_mismatch");
		}
		if (stage_sort_status == "success" && sort_route_uid.begins_with("COMMON.FAIL.")) {
			issues.push_back("sort_route_uid_status_mismatch");
		}
	}
	const bool has_meaningful_workload = MAX(visible_splats, total_splats) >= 1024;
	if (stage_valid && has_meaningful_workload && total_splats > 0 && stage_cull_status == "success" && cull_ms <= 0.0f) {
		issues.push_back("cull_ms_placeholder");
	}
	if (stage_valid && has_meaningful_workload && visible_splats > 0 && stage_sort_status == "success" && sort_ms <= 0.0f) {
		issues.push_back("sort_ms_placeholder");
	}
	if (stage_valid && has_meaningful_workload && visible_splats > 0 && stage_raster_status == "success" && raster_ms <= 0.0f) {
		issues.push_back("raster_ms_placeholder");
	}
	const bool has_gpu_breakdown = gpu_binning_ms > 0.0f ||
			gpu_prefix_ms > 0.0f ||
			gpu_raster_ms > 0.0f ||
			gpu_resolve_ms > 0.0f;
	if (stage_valid && has_meaningful_workload && visible_splats > 0 &&
			stage_raster_status == "success" &&
			gpu_timing_frame_serial > 0 &&
			gpu_frame_ms > 0.0f &&
			!has_gpu_breakdown) {
		issues.push_back("gpu_pass_timing_placeholder");
	}

	const String data_source = p_metrics.get("data_source", String());
	if (data_source.is_empty()) {
		issues.push_back("data_source_empty");
	}
	const String raster_path = p_metrics.get("raster_path", String());
	if (raster_path.is_empty()) {
		issues.push_back("raster_path_empty");
	}

	String issues_text;
	for (int i = 0; i < issues.size(); i++) {
		if (i > 0) {
			issues_text += ", ";
		}
		const String issue = issues[i];
		issues_text += issue;
	}
	if (!issues.is_empty()) {
		GS_LOG_WARN_DEFAULT(vformat("[Diagnostics] Production metrics contract violated: %s", issues_text));
#ifdef DEV_ENABLED
		WARN_PRINT(vformat("Production metrics contract violated: %s", issues_text));
#endif
	}

	Dictionary validation;
	validation["valid"] = issues.is_empty();
	validation["issues"] = issues;
	validation["frame"] = p_metrics.get("frame", int64_t(0));
	return validation;
}

static Dictionary _evaluate_perf_gate(const ProductionMetricsConfig &p_config, const Dictionary &p_metrics) {
	Dictionary result;
	result["enabled"] = p_config.perf_gate_enabled;
	result["applicable"] = false;
	result["passed"] = true;
	result["budget_ms"] = p_config.perf_gate_budget_ms;
	result["splat_threshold"] = static_cast<int64_t>(p_config.perf_gate_splat_threshold);

	if (!p_config.perf_gate_enabled) {
		result["reason"] = "disabled";
		return result;
	}

	const bool stage_valid = bool(p_metrics.get("stage_metrics_valid", false));
	const int64_t visible_splats = p_metrics.get("visible_splats", int64_t(0));
	const float stage_total_ms = static_cast<float>(p_metrics.get("stage_total_ms", 0.0f));
	result["visible_splats"] = visible_splats;
	result["stage_total_ms"] = stage_total_ms;

	if (!stage_valid) {
		result["reason"] = "stage_metrics_invalid";
		return result;
	}
	if (visible_splats < static_cast<int64_t>(p_config.perf_gate_splat_threshold)) {
		result["reason"] = "below_splat_threshold";
		return result;
	}

	result["applicable"] = true;
	const bool passed = stage_total_ms <= p_config.perf_gate_budget_ms;
	result["passed"] = passed;
	result["reason"] = passed ? "within_budget" : "budget_exceeded";
	result["over_budget_ms"] = passed ? 0.0f : (stage_total_ms - p_config.perf_gate_budget_ms);
	return result;
}

static void _update_production_summary(GaussianSplatRenderer::DiagnosticsState &p_state,
		const ProductionMetricsConfig &p_config,
		const Dictionary &p_metrics,
		const Dictionary &p_perf_gate_result,
		uint64_t p_frame_end_usec) {
	const uint64_t frame = static_cast<uint64_t>(p_metrics.get("frame", int64_t(0)));
	if (p_state.production_metrics_window_frames == 0) {
		p_state.production_metrics_window_start_frame = frame;
		p_state.production_metrics_window_start_usec = p_frame_end_usec;
	}

	p_state.production_metrics_window_frames++;

	const float render_ms = static_cast<float>(p_metrics.get("render_ms", 0.0f));
	const float frame_time_ms = static_cast<float>(p_metrics.get("frame_time_ms", 0.0f));
	const float cull_ms = static_cast<float>(p_metrics.get("cull_ms", 0.0f));
	const float sort_ms = static_cast<float>(p_metrics.get("sort_ms", 0.0f));
	const float raster_ms = static_cast<float>(p_metrics.get("raster_ms", 0.0f));
	const float composite_ms = static_cast<float>(p_metrics.get("composite_ms", 0.0f));
	const float stage_total_ms = static_cast<float>(p_metrics.get("stage_total_ms", 0.0f));
	const uint32_t visible_splats = static_cast<uint32_t>(static_cast<int64_t>(p_metrics.get("visible_splats", int64_t(0))));

	p_state.production_metrics_frame_ms_sum += static_cast<double>(frame_time_ms > 0.0f ? frame_time_ms : render_ms);
	p_state.production_metrics_cull_ms_sum += static_cast<double>(cull_ms);
	p_state.production_metrics_sort_ms_sum += static_cast<double>(sort_ms);
	p_state.production_metrics_raster_ms_sum += static_cast<double>(raster_ms);
	p_state.production_metrics_composite_ms_sum += static_cast<double>(composite_ms);
	p_state.production_metrics_stage_total_ms_sum += static_cast<double>(stage_total_ms);
	p_state.production_metrics_frame_ms_peak = MAX(p_state.production_metrics_frame_ms_peak,
			static_cast<double>(frame_time_ms > 0.0f ? frame_time_ms : render_ms));
	p_state.production_metrics_stage_ms_peak = MAX(p_state.production_metrics_stage_ms_peak, static_cast<double>(stage_total_ms));
	p_state.production_metrics_visible_peak = MAX(p_state.production_metrics_visible_peak, visible_splats);
	p_state.production_metrics_visible_sum += visible_splats;

	if (bool(p_perf_gate_result.get("enabled", false)) && bool(p_perf_gate_result.get("applicable", false))) {
		p_state.production_metrics_perf_gate_checks++;
		if (!bool(p_perf_gate_result.get("passed", true))) {
			p_state.production_metrics_perf_gate_failures++;
		}
	}

	if (p_state.production_metrics_window_frames < p_config.summary_interval_frames) {
		return;
	}

	const double frame_count = static_cast<double>(p_state.production_metrics_window_frames);
	Dictionary summary;
	summary["start_frame"] = static_cast<int64_t>(p_state.production_metrics_window_start_frame);
	summary["end_frame"] = static_cast<int64_t>(frame);
	summary["frame_count"] = static_cast<int64_t>(p_state.production_metrics_window_frames);
	summary["window_start_usec"] = static_cast<int64_t>(p_state.production_metrics_window_start_usec);
	summary["window_end_usec"] = static_cast<int64_t>(p_frame_end_usec);
	summary["avg_frame_ms"] = frame_count > 0.0 ? p_state.production_metrics_frame_ms_sum / frame_count : 0.0;
	summary["avg_cull_ms"] = frame_count > 0.0 ? p_state.production_metrics_cull_ms_sum / frame_count : 0.0;
	summary["avg_sort_ms"] = frame_count > 0.0 ? p_state.production_metrics_sort_ms_sum / frame_count : 0.0;
	summary["avg_raster_ms"] = frame_count > 0.0 ? p_state.production_metrics_raster_ms_sum / frame_count : 0.0;
	summary["avg_composite_ms"] = frame_count > 0.0 ? p_state.production_metrics_composite_ms_sum / frame_count : 0.0;
	summary["avg_stage_total_ms"] = frame_count > 0.0 ? p_state.production_metrics_stage_total_ms_sum / frame_count : 0.0;
	summary["peak_frame_ms"] = p_state.production_metrics_frame_ms_peak;
	summary["peak_stage_total_ms"] = p_state.production_metrics_stage_ms_peak;
	summary["max_visible_splats"] = static_cast<int64_t>(p_state.production_metrics_visible_peak);
	summary["avg_visible_splats"] = frame_count > 0.0
			? static_cast<double>(p_state.production_metrics_visible_sum) / frame_count
			: 0.0;
	summary["perf_gate_checks"] = static_cast<int64_t>(p_state.production_metrics_perf_gate_checks);
	summary["perf_gate_failures"] = static_cast<int64_t>(p_state.production_metrics_perf_gate_failures);

	p_state.production_metrics_summaries.push_back(summary);
	while (p_state.production_metrics_summaries.size() > p_config.summary_history_size) {
		p_state.production_metrics_summaries.remove_at(0);
	}

	p_state.production_metrics_window_start_frame = 0;
	p_state.production_metrics_window_start_usec = 0;
	p_state.production_metrics_window_frames = 0;
	p_state.production_metrics_frame_ms_sum = 0.0;
	p_state.production_metrics_cull_ms_sum = 0.0;
	p_state.production_metrics_sort_ms_sum = 0.0;
	p_state.production_metrics_raster_ms_sum = 0.0;
	p_state.production_metrics_composite_ms_sum = 0.0;
	p_state.production_metrics_stage_total_ms_sum = 0.0;
	p_state.production_metrics_frame_ms_peak = 0.0;
	p_state.production_metrics_stage_ms_peak = 0.0;
	p_state.production_metrics_visible_peak = 0;
	p_state.production_metrics_visible_sum = 0;
	p_state.production_metrics_perf_gate_checks = 0;
	p_state.production_metrics_perf_gate_failures = 0;
}
} // namespace

RenderDiagnosticsOrchestrator::RenderDiagnosticsOrchestrator(const Dependencies &p_dependencies) :
		renderer(p_dependencies.renderer),
		debug_state_orchestrator(p_dependencies.debug_state_orchestrator),
		build_device_capability_report(p_dependencies.build_device_capability_report),
		runtime_ports(p_dependencies.runtime_ports) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(debug_state_orchestrator);
	ERR_FAIL_COND_MSG(!build_device_capability_report, "RenderDiagnosticsOrchestrator requires device capability callback.");
	ERR_FAIL_COND_MSG(!runtime_ports.update_gpu_pass_metrics_from_tile_renderer,
			"RenderDiagnosticsOrchestrator requires GPU pass metric refresh callback.");
}

void RenderDiagnosticsOrchestrator::record_rendering_error(const RenderingError &p_error) {
	diagnostics_state.runtime_error_statistics.total_errors++;
	if (p_error.get_severity() == RenderingError::Severity::WARNING) {
		diagnostics_state.runtime_error_statistics.total_warnings++;
	}
	diagnostics_state.runtime_error_statistics.last_error = p_error;
	diagnostics_state.runtime_error_statistics.last_error_time_usec = OS::get_singleton()->get_ticks_usec();
	diagnostics_state.runtime_error_statistics.last_error_context = p_error.get_context();
	diagnostics_state.runtime_error_statistics.error_code_counts[p_error.get_code().id]++;
	diagnostics_state.runtime_error_statistics.recent_errors.push_back(p_error);
	if (diagnostics_state.runtime_error_statistics.recent_errors.size() > 16) {
		diagnostics_state.runtime_error_statistics.recent_errors.remove_at(0);
	}
	diagnostics_state.runtime_diagnostics_requested = true;
	GaussianSplatRenderer::ErrorRecoveryStateMachine::State next_state =
			diagnostics_state.recovery_state_machine.state == GaussianSplatRenderer::ErrorRecoveryStateMachine::State::DISABLED
			? GaussianSplatRenderer::ErrorRecoveryStateMachine::State::DISABLED
			: GaussianSplatRenderer::ErrorRecoveryStateMachine::State::DIAGNOSTIC;
	transition_recovery_state(next_state, p_error.get_message());
	GaussianRenderingDiagnostics::ensure_singleton();
	if (GaussianRenderingDiagnostics::get_singleton()) {
		GaussianRenderingDiagnostics::get_singleton()->notify_error(renderer, p_error);
	}
}

void RenderDiagnosticsOrchestrator::transition_recovery_state(GaussianSplatRenderer::ErrorRecoveryStateMachine::State p_state,
		const String &p_reason) {
	if (diagnostics_state.recovery_state_machine.state == p_state &&
			diagnostics_state.recovery_state_machine.reason == p_reason) {
		return;
	}
	diagnostics_state.recovery_state_machine.state = p_state;
	diagnostics_state.recovery_state_machine.reason = p_reason;
	diagnostics_state.recovery_state_machine.last_transition_frame = renderer->get_frame_state().frame_counter;
	diagnostics_state.recovery_state_machine.last_transition_time_usec = OS::get_singleton()->get_ticks_usec();
}

void RenderDiagnosticsOrchestrator::record_cross_device_operation(
		const GaussianSplatRenderer::CrossDeviceOperation &p_operation) {
	diagnostics_state.cross_device_operations.push_back(p_operation);
	const int MAX_OPS = 64;
	while (diagnostics_state.cross_device_operations.size() > MAX_OPS) {
		diagnostics_state.cross_device_operations.remove_at(0);
	}
}

void RenderDiagnosticsOrchestrator::capture_frame_timing_sample() {
	GaussianSplatRenderer::FrameTimingSample sample;
	sample.timestamp_usec = OS::get_singleton()->get_ticks_usec();
	sample.frame = renderer->get_frame_state().frame_counter;
	sample.render_ms = renderer->get_frame_state().render_time_ms;
	sample.sort_ms = renderer->get_frame_state().sort_time_ms;
	sample.total_ms = renderer->get_frame_state().render_time_ms + renderer->get_frame_state().sort_time_ms;
	sample.visible_splats = renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
	sample.used_gpu = renderer->get_sorting_state().gpu_sorter.is_valid();
	diagnostics_state.frame_timing_history.push_back(sample);
	const int MAX_SAMPLES = 240;
	while (diagnostics_state.frame_timing_history.size() > MAX_SAMPLES) {
		diagnostics_state.frame_timing_history.remove_at(0);
	}
}

void RenderDiagnosticsOrchestrator::increment_frame_counter() {
	capture_frame_timing_sample();
	renderer->get_frame_state().frame_counter++;
	GaussianRenderingDiagnostics::ensure_singleton();
	if (GaussianRenderingDiagnostics::get_singleton()) {
		GaussianRenderingDiagnostics::get_singleton()->notify_frame_completed(renderer);
	}
	emit_runtime_diagnostics_if_requested();
}

void RenderDiagnosticsOrchestrator::emit_runtime_diagnostics_if_requested() {
	if (!diagnostics_state.runtime_diagnostics_requested) {
		return;
	}
	GaussianRenderingDiagnostics::ensure_singleton();
	if (GaussianRenderingDiagnostics::get_singleton()) {
		GaussianRenderingDiagnostics::get_singleton()->request_runtime_report();
	}
	diagnostics_state.runtime_diagnostics_requested = false;
}

Array RenderDiagnosticsOrchestrator::serialize_texture_trace() const {
	Array result;
	for (const GaussianSplatRenderer::TextureTraceEntry &entry : diagnostics_state.texture_allocation_trace) {
		Dictionary dict;
		dict["timestamp_usec"] = static_cast<int64_t>(entry.timestamp_usec);
		dict["action"] = entry.action;
		dict["texture_rid"] = static_cast<int64_t>(entry.texture_rid);
		dict["device_instance_id"] = static_cast<int64_t>(entry.device_instance_id);
		dict["format"] = entry.format_label;
		dict["width"] = entry.extent.x;
		dict["height"] = entry.extent.y;
		result.push_back(dict);
	}
	return result;
}

Array RenderDiagnosticsOrchestrator::serialize_cross_device_operations() const {
	Array result;
	for (const GaussianSplatRenderer::CrossDeviceOperation &op : diagnostics_state.cross_device_operations) {
		Dictionary dict;
		dict["timestamp_usec"] = static_cast<int64_t>(op.timestamp_usec);
		dict["context"] = op.context;
		dict["source_device"] = static_cast<int64_t>(op.source_device);
		dict["target_device"] = static_cast<int64_t>(op.target_device);
		result.push_back(dict);
	}
	return result;
}

Array RenderDiagnosticsOrchestrator::serialize_frame_timing() const {
	Array result;
	for (const GaussianSplatRenderer::FrameTimingSample &sample : diagnostics_state.frame_timing_history) {
		Dictionary dict;
		dict["timestamp_usec"] = static_cast<int64_t>(sample.timestamp_usec);
		dict["frame"] = static_cast<int64_t>(sample.frame);
		dict["render_ms"] = sample.render_ms;
		dict["sort_ms"] = sample.sort_ms;
		dict["total_ms"] = sample.total_ms;
		dict["visible_splats"] = static_cast<int64_t>(sample.visible_splats);
		dict["used_gpu"] = sample.used_gpu;
		result.push_back(dict);
	}
	return result;
}

Dictionary RenderDiagnosticsOrchestrator::serialize_error_statistics() const {
	Dictionary dict;
	dict["total_errors"] = static_cast<int64_t>(diagnostics_state.runtime_error_statistics.total_errors);
	dict["total_warnings"] = static_cast<int64_t>(diagnostics_state.runtime_error_statistics.total_warnings);
	dict["total_recoveries"] = static_cast<int64_t>(diagnostics_state.runtime_error_statistics.total_recoveries);
	dict["last_error_time_usec"] = static_cast<int64_t>(diagnostics_state.runtime_error_statistics.last_error_time_usec);
	dict["last_recovery_time_usec"] = static_cast<int64_t>(diagnostics_state.runtime_error_statistics.last_recovery_time_usec);
	dict["last_error"] = diagnostics_state.runtime_error_statistics.last_error.to_dictionary();
	dict["last_error_context"] = diagnostics_state.runtime_error_statistics.last_error_context;

	Dictionary error_counts;
	for (const KeyValue<int, uint64_t> &kv : diagnostics_state.runtime_error_statistics.error_code_counts) {
		error_counts[String::num_int64(kv.key)] = static_cast<int64_t>(kv.value);
	}
	dict["error_code_counts"] = error_counts;

	Dictionary recovery_counts;
	for (const KeyValue<int, uint64_t> &kv : diagnostics_state.runtime_error_statistics.recovery_code_counts) {
		recovery_counts[String::num_int64(kv.key)] = static_cast<int64_t>(kv.value);
	}
	dict["recovery_code_counts"] = recovery_counts;

	Array history;
	for (const RenderingError &error : diagnostics_state.runtime_error_statistics.recent_errors) {
		history.push_back(error.to_dictionary());
	}
	dict["recent_errors"] = history;
	dict["recovery_state"] = static_cast<int64_t>(diagnostics_state.recovery_state_machine.state);
	dict["recovery_reason"] = diagnostics_state.recovery_state_machine.reason;
	dict["recovery_transition_frame"] = static_cast<int64_t>(diagnostics_state.recovery_state_machine.last_transition_frame);
	dict["recovery_transition_time_usec"] = static_cast<int64_t>(diagnostics_state.recovery_state_machine.last_transition_time_usec);
	return dict;
}

Dictionary RenderDiagnosticsOrchestrator::build_render_stats() const {
	Dictionary stats;
	GaussianSplatRenderer *mutable_renderer = const_cast<GaussianSplatRenderer *>(renderer);
	const Ref<DebugOverlaySystem> &overlay_system_ref = mutable_renderer->get_subsystem_state().debug_overlay_system;
	DebugOverlaySystem *overlay_system = overlay_system_ref.is_valid() ? overlay_system_ref.ptr() : nullptr;
	const DebugOverlayQueryView overlay_query = overlay_system
			? overlay_system->build_query_view(mutable_renderer)
			: DebugOverlayQueryView();
	const DebugOverlayOptions overlay_options = overlay_query.get_options();
	const DebugOverlayCommandSink overlay_command = overlay_system
			? overlay_system->build_command_sink(mutable_renderer)
			: DebugOverlayCommandSink();
	if (overlay_system && mutable_renderer->get_debug_state().overlay_dirty) {
		overlay_system->rebuild_renderer_overlay_statistics_from_cache(overlay_query, overlay_command);
	}
	if (overlay_system && mutable_renderer->get_debug_state().hud_dirty) {
		overlay_system->rebuild_renderer_performance_hud_lines(overlay_query, overlay_command);
	}
	if (!diagnostics_state.last_telemetry_snapshot.is_empty()) {
		_merge_dictionary(stats, diagnostics_state.last_telemetry_snapshot);
	} else {
		stats["visible_splats"] = renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
		stats["total_splats"] = renderer->get_scene_state().gaussian_data.is_valid() ? renderer->get_scene_state().gaussian_data->get_count() : 0;
		stats["sort_time_ms"] = renderer->get_frame_state().sort_time_ms;
		stats["render_time_ms"] = renderer->get_frame_state().render_time_ms;
		stats["frame_count"] = renderer->get_frame_state().frame_counter;
		stats["render_mode"] = renderer->get_render_config().render_mode;
		const bool stage_metrics_valid = mutable_renderer->get_debug_state().last_stage_metrics_valid;
		GaussianSplatRenderer::StageMetrics stage_metrics = stage_metrics_valid
				? mutable_renderer->get_debug_state().last_stage_metrics
				: GaussianSplatRenderer::StageMetrics();
		_append_telemetry_extras(*mutable_renderer, stage_metrics, stage_metrics_valid,
				renderer->get_frame_state().render_time_ms, stats);
	}
	stats["painterly_enabled"] = renderer->get_painterly_config().enabled;
	stats["painterly_low_end_mode"] = renderer->get_painterly_config().low_end_mode;
	PainterlyPassGraph *pass_graph = renderer->get_subsystem_state().painterly_renderer.is_valid()
			? renderer->get_subsystem_state().painterly_renderer->get_pass_graph()
			: nullptr;
	stats["painterly_internal_scale"] = pass_graph ? pass_graph->get_internal_scale() : renderer->get_painterly_config().internal_scale;
	stats["using_scene_data_camera"] = renderer->get_view_state().using_scene_data;
	// Debug: expose camera transform values to verify they're updating
	stats["debug_cam_origin_x"] = renderer->get_view_state().last_camera_to_world_transform.origin.x;
	stats["debug_cam_origin_y"] = renderer->get_view_state().last_camera_to_world_transform.origin.y;
	stats["debug_cam_origin_z"] = renderer->get_view_state().last_camera_to_world_transform.origin.z;
	// Expose view matrix basis[0][0] as rotation indicator
	stats["debug_cam_basis_00"] = renderer->get_view_state().last_camera_to_world_transform.basis[0][0];

	if (debug_state_orchestrator) {
		const GaussianSplatRenderer::DebugConfig &debug_config = renderer->get_debug_config();
		if (debug_config.enable_binning_counters || debug_config.dump_gpu_counters) {
			const Dictionary binning = debug_state_orchestrator->get_binning_debug_counters();
			if (!binning.is_empty()) {
				stats["sh_cache_hits"] = binning.get("sh_cache_hits", 0);
				stats["sh_cache_updates"] = binning.get("sh_cache_updates", 0);
				stats["sh_cache_forced_updates"] = binning.get("sh_cache_forced_updates", 0);
				stats["sh_cache_hit_rate"] = binning.get("sh_cache_hit_rate", 0.0);
			}
		}
	}

	PackedInt32Array sorted_preview;
	if (renderer->get_subsystem_state().gpu_culler.is_valid()) {
		int preview_count = MIN((int)renderer->get_subsystem_state().gpu_culler->get_state().culled_indices.size(), 32);
		if (preview_count > 0) {
			sorted_preview.resize(preview_count);
			for (int i = 0; i < preview_count; i++) {
				sorted_preview.set(i, (int)renderer->get_subsystem_state().gpu_culler->get_state().culled_indices[i]);
			}
		}
	}
	if (sorted_preview.is_empty() && renderer->get_subsystem_state().sorting_pipeline.is_valid()) {
		const uint32_t sorted_count = renderer->get_sorting_state().sorted_splat_count;
		const int preview_count = MIN((int)sorted_count, 32);
		if (preview_count > 0) {
			RID sort_indices_buffer = renderer->get_subsystem_state().sorting_pipeline->get_sort_indices_buffer();
			RenderingDevice *fallback_device = renderer->get_device_state().rd;
			RenderingDevice *owner = renderer->get_resource_owner(sort_indices_buffer, fallback_device);
			if (!owner) {
				owner = fallback_device;
			}
			if (owner && sort_indices_buffer.is_valid()) {
				Vector<uint8_t> preview_bytes = owner->buffer_get_data(sort_indices_buffer, 0,
						uint32_t(preview_count) * uint32_t(sizeof(uint32_t)));
				const int expected_bytes = preview_count * int(sizeof(uint32_t));
				if (preview_bytes.size() >= expected_bytes) {
					sorted_preview.resize(preview_count);
					const uint32_t *preview_indices = reinterpret_cast<const uint32_t *>(preview_bytes.ptr());
					for (int i = 0; i < preview_count; i++) {
						sorted_preview.set(i, int(preview_indices[i]));
					}
				}
			}
		}
	}
	stats["sorted_indices_preview"] = sorted_preview;

	// GPU sorter state and metrics
	const bool has_gpu_sorter = renderer->get_sorting_state().gpu_sorter.is_valid();
	stats["gpu_sorter_initialized"] = has_gpu_sorter;
	stats["gpu_sorter_async_pipeline_ready"] = false;
	stats["rendering_device_ready"] = renderer->get_device_state().rd != nullptr;
	stats["sort_active_algorithm"] = renderer->get_sorting_state().active_sort_algorithm;
	stats["sort_switch_reason"] = renderer->get_sorting_state().sort_switch_reason;
	stats["sort_override_force_cpu"] = renderer->get_sorting_state().override_force_cpu;
	stats["sort_override_force_algorithm"] = renderer->get_sorting_state().override_force_algorithm;
	stats["sort_override_forced_algorithm"] = renderer->get_sorting_state().override_forced_algorithm;

	if (has_gpu_sorter) {
		stats["gpu_sorter_algorithm"] = renderer->get_sorting_state().gpu_sorter->get_algorithm_name();
		stats["gpu_sorter_ready"] = renderer->get_sorting_state().gpu_sorter->is_ready();
		stats["gpu_sorter_max_elements"] = (int)renderer->get_sorting_state().gpu_sorter->get_max_elements();
		stats["gpu_sorter_last_sort_ms"] = renderer->get_sorting_state().gpu_sorter->get_last_sort_time_ms();

		SortingMetrics sorter_metrics = renderer->get_sorting_state().gpu_sorter->get_metrics();
		stats["gpu_sorter_avg_sort_ms"] = sorter_metrics.avg_sort_time_ms;
		stats["gpu_sorter_peak_sort_ms"] = sorter_metrics.peak_sort_time_ms;
		stats["gpu_sorter_total_sorts"] = (int64_t)sorter_metrics.total_sorts;
		stats["gpu_sorter_async_sorts"] = (int64_t)sorter_metrics.async_sorts;
		stats["gpu_sorter_total_elements"] = (int64_t)sorter_metrics.total_elements_sorted;
		stats["gpu_sorter_bandwidth_utilization"] = sorter_metrics.bandwidth_utilization;
		stats["gpu_sorter_fallback_events"] = (int64_t)sorter_metrics.fallback_events;
		stats["gpu_sorter_last_fallback_reason"] = sorter_metrics.last_fallback_reason;
		stats["gpu_sorter_fallback_reason_counts"] = sorter_metrics.fallback_reason_counts;
	} else {
		stats["gpu_sorter_algorithm"] = String();
		stats["gpu_sorter_ready"] = false;
		stats["gpu_sorter_max_elements"] = 0;
		stats["gpu_sorter_last_sort_ms"] = 0.0f;
		stats["gpu_sorter_avg_sort_ms"] = 0.0f;
		stats["gpu_sorter_peak_sort_ms"] = 0.0f;
		stats["gpu_sorter_total_sorts"] = 0;
		stats["gpu_sorter_async_sorts"] = 0;
		stats["gpu_sorter_total_elements"] = 0;
		stats["gpu_sorter_bandwidth_utilization"] = 0.0f;
		stats["gpu_sorter_fallback_events"] = 0;
		stats["gpu_sorter_last_fallback_reason"] = String();
		stats["gpu_sorter_fallback_reason_counts"] = Dictionary();
	}

	// GPU buffer manager stats
	if (renderer->get_resource_state().buffer_manager.is_valid() && renderer->get_resource_state().buffer_manager_initialized) {
		stats["buffer_manager_memory_mb"] = renderer->get_resource_state().buffer_manager->get_memory_usage_mb();
		stats["buffer_manager_capacity"] = renderer->get_resource_state().buffer_manager->get_buffer_capacity();
		stats["buffer_manager_count"] = renderer->get_resource_state().buffer_manager->get_gaussian_count();
	} else {
		stats["visible_ratio"] = 0.0;
		stats["culled_ratio"] = 0.0;
	}

	stats["debug_show_tile_grid"] = overlay_system ? overlay_options.show_tile_grid : renderer->get_debug_state().show_tile_grid;
	stats["debug_show_density_heatmap"] = overlay_system ? overlay_options.show_density_heatmap : renderer->get_debug_state().show_density_heatmap;
	stats["debug_show_performance_hud"] = overlay_system ? overlay_options.show_performance_hud : renderer->get_debug_state().show_performance_hud;
	stats["debug_show_residency_hud"] = overlay_system ? overlay_options.show_residency_hud : renderer->get_debug_state().show_residency_hud;
	const String normalized_route_uid = _normalize_route_uid_for_stats(renderer->get_debug_state().route_uid);
	const String normalized_sort_route_uid = _normalize_sort_route_uid_for_stats(renderer->get_debug_state().sort_route_uid);
	stats["route_uid"] = normalized_route_uid;
	stats["sort_route_uid"] = normalized_sort_route_uid;
	stats["route_uid_missing"] = RenderRouteUID::is_route_uid_missing(normalized_route_uid);
	stats["sort_route_uid_missing"] = RenderRouteUID::is_sort_route_uid_missing(normalized_sort_route_uid);
	stats["instance_pipeline_content_generation"] =
			static_cast<int64_t>(renderer->get_resource_state().instance_pipeline_content_generation);
	stats["cached_render_reuse_enabled"] = renderer->is_cached_render_reuse_enabled();
	if (renderer->get_subsystem_state().gpu_culler.is_valid()) {
		const auto &cull_state = renderer->get_subsystem_state().gpu_culler->get_state();
		stats["cull_static_chunk_total"] = static_cast<int64_t>(cull_state.static_chunks.size());
		stats["cull_visible_static_chunks"] = static_cast<int64_t>(cull_state.visible_static_chunk_indices.size());
		stats["cull_gpu_visible_count"] = static_cast<int64_t>(cull_state.gpu_visible_indices_count);
		stats["cull_cpu_visible_count"] = static_cast<int64_t>(cull_state.culled_indices.size());
		stats["cull_total_splats_pre_cull"] = static_cast<int64_t>(cull_state.total_splats_pre_cull);
	} else {
		stats["cull_static_chunk_total"] = static_cast<int64_t>(0);
		stats["cull_visible_static_chunks"] = static_cast<int64_t>(0);
		stats["cull_gpu_visible_count"] = static_cast<int64_t>(0);
		stats["cull_cpu_visible_count"] = static_cast<int64_t>(0);
		stats["cull_total_splats_pre_cull"] = static_cast<int64_t>(0);
	}
	stats["debug_overlay_version"] = renderer->get_debug_state().overlay_version;
	stats["debug_hud_version"] = renderer->get_debug_state().hud_version;
	stats["debug_tile_density_peak"] = (int)(overlay_system ? overlay_query.get_tile_density_peak() : renderer->get_debug_state().tile_density_peak);
	stats["debug_tile_density_average"] = overlay_system ? overlay_query.get_tile_density_average() : renderer->get_debug_state().tile_density_average;
	stats["debug_tile_density_size"] =
			overlay_system
					? Vector2i(overlay_query.get_tile_density_width(), overlay_query.get_tile_density_height())
					: Vector2i(renderer->get_debug_state().tile_density_width, renderer->get_debug_state().tile_density_height);

	Array hud_lines_array;
	const Vector<String> &hud_lines = overlay_system ? overlay_query.get_hud_lines() : renderer->get_debug_state().hud_lines;
	for (const String &line : hud_lines) {
		hud_lines_array.push_back(line);
	}
	stats["performance_hud_lines"] = hud_lines_array;
	stats["debug_preview_mode"] = renderer->get_debug_state().preview_mode;
	stats["telemetry"] = diagnostics_state.last_telemetry_snapshot;
	stats["production_metrics"] = diagnostics_state.last_production_metrics;
	stats["production_metrics_validation"] = diagnostics_state.last_production_metrics_validation;
	stats["production_metrics_invalid_count"] = static_cast<int64_t>(diagnostics_state.production_metrics_invalid_count);
	stats["perf_gate"] = diagnostics_state.last_perf_gate_result;
	return stats;
}

float RenderDiagnosticsOrchestrator::get_sort_time_ms_internal() const {
	return renderer->get_frame_state().sort_time_ms;
}

float RenderDiagnosticsOrchestrator::get_render_time_ms_internal() const {
	return renderer->get_frame_state().render_time_ms;
}

Dictionary RenderDiagnosticsOrchestrator::get_last_sort_metrics_internal() const {
	Dictionary metrics;
	SortingStrategyConfig config = SortingStrategyConfig::load_from_project_settings();
	if (!renderer->get_performance_state().sort_metrics_history.is_empty()) {
		const GaussianSplatRenderer::SortFrameMetrics &sample =
				renderer->get_performance_state().sort_metrics_history[renderer->get_performance_state().sort_metrics_history.size() - 1];
		metrics["frame"] = sample.frame_index;
		metrics["elements"] = sample.element_count;
		metrics["total_ms"] = sample.total_ms;
		metrics["gpu_ms"] = sample.gpu_ms;
		metrics["cpu_ms"] = sample.cpu_ms;
		metrics["cpu_selection_ms"] = sample.cpu_selection_ms;
		metrics["algorithm"] = String(sample.algorithm);
		metrics["used_gpu"] = sample.used_gpu;
		metrics["cpu_fallback"] = sample.used_cpu_fallback;
		metrics["hybrid"] = sample.used_hybrid;
	}
	metrics["target_ms"] = config.target_sort_time_ms;
	const auto &sorting_state = renderer->get_sorting_state();
	metrics["active_algorithm"] = sorting_state.active_sort_algorithm;
	metrics["switch_reason"] = sorting_state.sort_switch_reason;
	metrics["override_force_cpu"] = sorting_state.override_force_cpu;
	metrics["override_force_algorithm"] = sorting_state.override_force_algorithm;
	metrics["override_forced_algorithm"] = sorting_state.override_forced_algorithm;
	return metrics;
}

Array RenderDiagnosticsOrchestrator::get_sort_metrics_history_internal() const {
	Array history;
	for (const GaussianSplatRenderer::SortFrameMetrics &sample : renderer->get_performance_state().sort_metrics_history) {
		Dictionary entry;
		entry["frame"] = sample.frame_index;
		entry["elements"] = sample.element_count;
		entry["total_ms"] = sample.total_ms;
		entry["gpu_ms"] = sample.gpu_ms;
		entry["cpu_ms"] = sample.cpu_ms;
		entry["cpu_selection_ms"] = sample.cpu_selection_ms;
		entry["algorithm"] = String(sample.algorithm);
		entry["used_gpu"] = sample.used_gpu;
		entry["cpu_fallback"] = sample.used_cpu_fallback;
		entry["hybrid"] = sample.used_hybrid;
		history.push_back(entry);
	}
	return history;
}

void RenderDiagnosticsOrchestrator::record_sort_sample(const GaussianSplatRenderer::SortFrameMetrics &p_sample) {
	SortingStrategyConfig config = SortingStrategyConfig::load_from_project_settings();
	uint32_t history_limit = config.history_size > 0 ? config.history_size : 120;

	renderer->get_performance_state().sort_metrics_history.push_back(p_sample);
	while (renderer->get_performance_state().sort_metrics_history.size() > history_limit) {
		renderer->get_performance_state().sort_metrics_history.remove_at(0);
	}

	if (config.log_metrics && config.log_interval_frames > 0) {
		if (p_sample.frame_index % config.log_interval_frames == 0) {
			GS_LOG_GPU_SORT_INFO(vformat("[Sort Metrics] frame=%d elements=%d total=%.2f ms gpu=%.2f ms cpu=%.2f ms",
					p_sample.frame_index,
					p_sample.element_count,
					p_sample.total_ms,
					p_sample.gpu_ms,
					p_sample.cpu_ms));
		}
	}
}

void RenderDiagnosticsOrchestrator::finalize_frame_metrics(uint64_t p_frame_start_usec) {
	debug_state_orchestrator->update_frame_times(renderer->get_frame_state().render_time_ms, renderer->get_frame_state().sort_time_ms);
	const Ref<DebugOverlaySystem> &overlay_system_ref = renderer->get_subsystem_state().debug_overlay_system;
	DebugOverlaySystem *overlay_system = overlay_system_ref.is_valid() ? overlay_system_ref.ptr() : nullptr;
	const DebugOverlayOptions overlay_options = overlay_system
			? overlay_system->build_query_view(renderer).get_options()
			: DebugOverlayOptions();
	if ((overlay_system ? (overlay_options.show_performance_hud || overlay_options.show_residency_hud)
			: (renderer->get_debug_state().show_performance_hud || renderer->get_debug_state().show_residency_hud))) {
		if (overlay_system) {
			overlay_system->build_command_sink(renderer).invalidate_hud(false);
		}
	}

	// Update performance metrics
	uint64_t frame_end = OS::get_singleton()->get_ticks_usec();
	float frame_time_ms = (frame_end - p_frame_start_usec) / 1000.0f;

	// Frame-to-frame timing (measures actual throughput including GPU wait)
	if (renderer->get_performance_state().metrics.last_frame_start_usec > 0) {
		renderer->get_performance_state().metrics.frame_to_frame_time_ms =
				(p_frame_start_usec - renderer->get_performance_state().metrics.last_frame_start_usec) / 1000.0f;
	}
	renderer->get_performance_state().metrics.last_frame_start_usec = p_frame_start_usec;

	renderer->get_performance_state().metrics.total_frames_rendered++;
	renderer->get_performance_state().metrics.rendered_splat_count = renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);

	// Update rolling average frame time (use frame-to-frame for true FPS)
	float alpha = 0.1f; // Smoothing factor
	float timing_for_avg = renderer->get_performance_state().metrics.frame_to_frame_time_ms > 0.0f
			? renderer->get_performance_state().metrics.frame_to_frame_time_ms
			: frame_time_ms;
	if (renderer->get_performance_state().metrics.total_frames_rendered <= 2) {
		renderer->get_performance_state().metrics.avg_frame_time_ms = timing_for_avg;
		renderer->get_performance_state().metrics.avg_frame_to_frame_ms = timing_for_avg;
	} else {
		renderer->get_performance_state().metrics.avg_frame_time_ms =
				renderer->get_performance_state().metrics.avg_frame_time_ms * (1.0f - alpha) + frame_time_ms * alpha;
		renderer->get_performance_state().metrics.avg_frame_to_frame_ms =
				renderer->get_performance_state().metrics.avg_frame_to_frame_ms * (1.0f - alpha) + timing_for_avg * alpha;
	}

	renderer->get_performance_state().metrics.peak_frame_time_ms =
			MAX(renderer->get_performance_state().metrics.peak_frame_time_ms, renderer->get_performance_state().metrics.frame_to_frame_time_ms);

	// Ensure GPU metrics are up to date before logging
	(renderer->*runtime_ports.update_gpu_pass_metrics_from_tile_renderer)();

	ProductionMetricsConfig metrics_config = _load_production_metrics_config();
	const bool stage_metrics_valid = renderer->get_debug_state().last_stage_metrics_valid;
	GaussianSplatRenderer::StageMetrics stage_metrics = stage_metrics_valid
			? renderer->get_debug_state().last_stage_metrics
			: GaussianSplatRenderer::StageMetrics();
	Dictionary production_metrics = _build_production_metrics_snapshot(
			*renderer, stage_metrics, stage_metrics_valid, frame_time_ms);
	Dictionary telemetry_snapshot = production_metrics.duplicate();
	_append_telemetry_extras(*renderer, stage_metrics, stage_metrics_valid, frame_time_ms, telemetry_snapshot);
	diagnostics_state.last_production_metrics = production_metrics;
	diagnostics_state.last_telemetry_snapshot = telemetry_snapshot;

	if (metrics_config.validate_metrics) {
		diagnostics_state.last_production_metrics_validation =
				_validate_production_metrics(diagnostics_state.last_production_metrics);
		if (!bool(diagnostics_state.last_production_metrics_validation.get("valid", true))) {
			diagnostics_state.production_metrics_invalid_count++;
		}
	} else {
		Dictionary validation;
		validation["valid"] = true;
		validation["disabled"] = true;
		validation["frame"] = diagnostics_state.last_production_metrics.get("frame", int64_t(0));
		diagnostics_state.last_production_metrics_validation = validation;
	}

	diagnostics_state.last_perf_gate_result =
			_evaluate_perf_gate(metrics_config, diagnostics_state.last_production_metrics);

	_update_production_summary(diagnostics_state, metrics_config, diagnostics_state.last_production_metrics,
			diagnostics_state.last_perf_gate_result, frame_end);
}

Dictionary RenderDiagnosticsOrchestrator::get_runtime_diagnostic_snapshot() const {
	Dictionary snapshot;
	snapshot["frame"] = static_cast<int64_t>(renderer->get_frame_state().frame_counter);
	snapshot["device_capability_report"] = build_device_capability_report();
	snapshot["texture_allocation_trace"] = serialize_texture_trace();
	snapshot["cross_device_operation_log"] = serialize_cross_device_operations();
	snapshot["frame_timing_analysis"] = serialize_frame_timing();
	snapshot["error_statistics"] = serialize_error_statistics();
	snapshot["production_metrics_contract"] = _production_metrics_contract();
	snapshot["production_metrics"] = diagnostics_state.last_production_metrics;
	snapshot["production_metrics_validation"] = diagnostics_state.last_production_metrics_validation;
	snapshot["production_metrics_invalid_count"] = static_cast<int64_t>(diagnostics_state.production_metrics_invalid_count);
	snapshot["perf_gate"] = diagnostics_state.last_perf_gate_result;
	snapshot["telemetry"] = diagnostics_state.last_telemetry_snapshot;

	Array summaries;
	for (const Dictionary &summary : diagnostics_state.production_metrics_summaries) {
		summaries.push_back(summary);
	}
	snapshot["production_metrics_summaries"] = summaries;

	// Read debug modes from DebugOverlaySystem (1b.3 debug seam)
	Dictionary debug_modes;
	const Ref<DebugOverlaySystem> &overlay_system_ref = renderer->get_subsystem_state().debug_overlay_system;
	if (overlay_system_ref.is_valid()) {
		const DebugOverlaySystem *overlay_system = overlay_system_ref.ptr();
		const DebugOverlayOptions overlay_options = overlay_system->build_query_view(renderer).get_options();
		debug_modes["tile_bounds"] = overlay_options.show_tile_bounds;
		debug_modes["splat_coverage"] = overlay_options.show_splat_coverage;
		debug_modes["overflow_tiles"] = overlay_options.show_overflow_tiles;
		debug_modes["projection_issues"] = overlay_options.show_projection_issues;
		debug_modes["tile_grid"] = overlay_options.show_tile_grid;
		debug_modes["density_heatmap"] = overlay_options.show_density_heatmap;
		debug_modes["shadow_opacity"] = overlay_options.show_shadow_opacity;
		debug_modes["performance_hud"] = overlay_options.show_performance_hud;
		debug_modes["residency_hud"] = overlay_options.show_residency_hud;
		debug_modes["device_boundaries"] = overlay_options.show_device_boundaries;
		debug_modes["texture_states"] = overlay_options.show_texture_states;
		debug_modes["resolve_input"] = overlay_options.show_resolve_input;
		debug_modes["resolve_output"] = overlay_options.show_resolve_output;
		debug_modes["dump_gpu_counters"] = overlay_options.dump_gpu_counters;
	}
	snapshot["debug_modes"] = debug_modes;

	Dictionary gpu_performance;
	const auto &perf = renderer->get_performance_state().metrics;
	gpu_performance["utilization_percent"] = perf.gpu_utilization;
	gpu_performance["frame_gpu_ms"] = perf.gpu_frame_time_ms;
	gpu_performance["binning_gpu_ms"] = perf.gpu_tile_binning_time_ms;
	gpu_performance["raster_gpu_ms"] = perf.gpu_tile_raster_time_ms;
	gpu_performance["prefix_gpu_ms"] = perf.gpu_tile_prefix_time_ms;
	gpu_performance["resolve_gpu_ms"] = perf.gpu_tile_resolve_time_ms;
	gpu_performance["timing_frame_serial"] = (int64_t)perf.gpu_timing_frame_serial;
	gpu_performance["timing_frames_behind"] = (int64_t)perf.gpu_timing_frames_behind;
	gpu_performance["timeline_inflight_frames"] = (int64_t)perf.gpu_timeline_inflight_frames;
	gpu_performance["timeline_completed_frames"] = (int64_t)perf.gpu_timeline_completed_frames;
	gpu_performance["timeline_stall_count"] = (int64_t)perf.gpu_timeline_stall_count;
	gpu_performance["timeline_stall_ms"] = perf.gpu_timeline_stall_ms;
	gpu_performance["timeline_last_value"] = (int64_t)perf.gpu_timeline_last_value;
	snapshot["gpu_performance"] = gpu_performance;

	return snapshot;
}

Dictionary GaussianSplatRenderer::get_render_stats() const {
	ERR_FAIL_NULL_V(diagnostics_orchestrator, Dictionary());
	return diagnostics_orchestrator->build_render_stats();
}

float GaussianSplatRenderer::get_sort_time_ms() const {
	ERR_FAIL_NULL_V(diagnostics_orchestrator, 0.0f);
	return diagnostics_orchestrator->get_sort_time_ms_internal();
}

float GaussianSplatRenderer::get_render_time_ms() const {
	ERR_FAIL_NULL_V(diagnostics_orchestrator, 0.0f);
	return diagnostics_orchestrator->get_render_time_ms_internal();
}

Dictionary GaussianSplatRenderer::get_last_sort_metrics() const {
	ERR_FAIL_NULL_V(diagnostics_orchestrator, Dictionary());
	return diagnostics_orchestrator->get_last_sort_metrics_internal();
}

Array GaussianSplatRenderer::get_sort_metrics_history() const {
	ERR_FAIL_NULL_V(diagnostics_orchestrator, Array());
	return diagnostics_orchestrator->get_sort_metrics_history_internal();
}

void GaussianSplatRenderer::record_sort_sample(const SortFrameMetrics &p_sample) {
	ERR_FAIL_NULL(diagnostics_orchestrator);
	diagnostics_orchestrator->record_sort_sample(p_sample);
}

Dictionary GaussianSplatRenderer::get_runtime_diagnostic_snapshot() const {
	return diagnostics_orchestrator->get_runtime_diagnostic_snapshot();
}

GaussianSplatRenderer::MonitorStreamingSnapshot GaussianSplatRenderer::get_monitor_streaming_snapshot() const {
	MonitorStreamingSnapshot snapshot;
	const DebugState &debug_state = get_debug_state();
	const SceneState &scene_state = get_scene_state();
	const FrameState &frame_state = get_frame_state();
	const PerformanceMetrics &metrics = get_performance_state().metrics;

	snapshot.route_uid = debug_state.route_uid;
	snapshot.sort_route_uid = debug_state.sort_route_uid;
	snapshot.stage_metrics_valid = debug_state.last_stage_metrics_valid;
	if (debug_state.last_stage_metrics_valid) {
		snapshot.stage_cull_time_ms = debug_state.last_stage_metrics.cull.cull_time_ms;
		snapshot.stage_sort_time_ms = debug_state.last_stage_metrics.sort.sort_time_ms;
	}
	snapshot.frame_sort_time_ms = frame_state.sort_time_ms;
	snapshot.metrics_gpu_frame_time_ms = metrics.gpu_frame_time_ms;
	snapshot.metrics_culling_time_ms = metrics.culling_time_ms;
	snapshot.metrics_gpu_tile_binning_time_ms = metrics.gpu_tile_binning_time_ms;
	snapshot.metrics_gpu_tile_prefix_time_ms = metrics.gpu_tile_prefix_time_ms;
	snapshot.metrics_gpu_tile_raster_time_ms = metrics.gpu_tile_raster_time_ms;
	snapshot.metrics_gpu_tile_resolve_time_ms = metrics.gpu_tile_resolve_time_ms;
	snapshot.metrics_visible_after_culling = metrics.visible_after_culling;
	snapshot.metrics_rendered_splat_count = metrics.rendered_splat_count;
	snapshot.frame_visible_splat_count = frame_state.visible_splat_count.load(std::memory_order_acquire);
	snapshot.has_streaming_data = (scene_state.gaussian_data.is_valid() && scene_state.gaussian_data->get_count() > 0) ||
			scene_state.active_asset.is_valid();

	const StreamingState &streaming_state = get_streaming_state();
	if (!streaming_state.current_streaming_system.is_valid()) {
		return snapshot;
	}

	Ref<GaussianStreamingSystem> streaming_system = streaming_state.current_streaming_system;
	snapshot.has_streaming_system = true;
	snapshot.runtime_ready = streaming_system->is_runtime_ready();
	snapshot.streaming_analytics = streaming_system->get_streaming_analytics();
	snapshot.vram_debug_stats = streaming_system->get_vram_debug_stats();
	snapshot.chunk_culling_stats = streaming_system->get_chunk_culling_stats();
	snapshot.lod_debug_stats = streaming_system->get_lod_debug_stats();
	snapshot.vram_usage_bytes = streaming_system->get_vram_usage();
	snapshot.chunks_loaded_this_frame = streaming_system->get_chunks_loaded_this_frame();
	snapshot.chunks_evicted_this_frame = streaming_system->get_chunks_evicted_this_frame();
	snapshot.visible_splat_count = streaming_system->get_visible_count();
	snapshot.buffer_capacity_splats = streaming_system->get_buffer_capacity_splats();
	snapshot.effective_splat_count = streaming_system->get_effective_splat_count();
	snapshot.visible_chunk_change_ratio = streaming_system->get_visible_chunk_change_ratio();
	snapshot.global_lod_blend_factor = streaming_system->get_global_lod_blend_factor();
	snapshot.global_sh_band_level = streaming_system->get_global_sh_band_level();
	snapshot.lod_hysteresis_zone = streaming_system->get_lod_hysteresis_zone();
	snapshot.lod_blend_distance = streaming_system->get_lod_blend_distance();
	snapshot.sh_compression_metrics_valid = true;
	snapshot.sh_compression_metrics = streaming_system->get_total_sh_metrics();

	Ref<VRAMBudgetRegulator> regulator = streaming_system->get_vram_regulator();
	if (regulator.is_valid()) {
		snapshot.lod_distance_multiplier = regulator->get_lod_distance_multiplier();
	}

	if (streaming_state.memory_stream.is_valid()) {
		snapshot.memory_stream_valid = true;
		snapshot.memory_stream_stats = streaming_state.memory_stream->get_stats();
	}

	return snapshot;
}
