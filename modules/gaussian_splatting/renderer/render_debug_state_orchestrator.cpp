#include "render_debug_state_orchestrator.h"

#include "../core/gs_project_settings.h"
#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/variant/variant.h"
#include "core/io/json.h"
#include "../logger/gs_debug_trace.h"
#include "../interfaces/debug_overlay_system.h"
#include "../interfaces/output_compositor.h"

namespace {

// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).
static bool _get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
	return gs::settings::get_bool(ps, name, fallback);
}

static int _get_int_setting(ProjectSettings *ps, const StringName &name, int fallback) {
	return static_cast<int>(gs::settings::get_uint(ps, name, static_cast<uint32_t>(fallback)));
}

static float _get_float_setting(ProjectSettings *ps, const StringName &name, float fallback) {
	return gs::settings::get_float(ps, name, fallback);
}


static bool _default_pipeline_trace_enabled() {
#if 0 // Disabled for perf baseline
	return true;
#else
	return false;
#endif
}

static constexpr uint32_t kAnomalyDropDivisor = 4;
static constexpr uint32_t kAnomalyDropMinCount = 128;
static constexpr uint64_t kAnomalyDumpCooldownFrames = 30;

static bool _is_drop(uint32_t p_previous, uint32_t p_current, uint32_t p_min_count, uint32_t p_divisor) {
	if (p_previous < p_min_count) {
		return false;
	 }
	return p_current * p_divisor < p_previous;
}

static uint64_t _hash_combine(uint64_t p_seed, uint64_t p_value) {
	// 64-bit hash combine (boost-style).
	return p_seed ^ (p_value + 0x9e3779b97f4a7c15ULL + (p_seed << 6) + (p_seed >> 2));
}

static int64_t _quantize_float(float p_value, float p_step) {
	if (p_step <= 0.0f) {
		return 0;
	}
	return static_cast<int64_t>(Math::round(p_value / p_step));
}

static uint64_t _hash_camera_pose(const Transform3D &p_transform, float p_pos_step, float p_rot_step) {
	if (p_pos_step <= 0.0f || p_rot_step <= 0.0f) {
		return 0;
	}

	Vector3 pos = p_transform.origin;
	Quaternion rot = p_transform.basis.get_rotation_quaternion();
	if (rot.w < 0.0f) {
		rot = Quaternion(-rot.x, -rot.y, -rot.z, -rot.w);
	}

	uint64_t seed = 1469598103934665603ULL;
	seed = _hash_combine(seed, static_cast<uint64_t>(_quantize_float(pos.x, p_pos_step)));
	seed = _hash_combine(seed, static_cast<uint64_t>(_quantize_float(pos.y, p_pos_step)));
	seed = _hash_combine(seed, static_cast<uint64_t>(_quantize_float(pos.z, p_pos_step)));
	seed = _hash_combine(seed, static_cast<uint64_t>(_quantize_float(rot.x, p_rot_step)));
	seed = _hash_combine(seed, static_cast<uint64_t>(_quantize_float(rot.y, p_rot_step)));
	seed = _hash_combine(seed, static_cast<uint64_t>(_quantize_float(rot.z, p_rot_step)));
	seed = _hash_combine(seed, static_cast<uint64_t>(_quantize_float(rot.w, p_rot_step)));
	return seed;
}

static String _normalize_route_uid_for_snapshot(const String &p_route_uid) {
	if (p_route_uid.is_empty()) {
		return RenderRouteUID::COMMON_UNKNOWN_ROUTE;
	}
	return p_route_uid;
}

static String _normalize_sort_route_uid_for_snapshot(const String &p_sort_route_uid) {
	if (p_sort_route_uid.is_empty()) {
		return RenderRouteUID::COMMON_UNKNOWN_SORT_ROUTE;
	}
	return p_sort_route_uid;
}

static void _record_debug_pipeline_event(GaussianSplatRenderer *p_renderer, const String &p_message,
		uint32_t p_input_count, uint32_t p_output_count, bool p_is_error) {
	if (!p_renderer || !p_renderer->get_debug_config().enable_pipeline_trace) {
		return;
	}
	auto &debug_state = p_renderer->get_debug_state();
	if (!debug_state.pipeline_events_valid) {
		debug_state.pipeline_events_frame = p_renderer->get_frame_state().frame_counter;
		debug_state.pipeline_events_valid = true;
	}
	GaussianSplatRenderer::PipelineEvent event;
	event.stage = "debug";
	event.message = p_message;
	event.route_uid = debug_state.route_uid;
	event.input_count = p_input_count;
	event.output_count = p_output_count;
	event.is_error = p_is_error;
	debug_state.pipeline_events.push_back(event);
}

static Error _save_texture_snapshot(RenderingDevice *p_device, const RID &p_texture, const String &p_path) {
	if (!p_device || !p_texture.is_valid()) {
		return ERR_INVALID_PARAMETER;
	}

	RenderingDevice::TextureFormat format = p_device->texture_get_format(p_texture);
	Image::Format image_format = Image::FORMAT_MAX;
	switch (format.format) {
		case RD::DATA_FORMAT_R8G8B8A8_UNORM:
		case RD::DATA_FORMAT_R8G8B8A8_SRGB:
			image_format = Image::FORMAT_RGBA8;
			break;
		case RD::DATA_FORMAT_R16G16B16A16_SFLOAT:
			image_format = Image::FORMAT_RGBAH;
			break;
		case RD::DATA_FORMAT_R32G32B32A32_SFLOAT:
			image_format = Image::FORMAT_RGBAF;
			break;
		default:
			return ERR_UNAVAILABLE;
	}

	Vector<uint8_t> data = p_device->texture_get_data(p_texture, 0);
	if (data.is_empty()) {
		return ERR_CANT_ACQUIRE_RESOURCE;
	}

	Ref<Image> image = Image::create_from_data(format.width, format.height, false, image_format, data);
	if (!image.is_valid()) {
		return ERR_CANT_CREATE;
	}
	if (image_format != Image::FORMAT_RGBA8) {
		image->convert(Image::FORMAT_RGBA8);
	}

	return image->save_png(p_path);
}

static Dictionary _stage_io_to_dict(const GaussianSplatRenderer::StageIO &p_io) {
	Dictionary out;
	out["frame"] = static_cast<int64_t>(p_io.frame_id);
	out["input_count"] = static_cast<int64_t>(p_io.input_count);
	out["output_count"] = static_cast<int64_t>(p_io.output_count);
	out["input_buffer"] = p_io.input_buffer.is_valid() ? static_cast<int64_t>(p_io.input_buffer.get_id()) : int64_t(0);
	out["output_buffer"] = p_io.output_buffer.is_valid() ? static_cast<int64_t>(p_io.output_buffer.get_id()) : int64_t(0);
	out["validated"] = p_io.validated;
	out["validation_failed"] = p_io.validation_failed;
	out["validation_error"] = p_io.validation_error;
	return out;
}

static String _fallback_reason_to_string(GaussianSplatRenderer::RenderFallbackReason p_reason);

static Dictionary _pipeline_event_to_dict(const GaussianSplatRenderer::PipelineEvent &p_event) {
	Dictionary out;
	out["stage"] = p_event.stage;
	out["message"] = p_event.message;
	out["route_uid"] = p_event.route_uid;
	out["input_count"] = static_cast<int64_t>(p_event.input_count);
	out["output_count"] = static_cast<int64_t>(p_event.output_count);
	out["is_error"] = p_event.is_error;
	out["fallback_reason"] = _fallback_reason_to_string(p_event.fallback_reason);
	out["fallback_reason_code"] = static_cast<int64_t>(p_event.fallback_reason);
	return out;
}

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

static String _fallback_reason_to_string(GaussianSplatRenderer::RenderFallbackReason p_reason) {
	switch (p_reason) {
		case GaussianSplatRenderer::RenderFallbackReason::NONE:
			return "none";
		case GaussianSplatRenderer::RenderFallbackReason::DATA_UNAVAILABLE:
			return "data_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::STREAMING_DATA_UNAVAILABLE:
			return "streaming_data_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::GPU_CULLER_UNAVAILABLE:
			return "gpu_culler_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::NO_VISIBLE_SPLATS:
			return "no_visible_splats";
		case GaussianSplatRenderer::RenderFallbackReason::RENDERING_DEVICE_UNAVAILABLE:
			return "rendering_device_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::RASTER_REUSED_CACHED_RENDER:
			return "raster_reused_cached_render";
		case GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_UNAVAILABLE:
			return "painterly_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_PASS_GRAPH_UNAVAILABLE:
			return "painterly_pass_graph_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_MATERIAL_UNAVAILABLE:
			return "painterly_material_unavailable";
		case GaussianSplatRenderer::RenderFallbackReason::PAINTERLY_RENDER_FAILED:
			return "painterly_render_failed";
		case GaussianSplatRenderer::RenderFallbackReason::TILE_FALLBACK_FAILED:
			return "tile_fallback_failed";
		case GaussianSplatRenderer::RenderFallbackReason::OUTPUT_COMPOSITOR_UNAVAILABLE:
			return "output_compositor_unavailable";
	}
	return "unknown";
}

static Dictionary _stage_result_to_dict(const GaussianSplatRenderer::StageResult &p_result) {
	Dictionary out;
	out["status"] = _stage_status_to_string(p_result.status);
	out["is_error"] = p_result.is_error;
	out["reason"] = p_result.reason;
	out["fallback_reason"] = _fallback_reason_to_string(p_result.fallback_reason);
	return out;
}

static Dictionary _cull_output_to_dict(const GaussianSplatRenderer::CullStageOutput &p_output) {
	Dictionary out;
	out["has_visible"] = p_output.has_visible;
	out["visible_count"] = static_cast<int64_t>(p_output.visible_count);
	out["candidate_count"] = static_cast<int64_t>(p_output.candidate_count);
	out["time_ms"] = p_output.cull_time_ms;
	out["visible_domain"] = GaussianRenderState::index_domain_to_string(p_output.visible_domain);
	return out;
}

static Dictionary _sort_output_to_dict(const GaussianSplatRenderer::SortStageOutput &p_output) {
	Dictionary out;
	out["did_sort"] = p_output.did_sort;
	out["input_count"] = static_cast<int64_t>(p_output.input_count);
	out["sorted_count"] = static_cast<int64_t>(p_output.sorted_count);
	out["time_ms"] = p_output.sort_time_ms;
	out["input_domain"] = GaussianRenderState::index_domain_to_string(p_output.input_domain);
	out["output_domain"] = GaussianRenderState::index_domain_to_string(p_output.output_domain);
	return out;
}

static Dictionary _raster_output_to_dict(const GaussianSplatRenderer::RasterStageOutput &p_output) {
	Dictionary out;
	out["color"] = p_output.color.is_valid() ? static_cast<int64_t>(p_output.color.get_id()) : int64_t(0);
	out["depth"] = p_output.depth.is_valid() ? static_cast<int64_t>(p_output.depth.get_id()) : int64_t(0);
	out["internal_width"] = p_output.internal_size.x;
	out["internal_height"] = p_output.internal_size.y;
	out["painterly_active"] = p_output.painterly_active;
	out["reused_cached_render"] = p_output.reused_cached_render;
	out["render_time_ms"] = p_output.render_time_ms;
	out["raster_path"] = p_output.raster_path;
	out["sorted_splat_count"] = static_cast<int64_t>(p_output.sorted_splat_count);
	out["content_generation"] = static_cast<int64_t>(p_output.content_generation);
	out["shader_defines_hash"] = static_cast<int64_t>(p_output.shader_defines_hash);
	return out;
}

static Dictionary _splat_audit_to_dict(const GaussianSplatRenderer::SplatAuditSummary &p_summary) {
	Dictionary out;
	out["valid"] = p_summary.valid;
	out["sample_count"] = static_cast<int64_t>(p_summary.sample_count);
	out["projected_count"] = static_cast<int64_t>(p_summary.projected_count);
	out["in_viewport_count"] = static_cast<int64_t>(p_summary.in_viewport_count);
	out["iterated_count"] = static_cast<int64_t>(p_summary.iterated_count);
	out["contributed_count"] = static_cast<int64_t>(p_summary.contributed_count);
	out["alpha_skipped_count"] = static_cast<int64_t>(p_summary.alpha_skipped_count);
	out["missing_iterated_count"] = static_cast<int64_t>(p_summary.missing_iterated_count);
	out["missing_contrib_count"] = static_cast<int64_t>(p_summary.missing_contrib_count);
	if (p_summary.first_mismatch_flags != 0u) {
		Dictionary mismatch;
		mismatch["global_idx"] = static_cast<int64_t>(p_summary.first_mismatch_global_idx);
		mismatch["expected_x"] = static_cast<int64_t>(p_summary.first_mismatch_expected_x);
		mismatch["expected_y"] = static_cast<int64_t>(p_summary.first_mismatch_expected_y);
		mismatch["flags"] = static_cast<int64_t>(p_summary.first_mismatch_flags);
		out["first_mismatch"] = mismatch;
	}
	return out;
}

} // namespace

RenderDebugStateOrchestrator::RenderDebugStateOrchestrator(GaussianSplatRenderer *p_renderer,
		Ref<TileRenderer> *p_tile_renderer, Ref<DebugOverlaySystem> *p_debug_overlay_system,
		GaussianSplatRenderer::JacobianDebugConfig *p_jacobian_debug) :
		renderer(p_renderer),
		tile_renderer(p_tile_renderer),
		debug_overlay_system(p_debug_overlay_system),
		jacobian_debug(p_jacobian_debug) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(tile_renderer);
	ERR_FAIL_NULL(debug_overlay_system);
	ERR_FAIL_NULL(jacobian_debug);

	const bool default_trace = _default_pipeline_trace_enabled();
	if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
		debug_config.enable_all_debug = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_all_debug", false);
		debug_config.enable_pipeline_trace = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_pipeline_trace", default_trace);
		debug_config.enable_state_guardrails = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_state_guardrails", false);
		debug_config.enable_splat_audit = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_splat_audit", false);
		debug_config.enable_frame_logging = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_frame_logging", false);
		debug_config.enable_frame_logging_verbose = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_frame_logging_verbose", false);
		debug_config.frame_log_frequency = _get_int_setting(ps,
				"rendering/gaussian_splatting/debug/frame_log_frequency", debug_config.frame_log_frequency);
		debug_config.enable_sort_path_logs = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_sort_path_logs", false);
		debug_config.enable_tile_logs = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_tile_logs", false);
		debug_config.enable_tile_pipeline_logs = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_tile_pipeline_logs", false);
		debug_config.enable_tile_dispatch_logs = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_tile_dispatch_logs", false);
		debug_config.enable_gpu_counter_logs = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_gpu_counter_logs", false);
		debug_config.enable_binning_counters = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_binning_counters", false);
		debug_config.enable_cull_counters = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_cull_counters", false);
		debug_config.enable_cull_guardrails = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_cull_guardrails", false);
		debug_config.enable_autotune_logs = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_autotune_logs", false);
		debug_config.enable_data_logging = _get_bool_setting(ps,
				"rendering/gaussian_splatting/debug/enable_data_logging", false);
		debug_config.cull_guardrail_position_epsilon = _get_float_setting(ps,
				"rendering/gaussian_splatting/debug/cull_guardrail_position_epsilon",
				debug_config.cull_guardrail_position_epsilon);
		debug_config.cull_guardrail_rotation_epsilon = _get_float_setting(ps,
				"rendering/gaussian_splatting/debug/cull_guardrail_rotation_epsilon",
				debug_config.cull_guardrail_rotation_epsilon);
		debug_config.cull_guardrail_drop_ratio = _get_float_setting(ps,
				"rendering/gaussian_splatting/debug/cull_guardrail_drop_ratio",
				debug_config.cull_guardrail_drop_ratio);
		debug_config.cull_guardrail_min_visible = _get_int_setting(ps,
				"rendering/gaussian_splatting/debug/cull_guardrail_min_visible",
				debug_config.cull_guardrail_min_visible);
		const int default_samples = debug_config.splat_audit_sample_count;
		const int configured_samples = _get_int_setting(ps,
				"rendering/gaussian_splatting/debug/splat_audit_sample_count", default_samples);
		debug_config.splat_audit_sample_count = CLAMP(configured_samples, 1,
				static_cast<int>(TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES));
		if (debug_config.enable_all_debug) {
			debug_config.enable_pipeline_trace = true;
			debug_config.enable_state_guardrails = true;
			debug_config.enable_splat_audit = true;
			debug_config.enable_frame_logging = true;
			debug_config.enable_frame_logging_verbose = true;
			debug_config.enable_sort_path_logs = true;
			debug_config.enable_tile_logs = true;
			debug_config.enable_tile_pipeline_logs = true;
			debug_config.enable_tile_dispatch_logs = true;
			debug_config.enable_gpu_counter_logs = true;
			debug_config.enable_binning_counters = true;
			debug_config.enable_cull_counters = true;
			debug_config.enable_cull_guardrails = true;
			debug_config.enable_autotune_logs = true;
			debug_config.enable_data_logging = true;
			debug_config.dump_gpu_counters = true;
			if (debug_config.frame_log_frequency <= 0) {
				debug_config.frame_log_frequency = 1;
			}
		}
	} else {
		debug_config.enable_all_debug = false;
		debug_config.enable_pipeline_trace = default_trace;
		debug_config.enable_state_guardrails = false;
		debug_config.enable_splat_audit = false;
		debug_config.enable_frame_logging = false;
		debug_config.enable_frame_logging_verbose = false;
		debug_config.frame_log_frequency = debug_config.frame_log_frequency > 0 ? debug_config.frame_log_frequency : 300;
		debug_config.enable_sort_path_logs = false;
		debug_config.enable_tile_logs = false;
		debug_config.enable_tile_pipeline_logs = false;
		debug_config.enable_tile_dispatch_logs = false;
		debug_config.enable_gpu_counter_logs = false;
		debug_config.enable_binning_counters = false;
		debug_config.enable_cull_counters = false;
		debug_config.enable_cull_guardrails = false;
		debug_config.enable_autotune_logs = false;
		debug_config.enable_data_logging = false;
		debug_config.splat_audit_sample_count = CLAMP(debug_config.splat_audit_sample_count, 1,
				static_cast<int>(TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES));
	}

	if (debug_config.enable_frame_logging_verbose && !debug_config.enable_all_debug) {
		debug_config.enable_sort_path_logs = true;
		debug_config.enable_tile_logs = true;
		debug_config.enable_tile_pipeline_logs = true;
		debug_config.enable_tile_dispatch_logs = true;
		debug_config.enable_gpu_counter_logs = true;
		debug_config.enable_autotune_logs = true;
	}

	if (debug_config.enable_gpu_counter_logs) {
		debug_config.dump_gpu_counters = true;
	}

	debug_config.cull_guardrail_drop_ratio = CLAMP(debug_config.cull_guardrail_drop_ratio, 0.1f, 0.99f);
	debug_config.cull_guardrail_min_visible = MAX(1, debug_config.cull_guardrail_min_visible);
}

void RenderDebugStateOrchestrator::set_debug_preview_mode(GaussianSplatRenderer::DebugPreviewMode p_mode) {
	if (debug_state.preview_mode == p_mode) {
		return;
	}

	debug_state.preview_mode = p_mode;
}

Dictionary RenderDebugStateOrchestrator::get_binning_debug_counters() const {
	Dictionary out;
	if (!tile_renderer->is_valid()) {
		return out;
	}
	TileRenderer *tr = tile_renderer->ptr();
	TileRenderer::DebugCounterSnapshot counters = tr->get_debug_counters();
	out["total_processed"] = (int64_t)counters.total_processed;
	out["near_far_reject"] = (int64_t)counters.near_far_reject;
	out["view_distance_reject"] = (int64_t)counters.view_distance_reject;
	out["quaternion_reject"] = (int64_t)counters.quaternion_reject;
	out["scale_reject"] = (int64_t)counters.scale_reject;
	out["clip_w_reject"] = (int64_t)counters.clip_w_reject;
	out["clip_bounds_reject"] = (int64_t)counters.clip_bounds_reject;
	out["screen_nan_reject"] = (int64_t)counters.screen_nan_reject;
	out["focal_length_reject"] = (int64_t)counters.focal_length_reject;
	out["z_inverse_reject"] = (int64_t)counters.z_inverse_reject;
	out["covariance_nan_reject"] = (int64_t)counters.covariance_nan_reject;
	out["determinant_reject"] = (int64_t)counters.determinant_reject;
	out["radius_reject"] = (int64_t)counters.radius_reject;
	out["distance_cull_reject"] = (int64_t)counters.distance_cull_reject;
	out["viewport_bounds_reject"] = (int64_t)counters.viewport_bounds_reject;
	out["bbox_integrity_reject"] = (int64_t)counters.bbox_integrity_reject;
	out["tile_extent_reject"] = (int64_t)counters.tile_extent_reject;
	out["success_count"] = (int64_t)counters.success_count;
	out["extreme_conic_count"] = (int64_t)counters.extreme_conic_count;
	out["index_mismatch_count"] = (int64_t)counters.index_mismatch_count;
	// Diagnostic counters for radial stretching investigation
	out["depth_discrepancy_count"] = (int64_t)counters.depth_discrepancy_count;
	out["depth_discrepancy_sum_q8"] = (int64_t)counters.depth_discrepancy_sum_q8;
	out["high_aspect_ratio_count"] = (int64_t)counters.high_aspect_ratio_count;
	out["max_aspect_q8"] = (int64_t)counters.max_aspect_q8;
	// New diagnostic counters
	out["max_aspect_preclamp_q8"] = (int64_t)counters.max_aspect_preclamp_q8;
	out["j_col2_clamp_count"] = (int64_t)counters.j_col2_clamp_count;
	out["sh_cache_hits"] = (int64_t)counters.sh_cache_hits;
	out["sh_cache_updates"] = (int64_t)counters.sh_cache_updates;
	out["sh_cache_forced_updates"] = (int64_t)counters.sh_cache_forced_updates;
	// Subpixel culling diagnostics (Q8.8 fixed-point)
	out["tiny_splat_param_q8"] = (int64_t)counters.tiny_splat_param_q8;
	out["min_allowed_radius_q8"] = (int64_t)counters.min_allowed_radius_q8;
	out["min_radius_min_q8_inv"] = (int64_t)counters.min_radius_min_q8_inv;
	// Computed averages for easier debugging
	if (counters.depth_discrepancy_count > 0) {
		out["avg_depth_discrepancy"] = double(counters.depth_discrepancy_sum_q8) / 256.0 / double(counters.depth_discrepancy_count);
	} else {
		out["avg_depth_discrepancy"] = 0.0;
	}
	out["max_aspect_ratio"] = double(counters.max_aspect_q8) / 256.0;
	out["max_aspect_preclamp"] = double(counters.max_aspect_preclamp_q8) / 256.0;
	out["tiny_splat_param_px"] = double(counters.tiny_splat_param_q8) / 256.0;
	out["min_allowed_radius_px"] = double(counters.min_allowed_radius_q8) / 256.0;
	if (counters.min_radius_min_q8_inv != 0u) {
		const uint32_t min_radius_q8 = 0xFFFFFFFFu - counters.min_radius_min_q8_inv;
		out["min_radius_min_px"] = double(min_radius_q8) / 256.0;
	} else {
		out["min_radius_min_px"] = -1.0;
	}
	const uint32_t sh_cache_total = counters.sh_cache_hits + counters.sh_cache_updates;
	out["sh_cache_hit_rate"] = sh_cache_total > 0
			? (double(counters.sh_cache_hits) / double(sh_cache_total))
			: 0.0;

	TileRenderer::OverflowStatsSnapshot overflow = tr->get_overflow_stats();
	out["overflow_tile_count"] = (int64_t)overflow.overflow_tile_count;
	out["overflow_splats_clamped"] = (int64_t)overflow.overflow_splats_clamped;
	out["overflow_splats_aggregated"] = (int64_t)overflow.overflow_splats_aggregated;
	out["raster_sample_count"] = (int64_t)overflow.raster_sample_count;
	out["raster_splats_iterated"] = (int64_t)overflow.raster_splats_iterated;
	out["raster_splats_contributed"] = (int64_t)overflow.raster_splats_contributed;
	out["raster_reject_sorted_idx_oob"] = (int64_t)overflow.raster_reject_sorted_idx_oob;
	out["raster_reject_gaussian_idx_oob"] = (int64_t)overflow.raster_reject_gaussian_idx_oob;
	out["raster_reject_base_opacity"] = (int64_t)overflow.raster_reject_base_opacity;
	out["raster_reject_nan_inf"] = (int64_t)overflow.raster_reject_nan_inf;
	out["raster_reject_weight"] = (int64_t)overflow.raster_reject_weight;
	out["raster_reject_alpha"] = (int64_t)overflow.raster_reject_alpha;
	out["raster_break_remaining_alpha"] = (int64_t)overflow.raster_break_remaining_alpha;
	out["raster_break_final_alpha"] = (int64_t)overflow.raster_break_final_alpha;
	out["raster_has_depth"] = (int64_t)overflow.raster_has_depth;
	out["raster_alpha_sum_q10"] = (int64_t)overflow.raster_alpha_sum_q10;
	double raster_avg_alpha = 0.0;
	if (overflow.raster_sample_count > 0) {
		raster_avg_alpha = double(overflow.raster_alpha_sum_q10) / (double(overflow.raster_sample_count) * 1024.0);
	}
	out["raster_avg_alpha"] = raster_avg_alpha;

	return out;
}

void RenderDebugStateOrchestrator::store_stage_metrics(const GaussianSplatRenderer::StageMetrics &p_metrics) {
	debug_state.last_stage_metrics = p_metrics;
	debug_state.last_stage_metrics_valid = true;
}

void RenderDebugStateOrchestrator::clear_stage_metrics() {
	debug_state.last_stage_metrics = GaussianSplatRenderer::StageMetrics();
	debug_state.last_stage_metrics_valid = false;
}

void RenderDebugStateOrchestrator::update_frame_times(float p_render_ms, float p_sort_ms) {
	debug_state.last_render_time_ms = p_render_ms;
	debug_state.last_sort_time_ms = p_sort_ms;
}

bool RenderDebugStateOrchestrator::_check_cull_guardrails(uint64_t p_frame_id, uint32_t p_visible_count, String &r_message) {
	if (!debug_config.enable_cull_guardrails || !renderer) {
		return false;
	}

	const float pos_step = debug_config.cull_guardrail_position_epsilon;
	const float rot_step = debug_config.cull_guardrail_rotation_epsilon;
	uint64_t pose_key = _hash_camera_pose(renderer->get_view_state().last_camera_to_world_transform, pos_step, rot_step);
	if (pose_key == 0) {
		return false;
	}

	uint32_t static_visible = 0;
	uint32_t static_total = 0;
	uint32_t gpu_visible = 0;
	uint32_t cpu_visible = 0;
	GPUCuller *gpu_culler = renderer->get_subsystem_state().gpu_culler.ptr();
	if (gpu_culler) {
		const auto &cull_state = gpu_culler->get_state();
		static_visible = static_cast<uint32_t>(cull_state.visible_static_chunk_indices.size());
		static_total = static_cast<uint32_t>(cull_state.static_chunks.size());
		gpu_visible = cull_state.gpu_visible_indices_count;
		cpu_visible = static_cast<uint32_t>(cull_state.culled_indices.size());
	}

	bool streaming_active = false;
	uint32_t streaming_visible = 0;
	const auto &streaming_state = renderer->get_streaming_state();
	if (streaming_state.current_streaming_system.is_valid()) {
		streaming_active = true;
		streaming_visible = streaming_state.current_streaming_system->get_visible_count();
	}

	CullGuardrailSample *match = nullptr;
	for (uint32_t i = 0; i < kCullGuardrailSamples; i++) {
		if (cull_guardrail_samples[i].valid && cull_guardrail_samples[i].key == pose_key) {
			match = &cull_guardrail_samples[i];
			break;
		}
	}

	bool triggered = false;
	if (match && match->visible_count >= static_cast<uint32_t>(debug_config.cull_guardrail_min_visible)) {
		float ratio = match->visible_count > 0
				? float(p_visible_count) / float(match->visible_count)
				: 1.0f;
		if (ratio < debug_config.cull_guardrail_drop_ratio) {
			triggered = true;
			r_message = vformat(" cull_guardrail_drop=%u->%u ratio=%.3f static_chunks=%u/%u->%u/%u stream_visible=%u->%u gpu_visible=%u->%u cpu_visible=%u->%u",
					match->visible_count, p_visible_count, ratio,
					match->static_visible_chunks, match->static_chunk_total,
					static_visible, static_total,
					match->streaming_visible, streaming_visible,
					match->gpu_visible, gpu_visible,
					match->cpu_visible, cpu_visible);
		}
	}

	CullGuardrailSample &slot = match ? *match : cull_guardrail_samples[cull_guardrail_cursor++ % kCullGuardrailSamples];
	slot.valid = true;
	slot.key = pose_key;
	slot.visible_count = p_visible_count;
	slot.static_visible_chunks = static_visible;
	slot.static_chunk_total = static_total;
	slot.streaming_visible = streaming_visible;
	slot.streaming_active = streaming_active;
	slot.gpu_visible = gpu_visible;
	slot.cpu_visible = cpu_visible;
	slot.frame_id = static_cast<uint32_t>(p_frame_id);

	return triggered;
}

void RenderDebugStateOrchestrator::reset_debug_overlay_metrics(float p_sort_ms) {
	debug_state.last_render_time_ms = 0.0f;
	debug_state.last_sort_time_ms = p_sort_ms;
	debug_state.tile_density_cache.clear();
	debug_state.tile_density_width = 0;
	debug_state.tile_density_height = 0;
	debug_state.tile_density_peak = 0;
	debug_state.tile_density_average = 0.0f;
	debug_state.last_tile_assignment_ms = 0.0f;
	debug_state.last_tile_rasterization_ms = 0.0f;
	clear_overlay_dirty_flags();
}

void RenderDebugStateOrchestrator::update_raster_metrics(const RasterPerformance &p_perf, const RasterStats &p_stats) {
	debug_state.last_tile_assignment_ms = p_perf.tile_assignment_ms;
	debug_state.last_tile_rasterization_ms = p_perf.rasterization_ms;
	debug_state.tile_density_peak = p_stats.max_splats_in_tile;
	debug_state.tile_density_average = p_stats.average_splats_per_tile;

	uint64_t frame_id = renderer->get_frame_state().frame_counter;
	uint32_t visible_count = renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
	const bool trace_enabled = debug_config.enable_pipeline_trace;
	const bool audit_enabled = debug_config.enable_splat_audit;
	if (!trace_enabled && !audit_enabled) {
		last_visible_count = visible_count;
		has_prev_visible = true;
		has_prev_contrib = false;
		debug_state.splat_audit.valid = false;
		return;
	}

	uint32_t contrib_count = 0;
	uint32_t idx_mismatch = 0;
	bool has_overflow_stats = false;
	GaussianSplatRenderer::SplatAuditSummary audit_summary;
	audit_summary.valid = false;
	if (tile_renderer && tile_renderer->is_valid()) {
		TileRenderer *tr = tile_renderer->ptr();
		const TileRenderer::DebugCounterSnapshot counters = tr->get_debug_counters();
		idx_mismatch = counters.index_mismatch_count;
		const TileRenderer::OverflowStatsSnapshot overflow = tr->get_overflow_stats();
		contrib_count = overflow.raster_splats_contributed;
		has_overflow_stats = overflow.raster_sample_count > 0 ||
				overflow.raster_has_depth > 0 ||
				overflow.raster_splats_contributed > 0;
		if (audit_enabled) {
			const TileRenderer::SplatAuditSnapshot audit = tr->get_splat_audit_snapshot();
			if (audit.valid) {
				audit_summary.valid = true;
				audit_summary.sample_count = audit.sample_count;
				audit_summary.projected_count = audit.projected_count;
				audit_summary.in_viewport_count = audit.in_viewport_count;
				audit_summary.iterated_count = audit.iterated_count;
				audit_summary.contributed_count = audit.contributed_count;
				audit_summary.alpha_skipped_count = audit.alpha_skipped_count;
				audit_summary.missing_iterated_count = audit.missing_iterated_count;
				audit_summary.missing_contrib_count = audit.missing_contrib_count;
				audit_summary.first_mismatch_global_idx = audit.first_mismatch_global_idx;
				audit_summary.first_mismatch_expected_x = audit.first_mismatch_expected_x;
				audit_summary.first_mismatch_expected_y = audit.first_mismatch_expected_y;
				audit_summary.first_mismatch_flags = audit.first_mismatch_flags;
			}
		}
	}
	debug_state.splat_audit = audit_summary;

	const bool visible_drop = has_prev_visible &&
			_is_drop(last_visible_count, visible_count, kAnomalyDropMinCount, kAnomalyDropDivisor);
	const bool contrib_drop = has_prev_contrib && has_overflow_stats &&
			_is_drop(last_contrib_count, contrib_count, kAnomalyDropMinCount, kAnomalyDropDivisor);
	const bool audit_missing_iterated = audit_summary.valid && audit_summary.missing_iterated_count > 0;
	const bool audit_missing_contrib = audit_summary.valid && audit_summary.missing_contrib_count > 0;
	String guardrail_message;
	const bool cull_guardrail_drop = _check_cull_guardrails(frame_id, visible_count, guardrail_message);
	const bool anomaly_detected = (idx_mismatch > 0 || visible_drop || contrib_drop ||
			audit_missing_iterated || audit_missing_contrib || cull_guardrail_drop);

	if (anomaly_detected) {
		String message = "anomaly";
		message += " idx_mismatch=" + String::num_uint64(idx_mismatch);
		if (visible_drop) {
			message += " visible_drop=" + String::num_uint64(last_visible_count) + "->" + String::num_uint64(visible_count);
		}
		if (contrib_drop) {
			message += " contrib_drop=" + String::num_uint64(last_contrib_count) + "->" + String::num_uint64(contrib_count);
		}
		if (audit_missing_iterated) {
			message += " audit_missing_iterated=" + String::num_uint64(audit_summary.missing_iterated_count);
		}
		if (audit_missing_contrib) {
			message += " audit_missing_contrib=" + String::num_uint64(audit_summary.missing_contrib_count);
		}
		if (cull_guardrail_drop) {
			message += guardrail_message;
		}
		if (audit_summary.valid && audit_summary.first_mismatch_flags != 0u) {
			message += " audit_first_idx=" + String::num_uint64(audit_summary.first_mismatch_global_idx);
			message += " audit_first_px=" + String::num_uint64(audit_summary.first_mismatch_expected_x) + "," +
					String::num_uint64(audit_summary.first_mismatch_expected_y);
		}
		_record_debug_pipeline_event(renderer, message, last_visible_count, visible_count, true);

		const bool dump_ready = !has_anomaly_dump_frame ||
				frame_id >= last_anomaly_dump_frame + kAnomalyDumpCooldownFrames;
		// Artifact writes are opt-in to keep the runtime render path side-effect free by default.
		const bool artifact_writes_enabled = debug_config.enable_data_logging || debug_config.enable_all_debug;
		if (dump_ready && artifact_writes_enabled) {
			const String trace_path = "user://gs_pipeline_trace_" + String::num_uint64(frame_id) + ".json";
			renderer->dump_pipeline_trace_to_file(trace_path);

			OutputCompositor *output_compositor = renderer->get_subsystem_state().output_compositor.ptr();
			if (output_compositor) {
				RID final_texture = output_compositor->get_final_render_texture();
				if (final_texture.is_valid()) {
					RenderingDevice *device = renderer->get_resource_owner(final_texture, renderer->get_device_state().rd);
					const String image_path = "user://gs_anomaly_frame_" + String::num_uint64(frame_id) + ".png";
					_save_texture_snapshot(device, final_texture, image_path);
				}
			}

			has_anomaly_dump_frame = true;
			last_anomaly_dump_frame = frame_id;
		}
	}

	last_visible_count = visible_count;
	has_prev_visible = true;
	if (has_overflow_stats) {
		last_contrib_count = contrib_count;
		has_prev_contrib = true;
	} else {
		has_prev_contrib = false;
	}
}

void RenderDebugStateOrchestrator::clear_overlay_dirty_flags() {
	debug_state.overlay_dirty = false;
	debug_state.hud_dirty = false;
}

void RenderDebugStateOrchestrator::apply_debug_options_to_render_params(TileRenderer::RenderParams &r_params) const {
	// Phase 8: Centralized debug options setup for render params
	// Used by painterly G-buffer population and _render_tile_fallback
	if (tile_renderer->is_valid()) {
		tile_renderer->ptr()->set_debug_binning_counters_enabled(debug_config.enable_binning_counters);
	}
	const bool depth_visualization = (debug_state.preview_mode == GaussianSplatRenderer::DEBUG_PREVIEW_DEPTH);
	if (debug_overlay_system->is_valid()) {
		DebugOverlaySystem *overlay = debug_overlay_system->ptr();
		const DebugOverlayQueryView overlay_view = overlay->build_query_view(renderer);
		const DebugOverlayOptions overlay_options = overlay_view.get_options();
		r_params.debug_show_tile_bounds = overlay_options.show_tile_bounds;
		r_params.debug_show_splat_coverage = overlay_options.show_splat_coverage;
		r_params.debug_show_overflow_tiles = overlay_options.show_overflow_tiles;
		r_params.debug_show_projection_issues = overlay_options.show_projection_issues;
		r_params.debug_show_white_albedo = overlay_options.show_white_albedo;
		r_params.debug_show_shadow_opacity = overlay_options.show_shadow_opacity;
		r_params.debug_show_tile_grid = overlay_options.show_tile_grid;
		r_params.debug_show_density_heatmap = overlay_options.show_density_heatmap;
		r_params.debug_show_performance_hud = overlay_options.show_performance_hud;
		r_params.debug_show_depth_visualization = depth_visualization;
		r_params.debug_overlay_opacity = overlay_options.overlay_opacity;
		r_params.debug_dump_gpu_counters = overlay_options.dump_gpu_counters || debug_config.enable_gpu_counter_logs;
		r_params.debug_enable_tile_logs = debug_config.enable_tile_logs;
		r_params.debug_enable_tile_pipeline_logs = debug_config.enable_tile_pipeline_logs;
		r_params.debug_enable_tile_dispatch_logs = debug_config.enable_tile_dispatch_logs;
		r_params.debug_enable_gpu_counter_logs = debug_config.enable_gpu_counter_logs;
		r_params.debug_frame_log_frequency = debug_config.frame_log_frequency;
		r_params.debug_enable_splat_audit = debug_config.enable_splat_audit;
		r_params.debug_splat_audit_sample_count = debug_config.splat_audit_sample_count;
	} else {
		r_params.debug_show_tile_bounds = debug_state.show_tile_bounds;
		r_params.debug_show_splat_coverage = debug_state.show_splat_coverage;
		r_params.debug_show_overflow_tiles = debug_state.show_overflow_tiles;
		r_params.debug_show_projection_issues = debug_state.show_projection_issues;
		r_params.debug_show_white_albedo = debug_state.show_white_albedo;
		r_params.debug_show_shadow_opacity = debug_state.show_shadow_opacity;
		r_params.debug_show_tile_grid = debug_state.show_tile_grid;
		r_params.debug_show_density_heatmap = debug_state.show_density_heatmap;
		r_params.debug_show_performance_hud = debug_state.show_performance_hud;
		r_params.debug_show_depth_visualization = depth_visualization;
		r_params.debug_overlay_opacity = debug_config.overlay_opacity;
		r_params.debug_dump_gpu_counters = debug_config.dump_gpu_counters || debug_config.enable_gpu_counter_logs;
		r_params.debug_enable_tile_logs = debug_config.enable_tile_logs;
		r_params.debug_enable_tile_pipeline_logs = debug_config.enable_tile_pipeline_logs;
		r_params.debug_enable_tile_dispatch_logs = debug_config.enable_tile_dispatch_logs;
		r_params.debug_enable_gpu_counter_logs = debug_config.enable_gpu_counter_logs;
		r_params.debug_frame_log_frequency = debug_config.frame_log_frequency;
		r_params.debug_enable_splat_audit = debug_config.enable_splat_audit;
		r_params.debug_splat_audit_sample_count = debug_config.splat_audit_sample_count;
	}

	if (debug_config.compute_raster_policy_override != GaussianSplatting::ComputeRasterPolicy::Default) {
		r_params.compute_raster_policy = debug_config.compute_raster_policy_override;
	}
}

// Orchestrator setters - delegate to DebugOverlaySystem if available, otherwise update debug state directly.
void RenderDebugStateOrchestrator::set_debug_show_tile_grid(bool p_enabled) {
	if (debug_overlay_system->is_valid()) {
		debug_overlay_system->ptr()->build_command_sink(renderer).set_show_tile_grid(p_enabled);
		return;
	}
	if (debug_state.show_tile_grid == p_enabled) {
		return;
	}
	debug_state.show_tile_grid = p_enabled;
}

void RenderDebugStateOrchestrator::set_debug_show_density_heatmap(bool p_enabled) {
	if (debug_overlay_system->is_valid()) {
		debug_overlay_system->ptr()->build_command_sink(renderer).set_show_density_heatmap(p_enabled);
		return;
	}
	if (debug_state.show_density_heatmap == p_enabled) {
		return;
	}
	debug_state.show_density_heatmap = p_enabled;
}

void RenderDebugStateOrchestrator::set_debug_show_shadow_opacity(bool p_enabled) {
	if (debug_overlay_system->is_valid()) {
		debug_overlay_system->ptr()->build_command_sink(renderer).set_show_shadow_opacity(p_enabled);
		return;
	}
	if (debug_state.show_shadow_opacity == p_enabled) {
		return;
	}
	debug_state.show_shadow_opacity = p_enabled;
}

void RenderDebugStateOrchestrator::set_debug_show_performance_hud(bool p_enabled) {
	if (debug_overlay_system->is_valid()) {
		debug_overlay_system->ptr()->build_command_sink(renderer).set_show_performance_hud(p_enabled);
		return;
	}
	if (debug_state.show_performance_hud == p_enabled) {
		return;
	}
	debug_state.show_performance_hud = p_enabled;
}

void RenderDebugStateOrchestrator::set_debug_show_residency_hud(bool p_enabled) {
	if (debug_overlay_system->is_valid()) {
		debug_overlay_system->ptr()->build_command_sink(renderer).set_show_residency_hud(p_enabled);
		return;
	}
	if (debug_state.show_residency_hud == p_enabled) {
		return;
	}
	debug_state.show_residency_hud = p_enabled;
}

void RenderDebugStateOrchestrator::set_debug_show_device_boundaries(bool p_enabled) {
	if (debug_overlay_system->is_valid()) {
		debug_overlay_system->ptr()->build_command_sink(renderer).set_show_device_boundaries(p_enabled);
		return;
	}
	if (debug_state.show_device_boundaries == p_enabled) {
		return;
	}
	debug_state.show_device_boundaries = p_enabled;
}

void RenderDebugStateOrchestrator::set_debug_show_texture_states(bool p_enabled) {
	if (debug_overlay_system->is_valid()) {
		debug_overlay_system->ptr()->build_command_sink(renderer).set_show_texture_states(p_enabled);
		return;
	}
	if (debug_state.show_texture_states == p_enabled) {
		return;
	}
	debug_state.show_texture_states = p_enabled;
}

void RenderDebugStateOrchestrator::set_debug_overlay_opacity(float p_opacity) {
	float clamped = CLAMP(p_opacity, 0.0f, 1.0f);
	if (debug_overlay_system->is_valid()) {
		debug_overlay_system->ptr()->build_command_sink(renderer).set_overlay_opacity(clamped);
		return;
	}

	if (Math::is_equal_approx(debug_config.overlay_opacity, clamped)) {
		return;
	}

	debug_config.overlay_opacity = clamped;
}

void RenderDebugStateOrchestrator::set_debug_compute_raster_policy(int p_policy) {
	int clamped = CLAMP(p_policy, 0, 2);
	debug_config.compute_raster_policy_override =
			static_cast<GaussianSplatting::ComputeRasterPolicy>(clamped);
}

int RenderDebugStateOrchestrator::get_debug_compute_raster_policy() const {
	return static_cast<int>(debug_config.compute_raster_policy_override);
}

void RenderDebugStateOrchestrator::set_debug_dump_gpu_counters(bool p_enabled) {
	debug_config.dump_gpu_counters = p_enabled;
	if (debug_overlay_system->is_valid()) {
		debug_overlay_system->ptr()->build_command_sink(renderer).set_dump_gpu_counters(p_enabled);
	}
}

bool RenderDebugStateOrchestrator::get_debug_dump_gpu_counters() const {
	if (debug_overlay_system->is_valid()) {
		return debug_overlay_system->ptr()->build_query_view(renderer).get_options().dump_gpu_counters;
	}
	return debug_config.dump_gpu_counters;
}

void RenderDebugStateOrchestrator::set_debug_binning_counters_enabled(bool p_enabled) {
	debug_config.enable_binning_counters = p_enabled;
}

bool RenderDebugStateOrchestrator::get_debug_binning_counters_enabled() const {
	return debug_config.enable_binning_counters;
}

void RenderDebugStateOrchestrator::set_debug_pipeline_trace_enabled(bool p_enabled) {
	debug_config.enable_pipeline_trace = p_enabled;
}

bool RenderDebugStateOrchestrator::get_debug_pipeline_trace_enabled() const {
	return debug_config.enable_pipeline_trace;
}

void RenderDebugStateOrchestrator::set_debug_state_guardrails_enabled(bool p_enabled) {
	debug_config.enable_state_guardrails = p_enabled;
}

bool RenderDebugStateOrchestrator::get_debug_state_guardrails_enabled() const {
	return debug_config.enable_state_guardrails;
}

void RenderDebugStateOrchestrator::set_debug_cull_guardrails_enabled(bool p_enabled) {
	debug_config.enable_cull_guardrails = p_enabled;
}

bool RenderDebugStateOrchestrator::get_debug_cull_guardrails_enabled() const {
	return debug_config.enable_cull_guardrails;
}

void RenderDebugStateOrchestrator::set_debug_splat_audit_enabled(bool p_enabled) {
	debug_config.enable_splat_audit = p_enabled;
}

bool RenderDebugStateOrchestrator::get_debug_splat_audit_enabled() const {
	return debug_config.enable_splat_audit;
}

void RenderDebugStateOrchestrator::set_debug_splat_audit_sample_count(int p_count) {
	debug_config.splat_audit_sample_count = CLAMP(p_count, 1,
			static_cast<int>(TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES));
}

int RenderDebugStateOrchestrator::get_debug_splat_audit_sample_count() const {
	return debug_config.splat_audit_sample_count;
}

void RenderDebugStateOrchestrator::set_jacobian_bypass_radius_depth_floor(bool p_enabled) {
	jacobian_debug->bypass_radius_depth_floor = p_enabled;
}

void RenderDebugStateOrchestrator::set_jacobian_bypass_j_col2_clamp(bool p_enabled) {
	jacobian_debug->bypass_j_col2_clamp = p_enabled;
}

void RenderDebugStateOrchestrator::set_jacobian_invert_j_col2_sign(bool p_enabled) {
	jacobian_debug->invert_j_col2_sign = p_enabled;
}

void RenderDebugStateOrchestrator::set_max_conic_aspect(float p_aspect) {
	jacobian_debug->max_conic_aspect = CLAMP(p_aspect, 1.0f, 100.0f);
}

void GaussianSplatRenderer::set_debug_preview_mode(DebugPreviewMode p_mode) {
	debug_state_orchestrator->set_debug_preview_mode(p_mode);
}

Dictionary GaussianSplatRenderer::get_binning_debug_counters() const {
	return debug_state_orchestrator->get_binning_debug_counters();
}

int GaussianSplatRenderer::get_overflow_tile_count() const {
	if (!tile_renderer_state.renderer.is_valid()) {
		return 0;
	}
	return tile_renderer_state.renderer->get_overflow_tile_count();
}

int GaussianSplatRenderer::get_clamped_records() const {
	if (!tile_renderer_state.renderer.is_valid()) {
		return 0;
	}
	return tile_renderer_state.renderer->get_clamped_records();
}

int GaussianSplatRenderer::get_aggregated_count() const {
	if (!tile_renderer_state.renderer.is_valid()) {
		return 0;
	}
	return tile_renderer_state.renderer->get_aggregated_count();
}

Dictionary GaussianSplatRenderer::get_overflow_stats() const {
	Dictionary out;
	if (!tile_renderer_state.renderer.is_valid()) {
		return out;
	}

	const TileRenderer::OverflowStatsSnapshot overflow = tile_renderer_state.renderer->get_overflow_stats();
	out["overflow_tile_count"] = static_cast<int64_t>(overflow.overflow_tile_count);
	out["clamped_records"] = static_cast<int64_t>(overflow.overflow_splats_clamped);
	out["aggregated_records"] = static_cast<int64_t>(overflow.overflow_splats_aggregated);
	out["raster_sample_count"] = static_cast<int64_t>(overflow.raster_sample_count);
	out["raster_splats_iterated"] = static_cast<int64_t>(overflow.raster_splats_iterated);
	out["raster_splats_contributed"] = static_cast<int64_t>(overflow.raster_splats_contributed);
	return out;
}

Dictionary GaussianSplatRenderer::get_pipeline_trace_snapshot() const {
	const DebugState &debug_state = debug_state_orchestrator->get_state();
	const DebugConfig &debug_config = debug_state_orchestrator->get_config();

	Dictionary snapshot;
	const uint64_t frame_id = debug_state.pipeline_events_valid
			? debug_state.pipeline_events_frame
			: get_frame_state().frame_counter;
	const bool trace_fresh = debug_state.pipeline_events_valid;
	String route_uid = debug_state.route_uid;
	if (trace_fresh && RenderRouteUID::is_route_uid_missing(route_uid) && !debug_state.pipeline_events.is_empty()) {
		for (int i = debug_state.pipeline_events.size() - 1; i >= 0; --i) {
			const String &event_uid = debug_state.pipeline_events[i].route_uid;
			if (!RenderRouteUID::is_route_uid_missing(event_uid)) {
				route_uid = event_uid;
				break;
			}
		}
	}
	route_uid = _normalize_route_uid_for_snapshot(route_uid);
	snapshot["frame"] = static_cast<int64_t>(frame_id);
	snapshot["dump_frame"] = static_cast<int64_t>(get_frame_state().frame_counter);
	snapshot["trace_enabled"] = debug_config.enable_pipeline_trace;
	snapshot["events_valid"] = debug_state.pipeline_events_valid;
	snapshot["trace_fresh"] = trace_fresh;
	snapshot["trace_generation"] = static_cast<int64_t>(debug_state.pipeline_trace_generation);
	snapshot["route_uid"] = route_uid;
	snapshot["sort_route_uid"] = _normalize_sort_route_uid_for_snapshot(debug_state.sort_route_uid);

	Array events;
	if (trace_fresh) {
		events.resize(debug_state.pipeline_events.size());
		for (int i = 0; i < debug_state.pipeline_events.size(); i++) {
			events[i] = _pipeline_event_to_dict(debug_state.pipeline_events[i]);
		}
	}
	snapshot["events"] = events;

	snapshot["stage_metrics_valid"] = debug_state.last_stage_metrics_valid;
	if (debug_state.last_stage_metrics_valid) {
		Dictionary stage_results;
		stage_results["cull"] = _stage_result_to_dict(debug_state.last_stage_metrics.cull_result);
		stage_results["sort"] = _stage_result_to_dict(debug_state.last_stage_metrics.sort_result);
		stage_results["raster"] = _stage_result_to_dict(debug_state.last_stage_metrics.raster_result);
		stage_results["composite"] = _stage_result_to_dict(debug_state.last_stage_metrics.composite_result);
		snapshot["stage_results"] = stage_results;

		Dictionary stage_metrics;
		stage_metrics["cull"] = _cull_output_to_dict(debug_state.last_stage_metrics.cull);
		stage_metrics["sort"] = _sort_output_to_dict(debug_state.last_stage_metrics.sort);
		stage_metrics["raster"] = _raster_output_to_dict(debug_state.last_stage_metrics.raster);
		Dictionary composite_metrics;
		composite_metrics["executed"] = debug_state.last_stage_metrics.composite_executed;
		composite_metrics["time_ms"] = debug_state.last_stage_metrics.composite_time_ms;
		stage_metrics["composite"] = composite_metrics;
		snapshot["stage_metrics"] = stage_metrics;

		Dictionary stage_io;
		stage_io["cull"] = _stage_io_to_dict(debug_state.last_stage_metrics.cull_io);
		stage_io["sort"] = _stage_io_to_dict(debug_state.last_stage_metrics.sort_io);
		stage_io["raster"] = _stage_io_to_dict(debug_state.last_stage_metrics.raster_io);
		stage_io["composite"] = _stage_io_to_dict(debug_state.last_stage_metrics.composite_io);
		snapshot["stage_io"] = stage_io;
	}

	if (debug_state.splat_audit.valid || debug_config.enable_splat_audit) {
		snapshot["splat_audit"] = _splat_audit_to_dict(debug_state.splat_audit);
	}

	if (debug_config.enable_pipeline_trace) {
		Dictionary data_flow = GaussianSplatting::debug_trace_get_data_flow_snapshot();
		if (!data_flow.is_empty()) {
			snapshot["data_flow"] = data_flow;
		}
		Array debug_events = GaussianSplatting::debug_trace_get_recent_events();
		if (!debug_events.is_empty()) {
			snapshot["debug_events"] = debug_events;
		}
	}

	return snapshot;
}

String GaussianSplatRenderer::get_pipeline_trace_json() const {
	return JSON::stringify(get_pipeline_trace_snapshot(), String::utf8("  "));
}

Error GaussianSplatRenderer::dump_pipeline_trace_to_file(const String &p_path) const {
	if (p_path.is_empty()) {
		return ERR_INVALID_PARAMETER;
	}
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
	if (!file.is_valid()) {
		return FileAccess::get_open_error();
	}
	file->store_string(get_pipeline_trace_json());
	return OK;
}

void GaussianSplatRenderer::update_debug_raster_metrics(const RasterPerformance &p_perf, const RasterStats &p_stats) {
	ERR_FAIL_NULL(debug_state_orchestrator);
	debug_state_orchestrator->update_raster_metrics(p_perf, p_stats);
}

void GaussianSplatRenderer::clear_debug_overlay_dirty_flags() {
	ERR_FAIL_NULL(debug_state_orchestrator);
	debug_state_orchestrator->clear_overlay_dirty_flags();
}

void GaussianSplatRenderer::apply_debug_options_to_render_params(TileRenderer::RenderParams &r_params) const {
	ERR_FAIL_NULL(debug_state_orchestrator);
	debug_state_orchestrator->apply_debug_options_to_render_params(r_params);
}

void GaussianSplatRenderer::set_debug_show_tile_grid(bool p_enabled) {
	debug_state_orchestrator->set_debug_show_tile_grid(p_enabled);
}

void GaussianSplatRenderer::set_debug_show_density_heatmap(bool p_enabled) {
	debug_state_orchestrator->set_debug_show_density_heatmap(p_enabled);
}

void GaussianSplatRenderer::set_debug_show_performance_hud(bool p_enabled) {
	debug_state_orchestrator->set_debug_show_performance_hud(p_enabled);
}

void GaussianSplatRenderer::set_debug_show_residency_hud(bool p_enabled) {
	debug_state_orchestrator->set_debug_show_residency_hud(p_enabled);
}

void GaussianSplatRenderer::set_debug_show_device_boundaries(bool p_enabled) {
	debug_state_orchestrator->set_debug_show_device_boundaries(p_enabled);
}

void GaussianSplatRenderer::set_debug_show_texture_states(bool p_enabled) {
	debug_state_orchestrator->set_debug_show_texture_states(p_enabled);
}

void GaussianSplatRenderer::set_debug_compute_raster_policy(int p_policy) {
	debug_state_orchestrator->set_debug_compute_raster_policy(p_policy);
}

int GaussianSplatRenderer::get_debug_compute_raster_policy() const {
	return debug_state_orchestrator->get_debug_compute_raster_policy();
}

void GaussianSplatRenderer::set_debug_overlay_opacity(float p_opacity) {
	debug_state_orchestrator->set_debug_overlay_opacity(p_opacity);
}

void GaussianSplatRenderer::set_debug_dump_gpu_counters(bool p_enabled) {
	debug_state_orchestrator->set_debug_dump_gpu_counters(p_enabled);
}

bool GaussianSplatRenderer::get_debug_dump_gpu_counters() const {
	return debug_state_orchestrator->get_debug_dump_gpu_counters();
}

void GaussianSplatRenderer::set_debug_binning_counters_enabled(bool p_enabled) {
	debug_state_orchestrator->set_debug_binning_counters_enabled(p_enabled);
}

bool GaussianSplatRenderer::get_debug_binning_counters_enabled() const {
	return debug_state_orchestrator->get_debug_binning_counters_enabled();
}

void GaussianSplatRenderer::set_debug_pipeline_trace_enabled(bool p_enabled) {
	debug_state_orchestrator->set_debug_pipeline_trace_enabled(p_enabled);
}

bool GaussianSplatRenderer::get_debug_pipeline_trace_enabled() const {
	return debug_state_orchestrator->get_debug_pipeline_trace_enabled();
}

void GaussianSplatRenderer::set_debug_state_guardrails_enabled(bool p_enabled) {
	debug_state_orchestrator->set_debug_state_guardrails_enabled(p_enabled);
}

bool GaussianSplatRenderer::get_debug_state_guardrails_enabled() const {
	return debug_state_orchestrator->get_debug_state_guardrails_enabled();
}

void GaussianSplatRenderer::set_debug_cull_guardrails_enabled(bool p_enabled) {
	debug_state_orchestrator->set_debug_cull_guardrails_enabled(p_enabled);
}

bool GaussianSplatRenderer::get_debug_cull_guardrails_enabled() const {
	return debug_state_orchestrator->get_debug_cull_guardrails_enabled();
}

void GaussianSplatRenderer::set_debug_splat_audit_enabled(bool p_enabled) {
	debug_state_orchestrator->set_debug_splat_audit_enabled(p_enabled);
}

bool GaussianSplatRenderer::get_debug_splat_audit_enabled() const {
	return debug_state_orchestrator->get_debug_splat_audit_enabled();
}

void GaussianSplatRenderer::set_debug_splat_audit_sample_count(int p_count) {
	debug_state_orchestrator->set_debug_splat_audit_sample_count(p_count);
}

int GaussianSplatRenderer::get_debug_splat_audit_sample_count() const {
	return debug_state_orchestrator->get_debug_splat_audit_sample_count();
}

void GaussianSplatRenderer::reload_pipeline_feature_set() {
	g_pipeline_feature_set.load_from_project_settings();
	pipeline_features_warning_cache = String();
	update_pipeline_features(get_device_state().rd);
}

void GaussianSplatRenderer::set_jacobian_bypass_radius_depth_floor(bool p_enabled) {
	debug_state_orchestrator->set_jacobian_bypass_radius_depth_floor(p_enabled);
}

void GaussianSplatRenderer::set_jacobian_bypass_j_col2_clamp(bool p_enabled) {
	debug_state_orchestrator->set_jacobian_bypass_j_col2_clamp(p_enabled);
}

void GaussianSplatRenderer::set_jacobian_invert_j_col2_sign(bool p_enabled) {
	debug_state_orchestrator->set_jacobian_invert_j_col2_sign(p_enabled);
}

void GaussianSplatRenderer::set_max_conic_aspect(float p_aspect) {
	debug_state_orchestrator->set_max_conic_aspect(p_aspect);
}
