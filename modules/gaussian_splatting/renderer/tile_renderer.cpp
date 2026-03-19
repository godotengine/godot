#include "tile_renderer.h"

#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/math_funcs.h"
#include "core/object/class_db.h"
#include "core/object/callable_method_pointer.h"
#include "core/config/project_settings.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include "core/templates/hash_map.h"
#include "core/templates/hashfuncs.h"

#include "gaussian_gpu_layout.h"
#include "gpu_debug_utils.h"
#include "gpu_performance_monitor.h"
#include "gpu_sorting_config.h"
#include "instance_pipeline_contract.h"
#include "quantization_config.h"
#include "resource_owner_mismatch_contract.h"
#include "sh_config.h"
#include "shader_compilation_helper.h"
#include "../logger/gs_logger.h"
#include "../interfaces/render_device_manager.h"
#include "../interfaces/sync_policy.h"
#include "../core/performance_monitors.h"
#include "../shaders/tile_binning.glsl.gen.h"
#include "../shaders/tile_prefix_scan.glsl.gen.h"
#include "../shaders/tile_rasterizer.glsl.gen.h"
#include "../shaders/tile_rasterizer_compute.glsl.gen.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
#define VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT 0x00000080
#endif
#ifndef VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
#define VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT 0x00000800
#endif

namespace {

// RAII guard to ensure render() function cleans up state on all exit paths.
// This prevents "poisoned" renderer state from early returns leaving dirty counters,
// orphaned timestamps, or stale flags that affect subsequent frames.
class RenderScopeGuard {
public:
    RenderScopeGuard() = default;

    ~RenderScopeGuard() {
        // Emit end timestamp if begin was captured (ensures paired markers)
        if (timestamp_device && !timestamp_label.is_empty()) {
            timestamp_device->capture_timestamp(timestamp_label + String("_End"));
        }
    }

    // Call after successfully capturing begin timestamp
    void set_timestamp_context(RenderingDevice *p_device, const String &p_label) {
        timestamp_device = p_device;
        timestamp_label = p_label;
    }

    // Call when render completes successfully to prevent cleanup actions
    // (the normal path already emits end timestamp)
    void mark_success() {
        // Clear timestamp context so destructor doesn't emit duplicate _End
        timestamp_device = nullptr;
        timestamp_label = String();
    }

private:
    RenderingDevice *timestamp_device = nullptr;
    String timestamp_label;
};

bool _resolve_compute_raster_policy(GaussianSplatting::ComputeRasterPolicy p_policy) {
	switch (p_policy) {
		case GaussianSplatting::ComputeRasterPolicy::Default:
			return g_gpu_sorting_config.enable_compute_raster;
		case GaussianSplatting::ComputeRasterPolicy::ForceOn:
			return true;
		case GaussianSplatting::ComputeRasterPolicy::ForceOff:
			return false;
	}
	return g_gpu_sorting_config.enable_compute_raster;
}

bool _is_srgb_format(RD::DataFormat p_format) {
	switch (p_format) {
		case RD::DATA_FORMAT_R8G8B8A8_SRGB:
		case RD::DATA_FORMAT_B8G8R8A8_SRGB:
		case RD::DATA_FORMAT_A8B8G8R8_SRGB_PACK32:
			return true;
		default:
			return false;
	}
}

RD::DataFormat _resolve_storage_compatible_color_format(RD::DataFormat p_format) {
	switch (p_format) {
		case RD::DATA_FORMAT_R8G8B8A8_UNORM:
		case RD::DATA_FORMAT_R16G16B16A16_SFLOAT:
		case RD::DATA_FORMAT_R32G32B32A32_SFLOAT:
			return p_format;
		case RD::DATA_FORMAT_R8G8B8A8_SRGB:
		case RD::DATA_FORMAT_B8G8R8A8_UNORM:
		case RD::DATA_FORMAT_B8G8R8A8_SRGB:
		case RD::DATA_FORMAT_A8B8G8R8_UNORM_PACK32:
		case RD::DATA_FORMAT_A8B8G8R8_SRGB_PACK32:
			return RD::DATA_FORMAT_R8G8B8A8_UNORM;
		default:
			return RD::DATA_FORMAT_R8G8B8A8_UNORM;
	}
}

uint64_t _hash_shader_defines(const Vector<String> &p_defines) {
	String combined;
	for (int i = 0; i < p_defines.size(); i++) {
		combined += p_defines[i];
	}
	return combined.hash64();
}

static constexpr uint32_t ADAPTIVE_OVERLAP_BUDGET_RECENT_RAW_WINDOW = 16u;

struct AdaptiveOverlapBudgetRuntimeState {
	uint32_t suggested_budget_records = 0u;
	uint32_t low_utilization_frames = 0u;
	uint32_t recent_raw_usage_records[ADAPTIVE_OVERLAP_BUDGET_RECENT_RAW_WINDOW] = {};
	uint32_t recent_raw_usage_count = 0u;
	uint32_t recent_raw_usage_write_index = 0u;
	uint32_t recent_raw_usage_peak = 0u;
};

static HashMap<uint64_t, AdaptiveOverlapBudgetRuntimeState> adaptive_overlap_budget_runtime_states;

static uint64_t _adaptive_overlap_budget_key(const TileRenderer *p_renderer) {
	return uint64_t(reinterpret_cast<uintptr_t>(p_renderer));
}

static AdaptiveOverlapBudgetRuntimeState &_get_adaptive_overlap_budget_state(const TileRenderer *p_renderer) {
	const uint64_t key = _adaptive_overlap_budget_key(p_renderer);
	AdaptiveOverlapBudgetRuntimeState *state = adaptive_overlap_budget_runtime_states.getptr(key);
	if (!state) {
		adaptive_overlap_budget_runtime_states.insert(key, AdaptiveOverlapBudgetRuntimeState());
		state = adaptive_overlap_budget_runtime_states.getptr(key);
	}
	return *state;
}

static void _clear_adaptive_overlap_budget_state(const TileRenderer *p_renderer) {
	adaptive_overlap_budget_runtime_states.erase(_adaptive_overlap_budget_key(p_renderer));
}

static bool _is_adaptive_overlap_budget_enabled() {
	return g_gpu_sorting_config.adaptive_overlap_budget_enabled;
}

static uint32_t _get_adaptive_overlap_budget_hard_cap() {
	return g_gpu_sorting_config.get_overlap_records_hard_cap();
}

static uint32_t _get_adaptive_overlap_budget_min() {
	return MAX<uint32_t>(g_gpu_sorting_config.get_overlap_records_adaptive_min(), 1u);
}

static uint32_t _compute_overlap_budget_with_headroom(uint32_t p_raw_records) {
	const uint64_t raw = uint64_t(MAX<uint32_t>(p_raw_records, 1u));
	// +20% headroom (ceil) to avoid repeated clipping spikes.
	return uint32_t(MIN<uint64_t>((raw * 6u + 4u) / 5u, uint64_t(UINT32_MAX)));
}

static void _record_adaptive_overlap_raw_usage(AdaptiveOverlapBudgetRuntimeState &r_state, uint32_t p_raw_records) {
	r_state.recent_raw_usage_records[r_state.recent_raw_usage_write_index] = p_raw_records;
	r_state.recent_raw_usage_write_index = (r_state.recent_raw_usage_write_index + 1u) % ADAPTIVE_OVERLAP_BUDGET_RECENT_RAW_WINDOW;
	if (r_state.recent_raw_usage_count < ADAPTIVE_OVERLAP_BUDGET_RECENT_RAW_WINDOW) {
		r_state.recent_raw_usage_count++;
	}

	uint32_t peak = 0u;
	for (uint32_t i = 0u; i < r_state.recent_raw_usage_count; i++) {
		peak = MAX(peak, r_state.recent_raw_usage_records[i]);
	}
	r_state.recent_raw_usage_peak = peak;
}

static bool _adaptive_overlap_budget_has_history(const TileRenderer *p_renderer) {
	const AdaptiveOverlapBudgetRuntimeState *state =
			adaptive_overlap_budget_runtime_states.getptr(_adaptive_overlap_budget_key(p_renderer));
	return state && state->recent_raw_usage_count > 0u;
}

static void _log_adaptive_overlap_budget_change(uint32_t p_previous_budget, uint32_t p_new_budget,
		uint32_t p_raw_basis, bool p_hard_cap_clamped, const char *p_reason) {
	if (p_previous_budget == p_new_budget) {
		return;
	}
	GS_LOG_GPU_SORT_INFO(vformat("[TileRenderer] Adaptive overlap budget %s: %s -> %s (raw_basis=%s hard_cap_clamped=%s)",
			String(p_reason),
			String::num_uint64(p_previous_budget),
			String::num_uint64(p_new_budget),
			String::num_uint64(p_raw_basis),
			p_hard_cap_clamped ? "yes" : "no"));
}

static uint32_t _get_adaptive_overlap_budget_suggestion(const TileRenderer *p_renderer) {
	if (!_is_adaptive_overlap_budget_enabled()) {
		return 0u;
	}
	const AdaptiveOverlapBudgetRuntimeState *state = adaptive_overlap_budget_runtime_states.getptr(_adaptive_overlap_budget_key(p_renderer));
	if (!state || state->suggested_budget_records == 0u) {
		return 0u;
	}
	const uint32_t hard_cap = _get_adaptive_overlap_budget_hard_cap();
	const uint32_t min_budget = _get_adaptive_overlap_budget_min();
	return CLAMP(state->suggested_budget_records, min_budget, hard_cap);
}

static uint32_t _get_adaptive_overlap_budget_floor(const TileRenderer *p_renderer) {
	if (!_is_adaptive_overlap_budget_enabled()) {
		return 0u;
	}
	const uint32_t hard_cap = _get_adaptive_overlap_budget_hard_cap();
	const uint32_t min_budget = _get_adaptive_overlap_budget_min();
	AdaptiveOverlapBudgetRuntimeState &state = _get_adaptive_overlap_budget_state(p_renderer);
	if (state.suggested_budget_records == 0u) {
		state.suggested_budget_records = min_budget;
	}
	state.suggested_budget_records = CLAMP(state.suggested_budget_records, min_budget, hard_cap);
	return state.suggested_budget_records;
}

static uint32_t _update_adaptive_overlap_budget(const TileRenderer *p_renderer, uint32_t p_raw_records) {
	constexpr uint32_t SHRINK_TRIGGER_UTILIZATION_PERCENT = 55u;
	constexpr uint32_t SHRINK_HYSTERESIS_FRAMES = 120u;
	constexpr uint32_t SHRINK_STEP_PERCENT = 90u; // shrink by at most 10% per hysteresis window

	if (!_is_adaptive_overlap_budget_enabled()) {
		_clear_adaptive_overlap_budget_state(p_renderer);
		return 0u;
	}

	const uint32_t hard_cap = _get_adaptive_overlap_budget_hard_cap();
	const uint32_t min_budget = _get_adaptive_overlap_budget_min();
	AdaptiveOverlapBudgetRuntimeState &state = _get_adaptive_overlap_budget_state(p_renderer);
	_record_adaptive_overlap_raw_usage(state, p_raw_records);

	const uint32_t raw_budget_basis = MAX(p_raw_records, state.recent_raw_usage_peak);
	const uint32_t desired_budget_unclamped = _compute_overlap_budget_with_headroom(raw_budget_basis);
	const bool hard_cap_clamped = hard_cap != UINT32_MAX && desired_budget_unclamped > hard_cap;
	const uint32_t desired_budget = CLAMP(desired_budget_unclamped, min_budget, hard_cap);
	const uint32_t previous_budget = state.suggested_budget_records;

	if (state.suggested_budget_records == 0u) {
		state.suggested_budget_records = desired_budget;
		_log_adaptive_overlap_budget_change(0u, state.suggested_budget_records, raw_budget_basis, hard_cap_clamped, "init");
		return state.suggested_budget_records;
	}

	if (desired_budget > state.suggested_budget_records) {
		state.suggested_budget_records = desired_budget;
		state.low_utilization_frames = 0u;
		_log_adaptive_overlap_budget_change(previous_budget, state.suggested_budget_records, raw_budget_basis, hard_cap_clamped, "grow");
		return state.suggested_budget_records;
	}

	const uint64_t shrink_trigger = uint64_t(previous_budget) * SHRINK_TRIGGER_UTILIZATION_PERCENT / 100u;
	if (uint64_t(raw_budget_basis) <= shrink_trigger) {
		state.low_utilization_frames++;
		if (state.low_utilization_frames >= SHRINK_HYSTERESIS_FRAMES) {
			const uint32_t shrink_step_budget = uint32_t(MAX<uint64_t>(
					(uint64_t(previous_budget) * SHRINK_STEP_PERCENT + 99u) / 100u,
					uint64_t(min_budget)));
			uint32_t new_budget = MAX(shrink_step_budget, desired_budget);
				new_budget = CLAMP(new_budget, min_budget, hard_cap);
				state.low_utilization_frames = 0u;
				if (new_budget < previous_budget) {
					state.suggested_budget_records = new_budget;
					_log_adaptive_overlap_budget_change(previous_budget, state.suggested_budget_records, raw_budget_basis, false, "shrink");
				}
			}
		} else {
		state.low_utilization_frames = 0u;
	}

	return state.suggested_budget_records;
}

} // namespace

class TileRenderer::RenderFrameExecutor {
public:
	RenderFrameExecutor(TileRenderer &p_renderer, RenderingDevice *p_rendering_device, const RenderParams &p_params, RenderingDevice *p_resource_device)
			: renderer(p_renderer),
			  rendering_device(p_rendering_device),
			  params(p_params),
			  resource_device(p_resource_device) {}

	~RenderFrameExecutor() {
		_prepare_next_tile_counts_if_needed();
	}

	RID run() {
		// Clear debug counters FIRST to ensure clean state for this frame regardless of early returns.
		// This prevents "poisoned" state from affecting subsequent frames.
		renderer._clear_debug_counters();
		renderer._reset_timestamp_tracking();

		if (!_validate_and_configure_settings()) {
			return RID();
		}
		_setup_frame_timestamps();
		if (!_validate_viewport_and_resources()) {
			return RID();
		}
		if (!_setup_diagnostics_and_params()) {
			return RID();
		}
		if (!_execute_global_sort_pipeline()) {
			return RID();
		}
		if (_has_dispatch_work()) {
			if (!_select_and_prepare_raster_path()) {
				return RID();
			}
			_dispatch_rasterization();
		} else {
			// Ensure no-work frames don't report stale raster mode from prior frames.
			renderer.perf_metrics.last_raster_used_compute = false;
			renderer.perf_metrics.sorted_indices_blend_fallback_active = false;
			renderer.perf_metrics.sorted_indices_blend_fallback_reason = String();
		}
		return _finalize_frame();
	}

private:
	bool _has_dispatch_work() const {
		// Only consider actual splat/element counts as work. Buffer existence
		// alone (e.g. an indirect dispatch buffer with a GPU-written count of 0)
		// should not prevent the zero-work clear path from running.
		return params.splat_count > 0 || effective_visible_splats > 0;
	}

	void _prepare_next_tile_counts_if_needed() {
		if (!tile_counts_buffer_advanced || tile_counts_buffer_prepared) {
			return;
		}
		if (!renderer.render_settings.global_sort_enabled) {
			return;
		}
		renderer.global_sort_resources.prepare_next_tile_counts_buffer(resource_device);
		tile_counts_buffer_prepared = true;
	}

	bool _validate_and_configure_settings() {
		if (rendering_device && rendering_device != resource_device) {
			ERR_PRINT_ONCE("[TileRenderer] Render called with mismatched RenderingDevice instance");
			return false;
		}

		if (!params.gaussian_buffer.is_valid() || !params.sorted_indices.is_valid()) {
			ERR_PRINT_ONCE("[TileRenderer] Invalid buffers passed to tile renderer");
			return false;
		}

		renderer.instance_pipeline_buffers.instance_buffer = params.instance_buffer;
		renderer.instance_pipeline_buffers.splat_ref_buffer = params.splat_ref_buffer;
		renderer.instance_pipeline_buffers.quantization_buffer = params.quantization_buffer;
		renderer.instance_pipeline_buffers.indirect_count_buffer = params.instance_indirect_count_buffer;
		renderer.instance_pipeline_buffers.indirect_dispatch_buffer = params.instance_indirect_dispatch_buffer;

		const bool quantization_required = g_quantization_config.per_chunk_quantization;
		const GaussianSplatting::InstancePipelineContract::InvariantViolationReason invariant_reason =
				GaussianSplatting::InstancePipelineContract::first_tile_runtime_violation(
						renderer.instance_pipeline_buffers.instance_buffer,
						renderer.instance_pipeline_buffers.splat_ref_buffer,
						renderer.instance_pipeline_buffers.indirect_count_buffer,
						renderer.instance_pipeline_buffers.indirect_dispatch_buffer,
						quantization_required,
						renderer.instance_pipeline_buffers.quantization_buffer);
		if (invariant_reason != GaussianSplatting::InstancePipelineContract::InvariantViolationReason::NONE) {
			const char *route = GaussianSplatting::InstancePipelineContract::get_violation_route(invariant_reason);
			const char *violation_class_name = GaussianSplatting::InstancePipelineContract::get_violation_class_name(
					GaussianSplatting::InstancePipelineContract::get_violation_class(invariant_reason));
			const char *reason_name = GaussianSplatting::InstancePipelineContract::get_violation_reason_name(invariant_reason);
#if defined(DEBUG_ENABLED) || defined(TESTS_ENABLED)
			ERR_FAIL_V_MSG(false, vformat("[TileRenderer] Hard-fail: impossible instance pipeline activation invariant violation route=%s class=%s reason=%s",
					route, violation_class_name, reason_name));
#else
			ERR_PRINT_ONCE(vformat("[TileRenderer] Instance pipeline enabled but required buffers are missing route=%s class=%s reason=%s",
					route, violation_class_name, reason_name));
			return false;
#endif
		}

		if (params.splat_count == 0 && params.sorted_indices.is_valid() && params.total_gaussians > 0) {
			WARN_PRINT_ONCE(vformat("[TileRenderer] Zero splat_count with non-empty sorted indices (total_gaussians=%d, buffer_id=%d)",
					params.total_gaussians, params.sorted_indices.get_id()));
		}

		if (renderer.device_context.performance_monitor) {
			RenderingDevice *submission_device = renderer._get_submission_device();
			renderer.device_context.performance_monitor->set_rendering_device(submission_device ? submission_device : resource_device);
			renderer.device_context.performance_monitor->record_submission(renderer.frame_state.current_frame_serial, renderer.frame_state.current_frame_serial);
		}

		// Enable sync/readback only when HUD or density overlays are active.
		renderer.diagnostics.runtime_statistics_enabled = params.debug_show_performance_hud ||
				params.debug_show_tile_grid || params.debug_show_density_heatmap;
		renderer.perf_metrics.sorted_indices_blend_fallback_active = false;
		renderer.perf_metrics.sorted_indices_blend_fallback_reason = String();

		// Use caller's frame_serial if provided, otherwise auto-increment.
		// A non-zero frame_serial is required for async GPU counter readback.
		static uint64_t auto_frame_serial = 0;
		if (params.frame_serial > 0) {
			renderer.frame_state.current_frame_serial = params.frame_serial;
			auto_frame_serial = params.frame_serial; // Sync auto counter
		} else {
			renderer.frame_state.current_frame_serial = ++auto_frame_serial;
		}

		renderer.render_settings.global_sort_enabled = true;
		const uint32_t current_effective_visible_splats = _resolve_effective_visible_splats();
		effective_visible_splats = current_effective_visible_splats;
		const uint32_t effective_splat_floor = MAX(params.splat_count, current_effective_visible_splats);
		initial_overlap_record_count = (params.overlap_record_count > 0) ? params.overlap_record_count : effective_splat_floor;
		resolved_total_gaussians = params.total_gaussians;
		renderer.render_settings.allow_compute_raster = _resolve_compute_raster_policy(params.compute_raster_policy);
		bool packed_stage_requested = params.request_packed_stage_data;
		uint32_t packed_stage_count = resolved_total_gaussians > 0 ? resolved_total_gaussians : effective_splat_floor;
		if (packed_stage_requested && packed_stage_count > UINT16_MAX) {
			WARN_PRINT_ONCE("[TileRenderer] Packed stage data requires <= 65535 total splats; disabling packed projection payloads.");
			packed_stage_requested = false;
		}
		bool packed_stage_changed = (renderer.render_settings.enable_packed_stage_data != packed_stage_requested);
		renderer.render_settings.enable_packed_stage_data = packed_stage_requested;

		bool tighter_bounds_requested = params.request_tighter_bounds;
		bool tighter_bounds_changed = (renderer.render_settings.enable_tighter_bounds != tighter_bounds_requested);
		renderer.render_settings.enable_tighter_bounds = tighter_bounds_requested;

		bool sh_amortization_requested = params.request_sh_amortization;
		uint32_t sh_divisor = params.sh_amortization_divisor;
		if (sh_amortization_requested && resolved_total_gaussians == 0) {
			WARN_PRINT_ONCE("[TileRenderer] SH amortization requires total_gaussians; disabling for this frame.");
			sh_amortization_requested = false;
		}
		if (!sh_amortization_requested || sh_divisor <= 1u) {
			sh_amortization_requested = false;
			sh_divisor = 1u;
		}
		bool sh_cache_resized = false;
		if (sh_amortization_requested && resolved_total_gaussians > 0) {
			sh_cache_resized = renderer.sh_cache_buffers.ensure_color_cache(resolved_total_gaussians);
			if (!renderer.sh_cache_buffers.sh_color_cache.is_valid()) {
				WARN_PRINT_ONCE("[TileRenderer] SH amortization cache allocation failed; disabling for this frame.");
				sh_amortization_requested = false;
				sh_divisor = 1u;
			}
		}
		bool sh_amortization_changed = (renderer.render_settings.enable_sh_amortization != sh_amortization_requested);
		bool sh_divisor_changed = (renderer.render_settings.sh_amortization_divisor != sh_divisor);
		renderer.render_settings.enable_sh_amortization = sh_amortization_requested;
		renderer.render_settings.sh_amortization_divisor = sh_divisor;
		if (sh_cache_resized) {
			renderer.sh_cache_needs_full_update = true;
		}
		if (sh_amortization_changed || sh_divisor_changed) {
			renderer.sh_cache_needs_full_update = sh_amortization_requested;
		}

		const uint32_t subpixel_history_entries = MAX(resolved_total_gaussians, effective_splat_floor);
		if (subpixel_history_entries > 0) {
			renderer.subpixel_history_buffers.ensure_history_buffer(subpixel_history_entries);
			if (!renderer.subpixel_history_buffers.subpixel_history_buffer.is_valid()) {
				ERR_PRINT_ONCE("[TileRenderer] Failed to allocate subpixel history buffer for hysteresis.");
				return false;
			}
		}

		if (packed_stage_changed || tighter_bounds_changed || sh_amortization_changed) {
			renderer.shader_resources.reset_state();
			if (packed_stage_changed) {
				if (resource_device) {
					renderer.projection_buffers.release(resource_device);
				} else {
					renderer.projection_buffers.reset_state();
				}
			}
			renderer._invalidate_descriptor_cache();
		}

		return true;
	}

	void _setup_frame_timestamps() {
		if (!renderer.is_gpu_timestamp_capture_enabled()) {
			return;
		}
		// Bracket the whole render for coarse GPU timing. Only pre-flush markers survive on main RD;
		// post-flush markers may be discarded (see docs/GPU_TIMESTAMP_PROFILING.md).
		frame_timestamp_device = renderer._get_submission_device();
		if (!frame_timestamp_device) {
			frame_timestamp_device = resource_device;
		}
		if (frame_timestamp_device) {
			timestamp_label = "GaussianSplat_" + String::num_uint64(renderer.frame_state.current_frame_serial);
			frame_timestamp_device->capture_timestamp(timestamp_label + String("_Begin"));
			// Register with guard so early returns emit matching "_End" marker.
			scope_guard.set_timestamp_context(frame_timestamp_device, timestamp_label);
		}

		// NOTE: GaussianSplat_Begin/End timestamps are NOT captured because buffer_get_data()
		// in the prefix sum forces a flush that resets the timestamp buffer. Only TileOverlapCount
		// markers (pre-flush) can be read. See docs/GPU_TIMESTAMP_PROFILING.md.
	}

	uint32_t _resolve_effective_visible_splats() {
		// When using GPU-indirect dispatch, the actual visible splat count is
		// determined by the GPU and may exceed the CPU's stale readback value
		// (params.splat_count). The projection_buffer and other per-splat buffers
		// must be large enough for the maximum the GPU can produce, otherwise the
		// EMIT pass writes out-of-bounds causing tile corruption.
		if (renderer.instance_pipeline_buffers.indirect_dispatch_buffer.is_valid() &&
				params.max_visible_splats > 0) {
			return MAX(params.splat_count, params.max_visible_splats);
		}
		return params.splat_count;
	}

	bool _validate_viewport_and_resources() {
		target_viewport = params.viewport_size;
		if (target_viewport.x <= 0 || target_viewport.y <= 0) {
			target_viewport = renderer.grid_state.viewport_size;
		}

		if (target_viewport.x <= 0 || target_viewport.y <= 0) {
			ERR_PRINT_ONCE("[TileRenderer] Invalid viewport size for tile renderer");
			return false;
		}

		requested_tile_size = params.tile_size > 0 ? params.tile_size : renderer.config_state.tile_size;
		if (renderer._ensure_resources(target_viewport, requested_tile_size, renderer.config_state.desired_output_format) != OK) {
			GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to prepare GPU resources for rendering");
			return false;
		}

		effective_visible_splats = _resolve_effective_visible_splats();
		renderer._ensure_global_projection_buffer(effective_visible_splats);
		renderer.subpixel_visibility_buffers.ensure_visibility_buffer(effective_visible_splats);

		if (!renderer.projection_buffers.projection_buffer.is_valid() ||
				!renderer.subpixel_visibility_buffers.subpixel_visibility_buffer.is_valid() ||
				!renderer.render_targets.output_texture.is_valid() || !renderer.get_output_texture().is_valid() ||
				!renderer.render_targets.normal_texture.is_valid()) {
			ERR_PRINT_ONCE("[TileRenderer] Tile renderer resources are not valid after initialization");
			return false;
		}

		if (!renderer._validate_tile_grid("render")) {
			return false;
		}

#ifdef DEV_ENABLED
		renderer._validate_resource_owner(renderer.projection_buffers.projection_buffer, renderer.projection_buffers.projection_buffer_owner, resource_device, "projection_buffer");
		renderer._validate_resource_owner(renderer.subpixel_visibility_buffers.subpixel_visibility_buffer, renderer.subpixel_visibility_buffers.subpixel_visibility_owner, resource_device, "subpixel_visibility_buffer");
		renderer._validate_resource_owner(renderer.debug_stats.debug_counter_buffer, renderer.debug_stats.debug_counter_owner, resource_device, "debug_counter_buffer");
		renderer._validate_resource_owner(renderer.debug_stats.overflow_statistics_buffer, renderer.debug_stats.overflow_stats_owner, resource_device, "overflow_statistics_buffer");
		renderer._validate_resource_owner(renderer.debug_stats.debug_splat_audit_buffer, renderer.debug_stats.debug_splat_audit_owner, resource_device, "debug_splat_audit_buffer");
		if (renderer.render_settings.global_sort_enabled) {
			renderer._validate_resource_owner(renderer.global_sort_resources.keys_buffer, renderer.global_sort_resources.buffer_owner, resource_device, "global_sort_buffers");
		}
		renderer._validate_local_owner(renderer.render_targets.output_texture, renderer.render_targets.output_texture_local_owner, resource_device, "output_texture");
		renderer._validate_local_owner(renderer.render_targets.depth_texture, renderer.render_targets.depth_texture_local_owner, resource_device, "depth_texture");
		renderer._validate_local_owner(renderer.render_targets.normal_texture, renderer.render_targets.normal_texture_local_owner, resource_device, "normal_texture");
		renderer._validate_local_owner(renderer.render_targets.resolved_texture, renderer.render_targets.resolved_texture_local_owner, resource_device, "resolved_texture");
		renderer._validate_local_owner(renderer.render_targets.resolved_depth_texture, renderer.render_targets.resolved_depth_texture_local_owner,
				resource_device, "resolved_depth_texture");
		renderer._validate_resource_owner(renderer.render_targets.output_texture_external, renderer.render_targets.output_texture_owner,
				renderer.render_targets.output_texture_owner.device, "output_texture_external");
		renderer._validate_resource_owner(renderer.render_targets.depth_texture_external, renderer.render_targets.depth_texture_owner,
				renderer.render_targets.depth_texture_owner.device, "depth_texture_external");
		renderer._validate_resource_owner(renderer.render_targets.resolved_texture_external, renderer.render_targets.resolved_texture_owner,
				renderer.render_targets.resolved_texture_owner.device, "resolved_texture_external");
		renderer._validate_resource_owner(renderer.render_targets.resolved_depth_texture_external, renderer.render_targets.resolved_depth_texture_owner,
				renderer.render_targets.resolved_depth_texture_owner.device, "resolved_depth_texture_external");
		if (renderer.shader_resources.shader_device) {
			uint64_t actual_device_id = renderer.shader_resources.shader_device->get_device_instance_id();
			if (renderer.shader_resources.shader_device_instance != actual_device_id) {
				ERR_PRINT(vformat("[TileRenderer] Shader device instance mismatch (expected=%s actual=%s)",
						String::num_uint64(renderer.shader_resources.shader_device_instance), String::num_uint64(actual_device_id)));
				DEV_ASSERT(renderer.shader_resources.shader_device_instance == actual_device_id);
			}
		}
#endif

		return true;
	}

	void _setup_diagnostics_state() {
		renderer.diagnostics.debug_dump_gpu_counters = params.debug_dump_gpu_counters;
		renderer.diagnostics.debug_gpu_counter_logs_enabled = params.debug_enable_gpu_counter_logs;
		renderer.diagnostics.debug_tile_logs_enabled = params.debug_enable_tile_logs;
		renderer.diagnostics.debug_tile_pipeline_logs_enabled = params.debug_enable_tile_pipeline_logs;
		renderer.diagnostics.debug_tile_dispatch_logs_enabled = params.debug_enable_tile_dispatch_logs;
		renderer.diagnostics.debug_frame_log_frequency = params.debug_frame_log_frequency;
		renderer.diagnostics.last_overlap_record_budget_effective = 0;
		renderer.diagnostics.last_overlap_keep_ratio = 1.0f;

		// Note: _clear_debug_counters() already called at function entry for clean state.
		// The second call in the global sort path is intentional to reset
		// counters before emit+raster passes for per-stage diagnostics.

		log_interval = renderer.diagnostics.debug_frame_log_frequency;
		tile_logs_enabled = renderer.diagnostics.debug_tile_logs_enabled && log_interval > 0;

#ifdef DEV_ENABLED
		static uint64_t render_entry_count = 0;
		render_entry_count++;
		if (tile_logs_enabled && (render_entry_count % static_cast<uint64_t>(log_interval) == 0)) {
			GS_LOG_RENDERER_DEBUG(vformat("[TileRenderer::render] ENTRY #%d buf_id=%d", render_entry_count,
					renderer.debug_stats.overflow_statistics_buffer.get_id()));
		}
#endif

	}

	bool _upload_param_buffer(float p_overlap_keep_ratio) {
		TileRenderParamsGPU params_gpu = renderer.params_builder.build_params(params, initial_overlap_record_count,
				resolved_total_gaussians, resource_device, p_overlap_keep_ratio);

		uint32_t param_hash = hash_murmur3_buffer(&params_gpu, int(sizeof(TileRenderParamsGPU)));
		const RID param_buffer = renderer.uniform_buffers.param_uniform_buffer;
		const bool needs_update = !renderer.last_param_hash_valid ||
				renderer.last_param_hash != param_hash ||
				renderer.last_param_uniform_buffer != param_buffer;
		if (!needs_update) {
			return true;
		}

		Vector<uint8_t> param_data;
		param_data.resize(sizeof(TileRenderParamsGPU));
		std::memcpy(param_data.ptrw(), &params_gpu, sizeof(TileRenderParamsGPU));

		RenderingDevice *param_device = renderer.uniform_buffers.param_uniform_buffer_owner.device
				? renderer.uniform_buffers.param_uniform_buffer_owner.device
				: uniform_device;
		if (!param_device) {
			param_device = resource_device;
		}
		ERR_FAIL_NULL_V_MSG(param_device, false, "[TileRenderer] No RenderingDevice available to update parameter buffer contents");
		param_device->buffer_update(param_buffer, 0, param_data.size(), param_data.ptr());
		renderer.last_param_hash = param_hash;
		renderer.last_param_hash_valid = true;
		renderer.last_param_uniform_buffer = param_buffer;
		return true;
	}

	bool _build_and_upload_params() {
		uniform_device = renderer._acquire_submission_device();
		if (!uniform_device) {
			uniform_device = resource_device;
		}

		if (!uniform_device) {
			ERR_PRINT_ONCE("[TileRenderer] Unable to determine RenderingDevice for uniform updates");
			return false;
		}

#ifdef DEBUG_ENABLED
		// Pipeline debug logging disabled for performance (was gated by kLogPipelineDebug).
#endif

		if (!renderer._ensure_param_uniform_buffer(uniform_device)) {
			ERR_PRINT_ONCE("[TileRenderer] Failed to allocate parameter buffer");
			return false;
		}
		if (!_upload_param_buffer(overlap_keep_ratio)) {
			return false;
		}
		renderer._update_splat_audit_buffer(params);
		return true;
	}

	bool _setup_diagnostics_and_params() {
		_setup_diagnostics_state();
		return _build_and_upload_params();
	}

	bool _execute_global_sort_pipeline() {
		uint64_t assignment_start = OS::get_singleton()->get_ticks_usec();
		auto finish_assignment_metrics = [&]() {
			uint64_t assignment_end = OS::get_singleton()->get_ticks_usec();
			renderer.perf_metrics.tile_assignment_ms = (assignment_end - assignment_start) / 1000.0f;
			renderer.timing_state.last_setup_cpu_ms = renderer.perf_metrics.tile_assignment_ms;
		};

			if (renderer.render_settings.global_sort_enabled) {
				if (!_has_dispatch_work()) {
					renderer.diagnostics.last_overlap_record_count = 0;
					renderer.diagnostics.last_overlap_record_budget_effective = renderer._get_effective_overlap_capacity();
					renderer.diagnostics.last_overlap_keep_ratio = 1.0f;
					finish_assignment_metrics();
					return true;
				}

				bool debug_sync_requested = false;
#ifdef DEBUG_ENABLED
					debug_sync_requested = (g_gpu_sorting_config.enable_prefix_readback || g_gpu_sorting_config.debug_validate_prefix) &&
							!g_gpu_sorting_config.profiling_preserve_gpu_timestamps;
#endif
				const gs_sort_policy::ReadbackPolicy readback_policy =
						gs_sort_policy::resolve_readback_policy(debug_sync_requested,
								g_gpu_sorting_config.profiling_preserve_gpu_timestamps);
				const bool allow_sync_readback = readback_policy.allow_sync_readback;

				// Async auto-resize: If the previous frame detected overflow via async readback,
				// proactively resize buffers BEFORE the count pass. This allows disabling sync
				// readback for a pure GPU-driven pipeline with no CPU stalls.
				if (renderer.async_readback.overflow_state.first_frame_complete && renderer.async_readback.overflow_state.overflow_detected) {
					const uint32_t headroom_factor = 3u; // 50% headroom = multiply by 1.5 ≈ 3/2
					const uint64_t needed64 = uint64_t(renderer.async_readback.overflow_state.last_unclamped_total) * headroom_factor / 2u;
					const uint32_t needed = needed64 > UINT32_MAX ? UINT32_MAX : uint32_t(needed64);
					if (needed > renderer.global_sort_resources.capacity) {
						if (renderer.diagnostics.debug_tile_logs_enabled) {
							GS_LOG_GPU_SORT_DEBUG(vformat("[TileRenderer] Async auto-resize: overflow detected (unclamped=%d, capacity=%d), resizing to %d",
									renderer.async_readback.overflow_state.last_unclamped_total, renderer.global_sort_resources.capacity, needed));
						}
						renderer._ensure_global_sort_resources(needed);
					}
					// Clear the overflow flag after handling to prevent repeated resizes.
					renderer.async_readback.overflow_state.overflow_detected = false;
				}
				// Initial buffer sizing: use higher multiplier for close camera views where splats cover many tiles.
				// With camera close to geometry, each splat can cover 50-100+ tiles easily.
				// Using 50x provides headroom for close-up views without budget cutoff.
				// Use effective_visible_splats (accounts for max_visible_splats from instance pipeline)
				// to avoid undersized initial allocation when CPU readback is stale.
				const uint32_t overlap_base_count = MAX(params.splat_count, effective_visible_splats);
				uint32_t overlap_estimate = MIN<uint64_t>(uint64_t(overlap_base_count) * 50u, UINT32_MAX);
				if (renderer.global_sort_resources.capacity == 0 &&
						params.total_gaussians > 0 &&
						overlap_base_count < 1024u) {
					// Streaming warm-up can report a tiny visible set on the first frame while the
					// atlas already represents a much larger world. Avoid initializing sort capacity
					// to unusably small values (for example 50) that would hard-clamp overlap records.
					const uint32_t bootstrap_cap = 8000000u;
					const uint32_t bootstrap_floor = MIN(params.total_gaussians, bootstrap_cap);
					overlap_estimate = MAX(overlap_estimate, bootstrap_floor);
					if (renderer.diagnostics.debug_tile_logs_enabled) {
						GS_LOG_GPU_SORT_DEBUG(vformat("[TileRenderer] Bootstrap overlap estimate: splat_count=%u total_gaussians=%u estimate=%u",
								params.splat_count, params.total_gaussians, overlap_estimate));
					}
					}
					const uint32_t adaptive_overlap_floor = _get_adaptive_overlap_budget_floor(&renderer);
					if (adaptive_overlap_floor > 0u) {
						if (_adaptive_overlap_budget_has_history(&renderer)) {
							// After we have raw usage history, drive allocation from adaptive budget directly.
							overlap_estimate = adaptive_overlap_floor;
						} else {
							// Bootstrap from coarse estimate before first reliable overlap sample is available.
							overlap_estimate = MAX(overlap_estimate, adaptive_overlap_floor);
						}
					}
					renderer._ensure_global_sort_resources(overlap_estimate);
				renderer.global_sort_resources.advance_tile_counts_buffer();
			tile_counts_buffer_advanced = true;
			if (renderer.global_sort_resources.capacity == 0 ||
					!renderer.global_sort_resources.keys_buffer.is_valid() || !renderer.global_sort_resources.values_buffer.is_valid() ||
					!renderer.global_sort_resources.get_tile_counts_buffer().is_valid() || !renderer.global_sort_resources.tile_ranges_buffer.is_valid() ||
					!renderer.shader_resources.tile_binning_count_pipeline.is_valid()) {
				GS_LOG_ERROR_DEFAULT("[TileRenderer] Global composite sort enabled but resources are unavailable");
				return false;
			}
			if (!renderer.global_sort_resources.sorter.is_valid() && params.splat_count > 0) {
				if (!renderer.global_sort_resources.sorter_missing_logged) {
					GS_LOG_WARN_DEFAULT("[TileRenderer] Global composite sorter unavailable; rendering unsorted tiles");
					renderer.global_sort_resources.sorter_missing_logged = true;
				}
			}

			// Pass 1: count overlaps per tile.
			renderer.binning_stage.clear_tile_counts(resource_device);

			TileBinningStage::BinningUniformSets count_sets;
			renderer.binning_stage.prepare_count_uniform_sets(uniform_device, params.gaussian_buffer, params.sorted_indices, params, count_sets);
			if (!count_sets.param_uniform_set.is_valid()) {
				GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to acquire binning parameter uniform set (count pass)");
				return false;
			}

			if (!count_sets.buffer_uniform_set.is_valid()) {
				GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to acquire binning buffer uniform set (count pass)");
				return false;
			}

			// Sync only when prefix readback/validation is enabled.
			renderer._dispatch_tile_binning_count(params.splat_count, count_sets.buffer_uniform_set, count_sets.param_uniform_set,
					count_sets.lighting_uniform_set,
					uniform_device, allow_sync_readback);

			uint32_t overlap_record_count = 0;
			uint32_t raw_overlap_record_count = 0;
			if (!renderer._update_global_tile_ranges(params.gaussian_buffer, params.sorted_indices, uniform_device, overlap_record_count,
						raw_overlap_record_count, allow_sync_readback)) {
				GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to build global tile ranges for rasterization");
				return false;
			}
			const uint32_t adaptive_overlap_budget_suggested = _update_adaptive_overlap_budget(&renderer, raw_overlap_record_count);
#ifdef DEV_ENABLED
			if (renderer.diagnostics.debug_tile_pipeline_logs_enabled) {
				static int overlap_debug_log_counter = 0;
					if (++overlap_debug_log_counter == 1 || (overlap_debug_log_counter % MAX(1, renderer.diagnostics.debug_frame_log_frequency) == 0)) {
						GS_LOG_GPU_SORT_DEBUG(vformat("[OVERLAP] frame=%d splats=%d overlap=%d raw=%d capacity=%d adaptive_suggested=%d adaptive_enabled=%s",
								int(renderer.frame_state.current_frame_serial), params.splat_count,
								overlap_record_count, raw_overlap_record_count, renderer.global_sort_resources.capacity,
								adaptive_overlap_budget_suggested,
								g_gpu_sorting_config.adaptive_overlap_budget_enabled ? "yes" : "no"));
					}
				}
#endif

			// Ensure sort buffers can hold the actual overlap count from prefix scan.
			// Without this, the initial estimate (splat_count * 50) may be too small
			// for close-up views where splats cover 100+ tiles each, causing the
				// prefix scan to clamp and truncate tile allocations.
				uint32_t overlap_capacity_request = MAX(overlap_record_count, raw_overlap_record_count);
				if (adaptive_overlap_budget_suggested > 0u) {
					overlap_capacity_request = MAX(overlap_capacity_request, adaptive_overlap_budget_suggested);
				}
			if (overlap_capacity_request > renderer.global_sort_resources.capacity) {
				const uint64_t gen_before = renderer.descriptor_generation;
				renderer._ensure_global_sort_resources(overlap_capacity_request);

				if (renderer.descriptor_generation > gen_before) {
					// Descriptor cache was invalidated: buffers were reallocated.
					// Re-run COUNT + PREFIX so the new buffers contain valid data.
					renderer.global_sort_resources.advance_tile_counts_buffer();
					renderer.binning_stage.clear_tile_counts(resource_device);

					renderer.binning_stage.prepare_count_uniform_sets(uniform_device,
							params.gaussian_buffer, params.sorted_indices, params, count_sets);
					if (!count_sets.param_uniform_set.is_valid() || !count_sets.buffer_uniform_set.is_valid()) {
						GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to re-acquire binning uniform sets after resize");
						return false;
					}

					renderer._dispatch_tile_binning_count(params.splat_count,
							count_sets.buffer_uniform_set, count_sets.param_uniform_set,
							count_sets.lighting_uniform_set, uniform_device, allow_sync_readback);

					overlap_record_count = 0;
					raw_overlap_record_count = 0;
					if (!renderer._update_global_tile_ranges(params.gaussian_buffer, params.sorted_indices,
								uniform_device, overlap_record_count, raw_overlap_record_count, allow_sync_readback)) {
						GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to rebuild tile ranges after buffer resize");
						return false;
					}
					_update_adaptive_overlap_budget(&renderer, raw_overlap_record_count);
				}
			}

			const uint32_t effective_overlap_capacity = renderer._get_effective_overlap_capacity();
			if (effective_overlap_capacity == 0) {
				GS_LOG_ERROR_DEFAULT("[TileRenderer] Effective overlap capacity is zero in global sort mode");
				return false;
			}

			renderer.diagnostics.last_overlap_record_count = overlap_record_count;

			// NOTE: GaussianSplat_Begin/End markers are NOT captured here because they would
			// be in the post-flush timestamp buffer and cannot be read back. Only TileOverlapCount
			// markers (pre-flush) are accessible. See docs/GPU_TIMESTAMP_PROFILING.md.

			renderer.diagnostics.last_overlap_record_budget_effective = effective_overlap_capacity;
			renderer.diagnostics.last_overlap_keep_ratio = overlap_keep_ratio;

			renderer.global_sort_resources.mark_tile_counts_dirty();

			// Reset debug counters so diagnostics reflect emit+raster passes only.
			renderer._clear_debug_counters();

			// Overlap count is read from global_sort_resources.indirect_dispatch_buffer in emit/raster shaders.

			// Pass 2: emit overlap records into the global key/value arrays.
			// Clear tile_counts before EMIT - the shader reuses it as a per-tile cursor (starting from 0)
			// to track write offsets within each tile's range. Without this clear, the cursor would
			// start at the count value from the COUNT pass instead of 0.
			renderer.binning_stage.clear_tile_counts(resource_device);

			TileBinningStage::BinningUniformSets emit_sets;
			renderer.binning_stage.prepare_emit_uniform_sets(uniform_device, params.gaussian_buffer, params.sorted_indices, params, emit_sets);
			if (!emit_sets.param_uniform_set.is_valid()) {
				GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to acquire binning parameter uniform set (emit pass)");
				return false;
			}

			if (!emit_sets.buffer_uniform_set.is_valid()) {
				GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to acquire binning buffer uniform set (emit pass)");
				return false;
			}

			renderer._dispatch_tile_binning(params.splat_count, emit_sets.buffer_uniform_set, emit_sets.param_uniform_set,
					emit_sets.lighting_uniform_set,
					uniform_device, allow_sync_readback);

			// GPU-driven sorting: shaders read element_count directly from indirect buffer.
			// No CPU readback needed - the GPU wrote the count during prefix scan.
			// In async mode overlap_record_count can be stale, so allow indirect-backed
			// paths to proceed even when CPU splat_count transiently reports zero.
			const bool has_instance_indirect = renderer.instance_pipeline_buffers.indirect_dispatch_buffer.is_valid();
			const bool should_attempt_sort = renderer.global_sort_resources.sorter.is_valid() &&
					(allow_sync_readback ? (overlap_record_count > 0) : (params.splat_count > 0 || has_instance_indirect));
			// PERF: Use sort_indirect_async to avoid blocking CPU on GPU sort completion.
			if (should_attempt_sort) {
#ifdef DEV_ENABLED
				if (!allow_sync_readback && overlap_record_count == 0 && params.splat_count > 0 && renderer.diagnostics.debug_tile_logs_enabled) {
					static uint64_t stale_overlap_sort_log_counter = 0;
					stale_overlap_sort_log_counter++;
					const int stale_log_interval = MAX(1, renderer.diagnostics.debug_frame_log_frequency);
					if (stale_overlap_sort_log_counter == 1 || (stale_overlap_sort_log_counter % uint64_t(stale_log_interval) == 0)) {
						GS_LOG_GPU_SORT_DEBUG(vformat("[TileRenderer] Async overlap estimate is zero (splats=%d), still dispatching indirect sort",
								params.splat_count));
					}
				}
#endif
				uint64_t sort_timeline = renderer.global_sort_resources.sorter->sort_indirect_async(
						renderer.global_sort_resources.keys_buffer,
						renderer.global_sort_resources.values_buffer,
						renderer.global_sort_resources.indirect_dispatch_buffer);
				if (sort_timeline == 0) {
					if (readback_policy.allow_sync_sort_fallback) {
						renderer.perf_metrics.sort_sync_fallback_count++;
						Error sort_err = renderer.global_sort_resources.sorter->sort_indirect(
								renderer.global_sort_resources.keys_buffer,
								renderer.global_sort_resources.values_buffer,
								renderer.global_sort_resources.indirect_dispatch_buffer);
						if (sort_err != OK) {
							GS_LOG_WARN_DEFAULT(vformat("[TileRenderer] Global composite sort failed (%d); rendering unsorted tiles", sort_err));
						}
					} else {
						GS_LOG_WARN_DEFAULT(vformat("[TileRenderer] Global composite async sort returned timeline=0 (policy=%s); rendering unsorted tiles",
								gs_sort_policy::mode_name(readback_policy.mode)));
					}
				}
				if (allow_sync_readback) {
					renderer._flush_pending_submission(true);
				}
			} else if ((allow_sync_readback ? (overlap_record_count > 0) : (params.splat_count > 0)) &&
					!renderer.global_sort_resources.sorter.is_valid() && !renderer.global_sort_resources.sorter_missing_logged) {
				GS_LOG_WARN_DEFAULT("[TileRenderer] Global composite sorter unavailable; rendering unsorted tiles");
				renderer.global_sort_resources.sorter_missing_logged = true;
			}

			// GPU-driven: sort reads element_count directly from global_sort_resources.indirect_dispatch_buffer.
		}

		finish_assignment_metrics();

		// Note: On main RD, compute→raster synchronization is handled automatically by command ordering.
		// sync() only works on local devices and isn't needed here since compute_list_end() ensures
		// compute work completes before subsequent draw list operations.

		return true;
	}

	bool _select_and_prepare_raster_path() {
		RID state_uniform = params.interactive_state_uniform.is_valid() ? params.interactive_state_uniform : renderer._get_default_state_uniform(uniform_device);
		const bool blend_dependent_path = !params.force_solid_coverage && params.splat_count > 0;
		const bool valid_sorted_indices_for_blend = params.sorted_indices.is_valid() && resolved_total_gaussians > 0;
		if (blend_dependent_path && !valid_sorted_indices_for_blend) {
			use_compute_raster = false;
			raster_reason = "Fragment fallback: blend-dependent raster requires valid sorted indices";
			renderer.perf_metrics.sorted_indices_blend_fallback_active = true;
			renderer.perf_metrics.sorted_indices_blend_fallback_reason = raster_reason;
		} else {
			RasterDecision raster_decision = renderer._evaluate_raster_path(uniform_device);
			use_compute_raster = raster_decision.use_compute;
			raster_reason = raster_decision.reason;
			renderer.perf_metrics.sorted_indices_blend_fallback_active = false;
			renderer.perf_metrics.sorted_indices_blend_fallback_reason = String();
		}
		raster_sets = TileRasterizerStage::RasterUniformSets{};
		if (use_compute_raster) {
			if (!renderer.raster_stage.prepare_compute_uniform_sets(uniform_device, state_uniform, params.gaussian_buffer,
						params.sorted_indices, raster_sets)) {
				use_compute_raster = false;
				raster_reason = "Compute raster fallback: uniform set acquisition failed";
			}
		}
		if (!use_compute_raster) {
			renderer.raster_stage.prepare_fragment_uniform_sets(uniform_device, state_uniform, params.gaussian_buffer,
					params.sorted_indices, raster_sets);
			if (!raster_sets.param_uniform_set.is_valid()) {
				GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to acquire raster parameter uniform set");
				return false;
			}
			if (!raster_sets.buffer_uniform_set.is_valid()) {
				GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to acquire raster buffer uniform set");
				return false;
			}
		}

		renderer._log_raster_path_decision({ use_compute_raster, raster_reason });
		renderer.perf_metrics.last_raster_used_compute = use_compute_raster;
		if (use_compute_raster) {
			renderer.perf_metrics.compute_raster_frames++;
		} else {
			renderer.perf_metrics.fragment_raster_frames++;
		}

		return true;
	}

	void _dispatch_rasterization() {
		uint64_t raster_start = OS::get_singleton()->get_ticks_usec();
		if (use_compute_raster) {
#ifdef DEV_ENABLED
			static uint64_t compute_raster_log_counter = 0;
			compute_raster_log_counter++;
			if (renderer.diagnostics.debug_tile_logs_enabled &&
					(compute_raster_log_counter == 1 ||
							(log_interval > 0 && (compute_raster_log_counter % static_cast<uint64_t>(log_interval) == 0)))) {
				GS_LOG_RENDERER_DEBUG(vformat("[TileRenderer] Using COMPUTE rasterizer: tiles=%dx%d splats=%d format=%d",
						renderer.grid_state.tiles_x, renderer.grid_state.tiles_y, params.splat_count, int(renderer.config_state.output_format)));
			}
#endif
			renderer._dispatch_tile_rasterizer_compute(params.splat_count, raster_sets.buffer_uniform_set, raster_sets.param_uniform_set,
					raster_sets.image_uniform_set, uniform_device);
		} else {
			// Fragment shader path: uses render pipeline with framebuffer attachments.
			renderer._dispatch_tile_rasterizer(params.splat_count, raster_sets.buffer_uniform_set, raster_sets.param_uniform_set, uniform_device);
		}
		uint64_t raster_end = OS::get_singleton()->get_ticks_usec();
		renderer.perf_metrics.rasterization_ms = (raster_end - raster_start) / 1000.0f;

		// Note: On main RD, raster→compute synchronization is handled automatically by command ordering.
		// draw_list_end() ensures fragment work completes before subsequent compute list operations.
	}

	RID _finalize_frame() {
		bool skip_resolve_dispatch = false;
		if (!_has_dispatch_work()) {
			renderer.perf_metrics.rasterization_ms = 0.0f;
			auto clear_texture = [](RenderingDevice *p_owner, const RID &p_texture, const Color &p_clear) {
				if (!p_owner || !p_texture.is_valid() || !p_owner->texture_is_valid(p_texture)) {
					return;
				}
				RD::TextureFormat fmt = p_owner->texture_get_format(p_texture);
				uint32_t mipmaps = MAX<uint32_t>(1, fmt.mipmaps);
				uint32_t layers = MAX<uint32_t>(1, fmt.array_layers);
				p_owner->texture_clear(p_texture, p_clear, 0, mipmaps, 0, layers);
			};

			RenderingDevice *raw_color_owner = renderer.render_targets.output_texture_owner.device
					? renderer.render_targets.output_texture_owner.device
					: resource_device;
			RenderingDevice *raw_depth_owner = renderer.render_targets.depth_texture_owner.device
					? renderer.render_targets.depth_texture_owner.device
					: resource_device;
			RenderingDevice *resolve_color_owner = renderer.render_targets.resolved_texture_owner.device
					? renderer.render_targets.resolved_texture_owner.device
					: resource_device;
			RenderingDevice *resolve_depth_owner = renderer.render_targets.resolved_depth_texture_owner.device
					? renderer.render_targets.resolved_depth_texture_owner.device
					: resource_device;
			RID raw_color = renderer.render_targets.output_texture_external.is_valid()
					? renderer.render_targets.output_texture_external
					: renderer.render_targets.output_texture;
			RID raw_depth = renderer.render_targets.depth_texture_external.is_valid()
					? renderer.render_targets.depth_texture_external
					: renderer.render_targets.depth_texture;
			RID resolve_color = renderer.render_targets.resolved_texture_external.is_valid()
					? renderer.render_targets.resolved_texture_external
					: renderer.render_targets.resolved_texture;
			RID resolve_depth = renderer.render_targets.resolved_depth_texture_external.is_valid()
					? renderer.render_targets.resolved_depth_texture_external
					: renderer.render_targets.resolved_depth_texture;
			clear_texture(raw_color_owner, raw_color, Color(0.0f, 0.0f, 0.0f, 0.0f));
			clear_texture(raw_depth_owner, raw_depth, Color(1.0f, 0.0f, 0.0f, 0.0f));
			if (resolve_color != raw_color || resolve_color_owner != raw_color_owner) {
				clear_texture(resolve_color_owner, resolve_color, Color(0.0f, 0.0f, 0.0f, 0.0f));
			}
			if (resolve_depth != raw_depth || resolve_depth_owner != raw_depth_owner) {
				clear_texture(resolve_depth_owner, resolve_depth, Color(1.0f, 0.0f, 0.0f, 0.0f));
			}
			skip_resolve_dispatch = true;
			if (renderer.diagnostics.debug_log_resolve) {
				GS_LOG_EVERY_N(gs_logger::Category::RENDERER, gs_logger::Level::DEBUG,
						tile_resolve_clear_log_counter, renderer.diagnostics.debug_log_resolve_interval_frames,
						vformat("[TileResolve] frame=%s splats=0 mode=%d outputs cleared (color=%s depth=%s)",
								String::num_uint64(renderer.frame_state.current_frame_serial),
								int(renderer.render_settings.resolve_debug_mode),
								renderer.render_targets.resolved_texture.is_valid()
										? String::num_uint64(renderer.render_targets.resolved_texture.get_id())
										: String("0"),
								renderer.render_targets.resolved_depth_texture.is_valid()
										? String::num_uint64(renderer.render_targets.resolved_depth_texture.get_id())
										: String("0")));
			}
		}

		if (!skip_resolve_dispatch &&
				renderer.render_targets.resolved_texture.is_valid() && renderer.render_targets.resolved_depth_texture.is_valid()) {
			renderer._dispatch_tile_resolve(target_viewport, requested_tile_size, params.output_is_premultiplied, params);
		}

		// Debug counter readback is now on-demand via get_debug_counters()/get_overflow_stats().

		if (renderer.diagnostics.debug_log_resolve) {
			RID raw_color = renderer.render_targets.output_texture_external.is_valid()
					? renderer.render_targets.output_texture_external
					: renderer.render_targets.output_texture;
			RID resolved_color = renderer.render_targets.resolved_texture_external.is_valid()
					? renderer.render_targets.resolved_texture_external
					: renderer.render_targets.resolved_texture;
			bool using_resolved = resolved_color.is_valid() && (renderer.render_settings.resolve_debug_mode != RESOLVE_DEBUG_INPUT);
			GS_LOG_EVERY_N(gs_logger::Category::RENDERER, gs_logger::Level::DEBUG,
					tile_resolve_status_log_counter, renderer.diagnostics.debug_log_resolve_interval_frames,
					vformat("[TileResolve] frame=%s splats=%d mode=%d using_resolved=%s raw=%s resolved=%s depth_raw=%s depth_resolved=%s",
							String::num_uint64(renderer.frame_state.current_frame_serial),
							int(params.splat_count),
							int(renderer.render_settings.resolve_debug_mode),
							using_resolved ? "yes" : "no",
							raw_color.is_valid() ? String::num_uint64(raw_color.get_id()) : String("0"),
							resolved_color.is_valid() ? String::num_uint64(resolved_color.get_id()) : String("0"),
							renderer.render_targets.depth_texture.is_valid()
									? String::num_uint64(renderer.render_targets.depth_texture.get_id())
									: String("0"),
							renderer.render_targets.resolved_depth_texture.is_valid()
									? String::num_uint64(renderer.render_targets.resolved_depth_texture.get_id())
									: String("0")));
		}

		// Collect rendering statistics and validate results.
		renderer._collect_render_statistics();

		// Dump GPU debug counters for diagnosing splat disappearance.
		renderer._dump_gpu_debug_counters(params);

		if (renderer.render_settings.global_sort_enabled) {
			renderer.global_sort_resources.prepare_next_tile_counts_buffer(resource_device);
			tile_counts_buffer_prepared = true;
		}

		if (renderer.render_settings.enable_sh_amortization && renderer.sh_cache_needs_full_update) {
			renderer.sh_cache_needs_full_update = false;
		}

		RID final_output = renderer.get_output_texture();

		if (frame_timestamp_device && renderer.is_gpu_timestamp_capture_enabled()) {
			frame_timestamp_device->capture_timestamp(timestamp_label + String("_End"));
		}

		// Mark success so the RAII guard doesn't emit duplicate "_End" timestamp.
		scope_guard.mark_success();

		return final_output;
	}

	TileRenderer &renderer;
	RenderingDevice *rendering_device = nullptr;
	const RenderParams &params;
	RenderingDevice *resource_device = nullptr;
	RenderScopeGuard scope_guard;
	RenderingDevice *uniform_device = nullptr;
	RenderingDevice *frame_timestamp_device = nullptr;
	String timestamp_label;
	Vector2i target_viewport;
	int requested_tile_size = 0;
	uint32_t initial_overlap_record_count = 0;
	uint32_t resolved_total_gaussians = 0;
	uint32_t effective_visible_splats = 0;
	int log_interval = 0;
	bool tile_logs_enabled = false;
	float overlap_keep_ratio = 1.0f; // GPU handles overflow via prefix scan clamping.
	bool tile_counts_buffer_advanced = false;
	bool tile_counts_buffer_prepared = false;
	bool use_compute_raster = false;
	String raster_reason;
	TileRasterizerStage::RasterUniformSets raster_sets{};
};

GaussianSplatting::TileRenderParams::TileRenderParams() {
	viewport_size = Vector2i();
	world_to_camera_transform = Transform3D();
	projection = Projection();
	render_projection = projection;
	tile_size = TileRenderer::DEFAULT_TILE_SIZE;
	gaussian_buffer = RID();
	sorted_indices = RID();
	instance_buffer = RID();
	splat_ref_buffer = RID();
	quantization_buffer = RID();
	instance_indirect_count_buffer = RID();
	instance_indirect_dispatch_buffer = RID();
	interactive_state_uniform = RID();
	scene_uniform_buffer = RID();
	directional_light_buffer = RID();
	cluster_buffer = RID();
	shadow_atlas = RID();
	total_gaussians = 0;
	frame_serial = 0;
	omni_light_count = 0;
	spot_light_count = 0;
	cluster_size = 0;
	cluster_max_elements = 0;
	light_mask = 0xFFFFFFFFu;
	debug_splat_audit_sample_count = TileRenderer::DEBUG_SPLAT_AUDIT_MAX_SAMPLES;
	debug_enable_tile_logs = false;
	debug_enable_tile_pipeline_logs = false;
	debug_enable_tile_dispatch_logs = false;
	debug_enable_gpu_counter_logs = false;
	debug_frame_log_frequency = 0;
	compute_raster_policy = ComputeRasterPolicy::Default;
	request_packed_stage_data = false;
	request_tighter_bounds = false;
	request_sh_amortization = false;
	sh_amortization_divisor = 1;
	alpha_floor = 0.0f;
	force_solid_coverage = false;
	opacity_multiplier = 1.0f;
	direct_light_scale = 0.5f;
	indirect_sh_scale = 1.0f;
	shadow_strength = 1.0f;
	sh_dc_logit = false;
	shadow_receiver_bias_scale = 0.2f;
	shadow_receiver_bias_min = 0.0f;
	shadow_receiver_bias_max = 0.0f;
	enable_direct_lighting = true;
	normal_mode = 0;
	direct_lighting_mode = 1;
	cull_far_tolerance = 0.05f;
	tiny_splat_screen_radius = 0.3f;  // Drop subpixel splats to prevent tile overflow (#797)
	max_conic_aspect = 10.0f;
	low_pass_filter = 0.35f;
	jacobian_bypass_radius_depth_floor = false;
	jacobian_bypass_j_col2_clamp = false;
	jacobian_invert_j_col2_sign = false;
	// Opacity-aware bounding (FlashGS optimization) - enabled by default
	opacity_aware_culling = true;
	visibility_threshold = 0.01f;
	// Distance-based culling - enabled by default
	// Uses probabilistic culling to prevent per-tile overflow at distance
	distance_cull_enabled = true;
	distance_cull_start = 30.0f;
	distance_cull_max_rate = 0.5f;
	// LOD blending (LODGE technique) - enabled by default
	lod_blend_enabled = true;
	lod_blend_factor = 1.0f;
	lod_blend_distance = 5.0f;
}

TileRenderer::TileRenderer() : debug_stats(*this), params_builder(*this), prefix_scan_stage(*this), binning_stage(*this), raster_stage(*this),
        resolve_stage(*this), adaptive_controller(*this), async_readback(*this), global_sort_resources(*this),
        uniform_buffers(*this), projection_buffers(*this), sh_cache_buffers(*this), subpixel_history_buffers(*this),
        subpixel_visibility_buffers(*this), shader_resources(*this), render_targets(*this) {
    adaptive_controller.reset_state(config_state.tile_size);
    device_context.resource_rd = nullptr;
    device_context.submission_rd = nullptr;
    device_context.warned_missing_submission_device = false;
    device_context.performance_monitor = nullptr;
    frame_state = TileFrameState();
    shader_compilation_manager = std::make_unique<ShaderCompilationManager>(*this);

    // Register with performance monitors for editor debugger
    if (GaussianSplattingPerformanceMonitors *monitors = GaussianSplattingPerformanceMonitors::get_singleton()) {
        monitors->register_renderer(this);
    }
}

void TileRenderer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_debug_log_resolve", "enabled"), &TileRenderer::set_debug_log_resolve);
    ClassDB::bind_method(D_METHOD("get_debug_log_resolve"), &TileRenderer::get_debug_log_resolve);
    ClassDB::bind_method(D_METHOD("set_resolve_debug_visualize_tiles", "enabled"), &TileRenderer::set_resolve_debug_visualize_tiles);
    ClassDB::bind_method(D_METHOD("is_resolve_debug_visualize_tiles_enabled"), &TileRenderer::is_resolve_debug_visualize_tiles_enabled);
    ClassDB::bind_method(D_METHOD("set_resolve_use_texel_fetch", "enabled"), &TileRenderer::set_resolve_use_texel_fetch);
    ClassDB::bind_method(D_METHOD("is_resolve_use_texel_fetch_enabled"), &TileRenderer::is_resolve_use_texel_fetch_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_log_resolve"), "set_debug_log_resolve", "get_debug_log_resolve");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "resolve_debug_visualize_tiles"), "set_resolve_debug_visualize_tiles", "is_resolve_debug_visualize_tiles_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "resolve_use_texel_fetch"), "set_resolve_use_texel_fetch", "is_resolve_use_texel_fetch_enabled");
}

void TileRenderer::set_debug_log_resolve(bool p_enabled) {
    diagnostics.debug_log_resolve = p_enabled;
    // Throttle resolve logging aggressively to avoid per-frame spam in performance runs.
    // Default: log every 120 frames when enabled.
    diagnostics.debug_log_resolve_interval_frames = 120;
}

void TileRenderer::set_resolve_debug_visualize_tiles(bool p_enabled) {
    diagnostics.resolve_debug_visualize_tiles = p_enabled;
}

void TileRenderer::set_resolve_use_texel_fetch(bool p_enabled) {
    diagnostics.resolve_use_texel_fetch_sampling = p_enabled;
}

void TileRenderer::set_debug_binning_counters_enabled(bool p_enabled) {
	if (diagnostics.debug_binning_counters_enabled == p_enabled) {
		return;
	}
	diagnostics.debug_binning_counters_enabled = p_enabled;
	shader_resources.reset_state();
	_invalidate_descriptor_cache();
}

void TileRenderer::_on_overflow_flag_readback(const Vector<uint8_t> &p_data, int64_t p_request_frame_serial) {
    const uint64_t frame_serial = p_request_frame_serial > 0 ? uint64_t(p_request_frame_serial) : 0;
    async_readback.on_overflow_flag_readback(p_data, frame_serial);
}

void TileRenderer::_on_debug_counters_readback(const Vector<uint8_t> &p_data) {
    debug_stats.on_debug_counters_readback(p_data);
}

void TileRenderer::_on_overflow_stats_readback(const Vector<uint8_t> &p_data) {
    debug_stats.on_overflow_stats_readback(p_data);
}

void TileRenderer::_on_splat_audit_readback(const Vector<uint8_t> &p_data) {
    debug_stats.on_splat_audit_readback(p_data);
}

void TileRenderer::_on_tile_counts_readback(const Vector<uint8_t> &p_data, int64_t p_request_frame_serial) {
    const uint64_t frame_serial = p_request_frame_serial > 0 ? uint64_t(p_request_frame_serial) : 0;
    async_readback.on_tile_counts_readback(p_data, frame_serial);
}

RID TileRenderer::get_output_texture() const {
    const bool force_resolved = (render_settings.resolve_debug_mode == RESOLVE_DEBUG_OUTPUT);
    const bool prefer_raw = (render_settings.resolve_debug_mode == RESOLVE_DEBUG_INPUT) || (!shader_resources.resolve_pipeline_initialized && !force_resolved);
    return render_targets.get_output_texture(prefer_raw, force_resolved);
}

RID TileRenderer::get_depth_texture() const {
    const bool force_resolved = (render_settings.resolve_debug_mode == RESOLVE_DEBUG_OUTPUT);
    const bool resolved_depth_unavailable = !render_targets.depth_texture_copy_compatible;
    const bool prefer_raw = (render_settings.resolve_debug_mode == RESOLVE_DEBUG_INPUT) ||
            (!shader_resources.resolve_pipeline_initialized && !force_resolved) ||
            (!force_resolved && resolved_depth_unavailable);
    return render_targets.get_depth_texture(prefer_raw, force_resolved);
}

RenderingDevice *TileRenderer::get_output_texture_owner() const {
    const bool force_resolved = (render_settings.resolve_debug_mode == RESOLVE_DEBUG_OUTPUT);
    const bool prefer_raw = (render_settings.resolve_debug_mode == RESOLVE_DEBUG_INPUT) || (!shader_resources.resolve_pipeline_initialized && !force_resolved);
    return render_targets.get_output_texture_owner(prefer_raw, force_resolved);
}

RenderingDevice *TileRenderer::get_depth_texture_owner() const {
    const bool force_resolved = (render_settings.resolve_debug_mode == RESOLVE_DEBUG_OUTPUT);
    const bool resolved_depth_unavailable = !render_targets.depth_texture_copy_compatible;
    const bool prefer_raw = (render_settings.resolve_debug_mode == RESOLVE_DEBUG_INPUT) ||
            (!shader_resources.resolve_pipeline_initialized && !force_resolved) ||
            (!force_resolved && resolved_depth_unavailable);
    return render_targets.get_depth_texture_owner(prefer_raw, force_resolved);
}

bool TileRenderer::_is_main_rendering_device(RenderingDevice *p_device) {
    if (!p_device) {
        return false;
    }

    if (RenderingDevice *singleton = RenderingDevice::get_singleton()) {
        return p_device == singleton;
    }

    return false;
}

TileRenderer::~TileRenderer() {
    _clear_adaptive_overlap_budget_state(this);
    // Unregister from performance monitors
    if (GaussianSplattingPerformanceMonitors *monitors = GaussianSplattingPerformanceMonitors::get_singleton()) {
        monitors->unregister_renderer(this);
    }
    cleanup();
}

Error TileRenderer::initialize(RenderingDevice *p_rendering_device, const Vector2i &p_initial_viewport, int p_tile_size,
        RD::DataFormat p_format, RenderingDevice *p_submission_device) {
    ERR_FAIL_NULL_V_MSG(p_rendering_device, ERR_INVALID_PARAMETER, "[TileRenderer] Rendering device is required for initialization");

    device_context.resource_rd = p_rendering_device;
    device_context.submission_rd = p_submission_device ? p_submission_device : p_rendering_device;
    device_context.warned_missing_submission_device = false;

    RenderingDevice *device = _get_resource_device();
    ERR_FAIL_NULL_V_MSG(device, ERR_UNCONFIGURED, "[TileRenderer] Unable to acquire rendering device for initialization");
    config_state.tile_size = p_tile_size > 0 ? p_tile_size : DEFAULT_TILE_SIZE;
    adaptive_controller.reset_state(config_state.tile_size);

    if (p_format != RD::DATA_FORMAT_MAX) {
        config_state.desired_output_format = p_format;
    } else {
        config_state.desired_output_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    }
    config_state.output_format = RD::DATA_FORMAT_MAX;
    render_settings.allow_compute_raster = false;

    Error err = _compile_tile_shaders();
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("[TileRenderer] Failed to compile tile renderer shaders");
        return err;
    }

    if (p_initial_viewport.x > 0 && p_initial_viewport.y > 0) {
        err = _ensure_resources(p_initial_viewport, config_state.tile_size, config_state.desired_output_format);
        if (err != OK) {
            return err;
        }
    } else {
        grid_state.viewport_size = Vector2i();
        grid_state.tiles_x = 0;
        grid_state.tiles_y = 0;
        grid_state.total_tiles = 0;
    }

    return OK;
}

void TileRenderer::cleanup() {
	clear_output_resource_tracking();
	_clear_adaptive_overlap_budget_state(this);
    RenderingDevice *device = _get_resource_device();
    RenderingDevice *pipeline_owner = shader_resources.shader_device ? shader_resources.shader_device : device;
    if (device) {
        projection_buffers.release(device);
        sh_cache_buffers.release(device);
		subpixel_history_buffers.release(device);
		subpixel_visibility_buffers.release(device);
        debug_stats.free_buffers(device);
        global_sort_resources.release(device);
        render_settings.global_sort_enabled = false;
        shader_resources.release(device, pipeline_owner);
        if (render_targets.tile_framebuffer.is_valid() &&
                device->framebuffer_is_valid(render_targets.tile_framebuffer)) {
            device->free(render_targets.tile_framebuffer);
            render_targets.tile_framebuffer = RID();
        } else {
            render_targets.tile_framebuffer = RID();
        }
        if (resolve_stage.resolve_sampler.is_valid()) {
            RenderingDevice *sampler_device = resolve_stage.resolve_sampler_owner.device
                    ? resolve_stage.resolve_sampler_owner.device
                    : device;
            if (sampler_device) {
                sampler_device->free(resolve_stage.resolve_sampler);
            }
            resolve_stage.resolve_sampler = RID();
            resolve_stage.resolve_sampler_owner.clear();
        }
        if (resolve_stage.shadow_sampler.is_valid()) {
            RenderingDevice *shadow_sampler_device = resolve_stage.shadow_sampler_owner.device
                    ? resolve_stage.shadow_sampler_owner.device
                    : device;
            if (shadow_sampler_device) {
                shadow_sampler_device->free(resolve_stage.shadow_sampler);
            }
            resolve_stage.shadow_sampler = RID();
            resolve_stage.shadow_sampler_owner.clear();
        }
        resolve_stage.free_fallback_lighting_buffers(device);
        uniform_buffers.release(device);
    } else {
        uniform_buffers.reset_state();
    }

    _invalidate_descriptor_cache();

	projection_buffers.reset_state();
	sh_cache_buffers.reset_state();
	subpixel_history_buffers.reset_state();
	subpixel_visibility_buffers.reset_state();
    shader_resources.reset_state();
	render_settings.global_sort_enabled = true;
    render_settings.enable_sh_amortization = false;
    render_settings.sh_amortization_divisor = 1;
    sh_cache_needs_full_update = false;
	if (!device) {
		global_sort_resources.reset_state(true);
	}
	_destroy_output_textures();
    render_targets.depth_texture_copy_compatible = false;
    render_targets.tile_framebuffer = RID();
    render_targets.tile_framebuffer_format = RD::INVALID_ID;
    grid_state = TileGridState();
    perf_metrics = TilePerformanceMetrics();
    diagnostics = TileDiagnosticsState();
    config_state = TileConfigState();
    adaptive_controller.reset_state(config_state.tile_size);

    device_context.resource_rd = nullptr;
    device_context.submission_rd = nullptr;
    device_context.warned_missing_submission_device = false;
    device_context.performance_monitor = nullptr;
    frame_state = TileFrameState();
}

Error TileRenderer::resize(const Vector2i &p_size, RD::DataFormat p_format) {
    if (p_size.x <= 0 || p_size.y <= 0) {
        return ERR_INVALID_PARAMETER;
    }

    if (p_format != RD::DATA_FORMAT_MAX) {
        config_state.desired_output_format = p_format;
    }

    return _ensure_resources(p_size, config_state.tile_size, config_state.desired_output_format);
}

void TileRenderer::set_output_format(RD::DataFormat p_format) {
    if (p_format == RD::DATA_FORMAT_MAX) {
        p_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    }

    if (p_format == config_state.desired_output_format) {
        return;
    }

    config_state.desired_output_format = p_format;

    RenderingDevice *device = _get_resource_device();
    if (device && grid_state.viewport_size.x > 0 && grid_state.viewport_size.y > 0) {
        _ensure_resources(grid_state.viewport_size, config_state.tile_size, config_state.desired_output_format);
    }
}

void TileRenderer::set_output_invalidation_callback(std::function<void()> p_callback) {
    output_invalidation_callback = p_callback;
}

void TileRenderer::clear_output_invalidation_callback() {
    output_invalidation_callback = nullptr;
}

void TileRenderer::set_adaptive_settings(const AdaptiveSettings &p_settings) {
    adaptive_controller.set_settings(p_settings, config_state.tile_size);
}

Error TileRenderer::_ensure_resources(const Vector2i &p_size, int p_tile_size, RD::DataFormat p_format) {
    RenderingDevice *device = _get_resource_device();
    ERR_FAIL_NULL_V(device, ERR_UNCONFIGURED);

    if (p_size.x <= 0 || p_size.y <= 0) {
        return ERR_INVALID_PARAMETER;
    }

    const AdaptiveSettings &settings = adaptive_controller.get_settings();
    int requested_tile_size = p_tile_size > 0 ? p_tile_size : config_state.tile_size;
    int adaptive_tile_size = _compute_adaptive_tile_size(requested_tile_size, p_size);
    adaptive_tile_size = CLAMP(adaptive_tile_size, settings.min_tile_size, settings.max_tile_size);
    if (settings.clamp_to_power_of_two && adaptive_tile_size > 0) {
        int lower = 1;
        while ((lower << 1) <= adaptive_tile_size) {
            lower <<= 1;
        }
        int upper = lower << 1;
        if (lower < settings.min_tile_size) {
            lower = settings.min_tile_size;
        }
        if (upper < lower) {
            upper = lower;
        }
        if (upper > settings.max_tile_size) {
            upper = settings.max_tile_size;
        }
        int lower_diff = std::abs(adaptive_tile_size - lower);
        int upper_diff = std::abs(adaptive_tile_size - upper);
        adaptive_tile_size = (upper_diff < lower_diff) ? upper : lower;
    }

    bool tile_size_changed = adaptive_tile_size != config_state.tile_size;
    int previous_tile_size = config_state.tile_size;
    if (tile_size_changed) {
        config_state.tile_size = adaptive_tile_size;
        adaptive_controller.on_tile_size_applied(config_state.tile_size, true);
    } else {
        adaptive_controller.on_tile_size_applied(config_state.tile_size, false);
    }

    RD::DataFormat requested_format = p_format;
    if (requested_format == RD::DATA_FORMAT_MAX) {
        requested_format = config_state.desired_output_format;
    }
    if (requested_format == RD::DATA_FORMAT_MAX) {
        requested_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    }

    config_state.desired_output_format = requested_format;
    bool format_changed = requested_format != config_state.output_format || !render_targets.output_texture.is_valid();

    // Check if viewport size actually changed (not just initial difference)
    bool size_changed = grid_state.viewport_size.x > 0 && grid_state.viewport_size.y > 0 && grid_state.viewport_size != p_size;

    Error err = _compile_tile_shaders();
    if (err != OK) {
        return err;
    }

	bool recreate_buffers = tile_size_changed || format_changed || size_changed;
	// Debug/diagnostic buffers are required in both modes.
	recreate_buffers = recreate_buffers || !debug_stats.debug_counter_buffer.is_valid() ||
            !debug_stats.overflow_statistics_buffer.is_valid() || !debug_stats.debug_splat_audit_buffer.is_valid();

	if (recreate_buffers) {
		projection_buffers.release(device);
		debug_stats.free_buffers(device);
		_destroy_output_textures();
		render_targets.depth_texture_copy_compatible = false;

        _update_tile_dimensions(p_size);
        if (!_validate_tile_grid("ensure_resources")) {
            return ERR_INVALID_PARAMETER;
        }
        _create_aux_buffers();
        _create_output_texture(p_size, config_state.desired_output_format);

		bool allocation_failed = !render_targets.output_texture.is_valid() || !debug_stats.debug_counter_buffer.is_valid() ||
                !debug_stats.overflow_statistics_buffer.is_valid() || !debug_stats.debug_splat_audit_buffer.is_valid();
        if (allocation_failed && tile_size_changed) {
            GS_LOG_WARN_DEFAULT("[TileRenderer] Adaptive tile size allocation failed, reverting to previous tile size");
            config_state.tile_size = previous_tile_size;
            adaptive_controller.on_allocation_failure(config_state.tile_size);
            _update_tile_dimensions(p_size);
            if (!_validate_tile_grid("ensure_resources_fallback")) {
                return ERR_INVALID_PARAMETER;
            }
            _create_aux_buffers();
            _create_output_texture(p_size, config_state.desired_output_format);
        }
    }

    return OK;
}

int TileRenderer::_compute_adaptive_tile_size(int p_requested_tile_size, const Vector2i &p_size) {
    return adaptive_controller.compute_tile_size(p_requested_tile_size, p_size);
}

void TileRenderer::_update_tile_dimensions(const Vector2i &p_size) {
    grid_state.viewport_size = p_size;
    if (config_state.tile_size <= 0) {
        grid_state.tiles_x = 0;
        grid_state.tiles_y = 0;
        grid_state.total_tiles = 0;
        grid_state.total_tiles_wide = 0;
        grid_state.tile_grid_overflow = false;
        return;
    }
    grid_state.tiles_x = (grid_state.viewport_size.x + config_state.tile_size - 1) / config_state.tile_size;
    grid_state.tiles_y = (grid_state.viewport_size.y + config_state.tile_size - 1) / config_state.tile_size;
    grid_state.total_tiles_wide = uint64_t(grid_state.tiles_x) * uint64_t(grid_state.tiles_y);
    grid_state.tile_grid_overflow = grid_state.total_tiles_wide > UINT32_MAX;
    grid_state.total_tiles = grid_state.tile_grid_overflow ? UINT32_MAX : static_cast<uint32_t>(grid_state.total_tiles_wide);
}

bool TileRenderer::_validate_tile_grid(const char *p_context) const {
    const char *context = p_context ? p_context : "unknown";
    if (config_state.tile_size <= 0) {
        ERR_PRINT(vformat("[TileRenderer] Invalid tile size in %s (config_state.tile_size=%d)", context, config_state.tile_size));
#ifdef DEV_ENABLED
        DEV_ASSERT(config_state.tile_size > 0);
#endif
        return false;
    }
    if (grid_state.viewport_size.x <= 0 || grid_state.viewport_size.y <= 0) {
        ERR_PRINT(vformat("[TileRenderer] Invalid viewport in %s (viewport=%d x %d)", context,
                grid_state.viewport_size.x, grid_state.viewport_size.y));
#ifdef DEV_ENABLED
        DEV_ASSERT(grid_state.viewport_size.x > 0 && grid_state.viewport_size.y > 0);
#endif
        return false;
    }
    if (grid_state.tiles_x == 0 || grid_state.tiles_y == 0) {
        ERR_PRINT(vformat("[TileRenderer] Invalid tile grid in %s (tiles=%d x %d)", context, grid_state.tiles_x, grid_state.tiles_y));
#ifdef DEV_ENABLED
        DEV_ASSERT(grid_state.tiles_x > 0 && grid_state.tiles_y > 0);
#endif
        return false;
    }
    if (grid_state.tile_grid_overflow || grid_state.total_tiles_wide > UINT32_MAX) {
        ERR_PRINT(vformat("[TileRenderer] Tile grid overflow in %s (tiles=%d x %d, total=%s)",
                context, grid_state.tiles_x, grid_state.tiles_y, String::num_uint64(grid_state.total_tiles_wide)));
#ifdef DEV_ENABLED
        DEV_ASSERT(!grid_state.tile_grid_overflow);
#endif
        return false;
    }
    if (grid_state.total_tiles == 0 || grid_state.total_tiles != grid_state.total_tiles_wide) {
        ERR_PRINT(vformat("[TileRenderer] Tile grid mismatch in %s (grid_state.total_tiles=%d, grid_state.total_tiles_wide=%s)",
                context, grid_state.total_tiles, String::num_uint64(grid_state.total_tiles_wide)));
#ifdef DEV_ENABLED
        DEV_ASSERT(grid_state.total_tiles == grid_state.total_tiles_wide);
#endif
        return false;
    }
    return true;
}

bool TileRenderer::_validate_resource_owner(const RID &p_resource, const BufferOwnership &p_owner, RenderingDevice *p_device,
        const char *p_label) const {
    if (!p_resource.is_valid()) {
        return true;
    }
    if (!p_owner.device) {
        ERR_PRINT(vformat("[TileRenderer] Resource owner missing (%s)", p_label ? p_label : "unknown"));
#ifdef DEV_ENABLED
        DEV_ASSERT(p_owner.device != nullptr);
#endif
        return false;
    }
    if (!p_device) {
        return true;
    }
    if (!p_owner.matches(p_device)) {
        uint64_t expected_id = p_owner.device_id;
        uint64_t actual_id = p_device->get_device_instance_id();
        ERR_PRINT(vformat("[TileRenderer] Resource owner mismatch (%s): expected_device_id=%s actual_device_id=%s",
                p_label ? p_label : "unknown", String::num_uint64(expected_id), String::num_uint64(actual_id)));
#ifdef DEV_ENABLED
        DEV_ASSERT(p_owner.matches(p_device));
#endif
        return false;
    }
    return true;
}

bool TileRenderer::_validate_local_owner(const RID &p_resource, RenderingDevice *p_owner_device, RenderingDevice *p_expected_device,
        const char *p_label) const {
    if (!p_resource.is_valid()) {
        return true;
    }
    if (!p_owner_device) {
        ERR_PRINT(vformat("[TileRenderer] Local resource owner missing (%s)", p_label ? p_label : "unknown"));
#ifdef DEV_ENABLED
        DEV_ASSERT(p_owner_device != nullptr);
#endif
        return false;
    }
    if (!p_expected_device) {
        return true;
    }
    if (p_owner_device != p_expected_device) {
        uint64_t expected_id = p_expected_device->get_device_instance_id();
        uint64_t actual_id = p_owner_device->get_device_instance_id();
        ERR_PRINT(vformat("[TileRenderer] Local resource owner mismatch (%s): expected_device_id=%s actual_device_id=%s",
                p_label ? p_label : "unknown", String::num_uint64(expected_id), String::num_uint64(actual_id)));
#ifdef DEV_ENABLED
        DEV_ASSERT(p_owner_device == p_expected_device);
#endif
        return false;
    }
    return true;
}

/* static */ bool TileRenderer::_verify_texture_device_ownership(RenderingDevice *p_rd, RID p_resource, const char *p_label) {
    return ResourceOwnerMismatchContract::verify_texture_device_ownership(p_rd, p_resource, p_label);
}

/* static */ bool TileRenderer::_verify_buffer_device_ownership(RenderingDevice *p_rd, RID p_resource, const char *p_label) {
    return ResourceOwnerMismatchContract::verify_buffer_device_ownership(p_rd, p_resource, p_label);
}

void TileRenderer::_create_aux_buffers() {
	debug_stats.create_buffers(_get_resource_device());
}

void TileRenderer::_destroy_output_textures() {
    clear_output_resource_tracking();
    if (output_invalidation_callback) {
        output_invalidation_callback();
    }
    render_targets.destroy_output_textures();
}

void TileRenderer::_create_output_texture(const Vector2i &p_size, RD::DataFormat p_format) {
    render_targets.create_output_textures(p_size, p_format);
}

Vector<String> TileRenderer::_build_common_shader_defines(bool p_include_dispatch_group) const {
    Vector<String> defines;
    defines.push_back(vformat("#define GS_TILE_SIZE %d\n", config_state.tile_size));
    defines.push_back(vformat("#define GS_TILE_LOCAL_SIZE_X %d\n", config_state.tile_size));
    defines.push_back(vformat("#define GS_TILE_LOCAL_SIZE_Y %d\n", config_state.tile_size));
    if (p_include_dispatch_group) {
        defines.push_back(vformat("#define GS_DISPATCH_LOCAL_SIZE_X %d\n", int(BINNING_GROUP_SIZE)));
    }
    // Enable subgroup optimizations if GPU supports them
    if (shader_resources.subgroups_available) {
        defines.push_back("#define GS_ENABLE_SUBGROUPS 1\n");
    }
    // Disable debug counters in production builds for significant performance gains
    // (eliminates ~20 atomic operations per splat that cause L2 cache thrashing)
	if (!diagnostics.debug_binning_counters_enabled) {
		defines.push_back("#define GS_DEBUG_COUNTERS_DISABLED 1\n");
	}
	if (render_settings.enable_packed_stage_data) {
		defines.push_back("#define GS_PACKED_STAGE_DATA 1\n");
	}
	if (render_settings.enable_tighter_bounds) {
		defines.push_back("#define GS_TIGHTER_BOUNDS 1\n");
	}
	if (render_settings.enable_sh_amortization) {
		defines.push_back("#define GS_SH_AMORTIZATION 1\n");
	}
	if (g_quantization_config.per_chunk_quantization) {
		defines.push_back("#define USE_QUANTIZED_GAUSSIANS 1\n");
	}
	if (g_sh_config.dc_is_logit) {
		defines.push_back("#define GS_DC_LOGIT 1\n");
	}
	RendererRD::LightStorage *light_storage = RendererRD::LightStorage::get_singleton();
	uint32_t max_directional_lights = light_storage ? light_storage->get_max_directional_lights() : 1u;
	if (max_directional_lights == 0u) {
		max_directional_lights = 1u;
	}
	defines.push_back(vformat("#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS %d\n", max_directional_lights));
	defines.push_back(vformat("#define GS_MAX_OMNI_LIGHTS %d\n", uint32_t(MAX_OMNI_LIGHTS)));
	defines.push_back(vformat("#define GS_MAX_SPOT_LIGHTS %d\n", uint32_t(MAX_SPOT_LIGHTS)));
	return defines;
}

static uint32_t _required_tile_bits(uint32_t p_total_tiles) {
    if (p_total_tiles <= 1u) {
        return 0u;
    }
    uint32_t bits = 0u;
    uint32_t value = p_total_tiles - 1u;
    while (value > 0u) {
        ++bits;
        value >>= 1u;
    }
    return bits;
}

SortKeyConfig TileRenderer::_get_effective_sort_key_config() const {
    SortKeyConfig cfg = SortKeyConfig::from_settings();
    if (!render_settings.global_sort_enabled) {
        return cfg;
    }

    if (cfg.key_bits == 32) {
        uint32_t required_tile_bits = _required_tile_bits(grid_state.total_tiles);
        if (required_tile_bits > cfg.tile_bits) {
            WARN_PRINT_ONCE(vformat("[TileRenderer] 32-bit sort keys require tile_bits>=%d (tile_count=%d); falling back to 64-bit keys",
                    required_tile_bits, grid_state.total_tiles));
            cfg.key_bits = 64;
            cfg.tile_bits = 32;
            cfg.depth_bits = 32;
            cfg.enable_tie_breaker = true;
        } else if (cfg.depth_bits == 0) {
            WARN_PRINT_ONCE("[TileRenderer] 32-bit sort keys require depth_bits>0; falling back to 64-bit keys");
            cfg.key_bits = 64;
            cfg.tile_bits = 32;
            cfg.depth_bits = 32;
            cfg.enable_tie_breaker = true;
        } else if (cfg.depth_bits < 24) {
            WARN_PRINT_ONCE(vformat("[TileRenderer] 32-bit tile sort depth_bits=%d can cause motion shimmer; promoting to 64-bit keys",
                    cfg.depth_bits));
            cfg.key_bits = 64;
            cfg.tile_bits = 32;
            cfg.depth_bits = 32;
            cfg.enable_tie_breaker = true;
        }
    }

    return cfg;
}

Vector<String> TileRenderer::_build_binning_shader_defines() const {
	Vector<String> defines = _build_common_shader_defines(true);
	defines.push_back(vformat("#define GS_TILE_SPLAT_CAPACITY %d\n", MAX_SPLATS_PER_TILE));
	SortKeyConfig key_config = _get_effective_sort_key_config();
	defines.push_back(vformat("#define GS_SORT_KEY_BITS %d\n", key_config.key_bits));
	defines.push_back(vformat("#define GS_SORT_TILE_BITS %d\n", key_config.tile_bits));
	defines.push_back(vformat("#define GS_SORT_DEPTH_BITS %d\n", key_config.depth_bits));
	if (key_config.enable_tie_breaker) {
		// Keep most depth precision intact while injecting a deterministic low-bit tie-break.
		const uint32_t tie_bits = (key_config.key_bits > 32) ? 8u : 4u;
		defines.push_back("#define GS_SORT_TIE_BREAKER 1\n");
		defines.push_back(vformat("#define GS_SORT_TIE_BITS %d\n", tie_bits));
	}
	if (render_settings.global_sort_enabled) {
		defines.push_back("#define GS_TILE_GLOBAL_SORT 1\n");
		defines.push_back("#define GS_TILE_GLOBAL_SORT_EMIT_PASS 1\n");
	}
	return defines;
}

Vector<String> TileRenderer::_build_raster_shader_defines() const {
	Vector<String> defines = _build_common_shader_defines(false);
	defines.push_back(vformat("#define GS_TILE_SPLAT_CAPACITY %d\n", MAX_SPLATS_PER_TILE));
	const uint32_t raster_cap = CLAMP<uint32_t>(g_gpu_sorting_config.max_raster_splats_per_tile, 256u, 131072u);
	defines.push_back(vformat("#define GS_MAX_RASTER_SPLATS_PER_TILE %d\n", raster_cap));
	if (render_settings.global_sort_enabled) {
		defines.push_back("#define GS_TILE_GLOBAL_SORT 1\n");
	}
	// Enable compile-time raster stats collection when debug counters are active.
	// Without this define, the rasterizer hot loop has zero counter variables and
	// zero stat branches, eliminating register pressure and warp divergence.
#if defined(DEV_ENABLED) || defined(DEBUG_ENABLED) || defined(TESTS_ENABLED)
	if (diagnostics.debug_binning_counters_enabled) {
		defines.push_back("#define GS_COLLECT_RASTER_STATS 1\n");
	}
#else
	if (diagnostics.debug_binning_counters_enabled) {
		WARN_PRINT_ONCE("[TileRenderer] Raster stats collection requested but disabled in this build configuration");
	}
#endif
	return defines;
}


void TileRenderer::_detect_subgroup_support(RenderingDevice *p_device) {
	if (!p_device) {
		return;
	}

	static HashMap<uint64_t, bool> subgroup_support_cache;
	uint64_t device_id = p_device->get_device_instance_id();
	const bool *cached = subgroup_support_cache.getptr(device_id);
	if (cached) {
		shader_resources.subgroups_available = *cached;
		return;
	}

	uint64_t subgroup_ops = p_device->limit_get(RenderingDevice::LIMIT_SUBGROUP_OPERATIONS);
	uint64_t subgroup_stages = p_device->limit_get(RenderingDevice::LIMIT_SUBGROUP_IN_SHADERS);

	bool has_basic = (subgroup_ops & RenderingDevice::SUBGROUP_BASIC_BIT) != 0;
	bool has_ballot = (subgroup_ops & RenderingDevice::SUBGROUP_BALLOT_BIT) != 0;
	bool has_shuffle = (subgroup_ops & RenderingDevice::SUBGROUP_SHUFFLE_BIT) != 0;
	bool has_vote = (subgroup_ops & RenderingDevice::SUBGROUP_VOTE_BIT) != 0;
	bool has_compute = (subgroup_stages & RenderingDevice::SHADER_STAGE_COMPUTE_BIT) != 0;
	bool has_fragment = (subgroup_stages & RenderingDevice::SHADER_STAGE_FRAGMENT_BIT) != 0;

	shader_resources.subgroups_available = has_basic && has_ballot && has_shuffle && has_vote && has_compute && has_fragment;
	subgroup_support_cache.insert(device_id, shader_resources.subgroups_available);

	if (GaussianSplatting::is_debug_frame_logging_enabled()) {
		if (shader_resources.subgroups_available) {
			GS_LOG_WARN_DEFAULT(vformat("[TileRenderer] Subgroup operations available (basic=%d, ballot=%d, shuffle=%d, vote=%d, compute=%d, fragment=%d) - using optimized path",
					(int)has_basic, (int)has_ballot, (int)has_shuffle, (int)has_vote, (int)has_compute, (int)has_fragment));
		} else {
			GS_LOG_WARN_DEFAULT(vformat("[TileRenderer] Subgroup operations unavailable (basic=%d, ballot=%d, shuffle=%d, vote=%d, compute=%d, fragment=%d) - using atomicAdd fallback",
					(int)has_basic, (int)has_ballot, (int)has_shuffle, (int)has_vote, (int)has_compute, (int)has_fragment));
		}
	}
}

bool TileRenderer::_check_pipeline_validity(RenderingDevice *p_device, bool p_want_compute_raster) const {
	bool device_changed = shader_resources.shader_device != nullptr && shader_resources.shader_device != p_device;
	bool pipelines_valid = shader_resources.tile_binning_shader.is_valid() && shader_resources.tile_binning_pipeline.is_valid() &&
			shader_resources.tile_raster_shader.is_valid() && !device_changed;
	if (!p_want_compute_raster) {
		pipelines_valid = pipelines_valid && shader_resources.tile_raster_pipeline.is_valid();
	}
	if (render_settings.global_sort_enabled) {
		pipelines_valid = pipelines_valid && shader_resources.tile_binning_count_shader.is_valid() && shader_resources.tile_binning_count_pipeline.is_valid() &&
				shader_resources.tile_prefix_pipeline_pass1.is_valid() && shader_resources.tile_prefix_pipeline_pass2.is_valid() && shader_resources.tile_prefix_pipeline_pass3.is_valid();
	}
	if (p_want_compute_raster) {
		pipelines_valid = pipelines_valid && shader_resources.tile_raster_compute_shader.is_valid() && shader_resources.tile_raster_compute_pipeline.is_valid();
	}
	pipelines_valid = pipelines_valid && shader_resources.quantized_storage_enabled == g_quantization_config.per_chunk_quantization;
	pipelines_valid = pipelines_valid && shader_resources.shader_defines_hash == _compute_shader_defines_hash();
	return pipelines_valid;
}

void TileRenderer::_initialize_shader_sources() {
	if (!shader_resources.tile_binning_shader_initialized) {
		ERR_FAIL_NULL(shader_resources.tile_binning_shader_source.get());
		Vector<String> variants;
		variants.push_back("");
		shader_resources.tile_binning_shader_source->initialize(variants);
		shader_resources.tile_binning_shader_initialized = true;
	}

	if (!shader_resources.tile_raster_shader_initialized) {
		ERR_FAIL_NULL(shader_resources.tile_raster_shader_source.get());
		Vector<String> variants;
		variants.push_back("");
		shader_resources.tile_raster_shader_source->initialize(variants);
		shader_resources.tile_raster_shader_initialized = true;
	}
	if (!shader_resources.tile_raster_compute_shader_initialized) {
		ERR_FAIL_NULL(shader_resources.tile_raster_compute_shader_source.get());
		Vector<String> variants;
		variants.push_back("");
		shader_resources.tile_raster_compute_shader_source->initialize(variants);
		shader_resources.tile_raster_compute_shader_initialized = true;
	}
	if (!shader_resources.tile_prefix_shader_initialized) {
		ERR_FAIL_NULL(shader_resources.tile_prefix_shader_source.get());
		Vector<String> variants;
		variants.push_back("");
		shader_resources.tile_prefix_shader_source->initialize(variants);
		shader_resources.tile_prefix_shader_initialized = true;
	}
}

void TileRenderer::_free_existing_pipelines(RenderingDevice *p_shader_owner) {
	if (shader_resources.tile_binning_pipeline.is_valid()) {
		if (p_shader_owner && p_shader_owner->compute_pipeline_is_valid(shader_resources.tile_binning_pipeline)) {
			p_shader_owner->free(shader_resources.tile_binning_pipeline);
		}
		shader_resources.tile_binning_pipeline = RID();
	}
	if (shader_resources.tile_binning_count_pipeline.is_valid()) {
		if (p_shader_owner && p_shader_owner->compute_pipeline_is_valid(shader_resources.tile_binning_count_pipeline)) {
			p_shader_owner->free(shader_resources.tile_binning_count_pipeline);
		}
		shader_resources.tile_binning_count_pipeline = RID();
	}
	if (shader_resources.tile_prefix_pipeline_pass1.is_valid()) {
		if (p_shader_owner && p_shader_owner->compute_pipeline_is_valid(shader_resources.tile_prefix_pipeline_pass1)) {
			p_shader_owner->free(shader_resources.tile_prefix_pipeline_pass1);
		}
		shader_resources.tile_prefix_pipeline_pass1 = RID();
	}
	if (shader_resources.tile_prefix_pipeline_pass2.is_valid()) {
		if (p_shader_owner && p_shader_owner->compute_pipeline_is_valid(shader_resources.tile_prefix_pipeline_pass2)) {
			p_shader_owner->free(shader_resources.tile_prefix_pipeline_pass2);
		}
		shader_resources.tile_prefix_pipeline_pass2 = RID();
	}
	if (shader_resources.tile_prefix_pipeline_pass3.is_valid()) {
		if (p_shader_owner && p_shader_owner->compute_pipeline_is_valid(shader_resources.tile_prefix_pipeline_pass3)) {
			p_shader_owner->free(shader_resources.tile_prefix_pipeline_pass3);
		}
		shader_resources.tile_prefix_pipeline_pass3 = RID();
	}
	if (shader_resources.tile_prefix_shader_pass2.is_valid()) {
		if (p_shader_owner) {
			p_shader_owner->free(shader_resources.tile_prefix_shader_pass2);
		}
		shader_resources.tile_prefix_shader_pass2 = RID();
	}
	if (shader_resources.tile_prefix_shader_pass3.is_valid()) {
		if (p_shader_owner) {
			p_shader_owner->free(shader_resources.tile_prefix_shader_pass3);
		}
		shader_resources.tile_prefix_shader_pass3 = RID();
	}
	if (shader_resources.tile_raster_pipeline.is_valid()) {
		if (p_shader_owner && p_shader_owner->render_pipeline_is_valid(shader_resources.tile_raster_pipeline)) {
			p_shader_owner->free(shader_resources.tile_raster_pipeline);
		}
		shader_resources.tile_raster_pipeline = RID();
	}
	if (shader_resources.tile_raster_compute_pipeline.is_valid()) {
		if (p_shader_owner && p_shader_owner->compute_pipeline_is_valid(shader_resources.tile_raster_compute_pipeline)) {
			p_shader_owner->free(shader_resources.tile_raster_compute_pipeline);
		}
		shader_resources.tile_raster_compute_pipeline = RID();
	}

	if (shader_resources.tile_binning_shader.is_valid()) {
		if (p_shader_owner) {
			p_shader_owner->free(shader_resources.tile_binning_shader);
		}
		shader_resources.tile_binning_shader = RID();
	}
	if (shader_resources.tile_binning_count_shader.is_valid()) {
		if (p_shader_owner) {
			p_shader_owner->free(shader_resources.tile_binning_count_shader);
		}
		shader_resources.tile_binning_count_shader = RID();
	}
	if (shader_resources.tile_prefix_shader.is_valid()) {
		if (p_shader_owner) {
			p_shader_owner->free(shader_resources.tile_prefix_shader);
		}
		shader_resources.tile_prefix_shader = RID();
	}
	if (shader_resources.tile_raster_shader.is_valid()) {
		if (p_shader_owner) {
			p_shader_owner->free(shader_resources.tile_raster_shader);
		}
		shader_resources.tile_raster_shader = RID();
	}
	if (shader_resources.tile_raster_compute_shader.is_valid()) {
		if (p_shader_owner) {
			p_shader_owner->free(shader_resources.tile_raster_compute_shader);
		}
		shader_resources.tile_raster_compute_shader = RID();
	}
}

Error TileRenderer::_compile_binning_shaders(RenderingDevice *p_device) {
	ERR_FAIL_NULL_V(shader_compilation_manager, ERR_UNCONFIGURED);
	return shader_compilation_manager->compile_binning_shaders(p_device);
}

Error TileRenderer::_compile_prefix_shaders(RenderingDevice *p_device) {
	ERR_FAIL_NULL_V(shader_compilation_manager, ERR_UNCONFIGURED);
	return shader_compilation_manager->compile_prefix_shaders(p_device);
}

Error TileRenderer::_compile_raster_shaders(RenderingDevice *p_device, bool p_want_compute_raster) {
	ERR_FAIL_NULL_V(shader_compilation_manager, ERR_UNCONFIGURED);
	return shader_compilation_manager->compile_raster_shaders(p_device, p_want_compute_raster);
}

Error TileRenderer::_compile_tile_shaders() {
	ERR_FAIL_NULL_V(shader_compilation_manager, ERR_UNCONFIGURED);
	return shader_compilation_manager->compile_all_shaders(device_context.resource_rd);
}

uint64_t TileRenderer::_compute_shader_defines_hash() const {
	Vector<String> binning_defines = _build_binning_shader_defines();
	Vector<String> raster_defines = _build_raster_shader_defines();
	uint64_t seed = HASH_MURMUR3_SEED;
	seed = hash64_murmur3_64(_hash_shader_defines(binning_defines), seed);
	seed = hash64_murmur3_64(_hash_shader_defines(raster_defines), seed);
	return seed;
}

void TileRenderer::_clear_debug_counters() {
	debug_stats.clear_counters(_get_resource_device());
}

void TileRenderer::_update_splat_audit_buffer(const RenderParams &p_params) {
	debug_stats.update_splat_audit_buffer(_get_resource_device(), p_params, frame_state.current_frame_serial);
}

uint32_t TileRenderer::_get_projection_stride() const {
	return render_settings.enable_packed_stage_data ? TileProjectionLayout::STRIDE_PACKED : TileProjectionLayout::STRIDE_FULL;
}

uint64_t TileRenderer::_compute_projection_buffer_bytes(uint32_t p_visible_count, uint32_t &r_capacity) {
	const uint32_t required_elements = MAX<uint32_t>(p_visible_count, 1u);

	uint32_t target_capacity = 1u;
	while (target_capacity < required_elements && target_capacity < (1u << 31)) {
		target_capacity <<= 1u;
	}
	if (target_capacity < required_elements) {
		target_capacity = required_elements;
	}

	r_capacity = target_capacity;
	return uint64_t(target_capacity) * uint64_t(_get_projection_stride());
}

void TileRenderer::_ensure_global_projection_buffer(uint32_t p_visible_count) {
	projection_buffers.ensure_projection_buffer(p_visible_count);
}

void TileRenderer::_ensure_global_sort_resources(uint32_t p_visible_count) {
	global_sort_resources.ensure_resources(p_visible_count);
}
bool TileRenderer::_update_global_tile_ranges(const RID &p_gaussian_buffer, const RID &p_sorted_indices,
        RenderingDevice *p_device, uint32_t &r_record_count, uint32_t &r_raw_record_count, bool p_allow_sync_readback) {
    return prefix_scan_stage.update_global_tile_ranges(p_gaussian_buffer, p_sorted_indices, p_device, r_record_count,
            r_raw_record_count, p_allow_sync_readback);
}

void TileRenderer::_reset_timestamp_tracking() {
	timing_state.binning_timestamp.reset();
	timing_state.raster_timestamp.reset();
	timing_state.prefix_timestamp.reset();
	timing_state.resolve_timestamp.reset();
	if (!gpu_timestamp_capture_enabled) {
		timing_state.last_binning_gpu_ms = 0.0f;
		timing_state.last_raster_gpu_ms = 0.0f;
		timing_state.last_prefix_gpu_ms = 0.0f;
		timing_state.last_resolve_gpu_ms = 0.0f;
		timing_state.last_frame_gpu_ms = 0.0f;
		timing_state.gpu_timing_frame_serial = 0;
		timing_state.gpu_timing_frames_behind = 0;
		return;
	}
	// Keep last resolved values across frames. Timestamp availability can lag by
	// multiple frames on some backends; clearing every frame causes telemetry to
	// flap to zero and hides otherwise valid samples.
}

void TileRenderer::_resolve_timestamp_range(TimestampRange &p_range, float &r_duration_ms) {
    r_duration_ms = 0.0f;
    if (!p_range.is_valid()) {
        return;
    }

    RenderingDevice *device = p_range.device;
    if (!device) {
        p_range.reset();
        return;
    }

    uint32_t timestamp_count = device->get_captured_timestamps_count();
    if (p_range.start_index >= timestamp_count || p_range.end_index >= timestamp_count) {
        p_range.reset();
        return;
    }

    uint64_t start_gpu = device->get_captured_timestamp_gpu_time(p_range.start_index);
    uint64_t end_gpu = device->get_captured_timestamp_gpu_time(p_range.end_index);
    if (end_gpu > start_gpu) {
        r_duration_ms = (end_gpu - start_gpu) / 1000000.0f;
    }
#ifdef DEBUG_ENABLED
    if (!p_range.label.is_empty() && r_duration_ms > 0.0f) {
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[TileRenderer] %s GPU time: %.2f ms", p_range.label, r_duration_ms));
    }
#endif
    p_range.reset();
}

void TileRenderer::_parse_timestamps_into_frame_map(RenderingDevice *p_device, uint32_t p_available,
        HashMap<uint64_t, GpuTimestampFrameStages> &r_frames) const {
    if (!p_device) {
        return;
    }

    struct TimestampFieldEntry {
        const char *prefix;
        int prefix_length;
        const char *suffix;
        GpuTimestampStageTimes GpuTimestampFrameStages::*stage;
        double GpuTimestampStageTimes::*field;
    };

    static const TimestampFieldEntry timestamp_fields[] = {
        { "TileBinning_", int(sizeof("TileBinning_") - 1), "_Begin", &GpuTimestampFrameStages::binning, &GpuTimestampStageTimes::begin_ns },
        { "TileBinning_", int(sizeof("TileBinning_") - 1), "_End", &GpuTimestampFrameStages::binning, &GpuTimestampStageTimes::end_ns },
        { "TileRaster_", int(sizeof("TileRaster_") - 1), "_Begin", &GpuTimestampFrameStages::raster, &GpuTimestampStageTimes::begin_ns },
        { "TileRaster_", int(sizeof("TileRaster_") - 1), "_End", &GpuTimestampFrameStages::raster, &GpuTimestampStageTimes::end_ns },
        { "TileOverlapCount_", int(sizeof("TileOverlapCount_") - 1), "_Begin", &GpuTimestampFrameStages::overlap_count, &GpuTimestampStageTimes::begin_ns },
        { "TileOverlapCount_", int(sizeof("TileOverlapCount_") - 1), "_End", &GpuTimestampFrameStages::overlap_count, &GpuTimestampStageTimes::end_ns },
        { "TilePrefix_", int(sizeof("TilePrefix_") - 1), "_Begin", &GpuTimestampFrameStages::prefix, &GpuTimestampStageTimes::begin_ns },
        { "TilePrefix_", int(sizeof("TilePrefix_") - 1), "_End", &GpuTimestampFrameStages::prefix, &GpuTimestampStageTimes::end_ns },
        { "TileResolve_", int(sizeof("TileResolve_") - 1), "_Begin", &GpuTimestampFrameStages::resolve, &GpuTimestampStageTimes::begin_ns },
        { "TileResolve_", int(sizeof("TileResolve_") - 1), "_End", &GpuTimestampFrameStages::resolve, &GpuTimestampStageTimes::end_ns },
        { "GaussianSplat_", int(sizeof("GaussianSplat_") - 1), "_Begin", &GpuTimestampFrameStages::total, &GpuTimestampStageTimes::begin_ns },
        { "GaussianSplat_", int(sizeof("GaussianSplat_") - 1), "_End", &GpuTimestampFrameStages::total, &GpuTimestampStageTimes::end_ns },
    };

    auto parse_label = [](const String &p_label, const char *p_prefix, int p_prefix_length, const char *p_suffix, uint64_t &r_serial) -> bool {
        if (!p_label.begins_with(p_prefix)) {
            return false;
        }
        int64_t suffix_pos = p_label.rfind(p_suffix);
        if (suffix_pos < 0 || suffix_pos <= p_prefix_length) {
            return false;
        }
        String serial_str = p_label.substr(p_prefix_length, suffix_pos - p_prefix_length);
        r_serial = (uint64_t)serial_str.to_int();
        return true;
    };
    auto get_frame_entry = [&r_frames](uint64_t p_serial) -> GpuTimestampFrameStages * {
        GpuTimestampFrameStages *entry = r_frames.getptr(p_serial);
        if (!entry) {
            GpuTimestampFrameStages fresh;
            r_frames.insert(p_serial, fresh);
            entry = r_frames.getptr(p_serial);
        }
        return entry;
    };

    for (uint32_t i = 0; i < p_available; i++) {
        String name = p_device->get_captured_timestamp_name(i);
        double gpu_ns = double(p_device->get_captured_timestamp_gpu_time(i));

        for (const TimestampFieldEntry &entry : timestamp_fields) {
            uint64_t serial = 0;
            if (!parse_label(name, entry.prefix, entry.prefix_length, entry.suffix, serial)) {
                continue;
            }
            GpuTimestampFrameStages *frame_entry = get_frame_entry(serial);
            if (!frame_entry) {
                break;
            }
            GpuTimestampStageTimes &stage = frame_entry->*(entry.stage);
            stage.*(entry.field) = gpu_ns;
            break;
        }
    }
}

TileRenderer::GpuTimestampDurations TileRenderer::_compute_stage_durations(const HashMap<uint64_t, GpuTimestampFrameStages> &p_frames) const {
    GpuTimestampDurations durations;

    // Pick the latest frame serial with whatever timing survived the flush.
    for (const KeyValue<uint64_t, GpuTimestampFrameStages> &kv : p_frames) {
        const uint64_t serial = kv.key;
        const GpuTimestampFrameStages &stages = kv.value;

        // Check if we have valid markers (total, per-pass, or overlap count).
        // NOTE: Godot's buffer_get_data() flush resets the timestamp buffer mid-frame;
        // only pre-flush markers (TileOverlapCount) are reliably readable. Post-flush
        // markers (GaussianSplat_*, TileBinning_*, TileRaster_*) are usually discarded.
        // See docs/GPU_TIMESTAMP_PROFILING.md for details.
        bool has_total = stages.total.is_complete();
        bool has_per_pass = stages.binning.is_complete() && stages.raster.is_complete();
        bool has_overlap_count = stages.overlap_count.is_complete();
        bool has_prefix = stages.prefix.is_complete();
        bool has_resolve = stages.resolve.is_complete();

        if (!has_total && !has_per_pass && !has_overlap_count && !has_prefix && !has_resolve) {
            continue; // Need at least one complete timing marker pair
        }

        double bin_ms = has_per_pass ? (stages.binning.end_ns - stages.binning.begin_ns) / 1e6 : 0.0;
        double raster_ms = has_per_pass ? (stages.raster.end_ns - stages.raster.begin_ns) / 1e6 : 0.0;
        double count_ms = has_overlap_count ? (stages.overlap_count.end_ns - stages.overlap_count.begin_ns) / 1e6 : 0.0;
        double prefix_ms = has_prefix ? (stages.prefix.end_ns - stages.prefix.begin_ns) / 1e6 : 0.0;
        double resolve_ms = has_resolve ? (stages.resolve.end_ns - stages.resolve.begin_ns) / 1e6 : 0.0;
        double total_ms = has_total ? (stages.total.end_ns - stages.total.begin_ns) / 1e6 : 0.0;

        if (!durations.has_data || serial > durations.serial) {
            durations.has_data = true;
            durations.serial = serial;
            durations.bin_ms = bin_ms;
            durations.raster_ms = raster_ms;
            durations.count_ms = count_ms;
            durations.prefix_ms = prefix_ms;
            durations.resolve_ms = resolve_ms;
            durations.total_ms = total_ms;
        }
    }

    return durations;
}

void TileRenderer::_update_timing_metrics(const GpuTimestampDurations &p_durations) {
    if (!p_durations.has_data) {
        return;
    }

    timing_state.last_binning_gpu_ms = (float)p_durations.bin_ms;
    timing_state.last_raster_gpu_ms = (float)p_durations.raster_ms;
    timing_state.last_prefix_gpu_ms = (float)((p_durations.prefix_ms > 0.0) ? p_durations.prefix_ms : p_durations.count_ms);
    timing_state.last_resolve_gpu_ms = (float)p_durations.resolve_ms;
    float fallback_total = (float)(p_durations.bin_ms + p_durations.raster_ms + timing_state.last_prefix_gpu_ms + p_durations.resolve_ms);
    timing_state.last_frame_gpu_ms = (float)((p_durations.total_ms > 0.0) ? p_durations.total_ms : fallback_total);
    timing_state.gpu_timing_frame_serial = p_durations.serial;
    timing_state.gpu_timing_frames_behind = (frame_state.current_frame_serial > p_durations.serial)
            ? (frame_state.current_frame_serial - p_durations.serial)
            : 0;
}

void TileRenderer::resolve_gpu_timestamps_async() {
    RenderingDevice *device = _get_submission_device();
    if (!device) {
        return;
    }

    if (device_context.performance_monitor) {
        device_context.performance_monitor->set_rendering_device(device);
    }

    // NOTE: Godot resets timestamps each frame. get_captured_timestamps_count() returns
    // the count from the PREVIOUS frame's submission. We must read ALL timestamps from
    // index 0 each time, not incrementally.
    const uint32_t available = device->get_captured_timestamps_count();
    if (available == 0) {
        if (timing_state.gpu_timing_frame_serial > 0 && frame_state.current_frame_serial > timing_state.gpu_timing_frame_serial) {
            timing_state.gpu_timing_frames_behind = frame_state.current_frame_serial - timing_state.gpu_timing_frame_serial;
        }
        return;
    }

    HashMap<uint64_t, GpuTimestampFrameStages> frame_times;
    _parse_timestamps_into_frame_map(device, available, frame_times);

    GpuTimestampDurations durations = _compute_stage_durations(frame_times);
    _update_timing_metrics(durations);
    if (!durations.has_data && timing_state.gpu_timing_frame_serial > 0 && frame_state.current_frame_serial > timing_state.gpu_timing_frame_serial) {
        timing_state.gpu_timing_frames_behind = frame_state.current_frame_serial - timing_state.gpu_timing_frame_serial;
    }

    if (device_context.performance_monitor) {
        if (durations.has_data) {
            device_context.performance_monitor->record_completion(durations.serial, durations.serial);
        }
        device_context.performance_monitor->detect_pipeline_stalls(RID());
    }
}

RID TileRenderer::render(RenderingDevice *p_rendering_device, const RenderParams &p_params) {
    RenderingDevice *resource_device = _get_resource_device();
    ERR_FAIL_NULL_V_MSG(resource_device, RID(), "[TileRenderer] Rendering device is required");

    RenderFrameExecutor executor(*this, p_rendering_device, p_params, resource_device);
    return executor.run();
}

uint64_t TileRenderer::_dispatch_tile_binning(uint32_t p_gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
        RID p_lighting_uniform_set, RenderingDevice *p_submission_device, bool p_requires_sync) {
	return binning_stage.dispatch_tile_binning(p_gaussian_count, p_buffer_uniform_set, p_param_uniform_set, p_lighting_uniform_set, p_submission_device,
			p_requires_sync);
}

uint64_t TileRenderer::_dispatch_tile_binning_count(uint32_t p_gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
        RID p_lighting_uniform_set, RenderingDevice *p_submission_device, bool p_requires_sync) {
	return binning_stage.dispatch_tile_binning_count(p_gaussian_count, p_buffer_uniform_set, p_param_uniform_set, p_lighting_uniform_set, p_submission_device,
			p_requires_sync);
}

TileRenderer::RasterDecision TileRenderer::_evaluate_raster_path(RenderingDevice *p_device) const {
    RasterDecision decision;
    decision.use_compute = false;

    if (!p_device) {
        decision.reason = "Render device unavailable";
        return decision;
    }
    if (!render_settings.allow_compute_raster) {
        decision.reason = "Compute raster disabled by pipeline settings";
        return decision;
    }
    if (!shader_resources.tile_raster_compute_shader.is_valid() || !shader_resources.tile_raster_compute_pipeline.is_valid()) {
        decision.reason = "Compute raster shader/pipeline not compiled";
        return decision;
    }

    if (config_state.tile_size <= 0) {
        decision.reason = "Tile size not set";
        return decision;
    }

    if (!render_targets.output_texture.is_valid() || !render_targets.depth_texture.is_valid()) {
        decision.reason = "Output/depth textures are not allocated";
        return decision;
    }
    if (!p_device->texture_is_valid(render_targets.output_texture) ||
            !p_device->texture_is_valid(render_targets.depth_texture)) {
        decision.reason = "Output/depth textures are invalid on the device";
        return decision;
    }

    RD::TextureFormat output_tex_format = p_device->texture_get_format(render_targets.output_texture);
    const RD::DataFormat actual_output_format = output_tex_format.format;
    if (_is_srgb_format(actual_output_format) || _is_srgb_format(config_state.output_format)) {
        decision.reason = vformat("Compute raster blocked for sRGB output (requested=%d actual=%d); using fragment path until explicit conversion exists",
                int(config_state.output_format), int(actual_output_format));
        return decision;
    }

    if (actual_output_format != _resolve_storage_compatible_color_format(actual_output_format)) {
        decision.reason = vformat("Compute raster requires storage-compatible output format; actual format %d uses fragment path",
                int(actual_output_format));
        return decision;
    }

    if (actual_output_format != RD::DATA_FORMAT_R8G8B8A8_UNORM) {
        decision.reason = vformat("Compute raster supports RGBA8 UNORM imageStore only; actual format %d uses fragment path",
                int(actual_output_format));
        return decision;
    }

    const uint64_t max_workgroup_x = MAX<uint64_t>(1u, p_device->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X));
    const uint64_t max_workgroup_y = MAX<uint64_t>(1u, p_device->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Y));
    const uint64_t max_invocations = MAX<uint64_t>(1u, p_device->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS));
    const uint64_t tile_size_u64 = uint64_t(config_state.tile_size);
    if (tile_size_u64 > max_workgroup_x || tile_size_u64 > max_workgroup_y) {
        decision.reason = vformat("Tile size %d exceeds device workgroup size limit (max_x=%s max_y=%s)",
                config_state.tile_size, String::num_uint64(max_workgroup_x), String::num_uint64(max_workgroup_y));
        return decision;
    }
    if (tile_size_u64 * tile_size_u64 > max_invocations) {
        decision.reason = vformat("Tile size %d exceeds device workgroup invocation limit (%s)",
                config_state.tile_size, String::num_uint64(max_invocations));
        return decision;
    }

    if (!p_device->compute_pipeline_is_valid(shader_resources.tile_raster_compute_pipeline)) {
        decision.reason = "Compute raster pipeline validation failed";
        return decision;
    }

    decision.use_compute = true;
    decision.reason = "Compute raster enabled";
    return decision;
}

void TileRenderer::_log_raster_path_decision(const RasterDecision &p_decision) {

    if (perf_metrics.last_raster_choice_initialized &&
            perf_metrics.last_raster_choice_compute == p_decision.use_compute &&
            perf_metrics.last_raster_choice_reason == p_decision.reason) {
        return;
    }

    perf_metrics.last_raster_choice_initialized = true;
    perf_metrics.last_raster_choice_compute = p_decision.use_compute;
    perf_metrics.last_raster_choice_reason = p_decision.reason;

    if (p_decision.use_compute) {
        GS_LOG_INFO_DEFAULT(vformat("[TileRenderer] Raster path: compute (%s)", p_decision.reason));
    } else {
        GS_LOG_INFO_DEFAULT(vformat("[TileRenderer] Raster path: fragment (%s)", p_decision.reason));
    }
}

uint64_t TileRenderer::_dispatch_tile_rasterizer_compute(uint32_t p_gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
        RID p_image_uniform_set, RenderingDevice *p_submission_device) {
    return raster_stage.dispatch_tile_rasterizer_compute(p_gaussian_count, p_buffer_uniform_set, p_param_uniform_set,
            p_image_uniform_set, p_submission_device);
}

uint64_t TileRenderer::_dispatch_tile_rasterizer(uint32_t p_gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
        RenderingDevice *p_submission_device) {
    return raster_stage.dispatch_tile_rasterizer(p_gaussian_count, p_buffer_uniform_set, p_param_uniform_set, p_submission_device);
}

void TileRenderer::_dispatch_tile_resolve(const Vector2i &p_viewport, int p_tile_size, bool p_output_is_premultiplied,
        const RenderParams &p_params) {
    resolve_stage.dispatch_tile_resolve(p_viewport, p_tile_size, p_output_is_premultiplied, p_params);
}

void TileRenderer::_queue_submission(RenderingDevice *p_device, bool p_requires_sync) {
    if (!p_device) {
        return;
    }

    // Use safe submit - only submits on local devices (main device syncs automatically)
    gs_device_utils::safe_submit(p_device);

    // NOTE: diagnostics.runtime_statistics_enabled no longer forces sync - we use async readback
    // for tile counts statistics (see _collect_render_statistics and _on_tile_counts_readback).
    // This avoids the 40ms GPU stall that was caused by synchronous buffer readback.
    bool needs_sync = p_requires_sync;
    if (needs_sync) {
        gs_device_utils::safe_sync(p_device);
        _resolve_timestamp_range(timing_state.binning_timestamp, timing_state.last_binning_gpu_ms);
        _resolve_timestamp_range(timing_state.raster_timestamp, timing_state.last_raster_gpu_ms);
        _resolve_timestamp_range(timing_state.prefix_timestamp, timing_state.last_prefix_gpu_ms);
        _resolve_timestamp_range(timing_state.resolve_timestamp, timing_state.last_resolve_gpu_ms);
        timing_state.last_frame_gpu_ms = timing_state.last_binning_gpu_ms + timing_state.last_raster_gpu_ms + timing_state.last_prefix_gpu_ms + timing_state.last_resolve_gpu_ms;
        timing_state.last_submission_cpu_ms = 0.0f;
    }
}

void TileRenderer::_flush_pending_submission(bool p_block) {
    RenderingDevice *device = _get_submission_device();
    if (!device) {
        return;
    }

    if (p_block) {
        // Use safe sync - only syncs on local devices (main device syncs automatically)
        gs_device_utils::safe_sync(device);
        _resolve_timestamp_range(timing_state.binning_timestamp, timing_state.last_binning_gpu_ms);
        _resolve_timestamp_range(timing_state.raster_timestamp, timing_state.last_raster_gpu_ms);
        _resolve_timestamp_range(timing_state.prefix_timestamp, timing_state.last_prefix_gpu_ms);
        _resolve_timestamp_range(timing_state.resolve_timestamp, timing_state.last_resolve_gpu_ms);
        timing_state.last_frame_gpu_ms = timing_state.last_binning_gpu_ms + timing_state.last_raster_gpu_ms + timing_state.last_prefix_gpu_ms + timing_state.last_resolve_gpu_ms;
        resolve_gpu_timestamps_async();
    }
}

RenderingDevice *TileRenderer::_get_resource_device() const {
    return device_context.resource_rd;
}

RenderingDevice *TileRenderer::_get_submission_device() {
    if (device_context.submission_rd) {
        device_context.warned_missing_submission_device = false;
        return device_context.submission_rd;
    }

    if (RenderingDevice *resource = _get_resource_device()) {
        device_context.submission_rd = resource;
        device_context.warned_missing_submission_device = false;
        return device_context.submission_rd;
    }

    if (!device_context.warned_missing_submission_device) {
        GS_LOG_WARN_DEFAULT("[TileRenderer] Unable to acquire submission RenderingDevice");
        device_context.warned_missing_submission_device = true;
    }

    return nullptr;
}

RenderingDevice *TileRenderer::_acquire_submission_device() {
    return _get_submission_device();
}

void TileRenderer::_invalidate_descriptor_cache() {
    descriptor_generation++;

    // Release GPU uniform set objects. The generation counter will force
    // acquire_* functions to rebuild on next access.
    auto release_set = [](RID &p_rid, RenderingDevice *p_device) {
        if (p_rid.is_valid() && p_device && p_device->uniform_set_is_valid(p_rid)) {
            p_device->free(p_rid);
        }
        p_rid = RID();
    };

    // Binning stage
    release_set(binning_stage.cached_binning_buffer_uniform_set, binning_stage.cached_binning_buffer_device);
    release_set(binning_stage.cached_binning_buffer_uniform_set_alt, binning_stage.cached_binning_buffer_device);
    release_set(binning_stage.cached_binning_count_uniform_set, binning_stage.cached_binning_count_device);
    release_set(binning_stage.cached_binning_count_uniform_set_alt, binning_stage.cached_binning_count_device);
    release_set(binning_stage.cached_binning_param_uniform_set, binning_stage.cached_binning_param_device);
    binning_stage.cached_binning_buffer_device = nullptr;
    binning_stage.cached_binning_count_device = nullptr;
    binning_stage.cached_binning_param_device = nullptr;

    // Per-call params (vary per-frame).
    binning_stage.cached_binning_gaussian_buffer = RID();
    binning_stage.cached_binning_sorted_indices = RID();
    binning_stage.cached_binning_count_gaussian_buffer = RID();
    binning_stage.cached_binning_count_sorted_indices = RID();

    // Double-buffer tracking.
    binning_stage.cached_binning_tile_counts = RID();
    binning_stage.cached_binning_tile_counts_alt = RID();
    binning_stage.cached_binning_count_tile_counts = RID();
    binning_stage.cached_binning_count_tile_counts_alt = RID();

    // Prefix scan stage
    release_set(prefix_scan_stage.cached_binning_prefix_uniform_set, prefix_scan_stage.cached_binning_prefix_device);
    release_set(prefix_scan_stage.cached_binning_prefix_uniform_set_alt, prefix_scan_stage.cached_binning_prefix_device);
    prefix_scan_stage.cached_binning_prefix_device = nullptr;

    // Double-buffer tracking.
    prefix_scan_stage.cached_binning_prefix_tile_counts = RID();
    prefix_scan_stage.cached_binning_prefix_tile_counts_alt = RID();

    // Rasterizer stage
    release_set(raster_stage.cached_raster_buffer_uniform_set, raster_stage.cached_raster_buffer_device);
    release_set(raster_stage.cached_raster_param_uniform_set, raster_stage.cached_raster_param_device);
    release_set(raster_stage.cached_raster_compute_buffer_uniform_set, raster_stage.cached_raster_compute_buffer_device);
    release_set(raster_stage.cached_raster_compute_param_uniform_set, raster_stage.cached_raster_compute_param_device);
    release_set(raster_stage.cached_raster_image_uniform_set, raster_stage.cached_raster_image_device);
    raster_stage.cached_raster_buffer_device = nullptr;
    raster_stage.cached_raster_param_device = nullptr;
    raster_stage.cached_raster_compute_buffer_device = nullptr;
    raster_stage.cached_raster_compute_param_device = nullptr;
    raster_stage.cached_raster_image_device = nullptr;

    // Per-call params (vary per-frame).
    raster_stage.cached_raster_gaussian_buffer = RID();
    raster_stage.cached_raster_sorted_indices = RID();
    raster_stage.cached_raster_compute_gaussian_buffer = RID();
    raster_stage.cached_raster_compute_sorted_indices = RID();
    raster_stage.cached_state_uniform = RID();
    raster_stage.cached_raster_compute_state_uniform = RID();
    raster_stage.cached_raster_image_output = RID();
    raster_stage.cached_raster_image_depth = RID();
    raster_stage.cached_raster_image_normal = RID();
}

bool TileRenderer::_ensure_param_uniform_buffer(RenderingDevice *p_device) {
    return uniform_buffers.ensure_param_buffer(p_device);
}

RID TileRenderer::_get_default_state_uniform(RenderingDevice *p_device) {
    return uniform_buffers.get_default_state_uniform(p_device);
}

bool TileRenderer::_ensure_prefix_param_buffer(RenderingDevice *p_device, uint32_t p_size) {
	return uniform_buffers.ensure_prefix_param_buffer(p_device, p_size);
}

uint32_t TileRenderer::_clamp_overlap_record_budget(uint32_t p_requested) const {
	uint32_t capped = MAX<uint32_t>(p_requested, 1u);
	const uint32_t hard_cap = g_gpu_sorting_config.get_overlap_records_hard_cap();
	if (hard_cap != UINT32_MAX) {
		capped = MIN<uint32_t>(capped, hard_cap);
	}
	return capped;
}

uint32_t TileRenderer::_get_effective_overlap_capacity(uint32_t p_capacity_hint) const {
	const uint32_t capacity = p_capacity_hint > 0 ? p_capacity_hint : global_sort_resources.capacity;
	if (capacity == 0) {
		return 0;
	}
	return _clamp_overlap_record_budget(capacity);
}

void TileRenderer::_issue_async_tile_count_readback(RenderingDevice *p_device, uint32_t p_counts_bytes) {
	if (!p_device || async_readback.tile_counts_state.pending_readback) {
		return;
	}

	const uint64_t request_frame_serial = frame_state.current_frame_serial;
	async_readback.tile_counts_state.cached_total_tiles = grid_state.total_tiles;
	async_readback.tile_counts_state.pending_readback = true;
	async_readback.tile_counts_state.requested_frame_serial = request_frame_serial;
	Callable callback = callable_mp(this, &TileRenderer::_on_tile_counts_readback).bind(int64_t(request_frame_serial));
	Error err = p_device->buffer_get_data_async(global_sort_resources.get_tile_counts_buffer(), callback, 0, p_counts_bytes);
	if (err != OK) {
		async_readback.tile_counts_state.pending_readback = false;
		async_readback.tile_counts_state.requested_frame_serial = 0;
	}
}

void TileRenderer::_compute_density_metrics(const uint32_t *p_counts, uint32_t p_process_tiles, uint32_t p_total_tiles,
		uint32_t p_effective_capacity, uint32_t *p_density_write, RenderStats &r_stats, DensityMetrics &r_density_metrics) {
	uint64_t total_splats = 0;
	uint32_t max_splats = 0;
	uint32_t min_non_empty = std::numeric_limits<uint32_t>::max();
	uint32_t non_empty_tiles = 0;
	uint64_t non_empty_splat_sum = 0;

	r_density_metrics.density_histogram.fill(0);
	r_stats.empty_tiles = 0;

	for (uint32_t i = 0; i < p_process_tiles; i++) {
		uint32_t splat_count = p_counts[i];
		if (splat_count == 0) {
			r_stats.empty_tiles++;
			r_density_metrics.density_histogram[0]++;
		} else {
			non_empty_tiles++;
			non_empty_splat_sum += splat_count;
			min_non_empty = MIN(min_non_empty, splat_count);

			const float normalized = float(splat_count) / float(MAX<uint32_t>(1u, p_effective_capacity));
			if (normalized <= 0.25f) {
				r_density_metrics.density_histogram[1]++;
			} else if (normalized <= 0.5f) {
				r_density_metrics.density_histogram[2]++;
			} else if (normalized <= 0.75f) {
				r_density_metrics.density_histogram[3]++;
			} else {
				r_density_metrics.density_histogram[4]++;
			}
		}

		total_splats += splat_count;
		if (p_density_write && i < p_total_tiles) {
			p_density_write[i] = splat_count;
		}
		max_splats = MAX(max_splats, splat_count);
	}

	r_stats.max_splats_in_tile = max_splats;
	r_stats.average_splats_per_tile = p_process_tiles > 0 ? float(total_splats) / float(p_process_tiles) : 0.0f;
	r_stats.overlap_records = (uint32_t)MIN<uint64_t>(total_splats, UINT32_MAX);

	r_density_metrics.occupancy_ratio = r_stats.total_tiles > 0
			? float(r_stats.total_tiles - r_stats.empty_tiles) / float(r_stats.total_tiles)
			: 0.0f;
	r_density_metrics.overflow_ratio = 0.0f;
	uint32_t dense_tile_count = 0;
	if (DENSITY_BUCKET_COUNT >= 4) {
		dense_tile_count = r_density_metrics.density_histogram[3] + r_density_metrics.density_histogram[4];
	}
	r_density_metrics.dense_ratio = r_stats.total_tiles > 0 ? float(dense_tile_count) / float(r_stats.total_tiles) : 0.0f;
	r_density_metrics.average_non_empty_splats = non_empty_tiles > 0 ? float(non_empty_splat_sum) / float(non_empty_tiles) : 0.0f;
	r_density_metrics.min_non_empty_splats = non_empty_tiles > 0 ? min_non_empty : 0;
}

TileRenderer::RenderStats TileRenderer::_build_render_stats_from_cached_counts() {
	const uint32_t cached_tiles = async_readback.tile_counts_state.cached_total_tiles;
	const uint32_t *counts = async_readback.tile_counts_state.cached_counts.ptr();

	RenderStats stats;
	stats.total_tiles = grid_state.total_tiles;
	stats.has_rendering_errors = false;
	const uint32_t configured_overlap_budget = g_gpu_sorting_config.get_overlap_records_hard_cap();
	const uint32_t suggested_overlap_budget = _get_adaptive_overlap_budget_suggestion(this);
	stats.overlap_record_budget = suggested_overlap_budget > 0u ? suggested_overlap_budget : configured_overlap_budget;
	stats.overlap_record_budget_configured = configured_overlap_budget;
	stats.overlap_record_budget_effective = diagnostics.last_overlap_record_budget_effective;
	stats.overlap_thinning_keep_ratio = diagnostics.last_overlap_keep_ratio;

	uint32_t *density_write = nullptr;
	if (diagnostics.capture_tile_density_snapshot) {
		diagnostics.tile_density_snapshot.resize(grid_state.total_tiles);
		if (!diagnostics.tile_density_snapshot.is_empty()) {
			density_write = diagnostics.tile_density_snapshot.ptrw();
		}
	} else {
		diagnostics.tile_density_snapshot.clear();
	}

	DensityMetrics density_metrics;
	const uint32_t process_tiles = MIN(cached_tiles, grid_state.total_tiles);
	_compute_density_metrics(counts, process_tiles, grid_state.total_tiles, uint32_t(config_state.effective_splat_capacity),
			density_write, stats, density_metrics);

	stats.density_metrics = density_metrics;
	return stats;
}

void TileRenderer::_collect_render_statistics() {
	auto apply_raster_usage = [&](RenderStats &r_stats) {
		r_stats.compute_raster_frames = perf_metrics.compute_raster_frames;
		r_stats.fragment_raster_frames = perf_metrics.fragment_raster_frames;
		r_stats.last_raster_used_compute = perf_metrics.last_raster_used_compute;
		r_stats.sorted_indices_blend_fallback_active = perf_metrics.sorted_indices_blend_fallback_active;
		r_stats.sorted_indices_blend_fallback_reason = perf_metrics.sorted_indices_blend_fallback_reason;
	};
	auto reset_stats_and_disable = [&]() {
		diagnostics.last_render_stats = RenderStats();
		apply_raster_usage(diagnostics.last_render_stats);
		adaptive_controller.set_metrics_available(false);
	};
	if (!diagnostics.runtime_statistics_enabled && !adaptive_controller.is_enabled()) {
		reset_stats_and_disable();
		return;
	}

	RenderingDevice *device = _get_resource_device();
	if (!device || grid_state.total_tiles == 0) {
		reset_stats_and_disable();
		return;
	}

	if (render_settings.global_sort_enabled) {
		if (!global_sort_resources.get_tile_counts_buffer().is_valid() ||
				global_sort_resources.tile_buffer_tiles != grid_state.total_tiles) {
			diagnostics.last_render_stats = RenderStats();
			diagnostics.last_render_stats.total_tiles = grid_state.total_tiles;
			apply_raster_usage(diagnostics.last_render_stats);
			adaptive_controller.set_metrics_available(false);
			return;
		}

		const uint64_t counts_bytes64 = uint64_t(grid_state.total_tiles) * sizeof(uint32_t);
		if (counts_bytes64 > UINT32_MAX) {
			GS_LOG_ERROR_DEFAULT("[TileRenderer] Tile count statistics readback exceeds RD limits");
			diagnostics.last_render_stats = RenderStats();
			diagnostics.last_render_stats.total_tiles = grid_state.total_tiles;
			diagnostics.last_render_stats.has_rendering_errors = true;
			apply_raster_usage(diagnostics.last_render_stats);
			adaptive_controller.set_metrics_available(false);
			return;
		}
		const uint32_t counts_bytes = uint32_t(counts_bytes64);

		// Issue async readback if not already pending (avoids 40ms GPU sync stall)
		_issue_async_tile_count_readback(device, counts_bytes);

		// Use cached async results from previous frame (1-2 frame latency is acceptable for HUD)
		if (!async_readback.tile_counts_state.first_frame_complete ||
				async_readback.tile_counts_state.cached_counts.is_empty()) {
			// No async data yet - return minimal stats
			diagnostics.last_render_stats = RenderStats();
			diagnostics.last_render_stats.total_tiles = grid_state.total_tiles;
			const uint32_t configured_overlap_budget = g_gpu_sorting_config.get_overlap_records_hard_cap();
			const uint32_t suggested_overlap_budget = _get_adaptive_overlap_budget_suggestion(this);
			diagnostics.last_render_stats.overlap_record_budget =
					suggested_overlap_budget > 0u ? suggested_overlap_budget : configured_overlap_budget;
			diagnostics.last_render_stats.overlap_record_budget_configured = configured_overlap_budget;
			diagnostics.last_render_stats.overlap_record_budget_effective = diagnostics.last_overlap_record_budget_effective;
			diagnostics.last_render_stats.overlap_thinning_keep_ratio = diagnostics.last_overlap_keep_ratio;
			apply_raster_usage(diagnostics.last_render_stats);
			adaptive_controller.set_metrics_available(false);
			return;
		}

		diagnostics.last_render_stats = _build_render_stats_from_cached_counts();
		const OverflowStatsSnapshot overflow_stats = debug_stats.get_overflow_stats(device, frame_state.current_frame_serial);
		diagnostics.last_render_stats.tiles_with_overflow = overflow_stats.overflow_tile_count;
		if (diagnostics.last_render_stats.total_tiles > 0) {
			const uint32_t clamped_tiles = MIN(overflow_stats.overflow_tile_count, diagnostics.last_render_stats.total_tiles);
			diagnostics.last_render_stats.density_metrics.overflow_ratio =
					float(clamped_tiles) / float(diagnostics.last_render_stats.total_tiles);
		} else {
			diagnostics.last_render_stats.density_metrics.overflow_ratio = 0.0f;
		}
		apply_raster_usage(diagnostics.last_render_stats);
		_update_adaptive_state(diagnostics.last_render_stats);
		return;
	}
}

void TileRenderer::_dump_gpu_debug_counters(const RenderParams &p_params) {
	debug_stats.dump_gpu_debug_counters(_get_resource_device(), p_params, frame_state.current_frame_serial);
}

float TileRenderer::get_projection_success_rate_pct() const {
    if (debug_stats.cached_debug_counters.total_processed == 0) {
        return 0.0f;
    }
    return 100.0f * float(debug_stats.cached_debug_counters.success_count) /
            float(debug_stats.cached_debug_counters.total_processed);
}

float TileRenderer::get_sh_cache_hit_rate_pct() const {
    const uint32_t hits = debug_stats.cached_debug_counters.sh_cache_hits;
    const uint32_t updates = debug_stats.cached_debug_counters.sh_cache_updates;
    const uint32_t total = hits + updates;
    if (total == 0) {
        return 0.0f;
    }
    return 100.0f * float(hits) / float(total);
}

void TileRenderer::_update_adaptive_state(const RenderStats &p_stats) {
    adaptive_controller.update_state(p_stats);
}

SortingMetrics TileRenderer::get_sorter_metrics() const {
    if (global_sort_resources.sorter.is_valid()) {
        return global_sort_resources.sorter->get_metrics();
    }
    return SortingMetrics();
}

TileRenderer::DebugCounterSnapshot TileRenderer::get_debug_counters() const {
    RenderingDevice *device = const_cast<TileRenderer *>(this)->_get_resource_device();
    return debug_stats.get_debug_counters(device, frame_state.current_frame_serial);
}

TileRenderer::OverflowStatsSnapshot TileRenderer::get_overflow_stats() const {
    RenderingDevice *device = const_cast<TileRenderer *>(this)->_get_resource_device();
    return debug_stats.get_overflow_stats(device, frame_state.current_frame_serial);
}

TileRenderer::SplatAuditSnapshot TileRenderer::get_splat_audit_snapshot() const {
    RenderingDevice *device = const_cast<TileRenderer *>(this)->_get_resource_device();
    return debug_stats.get_splat_audit_snapshot(device, frame_state.current_frame_serial);
}

bool TileRenderer::_resolve_texture_owner(const char *p_label, const RID &p_texture, RenderingDevice *&r_owner,
		RenderingDevice *p_main_device, RenderDeviceManager *p_manager, bool p_log_errors) {
	if (!p_texture.is_valid()) {
		r_owner = nullptr;
		return true;
	}
	if (!r_owner && p_manager) {
		r_owner = p_manager->get_resource_owner(p_texture, nullptr);
	}
	if (!r_owner && p_main_device && p_main_device->texture_is_valid(p_texture)) {
		r_owner = p_main_device;
		if (p_log_errors) {
			GS_LOG_WARN_DEFAULT(vformat("[TileRenderer] %s owner missing; using main RenderingDevice for RID=%s",
					String(p_label), String::num_uint64(p_texture.get_id())));
		}
		return true;
	}
	if (!r_owner) {
		if (p_log_errors) {
			GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] %s ownership contract violation: missing owner for RID=%s",
					String(p_label), String::num_uint64(p_texture.get_id())));
		}
		return false;
	}
	if (!r_owner->texture_is_valid(p_texture)) {
		if (p_main_device && p_main_device != r_owner && p_main_device->texture_is_valid(p_texture)) {
			if (p_manager) {
				p_manager->push_cross_device_operation(
						String(p_label) + String("_owner_remap"), r_owner, p_main_device);
			}
			r_owner = p_main_device;
			return true;
		}
		if (p_log_errors) {
			GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] %s ownership contract violation: owner does not validate RID=%s",
					String(p_label), String::num_uint64(p_texture.get_id())));
		}
		return false;
	}
	if (p_main_device && r_owner != p_main_device) {
		const bool visible_on_main = p_main_device->texture_is_valid(p_texture);
		if (visible_on_main) {
			if (p_manager) {
				p_manager->push_cross_device_operation(
						String(p_label) + String("_owner_remap"), r_owner, p_main_device);
			}
			r_owner = p_main_device;
			return true;
		}
		if (p_manager) {
			p_manager->push_cross_device_operation(
					String(p_label) + String("_contract_violation"), r_owner, p_main_device);
		}
		if (p_log_errors) {
			GS_LOG_ERROR_DEFAULT(vformat("[TileRenderer] %s ownership contract violation: RID=%s is not visible on the main RenderingDevice",
					String(p_label), String::num_uint64(p_texture.get_id())));
		}
		return false;
	}
	return true;
}

void TileRenderer::set_device_manager(RenderDeviceManager *p_device_manager) {
	tracked_device_manager = p_device_manager;
}

void TileRenderer::track_output_resources(const RID &p_color_output, RenderingDevice *p_color_device,
		const RID &p_depth_output, RenderingDevice *p_depth_device) {
	if (!tracked_device_manager) {
		return;
	}

	RenderingDevice *main_device = _get_contract_main_device();

	if (p_color_output.is_valid()) {
		if (tracked_color_output.is_valid() && tracked_color_output != p_color_output) {
			tracked_device_manager->forget_resource(tracked_color_output);
		}
		RenderingDevice *owner = p_color_device ? p_color_device : main_device;
		if (_resolve_texture_owner("tile_color_output", p_color_output, owner, main_device, tracked_device_manager, false)) {
			tracked_device_manager->track_resource(p_color_output, owner, false, "tile_renderer_color_output");
			tracked_color_output = p_color_output;
		} else if (tracked_color_output.is_valid()) {
			tracked_device_manager->forget_resource(tracked_color_output);
			tracked_color_output = RID();
		}
	} else if (tracked_color_output.is_valid()) {
		tracked_device_manager->forget_resource(tracked_color_output);
		tracked_color_output = RID();
	}

	if (p_depth_output.is_valid()) {
		if (tracked_depth_output.is_valid() && tracked_depth_output != p_depth_output) {
			tracked_device_manager->forget_resource(tracked_depth_output);
		}
		RenderingDevice *owner = p_depth_device ? p_depth_device : main_device;
		if (_resolve_texture_owner("tile_depth_output", p_depth_output, owner, main_device, tracked_device_manager, false)) {
			tracked_device_manager->track_resource(p_depth_output, owner, false, "tile_renderer_depth_output");
			tracked_depth_output = p_depth_output;
		} else if (tracked_depth_output.is_valid()) {
			tracked_device_manager->forget_resource(tracked_depth_output);
			tracked_depth_output = RID();
		}
	} else if (tracked_depth_output.is_valid()) {
		tracked_device_manager->forget_resource(tracked_depth_output);
		tracked_depth_output = RID();
	}
}

void TileRenderer::clear_output_resource_tracking() {
	if (tracked_device_manager) {
		if (tracked_color_output.is_valid()) {
			tracked_device_manager->forget_resource(tracked_color_output);
		}
		if (tracked_depth_output.is_valid()) {
			tracked_device_manager->forget_resource(tracked_depth_output);
		}
	}
	tracked_color_output = RID();
	tracked_depth_output = RID();
}

TileRenderer::RenderResult TileRenderer::render_with_contract(RenderingDevice *p_device, const RenderParams &p_params) {
	RenderResult result;
	result.success = false;

	if (!is_initialized()) {
		return result;
	}

	set_frame_serial(p_params.frame_serial);

	RID output = render(p_device, p_params);

	if (!output.is_valid()) {
		return result;
	}

	result.output_texture = output;
	result.depth_texture = get_depth_texture();
	result.output_owner = get_output_texture_owner();
	result.depth_owner = get_depth_texture_owner();
	RenderingDevice *rd = _get_resource_device();
	if (!result.output_owner) {
		result.output_owner = p_device ? p_device : rd;
	}
	if (!result.depth_owner) {
		result.depth_owner = p_device ? p_device : rd;
	}
	result.has_depth = has_depth_output() && result.depth_texture.is_valid();
	result.depth_copy_compatible = result.has_depth && is_depth_copy_compatible();

	RenderingDevice *main_device = _get_contract_main_device();
	RenderDeviceManager *manager_ptr = tracked_device_manager;

	if (!_resolve_texture_owner("tile_color_output", result.output_texture, result.output_owner, main_device, manager_ptr, true)) {
		GS_LOG_ERROR_DEFAULT("[TileRenderer] Color output contract failed; output is disabled for this frame");
		clear_output_resource_tracking();
		result = RenderResult();
		return result;
	}

	if (result.has_depth) {
		if (!_resolve_texture_owner("tile_depth_output", result.depth_texture, result.depth_owner, main_device, manager_ptr, true)) {
			GS_LOG_WARN_DEFAULT("[TileRenderer] Depth output contract failed; depth output is disabled for this frame");
			result.depth_texture = RID();
			result.depth_owner = nullptr;
			result.has_depth = false;
			result.depth_copy_compatible = false;
		}
	}

	result.success = true;
	track_output_resources(result.output_texture, result.output_owner,
			result.depth_texture, result.has_depth ? result.depth_owner : nullptr);

	return result;
}
