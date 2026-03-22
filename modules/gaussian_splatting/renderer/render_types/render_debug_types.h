/**
 * @file render_debug_types.h
 * @brief Debug-related type definitions for GaussianSplatRenderer.
 *
 * This header contains debug configuration and state structs.
 * Extracted from inline .inc files for better code organization and IDE support.
 */

#ifndef GAUSSIAN_RENDER_DEBUG_TYPES_H
#define GAUSSIAN_RENDER_DEBUG_TYPES_H

#include "core/math/vector2.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "../tile_render_types.h"

/**
 * @namespace GaussianRenderDebug
 * @brief Contains standalone debug types that can be used independently.
 *
 * GaussianSplatRenderer re-exports these through class-local aliases.
 */
namespace GaussianRenderDebug {

/**
 * @struct SplatAuditSummary
 * @brief Summary statistics from splat audit validation.
 *
 * Used for debugging splat rendering issues by tracking projection,
 * viewport containment, and contribution statistics.
 */
struct SplatAuditSummary {
    bool valid = false;
    uint32_t sample_count = 0;
    uint32_t projected_count = 0;
    uint32_t in_viewport_count = 0;
    uint32_t iterated_count = 0;
    uint32_t contributed_count = 0;
    uint32_t alpha_skipped_count = 0;
    uint32_t missing_iterated_count = 0;
    uint32_t missing_contrib_count = 0;
    uint32_t first_mismatch_global_idx = 0;
    uint32_t first_mismatch_expected_x = 0;
    uint32_t first_mismatch_expected_y = 0;
    uint32_t first_mismatch_flags = 0;
};

/**
 * @struct DebugConfigBase
 * @brief Base debug configuration flags independent of renderer class.
 *
 * Contains debug options that don't depend on renderer-specific types.
 * The full DebugConfig struct in GaussianSplatRenderer extends this
 * with additional options that require class-specific types.
 */
struct DebugConfigBase {
    bool gpu_sort_readback_enabled = false; ///< Enable for debug readback; default off to avoid stalls
    bool dump_gpu_counters = false;
    bool enable_pipeline_trace = false;
    bool enable_state_guardrails = false;
    bool enable_splat_audit = false;
    bool enable_all_debug = false;
    bool enable_frame_logging = false;
    bool enable_frame_logging_verbose = false;
    int frame_log_frequency = 300;
    bool enable_sort_path_logs = false;
    bool enable_tile_logs = false;
    bool enable_tile_pipeline_logs = false;
    bool enable_tile_dispatch_logs = false;
    bool enable_gpu_counter_logs = false;
    bool enable_binning_counters = false;
    bool enable_cull_counters = false;
    bool enable_autotune_logs = false;
    bool enable_data_logging = false;
    bool enable_cull_guardrails = false;
    int splat_audit_sample_count = 64;
    int cull_guardrail_min_visible = 256;
    float cull_guardrail_position_epsilon = 0.05f;
    float cull_guardrail_rotation_epsilon = 0.01f;
    float cull_guardrail_drop_ratio = 0.75f;
    float overlay_opacity = 0.3f;
    float heatmap_exponent = 1.0f;
    Vector2 hud_origin = Vector2(16.0f, 16.0f);
	float hud_scale = 1.0f;
};

struct DebugConfig : public DebugConfigBase {
	GaussianSplatting::ComputeRasterPolicy compute_raster_policy_override = GaussianSplatting::ComputeRasterPolicy::Default;
};

/**
 * @struct JacobianDebugConfig
 * @brief Debug toggles for Jacobian radial stretching investigation.
 */
struct JacobianDebugConfig {
	bool bypass_radius_depth_floor = false;  ///< Use only near_plane, ignore radius*0.5 floor
	bool bypass_j_col2_clamp = false;        ///< Disable ±1e4 clamp on J_col2
	bool invert_j_col2_sign = false;         ///< Test opposite sign convention for J_col2
	float max_conic_aspect = 10.0f;          ///< Max aspect ratio for conic clamping (lower = less stretching)
};

template <typename PreviewModeT, typename StageMetricsT, typename PipelineEventT>
struct DebugState {
	bool show_tile_bounds = false;
	bool show_splat_coverage = false;
	bool show_overflow_tiles = false;
	bool show_projection_issues = false;
	bool show_white_albedo = false;
	bool show_shadow_opacity = false;
	bool show_tile_grid = false;
	bool show_density_heatmap = false;
	bool show_performance_hud = false;
	bool show_resolve_input = false;
	bool show_resolve_output = false;
	bool show_residency_hud = false;
	bool show_device_boundaries = false;
	bool show_texture_states = false;
	PreviewModeT preview_mode = {};
	bool overlay_dirty = false;
	bool hud_dirty = false;
	uint64_t overlay_version = 0;
	uint64_t hud_version = 0;
	Vector<String> hud_lines;
	Vector<uint32_t> tile_density_cache;
	int tile_density_width = 0;
	int tile_density_height = 0;
	uint32_t tile_density_peak = 0;
	float tile_density_average = 0.0f;
	float last_tile_assignment_ms = 0.0f;
	float last_tile_rasterization_ms = 0.0f;
	float last_render_time_ms = 0.0f;
	float last_sort_time_ms = 0.0f;
	String route_uid;
	String sort_route_uid;
	StageMetricsT last_stage_metrics;
	bool last_stage_metrics_valid = false;
	SplatAuditSummary splat_audit;
	Vector<PipelineEventT> pipeline_events;
	uint64_t pipeline_events_frame = 0;
	bool pipeline_events_valid = false;
	uint64_t pipeline_trace_generation = 0;
};

} // namespace GaussianRenderDebug

#endif // GAUSSIAN_RENDER_DEBUG_TYPES_H
