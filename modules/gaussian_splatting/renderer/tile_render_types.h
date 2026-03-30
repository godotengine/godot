#ifndef TILE_RENDER_TYPES_H
#define TILE_RENDER_TYPES_H

#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2i.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"

#include "../resources/color_grading_resource.h"

#include <array>
#include <cstdint>
#include <limits>

// ProjectedGaussian payload layout (must match tile_projection_common.glsl).
// Full layout (36 bytes):
//   data[0]: screen XY as packHalf2x16 (4 bytes)
//   data[1]: depth (float16) + opacity (unorm8) + flags (uint8)
//   data[2]: color as R11G11B10F packed format
//   data[3]: conic.x as float32
//   data[4]: conic.z as float32
//   data[5]: conic.y as float32
//   data[6]: global_idx as uint32
//   data[7]: normal.xy as half2
//   data[8]: normal.z as half (high 16 bits unused)
//
// Packed layout (32 bytes, optional): conic.y packed as float16 + 16-bit global_idx.
// This is lower precision and limited to 16-bit indices, so it is gated at runtime.
struct TileProjectionLayout {
	struct alignas(4) Payload {
		uint32_t data[9];
	};

	struct alignas(4) PackedPayload {
		uint32_t data[8];
	};

	static constexpr uint32_t STRIDE_FULL = sizeof(Payload);
	static constexpr uint32_t STRIDE_PACKED = sizeof(PackedPayload);
	static constexpr uint32_t STRIDE = STRIDE_FULL;
	static_assert(sizeof(Payload) == 36, "TileProjectionLayout::Payload must be 36 bytes");
	static_assert(sizeof(PackedPayload) == 32, "TileProjectionLayout::PackedPayload must be 32 bytes");
};

namespace GaussianSplatting {

struct TileAdaptiveSettings {
	bool enable_adaptive_tile_size = false;
	bool clamp_to_power_of_two = false;
	int min_tile_size = 8;
	int max_tile_size = 64;
	int tile_size_step = 4;
	float target_occupancy_ratio = 0.55f;
	float occupancy_hysteresis = 0.15f;
	float dense_ratio_threshold = 0.35f;
	float overflow_ratio_threshold = 0.02f;
	float max_average_splat_ratio = 0.65f;
	float smoothing_factor = 0.2f;
	uint32_t frames_before_adjustment = 3;
};

struct TileDebugCounterSnapshot {
	uint32_t total_processed = 0;
	uint32_t near_far_reject = 0;
	uint32_t view_distance_reject = 0;
	uint32_t quaternion_reject = 0;
	uint32_t scale_reject = 0;
	uint32_t clip_w_reject = 0;
	uint32_t clip_bounds_reject = 0;
	uint32_t screen_nan_reject = 0;
	uint32_t focal_length_reject = 0;
	uint32_t z_inverse_reject = 0;
	uint32_t covariance_nan_reject = 0;
	uint32_t determinant_reject = 0;
	uint32_t radius_reject = 0;
	uint32_t distance_cull_reject = 0;
	uint32_t viewport_bounds_reject = 0;
	uint32_t bbox_integrity_reject = 0;
	uint32_t tile_extent_reject = 0;
	uint32_t success_count = 0;
	uint32_t extreme_conic_count = 0;
	uint32_t index_mismatch_count = 0;
	uint32_t depth_discrepancy_count = 0;
	uint32_t depth_discrepancy_sum_q8 = 0;
	uint32_t high_aspect_ratio_count = 0;
	uint32_t max_aspect_q8 = 0;
	uint32_t max_aspect_preclamp_q8 = 0;
	uint32_t j_col2_clamp_count = 0;
	uint32_t sh_cache_hits = 0;
	uint32_t sh_cache_updates = 0;
	uint32_t sh_cache_forced_updates = 0;
	// Subpixel culling diagnostics (Q8.8 fixed-point, 256 = 1.0 px)
	uint32_t tiny_splat_param_q8 = 0;
	uint32_t min_allowed_radius_q8 = 0;
	uint32_t min_radius_min_q8_inv = 0;
};

struct TileOverflowStatsSnapshot {
	uint32_t overflow_tile_count = 0;
	uint32_t overflow_splats_clamped = 0;
	uint32_t overflow_splats_aggregated = 0;
	uint32_t raster_sample_count = 0;
	uint32_t raster_splats_iterated = 0;
	uint32_t raster_splats_contributed = 0;
	uint32_t raster_reject_sorted_idx_oob = 0;
	uint32_t raster_reject_gaussian_idx_oob = 0;
	uint32_t raster_reject_base_opacity = 0;
	uint32_t raster_reject_nan_inf = 0;
	uint32_t raster_reject_weight = 0;
	uint32_t raster_reject_alpha = 0;
	uint32_t raster_break_remaining_alpha = 0;
	uint32_t raster_break_final_alpha = 0;
	uint32_t raster_has_depth = 0;
	uint32_t raster_alpha_sum_q10 = 0;
	uint32_t raster_reject_index_mismatch = 0;
	uint32_t raster_break_subgroup_early_exit = 0;
};

struct TileSplatAuditSnapshot {
	bool valid = false;
	uint64_t frame_serial = 0;
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

struct TileDensityMetrics {
	std::array<uint32_t, 5> density_histogram{};
	float occupancy_ratio = 0.0f;
	float dense_ratio = 0.0f;
	float overflow_ratio = 0.0f;
	float average_non_empty_splats = 0.0f;
	uint32_t min_non_empty_splats = 0;
};

struct TileRenderStats {
	uint32_t total_tiles = 0;
	uint32_t tiles_with_overflow = 0;
	uint32_t empty_tiles = 0;
	uint32_t max_splats_in_tile = 0;
	float average_splats_per_tile = 0.0f;
	bool has_rendering_errors = false;
	TileDensityMetrics density_metrics;
	uint32_t overlap_records = 0;
	uint32_t overlap_record_budget = 0;
	uint32_t overlap_record_budget_effective = 0;
	uint32_t overlap_record_budget_configured = 0;
	float overlap_thinning_keep_ratio = 1.0f;
	uint64_t compute_raster_frames = 0;
	uint64_t fragment_raster_frames = 0;
	bool last_raster_used_compute = false;
	bool sorted_indices_blend_fallback_active = false;
	String sorted_indices_blend_fallback_reason;
};

enum TileResolveDebugMode {
	RESOLVE_DEBUG_NONE = 0,
	RESOLVE_DEBUG_INPUT,
	RESOLVE_DEBUG_OUTPUT,
};

enum class ComputeRasterPolicy {
	Default = 0, // Honor global config.
	ForceOn,
	ForceOff,
};

struct TileRenderSettings {
	bool global_sort_enabled = false;
	bool allow_compute_raster = false;
	bool enable_packed_stage_data = false;
	bool enable_tighter_bounds = false;
	bool enable_sh_amortization = false;
	uint32_t sh_amortization_divisor = 1;
	TileResolveDebugMode resolve_debug_mode = RESOLVE_DEBUG_NONE;
	float resolve_feather_pixels = 0.0f;
};

struct TileTimestampRange {
	RenderingDevice *device = nullptr;
	uint32_t start_index = std::numeric_limits<uint32_t>::max();
	uint32_t end_index = std::numeric_limits<uint32_t>::max();
	String label;
	void reset() {
		device = nullptr;
		start_index = std::numeric_limits<uint32_t>::max();
		end_index = std::numeric_limits<uint32_t>::max();
		label = String();
	}
	bool is_valid() const {
		return device != nullptr && start_index != std::numeric_limits<uint32_t>::max() &&
				end_index != std::numeric_limits<uint32_t>::max() && end_index >= start_index;
	}
};

struct TileTimingState {
	TileTimestampRange binning_timestamp;
	TileTimestampRange raster_timestamp;
	TileTimestampRange prefix_timestamp;
	TileTimestampRange resolve_timestamp;
	float last_submission_cpu_ms = 0.0f;
	float last_setup_cpu_ms = 0.0f;  // CPU time for buffer setup, allocation, uniform updates
	float last_binning_gpu_ms = 0.0f;
	float last_raster_gpu_ms = 0.0f;
	float last_prefix_gpu_ms = 0.0f;
	float last_resolve_gpu_ms = 0.0f;
	float last_frame_gpu_ms = 0.0f;
	uint64_t gpu_timing_frame_serial = 0;
	uint64_t gpu_timing_frames_behind = 0;
};

struct TilePerformanceMetrics {
	float tile_assignment_ms = 0.0f;
	float rasterization_ms = 0.0f;
	uint32_t profiling_cached_overlap_total = 0;
	uint64_t sort_sync_fallback_count = 0;
	uint64_t compute_raster_frames = 0;
	uint64_t fragment_raster_frames = 0;
	bool last_raster_used_compute = false;
	bool last_raster_choice_initialized = false;
	bool last_raster_choice_compute = false;
	String last_raster_choice_reason;
	bool sorted_indices_blend_fallback_active = false;
	String sorted_indices_blend_fallback_reason;
};

struct TileDiagnosticsState {
	mutable TileRenderStats last_render_stats;
	bool runtime_statistics_enabled = false;
	Vector<uint32_t> tile_density_snapshot;
	bool capture_tile_density_snapshot = false;
	bool debug_log_resolve = false;
	int debug_log_resolve_interval_frames = 60;
	bool debug_binning_counters_enabled = false;
	bool debug_dump_gpu_counters = false;
	bool debug_gpu_counter_logs_enabled = false;
	bool debug_tile_logs_enabled = false;
	bool debug_tile_pipeline_logs_enabled = false;
	bool debug_tile_dispatch_logs_enabled = false;
	int debug_frame_log_frequency = 0;
	bool resolve_debug_visualize_tiles = false;
	bool resolve_use_texel_fetch_sampling = true; // Use texelFetch for accurate depth - required for correct lighting
	uint32_t last_overlap_record_count = 0;
	uint32_t last_overlap_record_budget_effective = 0;
	float last_overlap_keep_ratio = 1.0f;
};

struct TileConfigState {
	int tile_size = 16;
	RD::DataFormat desired_output_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	RD::DataFormat output_format = RD::DATA_FORMAT_MAX;
	int effective_splat_capacity = 1024;
	RD::DataFormat resolve_target_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
};

struct TileGridState {
	Vector2i viewport_size;
	uint32_t tiles_x = 0;
	uint32_t tiles_y = 0;
	uint32_t total_tiles = 0;
	uint64_t total_tiles_wide = 0;
	bool tile_grid_overflow = false;
};

struct TileFrameState {
	uint64_t current_frame_serial = 0;
};

struct TileRenderParams {
	RID gaussian_buffer;
	RID sorted_indices;
	RID instance_buffer;
	RID splat_ref_buffer;
	RID chunk_meta_buffer;
	RID quantization_buffer;
	RID instance_indirect_count_buffer;
	RID instance_indirect_dispatch_buffer;
	RID interactive_state_uniform;
	RID scene_uniform_buffer;
	RID directional_light_buffer;
	RID cluster_buffer;
	RID shadow_atlas;
	uint32_t total_gaussians = 0;
	uint32_t splat_count = 0;
	// Maximum visible splats the GPU instance pipeline can produce.
	// Used to size projection_buffer for GPU-indirect dispatch paths where
	// splat_count (CPU readback) may lag behind the actual GPU-side count.
	uint32_t max_visible_splats = 0;
	uint32_t overlap_record_count = 0;
	uint32_t omni_light_count = 0;
	uint32_t spot_light_count = 0;
	uint32_t cluster_size = 0;
	uint32_t cluster_max_elements = 0;
	uint32_t light_mask = 0xFFFFFFFFu;
	Vector2i viewport_size;
	Transform3D world_to_camera_transform;
	Projection projection;
	Projection render_projection;
	int tile_size = 16;
	uint64_t frame_serial = 0;
	bool debug_show_tile_bounds = false;
	bool debug_show_splat_coverage = false;
	bool debug_show_overflow_tiles = false;
	bool debug_show_projection_issues = false;
	bool debug_show_white_albedo = false;
	bool debug_dump_gpu_counters = false;
	bool debug_enable_tile_logs = false;
	bool debug_enable_tile_pipeline_logs = false;
	bool debug_enable_tile_dispatch_logs = false;
	bool debug_enable_gpu_counter_logs = false;
	int debug_frame_log_frequency = 0;
	ComputeRasterPolicy compute_raster_policy = ComputeRasterPolicy::Default;
	bool request_packed_stage_data = false;
	bool request_tighter_bounds = false;
	bool request_sh_amortization = false;
	uint32_t sh_amortization_divisor = 1;
	bool debug_enable_splat_audit = false;
	bool debug_show_tile_grid = false;
	bool debug_show_density_heatmap = false;
	bool debug_show_depth_visualization = false;
	bool debug_show_shadow_opacity = false;
	bool debug_show_performance_hud = false;
	float debug_overlay_opacity = 0.3f;
	uint32_t debug_splat_audit_sample_count = 64;
	bool output_is_premultiplied = false;
	float opacity_multiplier = 1.0f;
	float direct_light_scale = 0.5f;
	float indirect_sh_scale = 1.0f;
	float shadow_strength = 1.0f;
	bool sh_dc_logit = false;
	float shadow_receiver_bias_scale = 0.2f;
	float shadow_receiver_bias_min = 0.0f;
	float shadow_receiver_bias_max = 0.0f;
	bool enable_direct_lighting = true;
	int normal_mode = 0;
	int direct_lighting_mode = 1; // 0=resolve, 1=per-splat (binning), 2=both
	float alpha_floor = 0.0f;
	bool force_solid_coverage = false;
	float cull_far_tolerance = 0.05f;
	float tiny_splat_screen_radius = 0.3f;  // Drop subpixel splats to prevent tile overflow (#797)
	float max_conic_aspect = 10.0f;
	float low_pass_filter = 0.35f; // Minimum covariance variance added in projection (lower = sharper)
	bool jacobian_bypass_radius_depth_floor = false;
	bool jacobian_bypass_j_col2_clamp = false;
	bool jacobian_invert_j_col2_sign = false;
	bool opacity_aware_culling = true;
	float visibility_threshold = 0.01f;
	bool distance_cull_enabled = true;
	float distance_cull_start = 30.0f;
	float distance_cull_max_rate = 0.5f;
	bool lod_blend_enabled = true;
	float lod_blend_factor = 1.0f;
	float lod_blend_distance = 5.0f;
	bool wind_enabled = false;
	Vector3 wind_direction = Vector3(1.0f, 0.0f, 0.0f);
	float wind_strength = 0.0f;
	float wind_frequency = 1.0f;
	float wind_spatial_frequency = 0.1f;
	float wind_time_seconds = 0.0f;
	bool sphere_effector_enabled = false;
	Vector3 sphere_effector_center = Vector3();
	float sphere_effector_radius = 0.0f;
	float sphere_effector_strength = 0.0f;
	float sphere_effector_falloff = 2.0f;
	float sphere_effector_frequency = 2.0f;
	Ref<class ColorGradingResource> color_grading;
	// Instance rotation inverse for SH view direction correction.
	// When a GaussianSplatNode3D has a rotation transform, SH coefficients (stored in
	// original capture coordinates) expect view directions in the same frame.
	// This matrix transforms view directions from the transformed local space back to
	// the original coordinate frame for correct SH evaluation.
	Basis instance_rotation_inverse = Basis();
	bool instance_rotation_valid = false;

	TileRenderParams();
};

struct BufferOwnership {
	RenderingDevice *device = nullptr;
	uint64_t device_id = 0;

	void set(RenderingDevice *p_device) {
		device = p_device;
		device_id = p_device ? p_device->get_device_instance_id() : 0;
	}

	void clear() {
		device = nullptr;
		device_id = 0;
	}

	bool matches(RenderingDevice *p_device) const {
		if (device == nullptr || p_device == nullptr) {
			return device == p_device;
		}
		return device == p_device && device_id == p_device->get_device_instance_id();
	}
};

} // namespace GaussianSplatting

#endif // TILE_RENDER_TYPES_H
