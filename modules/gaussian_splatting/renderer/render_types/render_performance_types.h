/**
 * @file render_performance_types.h
 * @brief Performance metrics and settings type definitions.
 *
 * Standalone types for performance monitoring and quality configuration.
 * Extracted from GaussianSplatRenderer so orchestrators and diagnostic
 * systems can depend on narrow contracts.
 */

#ifndef GAUSSIAN_RENDER_PERFORMANCE_TYPES_H
#define GAUSSIAN_RENDER_PERFORMANCE_TYPES_H

#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/variant/dictionary.h"
#include "render_pipeline_io_types.h"
#include <cstdint>

namespace GaussianRenderPerformance {

struct PerformanceSettings {
	int max_splats = 5000000;
};

/**
 * @struct SortFrameMetrics
 * @brief Per-frame sorting performance metrics.
 *
 * Captures timing and algorithm selection data for a single frame's
 * depth sorting pass. Used for performance monitoring and debugging.
 */
struct SortFrameMetrics {
	uint32_t frame_index = 0;       ///< Frame number for correlation.
	uint32_t element_count = 0;     ///< Number of splats sorted.
	float total_ms = 0.0f;          ///< Total sorting time in milliseconds.
	float gpu_ms = 0.0f;            ///< GPU-side sorting time.
	float cpu_ms = 0.0f;            ///< CPU-side sorting time (if fallback).
	float cpu_selection_ms = 0.0f;  ///< Time spent preparing sort input buffers (ms).
	StringName algorithm;           ///< Name of the sorting algorithm used.
	bool used_gpu = false;          ///< True if GPU sorting was used.
	bool used_cpu_fallback = false; ///< True if CPU fallback was triggered.
	bool used_hybrid = false;       ///< Reserved for future hybrid GPU/CPU sorting.
};

struct PerformanceMetrics {
	float buffer_upload_time_ms = 0.0f;
	float culling_time_ms = 0.0f;
	float gpu_memory_usage_mb = 0.0f;
	uint32_t uploaded_splat_count = 0;
	uint32_t rendered_splat_count = 0;
	bool using_real_data = false;
	String data_source = GaussianRenderPipeline::SplatDataSource::kSourceNone;
	String data_source_error;
	String raster_path = "unknown";
	uint64_t total_frames_rendered = 0;
	float avg_frame_time_ms = 0.0f;
	float peak_frame_time_ms = 0.0f;
	float sort_submission_time_ms = 0.0f;
	float sort_wait_time_ms = 0.0f;
	float sort_input_build_time_ms = 0.0f;
	uint64_t instance_sort_sync_fallback_count = 0;
	uint64_t tile_sort_sync_fallback_count = 0;
	uint64_t sort_cached_fallback_count = 0;
	uint64_t sort_identity_fallback_count = 0;
	uint64_t sort_cull_order_fallback_count = 0;
	bool async_sort_used = false;
	bool async_sort_waited = false;
	float async_overlap_efficiency = 0.0f;
	uint32_t culled_frustum_count = 0;
	uint32_t culled_distance_count = 0;
	uint32_t culled_screen_count = 0;
	uint32_t culled_importance_count = 0;
	uint32_t culling_candidate_count = 0;
	uint32_t visible_after_culling = 0;
	String cull_route_uid;
	String cull_route_reason;
	bool used_hierarchical_culling = false;
	Dictionary streaming_state;
	uint64_t sort_cache_hits = 0;
	uint64_t sort_cache_misses = 0;
	float gpu_utilization = 0.0f;
	float gpu_frame_time_ms = 0.0f;
	float gpu_tile_binning_time_ms = 0.0f;
	float gpu_tile_raster_time_ms = 0.0f;
	float gpu_tile_prefix_time_ms = 0.0f;
	float gpu_tile_resolve_time_ms = 0.0f;
	uint64_t gpu_timing_frame_serial = 0;
	uint64_t gpu_timing_frames_behind = 0;
	uint32_t gpu_timeline_inflight_frames = 0;
	uint32_t gpu_timeline_completed_frames = 0;
	uint32_t gpu_timeline_stall_count = 0;
	float gpu_timeline_stall_ms = 0.0f;
	uint64_t gpu_timeline_last_value = 0;
	uint64_t last_frame_start_usec = 0;
	float frame_to_frame_time_ms = 0.0f;
	float avg_frame_to_frame_ms = 0.0f;
	uint64_t cull_projection_contract_mismatch_count = 0;
};

struct PerformanceState {
	PerformanceMetrics metrics;
	LocalVector<SortFrameMetrics> sort_metrics_history;
};

} // namespace GaussianRenderPerformance

#endif // GAUSSIAN_RENDER_PERFORMANCE_TYPES_H
