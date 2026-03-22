/**
 * @file render_frame_types.h
 * @brief Frame context and pipeline type definitions for GaussianSplatRenderer.
 *
 * This header contains types related to frame rendering context, pipeline stages,
 * and diagnostics. Extracted from inline .inc files for better code organization.
 */

#ifndef GAUSSIAN_RENDER_FRAME_TYPES_H
#define GAUSSIAN_RENDER_FRAME_TYPES_H

#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/variant/dictionary.h"
#include "../rendering_error.h"

/**
 * @namespace GaussianRenderFrame
 * @brief Contains frame-related types that can be used independently.
 *
 * RenderFrameContext remains defined inside GaussianSplatRenderer because it
 * still wires together renderer-owned dependencies and state providers.
 */
namespace GaussianRenderFrame {

/**
 * @struct RuntimeErrorStatistics
 * @brief Accumulated error statistics for diagnostics.
 */
struct RuntimeErrorStatistics {
    uint64_t total_errors = 0;
    uint64_t total_warnings = 0;
    uint64_t total_recoveries = 0;
    RenderingError last_error;
    uint64_t last_error_time_usec = 0;
    uint64_t last_recovery_time_usec = 0;
    HashMap<int, uint64_t> error_code_counts;
    HashMap<int, uint64_t> recovery_code_counts;
    Vector<RenderingError> recent_errors;
    Dictionary last_error_context;
};

/**
 * @struct TextureTraceEntry
 * @brief Record of texture allocation/deallocation for debugging.
 */
struct TextureTraceEntry {
    uint64_t timestamp_usec = 0;
    String action;
    uint64_t texture_rid = 0;
    uint64_t device_instance_id = 0;
    String format_label;
    Size2i extent;
};

/**
 * @struct CrossDeviceOperation
 * @brief Record of operations spanning multiple rendering devices.
 */
struct CrossDeviceOperation {
    uint64_t timestamp_usec = 0;
    String context;
    uint64_t source_device = 0;
    uint64_t target_device = 0;
};

/**
 * @struct FrameTimingSample
 * @brief Single frame timing sample for performance tracking.
 */
struct FrameTimingSample {
    uint64_t timestamp_usec = 0;
    uint64_t frame = 0;
    double render_ms = 0.0;
    double sort_ms = 0.0;
    double total_ms = 0.0;
    uint32_t visible_splats = 0;
    bool used_gpu = false;
};

/**
 * @struct ErrorRecoveryStateMachine
 * @brief Tracks error recovery state transitions.
 */
struct ErrorRecoveryStateMachine {
	enum class State {
		HEALTHY,
		DIAGNOSTIC,
		RECOVERING,
		DEGRADED,
		DISABLED
	};

	State state = State::HEALTHY;
	String reason;
	uint64_t last_transition_frame = 0;
	uint64_t last_transition_time_usec = 0;
};

struct DiagnosticsState {
	RuntimeErrorStatistics runtime_error_statistics;
	ErrorRecoveryStateMachine recovery_state_machine;
	Vector<TextureTraceEntry> texture_allocation_trace;
	Vector<CrossDeviceOperation> cross_device_operations;
	Vector<FrameTimingSample> frame_timing_history;
	Dictionary last_telemetry_snapshot;
	Dictionary last_production_metrics;
	Dictionary last_production_metrics_validation;
	Dictionary last_perf_gate_result;
	Vector<Dictionary> production_metrics_summaries;
	uint64_t production_metrics_window_start_frame = 0;
	uint64_t production_metrics_window_start_usec = 0;
	uint32_t production_metrics_window_frames = 0;
	double production_metrics_frame_ms_sum = 0.0;
	double production_metrics_cull_ms_sum = 0.0;
	double production_metrics_sort_ms_sum = 0.0;
	double production_metrics_raster_ms_sum = 0.0;
	double production_metrics_composite_ms_sum = 0.0;
	double production_metrics_stage_total_ms_sum = 0.0;
	double production_metrics_frame_ms_peak = 0.0;
	double production_metrics_stage_ms_peak = 0.0;
	uint32_t production_metrics_visible_peak = 0;
	uint64_t production_metrics_visible_sum = 0;
	uint32_t production_metrics_perf_gate_checks = 0;
	uint32_t production_metrics_perf_gate_failures = 0;
	uint64_t production_metrics_invalid_count = 0;
	bool runtime_diagnostics_requested = false;
};

} // namespace GaussianRenderFrame

#endif // GAUSSIAN_RENDER_FRAME_TYPES_H
