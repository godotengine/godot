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
 * Note: StageMetrics, PipelineEvent, and RenderFrameContext remain defined
 * inside GaussianSplatRenderer class because they reference many class-specific
 * types (StageResult, CullStageOutput, etc.).
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

} // namespace GaussianRenderFrame

#endif // GAUSSIAN_RENDER_FRAME_TYPES_H
