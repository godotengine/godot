/**
 * @file render_pipeline_io_types.h
 * @brief Pipeline stage input/output type definitions.
 *
 * Standalone types for the renderer pipeline stages (cull, sort, raster,
 * composite). Extracted from GaussianSplatRenderer to allow orchestrators
 * and pipeline stages to depend on narrow contracts instead of the full
 * renderer header.
 */

#ifndef GAUSSIAN_RENDER_PIPELINE_IO_TYPES_H
#define GAUSSIAN_RENDER_PIPELINE_IO_TYPES_H

#include "core/math/vector2i.h"
#include "core/string/ustring.h"
#include "core/templates/rid.h"
#include "render_state_types.h"
#include "../tile_render_types.h"
#include <cstdint>

namespace GaussianRenderPipeline {

using IndexDomain = GaussianRenderState::IndexDomain;

struct RenderFrameSnapshot {
	bool valid = false;
	uint32_t visible_splats = 0;
	uint32_t sorted_splats = 0;
	IndexDomain cull_visible_domain = IndexDomain::UNKNOWN;
	IndexDomain sorted_index_domain = IndexDomain::UNKNOWN;
};

enum class RenderFallbackReason {
	NONE = 0,
	DATA_UNAVAILABLE,
	STREAMING_DATA_UNAVAILABLE,
	GPU_CULLER_UNAVAILABLE,
	NO_VISIBLE_SPLATS,
	RENDERING_DEVICE_UNAVAILABLE,
	RASTER_REUSED_CACHED_RENDER,
	PAINTERLY_UNAVAILABLE,
	PAINTERLY_PASS_GRAPH_UNAVAILABLE,
	PAINTERLY_MATERIAL_UNAVAILABLE,
	PAINTERLY_RENDER_FAILED,
	TILE_FALLBACK_FAILED,
	OUTPUT_COMPOSITOR_UNAVAILABLE,
};

// Note: Named StageStatus to avoid X11/Xlib.h macro conflict (#define Status int)
struct StageResult {
	enum class StageStatus {
		SUCCESS,
		SKIPPED,
		FALLBACK,
		FAILED
	};

	StageStatus status = StageStatus::SUCCESS;
	bool is_error = false;
	RenderFallbackReason fallback_reason = RenderFallbackReason::NONE;
	String reason;

	bool is_success() const { return status == StageStatus::SUCCESS || status == StageStatus::FALLBACK; }
	bool is_failure() const { return status == StageStatus::FAILED; }
	bool did_fallback() const { return status == StageStatus::FALLBACK; }
	bool was_skipped() const { return status == StageStatus::SKIPPED; }
};

struct StageIO {
	uint64_t frame_id = 0;
	uint32_t input_count = 0;
	uint32_t output_count = 0;
	RID input_buffer;
	RID output_buffer;
	bool validated = false;
	bool validation_failed = false;
	String validation_error;
};

struct SortStageOutput {
	bool did_sort = false;
	uint32_t sorted_count = 0;
	uint32_t input_count = 0;
	float sort_time_ms = 0.0f;
	IndexDomain input_domain = IndexDomain::UNKNOWN;
	IndexDomain output_domain = IndexDomain::UNKNOWN;
};

struct SplatDataSource {
	static constexpr const char *kSourceNone = "none";
	static constexpr const char *kSourceUnavailable = "Unavailable";
	static constexpr const char *kSourceBufferManager = "GPUBufferManager";
	static constexpr const char *kSourceStreaming = "StreamingGPU";
	static constexpr const char *kSourceCpuData = "gaussian_data";

	RID gaussian_buffer;
	RID sorted_indices;
	uint32_t splat_count = 0;
	uint32_t total_gaussians = 0;
	const char *source_name = kSourceUnavailable;
};

struct InstancePipelineBuffers {
	RID instance_buffer;
	RID asset_meta_buffer;
	RID asset_chunk_index_buffer;
	RID chunk_meta_buffer;
	RID visible_chunk_buffer;
	RID splat_ref_buffer;
	RID sort_key_buffer;
	RID sort_value_buffer;
	RID quantization_buffer;
	bool quantization_required = false;
	RID atlas_gaussian_buffer;
	uint32_t atlas_gaussian_count = 0;
	RID counter_buffer;
	RID chunk_dispatch_buffer;
	RID indirect_count_buffer;
	RID instance_count_buffer;
	uint32_t instance_count = 0;
	uint32_t dispatch_chunk_count = 0;
	uint32_t max_visible_chunks = 0;
	uint32_t max_visible_splats = 0;
	uint32_t max_chunk_splats = 0;
};

struct DataSourcePlan {
	SplatDataSource source;
	String error;
	bool using_real_data = false;
};

struct RasterStageOutput {
	RID color;
	RID depth;
	Size2i internal_size;
	bool painterly_active = false;
	bool reused_cached_render = false;
	float render_time_ms = 0.0f;
	String raster_path = "unknown";
	uint32_t sorted_splat_count = 0;
	uint64_t content_generation = 0;
	uint64_t shader_defines_hash = 0;
};

/**
 * @struct PainterlyCompositePushConstant
 * @brief GPU push constants for painterly compositing shader.
 */
struct PainterlyCompositePushConstant {
	float inv_viewport_size[2];
	float depth_bias;
	float blend_strength;
	float near_plane;
	float far_plane;
	float proj_22;
	float proj_32;
	float proj_23;
};

/**
 * @struct RenderFramePlan
 * @brief Planning data that determines the render path for a frame.
 */
struct RenderFramePlan {
	DataSourcePlan data_source;
	bool has_render_data = false;
	bool set_skip_metrics = false;
	bool clear_cull_state_on_skip = false;
	GaussianSplatting::ComputeRasterPolicy compute_raster_policy = GaussianSplatting::ComputeRasterPolicy::Default;
	String cull_skip_reason;
	String sort_skip_reason;
	RenderFallbackReason cull_skip_reason_code = RenderFallbackReason::NONE;
	RenderFallbackReason sort_skip_reason_code = RenderFallbackReason::NONE;
};

} // namespace GaussianRenderPipeline

#endif // GAUSSIAN_RENDER_PIPELINE_IO_TYPES_H
