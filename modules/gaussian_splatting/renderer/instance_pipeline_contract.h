#ifndef GS_INSTANCE_PIPELINE_CONTRACT_H
#define GS_INSTANCE_PIPELINE_CONTRACT_H

#include "gaussian_splat_renderer.h"
#include <cstdint>

namespace GaussianSplatting {
namespace InstancePipelineContract {

enum class InvariantViolationClass : uint8_t {
	NONE = 0,
	ATLAS,
	CULL,
	SORT,
	RASTER,
	TILE_RUNTIME
};

enum class InvariantViolationReason : uint8_t {
	NONE = 0,

	ATLAS_GAUSSIAN_BUFFER_MISSING,
	ATLAS_ASSET_META_BUFFER_MISSING,
	ATLAS_CHUNK_META_BUFFER_MISSING,
	ATLAS_ASSET_CHUNK_INDEX_BUFFER_MISSING,
	ATLAS_QUANTIZATION_BUFFER_MISSING,

	CULL_INSTANCE_BUFFER_MISSING,
	CULL_ASSET_META_BUFFER_MISSING,
	CULL_ASSET_CHUNK_INDEX_BUFFER_MISSING,
	CULL_CHUNK_META_BUFFER_MISSING,
	CULL_VISIBLE_CHUNK_BUFFER_MISSING,
	CULL_COUNTER_BUFFER_MISSING,
	CULL_INSTANCE_COUNT_ZERO,
	CULL_DISPATCH_CHUNK_COUNT_ZERO,
	CULL_MAX_VISIBLE_CHUNKS_ZERO,

	SORT_ATLAS_GAUSSIAN_BUFFER_MISSING,
	SORT_INSTANCE_BUFFER_MISSING,
	SORT_CHUNK_META_BUFFER_MISSING,
	SORT_VISIBLE_CHUNK_BUFFER_MISSING,
	SORT_SPLAT_REF_BUFFER_MISSING,
	SORT_SORT_KEY_BUFFER_MISSING,
	SORT_SORT_VALUE_BUFFER_MISSING,
	SORT_COUNTER_BUFFER_MISSING,
	SORT_CHUNK_DISPATCH_BUFFER_MISSING,
	SORT_INDIRECT_COUNT_BUFFER_MISSING,
	SORT_INSTANCE_COUNT_BUFFER_MISSING,
	SORT_MAX_VISIBLE_SPLATS_ZERO,
	SORT_MAX_CHUNK_SPLATS_ZERO,
	SORT_QUANTIZATION_BUFFER_MISSING,

	RASTER_ATLAS_GAUSSIAN_BUFFER_MISSING,
	RASTER_INSTANCE_BUFFER_MISSING,
	RASTER_SPLAT_REF_BUFFER_MISSING,
	RASTER_INSTANCE_COUNT_BUFFER_MISSING,
	RASTER_QUANTIZATION_BUFFER_MISSING,

	TILE_INSTANCE_BUFFER_MISSING,
	TILE_SPLAT_REF_BUFFER_MISSING,
	TILE_INDIRECT_COUNT_BUFFER_MISSING,
	TILE_INDIRECT_DISPATCH_BUFFER_MISSING,
	TILE_QUANTIZATION_BUFFER_MISSING,

	COUNT
};

struct InvariantViolation {
	InvariantViolationClass violation_class = InvariantViolationClass::NONE;
	InvariantViolationReason reason = InvariantViolationReason::NONE;

	bool has_violation() const {
		return reason != InvariantViolationReason::NONE;
	}
};

inline InvariantViolationClass get_violation_class(InvariantViolationReason p_reason) {
	switch (p_reason) {
		case InvariantViolationReason::NONE:
			return InvariantViolationClass::NONE;

		case InvariantViolationReason::ATLAS_GAUSSIAN_BUFFER_MISSING:
		case InvariantViolationReason::ATLAS_ASSET_META_BUFFER_MISSING:
		case InvariantViolationReason::ATLAS_CHUNK_META_BUFFER_MISSING:
		case InvariantViolationReason::ATLAS_ASSET_CHUNK_INDEX_BUFFER_MISSING:
		case InvariantViolationReason::ATLAS_QUANTIZATION_BUFFER_MISSING:
			return InvariantViolationClass::ATLAS;

		case InvariantViolationReason::CULL_INSTANCE_BUFFER_MISSING:
		case InvariantViolationReason::CULL_ASSET_META_BUFFER_MISSING:
		case InvariantViolationReason::CULL_ASSET_CHUNK_INDEX_BUFFER_MISSING:
		case InvariantViolationReason::CULL_CHUNK_META_BUFFER_MISSING:
		case InvariantViolationReason::CULL_VISIBLE_CHUNK_BUFFER_MISSING:
		case InvariantViolationReason::CULL_COUNTER_BUFFER_MISSING:
		case InvariantViolationReason::CULL_INSTANCE_COUNT_ZERO:
		case InvariantViolationReason::CULL_DISPATCH_CHUNK_COUNT_ZERO:
		case InvariantViolationReason::CULL_MAX_VISIBLE_CHUNKS_ZERO:
			return InvariantViolationClass::CULL;

		case InvariantViolationReason::SORT_ATLAS_GAUSSIAN_BUFFER_MISSING:
		case InvariantViolationReason::SORT_INSTANCE_BUFFER_MISSING:
		case InvariantViolationReason::SORT_CHUNK_META_BUFFER_MISSING:
		case InvariantViolationReason::SORT_VISIBLE_CHUNK_BUFFER_MISSING:
		case InvariantViolationReason::SORT_SPLAT_REF_BUFFER_MISSING:
		case InvariantViolationReason::SORT_SORT_KEY_BUFFER_MISSING:
		case InvariantViolationReason::SORT_SORT_VALUE_BUFFER_MISSING:
		case InvariantViolationReason::SORT_COUNTER_BUFFER_MISSING:
		case InvariantViolationReason::SORT_CHUNK_DISPATCH_BUFFER_MISSING:
		case InvariantViolationReason::SORT_INDIRECT_COUNT_BUFFER_MISSING:
		case InvariantViolationReason::SORT_INSTANCE_COUNT_BUFFER_MISSING:
		case InvariantViolationReason::SORT_MAX_VISIBLE_SPLATS_ZERO:
		case InvariantViolationReason::SORT_MAX_CHUNK_SPLATS_ZERO:
		case InvariantViolationReason::SORT_QUANTIZATION_BUFFER_MISSING:
			return InvariantViolationClass::SORT;

		case InvariantViolationReason::RASTER_ATLAS_GAUSSIAN_BUFFER_MISSING:
		case InvariantViolationReason::RASTER_INSTANCE_BUFFER_MISSING:
		case InvariantViolationReason::RASTER_SPLAT_REF_BUFFER_MISSING:
		case InvariantViolationReason::RASTER_INSTANCE_COUNT_BUFFER_MISSING:
		case InvariantViolationReason::RASTER_QUANTIZATION_BUFFER_MISSING:
			return InvariantViolationClass::RASTER;

		case InvariantViolationReason::TILE_INSTANCE_BUFFER_MISSING:
		case InvariantViolationReason::TILE_SPLAT_REF_BUFFER_MISSING:
		case InvariantViolationReason::TILE_INDIRECT_COUNT_BUFFER_MISSING:
		case InvariantViolationReason::TILE_INDIRECT_DISPATCH_BUFFER_MISSING:
		case InvariantViolationReason::TILE_QUANTIZATION_BUFFER_MISSING:
			return InvariantViolationClass::TILE_RUNTIME;

		case InvariantViolationReason::COUNT:
			break;
	}
	return InvariantViolationClass::NONE;
}

inline const char *get_violation_class_name(InvariantViolationClass p_class) {
	switch (p_class) {
		case InvariantViolationClass::NONE:
			return "none";
		case InvariantViolationClass::ATLAS:
			return "atlas";
		case InvariantViolationClass::CULL:
			return "cull";
		case InvariantViolationClass::SORT:
			return "sort";
		case InvariantViolationClass::RASTER:
			return "raster";
		case InvariantViolationClass::TILE_RUNTIME:
			return "tile_runtime";
	}
	return "none";
}

inline const char *get_violation_reason_name(InvariantViolationReason p_reason) {
	switch (p_reason) {
		case InvariantViolationReason::NONE:
			return "none";
		case InvariantViolationReason::ATLAS_GAUSSIAN_BUFFER_MISSING:
			return "atlas_gaussian_buffer_missing";
		case InvariantViolationReason::ATLAS_ASSET_META_BUFFER_MISSING:
			return "atlas_asset_meta_buffer_missing";
		case InvariantViolationReason::ATLAS_CHUNK_META_BUFFER_MISSING:
			return "atlas_chunk_meta_buffer_missing";
		case InvariantViolationReason::ATLAS_ASSET_CHUNK_INDEX_BUFFER_MISSING:
			return "atlas_asset_chunk_index_buffer_missing";
		case InvariantViolationReason::ATLAS_QUANTIZATION_BUFFER_MISSING:
			return "atlas_quantization_buffer_missing";
		case InvariantViolationReason::CULL_INSTANCE_BUFFER_MISSING:
			return "cull_instance_buffer_missing";
		case InvariantViolationReason::CULL_ASSET_META_BUFFER_MISSING:
			return "cull_asset_meta_buffer_missing";
		case InvariantViolationReason::CULL_ASSET_CHUNK_INDEX_BUFFER_MISSING:
			return "cull_asset_chunk_index_buffer_missing";
		case InvariantViolationReason::CULL_CHUNK_META_BUFFER_MISSING:
			return "cull_chunk_meta_buffer_missing";
		case InvariantViolationReason::CULL_VISIBLE_CHUNK_BUFFER_MISSING:
			return "cull_visible_chunk_buffer_missing";
		case InvariantViolationReason::CULL_COUNTER_BUFFER_MISSING:
			return "cull_counter_buffer_missing";
		case InvariantViolationReason::CULL_INSTANCE_COUNT_ZERO:
			return "cull_instance_count_zero";
		case InvariantViolationReason::CULL_DISPATCH_CHUNK_COUNT_ZERO:
			return "cull_dispatch_chunk_count_zero";
		case InvariantViolationReason::CULL_MAX_VISIBLE_CHUNKS_ZERO:
			return "cull_max_visible_chunks_zero";
		case InvariantViolationReason::SORT_ATLAS_GAUSSIAN_BUFFER_MISSING:
			return "sort_atlas_gaussian_buffer_missing";
		case InvariantViolationReason::SORT_INSTANCE_BUFFER_MISSING:
			return "sort_instance_buffer_missing";
		case InvariantViolationReason::SORT_CHUNK_META_BUFFER_MISSING:
			return "sort_chunk_meta_buffer_missing";
		case InvariantViolationReason::SORT_VISIBLE_CHUNK_BUFFER_MISSING:
			return "sort_visible_chunk_buffer_missing";
		case InvariantViolationReason::SORT_SPLAT_REF_BUFFER_MISSING:
			return "sort_splat_ref_buffer_missing";
		case InvariantViolationReason::SORT_SORT_KEY_BUFFER_MISSING:
			return "sort_sort_key_buffer_missing";
		case InvariantViolationReason::SORT_SORT_VALUE_BUFFER_MISSING:
			return "sort_sort_value_buffer_missing";
		case InvariantViolationReason::SORT_COUNTER_BUFFER_MISSING:
			return "sort_counter_buffer_missing";
		case InvariantViolationReason::SORT_CHUNK_DISPATCH_BUFFER_MISSING:
			return "sort_chunk_dispatch_buffer_missing";
		case InvariantViolationReason::SORT_INDIRECT_COUNT_BUFFER_MISSING:
			return "sort_indirect_count_buffer_missing";
		case InvariantViolationReason::SORT_INSTANCE_COUNT_BUFFER_MISSING:
			return "sort_instance_count_buffer_missing";
		case InvariantViolationReason::SORT_MAX_VISIBLE_SPLATS_ZERO:
			return "sort_max_visible_splats_zero";
		case InvariantViolationReason::SORT_MAX_CHUNK_SPLATS_ZERO:
			return "sort_max_chunk_splats_zero";
		case InvariantViolationReason::SORT_QUANTIZATION_BUFFER_MISSING:
			return "sort_quantization_buffer_missing";
		case InvariantViolationReason::RASTER_ATLAS_GAUSSIAN_BUFFER_MISSING:
			return "raster_atlas_gaussian_buffer_missing";
		case InvariantViolationReason::RASTER_INSTANCE_BUFFER_MISSING:
			return "raster_instance_buffer_missing";
		case InvariantViolationReason::RASTER_SPLAT_REF_BUFFER_MISSING:
			return "raster_splat_ref_buffer_missing";
		case InvariantViolationReason::RASTER_INSTANCE_COUNT_BUFFER_MISSING:
			return "raster_instance_count_buffer_missing";
		case InvariantViolationReason::RASTER_QUANTIZATION_BUFFER_MISSING:
			return "raster_quantization_buffer_missing";
		case InvariantViolationReason::TILE_INSTANCE_BUFFER_MISSING:
			return "tile_instance_buffer_missing";
		case InvariantViolationReason::TILE_SPLAT_REF_BUFFER_MISSING:
			return "tile_splat_ref_buffer_missing";
		case InvariantViolationReason::TILE_INDIRECT_COUNT_BUFFER_MISSING:
			return "tile_indirect_count_buffer_missing";
		case InvariantViolationReason::TILE_INDIRECT_DISPATCH_BUFFER_MISSING:
			return "tile_indirect_dispatch_buffer_missing";
		case InvariantViolationReason::TILE_QUANTIZATION_BUFFER_MISSING:
			return "tile_quantization_buffer_missing";
		case InvariantViolationReason::COUNT:
			return "count";
	}
	return "none";
}

inline const char *get_violation_route(InvariantViolationReason p_reason) {
	switch (get_violation_class(p_reason)) {
		case InvariantViolationClass::ATLAS:
			return "instance_pipeline/invariant/atlas";
		case InvariantViolationClass::CULL:
			return "instance_pipeline/invariant/cull";
		case InvariantViolationClass::SORT:
			return "instance_pipeline/invariant/sort";
		case InvariantViolationClass::RASTER:
			return "instance_pipeline/invariant/raster";
		case InvariantViolationClass::TILE_RUNTIME:
			return "instance_pipeline/invariant/tile_runtime";
		case InvariantViolationClass::NONE:
			return "instance_pipeline/invariant/none";
	}
	return "instance_pipeline/invariant/none";
}

inline InvariantViolation make_violation(InvariantViolationReason p_reason) {
	InvariantViolation violation;
	violation.reason = p_reason;
	violation.violation_class = get_violation_class(p_reason);
	return violation;
}

inline InvariantViolationReason first_atlas_violation(const GaussianSplatRenderer::InstancePipelineBuffers &p_buffers) {
	if (!p_buffers.atlas_gaussian_buffer.is_valid()) {
		return InvariantViolationReason::ATLAS_GAUSSIAN_BUFFER_MISSING;
	}
	if (!p_buffers.asset_meta_buffer.is_valid()) {
		return InvariantViolationReason::ATLAS_ASSET_META_BUFFER_MISSING;
	}
	if (!p_buffers.chunk_meta_buffer.is_valid()) {
		return InvariantViolationReason::ATLAS_CHUNK_META_BUFFER_MISSING;
	}
	if (!p_buffers.asset_chunk_index_buffer.is_valid()) {
		return InvariantViolationReason::ATLAS_ASSET_CHUNK_INDEX_BUFFER_MISSING;
	}
	if (p_buffers.quantization_required && !p_buffers.quantization_buffer.is_valid()) {
		return InvariantViolationReason::ATLAS_QUANTIZATION_BUFFER_MISSING;
	}
	return InvariantViolationReason::NONE;
}

inline InvariantViolationReason first_cull_violation(const GaussianSplatRenderer::InstancePipelineBuffers &p_buffers) {
	if (!p_buffers.instance_buffer.is_valid()) {
		return InvariantViolationReason::CULL_INSTANCE_BUFFER_MISSING;
	}
	if (!p_buffers.asset_meta_buffer.is_valid()) {
		return InvariantViolationReason::CULL_ASSET_META_BUFFER_MISSING;
	}
	if (!p_buffers.asset_chunk_index_buffer.is_valid()) {
		return InvariantViolationReason::CULL_ASSET_CHUNK_INDEX_BUFFER_MISSING;
	}
	if (!p_buffers.chunk_meta_buffer.is_valid()) {
		return InvariantViolationReason::CULL_CHUNK_META_BUFFER_MISSING;
	}
	if (!p_buffers.visible_chunk_buffer.is_valid()) {
		return InvariantViolationReason::CULL_VISIBLE_CHUNK_BUFFER_MISSING;
	}
	if (!p_buffers.counter_buffer.is_valid()) {
		return InvariantViolationReason::CULL_COUNTER_BUFFER_MISSING;
	}
	if (p_buffers.instance_count == 0) {
		return InvariantViolationReason::CULL_INSTANCE_COUNT_ZERO;
	}
	if (p_buffers.dispatch_chunk_count == 0) {
		return InvariantViolationReason::CULL_DISPATCH_CHUNK_COUNT_ZERO;
	}
	if (p_buffers.max_visible_chunks == 0) {
		return InvariantViolationReason::CULL_MAX_VISIBLE_CHUNKS_ZERO;
	}
	return InvariantViolationReason::NONE;
}

inline InvariantViolationReason first_sort_violation(const GaussianSplatRenderer::InstancePipelineBuffers &p_buffers) {
	if (!p_buffers.atlas_gaussian_buffer.is_valid()) {
		return InvariantViolationReason::SORT_ATLAS_GAUSSIAN_BUFFER_MISSING;
	}
	if (!p_buffers.instance_buffer.is_valid()) {
		return InvariantViolationReason::SORT_INSTANCE_BUFFER_MISSING;
	}
	if (!p_buffers.chunk_meta_buffer.is_valid()) {
		return InvariantViolationReason::SORT_CHUNK_META_BUFFER_MISSING;
	}
	if (!p_buffers.visible_chunk_buffer.is_valid()) {
		return InvariantViolationReason::SORT_VISIBLE_CHUNK_BUFFER_MISSING;
	}
	if (!p_buffers.splat_ref_buffer.is_valid()) {
		return InvariantViolationReason::SORT_SPLAT_REF_BUFFER_MISSING;
	}
	if (!p_buffers.sort_key_buffer.is_valid()) {
		return InvariantViolationReason::SORT_SORT_KEY_BUFFER_MISSING;
	}
	if (!p_buffers.sort_value_buffer.is_valid()) {
		return InvariantViolationReason::SORT_SORT_VALUE_BUFFER_MISSING;
	}
	if (!p_buffers.counter_buffer.is_valid()) {
		return InvariantViolationReason::SORT_COUNTER_BUFFER_MISSING;
	}
	if (!p_buffers.chunk_dispatch_buffer.is_valid()) {
		return InvariantViolationReason::SORT_CHUNK_DISPATCH_BUFFER_MISSING;
	}
	if (!p_buffers.indirect_count_buffer.is_valid()) {
		return InvariantViolationReason::SORT_INDIRECT_COUNT_BUFFER_MISSING;
	}
	if (!p_buffers.instance_count_buffer.is_valid()) {
		return InvariantViolationReason::SORT_INSTANCE_COUNT_BUFFER_MISSING;
	}
	if (p_buffers.max_visible_splats == 0) {
		return InvariantViolationReason::SORT_MAX_VISIBLE_SPLATS_ZERO;
	}
	if (p_buffers.max_chunk_splats == 0) {
		return InvariantViolationReason::SORT_MAX_CHUNK_SPLATS_ZERO;
	}
	if (p_buffers.quantization_required && !p_buffers.quantization_buffer.is_valid()) {
		return InvariantViolationReason::SORT_QUANTIZATION_BUFFER_MISSING;
	}
	return InvariantViolationReason::NONE;
}

inline InvariantViolationReason first_raster_violation(const GaussianSplatRenderer::InstancePipelineBuffers &p_buffers) {
	if (!p_buffers.atlas_gaussian_buffer.is_valid()) {
		return InvariantViolationReason::RASTER_ATLAS_GAUSSIAN_BUFFER_MISSING;
	}
	if (!p_buffers.instance_buffer.is_valid()) {
		return InvariantViolationReason::RASTER_INSTANCE_BUFFER_MISSING;
	}
	if (!p_buffers.splat_ref_buffer.is_valid()) {
		return InvariantViolationReason::RASTER_SPLAT_REF_BUFFER_MISSING;
	}
	if (!p_buffers.instance_count_buffer.is_valid()) {
		return InvariantViolationReason::RASTER_INSTANCE_COUNT_BUFFER_MISSING;
	}
	if (p_buffers.quantization_required && !p_buffers.quantization_buffer.is_valid()) {
		return InvariantViolationReason::RASTER_QUANTIZATION_BUFFER_MISSING;
	}
	return InvariantViolationReason::NONE;
}

inline InvariantViolation evaluate_streaming_activation(const GaussianSplatRenderer::InstancePipelineBuffers &p_buffers,
		bool p_atlas_buffers_required) {
	if (p_atlas_buffers_required) {
		const InvariantViolationReason atlas_violation = first_atlas_violation(p_buffers);
		if (atlas_violation != InvariantViolationReason::NONE) {
			return make_violation(atlas_violation);
		}
	}

	const InvariantViolationReason cull_violation = first_cull_violation(p_buffers);
	if (cull_violation != InvariantViolationReason::NONE) {
		return make_violation(cull_violation);
	}

	const InvariantViolationReason sort_violation = first_sort_violation(p_buffers);
	if (sort_violation != InvariantViolationReason::NONE) {
		return make_violation(sort_violation);
	}

	const InvariantViolationReason raster_violation = first_raster_violation(p_buffers);
	if (raster_violation != InvariantViolationReason::NONE) {
		return make_violation(raster_violation);
	}

	return InvariantViolation();
}

inline InvariantViolationReason first_tile_runtime_violation(const RID &p_instance_buffer, const RID &p_splat_ref_buffer,
		const RID &p_indirect_count_buffer, const RID &p_indirect_dispatch_buffer,
		bool p_quantization_required, const RID &p_quantization_buffer) {
	if (!p_instance_buffer.is_valid()) {
		return InvariantViolationReason::TILE_INSTANCE_BUFFER_MISSING;
	}
	if (!p_splat_ref_buffer.is_valid()) {
		return InvariantViolationReason::TILE_SPLAT_REF_BUFFER_MISSING;
	}
	if (!p_indirect_count_buffer.is_valid()) {
		return InvariantViolationReason::TILE_INDIRECT_COUNT_BUFFER_MISSING;
	}
	if (!p_indirect_dispatch_buffer.is_valid()) {
		return InvariantViolationReason::TILE_INDIRECT_DISPATCH_BUFFER_MISSING;
	}
	if (p_quantization_required && !p_quantization_buffer.is_valid()) {
		return InvariantViolationReason::TILE_QUANTIZATION_BUFFER_MISSING;
	}
	return InvariantViolationReason::NONE;
}

inline bool has_atlas_buffers(const GaussianSplatRenderer::InstancePipelineBuffers &p_buffers) {
	return first_atlas_violation(p_buffers) == InvariantViolationReason::NONE;
}

inline bool has_cull_buffers(const GaussianSplatRenderer::InstancePipelineBuffers &p_buffers) {
	return first_cull_violation(p_buffers) == InvariantViolationReason::NONE;
}

inline bool has_sort_buffers(const GaussianSplatRenderer::InstancePipelineBuffers &p_buffers) {
	return first_sort_violation(p_buffers) == InvariantViolationReason::NONE;
}

inline bool has_raster_buffers(const GaussianSplatRenderer::InstancePipelineBuffers &p_buffers) {
	return first_raster_violation(p_buffers) == InvariantViolationReason::NONE;
}

} // namespace InstancePipelineContract
} // namespace GaussianSplatting

#endif // GS_INSTANCE_PIPELINE_CONTRACT_H
