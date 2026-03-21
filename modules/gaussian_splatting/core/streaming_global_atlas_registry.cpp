/**************************************************************************/
/* streaming_global_atlas_registry.cpp                                    */
/*                                                                        */
/* Global atlas publication state for the streaming system.               */
/**************************************************************************/

#include "streaming_global_atlas_registry.h"

#include "gaussian_streaming.h"
#include "core/config/project_settings.h"
#include "core/templates/span.h"
#include "../logger/gs_logger.h"
#include <algorithm>
#include <cfloat>
#include <cstdint>

namespace {

bool _is_streaming_debug_enabled() {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	if (!ps) {
		return false;
	}
	if (ps->get_setting("rendering/gaussian_splatting/debug/enable_all_debug", false)) {
		return true;
	}
	if (ps->get_setting("rendering/gaussian_splatting/debug/enable_frame_logging", false)) {
		return true;
	}
	return ps->get_setting("rendering/gaussian_splatting/debug/enable_data_logging", false);
}

void _clear_dirty_flags(LocalVector<uint8_t> &flags) {
	for (uint32_t i = 0; i < flags.size(); i++) {
		flags[i] = 0;
	}
}

bool _ensure_placeholder_buffer(RenderingDevice *p_rd, RID &r_buffer, uint32_t &r_logical_size, const char *p_name) {
	ERR_FAIL_NULL_V(p_rd, false);

	if (r_buffer.is_valid() && r_logical_size == 0) {
		return false;
	}

	if (r_buffer.is_valid()) {
		p_rd->free(r_buffer);
	}

	const uint32_t zero = 0;
	Span<const uint32_t> zero_span(&zero, 1);
	r_buffer = p_rd->storage_buffer_create(sizeof(uint32_t), zero_span.reinterpret<uint8_t>());
	p_rd->set_resource_name(r_buffer, p_name);
	r_logical_size = 0;
	return true;
}

} // namespace

uint64_t StreamingGlobalAtlasRegistry::get_auxiliary_vram_overhead_bytes() const {
	return uint64_t(asset_meta_buffer_size) +
			uint64_t(chunk_meta_buffer_size) +
			uint64_t(asset_chunk_index_buffer_size);
}

void StreamingGlobalAtlasRegistry::cleanup(RenderingDevice *p_rd) {
	auto release_buffer = [p_rd](RID &buffer, uint32_t &size, const char *label) {
		if (!buffer.is_valid()) {
			size = 0;
			return;
		}
		if (p_rd) {
			p_rd->free(buffer);
		} else {
			WARN_PRINT(vformat("[Streaming] %s RID %s exists but RenderingDevice is null; dropping stale RID handle.",
					label,
					String::num_uint64(static_cast<uint64_t>(buffer.get_id()))));
		}
		buffer = RID();
		size = 0;
	};

	release_buffer(asset_meta_buffer, asset_meta_buffer_size, "asset_meta_buffer_release");
	release_buffer(chunk_meta_buffer, chunk_meta_buffer_size, "chunk_meta_buffer_release");
	release_buffer(asset_chunk_index_buffer, asset_chunk_index_buffer_size, "asset_chunk_index_buffer_release");

	asset_meta_cpu.clear();
	chunk_meta_cpu.clear();
	asset_chunk_index_cpu.clear();
	chunk_meta_dirty_indices.clear();
	chunk_meta_dirty_flags.clear();
	asset_meta_dirty = false;
	asset_chunk_index_dirty = false;
	chunk_meta_dirty_all = false;
	asset_registry_dirty = false;
	global_atlas_state.atlas_gaussian_buffer = RID();
	global_atlas_state.atlas_gaussian_count = 0;
	global_atlas_state.asset_meta_buffer = RID();
	global_atlas_state.chunk_meta_buffer = RID();
	global_atlas_state.asset_chunk_index_buffer = RID();
	global_atlas_state.quantization_buffer = RID();
	max_chunk_count_per_asset = 0;
	max_chunk_splats = 0;
}

void StreamingGlobalAtlasRegistry::build_cpu_state(GaussianStreamingSystem &system) {
	asset_registry_dirty = false;
	const bool log_enabled = _is_streaming_debug_enabled();

	uint32_t total_chunks = 0;
	max_chunk_count_per_asset = 0;
	max_chunk_splats = 0;
	for (uint32_t asset_id : system.asset_registry.atlas_asset_order) {
		GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
		if (!asset) {
			continue;
		}
		LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
		total_chunks += asset_chunks.size();
		max_chunk_count_per_asset = MAX(max_chunk_count_per_asset, asset_chunks.size());
		for (uint32_t i = 0; i < asset_chunks.size(); i++) {
			max_chunk_splats = MAX(max_chunk_splats, asset_chunks[i].count);
		}
	}
	const uint32_t dense_count = system.asset_registry.dense_to_asset_id.size();
	asset_meta_cpu.resize(dense_count);
	for (uint32_t i = 0; i < dense_count; i++) {
		asset_meta_cpu[i] = {};
	}

	chunk_meta_cpu.resize(total_chunks);
	asset_chunk_index_cpu.resize(total_chunks);
	chunk_meta_dirty_flags.resize(total_chunks);
	_clear_dirty_flags(chunk_meta_dirty_flags);
	chunk_meta_dirty_indices.clear();
	chunk_meta_dirty_indices.reserve(total_chunks);

	uint32_t chunk_meta_cursor = 0;
	uint32_t chunk_index_cursor = 0;

	for (uint32_t asset_id : system.asset_registry.atlas_asset_order) {
		GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
		if (!asset) {
			continue;
		}
		if (asset->dense_id == GaussianStreamingSystem::INVALID_ASSET_ID || asset->dense_id >= asset_meta_cpu.size()) {
			continue;
		}

		LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
		asset->chunk_meta_base = chunk_meta_cursor;
		asset->chunk_meta_count = asset_chunks.size();
		asset->chunk_index_base = chunk_index_cursor;
		asset->chunk_index_count = asset_chunks.size();

		if (log_enabled) {
			float min_radius = FLT_MAX;
			float max_radius = 0.0f;
			uint32_t invalid_count = 0;
			for (uint32_t i = 0; i < asset_chunks.size(); i++) {
				const GaussianStreamingSystem::StreamingChunk &chunk = asset_chunks[i];
				Vector3 half = chunk.bounds.size * 0.5f;
				const float radius = half.length();
				if (!Math::is_finite(radius) || radius <= 0.0f) {
					invalid_count++;
					continue;
				}
				min_radius = MIN(min_radius, radius);
				max_radius = MAX(max_radius, radius);
			}
			if (!asset_chunks.is_empty()) {
				const float safe_min = min_radius == FLT_MAX ? 0.0f : min_radius;
				GS_LOG_STREAMING_DEBUG(vformat("[Streaming] Asset %d chunks=%d radius_range=(%.3f..%.3f) invalid=%d",
						asset_id, asset_chunks.size(), safe_min, max_radius, invalid_count));
				if (invalid_count > 0) {
					GS_LOG_STREAMING_WARN(vformat("[Streaming] Asset %d has %d chunks with invalid bounds radius",
							asset_id, invalid_count));
				}
			}
		}

		AssetMetaGPU asset_meta = {};
		asset_meta.lod_count = MAX(uint32_t(1), asset->lod_count);
		asset_meta.sh_degree = asset->sh_degree;
		asset_meta.flags = 0;

		Vector3 asset_center = asset->bounds.get_center();
		Vector3 asset_half = asset->bounds.size * 0.5f;
		float asset_radius = asset_half.length();
		asset_meta.bounds_center_local[0] = asset_center.x;
		asset_meta.bounds_center_local[1] = asset_center.y;
		asset_meta.bounds_center_local[2] = asset_center.z;
		asset_meta.bounds_radius_local = asset_radius;

		asset_meta.chunk_index_base = asset->chunk_index_base;
		asset_meta.chunk_index_count = asset->chunk_index_count;
		asset_meta.quant_chunk_base = system.per_chunk_quantization_enabled ? asset->quant_base : 0;
		asset_meta.quant_chunk_count = system.per_chunk_quantization_enabled ? asset->quant_count : 0;
		asset_meta.lod_ranges[0].base = asset->chunk_index_base;
		asset_meta.lod_ranges[0].count = asset->chunk_index_count;

		static int asset_meta_debug_counter = 0;
		if (log_enabled && ++asset_meta_debug_counter <= 10) {
			GS_LOG_STREAMING_DEBUG(vformat("[ASSET-META] dense_id=%d lod_count=%d chunk_idx_base=%d chunk_idx_count=%d lod_range[0]=(base=%d,count=%d) chunks=%d radius=%.3f",
					asset->dense_id, asset_meta.lod_count, asset_meta.chunk_index_base, asset_meta.chunk_index_count,
					asset_meta.lod_ranges[0].base, asset_meta.lod_ranges[0].count, int(asset_chunks.size()), asset_radius));
		}

		asset_meta_cpu[asset->dense_id] = asset_meta;

		for (uint32_t i = 0; i < asset_chunks.size(); i++) {
			const uint32_t global_idx = chunk_meta_cursor + i;
			if (chunk_index_cursor < asset_chunk_index_cpu.size()) {
				asset_chunk_index_cpu[chunk_index_cursor].chunk_id = global_idx;
			}
			update_chunk_meta_entry(system, asset_id, i);
			chunk_index_cursor++;
		}

		chunk_meta_cursor += asset_chunks.size();
		asset->metadata_dirty = false;
	}

	uint32_t resident_meta_chunks = 0;
	for (uint32_t i = 0; i < chunk_meta_cpu.size(); i++) {
		if (chunk_meta_cpu[i].splat_count > 0) {
			resident_meta_chunks++;
		}
	}
	if (log_enabled) {
		const uint64_t system_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&system));
		GS_LOG_STREAMING_DEBUG(vformat("[Streaming DIAG] atlas cpu meta built: system=%s total_chunks=%d resident_meta_chunks=%d loaded_chunks_count=%d",
				String::num_uint64(system_id), chunk_meta_cpu.size(), resident_meta_chunks, system.budget.loaded_chunks_count));
	}

	asset_meta_dirty = true;
	asset_chunk_index_dirty = true;
	chunk_meta_dirty_all = true;
	asset_registry_dirty = false;
}

void StreamingGlobalAtlasRegistry::update_chunk_meta_entry(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx) {
	if (asset_registry_dirty) {
		build_cpu_state(system);
	}

	GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
	if (!asset) {
		return;
	}

	LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
	if (chunk_idx >= asset_chunks.size()) {
		return;
	}

	const uint32_t global_idx = asset->chunk_meta_base + chunk_idx;
	if (global_idx >= chunk_meta_cpu.size()) {
		return;
	}

	const GaussianStreamingSystem::StreamingChunk &chunk = asset_chunks[chunk_idx];
	ChunkMetaGPU meta = {};
	const bool resident = chunk.is_loaded && !chunk.upload_pending && chunk.buffer_slot != UINT32_MAX;
	const uint32_t effective_splat_count = MIN(chunk.effective_count, chunk.count);
	const uint32_t sh_band_limit = uint32_t(CLAMP(chunk.sh_band_level, 0, 3));
	if (resident) {
		meta.atlas_base = chunk.buffer_slot * GaussianStreamingSystem::CHUNK_SIZE;
		meta.splat_count = effective_splat_count;
	} else {
		meta.atlas_base = 0;
		meta.splat_count = 0;
	}
	if (system.per_chunk_quantization_enabled) {
		meta.quant_base = asset->quant_base + chunk_idx;
		meta.quant_count = 1;
	} else {
		meta.quant_base = 0;
		meta.quant_count = 0;
	}

	Vector3 center;
	float radius = 0.0f;
	if (chunk.count > 0) {
		center = chunk.bounds.get_center();
		Vector3 half = chunk.bounds.size * 0.5f;
		radius = half.length();
	}
	meta.bounds_center_local[0] = center.x;
	meta.bounds_center_local[1] = center.y;
	meta.bounds_center_local[2] = center.z;
	meta.bounds_radius_local = radius;

	meta.asset_id = asset->dense_id;
	meta.lod_level = chunk.current_lod_level;
	meta.flags = 0;
	meta.sh_limit = sh_band_limit;

	chunk_meta_cpu[global_idx] = meta;
}

void StreamingGlobalAtlasRegistry::mark_chunk_meta_dirty(GaussianStreamingSystem &system, uint32_t chunk_idx) {
	mark_chunk_meta_dirty(system, GaussianStreamingSystem::PRIMARY_ASSET_ID, chunk_idx);
}

void StreamingGlobalAtlasRegistry::mark_chunk_meta_dirty(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx) {
	if (asset_registry_dirty) {
		build_cpu_state(system);
	}

	GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
	if (!asset) {
		return;
	}
	const uint32_t global_idx = asset->chunk_meta_base + chunk_idx;
	if (global_idx >= chunk_meta_cpu.size()) {
		return;
	}

	update_chunk_meta_entry(system, asset_id, chunk_idx);

	if (chunk_meta_dirty_all) {
		return;
	}
	if (chunk_meta_dirty_flags.is_empty() || global_idx >= chunk_meta_dirty_flags.size()) {
		return;
	}
	if (chunk_meta_dirty_flags[global_idx] == 0) {
		chunk_meta_dirty_flags[global_idx] = 1;
		chunk_meta_dirty_indices.push_back(global_idx);
	}
}

void StreamingGlobalAtlasRegistry::sync_to_gpu(GaussianStreamingSystem &system, RenderingDevice *p_rd) {
	global_atlas_state.atlas_gaussian_buffer = system.persistent_buffer;
	global_atlas_state.atlas_gaussian_count = system.get_buffer_capacity_splats();
	global_atlas_state.quantization_buffer = system.per_chunk_quantization_enabled ? system.quantization_buffer : RID();

	const bool quantization_rebuild = system.quantization_dirty;
	const uint32_t quantization_expected_size = uint32_t(system.quantization_gpu_data.size()) * sizeof(ChunkQuantizationGPU);
	bool atlas_dirty = quantization_rebuild || asset_registry_dirty || asset_meta_dirty ||
			asset_chunk_index_dirty || chunk_meta_dirty_all || !chunk_meta_dirty_indices.is_empty();
	if (system.per_chunk_quantization_enabled &&
			(system.quantization_dirty || !system.quantization_buffer.is_valid() || system.quantization_buffer_size != quantization_expected_size)) {
		const bool quantization_resource_changed = system._upload_quantization_buffer(p_rd);
		atlas_dirty = atlas_dirty || quantization_resource_changed;
		global_atlas_state.quantization_buffer = system.quantization_buffer;
	} else if (!system.per_chunk_quantization_enabled) {
		if (system._release_quantization_buffer(p_rd, "_sync_global_atlas_state(disabled)", true)) {
			atlas_dirty = true;
		}
		system.quantization_dirty = false;
	}

	uint32_t total_chunks = 0;
	for (uint32_t asset_id : system.asset_registry.atlas_asset_order) {
		GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
		if (!asset) {
			continue;
		}
		total_chunks += system._get_asset_chunks(*asset).size();
	}
	const uint32_t dense_count = system.asset_registry.dense_to_asset_id.size();

	if (asset_registry_dirty ||
			quantization_rebuild ||
			asset_meta_cpu.is_empty() ||
			asset_meta_cpu.size() != dense_count ||
			chunk_meta_cpu.size() != total_chunks ||
			asset_chunk_index_cpu.size() != total_chunks) {
		atlas_dirty = true;
		build_cpu_state(system);
	}

	if (!p_rd) {
		if (atlas_dirty) {
			WARN_PRINT_ONCE("[Streaming DIAG] _sync_global_atlas_state skipped GPU upload because RenderingDevice is null while atlas is dirty.");
		}
		global_atlas_state.asset_meta_buffer = asset_meta_buffer;
		global_atlas_state.chunk_meta_buffer = chunk_meta_buffer;
		global_atlas_state.asset_chunk_index_buffer = asset_chunk_index_buffer;
		return;
	}

	const uint32_t asset_meta_size = asset_meta_cpu.size() * sizeof(AssetMetaGPU);
	if (asset_meta_size == 0) {
		asset_meta_dirty = false;
		atlas_dirty = _ensure_placeholder_buffer(p_rd, asset_meta_buffer, asset_meta_buffer_size,
				"GS_Streaming_AssetMetaBuffer_Empty") || atlas_dirty;
	} else {
		Span<const AssetMetaGPU> asset_span(asset_meta_cpu.ptr(), asset_meta_cpu.size());
		if (!asset_meta_buffer.is_valid() || asset_meta_buffer_size != asset_meta_size) {
			if (asset_meta_buffer.is_valid()) {
				p_rd->free(asset_meta_buffer);
			}
			asset_meta_buffer = p_rd->storage_buffer_create(asset_meta_size, asset_span.reinterpret<uint8_t>());
			p_rd->set_resource_name(asset_meta_buffer, "GS_Streaming_AssetMetaBuffer");
			asset_meta_buffer_size = asset_meta_size;
			asset_meta_dirty = false;
			atlas_dirty = true;
		} else if (asset_meta_dirty) {
			p_rd->buffer_update(asset_meta_buffer, 0, asset_meta_size, asset_span.reinterpret<uint8_t>().ptr());
			asset_meta_dirty = false;
			atlas_dirty = true;
		}
	}

	const uint32_t chunk_meta_size = chunk_meta_cpu.size() * sizeof(ChunkMetaGPU);
	if (chunk_meta_size == 0) {
		chunk_meta_dirty_all = false;
		chunk_meta_dirty_indices.clear();
		_clear_dirty_flags(chunk_meta_dirty_flags);
		atlas_dirty = _ensure_placeholder_buffer(p_rd, chunk_meta_buffer, chunk_meta_buffer_size,
				"GS_Streaming_ChunkMetaBuffer_Empty") || atlas_dirty;
	} else {
		Span<const ChunkMetaGPU> chunk_span(chunk_meta_cpu.ptr(), chunk_meta_cpu.size());
		if (!chunk_meta_buffer.is_valid() || chunk_meta_buffer_size != chunk_meta_size) {
			if (chunk_meta_buffer.is_valid()) {
				p_rd->free(chunk_meta_buffer);
			}
			chunk_meta_buffer = p_rd->storage_buffer_create(chunk_meta_size, chunk_span.reinterpret<uint8_t>());
			p_rd->set_resource_name(chunk_meta_buffer, "GS_Streaming_ChunkMetaBuffer");
			chunk_meta_buffer_size = chunk_meta_size;
			chunk_meta_dirty_all = false;
			chunk_meta_dirty_indices.clear();
			_clear_dirty_flags(chunk_meta_dirty_flags);
			atlas_dirty = true;
		} else if (chunk_meta_dirty_all) {
			p_rd->buffer_update(chunk_meta_buffer, 0, chunk_meta_size, chunk_span.reinterpret<uint8_t>().ptr());
			chunk_meta_dirty_all = false;
			chunk_meta_dirty_indices.clear();
			_clear_dirty_flags(chunk_meta_dirty_flags);
			atlas_dirty = true;
		} else if (!chunk_meta_dirty_indices.is_empty()) {
			uint32_t dirty_count = chunk_meta_dirty_indices.size();
			uint32_t *dirty_ptr = chunk_meta_dirty_indices.ptr();
			std::sort(dirty_ptr, dirty_ptr + dirty_count);

			uint32_t filtered_count = 0;
			uint32_t last_idx = UINT32_MAX;
			for (uint32_t i = 0; i < dirty_count; i++) {
				const uint32_t idx = dirty_ptr[i];
				if (idx >= chunk_meta_cpu.size() || idx == last_idx) {
					continue;
				}
				dirty_ptr[filtered_count++] = idx;
				last_idx = idx;
			}

			uint32_t range_start = UINT32_MAX;
			uint32_t range_end = UINT32_MAX;
			for (uint32_t i = 0; i <= filtered_count; i++) {
				const bool at_end = i == filtered_count;
				const uint32_t idx = at_end ? UINT32_MAX : dirty_ptr[i];
				if (range_start == UINT32_MAX) {
					if (!at_end) {
						range_start = idx;
						range_end = idx;
					}
					continue;
				}

				if (!at_end && idx == range_end + 1u) {
					range_end = idx;
					continue;
				}

				const uint32_t range_count = range_end - range_start + 1u;
				const uint32_t offset = range_start * sizeof(ChunkMetaGPU);
				const uint32_t update_size = range_count * sizeof(ChunkMetaGPU);
				p_rd->buffer_update(chunk_meta_buffer, offset, update_size, chunk_meta_cpu.ptr() + range_start);
				for (uint32_t clear_idx = range_start; clear_idx <= range_end; clear_idx++) {
					if (clear_idx < chunk_meta_dirty_flags.size()) {
						chunk_meta_dirty_flags[clear_idx] = 0;
					}
				}

				if (!at_end) {
					range_start = idx;
					range_end = idx;
				} else {
					range_start = UINT32_MAX;
					range_end = UINT32_MAX;
				}
			}

			chunk_meta_dirty_indices.clear();
			atlas_dirty = true;
		}
	}

	const uint32_t chunk_index_size = asset_chunk_index_cpu.size() * sizeof(AssetChunkIndexGPU);
	if (chunk_index_size == 0) {
		asset_chunk_index_dirty = false;
		atlas_dirty = _ensure_placeholder_buffer(p_rd, asset_chunk_index_buffer, asset_chunk_index_buffer_size,
				"GS_Streaming_AssetChunkIndexBuffer_Empty") || atlas_dirty;
	} else {
		Span<const AssetChunkIndexGPU> chunk_index_span(asset_chunk_index_cpu.ptr(), asset_chunk_index_cpu.size());
		if (!asset_chunk_index_buffer.is_valid() || asset_chunk_index_buffer_size != chunk_index_size) {
			if (asset_chunk_index_buffer.is_valid()) {
				p_rd->free(asset_chunk_index_buffer);
			}
			asset_chunk_index_buffer = p_rd->storage_buffer_create(chunk_index_size, chunk_index_span.reinterpret<uint8_t>());
			p_rd->set_resource_name(asset_chunk_index_buffer, "GS_Streaming_AssetChunkIndexBuffer");
			asset_chunk_index_buffer_size = chunk_index_size;
			asset_chunk_index_dirty = false;
			atlas_dirty = true;
		} else if (asset_chunk_index_dirty) {
			p_rd->buffer_update(asset_chunk_index_buffer, 0, chunk_index_size, chunk_index_span.reinterpret<uint8_t>().ptr());
			asset_chunk_index_dirty = false;
			atlas_dirty = true;
		}
	}

	global_atlas_state.asset_meta_buffer = asset_meta_buffer;
	global_atlas_state.chunk_meta_buffer = chunk_meta_buffer;
	global_atlas_state.asset_chunk_index_buffer = asset_chunk_index_buffer;
	static int atlas_sync_diag_counter = 0;
	if (++atlas_sync_diag_counter <= 20) {
		print_line(vformat("[Streaming DIAG] atlas sync publish: system=%s chunk_meta_rid=%d asset_meta_rid=%d chunk_index_rid=%d chunks=%d",
				String::num_uint64(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&system))),
				chunk_meta_buffer.get_id(),
				asset_meta_buffer.get_id(),
				asset_chunk_index_buffer.get_id(),
				chunk_meta_cpu.size()));
	}
	if (atlas_dirty) {
		global_atlas_state.atlas_generation++;
		if (global_atlas_state.atlas_generation == 0) {
			global_atlas_state.atlas_generation = 1;
		}
	}
}
