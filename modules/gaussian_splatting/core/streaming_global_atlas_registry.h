#ifndef STREAMING_GLOBAL_ATLAS_REGISTRY_H
#define STREAMING_GLOBAL_ATLAS_REGISTRY_H

#include "core/templates/local_vector.h"
#include "core/templates/rid.h"
#include "../renderer/gaussian_gpu_layout.h"
#include <cstdint>

class GaussianStreamingSystem;
class RenderingDevice;

struct GlobalAtlasState {
	RID atlas_gaussian_buffer;
	uint32_t atlas_gaussian_count = 0;
	RID asset_meta_buffer;
	RID chunk_meta_buffer;
	RID asset_chunk_index_buffer;
	RID quantization_buffer;
	uint64_t atlas_generation = 0;
};

class StreamingGlobalAtlasRegistry {
	friend class GaussianStreamingSystem;

public:
	void cleanup(RenderingDevice *p_rd);
	void mark_asset_registry_dirty() { asset_registry_dirty = true; }
	void build_cpu_state(GaussianStreamingSystem &system);
	void update_chunk_meta_entry(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx);
	void mark_chunk_meta_dirty(GaussianStreamingSystem &system, uint32_t chunk_idx);
	void mark_chunk_meta_dirty(GaussianStreamingSystem &system, uint32_t asset_id, uint32_t chunk_idx);
	void sync_to_gpu(GaussianStreamingSystem &system, RenderingDevice *p_rd);

	uint32_t get_max_chunk_count_per_asset() const { return max_chunk_count_per_asset; }
	uint32_t get_max_chunk_splats() const { return max_chunk_splats; }
	uint64_t get_auxiliary_vram_overhead_bytes() const;
	const GlobalAtlasState &get_global_atlas_state() const { return global_atlas_state; }
	uint64_t get_atlas_generation() const { return global_atlas_state.atlas_generation; }

private:
	GlobalAtlasState global_atlas_state;
	uint32_t max_chunk_count_per_asset = 0;
	uint32_t max_chunk_splats = 0;
	RID asset_meta_buffer;
	RID chunk_meta_buffer;
	RID asset_chunk_index_buffer;
	uint32_t asset_meta_buffer_size = 0;
	uint32_t chunk_meta_buffer_size = 0;
	uint32_t asset_chunk_index_buffer_size = 0;
	LocalVector<AssetMetaGPU> asset_meta_cpu;
	LocalVector<ChunkMetaGPU> chunk_meta_cpu;
	LocalVector<AssetChunkIndexGPU> asset_chunk_index_cpu;
	LocalVector<uint32_t> chunk_meta_dirty_indices;
	LocalVector<uint8_t> chunk_meta_dirty_flags;
	bool asset_meta_dirty = false;
	bool asset_chunk_index_dirty = false;
	bool chunk_meta_dirty_all = false;
	bool asset_registry_dirty = false;
};

#endif // STREAMING_GLOBAL_ATLAS_REGISTRY_H
