#ifndef STREAMING_ASSET_TYPES_H
#define STREAMING_ASSET_TYPES_H

#include "gaussian_data.h"
#include "core/math/aabb.h"
#include "core/math/vector3.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"
#include "streaming_quantization.h"
#include <cstdint>

namespace GaussianStreamingTypes {

struct ChunkLayoutHint {
    uint32_t start_idx = 0;
    uint32_t count = 0;
    uint32_t source_index_offset = 0;
    bool source_indices_remapped = false;
    AABB bounds;
    Vector3 center;
    float radius = 0.0f;
};

struct StreamingChunk {
    uint32_t start_idx = 0;
    uint32_t count = 0;
    bool source_index_remapped = false;
    RID gpu_buffer;
    Vector3 center;
    AABB bounds;
    float max_radius = 0.0f;
    float distance = 0.0f;
    bool is_loaded = false;
    bool is_visible = true;
    bool upload_pending = false;
    RenderingDevice *upload_device = nullptr;
    uint64_t last_used_frame = 0;
    uint64_t last_loaded_frame = 0;
    uint32_t buffer_slot = UINT32_MAX;

    float lod_blend_factor = 1.0f;
    float previous_distance = 0.0f;
    uint32_t current_lod_level = 0;
    uint32_t target_lod_level = 0;

    int sh_band_level = 3;
    int splat_skip_factor = 1;
    float opacity_multiplier = 1.0f;
    uint32_t effective_count = 0;

    ChunkQuantizationInfo quantization;
    bool quantization_computed = false;
};

struct RequestedChunkState {
    uint64_t stamp = 0;
    uint32_t lod_mask = 0;
};

struct AtlasAssetState {
    uint32_t asset_id = 0;
    uint32_t dense_id = 0;
    Ref<GaussianData> data;
    bool uses_primary_chunks = false;
    LocalVector<StreamingChunk> asset_chunks;
    LocalVector<uint32_t> requested_chunks;
    HashMap<uint32_t, RequestedChunkState> requested_chunk_state;
    uint32_t lod_count = 1;
    uint32_t sh_degree = 0;
    AABB bounds;
    uint32_t chunk_meta_base = 0;
    uint32_t chunk_meta_count = 0;
    uint32_t chunk_index_base = 0;
    uint32_t chunk_index_count = 0;
    uint32_t quant_base = 0;
    uint32_t quant_count = 0;
    bool metadata_dirty = true;
    uint32_t generation = 0;
};

struct AssetRegistryState {
    HashMap<uint32_t, AtlasAssetState> atlas_assets;
    LocalVector<uint32_t> atlas_asset_order;
    HashMap<uint32_t, uint32_t> asset_id_to_dense;
    LocalVector<uint32_t> dense_to_asset_id;
    LocalVector<uint32_t> dense_id_generation;
    LocalVector<uint32_t> free_dense_ids;
    HashMap<uint32_t, uint32_t> asset_generation_tracker;
    uint64_t request_generation = 1;
    bool request_collection_active = false;
    bool request_pending = false;
    Vector<ChunkLayoutHint> io_chunk_layout_hints;
    uint32_t io_chunk_layout_asset_id = UINT32_MAX;
    Vector<ChunkLayoutHint> primary_chunk_layout_hints;
    LocalVector<uint32_t> primary_chunk_layout_source_indices;
    LocalVector<uint32_t> primary_chunk_source_indices;
};

} // namespace GaussianStreamingTypes

#endif // STREAMING_ASSET_TYPES_H
