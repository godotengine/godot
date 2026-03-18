#ifndef GS_INSTANCE_LAYOUT_GLSL
#define GS_INSTANCE_LAYOUT_GLSL

// Instance pipeline layout (always active).

#ifndef GS_MAX_ASSET_LODS
#define GS_MAX_ASSET_LODS 8
#endif

// Per-instance data (std430, 16-byte aligned).
struct InstanceDataGPU {
    vec4 rotation;           // quat (x,y,z,w)
    vec4 inv_rotation;       // quat inverse
    vec4 translation_scale;  // xyz = translation, w = uniform_scale
    vec4 params;             // x = opacity, y = lod_bias, z = wind_intensity, w = wind_mode
    uvec2 ids;               // x = asset_id, y = flags
    uvec2 lod;               // x = resolved_lod_level, y = reserved
    vec4 wind_params;        // xyz = wind direction override (0,0,0=infer global), w = wind frequency scale
};

// Instance flag bits (ids.y).
const uint GS_INSTANCE_FLAG_IS_2D = 1u << 0u;
const uint GS_INSTANCE_FLAG_ROTATION_IDENTITY = 1u << 1u;
const uint GS_INSTANCE_FLAG_SCALE_IDENTITY = 1u << 2u;
const uint GS_INSTANCE_FLAG_TRANSLATION_ZERO = 1u << 3u;

// Instance wind override modes (params.w).
#ifndef GS_INSTANCE_WIND_MODE_INHERIT
#define GS_INSTANCE_WIND_MODE_INHERIT 0u
#define GS_INSTANCE_WIND_MODE_FORCE_DISABLED 1u
#define GS_INSTANCE_WIND_MODE_FORCE_ENABLED 2u
#endif

// Per-asset metadata table.
struct AssetLodRangeGPU {
    uint base;               // base into AssetChunkIndexBuffer
    uint count;              // number of chunk indices for this LOD
};

struct AssetMetaGPU {
    uint lod_count;
    uint sh_degree;          // max SH bands for asset
    uint flags;              // include is_2d bit
    uint pad0;

    vec3 bounds_center_local;
    float bounds_radius_local;

    uint chunk_index_base;   // base into AssetChunkIndexBuffer (all LODs)
    uint chunk_index_count;  // total indices across all LODs
    uint quant_chunk_base;   // base into QuantizationChunkBuffer (asset-wide range)
    uint quant_chunk_count;  // total quant chunks for asset (informational)

    AssetLodRangeGPU lod_ranges[GS_MAX_ASSET_LODS];
};

// Global chunk metadata table (asset-local bounds).
struct ChunkMetaGPU {
    uint atlas_base;         // start index into global atlas buffer
    uint splat_count;
    uint quant_base;         // absolute base into QuantizationChunkBuffer (authoritative)
    uint quant_count;

    vec3 bounds_center_local;
    float bounds_radius_local;

    uint asset_id;
    uint lod_level;
    uint flags;
    uint sh_limit;           // per-chunk SH band limit (optional)
    uint pad0;
    uint pad1;
    uint pad2;
};

// Global chunk index indirection (dense).
struct AssetChunkIndexGPU {
    uint chunk_id;           // index into ChunkMetaGPU[]
};

// Visible chunk list emitted by Stage A.
struct VisibleChunkRefGPU {
    uint instance_id;
    uint chunk_id;
};

// Visible splat list emitted by Stage B.
struct SplatRefGPU {
    uint instance_id;
    uint atlas_index;
};

const uint GS_INSTANCE_DATA_GPU_SIZE = 96u;
const uint GS_ASSET_LOD_RANGE_GPU_SIZE = 8u;
const uint GS_ASSET_META_GPU_SIZE = 48u + 8u * GS_MAX_ASSET_LODS;
const uint GS_CHUNK_META_GPU_SIZE = 64u;
const uint GS_ASSET_CHUNK_INDEX_GPU_SIZE = 4u;
const uint GS_VISIBLE_CHUNK_REF_GPU_SIZE = 8u;
const uint GS_SPLAT_REF_GPU_SIZE = 8u;

#endif // GS_INSTANCE_LAYOUT_GLSL
