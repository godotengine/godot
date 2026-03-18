#ifndef GAUSSIAN_GPU_LAYOUT_H
#define GAUSSIAN_GPU_LAYOUT_H

#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "../core/gaussian_data.h"
#include <cstddef>
#include <cstdint>

static constexpr uint32_t GS_RENDER_PARAMS_LAYOUT_VERSION = 16; // Keep in sync with shaders/includes/gs_render_params.glsl
static constexpr uint32_t GS_MAX_ASSET_LODS = 8;

// Instance pipeline only: uses SplatRef indirection and asset-local quantization.

struct SHCompressionMetrics {
    uint64_t raw_bytes = 0;
    uint64_t compressed_bytes = 0;
    uint32_t coefficient_count = 0;
};

// ============================================================================
// Instance Pipeline GPU Layout (std430)
// ============================================================================

// Per-instance data (uniform scale enforced).
struct alignas(16) InstanceDataGPU {
    float rotation[4];           // quat (x,y,z,w)
    float inv_rotation[4];       // quat inverse
    float translation_scale[4];  // xyz = translation, w = uniform_scale
    float params[4];             // x = opacity, y = lod_bias, z = wind_intensity, w = wind_mode
    uint32_t ids[2];             // x = asset_id, y = flags
    uint32_t lod[2];             // x = resolved_lod_level, y = reserved
    float wind_params[4];        // xyz = wind direction override (0,0,0=infer global), w = wind frequency scale
};

// Instance flag bits (ids[1]).
static constexpr uint32_t GS_INSTANCE_FLAG_IS_2D = 1u << 0;
static constexpr uint32_t GS_INSTANCE_FLAG_ROTATION_IDENTITY = 1u << 1;
static constexpr uint32_t GS_INSTANCE_FLAG_SCALE_IDENTITY = 1u << 2;
static constexpr uint32_t GS_INSTANCE_FLAG_TRANSLATION_ZERO = 1u << 3;

// Instance wind override modes (InstanceDataGPU.params.w).
static constexpr uint32_t GS_INSTANCE_WIND_MODE_INHERIT = 0u;
static constexpr uint32_t GS_INSTANCE_WIND_MODE_FORCE_DISABLED = 1u;
static constexpr uint32_t GS_INSTANCE_WIND_MODE_FORCE_ENABLED = 2u;

static_assert(alignof(InstanceDataGPU) == 16, "InstanceDataGPU must be 16-byte aligned");
static_assert(sizeof(InstanceDataGPU) == 96, "InstanceDataGPU must be 96 bytes");
static_assert(offsetof(InstanceDataGPU, rotation) == 0, "InstanceDataGPU.rotation offset mismatch");
static_assert(offsetof(InstanceDataGPU, inv_rotation) == 16, "InstanceDataGPU.inv_rotation offset mismatch");
static_assert(offsetof(InstanceDataGPU, translation_scale) == 32, "InstanceDataGPU.translation_scale offset mismatch");
static_assert(offsetof(InstanceDataGPU, params) == 48, "InstanceDataGPU.params offset mismatch");
static_assert(offsetof(InstanceDataGPU, ids) == 64, "InstanceDataGPU.ids offset mismatch");
static_assert(offsetof(InstanceDataGPU, lod) == 72, "InstanceDataGPU.lod offset mismatch");
static_assert(offsetof(InstanceDataGPU, wind_params) == 80, "InstanceDataGPU.wind_params offset mismatch");

// Per-asset metadata table.
struct AssetLodRangeGPU {
    uint32_t base;               // base into AssetChunkIndexBuffer
    uint32_t count;              // number of chunk indices for this LOD
};

static_assert(sizeof(AssetLodRangeGPU) == 8, "AssetLodRangeGPU must be 8 bytes");
static_assert(offsetof(AssetLodRangeGPU, base) == 0, "AssetLodRangeGPU.base offset mismatch");
static_assert(offsetof(AssetLodRangeGPU, count) == 4, "AssetLodRangeGPU.count offset mismatch");

struct alignas(16) AssetMetaGPU {
    uint32_t lod_count;
    uint32_t sh_degree;          // max SH bands for asset
    uint32_t flags;              // include is_2d bit
    uint32_t pad0;

    float bounds_center_local[3];
    float bounds_radius_local;

    uint32_t chunk_index_base;   // base into AssetChunkIndexBuffer (all LODs)
    uint32_t chunk_index_count;  // total indices across all LODs
    uint32_t quant_chunk_base;   // base into QuantizationChunkBuffer (asset-wide range)
    uint32_t quant_chunk_count;  // total quant chunks for asset (informational)

    AssetLodRangeGPU lod_ranges[GS_MAX_ASSET_LODS];
};

static_assert(alignof(AssetMetaGPU) == 16, "AssetMetaGPU must be 16-byte aligned");
static_assert(sizeof(AssetMetaGPU) == 48 + sizeof(AssetLodRangeGPU) * GS_MAX_ASSET_LODS,
        "AssetMetaGPU size mismatch");
static_assert(offsetof(AssetMetaGPU, lod_count) == 0, "AssetMetaGPU.lod_count offset mismatch");
static_assert(offsetof(AssetMetaGPU, bounds_center_local) == 16, "AssetMetaGPU.bounds_center_local offset mismatch");
static_assert(offsetof(AssetMetaGPU, bounds_radius_local) == 28, "AssetMetaGPU.bounds_radius_local offset mismatch");
static_assert(offsetof(AssetMetaGPU, chunk_index_base) == 32, "AssetMetaGPU.chunk_index_base offset mismatch");
static_assert(offsetof(AssetMetaGPU, lod_ranges) == 48, "AssetMetaGPU.lod_ranges offset mismatch");

// Global chunk metadata table (asset-local bounds).
struct alignas(16) ChunkMetaGPU {
    uint32_t atlas_base;         // start index into global atlas buffer
    uint32_t splat_count;
    uint32_t quant_base;         // absolute base into QuantizationChunkBuffer (authoritative)
    uint32_t quant_count;

    float bounds_center_local[3];
    float bounds_radius_local;

    uint32_t asset_id;
    uint32_t lod_level;
    uint32_t flags;
    uint32_t sh_limit;           // per-chunk SH band limit (optional)
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

static_assert(alignof(ChunkMetaGPU) == 16, "ChunkMetaGPU must be 16-byte aligned");
static_assert(sizeof(ChunkMetaGPU) == 64, "ChunkMetaGPU must be 64 bytes");
static_assert(offsetof(ChunkMetaGPU, atlas_base) == 0, "ChunkMetaGPU.atlas_base offset mismatch");
static_assert(offsetof(ChunkMetaGPU, bounds_center_local) == 16, "ChunkMetaGPU.bounds_center_local offset mismatch");
static_assert(offsetof(ChunkMetaGPU, bounds_radius_local) == 28, "ChunkMetaGPU.bounds_radius_local offset mismatch");
static_assert(offsetof(ChunkMetaGPU, asset_id) == 32, "ChunkMetaGPU.asset_id offset mismatch");
static_assert(offsetof(ChunkMetaGPU, pad2) == 56, "ChunkMetaGPU.pad2 offset mismatch");

// Global chunk index indirection (dense).
struct AssetChunkIndexGPU {
    uint32_t chunk_id;           // index into ChunkMetaGPU[]
};

static_assert(sizeof(AssetChunkIndexGPU) == 4, "AssetChunkIndexGPU must be 4 bytes");
static_assert(offsetof(AssetChunkIndexGPU, chunk_id) == 0, "AssetChunkIndexGPU.chunk_id offset mismatch");

// Visible chunk list emitted by Stage A.
struct VisibleChunkRefGPU {
    uint32_t instance_id;
    uint32_t chunk_id;
};

static_assert(sizeof(VisibleChunkRefGPU) == 8, "VisibleChunkRefGPU must be 8 bytes");
static_assert(offsetof(VisibleChunkRefGPU, instance_id) == 0, "VisibleChunkRefGPU.instance_id offset mismatch");
static_assert(offsetof(VisibleChunkRefGPU, chunk_id) == 4, "VisibleChunkRefGPU.chunk_id offset mismatch");

// Visible splat list emitted by Stage B.
struct SplatRefGPU {
    uint32_t instance_id;
    uint32_t atlas_index;
};

static_assert(sizeof(SplatRefGPU) == 8, "SplatRefGPU must be 8 bytes");
static_assert(offsetof(SplatRefGPU, instance_id) == 0, "SplatRefGPU.instance_id offset mismatch");
static_assert(offsetof(SplatRefGPU, atlas_index) == 4, "SplatRefGPU.atlas_index offset mismatch");

// ============================================================================
// Float16 Data Structures (for memory-optimized storage)
// ============================================================================

/**
 * @struct PackedSphericalHarmonicsF16
 * @brief Float16 version of SH storage for reduced memory footprint.
 *
 * DC coefficients remain Float32 for base color quality.
 * Higher-order coefficients use Float16 (already low dynamic range).
 */
struct alignas(16) PackedSphericalHarmonicsF16 {
    static constexpr uint32_t MAX_ENCODED_COEFFICIENTS = 12;

    float dc[4];                    // 16 bytes - DC coefficients (FP32 for quality)
    uint16_t encoded[MAX_ENCODED_COEFFICIENTS]; // 24 bytes - Higher-order coefficients (FP16)

    void clear();
};

static_assert(sizeof(PackedSphericalHarmonicsF16) == 48, "PackedSphericalHarmonicsF16 must be 48 bytes");

/**
 * @struct PackedGaussianF16
 * @brief Float16 version of PackedGaussian for reduced GPU memory usage.
 *
 * Provides approximately 2x compression vs PackedGaussian with minimal
 * quality impact for most fields. Scale and opacity remain FP32.
 *
 * Memory layout (96 bytes, 16-byte aligned):
 *   position_xy: 2 x half (4 bytes)
 *   position_z: 1 x half + padding (4 bytes)
 *   opacity: float32 (4 bytes)
 *   scale[3]: float32 (12 bytes) - precision-sensitive
 *   area: float32 (4 bytes)
 *   rotation_xy: 2 x half (4 bytes)
 *   rotation_zw: 2 x half (4 bytes)
 *   sh: PackedSphericalHarmonicsF16 (48 bytes)
 *   normal[3]: float32 (12 bytes)
 *   stroke_age: float32 (4 bytes)
 *   brush_axes[2]: float32 (8 bytes)
 *   painterly_meta: uint32 (4 bytes)
 *   sh_metadata: uint32 (4 bytes)
 *   _padding: 4 bytes (to reach 128 for alignment)
 */
struct alignas(16) PackedGaussianF16 {
    // Position as Float16 relative offsets (requires per-chunk center)
    uint32_t position_xy;           // packHalf2x16(x, y) - 4 bytes @0
    uint32_t position_z_pad;        // lower 16 bits: half(z), upper: chunk_id - 4 bytes @4

    float opacity;                  // Keep FP32 - 4 bytes @8

    float scale[3];                 // Keep FP32 (precision-sensitive) - 12 bytes @12
    float area;                     // 4 bytes @24

    uint32_t rotation_xy;           // packHalf2x16(qx, qy) - 4 bytes @28
    uint32_t rotation_zw;           // packHalf2x16(qz, qw) - 4 bytes @32
    uint32_t _pre_sh_padding[3];    // 12 bytes @36 - Explicit padding for 16-byte sh alignment

    PackedSphericalHarmonicsF16 sh; // 48 bytes @48

    float normal[3];                // 12 bytes @96
    float stroke_age;               // 4 bytes @108

    float brush_axes[2];            // 8 bytes @112
    uint32_t painterly_meta;        // 4 bytes @120
    uint32_t sh_metadata;           // 4 bytes @124
    uint32_t _padding[4];           // 16 bytes @128 - Align to 144 bytes total
};

static_assert(sizeof(PackedGaussianF16) == 144, "PackedGaussianF16 must be 144 bytes");
static_assert(sizeof(PackedGaussianF16) % 16 == 0, "PackedGaussianF16 must be 16-byte aligned");

/**
 * @struct QuantizationChunkGPU
 * @brief Per-chunk quantization data for GPU access.
 *
 * Each chunk stores a center point that positions are relative to,
 * improving Float16 precision for position data.
 */
struct alignas(16) QuantizationChunkGPU {
    float center[3];                // Asset-local center (instance transform applied later)
    uint32_t start_index;           // First Gaussian index
    float max_extent;               // Maximum distance from center
    uint32_t count;                 // Number of Gaussians
    uint32_t _padding[2];           // Align to 32 bytes
};

static_assert(sizeof(QuantizationChunkGPU) == 32, "QuantizationChunkGPU must be 32 bytes");

struct alignas(16) PackedSphericalHarmonics {
    static constexpr uint32_t MAX_ENCODED_COEFFICIENTS = 12;

    float dc[4];
    float encoded[MAX_ENCODED_COEFFICIENTS];

    void clear();
};

struct alignas(16) PackedGaussian {
    float position[3];
    float opacity;

    float scale[3];
    float area;

    float rotation[4];

    PackedSphericalHarmonics sh;

    float normal[3];
    float stroke_age;

    float brush_axes[2];
    uint32_t painterly_meta;
    uint32_t sh_metadata;
};

static_assert(sizeof(PackedGaussian) == 144, "PackedGaussian must match shader layout (144 bytes)");

/**
 * @struct PackedGaussianQuantized
 * @brief Per-chunk quantized Gaussian for maximum compression (Unity technique).
 *
 * Uses per-chunk min/max bounds to normalize positions and scales,
 * achieving up to 4x compression ratio with minimal quality loss.
 *
 * Memory layout (80 bytes, 16-byte aligned):
 *   quantized_position: uint16_t[3] - Normalized position (6 bytes)
 *   chunk_id: uint16_t - Index into chunk bounds buffer (2 bytes)
 *   opacity: float (4 bytes)
 *   quantized_scale: uint16_t[3] - Normalized scale (6 bytes, or zeros if not quantized)
 *   area_lo: uint16_t - Low 16 bits of area as float16 (2 bytes)
 *   rotation: uint16_t[4] - Quaternion as float16 (8 bytes)
 *   sh_dc: float[4] - DC coefficients (16 bytes)
 *   sh_encoded: uint32_t[6] - Higher-order SH as RGB9E5 pairs (24 bytes)
 *   normal_xy: uint32_t - Normal xy as half2 (4 bytes)
 *   normal_z_stroke: uint32_t - Normal z + stroke_age as half2 (4 bytes)
 *   painterly_data: uint32_t - Packed painterly meta (4 bytes)
 *   sh_metadata: uint32_t - SH encoding metadata (4 bytes)
 *
 * Total: 80 bytes (vs 144 bytes for PackedGaussian = 44% reduction)
 */
struct alignas(16) PackedGaussianQuantized {
    uint16_t quantized_position[3]; // 6 bytes @0 - Normalized position per-chunk
    uint16_t chunk_id;              // 2 bytes @6 - Index into ChunkQuantizationGPU buffer

    float opacity;                  // 4 bytes @8

    uint16_t quantized_scale[3];    // 6 bytes @12 - Normalized scale (0 if not quantized)
    uint16_t area_fp16;             // 2 bytes @18 - Area as float16

    uint16_t rotation[4];           // 8 bytes @20 - Quaternion as float16

    uint16_t _pre_sh_padding[2];    // 4 bytes @28 - Align to 32

    float sh_dc[4];                 // 16 bytes @32 - DC coefficients (FP32)
    uint32_t sh_encoded[6];         // 24 bytes @48 - RGB9E5 encoded higher-order

    uint32_t normal_xy;             // 4 bytes @72 - packHalf2x16(nx, ny)
    uint32_t normal_z_stroke;       // 4 bytes @76 - packHalf2x16(nz, stroke_age)
};

static_assert(sizeof(PackedGaussianQuantized) == 80, "PackedGaussianQuantized must be 80 bytes");
static_assert(sizeof(PackedGaussianQuantized) % 16 == 0, "PackedGaussianQuantized must be 16-byte aligned");

struct alignas(16) TileRenderParamsGPU {
    float view_matrix[16];
    float inv_view_matrix[16];
    float projection_matrix[16];
    float inv_projection_matrix[16];
    float viewport_size[2];
    float tile_count[2];
    uint32_t total_gaussians;
    uint32_t visible_gaussians;
    float near_plane;
    float far_plane;
    float debug_flags[4];
    float debug_overlay_opacity;
    float opacity_multiplier;
    float _pad_before_camera[2]; // std140: align camera_position to 16 bytes
    float camera_position[4]; // xyz used, w reserved
    float alpha_floor;
    uint32_t force_solid_coverage;
    uint32_t overlap_record_count;
    uint32_t _pad_overlap;
    float cull_far_tolerance;
    float tiny_splat_screen_radius;
    float max_conic_aspect;
    float _pad_before_jacobian; // std140: align jacobian_diag_flags (vec4) to 16 bytes
    float jacobian_diag_flags[4];
    float debug_overlay_flags[4];
    // Spherical Harmonics configuration:
    // x=sh_bands (0-3), y=amortization_divisor, z=amortization_phase, w=force_full_update
    // sh_bands: 0=DC only, 1=1st order, 2=2nd order, 3=3rd order (full)
    float sh_config[4];
    // SH decode configuration:
    // x=dc_logit (1.0=decode DC with sigmoid), yzw=reserved
    float sh_decode_config[4];
    // Opacity-aware culling (FlashGS): x=enabled (bool), y=visibility_threshold (tau), z=reserved, w=reserved
    // When enabled, splat radii are calculated as: r = sqrt(2 * ln(alpha/tau) * lambda_max)
    // This reduces tile-Gaussian pairs by ~94% for low-opacity splats
    float opacity_culling_config[4];
    // LOD blending configuration (LODGE technique):
    // x=lod_blend_factor (0-1, global blend factor for the frame)
    // y=lod_blend_enabled (0 or 1)
    // z=lod_blend_distance (world units)
    // w=reserved
    float lod_blend_config[4];
    // Distance-based culling configuration:
    // x=start_distance, y=max_cull_rate, z=enabled, w=reserved
    float distance_cull_config[4];
    // Color grading parameters (aligned to vec4):
    // color_grading_primary: x=enabled (0/1), y=exposure, z=contrast, w=saturation
    // color_grading_secondary: x=temperature, y=tint, z=hue_shift, w=reserved
    float color_grading_primary[4];
    float color_grading_secondary[4];
    // Lighting configuration:
    // lighting_config: x=direct_light_scale, y=indirect_sh_scale, z=enable_direct_lighting,
    // w=normal_mode (0=depth gradients, 1=view dir, 2=depth gradients with fallback)
    float lighting_config[4];
    // Shadow configuration:
    // shadow_strength: x=shadow_strength (0..1), yzw=reserved
    float shadow_strength[4];
    // Shadow receiver bias configuration:
    // shadow_bias_config: x=receiver_bias_scale (per-splat radius multiplier),
    // y=receiver_bias_min, z=receiver_bias_max (0=disabled), w=reserved
    float shadow_bias_config[4];
    // Lighting mode:
    // lighting_mode: x=direct_lighting_mode (0=resolve, 1=per-splat, 2=both), yzw=reserved
    uint32_t lighting_mode[4];
    // Light counts:
    // light_counts: x=omni_light_count, y=spot_light_count, z=cluster_enabled, w=light_mask
    uint32_t light_counts[4];
    // Cluster configuration:
    // cluster_config: x=cluster_shift, y=cluster_width, z=cluster_type_size, w=max_cluster_element_count_div_32
    uint32_t cluster_config[4];
    // Instance rotation inverse for SH view direction correction (mat3 stored as 3 vec4s for std140).
    // When a GaussianSplatNode3D has a rotation transform, we need to transform view directions
    // back to the original coordinate frame (where SH coefficients were computed).
    // instance_rotation_inv_col0.xyz = first column of rotation inverse matrix
    // instance_rotation_inv_col1.xyz = second column
    // instance_rotation_inv_col2.xyz = third column
    // w components reserved for future use
    float instance_rotation_inv_col0[4];
    float instance_rotation_inv_col1[4];
    float instance_rotation_inv_col2[4];
    float wind_dir_strength[4];
    float wind_time_config[4];
    // Single global sphere effector (foundation for capped multi-effector support):
    // effector_sphere: xyz=center (world), w=radius
    // effector_config: x=enabled (0/1), y=strength (meters), z=falloff exponent, w=reserved
    float effector_sphere[4];
    float effector_config[4];
};

static_assert(sizeof(TileRenderParamsGPU) == 720, "TileRenderParamsGPU must match RenderParams std140 layout (720 bytes)");
static_assert(alignof(TileRenderParamsGPU) == 16, "TileRenderParamsGPU must be 16-byte aligned");
static_assert(offsetof(TileRenderParamsGPU, view_matrix) == 0, "TileRenderParamsGPU.view_matrix offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, inv_view_matrix) == 64, "TileRenderParamsGPU.inv_view_matrix offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, projection_matrix) == 128, "TileRenderParamsGPU.projection_matrix offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, inv_projection_matrix) == 192, "TileRenderParamsGPU.inv_projection_matrix offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, viewport_size) == 256, "TileRenderParamsGPU.viewport_size offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, tile_count) == 264, "TileRenderParamsGPU.tile_count offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, total_gaussians) == 272, "TileRenderParamsGPU.total_gaussians offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, visible_gaussians) == 276, "TileRenderParamsGPU.visible_gaussians offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, near_plane) == 280, "TileRenderParamsGPU.near_plane offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, far_plane) == 284, "TileRenderParamsGPU.far_plane offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, debug_flags) == 288, "TileRenderParamsGPU.debug_flags offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, debug_overlay_opacity) == 304, "TileRenderParamsGPU.debug_overlay_opacity offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, opacity_multiplier) == 308, "TileRenderParamsGPU.opacity_multiplier offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, _pad_before_camera) == 312, "TileRenderParamsGPU._pad_before_camera offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, camera_position) == 320, "TileRenderParamsGPU.camera_position offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, alpha_floor) == 336, "TileRenderParamsGPU.alpha_floor offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, force_solid_coverage) == 340, "TileRenderParamsGPU.force_solid_coverage offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, overlap_record_count) == 344, "TileRenderParamsGPU.overlap_record_count offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, _pad_overlap) == 348, "TileRenderParamsGPU._pad_overlap offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, cull_far_tolerance) == 352, "TileRenderParamsGPU.cull_far_tolerance offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, tiny_splat_screen_radius) == 356, "TileRenderParamsGPU.tiny_splat_screen_radius offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, max_conic_aspect) == 360, "TileRenderParamsGPU.max_conic_aspect offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, _pad_before_jacobian) == 364, "TileRenderParamsGPU._pad_before_jacobian offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, jacobian_diag_flags) == 368, "TileRenderParamsGPU.jacobian_diag_flags offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, debug_overlay_flags) == 384, "TileRenderParamsGPU.debug_overlay_flags offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, sh_config) == 400, "TileRenderParamsGPU.sh_config offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, sh_decode_config) == 416, "TileRenderParamsGPU.sh_decode_config offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, opacity_culling_config) == 432, "TileRenderParamsGPU.opacity_culling_config offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, lod_blend_config) == 448, "TileRenderParamsGPU.lod_blend_config offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, distance_cull_config) == 464, "TileRenderParamsGPU.distance_cull_config offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, color_grading_primary) == 480, "TileRenderParamsGPU.color_grading_primary offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, color_grading_secondary) == 496, "TileRenderParamsGPU.color_grading_secondary offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, lighting_config) == 512, "TileRenderParamsGPU.lighting_config offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, shadow_strength) == 528, "TileRenderParamsGPU.shadow_strength offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, shadow_bias_config) == 544, "TileRenderParamsGPU.shadow_bias_config offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, lighting_mode) == 560, "TileRenderParamsGPU.lighting_mode offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, light_counts) == 576, "TileRenderParamsGPU.light_counts offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, cluster_config) == 592, "TileRenderParamsGPU.cluster_config offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, instance_rotation_inv_col0) == 608, "TileRenderParamsGPU.instance_rotation_inv_col0 offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, instance_rotation_inv_col1) == 624, "TileRenderParamsGPU.instance_rotation_inv_col1 offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, instance_rotation_inv_col2) == 640, "TileRenderParamsGPU.instance_rotation_inv_col2 offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, wind_dir_strength) == 656, "TileRenderParamsGPU.wind_dir_strength offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, wind_time_config) == 672, "TileRenderParamsGPU.wind_time_config offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, effector_sphere) == 688, "TileRenderParamsGPU.effector_sphere offset mismatch");
static_assert(offsetof(TileRenderParamsGPU, effector_config) == 704, "TileRenderParamsGPU.effector_config offset mismatch");

struct alignas(16) FrustumCullParamsGPU {
    static constexpr uint32_t kFrustumPlaneCount = 6;

    float view_matrix[16];
    float proj_matrix[16];
    float frustum_planes[kFrustumPlaneCount][4];
    float camera_position[3];
    float pixel_scale_y;
    float near_tolerance;
    float min_screen_size;
    float max_distance_sq;
    float importance_threshold;
    float radius_multiplier;
    float far_tolerance;
    float tiny_splat_screen_radius;
    float frustum_plane_slack;
    uint32_t total_splats;
    uint32_t max_visible;
    uint32_t enable_frustum;
    uint32_t orthographic;
    uint32_t use_consolidated_readback; // 1 = write to consolidated buffer for single readback
};

static_assert(alignof(FrustumCullParamsGPU) == 16, "FrustumCullParamsGPU must be 16-byte aligned");
static_assert(sizeof(FrustumCullParamsGPU) == 304, "FrustumCullParamsGPU must match std140 layout (304 bytes)");
static_assert(sizeof(FrustumCullParamsGPU) % 16 == 0, "FrustumCullParamsGPU must align to 16 bytes for std140");
static_assert(offsetof(FrustumCullParamsGPU, view_matrix) == 0, "FrustumCullParamsGPU.view_matrix offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, proj_matrix) == 64, "FrustumCullParamsGPU.proj_matrix offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, frustum_planes) == 128, "FrustumCullParamsGPU.frustum_planes offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, camera_position) == 224, "FrustumCullParamsGPU.camera_position offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, pixel_scale_y) == 236, "FrustumCullParamsGPU.pixel_scale_y offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, near_tolerance) == 240, "FrustumCullParamsGPU.near_tolerance offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, min_screen_size) == 244, "FrustumCullParamsGPU.min_screen_size offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, max_distance_sq) == 248, "FrustumCullParamsGPU.max_distance_sq offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, importance_threshold) == 252, "FrustumCullParamsGPU.importance_threshold offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, radius_multiplier) == 256, "FrustumCullParamsGPU.radius_multiplier offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, far_tolerance) == 260, "FrustumCullParamsGPU.far_tolerance offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, tiny_splat_screen_radius) == 264, "FrustumCullParamsGPU.tiny_splat_screen_radius offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, frustum_plane_slack) == 268, "FrustumCullParamsGPU.frustum_plane_slack offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, total_splats) == 272, "FrustumCullParamsGPU.total_splats offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, max_visible) == 276, "FrustumCullParamsGPU.max_visible offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, enable_frustum) == 280, "FrustumCullParamsGPU.enable_frustum offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, orthographic) == 284, "FrustumCullParamsGPU.orthographic offset mismatch");
static_assert(offsetof(FrustumCullParamsGPU, use_consolidated_readback) == 288, "FrustumCullParamsGPU.use_consolidated_readback offset mismatch");

struct alignas(16) InstanceCullParamsGPU {
    static constexpr uint32_t kFrustumPlaneCount = 6;

    float view_matrix[16];
    float proj_matrix[16];
    float frustum_planes[kFrustumPlaneCount][4];
    float frustum_plane_slack;
    uint32_t instance_count;
    uint32_t max_visible_chunks;
    uint32_t enable_frustum;
};

static_assert(alignof(InstanceCullParamsGPU) == 16, "InstanceCullParamsGPU must be 16-byte aligned");
static_assert(sizeof(InstanceCullParamsGPU) == 240, "InstanceCullParamsGPU must match std140 layout (240 bytes)");
static_assert(sizeof(InstanceCullParamsGPU) % 16 == 0, "InstanceCullParamsGPU must align to 16 bytes for std140");
static_assert(offsetof(InstanceCullParamsGPU, view_matrix) == 0, "InstanceCullParamsGPU.view_matrix offset mismatch");
static_assert(offsetof(InstanceCullParamsGPU, proj_matrix) == 64, "InstanceCullParamsGPU.proj_matrix offset mismatch");
static_assert(offsetof(InstanceCullParamsGPU, frustum_planes) == 128, "InstanceCullParamsGPU.frustum_planes offset mismatch");
static_assert(offsetof(InstanceCullParamsGPU, frustum_plane_slack) == 224, "InstanceCullParamsGPU.frustum_plane_slack offset mismatch");
static_assert(offsetof(InstanceCullParamsGPU, instance_count) == 228, "InstanceCullParamsGPU.instance_count offset mismatch");
static_assert(offsetof(InstanceCullParamsGPU, max_visible_chunks) == 232, "InstanceCullParamsGPU.max_visible_chunks offset mismatch");
static_assert(offsetof(InstanceCullParamsGPU, enable_frustum) == 236, "InstanceCullParamsGPU.enable_frustum offset mismatch");

struct alignas(16) InstanceDepthParamsGPU {
    float view_matrix[16];
    uint32_t visible_chunk_count;
    uint32_t max_visible_splats;
    uint32_t pad0; // Used as dispatch_group_x for instance chunk indirect dispatch.
    uint32_t pad1;
    float wind_dir_strength[4];
    float wind_time_config[4];
    float effector_sphere[4];
    float effector_config[4];
    float frustum_planes[InstanceCullParamsGPU::kFrustumPlaneCount][4];
    float camera_position_ortho[4]; // xyz = camera position, w = orthographic flag (1.0 or 0.0)
    float cull_screen_distance[4]; // x = pixel_scale_y, y = tiny_splat_radius_px, z = min_screen_threshold_px, w = max_distance_sq
    float cull_frustum_radius[4]; // x = radius_multiplier, y = frustum_plane_slack, z = enable_frustum, w = reserved
};

static_assert(alignof(InstanceDepthParamsGPU) == 16, "InstanceDepthParamsGPU must be 16-byte aligned");
static_assert(sizeof(InstanceDepthParamsGPU) == 288, "InstanceDepthParamsGPU must match std140 layout (288 bytes)");
static_assert(sizeof(InstanceDepthParamsGPU) % 16 == 0, "InstanceDepthParamsGPU must align to 16 bytes for std140");
static_assert(offsetof(InstanceDepthParamsGPU, view_matrix) == 0, "InstanceDepthParamsGPU.view_matrix offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, visible_chunk_count) == 64, "InstanceDepthParamsGPU.visible_chunk_count offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, max_visible_splats) == 68, "InstanceDepthParamsGPU.max_visible_splats offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, pad0) == 72, "InstanceDepthParamsGPU.pad0 offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, pad1) == 76, "InstanceDepthParamsGPU.pad1 offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, wind_dir_strength) == 80, "InstanceDepthParamsGPU.wind_dir_strength offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, wind_time_config) == 96, "InstanceDepthParamsGPU.wind_time_config offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, effector_sphere) == 112, "InstanceDepthParamsGPU.effector_sphere offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, effector_config) == 128, "InstanceDepthParamsGPU.effector_config offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, frustum_planes) == 144, "InstanceDepthParamsGPU.frustum_planes offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, camera_position_ortho) == 240, "InstanceDepthParamsGPU.camera_position_ortho offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, cull_screen_distance) == 256, "InstanceDepthParamsGPU.cull_screen_distance offset mismatch");
static_assert(offsetof(InstanceDepthParamsGPU, cull_frustum_radius) == 272, "InstanceDepthParamsGPU.cull_frustum_radius offset mismatch");

void pack_gaussian(const Gaussian &src,
        PackedGaussian &dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs = nullptr,
        uint32_t first_order_count = 3,
        uint32_t higher_order_count = 0,
        uint32_t coefficient_limit = PackedSphericalHarmonics::MAX_ENCODED_COEFFICIENTS);

void pack_gaussians_range(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        Vector<PackedGaussian> &dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs = nullptr,
        uint32_t first_order_count = 3,
        uint32_t higher_order_count = 0,
        uint32_t coefficient_limit = PackedSphericalHarmonics::MAX_ENCODED_COEFFICIENTS);

void pack_gaussians_range_raw(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        PackedGaussian *dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs = nullptr,
        uint32_t first_order_count = 3,
        uint32_t higher_order_count = 0,
        uint32_t coefficient_limit = PackedSphericalHarmonics::MAX_ENCODED_COEFFICIENTS);

void pack_gaussians_range_raw(const Gaussian *src,
        uint32_t start,
        uint32_t count,
        PackedGaussian *dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs = nullptr,
        uint32_t first_order_count = 3,
        uint32_t higher_order_count = 0,
        uint32_t coefficient_limit = PackedSphericalHarmonics::MAX_ENCODED_COEFFICIENTS);

void pack_gaussians_range_limited(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        Vector<PackedGaussian> &dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        const uint8_t *coefficient_limits,
        uint32_t coefficient_limit = PackedSphericalHarmonics::MAX_ENCODED_COEFFICIENTS);

// ============================================================================
// Float16 Packing Functions
// ============================================================================

/**
 * @brief Packs a single Gaussian into Float16 format.
 * @param src Source Gaussian data.
 * @param dst Destination packed Float16 structure.
 * @param metrics SH compression metrics to update.
 * @param chunk_center Asset-local center for position quantization.
 * @param higher_order_coeffs Higher-order SH coefficients (optional).
 * @param first_order_count Number of first-order SH coefficients.
 * @param higher_order_count Number of higher-order SH coefficients.
 * @param coefficient_limit Maximum encoded coefficients.
 */
void pack_gaussian_f16(const Gaussian &src,
        PackedGaussianF16 &dst,
        SHCompressionMetrics &metrics,
        const Vector3 &chunk_center,
        const Vector3 *higher_order_coeffs = nullptr,
        uint32_t first_order_count = 3,
        uint32_t higher_order_count = 0,
        uint32_t coefficient_limit = PackedSphericalHarmonicsF16::MAX_ENCODED_COEFFICIENTS);

/**
 * @brief Packs a range of Gaussians into Float16 format.
 * @param src Source Gaussian array.
 * @param start Starting index.
 * @param count Number of Gaussians to pack.
 * @param dst Destination packed array.
 * @param metrics SH compression metrics to update.
 * @param chunk_center Asset-local center for position quantization.
 * @param higher_order_coeffs Higher-order SH coefficients (optional).
 * @param first_order_count Number of first-order SH coefficients.
 * @param higher_order_count Number of higher-order SH coefficients.
 * @param coefficient_limit Maximum encoded coefficients.
 */
void pack_gaussians_range_f16(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        Vector<PackedGaussianF16> &dst,
        SHCompressionMetrics &metrics,
        const Vector3 &chunk_center,
        const Vector3 *higher_order_coeffs = nullptr,
        uint32_t first_order_count = 3,
        uint32_t higher_order_count = 0,
        uint32_t coefficient_limit = PackedSphericalHarmonicsF16::MAX_ENCODED_COEFFICIENTS);

/**
 * @brief Packs Gaussians with per-chunk quantization for optimal FP16 precision.
 * @param src Source Gaussian array.
 * @param start Starting index.
 * @param count Number of Gaussians to pack.
 * @param chunk_size Number of Gaussians per quantization chunk.
 * @param dst Destination packed array.
 * @param chunks Output quantization chunk data for GPU.
 * @param metrics SH compression metrics to update.
 * @param higher_order_coeffs Higher-order SH coefficients (optional).
 * @param first_order_count Number of first-order SH coefficients.
 * @param higher_order_count Number of higher-order SH coefficients.
 */
void pack_gaussians_chunked_f16(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        uint32_t chunk_size,
        Vector<PackedGaussianF16> &dst,
        Vector<QuantizationChunkGPU> &chunks,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs = nullptr,
        uint32_t first_order_count = 3,
        uint32_t higher_order_count = 0);

/**
 * @brief Returns whether Float16 storage is currently enabled.
 */
bool is_float16_storage_enabled();

/**
 * @brief Returns the packed Gaussian size based on current configuration.
 */
uint32_t get_packed_gaussian_size();

#endif // GAUSSIAN_GPU_LAYOUT_H
