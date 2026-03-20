#ifndef QUANTIZATION_DEQUANT_GLSL
#define QUANTIZATION_DEQUANT_GLSL

// ============================================================================
// Per-Chunk Quantization/Dequantization for Gaussian Splatting (Unity Technique)
// ============================================================================
//
// This file provides GPU-side dequantization for per-chunk quantized Gaussian data.
// Based on Unity's Gaussian Splatting implementation, this achieves 4x compression
// for position data with minimal quality loss.
//
// Quantization formula (CPU-side):
//   quantized = (position - min) / (max - min) * ((1 << bits) - 1)
//
// Dequantization formula (GPU-side, implemented here):
//   position = min + quantized / ((1 << bits) - 1) * range
//            = min + quantized * inv_max * range
//
// Chunk bounds are asset-local in the instance pipeline.


// Per-chunk quantization bounds (matches ChunkQuantizationGPU in C++)
struct ChunkQuantization {
    vec3 position_min;      // Asset-local minimum position
    uint position_bits;     // Bit depth for positions (8-24)
    vec3 position_range;    // Range (max - min) for dequantization
    uint scale_bits;        // Bit depth for scales (8-16), 0 if not quantized
    vec3 scale_min;         // Scale minimum (if quantized, asset-local)
    uint start_index;       // First Gaussian index in this chunk
    vec3 scale_range;       // Scale range (if quantized)
    uint count;             // Number of Gaussians in chunk
};

// Quantized Gaussian data (matches PackedGaussianQuantized in C++)
// IMPORTANT: Use scalar/uint types to avoid std430 uvec2 alignment padding mismatch
// C++ layout: 80 bytes tightly packed, GLSL uvec2 would add implicit padding
struct GaussianQuantized {
    uvec2 position_chunk;   // 8 bytes @0: quantized_position[3] + chunk_id
    float opacity;          // 4 bytes @8
    // Use uint instead of uvec2 to avoid 8-byte alignment at offset 12
    uint scale_area_lo;     // 4 bytes @12: quantized_scale[0-1]
    uint scale_area_hi;     // 4 bytes @16: quantized_scale[2] + area_fp16
    uint rotation_lo;       // 4 bytes @20: rotation[0-1]
    uint rotation_hi;       // 4 bytes @24: rotation[2-3]
    uint _padding;          // 4 bytes @28: alignment padding (matches C++ _pre_sh_padding)
    vec4 sh_dc;             // 16 bytes @32: DC coefficients (FP32)
    uvec2 sh_encoded_01;    // 8 bytes @48: sh_encoded[0-1]
    uvec2 sh_encoded_23;    // 8 bytes @56: sh_encoded[2-3]
    uvec2 sh_encoded_45;    // 8 bytes @64: sh_encoded[4-5]
    uint normal_xy;         // 4 bytes @72: packHalf2x16(nx, ny)
    uint normal_z_stroke;   // 4 bytes @76: packHalf2x16(nz, stroke_age)
    // Total: 80 bytes (matches C++)
};

// ============================================================================
// Dequantization Functions
// ============================================================================

const uint POSITION_BITS_MIN = 8u;
const uint POSITION_BITS_MAX = 24u;
const uint SCALE_BITS_MIN = 8u;
const uint SCALE_BITS_MAX = 16u;

// Clamp quantization bit depth to the supported range.
uint sanitize_quant_bits(uint bits, uint min_bits, uint max_bits) {
    return min(max(bits, min_bits), max_bits);
}

// Compute the reciprocal of the maximum quantized integer for a bit depth.
float compute_inv_quant_max(uint bits, uint min_bits, uint max_bits) {
    uint safe_bits = sanitize_quant_bits(bits, min_bits, max_bits);
    uint safe_max = (1u << safe_bits) - 1u;
    return 1.0 / float(safe_max);
}

// Extract quantized position components from packed data
uvec3 extract_quantized_position(uvec2 position_chunk) {
    // position_chunk.x = quantized_position[0] | (quantized_position[1] << 16)
    // position_chunk.y = quantized_position[2] | (chunk_id << 16)
    return uvec3(
        position_chunk.x & 0xFFFFu,
        position_chunk.x >> 16u,
        position_chunk.y & 0xFFFFu
    );
}

// Extract chunk ID from packed position data
uint extract_chunk_id(uvec2 position_chunk) {
    return position_chunk.y >> 16u;
}

// Extract quantized scale components from packed data (scalar version for struct layout)
uvec3 extract_quantized_scale(uint scale_lo, uint scale_hi) {
    // scale_lo = quantized_scale[0] | (quantized_scale[1] << 16)
    // scale_hi = quantized_scale[2] | (area_fp16 << 16)
    return uvec3(
        scale_lo & 0xFFFFu,
        scale_lo >> 16u,
        scale_hi & 0xFFFFu
    );
}

// Legacy overload for uvec2 (backward compatibility)
uvec3 extract_quantized_scale(uvec2 scale_area) {
    return extract_quantized_scale(scale_area.x, scale_area.y);
}

// Extract area from packed scale/area data (scalar version)
float extract_area(uint scale_hi) {
    uint area_bits = scale_hi >> 16u;
    return unpackHalf2x16(area_bits).x;
}

// Legacy overload for uvec2
float extract_area(uvec2 scale_area) {
    return extract_area(scale_area.y);
}

// Extract rotation quaternion from packed data (scalar version for struct layout)
vec4 extract_rotation(uint rotation_lo, uint rotation_hi) {
    vec2 xy = unpackHalf2x16(rotation_lo);
    vec2 zw = unpackHalf2x16(rotation_hi);
    return normalize(vec4(xy.x, xy.y, zw.x, zw.y));
}

// Legacy overload for uvec2 (backward compatibility)
vec4 extract_rotation(uvec2 rotation_packed) {
    return extract_rotation(rotation_packed.x, rotation_packed.y);
}

// Dequantize position using asset-local chunk bounds.
vec3 dequantize_position(uvec3 quantized, ChunkQuantization chunk) {
    float inv_max = compute_inv_quant_max(chunk.position_bits, POSITION_BITS_MIN, POSITION_BITS_MAX);
    return chunk.position_min + vec3(quantized) * inv_max * chunk.position_range;
}

// Dequantize scale using asset-local chunk bounds.
vec3 dequantize_scale(uvec3 quantized, ChunkQuantization chunk) {
    if (chunk.scale_bits == 0u) {
        // Scale not quantized - return zero (caller should use original scale)
        return vec3(0.0);
    }
    float inv_max = compute_inv_quant_max(chunk.scale_bits, SCALE_BITS_MIN, SCALE_BITS_MAX);
    return chunk.scale_min + vec3(quantized) * inv_max * chunk.scale_range;
}

// Extract normal from packed data
vec3 extract_normal(uint normal_xy, uint normal_z_stroke) {
    vec2 xy = unpackHalf2x16(normal_xy);
    vec2 z_stroke = unpackHalf2x16(normal_z_stroke);
    return vec3(xy.x, xy.y, z_stroke.x);
}

// Extract stroke age from packed normal data
float extract_stroke_age(uint normal_z_stroke) {
    vec2 z_stroke = unpackHalf2x16(normal_z_stroke);
    return z_stroke.y;
}

// ============================================================================
// Full Gaussian Unpacking
// ============================================================================

// Unpack a fully quantized Gaussian to standard fields
void unpack_gaussian_quantized(
    in GaussianQuantized g,
    in ChunkQuantization chunk,
    out vec3 position,
    out float opacity,
    out vec3 scale,
    out vec4 rotation,
    out vec4 sh_dc,
    out vec3 normal,
    out float stroke_age
) {
    uvec3 pos_q = extract_quantized_position(g.position_chunk);
    position = dequantize_position(pos_q, chunk);

    opacity = g.opacity;

    uvec3 scale_q = extract_quantized_scale(g.scale_area_lo, g.scale_area_hi);
    vec3 dequant_scale = dequantize_scale(scale_q, chunk);
    // If scale not quantized (scale_bits == 0), we'd need fallback
    // For now, assume quantized if using this path
    scale = (chunk.scale_bits > 0u) ? dequant_scale : vec3(1.0);

    rotation = extract_rotation(g.rotation_lo, g.rotation_hi);

    sh_dc = g.sh_dc;

    normal = extract_normal(g.normal_xy, g.normal_z_stroke);
    stroke_age = extract_stroke_age(g.normal_z_stroke);
}

// ============================================================================
// Conditional Loading Macros (for shader compatibility)
// ============================================================================

#ifdef USE_QUANTIZED_GAUSSIANS

// When quantization is enabled, use quantized structures
#define GAUSSIAN_STRUCT GaussianQuantized
#define CHUNK_STRUCT ChunkQuantization

// Macro to load position with dequantization
#define LOAD_POSITION_QUANTIZED(g, chunk) \
    dequantize_position(extract_quantized_position((g).position_chunk), chunk)

// Macro to load scale with dequantization
#define LOAD_SCALE_QUANTIZED(g, chunk) \
    ((chunk).scale_bits > 0u ? dequantize_scale(extract_quantized_scale((g).scale_area_lo, (g).scale_area_hi), chunk) : vec3(1.0))

// Macro to load rotation
#define LOAD_ROTATION_QUANTIZED(g) extract_rotation((g).rotation_lo, (g).rotation_hi)

// Macro to load opacity
#define LOAD_OPACITY_QUANTIZED(g) ((g).opacity)

// Macro to load SH DC
#define LOAD_SH_DC_QUANTIZED(g) ((g).sh_dc)

// Macro to load normal
#define LOAD_NORMAL_QUANTIZED(g) extract_normal((g).normal_xy, (g).normal_z_stroke)

#endif // USE_QUANTIZED_GAUSSIANS

// ============================================================================
// Utility: Compute quantization error bounds
// ============================================================================

// Calculate maximum position quantization error for a chunk
float get_max_position_error(ChunkQuantization chunk) {
    float inv_max = compute_inv_quant_max(chunk.position_bits, POSITION_BITS_MIN, POSITION_BITS_MAX);
    float half_step = 0.5 * inv_max;
    return max(max(
        chunk.position_range.x * half_step,
        chunk.position_range.y * half_step),
        chunk.position_range.z * half_step
    );
}

// Calculate maximum scale quantization error for a chunk
float get_max_scale_error(ChunkQuantization chunk) {
    if (chunk.scale_bits == 0u) {
        return 0.0;
    }
    float inv_max = compute_inv_quant_max(chunk.scale_bits, SCALE_BITS_MIN, SCALE_BITS_MAX);
    float half_step = 0.5 * inv_max;
    return max(max(
        chunk.scale_range.x * half_step,
        chunk.scale_range.y * half_step),
        chunk.scale_range.z * half_step
    );
}

#endif // QUANTIZATION_DEQUANT_GLSL
