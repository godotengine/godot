#ifndef STREAMING_QUANTIZATION_H
#define STREAMING_QUANTIZATION_H

#include "gaussian_data.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include <cstdint>

// ============================================================================
// Per-Chunk Quantization (Unity Technique)
// ============================================================================

/**
 * @struct ChunkQuantizationInfo
 * @brief Per-chunk quantization bounds for position and scale compression.
 *
 * Based on Unity's Gaussian Splatting implementation, this provides 4x
 * compression for position data with minimal quality loss. Each chunk
 * stores min/max bounds which are used to normalize positions to [0, 1]
 * range before quantizing to the configured bit depth.
 *
 * Quantization formula:
 *   quantized = (position - min) / (max - min) * ((1 << bits) - 1)
 * Dequantization formula:
 *   position = min + quantized / ((1 << bits) - 1) * (max - min)
 *
 * Pattern notes:
 * - Value object with invariants (Pattern 3): validation is centralized
 *   in compute_from_gaussians and clear.
 * - position_bits / scale_bits are uint32_t bit-depth values (8-24 and 8-16
 *   respectively). scales_quantized gates whether scale fields are meaningful.
 */
struct ChunkQuantizationInfo {
    // Position bounds for normalization
    Vector3 position_min;
    Vector3 position_max;

    // Scale bounds for normalization (optional)
    Vector3 scale_min;
    Vector3 scale_max;

    // Computed derived values (for fast GPU dequantization)
    Vector3 position_range;  // position_max - position_min
    Vector3 scale_range;     // scale_max - scale_min

    // Quantization parameters from config
    uint32_t position_bits = 16;
    uint32_t scale_bits = 12;
    bool scales_quantized = false;

    // Initialize from chunk data
    void compute_from_gaussians(const LocalVector<Gaussian> &gaussians,
                                 uint32_t start_idx, uint32_t count,
                                 uint32_t pos_bits, uint32_t sc_bits,
                                 bool quantize_scale);

    // Quantize a position value
    void quantize_position(const Vector3 &pos, uint32_t &out_x, uint32_t &out_y, uint32_t &out_z) const;

    // Quantize a scale value
    void quantize_scale(const Vector3 &scale, uint32_t &out_x, uint32_t &out_y, uint32_t &out_z) const;

    // Dequantize a position value (for CPU-side verification)
    Vector3 dequantize_position(uint32_t x, uint32_t y, uint32_t z) const;

    // Dequantize a scale value (for CPU-side verification)
    Vector3 dequantize_scale(uint32_t x, uint32_t y, uint32_t z) const;

    // Clear/reset
    void clear();

    // Get maximum quantization error for this chunk
    float get_max_position_error() const;
    float get_max_scale_error() const;
};

/**
 * @struct ChunkQuantizationGPU
 * @brief GPU-uploadable per-chunk quantization data (64 bytes, 16-byte aligned).
 *
 * This is the GPU-side representation of ChunkQuantizationInfo, designed
 * for efficient upload and access in shaders.
 */
struct alignas(16) ChunkQuantizationGPU {
    float position_min[3];      // 12 bytes - World-space minimum position
    uint32_t position_bits;     // 4 bytes - Bit depth for positions (8-24)
    float position_range[3];    // 12 bytes - Range (max - min) for dequantization
    uint32_t scale_bits;        // 4 bytes - Bit depth for scales (8-16), 0 if not quantized
    float scale_min[3];         // 12 bytes - Scale minimum (if quantized)
    uint32_t start_index;       // 4 bytes - First Gaussian index in this chunk
    float scale_range[3];       // 12 bytes - Scale range (if quantized)
    uint32_t count;             // 4 bytes - Number of Gaussians in chunk
};

static_assert(sizeof(ChunkQuantizationGPU) == 64, "ChunkQuantizationGPU must be 64 bytes");

#endif // STREAMING_QUANTIZATION_H
