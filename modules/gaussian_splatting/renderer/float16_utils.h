#ifndef FLOAT16_UTILS_H
#define FLOAT16_UTILS_H

#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include "core/math/color.h"
#include "core/templates/local_vector.h"
#include <cstdint>

/**
 * @file float16_utils.h
 * @brief Float16 (half-precision) conversion utilities for Gaussian Splatting data.
 *
 * Provides efficient conversion between float32 and float16 formats for GPU upload,
 * including quantized position encoding for improved precision.
 */

namespace Float16Utils {

// ============================================================================
// Basic Float16 Conversion
// ============================================================================

/**
 * @brief Converts a float32 value to float16 (IEEE 754 half-precision).
 * @param value Input float32 value.
 * @return Packed uint16 representation of the float16.
 *
 * Handles denormals, infinities, and NaN correctly.
 */
uint16_t float_to_half(float value);

/**
 * @brief Converts a float16 value back to float32.
 * @param half_bits Packed uint16 representation of the float16.
 * @return Converted float32 value.
 */
float half_to_float(uint16_t half_bits);

/**
 * @brief Packs two float32 values into a single uint32 (two float16s).
 * @param a First float value.
 * @param b Second float value.
 * @return Packed uint32 with a in lower 16 bits, b in upper 16 bits.
 */
uint32_t pack_half2(float a, float b);

/**
 * @brief Unpacks two float16 values from a uint32.
 * @param packed Packed uint32 containing two float16 values.
 * @param a Output first float value.
 * @param b Output second float value.
 */
void unpack_half2(uint32_t packed, float &a, float &b);

// ============================================================================
// Vector Conversions
// ============================================================================

/**
 * @brief Packs a Vector3 into two uint32s (6 bytes total, padded to 8).
 * @param v Input Vector3.
 * @param out_xy Output packed xy components (2 halfs).
 * @param out_z Output packed z component (1 half in lower bits).
 */
void pack_vector3_half(const Vector3 &v, uint32_t &out_xy, uint32_t &out_z);

/**
 * @brief Unpacks a Vector3 from two uint32s.
 * @param xy Packed xy components.
 * @param z Packed z component.
 * @return Unpacked Vector3.
 */
Vector3 unpack_vector3_half(uint32_t xy, uint32_t z);

/**
 * @brief Packs a Quaternion into two uint32s (8 bytes total).
 * @param q Input quaternion.
 * @param out_xy Output packed xy components.
 * @param out_zw Output packed zw components.
 */
void pack_quaternion_half(const Quaternion &q, uint32_t &out_xy, uint32_t &out_zw);

/**
 * @brief Unpacks a Quaternion from two uint32s.
 * @param xy Packed xy components.
 * @param zw Packed zw components.
 * @return Unpacked quaternion (normalized).
 */
Quaternion unpack_quaternion_half(uint32_t xy, uint32_t zw);

// ============================================================================
// Quantized Position Encoding
// ============================================================================

/**
 * @struct QuantizationChunk
 * @brief Per-chunk quantization data for improved FP16 position precision.
 *
 * Stores the center offset for a chunk of Gaussians, allowing positions
 * to be encoded as FP16 relative offsets from the center.
 */
struct QuantizationChunk {
    Vector3 center;     ///< World-space center of the chunk
    float max_extent;   ///< Maximum distance from center (for normalization)
    uint32_t start_idx; ///< First Gaussian index in this chunk
    uint32_t count;     ///< Number of Gaussians in this chunk
};

/**
 * @brief Computes quantization chunks for a set of positions.
 * @param positions Input position array.
 * @param chunk_size Number of Gaussians per chunk.
 * @param out_chunks Output vector of quantization chunks.
 */
void compute_quantization_chunks(
    const LocalVector<Vector3> &positions,
    uint32_t chunk_size,
    LocalVector<QuantizationChunk> &out_chunks);

/**
 * @brief Encodes a position as FP16 relative to a chunk center.
 * @param position World-space position.
 * @param chunk Quantization chunk containing the center offset.
 * @param out_xy Output packed xy offset (2 halfs).
 * @param out_z Output packed z offset (1 half).
 */
void encode_position_quantized(
    const Vector3 &position,
    const QuantizationChunk &chunk,
    uint32_t &out_xy,
    uint32_t &out_z);

/**
 * @brief Decodes a quantized FP16 position back to world-space.
 * @param xy Packed xy offset.
 * @param z Packed z offset.
 * @param chunk Quantization chunk containing the center offset.
 * @return World-space position.
 */
Vector3 decode_position_quantized(
    uint32_t xy,
    uint32_t z,
    const QuantizationChunk &chunk);

// ============================================================================
// SH Coefficient Conversions
// ============================================================================

/**
 * @brief Packs SH coefficients as float16 values.
 * @param coeffs Input float32 coefficient array.
 * @param count Number of coefficients.
 * @param out_packed Output packed data (half the size).
 */
void pack_sh_coefficients_half(
    const float *coeffs,
    uint32_t count,
    uint16_t *out_packed);

/**
 * @brief Unpacks SH coefficients from float16 to float32.
 * @param packed Input packed float16 array.
 * @param count Number of coefficients.
 * @param out_coeffs Output float32 coefficient array.
 */
void unpack_sh_coefficients_half(
    const uint16_t *packed,
    uint32_t count,
    float *out_coeffs);

// ============================================================================
// Bulk Conversion Utilities
// ============================================================================

/**
 * @struct Float16ConversionStats
 * @brief Statistics for Float16 conversion operations.
 */
struct Float16ConversionStats {
    uint64_t values_converted = 0;
    uint64_t denormals_encountered = 0;
    uint64_t infinities_clamped = 0;
    uint64_t nans_replaced = 0;
    float max_relative_error = 0.0f;
    float avg_relative_error = 0.0f;
};

/**
 * @brief Converts a batch of float32 values to float16.
 * @param src Source float32 array.
 * @param dst Destination uint16 array.
 * @param count Number of values.
 * @param stats Optional statistics output.
 */
void convert_float32_to_float16_batch(
    const float *src,
    uint16_t *dst,
    uint32_t count,
    Float16ConversionStats *stats = nullptr);

/**
 * @brief Measures the precision loss for a batch of values.
 * @param original Original float32 values.
 * @param count Number of values.
 * @return Statistics including max and average relative error.
 */
Float16ConversionStats measure_precision_loss(
    const float *original,
    uint32_t count);

} // namespace Float16Utils

#endif // FLOAT16_UTILS_H
