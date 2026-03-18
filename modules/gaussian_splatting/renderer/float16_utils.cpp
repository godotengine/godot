#include "float16_utils.h"
#include "core/math/math_funcs.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace Float16Utils {

// ============================================================================
// Basic Float16 Conversion
// ============================================================================

uint16_t float_to_half(float value) {
    // IEEE 754 float32 to float16 conversion
    uint32_t f32;
    memcpy(&f32, &value, sizeof(float));

    uint32_t sign = (f32 >> 16) & 0x8000;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = f32 & 0x7FFFFF;

    // Handle special cases
    if (exponent <= 0) {
        // Denormalized or zero
        if (exponent < -10) {
            // Too small, return zero with sign
            return (uint16_t)sign;
        }
        // Denormalized half
        mantissa |= 0x800000; // Add implicit 1
        int shift = 14 - exponent;
        mantissa >>= shift;
        return (uint16_t)(sign | (mantissa >> 13));
    } else if (exponent >= 31) {
        // Infinity or NaN
        if ((f32 & 0x7FFFFFFF) > 0x7F800000) {
            // NaN - preserve some mantissa bits
            return (uint16_t)(sign | 0x7E00 | (mantissa >> 13));
        }
        // Infinity
        return (uint16_t)(sign | 0x7C00);
    }

    // Normal case
    uint16_t half = (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));

    // Round to nearest even
    if (mantissa & 0x1000) {
        if ((mantissa & 0x2FFF) || (half & 1)) {
            half++;
            // Handle overflow to infinity
            if ((half & 0x7C00) == 0x7C00) {
                return (uint16_t)(sign | 0x7C00);
            }
        }
    }

    return half;
}

float half_to_float(uint16_t half_bits) {
    uint32_t sign = (half_bits & 0x8000) << 16;
    int32_t exponent = (half_bits >> 10) & 0x1F;
    uint32_t mantissa = half_bits & 0x3FF;

    uint32_t f32;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            f32 = sign;
        } else {
            // Denormalized - normalize it
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= ~0x400;
            exponent = exponent - 15 + 127;
            f32 = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Infinity or NaN
        f32 = sign | 0x7F800000 | (mantissa << 13);
    } else {
        // Normalized
        exponent = exponent - 15 + 127;
        f32 = sign | (exponent << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

uint32_t pack_half2(float a, float b) {
    uint16_t ha = float_to_half(a);
    uint16_t hb = float_to_half(b);
    return (uint32_t)ha | ((uint32_t)hb << 16);
}

void unpack_half2(uint32_t packed, float &a, float &b) {
    a = half_to_float((uint16_t)(packed & 0xFFFF));
    b = half_to_float((uint16_t)(packed >> 16));
}

// ============================================================================
// Vector Conversions
// ============================================================================

void pack_vector3_half(const Vector3 &v, uint32_t &out_xy, uint32_t &out_z) {
    out_xy = pack_half2(v.x, v.y);
    out_z = (uint32_t)float_to_half(v.z);
}

Vector3 unpack_vector3_half(uint32_t xy, uint32_t z) {
    Vector3 result;
    unpack_half2(xy, result.x, result.y);
    result.z = half_to_float((uint16_t)(z & 0xFFFF));
    return result;
}

void pack_quaternion_half(const Quaternion &q, uint32_t &out_xy, uint32_t &out_zw) {
    out_xy = pack_half2(q.x, q.y);
    out_zw = pack_half2(q.z, q.w);
}

Quaternion unpack_quaternion_half(uint32_t xy, uint32_t zw) {
    Quaternion result;
    unpack_half2(xy, result.x, result.y);
    unpack_half2(zw, result.z, result.w);
    result.normalize();
    return result;
}

// ============================================================================
// Quantized Position Encoding
// ============================================================================

void compute_quantization_chunks(
    const LocalVector<Vector3> &positions,
    uint32_t chunk_size,
    LocalVector<QuantizationChunk> &out_chunks) {

    if (positions.size() == 0) {
        out_chunks.clear();
        return;
    }

    uint32_t total = positions.size();
    uint32_t num_chunks = (total + chunk_size - 1) / chunk_size;
    out_chunks.resize(num_chunks);

    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        uint32_t start = chunk_idx * chunk_size;
        uint32_t end = MIN(start + chunk_size, total);
        uint32_t count = end - start;

        // Compute bounding box for this chunk
        Vector3 min_pos = positions[start];
        Vector3 max_pos = positions[start];

        for (uint32_t i = start + 1; i < end; i++) {
            min_pos.x = MIN(min_pos.x, positions[i].x);
            min_pos.y = MIN(min_pos.y, positions[i].y);
            min_pos.z = MIN(min_pos.z, positions[i].z);
            max_pos.x = MAX(max_pos.x, positions[i].x);
            max_pos.y = MAX(max_pos.y, positions[i].y);
            max_pos.z = MAX(max_pos.z, positions[i].z);
        }

        // Compute center and max extent
        Vector3 center = (min_pos + max_pos) * 0.5f;
        Vector3 extent = (max_pos - min_pos) * 0.5f;
        float max_extent = MAX(MAX(extent.x, extent.y), extent.z);

        // Ensure non-zero extent for degenerate cases
        max_extent = MAX(max_extent, 0.001f);

        out_chunks[chunk_idx].center = center;
        out_chunks[chunk_idx].max_extent = max_extent;
        out_chunks[chunk_idx].start_idx = start;
        out_chunks[chunk_idx].count = count;
    }
}

void encode_position_quantized(
    const Vector3 &position,
    const QuantizationChunk &chunk,
    uint32_t &out_xy,
    uint32_t &out_z) {

    // Compute relative offset from chunk center
    Vector3 offset = position - chunk.center;

    // Pack as float16
    pack_vector3_half(offset, out_xy, out_z);
}

Vector3 decode_position_quantized(
    uint32_t xy,
    uint32_t z,
    const QuantizationChunk &chunk) {

    // Unpack relative offset
    Vector3 offset = unpack_vector3_half(xy, z);

    // Add chunk center
    return offset + chunk.center;
}

// ============================================================================
// SH Coefficient Conversions
// ============================================================================

void pack_sh_coefficients_half(
    const float *coeffs,
    uint32_t count,
    uint16_t *out_packed) {

    for (uint32_t i = 0; i < count; i++) {
        out_packed[i] = float_to_half(coeffs[i]);
    }
}

void unpack_sh_coefficients_half(
    const uint16_t *packed,
    uint32_t count,
    float *out_coeffs) {

    for (uint32_t i = 0; i < count; i++) {
        out_coeffs[i] = half_to_float(packed[i]);
    }
}

// ============================================================================
// Bulk Conversion Utilities
// ============================================================================

void convert_float32_to_float16_batch(
    const float *src,
    uint16_t *dst,
    uint32_t count,
    Float16ConversionStats *stats) {

    if (stats) {
        stats->values_converted = count;
        stats->denormals_encountered = 0;
        stats->infinities_clamped = 0;
        stats->nans_replaced = 0;
    }

    for (uint32_t i = 0; i < count; i++) {
        float val = src[i];

        // Track special cases
        if (stats) {
            if (std::isnan(val)) {
                stats->nans_replaced++;
            } else if (std::isinf(val)) {
                stats->infinities_clamped++;
            } else if (val != 0.0f && std::abs(val) < 6.1e-5f) {
                // Float16 smallest normal is ~6.1e-5
                stats->denormals_encountered++;
            }
        }

        dst[i] = float_to_half(val);
    }
}

Float16ConversionStats measure_precision_loss(
    const float *original,
    uint32_t count) {

    Float16ConversionStats stats;
    stats.values_converted = count;

    double total_error = 0.0;
    uint32_t valid_count = 0;

    for (uint32_t i = 0; i < count; i++) {
        float val = original[i];

        // Skip special values
        if (std::isnan(val) || std::isinf(val) || val == 0.0f) {
            continue;
        }

        // Convert to half and back
        uint16_t half = float_to_half(val);
        float reconstructed = half_to_float(half);

        // Compute relative error
        float abs_val = std::abs(val);
        float error = std::abs(reconstructed - val) / abs_val;

        total_error += error;
        valid_count++;

        if (error > stats.max_relative_error) {
            stats.max_relative_error = error;
        }
    }

    if (valid_count > 0) {
        stats.avg_relative_error = (float)(total_error / valid_count);
    }

    return stats;
}

} // namespace Float16Utils
