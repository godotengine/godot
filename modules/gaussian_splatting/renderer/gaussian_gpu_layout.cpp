#include "gaussian_gpu_layout.h"
#include "float16_config.h"
#include "float16_utils.h"
#include "gpu_debug_utils.h"

#include "../core/gs_project_settings.h"
#include "../logger/gs_logger.h"
#include "../logger/gs_debug_trace.h"
#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/variant/variant.h"
#include <cstring>

namespace {
static bool _is_data_log_enabled() { return gs::settings::is_data_log_enabled(); }

static float bits_to_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(uint32_t));
    return value;
}

static uint32_t encode_rgb9e5(const Vector3 &value) {
    const float max_channel = 65408.0f; // Maximum representable value in RGB9E5
    float r = CLAMP(value.x, 0.0f, max_channel);
    float g = CLAMP(value.y, 0.0f, max_channel);
    float b = CLAMP(value.z, 0.0f, max_channel);

    float max_component = MAX(MAX(r, g), b);
    if (max_component < 1.5258789e-5f) { // 2^-16
        return 0;
    }

    float exponent = Math::floor(Math::log(max_component) / Math::log(2.0f));
    int exp_shared = int(exponent) + 1 + 15; // Bias of 15
    exp_shared = CLAMP(exp_shared, 0, 31);

    float denom = Math::pow(2.0f, float(exp_shared - 15 - 9));
    int rm = int(Math::round(r / denom));
    int gm = int(Math::round(g / denom));
    int bm = int(Math::round(b / denom));

    if (rm > 511 || gm > 511 || bm > 511) {
        exp_shared = MIN(exp_shared + 1, 31);
        denom = Math::pow(2.0f, float(exp_shared - 15 - 9));
        rm = int(Math::round(r / denom));
        gm = int(Math::round(g / denom));
        bm = int(Math::round(b / denom));
    }

    rm = CLAMP(rm, 0, 511);
    gm = CLAMP(gm, 0, 511);
    bm = CLAMP(bm, 0, 511);

    return (uint32_t(exp_shared) << 27) | (uint32_t(bm) << 18) | (uint32_t(gm) << 9) | uint32_t(rm);
}

static uint32_t pack_sh_metadata(uint32_t stored_first, uint32_t stored_high, uint32_t encoded_total) {
    const uint32_t encoding = 1u; // RGB9E5 encoding identifier
    stored_first = MIN(stored_first, 0xFFu);
    stored_high = MIN(stored_high, 0xFFu);
    encoded_total = MIN(encoded_total, 0xFFu);
    return stored_first | (stored_high << 8) | (encoded_total << 16) | (encoding << 24);
}

} // namespace

void PackedSphericalHarmonics::clear() {
    for (int i = 0; i < 4; i++) {
        dc[i] = 0.0f;
    }
    for (uint32_t i = 0; i < MAX_ENCODED_COEFFICIENTS; i++) {
        encoded[i] = 0.0f;
    }
}

void pack_gaussian(const Gaussian &src,
        PackedGaussian &dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        uint32_t coefficient_limit) {
    dst.position[0] = src.position.x;
    dst.position[1] = src.position.y;
    dst.position[2] = src.position.z;
    dst.opacity = src.opacity;

    dst.scale[0] = src.scale.x;
    dst.scale[1] = src.scale.y;
    dst.scale[2] = src.scale.z;
    dst.area = src.area;

    dst.rotation[0] = src.rotation.x;
    dst.rotation[1] = src.rotation.y;
    dst.rotation[2] = src.rotation.z;
    dst.rotation[3] = src.rotation.w;
    // Sample a few SH DC values for debug trace (avoids per-gaussian logging).
    static int pack_debug_count = 0;
    if (++pack_debug_count <= 5) {
        GaussianSplatting::debug_trace_record_pack_sh(src.sh_dc, src.opacity);
    }

    dst.sh.clear();
    dst.sh.dc[0] = src.sh_dc.r;
    dst.sh.dc[1] = src.sh_dc.g;
    dst.sh.dc[2] = src.sh_dc.b;
    dst.sh.dc[3] = src.sh_dc.a;

    uint32_t encoded_capacity = MIN<uint32_t>(coefficient_limit, PackedSphericalHarmonics::MAX_ENCODED_COEFFICIENTS);
    uint32_t first_count = MIN<uint32_t>(first_order_count, 3u);
    uint32_t stored_first = MIN<uint32_t>(first_count, encoded_capacity);
    uint32_t encoded_total = 0;

    for (uint32_t i = 0; i < stored_first; i++) {
        uint32_t packed = encode_rgb9e5(src.sh_1[i]);
        dst.sh.encoded[encoded_total++] = bits_to_float(packed);
    }

    uint32_t stored_high = 0;
    if (higher_order_count > 0 && encoded_total < encoded_capacity) {
        stored_high = MIN<uint32_t>(higher_order_count, encoded_capacity - encoded_total);
        for (uint32_t i = 0; i < stored_high; i++) {
            Vector3 coeff = higher_order_coeffs ? higher_order_coeffs[i] : Vector3();
            uint32_t packed = encode_rgb9e5(coeff);
            dst.sh.encoded[encoded_total++] = bits_to_float(packed);
        }
    }

    dst.normal[0] = src.normal.x;
    dst.normal[1] = src.normal.y;
    dst.normal[2] = src.normal.z;
    dst.stroke_age = src.stroke_age;

    dst.brush_axes[0] = src.brush_axes.x;
    dst.brush_axes[1] = src.brush_axes.y;
    dst.painterly_meta = src.painterly_meta;

    // DEBUG: Log first gaussian packing.
    static bool logged_once = false;
    if (_is_data_log_enabled() && !logged_once) {
        GS_LOG_RENDERER_DEBUG(vformat("[GPU Pack] First gaussian: first_order_count=%d, stored_first=%d, stored_high=%d, encoded_total=%d",
            first_order_count, stored_first, stored_high, encoded_total));
        GS_LOG_RENDERER_DEBUG(vformat("[GPU Pack] First gaussian: src.sh_1[0] = (%f, %f, %f)",
            src.sh_1[0].x, src.sh_1[0].y, src.sh_1[0].z));
    }

    dst.sh_metadata = pack_sh_metadata(stored_first, stored_high, encoded_total);

    if (_is_data_log_enabled() && !logged_once) {
        GS_LOG_RENDERER_DEBUG(vformat("[GPU Pack] sh_metadata = 0x%08X", dst.sh_metadata));
        logged_once = true;
    }

    metrics.raw_bytes += sizeof(Color);
    if (encoded_total > 0) {
        metrics.raw_bytes += sizeof(Vector3) * encoded_total;
    }
    metrics.compressed_bytes += sizeof(dst.sh.dc) + sizeof(float) * encoded_total;
    metrics.coefficient_count += encoded_total;
}

void pack_gaussians_range(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        Vector<PackedGaussian> &dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        uint32_t coefficient_limit) {
    static int pack_range_call_count = 0;
    if (++pack_range_call_count <= 3) {
        if (src.size() > 0 && start < src.size()) {
            const Gaussian &g0 = src[start];
            GaussianSplatting::debug_trace_record_pack_range(count, start, src.size(), g0.sh_dc, g0.opacity);
        } else {
            GaussianSplatting::debug_trace_record_pack_range(count, start, src.size(), Color(), 0.0f);
        }
    }
    if (count == 0) {
        dst.clear();
        return;
    }

    ERR_FAIL_COND_MSG(start + count > src.size(), "pack_gaussians_range out of bounds");

    dst.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        const Vector3 *coeff_ptr = nullptr;
        if (higher_order_coeffs && higher_order_count > 0) {
            coeff_ptr = higher_order_coeffs + (size_t)(start + i) * higher_order_count;
        }
        pack_gaussian(src[start + i], dst.write[i], metrics, coeff_ptr, first_order_count, higher_order_count, coefficient_limit);
    }
}

void pack_gaussians_range_raw(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        PackedGaussian *dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        uint32_t coefficient_limit) {
    if (count == 0 || dst == nullptr) {
        return;
    }

    ERR_FAIL_COND_MSG(start + count > src.size(), "pack_gaussians_range_raw out of bounds");

    for (uint32_t i = 0; i < count; i++) {
        const Vector3 *coeff_ptr = nullptr;
        if (higher_order_coeffs && higher_order_count > 0) {
            coeff_ptr = higher_order_coeffs + (size_t)(start + i) * higher_order_count;
        }
        pack_gaussian(src[start + i], dst[i], metrics, coeff_ptr, first_order_count, higher_order_count, coefficient_limit);
    }
}

void pack_gaussians_range_raw(const Gaussian *src,
        uint32_t start,
        uint32_t count,
        PackedGaussian *dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        uint32_t coefficient_limit) {
    if (count == 0 || dst == nullptr || src == nullptr) {
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        const Vector3 *coeff_ptr = nullptr;
        if (higher_order_coeffs && higher_order_count > 0) {
            coeff_ptr = higher_order_coeffs + (size_t)(start + i) * higher_order_count;
        }
        pack_gaussian(src[start + i], dst[i], metrics, coeff_ptr, first_order_count, higher_order_count, coefficient_limit);
    }
}

void pack_gaussians_range_limited(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        Vector<PackedGaussian> &dst,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        const uint8_t *coefficient_limits,
        uint32_t coefficient_limit) {
    if (count == 0) {
        dst.clear();
        return;
    }

    ERR_FAIL_COND_MSG(start + count > src.size(), "pack_gaussians_range_limited out of bounds");

    dst.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        const Vector3 *coeff_ptr = nullptr;
        if (higher_order_coeffs && higher_order_count > 0) {
            coeff_ptr = higher_order_coeffs + (size_t)(start + i) * higher_order_count;
        }
        uint32_t limit = coefficient_limit;
        if (coefficient_limits) {
            limit = MIN<uint32_t>(limit, coefficient_limits[i]);
        }
        pack_gaussian(src[start + i], dst.write[i], metrics, coeff_ptr, first_order_count, higher_order_count, limit);
    }
}

// ============================================================================
// Float16 Packing Implementations
// ============================================================================

void PackedSphericalHarmonicsF16::clear() {
    for (int i = 0; i < 4; i++) {
        dc[i] = 0.0f;
    }
    for (uint32_t i = 0; i < MAX_ENCODED_COEFFICIENTS; i++) {
        encoded[i] = 0;
    }
}

void pack_gaussian_f16(const Gaussian &src,
        PackedGaussianF16 &dst,
        SHCompressionMetrics &metrics,
        const Vector3 &chunk_center,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        uint32_t coefficient_limit) {

    // Position: encode as Float16 relative to chunk center
    Vector3 relative_pos = src.position - chunk_center;
    dst.position_xy = Float16Utils::pack_half2(relative_pos.x, relative_pos.y);
    dst.position_z_pad = (uint32_t)Float16Utils::float_to_half(relative_pos.z);

    // Opacity: keep FP32
    dst.opacity = src.opacity;

    // Scale: keep FP32 (precision-sensitive)
    dst.scale[0] = src.scale.x;
    dst.scale[1] = src.scale.y;
    dst.scale[2] = src.scale.z;
    dst.area = src.area;

    // Rotation: encode as Float16
    dst.rotation_xy = Float16Utils::pack_half2(src.rotation.x, src.rotation.y);
    dst.rotation_zw = Float16Utils::pack_half2(src.rotation.z, src.rotation.w);

    // Spherical Harmonics: DC as FP32, higher-order as FP16
    dst.sh.clear();
    dst.sh.dc[0] = src.sh_dc.r;
    dst.sh.dc[1] = src.sh_dc.g;
    dst.sh.dc[2] = src.sh_dc.b;
    dst.sh.dc[3] = src.sh_dc.a;

    uint32_t encoded_capacity = MIN<uint32_t>(coefficient_limit, PackedSphericalHarmonicsF16::MAX_ENCODED_COEFFICIENTS);
    uint32_t first_count = MIN<uint32_t>(first_order_count, 3);
    uint32_t stored_first = MIN<uint32_t>(first_count, encoded_capacity);
    uint32_t encoded_total = 0;

    // Encode first-order SH coefficients
    // Each sh_1[i] is a Vector3 (RGB), we pack as RGB9E5 then store as FP16 encoded bits
    for (uint32_t i = 0; i < stored_first; i++) {
        uint32_t packed = encode_rgb9e5(src.sh_1[i]);
        // Store as uint16 (the packed RGB9E5 is 32-bit, but we can store low 16 bits)
        // Actually, RGB9E5 is 32-bit, so we need to handle this differently
        // For FP16 SH, we'll convert each channel to FP16 separately
        // But the existing code uses RGB9E5 encoding for compact storage
        // For now, keep the same RGB9E5 encoding but store the bits
        dst.sh.encoded[encoded_total++] = (uint16_t)(packed & 0xFFFF);
        if (encoded_total < encoded_capacity) {
            dst.sh.encoded[encoded_total++] = (uint16_t)(packed >> 16);
        }
    }

    // Higher-order SH coefficients
    uint32_t stored_high = 0;
    if (higher_order_count > 0 && encoded_total < encoded_capacity) {
        stored_high = MIN<uint32_t>(higher_order_count, (encoded_capacity - encoded_total) / 2);
        for (uint32_t i = 0; i < stored_high && encoded_total + 1 < encoded_capacity; i++) {
            Vector3 coeff = higher_order_coeffs ? higher_order_coeffs[i] : Vector3();
            uint32_t packed = encode_rgb9e5(coeff);
            dst.sh.encoded[encoded_total++] = (uint16_t)(packed & 0xFFFF);
            if (encoded_total < encoded_capacity) {
                dst.sh.encoded[encoded_total++] = (uint16_t)(packed >> 16);
            }
        }
    }

    // Normal, stroke_age, brush_axes (keep FP32)
    dst.normal[0] = src.normal.x;
    dst.normal[1] = src.normal.y;
    dst.normal[2] = src.normal.z;
    dst.stroke_age = src.stroke_age;

    dst.brush_axes[0] = src.brush_axes.x;
    dst.brush_axes[1] = src.brush_axes.y;
    dst.painterly_meta = src.painterly_meta;

    // Pack SH metadata with FP16 encoding indicator
    const uint32_t SH_ENCODING_F16 = 2u; // New encoding type for FP16
    dst.sh_metadata = (stored_first & 0xFFu) |
                      ((stored_high & 0xFFu) << 8) |
                      ((encoded_total & 0xFFu) << 16) |
                      (SH_ENCODING_F16 << 24);

    // Zero padding arrays for alignment
    memset(dst._pre_sh_padding, 0, sizeof(dst._pre_sh_padding));
    memset(dst._padding, 0, sizeof(dst._padding));

    // Update metrics
    metrics.raw_bytes += sizeof(Color);
    if (encoded_total > 0) {
        metrics.raw_bytes += sizeof(Vector3) * ((encoded_total + 1) / 2);
    }
    metrics.compressed_bytes += sizeof(dst.sh.dc) + sizeof(uint16_t) * encoded_total;
    metrics.coefficient_count += (encoded_total + 1) / 2;
}

void pack_gaussians_range_f16(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        Vector<PackedGaussianF16> &dst,
        SHCompressionMetrics &metrics,
        const Vector3 &chunk_center,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count,
        uint32_t coefficient_limit) {

    if (count == 0) {
        dst.clear();
        return;
    }

    ERR_FAIL_COND_MSG(start + count > src.size(), "pack_gaussians_range_f16 out of bounds");

    dst.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        const Vector3 *coeff_ptr = nullptr;
        if (higher_order_coeffs && higher_order_count > 0) {
            coeff_ptr = higher_order_coeffs + (size_t)(start + i) * higher_order_count;
        }
        pack_gaussian_f16(src[start + i], dst.write[i], metrics, chunk_center,
                coeff_ptr, first_order_count, higher_order_count, coefficient_limit);
    }
}

void pack_gaussians_chunked_f16(const LocalVector<Gaussian> &src,
        uint32_t start,
        uint32_t count,
        uint32_t chunk_size,
        Vector<PackedGaussianF16> &dst,
        Vector<QuantizationChunkGPU> &chunks,
        SHCompressionMetrics &metrics,
        const Vector3 *higher_order_coeffs,
        uint32_t first_order_count,
        uint32_t higher_order_count) {

    if (count == 0) {
        dst.clear();
        chunks.clear();
        return;
    }

    ERR_FAIL_COND_MSG(start + count > src.size(), "pack_gaussians_chunked_f16 out of bounds");

    // Compute number of chunks
    uint32_t num_chunks = (count + chunk_size - 1) / chunk_size;
    chunks.resize(num_chunks);
    dst.resize(count);

    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        uint32_t chunk_start = start + chunk_idx * chunk_size;
        uint32_t chunk_end = MIN(chunk_start + chunk_size, start + count);
        uint32_t chunk_count = chunk_end - chunk_start;

        // Compute bounding box for this chunk
        Vector3 min_pos = src[chunk_start].position;
        Vector3 max_pos = src[chunk_start].position;

        for (uint32_t i = chunk_start + 1; i < chunk_end; i++) {
            min_pos.x = MIN(min_pos.x, src[i].position.x);
            min_pos.y = MIN(min_pos.y, src[i].position.y);
            min_pos.z = MIN(min_pos.z, src[i].position.z);
            max_pos.x = MAX(max_pos.x, src[i].position.x);
            max_pos.y = MAX(max_pos.y, src[i].position.y);
            max_pos.z = MAX(max_pos.z, src[i].position.z);
        }

        // Compute center and extent
        Vector3 center = (min_pos + max_pos) * 0.5f;
        Vector3 extent = (max_pos - min_pos) * 0.5f;
        float max_extent = MAX(MAX(extent.x, extent.y), extent.z);
        max_extent = MAX(max_extent, 0.001f); // Avoid zero extent

        // Fill GPU chunk data
        QuantizationChunkGPU &gpu_chunk = chunks.write[chunk_idx];
        gpu_chunk.center[0] = center.x;
        gpu_chunk.center[1] = center.y;
        gpu_chunk.center[2] = center.z;
        gpu_chunk.start_index = chunk_start - start; // Relative to output buffer
        gpu_chunk.max_extent = max_extent;
        gpu_chunk.count = chunk_count;
        gpu_chunk._padding[0] = 0;
        gpu_chunk._padding[1] = 0;

        // Pack Gaussians in this chunk
        for (uint32_t i = 0; i < chunk_count; i++) {
            uint32_t src_idx = chunk_start + i;
            uint32_t dst_idx = chunk_idx * chunk_size + i;

            const Vector3 *coeff_ptr = nullptr;
            if (higher_order_coeffs && higher_order_count > 0) {
                coeff_ptr = higher_order_coeffs + (size_t)src_idx * higher_order_count;
            }

            pack_gaussian_f16(src[src_idx], dst.write[dst_idx], metrics, center,
                    coeff_ptr, first_order_count, higher_order_count,
                    PackedSphericalHarmonicsF16::MAX_ENCODED_COEFFICIENTS);
        }
    }
}

bool is_float16_storage_enabled() {
    return g_float16_config.use_float16_storage;
}

uint32_t get_packed_gaussian_size() {
    if (g_float16_config.use_float16_storage) {
        return sizeof(PackedGaussianF16);
    }
    return sizeof(PackedGaussian);
}
