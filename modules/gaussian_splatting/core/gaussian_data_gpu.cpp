/**
 * @file gaussian_data_gpu.cpp
 * @brief Companion .cpp for gaussian_data.h -- GPU buffer management and payload validation.
 *
 * Contains the GaussianData methods responsible for creating and updating GPU
 * storage buffers as well as the finite-range validation helpers that guard
 * against uploading corrupt data to the GPU.
 *
 * Split from gaussian_data.cpp to keep the translation unit size manageable
 * and to make build dependencies on the rendering back-end explicit.
 */

#include "gaussian_data.h"
#include "core/math/math_funcs.h"
#include "core/templates/span.h"
#include "servers/rendering/rendering_device.h"
#include "../interfaces/sync_policy.h"
#include "../renderer/gaussian_gpu_layout.h"

// ---------------------------------------------------------------------------
// Finite-range validation helpers
// ---------------------------------------------------------------------------

bool GaussianData::_is_finite_and_bounded(float p_value, float p_abs_max) {
    return Math::is_finite(p_value) && Math::abs(p_value) <= p_abs_max;
}

bool GaussianData::_is_finite_vector2(const Vector2 &p_value, float p_abs_max) {
    return _is_finite_and_bounded(p_value.x, p_abs_max) &&
            _is_finite_and_bounded(p_value.y, p_abs_max);
}

bool GaussianData::_is_finite_vector3(const Vector3 &p_value, float p_abs_max) {
    return _is_finite_and_bounded(p_value.x, p_abs_max) &&
            _is_finite_and_bounded(p_value.y, p_abs_max) &&
            _is_finite_and_bounded(p_value.z, p_abs_max);
}

bool GaussianData::_is_finite_quaternion(const Quaternion &p_value, float p_abs_max) {
    return _is_finite_and_bounded(p_value.x, p_abs_max) &&
            _is_finite_and_bounded(p_value.y, p_abs_max) &&
            _is_finite_and_bounded(p_value.z, p_abs_max) &&
            _is_finite_and_bounded(p_value.w, p_abs_max);
}

bool GaussianData::_is_finite_color(const Color &p_value, float p_abs_max) {
    return _is_finite_and_bounded(p_value.r, p_abs_max) &&
            _is_finite_and_bounded(p_value.g, p_abs_max) &&
            _is_finite_and_bounded(p_value.b, p_abs_max) &&
            _is_finite_and_bounded(p_value.a, p_abs_max);
}

// ---------------------------------------------------------------------------
// GPU payload validation (called under data_rwlock)
// ---------------------------------------------------------------------------

bool GaussianData::_validate_gpu_payload_locked(String *r_error_message) const {
    static constexpr float kMaxAbsPosition = 1.0e7f;
    static constexpr float kMaxAbsScale = 1.0e5f;
    static constexpr float kMaxAbsRotation = 1.0e4f;
    static constexpr float kMaxAbsColor = 6.5408e4f;
    static constexpr float kMaxAbsNormal = 1.0e4f;
    static constexpr float kMaxAbsArea = 1.0e8f;
    static constexpr float kMaxAbsBrushAxis = 1.0e4f;
    static constexpr float kMaxAbsStrokeAge = 1.0e8f;

    auto fail = [&](const String &p_message) -> bool {
        if (r_error_message != nullptr) {
            *r_error_message = p_message;
        }
        return false;
    };

    if (sh_first_order_count > 3u) {
        return fail(vformat("[GaussianData] Invalid SH first-order count: %d", sh_first_order_count));
    }

    const uint64_t expected_high_order_size = uint64_t(gaussians.size()) * uint64_t(sh_high_order_count);
    if (uint64_t(sh_high_order_coefficients.size()) != expected_high_order_size) {
        return fail(vformat(
                "[GaussianData] Invalid high-order SH storage size: expected=%d got=%d",
                int64_t(expected_high_order_size), int64_t(sh_high_order_coefficients.size())));
    }

    for (int i = 0; i < gaussians.size(); i++) {
        const Gaussian &g = gaussians[i];
        if (!_is_finite_vector3(g.position, kMaxAbsPosition)) {
            return fail(vformat("[GaussianData] Gaussian[%d] has invalid position", i));
        }
        if (!_is_finite_vector3(g.scale, kMaxAbsScale) || g.scale.x <= 0.0f || g.scale.y <= 0.0f || g.scale.z <= 0.0f) {
            return fail(vformat("[GaussianData] Gaussian[%d] has invalid scale", i));
        }
        if (!_is_finite_quaternion(g.rotation, kMaxAbsRotation) || g.rotation.length_squared() <= CMP_EPSILON) {
            return fail(vformat("[GaussianData] Gaussian[%d] has invalid rotation", i));
        }
        if (!Math::is_finite(g.opacity) || g.opacity < 0.0f || g.opacity > 1.0f) {
            return fail(vformat("[GaussianData] Gaussian[%d] has invalid opacity", i));
        }
        if (!_is_finite_color(g.sh_dc, kMaxAbsColor)) {
            return fail(vformat("[GaussianData] Gaussian[%d] has invalid SH DC coefficients", i));
        }
        for (int sh = 0; sh < 3; sh++) {
            if (!_is_finite_vector3(g.sh_1[sh], kMaxAbsColor)) {
                return fail(vformat("[GaussianData] Gaussian[%d] has invalid SH coefficient in sh_1[%d]", i, sh));
            }
        }
        if (!_is_finite_vector3(g.normal, kMaxAbsNormal)) {
            return fail(vformat("[GaussianData] Gaussian[%d] has invalid normal", i));
        }
        if (!_is_finite_and_bounded(g.area, kMaxAbsArea) || g.area < 0.0f) {
            return fail(vformat("[GaussianData] Gaussian[%d] has invalid area", i));
        }
        if (!_is_finite_and_bounded(g.stroke_age, kMaxAbsStrokeAge) || g.stroke_age < 0.0f) {
            return fail(vformat("[GaussianData] Gaussian[%d] has invalid stroke age", i));
        }
        if (!_is_finite_vector2(g.brush_axes, kMaxAbsBrushAxis)) {
            return fail(vformat("[GaussianData] Gaussian[%d] has invalid brush axes", i));
        }
    }

    for (int i = 0; i < sh_high_order_coefficients.size(); i++) {
        if (!_is_finite_vector3(sh_high_order_coefficients[i], kMaxAbsColor)) {
            return fail(vformat("[GaussianData] Invalid high-order SH coefficient at index %d", i));
        }
    }

    if (r_error_message != nullptr) {
        r_error_message->clear();
    }
    return true;
}

// ---------------------------------------------------------------------------
// GPU buffer creation and update
// ---------------------------------------------------------------------------

RID GaussianData::create_gpu_buffer(RenderingDevice *p_rd) const {
    // Create GPU buffer on the caller-provided owner device.
    // Callers must use the same RenderingDevice when releasing the returned RID.
    RenderingDevice *rd = p_rd;
    if (!rd) {
        // Legacy fallback for older call sites that do not pass an owner device.
        WARN_PRINT_ONCE("create_gpu_buffer() called without explicit owner device; falling back to main RenderingDevice");
        rd = get_rendering_device();
    }
    ERR_FAIL_NULL_V_MSG(rd, RID(), "RenderingDevice required for GPU buffer creation");

    Vector<PackedGaussian> packed_gaussians;
    SHCompressionMetrics metrics;
    {
        RWLockRead lock(data_rwlock);
        uint32_t gaussian_count = gaussians.size();
        if (gaussian_count == 0) {
            return RID();
        }
        String validation_error;
        ERR_FAIL_COND_V_MSG(!_validate_gpu_payload_locked(&validation_error), RID(), validation_error);
        const Vector3 *sh_coeff_ptr = sh_high_order_coefficients.is_empty()
                ? nullptr
                : sh_high_order_coefficients.ptr();
        pack_gaussians_range(gaussians,
                0,
                gaussian_count,
                packed_gaussians,
                metrics,
                sh_coeff_ptr,
                sh_first_order_count,
                sh_high_order_count);
    }

    uint32_t buffer_size = sizeof(PackedGaussian) * packed_gaussians.size();
    Span<const PackedGaussian> packed_span(packed_gaussians.ptr(), packed_gaussians.size());
    RID buffer = rd->storage_buffer_create(buffer_size, packed_span.reinterpret<uint8_t>());
    rd->set_resource_name(buffer, "GS_GaussianData_PackedGaussianBuffer");
    return buffer;
}

void GaussianData::update_gpu_buffer(RID p_buffer, RenderingDevice *p_rd) const {
    // Update existing GPU buffer on its owner RenderingDevice.
    // Passing the correct owner device is required to keep RID/device invariants.
    if (!p_buffer.is_valid()) {
        return;
    }

    RenderingDevice *rd = p_rd;
    if (!rd) {
        // Legacy fallback for older call sites that do not pass an owner device.
        WARN_PRINT_ONCE("update_gpu_buffer() called without explicit owner device; falling back to main RenderingDevice");
        rd = get_rendering_device();
    }
    ERR_FAIL_NULL_MSG(rd, "RenderingDevice required for GPU buffer update");

    Vector<PackedGaussian> packed_gaussians;
    SHCompressionMetrics metrics;
    {
        RWLockRead lock(data_rwlock);
        uint32_t gaussian_count = gaussians.size();
        if (gaussian_count == 0) {
            return;
        }
        String validation_error;
        ERR_FAIL_COND_MSG(!_validate_gpu_payload_locked(&validation_error), validation_error);
        const Vector3 *sh_coeff_ptr = sh_high_order_coefficients.is_empty()
                ? nullptr
                : sh_high_order_coefficients.ptr();
        pack_gaussians_range(gaussians,
                0,
                gaussian_count,
                packed_gaussians,
                metrics,
                sh_coeff_ptr,
                sh_first_order_count,
                sh_high_order_count);
    }

    uint32_t buffer_size = sizeof(PackedGaussian) * packed_gaussians.size();
    rd->buffer_update(p_buffer, 0, buffer_size, packed_gaussians.ptr());
    gs_device_utils::safe_submit_and_sync(rd);
}

Error GaussianData::validate_gpu_payload(String *r_error_message) const {
    RWLockRead lock(data_rwlock);
    if (_validate_gpu_payload_locked(r_error_message)) {
        return OK;
    }
    return ERR_INVALID_DATA;
}
