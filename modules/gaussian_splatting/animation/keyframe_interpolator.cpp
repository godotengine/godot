#include "keyframe_interpolator.h"
#include "core/math/math_funcs.h"

namespace GaussianSplatting {

Variant KeyframeInterpolator::interpolate(const LocalVector<Keyframe>& keyframes, float time) const {
    if (keyframes.is_empty()) {
        return Variant();
    }

    if (keyframes.size() == 1) {
        return keyframes[0].value;
    }

    // Find surrounding keyframes
    int index_a, index_b;
    _find_keyframe_indices(keyframes, time, index_a, index_b);

    if (index_a == index_b) {
        return keyframes[index_a].value;
    }

    const Keyframe& kf_a = keyframes[index_a];
    const Keyframe& kf_b = keyframes[index_b];

    float time_diff = kf_b.time - kf_a.time;
    if (time_diff <= 0.0f) {
        return kf_a.value;
    }

    float t = (time - kf_a.time) / time_diff;
    t = CLAMP(t, 0.0f, 1.0f);

    // Handle different value types
    Variant::Type type = kf_a.value.get_type();
    Variant::Type other_type = kf_b.value.get_type();
    const bool numeric_pair = (type == Variant::FLOAT || type == Variant::INT) &&
            (other_type == Variant::FLOAT || other_type == Variant::INT);
    if (!numeric_pair && type != other_type) {
        // Type mismatch, return first keyframe.
        return kf_a.value;
    }

    InterpolationType interp_type = kf_a.interpolation;

    if (numeric_pair) {
        float a = kf_a.value.operator float();
        float b = kf_b.value.operator float();
        return _interpolate_float(a, b, t, interp_type, kf_a.out_handle, kf_b.in_handle);
    }

    switch (type) {
        case Variant::FLOAT: {
            float a = kf_a.value.operator float();
            float b = kf_b.value.operator float();
            return _interpolate_float(a, b, t, interp_type, kf_a.out_handle, kf_b.in_handle);
        }
        case Variant::INT: {
            float a = kf_a.value.operator float();
            float b = kf_b.value.operator float();
            return _interpolate_float(a, b, t, interp_type, kf_a.out_handle, kf_b.in_handle);
        }
        case Variant::VECTOR3: {
            Vector3 a = kf_a.value.operator Vector3();
            Vector3 b = kf_b.value.operator Vector3();
            return _interpolate_vector3(a, b, t, interp_type, kf_a.out_handle, kf_b.in_handle);
        }
        case Variant::COLOR: {
            Color a = kf_a.value.operator Color();
            Color b = kf_b.value.operator Color();
            return _interpolate_color(a, b, t, interp_type);
        }
        case Variant::QUATERNION: {
            Quaternion a = kf_a.value.operator Quaternion();
            Quaternion b = kf_b.value.operator Quaternion();
            return _interpolate_quaternion(a, b, t, interp_type);
        }
        default:
            // Unsupported type, return constant
            return kf_a.value;
    }
}

Vector3 KeyframeInterpolator::_interpolate_vector3(const Vector3& a, const Vector3& b, float t, InterpolationType type, const Vector2& handle_a, const Vector2& handle_b) {
    switch (type) {
        case InterpolationType::CONSTANT:
            return a;
        case InterpolationType::LINEAR:
            return a.lerp(b, t);
        case InterpolationType::CUBIC_BEZIER:
            return _cubic_bezier_vector3(t, a, b, handle_a, handle_b);
        case InterpolationType::SMOOTH_STEP:
            return a.lerp(b, smooth_step(t));
        case InterpolationType::SMOOTHER_STEP:
            return a.lerp(b, smoother_step(t));
        default:
            return a.lerp(b, t);
    }
}

Color KeyframeInterpolator::_interpolate_color(const Color& a, const Color& b, float t, InterpolationType type) {
    switch (type) {
        case InterpolationType::CONSTANT:
            return a;
        case InterpolationType::LINEAR:
            return a.lerp(b, t);
        case InterpolationType::SMOOTH_STEP:
            return a.lerp(b, smooth_step(t));
        case InterpolationType::SMOOTHER_STEP:
            return a.lerp(b, smoother_step(t));
        default:
            return a.lerp(b, t);
    }
}

Quaternion KeyframeInterpolator::_interpolate_quaternion(const Quaternion& a, const Quaternion& b, float t, InterpolationType type) {
    switch (type) {
        case InterpolationType::CONSTANT:
            return a;
        case InterpolationType::LINEAR:
            return a.slerp(b, t);
        case InterpolationType::SMOOTH_STEP:
            return a.slerp(b, smooth_step(t));
        case InterpolationType::SMOOTHER_STEP:
            return a.slerp(b, smoother_step(t));
        default:
            return a.slerp(b, t);
    }
}

float KeyframeInterpolator::_interpolate_float(float a, float b, float t, InterpolationType type, const Vector2& handle_a, const Vector2& handle_b) {
    switch (type) {
        case InterpolationType::CONSTANT:
            return a;
        case InterpolationType::LINEAR:
            return Math::lerp(a, b, t);
        case InterpolationType::CUBIC_BEZIER:
            return a + (b - a) * _cubic_bezier(t, handle_a, handle_b);
        case InterpolationType::SMOOTH_STEP:
            return Math::lerp(a, b, smooth_step(t));
        case InterpolationType::SMOOTHER_STEP:
            return Math::lerp(a, b, smoother_step(t));
        default:
            return Math::lerp(a, b, t);
    }
}

float KeyframeInterpolator::_cubic_bezier(float t, const Vector2& p1, const Vector2& p2) {
    // Cubic Bezier with control points p1 and p2
    // Assume start point (0,0) and end point (1,1)
    float u = 1.0f - t;
    float tt = t * t;
    float uu = u * u;
    float uuu = uu * u;
    float ttt = tt * t;

    // Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    return 3 * uu * t * p1.y + 3 * u * tt * p2.y + ttt;
}

Vector3 KeyframeInterpolator::_cubic_bezier_vector3(float t, const Vector3& start, const Vector3& end, const Vector2& handle_a, const Vector2& handle_b) {
    Vector3 result;
    result.x = start.x + (end.x - start.x) * _cubic_bezier(t, handle_a, handle_b);
    result.y = start.y + (end.y - start.y) * _cubic_bezier(t, handle_a, handle_b);
    result.z = start.z + (end.z - start.z) * _cubic_bezier(t, handle_a, handle_b);
    return result;
}

void KeyframeInterpolator::_find_keyframe_indices(const LocalVector<Keyframe>& keyframes, float time, int& index_a, int& index_b) {
    int size = keyframes.size();

    if (size == 0) {
        index_a = index_b = 0;
        return;
    }

    if (size == 1 || time <= keyframes[0].time) {
        index_a = index_b = 0;
        return;
    }

    if (time >= keyframes[size - 1].time) {
        index_a = index_b = size - 1;
        return;
    }

    // Binary search for efficiency
    int left = 0;
    int right = size - 1;

    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (keyframes[mid].time <= time) {
            left = mid;
        } else {
            right = mid;
        }
    }

    index_a = left;
    index_b = right;
}

float KeyframeInterpolator::smooth_step(float t) {
    return t * t * (3.0f - 2.0f * t);
}

float KeyframeInterpolator::smoother_step(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

int KeyframeInterpolator::add_keyframe_sorted(LocalVector<Keyframe>& keyframes, const Keyframe& keyframe) {
    // Find insertion point
    uint32_t insertion_point = 0;
    for (uint32_t i = 0; i < keyframes.size(); i++) {
        if (keyframes[i].time > keyframe.time) {
            insertion_point = i;
            break;
        }
        insertion_point = i + 1;
    }

    keyframes.insert(insertion_point, keyframe);
    return insertion_point;
}

} // namespace GaussianSplatting
