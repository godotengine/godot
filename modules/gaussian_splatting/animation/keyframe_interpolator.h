#ifndef GAUSSIAN_KEYFRAME_INTERPOLATOR_H
#define GAUSSIAN_KEYFRAME_INTERPOLATOR_H

#include "core/variant/variant.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/color.h"
#include "core/math/quaternion.h"
#include "core/templates/local_vector.h"

namespace GaussianSplatting {

enum class InterpolationType {
    CONSTANT,
    LINEAR,
    CUBIC_BEZIER,
    SMOOTH_STEP,
    SMOOTHER_STEP
};

struct Keyframe {
    float time;
    Variant value;
    InterpolationType interpolation = InterpolationType::LINEAR;

    // Bezier control points for cubic interpolation
    Vector2 in_handle = Vector2(0, 0);
    Vector2 out_handle = Vector2(0, 0);

    Keyframe() = default;
    Keyframe(float p_time, const Variant& p_value, InterpolationType p_interp = InterpolationType::LINEAR)
        : time(p_time), value(p_value), interpolation(p_interp) {}

    bool operator<(const Keyframe& other) const {
        return time < other.time;
    }
};

class KeyframeInterpolator {
private:
    // Helper methods for different data types
    static Vector3 _interpolate_vector3(const Vector3& a, const Vector3& b, float t, InterpolationType type, const Vector2& handle_a = Vector2(), const Vector2& handle_b = Vector2());
    static Color _interpolate_color(const Color& a, const Color& b, float t, InterpolationType type);
    static Quaternion _interpolate_quaternion(const Quaternion& a, const Quaternion& b, float t, InterpolationType type);
    static float _interpolate_float(float a, float b, float t, InterpolationType type, const Vector2& handle_a = Vector2(), const Vector2& handle_b = Vector2());

    // Cubic Bezier interpolation
    static float _cubic_bezier(float t, const Vector2& p1, const Vector2& p2);
    static Vector3 _cubic_bezier_vector3(float t, const Vector3& start, const Vector3& end, const Vector2& handle_a, const Vector2& handle_b);

    // Find keyframe indices for interpolation
    static void _find_keyframe_indices(const LocalVector<Keyframe>& keyframes, float time, int& index_a, int& index_b);

public:
    KeyframeInterpolator() = default;

    // Main interpolation method
    Variant interpolate(const LocalVector<Keyframe>& keyframes, float time) const;

    // Utility methods
    static float smooth_step(float t);
    static float smoother_step(float t);

    // Keyframe management helpers
    static int add_keyframe_sorted(LocalVector<Keyframe>& keyframes, const Keyframe& keyframe);
};

} // namespace GaussianSplatting

#endif // GAUSSIAN_KEYFRAME_INTERPOLATOR_H
