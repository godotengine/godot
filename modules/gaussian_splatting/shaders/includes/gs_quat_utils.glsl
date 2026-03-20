// gs_quat_utils.glsl — Quaternion utility functions for GPU shaders.
//
// quaternion_to_matrix() does NOT normalize its input, unlike the version in
// gaussian_splat_common_inc.glsl.  Callers must provide a unit-length (or
// pre-corrected) quaternion.  This avoids a redundant normalize() when the
// caller has already applied fast Newton-Raphson correction (PERF-5, #676).

#ifndef GS_QUAT_UTILS_GLSL_INCLUDED
#define GS_QUAT_UTILS_GLSL_INCLUDED

// Convert a quaternion to a rotation matrix.
mat3 quaternion_to_matrix(vec4 q) {
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;

    // Construct columns explicitly (GLSL matrices are column-major)
    vec3 col0 = vec3(
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy + wz),
        2.0 * (xz - wy));
    vec3 col1 = vec3(
        2.0 * (xy - wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz + wx));
    vec3 col2 = vec3(
        2.0 * (xz + wy),
        2.0 * (yz - wx),
        1.0 - 2.0 * (xx + yy));

    return mat3(col0, col1, col2);
}

// Rotate a vector by a quaternion.
vec3 gs_quat_rotate(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Multiply two quaternions.
vec4 gs_quat_mul(vec4 a, vec4 b) {
    return vec4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

#endif // GS_QUAT_UTILS_GLSL_INCLUDED
