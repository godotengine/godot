#ifndef PAINTERLY_COMMON_GLSL
#define PAINTERLY_COMMON_GLSL

const float PAINTERLY_MIN_VARIANCE = 0.0001;
const float PAINTERLY_VARIANCE_EPSILON = 0.3;
const float PAINTERLY_MIN_DETERMINANT = 1e-8;

struct PainterlyConicData {
    vec3 conic;
    float cov_xx;
    float cov_yy;
    float cov_xy;
    float determinant;
};

// Convert a quaternion rotation to a 3x3 rotation matrix.
mat3 painterly_quaternion_to_matrix(vec4 q) {
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;

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

// Build a diagonal scale matrix from per-axis scale values.
mat3 painterly_scale_matrix(vec3 scale) {
    return mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
}

// Build a covariance matrix from rotation and scale in 3D space.
mat3 painterly_build_covariance(mat3 rotation_matrix, vec3 scale) {
    vec3 scale_sq = scale * scale;
    mat3 scale_matrix = painterly_scale_matrix(scale_sq);
    return rotation_matrix * scale_matrix * transpose(rotation_matrix);
}

// Build a covariance matrix from scale and quaternion inputs.
mat3 painterly_build_covariance(vec3 scale, vec4 rotation) {
    mat3 rotation_matrix = painterly_quaternion_to_matrix(rotation);
    return painterly_build_covariance(rotation_matrix, scale);
}

// Project a 3D covariance into view space and derive the 2D conic form.
PainterlyConicData painterly_project_gaussian(mat3 view_matrix, mat3 covariance_3d) {
    PainterlyConicData result;

    mat3 covariance_2d = view_matrix * covariance_3d * transpose(view_matrix);

    float cov_xx = covariance_2d[0][0];
    float cov_xy = covariance_2d[0][1];
    float cov_yy = covariance_2d[1][1];

    cov_xx += PAINTERLY_VARIANCE_EPSILON;
    cov_yy += PAINTERLY_VARIANCE_EPSILON;

    cov_xx = max(cov_xx, PAINTERLY_MIN_VARIANCE);
    cov_yy = max(cov_yy, PAINTERLY_MIN_VARIANCE);

    float det = cov_xx * cov_yy - cov_xy * cov_xy;
    result.determinant = det;

    det = max(det, PAINTERLY_MIN_DETERMINANT);

    float det_inv = 1.0 / det;
    result.conic = vec3(cov_yy * det_inv, -cov_xy * det_inv, cov_xx * det_inv);
    result.cov_xx = cov_xx;
    result.cov_yy = cov_yy;
    result.cov_xy = cov_xy;

    return result;
}

// Compute a screen-space radius from projected covariance.
float painterly_compute_radius(const PainterlyConicData data, float sigma_multiplier) {
    float variance = max(data.cov_xx, data.cov_yy);
    variance = max(variance, PAINTERLY_MIN_VARIANCE);
    return sigma_multiplier * sqrt(variance);
}

// Evaluate the quadratic form for a projected Gaussian at a pixel offset.
float painterly_gaussian_power(vec2 uv, vec3 conic) {
    float dx = uv.x;
    float dy = uv.y;
    return -0.5 * (conic.x * dx * dx + conic.z * dy * dy) - conic.y * dx * dy;
}

// Convert Gaussian power into a clamped alpha contribution.
float painterly_gaussian_alpha(float opacity, float power) {
    float alpha = opacity * exp(power);
    return clamp(alpha, 0.0, 0.99);
}

// Hash a 3D value to a stable scalar in [0, 1).
float painterly_hash_scalar(vec3 value) {
    return fract(sin(dot(value, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}

// Hash a 3D value to a stable 2D seed vector.
vec2 painterly_hash_vector(vec3 value) {
    float h1 = painterly_hash_scalar(value);
    float h2 = painterly_hash_scalar(value + vec3(19.19, 93.43, 42.43));
    return vec2(h1, h2);
}

// Normalize a vector with a fallback for near-zero length inputs.
vec3 painterly_safe_normalize(vec3 v, vec3 fallback) {
    float len_sq = dot(v, v);
    if (len_sq < 1e-6) {
        return fallback;
    }
    return v * inversesqrt(len_sq);
}

#endif // PAINTERLY_COMMON_GLSL
