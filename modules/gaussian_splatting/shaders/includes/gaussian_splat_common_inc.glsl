const float GAUSSIAN_SIGMA_MULTIPLIER = 3.0;
const float MIN_DETERMINANT = 1e-6;
// MIN_VARIANCE ensures eigenvalue clamps yield radius >= MIN_SPLAT_RADIUS/SIGMA
const float MIN_VARIANCE = 0.002;
const float JACOBIAN_EPSILON = 1e-5;
const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2_0 = 1.0925484305920792;
const float SH_C2_1 = 0.31539156525252005;
const float SH_C2_2 = 0.5462742152960396;
const float SH_C3_0 = 0.5900435899266435;
const float SH_C3_1 = 2.890611442640554;
const float SH_C3_2 = 0.4570457994644658;
const float SH_C3_3 = 0.3731763325901154;

// Dithering constants for 8-bit output quantization artifact mitigation
// For R8G8B8A8_UNORM (8-bit per channel), 1 LSB = 1/255
const float DITHER_AMPLITUDE = 1.0 / 255.0;

const uint SH_METADATA_FIRST_ORDER_MASK = 0x000000FFu;
const uint SH_METADATA_HIGH_ORDER_MASK = 0x0000FF00u;
const uint SH_METADATA_ENCODED_COUNT_MASK = 0x00FF0000u;
const uint SH_METADATA_ENCODING_MASK = 0xFF000000u;
const uint SH_ENCODING_RGB9E5 = 1u;

struct EigenBasis {
    vec2 axis0;
    vec2 axis1;
    float radius0;
    float radius1;
};

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view_matrix;
    mat4 projection_matrix;
    mat4 view_projection_matrix;
    vec4 camera_position; // xyz: world camera position, w: time or unused
    vec4 viewport;        // x: width, y: height, z: near plane, w: far plane
    vec4 misc;            // x: sigma multiplier override, y: gaussian count, z,w: unused
} scene_data;

layout(set = 1, binding = 0, std430) readonly buffer PositionBuffer {
    vec4 position_opacity[]; // xyz position, w opacity
} splat_positions;

layout(set = 1, binding = 1, std430) readonly buffer ScaleBuffer {
    vec4 scale_data[]; // xyz scales
} splat_scales;

layout(set = 1, binding = 2, std430) readonly buffer RotationBuffer {
    vec4 rotation_data[]; // xyzw quaternion
} splat_rotations;

layout(set = 1, binding = 3, std430) readonly buffer SHBuffer {
    vec4 sh_data[]; // 4 vec4 entries per gaussian: dc + metadata, followed by encoded coefficients
} splat_sh;

const vec2 QUAD_CORNERS[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0)
);

mat3 quaternion_to_matrix(vec4 q) {
    vec4 n = normalize(q);
    float xx = n.x * n.x;
    float yy = n.y * n.y;
    float zz = n.z * n.z;
    float xy = n.x * n.y;
    float xz = n.x * n.z;
    float yz = n.y * n.z;
    float wx = n.w * n.x;
    float wy = n.w * n.y;
    float wz = n.w * n.z;

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

mat3 build_covariance(vec3 scale, vec4 rotation) {
    mat3 rot = quaternion_to_matrix(rotation);
    vec3 s2 = scale * scale;
    mat3 scale_mat = mat3(
        s2.x, 0.0, 0.0,
        0.0, s2.y, 0.0,
        0.0, 0.0, s2.z
    );
    return rot * scale_mat * transpose(rot);
}

EigenBasis compute_eigen(mat2 cov) {
    EigenBasis basis;
    float trace = cov[0][0] + cov[1][1];
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
    det = max(det, MIN_DETERMINANT);

    float disc = max(trace * trace * 0.25 - det, 0.0);
    float root = sqrt(disc);

    float lambda0 = trace * 0.5 + root;
    float lambda1 = trace * 0.5 - root;

    lambda0 = max(lambda0, MIN_VARIANCE);
    lambda1 = max(lambda1, MIN_VARIANCE);

    vec2 axis0;
    if (abs(cov[0][1]) > JACOBIAN_EPSILON) {
        axis0 = normalize(vec2(lambda0 - cov[1][1], cov[0][1]));
    } else {
        axis0 = vec2(1.0, 0.0);
    }
    vec2 axis1 = vec2(-axis0.y, axis0.x);

    basis.axis0 = axis0;
    basis.axis1 = axis1;
    basis.radius0 = sqrt(lambda0);
    basis.radius1 = sqrt(lambda1);
    return basis;
}

float get_sigma_multiplier() {
    return scene_data.misc.x > 0.0 ? scene_data.misc.x : GAUSSIAN_SIGMA_MULTIPLIER;
}

uint get_gaussian_count() {
    return uint(max(scene_data.misc.y, 0.0));
}

mat2 compute_projected_covariance(vec3 view_pos, vec3 scale, vec4 rotation, vec2 viewport_size) {
    mat3 cov3d = build_covariance(scale, rotation);

    float focal_x = scene_data.projection_matrix[0][0] * viewport_size.x * 0.5;
    float focal_y = scene_data.projection_matrix[1][1] * viewport_size.y * 0.5;

    float z = view_pos.z;
    if (abs(z) < 1e-6) {
        z = z >= 0.0 ? 1e-6 : -1e-6;
    }

    float z_inv = 1.0 / z;
    float z_inv_sq = z_inv * z_inv;

    mat3 J = mat3(
        focal_x * z_inv, 0.0, -focal_x * view_pos.x * z_inv_sq,
        0.0, focal_y * z_inv, -focal_y * view_pos.y * z_inv_sq,
        0.0, 0.0, 0.0
    );

    mat3 cov_proj = J * cov3d * transpose(J);
    mat2 cov2d = mat2(cov_proj[0][0], cov_proj[0][1], cov_proj[1][0], cov_proj[1][1]);
    cov2d[0][0] = max(cov2d[0][0] + 0.3, MIN_VARIANCE);
    cov2d[1][1] = max(cov2d[1][1] + 0.3, MIN_VARIANCE);
    return cov2d;
}

vec3 covariance_to_conic(mat2 cov2d) {
    float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
    det = max(det, MIN_DETERMINANT);
    float inv_det = 1.0 / det;
    return vec3(cov2d[1][1] * inv_det, -cov2d[0][1] * inv_det, cov2d[0][0] * inv_det);
}

uint gaussian_get_first_order_count(uint meta) {
    return meta & SH_METADATA_FIRST_ORDER_MASK;
}

uint gaussian_get_high_order_count(uint meta) {
    return (meta & SH_METADATA_HIGH_ORDER_MASK) >> 8u;
}

uint gaussian_get_encoded_count(uint meta) {
    return (meta & SH_METADATA_ENCODED_COUNT_MASK) >> 16u;
}

uint gaussian_get_sh_encoding(uint meta) {
    return (meta & SH_METADATA_ENCODING_MASK) >> 24u;
}

vec3 decode_rgb9e5(uint packed) {
    uint exponent = (packed >> 27u) & 0x1Fu;
    float scale = exp2(float(exponent) - 24.0);
    vec3 mantissa = vec3(
        float(packed & 0x1FFu),
        float((packed >> 9u) & 0x1FFu),
        float((packed >> 18u) & 0x1FFu)
    );
    return mantissa * scale;
}

// Simple hash-based noise for dithering (spatially varying)
// Uses fragment position to generate pseudo-random value in [-0.5, 0.5]
float dither_noise(vec2 frag_coord) {
    // Simple but effective hash function for dithering
    vec2 p = frag_coord * 0.06711056 + 0.00583715;
    return fract(52.9829189 * fract(dot(p, vec2(12.9898, 78.233)))) - 0.5;
}

// Generate RGB dither noise using different offsets for each channel
// This breaks up color banding from RGB9E5 quantization
vec3 dither_noise_rgb(vec2 frag_coord) {
    return vec3(
        dither_noise(frag_coord),
        dither_noise(frag_coord + vec2(17.0, 23.0)),
        dither_noise(frag_coord + vec2(37.0, 41.0))
    );
}

// -----------------------------------------------------------------
// SH sign convention (ISSUE-038): Condon-Shortley phase included.
//
// This evaluation uses the real spherical harmonics basis with the
// Condon-Shortley (CS) phase factor applied.  Odd-m basis functions
// carry a leading minus sign (e.g. Y_1^{-1} = -C1*y, Y_1^1 = -C1*x).
//
// PLY coefficients from 3DGS training (Kerbl et al. 2023) are stored
// with CS phase already baked in, so they are consumed here without
// any sign adjustment.  The import side (ply_loader.cpp,
// assemble_sh_coefficients) documents the same convention.
// -----------------------------------------------------------------
//
// Compute SH basis functions up to the specified band level
// basis_values[0] = DC (l=0)
// basis_values[1-3] = 1st order (l=1)
// basis_values[4-8] = 2nd order (l=2)
// basis_values[9-15] = 3rd order (l=3)
void compute_sh_basis_with_bands(vec3 dir, uint max_band, out float basis_values[16]) {
    // Initialize all to zero
    for (int i = 0; i < 16; i++) {
        basis_values[i] = 0.0;
    }

    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    // Band 0 (DC) - always computed
    basis_values[0] = SH_C0;

    if (max_band < 1u) return;

    // Band 1 (1st order)
    basis_values[1] = -SH_C1 * y;
    basis_values[2] = SH_C1 * z;
    basis_values[3] = -SH_C1 * x;

    if (max_band < 2u) return;

    // Band 2 (2nd order)
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    basis_values[4] = SH_C2_0 * x * y;
    basis_values[5] = -SH_C2_0 * y * z;
    basis_values[6] = SH_C2_1 * (3.0 * zz - 1.0);
    basis_values[7] = -SH_C2_0 * x * z;
    basis_values[8] = SH_C2_2 * (xx - yy);

    if (max_band < 3u) return;

    // Band 3 (3rd order)
    basis_values[9] = -SH_C3_0 * y * (3.0 * xx - yy);
    basis_values[10] = SH_C3_1 * x * y * z;
    basis_values[11] = -SH_C3_2 * y * (1.0 - 5.0 * zz);
    basis_values[12] = SH_C3_3 * z * (5.0 * zz - 3.0);
    basis_values[13] = -SH_C3_2 * x * (1.0 - 5.0 * zz);
    basis_values[14] = SH_C3_1 * z * (xx - yy);
    basis_values[15] = -SH_C3_0 * x * (xx - 3.0 * yy);
}

// Legacy compute_sh_basis for backwards compatibility (computes all bands)
void compute_sh_basis(vec3 dir, out float basis_values[16]) {
    compute_sh_basis_with_bands(dir, 3u, basis_values);
}

// Evaluate SH color with configurable band level
// sh_band_level: 0=DC only, 1=1st order, 2=2nd order, 3=3rd order
vec3 evaluate_sh_color_with_bands(uint gaussian_index, vec3 dir, uint sh_band_level) {
    const int VEC4_STRIDE = 4;
    int base = int(gaussian_index) * VEC4_STRIDE;

    vec4 dc_meta = splat_sh.sh_data[base];
    vec3 color = dc_meta.rgb;

    // If band level is 0 (DC only), return immediately
    if (sh_band_level == 0u) {
        return color;
    }

    uint metadata = floatBitsToUint(dc_meta.w);

    uint encoding = gaussian_get_sh_encoding(metadata);
    if (encoding != SH_ENCODING_RGB9E5) {
        return color;
    }

    uint first_count = gaussian_get_first_order_count(metadata);
    uint high_count = gaussian_get_high_order_count(metadata);
    uint encoded_total = gaussian_get_encoded_count(metadata);

    // Early exit if no coefficients available
    if (first_count == 0u && high_count == 0u) {
        return color;
    }

    float encoded_coeffs[12];
    for (int i = 0; i < 12; i++) {
        int vec_index = base + 1 + i / 4;
        int component = i % 4;
        encoded_coeffs[i] = splat_sh.sh_data[vec_index][component];
    }

    float basis_values[16];
    compute_sh_basis_with_bands(dir, sh_band_level, basis_values);

    // Add first-order SH terms (if available and band >= 1)
    if (sh_band_level >= 1u) {
        for (uint i = 0u; i < first_count && i < 3u && i < encoded_total; i++) {
            vec3 coeff = decode_rgb9e5(floatBitsToUint(encoded_coeffs[i]));
            float weight = basis_values[1u + i];
            color += coeff * weight;
        }
    }

    // Add higher-order SH terms (if available and band >= 2)
    if (sh_band_level >= 2u) {
        // Determine how many coefficients to use based on band level
        uint coeff_limit = (sh_band_level == 2u) ? 5u : 12u;  // l=2 has 5 coefficients, l=3 adds 7 more

        for (uint i = 0u; i < high_count && (i + first_count) < encoded_total && (i + 4u) < 16u && i < coeff_limit; i++) {
            vec3 coeff = decode_rgb9e5(floatBitsToUint(encoded_coeffs[first_count + i]));
            float weight = basis_values[4u + i];
            color += coeff * weight;
        }
    }

    // Clamp final color to ensure non-negative after SH evaluation
    // SH basis functions can produce negative contributions that shouldn't result in negative color
    color = max(color, vec3(0.0));

    return color;
}

// Legacy evaluate_sh_color for backwards compatibility (uses all available bands)
vec3 evaluate_sh_color(uint gaussian_index, vec3 dir) {
    return evaluate_sh_color_with_bands(gaussian_index, dir, 3u);
}

// Evaluate SH color with dithering to mitigate RGB9E5 quantization banding
// frag_coord: fragment screen position (gl_FragCoord.xy) for spatially-varying dither
// sh_band_level: 0=DC only, 1=1st order, 2=2nd order, 3=3rd order
vec3 evaluate_sh_color_dithered(uint gaussian_index, vec3 dir, uint sh_band_level, vec2 frag_coord) {
    const int VEC4_STRIDE = 4;
    int base = int(gaussian_index) * VEC4_STRIDE;

    vec4 dc_meta = splat_sh.sh_data[base];
    vec3 color = dc_meta.rgb;

    // If band level is 0 (DC only), return immediately (no dithering needed for DC)
    if (sh_band_level == 0u) {
        return color;
    }

    uint metadata = floatBitsToUint(dc_meta.w);

    uint encoding = gaussian_get_sh_encoding(metadata);
    if (encoding != SH_ENCODING_RGB9E5) {
        return color;
    }

    uint first_count = gaussian_get_first_order_count(metadata);
    uint high_count = gaussian_get_high_order_count(metadata);
    uint encoded_total = gaussian_get_encoded_count(metadata);

    // Early exit if no coefficients available
    if (first_count == 0u && high_count == 0u) {
        return color;
    }

    float encoded_coeffs[12];
    for (int i = 0; i < 12; i++) {
        int vec_index = base + 1 + i / 4;
        int component = i % 4;
        encoded_coeffs[i] = splat_sh.sh_data[vec_index][component];
    }

    float basis_values[16];
    compute_sh_basis_with_bands(dir, sh_band_level, basis_values);

    // Add first-order SH terms (if available and band >= 1)
    if (sh_band_level >= 1u) {
        for (uint i = 0u; i < first_count && i < 3u && i < encoded_total; i++) {
            vec3 coeff = decode_rgb9e5(floatBitsToUint(encoded_coeffs[i]));
            float weight = basis_values[1u + i];
            color += coeff * weight;
        }
    }

    // Add higher-order SH terms (if available and band >= 2)
    if (sh_band_level >= 2u) {
        // Determine how many coefficients to use based on band level
        uint coeff_limit = (sh_band_level == 2u) ? 5u : 12u;  // l=2 has 5 coefficients, l=3 adds 7 more

        for (uint i = 0u; i < high_count && (i + first_count) < encoded_total && (i + 4u) < 16u && i < coeff_limit; i++) {
            vec3 coeff = decode_rgb9e5(floatBitsToUint(encoded_coeffs[first_count + i]));
            float weight = basis_values[4u + i];
            color += coeff * weight;
        }
    }

    // Clamp final color to ensure non-negative after SH evaluation
    color = max(color, vec3(0.0));

    // Apply dither to final color to break up 8-bit quantization banding
    vec3 dither = dither_noise_rgb(frag_coord) * DITHER_AMPLITUDE;
    color += dither;

    return color;
}
