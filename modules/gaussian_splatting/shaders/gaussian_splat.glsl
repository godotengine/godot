#[vertex]

#version 450
#extension GL_GOOGLE_include_directive : enable

#include "includes/gaussian_splat_common_inc.glsl"

layout(location = 0) flat out uint v_gaussian_index;
layout(location = 1) flat out vec3 v_conic;
layout(location = 2) flat out vec2 v_screen_center;
layout(location = 3) flat out float v_opacity;
layout(location = 4) out vec2 v_ndc_offset;
layout(location = 5) flat out vec3 v_view_dir;

void main() {
    uint gaussian_index = get_gaussian_count() > 0u ? uint(gl_InstanceIndex) : uint(gl_VertexIndex) / 6u;
    if (gaussian_index >= get_gaussian_count()) {
        gl_Position = vec4(0.0);
        return;
    }

    vec4 pos_opacity = splat_positions.position_opacity[gaussian_index];
    vec3 world_pos = pos_opacity.xyz;
    float opacity = pos_opacity.w;

    vec3 scale = splat_scales.scale_data[gaussian_index].xyz;
    vec4 rotation = splat_rotations.rotation_data[gaussian_index];

    vec4 view_pos4 = scene_data.view_matrix * vec4(world_pos, 1.0);
    vec3 view_pos = view_pos4.xyz;

    float near_plane = scene_data.viewport.z;
    float far_plane = scene_data.viewport.w;
    if (view_pos.z > -near_plane || view_pos.z < -far_plane) {
        gl_Position = vec4(0.0);
        return;
    }

    vec4 clip_pos = scene_data.projection_matrix * view_pos4;
    if (abs(clip_pos.w) < 1e-6) {
        gl_Position = vec4(0.0);
        return;
    }

    vec2 viewport_size = scene_data.viewport.xy;
    mat2 cov2d = compute_projected_covariance(view_pos, scale, rotation, viewport_size);
    vec3 conic = covariance_to_conic(cov2d);

    EigenBasis basis = compute_eigen(cov2d);
    float sigma = get_sigma_multiplier();
    vec2 axis0 = basis.axis0 * basis.radius0 * sigma;
    vec2 axis1 = basis.axis1 * basis.radius1 * sigma;

    int vertex_id = gl_VertexIndex % 6;
    vec2 corner = QUAD_CORNERS[vertex_id];
    vec2 offset_pixels = axis0 * corner.x + axis1 * corner.y;
    vec2 offset_ndc = offset_pixels * (2.0 / viewport_size);

    vec2 ndc = clip_pos.xy / clip_pos.w;
    vec2 projected = ndc + offset_ndc;

    gl_Position = vec4(projected, clip_pos.z / clip_pos.w, 1.0);

    vec2 screen_center = (ndc * 0.5 + 0.5) * viewport_size;

    v_gaussian_index = gaussian_index;
    v_conic = conic;
    v_screen_center = screen_center;
    v_opacity = opacity;
    v_ndc_offset = offset_pixels;
    v_view_dir = normalize(scene_data.camera_position.xyz - world_pos);
}

#[fragment]

#version 450
#extension GL_GOOGLE_include_directive : enable

#include "includes/gaussian_splat_common_inc.glsl"

layout(location = 0) flat in uint f_gaussian_index;
layout(location = 1) flat in vec3 f_conic;
layout(location = 2) flat in vec2 f_screen_center;
layout(location = 3) flat in float f_opacity;
layout(location = 4) in vec2 f_ndc_offset;
layout(location = 5) flat in vec3 f_view_dir;

layout(location = 0) out vec4 frag_color;

void main() {
    vec2 pixel_pos = gl_FragCoord.xy;
    vec2 delta = pixel_pos - f_screen_center;

    float power = -0.5 * (f_conic.x * delta.x * delta.x + f_conic.z * delta.y * delta.y) - f_conic.y * delta.x * delta.y;
    if (power > 0.0) {
        discard;
    }

    float alpha = f_opacity * exp(power);
    if (alpha <= (1.0 / 255.0)) {
        discard;
    }

    // Use dithered SH evaluation to reduce RGB9E5 quantization banding artifacts
    vec3 color = evaluate_sh_color_dithered(f_gaussian_index, f_view_dir, 3u, gl_FragCoord.xy);

    frag_color = vec4(clamp(color, vec3(0.0), vec3(1.0)), clamp(alpha, 0.0, 1.0));
}

