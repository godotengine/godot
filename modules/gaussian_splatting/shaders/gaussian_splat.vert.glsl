#[vertex]

#version 450
#extension GL_GOOGLE_include_directive : enable

#include "includes/painterly_common.glsl"

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 splat_position;
layout(location = 2) in vec4 splat_color;
layout(location = 3) in vec3 splat_scale;
layout(location = 4) in vec4 splat_rotation;

layout(set = 0, binding = 0, std140) uniform Matrices {
    mat4 view_matrix;
    mat4 projection_matrix;
    mat4 view_projection_matrix;
    vec3 camera_position;
    float time;
} matrices;

layout(location = 0) out vec2 uv_coord;
layout(location = 1) out vec4 color;
layout(location = 2) out vec3 conic;
layout(location = 3) out float opacity;
layout(location = 4) out vec3 painterly_view_dir;
layout(location = 5) out vec3 painterly_normal_vs;
layout(location = 6) out vec2 stylization_seed;

void main() {
    vec4 view_pos = matrices.view_matrix * vec4(splat_position, 1.0);

    if (view_pos.z > 0.0) {
        gl_Position = vec4(0.0);
        return;
    }

    mat3 rotation = painterly_quaternion_to_matrix(splat_rotation);
    mat3 covariance_3d = painterly_build_covariance(rotation, splat_scale);
    mat3 view_basis = mat3(matrices.view_matrix);
    PainterlyConicData projection = painterly_project_gaussian(view_basis, covariance_3d);

    if (projection.determinant <= 0.0) {
        gl_Position = vec4(0.0);
        return;
    }

    conic = projection.conic;

    float radius = painterly_compute_radius(projection, 3.0);

    vec4 clip_pos = matrices.projection_matrix * view_pos;
    vec2 screen_pos = clip_pos.xy / clip_pos.w;

    float safe_depth = max(abs(view_pos.z), 0.0001);
    vec2 quad_offset = vertex_position.xy * radius / safe_depth;
    gl_Position = vec4(screen_pos + quad_offset, clip_pos.z / clip_pos.w, 1.0);

    uv_coord = vertex_position.xy;
    color = splat_color;
    opacity = min(1.0, splat_color.a);

    vec3 view_normal = view_basis * (rotation * vec3(0.0, 0.0, 1.0));
    painterly_normal_vs = painterly_safe_normalize(view_normal, vec3(0.0, 0.0, 1.0));
    painterly_view_dir = painterly_safe_normalize(-view_pos.xyz, vec3(0.0, 0.0, 1.0));

    stylization_seed = painterly_hash_vector(splat_position);
}
