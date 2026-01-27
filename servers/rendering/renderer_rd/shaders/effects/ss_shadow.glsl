#[compute]

#version 450

#VERSION_DEFINES

#define MAX_VIEWS 2

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D depth_tex;

layout(set = 0, binding = 1, std140) uniform CameraData {
	mat4 inv_view_projection[MAX_VIEWS];
	mat4 projection[MAX_VIEWS];
	vec4 light_direction[MAX_VIEWS];
}
camera;

layout(push_constant, std430) uniform Params {
	ivec2 resolution;
	int view_index;
	int sample_count;
	float max_distance;
	float thickness;
	float intensity;
}
params;

layout(set = 1, binding = 0, r8) uniform writeonly image2D shadow_image;

vec3 reconstruct_position(vec2 uv, float depth) {
	vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
	vec4 position = camera.inv_view_projection[params.view_index] * ndc;
	return position.xyz / position.w;
}

vec2 project_position(vec3 view_pos) {
	vec4 clip = camera.projection[params.view_index] * vec4(view_pos, 1.0);
	float inv_w = 1.0 / clip.w;
	vec2 ndc = clip.xy * inv_w;
	return ndc * 0.5 + 0.5;
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	imageStore(shadow_image, pixel, vec4(0.0, 0.0, 0.0, 0.0));
}
