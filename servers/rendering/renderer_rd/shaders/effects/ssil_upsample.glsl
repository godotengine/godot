#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16, set = 0, binding = 0) uniform restrict writeonly image2D dest_image;
layout(set = 0, binding = 1) uniform sampler2D source_ssil;

layout(set = 1, binding = 0) uniform sampler2D depth_buffer;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	ivec2 half_pixel_size;
}
params;

void add_sample(vec2 p_uv, vec4 color_sample, float p_edge, float p_center_depth, inout vec4 sum_color, inout float sum_weight) {

    float diff = clamp(1.0 - (abs(p_edge - p_center_depth)), 0.0, 1.0);

    sum_color += color_sample * diff;
    sum_weight += diff;
}

vec4 edges_LRTB(vec2 p_uv) {
    float L = textureLodOffset(depth_buffer, p_uv, 0.0, ivec2(-2, 0)).r;
    float R = textureLodOffset(depth_buffer, p_uv, 0.0, ivec2(2, 0)).r;
    float T = textureLodOffset(depth_buffer, p_uv, 0.0, ivec2(0, 2)).r;
    float B = textureLodOffset(depth_buffer, p_uv, 0.0, ivec2(0, -2)).r;

    return vec4(L, R, T, B);
}

vec4 bilateral_upsample(vec2 p_uv, float p_linear_depth) {
    vec4 center_color = textureLod(source_ssil, p_uv, 0.0);
    vec4 edges = edges_LRTB(p_uv);

    vec4 sum_color = center_color;
    float sum_weight = 1.0;

    add_sample(p_uv, textureLodOffset(source_ssil, p_uv, 0.0, ivec2(-1, 0)), edges.x, p_linear_depth, sum_color, sum_weight);
    add_sample(p_uv, textureLodOffset(source_ssil, p_uv, 0.0, ivec2(1, 0)), edges.y, p_linear_depth, sum_color, sum_weight);
    add_sample(p_uv, textureLodOffset(source_ssil, p_uv, 0.0, ivec2(0, 1)), edges.z, p_linear_depth, sum_color, sum_weight);
    add_sample(p_uv, textureLodOffset(source_ssil, p_uv, 0.0, ivec2(0, -1)), edges.w, p_linear_depth, sum_color, sum_weight);

    return sum_color / sum_weight;
}

void main() {
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(ssC, params.screen_size))) { //too large, do nothing
		return;
	}

	vec2 uv = (vec2(ssC) + 0.5) / vec2(params.screen_size);
    float depth = textureLod(depth_buffer, uv, 0.0).r;

    vec4 upsampled_ssil = bilateral_upsample(uv, depth);

	imageStore(dest_image, ssC, upsampled_ssil);
}
