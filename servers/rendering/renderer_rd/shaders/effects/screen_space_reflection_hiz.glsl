#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source;
layout(r32f, set = 0, binding = 1) uniform restrict writeonly image2D dest;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	bool is_width_odd;
	bool is_height_odd;
}
params;

void main() {
	ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(pixel_pos, params.screen_size))) {
		return;
	}

	float depth_0 = texelFetch(source, pixel_pos * 2 + ivec2(0, 0), 0).x;
	float depth_1 = texelFetch(source, pixel_pos * 2 + ivec2(1, 0), 0).x;
	float depth_2 = texelFetch(source, pixel_pos * 2 + ivec2(0, 1), 0).x;
	float depth_3 = texelFetch(source, pixel_pos * 2 + ivec2(1, 1), 0).x;

	float depth_4 = depth_0;
	float depth_5 = depth_0;
	float depth_6 = depth_0;
	if (params.is_width_odd) {
		depth_4 = texelFetch(source, pixel_pos * 2 + ivec2(2, 0), 0).x;
		depth_5 = texelFetch(source, pixel_pos * 2 + ivec2(2, 1), 0).x;
		if (params.is_height_odd) {
			depth_6 = texelFetch(source, pixel_pos * 2 + ivec2(2, 2), 0).x;
		}
	}

	float depth_7 = depth_0;
	float depth_8 = depth_0;
	if (params.is_height_odd) {
		depth_7 = texelFetch(source, pixel_pos * 2 + ivec2(0, 2), 0).x;
		depth_8 = texelFetch(source, pixel_pos * 2 + ivec2(1, 2), 0).x;
	}

	float depth = max(max(max(max(max(depth_0, depth_1), max(depth_2, depth_3)), max(depth_4, depth_5)), max(depth_6, depth_7)), depth_8);

	imageStore(dest, pixel_pos, vec4(depth, 0.0, 0.0, 0.0));
}
