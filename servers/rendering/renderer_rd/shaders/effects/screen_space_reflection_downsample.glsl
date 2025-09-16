#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_depth;
layout(set = 0, binding = 1) uniform sampler2D source_normal_roughness;
layout(r32f, set = 0, binding = 2) uniform restrict writeonly image2D dest_depth;
layout(rgba8, set = 0, binding = 3) uniform restrict writeonly image2D dest_normal_roughness;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
}
params;

void get_sample(ivec2 sample_pos, inout float depth, inout ivec2 winner_sample_pos) {
	float sample_depth = texelFetch(source_depth, sample_pos, 0).x;

	if (depth < sample_depth) {
		depth = sample_depth;
		winner_sample_pos = sample_pos;
	}
}

void main() {
	ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(pixel_pos, params.screen_size))) {
		return;
	}

	ivec2 sample_pos = pixel_pos * 2 + ivec2(0, 0);
	float depth = texelFetch(source_depth, sample_pos, 0).x;

	get_sample(pixel_pos * 2 + ivec2(1, 0), depth, sample_pos);
	get_sample(pixel_pos * 2 + ivec2(0, 1), depth, sample_pos);
	get_sample(pixel_pos * 2 + ivec2(1, 1), depth, sample_pos);

#ifdef MODE_ODD_WIDTH
	get_sample(pixel_pos * 2 + ivec2(2, 0), depth, sample_pos);
	get_sample(pixel_pos * 2 + ivec2(2, 1), depth, sample_pos);
#endif

#ifdef MODE_ODD_HEIGHT
	get_sample(pixel_pos * 2 + ivec2(0, 2), depth, sample_pos);
	get_sample(pixel_pos * 2 + ivec2(1, 2), depth, sample_pos);
#endif

#if defined(MODE_ODD_WIDTH) && defined(MODE_ODD_HEIGHT)
	get_sample(pixel_pos * 2 + ivec2(2, 2), depth, sample_pos);
#endif

	imageStore(dest_depth, pixel_pos, vec4(depth, 0.0, 0.0, 0.0));
	imageStore(dest_normal_roughness, pixel_pos, texelFetch(source_normal_roughness, sample_pos, 0));
}
