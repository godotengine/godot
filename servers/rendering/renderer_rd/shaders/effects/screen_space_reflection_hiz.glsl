#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source;
layout(r32f, set = 0, binding = 1) uniform restrict writeonly image2D dest;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	ivec2 pad;
}
params;

float load_depth(ivec2 p_position) {
	ivec2 source_size = textureSize(source, 0);
	return texelFetch(source, clamp(p_position, ivec2(0), source_size - ivec2(1)), 0).x;
}

void main() {
	ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(pixel_pos, params.screen_size))) {
		return;
	}

	float depth = load_depth(pixel_pos * 2 + ivec2(0, 0));
	depth = max(depth, load_depth(pixel_pos * 2 + ivec2(1, 0)));
	depth = max(depth, load_depth(pixel_pos * 2 + ivec2(0, 1)));
	depth = max(depth, load_depth(pixel_pos * 2 + ivec2(1, 1)));

#ifdef MODE_ODD_WIDTH
	depth = max(depth, load_depth(pixel_pos * 2 + ivec2(2, 0)));
	depth = max(depth, load_depth(pixel_pos * 2 + ivec2(2, 1)));
#endif

#ifdef MODE_ODD_HEIGHT
	depth = max(depth, load_depth(pixel_pos * 2 + ivec2(0, 2)));
	depth = max(depth, load_depth(pixel_pos * 2 + ivec2(1, 2)));
#endif

#if defined(MODE_ODD_WIDTH) && defined(MODE_ODD_HEIGHT)
	depth = max(depth, load_depth(pixel_pos * 2 + ivec2(2, 2)));
#endif

	imageStore(dest, pixel_pos, vec4(depth, 0.0, 0.0, 0.0));
}
