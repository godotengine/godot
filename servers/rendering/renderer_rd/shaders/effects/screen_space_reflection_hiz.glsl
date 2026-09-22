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

void main() {
	ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(pixel_pos, params.screen_size))) {
		return;
	}

	float depth = texelFetch(source, pixel_pos * 2 + ivec2(0, 0), 0).x;
	depth = max(depth, texelFetch(source, pixel_pos * 2 + ivec2(1, 0), 0).x);
	depth = max(depth, texelFetch(source, pixel_pos * 2 + ivec2(0, 1), 0).x);
	depth = max(depth, texelFetch(source, pixel_pos * 2 + ivec2(1, 1), 0).x);

#ifdef MODE_ODD_WIDTH
	depth = max(depth, texelFetch(source, pixel_pos * 2 + ivec2(2, 0), 0).x);
	depth = max(depth, texelFetch(source, pixel_pos * 2 + ivec2(2, 1), 0).x);
#endif

#ifdef MODE_ODD_HEIGHT
	depth = max(depth, texelFetch(source, pixel_pos * 2 + ivec2(0, 2), 0).x);
	depth = max(depth, texelFetch(source, pixel_pos * 2 + ivec2(1, 2), 0).x);
#endif

#if defined(MODE_ODD_WIDTH) && defined(MODE_ODD_HEIGHT)
	depth = max(depth, texelFetch(source, pixel_pos * 2 + ivec2(2, 2), 0).x);
#endif

	imageStore(dest, pixel_pos, vec4(depth, 0.0, 0.0, 0.0));
}
