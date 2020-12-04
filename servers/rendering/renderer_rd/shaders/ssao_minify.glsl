#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant, binding = 1, std430) uniform Params {
	vec2 pixel_size;
	float z_far;
	float z_near;
	ivec2 source_size;
	bool orthogonal;
	uint pad;
}
params;

#ifdef MINIFY_START
layout(set = 0, binding = 0) uniform sampler2D source_texture;
#else
layout(r32f, set = 0, binding = 0) uniform restrict readonly image2D source_image;
#endif
layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D dest_image;

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThan(pos, params.source_size >> 1))) { //too large, do nothing
		return;
	}

#ifdef MINIFY_START
	float depth = texelFetch(source_texture, pos << 1, 0).r * 2.0 - 1.0;
	if (params.orthogonal) {
		depth = ((depth + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / 2.0;
	} else {
		depth = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - depth * (params.z_far - params.z_near));
	}
#else
	float depth = imageLoad(source_image, pos << 1).r;
#endif

	imageStore(dest_image, pos, vec4(depth));
}
