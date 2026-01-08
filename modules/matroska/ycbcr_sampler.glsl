#[compute]
#version 450

layout(local_size_x = 1, //
		local_size_y = 1, //
		local_size_z = 1) in;

layout(binding = 0) uniform sampler2D src_yuv;
layout(binding = 1, rgba8) uniform writeonly image2D dst_rgba;

void main() {
	ivec2 dimensions = textureSize(src_yuv, 0);
	vec2 uv = gl_GlobalInvocationID.xy / vec2(dimensions);
	vec4 pixel = textureLod(src_yuv, uv, 0);
	imageStore(dst_rgba, ivec2(gl_GlobalInvocationID.xy), pixel);
}
