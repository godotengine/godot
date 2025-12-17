#[compute]
#version 450

layout(local_size_x = 1, //
		local_size_y = 1, //
		local_size_z = 1) in;

layout(binding = 0) uniform sampler2D src_yuv;
layout(binding = 1, rgba8) uniform writeonly image2D dst_rgba;

void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	vec4 pixel = texelFetch(src_yuv, uv, 0);
	imageStore(dst_rgba, uv, pixel);
}
