#[compute]
#version 450

layout(local_size_x = 16, //
		local_size_y = 16, //
		local_size_z = 1) in;

layout(binding = 0) uniform sampler2D ycbcr_sampler;
layout(binding = 1, rgba8) uniform writeonly image2D rgba_dst;

void main() {
	vec4 pixel = texture(ycbcr_sampler, gl_GlobalInvocationID.xy);
	imageStore(rgba_dst, ivec2(gl_GlobalInvocationID.xy), pixel);
}
