#[compute]
#version 450

layout(local_size_x = 16, //
		local_size_y = 16, //
		local_size_z = 1) in;

layout(binding = 0) uniform sampler2D ycbcr_sampler;
layout(binding = 1, rgba8ui) uniform writeonly uimage2D rgba_dst;

void main() {
	vec2 srcUV = gl_GlobalInvocationID.xy;
	ivec2 dstUV = ivec2(gl_GlobalInvocationID.xy);

	vec3 color = texture(ycbcr_sampler, srcUV).rgb;
	uvec4 value = uvec4(color, 255u);
	imageStore(rgba_dst, dstUV, value);
}
