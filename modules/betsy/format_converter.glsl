#[compute]

#version 450

layout(binding = 0) uniform restrict readonly image2D srcTexture;
layout(rg32ui, binding = 1) uniform restrict writeonly image2D dstTexture;

layout(local_size_x = 8, //
		local_size_y = 8, //
		local_size_z = 1) in;

void main() {
	vec4 src = imageLoad(srcTexture, gl_GlobalInvocationID.xy);
	imageStore(dstTexture, gl_GlobalInvocationID.xy, src);
}
