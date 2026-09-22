// RGB and Alpha components of ETC2 RGBA/DXT5 are computed separately.
// This compute shader merely stitches them together to form the final result
// It's also used by RG11/BC4 driver to stitch two R11/BC4 into one RG11/BC5

#[compute]
#version 450

layout(local_size_x = 8, //
		local_size_y = 8, //
		local_size_z = 1) in;

layout(binding = 0) uniform usampler2D srcRGB;
layout(binding = 1) uniform usampler2D srcAlpha;
layout(binding = 2, rgba32ui) uniform restrict writeonly uimage2D dstTexture;

void main() {
	uvec2 rgbBlock = texelFetch(srcRGB, ivec2(gl_GlobalInvocationID.xy), 0).xy;
	uvec2 alphaBlock = texelFetch(srcAlpha, ivec2(gl_GlobalInvocationID.xy), 0).xy;

	imageStore(dstTexture, ivec2(gl_GlobalInvocationID.xy), uvec4(rgbBlock.xy, alphaBlock.xy));
}
