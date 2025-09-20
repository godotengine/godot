#[versions]

version_float = "#define VER_FLOAT";
version_half = "#define VER_HALF";
version_uint8 = "#define VER_UINT8";

#[compute]
#version 450

#VERSION_DEFINES

//#define TEXTURE_BUFFER

#extension GL_EXT_samplerless_texture_functions : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef TEXTURE_BUFFER
layout(binding = 0) uniform textureBuffer source;
#else
layout(binding = 0) uniform texture2DArray source;
#endif

#if defined(VER_FLOAT)
layout(binding = 1, rgba32f) uniform writeonly image2D dest;
#elif defined(VER_HALF)
layout(binding = 1, rgba16f) uniform writeonly image2D dest;
#elif defined(VER_UINT8)
layout(binding = 1, rgba8) uniform writeonly image2D dest;
#endif

layout(push_constant, std430) uniform Params {
	uint p_width;
	uint p_height;
	uint p_padding[2];
}
params;

void main() {
	if (gl_GlobalInvocationID.x >= params.p_width || gl_GlobalInvocationID.y >= params.p_height) {
		return;
	}

	int index = int(gl_GlobalInvocationID.y * params.p_width + gl_GlobalInvocationID.x) * 3;

#ifdef TEXTURE_BUFFER
	vec4 color = vec4(texelFetch(source, index).x, texelFetch(source, index + 1).x, texelFetch(source, index + 2).x, 1.0);
#else
	ivec3 indices = ivec3(index, index + 1, index + 2);
	ivec3 layers = indices / int(params.p_width * params.p_height);
	ivec3 layer_offsets = indices % int(params.p_width * params.p_height);

	ivec3 coords[3] = ivec3[](
			ivec3(layer_offsets.x % int(params.p_width), layer_offsets.x / int(params.p_width), layers.x),
			ivec3(layer_offsets.y % int(params.p_width), layer_offsets.y / int(params.p_width), layers.y),
			ivec3(layer_offsets.z % int(params.p_width), layer_offsets.z / int(params.p_width), layers.z));

	vec4 color = vec4(texelFetch(source, coords[0], 0).x, texelFetch(source, coords[1], 0).x, texelFetch(source, coords[2], 0).x, 1.0);
#endif

	imageStore(dest, ivec2(gl_GlobalInvocationID.xy), color);
}
