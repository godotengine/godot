#[versions]

version_float = "#define VER_FLOAT";
version_half = "#define VER_HALF";
version_unorm8 = "#define VER_UINT8";
version_unorm16 = "#define VER_UINT16";

#[compute]
#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std430, binding = 0) buffer Source {
#if defined(VER_FLOAT)
	float data[];
#else
	uint data[];
#endif
}
source;

#if defined(VER_FLOAT)
layout(binding = 1, rgba32f) uniform writeonly image2D dest;
#elif defined(VER_HALF)
layout(binding = 1, rgba16f) uniform writeonly image2D dest;
#elif defined(VER_UINT8)
layout(binding = 1, rgba8) uniform writeonly image2D dest;
#elif defined(VER_UINT16)
layout(binding = 1, rgba16) uniform writeonly image2D dest;
#endif

layout(push_constant, std430) uniform Params {
	uint p_width;
	uint p_height;
	uint p_padding[2];
}
params;

void main() {
	// gl_GlobalInvocationID is equivalent to the current texel coordinates.
	if (gl_GlobalInvocationID.x >= params.p_width || gl_GlobalInvocationID.y >= params.p_height) {
		return;
	}

	// The index of a texel in the source buffer, NOT an index of source.data[]
	const int texel_index = int(gl_GlobalInvocationID.y * params.p_width + gl_GlobalInvocationID.x);

#if defined(VER_FLOAT)
	// Since 32-bit floats are aligned with RGBF texel data, just retrieve the values from the array.
	// Multiply by 3 to align with the components.

	int data_index = texel_index * 3;
	vec3 color_rgb = vec3(source.data[data_index], source.data[data_index + 1], source.data[data_index + 2]);

#elif defined(VER_UINT8)
	// RGB8 texel data and 32-bit uints are not aligned, so we have to use a bit of magic.
	// The source texel can be in either of 4 alignment 'states':
	// 0 - [ XYZ_-____ ]
	// 1 - [ _YZW-____ ]
	// 2 - [ __ZW-X___ ]
	// 3 - [ ___W-XY__ ]
	// The texel index additionally needs to be decremented after every 'cycle' in order to properly fit into the source array.

	vec3 color_rgb = vec3(0.0);
	int data_index = texel_index - (texel_index / 4);

	switch ((texel_index * 3) % 4) {
		case 0:
			color_rgb = unpackUnorm4x8(source.data[data_index]).xyz;
			break;
		case 1:
			color_rgb = unpackUnorm4x8(source.data[data_index - 1]).yzw;
			break;
		case 2:
			color_rgb.rg = unpackUnorm4x8(source.data[data_index - 1]).zw;
			color_rgb.b = unpackUnorm4x8(source.data[data_index]).x;
			break;
		case 3:
			color_rgb.r = unpackUnorm4x8(source.data[data_index - 1]).w;
			color_rgb.gb = unpackUnorm4x8(source.data[data_index]).xy;
			break;
		default:
			break;
	}

#else
	// In a similar vein to RGB8, the RGBH/RGB16 source texel can be in either of 2 alignment 'states':
	// 0 - [ XY-X_ ]
	// 1 - [ _Y-XY ]
	// The texel index has to be incremented this time, as the size of a texel (6 bytes) is greater than that of a 32-bit uint (4 bytes).

	vec3 color_rgb = vec3(0.0);
	int data_index = texel_index + (texel_index / 2);

	switch ((texel_index * 3) % 2) {
#if defined(VER_HALF)
		case 0:
			color_rgb.xy = unpackHalf2x16(source.data[data_index]);
			color_rgb.z = unpackHalf2x16(source.data[data_index + 1]).x;
			break;
		case 1:
			color_rgb.x = unpackHalf2x16(source.data[data_index]).y;
			color_rgb.yz = unpackHalf2x16(source.data[data_index + 1]);
			break;
#elif defined(VER_UINT16)
		case 0:
			color_rgb.xy = unpackUnorm2x16(source.data[data_index]);
			color_rgb.z = unpackUnorm2x16(source.data[data_index + 1]).x;
			break;
		case 1:
			color_rgb.x = unpackUnorm2x16(source.data[data_index]).y;
			color_rgb.yz = unpackUnorm2x16(source.data[data_index + 1]);
			break;
#endif
		default:
			break;
	}
#endif

	// Store the resulting RGBA color.
	imageStore(dest, ivec2(gl_GlobalInvocationID.xy), vec4(color_rgb, 1.0));
}
