#[vertex]

#version 450

#VERSION_DEFINES

#ifdef USE_MULTIVIEW
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#endif //USE_MULTIVIEW

#ifdef USE_MULTIVIEW
layout(location = 0) out vec3 uv_interp;
#else
layout(location = 0) out vec2 uv_interp;
#endif

layout(push_constant, std430) uniform Params {
	float max_texel_factor;
	float res1;
	float res2;
	float res3;
}
params;

void main() {
	vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	uv_interp.xy = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
#ifdef USE_MULTIVIEW
	uv_interp.z = ViewIndex;
#endif
}

#[fragment]

#version 450

#VERSION_DEFINES

#ifdef USE_MULTIVIEW
layout(location = 0) in vec3 uv_interp;
layout(set = 0, binding = 0) uniform sampler2DArray source_color;
#else /* USE_MULTIVIEW */
layout(location = 0) in vec2 uv_interp;
layout(set = 0, binding = 0) uniform sampler2D source_color;
#endif /* USE_MULTIVIEW */

#ifdef SPLIT_RG
layout(location = 0) out vec2 frag_color;
#else
layout(location = 0) out uint frag_color;
#endif

layout(push_constant, std430) uniform Params {
	float max_texel_factor;
	float res1;
	float res2;
	float res3;
}
params;

void main() {
#ifdef USE_MULTIVIEW
	vec3 uv = uv_interp;
#else
	vec2 uv = uv_interp;
#endif

	// Input is standardized. R for X, G for Y, 0.0 (0) = 1, 0.33 (85) = 2, 0.66 (170) = 3, 1.0 (255) = 8
	vec4 color = textureLod(source_color, uv, 0.0);

#ifdef SPLIT_RG
	// Density map for VRS according to VK_EXT_fragment_density_map, we can use as is.
	frag_color = max(vec2(1.0f) - color.rg, vec2(1.0f / 255.0f));
#else
	// Output image shading rate image for VRS according to VK_KHR_fragment_shading_rate.
	color.r = clamp(floor(color.r * params.max_texel_factor + 0.1), 0.0, params.max_texel_factor);
	color.g = clamp(floor(color.g * params.max_texel_factor + 0.1), 0.0, params.max_texel_factor);

	// Note 1x4, 4x1, 1x8, 8x1, 2x8 and 8x2 are not supported:
	if (color.r < (color.g - 1.0)) {
		color.r = color.g - 1.0;
	}
	if (color.g < (color.r - 1.0)) {
		color.g = color.r - 1.0;
	}

	// Encode to frag_color;
	frag_color = int(color.r + 0.1) << 2;
	frag_color += int(color.g + 0.1);
#endif
}
