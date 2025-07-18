/* clang-format off */

// TODO: Correct this file for Renderer_RD

#[vertex]

#version 450

// Has to be same push_constant for vertex & frag
layout(push_constant, std430) uniform TexBlitData {
	vec2 offset;
	vec2 size;
	vec4 modulate;
	vec3 pad;
	bool convert_to_srgb;
} data;

layout(location = 0) out vec2 uv;

void main() {
	vec2 base_arr[6] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0), vec2(1.0, 0.0), vec2(1.0, 1.0));
	uv = base_arr[gl_VertexIndex];
	// gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);

	gl_Position = vec4( (data.offset + (uv * data.size)) * 2.0 - 1.0, 1.0, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#include "samplers_inc.glsl"

layout(push_constant, std430) uniform TexBlitData {
	vec2 offset;
	vec2 size;
	vec4 modulate;
	vec3 pad;
	bool convert_to_srgb;
} data;

// uniform type sampler with image
// Samplers Binding First Index
	// Look at Sky.glsl again
layout(set = 0, binding = 0) uniform texture2D source;

layout(set = 0, binding = 1) uniform texture2D source2;

layout(set = 0, binding = 2) uniform texture2D source3;

layout(set = 0, binding = 3) uniform texture2D source4;

layout(location = 0) in vec2 uv;

layout (location = 0) out vec4 out_color;

#ifdef USE_OUTPUT2
layout (location = 1) out vec4 out_color2;
#endif

#ifdef USE_OUTPUT3
layout (location = 2) out vec4 out_color3;
#endif

#ifdef USE_OUTPUT4
layout (location = 3) out vec4 out_color4;
#endif

#ifdef MATERIAL_UNIFORMS_USED
layout(set = 1, binding = 0, std140) uniform MaterialUniforms {
#MATERIAL_UNIFORMS
} material;
#endif

#GLOBALS

vec3 linear_to_srgb(vec3 color) {
	// If going to srgb, clamp from 0 to 1.
	color = clamp(color, vec3(0.0), vec3(1.0));
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

void main() {
	// Handles the case where user code uses extra outputs, but extra output targets were not bound
	vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
	vec4 color2 = vec4(0.0, 0.0, 0.0, 1.0);
	vec4 color3 = vec4(0.0, 0.0, 0.0, 1.0);
	vec4 color4 = vec4(0.0, 0.0, 0.0, 1.0);

#CODE : BLIT

	// Discards extra outputs if extra output targets were not bound
	out_color = color;

#ifdef USE_OUTPUT2
	out_color2 = color2;
#endif
#ifdef USE_OUTPUT3
	out_color3 = color3;
#endif
#ifdef USE_OUTPUT4
	out_color4 = color4;
#endif

	if (data.convert_to_srgb) {
		out_color.rgb = linear_to_srgb(out_color.rgb); // Regular linear -> SRGB conversion.
#ifdef USE_OUTPUT2
		out_color2.rgb = linear_to_srgb(out_color2.rgb);
#endif
#ifdef USE_OUTPUT3
		out_color3.rgb = linear_to_srgb(out_color3.rgb);
#endif
#ifdef USE_OUTPUT4
		out_color4.rgb = linear_to_srgb(out_color4.rgb);
#endif
	}
}
