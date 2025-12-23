/* clang-format off */

#[vertex]

#version 450

// Has to be same push_constant for vertex & frag
layout(push_constant, std430) uniform TexBlitData {
	vec2 offset;
	vec2 size;
	vec4 modulate;
	vec2 pad;
	int convert_to_srgb;
	float time;
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

#define OUTPUT0_SRGB uint(1)
#define OUTPUT1_SRGB uint(2)
#define OUTPUT2_SRGB uint(4)
#define OUTPUT3_SRGB uint(8)

layout(push_constant, std430) uniform TexBlitData {
	vec2 offset;
	vec2 size;
	vec4 modulate;
	vec2 pad;
	int convert_to_srgb;
	float time;
} data;

// uniform type sampler with image
// Samplers Binding First Index
	// Look at Sky.glsl again
layout(set = 0, binding = 0) uniform texture2D source0;

layout(set = 0, binding = 1) uniform texture2D source1;

layout(set = 0, binding = 2) uniform texture2D source2;

layout(set = 0, binding = 3) uniform texture2D source3;

layout(location = 0) in vec2 uv;

layout (location = 0) out vec4 out_color0;

#ifdef USE_OUTPUT1
layout (location = 1) out vec4 out_color1;
#endif

#ifdef USE_OUTPUT2
layout (location = 2) out vec4 out_color2;
#endif

#ifdef USE_OUTPUT3
layout (location = 3) out vec4 out_color3;
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
	vec4 color0 = vec4(0.0, 0.0, 0.0, 1.0);
	vec4 color1 = vec4(0.0, 0.0, 0.0, 1.0);
	vec4 color2 = vec4(0.0, 0.0, 0.0, 1.0);
	vec4 color3 = vec4(0.0, 0.0, 0.0, 1.0);

#CODE : BLIT

	// Discards extra outputs if extra output targets were not bound
	out_color0 = color0;

#ifdef USE_OUTPUT1
	out_color1 = color1;
#endif
#ifdef USE_OUTPUT2
	out_color2 = color2;
#endif
#ifdef USE_OUTPUT3
	out_color3 = color3;
#endif

	if (bool(data.convert_to_srgb & OUTPUT0_SRGB)) {
		out_color0.rgb = linear_to_srgb(out_color0.rgb); // Regular linear -> SRGB conversion.
	}
#ifdef USE_OUTPUT1
	if (bool(data.convert_to_srgb & OUTPUT1_SRGB)) {
		out_color1.rgb = linear_to_srgb(out_color1.rgb);
	}
#endif
#ifdef USE_OUTPUT2
	if (bool(data.convert_to_srgb & OUTPUT2_SRGB)) {
		out_color2.rgb = linear_to_srgb(out_color2.rgb);
	}
#endif
#ifdef USE_OUTPUT3
	if (bool(data.convert_to_srgb & OUTPUT3_SRGB)) {
		out_color3.rgb = linear_to_srgb(out_color3.rgb);
	}
#endif
}
