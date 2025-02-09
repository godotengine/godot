/* clang-format off */

// TODO: Correct this file for Renderer_RD

#[modes]

mode_default =

#[specializations]

USE_OUTPUT2 = true
USE_OUTPUT3 = true
USE_OUTPUT4 = true

#[vertex]

layout(location = 0) in vec2 vertex_attrib;

uniform vec2 offset;
uniform vec2 size;

out vec2 uv;

void main() {
	uv = vertex_attrib * 0.5 + 0.5;
	// This math scales the Vertex Attribute Quad to match the Rect the user passed in, based on Offset & Size
	gl_Position = vec4( (offset * 2 - 1) + (size * (vertex_attrib + 1)), 1.0, 1.0);
}

#[fragment]

#GLOBALS

uniform sampler2D source; // texunit:0

uniform sampler2D source2; // texunit:1

uniform sampler2D source3; // texunit:2

uniform sampler2D source4; // texunit:3

uniform bool convert_to_srgb;
uniform vec4 modulate;

in vec2 uv;

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

	if (convert_to_srgb) {
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
