#[vertex]
#version 450

#VERSION_DEFINES

layout(set = 0, binding = 0, std430) restrict readonly buffer GlobalShaderUniformData {
	vec4 data[];
}
global_shader_uniforms;

#include "samplers_inc.glsl"

layout(location = 0) in vec3 vertex_attrib;
layout(location = 1) in vec2 uv_attrib;
layout(location = 2) in vec4 color_attrib;

layout(location = 0) out vec2 uv_interp;
layout(location = 1) out vec4 color_interp;

#ifdef MATERIAL_UNIFORMS_USED
/* clang-format off */
layout(set = 1, binding = 0, std140) uniform MaterialUniforms {
#MATERIAL_UNIFORMS
}
material;
/* clang-format on */
#endif

#GLOBALS

void main() {
	vec4 vertex = vec4(vertex_attrib, 1);
	vec2 uv = uv_attrib;
	vec4 color = color_attrib;

	{
#CODE : VERTEX
	}

	uv_interp = uv;
	color_interp = color;
	gl_Position = vertex;
}

#[fragment]
#version 450

#VERSION_DEFINES

layout(set = 0, binding = 0, std430) restrict readonly buffer GlobalShaderUniformData {
	vec4 data[];
}
global_shader_uniforms;

#include "samplers_inc.glsl"

layout(location = 0) in vec2 uv_interp;
layout(location = 1) in vec4 color_interp;

layout(location = 0) out vec4 frag_color;

#ifdef MATERIAL_UNIFORMS_USED
/* clang-format off */
layout(set = 1, binding = 0, std140) uniform MaterialUniforms {
#MATERIAL_UNIFORMS
}
material;
/* clang-format on */
#endif

#GLOBALS

void main() {
	vec2 uv = uv_interp;
	vec4 color = color_interp;

	{
#CODE : FRAGMENT
	}

	frag_color = color;
}
