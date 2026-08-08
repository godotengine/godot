#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

void main() {
	vec2 base_arr[3] = vec2[](vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
	uv_interp = base_arr[gl_VertexIndex];
	gl_Position = vec4(uv_interp, 0.0, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#include "../oct_inc.glsl"

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform samplerCube source_cube;

layout(push_constant, std430) uniform Params {
	float border_size;
	uint pad1;
	uint pad2;
	uint pad3;
}
params;

void main() {
	vec3 dir = oct_to_vec3_with_border(uv_interp * 0.5 + 0.5, params.border_size);
	frag_color = vec4(texture(source_cube, dir).rgb, 1.0);
}
