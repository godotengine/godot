#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

void main() {
	vec2 vertex_base;
	if (gl_VertexIndex == 0) {
		vertex_base = vec2(-1.0, -1.0);
	} else if (gl_VertexIndex == 1) {
		vertex_base = vec2(-1.0, 3.0);
	} else {
		vertex_base = vec2(3.0, -1.0);
	}
	gl_Position = vec4(vertex_base, 0.0, 1.0);
	uv_interp = clamp(vertex_base, vec2(0.0), vec2(1.0)) * 2.0;
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(set = 0, binding = 0) uniform sampler2D source_color;
layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

layout(push_constant, std430) uniform Params {
	vec2 uv_offset;
	vec2 uv_scale;
	uint force_opaque;
	uint pad1;
	uint pad2;
	uint pad3;
}
params;

void main() {
	frag_color = texture(source_color, params.uv_offset + uv_interp * params.uv_scale);
	if (params.force_opaque != 0u) {
		frag_color.a = 1.0;
	}
}
