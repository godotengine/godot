/* clang-format off */
#[vertex]

#version 450

#VERSION_DEFINES

/* clang-format on */

layout(push_constant, std430) uniform Info {
	mat4 mvp;
	vec4 color;
}
info;

layout(location = 0) in vec3 vertex_attrib;

void main() {
	vec4 vertex = info.mvp * vec4(vertex_attrib, 1.0);
	vertex.xyz /= vertex.w;
	gl_Position = vec4(vertex.xy, 0.0, 1.0);
}

/* clang-format off */
#[fragment]

#version 450

#VERSION_DEFINES

layout(push_constant, std430) uniform Info {
	mat4 mvp;
	vec4 color;
}
info;

layout(location = 0) out vec4 frag_color;

void main() {
	frag_color = info.color;
}
