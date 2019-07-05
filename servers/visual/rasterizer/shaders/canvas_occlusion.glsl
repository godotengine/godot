/* clang-format off */
[vertex]
/* clang-format on */

#version 450

layout(location = 0) in highp vec3 vertex;

layout(push_constant, binding = 0, std430) uniform Constants {

	mat4 modelview;
	mat4 projection;
} constants;

layout(location = 0) out highp float depth;

void main() {

	highp vec4 vtx = (constants.modelview * vec4(vertex, 1.0));
	depth = length(vtx.xy);

	gl_Position = constants.projection * vtx;

}

/* clang-format off */
[fragment]
/* clang-format on */

#version 450

layout(location = 0) in highp float depth;
layout(location = 0) out highp float distance_buf;

void main() {

	distance_buf=depth;
}
