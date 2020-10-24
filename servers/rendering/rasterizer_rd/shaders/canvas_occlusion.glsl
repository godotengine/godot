#[vertex]

#version 450

layout(location = 0) in highp vec3 vertex;

layout(push_constant, binding = 0, std430) uniform Constants {
	mat4 projection;
	mat2x4 modelview;
	vec2 direction;
	float z_far;
	float pad;
}
constants;

layout(location = 0) out highp float depth;

void main() {
	highp vec4 vtx = vec4(vertex, 1.0) * mat4(constants.modelview[0], constants.modelview[1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
	depth = dot(constants.direction, vtx.xy);

	gl_Position = constants.projection * vtx;
}

#[fragment]

#version 450

layout(push_constant, binding = 0, std430) uniform Constants {
	mat4 projection;
	mat2x4 modelview;
	vec2 direction;
	float z_far;
	float pad;
}
constants;

layout(location = 0) in highp float depth;
layout(location = 0) out highp float distance_buf;

void main() {
	distance_buf = depth / constants.z_far;
}
