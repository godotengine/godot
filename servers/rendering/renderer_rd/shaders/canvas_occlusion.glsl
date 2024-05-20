#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) in highp vec3 vertex;

layout(push_constant, std430) uniform Constants {
	mat4 projection;
	mat2x4 modelview;
	vec2 direction;
	float z_far;
	float pad;
}
constants;

#ifdef MODE_SHADOW
layout(location = 0) out highp float depth;
#endif

void main() {
	highp vec4 vtx = vec4(vertex, 1.0) * mat4(constants.modelview[0], constants.modelview[1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));

#ifdef MODE_SHADOW
	depth = dot(constants.direction, vtx.xy);
#endif
	gl_Position = constants.projection * vtx;
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(push_constant, std430) uniform Constants {
	mat4 projection;
	mat2x4 modelview;
	vec2 direction;
	float z_far;
	float pad;
}
constants;

#ifdef MODE_SHADOW
layout(location = 0) in highp float depth;
layout(location = 0) out highp float distance_buf;
#else
layout(location = 0) out highp float sdf_buf;
#endif

void main() {
#ifdef MODE_SHADOW
	distance_buf = depth / constants.z_far;
#else
	sdf_buf = 1.0;
#endif
}
