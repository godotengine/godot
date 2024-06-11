#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) in highp vec2 vertex;

layout(push_constant, std430) uniform Constants {
	mat2x4 modelview;

	vec2 scale;
	uint pad1;
	uint pad2;
}
constants;

void main() {
	highp vec4 vtx = vec4(vertex, 0.0, 1.0) * mat4(constants.modelview[0], constants.modelview[1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
//    highp vec4 vtx = vec4(vertex.xy, 0.0, 1.0);

//	gl_Position = constants.projection * vtx;
	gl_Position = vec4(vtx.xy * constants.scale, vtx.zw);
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(push_constant, std430) uniform Constants {
	mat2x4 modelview;
	vec2 scale;
}
constants;

layout(location = 0) out highp float color;

void main() {
	color = 0.0;
}
