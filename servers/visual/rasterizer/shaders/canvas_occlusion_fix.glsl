/* clang-format off */
[vertex]
/* clang-format on */

#version 450

layout(location = 0) out highp float u;

void main() {

	if (gl_VertexIndex==0) {
		u=0.0;
		gl_Position=vec4(-1.0,-1.0,0.0,1.0);
	} else if (gl_VertexIndex==1) {
		u=0.0;
		gl_Position=vec4(-1.0,1.0,0.0,1.0);
	} else if (gl_VertexIndex==2) {
		u=1.0;
		gl_Position=vec4(1.0,1.0,0.0,1.0);
	} else {
		u=1.0;
		gl_Position=vec4(1.0,-1.0,0.0,1.0);
	}
}

/* clang-format off */
[fragment]
/* clang-format on */

#version 450

#define PI 3.14159265359

layout(set=0, binding=0) uniform sampler2D textures[4];
layout(location = 0) in highp float u;
layout(location = 0) out highp float distance;

layout(push_constant, binding = 0, std430) uniform Constants {
	mat4 projection;
	float far;
	float pad[3];
} constants;

void main() {

	//0-1 in the texture we are writing to represents a circle, 0-2PI)
	//obtain the quarter circle from the source textures
	float angle=fract(u+1.0-0.125);

	float depth;
#if 0
	if (angle<0.25) {
		highp float sub_angle = ((angle/0.25)*2.0-1.0)*(PI/4.0);
		highp float x=tan(sub_angle)*0.5+0.5;
		depth=texture(textures[0],vec2(x,0.0)).x;
	} else if (angle<0.50) {
		highp float sub_angle = (((angle-0.25)/0.25)*2.0-1.0)*(PI/4.0);
		highp float x=tan(sub_angle)*0.5+0.5;
		depth=texture(textures[1],vec2(x,0.0)).x;
	} else if (angle<0.75) {
		highp float sub_angle = (((angle-0.5)/0.25)*2.0-1.0)*(PI/4.0);
		highp float x=tan(sub_angle)*0.5+0.5;
		depth=texture(textures[2],vec2(x,0.0)).x;
	} else {
		highp float sub_angle = (((angle-0.75)/0.25)*2.0-1.0)*(PI/4.0);
		highp float x=tan(sub_angle)*0.5+0.5;
		depth=texture(textures[3],vec2(x,0.0)).x;
	}
#else
	if (angle<0.25) {
		highp float sub_angle = ((angle/0.25)*2.0-1.0)*(PI/4.0);
		vec2 pos = vec2(cos(sub_angle),sin(sub_angle))*constants.far;
		vec4 proj = constants.projection * vec4(pos,0.0,1.0);
		float coord = (proj.x/proj.w)*0.5+0.5;
		depth=texture(textures[0],vec2(coord,0.0)).x;
	} else if (angle<0.50) {
		highp float sub_angle = (((angle-0.25)/0.25)*2.0-1.0)*(PI/4.0);
		vec2 pos = vec2(cos(sub_angle),sin(sub_angle))*constants.far;
		vec4 proj = constants.projection * vec4(pos,0.0,1.0);
		float coord = (proj.x/proj.w)*0.5+0.5;
		depth=texture(textures[1],vec2(coord,0.0)).x;
	} else if (angle<0.75) {
		highp float sub_angle = (((angle-0.5)/0.25)*2.0-1.0)*(PI/4.0);
		vec2 pos = vec2(cos(sub_angle),sin(sub_angle))*constants.far;
		vec4 proj = constants.projection * vec4(pos,0.0,1.0);
		float coord = (proj.x/proj.w)*0.5+0.5;
		depth=texture(textures[2],vec2(coord,0.0)).x;
	} else {
		highp float sub_angle = (((angle-0.75)/0.25)*2.0-1.0)*(PI/4.0);
		vec2 pos = vec2(cos(sub_angle),sin(sub_angle))*constants.far;
		vec4 proj = constants.projection * vec4(pos,0.0,1.0);
		float coord = (proj.x/proj.w)*0.5+0.5;
		depth=texture(textures[3],vec2(coord,0.0)).x;
	}


#endif
	distance=depth;
}
