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

void main() {

	//0-1 in the texture we are writing to represents a circle, 0-2PI)
	//obtain the quarter circle from the source textures
	highp float sub_angle = ((mod(u,0.25)/0.25)*2.0-1.0)*(PI/4.0);
	highp float x=tan(sub_angle)*0.5+0.5;

	float depth;
	if (u<0.25) {
		depth=texture(textures[0],vec2(x,0.0)).x;
	} else if (u<0.50) {
		depth=texture(textures[1],vec2(x,0.0)).x;
	} else if (u<0.75) {
		depth=texture(textures[2],vec2(x,0.0)).x;
	} else {
		depth=texture(textures[3],vec2(x,0.0)).x;
	}
	distance=depth;
}
