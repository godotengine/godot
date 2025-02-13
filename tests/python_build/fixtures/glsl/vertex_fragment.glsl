#[versions]

lines = "#define MODE_LINES";

#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec3 uv_interp;

void main() {

#ifdef MODE_LINES
	uv_interp = vec3(0,0,1);
#endif
}

#[fragment]

#version 450

#VERSION_DEFINES

#include "_included.glsl"

layout(location = 0) out vec4 dst_color;

void main() {
	dst_color = vec4(1,1,0,0);
}
