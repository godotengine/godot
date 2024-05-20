#[vertex]

#version 450

#VERSION_DEFINES

#include "_included.glsl"

layout(location = 0) out vec2 uv_interp;

void main() {
	uv_interp = vec2(0, 1);
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(location = 0) in vec2 uv_interp;

void main() {
	uv_interp = vec2(1, 0);
}
