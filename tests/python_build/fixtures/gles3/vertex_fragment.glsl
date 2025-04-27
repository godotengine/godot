#include "_included.glsl"

#[modes]

mode_ninepatch = #define USE_NINEPATCH
/* clang-format off */
#[specializations]

DISABLE_LIGHTING = false

#[vertex]

precision highp float;
precision highp int;
/* clang-format on */
layout(location = 0) in highp vec3 vertex;

out highp vec4 position_interp;

void main() {
	position_interp = vec4(vertex.x, 1, 0, 1);
}

#[fragment]

precision highp float;
precision highp int;

in highp vec4 position_interp;

void main() {
	highp float depth = ((position_interp.z / position_interp.w) + 1.0);
	frag_color = vec4(depth);
}
