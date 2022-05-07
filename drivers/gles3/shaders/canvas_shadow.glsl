/* clang-format off */
[vertex]

uniform highp mat4 projection_matrix;
/* clang-format on */
uniform highp mat4 light_matrix;
uniform highp mat4 world_matrix;
uniform highp float distance_norm;

layout(location = 0) in highp vec3 vertex;

out highp vec4 position_interp;

void main() {
	gl_Position = projection_matrix * (light_matrix * (world_matrix * vec4(vertex, 1.0)));
	position_interp = gl_Position;
}

/* clang-format off */
[fragment]

in highp vec4 position_interp;
/* clang-format on */

#ifdef USE_RGBA_SHADOWS
layout(location = 0) out lowp vec4 distance_buf;
#else
layout(location = 0) out highp float distance_buf;
#endif

void main() {
	highp float depth = ((position_interp.z / position_interp.w) + 1.0) * 0.5 + 0.0; // bias

#ifdef USE_RGBA_SHADOWS

	highp vec4 comp = fract(depth * vec4(255.0 * 255.0 * 255.0, 255.0 * 255.0, 255.0, 1.0));
	comp -= comp.xxyz * vec4(0.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
	distance_buf = comp;
#else

	distance_buf = depth;
#endif
}
