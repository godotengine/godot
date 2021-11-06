/* clang-format off */
[vertex]

#ifdef USE_GLES_OVER_GL
#define lowp
#define mediump
#define highp
#else
precision highp float;
precision highp int;
#endif

layout(location = 0) highp vec3 vertex;

uniform highp mat4 projection_matrix;
/* clang-format on */
uniform highp mat4 light_matrix;
uniform highp mat4 world_matrix;
uniform highp float distance_norm;

out highp vec4 position_interp;

void main() {
	gl_Position = projection_matrix * (light_matrix * (world_matrix * vec4(vertex, 1.0)));
	position_interp = gl_Position;
}

/* clang-format off */
[fragment]

#ifdef USE_GLES_OVER_GL
#define lowp
#define mediump
#define highp
#else
#if defined(USE_HIGHP_PRECISION)
precision highp float;
precision highp int;
#else
precision mediump float;
precision mediump int;
#endif
#endif

in highp vec4 position_interp;
/* clang-format on */

void main() {
	highp float depth = ((position_interp.z / position_interp.w) + 1.0) * 0.5 + 0.0; // bias

#ifdef USE_RGBA_SHADOWS

	highp vec4 comp = fract(depth * vec4(255.0 * 255.0 * 255.0, 255.0 * 255.0, 255.0, 1.0));
	comp -= comp.xxyz * vec4(0.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
	frag_color = comp;
#else

	frag_color = vec4(depth);
#endif
}
