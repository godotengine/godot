[vertex]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

uniform highp mat4 projection_matrix;
uniform highp mat4 light_matrix;
uniform highp mat4 world_matrix;

attribute highp vec3 vertex; // attrib:0

#ifndef USE_DEPTH_SHADOWS

varying vec4 position_interp;

#endif


void main() {

	gl_Position = projection_matrix * (light_matrix * (world_matrix *  vec4(vertex,1.0)));

#ifndef USE_DEPTH_SHADOWS
	position_interp = gl_Position;
#endif

}

[fragment]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

#ifndef USE_DEPTH_SHADOWS

varying vec4 position_interp;

#endif

void main() {

#ifdef USE_DEPTH_SHADOWS

#else
	highp float depth = ((position_interp.z / position_interp.w) + 1.0) * 0.5 + 0.0;//bias;
	highp vec4 comp = fract(depth * vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0));
	comp -= comp.xxyz * vec4(0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
	gl_FragColor = comp;
#endif

}

