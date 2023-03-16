/* clang-format off */
#[modes]

mode_sdf =
mode_shadow = #define MODE_SHADOW
mode_shadow_RGBA = #define MODE_SHADOW \n#define USE_RGBA_SHADOWS

#[specializations]

#[vertex]

layout(location = 0) in vec3 vertex;

uniform highp mat4 projection;
uniform highp vec4 modelview1;
uniform highp vec4 modelview2;
uniform highp vec2 direction;
uniform highp float z_far;

#ifdef MODE_SHADOW
out float depth;
#endif

void main() {
	highp vec4 vtx = vec4(vertex, 1.0) * mat4(modelview1, modelview2, vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));

#ifdef MODE_SHADOW
	depth = dot(direction, vtx.xy);
#endif
	gl_Position = projection * vtx;
}

#[fragment]


uniform highp mat4 projection;
uniform highp vec4 modelview1;
uniform highp vec4 modelview2;
uniform highp vec2 direction;
uniform highp float z_far;

#ifdef MODE_SHADOW
in highp float depth;
#endif

#ifdef USE_RGBA_SHADOWS
layout(location = 0) out lowp vec4 out_buf;
#else
layout(location = 0) out highp float out_buf;
#endif

void main() {
    float out_depth = 1.0;

#ifdef MODE_SHADOW
	out_depth = depth / z_far;
#endif

#ifdef USE_RGBA_SHADOWS
	out_depth = clamp(out_depth, -1.0, 1.0);
	out_depth = out_depth * 0.5 + 0.5;
	highp vec4 comp = fract(out_depth * vec4(255.0 * 255.0 * 255.0, 255.0 * 255.0, 255.0, 1.0));
	comp -= comp.xxyz * vec4(0.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
	out_buf = comp;
#else
	out_buf = out_depth;
#endif
}
