[vertex]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

uniform highp mat4 projection_matrix;
uniform highp mat4 modelview_matrix;
uniform highp mat4 extra_matrix;
attribute highp vec2 vertex; // attrib:0
attribute vec4 color_attrib; // attrib:3
attribute vec2 uv_attrib; // attrib:4

varying vec2 uv_interp;
varying vec4 color_interp;

uniform highp vec2 color_texpixel_size;

uniform highp float time;

VERTEX_SHADER_GLOBALS

vec2 select(vec2 a, vec2 b, bvec2 c) {
	vec2 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;

	return ret;
}

void main() {

	vec4 color = color_attrib;

	vec4 outvec = vec4(vertex.xy, 0.0, 1.0);

	uv_interp = uv_attrib;

{
        vec2 src_vtx=outvec.xy;
VERTEX_SHADER_CODE

}

	color_interp = color;

	gl_Position = projection_matrix * modelview_matrix * outvec;

}

[fragment]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

uniform sampler2D color_texture; // texunit:0
uniform highp vec2 color_texpixel_size;
uniform mediump sampler2D normal_texture; // texunit:1

varying mediump vec2 uv_interp;
varying mediump vec4 color_interp;

uniform highp float time;

uniform vec4 final_modulate;

#ifdef SCREEN_TEXTURE_USED

uniform sampler2D screen_texture; // texunit:2

#endif

#ifdef SCREEN_UV_USED

uniform vec2 screen_pixel_size;

#endif

FRAGMENT_SHADER_GLOBALS


void main() {

	vec4 color = color_interp;

	color *= texture2D(color_texture, uv_interp);

#ifdef SCREEN_UV_USED
	vec2 screen_uv = gl_FragCoord.xy * screen_pixel_size;
#endif
{

FRAGMENT_SHADER_CODE


}

	color *= final_modulate;

	gl_FragColor = color;

}
