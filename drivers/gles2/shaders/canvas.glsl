/* clang-format off */
[vertex]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

uniform highp mat4 projection_matrix;
/* clang-format on */
uniform highp mat4 modelview_matrix;
uniform highp mat4 extra_matrix;
attribute highp vec2 vertex; // attrib:0
attribute vec4 color_attrib; // attrib:3
attribute vec2 uv_attrib; // attrib:4

varying vec2 uv_interp;
varying vec4 color_interp;

uniform highp vec2 color_texpixel_size;

#ifdef USE_TEXTURE_RECT

uniform vec4 dst_rect;
uniform vec4 src_rect;

#endif

uniform highp float time;

/* clang-format off */

VERTEX_SHADER_GLOBALS

/* clang-format on */

vec2 select(vec2 a, vec2 b, bvec2 c) {
	vec2 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;

	return ret;
}

void main() {

	vec4 color = color_attrib;

#ifdef USE_TEXTURE_RECT

	if (dst_rect.z < 0.0) { // Transpose is encoded as negative dst_rect.z
		uv_interp = src_rect.xy + abs(src_rect.zw) * vertex.yx;
	} else {
		uv_interp = src_rect.xy + abs(src_rect.zw) * vertex;
	}

	vec4 outvec = vec4(0.0, 0.0, 0.0, 1.0);

	// This is what is done in the GLES 3 bindings and should
	// take care of flipped rects.
	//
	// But it doesn't.
	// I don't know why, will need to investigate further.

	outvec.xy = dst_rect.xy + abs(dst_rect.zw) * select(vertex, vec2(1.0, 1.0) - vertex, lessThan(src_rect.zw, vec2(0.0, 0.0)));

	// outvec.xy = dst_rect.xy + abs(dst_rect.zw) * vertex;
#else
	vec4 outvec = vec4(vertex.xy, 0.0, 1.0);

#ifdef USE_UV_ATTRIBUTE
	uv_interp = uv_attrib;
#else
	uv_interp = vertex.xy;
#endif

#endif

	{
		vec2 src_vtx = outvec.xy;
		/* clang-format off */

VERTEX_SHADER_CODE

		/* clang-format on */
	}

#if !defined(SKIP_TRANSFORM_USED)
	outvec = extra_matrix * outvec;
	outvec = modelview_matrix * outvec;
#endif

	color_interp = color;

#ifdef USE_PIXEL_SNAP
	outvec.xy = floor(outvec + 0.5).xy;
#endif

	gl_Position = projection_matrix * outvec;
}

/* clang-format off */
[fragment]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

uniform sampler2D color_texture; // texunit:-1
/* clang-format on */
uniform highp vec2 color_texpixel_size;
uniform mediump sampler2D normal_texture; // texunit:-2

varying mediump vec2 uv_interp;
varying mediump vec4 color_interp;

uniform highp float time;

uniform vec4 final_modulate;

#ifdef SCREEN_TEXTURE_USED

uniform sampler2D screen_texture; // texunit:-3

#endif

#ifdef SCREEN_UV_USED

uniform vec2 screen_pixel_size;

#endif

/* clang-format off */

FRAGMENT_SHADER_GLOBALS

/* clang-format on */

void main() {

	vec4 color = color_interp;

#if !defined(COLOR_USED)
	//default behavior, texture by color
	color *= texture2D(color_texture, uv_interp);
#endif

#ifdef SCREEN_UV_USED
	vec2 screen_uv = gl_FragCoord.xy * screen_pixel_size;
#endif
	{
		/* clang-format off */

FRAGMENT_SHADER_CODE

		/* clang-format on */
	}

	color *= final_modulate;

	gl_FragColor = color;
}
