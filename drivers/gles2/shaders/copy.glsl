[vertex]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

attribute highp vec4 vertex_attrib; // attrib:0
attribute vec2 uv_in; // attrib:4
attribute vec2 uv2_in; // attrib:5

varying vec2 uv_interp;

varying vec2 uv2_interp;

#ifdef USE_COPY_SECTION
uniform vec4 copy_section;
#endif

void main() {

	uv_interp = uv_in;
	uv2_interp = uv2_in;
	gl_Position = vertex_attrib;

#ifdef USE_COPY_SECTION
	uv_interp = copy_section.xy + uv_interp * copy_section.zw;
	gl_Position.xy = (copy_section.xy + (gl_Position.xy * 0.5 + 0.5) * copy_section.zw) * 2.0 - 1.0;
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


varying vec2 uv_interp;
uniform sampler2D source; // texunit:0

varying vec2 uv2_interp;

#ifdef USE_CUSTOM_ALPHA
uniform float custom_alpha;
#endif


void main() {

	//vec4 color = color_interp;
	vec4 color = texture2D( source,  uv_interp );


#ifdef USE_NO_ALPHA
	color.a=1.0;
#endif

#ifdef USE_CUSTOM_ALPHA
	color.a=custom_alpha;
#endif


	gl_FragColor = color;
}
