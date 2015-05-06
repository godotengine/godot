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
attribute highp vec3 vertex; // attrib:0
attribute vec4 color_attrib; // attrib:3
attribute highp vec2 uv_attrib; // attrib:4

varying vec2 uv_interp;
varying vec4 color_interp;

//uniform bool snap_pixels;

void main() {

	color_interp = color_attrib;
	uv_interp = uv_attrib;		
	highp vec4 outvec = vec4(vertex, 1.0);
	outvec = extra_matrix * outvec;
	outvec = modelview_matrix * outvec;
#ifdef USE_PIXEL_SNAP

		outvec.xy=floor(outvec.xy+0.5);
#endif
	gl_Position = projection_matrix * outvec;
}

[fragment]

#ifdef USE_GLES_OVER_GL
#define mediump
#define highp
#else
precision mediump float;
precision mediump int;
#endif

 // texunit:0
uniform sampler2D texture;

varying vec2 uv_interp;
varying vec4 color_interp;

#ifdef MOMO

#endif

void main() {

	vec4 color = color_interp;
	
	color *= texture2D( texture,  uv_interp );

#ifdef DEBUG_ENCODED_32
	highp float enc32 = dot( color,highp vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  );
	color = vec4(vec3(enc32),1.0);
#endif

//	color.rgb*=color.a;
	gl_FragColor = color;

}

