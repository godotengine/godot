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

varying vec2 uv_out;



void main() {

        color_interp = color_attrib;
        uv_interp = uv_attrib;
        vec4 outvec = vec4(vertex, 1.0);
        outvec = extra_matrix * outvec;
        outvec = modelview_matrix * outvec;
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
varying vec2 uv_out;


void main() {

        vec4 color = color_interp;

        color *= texture2D( texture,  uv_interp );

        gl_FragColor = color;
}

