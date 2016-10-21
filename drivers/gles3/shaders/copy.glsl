[vertex]


layout(location=0) in highp vec4 vertex_attrib;
#ifdef USE_CUBEMAP
layout(location=4) in vec3 cube_in;
#else
layout(location=4) in vec2 uv_in; // attrib:4
#endif
layout(location=5) in vec2 uv2_in; // attrib:5

#ifdef USE_CUBEMAP
out vec3 cube_interp;
#else
out vec2 uv_interp;
#endif

out vec2 uv2_interp;

void main() {

#ifdef USE_CUBEMAP
	cube_interp = cube_in;
#else
	uv_interp = uv_in;
#endif
	uv2_interp = uv2_in;
	gl_Position = vertex_attrib;
}

[fragment]


#ifdef USE_CUBEMAP
in vec3 cube_interp;
uniform samplerCube source_cube; //texunit:0
#else
in vec2 uv_interp;
uniform sampler2D source; //texunit:0
#endif


uniform float stuff;

in vec2 uv2_interp;

layout(location = 0) out vec4 frag_color;

void main() {

	//vec4 color = color_interp;

#ifdef USE_CUBEMAP
	vec4 color = texture( source_cube,  normalize(cube_interp) );

#else
	vec4 color = texture( source,  uv_interp );
#endif


	frag_color = color;
}

