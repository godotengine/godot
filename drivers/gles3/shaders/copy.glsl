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
uniform samplerCube source_cube;
#else
in vec2 uv_interp;
uniform sampler2D source;
#endif

in vec2 uv2_interp;

layout(location = 0) vec4 frag_color; //color:0

void main() {

	//vec4 color = color_interp;

	frag_color = color;
}

