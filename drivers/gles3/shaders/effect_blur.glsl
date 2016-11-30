[vertex]


layout(location=0) in highp vec4 vertex_attrib;
layout(location=4) in vec2 uv_in;

out vec2 uv_interp;


void main() {

	uv_interp = uv_in;
	gl_Position = vertex_attrib;
}

[fragment]


in vec2 uv_interp;
uniform sampler2D source_color; //texunit:0

uniform float lod;
uniform vec2 pixel_size;


layout(location = 0) out vec4 frag_color;

void main() {



#ifdef GAUSSIAN_HORIZONTAL
	vec4 color =textureLod( source_color,  uv_interp+vec2( 0.0, 0.0)*pixel_size,lod )*0.38774;
	color+=textureLod( source_color,  uv_interp+vec2( 1.0, 0.0)*pixel_size,lod )*0.24477;
	color+=textureLod( source_color,  uv_interp+vec2( 2.0, 0.0)*pixel_size,lod )*0.06136;
	color+=textureLod( source_color,  uv_interp+vec2(-1.0, 0.0)*pixel_size,lod )*0.24477;
	color+=textureLod( source_color,  uv_interp+vec2(-2.0, 0.0)*pixel_size,lod )*0.06136;
	frag_color = color;
#endif

#ifdef GAUSSIAN_VERTICAL
	vec4 color =textureLod( source_color,  uv_interp+vec2( 0.0, 0.0)*pixel_size,lod )*0.38774;
	color+=textureLod( source_color,  uv_interp+vec2( 0.0, 1.0)*pixel_size,lod )*0.24477;
	color+=textureLod( source_color,  uv_interp+vec2( 0.0, 2.0)*pixel_size,lod )*0.06136;
	color+=textureLod( source_color,  uv_interp+vec2( 0.0,-1.0)*pixel_size,lod )*0.24477;
	color+=textureLod( source_color,  uv_interp+vec2( 0.0,-2.0)*pixel_size,lod )*0.06136;
	frag_color = color;
#endif

#ifdef SIMPLE_COPY
	vec4 color =textureLod( source_color,  uv_interp,0.0);
	frag_color = color;
#endif


}

