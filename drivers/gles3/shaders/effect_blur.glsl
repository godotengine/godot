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

#ifdef SSAO_MERGE
uniform sampler2D source_ssao; //texunit:1
#endif

uniform float lod;
uniform vec2 pixel_size;


layout(location = 0) out vec4 frag_color;

#ifdef SSAO_MERGE

uniform vec4 ssao_color;

#endif

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

#ifdef SSAO_MERGE

	vec4 color =textureLod( source_color,  uv_interp,0.0);
	float ssao =textureLod( source_ssao,  uv_interp,0.0).r;

	frag_color = vec4( mix(color.rgb,color.rgb*mix(ssao_color.rgb,vec3(1.0),ssao),color.a), 1.0 );

#endif
}

