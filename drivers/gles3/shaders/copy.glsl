[vertex]


layout(location=0) in highp vec4 vertex_attrib;
#if defined(USE_CUBEMAP) || defined(USE_PANORAMA)
layout(location=4) in vec3 cube_in;
#else
layout(location=4) in vec2 uv_in;
#endif
layout(location=5) in vec2 uv2_in;

#if defined(USE_CUBEMAP) || defined(USE_PANORAMA)
out vec3 cube_interp;
#else
out vec2 uv_interp;
#endif

out vec2 uv2_interp;

#ifdef USE_COPY_SECTION

uniform vec4 copy_section;

#endif

void main() {

#if defined(USE_CUBEMAP) || defined(USE_PANORAMA)
	cube_interp = cube_in;
#elif defined(USE_ASYM_PANO)
	uv_interp = vertex_attrib.xy;
#else
	uv_interp = uv_in;
#ifdef V_FLIP
	uv_interp.y = 1.0-uv_interp.y;
#endif

#endif
	uv2_interp = uv2_in;
	gl_Position = vertex_attrib;

#ifdef USE_COPY_SECTION

	uv_interp = copy_section.xy + uv_interp * copy_section.zw;
	gl_Position.xy = (copy_section.xy + (gl_Position.xy * 0.5 + 0.5) * copy_section.zw) * 2.0 - 1.0;
#endif

}

[fragment]

#define M_PI 3.14159265359

#if !defined(USE_GLES_OVER_GL)
precision mediump float;
#endif

#if defined(USE_CUBEMAP) || defined(USE_PANORAMA)
in vec3 cube_interp;
#else
in vec2 uv_interp;
#endif

#ifdef USE_ASYM_PANO
uniform highp mat4 pano_transform;
uniform highp vec4 asym_proj;
#endif

#ifdef USE_CUBEMAP
uniform samplerCube source_cube; //texunit:0
#else
uniform sampler2D source; //texunit:0
#endif


#ifdef USE_MULTIPLIER
uniform float multiplier;
#endif

#if defined(USE_PANORAMA) || defined(USE_ASYM_PANO)

vec4 texturePanorama(vec3 normal,sampler2D pano ) {

	vec2 st = vec2(
		atan(normal.x, normal.z),
		acos(normal.y)
	);

	if(st.x < 0.0)
		st.x += M_PI*2.0;

	st/=vec2(M_PI*2.0,M_PI);

	return textureLod(pano,st,0.0);

}

#endif


uniform float stuff;
uniform vec2 pixel_size;

in vec2 uv2_interp;


#ifdef USE_BCS

uniform vec3 bcs;

#endif

#ifdef USE_COLOR_CORRECTION

uniform sampler2D color_correction; //texunit:1

#endif

layout(location = 0) out vec4 frag_color;




void main() {

	//vec4 color = color_interp;

#ifdef USE_PANORAMA

	vec4 color = texturePanorama(  normalize(cube_interp), source );

#elif defined(USE_ASYM_PANO)

	// When an assymetrical projection matrix is used (applicable for stereoscopic rendering i.e. VR) we need to do this calculation per fragment to get a perspective correct result. 
	// Note that we're ignoring the x-offset for IPD, with Z sufficiently in the distance it becomes neglectible, as a result we could probably just set cube_normal.z to -1.
	// The Matrix[2][0] (= asym_proj.x) and Matrix[2][1] (= asym_proj.z) values are what provide the right shift in the image.

	vec3 cube_normal;
	cube_normal.z = -1000000.0;
	cube_normal.x = (cube_normal.z * (-uv_interp.x - asym_proj.x)) / asym_proj.y;
	cube_normal.y = (cube_normal.z * (-uv_interp.y - asym_proj.z)) / asym_proj.a;
	cube_normal = mat3(pano_transform) * cube_normal;
	cube_normal.z = -cube_normal.z;

	vec4 color = texturePanorama(  normalize(cube_normal.xyz), source );

#elif defined(USE_CUBEMAP)
	vec4 color = texture( source_cube,  normalize(cube_interp) );

#else
	vec4 color = textureLod( source,  uv_interp,0.0 );
#endif



#ifdef LINEAR_TO_SRGB
	//regular Linear -> SRGB conversion
	vec3 a = vec3(0.055);
	color.rgb = mix( (vec3(1.0)+a)*pow(color.rgb,vec3(1.0/2.4))-a , 12.92*color.rgb , lessThan(color.rgb,vec3(0.0031308)));
#endif

#ifdef SRGB_TO_LINEAR

	color.rgb = mix(pow((color.rgb + vec3(0.055)) * (1.0 / (1 + 0.055)),vec3(2.4)),color.rgb * (1.0 / 12.92),lessThan(color.rgb,vec3(0.04045)));
#endif

#ifdef DEBUG_GRADIENT
	color.rg=uv_interp;
	color.b=0.0;
#endif

#ifdef DISABLE_ALPHA
	color.a=1.0;
#endif


#ifdef GAUSSIAN_HORIZONTAL
	color*=0.38774;
	color+=texture( source,  uv_interp+vec2( 1.0, 0.0)*pixel_size )*0.24477;
	color+=texture( source,  uv_interp+vec2( 2.0, 0.0)*pixel_size )*0.06136;
	color+=texture( source,  uv_interp+vec2(-1.0, 0.0)*pixel_size )*0.24477;
	color+=texture( source,  uv_interp+vec2(-2.0, 0.0)*pixel_size )*0.06136;
#endif

#ifdef GAUSSIAN_VERTICAL
	color*=0.38774;
	color+=texture( source,  uv_interp+vec2( 0.0, 1.0)*pixel_size )*0.24477;
	color+=texture( source,  uv_interp+vec2( 0.0, 2.0)*pixel_size )*0.06136;
	color+=texture( source,  uv_interp+vec2( 0.0,-1.0)*pixel_size )*0.24477;
	color+=texture( source,  uv_interp+vec2( 0.0,-2.0)*pixel_size )*0.06136;
#endif

#ifdef USE_BCS

	color.rgb = mix(vec3(0.0),color.rgb,bcs.x);
	color.rgb = mix(vec3(0.5),color.rgb,bcs.y);
	color.rgb = mix(vec3(dot(vec3(1.0),color.rgb)*0.33333),color.rgb,bcs.z);

#endif

#ifdef USE_COLOR_CORRECTION

	color.r = texture(color_correction,vec2(color.r,0.0)).r;
	color.g = texture(color_correction,vec2(color.g,0.0)).g;
	color.b = texture(color_correction,vec2(color.b,0.0)).b;
#endif

#ifdef USE_MULTIPLIER
	color.rgb*=multiplier;
#endif
	frag_color = color;
}

