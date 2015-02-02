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

#if defined(USE_TIME)
uniform float time;
#endif


#ifdef USE_LIGHTING

uniform highp mat4 light_matrix;
varying vec4 light_tex_pos;

#endif

#if defined(ENABLE_VAR1_INTERP)
varying vec4 var1_interp;
#endif

#if defined(ENABLE_VAR2_INTERP)
varying vec4 var2_interp;
#endif

//uniform bool snap_pixels;

VERTEX_SHADER_GLOBALS

void main() {

	color_interp = color_attrib;
	uv_interp = uv_attrib;		
        highp vec4 outvec = vec4(vertex, 1.0);
{
        vec2 src_vtx=outvec.xy;
VERTEX_SHADER_CODE

}
#if !defined(USE_WORLD_VEC)
        outvec = extra_matrix * outvec;
        outvec = modelview_matrix * outvec;
#endif

#ifdef USE_PIXEL_SNAP

	outvec.xy=floor(outvec.xy+0.5);
#endif


	gl_Position = projection_matrix * outvec;

#ifdef USE_LIGHTING

	light_tex_pos.xy = light_matrix * gl_Position;
	light_tex_pos.zw=outvec.xy - light_matrix[4].xy; //likely wrong

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

 // texunit:0
uniform sampler2D texture;

varying vec2 uv_interp;
varying vec4 color_interp;

#ifdef MOMO

#endif

#if defined(ENABLE_SCREEN_UV)

uniform vec2 screen_uv_mult;

#endif

#if defined(ENABLE_TEXSCREEN)

uniform vec2 texscreen_screen_mult;
uniform vec4 texscreen_screen_clamp;
uniform sampler2D texscreen_tex;

#endif


#if defined(ENABLE_VAR1_INTERP)
varying vec4 var1_interp;
#endif

#if defined(ENABLE_VAR2_INTERP)
varying vec4 var2_interp;
#endif

#if defined(USE_TIME)
uniform float time;
#endif


#ifdef USE_LIGHTING

uniform sampler2D light_texture;
varying vec4 light_tex_pos;

#ifdef USE_SHADOWS

uniform sampler2D shadow_texture;
uniform float shadow_attenuation;

#endif

#endif

#if defined(USE_TEXPIXEL_SIZE)
uniform vec2 texpixel_size;
#endif


FRAGMENT_SHADER_GLOBALS


void main() {

	vec4 color = color_interp;
#if defined(NORMAL_USED)
	vec3 normal = vec3(0,0,1);
#endif

	color *= texture2D( texture,  uv_interp );
#if defined(ENABLE_SCREEN_UV)
	vec2 screen_uv = gl_FragCoord.xy*screen_uv_mult;
#endif

{
FRAGMENT_SHADER_CODE
}
#ifdef DEBUG_ENCODED_32
	highp float enc32 = dot( color,highp vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  );
	color = vec4(vec3(enc32),1.0);
#endif

#ifdef USE_LIGHTING

	float att=1.0;

	vec3 light = texture2D(light_texture,light_tex_pos).rgb;
#ifdef USE_SHADOWS
	//this might not be that great on mobile?
	float light_dist = length(light_texture.zw);
	float light_angle = atan2(light_texture.x,light_texture.z) + 1.0 * 0.5;
	float shadow_dist = texture2D(shadow_texture,vec2(light_angle,0));
	if (light_dist>shadow_dist) {
		light*=shadow_attenuation;
	}
//use shadows
#endif

#if defined(USE_LIGHT_SHADER_CODE)
//light is written by the light shader
{
	vec2 light_dir = normalize(light_tex_pos.zw);
	float light_distance = length(light_tex_pos.zw);
LIGHT_SHADER_CODE
}
#else

#if defined(NORMAL_USED)
	vec2 light_normal = normalize(light_tex_pos.zw);
	light = color.rgb * light * max(dot(light_normal,normal),0);
#endif

	color.rgb=light;
//light shader code
#endif

//use lighting
#endif
//	color.rgb*=color.a;
	gl_FragColor = color;

}

