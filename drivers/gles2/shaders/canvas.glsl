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
uniform highp mat4 light_local_matrix;
uniform vec2 light_pos;
varying vec4 light_uv_interp;

#if defined(NORMAL_USED)
varying vec4 local_rot;
uniform vec2 normal_flip;
#endif

#ifdef USE_SHADOWS
varying highp vec2 pos;
#endif

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

	light_uv_interp.xy = (light_matrix * outvec).xy;
	light_uv_interp.zw =(light_local_matrix * outvec).xy;
#ifdef USE_SHADOWS
	pos=outvec.xy;
#endif

#if defined(NORMAL_USED)
	local_rot.xy=normalize( (modelview_matrix * ( extra_matrix * vec4(1.0,0.0,0.0,0.0) )).xy  )*normal_flip.x;
	local_rot.zw=normalize( (modelview_matrix * ( extra_matrix * vec4(0.0,1.0,0.0,0.0) )).xy  )*normal_flip.y;
#endif

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


uniform sampler2D texture; // texunit:0

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

#ifdef USE_MODULATE

uniform vec4 modulate;

#endif

#ifdef USE_LIGHTING

uniform sampler2D light_texture;
uniform vec4 light_color;
uniform vec4 light_shadow_color;
uniform float light_height;
varying vec4 light_uv_interp;
uniform float light_outside_alpha;

#if defined(NORMAL_USED)
varying vec4 local_rot;
#endif

#ifdef USE_SHADOWS

uniform highp sampler2D shadow_texture;
uniform float shadow_attenuation;

uniform highp mat4 shadow_matrix;
varying highp vec2 pos;
uniform float shadowpixel_size;

#ifdef SHADOW_ESM
uniform float shadow_esm_multiplier;
#endif

#endif

#endif

#if defined(USE_TEXPIXEL_SIZE)
uniform vec2 texpixel_size;
#endif


FRAGMENT_SHADER_GLOBALS


void main() {

	vec4 color = color_interp;
#if defined(NORMAL_USED)
	vec3 normal = vec3(0.0,0.0,1.0);
#endif

#ifdef USE_DISTANCE_FIELD
	const float smoothing = 1.0/32.0;
	float distance = texture2D(texture, uv_interp).a;
	color.a = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance) * color.a;
#else
	color *= texture2D( texture,  uv_interp );

#endif


#if defined(ENABLE_SCREEN_UV)
	vec2 screen_uv = gl_FragCoord.xy*screen_uv_mult;
#endif


{
#if defined(USE_NORMALMAP)
	vec3 normal_map=vec3(0.0,0.0,1.0);
	float normal_depth=1.0;
#endif

FRAGMENT_SHADER_CODE

#if defined(USE_NORMALMAP)
	normal = mix(vec3(0.0,0.0,1.0), normal_map * vec3(2.0,-2.0,1.0) - vec3( 1.0, -1.0, 0.0 ), normal_depth );
#endif

}
#ifdef DEBUG_ENCODED_32
	highp float enc32 = dot( color,highp vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  );
	color = vec4(vec3(enc32),1.0);
#endif

#ifdef USE_MODULATE

	color*=modulate;
#endif


#ifdef USE_LIGHTING

	vec2 light_vec = light_uv_interp.zw;; //for shadow and normal mapping

#if defined(NORMAL_USED)
	normal.xy =  mat2(local_rot.xy,local_rot.zw) * normal.xy;
#endif

	float att=1.0;

	vec2 light_uv = light_uv_interp.xy;
	vec4 light = texture2D(light_texture,light_uv) * light_color;
#if defined(USE_OUTPUT_SHADOW_COLOR)
	vec4 shadow_color=vec4(0.0,0.0,0.0,0.0);
#endif

	if (any(lessThan(light_uv_interp.xy,vec2(0.0,0.0))) || any(greaterThanEqual(light_uv_interp.xy,vec2(1.0,1.0)))) {
		color.a*=light_outside_alpha; //invisible

	} else {

#if defined(USE_LIGHT_SHADER_CODE)
//light is written by the light shader
		{
			vec4 light_out=light*color;
LIGHT_SHADER_CODE
			color=light_out;
		}

#else

#if defined(NORMAL_USED)
		vec3 light_normal = normalize(vec3(light_vec,-light_height));
		light*=max(dot(-light_normal,normal),0.0);
#endif

		color*=light;
/*
#ifdef USE_NORMAL
	color.xy=local_rot.xy;//normal.xy;
	color.zw=vec2(0.0,1.0);
#endif
*/

//light shader code
#endif


#ifdef USE_SHADOWS

		float angle_to_light = -atan(light_vec.x,light_vec.y);
		float PI = 3.14159265358979323846264;
		/*int i = int(mod(floor((angle_to_light+7.0*PI/6.0)/(4.0*PI/6.0))+1.0, 3.0)); // +1 pq os indices estao em ordem 2,0,1 nos arrays
		float ang*/

		float su,sz;

		float abs_angle = abs(angle_to_light);
		vec2 point;
		float sh;
		if (abs_angle<45.0*PI/180.0) {
			point = light_vec;
			sh=0.0+(1.0/8.0);
		} else if (abs_angle>135.0*PI/180.0) {
			point = -light_vec;
			sh = 0.5+(1.0/8.0);
		} else if (angle_to_light>0.0) {

			point = vec2(light_vec.y,-light_vec.x);
			sh = 0.25+(1.0/8.0);
		} else {

			point = vec2(-light_vec.y,light_vec.x);
			sh = 0.75+(1.0/8.0);

		}


		highp vec4 s = shadow_matrix * vec4(point,0.0,1.0);
		s.xyz/=s.w;
		su=s.x*0.5+0.5;
		sz=s.z*0.5+0.5;

		highp float shadow_attenuation=0.0;

#ifdef USE_DEPTH_SHADOWS

#define SHADOW_DEPTH(m_tex,m_uv) (texture2D((m_tex),(m_uv)).r)

#else

//#define SHADOW_DEPTH(m_tex,m_uv) dot(texture2D((m_tex),(m_uv)),highp vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  )
#define SHADOW_DEPTH(m_tex,m_uv) dot(texture2D((m_tex),(m_uv)),vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  )

#endif



#ifdef SHADOW_PCF5

		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su+shadowpixel_size,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su+shadowpixel_size*2.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su-shadowpixel_size,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su-shadowpixel_size*2.0,sh))<sz?0.0:1.0;
		shadow_attenuation/=5.0;

#endif

#ifdef SHADOW_PCF13

		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su+shadowpixel_size,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su+shadowpixel_size*2.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su+shadowpixel_size*3.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su+shadowpixel_size*4.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su+shadowpixel_size*5.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su+shadowpixel_size*6.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su-shadowpixel_size*2.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su-shadowpixel_size*3.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su-shadowpixel_size*4.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su-shadowpixel_size*5.0,sh))<sz?0.0:1.0;
		shadow_attenuation += SHADOW_DEPTH(shadow_texture,vec2(su-shadowpixel_size*6.0,sh))<sz?0.0:1.0;
		shadow_attenuation/=13.0;

#endif

#ifdef SHADOW_ESM


	{
			float unnormalized = su/shadowpixel_size;
			float fractional = fract(unnormalized);
			unnormalized = floor(unnormalized);
			float zc = SHADOW_DEPTH(shadow_texture,vec2((unnormalized-0.5)*shadowpixel_size,sh));
			float zn = SHADOW_DEPTH(shadow_texture,vec2((unnormalized+0.5)*shadowpixel_size,sh));
			float z = mix(zc,zn,fractional);
			shadow_attenuation=clamp(exp(shadow_esm_multiplier* ( z - sz )),0.0,1.0);
	}

#endif

#if !defined(SHADOW_PCF5) && !defined(SHADOW_PCF13) && !defined(SHADOW_ESM)

		shadow_attenuation = SHADOW_DEPTH(shadow_texture,vec2(su+shadowpixel_size,sh))<sz?0.0:1.0;

#endif

#if defined(USE_OUTPUT_SHADOW_COLOR)
	color=mix(shadow_color,color,shadow_attenuation);
#else
	//color*=shadow_attenuation;
	color=mix(light_shadow_color,color,shadow_attenuation);
#endif
//use shadows
#endif
	}

//use lighting
#endif
	//color.rgb*=color.a;
	gl_FragColor = color;


}

