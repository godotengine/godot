[vertex]


layout(location=0) in highp vec2 vertex;
layout(location=3) in vec4 color_attrib;

#ifdef USE_TEXTURE_RECT

layout(location=1) in highp vec4 dst_rect;
layout(location=2) in highp vec4 src_rect;

#else

layout(location=4) in highp vec2 uv_attrib;

//skeletn
#endif


layout(std140) uniform CanvasItemData { //ubo:0

	highp mat4 projection_matrix;
	highp vec4 time;
};

uniform highp mat4 modelview_matrix;
uniform highp mat4 extra_matrix;


out mediump vec2 uv_interp;
out mediump vec4 color_interp;

#ifdef USE_LIGHTING

layout(std140) uniform LightData { //ubo:1

	//light matrices
	highp mat4 light_matrix;
	highp mat4 light_local_matrix;
	highp mat4 shadow_matrix;
	highp vec4 light_color;
	highp vec4 light_shadow_color;
	highp vec2 light_pos;
	highp float shadowpixel_size;
	highp float shadow_gradient;
	highp float light_height;
	highp float light_outside_alpha;
	highp float shadow_distance_mult;
};


out vec4 light_uv_interp;

#if defined(NORMAL_USED)
out vec4 local_rot;
#endif

#ifdef USE_SHADOWS
out highp vec2 pos;
#endif

#endif


VERTEX_SHADER_GLOBALS

#if defined(USE_MATERIAL)

layout(std140) uniform UniformData { //ubo:2

MATERIAL_UNIFORMS

};

#endif

void main() {

	vec4 vertex_color = color_attrib;


#ifdef USE_TEXTURE_RECT


	uv_interp = src_rect.xy + abs(src_rect.zw) * vertex;
	highp vec4 outvec = vec4(dst_rect.xy + dst_rect.zw * mix(vertex,vec2(1.0,1.0)-vertex,lessThan(src_rect.zw,vec2(0.0,0.0))),0.0,1.0);

#else
	uv_interp = uv_attrib;
	highp vec4 outvec = vec4(vertex,0.0,1.0);
#endif


{
	vec2 src_vtx=outvec.xy;

VERTEX_SHADER_CODE

}

#if !defined(SKIP_TRANSFORM_USED)
	outvec = extra_matrix * outvec;
	outvec = modelview_matrix * outvec;
#endif

	color_interp = vertex_color;

#ifdef USE_PIXEL_SNAP

	outvec.xy=floor(outvec+0.5);
#endif


	gl_Position = projection_matrix * outvec;

#ifdef USE_LIGHTING

	light_uv_interp.xy = (light_matrix * outvec).xy;
	light_uv_interp.zw =(light_local_matrix * outvec).xy;
#ifdef USE_SHADOWS
	pos=outvec.xy;
#endif

#if defined(NORMAL_USED)
	local_rot.xy=normalize( (modelview_matrix * ( extra_matrix * vec4(1.0,0.0,0.0,0.0) )).xy  );
	local_rot.zw=normalize( (modelview_matrix * ( extra_matrix * vec4(0.0,1.0,0.0,0.0) )).xy  );
#ifdef USE_TEXTURE_RECT
	local_rot.xy*=sign(src_rect.z);
	local_rot.zw*=sign(src_rect.w);
#endif

#endif

#endif

}

[fragment]



uniform mediump sampler2D color_texture; // texunit:0
uniform highp vec2 color_texpixel_size;

in mediump vec2 uv_interp;
in mediump vec4 color_interp;


#if defined(SCREEN_TEXTURE_USED)

uniform sampler2D screen_texture; // texunit:-3

#endif

layout(std140) uniform CanvasItemData {

	highp mat4 projection_matrix;
	highp vec4 time;
};


#ifdef USE_LIGHTING

layout(std140) uniform LightData {

	highp mat4 light_matrix;
	highp mat4 light_local_matrix;
	highp mat4 shadow_matrix;
	highp vec4 light_color;
	highp vec4 light_shadow_color;
	highp vec2 light_pos;
	highp float shadowpixel_size;
	highp float shadow_gradient;
	highp float light_height;
	highp float light_outside_alpha;
	highp float shadow_distance_mult;
};

uniform lowp sampler2D light_texture; // texunit:-1
in vec4 light_uv_interp;


#if defined(NORMAL_USED)
in vec4 local_rot;
#endif

#ifdef USE_SHADOWS

uniform highp sampler2D shadow_texture; // texunit:-2
in highp vec2 pos;

#endif

#endif

uniform mediump vec4 final_modulate;

FRAGMENT_SHADER_GLOBALS


layout(location=0) out mediump vec4 frag_color;


#if defined(USE_MATERIAL)

layout(std140) uniform UniformData {

MATERIAL_UNIFORMS

};

#endif

void main() {

	vec4 color = color_interp;
#if defined(NORMAL_USED)
	vec3 normal = vec3(0.0,0.0,1.0);
#endif

#if !defined(COLOR_USED)
//default behavior, texture by color

#ifdef USE_DISTANCE_FIELD
	const float smoothing = 1.0/32.0;
	float distance = texture(color_texture, uv_interp).a;
	color.a = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance) * color.a;
#else
	color *= texture( color_texture,  uv_interp );

#endif

#endif

#if defined(ENABLE_SCREEN_UV)
	vec2 screen_uv = gl_FragCoord.xy*screen_uv_mult;
#endif


{
	float normal_depth=1.0;

#if defined(NORMALMAP_USED)
	vec3 normal_map=vec3(0.0,0.0,1.0);
#endif

FRAGMENT_SHADER_CODE

#if defined(NORMALMAP_USED)
	normal = mix(vec3(0.0,0.0,1.0), normal_map * vec3(2.0,-2.0,1.0) - vec3( 1.0, -1.0, 0.0 ), normal_depth );
#endif

}
#ifdef DEBUG_ENCODED_32
	highp float enc32 = dot( color,highp vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  );
	color = vec4(vec3(enc32),1.0);
#endif


	color*=final_modulate;




#ifdef USE_LIGHTING

	vec2 light_vec = light_uv_interp.zw;; //for shadow and normal mapping

#if defined(NORMAL_USED)
	normal.xy =  mat2(local_rot.xy,local_rot.zw) * normal.xy;
#endif

	float att=1.0;

	vec2 light_uv = light_uv_interp.xy;
	vec4 light = texture(light_texture,light_uv) * light_color;
#if defined(SHADOW_COLOR_USED)
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
		//sz=lightlength(light_vec);

		highp float shadow_attenuation=0.0;

#ifdef USE_RGBA_SHADOWS

#define SHADOW_DEPTH(m_tex,m_uv) dot(texture2D((m_tex),(m_uv)),vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  )

#else

#define SHADOW_DEPTH(m_tex,m_uv) (texture2D((m_tex),(m_uv)).r)

#endif



#ifdef SHADOW_USE_GRADIENT

#define SHADOW_TEST(m_ofs) { highp float sd = SHADOW_DEPTH(shadow_texture,vec2(m_ofs,sh)); shadow_attenuation+=1.0-smoothstep(sd,sd+shadow_gradient,sz); }

#else

#define SHADOW_TEST(m_ofs) { highp float sd = SHADOW_DEPTH(shadow_texture,vec2(m_ofs,sh)); shadow_attenuation+=step(sz,sd); }

#endif


#ifdef SHADOW_FILTER_NEAREST

		SHADOW_TEST(su+shadowpixel_size);

#endif


#ifdef SHADOW_FILTER_PCF3

		SHADOW_TEST(su+shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su-shadowpixel_size);
		shadow_attenuation/=3.0;

#endif


#ifdef SHADOW_FILTER_PCF5

		SHADOW_TEST(su+shadowpixel_size*3.0);
		SHADOW_TEST(su+shadowpixel_size*2.0);
		SHADOW_TEST(su+shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su-shadowpixel_size);
		SHADOW_TEST(su-shadowpixel_size*2.0);
		SHADOW_TEST(su-shadowpixel_size*3.0);
		shadow_attenuation/=5.0;

#endif


#ifdef SHADOW_FILTER_PCF9

		SHADOW_TEST(su+shadowpixel_size*4.0);
		SHADOW_TEST(su+shadowpixel_size*3.0);
		SHADOW_TEST(su+shadowpixel_size*2.0);
		SHADOW_TEST(su+shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su-shadowpixel_size);
		SHADOW_TEST(su-shadowpixel_size*2.0);
		SHADOW_TEST(su-shadowpixel_size*3.0);
		SHADOW_TEST(su-shadowpixel_size*4.0);
		shadow_attenuation/=9.0;

#endif

#ifdef SHADOW_FILTER_PCF13

		SHADOW_TEST(su+shadowpixel_size*6.0);
		SHADOW_TEST(su+shadowpixel_size*5.0);
		SHADOW_TEST(su+shadowpixel_size*4.0);
		SHADOW_TEST(su+shadowpixel_size*3.0);
		SHADOW_TEST(su+shadowpixel_size*2.0);
		SHADOW_TEST(su+shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su-shadowpixel_size);
		SHADOW_TEST(su-shadowpixel_size*2.0);
		SHADOW_TEST(su-shadowpixel_size*3.0);
		SHADOW_TEST(su-shadowpixel_size*4.0);
		SHADOW_TEST(su-shadowpixel_size*5.0);
		SHADOW_TEST(su-shadowpixel_size*6.0);
		shadow_attenuation/=13.0;

#endif


#if defined(SHADOW_COLOR_USED)
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
	frag_color = color;

}

