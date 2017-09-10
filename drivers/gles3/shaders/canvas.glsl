[vertex]


layout(location=0) in highp vec2 vertex;
layout(location=3) in vec4 color_attrib;

#ifdef USE_TEXTURE_RECT

uniform vec4 dst_rect;
uniform vec4 src_rect;

#else

#ifdef USE_INSTANCING

layout(location=8) in highp vec4 instance_xform0;
layout(location=9) in highp vec4 instance_xform1;
layout(location=10) in highp vec4 instance_xform2;
layout(location=11) in lowp vec4 instance_color;

#ifdef USE_INSTANCE_CUSTOM
layout(location=12) in highp vec4 instance_custom_data;
#endif

#endif

layout(location=4) in highp vec2 uv_attrib;

//skeletn
#endif

uniform highp vec2 color_texpixel_size;


layout(std140) uniform CanvasItemData { //ubo:0

	highp mat4 projection_matrix;
	highp float time;
};

uniform highp mat4 modelview_matrix;
uniform highp mat4 extra_matrix;


out mediump vec2 uv_interp;
out mediump vec4 color_interp;

#ifdef USE_NINEPATCH

out highp vec2 pixel_size_interp;
#endif


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


out vec4 local_rot;


#ifdef USE_SHADOWS
out highp vec2 pos;
#endif

const bool at_light_pass = true;
#else
const bool at_light_pass = false;
#endif

#ifdef USE_PARTICLES
uniform int h_frames;
uniform int v_frames;
#endif


#if defined(USE_MATERIAL)

layout(std140) uniform UniformData { //ubo:2

MATERIAL_UNIFORMS

};

#endif

VERTEX_SHADER_GLOBALS

void main() {

	vec4 vertex_color = color_attrib;

#ifdef USE_INSTANCING
	mat4 extra_matrix2 = extra_matrix * transpose(mat4(instance_xform0,instance_xform1,instance_xform2,vec4(0.0,0.0,0.0,1.0)));
	vertex_color*=instance_color;
#else
	mat4 extra_matrix2 = extra_matrix;
#endif

#ifdef USE_TEXTURE_RECT

	if (dst_rect.z < 0.0) { // Transpose is encoded as negative dst_rect.z
		uv_interp = src_rect.xy + abs(src_rect.zw) * vertex.yx;
	} else {
		uv_interp = src_rect.xy + abs(src_rect.zw) * vertex;
	}
	highp vec4 outvec = vec4(dst_rect.xy + abs(dst_rect.zw) * mix(vertex,vec2(1.0,1.0)-vertex,lessThan(src_rect.zw,vec2(0.0,0.0))),0.0,1.0);

#else
	uv_interp = uv_attrib;
	highp vec4 outvec = vec4(vertex,0.0,1.0);
#endif


#ifdef USE_PARTICLES
	//scale by texture size
	outvec.xy/=color_texpixel_size;

	//compute h and v frames and adjust UV interp for animation
	int total_frames = h_frames * v_frames;
	int frame = min(int(float(total_frames) *instance_custom_data.z),total_frames-1);
	float frame_w = 1.0/float(h_frames);
	float frame_h = 1.0/float(v_frames);
	uv_interp.x = uv_interp.x * frame_w + frame_w * float(frame % h_frames);
	uv_interp.y = uv_interp.y * frame_h + frame_h * float(frame / h_frames);

#endif

#define extra_matrix extra_matrix2

{
	vec2 src_vtx=outvec.xy;

VERTEX_SHADER_CODE

}


#ifdef USE_NINEPATCH

	pixel_size_interp=abs(dst_rect.zw) * vertex;
#endif

#if !defined(SKIP_TRANSFORM_USED)
	outvec = extra_matrix * outvec;
	outvec = modelview_matrix * outvec;
#endif

#undef extra_matrix

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


	local_rot.xy=normalize( (modelview_matrix * ( extra_matrix * vec4(1.0,0.0,0.0,0.0) )).xy  );
	local_rot.zw=normalize( (modelview_matrix * ( extra_matrix * vec4(0.0,1.0,0.0,0.0) )).xy  );
#ifdef USE_TEXTURE_RECT
	local_rot.xy*=sign(src_rect.z);
	local_rot.zw*=sign(src_rect.w);
#endif



#endif

}

[fragment]



uniform mediump sampler2D color_texture; // texunit:0
uniform highp vec2 color_texpixel_size;
uniform mediump sampler2D normal_texture; // texunit:1

in mediump vec2 uv_interp;
in mediump vec4 color_interp;


#if defined(SCREEN_TEXTURE_USED)

uniform sampler2D screen_texture; // texunit:-3

#endif

#if defined(SCREEN_UV_USED)

uniform vec2 screen_pixel_size;
#endif

layout(std140) uniform CanvasItemData {

	highp mat4 projection_matrix;
	highp float time;
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


in vec4 local_rot;


#ifdef USE_SHADOWS

uniform highp sampler2D shadow_texture; // texunit:-2
in highp vec2 pos;

#endif

const bool at_light_pass = true;
#else
const bool at_light_pass = false;
#endif

uniform mediump vec4 final_modulate;




layout(location=0) out mediump vec4 frag_color;


#if defined(USE_MATERIAL)

layout(std140) uniform UniformData {

MATERIAL_UNIFORMS

};

#endif

FRAGMENT_SHADER_GLOBALS

void light_compute(inout vec3 light,vec3 light_vec,float light_height,vec4 light_color,vec2 light_uv,vec4 shadow,vec3 normal,vec2 uv,vec2 screen_uv,vec4 color) {

#if defined(USE_LIGHT_SHADER_CODE)

LIGHT_SHADER_CODE

#endif

}

#ifdef USE_TEXTURE_RECT

uniform vec4 dst_rect;
uniform vec4 src_rect;
uniform bool clip_rect_uv;

#ifdef USE_NINEPATCH

in highp vec2 pixel_size_interp;

uniform int np_repeat_v;
uniform int np_repeat_h;
uniform bool np_draw_center;
//left top right bottom in pixel coordinates
uniform vec4 np_margins;



float map_ninepatch_axis(float pixel, float draw_size,float tex_pixel_size,float margin_begin,float margin_end,int np_repeat,inout int draw_center) {


	float tex_size = 1.0/tex_pixel_size;

	if (pixel < margin_begin) {
		return pixel * tex_pixel_size;
	} else if (pixel >= draw_size-margin_end) {
		return (tex_size-(draw_size-pixel)) * tex_pixel_size;
	} else {
		if (!np_draw_center){
			draw_center--;
		}

		if (np_repeat==0) { //stretch
			//convert to ratio
			float ratio = (pixel - margin_begin) / (draw_size - margin_begin - margin_end);
			//scale to source texture
			return (margin_begin + ratio * (tex_size - margin_begin - margin_end)) * tex_pixel_size;
		} else if (np_repeat==1) { //tile
			//convert to ratio
			float ofs = mod((pixel - margin_begin), tex_size - margin_begin - margin_end);
			//scale to source texture
			return (margin_begin + ofs) * tex_pixel_size;
		} else if (np_repeat==2) { //tile fit
			//convert to ratio
			float src_area = draw_size - margin_begin - margin_end;
			float dst_area = tex_size - margin_begin - margin_end;
			float scale = max(1.0,floor(src_area / max(dst_area,0.0000001) + 0.5));

			//convert to ratio
			float ratio = (pixel - margin_begin) / src_area;
			ratio = mod(ratio * scale,1.0);
			return (margin_begin + ratio * dst_area) * tex_pixel_size;
		}
	}

}

#endif
#endif

uniform bool use_default_normal;

void main() {

	vec4 color = color_interp;
	vec2 uv = uv_interp;

#ifdef USE_TEXTURE_RECT

#ifdef USE_NINEPATCH

	int draw_center=2;
	uv = vec2(
				map_ninepatch_axis(pixel_size_interp.x,abs(dst_rect.z),color_texpixel_size.x,np_margins.x,np_margins.z,np_repeat_h,draw_center),
				map_ninepatch_axis(pixel_size_interp.y,abs(dst_rect.w),color_texpixel_size.y,np_margins.y,np_margins.w,np_repeat_v,draw_center)
				);

	if (draw_center==0) {
		color.a=0.0;
	}

	uv = uv*src_rect.zw+src_rect.xy; //apply region if needed
#endif

	if (clip_rect_uv) {

		vec2 half_texpixel = color_texpixel_size * 0.5;
		uv = clamp(uv,src_rect.xy+half_texpixel,src_rect.xy+abs(src_rect.zw)-color_texpixel_size);
	}

#endif

#if !defined(COLOR_USED)
//default behavior, texture by color

#ifdef USE_DISTANCE_FIELD
	const float smoothing = 1.0/32.0;
	float distance = textureLod(color_texture, uv,0.0).a;
	color.a = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance) * color.a;
#else
	color *= texture( color_texture,  uv );

#endif

#endif



	vec3 normal;

#if defined(NORMAL_USED)

	bool normal_used = true;
#else
	bool normal_used = false;
#endif

	if (use_default_normal) {
		normal.xy = textureLod(normal_texture, uv,0.0).xy * 2.0 - 1.0;
		normal.z = sqrt(1.0-dot(normal.xy,normal.xy));
		normal_used=true;
	} else {
		normal = vec3(0.0,0.0,1.0);
	}



#if defined(SCREEN_UV_USED)
	vec2 screen_uv = gl_FragCoord.xy*screen_pixel_size;
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

	if (normal_used) {
		normal.xy =  mat2(local_rot.xy,local_rot.zw) * normal.xy;
	}

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
		light_compute(light,light_vec,light_height,light_color,light_uv,shadow,normal,uv,screen_uv,color);

#else

		if (normal_used) {

			vec3 light_normal = normalize(vec3(light_vec,-light_height));
			light*=max(dot(-light_normal,normal),0.0);
		}

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

#define SHADOW_DEPTH(m_tex,m_uv) dot(texture((m_tex),(m_uv)),vec4(1.0 / (256.0 * 256.0 * 256.0),1.0 / (256.0 * 256.0),1.0 / 256.0,1)  )

#else

#define SHADOW_DEPTH(m_tex,m_uv) (texture((m_tex),(m_uv)).r)

#endif



#ifdef SHADOW_USE_GRADIENT

#define SHADOW_TEST(m_ofs) { highp float sd = SHADOW_DEPTH(shadow_texture,vec2(m_ofs,sh)); shadow_attenuation+=1.0-smoothstep(sd,sd+shadow_gradient,sz); }

#else

#define SHADOW_TEST(m_ofs) { highp float sd = SHADOW_DEPTH(shadow_texture,vec2(m_ofs,sh)); shadow_attenuation+=step(sz,sd); }

#endif


#ifdef SHADOW_FILTER_NEAREST

		SHADOW_TEST(su);

#endif


#ifdef SHADOW_FILTER_PCF3

		SHADOW_TEST(su+shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su-shadowpixel_size);
		shadow_attenuation/=3.0;

#endif


#ifdef SHADOW_FILTER_PCF5

		SHADOW_TEST(su+shadowpixel_size*2.0);
		SHADOW_TEST(su+shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su-shadowpixel_size);
		SHADOW_TEST(su-shadowpixel_size*2.0);
		shadow_attenuation/=5.0;

#endif


#ifdef SHADOW_FILTER_PCF7

		SHADOW_TEST(su+shadowpixel_size*3.0);
		SHADOW_TEST(su+shadowpixel_size*2.0);
		SHADOW_TEST(su+shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su-shadowpixel_size);
		SHADOW_TEST(su-shadowpixel_size*2.0);
		SHADOW_TEST(su-shadowpixel_size*3.0);
		shadow_attenuation/=7.0;

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
