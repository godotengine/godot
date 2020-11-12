/* clang-format off */
[vertex]

#ifdef USE_GLES_OVER_GL
#define lowp
#define mediump
#define highp
#else
precision highp float;
precision highp int;
#endif

uniform highp mat4 projection_matrix;
/* clang-format on */

#include "stdlib.glsl"

uniform highp mat4 modelview_matrix;
uniform highp mat4 extra_matrix;
attribute highp vec2 vertex; // attrib:0

#ifdef USE_ATTRIB_LIGHT_ANGLE
// shared with tangent, not used in canvas shader
attribute highp float light_angle; // attrib:2
#endif

attribute vec4 color_attrib; // attrib:3
attribute vec2 uv_attrib; // attrib:4

#ifdef USE_ATTRIB_MODULATE
attribute highp vec4 modulate_attrib; // attrib:5
#endif

#ifdef USE_ATTRIB_LARGE_VERTEX
// shared with skeleton attributes, not used in batched shader
attribute highp vec2 translate_attrib; // attrib:6
attribute highp vec4 basis_attrib; // attrib:7
#endif

#ifdef USE_SKELETON
attribute highp vec4 bone_indices; // attrib:6
attribute highp vec4 bone_weights; // attrib:7
#endif

#ifdef USE_INSTANCING

attribute highp vec4 instance_xform0; //attrib:8
attribute highp vec4 instance_xform1; //attrib:9
attribute highp vec4 instance_xform2; //attrib:10
attribute highp vec4 instance_color; //attrib:11

#ifdef USE_INSTANCE_CUSTOM
attribute highp vec4 instance_custom_data; //attrib:12
#endif

#endif

#ifdef USE_SKELETON
uniform highp sampler2D skeleton_texture; // texunit:-3
uniform highp ivec2 skeleton_texture_size;
uniform highp mat4 skeleton_transform;
uniform highp mat4 skeleton_transform_inverse;
#endif

varying vec2 uv_interp;
varying vec4 color_interp;

#ifdef USE_ATTRIB_MODULATE
// modulate doesn't need interpolating but we need to send it to the fragment shader
varying vec4 modulate_interp;
#endif

#ifdef MODULATE_USED
uniform vec4 final_modulate;
#endif

uniform highp vec2 color_texpixel_size;

#ifdef USE_TEXTURE_RECT

uniform vec4 dst_rect;
uniform vec4 src_rect;

#endif

uniform highp float time;

#ifdef USE_LIGHTING

// light matrices
uniform highp mat4 light_matrix;
uniform highp mat4 light_matrix_inverse;
uniform highp mat4 light_local_matrix;
uniform highp mat4 shadow_matrix;
uniform highp vec4 light_color;
uniform highp vec4 light_shadow_color;
uniform highp vec2 light_pos;
uniform highp float shadowpixel_size;
uniform highp float shadow_gradient;
uniform highp float light_height;
uniform highp float light_outside_alpha;
uniform highp float shadow_distance_mult;

varying vec4 light_uv_interp;
varying vec2 transformed_light_uv;
varying vec4 local_rot;

#ifdef USE_SHADOWS
varying highp vec2 pos;
#endif

const bool at_light_pass = true;
#else
const bool at_light_pass = false;
#endif

/* clang-format off */

VERTEX_SHADER_GLOBALS

/* clang-format on */

vec2 select(vec2 a, vec2 b, bvec2 c) {
	vec2 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;

	return ret;
}

void main() {

	vec4 color = color_attrib;
	vec2 uv;

#ifdef USE_INSTANCING
	mat4 extra_matrix_instance = extra_matrix * transpose(mat4(instance_xform0, instance_xform1, instance_xform2, vec4(0.0, 0.0, 0.0, 1.0)));
	color *= instance_color;

#ifdef USE_INSTANCE_CUSTOM
	vec4 instance_custom = instance_custom_data;
#else
	vec4 instance_custom = vec4(0.0);
#endif

#else
	mat4 extra_matrix_instance = extra_matrix;
	vec4 instance_custom = vec4(0.0);
#endif

#ifdef USE_TEXTURE_RECT

	if (dst_rect.z < 0.0) { // Transpose is encoded as negative dst_rect.z
		uv = src_rect.xy + abs(src_rect.zw) * vertex.yx;
	} else {
		uv = src_rect.xy + abs(src_rect.zw) * vertex;
	}

	vec4 outvec = vec4(0.0, 0.0, 0.0, 1.0);

	// This is what is done in the GLES 3 bindings and should
	// take care of flipped rects.
	//
	// But it doesn't.
	// I don't know why, will need to investigate further.

	outvec.xy = dst_rect.xy + abs(dst_rect.zw) * select(vertex, vec2(1.0, 1.0) - vertex, lessThan(src_rect.zw, vec2(0.0, 0.0)));

	// outvec.xy = dst_rect.xy + abs(dst_rect.zw) * vertex;
#else
	vec4 outvec = vec4(vertex.xy, 0.0, 1.0);

	uv = uv_attrib;
#endif

	float point_size = 1.0;

	{
		vec2 src_vtx = outvec.xy;
		/* clang-format off */

VERTEX_SHADER_CODE

		/* clang-format on */
	}

	gl_PointSize = point_size;

#ifdef USE_ATTRIB_MODULATE
	// modulate doesn't need interpolating but we need to send it to the fragment shader
	modulate_interp = modulate_attrib;
#endif

#ifdef USE_ATTRIB_LARGE_VERTEX
	// transform is in attributes
	vec2 temp;

	temp = outvec.xy;
	temp.x = (outvec.x * basis_attrib.x) + (outvec.y * basis_attrib.z);
	temp.y = (outvec.x * basis_attrib.y) + (outvec.y * basis_attrib.w);

	temp += translate_attrib;
	outvec.xy = temp;

#else

	// transform is in uniforms
#if !defined(SKIP_TRANSFORM_USED)
	outvec = extra_matrix_instance * outvec;
	outvec = modelview_matrix * outvec;
#endif

#endif // not large integer

	color_interp = color;

#ifdef USE_PIXEL_SNAP
	outvec.xy = floor(outvec + 0.5).xy;
	// precision issue on some hardware creates artifacts within texture
	// offset uv by a small amount to avoid
	uv += 1e-5;
#endif

#ifdef USE_SKELETON

	// look up transform from the "pose texture"
	if (bone_weights != vec4(0.0)) {

		highp mat4 bone_transform = mat4(0.0);

		for (int i = 0; i < 4; i++) {
			ivec2 tex_ofs = ivec2(int(bone_indices[i]) * 2, 0);

			highp mat4 b = mat4(
					texel2DFetch(skeleton_texture, skeleton_texture_size, tex_ofs + ivec2(0, 0)),
					texel2DFetch(skeleton_texture, skeleton_texture_size, tex_ofs + ivec2(1, 0)),
					vec4(0.0, 0.0, 1.0, 0.0),
					vec4(0.0, 0.0, 0.0, 1.0));

			bone_transform += b * bone_weights[i];
		}

		mat4 bone_matrix = skeleton_transform * transpose(bone_transform) * skeleton_transform_inverse;

		outvec = bone_matrix * outvec;
	}

#endif

	uv_interp = uv;
	gl_Position = projection_matrix * outvec;

#ifdef USE_LIGHTING

	light_uv_interp.xy = (light_matrix * outvec).xy;
	light_uv_interp.zw = (light_local_matrix * outvec).xy;

	transformed_light_uv = (mat3(light_matrix_inverse) * vec3(light_uv_interp.zw, 0.0)).xy; //for normal mapping

#ifdef USE_SHADOWS
	pos = outvec.xy;
#endif

#ifdef USE_ATTRIB_LIGHT_ANGLE
	// we add a fixed offset because we are using the sign later,
	// and don't want floating point error around 0.0
	float la = abs(light_angle) - 1.0;

	// vector light angle
	vec4 vla;
	vla.xy = vec2(cos(la), sin(la));
	vla.zw = vec2(-vla.y, vla.x);

	// vertical flip encoded in the sign
	vla.zw *= sign(light_angle);

	// apply the transform matrix.
	// The rotate will be encoded in the transform matrix for single rects,
	// and just the flips in the light angle.
	// For batching we will encode the rotation and the flips
	// in the light angle, and can use the same shader.
	local_rot.xy = normalize((modelview_matrix * (extra_matrix_instance * vec4(vla.xy, 0.0, 0.0))).xy);
	local_rot.zw = normalize((modelview_matrix * (extra_matrix_instance * vec4(vla.zw, 0.0, 0.0))).xy);
#else
	local_rot.xy = normalize((modelview_matrix * (extra_matrix_instance * vec4(1.0, 0.0, 0.0, 0.0))).xy);
	local_rot.zw = normalize((modelview_matrix * (extra_matrix_instance * vec4(0.0, 1.0, 0.0, 0.0))).xy);
#ifdef USE_TEXTURE_RECT
	local_rot.xy *= sign(src_rect.z);
	local_rot.zw *= sign(src_rect.w);
#endif
#endif // not using light angle

#endif
}

/* clang-format off */
[fragment]

// texture2DLodEXT and textureCubeLodEXT are fragment shader specific.
// Do not copy these defines in the vertex section.
#ifndef USE_GLES_OVER_GL
#ifdef GL_EXT_shader_texture_lod
#extension GL_EXT_shader_texture_lod : enable
#define texture2DLod(img, coord, lod) texture2DLodEXT(img, coord, lod)
#define textureCubeLod(img, coord, lod) textureCubeLodEXT(img, coord, lod)
#endif
#endif // !USE_GLES_OVER_GL

#ifdef GL_ARB_shader_texture_lod
#extension GL_ARB_shader_texture_lod : enable
#endif

#if !defined(GL_EXT_shader_texture_lod) && !defined(GL_ARB_shader_texture_lod)
#define texture2DLod(img, coord, lod) texture2D(img, coord, lod)
#define textureCubeLod(img, coord, lod) textureCube(img, coord, lod)
#endif

#ifdef USE_GLES_OVER_GL
#define lowp
#define mediump
#define highp
#else
#if defined(USE_HIGHP_PRECISION)
precision highp float;
precision highp int;
#else
precision mediump float;
precision mediump int;
#endif
#endif

#include "stdlib.glsl"

uniform sampler2D color_texture; // texunit:-1
/* clang-format on */
uniform highp vec2 color_texpixel_size;
uniform mediump sampler2D normal_texture; // texunit:-2

varying mediump vec2 uv_interp;
varying mediump vec4 color_interp;

#ifdef USE_ATTRIB_MODULATE
varying mediump vec4 modulate_interp;
#endif

uniform highp float time;

uniform vec4 final_modulate;

#ifdef SCREEN_TEXTURE_USED

uniform sampler2D screen_texture; // texunit:-4

#endif

#ifdef SCREEN_UV_USED

uniform vec2 screen_pixel_size;

#endif

#ifdef USE_LIGHTING

uniform highp mat4 light_matrix;
uniform highp mat4 light_local_matrix;
uniform highp mat4 shadow_matrix;
uniform highp vec4 light_color;
uniform highp vec4 light_shadow_color;
uniform highp vec2 light_pos;
uniform highp float shadowpixel_size;
uniform highp float shadow_gradient;
uniform highp float light_height;
uniform highp float light_outside_alpha;
uniform highp float shadow_distance_mult;

uniform lowp sampler2D light_texture; // texunit:-6
varying vec4 light_uv_interp;
varying vec2 transformed_light_uv;

varying vec4 local_rot;

#ifdef USE_SHADOWS

uniform highp sampler2D shadow_texture; // texunit:-5
varying highp vec2 pos;

#endif

const bool at_light_pass = true;
#else
const bool at_light_pass = false;
#endif

uniform bool use_default_normal;

/* clang-format off */

FRAGMENT_SHADER_GLOBALS

/* clang-format on */

void light_compute(
		inout vec4 light,
		inout vec2 light_vec,
		inout float light_height,
		inout vec4 light_color,
		vec2 light_uv,
		inout vec4 shadow_color,
		inout vec2 shadow_vec,
		vec3 normal,
		vec2 uv,
#if defined(SCREEN_UV_USED)
		vec2 screen_uv,
#endif
		vec4 color) {

#if defined(USE_LIGHT_SHADER_CODE)

	/* clang-format off */

LIGHT_SHADER_CODE

	/* clang-format on */

#endif
}

void main() {

	vec4 color = color_interp;
	vec2 uv = uv_interp;
#ifdef USE_FORCE_REPEAT
	//needs to use this to workaround GLES2/WebGL1 forcing tiling that textures that don't support it
	uv = mod(uv, vec2(1.0, 1.0));
#endif

#if !defined(COLOR_USED)
	//default behavior, texture by color
	color *= texture2D(color_texture, uv);
#endif

#ifdef SCREEN_UV_USED
	vec2 screen_uv = gl_FragCoord.xy * screen_pixel_size;
#endif

	vec3 normal;

#if defined(NORMAL_USED)

	bool normal_used = true;
#else
	bool normal_used = false;
#endif

	if (use_default_normal) {
		normal.xy = texture2D(normal_texture, uv).xy * 2.0 - 1.0;
		normal.z = sqrt(1.0 - dot(normal.xy, normal.xy));
		normal_used = true;
	} else {
		normal = vec3(0.0, 0.0, 1.0);
	}

	{
		float normal_depth = 1.0;

#if defined(NORMALMAP_USED)
		vec3 normal_map = vec3(0.0, 0.0, 1.0);
		normal_used = true;
#endif

		/* clang-format off */

FRAGMENT_SHADER_CODE

		/* clang-format on */

#if defined(NORMALMAP_USED)
		normal = mix(vec3(0.0, 0.0, 1.0), normal_map * vec3(2.0, -2.0, 1.0) - vec3(1.0, -1.0, 0.0), normal_depth);
#endif
	}

#ifdef USE_ATTRIB_MODULATE
	color *= modulate_interp;
#else
#if !defined(MODULATE_USED)
	color *= final_modulate;
#endif
#endif

#ifdef USE_LIGHTING

	vec2 light_vec = transformed_light_uv;
	vec2 shadow_vec = transformed_light_uv;

	if (normal_used) {
		normal.xy = mat2(local_rot.xy, local_rot.zw) * normal.xy;
	}

	float att = 1.0;

	vec2 light_uv = light_uv_interp.xy;
	vec4 light = texture2D(light_texture, light_uv);

	if (any(lessThan(light_uv_interp.xy, vec2(0.0, 0.0))) || any(greaterThanEqual(light_uv_interp.xy, vec2(1.0, 1.0)))) {
		color.a *= light_outside_alpha; //invisible

	} else {
		float real_light_height = light_height;
		vec4 real_light_color = light_color;
		vec4 real_light_shadow_color = light_shadow_color;

#if defined(USE_LIGHT_SHADER_CODE)
		//light is written by the light shader
		light_compute(
				light,
				light_vec,
				real_light_height,
				real_light_color,
				light_uv,
				real_light_shadow_color,
				shadow_vec,
				normal,
				uv,
#if defined(SCREEN_UV_USED)
				screen_uv,
#endif
				color);
#endif

		light *= real_light_color;

		if (normal_used) {
			vec3 light_normal = normalize(vec3(light_vec, -real_light_height));
			light *= max(dot(-light_normal, normal), 0.0);
		}

		color *= light;

#ifdef USE_SHADOWS

#ifdef SHADOW_VEC_USED
		mat3 inverse_light_matrix = mat3(light_matrix);
		inverse_light_matrix[0] = normalize(inverse_light_matrix[0]);
		inverse_light_matrix[1] = normalize(inverse_light_matrix[1]);
		inverse_light_matrix[2] = normalize(inverse_light_matrix[2]);
		shadow_vec = (inverse_light_matrix * vec3(shadow_vec, 0.0)).xy;
#else
		shadow_vec = light_uv_interp.zw;
#endif

		float angle_to_light = -atan(shadow_vec.x, shadow_vec.y);
		float PI = 3.14159265358979323846264;
		/*int i = int(mod(floor((angle_to_light+7.0*PI/6.0)/(4.0*PI/6.0))+1.0, 3.0)); // +1 pq os indices estao em ordem 2,0,1 nos arrays
		float ang*/

		float su, sz;

		float abs_angle = abs(angle_to_light);
		vec2 point;
		float sh;
		if (abs_angle < 45.0 * PI / 180.0) {
			point = shadow_vec;
			sh = 0.0 + (1.0 / 8.0);
		} else if (abs_angle > 135.0 * PI / 180.0) {
			point = -shadow_vec;
			sh = 0.5 + (1.0 / 8.0);
		} else if (angle_to_light > 0.0) {

			point = vec2(shadow_vec.y, -shadow_vec.x);
			sh = 0.25 + (1.0 / 8.0);
		} else {

			point = vec2(-shadow_vec.y, shadow_vec.x);
			sh = 0.75 + (1.0 / 8.0);
		}

		highp vec4 s = shadow_matrix * vec4(point, 0.0, 1.0);
		s.xyz /= s.w;
		su = s.x * 0.5 + 0.5;
		sz = s.z * 0.5 + 0.5;
		//sz=lightlength(light_vec);

		highp float shadow_attenuation = 0.0;

#ifdef USE_RGBA_SHADOWS
#define SHADOW_DEPTH(m_tex, m_uv) dot(texture2D((m_tex), (m_uv)), vec4(1.0 / (255.0 * 255.0 * 255.0), 1.0 / (255.0 * 255.0), 1.0 / 255.0, 1.0))

#else

#define SHADOW_DEPTH(m_tex, m_uv) (texture2D((m_tex), (m_uv)).r)

#endif

#ifdef SHADOW_USE_GRADIENT

		/* clang-format off */
		/* GLSL es 100 doesn't support line continuation characters(backslashes) */
#define SHADOW_TEST(m_ofs) { highp float sd = SHADOW_DEPTH(shadow_texture, vec2(m_ofs, sh)); shadow_attenuation += 1.0 - smoothstep(sd, sd + shadow_gradient, sz); }

#else

#define SHADOW_TEST(m_ofs) { highp float sd = SHADOW_DEPTH(shadow_texture, vec2(m_ofs, sh)); shadow_attenuation += step(sz, sd); }
		/* clang-format on */

#endif

#ifdef SHADOW_FILTER_NEAREST

		SHADOW_TEST(su);

#endif

#ifdef SHADOW_FILTER_PCF3

		SHADOW_TEST(su + shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su - shadowpixel_size);
		shadow_attenuation /= 3.0;

#endif

#ifdef SHADOW_FILTER_PCF5

		SHADOW_TEST(su + shadowpixel_size * 2.0);
		SHADOW_TEST(su + shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su - shadowpixel_size);
		SHADOW_TEST(su - shadowpixel_size * 2.0);
		shadow_attenuation /= 5.0;

#endif

#ifdef SHADOW_FILTER_PCF7

		SHADOW_TEST(su + shadowpixel_size * 3.0);
		SHADOW_TEST(su + shadowpixel_size * 2.0);
		SHADOW_TEST(su + shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su - shadowpixel_size);
		SHADOW_TEST(su - shadowpixel_size * 2.0);
		SHADOW_TEST(su - shadowpixel_size * 3.0);
		shadow_attenuation /= 7.0;

#endif

#ifdef SHADOW_FILTER_PCF9

		SHADOW_TEST(su + shadowpixel_size * 4.0);
		SHADOW_TEST(su + shadowpixel_size * 3.0);
		SHADOW_TEST(su + shadowpixel_size * 2.0);
		SHADOW_TEST(su + shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su - shadowpixel_size);
		SHADOW_TEST(su - shadowpixel_size * 2.0);
		SHADOW_TEST(su - shadowpixel_size * 3.0);
		SHADOW_TEST(su - shadowpixel_size * 4.0);
		shadow_attenuation /= 9.0;

#endif

#ifdef SHADOW_FILTER_PCF13

		SHADOW_TEST(su + shadowpixel_size * 6.0);
		SHADOW_TEST(su + shadowpixel_size * 5.0);
		SHADOW_TEST(su + shadowpixel_size * 4.0);
		SHADOW_TEST(su + shadowpixel_size * 3.0);
		SHADOW_TEST(su + shadowpixel_size * 2.0);
		SHADOW_TEST(su + shadowpixel_size);
		SHADOW_TEST(su);
		SHADOW_TEST(su - shadowpixel_size);
		SHADOW_TEST(su - shadowpixel_size * 2.0);
		SHADOW_TEST(su - shadowpixel_size * 3.0);
		SHADOW_TEST(su - shadowpixel_size * 4.0);
		SHADOW_TEST(su - shadowpixel_size * 5.0);
		SHADOW_TEST(su - shadowpixel_size * 6.0);
		shadow_attenuation /= 13.0;

#endif

		//color *= shadow_attenuation;
		color = mix(real_light_shadow_color, color, shadow_attenuation);
//use shadows
#endif
	}

//use lighting
#endif

	gl_FragColor = color;
}
