/* clang-format off */
[vertex]

layout(location = 0) in highp vec2 vertex;

#ifdef USE_ATTRIB_LIGHT_ANGLE
layout(location = 2) in highp float light_angle;
#endif

/* clang-format on */
layout(location = 3) in vec4 color_attrib;

#ifdef USE_ATTRIB_MODULATE
layout(location = 5) in vec4 modulate_attrib; // attrib:5
#endif

// Usually, final_modulate is passed as a uniform. However during batching
// If larger fvfs are used, final_modulate is passed as an attribute.
// we need to read from the attribute in custom vertex shader
// rather than the uniform. We do this by specifying final_modulate_alias
// in shaders rather than final_modulate directly.
#ifdef USE_ATTRIB_MODULATE
#define final_modulate_alias modulate_attrib
#else
#define final_modulate_alias final_modulate
#endif

#ifdef USE_ATTRIB_LARGE_VERTEX
// shared with skeleton attributes, not used in batched shader
layout(location = 6) in vec2 translate_attrib; // attrib:6
layout(location = 7) in vec4 basis_attrib; // attrib:7
#endif

#ifdef USE_SKELETON
layout(location = 6) in uvec4 bone_indices; // attrib:6
layout(location = 7) in vec4 bone_weights; // attrib:7
#endif

#ifdef USE_TEXTURE_RECT

uniform vec4 dst_rect;
uniform vec4 src_rect;

#else

#ifdef USE_INSTANCING

layout(location = 8) in highp vec4 instance_xform0;
layout(location = 9) in highp vec4 instance_xform1;
layout(location = 10) in highp vec4 instance_xform2;
layout(location = 11) in lowp vec4 instance_color;

#ifdef USE_INSTANCE_CUSTOM
layout(location = 12) in highp vec4 instance_custom_data;
#endif

#endif

layout(location = 4) in highp vec2 uv_attrib;

// skeleton
#endif

uniform highp vec2 color_texpixel_size;

layout(std140) uniform CanvasItemData { //ubo:0

	highp mat4 projection_matrix;
	highp float time;
};

uniform highp mat4 modelview_matrix;
uniform highp mat4 extra_matrix;

out highp vec2 uv_interp;
out mediump vec4 color_interp;

#ifdef USE_ATTRIB_MODULATE
// modulate doesn't need interpolating but we need to send it to the fragment shader
flat out mediump vec4 modulate_interp;
#endif

#ifdef MODULATE_USED
uniform mediump vec4 final_modulate;
#endif

#ifdef USE_NINEPATCH

out highp vec2 pixel_size_interp;
#endif

#ifdef USE_SKELETON
uniform mediump sampler2D skeleton_texture; // texunit:-4
uniform highp mat4 skeleton_transform;
uniform highp mat4 skeleton_transform_inverse;
#endif

#ifdef USE_LIGHTING

layout(std140) uniform LightData { //ubo:1

	// light matrices
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
out vec2 transformed_light_uv;

out vec4 local_rot;

#ifdef USE_SHADOWS
out highp vec2 pos;
#endif

const bool at_light_pass = true;
#else
const bool at_light_pass = false;
#endif

#if defined(USE_MATERIAL)

/* clang-format off */
layout(std140) uniform UniformData { //ubo:2

MATERIAL_UNIFORMS

};
/* clang-format on */

#endif

/* clang-format off */

VERTEX_SHADER_GLOBALS

/* clang-format on */

void main() {
	vec4 color = color_attrib;

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
		uv_interp = src_rect.xy + abs(src_rect.zw) * vertex.yx;
	} else {
		uv_interp = src_rect.xy + abs(src_rect.zw) * vertex;
	}
	highp vec4 outvec = vec4(dst_rect.xy + abs(dst_rect.zw) * mix(vertex, vec2(1.0, 1.0) - vertex, lessThan(src_rect.zw, vec2(0.0, 0.0))), 0.0, 1.0);

#else
	uv_interp = uv_attrib;
	highp vec4 outvec = vec4(vertex, 0.0, 1.0);
#endif

#ifdef USE_PARTICLES
	//scale by texture size
	outvec.xy /= color_texpixel_size;
#endif

#define extra_matrix extra_matrix_instance

	float point_size = 1.0;
	//for compatibility with the fragment shader we need to use uv here
	vec2 uv = uv_interp;
	{
		/* clang-format off */

VERTEX_SHADER_CODE

		/* clang-format on */
	}

	gl_PointSize = point_size;
	uv_interp = uv;

#ifdef USE_NINEPATCH

	pixel_size_interp = abs(dst_rect.zw) * vertex;
#endif

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
	outvec = extra_matrix * outvec;
	outvec = modelview_matrix * outvec;
#endif

#endif // not large integer

#undef extra_matrix

	color_interp = color;

#ifdef USE_PIXEL_SNAP
	outvec.xy = floor(outvec + 0.5).xy;
	// precision issue on some hardware creates artifacts within texture
	// offset uv by a small amount to avoid
	uv_interp += 1e-5;
#endif

#ifdef USE_SKELETON

	if (bone_weights != vec4(0.0)) { //must be a valid bone
		//skeleton transform

		ivec4 bone_indicesi = ivec4(bone_indices);

		ivec2 tex_ofs = ivec2(bone_indicesi.x % 256, (bone_indicesi.x / 256) * 2);

		highp mat2x4 m;
		m = mat2x4(
					texelFetch(skeleton_texture, tex_ofs, 0),
					texelFetch(skeleton_texture, tex_ofs + ivec2(0, 1), 0)) *
				bone_weights.x;

		tex_ofs = ivec2(bone_indicesi.y % 256, (bone_indicesi.y / 256) * 2);

		m += mat2x4(
					 texelFetch(skeleton_texture, tex_ofs, 0),
					 texelFetch(skeleton_texture, tex_ofs + ivec2(0, 1), 0)) *
				bone_weights.y;

		tex_ofs = ivec2(bone_indicesi.z % 256, (bone_indicesi.z / 256) * 2);

		m += mat2x4(
					 texelFetch(skeleton_texture, tex_ofs, 0),
					 texelFetch(skeleton_texture, tex_ofs + ivec2(0, 1), 0)) *
				bone_weights.z;

		tex_ofs = ivec2(bone_indicesi.w % 256, (bone_indicesi.w / 256) * 2);

		m += mat2x4(
					 texelFetch(skeleton_texture, tex_ofs, 0),
					 texelFetch(skeleton_texture, tex_ofs + ivec2(0, 1), 0)) *
				bone_weights.w;

		mat4 bone_matrix = skeleton_transform * transpose(mat4(m[0], m[1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0))) * skeleton_transform_inverse;

		outvec = bone_matrix * outvec;
	}

#endif

	gl_Position = projection_matrix * outvec;

#ifdef USE_LIGHTING

	light_uv_interp.xy = (light_matrix * outvec).xy;
	light_uv_interp.zw = (light_local_matrix * outvec).xy;

	mat3 inverse_light_matrix = mat3(inverse(light_matrix));
	inverse_light_matrix[0] = normalize(inverse_light_matrix[0]);
	inverse_light_matrix[1] = normalize(inverse_light_matrix[1]);
	inverse_light_matrix[2] = normalize(inverse_light_matrix[2]);
	transformed_light_uv = (inverse_light_matrix * vec3(light_uv_interp.zw, 0.0)).xy; //for normal mapping

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

uniform mediump sampler2D color_texture; // texunit:0
/* clang-format on */
uniform highp vec2 color_texpixel_size;
uniform mediump sampler2D normal_texture; // texunit:1

in highp vec2 uv_interp;
in mediump vec4 color_interp;

#ifdef USE_ATTRIB_MODULATE
flat in mediump vec4 modulate_interp;
#endif

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
in vec2 transformed_light_uv;

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

layout(location = 0) out mediump vec4 frag_color;

#if defined(USE_MATERIAL)

/* clang-format off */
layout(std140) uniform UniformData {

MATERIAL_UNIFORMS

};
/* clang-format on */

#endif

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

#ifdef USE_TEXTURE_RECT

uniform vec4 dst_rect;
uniform vec4 src_rect;
uniform bool clip_rect_uv;

#ifdef USE_NINEPATCH

in highp vec2 pixel_size_interp;

uniform int np_repeat_v;
uniform int np_repeat_h;
uniform bool np_draw_center;
// left top right bottom in pixel coordinates
uniform vec4 np_margins;

// there are two ninepatch modes, and we don't want to waste a conditional
#if defined USE_NINEPATCH_SCALING
float map_ninepatch_axis(float pixel, float draw_size, float tex_pixel_size, float margin_begin, float margin_end, float s_ratio, int np_repeat, inout int draw_center) {
	float tex_size = 1.0 / tex_pixel_size;

	float screen_margin_begin = margin_begin / s_ratio;
	float screen_margin_end = margin_end / s_ratio;
	if (pixel < screen_margin_begin) {
		return pixel * s_ratio * tex_pixel_size;
	} else if (pixel >= draw_size - screen_margin_end) {
		return (tex_size - (draw_size - pixel) * s_ratio) * tex_pixel_size;
	} else {
		if (!np_draw_center) {
			draw_center--;
		}

		if (np_repeat == 0) { //stretch
			//convert to ratio
			float ratio = (pixel - screen_margin_begin) / (draw_size - screen_margin_begin - screen_margin_end);
			//scale to source texture
			return (margin_begin + ratio * (tex_size - margin_begin - margin_end)) * tex_pixel_size;
		} else if (np_repeat == 1) { //tile
			//convert to ratio
			float ofs = mod((pixel - screen_margin_begin), tex_size - margin_begin - margin_end);
			//scale to source texture
			return (margin_begin + ofs) * tex_pixel_size;
		} else if (np_repeat == 2) { //tile fit
			//convert to ratio
			float src_area = draw_size - screen_margin_begin - screen_margin_end;
			float dst_area = tex_size - margin_begin - margin_end;
			float scale = max(1.0, floor(src_area / max(dst_area, 0.0000001) + 0.5));

			//convert to ratio
			float ratio = (pixel - screen_margin_begin) / src_area;
			ratio = mod(ratio * scale, 1.0);
			return (margin_begin + ratio * dst_area) * tex_pixel_size;
		}
	}
}
#else
float map_ninepatch_axis(float pixel, float draw_size, float tex_pixel_size, float margin_begin, float margin_end, int np_repeat, inout int draw_center) {
	float tex_size = 1.0 / tex_pixel_size;

	if (pixel < margin_begin) {
		return pixel * tex_pixel_size;
	} else if (pixel >= draw_size - margin_end) {
		return (tex_size - (draw_size - pixel)) * tex_pixel_size;
	} else {
		if (!np_draw_center) {
			draw_center--;
		}

		// np_repeat is passed as uniform using NinePatchRect::AxisStretchMode enum.
		if (np_repeat == 0) { // Stretch.
			// Convert to ratio.
			float ratio = (pixel - margin_begin) / (draw_size - margin_begin - margin_end);
			// Scale to source texture.
			return (margin_begin + ratio * (tex_size - margin_begin - margin_end)) * tex_pixel_size;
		} else if (np_repeat == 1) { // Tile.
			// Convert to offset.
			float ofs = mod((pixel - margin_begin), tex_size - margin_begin - margin_end);
			// Scale to source texture.
			return (margin_begin + ofs) * tex_pixel_size;
		} else if (np_repeat == 2) { // Tile Fit.
			// Calculate scale.
			float src_area = draw_size - margin_begin - margin_end;
			float dst_area = tex_size - margin_begin - margin_end;
			float scale = max(1.0, floor(src_area / max(dst_area, 0.0000001) + 0.5));
			// Convert to ratio.
			float ratio = (pixel - margin_begin) / src_area;
			ratio = mod(ratio * scale, 1.0);
			// Scale to source texture.
			return (margin_begin + ratio * dst_area) * tex_pixel_size;
		} else { // Shouldn't happen, but silences compiler warning.
			return 0.0;
		}
	}
}
#endif

#endif
#endif

uniform bool use_default_normal;

void main() {
	vec4 color = color_interp;
	vec2 uv = uv_interp;

#ifdef USE_TEXTURE_RECT

#ifdef USE_NINEPATCH

	int draw_center = 2;
#if defined USE_NINEPATCH_SCALING
	float s_ratio = max((1.0 / color_texpixel_size.x) / abs(dst_rect.z), (1.0 / color_texpixel_size.y) / abs(dst_rect.w));
	s_ratio = max(1.0, s_ratio);
	uv = vec2(
			map_ninepatch_axis(pixel_size_interp.x, abs(dst_rect.z), color_texpixel_size.x, np_margins.x, np_margins.z, s_ratio, np_repeat_h, draw_center),
			map_ninepatch_axis(pixel_size_interp.y, abs(dst_rect.w), color_texpixel_size.y, np_margins.y, np_margins.w, s_ratio, np_repeat_v, draw_center));

	if (draw_center == 0) {
		color.a = 0.0;
	}
#else
	uv = vec2(
			map_ninepatch_axis(pixel_size_interp.x, abs(dst_rect.z), color_texpixel_size.x, np_margins.x, np_margins.z, np_repeat_h, draw_center),
			map_ninepatch_axis(pixel_size_interp.y, abs(dst_rect.w), color_texpixel_size.y, np_margins.y, np_margins.w, np_repeat_v, draw_center));

	if (draw_center == 0) {
		color.a = 0.0;
	}
#endif
	uv = uv * src_rect.zw + src_rect.xy; //apply region if needed
#endif

	if (clip_rect_uv) {
		uv = clamp(uv, src_rect.xy, src_rect.xy + abs(src_rect.zw));
	}

#endif

#if !defined(COLOR_USED)
	//default behavior, texture by color

#ifdef USE_DISTANCE_FIELD
	const float smoothing = 1.0 / 32.0;
	float distance = textureLod(color_texture, uv, 0.0).a;
	color.a = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance) * color.a;
#else
	color *= texture(color_texture, uv);

#endif

#endif

	vec3 normal;

#if defined(NORMAL_USED)

	bool normal_used = true;
#else
	bool normal_used = false;
#endif

	if (use_default_normal) {
		normal.xy = textureLod(normal_texture, uv, 0.0).xy * 2.0 - 1.0;
		normal.z = sqrt(max(0.0, 1.0 - dot(normal.xy, normal.xy)));
		normal_used = true;
	} else {
		normal = vec3(0.0, 0.0, 1.0);
	}

#if defined(SCREEN_UV_USED)
	vec2 screen_uv = gl_FragCoord.xy * screen_pixel_size;
#endif

	{
		float normal_depth = 1.0;

#if defined(NORMALMAP_USED)
		vec3 normal_map = vec3(0.0, 0.0, 1.0);
		normal_used = true;
#endif

		// If larger fvfs are used, final_modulate is passed as an attribute.
		// we need to read from this in custom fragment shaders or applying in the post step,
		// rather than using final_modulate directly.
#if defined(final_modulate_alias)
#undef final_modulate_alias
#endif
#ifdef USE_ATTRIB_MODULATE
#define final_modulate_alias modulate_interp
#else
#define final_modulate_alias final_modulate
#endif

		/* clang-format off */

FRAGMENT_SHADER_CODE

		/* clang-format on */

#if defined(NORMALMAP_USED)
		normal = mix(vec3(0.0, 0.0, 1.0), normal_map * vec3(2.0, -2.0, 1.0) - vec3(1.0, -1.0, 0.0), normal_depth);
#endif
	}
#ifdef DEBUG_ENCODED_32
	highp float enc32 = dot(color, highp vec4(1.0 / (256.0 * 256.0 * 256.0), 1.0 / (256.0 * 256.0), 1.0 / 256.0, 1.0));
	color = vec4(vec3(enc32), 1.0);
#endif

#if !defined(MODULATE_USED)
	color *= final_modulate_alias;
#endif

#ifdef USE_LIGHTING

	vec2 light_vec = transformed_light_uv;
	vec2 shadow_vec = transformed_light_uv;

	if (normal_used) {
		normal.xy = mat2(local_rot.xy, local_rot.zw) * normal.xy;
	}

	float att = 1.0;

	vec2 light_uv = light_uv_interp.xy;
	vec4 light = texture(light_texture, light_uv);

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
		shadow_vec = (mat3(inverse_light_matrix) * vec3(shadow_vec, 0.0)).xy;
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

#define SHADOW_DEPTH(m_tex, m_uv) dot(texture((m_tex), (m_uv)), vec4(1.0 / (255.0 * 255.0 * 255.0), 1.0 / (255.0 * 255.0), 1.0 / 255.0, 1.0))

#else

#define SHADOW_DEPTH(m_tex, m_uv) (texture((m_tex), (m_uv)).r)

#endif

#ifdef SHADOW_USE_GRADIENT

#define SHADOW_TEST(m_ofs)                                                    \
	{                                                                         \
		highp float sd = SHADOW_DEPTH(shadow_texture, vec2(m_ofs, sh));       \
		shadow_attenuation += 1.0 - smoothstep(sd, sd + shadow_gradient, sz); \
	}

#else

#define SHADOW_TEST(m_ofs)                                              \
	{                                                                   \
		highp float sd = SHADOW_DEPTH(shadow_texture, vec2(m_ofs, sh)); \
		shadow_attenuation += step(sz, sd);                             \
	}

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

#ifdef LINEAR_TO_SRGB
	// regular Linear -> SRGB conversion
	vec3 a = vec3(0.055);
	color.rgb = mix((vec3(1.0) + a) * pow(color.rgb, vec3(1.0 / 2.4)) - a, 12.92 * color.rgb, lessThan(color.rgb, vec3(0.0031308)));
#endif

	//color.rgb *= color.a;
	frag_color = color;
}
