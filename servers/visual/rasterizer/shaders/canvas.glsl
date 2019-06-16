/* clang-format off */
[vertex]
/* clang-format on */

#version 450

/* clang-format off */
VERSION_DEFINES
/* clang-format on */

#ifdef USE_VERTEX_ARRAYS
layout(location = 0) in vec2 vertex_attrib;
layout(location = 3) in vec4 color_attrib;
layout(location = 4) in vec2 uv_attrib;

layout(location = 6) in uvec4 bone_indices_attrib;
layout(location = 7) in vec4 bone_weights_attrib;
#endif

#include "canvas_uniforms_inc.glsl"


layout(location=0) out vec2 uv_interp;
layout(location=1) out vec4 color_interp;

#ifdef USE_NINEPATCH

layout(location=3) out vec2 pixel_size_interp;

#endif

/* clang-format off */
MATERIAL_UNIFORMS
/* clang-format on */


/* clang-format off */
VERTEX_SHADER_GLOBALS
/* clang-format on */


void main() {

	vec4 instance_custom = vec4(0.0);

#ifdef USE_VERTEX_ARRAYS

	vec2 vertex = vertex_attrib;
	vec4 color = color_attrib;
	vec2 uv = uv_attrib;
	uvec4 bone_indices = bone_indices_attrib;
	vec4 bone_weights = bone_weights_attrib;
#else

	vec2 vertex_base_arr[4] = vec2[](vec2(0.0,0.0),vec2(0.0,1.0),vec2(1.0,1.0),vec2(1.0,0.0));
	vec2 vertex_base = vertex_base_arr[gl_VertexIndex];

	vec2 uv = draw_data.src_rect.xy + draw_data.src_rect.zw * ((draw_data.flags&FLAGS_TRANSPOSE_RECT)!=0 ? vertex_base.xy : vertex_base.yx);
	vec4 color = vec4(1.0);
	vec2 vertex = draw_data.dst_rect.xy + abs(draw_data.dst_rect.zw) * mix(vertex_base, vec2(1.0, 1.0) - vertex_base, lessThan(draw_data.src_rect.zw, vec2(0.0, 0.0)));
	uvec4 bone_indices = uvec4(0,0,0,0);
	vec4 bone_weights = vec4(0,0,0,0);

#endif

	mat4 world_matrix  = mat4(draw_data.world[0],draw_data.world[1],vec4(0.0,0.0,1.0,0.0),vec4(0.0,0.0,0.0,1.0));
#if 0
	if (draw_data.flags&FLAGS_INSTANCING_ENABLED) {

		uint offset = draw_data.flags&FLAGS_INSTANCING_STRIDE_MASK;
		offset *= gl_InstanceIndex;
		mat4 instance_xform  = mat4(
					vec4( texelFetch(instancing_buffer,offset+0),texelFetch(instancing_buffer,offset+1),0.0,texelFetch(instancing_buffer,offset+3) ),
					vec4( texelFetch(instancing_buffer,offset+4),texelFetch(instancing_buffer,offset+5),0.0,texelFetch(instancing_buffer,offset+7) ),
					vec4( 0.0,0.0,1.0,0.0),
					vec4( 0.0,0.0,0.0,1.0 ) );
		offset+=8;
		if ( draw_data.flags&FLAGS_INSTANCING_HAS_COLORS ) {
			vec4 instance_color;
			if (draw_data.flags&FLAGS_INSTANCING_COLOR_8_BIT ) {
				uint bits = floatBitsToUint(texelFetch(instancing_buffer,offset));
				instance_color = unpackUnorm4x8(bits);
				offset+=1;
			} else {
				instance_color = vec4(texelFetch(instancing_buffer,offset+0),texelFetch(instancing_buffer,offset+1),texelFetch(instancing_buffer,offset+2),texelFetch(instancing_buffer,offset+3));
				offser+=4;
			}

			color*=instance_color;
		}
		if ( draw_data.flags&FLAGS_INSTANCING_HAS_CUSTOM_DATA ) {
			if (draw_data.flags&FLAGS_INSTANCING_CUSTOM_DATA_8_BIT ) {
				uint bits = floatBitsToUint(texelFetch(instancing_buffer,offset));
				instance_custom = unpackUnorm4x8(bits);
			} else {
				instance_custom = vec4(texelFetch(instancing_buffer,offset+0),texelFetch(instancing_buffer,offset+1),texelFetch(instancing_buffer,offset+2),texelFetch(instancing_buffer,offset+3));
			}
		}

	}

#endif

	if (bool(draw_data.flags&FLAGS_USING_PARTICLES)) {
		//scale by texture size
		vertex /= draw_data.color_texture_pixel_size;
	}
#ifdef USE_POINT_SIZE
	float point_size = 1.0;
#endif
	{
		/* clang-format off */
VERTEX_SHADER_CODE
		/* clang-format on */
	}



#ifdef USE_NINEPATCH
	pixel_size_interp = abs(draw_data.dst_rect.zw) * vertex;
#endif

#if !defined(SKIP_TRANSFORM_USED)
	vertex = (world_matrix * vec4(vertex,0.0,1.0)).xy;
#endif

	color_interp = color;

	if (bool(draw_data.flags&FLAGS_USE_PIXEL_SNAP)) {

		vertex = floor(vertex + 0.5);
		// precision issue on some hardware creates artifacts within texture
		// offset uv by a small amount to avoid
		uv += 1e-5;
	}

#if 0
	if (bool(draw_data.flags&FLAGS_USE_SKELETON) && bone_weights != vec4(0.0)) { //must be a valid bone
		//skeleton transform

		ivec4 bone_indicesi = ivec4(bone_indices);

		uvec2 tex_ofs = bone_indicesi.x *2;

		mat2x4 m;
		m = mat2x4(
					texelFetch(skeleton_buffer, tex_ofs+0),
					texelFetch(skeleton_buffer, tex_ofs+1) ) *
			bone_weights.x;

		tex_ofs = bone_indicesi.y * 2;

		m += mat2x4(
					texelFetch(skeleton_buffer, tex_ofs+0),
					texelFetch(skeleton_buffer, tex_ofs+1) ) *
			 bone_weights.y;

		tex_ofs = bone_indicesi.z * 2;

		m += mat2x4(
					texelFetch(skeleton_buffer, tex_ofs+0),
					texelFetch(skeleton_buffer, tex_ofs+1) ) *
			 bone_weights.z;

		tex_ofs = bone_indicesi.w * 2;

		m += mat2x4(
					texelFetch(skeleton_buffer, tex_ofs+0),
					texelFetch(skeleton_buffer, tex_ofs+1) ) *
			 bone_weights.w;

		mat4 bone_matrix = skeleton_data.skeleton_transform * transpose(mat4(m[0], m[1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0))) * skeleton_data.skeleton_transform_inverse;

		//outvec = bone_matrix * outvec;
	}
#endif

#if !defined(SKIP_TRANSFORM_USED)
	gl_Position = (canvas_data.screen_transform * canvas_data.canvas_transform) * vec4(vertex,0.0,1.0);
#else
	gl_Position = vec4(vertex,0.0,1.0);
#endif

#ifdef USE_POINT_SIZE
	gl_PointSize=point_size;
#endif

}

/* clang-format off */
[fragment]

#version 450

/* clang-format off */
VERSION_DEFINES
/* clang-format on */

#include "canvas_uniforms_inc.glsl"

layout(location=0) in vec2 uv_interp;
layout(location=1) in vec4 color_interp;

#ifdef USE_NINEPATCH

layout(location=3) in vec2 pixel_size_interp;

#endif

layout(location = 0) out vec4 frag_color;

/* clang-format off */
MATERIAL_UNIFORMS
/* clang-format on */


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
		vec3 normal,
		vec2 uv,
		vec2 screen_uv,
		vec4 color) {

	/* clang-format off */
LIGHT_SHADER_CODE
	/* clang-format on */
}


#ifdef USE_NINEPATCH

float map_ninepatch_axis(float pixel, float draw_size, float tex_pixel_size, float margin_begin, float margin_end, int np_repeat, inout int draw_center) {

	float tex_size = 1.0 / tex_pixel_size;

	if (pixel < margin_begin) {
		return pixel * tex_pixel_size;
	} else if (pixel >= draw_size - margin_end) {
		return (tex_size - (draw_size - pixel)) * tex_pixel_size;
	} else {
		if (!bool(draw_data.flags&FLAGS_NINEPACH_DRAW_CENTER)) {
			draw_center--;
		}

		if (np_repeat == 0) { //stretch
			//convert to ratio
			float ratio = (pixel - margin_begin) / (draw_size - margin_begin - margin_end);
			//scale to source texture
			return (margin_begin + ratio * (tex_size - margin_begin - margin_end)) * tex_pixel_size;
		} else if (np_repeat == 1) { //tile
			//convert to ratio
			float ofs = mod((pixel - margin_begin), tex_size - margin_begin - margin_end);
			//scale to source texture
			return (margin_begin + ofs) * tex_pixel_size;
		} else if (np_repeat == 2) { //tile fit
			//convert to ratio
			float src_area = draw_size - margin_begin - margin_end;
			float dst_area = tex_size - margin_begin - margin_end;
			float scale = max(1.0, floor(src_area / max(dst_area, 0.0000001) + 0.5));

			//convert to ratio
			float ratio = (pixel - margin_begin) / src_area;
			ratio = mod(ratio * scale, 1.0);
			return (margin_begin + ratio * dst_area) * tex_pixel_size;
		}
	}
}


#endif

void main() {

	vec4 color = color_interp;
	vec2 uv = uv_interp;

#ifdef USE_TEXTURE_RECT

#ifdef USE_NINEPATCH

	int draw_center = 2;
	uv = vec2(
			map_ninepatch_axis(pixel_size_interp.x, abs(draw_data.dst_rect.z), draw_data.color_texture_pixel_size.x, draw_data.ninepatch_margins.x, draw_data.ninepatch_margins.z, (draw_data.ninepatch_repeat>>16), draw_center),
			map_ninepatch_axis(pixel_size_interp.y, abs(draw_data.dst_rect.w), draw_data.color_texture_pixel_size.y, draw_data.ninepatch_margins.y, draw_data.ninepatch_margins.w, (draw_data.ninepatch_repeat&0xFFFF), draw_center));

	if (draw_center == 0) {
		color.a = 0.0;
	}

	uv = uv * draw_data.src_rect.zw + draw_data.src_rect.xy; //apply region if needed
#endif

	if (bool(draw_data.flags&FLAGS_CLIP_RECT_UV)) {

		uv = clamp(uv, draw_data.src_rect.xy, draw_data.src_rect.xy + abs(draw_data.src_rect.zw));
	}

#endif

#if !defined(COLOR_USED)
	//default behavior, texture by color

	color *= texture(sampler2D(color_texture,texture_sampler), uv);

#endif



	vec3 normal;

#if defined(NORMAL_USED)

	bool normal_used = true;
#else
	bool normal_used = false;
#endif

#if 0
	if (false /*normal_used || canvas_data.light_count > 0*/ ) {
		normal.xy = texture(sampler2D(normal_texture,texture_sampler	), uv).xy * 2.0 - 1.0;
		normal.z = sqrt(1.0 - dot(normal.xy, normal.xy));
		normal_used = true;
	} else {
#endif
		normal = vec3(0.0, 0.0, 1.0);
#if 0
	}
#endif

#if defined(SCREEN_UV_USED)
	vec2 screen_uv = gl_FragCoord.xy * screen_pixel_size;
#endif

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

	color *= draw_data.modulation;
#if 0
	if (canvas_data.light_count > 0 ) {
		//do lighting

	}
#endif
	//color.rgb *= color.a;
	frag_color = color;
}
