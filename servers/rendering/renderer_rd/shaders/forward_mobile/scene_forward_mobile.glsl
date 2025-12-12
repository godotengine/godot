#[vertex]

#version 450

#VERSION_DEFINES

/* Include half precision types. */
#include "../half_inc.glsl"

/* Include our forward mobile UBOs definitions etc. */
#include "scene_forward_mobile_inc.glsl"

#define SHADER_IS_SRGB false
#define SHADER_SPACE_FAR 0.0

#ifdef SHADOW_PASS
#define IN_SHADOW_PASS true
#else
#define IN_SHADOW_PASS false
#endif

/* INPUT ATTRIBS */

// Always contains vertex position in XYZ, can contain tangent angle in W.
layout(location = 0) in vec4 vertex_angle_attrib;

//only for pure render depth when normal is not used

#ifdef NORMAL_USED
// Contains Normal/Axis in RG, can contain tangent in BA.
layout(location = 1) in vec4 axis_tangent_attrib;
#endif

// Location 2 is unused.

#if defined(COLOR_USED)
layout(location = 3) in vec4 color_attrib;
#endif

#ifdef UV_USED
layout(location = 4) in vec2 uv_attrib;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP) || defined(MODE_RENDER_MATERIAL)
layout(location = 5) in vec2 uv2_attrib;
#endif // MODE_RENDER_MATERIAL

#if defined(CUSTOM0_USED)
layout(location = 6) in vec4 custom0_attrib;
#endif

#if defined(CUSTOM1_USED)
layout(location = 7) in vec4 custom1_attrib;
#endif

#if defined(CUSTOM2_USED)
layout(location = 8) in vec4 custom2_attrib;
#endif

#if defined(CUSTOM3_USED)
layout(location = 9) in vec4 custom3_attrib;
#endif

#if defined(BONES_USED) || defined(USE_PARTICLE_TRAILS)
layout(location = 10) in uvec4 bone_attrib;
#endif

#if defined(WEIGHTS_USED) || defined(USE_PARTICLE_TRAILS)
layout(location = 11) in vec4 weight_attrib;
#endif

#if defined(MODE_RENDER_MOTION_VECTORS)
layout(location = 12) in vec4 previous_vertex_attrib;

#if defined(NORMAL_USED) || defined(TANGENT_USED)
layout(location = 13) in vec4 previous_normal_attrib;
#endif

#endif // MODE_RENDER_MOTION_VECTORS

void axis_angle_to_tbn(vec3 axis, float angle, out vec3 tangent, out vec3 binormal, out vec3 normal) {
	float c = cos(angle);
	float s = sin(angle);
	vec3 omc_axis = (1.0 - c) * axis;
	vec3 s_axis = s * axis;
	tangent = omc_axis.xxx * axis + vec3(c, -s_axis.z, s_axis.y);
	binormal = omc_axis.yyy * axis + vec3(s_axis.z, c, -s_axis.x);
	normal = omc_axis.zzz * axis + vec3(-s_axis.y, s_axis.x, c);
}

/* Varyings */

layout(location = 0) out vec3 vertex_interp;

#ifdef NORMAL_USED
layout(location = 1) out vec3 normal_interp;
#endif

#if defined(COLOR_USED)
layout(location = 2) out vec4 color_interp;
#endif

#ifdef UV_USED
layout(location = 3) out vec2 uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 4) out vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(BENT_NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 5) out vec3 tangent_interp;
layout(location = 6) out vec3 binormal_interp;
#endif
#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED) && defined(USE_VERTEX_LIGHTING)
layout(location = 7) out vec4 diffuse_light_interp;
layout(location = 8) out vec4 specular_light_interp;

#include "../scene_forward_vertex_lights_inc.glsl"
#endif // !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED) && defined(USE_VERTEX_LIGHTING)
#ifdef MATERIAL_UNIFORMS_USED
/* clang-format off */
layout(set = MATERIAL_UNIFORM_SET, binding = 0, std140) uniform MaterialUniforms {
#MATERIAL_UNIFORMS
} material;
/* clang-format on */
#endif

#ifdef MODE_DUAL_PARABOLOID

layout(location = 9) out float dp_clip;

#endif

#if defined(MODE_RENDER_MOTION_VECTORS)
layout(location = 12) out highp vec4 screen_position;
layout(location = 13) out highp vec4 prev_screen_position;
#endif

#ifdef USE_MULTIVIEW
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
vec3 multiview_uv(vec2 uv) {
	return vec3(uv, ViewIndex);
}
ivec3 multiview_uv(ivec2 uv) {
	return ivec3(uv, int(ViewIndex));
}
#else // USE_MULTIVIEW
#define ViewIndex 0
vec2 multiview_uv(vec2 uv) {
	return uv;
}
ivec2 multiview_uv(ivec2 uv) {
	return uv;
}
#endif // !USE_MULTIVIEW

#if defined(POINT_SIZE_USED) && defined(POINT_COORD_USED)
layout(location = 14) out vec2 point_coord_interp;
#endif

invariant gl_Position;

#GLOBALS

#define scene_data scene_data_block.data

#ifdef USE_DOUBLE_PRECISION
// Helper functions for emulating double precision when adding floats.
vec3 quick_two_sum(vec3 a, vec3 b, out vec3 out_p) {
	vec3 s = a + b;
	out_p = b - (s - a);
	return s;
}

vec3 two_sum(vec3 a, vec3 b, out vec3 out_p) {
	vec3 s = a + b;
	vec3 v = s - a;
	out_p = (a - (s - v)) + (b - v);
	return s;
}

vec3 double_add_vec3(vec3 base_a, vec3 prec_a, vec3 base_b, vec3 prec_b, out vec3 out_precision) {
	vec3 s, t, se, te;
	s = two_sum(base_a, base_b, se);
	t = two_sum(prec_a, prec_b, te);
	se += t;
	s = quick_two_sum(s, se, se);
	se += te;
	s = quick_two_sum(s, se, out_precision);
	return s;
}
#endif

uint multimesh_stride() {
	uint stride = sc_multimesh_format_2d() ? 2 : 3;
	stride += sc_multimesh_has_color() ? 1 : 0;
	stride += sc_multimesh_has_custom_data() ? 1 : 0;
	return stride;
}

void _unpack_vertex_attributes(vec4 p_vertex_in, vec3 p_compressed_aabb_position, vec3 p_compressed_aabb_size,
#if defined(NORMAL_USED) || defined(TANGENT_USED)
		vec4 p_normal_in,
#ifdef NORMAL_USED
		out vec3 r_normal,
#endif
		out vec3 r_tangent,
		out vec3 r_binormal,
#endif
		out vec3 r_vertex) {

	r_vertex = p_vertex_in.xyz * p_compressed_aabb_size + p_compressed_aabb_position;
#ifdef NORMAL_USED
	r_normal = oct_to_vec3(p_normal_in.xy * 2.0 - 1.0);
#endif

#if defined(NORMAL_USED) || defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(BENT_NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	float binormal_sign;

	// This works because the oct value (0, 1) maps onto (0, 0, -1) which encodes to (1, 1).
	// Accordingly, if p_normal_in.z contains octahedral values, it won't equal (0, 1).
	if (p_normal_in.z > 0.0 || p_normal_in.w < 1.0) {
		// Uncompressed format.
		vec2 signed_tangent_attrib = p_normal_in.zw * 2.0 - 1.0;
		r_tangent = oct_to_vec3(vec2(signed_tangent_attrib.x, abs(signed_tangent_attrib.y) * 2.0 - 1.0));
		binormal_sign = sign(signed_tangent_attrib.y);
		r_binormal = normalize(cross(r_normal, r_tangent) * binormal_sign);
	} else {
		// Compressed format.
		float angle = p_vertex_in.w;
		binormal_sign = angle > 0.5 ? 1.0 : -1.0; // 0.5 does not exist in UNORM16, so values are either greater or smaller.
		angle = abs(angle * 2.0 - 1.0) * M_PI; // 0.5 is basically zero, allowing to encode both signs reliably.
		vec3 axis = r_normal;
		axis_angle_to_tbn(axis, angle, r_tangent, r_binormal, r_normal);
		r_binormal *= binormal_sign;
	}
#endif
}

void vertex_shader(in vec3 vertex,
#ifdef NORMAL_USED
		in vec3 normal_highp,
#endif
#if defined(NORMAL_USED) || defined(TANGENT_USED)
		in vec3 tangent_highp,
		in vec3 binormal_highp,
#endif
		in uint instance_index, in uint multimesh_offset, in mat3x4 in_model_matrix,
#ifdef USE_DOUBLE_PRECISION
		in vec3 model_precision,
		in vec3 view_precision,
#endif
#ifdef MODE_DUAL_PARABOLOID
		in float dual_paraboloid_side,
		in float z_far,
#endif
#if defined(MODE_RENDER_DEPTH) || defined(MODE_RENDER_MATERIAL)
		in uint scene_flags,
#endif
		in mat4 projection_matrix,
		in mat4 inv_projection_matrix,
#ifdef USE_MULTIVIEW
		in vec4 scene_eye_offset,
#endif
		in mat3x4 in_view_matrix,
		in mat3x4 in_inv_view_matrix,
		in vec2 viewport_size,
		in uint scene_directional_light_count,
		out vec4 screen_position_output) {
	vec4 instance_custom = vec4(0.0);
#if defined(COLOR_USED)
	vec4 color_highp = color_attrib;
#endif

	mat4 inv_view_matrix = transpose(mat4(in_inv_view_matrix[0],
			in_inv_view_matrix[1],
			in_inv_view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

	mat4 model_matrix = transpose(mat4(in_model_matrix[0],
			in_model_matrix[1],
			in_model_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

	mat3 model_normal_matrix;
	if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_NON_UNIFORM_SCALE)) {
		model_normal_matrix = transpose(inverse(mat3(model_matrix)));
	} else {
		model_normal_matrix = mat3(model_matrix);
	}

	mat4 matrix;
	mat4 read_model_matrix = model_matrix;

	if (sc_multimesh()) {
		//multimesh, instances are for it

#ifdef USE_PARTICLE_TRAILS
		uint trail_size = (instances.data[instance_index].flags >> INSTANCE_FLAGS_PARTICLE_TRAIL_SHIFT) & INSTANCE_FLAGS_PARTICLE_TRAIL_MASK;
		uint stride = 3 + 1 + 1; //particles always uses this format

		uint offset = trail_size * stride * INSTANCE_INDEX;

#ifdef COLOR_USED
		vec4 pcolor;
#endif
		{
			uint boffset = offset + bone_attrib.x * stride;
			matrix = mat4(transforms.data[boffset + 0], transforms.data[boffset + 1], transforms.data[boffset + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weight_attrib.x;
#ifdef COLOR_USED
			pcolor = transforms.data[boffset + 3] * weight_attrib.x;
#endif
		}
		if (weight_attrib.y > 0.001) {
			uint boffset = offset + bone_attrib.y * stride;
			matrix += mat4(transforms.data[boffset + 0], transforms.data[boffset + 1], transforms.data[boffset + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weight_attrib.y;
#ifdef COLOR_USED
			pcolor += transforms.data[boffset + 3] * weight_attrib.y;
#endif
		}
		if (weight_attrib.z > 0.001) {
			uint boffset = offset + bone_attrib.z * stride;
			matrix += mat4(transforms.data[boffset + 0], transforms.data[boffset + 1], transforms.data[boffset + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weight_attrib.z;
#ifdef COLOR_USED
			pcolor += transforms.data[boffset + 3] * weight_attrib.z;
#endif
		}
		if (weight_attrib.w > 0.001) {
			uint boffset = offset + bone_attrib.w * stride;
			matrix += mat4(transforms.data[boffset + 0], transforms.data[boffset + 1], transforms.data[boffset + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weight_attrib.w;
#ifdef COLOR_USED
			pcolor += transforms.data[boffset + 3] * weight_attrib.w;
#endif
		}

		instance_custom = transforms.data[offset + 4];

#ifdef COLOR_USED
		color_highp *= pcolor;
#endif

#else
		uint stride = multimesh_stride();
		uint offset = stride * (INSTANCE_INDEX + multimesh_offset);

		if (sc_multimesh_format_2d()) {
			matrix = mat4(transforms.data[offset + 0], transforms.data[offset + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
			offset += 2;
		} else {
			matrix = mat4(transforms.data[offset + 0], transforms.data[offset + 1], transforms.data[offset + 2], vec4(0.0, 0.0, 0.0, 1.0));
			offset += 3;
		}

		if (sc_multimesh_has_color()) {
#ifdef COLOR_USED
			color_highp *= transforms.data[offset];
#endif
			offset += 1;
		}

		if (sc_multimesh_has_custom_data()) {
			instance_custom = transforms.data[offset];
		}

#endif
		//transpose
		matrix = transpose(matrix);

#if !defined(USE_DOUBLE_PRECISION) || defined(SKIP_TRANSFORM_USED) || defined(VERTEX_WORLD_COORDS_USED) || defined(MODEL_MATRIX_USED)
		// Normally we can bake the multimesh transform into the model matrix, but when using double precision
		// we avoid baking it in so we can emulate high precision.
		read_model_matrix = model_matrix * matrix;
#if !defined(USE_DOUBLE_PRECISION) || defined(SKIP_TRANSFORM_USED) || defined(VERTEX_WORLD_COORDS_USED)
		model_matrix = read_model_matrix;
#endif // !defined(USE_DOUBLE_PRECISION) || defined(SKIP_TRANSFORM_USED) || defined(VERTEX_WORLD_COORDS_USED)
#endif // !defined(USE_DOUBLE_PRECISION) || defined(SKIP_TRANSFORM_USED) || defined(VERTEX_WORLD_COORDS_USED) || defined(MODEL_MATRIX_USED)
		model_normal_matrix = model_normal_matrix * mat3(matrix);
	}

#ifdef UV_USED
	uv_interp = uv_attrib;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
	uv2_interp = uv2_attrib;
#endif

	vec4 uv_scale = instances.data[instance_index].uv_scale;

	if (uv_scale != vec4(0.0)) { // Compression enabled
#ifdef UV_USED
		uv_interp = (uv_interp - 0.5) * uv_scale.xy;
#endif
#if defined(UV2_USED) || defined(USE_LIGHTMAP)
		uv2_interp = (uv2_interp - 0.5) * uv_scale.zw;
#endif
	}

#ifdef OVERRIDE_POSITION
	vec4 position = vec4(1.0);
#endif

#ifdef USE_MULTIVIEW
	vec3 eye_offset = scene_eye_offset.xyz;
#else
	vec3 eye_offset = vec3(0.0, 0.0, 0.0);
#endif // USE_MULTIVIEW

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (model_matrix * vec4(vertex, 1.0)).xyz;

#ifdef NORMAL_USED
	normal_highp = model_normal_matrix * normal_highp;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(BENT_NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	tangent_highp = model_normal_matrix * tangent_highp;
	binormal_highp = model_normal_matrix * binormal_highp;

#endif
#endif

#ifdef Z_CLIP_SCALE_USED
	float z_clip_scale = 1.0;
#endif

	float roughness_highp = 1.0;

	mat4 read_view_matrix = transpose(mat4(in_view_matrix[0],
			in_view_matrix[1],
			in_view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

#ifdef USE_DOUBLE_PRECISION
	mat4 modelview = read_view_matrix * model_matrix;

	// We separate the basis from the origin because the basis is fine with single point precision.
	// Then we combine the translations from the model matrix and the view matrix using emulated doubles.
	// We add the result to the vertex and ignore the final lost precision.
	vec3 model_origin = model_matrix[3].xyz;
	if (sc_multimesh()) {
		modelview = modelview * matrix;

		vec3 instance_origin = mat3(model_matrix) * matrix[3].xyz;
		model_origin = double_add_vec3(model_origin, model_precision, instance_origin, vec3(0.0), model_precision);
	}

	// Overwrite the translation part of modelview with improved precision.
	vec3 temp_precision; // Will be ignored.
	modelview[3].xyz = double_add_vec3(model_origin, model_precision, inv_view_matrix[3].xyz, view_precision, temp_precision);
	modelview[3].xyz = mat3(read_view_matrix) * modelview[3].xyz;
#else
	mat4 modelview = read_view_matrix * model_matrix;
#endif
	mat3 modelview_normal = mat3(read_view_matrix) * model_normal_matrix;
	vec2 read_viewport_size = scene_data.viewport_size;

#ifdef POINT_SIZE_USED
	float point_size = 1.0;
#endif

	{
#CODE : VERTEX
	}

#if defined(COLOR_USED)
	color_interp = hvec4(color_highp);
#endif

	half roughness = half(roughness_highp);

// using local coordinates (default)
#if !defined(SKIP_TRANSFORM_USED) && !defined(VERTEX_WORLD_COORDS_USED)

	vertex = (modelview * vec4(vertex, 1.0)).xyz;

#ifdef NORMAL_USED
	normal_highp = modelview_normal * normal_highp;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(BENT_NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	binormal_highp = modelview_normal * binormal_highp;
	tangent_highp = modelview_normal * tangent_highp;
#endif
#endif // !defined(SKIP_TRANSFORM_USED) && !defined(VERTEX_WORLD_COORDS_USED)

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (read_view_matrix * vec4(vertex, 1.0)).xyz;
#ifdef NORMAL_USED
	normal_highp = (read_view_matrix * vec4(normal_highp, 0.0)).xyz;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(BENT_NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	binormal_highp = (read_view_matrix * vec4(binormal_highp, 0.0)).xyz;
	tangent_highp = (read_view_matrix * vec4(tangent_highp, 0.0)).xyz;
#endif
#endif

	vertex_interp = vertex;

	// Normalize TBN vectors before interpolation, per MikkTSpace.
	// See: http://www.mikktspace.com/
#ifdef NORMAL_USED
	normal_interp = hvec3(normalize(normal_highp));
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED) || defined(BENT_NORMAL_MAP_USED)
	tangent_interp = hvec3(normalize(tangent_highp));
	binormal_interp = hvec3(normalize(binormal_highp));
#endif

// VERTEX LIGHTING
#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED) && defined(USE_VERTEX_LIGHTING)
	hvec3 normal = hvec3(normal_interp);

#ifdef USE_MULTIVIEW
	hvec3 view = hvec3(-normalize(vertex_interp - eye_offset));
#else
	hvec3 view = hvec3(-normalize(vertex_interp));
#endif

	hvec4 diffuse_light = hvec4(0.0);
	hvec4 specular_light = hvec4(0.0);

	uint omni_light_count = sc_omni_lights(8);
	uvec2 omni_light_indices = instances.data[instance_index].omni_lights;
	for (uint i = 0; i < omni_light_count; i++) {
		uint light_index = (i > 3) ? ((omni_light_indices.y >> ((i - 4) * 8)) & 0xFF) : ((omni_light_indices.x >> (i * 8)) & 0xFF);
		if (i > 0 && light_index == 0xFF) {
			break;
		}

		light_process_omni_vertex(light_index, vertex, view, normal, roughness, diffuse_light.rgb, specular_light.rgb);
	}

	uint spot_light_count = sc_spot_lights(8);
	uvec2 spot_light_indices = instances.data[instance_index].spot_lights;
	for (uint i = 0; i < spot_light_count; i++) {
		uint light_index = (i > 3) ? ((spot_light_indices.y >> ((i - 4) * 8)) & 0xFF) : ((spot_light_indices.x >> (i * 8)) & 0xFF);
		if (i > 0 && light_index == 0xFF) {
			break;
		}

		light_process_spot_vertex(light_index, vertex, view, normal, roughness, diffuse_light.rgb, specular_light.rgb);
	}

	uint directional_lights_count = sc_directional_lights(scene_directional_light_count);
	if (directional_lights_count > 0) {
		// We process the first directional light separately as it may have shadows.
		hvec3 directional_diffuse = hvec3(0.0);
		hvec3 directional_specular = hvec3(0.0);

		for (uint i = 0; i < directional_lights_count; i++) {
			if (!bool(directional_lights.data[i].mask & instances.data[instance_index].layer_mask)) {
				continue; // Not masked, skip.
			}

			if (directional_lights.data[i].bake_mode == LIGHT_BAKE_STATIC && bool(instances.data[instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
				continue; // Statically baked light and object uses lightmap, skip.
			}
			if (i == 0) {
				light_compute_vertex(normal, hvec3(directional_lights.data[0].direction), view,
						hvec3(directional_lights.data[0].color * directional_lights.data[0].energy),
						true, roughness,
						directional_diffuse,
						directional_specular);
			} else {
				light_compute_vertex(normal, hvec3(directional_lights.data[i].direction), view,
						hvec3(directional_lights.data[i].color * directional_lights.data[i].energy),
						true, roughness,
						diffuse_light.rgb,
						specular_light.rgb);
			}
		}

		// Calculate the contribution from the shadowed light so we can scale the shadows accordingly.
		half diff_avg = dot(diffuse_light.rgb, hvec3(0.33333));
		half diff_dir_avg = dot(directional_diffuse, hvec3(0.33333));
		if (diff_avg > half(0.0)) {
			diffuse_light.a = diff_dir_avg / (diff_avg + diff_dir_avg);
		} else {
			diffuse_light.a = half(1.0);
		}

		diffuse_light.rgb += directional_diffuse;

		half spec_avg = dot(specular_light.rgb, hvec3(0.33333));
		half spec_dir_avg = dot(directional_specular, hvec3(0.33333));
		if (spec_avg > half(0.0)) {
			specular_light.a = spec_dir_avg / (spec_avg + spec_dir_avg);
		} else {
			specular_light.a = half(1.0);
		}

		specular_light.rgb += directional_specular;
	}

	diffuse_light_interp = hvec4(diffuse_light);
	specular_light_interp = hvec4(specular_light);

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED) && defined(USE_VERTEX_LIGHTING)

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_DUAL_PARABOLOID

	vertex_interp.z *= dual_paraboloid_side;

	dp_clip = vertex_interp.z; //this attempts to avoid noise caused by objects sent to the other parabolloid side due to bias

	//for dual paraboloid shadow mapping, this is the fastest but least correct way, as it curves straight edges

	vec3 vtx = vertex_interp;
	float distance = length(vtx);
	vtx = normalize(vtx);
	vtx.xy /= 1.0 - vtx.z;
	vtx.z = (distance / z_far);
	vtx.z = vtx.z * 2.0 - 1.0;
	vertex_interp = vtx;

#endif

#endif //MODE_RENDER_DEPTH

#ifdef OVERRIDE_POSITION
	gl_Position = position;
#else
	gl_Position = projection_matrix * vec4(vertex_interp, 1.0);
#endif // OVERRIDE_POSITION

#if defined(Z_CLIP_SCALE_USED) && !defined(SHADOW_PASS)
	gl_Position.z = mix(gl_Position.w, gl_Position.z, z_clip_scale);
#endif

#ifdef MODE_RENDER_DEPTH
	if (bool(scene_flags & SCENE_DATA_FLAGS_USE_PANCAKE_SHADOWS)) {
		if (gl_Position.z >= 0.9999) {
			gl_Position.z = 0.9999;
		}
	}
#endif // MODE_RENDER_DEPTH
#ifdef MODE_RENDER_MATERIAL
	if (bool(scene_flags & SCENE_DATA_FLAGS_USE_UV2_MATERIAL)) {
		vec2 uv_dest_attrib;
		if (uv_scale != vec4(0.0)) {
			uv_dest_attrib = (uv2_attrib.xy - 0.5) * uv_scale.zw;
		} else {
			uv_dest_attrib = uv2_attrib.xy;
		}

		vec2 uv_offset = unpackHalf2x16(draw_call.uv_offset);
		gl_Position.xy = (uv_dest_attrib + uv_offset) * 2.0 - 1.0;
		gl_Position.z = 0.00001;
		gl_Position.w = 1.0;
	}
#endif // MODE_RENDER_MATERIAL
#ifdef MODE_RENDER_MOTION_VECTORS
	screen_position_output = gl_Position;
#endif // MODE_RENDER_MOTION_VECTORS

#ifdef POINT_SIZE_USED
	if (sc_emulate_point_size) {
		vec2 point_coords[6] = vec2[](
				vec2(0, 1),
				vec2(0, 0),
				vec2(1, 1),
				vec2(0, 0),
				vec2(1, 0),
				vec2(1, 1));

		vec2 point_coord = point_coords[gl_VertexIndex % 6];
		gl_Position.xy += (point_coord * 2.0 - 1.0) * point_size * scene_data.screen_pixel_size * gl_Position.w;

#ifdef POINT_COORD_USED
		point_coord_interp = point_coord;
#endif
	} else {
		gl_PointSize = point_size;
	}
#endif
}

void main() {
#if defined(MODE_RENDER_MOTION_VECTORS)
	vec3 prev_vertex;
#ifdef NORMAL_USED
	vec3 prev_normal;
#endif
#if defined(NORMAL_USED) || defined(TANGENT_USED)
	vec3 prev_tangent;
	vec3 prev_binormal;
#endif

	_unpack_vertex_attributes(
			previous_vertex_attrib,
			instances.data[draw_call.instance_index].compressed_aabb_position_pad.xyz,
			instances.data[draw_call.instance_index].compressed_aabb_size_pad.xyz,
#if defined(NORMAL_USED) || defined(TANGENT_USED)
			previous_normal_attrib,
#ifdef NORMAL_USED
			prev_normal,
#endif
			prev_tangent,
			prev_binormal,
#endif
			prev_vertex);

	vertex_shader(prev_vertex,
#ifdef NORMAL_USED
			prev_normal,
#endif
#if defined(NORMAL_USED) || defined(TANGENT_USED)
			prev_tangent,
			prev_binormal,
#endif
			draw_call.instance_index, draw_call.multimesh_motion_vectors_previous_offset, instances.data[draw_call.instance_index].prev_transform,
#ifdef USE_DOUBLE_PRECISION
			instances.data[draw_call.instance_index].prev_model_precision.xyz,
			scene_data_block.prev_data.inv_view_precision,
#endif

#ifdef MODE_DUAL_PARABOLOID
			scene_data_block.prev_data.dual_paraboloid_side,
			scene_data_block.prev_data.z_far,
#endif
#if defined(MODE_RENDER_DEPTH) || defined(MODE_RENDER_MATERIAL)
			scene_data_block.prev_data.flags,
#endif
#ifdef USE_MULTIVIEW
			scene_data_block.prev_data.projection_matrix_view[ViewIndex],
			scene_data_block.prev_data.inv_projection_matrix_view[ViewIndex],
			scene_data_block.prev_data.eye_offset[ViewIndex],
#else
			scene_data_block.prev_data.projection_matrix,
			scene_data_block.prev_data.inv_projection_matrix,
#endif
			scene_data_block.prev_data.view_matrix,
			scene_data_block.prev_data.inv_view_matrix,
			scene_data_block.prev_data.viewport_size,
			scene_data_block.prev_data.directional_light_count,
			prev_screen_position);
#else
	// Unused output.
	vec4 screen_position;
#endif // MODE_RENDER_MOTION_VECTORS

	vec3 vertex;
#ifdef NORMAL_USED
	vec3 normal;
#endif
#if defined(NORMAL_USED) || defined(TANGENT_USED)
	vec3 tangent;
	vec3 binormal;
#endif

	_unpack_vertex_attributes(
			vertex_angle_attrib,
			instances.data[draw_call.instance_index].compressed_aabb_position_pad.xyz,
			instances.data[draw_call.instance_index].compressed_aabb_size_pad.xyz,
#if defined(NORMAL_USED) || defined(TANGENT_USED)
			axis_tangent_attrib,
#ifdef NORMAL_USED
			normal,
#endif
			tangent,
			binormal,
#endif
			vertex);

	vertex_shader(vertex,
#ifdef NORMAL_USED
			normal,
#endif
#if defined(NORMAL_USED) || defined(TANGENT_USED)
			tangent,
			binormal,
#endif
			draw_call.instance_index, draw_call.multimesh_motion_vectors_current_offset, instances.data[draw_call.instance_index].transform,
#ifdef USE_DOUBLE_PRECISION
			instances.data[draw_call.instance_index].model_precision.xyz,
			scene_data_block.data.inv_view_precision,
#endif
#ifdef MODE_DUAL_PARABOLOID
			scene_data_block.data.dual_paraboloid_side,
			scene_data_block.data.z_far,
#endif
#if defined(MODE_RENDER_DEPTH) || defined(MODE_RENDER_MATERIAL)
			scene_data_block.data.flags,
#endif
#ifdef USE_MULTIVIEW
			scene_data_block.data.projection_matrix_view[ViewIndex],
			scene_data_block.data.inv_projection_matrix_view[ViewIndex],
			scene_data_block.data.eye_offset[ViewIndex],
#else
			scene_data_block.data.projection_matrix,
			scene_data_block.data.inv_projection_matrix,
#endif
			scene_data_block.data.view_matrix,
			scene_data_block.data.inv_view_matrix,
			scene_data_block.data.viewport_size,
			scene_data_block.data.directional_light_count,
			screen_position);
}

#[fragment]

#version 450

#VERSION_DEFINES

#define SHADER_IS_SRGB false
#define SHADER_SPACE_FAR 0.0

#ifdef SHADOW_PASS
#define IN_SHADOW_PASS true
#else
#define IN_SHADOW_PASS false
#endif

/* Include half precision types. */
#include "../half_inc.glsl"

/* Include our forward mobile UBOs definitions etc. */
#include "scene_forward_mobile_inc.glsl"

/* Varyings */

// All interpolators are intentionally kept at full precision as storageInputOutput16 is not
// checked for support. Devices with Adreno GPUs don't usually support this capability.

layout(location = 0) in vec3 vertex_interp;

#ifdef NORMAL_USED
layout(location = 1) in vec3 normal_interp;
#endif

#if defined(COLOR_USED)
layout(location = 2) in vec4 color_interp;
#endif

#ifdef UV_USED
layout(location = 3) in vec2 uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 4) in vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(BENT_NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 5) in vec3 tangent_interp;
layout(location = 6) in vec3 binormal_interp;
#endif

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED) && defined(USE_VERTEX_LIGHTING)
layout(location = 7) in vec4 diffuse_light_interp;
layout(location = 8) in vec4 specular_light_interp;
#endif

#ifdef MODE_DUAL_PARABOLOID

layout(location = 9) in float dp_clip;

#endif

#if defined(MODE_RENDER_MOTION_VECTORS)
layout(location = 12) in highp vec4 screen_position;
layout(location = 13) in highp vec4 prev_screen_position;
#endif

#ifdef USE_LIGHTMAP
// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float w0(float a) {
	return (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0);
}

float w1(float a) {
	return (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0);
}

float w2(float a) {
	return (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0);
}

float w3(float a) {
	return (1.0 / 6.0) * (a * a * a);
}

// g0 and g1 are the two amplitude functions
float g0(float a) {
	return w0(a) + w1(a);
}

float g1(float a) {
	return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
float h0(float a) {
	return -1.0 + w1(a) / (w0(a) + w1(a));
}

float h1(float a) {
	return 1.0 + w3(a) / (w2(a) + w3(a));
}

vec4 textureArray_bicubic(texture2DArray tex, vec3 uv, vec2 texture_size) {
	vec2 texel_size = vec2(1.0) / texture_size;

	uv.xy = uv.xy * texture_size + vec2(0.5);

	vec2 iuv = floor(uv.xy);
	vec2 fuv = fract(uv.xy);

	float g0x = g0(fuv.x);
	float g1x = g1(fuv.x);
	float h0x = h0(fuv.x);
	float h1x = h1(fuv.x);
	float h0y = h0(fuv.y);
	float h1y = h1(fuv.y);

	vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - vec2(0.5)) * texel_size;
	vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - vec2(0.5)) * texel_size;
	vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - vec2(0.5)) * texel_size;
	vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - vec2(0.5)) * texel_size;

	return (g0(fuv.y) * (g0x * texture(sampler2DArray(tex, SAMPLER_LINEAR_CLAMP), vec3(p0, uv.z)) + g1x * texture(sampler2DArray(tex, SAMPLER_LINEAR_CLAMP), vec3(p1, uv.z)))) +
			(g1(fuv.y) * (g0x * texture(sampler2DArray(tex, SAMPLER_LINEAR_CLAMP), vec3(p2, uv.z)) + g1x * texture(sampler2DArray(tex, SAMPLER_LINEAR_CLAMP), vec3(p3, uv.z))));
}
#endif //USE_LIGHTMAP

#ifdef USE_MULTIVIEW
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
vec3 multiview_uv(vec2 uv) {
	return vec3(uv, ViewIndex);
}
ivec3 multiview_uv(ivec2 uv) {
	return ivec3(uv, int(ViewIndex));
}
#else // USE_MULTIVIEW
#define ViewIndex 0
vec2 multiview_uv(vec2 uv) {
	return uv;
}
ivec2 multiview_uv(ivec2 uv) {
	return uv;
}
#endif // !USE_MULTIVIEW

#if defined(POINT_SIZE_USED) && defined(POINT_COORD_USED)
layout(location = 14) in vec2 point_coord_interp;
#endif

//defines to keep compatibility with vertex

#ifdef USE_MULTIVIEW
#define projection_matrix scene_data.projection_matrix_view[ViewIndex]
#define inv_projection_matrix scene_data.inv_projection_matrix_view[ViewIndex]
#else
#define projection_matrix scene_data.projection_matrix
#define inv_projection_matrix scene_data.inv_projection_matrix
#endif

#if defined(ENABLE_SSS) && defined(ENABLE_TRANSMITTANCE)
//both required for transmittance to be enabled
#define LIGHT_TRANSMITTANCE_USED
#endif

#ifdef MATERIAL_UNIFORMS_USED
/* clang-format off */
layout(set = MATERIAL_UNIFORM_SET, binding = 0, std140) uniform MaterialUniforms {
#MATERIAL_UNIFORMS
} material;
/* clang-format on */
#endif

#GLOBALS

#define scene_data scene_data_block.data

/* clang-format on */

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_RENDER_MATERIAL

layout(location = 0) out vec4 albedo_output_buffer;
layout(location = 1) out vec4 normal_output_buffer;
layout(location = 2) out vec4 orm_output_buffer;
layout(location = 3) out vec4 emission_output_buffer;
layout(location = 4) out float depth_output_buffer;

#endif // MODE_RENDER_MATERIAL

#else // RENDER DEPTH

#ifdef MODE_MULTIPLE_RENDER_TARGETS

layout(location = 0) out vec4 diffuse_buffer; //diffuse (rgb) and roughness
layout(location = 1) out vec4 specular_buffer; //specular and SSS (subsurface scatter)
#else

layout(location = 0) out vec4 frag_color;
#endif // MODE_MULTIPLE_RENDER_TARGETS

#endif // RENDER DEPTH

#include "../scene_forward_aa_inc.glsl"

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED) // && !defined(USE_VERTEX_LIGHTING)

// Default to SPECULAR_SCHLICK_GGX.
#if !defined(SPECULAR_DISABLED) && !defined(SPECULAR_SCHLICK_GGX) && !defined(SPECULAR_TOON)
#define SPECULAR_SCHLICK_GGX
#endif

#include "../scene_forward_lights_inc.glsl"

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED) && !defined(USE_VERTEX_LIGHTING)

#ifndef MODE_RENDER_DEPTH

/*
	Only supporting normal fog here.
*/

hvec4 fog_process(vec3 vertex) {
	vec3 fog_color = scene_data_block.data.fog_light_color;

	if (sc_use_fog_aerial_perspective()) {
		vec3 sky_fog_color = vec3(0.0);
		vec3 cube_view = scene_data_block.data.radiance_inverse_xform * vertex;
		// mip_level always reads from the second mipmap and higher so the fog is always slightly blurred
		float mip_level = mix(1.0 / MAX_ROUGHNESS_LOD, 1.0, 1.0 - (abs(vertex.z) - scene_data_block.data.z_near) / (scene_data_block.data.z_far - scene_data_block.data.z_near));
#ifdef USE_RADIANCE_OCTMAP_ARRAY
		float roughness_lod, blend;
		blend = modf(mip_level * MAX_ROUGHNESS_LOD, roughness_lod);
		float cube_lod = vec3_to_oct_lod(dFdx(cube_view), dFdy(cube_view), scene_data_block.data.radiance_pixel_size);
		vec2 cube_uv = vec3_to_oct_with_border(cube_view, vec2(scene_data_block.data.radiance_border_size, 1.0 - scene_data_block.data.radiance_border_size * 2.0));
		vec3 sky_sample_a = textureLod(sampler2DArray(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(cube_uv, roughness_lod), cube_lod).rgb;
		vec3 sky_sample_b = textureLod(sampler2DArray(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(cube_uv, roughness_lod + 1), cube_lod).rgb;
		sky_fog_color = mix(sky_sample_a, sky_sample_b, blend);
#else
		float roughness_lod = mip_level * MAX_ROUGHNESS_LOD;
		vec2 cube_uv = vec3_to_oct_with_border(cube_view, vec2(scene_data_block.data.radiance_border_size, 1.0 - scene_data_block.data.radiance_border_size * 2.0));
		sky_fog_color = textureLod(sampler2D(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), cube_uv, roughness_lod).rgb;
#endif //USE_RADIANCE_OCTMAP_ARRAY
		fog_color = mix(fog_color, sky_fog_color, scene_data_block.data.fog_aerial_perspective);
	}

	if (sc_use_fog_sun_scatter()) {
		vec4 sun_scatter = vec4(0.0);
		float sun_total = 0.0;
		vec3 view = normalize(vertex);

		uint directional_lights_count = sc_directional_lights(scene_data.directional_light_count);
		for (uint i = 0; i < directional_lights_count; i++) {
			vec3 light_color = directional_lights.data[i].color * directional_lights.data[i].energy;
			float light_amount = pow(max(dot(view, directional_lights.data[i].direction), 0.0), 8.0);
			fog_color += light_color * light_amount * scene_data_block.data.fog_sun_scatter;
		}
	}

	float fog_amount = 0.0;

	if (sc_use_depth_fog()) {
		float fog_z = smoothstep(scene_data_block.data.fog_depth_begin, scene_data_block.data.fog_depth_end, length(vertex));
		float fog_quad_amount = pow(fog_z, scene_data_block.data.fog_depth_curve) * scene_data_block.data.fog_density;
		fog_amount = fog_quad_amount;
	} else {
		fog_amount = 1 - exp(min(0.0, -length(vertex) * scene_data_block.data.fog_density));
	}

	if (sc_use_fog_height_density()) {
		mat4 inv_view_matrix = transpose(mat4(scene_data_block.data.inv_view_matrix[0],
				scene_data_block.data.inv_view_matrix[1],
				scene_data_block.data.inv_view_matrix[2],
				vec4(0.0, 0.0, 0.0, 1.0)));

		float y = (inv_view_matrix * vec4(vertex, 1.0)).y;

		float y_dist = y - scene_data_block.data.fog_height;

		float vfog_amount = 1.0 - exp(min(0.0, y_dist * scene_data_block.data.fog_height_density));

		fog_amount = max(vfog_amount, fog_amount);
	}

	return hvec4(fog_color, fog_amount);
}

#endif //!MODE_RENDER DEPTH

void main() {
#ifdef UBERSHADER
	bool front_facing = gl_FrontFacing;
	if (uc_cull_mode() == POLYGON_CULL_BACK && !front_facing) {
		discard;
	} else if (uc_cull_mode() == POLYGON_CULL_FRONT && front_facing) {
		discard;
	}
#endif
#ifdef MODE_DUAL_PARABOLOID

	if (dp_clip > 0.0) {
		discard;
	}
#endif

	//lay out everything, whatever is unused is optimized away anyway
	vec3 vertex = vertex_interp;
#ifdef USE_MULTIVIEW
	vec3 eye_offset = scene_data.eye_offset[ViewIndex].xyz;
	vec3 view_highp = -normalize(vertex_interp - eye_offset);
#else
	vec3 eye_offset = vec3(0.0, 0.0, 0.0);
	vec3 view_highp = -normalize(vertex_interp);
#endif
	vec3 albedo_highp = vec3(1.0);
	vec3 backlight_highp = vec3(0.0);
	vec4 transmittance_color_highp = vec4(0.0);
	float transmittance_depth_highp = 0.0;
	float transmittance_boost_highp = 0.0;
	float metallic_highp = 0.0;
	float specular_highp = 0.5;
	vec3 emission_highp = vec3(0.0);
	float roughness_highp = 1.0;
	float rim_highp = 0.0;
	float rim_tint_highp = 0.0;
	float clearcoat_highp = 0.0;
	float clearcoat_roughness_highp = 0.0;
	float anisotropy_highp = 0.0;
	vec2 anisotropy_flow_highp = vec2(1.0, 0.0);
#ifdef PREMUL_ALPHA_USED
	float premul_alpha_highp = 1.0;
#endif
#ifndef FOG_DISABLED
	vec4 fog_highp = vec4(0.0);
#endif // !FOG_DISABLED
#if defined(CUSTOM_RADIANCE_USED)
	vec4 custom_radiance_highp = vec4(0.0);
#endif
#if defined(CUSTOM_IRRADIANCE_USED)
	vec4 custom_irradiance_highp = vec4(0.0);
#endif

	float ao_highp = 1.0;
	float ao_light_affect_highp = 0.0;

	float alpha_highp = 1.0;

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED) || defined(BENT_NORMAL_MAP_USED)
	vec3 binormal_highp = binormal_interp;
	vec3 tangent_highp = tangent_interp;
#else // TANGENT_USED || NORMAL_MAP_USED || LIGHT_ANISOTROPY_USED || BENT_NORMAL_MAP_USED
	vec3 binormal_highp = vec3(0.0);
	vec3 tangent_highp = vec3(0.0);
#endif

#ifdef NORMAL_USED
	vec3 normal_highp = normal_interp;
#if defined(DO_SIDE_CHECK)
	if (!gl_FrontFacing) {
		normal_highp = -normal_highp;
	}
#endif // DO_SIDE_CHECK
#endif // NORMAL_USED

#ifdef UV_USED
	vec2 uv = uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
	vec2 uv2 = uv2_interp;
#endif

#if defined(COLOR_USED)
	vec4 color_highp = color_interp;
#endif

#if defined(NORMAL_MAP_USED)

	vec3 normal_map_highp = vec3(0.5);
#endif

#if defined(BENT_NORMAL_MAP_USED)
	hvec3 bent_normal_vector;
	vec3 bent_normal_map_highp = vec3(0.5);
#endif

	float normal_map_depth_highp = 1.0;

	vec2 screen_uv = gl_FragCoord.xy * scene_data.screen_pixel_size;

	float sss_strength_highp = 0.0;

#ifdef ALPHA_SCISSOR_USED
	float alpha_scissor_threshold_highp = 1.0;
#endif // ALPHA_SCISSOR_USED

#ifdef ALPHA_HASH_USED
	float alpha_hash_scale_highp = 1.0;
#endif // ALPHA_HASH_USED

#ifdef ALPHA_ANTIALIASING_EDGE_USED
	float alpha_antialiasing_edge_highp = 0.0;
	vec2 alpha_texture_coordinate = vec2(0.0, 0.0);
#endif // ALPHA_ANTIALIASING_EDGE_USED

	mat4 inv_view_matrix = transpose(mat4(scene_data.inv_view_matrix[0],
			scene_data.inv_view_matrix[1],
			scene_data.inv_view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));
	mat4 read_model_matrix = transpose(mat4(instances.data[draw_call.instance_index].transform[0],
			instances.data[draw_call.instance_index].transform[1],
			instances.data[draw_call.instance_index].transform[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

#ifdef LIGHT_VERTEX_USED
	vec3 light_vertex = vertex;
#endif //LIGHT_VERTEX_USED

	mat3 model_normal_matrix;
	if (bool(instances.data[draw_call.instance_index].flags & INSTANCE_FLAGS_NON_UNIFORM_SCALE)) {
		model_normal_matrix = transpose(inverse(mat3(read_model_matrix)));
	} else {
		model_normal_matrix = mat3(read_model_matrix);
	}

	mat4 read_view_matrix = transpose(mat4(scene_data.view_matrix[0],
			scene_data.view_matrix[1],
			scene_data.view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));
	vec2 read_viewport_size = scene_data.viewport_size;

#ifdef POINT_COORD_USED
#ifdef POINT_SIZE_USED
	vec2 point_coord;
	if (sc_emulate_point_size) {
		point_coord = point_coord_interp;
	} else {
		point_coord = gl_PointCoord;
	}
#else // !POINT_SIZE_USED
	vec2 point_coord = vec2(0.5);
#endif
#endif

	{
#CODE : FRAGMENT
	}

	// Store variables in half precision after user shader code is run.
	hvec3 view = hvec3(view_highp);
	hvec3 albedo = hvec3(albedo_highp);
	hvec3 backlight = hvec3(backlight_highp);
	hvec4 transmittance_color = hvec4(transmittance_color_highp);
	half transmittance_depth = half(transmittance_depth_highp);
	half transmittance_boost = half(transmittance_boost_highp);
	half metallic = half(metallic_highp);
	half specular = half(specular_highp);
	hvec3 emission = hvec3(emission_highp);
	half roughness = half(roughness_highp);
	half rim = half(rim_highp);
	half rim_tint = half(rim_tint_highp);
	half clearcoat = half(clearcoat_highp);
	half clearcoat_roughness = half(clearcoat_roughness_highp);
	half anisotropy = half(anisotropy_highp);
	hvec2 anisotropy_flow = hvec2(anisotropy_flow_highp);
	half ao = half(ao_highp);
	half ao_light_affect = half(ao_light_affect_highp);
	half alpha = half(alpha_highp);
	half normal_map_depth = half(normal_map_depth_highp);
	half sss_strength = half(sss_strength_highp);
#ifdef PREMUL_ALPHA_USED
	half premul_alpha = half(premul_alpha_highp);
#endif
#ifndef FOG_DISABLED
	hvec4 fog = hvec4(fog_highp);
#endif
#ifdef CUSTOM_RADIANCE_USED
	hvec4 custom_radiance = hvec4(custom_radiance_highp);
#endif
#ifdef CUSTOM_IRRADIANCE_USED
	hvec4 custom_irradiance = hvec4(custom_irradiance_highp);
#endif
#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED) || defined(BENT_NORMAL_MAP_USED)
	hvec3 binormal = hvec3(binormal_highp);
	hvec3 tangent = hvec3(tangent_highp);
#else
	hvec3 binormal = hvec3(binormal_highp);
	hvec3 tangent = hvec3(tangent_highp);
#endif
#ifdef NORMAL_USED
	hvec3 normal = hvec3(normal_highp);
#endif
#if defined(COLOR_USED)
	hvec4 color = hvec4(color_highp);
#endif
#if defined(NORMAL_MAP_USED)
	hvec3 normal_map = hvec3(normal_map_highp);
#endif
#if defined(BENT_NORMAL_MAP_USED)
	hvec3 bent_normal_map = hvec3(bent_normal_map_highp);
#endif
#ifdef ALPHA_SCISSOR_USED
	half alpha_scissor_threshold = half(alpha_scissor_threshold_highp);
#endif
#ifdef ALPHA_HASH_USED
	half alpha_hash_scale = half(alpha_hash_scale_highp);
#endif
#ifdef ALPHA_ANTIALIASING_EDGE_USED
	half alpha_antialiasing_edge = half(alpha_antialiasing_edge_highp);
#endif

#ifdef LIGHT_VERTEX_USED
	vertex = light_vertex;
#ifdef USE_MULTIVIEW
	view = hvec3(-normalize(vertex - eye_offset));
#else
	view = hvec3(-normalize(vertex));
#endif //USE_MULTIVIEW
#endif //LIGHT_VERTEX_USED

#ifdef NORMAL_USED
	hvec3 geo_normal = normalize(normal);
#endif // NORMAL_USED

#ifdef LIGHT_TRANSMITTANCE_USED
#ifdef SSS_MODE_SKIN
	transmittance_color.a = sss_strength;
#else
	transmittance_color.a *= sss_strength;
#endif
#endif

#ifndef USE_SHADOW_TO_OPACITY

#ifdef ALPHA_SCISSOR_USED
#ifdef MODE_RENDER_MATERIAL
	if (alpha < alpha_scissor_threshold) {
		alpha = half(0.0);
	} else {
		alpha = half(1.0);
	}
#else
	if (alpha < alpha_scissor_threshold) {
		discard;
	}
#endif // MODE_RENDER_MATERIAL
#endif // ALPHA_SCISSOR_USED

// alpha hash can be used in unison with alpha antialiasing
#ifdef ALPHA_HASH_USED
	vec3 object_pos = (inverse(read_model_matrix) * inv_view_matrix * vec4(vertex, 1.0)).xyz;
#ifdef MODE_RENDER_MATERIAL
	if (alpha < compute_alpha_hash_threshold(object_pos, alpha_hash_scale)) {
		alpha = half(0.0);
	} else {
		alpha = half(1.0);
	}
#else
	if (alpha < compute_alpha_hash_threshold(object_pos, alpha_hash_scale)) {
		discard;
	}
#endif // MODE_RENDER_MATERIAL
#endif // ALPHA_HASH_USED

// If we are not edge antialiasing, we need to remove the output alpha channel from scissor and hash
#if (defined(ALPHA_SCISSOR_USED) || defined(ALPHA_HASH_USED)) && !defined(ALPHA_ANTIALIASING_EDGE_USED) && !defined(MODE_RENDER_MATERIAL)
	alpha = half(1.0);
#endif

#ifdef ALPHA_ANTIALIASING_EDGE_USED
// If alpha scissor is used, we must further the edge threshold, otherwise we won't get any edge feather
#ifdef ALPHA_SCISSOR_USED
	alpha_antialiasing_edge = clamp(alpha_scissor_threshold + alpha_antialiasing_edge, half(0.0), half(1.0));
#endif
	alpha = compute_alpha_antialiasing_edge(alpha, alpha_texture_coordinate, alpha_antialiasing_edge);
#endif // ALPHA_ANTIALIASING_EDGE_USED

#ifdef MODE_RENDER_DEPTH
#if defined(USE_OPAQUE_PREPASS) || defined(ALPHA_ANTIALIASING_EDGE_USED)
	if (alpha < half(scene_data.opaque_prepass_threshold)) {
		discard;
	}
#endif // USE_OPAQUE_PREPASS || ALPHA_ANTIALIASING_EDGE_USED
#endif // MODE_RENDER_DEPTH

#endif // !USE_SHADOW_TO_OPACITY

#if defined(NORMAL_MAP_USED)
	normal_map.xy = normal_map.xy * half(2.0) - half(1.0);
	normal_map.z = sqrt(max(half(0.0), half(1.0) - dot(normal_map.xy, normal_map.xy))); //always ignore Z, as it can be RG packed, Z may be pos/neg, etc.

	// Tangent-space transformation is performed using unnormalized TBN vectors, per MikkTSpace.
	// See: http://www.mikktspace.com/
	normal = normalize(mix(normal, tangent * normal_map.x + binormal * normal_map.y + normal * normal_map.z, normal_map_depth));
#elif defined(NORMAL_USED)
	normal = geo_normal;
#endif // NORMAL_MAP_USED

#ifdef BENT_NORMAL_MAP_USED
	bent_normal_map.xy = bent_normal_map.xy * half(2.0) - half(1.0);
	bent_normal_map.z = sqrt(max(half(0.0), half(1.0) - dot(bent_normal_map.xy, bent_normal_map.xy)));

	bent_normal_vector = normalize(tangent * bent_normal_map.x + binormal * bent_normal_map.y + normal * bent_normal_map.z);
#endif

#ifdef LIGHT_ANISOTROPY_USED

	if (anisotropy > half(0.01)) {
		hmat3 rot = hmat3(tangent, binormal, normal);
		// Make local to space.
		tangent = normalize(rot * hvec3(anisotropy_flow.x, anisotropy_flow.y, 0.0));
		binormal = normalize(rot * hvec3(-anisotropy_flow.y, anisotropy_flow.x, 0.0));
	}

#endif

#ifdef ENABLE_CLIP_ALPHA
#ifdef MODE_RENDER_MATERIAL
	if (albedo.a < half(0.99)) {
		// Used for doublepass and shadowmapping.
		albedo.a = half(0.0);
		alpha = half(0.0);
	} else {
		albedo.a = half(1.0);
		alpha = half(1.0);
	}
#else
	if (albedo.a < half(0.99)) {
		//used for doublepass and shadowmapping
		discard;
	}
#endif // MODE_RENDER_MATERIAL
#endif

	/////////////////////// FOG //////////////////////
#ifndef MODE_RENDER_DEPTH

#ifndef FOG_DISABLED
#ifndef CUSTOM_FOG_USED
	// Draw "fixed" fog before volumetric fog to ensure volumetric fog can appear in front of the sky.

	if (!sc_disable_fog() && bool(scene_data.flags & SCENE_DATA_FLAGS_USE_FOG)) {
		fog = fog_process(vertex);
	}

#endif //!CUSTOM_FOG_USED

#endif //!FOG_DISABLED
#endif //!MODE_RENDER_DEPTH

	/////////////////////// DECALS ////////////////////////////////

#ifndef MODE_RENDER_DEPTH

	vec3 vertex_ddx = dFdx(vertex);
	vec3 vertex_ddy = dFdy(vertex);

	uint decal_count = sc_decals(8);
	uvec2 decal_indices = instances.data[draw_call.instance_index].decals;
	for (uint i = 0; i < decal_count; i++) {
		uint decal_index = (i > 3) ? ((decal_indices.y >> ((i - 4) * 8)) & 0xFF) : ((decal_indices.x >> (i * 8)) & 0xFF);
		if (decal_index == 0xFF) {
			break;
		}

		vec3 uv_local = (decals.data[decal_index].xform * vec4(vertex, 1.0)).xyz;
		if (any(lessThan(uv_local, vec3(0.0, -1.0, 0.0))) || any(greaterThan(uv_local, vec3(1.0)))) {
			continue; //out of decal
		}

		float fade = pow(1.0 - (uv_local.y > 0.0 ? uv_local.y : -uv_local.y), uv_local.y > 0.0 ? decals.data[decal_index].upper_fade : decals.data[decal_index].lower_fade);

		if (decals.data[decal_index].normal_fade > 0.0) {
			fade *= smoothstep(decals.data[decal_index].normal_fade, 1.0, dot(vec3(geo_normal), decals.data[decal_index].normal) * 0.5 + 0.5);
		}

		//we need ddx/ddy for mipmaps, so simulate them
		vec2 ddx = (decals.data[decal_index].xform * vec4(vertex_ddx, 0.0)).xz;
		vec2 ddy = (decals.data[decal_index].xform * vec4(vertex_ddy, 0.0)).xz;

		if (decals.data[decal_index].albedo_rect != vec4(0.0)) {
			//has albedo
			vec4 decal_albedo;
			if (sc_decal_use_mipmaps()) {
				decal_albedo = textureGrad(sampler2D(decal_atlas_srgb, decal_sampler), uv_local.xz * decals.data[decal_index].albedo_rect.zw + decals.data[decal_index].albedo_rect.xy, ddx * decals.data[decal_index].albedo_rect.zw, ddy * decals.data[decal_index].albedo_rect.zw);
			} else {
				decal_albedo = textureLod(sampler2D(decal_atlas_srgb, decal_sampler), uv_local.xz * decals.data[decal_index].albedo_rect.zw + decals.data[decal_index].albedo_rect.xy, 0.0);
			}
			decal_albedo *= decals.data[decal_index].modulate;
			decal_albedo.a *= fade;
			albedo = hvec3(mix(vec3(albedo), decal_albedo.rgb, decal_albedo.a * decals.data[decal_index].albedo_mix));

			if (decals.data[decal_index].normal_rect != vec4(0.0)) {
				vec3 decal_normal;
				if (sc_decal_use_mipmaps()) {
					decal_normal = textureGrad(sampler2D(decal_atlas, decal_sampler), uv_local.xz * decals.data[decal_index].normal_rect.zw + decals.data[decal_index].normal_rect.xy, ddx * decals.data[decal_index].normal_rect.zw, ddy * decals.data[decal_index].normal_rect.zw).xyz;
				} else {
					decal_normal = textureLod(sampler2D(decal_atlas, decal_sampler), uv_local.xz * decals.data[decal_index].normal_rect.zw + decals.data[decal_index].normal_rect.xy, 0.0).xyz;
				}
				decal_normal.xy = decal_normal.xy * vec2(2.0, -2.0) - vec2(1.0, -1.0); //users prefer flipped y normal maps in most authoring software
				decal_normal.z = sqrt(max(0.0, 1.0 - dot(decal_normal.xy, decal_normal.xy)));
				//convert to view space, use xzy because y is up
				decal_normal = (decals.data[decal_index].normal_xform * decal_normal.xzy).xyz;

				normal = hvec3(normalize(mix(vec3(normal), decal_normal, decal_albedo.a)));
			}

			if (decals.data[decal_index].orm_rect != vec4(0.0)) {
				vec3 decal_orm;
				if (sc_decal_use_mipmaps()) {
					decal_orm = textureGrad(sampler2D(decal_atlas, decal_sampler), uv_local.xz * decals.data[decal_index].orm_rect.zw + decals.data[decal_index].orm_rect.xy, ddx * decals.data[decal_index].orm_rect.zw, ddy * decals.data[decal_index].orm_rect.zw).xyz;
				} else {
					decal_orm = textureLod(sampler2D(decal_atlas, decal_sampler), uv_local.xz * decals.data[decal_index].orm_rect.zw + decals.data[decal_index].orm_rect.xy, 0.0).xyz;
				}
				ao = half(mix(float(ao), decal_orm.r, decal_albedo.a));
				roughness = half(mix(float(roughness), decal_orm.g, decal_albedo.a));
				metallic = half(mix(float(metallic), decal_orm.b, decal_albedo.a));
			}
		}

		if (decals.data[decal_index].emission_rect != vec4(0.0)) {
			//emission is additive, so its independent from albedo
			if (sc_decal_use_mipmaps()) {
				emission += hvec3(textureGrad(sampler2D(decal_atlas_srgb, decal_sampler), uv_local.xz * decals.data[decal_index].emission_rect.zw + decals.data[decal_index].emission_rect.xy, ddx * decals.data[decal_index].emission_rect.zw, ddy * decals.data[decal_index].emission_rect.zw).xyz * decals.data[decal_index].emission_energy * fade);
			} else {
				emission += hvec3(textureLod(sampler2D(decal_atlas_srgb, decal_sampler), uv_local.xz * decals.data[decal_index].emission_rect.zw + decals.data[decal_index].emission_rect.xy, 0.0).xyz * decals.data[decal_index].emission_energy * fade);
			}
		}
	}
#endif //!MODE_RENDER_DEPTH

	/////////////////////// LIGHTING //////////////////////////////

#ifdef NORMAL_USED
	if (sc_scene_roughness_limiter_enabled()) {
		//https://www.jp.square-enix.com/tech/library/pdf/ImprovedGeometricSpecularAA.pdf
		// SPIR-V Validation claims that derivatives of FP16 vectors are not valid code generation (see #108009).
		vec3 dn = vec3(normal);
		vec3 dndu = dFdx(dn), dndv = dFdy(dn);
		half roughness2 = roughness * roughness;
		half variance = half(scene_data.roughness_limiter_amount) * half(dot(dndu, dndu) + dot(dndv, dndv));
		half kernelRoughness2 = min(half(2.0) * variance, half(scene_data.roughness_limiter_limit));
		half filteredRoughness2 = min(half(1.0), roughness2 + kernelRoughness2);
		roughness = sqrt(filteredRoughness2);
	}
#endif // NORMAL_USED
	//apply energy conservation

	hvec3 indirect_specular_light = hvec3(0.0);
	hvec3 direct_specular_light = hvec3(0.0);
	hvec3 diffuse_light = hvec3(0.0);
	hvec3 ambient_light = hvec3(0.0);

#ifndef MODE_UNSHADED
	// Used in regular draw pass and when drawing SDFs for SDFGI and materials for VoxelGI.
	emission *= half(scene_data.emissive_exposure_normalization);
#endif

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#ifndef AMBIENT_LIGHT_DISABLED
#ifdef BENT_NORMAL_MAP_USED
	hvec3 indirect_normal = bent_normal_vector;
#else
	hvec3 indirect_normal = normal;
#endif

	if (sc_scene_use_reflection_cubemap()) {
#ifdef LIGHT_ANISOTROPY_USED
		// https://google.github.io/filament/Filament.html#lighting/imagebasedlights/anisotropy
		hvec3 anisotropic_direction = anisotropy >= 0.0 ? binormal : tangent;
		hvec3 anisotropic_tangent = cross(anisotropic_direction, view);
		hvec3 anisotropic_normal = cross(anisotropic_tangent, anisotropic_direction);
		hvec3 bent_normal = normalize(mix(indirect_normal, anisotropic_normal, anisotropy * clamp(half(5.0) * roughness, half(0.0), half(1.0))));
		hvec3 ref_vec = reflect(-view, bent_normal);
		ref_vec = mix(ref_vec, bent_normal, roughness * roughness);
#else
		hvec3 ref_vec = reflect(-view, indirect_normal);
		ref_vec = mix(ref_vec, indirect_normal, roughness * roughness);
#endif
		half horizon = min(half(1.0) + dot(ref_vec, indirect_normal), half(1.0));
		ref_vec = hvec3(scene_data.radiance_inverse_xform * vec3(ref_vec));
#ifdef USE_RADIANCE_OCTMAP_ARRAY
		float lod;
		half blend = half(modf(float(sqrt(roughness) * MAX_ROUGHNESS_LOD), lod));

		float ref_lod = vec3_to_oct_lod(dFdx(ref_vec), dFdy(ref_vec), scene_data_block.data.radiance_pixel_size);
		vec2 ref_uv = vec3_to_oct_with_border(ref_vec, vec2(scene_data_block.data.radiance_border_size, 1.0 - scene_data_block.data.radiance_border_size * 2.0));
		hvec3 indirect_sample_a = hvec3(textureLod(sampler2DArray(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(ref_uv, float(lod)), ref_lod).rgb);
		hvec3 indirect_sample_b = hvec3(textureLod(sampler2DArray(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(ref_uv, float(lod) + 1.0), ref_lod).rgb);
		indirect_specular_light = mix(indirect_sample_a, indirect_sample_b, blend);
#else // USE_RADIANCE_OCTMAP_ARRAY
		float lod = sqrt(roughness) * MAX_ROUGHNESS_LOD;
		vec2 ref_uv = vec3_to_oct_with_border(ref_vec, vec2(scene_data_block.data.radiance_border_size, 1.0 - scene_data_block.data.radiance_border_size * 2.0));
		indirect_specular_light = hvec3(textureLod(sampler2D(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), ref_uv, lod).rgb);
#endif //USE_RADIANCE_OCTMAP_ARRAY
		indirect_specular_light *= REFLECTION_MULTIPLIER;
		indirect_specular_light *= half(scene_data.IBL_exposure_normalization);
		indirect_specular_light *= horizon * horizon;
		indirect_specular_light *= half(scene_data.ambient_light_color_energy.a);
	}

#if defined(CUSTOM_RADIANCE_USED)
	indirect_specular_light = mix(indirect_specular_light, custom_radiance.rgb, custom_radiance.a);
#endif // CUSTOM_RADIANCE_USED

#ifndef USE_LIGHTMAP
	//lightmap overrides everything
	if (bool(scene_data.flags & SCENE_DATA_FLAGS_USE_AMBIENT_LIGHT)) {
		ambient_light = hvec3(scene_data.ambient_light_color_energy.rgb);

		if (sc_scene_use_ambient_cubemap()) {
			vec3 ambient_dir = scene_data.radiance_inverse_xform * indirect_normal;
#ifdef USE_RADIANCE_OCTMAP_ARRAY
			float ambient_lod = vec3_to_oct_lod(dFdx(ambient_dir), dFdy(ambient_dir), scene_data_block.data.radiance_pixel_size);
			vec2 ambient_uv = vec3_to_oct_with_border(ambient_dir, vec2(scene_data_block.data.radiance_border_size, 1.0 - scene_data_block.data.radiance_border_size * 2.0));
			hvec3 octmap_ambient = hvec3(textureLod(sampler2DArray(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(ambient_uv, MAX_ROUGHNESS_LOD), ambient_lod).rgb);
#else
			float roughness_lod = MAX_ROUGHNESS_LOD;
			vec2 ambient_uv = vec3_to_oct_with_border(ambient_dir, vec2(scene_data_block.data.radiance_border_size, 1.0 - scene_data_block.data.radiance_border_size * 2.0));
			hvec3 octmap_ambient = hvec3(textureLod(sampler2D(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), ambient_uv, roughness_lod).rgb);
#endif //USE_RADIANCE_OCTMAP_ARRAY
			octmap_ambient *= REFLECTION_MULTIPLIER;
			octmap_ambient *= half(scene_data.IBL_exposure_normalization);
			ambient_light = mix(ambient_light, octmap_ambient * half(scene_data.ambient_light_color_energy.a), half(scene_data.ambient_color_sky_mix));
		}
	}
#endif // !USE_LIGHTMAP

#if defined(CUSTOM_IRRADIANCE_USED)
	ambient_light = mix(ambient_light, custom_irradiance.rgb, custom_irradiance.a);
#endif // CUSTOM_IRRADIANCE_USED
#ifdef LIGHT_CLEARCOAT_USED

	if (sc_scene_use_reflection_cubemap()) {
		half NoV = max(dot(geo_normal, view), half(0.0001));
		hvec3 ref_vec = reflect(-view, geo_normal);
		ref_vec = mix(ref_vec, geo_normal, clearcoat_roughness * clearcoat_roughness);
		// The clear coat layer assumes an IOR of 1.5 (4% reflectance)
		half Fc = clearcoat * (half(0.04) + half(0.96) * SchlickFresnel(NoV));
		half attenuation = half(1.0) - Fc;
		ambient_light *= attenuation;
		indirect_specular_light *= attenuation;

		half horizon = min(half(1.0) + dot(ref_vec, indirect_normal), half(1.0));
		ref_vec = hvec3(scene_data.radiance_inverse_xform * vec3(ref_vec));
		float roughness_lod = mix(0.001, 0.1, sqrt(float(clearcoat_roughness))) * MAX_ROUGHNESS_LOD;
#ifdef USE_RADIANCE_OCTMAP_ARRAY
		float lod;
		half blend = half(modf(roughness_lod, lod));

		float ref_lod = vec3_to_oct_lod(dFdx(ref_vec), dFdy(ref_vec), scene_data_block.data.radiance_pixel_size);
		vec2 ref_uv = vec3_to_oct_with_border(ref_vec, vec2(scene_data_block.data.radiance_border_size, 1.0 - scene_data_block.data.radiance_border_size * 2.0));
		hvec3 clearcoat_sample_a = hvec3(textureLod(sampler2DArray(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(ref_uv, lod), ref_lod).rgb);
		hvec3 clearcoat_sample_b = hvec3(textureLod(sampler2DArray(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec3(ref_uv, lod + 1), ref_lod).rgb);
		hvec3 clearcoat_light = mix(clearcoat_sample_a, clearcoat_sample_b, blend);
#else
		vec2 ref_uv = vec3_to_oct_with_border(ref_vec, vec2(scene_data_block.data.radiance_border_size, 1.0 - scene_data_block.data.radiance_border_size * 2.0));
		hvec3 clearcoat_light = hvec3(textureLod(sampler2D(radiance_octmap, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), ref_uv, roughness_lod).rgb);
#endif //USE_RADIANCE_OCTMAP_ARRAY
		indirect_specular_light += clearcoat_light * horizon * horizon * Fc * half(scene_data.ambient_light_color_energy.a);
	}
#endif // LIGHT_CLEARCOAT_USED
#endif // !AMBIENT_LIGHT_DISABLED
#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

	//radiance

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)
#ifndef AMBIENT_LIGHT_DISABLED
#ifdef USE_LIGHTMAP

	//lightmap
	if (bool(instances.data[draw_call.instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP_CAPTURE)) { //has lightmap capture
		uint index = instances.data[draw_call.instance_index].gi_offset;

		// The world normal.
		hvec3 wnormal = hmat3(inv_view_matrix) * indirect_normal;

		// The SH coefficients used for evaluating diffuse data from SH probes.
		const half c[5] = half[](
				half(0.886227), // l0 				sqrt(1.0/(4.0*PI)) 	* PI
				half(1.023327), // l1 				sqrt(3.0/(4.0*PI)) 	* PI*2.0/3.0
				half(0.858086), // l2n2, l2n1, l2p1	sqrt(15.0/(4.0*PI)) * PI*1.0/4.0
				half(0.247708), // l20 				sqrt(5.0/(16.0*PI)) * PI*1.0/4.0
				half(0.429043) // l2p2 			sqrt(15.0/(16.0*PI))* PI*1.0/4.0
		);

		half norm = half(scene_data.IBL_exposure_normalization);
		ambient_light += c[0] * hvec3(lightmap_captures.data[index].sh[0].rgb) * norm;
		ambient_light += c[1] * hvec3(lightmap_captures.data[index].sh[1].rgb) * wnormal.y * norm;
		ambient_light += c[1] * hvec3(lightmap_captures.data[index].sh[2].rgb) * wnormal.z * norm;
		ambient_light += c[1] * hvec3(lightmap_captures.data[index].sh[3].rgb) * wnormal.x * norm;
		ambient_light += c[2] * hvec3(lightmap_captures.data[index].sh[4].rgb) * wnormal.x * wnormal.y * norm;
		ambient_light += c[2] * hvec3(lightmap_captures.data[index].sh[5].rgb) * wnormal.y * wnormal.z * norm;
		ambient_light += c[3] * hvec3(lightmap_captures.data[index].sh[6].rgb) * (half(3.0) * wnormal.z * wnormal.z - half(1.0)) * norm;
		ambient_light += c[2] * hvec3(lightmap_captures.data[index].sh[7].rgb) * wnormal.x * wnormal.z * norm;
		ambient_light += c[4] * hvec3(lightmap_captures.data[index].sh[8].rgb) * (wnormal.x * wnormal.x - wnormal.y * wnormal.y) * norm;

	} else if (bool(instances.data[draw_call.instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) { // has actual lightmap
		bool uses_sh = bool(instances.data[draw_call.instance_index].flags & INSTANCE_FLAGS_USE_SH_LIGHTMAP);
		uint ofs = instances.data[draw_call.instance_index].gi_offset & 0xFFFF;
		uint slice = instances.data[draw_call.instance_index].gi_offset >> 16;
		vec3 uvw;
		uvw.xy = uv2 * instances.data[draw_call.instance_index].lightmap_uv_scale.zw + instances.data[draw_call.instance_index].lightmap_uv_scale.xy;
		uvw.z = float(slice);

		if (uses_sh) {
			uvw.z *= 4.0; //SH textures use 4 times more data
			hvec3 lm_light_l0;
			hvec3 lm_light_l1n1;
			hvec3 lm_light_l1_0;
			hvec3 lm_light_l1p1;

			if (sc_use_lightmap_bicubic_filter()) {
				lm_light_l0 = hvec3(textureArray_bicubic(lightmap_textures[ofs], uvw + vec3(0.0, 0.0, 0.0), lightmaps.data[ofs].light_texture_size).rgb);
				lm_light_l1n1 = hvec3((textureArray_bicubic(lightmap_textures[ofs], uvw + vec3(0.0, 0.0, 1.0), lightmaps.data[ofs].light_texture_size).rgb - vec3(0.5)) * 2.0);
				lm_light_l1_0 = hvec3((textureArray_bicubic(lightmap_textures[ofs], uvw + vec3(0.0, 0.0, 2.0), lightmaps.data[ofs].light_texture_size).rgb - vec3(0.5)) * 2.0);
				lm_light_l1p1 = hvec3((textureArray_bicubic(lightmap_textures[ofs], uvw + vec3(0.0, 0.0, 3.0), lightmaps.data[ofs].light_texture_size).rgb - vec3(0.5)) * 2.0);
			} else {
				lm_light_l0 = hvec3(textureLod(sampler2DArray(lightmap_textures[ofs], SAMPLER_LINEAR_CLAMP), uvw + vec3(0.0, 0.0, 0.0), 0.0).rgb);
				lm_light_l1n1 = hvec3((textureLod(sampler2DArray(lightmap_textures[ofs], SAMPLER_LINEAR_CLAMP), uvw + vec3(0.0, 0.0, 1.0), 0.0).rgb - vec3(0.5)) * 2.0);
				lm_light_l1_0 = hvec3((textureLod(sampler2DArray(lightmap_textures[ofs], SAMPLER_LINEAR_CLAMP), uvw + vec3(0.0, 0.0, 2.0), 0.0).rgb - vec3(0.5)) * 2.0);
				lm_light_l1p1 = hvec3((textureLod(sampler2DArray(lightmap_textures[ofs], SAMPLER_LINEAR_CLAMP), uvw + vec3(0.0, 0.0, 3.0), 0.0).rgb - vec3(0.5)) * 2.0);
			}

			hvec3 n = hvec3(normalize(lightmaps.data[ofs].normal_xform * indirect_normal));
			half exposure_normalization = half(lightmaps.data[ofs].exposure_normalization);

			ambient_light += lm_light_l0 * exposure_normalization;
			ambient_light += lm_light_l1n1 * n.y * lm_light_l0 * exposure_normalization * half(4.0);
			ambient_light += lm_light_l1_0 * n.z * lm_light_l0 * exposure_normalization * half(4.0);
			ambient_light += lm_light_l1p1 * n.x * lm_light_l0 * exposure_normalization * half(4.0);
		} else {
			if (sc_use_lightmap_bicubic_filter()) {
				ambient_light += hvec3(textureArray_bicubic(lightmap_textures[ofs], uvw, lightmaps.data[ofs].light_texture_size).rgb * lightmaps.data[ofs].exposure_normalization);
			} else {
				ambient_light += hvec3(textureLod(sampler2DArray(lightmap_textures[ofs], SAMPLER_LINEAR_CLAMP), uvw, 0.0).rgb * lightmaps.data[ofs].exposure_normalization);
			}
		}
	}

	// No GI nor non low end mode...

#endif // USE_LIGHTMAP

	// skipping ssao, do we remove ssao totally?

	uint reflection_probe_count = sc_reflection_probes(8);
	if (reflection_probe_count > 0) {
		hvec4 reflection_accum = hvec4(0.0);
		hvec4 ambient_accum = hvec4(0.0);

#ifdef LIGHT_ANISOTROPY_USED
		// https://google.github.io/filament/Filament.html#lighting/imagebasedlights/anisotropy
		hvec3 anisotropic_direction = anisotropy >= 0.0 ? binormal : tangent;
		hvec3 anisotropic_tangent = cross(anisotropic_direction, view);
		hvec3 anisotropic_normal = cross(anisotropic_tangent, anisotropic_direction);
		hvec3 bent_normal = normalize(mix(normal, anisotropic_normal, abs(anisotropy) * clamp(half(5.0) * roughness, half(0.0), half(1.0))));
#else
		hvec3 bent_normal = normal;
#endif
		hvec3 ref_vec = normalize(reflect(-view, bent_normal));
		// Interpolate between mirror and rough reflection by using linear_roughness * linear_roughness.
		ref_vec = mix(ref_vec, bent_normal, roughness * roughness * roughness * roughness);

		uvec2 reflection_indices = instances.data[draw_call.instance_index].reflection_probes;
		for (uint i = 0; i < reflection_probe_count; i++) {
			uint reflection_index = (i > 3) ? ((reflection_indices.y >> ((i - 4) * 8)) & 0xFF) : ((reflection_indices.x >> (i * 8)) & 0xFF);
			if (reflection_index == 0xFF) {
				break;
			}

			if (reflection_accum.a >= half(1.0) && ambient_accum.a >= half(1.0)) {
				break;
			}

			reflection_process(reflection_index, vertex, ref_vec, normal, roughness, ambient_light, indirect_specular_light, ambient_accum, reflection_accum);
		}

		if (ambient_accum.a < half(1.0)) {
			ambient_accum.rgb = ambient_light * (half(1.0) - ambient_accum.a) + ambient_accum.rgb;
		}

		if (reflection_accum.a < half(1.0)) {
			reflection_accum.rgb = indirect_specular_light * (half(1.0) - reflection_accum.a) + reflection_accum.rgb;
		}

		if (reflection_accum.a > half(0.0)) {
			indirect_specular_light = reflection_accum.rgb;
		}

#if !defined(USE_LIGHTMAP)
		if (ambient_accum.a > half(0.0)) {
			ambient_light = ambient_accum.rgb;
		}
#endif
	} //Reflection probes

	// finalize ambient light here
	ambient_light *= ao;
#ifndef SPECULAR_OCCLUSION_DISABLED
#ifdef BENT_NORMAL_MAP_USED
	// Simplified bent normal occlusion.
	half cos_b = max(dot(reflect(-view, normal), bent_normal_vector), half(0.0));
	half specular_occlusion = clamp((ao - (half(1.0) - cos_b)) / roughness, half(0.0), half(1.0));
	specular_occlusion = mix(specular_occlusion, cos_b * (half(1.0) - ao), roughness);
	indirect_specular_light *= specular_occlusion;
#else // BENT_NORMAL_MAP_USED
	half specular_occlusion = (ambient_light.r * half(0.3) + ambient_light.g * half(0.59) + ambient_light.b * half(0.11)) * half(2.0); // Luminance of ambient light.
	specular_occlusion = min(specular_occlusion * half(4.0), half(1.0)); // This multiplication preserves speculars on bright areas.

	half reflective_f = (half(1.0) - roughness) * metallic;
	// 10.0 is a magic number, it gives the intended effect in most scenarios.
	// Low enough for occlusion, high enough for reaction to lights and shadows.
	specular_occlusion = max(min(reflective_f * specular_occlusion * half(10.0), half(1.0)), specular_occlusion);
	indirect_specular_light *= specular_occlusion;
#endif // BENT_NORMAL_MAP_USED
#endif // USE_SPECULAR_OCCLUSION
	ambient_light *= albedo.rgb;

#endif // !AMBIENT_LIGHT_DISABLED

	// convert ao to direct light ao
	ao = mix(half(1.0), ao, ao_light_affect);

	//this saves some VGPRs
	hvec3 f0 = F0(metallic, specular, albedo);

#ifndef AMBIENT_LIGHT_DISABLED
	{
#if defined(DIFFUSE_TOON)
		//simplify for toon, as
		indirect_specular_light *= specular * metallic * albedo * half(2.0);
#else

		// scales the specular reflections, needs to be computed before lighting happens,
		// but after environment, GI, and reflection probes are added
		// Environment brdf approximation (Lazarov 2013)
		// see https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
		const hvec4 c0 = hvec4(-1.0, -0.0275, -0.572, 0.022);
		const hvec4 c1 = hvec4(1.0, 0.0425, 1.04, -0.04);
		hvec4 r = roughness * c0 + c1;
		half ndotv = clamp(dot(normal, view), half(0.0), half(1.0));
		half a004 = min(r.x * r.x, exp2(half(-9.28) * ndotv)) * r.x + r.y;
		hvec2 env = hvec2(-1.04, 1.04) * a004 + r.zw;

		indirect_specular_light *= env.x * f0 + env.y * clamp(half(50.0) * f0.g, metallic, half(1.0));
#endif
	}

#endif // !AMBIENT_LIGHT_DISABLED
#endif // !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

// LIGHTING
#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)
#ifdef USE_VERTEX_LIGHTING
	diffuse_light += hvec3(diffuse_light_interp.rgb);
	direct_specular_light += hvec3(specular_light_interp.rgb) * f0;
#endif

	uint directional_lights_count = sc_directional_lights(scene_data.directional_light_count);
	if (directional_lights_count > 0) {
#ifndef SHADOWS_DISABLED
		// Do shadow and lighting in two passes to reduce register pressure
		half shadows[8];

		half shadowmask = half(1.0);

#ifdef USE_LIGHTMAP
		uint shadowmask_mode = LIGHTMAP_SHADOWMASK_MODE_NONE;

		if (bool(instances.data[draw_call.instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
			const uint ofs = instances.data[draw_call.instance_index].gi_offset & 0xFFFF;
			shadowmask_mode = lightmaps.data[ofs].flags;

			if (shadowmask_mode != LIGHTMAP_SHADOWMASK_MODE_NONE) {
				const uint slice = instances.data[draw_call.instance_index].gi_offset >> 16;
				const vec2 scaled_uv = uv2 * instances.data[draw_call.instance_index].lightmap_uv_scale.zw + instances.data[draw_call.instance_index].lightmap_uv_scale.xy;
				const vec3 uvw = vec3(scaled_uv, float(slice));

				if (sc_use_lightmap_bicubic_filter()) {
					shadowmask = half(textureArray_bicubic(lightmap_textures[MAX_LIGHTMAP_TEXTURES + ofs], uvw, lightmaps.data[ofs].light_texture_size).x);
				} else {
					shadowmask = half(textureLod(sampler2DArray(lightmap_textures[MAX_LIGHTMAP_TEXTURES + ofs], SAMPLER_LINEAR_CLAMP), uvw, 0.0).x);
				}
			}
		}

		if (shadowmask_mode != LIGHTMAP_SHADOWMASK_MODE_ONLY) {
#endif // USE_LIGHTMAP

#ifdef USE_VERTEX_LIGHTING
			// Only process the first light's shadow for vertex lighting.
			for (uint i = 0; i < 1; i++) {
#else
		for (uint i = 0; i < directional_lights_count; i++) {
#endif
				if (!bool(directional_lights.data[i].mask & instances.data[draw_call.instance_index].layer_mask)) {
					continue; //not masked
				}

				if (directional_lights.data[i].bake_mode == LIGHT_BAKE_STATIC && bool(instances.data[draw_call.instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
					continue; // Statically baked light and object uses lightmap, skip.
				}

				half shadow = half(1.0);

				if (directional_lights.data[i].shadow_opacity > 0.001) {
					float depth_z = -vertex.z;

					vec4 pssm_coord;
					float blur_factor;
					hvec3 light_dir = hvec3(directional_lights.data[i].direction);
					hvec3 base_normal_bias = geo_normal * (half(1.0) - max(half(0.0), dot(light_dir, -geo_normal)));

#define BIAS_FUNC(m_var, m_idx)                                                                        \
	hvec3 normal_bias = base_normal_bias * half(directional_lights.data[i].shadow_normal_bias[m_idx]); \
	normal_bias -= light_dir * dot(light_dir, normal_bias);                                            \
	normal_bias += light_dir * half(directional_lights.data[i].shadow_bias[m_idx]);                    \
	m_var.xyz += vec3(normal_bias);

					if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
						vec4 v = vec4(vertex, 1.0);

						BIAS_FUNC(v, 0)

						pssm_coord = (directional_lights.data[i].shadow_matrix1 * v);
						blur_factor = 1.0;
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
						vec4 v = vec4(vertex, 1.0);

						BIAS_FUNC(v, 1)

						pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
						// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
						blur_factor = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.y;
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
						vec4 v = vec4(vertex, 1.0);

						BIAS_FUNC(v, 2)

						pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
						// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
						blur_factor = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.z;
					} else {
						vec4 v = vec4(vertex, 1.0);

						BIAS_FUNC(v, 3)

						pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
						// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
						blur_factor = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.w;
					}

					pssm_coord /= pssm_coord.w;

					bool blend_split = sc_directional_light_blend_split(i);
					float blend_split_weight = blend_split ? 1.0f : 0.0f;
					shadow = half(sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale * (blur_factor + (1.0 - blur_factor) * blend_split_weight), pssm_coord, scene_data.taa_frame_count));

					if (blend_split) {
						half pssm_blend;
						float blur_factor2;

						if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
							vec4 v = vec4(vertex, 1.0);
							BIAS_FUNC(v, 1)
							pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
							pssm_blend = half(smoothstep(directional_lights.data[i].shadow_split_offsets.x - directional_lights.data[i].shadow_split_offsets.x * 0.1, directional_lights.data[i].shadow_split_offsets.x, depth_z));
							// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
							blur_factor2 = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.y;
						} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
							vec4 v = vec4(vertex, 1.0);
							BIAS_FUNC(v, 2)
							pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
							pssm_blend = half(smoothstep(directional_lights.data[i].shadow_split_offsets.y - directional_lights.data[i].shadow_split_offsets.y * 0.1, directional_lights.data[i].shadow_split_offsets.y, depth_z));
							// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
							blur_factor2 = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.z;
						} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
							vec4 v = vec4(vertex, 1.0);
							BIAS_FUNC(v, 3)
							pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
							pssm_blend = half(smoothstep(directional_lights.data[i].shadow_split_offsets.z - directional_lights.data[i].shadow_split_offsets.z * 0.1, directional_lights.data[i].shadow_split_offsets.z, depth_z));
							// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
							blur_factor2 = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.w;
						} else {
							pssm_blend = half(0.0); //if no blend, same coord will be used (divide by z will result in same value, and already cached)
							blur_factor2 = 1.0;
						}

						pssm_coord /= pssm_coord.w;

						half shadow2 = half(sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale * (blur_factor2 + (1.0 - blur_factor2) * blend_split_weight), pssm_coord, scene_data.taa_frame_count));
						shadow = mix(shadow, shadow2, pssm_blend);
					}

#ifdef USE_LIGHTMAP
					if (shadowmask_mode == LIGHTMAP_SHADOWMASK_MODE_REPLACE) {
						shadow = mix(shadow, shadowmask, half(smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z))); //done with negative values for performance
					} else if (shadowmask_mode == LIGHTMAP_SHADOWMASK_MODE_OVERLAY) {
						shadow = shadowmask * mix(shadow, half(1.0), half(smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z))); //done with negative values for performance
					} else {
#endif
						shadow = mix(shadow, half(1.0), half(smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z)));
#ifdef USE_LIGHTMAP
					}
#endif

#ifdef USE_VERTEX_LIGHTING
					diffuse_light *= mix(half(1.0), shadow, half(diffuse_light_interp.a));
					direct_specular_light *= mix(half(1.0), shadow, half(specular_light_interp.a));
#endif
#undef BIAS_FUNC
				}

				shadows[i] = shadow;
			}

#ifdef USE_LIGHTMAP
		} else { // shadowmask_mode == LIGHTMAP_SHADOWMASK_MODE_ONLY

#ifdef USE_VERTEX_LIGHTING
			diffuse_light *= mix(half(1.0), shadowmask, half(diffuse_light_interp.a));
			direct_specular_light *= mix(half(1.0), shadowmask, half(specular_light_interp.a));
#endif

			shadows[0] = shadowmask;
		}
#endif // USE_LIGHTMAP

#endif // SHADOWS_DISABLED

#ifndef USE_VERTEX_LIGHTING
		uint directional_lights_count = sc_directional_lights(scene_data.directional_light_count);
		for (uint i = 0; i < directional_lights_count; i++) {
			if (!bool(directional_lights.data[i].mask & instances.data[draw_call.instance_index].layer_mask)) {
				continue; //not masked
			}

			if (directional_lights.data[i].bake_mode == LIGHT_BAKE_STATIC && bool(instances.data[draw_call.instance_index].flags & INSTANCE_FLAGS_USE_LIGHTMAP)) {
				continue; // Statically baked light and object uses lightmap, skip.
			}

			// We're not doing light transmittence

			half shadow = half(1.0);
#ifndef SHADOWS_DISABLED
			shadow = mix(half(1.0), shadows[i], half(directional_lights.data[i].shadow_opacity));
#endif
			blur_shadow(shadow);

			vec3 tint = vec3(1.0);
#ifdef DEBUG_DRAW_PSSM_SPLITS
			if (-vertex.z < directional_lights.data[i].shadow_split_offsets.x) {
				tint = vec3(1.0, 0.0, 0.0);
			} else if (-vertex.z < directional_lights.data[i].shadow_split_offsets.y) {
				tint = vec3(0.0, 1.0, 0.0);
			} else if (-vertex.z < directional_lights.data[i].shadow_split_offsets.z) {
				tint = vec3(0.0, 0.0, 1.0);
			} else {
				tint = vec3(1.0, 1.0, 0.0);
			}
			tint = mix(tint, vec3(1.0), float(shadow));
			shadow = half(1.0);
#endif

			float size_A = sc_use_light_soft_shadows() ? directional_lights.data[i].size : 0.0;

			light_compute(normal, hvec3(directional_lights.data[i].direction), view, saturateHalf(size_A),
					hvec3(directional_lights.data[i].color * directional_lights.data[i].energy * tint),
					true, shadow, f0, roughness, metallic, half(directional_lights.data[i].specular), albedo, alpha,
					screen_uv, hvec3(1.0),
#ifdef LIGHT_BACKLIGHT_USED
					backlight,
#endif
/* not supported here
#ifdef LIGHT_TRANSMITTANCE_USED
					transmittance_color,
					transmittance_depth,
					transmittance_boost,
					transmittance_z,
#endif
*/
#ifdef LIGHT_RIM_USED
					rim, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
					clearcoat, clearcoat_roughness, geo_normal,
#endif // LIGHT_CLEARCOAT_USED
#ifdef LIGHT_ANISOTROPY_USED
					binormal, tangent, anisotropy,
#endif
					diffuse_light,
					direct_specular_light);
		}
#endif // USE_VERTEX_LIGHTING
	} //directional light

#ifndef USE_VERTEX_LIGHTING
	uint omni_light_count = sc_omni_lights(8);
	uvec2 omni_indices = instances.data[draw_call.instance_index].omni_lights;
	for (uint i = 0; i < omni_light_count; i++) {
		uint light_index = (i > 3) ? ((omni_indices.y >> ((i - 4) * 8)) & 0xFF) : ((omni_indices.x >> (i * 8)) & 0xFF);
		if (i > 0 && light_index == 0xFF) {
			break;
		}

		light_process_omni(light_index, vertex, view, normal, vertex_ddx, vertex_ddy, f0, roughness, metallic, scene_data.taa_frame_count, albedo, alpha, screen_uv, hvec3(1.0),
#ifdef LIGHT_BACKLIGHT_USED
				backlight,
#endif
/*
#ifdef LIGHT_TRANSMITTANCE_USED
				transmittance_color,
				transmittance_depth,
				transmittance_boost,
#endif
*/
#ifdef LIGHT_RIM_USED
				rim,
				rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
				clearcoat, clearcoat_roughness, geo_normal,
#endif // LIGHT_CLEARCOAT_USED
#ifdef LIGHT_ANISOTROPY_USED
				binormal, tangent, anisotropy,
#endif
				diffuse_light, direct_specular_light);
	}

	uint spot_light_count = sc_spot_lights(8);
	uvec2 spot_indices = instances.data[draw_call.instance_index].spot_lights;
	for (uint i = 0; i < spot_light_count; i++) {
		uint light_index = (i > 3) ? ((spot_indices.y >> ((i - 4) * 8)) & 0xFF) : ((spot_indices.x >> (i * 8)) & 0xFF);
		if (i > 0 && light_index == 0xFF) {
			break;
		}

		light_process_spot(light_index, vertex, view, normal, vertex_ddx, vertex_ddy, f0, roughness, metallic, scene_data.taa_frame_count, albedo, alpha, screen_uv, hvec3(1.0),
#ifdef LIGHT_BACKLIGHT_USED
				backlight,
#endif
/*
#ifdef LIGHT_TRANSMITTANCE_USED
				transmittance_color,
				transmittance_depth,
				transmittance_boost,
#endif
*/
#ifdef LIGHT_RIM_USED
				rim,
				rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
				clearcoat, clearcoat_roughness, geo_normal,
#endif // LIGHT_CLEARCOAT_USED
#ifdef LIGHT_ANISOTROPY_USED
				binormal, tangent, anisotropy,
#endif
				diffuse_light, direct_specular_light);
	}
#endif // !VERTEX_LIGHTING

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#ifdef USE_SHADOW_TO_OPACITY
#ifndef MODE_RENDER_DEPTH
	alpha = min(alpha, clamp(length(ambient_light), half(0.0), half(1.0)));

#if defined(ALPHA_SCISSOR_USED)
#ifdef MODE_RENDER_MATERIAL
	if (alpha < alpha_scissor_threshold) {
		alpha = half(0.0);
	} else {
		alpha = half(1.0);
	}
#else
	if (alpha < alpha_scissor_threshold) {
		discard;
	}
#endif // MODE_RENDER_MATERIAL
#endif // !ALPHA_SCISSOR_USED

#endif // !MODE_RENDER_DEPTH
#endif // USE_SHADOW_TO_OPACITY

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_RENDER_MATERIAL

	albedo_output_buffer.rgb = albedo;
	albedo_output_buffer.a = alpha;

	normal_output_buffer.rgb = normal * 0.5 + 0.5;
	normal_output_buffer.a = 0.0;
	depth_output_buffer.r = -vertex.z;

	orm_output_buffer.r = ao;
	orm_output_buffer.g = roughness;
	orm_output_buffer.b = metallic;
	orm_output_buffer.a = sss_strength;

	emission_output_buffer.rgb = emission;
	emission_output_buffer.a = 0.0;
#endif // MODE_RENDER_MATERIAL

#else // MODE_RENDER_DEPTH

	// multiply by albedo
	diffuse_light *= albedo; // ambient must be multiplied by albedo at the end

	// apply direct light AO
	diffuse_light *= ao;
	direct_specular_light *= ao;

	// apply metallic
	diffuse_light *= half(1.0) - metallic;
	ambient_light *= half(1.0) - metallic;

#ifdef MODE_MULTIPLE_RENDER_TARGETS

#ifdef MODE_UNSHADED
	diffuse_buffer = vec4(albedo.rgb, 0.0);
	specular_buffer = vec4(0.0);

#else // MODE_UNSHADED

#ifdef SSS_MODE_SKIN
	sss_strength = -sss_strength;
#endif // SSS_MODE_SKIN
	diffuse_buffer = vec4(emission + diffuse_light + ambient_light, sss_strength);
	specular_buffer = vec4(direct_specular_light + indirect_specular_light, metallic);
#endif // MODE_UNSHADED

#ifndef FOG_DISABLED
	diffuse_buffer.rgb = mix(diffuse_buffer.rgb, fog.rgb, fog.a);
	specular_buffer.rgb = mix(specular_buffer.rgb, vec3(0.0), fog.a);
#endif // !FOG_DISABLED

#else //MODE_MULTIPLE_RENDER_TARGETS

#ifdef MODE_UNSHADED
	hvec4 out_color = hvec4(albedo, alpha);
#else // MODE_UNSHADED
	hvec4 out_color = hvec4(emission + ambient_light + diffuse_light + direct_specular_light + indirect_specular_light, alpha);
#endif // MODE_UNSHADED

#ifndef FOG_DISABLED
	// Draw "fixed" fog before volumetric fog to ensure volumetric fog can appear in front of the sky.
	out_color.rgb = mix(out_color.rgb, fog.rgb, fog.a);
#endif // !FOG_DISABLED

	// On mobile we use a UNORM buffer with 10bpp which results in a range from 0.0 - 1.0 resulting in HDR breaking
	// We divide by sc_luminance_multiplier to support a range from 0.0 - 2.0 both increasing precision on bright and darker images
	out_color.rgb = out_color.rgb / sc_luminance_multiplier();
#ifdef PREMUL_ALPHA_USED
	out_color.rgb *= premul_alpha;
#endif

	frag_color = out_color;

	if (sc_use_material_debanding()) {
		// From https://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
		// and https://www.shadertoy.com/view/MslGR8 (5th one starting from the bottom)
		// NOTE: `gl_FragCoord` is in pixels (i.e. not normalized UV).
		// This dithering must be applied after encoding changes (linear/nonlinear) have been applied
		// as the final step before quantization from floating point to integer values.

		// Iestyn's RGB dither (7 asm instructions) from Portal 2 X360, slightly modified for VR.
		// Removed the time component to avoid passing time into this shader.
		// This dither offset was chosen because it meshes nicely with the no-offset dither that
		// is used for Viewport debanding.
		const vec2 dither_offset = vec2(0.535, 8.715);
		vec3 dither = vec3(dot(vec2(171.0, 231.0), gl_FragCoord.xy + dither_offset));
		dither.rgb = fract(dither.rgb / vec3(103.0, 71.0, 97.0));

		// Subtract 0.5 to avoid slightly brightening the whole viewport.
		// Use a dither strength of 100% rather than the 37.5% suggested by the original source.
		// Assume that this shader always writes to a 10-bit buffer, so divide by 1023 to align
		// to 10-bit quantization.
		frag_color.rgb += (dither.rgb - 0.5) / 1023.0;
	}

#endif //MODE_MULTIPLE_RENDER_TARGETS

#endif //MODE_RENDER_DEPTH

#ifdef MODE_RENDER_MOTION_VECTORS
	// These motion vectors are in NDC space (as opposed to screen space) to fit the OpenXR XR_FB_space_warp specification.
	// https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_FB_space_warp

	vec3 ndc = screen_position.xyz / screen_position.w;
	ndc.y = -ndc.y;
	vec3 prev_ndc = prev_screen_position.xyz / prev_screen_position.w;
	prev_ndc.y = -prev_ndc.y;
	frag_color = vec4(ndc - prev_ndc, 0.0);
#endif
}
