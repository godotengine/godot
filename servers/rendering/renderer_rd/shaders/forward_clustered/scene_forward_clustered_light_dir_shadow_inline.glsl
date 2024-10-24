// Inlined directional shadow logic, use scene_data_block to access data
if (directional_lights.data[i].shadow_opacity > 0.001) {
	float depth_z = -vertex.z;
	vec3 light_dir = directional_lights.data[i].direction;
	vec3 base_normal_bias = normalize(normal_interp) * (1.0 - max(0.0, dot(light_dir, -normalize(normal_interp))));

#define BIAS_FUNC(m_var, m_idx)                                                                 \
	m_var.xyz += light_dir * directional_lights.data[i].shadow_bias[m_idx];                     \
	vec3 normal_bias = base_normal_bias * directional_lights.data[i].shadow_normal_bias[m_idx]; \
	normal_bias -= light_dir * dot(light_dir, normal_bias);                                     \
	m_var.xyz += normal_bias;

	//version with soft shadows, more expensive
	if (sc_use_directional_soft_shadows() && directional_lights.data[i].softshadow_angle > 0) {
		uint blend_count = 0;
		const uint blend_max = directional_lights.data[i].blend_splits ? 2 : 1;

		if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
			vec4 v = vec4(vertex, 1.0);

			BIAS_FUNC(v, 0)

			vec4 pssm_coord = (directional_lights.data[i].shadow_matrix1 * v);
			pssm_coord /= pssm_coord.w;

			float range_pos = dot(directional_lights.data[i].direction, v.xyz);
			float range_begin = directional_lights.data[i].shadow_range_begin.x;
			float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
			vec2 tex_scale = directional_lights.data[i].uv_scale1 * test_radius;
			shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale, scene_data_block.data.taa_frame_count);
			blend_count++;
		}

		if (blend_count < blend_max && depth_z < directional_lights.data[i].shadow_split_offsets.y) {
			vec4 v = vec4(vertex, 1.0);

			BIAS_FUNC(v, 1)

			vec4 pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
			pssm_coord /= pssm_coord.w;

			float range_pos = dot(directional_lights.data[i].direction, v.xyz);
			float range_begin = directional_lights.data[i].shadow_range_begin.y;
			float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
			vec2 tex_scale = directional_lights.data[i].uv_scale2 * test_radius;
			float s = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale, scene_data_block.data.taa_frame_count);

			if (blend_count == 0) {
				shadow = s;
			} else {
				//blend
				float blend = smoothstep(0.0, directional_lights.data[i].shadow_split_offsets.x, depth_z);
				shadow = mix(shadow, s, blend);
			}

			blend_count++;
		}

		if (blend_count < blend_max && depth_z < directional_lights.data[i].shadow_split_offsets.z) {
			vec4 v = vec4(vertex, 1.0);

			BIAS_FUNC(v, 2)

			vec4 pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
			pssm_coord /= pssm_coord.w;

			float range_pos = dot(directional_lights.data[i].direction, v.xyz);
			float range_begin = directional_lights.data[i].shadow_range_begin.z;
			float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
			vec2 tex_scale = directional_lights.data[i].uv_scale3 * test_radius;
			float s = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale, scene_data_block.data.taa_frame_count);

			if (blend_count == 0) {
				shadow = s;
			} else {
				//blend
				float blend = smoothstep(directional_lights.data[i].shadow_split_offsets.x, directional_lights.data[i].shadow_split_offsets.y, depth_z);
				shadow = mix(shadow, s, blend);
			}

			blend_count++;
		}

		if (blend_count < blend_max) {
			vec4 v = vec4(vertex, 1.0);

			BIAS_FUNC(v, 3)

			vec4 pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
			pssm_coord /= pssm_coord.w;

			float range_pos = dot(directional_lights.data[i].direction, v.xyz);
			float range_begin = directional_lights.data[i].shadow_range_begin.w;
			float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
			vec2 tex_scale = directional_lights.data[i].uv_scale4 * test_radius;
			float s = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale, scene_data_block.data.taa_frame_count);

			if (blend_count == 0) {
				shadow = s;
			} else {
				//blend
				float blend = smoothstep(directional_lights.data[i].shadow_split_offsets.y, directional_lights.data[i].shadow_split_offsets.z, depth_z);
				shadow = mix(shadow, s, blend);
			}
		}

	} else { //no soft shadows

		vec4 pssm_coord;
		float blur_factor;

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

		shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data_block.data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale * (blur_factor + (1.0 - blur_factor) * float(directional_lights.data[i].blend_splits)), pssm_coord, scene_data_block.data.taa_frame_count);

		if (directional_lights.data[i].blend_splits) {
			float pssm_blend;
			float blur_factor2;

			if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
				vec4 v = vec4(vertex, 1.0);
				BIAS_FUNC(v, 1)
				pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
				pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.x - directional_lights.data[i].shadow_split_offsets.x * 0.1, directional_lights.data[i].shadow_split_offsets.x, depth_z);
				// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
				blur_factor2 = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.y;
			} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
				vec4 v = vec4(vertex, 1.0);
				BIAS_FUNC(v, 2)
				pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
				pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.y - directional_lights.data[i].shadow_split_offsets.y * 0.1, directional_lights.data[i].shadow_split_offsets.y, depth_z);
				// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
				blur_factor2 = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.z;
			} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
				vec4 v = vec4(vertex, 1.0);
				BIAS_FUNC(v, 3)
				pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
				pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.z - directional_lights.data[i].shadow_split_offsets.z * 0.1, directional_lights.data[i].shadow_split_offsets.z, depth_z);
				// Adjust shadow blur with reference to the first split to reduce discrepancy between shadow splits.
				blur_factor2 = directional_lights.data[i].shadow_split_offsets.x / directional_lights.data[i].shadow_split_offsets.w;
			} else {
				pssm_blend = 0.0; //if no blend, same coord will be used (divide by z will result in same value, and already cached)
				blur_factor2 = 1.0;
			}

			pssm_coord /= pssm_coord.w;

			float shadow2 = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data_block.data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale * (blur_factor2 + (1.0 - blur_factor2) * float(directional_lights.data[i].blend_splits)), pssm_coord, scene_data_block.data.taa_frame_count);
			shadow = mix(shadow, shadow2, pssm_blend);
		}
	}

	shadow = mix(shadow, 1.0, smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z)); //done with negative values for performance

#undef BIAS_FUNC
} // shadows

if (i < 4) {
	shadow0 |= uint(clamp(shadow * 255.0, 0.0, 255.0)) << (i * 8);
} else {
	shadow1 |= uint(clamp(shadow * 255.0, 0.0, 255.0)) << ((i - 4) * 8);
}
