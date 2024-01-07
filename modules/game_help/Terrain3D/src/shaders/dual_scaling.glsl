// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

R"(

//INSERT: DUAL_SCALING_UNIFORMS
uniform int dual_scale_texture : hint_range(0,31) = 0;
uniform float dual_scale_reduction : hint_range(0.001,1) = 0.3;
uniform float tri_scale_reduction : hint_range(0.001,1) = 0.3;
uniform float dual_scale_far : hint_range(0,1000) = 170.0;
uniform float dual_scale_near : hint_range(0,1000) = 100.0;
varying float v_far_factor;

//INSERT: DUAL_SCALING_VERTEX
		v_far_factor = clamp(smoothstep(dual_scale_near, dual_scale_far, v_vertex_dist), 0.0, 1.0);

//INSERT: DUAL_SCALING_BASE
	// If dual scaling, apply to base texture
	if(region < 0) {
		matUV *= tri_scale_reduction;
	}
	albedo_ht = texture(_texture_array_albedo, vec3(matUV, float(out_mat.base)));
	normal_rg = texture(_texture_array_normal, vec3(matUV, float(out_mat.base)));
	if(out_mat.base == dual_scale_texture || out_mat.over == dual_scale_texture) {
		albedo_far = texture(_texture_array_albedo, vec3(matUV*dual_scale_reduction, float(dual_scale_texture)));
		normal_far = texture(_texture_array_normal, vec3(matUV*dual_scale_reduction, float(dual_scale_texture)));
	}
	if(out_mat.base == dual_scale_texture) {
		albedo_ht = mix(albedo_ht, albedo_far, v_far_factor);
		normal_rg = mix(normal_rg, normal_far, v_far_factor);
	}

//INSERT: UNI_SCALING_BASE
	albedo_ht = texture(_texture_array_albedo, vec3(matUV, float(out_mat.base)));
	normal_rg = texture(_texture_array_normal, vec3(matUV, float(out_mat.base)));

//INSERT: DUAL_SCALING_OVERLAY
		// If dual scaling, apply to overlay texture
		if(out_mat.over == dual_scale_texture) {
			albedo_ht2 = mix(albedo_ht2, albedo_far, v_far_factor);
			normal_rg2 = mix(normal_rg2, normal_far, v_far_factor);
		}

)"