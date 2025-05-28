// Simplified versions of light functions intended for the vertex shader.

// Eyeballed approximation of `exp2(15.0 * (1.0 - roughness) + 1.0) * 0.25`.
// Uses slightly more FMA instructions (2x rate) to avoid special instructions (0.25x rate).
// Range is reduced to [0.64,4977] from [068,2,221,528] which makes mediump feasible for the rest of the shader.
half roughness_to_shininess(half roughness) {
	half r = half(1.2) - roughness;
	half r2 = r * r;
	return r * r2 * r2 * half(2000.0);
}

void light_compute_vertex(hvec3 N, hvec3 L, hvec3 V, hvec3 light_color, bool is_directional, half roughness,
		inout hvec3 diffuse_light, inout hvec3 specular_light) {
	half NdotL = min(dot(N, L), half(1.0));
	half cNdotL = max(NdotL, half(0.0)); // clamped NdotL

#if defined(DIFFUSE_LAMBERT_WRAP)
	// Energy conserving lambert wrap shader.
	// https://web.archive.org/web/20210228210901/http://blog.stevemcauley.com/2011/12/03/energy-conserving-wrapped-diffuse/
	half diffuse_brdf_NL = max(half(0.0), (cNdotL + roughness) / ((half(1.0) + roughness) * (half(1.0) + roughness))) * half(1.0 / M_PI);
#else
	// lambert
	half diffuse_brdf_NL = cNdotL * half(1.0 / M_PI);
#endif

	diffuse_light += light_color * diffuse_brdf_NL;

#if !defined(SPECULAR_DISABLED)
	half specular_brdf_NL = half(0.0);
	// Normalized blinn always unless disabled.
	hvec3 H = normalize(V + L);
	half cNdotH = clamp(dot(N, H), half(0.0), half(1.0));
	half shininess = roughness_to_shininess(roughness);
	half blinn = pow(cNdotH, shininess);
	blinn *= (shininess + half(2.0)) * half(1.0 / (8.0 * M_PI)) * cNdotL;
	specular_brdf_NL = blinn;
	specular_light += specular_brdf_NL * light_color;
#endif
}

half get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return half(nd * pow(max(distance, 0.0001), -decay));
}

void light_process_omni_vertex(uint idx, vec3 vertex, hvec3 eye_vec, hvec3 normal, half roughness,
		inout hvec3 diffuse_light, inout hvec3 specular_light) {
	vec3 light_rel_vec = omni_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	hvec3 light_rel_vec_norm = hvec3(light_rel_vec / light_length);
	half omni_attenuation = get_omni_attenuation(light_length, omni_lights.data[idx].inv_radius, omni_lights.data[idx].attenuation);
	hvec3 color = hvec3(omni_lights.data[idx].color * omni_attenuation);

	light_compute_vertex(normal, light_rel_vec_norm, eye_vec, color, false, roughness,
			diffuse_light,
			specular_light);
}

void light_process_spot_vertex(uint idx, vec3 vertex, hvec3 eye_vec, hvec3 normal, half roughness,
		inout hvec3 diffuse_light,
		inout hvec3 specular_light) {
	vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	hvec3 light_rel_vec_norm = hvec3(light_rel_vec / light_length);
	half spot_attenuation = get_omni_attenuation(light_length, spot_lights.data[idx].inv_radius, spot_lights.data[idx].attenuation);
	hvec3 spot_dir = hvec3(spot_lights.data[idx].direction);

	half cone_angle = half(spot_lights.data[idx].cone_angle);
	half scos = max(dot(-light_rel_vec_norm, spot_dir), cone_angle);

	// This conversion to a highp float is crucial to prevent light leaking due to precision errors.
	float spot_rim = max(1e-4, float(half(1.0) - scos) / float(half(1.0) - cone_angle));
	spot_attenuation *= half(1.0 - pow(spot_rim, spot_lights.data[idx].cone_attenuation));

	hvec3 color = hvec3(spot_lights.data[idx].color * spot_attenuation);

	light_compute_vertex(normal, light_rel_vec_norm, eye_vec, color, false, roughness,
			diffuse_light, specular_light);
}
