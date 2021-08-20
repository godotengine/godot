

vec3 F0(float metallic, float specular, vec3 albedo) {
	float dielectric = 0.16 * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(vec3(dielectric), albedo, vec3(metallic));
}

// Optimized version based on Lagrange's identity
// see: https://github.com/google/filament/blob/main/shaders/src/brdf.fs
float D_GGX(float roughness, float NoH, const vec3 h) {
	float oneMinusNoHSquared = 1.0 - NoH * NoH;
	float a = NoH * roughness;
	float k = roughness / (oneMinusNoHSquared + a * a);
	float d = k * k * (1.0 / M_PI);
	return min(d, MEDIUMP_FLT_MAX);
}

float D_GGX_Anisotropic(float at, float ab, float ToH, float BoH, float NoH) {
	// Burley 2012, "Physically-Based Shading at Disney"

	// The values at and ab are perceptualRoughness^2, a2 is therefore perceptualRoughness^4
	// The dot product below computes perceptualRoughness^8. We cannot fit in fp16 without clamping
	// the roughness to too high values so we perform the dot product and the division in fp32
	float a2 = at * ab;
	highp vec3 d = vec3(ab * ToH, at * BoH, a2 * NoH);
	highp float d2 = dot(d, d);
	float b2 = a2 / d2;
	return a2 * b2 * b2 * (1.0 / M_PI);
}

float V_SmithGGXCorrelated_Fast(float roughness, float NoV, float NoL) {
	// Hammon 2017, "PBR Diffuse Lighting for GGX+Smith Microsurfaces"
	float v = 0.5 / mix(2.0 * NoL * NoV, NoL + NoV, roughness);
	return min(v, MEDIUMP_FLT_MAX);
}

vec3 F_Schlick(const vec3 f0, float VoH) {
	float f = pow(1.0 - VoH, 5.0);
	return f + f0 * (1.0 - f);
}

float sample_directional_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec4 coord) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
}

float sample_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec4 coord) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
}

void light_compute(vec3 N, vec3 L, vec3 V, float A, vec3 light_color, float attenuation, vec3 f0, float metallic, float roughness, float specular_amount,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
		float transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint, vec3 rim_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 B, vec3 T, float anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
		inout float alpha,
#endif
		inout vec3 diffuse_light, inout vec3 specular_light) {

#if defined(LIGHT_CODE_USED)
	// light is written by the light shader

	vec3 normal = N;
	vec3 light = L;
	vec3 view = V;

#CODE : LIGHT

#else

	float NdotL = min(dot(N, L), 1.0);
	float cNdotL = max(NdotL, 0.0); // clamped NdotL
	float NdotV = dot(N, V);
	float cNdotV = max(NdotV, 0.0);

#if defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	vec3 H = normalize(V + L);
#endif

#if defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	float cNdotH = clamp(dot(N, H), 0.0, 1.0);
#endif

#if defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	float cLdotH = clamp(dot(L, H), 0.0, 1.0);
#endif

	if (metallic < 1.0) {
		float diffuse_brdf_NL; // BRDF times N.L for calculating diffuse radiance

#if defined(DIFFUSE_LAMBERT_WRAP)
		// energy conserving lambert wrap shader
		diffuse_brdf_NL = max(0.0, (NdotL + roughness) / ((1.0 + roughness) * (1.0 + roughness)));
#elif defined(DIFFUSE_TOON)

		diffuse_brdf_NL = smoothstep(-roughness, max(roughness, 0.01), NdotL);
#else
		// lambert
		diffuse_brdf_NL = cNdotL * (1.0 / M_PI);
#endif

		diffuse_light += light_color * diffuse_brdf_NL * attenuation;

#if defined(LIGHT_BACKLIGHT_USED)
		diffuse_light += light_color * (vec3(1.0 / M_PI) - diffuse_brdf_NL) * backlight * attenuation;
#endif

#if defined(LIGHT_RIM_USED)
		float rim_light = pow(max(0.0, 1.0 - cNdotV), max(0.0, (1.0 - roughness) * 16.0));
		diffuse_light += rim_light * rim * mix(vec3(1.0), rim_color, rim_tint) * light_color;
#endif
	}

	if (roughness > 0.0) { // FIXME: roughness == 0 should not disable specular light entirely

#if defined(SPECULAR_TOON)

		vec3 R = normalize(-reflect(L, N));
		float RdotV = dot(R, V);
		float mid = 1.0 - roughness;
		mid *= mid;
		float intensity = smoothstep(mid - roughness * 0.5, mid + roughness * 0.5, RdotV) * mid;
		diffuse_light += light_color * intensity * attenuation * specular_amount; // write to diffuse_light, as in toon shading you generally want no reflection

#elif defined(SPECULAR_DISABLED)
		// none..

#elif defined(SPECULAR_SCHLICK_GGX)
		// shlick+ggx as default

		float alpha_ggx = roughness * roughness;

		float D = D_GGX(alpha_ggx, cNdotH, H);
		float V = V_SmithGGXCorrelated_Fast(alpha_ggx, cNdotV, cNdotL);
		vec3 F = F_Schlick(f0, cLdotH);

		vec3 specular_brdf_NL = cNdotL * D * V * F;

		specular_light += specular_brdf_NL * light_color * attenuation * specular_amount;
#endif
	}

#ifdef USE_SHADOW_TO_OPACITY
	alpha = min(alpha, clamp(1.0 - attenuation), 0.0, 1.0));
#endif

#endif //defined(LIGHT_CODE_USED)
}

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

float light_process_omni_shadow(uint idx, highp vec3 vertex, vec3 normal) {
#ifndef USE_NO_SHADOWS
	if (omni_lights.data[idx].shadow_enabled) {
		// there is a shadowmap

		highp vec3 light_rel_vec = omni_lights.data[idx].position - vertex;
		highp float light_length = length(light_rel_vec);

		highp vec4 v = vec4(vertex, 1.0);

		highp vec4 splane = (omni_lights.data[idx].shadow_matrix * v);

		highp float shadow_len = length(splane.xyz); //need to remember shadow len from here

		float shadow;

		splane.xyz = normalize(splane.xyz);
		highp vec4 clamp_rect = omni_lights.data[idx].atlas_rect;

		if (splane.z >= 0.0) {
			splane.z += 1.0;

			clamp_rect.y += clamp_rect.w;

		} else {
			splane.z = 1.0 - splane.z;
		}

		splane.xy /= splane.z;

		splane.xy = splane.xy * 0.5 + 0.5;
		splane.z = shadow_len * omni_lights.data[idx].inv_radius;
		splane.z -= omni_lights.data[idx].shadow_bias;
		splane.xy = clamp_rect.xy + splane.xy * clamp_rect.zw;
		splane.w = 1.0; //needed? i think it should be 1 already
		shadow = sample_pcf_shadow(shadow_atlas, omni_lights.data[idx].soft_shadow_scale * scene_data.shadow_atlas_pixel_size, splane);

		return shadow;
	}
#endif

	return 1.0;
}

void light_process_omni(uint idx, highp vec3 vertex, vec3 eye_vec, vec3 normal, highp vec3 vertex_ddx, highp vec3 vertex_ddy, vec3 f0, float metallic, float roughness, float shadow,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint, vec3 rim_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 binormal, vec3 tangent, float anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
		inout float alpha,
#endif
		inout vec3 diffuse_light, inout vec3 specular_light) {
	vec3 light_rel_vec = omni_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float omni_attenuation = get_omni_attenuation(light_length, omni_lights.data[idx].inv_radius, omni_lights.data[idx].attenuation);
	float light_attenuation = omni_attenuation;
	vec3 color = omni_lights.data[idx].color;

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth; //no transmittance by default
	transmittance_color.a *= light_attenuation;
	{
		highp vec4 clamp_rect = omni_lights.data[idx].atlas_rect;

		//redo shadowmapping, but shrink the model a bit to avoid arctifacts
		highp vec4 splane = (omni_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * omni_lights.data[idx].transmittance_bias, 1.0));

		highp float shadow_len = length(splane.xyz);
		splane.xyz = normalize(splane.xyz);

		if (splane.z >= 0.0) {
			splane.z += 1.0;
			clamp_rect.y += clamp_rect.w;
		} else {
			splane.z = 1.0 - splane.z;
		}

		splane.xy /= splane.z;

		splane.xy = splane.xy * 0.5 + 0.5;
		splane.z = shadow_len * omni_lights.data[idx].inv_radius;
		splane.xy = clamp_rect.xy + splane.xy * clamp_rect.zw;
		//		splane.xy = clamp(splane.xy,clamp_rect.xy + scene_data.shadow_atlas_pixel_size,clamp_rect.xy + clamp_rect.zw - scene_data.shadow_atlas_pixel_size );
		splane.w = 1.0; //needed? i think it should be 1 already

		float shadow_z = textureLod(sampler2D(shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), splane.xy, 0.0).r;
		transmittance_z = (splane.z - shadow_z) / omni_lights.data[idx].inv_radius;
	}
#endif

	if (sc_use_light_projector && omni_lights.data[idx].projector_rect != vec4(0.0)) {
		highp vec3 local_v = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;
		local_v = normalize(local_v);

		highp vec4 atlas_rect = omni_lights.data[idx].projector_rect;

		if (local_v.z >= 0.0) {
			local_v.z += 1.0;
			atlas_rect.y += atlas_rect.w;

		} else {
			local_v.z = 1.0 - local_v.z;
		}

		local_v.xy /= local_v.z;
		local_v.xy = local_v.xy * 0.5 + 0.5;
		highp vec2 proj_uv = local_v.xy * atlas_rect.zw;

		if (sc_projector_use_mipmaps) {
			highp vec2 proj_uv_ddx;
			highp vec2 proj_uv_ddy;
			{
				highp vec3 local_v_ddx = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0)).xyz;
				local_v_ddx = normalize(local_v_ddx);

				if (local_v_ddx.z >= 0.0) {
					local_v_ddx.z += 1.0;
				} else {
					local_v_ddx.z = 1.0 - local_v_ddx.z;
				}

				local_v_ddx.xy /= local_v_ddx.z;
				local_v_ddx.xy = local_v_ddx.xy * 0.5 + 0.5;

				proj_uv_ddx = local_v_ddx.xy * atlas_rect.zw - proj_uv;

				highp vec3 local_v_ddy = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0)).xyz;
				local_v_ddy = normalize(local_v_ddy);

				if (local_v_ddy.z >= 0.0) {
					local_v_ddy.z += 1.0;
				} else {
					local_v_ddy.z = 1.0 - local_v_ddy.z;
				}

				local_v_ddy.xy /= local_v_ddy.z;
				local_v_ddy.xy = local_v_ddy.xy * 0.5 + 0.5;

				proj_uv_ddy = local_v_ddy.xy * atlas_rect.zw - proj_uv;
			}

			highp vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			highp vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}

	light_attenuation *= shadow;

	light_compute(normal, normalize(light_rel_vec), eye_vec, 0, color, light_attenuation, f0, metallic, roughness, omni_lights.data[idx].specular_amount,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_boost,
			transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
			rim * omni_attenuation, rim_tint, rim_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal, tangent, anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
			alpha,
#endif
			diffuse_light,
			specular_light);
}

float light_process_spot_shadow(uint idx, highp vec3 vertex, vec3 normal) {
#ifndef USE_NO_SHADOWS
	if (spot_lights.data[idx].shadow_enabled) {
		highp vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
		highp float light_length = length(light_rel_vec);
		highp vec3 spot_dir = spot_lights.data[idx].direction;
		//there is a shadowmap
		highp vec4 v = vec4(vertex, 1.0);

		float shadow;

		highp vec4 splane = (spot_lights.data[idx].shadow_matrix * v);
		splane /= splane.w;
		splane.z -= spot_lights.data[idx].shadow_bias;

		//hard shadow
		highp vec4 shadow_uv = vec4(splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy, splane.z, 1.0);

		shadow = sample_pcf_shadow(shadow_atlas, spot_lights.data[idx].soft_shadow_scale * scene_data.shadow_atlas_pixel_size, shadow_uv);

		return shadow;
	}

#endif //USE_NO_SHADOWS

	return 1.0;
}

vec2 normal_to_panorama(highp vec3 n) {
	n = normalize(n);
	highp vec2 panorama_coords = vec2(atan(n.x, n.z), acos(-n.y));

	if (panorama_coords.x < 0.0) {
		panorama_coords.x += M_PI * 2.0;
	}

	panorama_coords /= vec2(M_PI * 2.0, M_PI);
	return panorama_coords;
}

void light_process_spot(uint idx, highp vec3 vertex, vec3 eye_vec, vec3 normal, highp vec3 vertex_ddx, highp vec3 vertex_ddy, vec3 f0, float metallic, float roughness, float shadow,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint, vec3 rim_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 binormal, vec3 tangent, float anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
		inout float alpha,
#endif
		inout vec3 diffuse_light,
		inout vec3 specular_light) {
	highp vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float spot_attenuation = get_omni_attenuation(light_length, spot_lights.data[idx].inv_radius, spot_lights.data[idx].attenuation);
	vec3 spot_dir = spot_lights.data[idx].direction;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir), spot_lights.data[idx].cone_angle);
	float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - spot_lights.data[idx].cone_angle));
	spot_attenuation *= 1.0 - pow(spot_rim, spot_lights.data[idx].cone_attenuation);
	float light_attenuation = spot_attenuation;
	vec3 color = spot_lights.data[idx].color;
	float specular_amount = spot_lights.data[idx].specular_amount;

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth;
	transmittance_color.a *= light_attenuation;
	{
		highp vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * spot_lights.data[idx].transmittance_bias, 1.0));
		splane /= splane.w;
		splane.xy = splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy;

		highp float shadow_z = textureLod(sampler2D(shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), splane.xy, 0.0).r;

		shadow_z = shadow_z * 2.0 - 1.0;
		highp float z_far = 1.0 / spot_lights.data[idx].inv_radius;
		highp float z_near = 0.01;
		shadow_z = 2.0 * z_near * z_far / (z_far + z_near - shadow_z * (z_far - z_near));

		//distance to light plane
		highp float z = dot(spot_dir, -light_rel_vec);
		transmittance_z = z - shadow_z;
	}
#endif //LIGHT_TRANSMITTANCE_USED

	if (sc_use_light_projector && spot_lights.data[idx].projector_rect != vec4(0.0)) {
		highp vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex, 1.0));
		splane /= splane.w;

		vec2 proj_uv = normal_to_panorama(splane.xyz) * spot_lights.data[idx].projector_rect.zw;

		if (sc_projector_use_mipmaps) {
			//ensure we have proper mipmaps
			highp vec4 splane_ddx = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0));
			splane_ddx /= splane_ddx.w;
			highp vec2 proj_uv_ddx = normal_to_panorama(splane_ddx.xyz) * spot_lights.data[idx].projector_rect.zw - proj_uv;

			highp vec4 splane_ddy = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0));
			splane_ddy /= splane_ddy.w;
			highp vec2 proj_uv_ddy = normal_to_panorama(splane_ddy.xyz) * spot_lights.data[idx].projector_rect.zw - proj_uv;

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}
	light_attenuation *= shadow;

	light_compute(normal, normalize(light_rel_vec), eye_vec, 0, color, light_attenuation, f0, metallic, roughness, spot_lights.data[idx].specular_amount,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_boost,
			transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
			rim * spot_attenuation, rim_tint, rim_color,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal, tangent, anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
			alpha,
#endif
			diffuse_light, specular_light);
}

void reflection_process(uint ref_index, highp vec3 vertex, vec3 normal, float roughness, vec3 ambient_light, vec3 specular_light, inout vec4 ambient_accum, inout vec4 reflection_accum) {
	vec3 box_extents = reflections.data[ref_index].box_extents;
	vec3 local_pos = (reflections.data[ref_index].local_matrix * vec4(vertex, 1.0)).xyz;

	if (any(greaterThan(abs(local_pos), box_extents))) { //out of the reflection box
		return;
	}

	vec3 ref_vec = normalize(reflect(vertex, normal));

	vec3 inner_pos = abs(local_pos / box_extents);
	float blend = max(inner_pos.x, max(inner_pos.y, inner_pos.z));
	//make blend more rounded
	blend = mix(length(inner_pos), blend, blend);
	blend *= blend;
	blend = max(0.0, 1.0 - blend);

	if (reflections.data[ref_index].intensity > 0.0) { // compute reflection

		vec3 local_ref_vec = (reflections.data[ref_index].local_matrix * vec4(ref_vec, 0.0)).xyz;

		if (reflections.data[ref_index].box_project) { //box project

			vec3 nrdir = normalize(local_ref_vec);
			vec3 rbmax = (box_extents - local_pos) / nrdir;
			vec3 rbmin = (-box_extents - local_pos) / nrdir;

			vec3 rbminmax = mix(rbmin, rbmax, greaterThan(nrdir, vec3(0.0, 0.0, 0.0)));

			float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			vec3 posonbox = local_pos + nrdir * fa;
			local_ref_vec = posonbox - reflections.data[ref_index].box_offset;
		}

		vec4 reflection;

		reflection.rgb = textureLod(samplerCubeArray(reflection_atlas, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(local_ref_vec, reflections.data[ref_index].index), roughness * MAX_ROUGHNESS_LOD).rgb;

		if (reflections.data[ref_index].exterior) {
			reflection.rgb = mix(specular_light, reflection.rgb, blend);
		}

		reflection.rgb *= reflections.data[ref_index].intensity; //intensity
		reflection.a = blend;
		reflection.rgb *= reflection.a;

		reflection_accum += reflection;
	}

	switch (reflections.data[ref_index].ambient_mode) {
		case REFLECTION_AMBIENT_DISABLED: {
			//do nothing
		} break;
		case REFLECTION_AMBIENT_ENVIRONMENT: {
			//do nothing
			vec3 local_amb_vec = (reflections.data[ref_index].local_matrix * vec4(normal, 0.0)).xyz;

			vec4 ambient_out;

			ambient_out.rgb = textureLod(samplerCubeArray(reflection_atlas, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(local_amb_vec, reflections.data[ref_index].index), MAX_ROUGHNESS_LOD).rgb;
			ambient_out.a = blend;
			if (reflections.data[ref_index].exterior) {
				ambient_out.rgb = mix(ambient_light, ambient_out.rgb, blend);
			}

			ambient_out.rgb *= ambient_out.a;
			ambient_accum += ambient_out;
		} break;
		case REFLECTION_AMBIENT_COLOR: {
			vec4 ambient_out;
			ambient_out.a = blend;
			ambient_out.rgb = reflections.data[ref_index].ambient;
			if (reflections.data[ref_index].exterior) {
				ambient_out.rgb = mix(ambient_light, ambient_out.rgb, blend);
			}
			ambient_out.rgb *= ambient_out.a;
			ambient_accum += ambient_out;
		} break;
	}
}

float blur_shadow(float shadow) {
	return shadow;
#if 0
	//disabling for now, will investigate later
	float interp_shadow = shadow;
	if (gl_HelperInvocation) {
		interp_shadow = -4.0; // technically anything below -4 will do but just to make sure
	}

	uvec2 fc2 = uvec2(gl_FragCoord.xy);
	interp_shadow -= dFdx(interp_shadow) * (float(fc2.x & 1) - 0.5);
	interp_shadow -= dFdy(interp_shadow) * (float(fc2.y & 1) - 0.5);

	if (interp_shadow >= 0.0) {
		shadow = interp_shadow;
	}
	return shadow;
#endif
}
