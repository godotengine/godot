// Functions related to lighting

float D_GGX(float cos_theta_m, float alpha) {
	float a = cos_theta_m * alpha;
	float k = alpha / (1.0 - cos_theta_m * cos_theta_m + a * a);
	return k * k * (1.0 / M_PI);
}

// From Earl Hammon, Jr. "PBR Diffuse Lighting for GGX+Smith Microsurfaces" https://www.gdcvault.com/play/1024478/PBR-Diffuse-Lighting-for-GGX
float V_GGX(float NdotL, float NdotV, float alpha) {
	return 0.5 / mix(2.0 * NdotL * NdotV, NdotL + NdotV, alpha);
}

float D_GGX_anisotropic(float cos_theta_m, float alpha_x, float alpha_y, float cos_phi, float sin_phi) {
	float alpha2 = alpha_x * alpha_y;
	highp vec3 v = vec3(alpha_y * cos_phi, alpha_x * sin_phi, alpha2 * cos_theta_m);
	highp float v2 = dot(v, v);
	float w2 = alpha2 / v2;
	float D = alpha2 * w2 * w2 * (1.0 / M_PI);
	return D;
}

float V_GGX_anisotropic(float alpha_x, float alpha_y, float TdotV, float TdotL, float BdotV, float BdotL, float NdotV, float NdotL) {
	float Lambda_V = NdotL * length(vec3(alpha_x * TdotV, alpha_y * BdotV, NdotV));
	float Lambda_L = NdotV * length(vec3(alpha_x * TdotL, alpha_y * BdotL, NdotL));
	return 0.5 / (Lambda_V + Lambda_L);
}

float SchlickFresnel(float u) {
	float m = 1.0 - u;
	float m2 = m * m;
	return m2 * m2 * m; // pow(m,5)
}

vec3 F0(float metallic, float specular, vec3 albedo) {
	float dielectric = 0.16 * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(vec3(dielectric), albedo, vec3(metallic));
}

void light_compute(vec3 N, vec3 L, vec3 V, float A, vec3 light_color, bool is_directional, float attenuation, vec3 f0, uint orms, float specular_amount, vec3 albedo, inout float alpha,
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
		float rim, float rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 B, vec3 T, float anisotropy,
#endif
		inout vec3 diffuse_light, inout vec3 specular_light) {

	vec4 orms_unpacked = unpackUnorm4x8(orms);

	float roughness = orms_unpacked.y;
	float metallic = orms_unpacked.z;

#if defined(LIGHT_CODE_USED)
	// light is written by the light shader

	mat4 inv_view_matrix = scene_data_block.data.inv_view_matrix;

#ifdef USING_MOBILE_RENDERER
	mat4 read_model_matrix = instances.data[draw_call.instance_index].transform;
#else
	mat4 read_model_matrix = instances.data[instance_index_interp].transform;
#endif

	mat4 read_view_matrix = scene_data_block.data.view_matrix;

#undef projection_matrix
#define projection_matrix scene_data_block.data.projection_matrix
#undef inv_projection_matrix
#define inv_projection_matrix scene_data_block.data.inv_projection_matrix

	vec2 read_viewport_size = scene_data_block.data.viewport_size;

	vec3 normal = N;
	vec3 light = L;
	vec3 view = V;

#CODE : LIGHT

#else

	float NdotL = min(A + dot(N, L), 1.0);
	float cNdotL = max(NdotL, 0.0); // clamped NdotL
	float NdotV = dot(N, V);
	float cNdotV = max(NdotV, 1e-4);

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	vec3 H = normalize(V + L);
#endif

#if defined(SPECULAR_SCHLICK_GGX)
	float cNdotH = clamp(A + dot(N, H), 0.0, 1.0);
#endif

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	float cLdotH = clamp(A + dot(L, H), 0.0, 1.0);
#endif

	if (metallic < 1.0) {
		float diffuse_brdf_NL; // BRDF times N.L for calculating diffuse radiance

#if defined(DIFFUSE_LAMBERT_WRAP)
		// Energy conserving lambert wrap shader.
		// https://web.archive.org/web/20210228210901/http://blog.stevemcauley.com/2011/12/03/energy-conserving-wrapped-diffuse/
		diffuse_brdf_NL = max(0.0, (NdotL + roughness) / ((1.0 + roughness) * (1.0 + roughness))) * (1.0 / M_PI);
#elif defined(DIFFUSE_TOON)

		diffuse_brdf_NL = smoothstep(-roughness, max(roughness, 0.01), NdotL) * (1.0 / M_PI);

#elif defined(DIFFUSE_BURLEY)

		{
			float FD90_minus_1 = 2.0 * cLdotH * cLdotH * roughness - 0.5;
			float FdV = 1.0 + FD90_minus_1 * SchlickFresnel(cNdotV);
			float FdL = 1.0 + FD90_minus_1 * SchlickFresnel(cNdotL);
			diffuse_brdf_NL = (1.0 / M_PI) * FdV * FdL * cNdotL;
			/*
			float energyBias = mix(roughness, 0.0, 0.5);
			float energyFactor = mix(roughness, 1.0, 1.0 / 1.51);
			float fd90 = energyBias + 2.0 * VoH * VoH * roughness;
			float f0 = 1.0;
			float lightScatter = f0 + (fd90 - f0) * pow(1.0 - cNdotL, 5.0);
			float viewScatter = f0 + (fd90 - f0) * pow(1.0 - cNdotV, 5.0);

			diffuse_brdf_NL = lightScatter * viewScatter * energyFactor;
			*/
		}
#else
		// lambert
		diffuse_brdf_NL = cNdotL * (1.0 / M_PI);
#endif

		diffuse_light += light_color * diffuse_brdf_NL * attenuation;

#if defined(LIGHT_BACKLIGHT_USED)
		diffuse_light += light_color * (vec3(1.0 / M_PI) - diffuse_brdf_NL) * backlight * attenuation;
#endif

#if defined(LIGHT_RIM_USED)
		// Epsilon min to prevent pow(0, 0) singularity which results in undefined behavior.
		float rim_light = pow(max(1e-4, 1.0 - cNdotV), max(0.0, (1.0 - roughness) * 16.0));
		diffuse_light += rim_light * rim * mix(vec3(1.0), albedo, rim_tint) * light_color;
#endif

#ifdef LIGHT_TRANSMITTANCE_USED

		{
#ifdef SSS_MODE_SKIN
			float scale = 8.25 / transmittance_depth;
			float d = scale * abs(transmittance_z);
			float dd = -d * d;
			vec3 profile = vec3(0.233, 0.455, 0.649) * exp(dd / 0.0064) +
					vec3(0.1, 0.336, 0.344) * exp(dd / 0.0484) +
					vec3(0.118, 0.198, 0.0) * exp(dd / 0.187) +
					vec3(0.113, 0.007, 0.007) * exp(dd / 0.567) +
					vec3(0.358, 0.004, 0.0) * exp(dd / 1.99) +
					vec3(0.078, 0.0, 0.0) * exp(dd / 7.41);

			diffuse_light += profile * transmittance_color.a * light_color * clamp(transmittance_boost - NdotL, 0.0, 1.0) * (1.0 / M_PI);
#else

			float scale = 8.25 / transmittance_depth;
			float d = scale * abs(transmittance_z);
			float dd = -d * d;
			diffuse_light += exp(dd) * transmittance_color.rgb * transmittance_color.a * light_color * clamp(transmittance_boost - NdotL, 0.0, 1.0) * (1.0 / M_PI);
#endif
		}
#else

#endif //LIGHT_TRANSMITTANCE_USED
	}

	if (roughness > 0.0) { // FIXME: roughness == 0 should not disable specular light entirely

		// D

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
#if defined(LIGHT_ANISOTROPY_USED)

		float aspect = sqrt(1.0 - anisotropy * 0.9);
		float ax = alpha_ggx / aspect;
		float ay = alpha_ggx * aspect;
		float XdotH = dot(T, H);
		float YdotH = dot(B, H);
		float D = D_GGX_anisotropic(cNdotH, ax, ay, XdotH, YdotH);
		float G = V_GGX_anisotropic(ax, ay, dot(T, V), dot(T, L), dot(B, V), dot(B, L), cNdotV, cNdotL);
#else // LIGHT_ANISOTROPY_USED
		float D = D_GGX(cNdotH, alpha_ggx);
		float G = V_GGX(cNdotL, cNdotV, alpha_ggx);
#endif // LIGHT_ANISOTROPY_USED
	   // F
		float cLdotH5 = SchlickFresnel(cLdotH);
		// Calculate Fresnel using specular occlusion term from Filament:
		// https://google.github.io/filament/Filament.html#lighting/occlusion/specularocclusion
		float f90 = clamp(dot(f0, vec3(50.0 * 0.33)), metallic, 1.0);
		vec3 F = f0 + (f90 - f0) * cLdotH5;

		vec3 specular_brdf_NL = cNdotL * D * F * G;

		specular_light += specular_brdf_NL * light_color * attenuation * specular_amount;
#endif

#if defined(LIGHT_CLEARCOAT_USED)
		// Clearcoat ignores normal_map, use vertex normal instead
		float ccNdotL = max(min(A + dot(vertex_normal, L), 1.0), 0.0);
		float ccNdotH = clamp(A + dot(vertex_normal, H), 0.0, 1.0);
		float ccNdotV = max(dot(vertex_normal, V), 1e-4);

#if !defined(SPECULAR_SCHLICK_GGX)
		float cLdotH5 = SchlickFresnel(cLdotH);
#endif
		float Dr = D_GGX(ccNdotH, mix(0.001, 0.1, clearcoat_roughness));
		float Gr = 0.25 / (cLdotH * cLdotH);
		float Fr = mix(.04, 1.0, cLdotH5);
		float clearcoat_specular_brdf_NL = clearcoat * Gr * Fr * Dr * cNdotL;

		specular_light += clearcoat_specular_brdf_NL * light_color * attenuation * specular_amount;
		// TODO: Clearcoat adds light to the scene right now (it is non-energy conserving), both diffuse and specular need to be scaled by (1.0 - FR)
		// but to do so we need to rearrange this entire function
#endif // LIGHT_CLEARCOAT_USED
	}

#ifdef USE_SHADOW_TO_OPACITY
	alpha = min(alpha, clamp(1.0 - attenuation, 0.0, 1.0));
#endif

#endif //defined(LIGHT_CODE_USED)
}

#ifndef SHADOWS_DISABLED

// Interleaved Gradient Noise
// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
float quick_hash(vec2 pos) {
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

float sample_directional_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec4 coord) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (sc_directional_soft_shadow_samples == 0) {
		return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;

	for (uint i = 0; i < sc_directional_soft_shadow_samples; i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data_block.data.directional_soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return avg * (1.0 / float(sc_directional_soft_shadow_samples));
}

float sample_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec3 coord) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (sc_soft_shadow_samples == 0) {
		return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;

	for (uint i = 0; i < sc_soft_shadow_samples; i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data_block.data.soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return avg * (1.0 / float(sc_soft_shadow_samples));
}

float sample_omni_pcf_shadow(texture2D shadow, float blur_scale, vec2 coord, vec4 uv_rect, vec2 flip_offset, float depth) {
	//if only one sample is taken, take it from the center
	if (sc_soft_shadow_samples == 0) {
		vec2 pos = coord * 0.5 + 0.5;
		pos = uv_rect.xy + pos * uv_rect.zw;
		return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;
	vec2 offset_scale = blur_scale * 2.0 * scene_data_block.data.shadow_atlas_pixel_size / uv_rect.zw;

	for (uint i = 0; i < sc_soft_shadow_samples; i++) {
		vec2 offset = offset_scale * (disk_rotation * scene_data_block.data.soft_shadow_kernel[i].xy);
		vec2 sample_coord = coord + offset;

		float sample_coord_length_sqaured = dot(sample_coord, sample_coord);
		bool do_flip = sample_coord_length_sqaured > 1.0;

		if (do_flip) {
			float len = sqrt(sample_coord_length_sqaured);
			sample_coord = sample_coord * (2.0 / len - 1.0);
		}

		sample_coord = sample_coord * 0.5 + 0.5;
		sample_coord = uv_rect.xy + sample_coord * uv_rect.zw;

		if (do_flip) {
			sample_coord += flip_offset;
		}
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(sample_coord, depth, 1.0));
	}

	return avg * (1.0 / float(sc_soft_shadow_samples));
}

float sample_directional_soft_shadow(texture2D shadow, vec3 pssm_coord, vec2 tex_scale) {
	//find blocker
	float blocker_count = 0.0;
	float blocker_average = 0.0;

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	for (uint i = 0; i < sc_directional_penumbra_shadow_samples; i++) {
		vec2 suv = pssm_coord.xy + (disk_rotation * scene_data_block.data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
		float d = textureLod(sampler2D(shadow, SAMPLER_LINEAR_CLAMP), suv, 0.0).r;
		if (d < pssm_coord.z) {
			blocker_average += d;
			blocker_count += 1.0;
		}
	}

	if (blocker_count > 0.0) {
		//blockers found, do soft shadow
		blocker_average /= blocker_count;
		float penumbra = (pssm_coord.z - blocker_average) / blocker_average;
		tex_scale *= penumbra;

		float s = 0.0;
		for (uint i = 0; i < sc_directional_penumbra_shadow_samples; i++) {
			vec2 suv = pssm_coord.xy + (disk_rotation * scene_data_block.data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
			s += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(suv, pssm_coord.z, 1.0));
		}

		return s / float(sc_directional_penumbra_shadow_samples);

	} else {
		//no blockers found, so no shadow
		return 1.0;
	}
}

#endif // SHADOWS_DISABLED

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

float light_process_omni_shadow(uint idx, vec3 vertex, vec3 normal) {
#ifndef SHADOWS_DISABLED
	if (omni_lights.data[idx].shadow_opacity > 0.001) {
		// there is a shadowmap
		vec2 texel_size = scene_data_block.data.shadow_atlas_pixel_size;
		vec4 base_uv_rect = omni_lights.data[idx].atlas_rect;
		base_uv_rect.xy += texel_size;
		base_uv_rect.zw -= texel_size * 2.0;

		// Omni lights use direction.xy to store to store the offset between the two paraboloid regions
		vec2 flip_offset = omni_lights.data[idx].direction.xy;

		vec3 local_vert = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;

		float shadow_len = length(local_vert); //need to remember shadow len from here
		vec3 shadow_dir = normalize(local_vert);

		vec3 local_normal = normalize(mat3(omni_lights.data[idx].shadow_matrix) * normal);
		vec3 normal_bias = local_normal * omni_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(local_normal, shadow_dir)));

		float shadow;

		if (sc_use_light_soft_shadows && omni_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			vec3 basis_normal = shadow_dir;
			vec3 v0 = abs(basis_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
			vec3 tangent = normalize(cross(v0, basis_normal));
			vec3 bitangent = normalize(cross(tangent, basis_normal));
			float z_norm = shadow_len * omni_lights.data[idx].inv_radius;

			tangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;
			bitangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;

			for (uint i = 0; i < sc_penumbra_shadow_samples; i++) {
				vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;

				vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

				pos = normalize(pos);

				vec4 uv_rect = base_uv_rect;

				if (pos.z >= 0.0) {
					uv_rect.xy += flip_offset;
				}

				pos.z = 1.0 + abs(pos.z);
				pos.xy /= pos.z;

				pos.xy = pos.xy * 0.5 + 0.5;
				pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;

				float d = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), pos.xy, 0.0).r;
				if (d < z_norm) {
					blocker_average += d;
					blocker_count += 1.0;
				}
			}

			if (blocker_count > 0.0) {
				//blockers found, do soft shadow
				blocker_average /= blocker_count;
				float penumbra = (z_norm - blocker_average) / blocker_average;
				tangent *= penumbra;
				bitangent *= penumbra;

				z_norm -= omni_lights.data[idx].inv_radius * omni_lights.data[idx].shadow_bias;

				shadow = 0.0;
				for (uint i = 0; i < sc_penumbra_shadow_samples; i++) {
					vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;
					vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

					pos = normalize(pos);
					pos = normalize(pos + normal_bias);

					vec4 uv_rect = base_uv_rect;

					if (pos.z >= 0.0) {
						uv_rect.xy += flip_offset;
					}

					pos.z = 1.0 + abs(pos.z);
					pos.xy /= pos.z;

					pos.xy = pos.xy * 0.5 + 0.5;
					pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;
					shadow += textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(pos.xy, z_norm, 1.0));
				}

				shadow /= float(sc_penumbra_shadow_samples);
				shadow = mix(1.0, shadow, omni_lights.data[idx].shadow_opacity);

			} else {
				//no blockers found, so no shadow
				shadow = 1.0;
			}
		} else {
			vec4 uv_rect = base_uv_rect;

			vec3 shadow_sample = normalize(shadow_dir + normal_bias);
			if (shadow_sample.z >= 0.0) {
				uv_rect.xy += flip_offset;
				flip_offset *= -1.0;
			}

			shadow_sample.z = 1.0 + abs(shadow_sample.z);
			vec2 pos = shadow_sample.xy / shadow_sample.z;
			float depth = shadow_len - omni_lights.data[idx].shadow_bias;
			depth *= omni_lights.data[idx].inv_radius;
			shadow = mix(1.0, sample_omni_pcf_shadow(shadow_atlas, omni_lights.data[idx].soft_shadow_scale / shadow_sample.z, pos, uv_rect, flip_offset, depth), omni_lights.data[idx].shadow_opacity);
		}

		return shadow;
	}
#endif

	return 1.0;
}

void light_process_omni(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float shadow, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 binormal, vec3 tangent, float anisotropy,
#endif
		inout vec3 diffuse_light, inout vec3 specular_light) {
	vec3 light_rel_vec = omni_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float omni_attenuation = get_omni_attenuation(light_length, omni_lights.data[idx].inv_radius, omni_lights.data[idx].attenuation);
	float light_attenuation = omni_attenuation;
	vec3 color = omni_lights.data[idx].color;

	float size_A = 0.0;

	if (sc_use_light_soft_shadows && omni_lights.data[idx].size > 0.0) {
		float t = omni_lights.data[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth; //no transmittance by default
	transmittance_color.a *= light_attenuation;
	{
		vec4 clamp_rect = omni_lights.data[idx].atlas_rect;

		//redo shadowmapping, but shrink the model a bit to avoid artifacts
		vec4 splane = (omni_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * omni_lights.data[idx].transmittance_bias, 1.0));

		float shadow_len = length(splane.xyz);
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
		//		splane.xy = clamp(splane.xy,clamp_rect.xy + scene_data_block.data.shadow_atlas_pixel_size,clamp_rect.xy + clamp_rect.zw - scene_data_block.data.shadow_atlas_pixel_size );
		splane.w = 1.0; //needed? i think it should be 1 already

		float shadow_z = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), splane.xy, 0.0).r;
		transmittance_z = (splane.z - shadow_z) / omni_lights.data[idx].inv_radius;
	}
#endif

	if (sc_use_light_projector && omni_lights.data[idx].projector_rect != vec4(0.0)) {
		vec3 local_v = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;
		local_v = normalize(local_v);

		vec4 atlas_rect = omni_lights.data[idx].projector_rect;

		if (local_v.z >= 0.0) {
			atlas_rect.y += atlas_rect.w;
		}

		local_v.z = 1.0 + abs(local_v.z);

		local_v.xy /= local_v.z;
		local_v.xy = local_v.xy * 0.5 + 0.5;
		vec2 proj_uv = local_v.xy * atlas_rect.zw;

		if (sc_projector_use_mipmaps) {
			vec2 proj_uv_ddx;
			vec2 proj_uv_ddy;
			{
				vec3 local_v_ddx = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0)).xyz;
				local_v_ddx = normalize(local_v_ddx);

				if (local_v_ddx.z >= 0.0) {
					local_v_ddx.z += 1.0;
				} else {
					local_v_ddx.z = 1.0 - local_v_ddx.z;
				}

				local_v_ddx.xy /= local_v_ddx.z;
				local_v_ddx.xy = local_v_ddx.xy * 0.5 + 0.5;

				proj_uv_ddx = local_v_ddx.xy * atlas_rect.zw - proj_uv;

				vec3 local_v_ddy = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0)).xyz;
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

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}

	light_attenuation *= shadow;

	light_compute(normal, normalize(light_rel_vec), eye_vec, size_A, color, false, light_attenuation, f0, orms, omni_lights.data[idx].specular_amount, albedo, alpha,
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
			rim * omni_attenuation, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal, tangent, anisotropy,
#endif
			diffuse_light,
			specular_light);
}

float light_process_spot_shadow(uint idx, vec3 vertex, vec3 normal) {
#ifndef SHADOWS_DISABLED
	if (spot_lights.data[idx].shadow_opacity > 0.001) {
		vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
		float light_length = length(light_rel_vec);
		vec3 spot_dir = spot_lights.data[idx].direction;

		vec3 shadow_dir = light_rel_vec / light_length;
		vec3 normal_bias = normal * light_length * spot_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(normal, shadow_dir)));

		//there is a shadowmap
		vec4 v = vec4(vertex + normal_bias, 1.0);

		vec4 splane = (spot_lights.data[idx].shadow_matrix * v);
		splane.z -= spot_lights.data[idx].shadow_bias / (light_length * spot_lights.data[idx].inv_radius);
		splane /= splane.w;

		float shadow;
		if (sc_use_light_soft_shadows && spot_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker
			float z_norm = dot(spot_dir, -light_rel_vec) * spot_lights.data[idx].inv_radius;

			vec2 shadow_uv = splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy;

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			float uv_size = spot_lights.data[idx].soft_shadow_size * z_norm * spot_lights.data[idx].soft_shadow_scale;
			vec2 clamp_max = spot_lights.data[idx].atlas_rect.xy + spot_lights.data[idx].atlas_rect.zw;
			for (uint i = 0; i < sc_penumbra_shadow_samples; i++) {
				vec2 suv = shadow_uv + (disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy) * uv_size;
				suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
				float d = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), suv, 0.0).r;
				if (d < splane.z) {
					blocker_average += d;
					blocker_count += 1.0;
				}
			}

			if (blocker_count > 0.0) {
				//blockers found, do soft shadow
				blocker_average /= blocker_count;
				float penumbra = (z_norm - blocker_average) / blocker_average;
				uv_size *= penumbra;

				shadow = 0.0;
				for (uint i = 0; i < sc_penumbra_shadow_samples; i++) {
					vec2 suv = shadow_uv + (disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy) * uv_size;
					suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
					shadow += textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(suv, splane.z, 1.0));
				}

				shadow /= float(sc_penumbra_shadow_samples);
				shadow = mix(1.0, shadow, spot_lights.data[idx].shadow_opacity);

			} else {
				//no blockers found, so no shadow
				shadow = 1.0;
			}
		} else {
			//hard shadow
			vec3 shadow_uv = vec3(splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy, splane.z);
			shadow = mix(1.0, sample_pcf_shadow(shadow_atlas, spot_lights.data[idx].soft_shadow_scale * scene_data_block.data.shadow_atlas_pixel_size, shadow_uv), spot_lights.data[idx].shadow_opacity);
		}

		return shadow;
	}

#endif // SHADOWS_DISABLED

	return 1.0;
}

vec2 normal_to_panorama(vec3 n) {
	n = normalize(n);
	vec2 panorama_coords = vec2(atan(n.x, n.z), acos(-n.y));

	if (panorama_coords.x < 0.0) {
		panorama_coords.x += M_PI * 2.0;
	}

	panorama_coords /= vec2(M_PI * 2.0, M_PI);
	return panorama_coords;
}

void light_process_spot(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float shadow, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 binormal, vec3 tangent, float anisotropy,
#endif
		inout vec3 diffuse_light,
		inout vec3 specular_light) {
	vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float spot_attenuation = get_omni_attenuation(light_length, spot_lights.data[idx].inv_radius, spot_lights.data[idx].attenuation);
	vec3 spot_dir = spot_lights.data[idx].direction;

	// This conversion to a highp float is crucial to prevent light leaking
	// due to precision errors in the following calculations (cone angle is mediump).
	highp float cone_angle = spot_lights.data[idx].cone_angle;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir), cone_angle);
	float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - cone_angle));

	spot_attenuation *= 1.0 - pow(spot_rim, spot_lights.data[idx].cone_attenuation);
	float light_attenuation = spot_attenuation;
	vec3 color = spot_lights.data[idx].color;
	float specular_amount = spot_lights.data[idx].specular_amount;

	float size_A = 0.0;

	if (sc_use_light_soft_shadows && spot_lights.data[idx].size > 0.0) {
		float t = spot_lights.data[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth;
	transmittance_color.a *= light_attenuation;
	{
		vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * spot_lights.data[idx].transmittance_bias, 1.0));
		splane /= splane.w;
		splane.xy = splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy;

		float shadow_z = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), splane.xy, 0.0).r;

		shadow_z = shadow_z * 2.0 - 1.0;
		float z_far = 1.0 / spot_lights.data[idx].inv_radius;
		float z_near = 0.01;
		shadow_z = 2.0 * z_near * z_far / (z_far + z_near - shadow_z * (z_far - z_near));

		//distance to light plane
		float z = dot(spot_dir, -light_rel_vec);
		transmittance_z = z - shadow_z;
	}
#endif //LIGHT_TRANSMITTANCE_USED

	if (sc_use_light_projector && spot_lights.data[idx].projector_rect != vec4(0.0)) {
		vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex, 1.0));
		splane /= splane.w;

		vec2 proj_uv = splane.xy * spot_lights.data[idx].projector_rect.zw;

		if (sc_projector_use_mipmaps) {
			//ensure we have proper mipmaps
			vec4 splane_ddx = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0));
			splane_ddx /= splane_ddx.w;
			vec2 proj_uv_ddx = splane_ddx.xy * spot_lights.data[idx].projector_rect.zw - proj_uv;

			vec4 splane_ddy = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0));
			splane_ddy /= splane_ddy.w;
			vec2 proj_uv_ddy = splane_ddy.xy * spot_lights.data[idx].projector_rect.zw - proj_uv;

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}
	light_attenuation *= shadow;

	light_compute(normal, normalize(light_rel_vec), eye_vec, size_A, color, false, light_attenuation, f0, orms, spot_lights.data[idx].specular_amount, albedo, alpha,
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
			rim * spot_attenuation, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal, tangent, anisotropy,
#endif
			diffuse_light, specular_light);
}

void reflection_process(uint ref_index, vec3 vertex, vec3 ref_vec, vec3 normal, float roughness, vec3 ambient_light, vec3 specular_light, inout vec4 ambient_accum, inout vec4 reflection_accum) {
	vec3 box_extents = reflections.data[ref_index].box_extents;
	vec3 local_pos = (reflections.data[ref_index].local_matrix * vec4(vertex, 1.0)).xyz;

	if (any(greaterThan(abs(local_pos), box_extents))) { //out of the reflection box
		return;
	}

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

		reflection.rgb = textureLod(samplerCubeArray(reflection_atlas, SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec4(local_ref_vec, reflections.data[ref_index].index), roughness * MAX_ROUGHNESS_LOD).rgb * sc_luminance_multiplier;
		reflection.rgb *= reflections.data[ref_index].exposure_normalization;
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

			ambient_out.rgb = textureLod(samplerCubeArray(reflection_atlas, SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec4(local_amb_vec, reflections.data[ref_index].index), MAX_ROUGHNESS_LOD).rgb;
			ambient_out.rgb *= reflections.data[ref_index].exposure_normalization;
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
