// Functions related to lighting

// This returns the G_GGX function divided by 2 cos_theta_m, where in practice cos_theta_m is either N.L or N.V.
// We're dividing this factor off because the overall term we'll end up looks like
// (see, for example, the first unnumbered equation in B. Burley, "Physically Based Shading at Disney", SIGGRAPH 2012):
//
//   F(L.V) D(N.H) G(N.L) G(N.V) / (4 N.L N.V)
//
// We're basically regouping this as
//
//   F(L.V) D(N.H) [G(N.L)/(2 N.L)] [G(N.V) / (2 N.V)]
//
// and thus, this function implements the [G(N.m)/(2 N.m)] part with m = L or V.
//
// The contents of the D and G (G1) functions (GGX) are taken from
// E. Heitz, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs", J. Comp. Graph. Tech. 3 (2) (2014).
// Eqns 71-72 and 85-86 (see also Eqns 43 and 80).

float G_GGX_2cos(float cos_theta_m, float alpha) {
	// Schlick's approximation
	// C. Schlick, "An Inexpensive BRDF Model for Physically-based Rendering", Computer Graphics Forum. 13 (3): 233 (1994)
	// Eq. (19), although see Heitz (2014) the about the problems with his derivation.
	// It nevertheless approximates GGX well with k = alpha/2.
	float k = 0.5 * alpha;
	return 0.5 / (cos_theta_m * (1.0 - k) + k);

	// float cos2 = cos_theta_m * cos_theta_m;
	// float sin2 = (1.0 - cos2);
	// return 1.0 / (cos_theta_m + sqrt(cos2 + alpha * alpha * sin2));
}

float D_GGX(float cos_theta_m, float alpha) {
	float alpha2 = alpha * alpha;
	float d = 1.0 + (alpha2 - 1.0) * cos_theta_m * cos_theta_m;
	return alpha2 / (M_PI * d * d);
}

float G_GGX_anisotropic_2cos(float cos_theta_m, float alpha_x, float alpha_y, float cos_phi, float sin_phi) {
	float cos2 = cos_theta_m * cos_theta_m;
	float sin2 = (1.0 - cos2);
	float s_x = alpha_x * cos_phi;
	float s_y = alpha_y * sin_phi;
	return 1.0 / max(cos_theta_m + sqrt(cos2 + (s_x * s_x + s_y * s_y) * sin2), 0.001);
}

float D_GGX_anisotropic(float cos_theta_m, float alpha_x, float alpha_y, float cos_phi, float sin_phi) {
	float cos2 = cos_theta_m * cos_theta_m;
	float sin2 = (1.0 - cos2);
	float r_x = cos_phi / alpha_x;
	float r_y = sin_phi / alpha_y;
	float d = cos2 + sin2 * (r_x * r_x + r_y * r_y);
	return 1.0 / max(M_PI * alpha_x * alpha_y * d * d, 0.001);
}

float SchlickFresnel(float u) {
	float m = 1.0 - u;
	float m2 = m * m;
	return m2 * m2 * m; // pow(m,5)
}

float GTR1(float NdotH, float a) {
	if (a >= 1.0)
		return 1.0 / M_PI;
	float a2 = a * a;
	float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return (a2 - 1.0) / (M_PI * log(a2) * t);
}

vec3 F0(float metallic, float specular, vec3 albedo) {
	float dielectric = 0.16 * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(vec3(dielectric), albedo, vec3(metallic));
}

void light_compute(vec3 N, vec3 L, vec3 V, vec3 light_color, float attenuation, vec3 f0, uint orms, float specular_amount,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_curve,
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
#ifdef USE_SOFT_SHADOWS
		float A,
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

#ifdef USE_SOFT_SHADOWS
	float NdotL = min(A + dot(N, L), 1.0);
#else
	float NdotL = dot(N, L);
#endif
	float cNdotL = max(NdotL, 0.0); // clamped NdotL
	float NdotV = dot(N, V);
	float cNdotV = max(NdotV, 0.0);

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_BLINN) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	vec3 H = normalize(V + L);
#endif

#if defined(SPECULAR_BLINN) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
#ifdef USE_SOFT_SHADOWS
	float cNdotH = clamp(A + dot(N, H), 0.0, 1.0);
#else
	float cNdotH = clamp(dot(N, H), 0.0, 1.0);
#endif
#endif

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
#ifdef USE_SOFT_SHADOWS
	float cLdotH = clamp(A + dot(L, H), 0.0, 1.0);
#else
	float cLdotH = clamp(dot(L, H), 0.0, 1.0);
#endif
#endif

	float metallic = unpackUnorm4x8(orms).z;
	if (metallic < 1.0) {
		float roughness = unpackUnorm4x8(orms).y;

#if defined(DIFFUSE_OREN_NAYAR)
		vec3 diffuse_brdf_NL;
#else
		float diffuse_brdf_NL; // BRDF times N.L for calculating diffuse radiance
#endif

#if defined(DIFFUSE_LAMBERT_WRAP)
		// energy conserving lambert wrap shader
		diffuse_brdf_NL = max(0.0, (NdotL + roughness) / ((1.0 + roughness) * (1.0 + roughness)));
#elif defined(DIFFUSE_TOON)

		diffuse_brdf_NL = smoothstep(-roughness, max(roughness, 0.01), NdotL);

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
		float rim_light = pow(max(0.0, 1.0 - cNdotV), max(0.0, (1.0 - roughness) * 16.0));
		diffuse_light += rim_light * rim * mix(vec3(1.0), rim_color, rim_tint) * light_color;
#endif

#ifdef LIGHT_TRANSMITTANCE_USED

#ifdef SSS_MODE_SKIN

		{
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
		}
#else

		if (transmittance_depth > 0.0) {
			float fade = clamp(abs(transmittance_z / transmittance_depth), 0.0, 1.0);

			fade = pow(max(0.0, 1.0 - fade), transmittance_curve);
			fade *= clamp(transmittance_boost - NdotL, 0.0, 1.0);

			diffuse_light += transmittance_color.rgb * light_color * (1.0 / M_PI) * transmittance_color.a * fade;
		}

#endif //SSS_MODE_SKIN

#endif //LIGHT_TRANSMITTANCE_USED
	}

	float roughness = unpackUnorm4x8(orms).y;
	if (roughness > 0.0) { // FIXME: roughness == 0 should not disable specular light entirely

		// D

#if defined(SPECULAR_BLINN)

		//normalized blinn
		float shininess = exp2(15.0 * (1.0 - roughness) + 1.0) * 0.25;
		float blinn = pow(cNdotH, shininess) * cNdotL;
		blinn *= (shininess + 8.0) * (1.0 / (8.0 * M_PI));
		float intensity = blinn;

		specular_light += light_color * intensity * attenuation * specular_amount;

#elif defined(SPECULAR_PHONG)

		vec3 R = normalize(-reflect(L, N));
		float cRdotV = clamp(A + dot(R, V), 0.0, 1.0);
		float shininess = exp2(15.0 * (1.0 - roughness) + 1.0) * 0.25;
		float phong = pow(cRdotV, shininess);
		phong *= (shininess + 8.0) * (1.0 / (8.0 * M_PI));
		float intensity = (phong) / max(4.0 * cNdotV * cNdotL, 0.75);

		specular_light += light_color * intensity * attenuation * specular_amount;

#elif defined(SPECULAR_TOON)

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

#if defined(LIGHT_ANISOTROPY_USED)

		float alpha_ggx = roughness * roughness;
		float aspect = sqrt(1.0 - anisotropy * 0.9);
		float ax = alpha_ggx / aspect;
		float ay = alpha_ggx * aspect;
		float XdotH = dot(T, H);
		float YdotH = dot(B, H);
		float D = D_GGX_anisotropic(cNdotH, ax, ay, XdotH, YdotH);
		float G = G_GGX_anisotropic_2cos(cNdotL, ax, ay, XdotH, YdotH) * G_GGX_anisotropic_2cos(cNdotV, ax, ay, XdotH, YdotH);

#else
		float alpha_ggx = roughness * roughness;
		float D = D_GGX(cNdotH, alpha_ggx);
		float G = G_GGX_2cos(cNdotL, alpha_ggx) * G_GGX_2cos(cNdotV, alpha_ggx);
#endif
		// F
		float cLdotH5 = SchlickFresnel(cLdotH);
		vec3 F = mix(vec3(cLdotH5), vec3(1.0), f0);

		vec3 specular_brdf_NL = cNdotL * D * F * G;

		specular_light += specular_brdf_NL * light_color * attenuation * specular_amount;
#endif

#if defined(LIGHT_CLEARCOAT_USED)

#if !defined(SPECULAR_SCHLICK_GGX)
		float cLdotH5 = SchlickFresnel(cLdotH);
#endif
		float Dr = GTR1(cNdotH, mix(.1, .001, clearcoat_gloss));
		float Fr = mix(.04, 1.0, cLdotH5);
		float Gr = G_GGX_2cos(cNdotL, .25) * G_GGX_2cos(cNdotV, .25);

		float clearcoat_specular_brdf_NL = 0.25 * clearcoat * Gr * Fr * Dr * cNdotL;

		specular_light += clearcoat_specular_brdf_NL * light_color * attenuation * specular_amount;
#endif
	}

#ifdef USE_SHADOW_TO_OPACITY
	alpha = min(alpha, clamp(1.0 - attenuation), 0.0, 1.0));
#endif

#endif //defined(LIGHT_CODE_USED)
}

#ifndef USE_NO_SHADOWS

// Interleaved Gradient Noise
// http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
float quick_hash(vec2 pos) {
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

float sample_directional_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec4 coord) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (scene_data.directional_soft_shadow_samples == 1) {
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

	for (uint i = 0; i < scene_data.directional_soft_shadow_samples; i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data.directional_soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return avg * (1.0 / float(scene_data.directional_soft_shadow_samples));
}

float sample_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec4 coord) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (scene_data.soft_shadow_samples == 1) {
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

	for (uint i = 0; i < scene_data.soft_shadow_samples; i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data.soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return avg * (1.0 / float(scene_data.soft_shadow_samples));
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

	for (uint i = 0; i < scene_data.directional_penumbra_shadow_samples; i++) {
		vec2 suv = pssm_coord.xy + (disk_rotation * scene_data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
		float d = textureLod(sampler2D(shadow, material_samplers[SAMPLER_LINEAR_CLAMP]), suv, 0.0).r;
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
		for (uint i = 0; i < scene_data.directional_penumbra_shadow_samples; i++) {
			vec2 suv = pssm_coord.xy + (disk_rotation * scene_data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
			s += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(suv, pssm_coord.z, 1.0));
		}

		return s / float(scene_data.directional_penumbra_shadow_samples);

	} else {
		//no blockers found, so no shadow
		return 1.0;
	}
}

#endif //USE_NO_SHADOWS

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

float light_process_omni_shadow(uint idx, vec3 vertex, vec3 normal) {
#ifndef USE_NO_SHADOWS
	if (omni_lights.data[idx].shadow_enabled) {
		// there is a shadowmap

		vec3 light_rel_vec = omni_lights.data[idx].position - vertex;
		float light_length = length(light_rel_vec);

		vec4 v = vec4(vertex, 1.0);

		vec4 splane = (omni_lights.data[idx].shadow_matrix * v);
		float shadow_len = length(splane.xyz); //need to remember shadow len from here

		{
			vec3 nofs = normal_interp * omni_lights.data[idx].shadow_normal_bias / omni_lights.data[idx].inv_radius;
			nofs *= (1.0 - max(0.0, dot(normalize(light_rel_vec), normalize(normal_interp))));
			v.xyz += nofs;
			splane = (omni_lights.data[idx].shadow_matrix * v);
		}

		float shadow;

#ifdef USE_SOFT_SHADOWS
		if (omni_lights.data[idx].soft_shadow_size > 0.0) {
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

			vec3 normal = normalize(splane.xyz);
			vec3 v0 = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
			vec3 tangent = normalize(cross(v0, normal));
			vec3 bitangent = normalize(cross(tangent, normal));
			float z_norm = shadow_len * omni_lights.data[idx].inv_radius;

			tangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;
			bitangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;

			for (uint i = 0; i < scene_data.penumbra_shadow_samples; i++) {
				vec2 disk = disk_rotation * scene_data.penumbra_shadow_kernel[i].xy;

				vec3 pos = splane.xyz + tangent * disk.x + bitangent * disk.y;

				pos = normalize(pos);
				vec4 uv_rect = omni_lights.data[idx].atlas_rect;

				if (pos.z >= 0.0) {
					pos.z += 1.0;
					uv_rect.y += uv_rect.w;
				} else {
					pos.z = 1.0 - pos.z;
				}

				pos.xy /= pos.z;

				pos.xy = pos.xy * 0.5 + 0.5;
				pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;

				float d = textureLod(sampler2D(shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), pos.xy, 0.0).r;
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
				for (uint i = 0; i < scene_data.penumbra_shadow_samples; i++) {
					vec2 disk = disk_rotation * scene_data.penumbra_shadow_kernel[i].xy;
					vec3 pos = splane.xyz + tangent * disk.x + bitangent * disk.y;

					pos = normalize(pos);
					vec4 uv_rect = omni_lights.data[idx].atlas_rect;

					if (pos.z >= 0.0) {
						pos.z += 1.0;
						uv_rect.y += uv_rect.w;
					} else {
						pos.z = 1.0 - pos.z;
					}

					pos.xy /= pos.z;

					pos.xy = pos.xy * 0.5 + 0.5;
					pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;
					shadow += textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(pos.xy, z_norm, 1.0));
				}

				shadow /= float(scene_data.penumbra_shadow_samples);

			} else {
				//no blockers found, so no shadow
				shadow = 1.0;
			}
		} else {
#endif
			splane.xyz = normalize(splane.xyz);
			vec4 clamp_rect = omni_lights.data[idx].atlas_rect;

			if (splane.z >= 0.0) {
				splane.z += 1.0;

				clamp_rect.y += clamp_rect.w;

			} else {
				splane.z = 1.0 - splane.z;
			}

			splane.xy /= splane.z;

			splane.xy = splane.xy * 0.5 + 0.5;
			splane.z = (shadow_len - omni_lights.data[idx].shadow_bias) * omni_lights.data[idx].inv_radius;
			splane.xy = clamp_rect.xy + splane.xy * clamp_rect.zw;
			splane.w = 1.0; //needed? i think it should be 1 already
			shadow = sample_pcf_shadow(shadow_atlas, omni_lights.data[idx].soft_shadow_scale * scene_data.shadow_atlas_pixel_size, splane);
#ifdef USE_SOFT_SHADOWS
		}
#endif

		return shadow;
	}
#endif

	return 1.0;
}

void light_process_omni(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float shadow,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_curve,
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

#ifdef USE_SOFT_SHADOWS
	float size_A = 0.0;

	if (omni_lights.data[idx].size > 0.0) {
		float t = omni_lights.data[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}
#endif

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth; //no transmittance by default
	transmittance_color.a *= light_attenuation;
	{
		vec4 clamp_rect = omni_lights.data[idx].atlas_rect;

		//redo shadowmapping, but shrink the model a bit to avoid arctifacts
		vec4 splane = (omni_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * omni_lights.data[idx].transmittance_bias, 1.0));

		shadow_len = length(splane.xyz);
		splane = normalize(splane.xyz);

		if (splane.z >= 0.0) {
			splane.z += 1.0;

		} else {
			splane.z = 1.0 - splane.z;
		}

		splane.xy /= splane.z;
		splane.xy = splane.xy * 0.5 + 0.5;
		splane.z = shadow_len * omni_lights.data[idx].inv_radius;
		splane.xy = clamp_rect.xy + splane.xy * clamp_rect.zw;
		splane.w = 1.0; //needed? i think it should be 1 already

		float shadow_z = textureLod(sampler2D(shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), splane.xy, 0.0).r;
		transmittance_z = (splane.z - shadow_z) / omni_lights.data[idx].inv_radius;
	}
#endif

#if 0

	if (omni_lights.data[idx].projector_rect != vec4(0.0)) {
		vec3 local_v = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;
		local_v = normalize(local_v);

		vec4 atlas_rect = omni_lights.data[idx].projector_rect;

		if (local_v.z >= 0.0) {
			local_v.z += 1.0;
			atlas_rect.y += atlas_rect.w;

		} else {
			local_v.z = 1.0 - local_v.z;
		}

		local_v.xy /= local_v.z;
		local_v.xy = local_v.xy * 0.5 + 0.5;
		vec2 proj_uv = local_v.xy * atlas_rect.zw;

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

		vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), proj_uv + atlas_rect.xy, proj_uv_ddx, proj_uv_ddy);
		no_shadow = mix(no_shadow, proj.rgb, proj.a);
	}
#endif

	light_attenuation *= shadow;

	light_compute(normal, normalize(light_rel_vec), eye_vec, color, light_attenuation, f0, orms, omni_lights.data[idx].specular_amount,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_curve,
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
#ifdef USE_SOFT_SHADOWS
			size_A,
#endif
#ifdef USE_SHADOW_TO_OPACITY
			alpha,
#endif
			diffuse_light,
			specular_light);
}

float light_process_spot_shadow(uint idx, vec3 vertex, vec3 normal) {
#ifndef USE_NO_SHADOWS
	if (spot_lights.data[idx].shadow_enabled) {
		vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
		float light_length = length(light_rel_vec);
		vec3 spot_dir = spot_lights.data[idx].direction;
		//there is a shadowmap
		vec4 v = vec4(vertex, 1.0);

		v.xyz -= spot_dir * spot_lights.data[idx].shadow_bias;

		float z_norm = dot(spot_dir, -light_rel_vec) * spot_lights.data[idx].inv_radius;

		float depth_bias_scale = 1.0 / (max(0.0001, z_norm)); //the closer to the light origin, the more you have to offset to reach 1px in the map
		vec3 normal_bias = normalize(normal_interp) * (1.0 - max(0.0, dot(spot_dir, -normalize(normal_interp)))) * spot_lights.data[idx].shadow_normal_bias * depth_bias_scale;
		normal_bias -= spot_dir * dot(spot_dir, normal_bias); //only XY, no Z
		v.xyz += normal_bias;

		//adjust with bias
		z_norm = dot(spot_dir, v.xyz - spot_lights.data[idx].position) * spot_lights.data[idx].inv_radius;

		float shadow;

		vec4 splane = (spot_lights.data[idx].shadow_matrix * v);
		splane /= splane.w;

#ifdef USE_SOFT_SHADOWS
		if (spot_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker

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
			for (uint i = 0; i < scene_data.penumbra_shadow_samples; i++) {
				vec2 suv = shadow_uv + (disk_rotation * scene_data.penumbra_shadow_kernel[i].xy) * uv_size;
				suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
				float d = textureLod(sampler2D(shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), suv, 0.0).r;
				if (d < z_norm) {
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
				for (uint i = 0; i < scene_data.penumbra_shadow_samples; i++) {
					vec2 suv = shadow_uv + (disk_rotation * scene_data.penumbra_shadow_kernel[i].xy) * uv_size;
					suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
					shadow += textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(suv, z_norm, 1.0));
				}

				shadow /= float(scene_data.penumbra_shadow_samples);

			} else {
				//no blockers found, so no shadow
				shadow = 1.0;
			}

		} else {
#endif
			//hard shadow
			vec4 shadow_uv = vec4(splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy, splane.z, 1.0);

			shadow = sample_pcf_shadow(shadow_atlas, spot_lights.data[idx].soft_shadow_scale * scene_data.shadow_atlas_pixel_size, shadow_uv);
#ifdef USE_SOFT_SHADOWS
		}
#endif

		return shadow;
	}

#endif //USE_NO_SHADOWS

	return 1.0;
}

void light_process_spot(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float shadow,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_curve,
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
	vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float spot_attenuation = get_omni_attenuation(light_length, spot_lights.data[idx].inv_radius, spot_lights.data[idx].attenuation);
	vec3 spot_dir = spot_lights.data[idx].direction;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir), spot_lights.data[idx].cone_angle);
	float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - spot_lights.data[idx].cone_angle));
	spot_attenuation *= 1.0 - pow(spot_rim, spot_lights.data[idx].cone_attenuation);
	float light_attenuation = spot_attenuation;
	vec3 color = spot_lights.data[idx].color;
	float specular_amount = spot_lights.data[idx].specular_amount;

#ifdef USE_SOFT_SHADOWS
	float size_A = 0.0;

	if (spot_lights.data[idx].size > 0.0) {
		float t = spot_lights.data[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}
#endif

	/*
	if (spot_lights.data[idx].atlas_rect!=vec4(0.0)) {
		//use projector texture
	}
	*/

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth;
	transmittance_color.a *= light_attenuation;
	{
		splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * spot_lights.data[idx].transmittance_bias, 1.0));
		splane /= splane.w;
		splane.xy = splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy;

		float shadow_z = textureLod(sampler2D(shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), splane.xy, 0.0).r;
		//reconstruct depth
		shadow_z /= spot_lights.data[idx].inv_radius;
		//distance to light plane
		float z = dot(spot_dir, -light_rel_vec);
		transmittance_z = z - shadow_z;
	}
#endif //LIGHT_TRANSMITTANCE_USED

	light_attenuation *= shadow;

	light_compute(normal, normalize(light_rel_vec), eye_vec, color, light_attenuation, f0, orms, spot_lights.data[idx].specular_amount,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_curve,
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
#ifdef USE_SOFT_SHADOW
			size_A,
#endif
#ifdef USE_SHADOW_TO_OPACITY
			alpha,
#endif
			diffuse_light, specular_light);
}

void reflection_process(uint ref_index, vec3 vertex, vec3 normal, float roughness, vec3 ambient_light, vec3 specular_light, inout vec4 ambient_accum, inout vec4 reflection_accum) {
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
