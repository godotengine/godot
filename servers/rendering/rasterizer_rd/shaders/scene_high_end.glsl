/* clang-format off */
[vertex]

#version 450

VERSION_DEFINES

#include "scene_high_end_inc.glsl"

/* INPUT ATTRIBS */

layout(location = 0) in vec3 vertex_attrib;
/* clang-format on */
layout(location = 1) in vec3 normal_attrib;
#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 2) in vec4 tangent_attrib;
#endif

#if defined(COLOR_USED)
layout(location = 3) in vec4 color_attrib;
#endif

layout(location = 4) in vec2 uv_attrib;

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 5) in vec2 uv2_attrib;
#endif

layout(location = 6) in uvec4 bone_attrib; // always bound, even if unused

/* Varyings */

layout(location = 0) out vec3 vertex_interp;
layout(location = 1) out vec3 normal_interp;

#if defined(COLOR_USED)
layout(location = 2) out vec4 color_interp;
#endif

layout(location = 3) out vec2 uv_interp;

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 4) out vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 5) out vec3 tangent_interp;
layout(location = 6) out vec3 binormal_interp;
#endif

#ifdef USE_MATERIAL_UNIFORMS
layout(set = 5, binding = 0, std140) uniform MaterialUniforms{
	/* clang-format off */
MATERIAL_UNIFORMS
	/* clang-format on */
} material;
#endif

/* clang-format off */

VERTEX_SHADER_GLOBALS

/* clang-format on */

// FIXME: This triggers a Mesa bug that breaks rendering, so disabled for now.
// See GH-13450 and https://bugs.freedesktop.org/show_bug.cgi?id=100316
invariant gl_Position;

layout(location = 7) flat out uint instance_index;

#ifdef MODE_DUAL_PARABOLOID

layout(location = 8) out float dp_clip;

#endif

void main() {

	instance_index = draw_call.instance_index;
	vec4 instance_custom = vec4(0.0);
#if defined(COLOR_USED)
	color_interp = color_attrib;
#endif

	mat4 world_matrix = instances.data[instance_index].transform;
	mat3 world_normal_matrix = mat3(instances.data[instance_index].normal_transform);

	if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH)) {
		//multimesh, instances are for it
		uint offset = (instances.data[instance_index].flags >> INSTANCE_FLAGS_MULTIMESH_STRIDE_SHIFT) & INSTANCE_FLAGS_MULTIMESH_STRIDE_MASK;
		offset *= gl_InstanceIndex;

		mat4 matrix;
		if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH_FORMAT_2D)) {
			matrix = mat4(transforms.data[offset + 0], transforms.data[offset + 1], vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
			offset += 2;
		} else {
			matrix = mat4(transforms.data[offset + 0], transforms.data[offset + 1], transforms.data[offset + 2], vec4(0.0, 0.0, 0.0, 1.0));
			offset += 3;
		}

		if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH_HAS_COLOR)) {
#ifdef COLOR_USED
			color_interp *= transforms.data[offset];
#endif
			offset += 1;
		}

		if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_MULTIMESH_HAS_CUSTOM_DATA)) {
			instance_custom = transforms.data[offset];
		}

		//transpose
		matrix = transpose(matrix);
		world_matrix = world_matrix * matrix;
		world_normal_matrix = world_normal_matrix * mat3(matrix);

	} else {
		//not a multimesh, instances are for multiple draw calls
		instance_index += gl_InstanceIndex;
	}

	vec3 vertex = vertex_attrib;
	vec3 normal = normal_attrib;

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	vec3 tangent = tangent_attrib.xyz;
	float binormalf = tangent_attrib.a;
	vec3 binormal = normalize(cross(normal, tangent) * binormalf);
#endif

	if (bool(instances.data[instance_index].flags & INSTANCE_FLAGS_SKELETON)) {
		//multimesh, instances are for it

		uvec2 bones_01 = uvec2(bone_attrib.x & 0xFFFF, bone_attrib.x >> 16) * 3;
		uvec2 bones_23 = uvec2(bone_attrib.y & 0xFFFF, bone_attrib.y >> 16) * 3;
		vec2 weights_01 = unpackUnorm2x16(bone_attrib.z);
		vec2 weights_23 = unpackUnorm2x16(bone_attrib.w);

		mat4 m = mat4(transforms.data[bones_01.x], transforms.data[bones_01.x + 1], transforms.data[bones_01.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.x;
		m += mat4(transforms.data[bones_01.y], transforms.data[bones_01.y + 1], transforms.data[bones_01.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_01.y;
		m += mat4(transforms.data[bones_23.x], transforms.data[bones_23.x + 1], transforms.data[bones_23.x + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.x;
		m += mat4(transforms.data[bones_23.y], transforms.data[bones_23.y + 1], transforms.data[bones_23.y + 2], vec4(0.0, 0.0, 0.0, 1.0)) * weights_23.y;

		//reverse order because its transposed
		vertex = (vec4(vertex, 1.0) * m).xyz;
		normal = (vec4(normal, 0.0) * m).xyz;

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)

		tangent = (vec4(tangent, 0.0) * m).xyz;
		binormal = (vec4(binormal, 0.0) * m).xyz;
#endif
	}

	uv_interp = uv_attrib;

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
	uv2_interp = uv2_attrib;
#endif

#ifdef USE_OVERRIDE_POSITION
	vec4 position;
#endif

	mat4 projection_matrix = scene_data.projection_matrix;

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (world_matrix * vec4(vertex, 1.0)).xyz;

	normal = world_normal_matrix * normal;

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	tangent = world_normal_matrix * tangent;
	binormal = world_normal_matrix * binormal;

#endif
#endif

	float roughness = 1.0;

	mat4 modelview = scene_data.inv_camera_matrix * world_matrix;
	mat3 modelview_normal = mat3(scene_data.inv_camera_matrix) * world_normal_matrix;

	{
		/* clang-format off */

VERTEX_SHADER_CODE

		/* clang-format on */
	}

// using local coordinates (default)
#if !defined(SKIP_TRANSFORM_USED) && !defined(VERTEX_WORLD_COORDS_USED)

	vertex = (modelview * vec4(vertex, 1.0)).xyz;
	normal = modelview_normal * normal;
#endif

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	binormal = modelview_normal * binormal;
	tangent = modelview_normal * tangent;
#endif

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (scene_data.inv_camera_matrix * vec4(vertex, 1.0)).xyz;
	normal = mat3(scene_data.inverse_normal_matrix) * normal;

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	binormal = mat3(scene_data.camera_inverse_binormal_matrix) * binormal;
	tangent = mat3(scene_data.camera_inverse_tangent_matrix) * tangent;
#endif
#endif

	vertex_interp = vertex;
	normal_interp = normal;

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	tangent_interp = tangent;
	binormal_interp = binormal;
#endif

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_DUAL_PARABOLOID

	vertex_interp.z *= scene_data.dual_paraboloid_side;
	normal_interp.z *= scene_data.dual_paraboloid_side;

	dp_clip = vertex_interp.z; //this attempts to avoid noise caused by objects sent to the other parabolloid side due to bias

	//for dual paraboloid shadow mapping, this is the fastest but least correct way, as it curves straight edges

	vec3 vtx = vertex_interp;
	float distance = length(vtx);
	vtx = normalize(vtx);
	vtx.xy /= 1.0 - vtx.z;
	vtx.z = (distance / scene_data.z_far);
	vtx.z = vtx.z * 2.0 - 1.0;
	vertex_interp = vtx;

#endif

#endif //MODE_RENDER_DEPTH

#ifdef USE_OVERRIDE_POSITION
	gl_Position = position;
#else
	gl_Position = projection_matrix * vec4(vertex_interp, 1.0);
#endif

#ifdef MODE_RENDER_DEPTH
	if (scene_data.pancake_shadows) {
		if (gl_Position.z <= 0.00001) {
			gl_Position.z = 0.00001;
		}
	}
#endif
}

/* clang-format off */
[fragment]

#version 450

VERSION_DEFINES

#include "scene_high_end_inc.glsl"

/* Varyings */

layout(location = 0) in vec3 vertex_interp;
/* clang-format on */
layout(location = 1) in vec3 normal_interp;

#if defined(COLOR_USED)
layout(location = 2) in vec4 color_interp;
#endif

layout(location = 3) in vec2 uv_interp;

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 4) in vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 5) in vec3 tangent_interp;
layout(location = 6) in vec3 binormal_interp;
#endif

layout(location = 7) flat in uint instance_index;

#ifdef MODE_DUAL_PARABOLOID

layout(location = 8) in float dp_clip;

#endif

//defines to keep compatibility with vertex

#define world_matrix instances.data[instance_index].transform
#define world_normal_matrix instances.data[instance_index].normal_transform
#define projection_matrix scene_data.projection_matrix

#if defined(ENABLE_SSS) && defined(ENABLE_TRANSMITTANCE)
//both required for transmittance to be enabled
#define LIGHT_TRANSMITTANCE_USED
#endif

#ifdef USE_MATERIAL_UNIFORMS
layout(set = 5, binding = 0, std140) uniform MaterialUniforms{
	/* clang-format off */
MATERIAL_UNIFORMS
	/* clang-format on */
} material;
#endif

/* clang-format off */

FRAGMENT_SHADER_GLOBALS

/* clang-format on */

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_RENDER_MATERIAL

layout(location = 0) out vec4 albedo_output_buffer;
layout(location = 1) out vec4 normal_output_buffer;
layout(location = 2) out vec4 orm_output_buffer;
layout(location = 3) out vec4 emission_output_buffer;
layout(location = 4) out float depth_output_buffer;

#endif

#ifdef MODE_RENDER_NORMAL
layout(location = 0) out vec4 normal_output_buffer;
#ifdef MODE_RENDER_ROUGHNESS
layout(location = 1) out float roughness_output_buffer;
#endif //MODE_RENDER_ROUGHNESS
#endif //MODE_RENDER_NORMAL
#else // RENDER DEPTH

#ifdef MODE_MULTIPLE_RENDER_TARGETS

layout(location = 0) out vec4 diffuse_buffer; //diffuse (rgb) and roughness
layout(location = 1) out vec4 specular_buffer; //specular and SSS (subsurface scatter)
#else

layout(location = 0) out vec4 frag_color;
#endif

#endif // RENDER DEPTH

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

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

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
	if (a >= 1.0) return 1.0 / M_PI;
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

void light_compute(vec3 N, vec3 L, vec3 V, float A, vec3 light_color, float attenuation, vec3 shadow_attenuation, vec3 diffuse_color, float roughness, float metallic, float specular, float specular_blob_intensity,
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
		float rim, float rim_tint,
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

#if defined(USE_LIGHT_SHADER_CODE)
	// light is written by the light shader

	vec3 normal = N;
	vec3 albedo = diffuse_color;
	vec3 light = L;
	vec3 view = V;

	/* clang-format off */

LIGHT_SHADER_CODE

	/* clang-format on */

#else
	float NdotL = min(A + dot(N, L), 1.0);
	float cNdotL = max(NdotL, 0.0); // clamped NdotL
	float NdotV = dot(N, V);
	float cNdotV = max(NdotV, 0.0);

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_BLINN) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	vec3 H = normalize(V + L);
#endif

#if defined(SPECULAR_BLINN) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	float cNdotH = clamp(A + dot(N, H), 0.0, 1.0);
#endif

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	float cLdotH = clamp(A + dot(L, H), 0.0, 1.0);
#endif

	if (metallic < 1.0) {
#if defined(DIFFUSE_OREN_NAYAR)
		vec3 diffuse_brdf_NL;
#else
		float diffuse_brdf_NL; // BRDF times N.L for calculating diffuse radiance
#endif

#if defined(DIFFUSE_LAMBERT_WRAP)
		// energy conserving lambert wrap shader
		diffuse_brdf_NL = max(0.0, (NdotL + roughness) / ((1.0 + roughness) * (1.0 + roughness)));

#elif defined(DIFFUSE_OREN_NAYAR)

		{
			// see http://mimosa-pudica.net/improved-oren-nayar.html
			float LdotV = dot(L, V);

			float s = LdotV - NdotL * NdotV;
			float t = mix(1.0, max(NdotL, NdotV), step(0.0, s));

			float sigma2 = roughness * roughness; // TODO: this needs checking
			vec3 A = 1.0 + sigma2 * (-0.5 / (sigma2 + 0.33) + 0.17 * diffuse_color / (sigma2 + 0.13));
			float B = 0.45 * sigma2 / (sigma2 + 0.09);

			diffuse_brdf_NL = cNdotL * (A + vec3(B) * s / t) * (1.0 / M_PI);
		}

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

		diffuse_light += light_color * diffuse_color * shadow_attenuation * diffuse_brdf_NL * attenuation;

#if defined(LIGHT_BACKLIGHT_USED)
		diffuse_light += light_color * diffuse_color * (vec3(1.0 / M_PI) - diffuse_brdf_NL) * backlight * attenuation;
#endif

#if defined(LIGHT_RIM_USED)
		float rim_light = pow(max(0.0, 1.0 - cNdotV), max(0.0, (1.0 - roughness) * 16.0));
		diffuse_light += rim_light * rim * mix(vec3(1.0), diffuse_color, rim_tint) * light_color;
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

			diffuse_light += profile * transmittance_color.a * diffuse_color * light_color * clamp(transmittance_boost - NdotL, 0.0, 1.0) * (1.0 / M_PI) * attenuation;
		}
#else

		if (transmittance_depth > 0.0) {
			float fade = clamp(abs(transmittance_z / transmittance_depth), 0.0, 1.0);

			fade = pow(max(0.0, 1.0 - fade), transmittance_curve);
			fade *= clamp(transmittance_boost - NdotL, 0.0, 1.0);

			diffuse_light += diffuse_color * transmittance_color.rgb * light_color * (1.0 / M_PI) * transmittance_color.a * fade * attenuation;
		}

#endif //SSS_MODE_SKIN

#endif //LIGHT_TRANSMITTANCE_USED
	}

	if (roughness > 0.0) { // FIXME: roughness == 0 should not disable specular light entirely

		// D

#if defined(SPECULAR_BLINN)

		//normalized blinn
		float shininess = exp2(15.0 * (1.0 - roughness) + 1.0) * 0.25;
		float blinn = pow(cNdotH, shininess) * cNdotL;
		blinn *= (shininess + 8.0) * (1.0 / (8.0 * M_PI));
		float intensity = blinn;

		specular_light += light_color * shadow_attenuation * intensity * specular_blob_intensity * attenuation;

#elif defined(SPECULAR_PHONG)

		vec3 R = normalize(-reflect(L, N));
		float cRdotV = clamp(A + dot(R, V), 0.0, 1.0);
		float shininess = exp2(15.0 * (1.0 - roughness) + 1.0) * 0.25;
		float phong = pow(cRdotV, shininess);
		phong *= (shininess + 8.0) * (1.0 / (8.0 * M_PI));
		float intensity = (phong) / max(4.0 * cNdotV * cNdotL, 0.75);

		specular_light += light_color * shadow_attenuation * intensity * specular_blob_intensity * attenuation;

#elif defined(SPECULAR_TOON)

		vec3 R = normalize(-reflect(L, N));
		float RdotV = dot(R, V);
		float mid = 1.0 - roughness;
		mid *= mid;
		float intensity = smoothstep(mid - roughness * 0.5, mid + roughness * 0.5, RdotV) * mid;
		diffuse_light += light_color * shadow_attenuation * intensity * specular_blob_intensity * attenuation; // write to diffuse_light, as in toon shading you generally want no reflection

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
		vec3 f0 = F0(metallic, specular, diffuse_color);
		float cLdotH5 = SchlickFresnel(cLdotH);
		vec3 F = mix(vec3(cLdotH5), vec3(1.0), f0);

		vec3 specular_brdf_NL = cNdotL * D * F * G;

		specular_light += specular_brdf_NL * light_color * shadow_attenuation * specular_blob_intensity * attenuation;
#endif

#if defined(LIGHT_CLEARCOAT_USED)

#if !defined(SPECULAR_SCHLICK_GGX)
		float cLdotH5 = SchlickFresnel(cLdotH);
#endif
		float Dr = GTR1(cNdotH, mix(.1, .001, clearcoat_gloss));
		float Fr = mix(.04, 1.0, cLdotH5);
		float Gr = G_GGX_2cos(cNdotL, .25) * G_GGX_2cos(cNdotV, .25);

		float clearcoat_specular_brdf_NL = 0.25 * clearcoat * Gr * Fr * Dr * cNdotL;

		specular_light += clearcoat_specular_brdf_NL * light_color * shadow_attenuation * specular_blob_intensity * attenuation;
#endif
	}

#ifdef USE_SHADOW_TO_OPACITY
	alpha = min(alpha, clamp(1.0 - length(shadow_attenuation * attenuation), 0.0, 1.0));
#endif

#endif //defined(USE_LIGHT_SHADER_CODE)
}

#ifndef USE_NO_SHADOWS

// Produces cheap but low-quality white noise, nothing special
float quick_hash(vec2 pos) {
	return fract(sin(dot(pos * 19.19, vec2(49.5791, 97.413))) * 49831.189237);
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

void light_process_omni(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 albedo, float roughness, float metallic, float specular, float p_blob_intensity,
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
		float rim, float rim_tint,
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

	vec3 light_rel_vec = lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float normalized_distance = light_length * lights.data[idx].inv_radius;
	vec2 attenuation_energy = unpackHalf2x16(lights.data[idx].attenuation_energy);
	float omni_attenuation = pow(max(1.0 - normalized_distance, 0.0), attenuation_energy.x);
	float light_attenuation = omni_attenuation;
	vec3 shadow_attenuation = vec3(1.0);
	vec4 color_specular = unpackUnorm4x8(lights.data[idx].color_specular);
	color_specular.rgb *= attenuation_energy.y;
	float size_A = 0.0;

	if (lights.data[idx].size > 0.0) {

		float t = lights.data[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth; //no transmittance by default
#endif

#ifndef USE_NO_SHADOWS
	vec4 shadow_color_enabled = unpackUnorm4x8(lights.data[idx].shadow_color_enabled);
	if (shadow_color_enabled.w > 0.5) {
		// there is a shadowmap

		vec4 v = vec4(vertex, 1.0);

		vec4 splane = (lights.data[idx].shadow_matrix * v);
		float shadow_len = length(splane.xyz); //need to remember shadow len from here

		{
			vec3 nofs = normal_interp * lights.data[idx].shadow_normal_bias / lights.data[idx].inv_radius;
			nofs *= (1.0 - max(0.0, dot(normalize(light_rel_vec), normalize(normal_interp))));
			v.xyz += nofs;
			splane = (lights.data[idx].shadow_matrix * v);
		}

		float shadow;

		if (lights.data[idx].soft_shadow_size > 0.0) {
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
			float z_norm = shadow_len * lights.data[idx].inv_radius;

			tangent *= lights.data[idx].soft_shadow_size * lights.data[idx].soft_shadow_scale;
			bitangent *= lights.data[idx].soft_shadow_size * lights.data[idx].soft_shadow_scale;

			for (uint i = 0; i < scene_data.penumbra_shadow_samples; i++) {

				vec2 disk = disk_rotation * scene_data.penumbra_shadow_kernel[i].xy;

				vec3 pos = splane.xyz + tangent * disk.x + bitangent * disk.y;

				pos = normalize(pos);
				vec4 uv_rect = lights.data[idx].atlas_rect;

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

				z_norm -= lights.data[idx].inv_radius * lights.data[idx].shadow_bias;

				shadow = 0.0;
				for (uint i = 0; i < scene_data.penumbra_shadow_samples; i++) {

					vec2 disk = disk_rotation * scene_data.penumbra_shadow_kernel[i].xy;
					vec3 pos = splane.xyz + tangent * disk.x + bitangent * disk.y;

					pos = normalize(pos);
					vec4 uv_rect = lights.data[idx].atlas_rect;

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

			splane.xyz = normalize(splane.xyz);
			vec4 clamp_rect = lights.data[idx].atlas_rect;

			if (splane.z >= 0.0) {

				splane.z += 1.0;

				clamp_rect.y += clamp_rect.w;

			} else {
				splane.z = 1.0 - splane.z;
			}

			splane.xy /= splane.z;

			splane.xy = splane.xy * 0.5 + 0.5;
			splane.z = (shadow_len - lights.data[idx].shadow_bias) * lights.data[idx].inv_radius;
			splane.xy = clamp_rect.xy + splane.xy * clamp_rect.zw;
			splane.w = 1.0; //needed? i think it should be 1 already
			shadow = sample_pcf_shadow(shadow_atlas, lights.data[idx].soft_shadow_scale * scene_data.shadow_atlas_pixel_size, splane);
		}

#ifdef LIGHT_TRANSMITTANCE_USED
		{

			vec4 clamp_rect = lights.data[idx].atlas_rect;

			//redo shadowmapping, but shrink the model a bit to avoid arctifacts
			splane = (lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * lights.data[idx].transmittance_bias, 1.0));

			shadow_len = length(splane.xyz);
			splane = normalize(splane.xyz);

			if (splane.z >= 0.0) {

				splane.z += 1.0;

			} else {

				splane.z = 1.0 - splane.z;
			}

			splane.xy /= splane.z;
			splane.xy = splane.xy * 0.5 + 0.5;
			splane.z = shadow_len * lights.data[idx].inv_radius;
			splane.xy = clamp_rect.xy + splane.xy * clamp_rect.zw;
			splane.w = 1.0; //needed? i think it should be 1 already

			float shadow_z = textureLod(sampler2D(shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), splane.xy, 0.0).r;
			transmittance_z = (splane.z - shadow_z) / lights.data[idx].inv_radius;
		}
#endif

		vec3 no_shadow = vec3(1.0);

		if (lights.data[idx].projector_rect != vec4(0.0)) {

			vec3 local_v = (lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;
			local_v = normalize(local_v);

			vec4 atlas_rect = lights.data[idx].projector_rect;

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
				vec3 local_v_ddx = (lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0)).xyz;
				local_v_ddx = normalize(local_v_ddx);

				if (local_v_ddx.z >= 0.0) {

					local_v_ddx.z += 1.0;
				} else {

					local_v_ddx.z = 1.0 - local_v_ddx.z;
				}

				local_v_ddx.xy /= local_v_ddx.z;
				local_v_ddx.xy = local_v_ddx.xy * 0.5 + 0.5;

				proj_uv_ddx = local_v_ddx.xy * atlas_rect.zw - proj_uv;

				vec3 local_v_ddy = (lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0)).xyz;
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

		shadow_attenuation = mix(shadow_color_enabled.rgb, no_shadow, shadow);
	}
#endif //USE_NO_SHADOWS

	light_compute(normal, normalize(light_rel_vec), eye_vec, size_A, color_specular.rgb, light_attenuation, shadow_attenuation, albedo, roughness, metallic, specular, color_specular.a * p_blob_intensity,
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
			rim * omni_attenuation, rim_tint,
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

void light_process_spot(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 albedo, float roughness, float metallic, float specular, float p_blob_intensity,
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
		float rim, float rim_tint,
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

	vec3 light_rel_vec = lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float normalized_distance = light_length * lights.data[idx].inv_radius;
	vec2 attenuation_energy = unpackHalf2x16(lights.data[idx].attenuation_energy);
	float spot_attenuation = pow(max(1.0 - normalized_distance, 0.001), attenuation_energy.x);
	vec3 spot_dir = lights.data[idx].direction;
	vec2 spot_att_angle = unpackHalf2x16(lights.data[idx].cone_attenuation_angle);
	float scos = max(dot(-normalize(light_rel_vec), spot_dir), spot_att_angle.y);
	float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - spot_att_angle.y));
	spot_attenuation *= 1.0 - pow(spot_rim, spot_att_angle.x);
	float light_attenuation = spot_attenuation;
	vec3 shadow_attenuation = vec3(1.0);
	vec4 color_specular = unpackUnorm4x8(lights.data[idx].color_specular);
	color_specular.rgb *= attenuation_energy.y;

	float size_A = 0.0;

	if (lights.data[idx].size > 0.0) {

		float t = lights.data[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}
/*
	if (lights.data[idx].atlas_rect!=vec4(0.0)) {
		//use projector texture
	}
	*/
#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth;
#endif

#ifndef USE_NO_SHADOWS
	vec4 shadow_color_enabled = unpackUnorm4x8(lights.data[idx].shadow_color_enabled);
	if (shadow_color_enabled.w > 0.5) {
		//there is a shadowmap
		vec4 v = vec4(vertex, 1.0);

		v.xyz -= spot_dir * lights.data[idx].shadow_bias;

		float z_norm = dot(spot_dir, -light_rel_vec) * lights.data[idx].inv_radius;

		float depth_bias_scale = 1.0 / (max(0.0001, z_norm)); //the closer to the light origin, the more you have to offset to reach 1px in the map
		vec3 normal_bias = normalize(normal_interp) * (1.0 - max(0.0, dot(spot_dir, -normalize(normal_interp)))) * lights.data[idx].shadow_normal_bias * depth_bias_scale;
		normal_bias -= spot_dir * dot(spot_dir, normal_bias); //only XY, no Z
		v.xyz += normal_bias;

		//adjust with bias
		z_norm = dot(spot_dir, v.xyz - lights.data[idx].position) * lights.data[idx].inv_radius;

		float shadow;

		vec4 splane = (lights.data[idx].shadow_matrix * v);
		splane /= splane.w;

		if (lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker

			vec2 shadow_uv = splane.xy * lights.data[idx].atlas_rect.zw + lights.data[idx].atlas_rect.xy;

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			float uv_size = lights.data[idx].soft_shadow_size * z_norm * lights.data[idx].soft_shadow_scale;
			vec2 clamp_max = lights.data[idx].atlas_rect.xy + lights.data[idx].atlas_rect.zw;
			for (uint i = 0; i < scene_data.penumbra_shadow_samples; i++) {

				vec2 suv = shadow_uv + (disk_rotation * scene_data.penumbra_shadow_kernel[i].xy) * uv_size;
				suv = clamp(suv, lights.data[idx].atlas_rect.xy, clamp_max);
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
					suv = clamp(suv, lights.data[idx].atlas_rect.xy, clamp_max);
					shadow += textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(suv, z_norm, 1.0));
				}

				shadow /= float(scene_data.penumbra_shadow_samples);

			} else {
				//no blockers found, so no shadow
				shadow = 1.0;
			}

		} else {
			//hard shadow
			vec4 shadow_uv = vec4(splane.xy * lights.data[idx].atlas_rect.zw + lights.data[idx].atlas_rect.xy, z_norm, 1.0);

			shadow = sample_pcf_shadow(shadow_atlas, lights.data[idx].soft_shadow_scale * scene_data.shadow_atlas_pixel_size, shadow_uv);
		}

		vec3 no_shadow = vec3(1.0);

		if (lights.data[idx].projector_rect != vec4(0.0)) {

			splane = (lights.data[idx].shadow_matrix * vec4(vertex, 1.0));
			splane /= splane.w;

			vec2 proj_uv = splane.xy * lights.data[idx].projector_rect.zw;

			//ensure we have proper mipmaps
			vec4 splane_ddx = (lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0));
			splane_ddx /= splane_ddx.w;
			vec2 proj_uv_ddx = splane_ddx.xy * lights.data[idx].projector_rect.zw - proj_uv;

			vec4 splane_ddy = (lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0));
			splane_ddy /= splane_ddy.w;
			vec2 proj_uv_ddy = splane_ddy.xy * lights.data[idx].projector_rect.zw - proj_uv;

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), proj_uv + lights.data[idx].projector_rect.xy, proj_uv_ddx, proj_uv_ddy);
			no_shadow = mix(no_shadow, proj.rgb, proj.a);
		}

		shadow_attenuation = mix(shadow_color_enabled.rgb, no_shadow, shadow);

#ifdef LIGHT_TRANSMITTANCE_USED
		{

			splane = (lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * lights.data[idx].transmittance_bias, 1.0));
			splane /= splane.w;
			splane.xy = splane.xy * lights.data[idx].atlas_rect.zw + lights.data[idx].atlas_rect.xy;

			float shadow_z = textureLod(sampler2D(shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), splane.xy, 0.0).r;
			//reconstruct depth
			shadow_z / lights.data[idx].inv_radius;
			//distance to light plane
			float z = dot(spot_dir, -light_rel_vec);
			transmittance_z = z - shadow_z;
		}
#endif //LIGHT_TRANSMITTANCE_USED
	}

#endif //USE_NO_SHADOWS

	light_compute(normal, normalize(light_rel_vec), eye_vec, size_A, color_specular.rgb, light_attenuation, shadow_attenuation, albedo, roughness, metallic, specular, color_specular.a * p_blob_intensity,
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
			rim * spot_attenuation, rim_tint,
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

	if (reflections.data[ref_index].params.x > 0.0) { // compute reflection

		vec3 local_ref_vec = (reflections.data[ref_index].local_matrix * vec4(ref_vec, 0.0)).xyz;

		if (reflections.data[ref_index].params.w > 0.5) { //box project

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

		if (reflections.data[ref_index].params.z < 0.5) {
			reflection.rgb = mix(specular_light, reflection.rgb, blend);
		}

		reflection.rgb *= reflections.data[ref_index].params.x;
		reflection.a = blend;
		reflection.rgb *= reflection.a;

		reflection_accum += reflection;
	}

#if !defined(USE_LIGHTMAP) && !defined(USE_VOXEL_CONE_TRACING)
	if (reflections.data[ref_index].ambient.a > 0.0) { //compute ambient using skybox

		vec3 local_amb_vec = (reflections.data[ref_index].local_matrix * vec4(normal, 0.0)).xyz;

		vec4 ambient_out;

		ambient_out.rgb = textureLod(samplerCubeArray(reflection_atlas, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(local_amb_vec, reflections.data[ref_index].index), MAX_ROUGHNESS_LOD).rgb;

		ambient_out.a = blend;
		ambient_out.rgb = mix(reflections.data[ref_index].ambient.rgb, ambient_out.rgb, reflections.data[ref_index].ambient.a);
		if (reflections.data[ref_index].params.z < 0.5) {
			ambient_out.rgb = mix(ambient_light, ambient_out.rgb, blend);
		}

		ambient_out.rgb *= ambient_out.a;
		ambient_accum += ambient_out;
	} else {

		vec4 ambient_out;
		ambient_out.a = blend;
		ambient_out.rgb = reflections.data[ref_index].ambient.rgb;
		if (reflections.data[ref_index].params.z < 0.5) {
			ambient_out.rgb = mix(ambient_light, ambient_out.rgb, blend);
		}
		ambient_out.rgb *= ambient_out.a;
		ambient_accum += ambient_out;
	}
#endif //USE_LIGHTMAP or VCT
}

#ifdef USE_VOXEL_CONE_TRACING

//standard voxel cone trace
vec4 voxel_cone_trace(texture3D probe, vec3 cell_size, vec3 pos, vec3 direction, float tan_half_angle, float max_distance, float p_bias) {

	float dist = p_bias;
	vec4 color = vec4(0.0);

	while (dist < max_distance && color.a < 0.95) {
		float diameter = max(1.0, 2.0 * tan_half_angle * dist);
		vec3 uvw_pos = (pos + dist * direction) * cell_size;
		float half_diameter = diameter * 0.5;
		//check if outside, then break
		if (any(greaterThan(abs(uvw_pos - 0.5), vec3(0.5f + half_diameter * cell_size)))) {
			break;
		}
		vec4 scolor = textureLod(sampler3D(probe, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uvw_pos, log2(diameter));
		float a = (1.0 - color.a);
		color += a * scolor;
		dist += half_diameter;
	}

	return color;
}

#ifndef GI_PROBE_HIGH_QUALITY
//faster version for 45 degrees

#ifdef GI_PROBE_USE_ANISOTROPY

vec4 voxel_cone_trace_anisotropic_45_degrees(texture3D probe, texture3D aniso_pos, texture3D aniso_neg, vec3 normal, vec3 cell_size, vec3 pos, vec3 direction, float tan_half_angle, float max_distance, float p_bias) {

	float dist = p_bias;
	vec4 color = vec4(0.0);
	float radius = max(0.5, tan_half_angle * dist);
	float lod_level = log2(radius * 2.0);

	while (dist < max_distance && color.a < 0.95) {
		vec3 uvw_pos = (pos + dist * direction) * cell_size;
		//check if outside, then break
		if (any(greaterThan(abs(uvw_pos - 0.5), vec3(0.5f + radius * cell_size)))) {
			break;
		}

		vec4 scolor = textureLod(sampler3D(probe, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uvw_pos, lod_level);
		vec3 aniso_neg = textureLod(sampler3D(aniso_neg, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uvw_pos, lod_level).rgb;
		vec3 aniso_pos = textureLod(sampler3D(aniso_pos, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uvw_pos, lod_level).rgb;

		scolor.rgb *= dot(max(vec3(0.0), (normal * aniso_pos)), vec3(1.0)) + dot(max(vec3(0.0), (-normal * aniso_neg)), vec3(1.0));
		lod_level += 1.0;

		float a = (1.0 - color.a);
		scolor *= a;
		color += scolor;
		dist += radius;
		radius = max(0.5, tan_half_angle * dist);
	}

	return color;
}
#else

vec4 voxel_cone_trace_45_degrees(texture3D probe, vec3 cell_size, vec3 pos, vec3 direction, float tan_half_angle, float max_distance, float p_bias) {

	float dist = p_bias;
	vec4 color = vec4(0.0);
	float radius = max(0.5, tan_half_angle * dist);
	float lod_level = log2(radius * 2.0);

	while (dist < max_distance && color.a < 0.95) {
		vec3 uvw_pos = (pos + dist * direction) * cell_size;

		//check if outside, then break
		if (any(greaterThan(abs(uvw_pos - 0.5), vec3(0.5f + radius * cell_size)))) {
			break;
		}
		vec4 scolor = textureLod(sampler3D(probe, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uvw_pos, lod_level);
		lod_level += 1.0;

		float a = (1.0 - color.a);
		scolor *= a;
		color += scolor;
		dist += radius;
		radius = max(0.5, tan_half_angle * dist);
	}

	return color;
}

#endif

#elif defined(GI_PROBE_USE_ANISOTROPY)

//standard voxel cone trace
vec4 voxel_cone_trace_anisotropic(texture3D probe, texture3D aniso_pos, texture3D aniso_neg, vec3 normal, vec3 cell_size, vec3 pos, vec3 direction, float tan_half_angle, float max_distance, float p_bias) {

	float dist = p_bias;
	vec4 color = vec4(0.0);

	while (dist < max_distance && color.a < 0.95) {
		float diameter = max(1.0, 2.0 * tan_half_angle * dist);
		vec3 uvw_pos = (pos + dist * direction) * cell_size;
		float half_diameter = diameter * 0.5;
		//check if outside, then break
		if (any(greaterThan(abs(uvw_pos - 0.5), vec3(0.5f + half_diameter * cell_size)))) {
			break;
		}
		float log2_diameter = log2(diameter);
		vec4 scolor = textureLod(sampler3D(probe, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uvw_pos, log2_diameter);
		vec3 aniso_neg = textureLod(sampler3D(aniso_neg, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uvw_pos, log2_diameter).rgb;
		vec3 aniso_pos = textureLod(sampler3D(aniso_pos, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uvw_pos, log2_diameter).rgb;

		scolor.rgb *= dot(max(vec3(0.0), (normal * aniso_pos)), vec3(1.0)) + dot(max(vec3(0.0), (-normal * aniso_neg)), vec3(1.0));

		float a = (1.0 - color.a);
		scolor *= a;
		color += scolor;
		dist += half_diameter;
	}

	return color;
}

#endif

void gi_probe_compute(uint index, vec3 position, vec3 normal, vec3 ref_vec, mat3 normal_xform, float roughness, vec3 ambient, vec3 environment, inout vec4 out_spec, inout vec4 out_diff) {

	position = (gi_probes.data[index].xform * vec4(position, 1.0)).xyz;
	ref_vec = normalize((gi_probes.data[index].xform * vec4(ref_vec, 0.0)).xyz);
	normal = normalize((gi_probes.data[index].xform * vec4(normal, 0.0)).xyz);

	position += normal * gi_probes.data[index].normal_bias;

	//this causes corrupted pixels, i have no idea why..
	if (any(bvec2(any(lessThan(position, vec3(0.0))), any(greaterThan(position, gi_probes.data[index].bounds))))) {
		return;
	}

	vec3 blendv = abs(position / gi_probes.data[index].bounds * 2.0 - 1.0);
	float blend = clamp(1.0 - max(blendv.x, max(blendv.y, blendv.z)), 0.0, 1.0);
	//float blend=1.0;

	float max_distance = length(gi_probes.data[index].bounds);
	vec3 cell_size = 1.0 / gi_probes.data[index].bounds;

	//radiance

#ifdef GI_PROBE_HIGH_QUALITY

#define MAX_CONE_DIRS 6
	vec3 cone_dirs[MAX_CONE_DIRS] = vec3[](
			vec3(0.0, 0.0, 1.0),
			vec3(0.866025, 0.0, 0.5),
			vec3(0.267617, 0.823639, 0.5),
			vec3(-0.700629, 0.509037, 0.5),
			vec3(-0.700629, -0.509037, 0.5),
			vec3(0.267617, -0.823639, 0.5));

	float cone_weights[MAX_CONE_DIRS] = float[](0.25, 0.15, 0.15, 0.15, 0.15, 0.15);
	float cone_angle_tan = 0.577;

#elif defined(GI_PROBE_LOW_QUALITY)

#define MAX_CONE_DIRS 1

	vec3 cone_dirs[MAX_CONE_DIRS] = vec3[](
			vec3(0.0, 0.0, 1.0));

	float cone_weights[MAX_CONE_DIRS] = float[](1.0);
	float cone_angle_tan = 4; //~76 degrees
#else // MEDIUM QUALITY

#define MAX_CONE_DIRS 4

	vec3 cone_dirs[MAX_CONE_DIRS] = vec3[](
			vec3(0.707107, 0.0, 0.707107),
			vec3(0.0, 0.707107, 0.707107),
			vec3(-0.707107, 0.0, 0.707107),
			vec3(0.0, -0.707107, 0.707107));

	float cone_weights[MAX_CONE_DIRS] = float[](0.25, 0.25, 0.25, 0.25);
	float cone_angle_tan = 0.98269;

#endif
	vec3 light = vec3(0.0);

	for (int i = 0; i < MAX_CONE_DIRS; i++) {

		vec3 dir = normalize((gi_probes.data[index].xform * vec4(normal_xform * cone_dirs[i], 0.0)).xyz);

#if defined(GI_PROBE_HIGH_QUALITY) || defined(GI_PROBE_LOW_QUALITY)

#ifdef GI_PROBE_USE_ANISOTROPY
		vec4 cone_light = voxel_cone_trace_anisotropic(gi_probe_textures[gi_probes.data[index].texture_slot], gi_probe_textures[gi_probes.data[index].texture_slot + 1], gi_probe_textures[gi_probes.data[index].texture_slot + 2], normalize(mix(dir, normal, gi_probes.data[index].anisotropy_strength)), cell_size, position, dir, cone_angle_tan, max_distance, gi_probes.data[index].bias);
#else

		vec4 cone_light = voxel_cone_trace(gi_probe_textures[gi_probes.data[index].texture_slot], cell_size, position, dir, cone_angle_tan, max_distance, gi_probes.data[index].bias);

#endif // GI_PROBE_USE_ANISOTROPY

#else

#ifdef GI_PROBE_USE_ANISOTROPY
		vec4 cone_light = voxel_cone_trace_anisotropic_45_degrees(gi_probe_textures[gi_probes.data[index].texture_slot], gi_probe_textures[gi_probes.data[index].texture_slot + 1], gi_probe_textures[gi_probes.data[index].texture_slot + 2], normalize(mix(dir, normal, gi_probes.data[index].anisotropy_strength)), cell_size, position, dir, cone_angle_tan, max_distance, gi_probes.data[index].bias);
#else
		vec4 cone_light = voxel_cone_trace_45_degrees(gi_probe_textures[gi_probes.data[index].texture_slot], cell_size, position, dir, cone_angle_tan, max_distance, gi_probes.data[index].bias);
#endif // GI_PROBE_USE_ANISOTROPY

#endif
		if (gi_probes.data[index].blend_ambient) {
			cone_light.rgb = mix(ambient, cone_light.rgb, min(1.0, cone_light.a / 0.95));
		}

		light += cone_weights[i] * cone_light.rgb;
	}

	light *= gi_probes.data[index].dynamic_range;

	if (gi_probes.data[index].ambient_occlusion > 0.001) {

		float size = 1.0 + gi_probes.data[index].ambient_occlusion_size * 7.0;

		float taps, blend;
		blend = modf(size, taps);
		float ao = 0.0;
		for (float i = 1.0; i <= taps; i++) {
			vec3 ofs = (position + normal * (i * 0.5 + 1.0)) * cell_size;
			ao += textureLod(sampler3D(gi_probe_textures[gi_probes.data[index].texture_slot], material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), ofs, i - 1.0).a * i;
		}

		if (blend > 0.001) {
			vec3 ofs = (position + normal * ((taps + 1.0) * 0.5 + 1.0)) * cell_size;
			ao += textureLod(sampler3D(gi_probe_textures[gi_probes.data[index].texture_slot], material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), ofs, taps).a * (taps + 1.0) * blend;
		}

		ao = 1.0 - min(1.0, ao);

		light = mix(scene_data.ao_color.rgb, light, mix(1.0, ao, gi_probes.data[index].ambient_occlusion));
	}

	out_diff += vec4(light * blend, blend);

	//irradiance
#ifndef GI_PROBE_LOW_QUALITY
	vec4 irr_light = voxel_cone_trace(gi_probe_textures[gi_probes.data[index].texture_slot], cell_size, position, ref_vec, tan(roughness * 0.5 * M_PI * 0.99), max_distance, gi_probes.data[index].bias);
	if (gi_probes.data[index].blend_ambient) {
		irr_light.rgb = mix(environment, irr_light.rgb, min(1.0, irr_light.a / 0.95));
	}
	irr_light.rgb *= gi_probes.data[index].dynamic_range;
	//irr_light=vec3(0.0);

	out_spec += vec4(irr_light.rgb * blend, blend);
#endif
}

#endif //USE_VOXEL_CONE_TRACING

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

void main() {

#ifdef MODE_DUAL_PARABOLOID

	if (dp_clip > 0.0)
		discard;
#endif

	//lay out everything, whathever is unused is optimized away anyway
	vec3 vertex = vertex_interp;
	vec3 view = -normalize(vertex_interp);
	vec3 albedo = vec3(1.0);
	vec3 backlight = vec3(0.0);
	vec4 transmittance_color = vec4(0.0);
	float transmittance_depth = 0.0;
	float transmittance_curve = 1.0;
	float transmittance_boost = 0.0;
	float metallic = 0.0;
	float specular = 0.5;
	vec3 emission = vec3(0.0);
	float roughness = 1.0;
	float rim = 0.0;
	float rim_tint = 0.0;
	float clearcoat = 0.0;
	float clearcoat_gloss = 0.0;
	float anisotropy = 0.0;
	vec2 anisotropy_flow = vec2(1.0, 0.0);

#if defined(AO_USED)
	float ao = 1.0;
	float ao_light_affect = 0.0;
#endif

	float alpha = 1.0;

#if defined(ALPHA_SCISSOR_USED)
	float alpha_scissor = 0.5;
#endif

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	vec3 binormal = normalize(binormal_interp);
	vec3 tangent = normalize(tangent_interp);
#else
	vec3 binormal = vec3(0.0);
	vec3 tangent = vec3(0.0);
#endif
	vec3 normal = normalize(normal_interp);

#if defined(DO_SIDE_CHECK)
	if (!gl_FrontFacing) {
		normal = -normal;
	}
#endif

	vec2 uv = uv_interp;

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
	vec2 uv2 = uv2_interp;
#endif

#if defined(COLOR_USED)
	vec4 color = color_interp;
#endif

#if defined(NORMALMAP_USED)

	vec3 normalmap = vec3(0.5);
#endif

	float normaldepth = 1.0;

	vec2 screen_uv = gl_FragCoord.xy * scene_data.screen_pixel_size + scene_data.screen_pixel_size * 0.5; //account for center

	float sss_strength = 0.0;

	{
		/* clang-format off */

FRAGMENT_SHADER_CODE

		/* clang-format on */
	}

#if defined(LIGHT_TRANSMITTANCE_USED)
#ifdef SSS_MODE_SKIN
	transmittance_color.a = sss_strength;
#else
	transmittance_color.a *= sss_strength;
#endif
#endif

#if !defined(USE_SHADOW_TO_OPACITY)

#if defined(ALPHA_SCISSOR_USED)
	if (alpha < alpha_scissor) {
		discard;
	}
#endif // ALPHA_SCISSOR_USED

#ifdef USE_OPAQUE_PREPASS

	if (alpha < opaque_prepass_threshold) {
		discard;
	}

#endif // USE_OPAQUE_PREPASS

#endif // !USE_SHADOW_TO_OPACITY

#if defined(NORMALMAP_USED)

	normalmap.xy = normalmap.xy * 2.0 - 1.0;
	normalmap.z = sqrt(max(0.0, 1.0 - dot(normalmap.xy, normalmap.xy))); //always ignore Z, as it can be RG packed, Z may be pos/neg, etc.

	normal = normalize(mix(normal, tangent * normalmap.x + binormal * normalmap.y + normal * normalmap.z, normaldepth));

#endif

#if defined(LIGHT_ANISOTROPY_USED)

	if (anisotropy > 0.01) {
		//rotation matrix
		mat3 rot = mat3(tangent, binormal, normal);
		//make local to space
		tangent = normalize(rot * vec3(anisotropy_flow.x, anisotropy_flow.y, 0.0));
		binormal = normalize(rot * vec3(-anisotropy_flow.y, anisotropy_flow.x, 0.0));
	}

#endif

#ifdef ENABLE_CLIP_ALPHA
	if (albedo.a < 0.99) {
		//used for doublepass and shadowmapping
		discard;
	}
#endif
	/////////////////////// DECALS ////////////////////////////////

#ifndef MODE_RENDER_DEPTH

	uvec4 cluster_cell = texture(usampler3D(cluster_texture, material_samplers[SAMPLER_NEAREST_CLAMP]), vec3(screen_uv, (abs(vertex.z) - scene_data.z_near) / (scene_data.z_far - scene_data.z_near)));
	//used for interpolating anything cluster related
	vec3 vertex_ddx = dFdx(vertex);
	vec3 vertex_ddy = dFdy(vertex);

	{ // process decals

		uint decal_count = cluster_cell.w >> CLUSTER_COUNTER_SHIFT;
		uint decal_pointer = cluster_cell.w & CLUSTER_POINTER_MASK;

		//do outside for performance and avoiding arctifacts

		for (uint i = 0; i < decal_count; i++) {

			uint decal_index = cluster_data.indices[decal_pointer + i];
			if (!bool(decals.data[decal_index].mask & instances.data[instance_index].layer_mask)) {
				continue; //not masked
			}

			vec3 uv_local = (decals.data[decal_index].xform * vec4(vertex, 1.0)).xyz;
			if (any(lessThan(uv_local, vec3(0.0, -1.0, 0.0))) || any(greaterThan(uv_local, vec3(1.0)))) {
				continue; //out of decal
			}

			//we need ddx/ddy for mipmaps, so simulate them
			vec2 ddx = (decals.data[decal_index].xform * vec4(vertex_ddx, 0.0)).xz;
			vec2 ddy = (decals.data[decal_index].xform * vec4(vertex_ddy, 0.0)).xz;

			float fade = pow(1.0 - (uv_local.y > 0.0 ? uv_local.y : -uv_local.y), uv_local.y > 0.0 ? decals.data[decal_index].upper_fade : decals.data[decal_index].lower_fade);

			if (decals.data[decal_index].normal_fade > 0.0) {
				fade *= smoothstep(decals.data[decal_index].normal_fade, 1.0, dot(normal_interp, decals.data[decal_index].normal) * 0.5 + 0.5);
			}

			if (decals.data[decal_index].albedo_rect != vec4(0.0)) {
				//has albedo
				vec4 decal_albedo = textureGrad(sampler2D(decal_atlas_srgb, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uv_local.xz * decals.data[decal_index].albedo_rect.zw + decals.data[decal_index].albedo_rect.xy, ddx * decals.data[decal_index].albedo_rect.zw, ddy * decals.data[decal_index].albedo_rect.zw);
				decal_albedo *= decals.data[decal_index].modulate;
				decal_albedo.a *= fade;
				albedo = mix(albedo, decal_albedo.rgb, decal_albedo.a * decals.data[decal_index].albedo_mix);

				if (decals.data[decal_index].normal_rect != vec4(0.0)) {

					vec3 decal_normal = textureGrad(sampler2D(decal_atlas, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uv_local.xz * decals.data[decal_index].normal_rect.zw + decals.data[decal_index].normal_rect.xy, ddx * decals.data[decal_index].normal_rect.zw, ddy * decals.data[decal_index].normal_rect.zw).xyz;
					decal_normal.xy = decal_normal.xy * vec2(2.0, -2.0) - vec2(1.0, -1.0); //users prefer flipped y normal maps in most authoring software
					decal_normal.z = sqrt(max(0.0, 1.0 - dot(decal_normal.xy, decal_normal.xy)));
					//convert to view space, use xzy because y is up
					decal_normal = (decals.data[decal_index].normal_xform * decal_normal.xzy).xyz;

					normal = normalize(mix(normal, decal_normal, decal_albedo.a));
				}

				if (decals.data[decal_index].orm_rect != vec4(0.0)) {

					vec3 decal_orm = textureGrad(sampler2D(decal_atlas, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uv_local.xz * decals.data[decal_index].orm_rect.zw + decals.data[decal_index].orm_rect.xy, ddx * decals.data[decal_index].orm_rect.zw, ddy * decals.data[decal_index].orm_rect.zw).xyz;
#if defined(AO_USED)
					ao = mix(ao, decal_orm.r, decal_albedo.a);
#endif
					roughness = mix(roughness, decal_orm.g, decal_albedo.a);
					metallic = mix(metallic, decal_orm.b, decal_albedo.a);
				}
			}

			if (decals.data[decal_index].emission_rect != vec4(0.0)) {
				//emission is additive, so its independent from albedo
				emission += textureGrad(sampler2D(decal_atlas_srgb, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), uv_local.xz * decals.data[decal_index].emission_rect.zw + decals.data[decal_index].emission_rect.xy, ddx * decals.data[decal_index].emission_rect.zw, ddy * decals.data[decal_index].emission_rect.zw).xyz * decals.data[decal_index].emission_energy * fade;
			}
		}
	}

#endif //not render depth
	/////////////////////// LIGHTING //////////////////////////////

	//apply energy conservation

	vec3 specular_light = vec3(0.0, 0.0, 0.0);
	vec3 diffuse_light = vec3(0.0, 0.0, 0.0);
	vec3 ambient_light = vec3(0.0, 0.0, 0.0);

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

	if (scene_data.roughness_limiter_enabled) {
		float limit = texelFetch(sampler2D(roughness_buffer, material_samplers[SAMPLER_NEAREST_CLAMP]), ivec2(gl_FragCoord.xy), 0).r;
		roughness = max(roughness, limit);
	}

	if (scene_data.use_reflection_cubemap) {

		vec3 ref_vec = reflect(-view, normal);
		ref_vec = scene_data.radiance_inverse_xform * ref_vec;
#ifdef USE_RADIANCE_CUBEMAP_ARRAY

		float lod, blend;
		blend = modf(roughness * MAX_ROUGHNESS_LOD, lod);
		specular_light = texture(samplerCubeArray(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(ref_vec, lod)).rgb;
		specular_light = mix(specular_light, texture(samplerCubeArray(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(ref_vec, lod + 1)).rgb, blend);

#else
		specular_light = textureLod(samplerCube(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), ref_vec, roughness * MAX_ROUGHNESS_LOD).rgb;

#endif //USE_RADIANCE_CUBEMAP_ARRAY
		specular_light *= scene_data.ambient_light_color_energy.a;
	}

#ifndef USE_LIGHTMAP
	//lightmap overrides everything
	if (scene_data.use_ambient_light) {

		ambient_light = scene_data.ambient_light_color_energy.rgb;

		if (scene_data.use_ambient_cubemap) {
			vec3 ambient_dir = scene_data.radiance_inverse_xform * normal;
#ifdef USE_RADIANCE_CUBEMAP_ARRAY
			vec3 cubemap_ambient = texture(samplerCubeArray(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), vec4(ambient_dir, MAX_ROUGHNESS_LOD)).rgb;
#else
			vec3 cubemap_ambient = textureLod(samplerCube(radiance_cubemap, material_samplers[SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP]), ambient_dir, MAX_ROUGHNESS_LOD).rgb;
#endif //USE_RADIANCE_CUBEMAP_ARRAY

			ambient_light = mix(ambient_light, cubemap_ambient * scene_data.ambient_light_color_energy.a, scene_data.ambient_color_sky_mix);
		}
	}
#endif // USE_LIGHTMAP

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

	//radiance

	float specular_blob_intensity = 1.0;

#if defined(SPECULAR_TOON)
	specular_blob_intensity *= specular * 2.0;
#endif

#if !defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)
	//gi probes

	//lightmap

	//lightmap capture

#ifdef USE_VOXEL_CONE_TRACING
	{ // process giprobes
		uint index1 = instances.data[instance_index].gi_offset & 0xFFFF;
		if (index1 != 0xFFFF) {
			vec3 ref_vec = normalize(reflect(normalize(vertex), normal));
			//find arbitrary tangent and bitangent, then build a matrix
			vec3 v0 = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
			vec3 tangent = normalize(cross(v0, normal));
			vec3 bitangent = normalize(cross(tangent, normal));
			mat3 normal_mat = mat3(tangent, bitangent, normal);

			vec4 amb_accum = vec4(0.0);
			vec4 spec_accum = vec4(0.0);
			gi_probe_compute(index1, vertex, normal, ref_vec, normal_mat, roughness * roughness, ambient_light, specular_light, spec_accum, amb_accum);

			uint index2 = instances.data[instance_index].gi_offset >> 16;

			if (index2 != 0xFFFF) {
				gi_probe_compute(index2, vertex, normal, ref_vec, normal_mat, roughness * roughness, ambient_light, specular_light, spec_accum, amb_accum);
			}

			if (amb_accum.a > 0.0) {
				amb_accum.rgb /= amb_accum.a;
			}

			if (spec_accum.a > 0.0) {
				spec_accum.rgb /= spec_accum.a;
			}

			specular_light = spec_accum.rgb;
			ambient_light = amb_accum.rgb;
		}
	}
#endif

	{ // process reflections

		vec4 reflection_accum = vec4(0.0, 0.0, 0.0, 0.0);
		vec4 ambient_accum = vec4(0.0, 0.0, 0.0, 0.0);

		uint reflection_probe_count = cluster_cell.z >> CLUSTER_COUNTER_SHIFT;
		uint reflection_probe_pointer = cluster_cell.z & CLUSTER_POINTER_MASK;

		for (uint i = 0; i < reflection_probe_count; i++) {

			uint ref_index = cluster_data.indices[reflection_probe_pointer + i];
			reflection_process(ref_index, vertex, normal, roughness, ambient_light, specular_light, ambient_accum, reflection_accum);
		}

		if (reflection_accum.a > 0.0) {
			specular_light = reflection_accum.rgb / reflection_accum.a;
		}

#if !defined(USE_LIGHTMAP)
		if (ambient_accum.a > 0.0) {
			ambient_light = ambient_accum.rgb / ambient_accum.a;
		}
#endif
	}

	{

#if defined(DIFFUSE_TOON)
		//simplify for toon, as
		specular_light *= specular * metallic * albedo * 2.0;
#else

		// scales the specular reflections, needs to be be computed before lighting happens,
		// but after environment, GI, and reflection probes are added
		// Environment brdf approximation (Lazarov 2013)
		// see https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
		const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
		const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
		vec4 r = roughness * c0 + c1;
		float ndotv = clamp(dot(normal, view), 0.0, 1.0);
		float a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
		vec2 env = vec2(-1.04, 1.04) * a004 + r.zw;

		vec3 f0 = F0(metallic, specular, albedo);
		specular_light *= env.x * f0 + env.y;
#endif
	}

	{ //directional light

		for (uint i = 0; i < scene_data.directional_light_count; i++) {

			if (!bool(directional_lights.data[i].mask & instances.data[instance_index].layer_mask)) {
				continue; //not masked
			}

			vec3 shadow_attenuation = vec3(1.0);

#ifdef LIGHT_TRANSMITTANCE_USED
			float transmittance_z = transmittance_depth;
#endif

			if (directional_lights.data[i].shadow_enabled) {
				float depth_z = -vertex.z;

				vec4 pssm_coord;
				vec3 shadow_color = vec3(0.0);
				vec3 light_dir = directional_lights.data[i].direction;

#define BIAS_FUNC(m_var, m_idx)                                                                                                                                       \
	m_var.xyz += light_dir * directional_lights.data[i].shadow_bias[m_idx];                                                                                           \
	vec3 normal_bias = normalize(normal_interp) * (1.0 - max(0.0, dot(light_dir, -normalize(normal_interp)))) * directional_lights.data[i].shadow_normal_bias[m_idx]; \
	normal_bias -= light_dir * dot(light_dir, normal_bias);                                                                                                           \
	m_var.xyz += normal_bias;

				float shadow = 0.0;

				if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
					vec4 v = vec4(vertex, 1.0);

					BIAS_FUNC(v, 0)

					pssm_coord = (directional_lights.data[i].shadow_matrix1 * v);
					pssm_coord /= pssm_coord.w;

					if (directional_lights.data[i].softshadow_angle > 0) {
						float range_pos = dot(directional_lights.data[i].direction, v.xyz);
						float range_begin = directional_lights.data[i].shadow_range_begin.x;
						float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
						vec2 tex_scale = directional_lights.data[i].uv_scale1 * test_radius;
						shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
					} else {
						shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
					}

					shadow_color = directional_lights.data[i].shadow_color1.rgb;

#ifdef LIGHT_TRANSMITTANCE_USED
					{
						vec4 trans_vertex = vec4(vertex - normalize(normal_interp) * directional_lights.data[i].shadow_transmittance_bias.x, 1.0);
						vec4 trans_coord = directional_lights.data[i].shadow_matrix1 * trans_vertex;
						trans_coord /= trans_coord.w;

						float shadow_z = textureLod(sampler2D(directional_shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), trans_coord.xy, 0.0).r;
						shadow_z *= directional_lights.data[i].shadow_transmittance_z_scale.x;
						float z = trans_coord.z * directional_lights.data[i].shadow_transmittance_z_scale.x;

						transmittance_z = z - shadow_z;
					}
#endif
				} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {

					vec4 v = vec4(vertex, 1.0);

					BIAS_FUNC(v, 1)

					pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
					pssm_coord /= pssm_coord.w;

					if (directional_lights.data[i].softshadow_angle > 0) {
						float range_pos = dot(directional_lights.data[i].direction, v.xyz);
						float range_begin = directional_lights.data[i].shadow_range_begin.y;
						float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
						vec2 tex_scale = directional_lights.data[i].uv_scale2 * test_radius;
						shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
					} else {
						shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
					}

					shadow_color = directional_lights.data[i].shadow_color2.rgb;
#ifdef LIGHT_TRANSMITTANCE_USED
					{
						vec4 trans_vertex = vec4(vertex - normalize(normal_interp) * directional_lights.data[i].shadow_transmittance_bias.y, 1.0);
						vec4 trans_coord = directional_lights.data[i].shadow_matrix2 * trans_vertex;
						trans_coord /= trans_coord.w;

						float shadow_z = textureLod(sampler2D(directional_shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), trans_coord.xy, 0.0).r;
						shadow_z *= directional_lights.data[i].shadow_transmittance_z_scale.y;
						float z = trans_coord.z * directional_lights.data[i].shadow_transmittance_z_scale.y;

						transmittance_z = z - shadow_z;
					}
#endif
				} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {

					vec4 v = vec4(vertex, 1.0);

					BIAS_FUNC(v, 2)

					pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
					pssm_coord /= pssm_coord.w;

					if (directional_lights.data[i].softshadow_angle > 0) {
						float range_pos = dot(directional_lights.data[i].direction, v.xyz);
						float range_begin = directional_lights.data[i].shadow_range_begin.z;
						float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
						vec2 tex_scale = directional_lights.data[i].uv_scale3 * test_radius;
						shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
					} else {
						shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
					}

					shadow_color = directional_lights.data[i].shadow_color3.rgb;
#ifdef LIGHT_TRANSMITTANCE_USED
					{
						vec4 trans_vertex = vec4(vertex - normalize(normal_interp) * directional_lights.data[i].shadow_transmittance_bias.z, 1.0);
						vec4 trans_coord = directional_lights.data[i].shadow_matrix3 * trans_vertex;
						trans_coord /= trans_coord.w;

						float shadow_z = textureLod(sampler2D(directional_shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), trans_coord.xy, 0.0).r;
						shadow_z *= directional_lights.data[i].shadow_transmittance_z_scale.z;
						float z = trans_coord.z * directional_lights.data[i].shadow_transmittance_z_scale.z;

						transmittance_z = z - shadow_z;
					}
#endif

				} else {

					vec4 v = vec4(vertex, 1.0);

					BIAS_FUNC(v, 3)

					pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
					pssm_coord /= pssm_coord.w;

					if (directional_lights.data[i].softshadow_angle > 0) {
						float range_pos = dot(directional_lights.data[i].direction, v.xyz);
						float range_begin = directional_lights.data[i].shadow_range_begin.w;
						float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
						vec2 tex_scale = directional_lights.data[i].uv_scale4 * test_radius;
						shadow = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
					} else {
						shadow = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
					}

					shadow_color = directional_lights.data[i].shadow_color4.rgb;

#ifdef LIGHT_TRANSMITTANCE_USED
					{
						vec4 trans_vertex = vec4(vertex - normalize(normal_interp) * directional_lights.data[i].shadow_transmittance_bias.w, 1.0);
						vec4 trans_coord = directional_lights.data[i].shadow_matrix4 * trans_vertex;
						trans_coord /= trans_coord.w;

						float shadow_z = textureLod(sampler2D(directional_shadow_atlas, material_samplers[SAMPLER_LINEAR_CLAMP]), trans_coord.xy, 0.0).r;
						shadow_z *= directional_lights.data[i].shadow_transmittance_z_scale.w;
						float z = trans_coord.z * directional_lights.data[i].shadow_transmittance_z_scale.w;

						transmittance_z = z - shadow_z;
					}
#endif
				}

				if (directional_lights.data[i].blend_splits) {

					vec3 shadow_color_blend = vec3(0.0);
					float pssm_blend;
					float shadow2;

					if (depth_z < directional_lights.data[i].shadow_split_offsets.x) {
						vec4 v = vec4(vertex, 1.0);
						BIAS_FUNC(v, 1)
						pssm_coord = (directional_lights.data[i].shadow_matrix2 * v);
						pssm_coord /= pssm_coord.w;

						if (directional_lights.data[i].softshadow_angle > 0) {
							float range_pos = dot(directional_lights.data[i].direction, v.xyz);
							float range_begin = directional_lights.data[i].shadow_range_begin.y;
							float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
							vec2 tex_scale = directional_lights.data[i].uv_scale2 * test_radius;
							shadow2 = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
						} else {
							shadow2 = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
						}

						pssm_blend = smoothstep(0.0, directional_lights.data[i].shadow_split_offsets.x, depth_z);
						shadow_color_blend = directional_lights.data[i].shadow_color2.rgb;
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.y) {
						vec4 v = vec4(vertex, 1.0);
						BIAS_FUNC(v, 2)
						pssm_coord = (directional_lights.data[i].shadow_matrix3 * v);
						pssm_coord /= pssm_coord.w;

						if (directional_lights.data[i].softshadow_angle > 0) {
							float range_pos = dot(directional_lights.data[i].direction, v.xyz);
							float range_begin = directional_lights.data[i].shadow_range_begin.z;
							float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
							vec2 tex_scale = directional_lights.data[i].uv_scale3 * test_radius;
							shadow2 = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
						} else {
							shadow2 = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
						}

						pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.x, directional_lights.data[i].shadow_split_offsets.y, depth_z);

						shadow_color_blend = directional_lights.data[i].shadow_color3.rgb;
					} else if (depth_z < directional_lights.data[i].shadow_split_offsets.z) {
						vec4 v = vec4(vertex, 1.0);
						BIAS_FUNC(v, 3)
						pssm_coord = (directional_lights.data[i].shadow_matrix4 * v);
						pssm_coord /= pssm_coord.w;
						if (directional_lights.data[i].softshadow_angle > 0) {
							float range_pos = dot(directional_lights.data[i].direction, v.xyz);
							float range_begin = directional_lights.data[i].shadow_range_begin.w;
							float test_radius = (range_pos - range_begin) * directional_lights.data[i].softshadow_angle;
							vec2 tex_scale = directional_lights.data[i].uv_scale4 * test_radius;
							shadow2 = sample_directional_soft_shadow(directional_shadow_atlas, pssm_coord.xyz, tex_scale * directional_lights.data[i].soft_shadow_scale);
						} else {
							shadow2 = sample_directional_pcf_shadow(directional_shadow_atlas, scene_data.directional_shadow_pixel_size * directional_lights.data[i].soft_shadow_scale, pssm_coord);
						}

						pssm_blend = smoothstep(directional_lights.data[i].shadow_split_offsets.y, directional_lights.data[i].shadow_split_offsets.z, depth_z);
						shadow_color_blend = directional_lights.data[i].shadow_color4.rgb;
					} else {
						pssm_blend = 0.0; //if no blend, same coord will be used (divide by z will result in same value, and already cached)
					}

					pssm_blend = sqrt(pssm_blend);

					shadow = mix(shadow, shadow2, pssm_blend);
					shadow_color = mix(shadow_color, shadow_color_blend, pssm_blend);
				}

				shadow = mix(shadow, 1.0, smoothstep(directional_lights.data[i].fade_from, directional_lights.data[i].fade_to, vertex.z)); //done with negative values for performance

				shadow_attenuation = mix(shadow_color, vec3(1.0), shadow);

#undef BIAS_FUNC
			}

			light_compute(normal, directional_lights.data[i].direction, normalize(view), directional_lights.data[i].size, directional_lights.data[i].color * directional_lights.data[i].energy, 1.0, shadow_attenuation, albedo, roughness, metallic, specular, directional_lights.data[i].specular * specular_blob_intensity,
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
					rim, rim_tint,
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
	}

	{ //omni lights

		uint omni_light_count = cluster_cell.x >> CLUSTER_COUNTER_SHIFT;
		uint omni_light_pointer = cluster_cell.x & CLUSTER_POINTER_MASK;

		for (uint i = 0; i < omni_light_count; i++) {

			uint light_index = cluster_data.indices[omni_light_pointer + i];

			if (!bool(lights.data[light_index].mask & instances.data[instance_index].layer_mask)) {
				continue; //not masked
			}

			light_process_omni(light_index, vertex, view, normal, vertex_ddx, vertex_ddy, albedo, roughness, metallic, specular, specular_blob_intensity,
#ifdef LIGHT_BACKLIGHT_USED
					backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
					transmittance_color,
					transmittance_depth,
					transmittance_curve,
					transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
					rim,
					rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
					clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
					tangent, binormal, anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
					alpha,
#endif
					diffuse_light, specular_light);
		}
	}

	{ //spot lights
		uint spot_light_count = cluster_cell.y >> CLUSTER_COUNTER_SHIFT;
		uint spot_light_pointer = cluster_cell.y & CLUSTER_POINTER_MASK;

		for (uint i = 0; i < spot_light_count; i++) {

			uint light_index = cluster_data.indices[spot_light_pointer + i];

			if (!bool(lights.data[light_index].mask & instances.data[instance_index].layer_mask)) {
				continue; //not masked
			}

			light_process_spot(light_index, vertex, view, normal, vertex_ddx, vertex_ddy, albedo, roughness, metallic, specular, specular_blob_intensity,
#ifdef LIGHT_BACKLIGHT_USED
					backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
					transmittance_color,
					transmittance_depth,
					transmittance_curve,
					transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
					rim,
					rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
					clearcoat, clearcoat_gloss,
#endif
#ifdef LIGHT_ANISOTROPY_USED
					tangent, binormal, anisotropy,
#endif
#ifdef USE_SHADOW_TO_OPACITY
					alpha,
#endif
					diffuse_light, specular_light);
		}
	}

#ifdef USE_SHADOW_TO_OPACITY
	alpha = min(alpha, clamp(length(ambient_light), 0.0, 1.0));

#if defined(ALPHA_SCISSOR_USED)
	if (alpha < alpha_scissor) {
		discard;
	}
#endif // ALPHA_SCISSOR_USED

#ifdef USE_OPAQUE_PREPASS

	if (alpha < opaque_prepass_threshold) {
		discard;
	}

#endif // USE_OPAQUE_PREPASS

#endif // USE_SHADOW_TO_OPACITY

#endif //!defined(MODE_RENDER_DEPTH) && !defined(MODE_UNSHADED)

#ifdef MODE_RENDER_DEPTH

#ifdef MODE_RENDER_MATERIAL

	albedo_output_buffer.rgb = albedo;
	albedo_output_buffer.a = alpha;

	normal_output_buffer.rgb = normal * 0.5 + 0.5;
	normal_output_buffer.a = 0.0;
	depth_output_buffer.r = -vertex.z;

#if defined(AO_USED)
	orm_output_buffer.r = ao;
#else
	orm_output_buffer.r = 0.0;
#endif
	orm_output_buffer.g = roughness;
	orm_output_buffer.b = metallic;
	orm_output_buffer.a = sss_strength;

	emission_output_buffer.rgb = emission;
	emission_output_buffer.a = 0.0;
#endif

#ifdef MODE_RENDER_NORMAL
	normal_output_buffer = vec4(normal * 0.5 + 0.5, 0.0);
#ifdef MODE_RENDER_ROUGHNESS
	roughness_output_buffer = roughness;
#endif //MODE_RENDER_ROUGHNESS
#endif //MODE_RENDER_NORMAL

//nothing happens, so a tree-ssa optimizer will result in no fragment shader :)
#else

	specular_light *= scene_data.reflection_multiplier;
	ambient_light *= albedo; //ambient must be multiplied by albedo at the end

//ambient occlusion
#if defined(AO_USED)

	if (scene_data.ssao_enabled && scene_data.ssao_ao_affect > 0.0) {
		float ssao = texture(sampler2D(ao_buffer, material_samplers[SAMPLER_LINEAR_CLAMP]), screen_uv).r;
		ao = mix(ao, min(ao, ssao), scene_data.ssao_ao_affect);
		ao_light_affect = mix(ao_light_affect, max(ao_light_affect, scene_data.ssao_light_affect), scene_data.ssao_ao_affect);
	}

	ambient_light = mix(scene_data.ao_color.rgb, ambient_light, ao);
	ao_light_affect = mix(1.0, ao, ao_light_affect);
	specular_light = mix(scene_data.ao_color.rgb, specular_light, ao_light_affect);
	diffuse_light = mix(scene_data.ao_color.rgb, diffuse_light, ao_light_affect);

#else

	if (scene_data.ssao_enabled) {
		float ao = texture(sampler2D(ao_buffer, material_samplers[SAMPLER_LINEAR_CLAMP]), screen_uv).r;
		ambient_light = mix(scene_data.ao_color.rgb, ambient_light, ao);
		float ao_light_affect = mix(1.0, ao, scene_data.ssao_light_affect);
		specular_light = mix(scene_data.ao_color.rgb, specular_light, ao_light_affect);
		diffuse_light = mix(scene_data.ao_color.rgb, diffuse_light, ao_light_affect);
	}

#endif // AO_USED

	// base color remapping
	diffuse_light *= 1.0 - metallic; // TODO: avoid all diffuse and ambient light calculations when metallic == 1 up to this point
	ambient_light *= 1.0 - metallic;

	//fog

#ifdef MODE_MULTIPLE_RENDER_TARGETS

#ifdef MODE_UNSHADED
	diffuse_buffer = vec4(albedo.rgb, 0.0);
	specular_buffer = vec4(0.0);

#else

#ifdef SSS_MODE_SKIN
	sss_strength = -sss_strength;
#endif
	diffuse_buffer = vec4(emission + diffuse_light + ambient_light, sss_strength);
	specular_buffer = vec4(specular_light, metallic);

#endif

#else //MODE_MULTIPLE_RENDER_TARGETS

#ifdef MODE_UNSHADED
	frag_color = vec4(albedo, alpha);
#else
	frag_color = vec4(emission + ambient_light + diffuse_light + specular_light, alpha);
	//frag_color = vec4(1.0);

#endif //USE_NO_SHADING

#endif //MODE_MULTIPLE_RENDER_TARGETS

#endif //MODE_RENDER_DEPTH
}
