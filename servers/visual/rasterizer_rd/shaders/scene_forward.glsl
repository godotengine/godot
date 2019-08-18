/* clang-format off */
[vertex]
/* clang-format on */

#version 450

/* clang-format off */
VERSION_DEFINES
/* clang-format on */

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

#if defined(UV_USED)
layout(location = 4) in vec2 uv_attrib;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 5) in vec2 uv2_attrib;
#endif

layout(location = 6) in uvec4 bone_attrib; // always bound, even if unused

/* Varyings */

out vec3 vertex_interp;
out vec3 normal_interp;

#if defined(COLOR_USED)
out vec4 color_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
out vec2 uv_interp;
#endif

//uv2 may be used for lightmapping, so always pass.
#if !defined(MODE_RENDER_DEPTH)
out vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)
out vec3 tangent_interp;
out vec3 binormal_interp;
#endif

#ifdef USE_MATERIAL_UNIFORMS
layout(set = 2, binding = 0, std140) uniform MaterialUniforms {
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
//invariant gl_Position;

void main() {

	vec3 vertex = vertex_attrib;

	mat4 world_matrix = instance_data.transform;
	mat3 world_normal_matrix= instance_data.normal_transform;

	vec3 normal = normal_attrib;

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	vec3 tangent = tangent_attrib.xyz;
	float binormalf = tangent_attrib.a;
#endif

#if defined(COLOR_USED)
	color_interp = color_attrib;
#endif

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	vec3 binormal = normalize(cross(normal, tangent) * binormalf);
#endif

#if defined(UV_USED)
	uv_interp = uv_attrib;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
	uv2_interp = uv2_attrib;
#endif

#ifdef USE_OVERRIDE_POSITION
	vec4 position;
#endif

	vec4 instance_custom = vec4(0.0);

	mat4 projection_matrix = scene_data.projection_matrix;

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (world_matrix * vec4(vertex,1.0)).xyz;

	normal = world_normal_matrix * normal;

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	tangent = world_normal_matrix * tangent;
	binormal = world_normal_matrix * binormal;

#endif
#endif

	float roughness = 1.0;

	mat4 modelview = scene_data.inv_camera_matrix * instance_data.transform;
	mat3 modelview_normal = mat3(scene_data.inv_camera_matrix) * instance_data.normal_transform;

	{
		/* clang-format off */

VERTEX_SHADER_CODE

		/* clang-format on */
	}

// using local coordinates (default)
#if !defined(SKIP_TRANSFORM_USED) && !defined(VERTEX_WORLD_COORDS_USED)

	vertex = (modelview * vec4(vertex,1.0)).xyz;
	normal = modelview_normal * normal;
#endif

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	binormal = modelview_normal * binormal;
	tangent = modelview_normal * tangent;
#endif
#endif

//using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (scene_data.inv_camera_matrix * vec4(vertex,1.0)).xyz;
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

	float z_ofs = scene_data.z_offset;
	z_ofs += (1.0 - abs(normal_interp.z)) * scene_data.z_slope_scale;
	vertex_interp.z -= z_ofs;

#endif //MODE_RENDER_DEPTH

#ifdef USE_OVERRIDE_POSITION
	gl_Position = position;
#else
	gl_Position = projection_matrix * vec4(vertex_interp, 1.0);
#endif

}

/* clang-format off */
[fragment]
/* clang-format on */

#version 450

/* clang-format off */
VERSION_DEFINES
/* clang-format on */


/* Varyings */

#if defined(COLOR_USED)
in vec4 color_interp;
#endif

#if defined(UV_USED)
in vec2 uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
in vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMALMAP_USED) || defined(LIGHT_ANISOTROPY_USED)
in vec3 tangent_interp;
in vec3 binormal_interp;
#endif

in highp vec3 vertex_interp;
in vec3 normal_interp;

//defines to keep compatibility with vertex

#define world_matrix instance_data.transform;
#define world_normal_matrix instance_data.normal_transform;
#define projection_matrix scene_data.projection_matrix;

#ifdef USE_MATERIAL_UNIFORMS
layout(set = 2, binding = 0, std140) uniform MaterialUniforms {
/* clang-format off */
MATERIAL_UNIFORMS
/* clang-format on */
} material;
#endif

/* clang-format off */

FRAGMENT_SHADER_GLOBALS

/* clang-format on */

#ifdef MODE_MULTIPLE_RENDER_TARGETS

layout(location = 0) out vec4 diffuse_buffer; //diffuse (rgb) and roughness
layout(location = 1) out vec4 specular_buffer; //specular and SSS (subsurface scatter)
#else

layout(location = 0) out vec4 frag_color;

#endif

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

void light_compute(vec3 N, vec3 L, vec3 V, vec3 B, vec3 T, vec3 light_color, vec3 attenuation, vec3 diffuse_color, vec3 transmission, float specular_blob_intensity, float roughness, float metallic, float specular, float rim, float rim_tint, float clearcoat, float clearcoat_gloss, float anisotropy, inout vec3 diffuse_light, inout vec3 specular_light, inout float alpha) {

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
	float NdotL = dot(N, L);
	float cNdotL = max(NdotL, 0.0); // clamped NdotL
	float NdotV = dot(N, V);
	float cNdotV = max(NdotV, 0.0);

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_BLINN) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_USE_CLEARCOAT)
	vec3 H = normalize(V + L);
#endif

#if defined(SPECULAR_BLINN) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_USE_CLEARCOAT)
	float cNdotH = max(dot(N, H), 0.0);
#endif

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_USE_CLEARCOAT)
	float cLdotH = max(dot(L, H), 0.0);
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

		diffuse_light += light_color * diffuse_color * diffuse_brdf_NL * attenuation;

#if defined(TRANSMISSION_USED)
		diffuse_light += light_color * diffuse_color * (vec3(1.0 / M_PI) - diffuse_brdf_NL) * transmission * attenuation;
#endif

#if defined(LIGHT_USE_RIM)
		float rim_light = pow(max(0.0, 1.0 - cNdotV), max(0.0, (1.0 - roughness) * 16.0));
		diffuse_light += rim_light * rim * mix(vec3(1.0), diffuse_color, rim_tint) * light_color;
#endif
	}

	if (roughness > 0.0) { // FIXME: roughness == 0 should not disable specular light entirely

		// D

#if defined(SPECULAR_BLINN)

		//normalized blinn
		float shininess = exp2(15.0 * (1.0 - roughness) + 1.0) * 0.25;
		float blinn = pow(cNdotH, shininess);
		blinn *= (shininess + 8.0) * (1.0 / (8.0 * M_PI));
		float intensity = (blinn) / max(4.0 * cNdotV * cNdotL, 0.75);

		specular_light += light_color * intensity * specular_blob_intensity * attenuation;

#elif defined(SPECULAR_PHONG)

		vec3 R = normalize(-reflect(L, N));
		float cRdotV = max(0.0, dot(R, V));
		float shininess = exp2(15.0 * (1.0 - roughness) + 1.0) * 0.25;
		float phong = pow(cRdotV, shininess);
		phong *= (shininess + 8.0) * (1.0 / (8.0 * M_PI));
		float intensity = (phong) / max(4.0 * cNdotV * cNdotL, 0.75);

		specular_light += light_color * intensity * specular_blob_intensity * attenuation;

#elif defined(SPECULAR_TOON)

		vec3 R = normalize(-reflect(L, N));
		float RdotV = dot(R, V);
		float mid = 1.0 - roughness;
		mid *= mid;
		float intensity = smoothstep(mid - roughness * 0.5, mid + roughness * 0.5, RdotV) * mid;
		diffuse_light += light_color * intensity * specular_blob_intensity * attenuation; // write to diffuse_light, as in toon shading you generally want no reflection

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

		specular_light += specular_brdf_NL * light_color * specular_blob_intensity * attenuation;
#endif

#if defined(LIGHT_USE_CLEARCOAT)

#if !defined(SPECULAR_SCHLICK_GGX)
		float cLdotH5 = SchlickFresnel(cLdotH);
#endif
		float Dr = GTR1(cNdotH, mix(.1, .001, clearcoat_gloss));
		float Fr = mix(.04, 1.0, cLdotH5);
		float Gr = G_GGX_2cos(cNdotL, .25) * G_GGX_2cos(cNdotV, .25);

		float clearcoat_specular_brdf_NL = 0.25 * clearcoat * Gr * Fr * Dr * cNdotL;

		specular_light += clearcoat_specular_brdf_NL * light_color * specular_blob_intensity * attenuation;
#endif
	}

#ifdef USE_SHADOW_TO_OPACITY
	alpha = min(alpha, clamp(1.0 - length(attenuation), 0.0, 1.0));
#endif

#endif //defined(USE_LIGHT_SHADER_CODE)
}


void main() {


	//lay out everything, whathever is unused is optimized away anyway
	vec3 vertex = vertex_interp;
	vec3 view = -normalize(vertex_interp);
	vec3 albedo = vec3(1.0);
	vec3 transmission = vec3(0.0);
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

#if defined(ENABLE_AO)
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

#if defined(UV_USED)
	vec2 uv = uv_interp;
#endif

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

#if defined(SCREEN_UV_USED)
	vec2 screen_uv = gl_FragCoord.xy * screen_pixel_size;
#endif

#if defined(SSS_USED)
	float sss_strength = 0.0;
#endif

#define
	{
		/* clang-format off */

FRAGMENT_SHADER_CODE

		/* clang-format on */
	}

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

	/////////////////////// LIGHTING //////////////////////////////

	//apply energy conservation

	vec3 specular_light = vec3(0.0, 0.0, 0.0);
	vec3 diffuse_light = vec3(0.0, 0.0, 0.0);
	vec3 ambient_light = vec3( 0.0, 0.0, 0.0);

	//radiance

	float specular_blob_intensity = 1.0;

#if defined(SPECULAR_TOON)
	specular_blob_intensity *= specular * 2.0;
#endif

	//gi probes

	//lightmap

	//lightmap capture

	//process reflections

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
		float ndotv = clamp(dot(normal, eye_vec), 0.0, 1.0);
		float a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
		vec2 env = vec2(-1.04, 1.04) * a004 + r.zw;

		vec3 f0 = F0(metallic, specular, albedo);
		specular_light *= env.x * f0 + env.y;
#endif
	}

	//directional light


	//process omni and spots

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


#ifdef MODE_RENDER_DEPTH
//nothing happens, so a tree-ssa optimizer will result in no fragment shader :)
#else

	specular_light *= reflection_multiplier;
	ambient_light *= albedo; //ambient must be multiplied by albedo at the end

#if defined(ENABLE_AO)
	ambient_light *= ao;
	ao_light_affect = mix(1.0, ao, ao_light_affect);
	specular_light *= ao_light_affect;
	diffuse_light *= ao_light_affect;
#endif

	// base color remapping
	diffuse_light *= 1.0 - metallic; // TODO: avoid all diffuse and ambient light calculations when metallic == 1 up to this point
	ambient_light *= 1.0 - metallic;

	//fog

#ifdef MODE_MULTIPLE_RENDER_TARGETS

#ifdef USE_NO_SHADING
	diffuse_buffer = vec4(albedo.rgb, 0.0);
	specular_buffer = vec4(0.0);

#else

	diffuse_buffer = vec4(emission + diffuse_light + ambient_light, sss_strenght);
	specular_buffer = vec4(specular_light, metallic);

#endif

#else //MODE_MULTIPLE_RENDER_TARGETS

#ifdef USE_NO_SHADING
	frag_color = vec4(albedo, alpha);
#else
	frag_color = vec4(emission + ambient_light + diffuse_light + specular_light, alpha);

#endif //USE_NO_SHADING

#endif //MODE_MULTIPLE_RENDER_TARGETS

#endif //MODE_RENDER_DEPTH
}
