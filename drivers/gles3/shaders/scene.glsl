/* clang-format off */
#[modes]

mode_color = #define BASE_PASS
mode_additive = #define USE_ADDITIVE_LIGHTING
mode_depth = #define MODE_RENDER_DEPTH

#[specializations]

USE_LIGHTMAP = false
USE_LIGHT_DIRECTIONAL = false
USE_LIGHT_POSITIONAL = false


#[vertex]

#define M_PI 3.14159265359

#define SHADER_IS_SRGB true

#include "stdlib_inc.glsl"

#if !defined(MODE_RENDER_DEPTH) || defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED) ||defined(LIGHT_CLEARCOAT_USED)
#ifndef NORMAL_USED
#define NORMAL_USED
#endif
#endif

/*
from RenderingServer:
ARRAY_VERTEX = 0, // RG32F or RGB32F (depending on 2D bit)
ARRAY_NORMAL = 1, // A2B10G10R10, A is ignored.
ARRAY_TANGENT = 2, // A2B10G10R10, A flips sign of binormal.
ARRAY_COLOR = 3, // RGBA8
ARRAY_TEX_UV = 4, // RG32F
ARRAY_TEX_UV2 = 5, // RG32F
ARRAY_CUSTOM0 = 6, // Depends on ArrayCustomFormat.
ARRAY_CUSTOM1 = 7,
ARRAY_CUSTOM2 = 8,
ARRAY_CUSTOM3 = 9,
ARRAY_BONES = 10, // RGBA16UI (x2 if 8 weights)
ARRAY_WEIGHTS = 11, // RGBA16UNORM (x2 if 8 weights)
ARRAY_INDEX = 12, // 16 or 32 bits depending on length > 0xFFFF.
ARRAY_MAX = 13
*/

/* INPUT ATTRIBS */

layout(location = 0) in highp vec3 vertex_attrib;
/* clang-format on */

#ifdef NORMAL_USED
layout(location = 1) in vec3 normal_attrib;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
layout(location = 2) in vec4 tangent_attrib;
#endif

#if defined(COLOR_USED)
layout(location = 3) in vec4 color_attrib;
#endif

#ifdef UV_USED
layout(location = 4) in vec2 uv_attrib;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
layout(location = 5) in vec2 uv2_attrib;
#endif

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

#if defined(BONES_USED)
layout(location = 10) in uvec4 bone_attrib;
#endif

#if defined(WEIGHTS_USED)
layout(location = 11) in vec4 weight_attrib;
#endif

layout(std140) uniform GlobalVariableData { //ubo:1
	vec4 global_variables[MAX_GLOBAL_VARIABLES];
};

layout(std140) uniform SceneData { // ubo:2
	highp mat4 projection_matrix;
	highp mat4 inv_projection_matrix;
	highp mat4 inv_view_matrix;
	highp mat4 view_matrix;

	vec2 viewport_size;
	vec2 screen_pixel_size;

	mediump vec4 ambient_light_color_energy;

	mediump float ambient_color_sky_mix;
	uint ambient_flags;
	bool material_uv2_mode;
	float opaque_prepass_threshold;
	//bool use_ambient_light;
	//bool use_ambient_cubemap;
	//bool use_reflection_cubemap;

	mat3 radiance_inverse_xform;

	uint directional_light_count;
	float z_far;
	float z_near;
	float pad;

	bool fog_enabled;
	float fog_density;
	float fog_height;
	float fog_height_density;

	vec3 fog_light_color;
	float fog_sun_scatter;

	float fog_aerial_perspective;

	float time;
	float reflection_multiplier; // one normally, zero when rendering reflections

	bool pancake_shadows;
}
scene_data;

uniform highp mat4 world_transform;

#ifdef USE_LIGHTMAP
uniform highp vec4 lightmap_uv_rect;
#endif

/* Varyings */

out highp vec3 vertex_interp;
#ifdef NORMAL_USED
out vec3 normal_interp;
#endif

#if defined(COLOR_USED)
out vec4 color_interp;
#endif

#if defined(UV_USED)
out vec2 uv_interp;
#endif

#if defined(UV2_USED)
out vec2 uv2_interp;
#else
#ifdef USE_LIGHTMAP
out vec2 uv2_interp;
#endif
#endif

#if defined(TANGENT_USED) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
out vec3 tangent_interp;
out vec3 binormal_interp;
#endif

#if defined(MATERIAL_UNIFORMS_USED)

/* clang-format off */
layout(std140) uniform MaterialUniforms { // ubo:3

#MATERIAL_UNIFORMS

};
/* clang-format on */

#endif

/* clang-format off */

#GLOBALS

/* clang-format on */

out highp vec4 position_interp;

invariant gl_Position;

void main() {
	highp vec3 vertex = vertex_attrib;

	highp mat4 model_matrix = world_transform;

#ifdef NORMAL_USED
	vec3 normal = normal_attrib * 2.0 - 1.0;
#endif
	highp mat3 model_normal_matrix = mat3(model_matrix);

#if defined(TANGENT_USED) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
	vec3 tangent;
	float binormalf;
	tangent = normal_tangent_attrib.xyz;
	binormalf = normal_tangent_attrib.a;
#endif

#if defined(COLOR_USED)
	color_interp = color_attrib;
#endif

#if defined(TANGENT_USED) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
	vec3 binormal = normalize(cross(normal, tangent) * binormalf);
#endif

#if defined(UV_USED)
	uv_interp = uv_attrib;
#endif

#ifdef USE_LIGHTMAP
	uv2_interp = lightmap_uv_rect.zw * uv2_attrib + lightmap_uv_rect.xy;
#else
#if defined(UV2_USED)
	uv2_interp = uv2_attrib;
#endif
#endif

#if defined(OVERRIDE_POSITION)
	highp vec4 position;
#endif
	highp mat4 projection_matrix = scene_data.projection_matrix;
	highp mat4 inv_projection_matrix = scene_data.inv_projection_matrix;

	vec4 instance_custom = vec4(0.0);

	// Using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (model_matrix * vec4(vertex, 1.0)).xyz;

#ifdef NORMAL_USED
	normal = model_normal_matrix * normal;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	tangent = model_normal_matrix * tangent;
	binormal = model_normal_matrix * binormal;

#endif
#endif

	float roughness = 1.0;

	highp mat4 modelview = scene_data.view_matrix * model_matrix;
	highp mat3 modelview_normal = mat3(scene_data.view_matrix) * model_normal_matrix;

	float point_size = 1.0;

	{
#CODE : VERTEX
	}

	gl_PointSize = point_size;

	// Using local coordinates (default)
#if !defined(SKIP_TRANSFORM_USED) && !defined(VERTEX_WORLD_COORDS_USED)

	vertex = (modelview * vec4(vertex, 1.0)).xyz;
#ifdef NORMAL_USED
	normal = modelview_normal * normal;
#endif

#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	binormal = modelview_normal * binormal;
	tangent = modelview_normal * tangent;
#endif

	// Using world coordinates
#if !defined(SKIP_TRANSFORM_USED) && defined(VERTEX_WORLD_COORDS_USED)

	vertex = (scene_data.view_matrix * vec4(vertex, 1.0)).xyz;
#ifdef NORMAL_USED
	normal = (scene_data.view_matrix * vec4(normal, 0.0)).xyz;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	binormal = (scene_data.view_matrix * vec4(binormal, 0.0)).xyz;
	tangent = (scene_data.view_matrix * vec4(tangent, 0.0)).xyz;
#endif
#endif

	vertex_interp = vertex;
#ifdef NORMAL_USED
	normal_interp = normal;
#endif

#if defined(TANGENT_USED) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
	tangent_interp = tangent;
	binormal_interp = binormal;
#endif

#if defined(OVERRIDE_POSITION)
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

	position_interp = gl_Position;
}

/* clang-format off */
#[fragment]


// Default to SPECULAR_SCHLICK_GGX.
#if !defined(SPECULAR_DISABLED) && !defined(SPECULAR_SCHLICK_GGX) && !defined(SPECULAR_TOON)
#define SPECULAR_SCHLICK_GGX
#endif

#if !defined(MODE_RENDER_DEPTH) || defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED) ||defined(LIGHT_CLEARCOAT_USED)
#ifndef NORMAL_USED
#define NORMAL_USED
#endif
#endif

#include "tonemap_inc.glsl"
#include "stdlib_inc.glsl"

/* texture unit usage, N is max_texture_unity-N

1-color correction // In tonemap_inc.glsl
2-radiance
3-directional_shadow
4-positional_shadow
5-screen
6-depth

*/

uniform highp mat4 world_transform;
/* clang-format on */

#define M_PI 3.14159265359
#define SHADER_IS_SRGB true

/* Varyings */

#if defined(COLOR_USED)
in vec4 color_interp;
#endif

#if defined(UV_USED)
in vec2 uv_interp;
#endif

#if defined(UV2_USED)
in vec2 uv2_interp;
#else
#ifdef USE_LIGHTMAP
in vec2 uv2_interp;
#endif
#endif

#if defined(TANGENT_USED) || defined(ENABLE_NORMALMAP) || defined(LIGHT_USE_ANISOTROPY)
in vec3 tangent_interp;
in vec3 binormal_interp;
#endif

#ifdef NORMAL_USED
in vec3 normal_interp;
#endif

in highp vec3 vertex_interp;

/* PBR CHANNELS */

#ifdef USE_RADIANCE_MAP

layout(std140) uniform Radiance { // ubo:4

	mat4 radiance_inverse_xform;
	float radiance_ambient_contribution;
};

#define RADIANCE_MAX_LOD 5.0

uniform sampler2D radiance_map; // texunit:-2

vec3 textureDualParaboloid(sampler2D p_tex, vec3 p_vec, float p_roughness) {
	vec3 norm = normalize(p_vec);
	norm.xy /= 1.0 + abs(norm.z);
	norm.xy = norm.xy * vec2(0.5, 0.25) + vec2(0.5, 0.25);
	if (norm.z > 0.0) {
		norm.y = 0.5 - norm.y + 0.5;
	}
	return textureLod(p_tex, norm.xy, p_roughness * RADIANCE_MAX_LOD).xyz;
}

#endif

layout(std140) uniform GlobalVariableData { //ubo:1
	vec4 global_variables[MAX_GLOBAL_VARIABLES];
};

	/* Material Uniforms */

#if defined(MATERIAL_UNIFORMS_USED)

/* clang-format off */
layout(std140) uniform MaterialUniforms { // ubo:3

#MATERIAL_UNIFORMS

};
/* clang-format on */

#endif

layout(std140) uniform SceneData { // ubo:2
	highp mat4 projection_matrix;
	highp mat4 inv_projection_matrix;
	highp mat4 inv_view_matrix;
	highp mat4 view_matrix;

	vec2 viewport_size;
	vec2 screen_pixel_size;

	mediump vec4 ambient_light_color_energy;

	mediump float ambient_color_sky_mix;
	uint ambient_flags;
	bool material_uv2_mode;
	float opaque_prepass_threshold;
	//bool use_ambient_light;
	//bool use_ambient_cubemap;
	//bool use_reflection_cubemap;

	mat3 radiance_inverse_xform;

	uint directional_light_count;
	float z_far;
	float z_near;
	float pad;

	bool fog_enabled;
	float fog_density;
	float fog_height;
	float fog_height_density;

	vec3 fog_light_color;
	float fog_sun_scatter;

	float fog_aerial_perspective;

	float time;
	float reflection_multiplier; // one normally, zero when rendering reflections

	bool pancake_shadows;
}
scene_data;

/* clang-format off */

#GLOBALS

/* clang-format on */

//directional light data

#ifdef USE_LIGHT_DIRECTIONAL

struct DirectionalLightData {
	mediump vec3 direction;
	mediump float energy;
	mediump vec3 color;
	mediump float size;
	mediump vec3 pad;
	mediump float specular;
};

#endif

// omni and spot
#ifdef USE_LIGHT_POSITIONAL
struct LightData { //this structure needs to be as packed as possible
	highp vec3 position;
	highp float inv_radius;

	mediump vec3 direction;
	highp float size;

	mediump vec3 color;
	mediump float attenuation;

	mediump float cone_attenuation;
	mediump float cone_angle;
	mediump float specular_amount;
	bool shadow_enabled;
};

layout(std140) uniform OmniLightData { // ubo:5

	LightData omni_lights[MAX_LIGHT_DATA_STRUCTS];
};

layout(std140) uniform SpotLightData { // ubo:6

	LightData spot_lights[MAX_LIGHT_DATA_STRUCTS];
};

uniform highp samplerCubeShadow positional_shadow; // texunit:-6

uniform int omni_light_indices[MAX_FORWARD_LIGHTS];
uniform int omni_light_count;

uniform int spot_light_indices[MAX_FORWARD_LIGHTS];
uniform int spot_light_count;

uniform int reflection_indices[MAX_FORWARD_LIGHTS];
uniform int reflection_count;

#endif

uniform highp sampler2D screen_texture; // texunit:-5
uniform highp sampler2D depth_buffer; // texunit:-6

layout(location = 0) out vec4 frag_color;

in highp vec4 position_interp;

vec3 F0(float metallic, float specular, vec3 albedo) {
	float dielectric = 0.16 * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(vec3(dielectric), albedo, vec3(metallic));
}

#if defined(USE_LIGHT_DIRECTIONAL) || defined(USE_LIGHT_POSITIONAL)
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

void light_compute(vec3 N, vec3 L, vec3 V, float A, vec3 light_color, float attenuation, vec3 f0, uint orms, float specular_amount, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
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

#if defined(USE_LIGHT_SHADER_CODE)
	// light is written by the light shader

	vec3 normal = N;
	vec3 light = L;
	vec3 view = V;

	/* clang-format off */


#CODE : LIGHT

	/* clang-format on */

#else
	float NdotL = min(A + dot(N, L), 1.0);
	float cNdotL = max(NdotL, 0.0); // clamped NdotL
	float NdotV = dot(N, V);
	float cNdotV = max(NdotV, 0.0);

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
		diffuse_light += rim_light * rim * mix(vec3(1.0), albedo, rim_tint) * light_color;
#endif
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
		vec3 F = mix(vec3(cLdotH5), vec3(1.0), f0);

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

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

void light_process_omni(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 f0, uint orms, float shadow, vec3 albedo, inout float alpha,
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
	vec3 light_rel_vec = omni_lights[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float omni_attenuation = get_omni_attenuation(light_length, omni_lights[idx].inv_radius, omni_lights[idx].attenuation);
	vec3 light_attenuation = vec3(omni_attenuation);
	vec3 color = omni_lights[idx].color;
	float size_A = 0.0;

	if (omni_lights.data[idx].size > 0.0) {
		float t = omni_lights[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}

	light_compute(normal, normalize(light_rel_vec), eye_vec, size_A, color, light_attenuation, f0, orms, omni_lights[idx].specular_amount, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
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

void light_process_spot(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 f0, uint orms, float shadow, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
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

	vec3 light_rel_vec = spot_lights[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float spot_attenuation = get_omni_attenuation(light_length, spot_lights[idx].inv_radius, spot_lights[idx].attenuation);
	vec3 spot_dir = spot_lights[idx].direction;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir), spot_lights[idx].cone_angle);
	float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - spot_lights[idx].cone_angle));
	spot_attenuation *= 1.0 - pow(spot_rim, spot_lights[idx].cone_attenuation);
	float light_attenuation = spot_attenuation;
	vec3 color = spot_lights[idx].color;

	float size_A = 0.0;

	if (spot_lights.data[idx].size > 0.0) {
		float t = spot_lights.data[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}

	light_compute(normal, normalize(light_rel_vec), eye_vec, size_A, color, light_attenuation, f0, orms, spot_lights[idx].specular_amount, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
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
#endif // defined(USE_LIGHT_DIRECTIONAL) || defined(USE_LIGHT_POSITIONAL)

void main() {
	//lay out everything, whatever is unused is optimized away anyway
	vec3 vertex = vertex_interp;
	vec3 view = -normalize(vertex_interp);
	vec3 albedo = vec3(1.0);
	vec3 backlight = vec3(0.0);
	vec4 transmittance_color = vec4(0.0, 0.0, 0.0, 1.0);
	float transmittance_depth = 0.0;
	float transmittance_boost = 0.0;
	float metallic = 0.0;
	float specular = 0.5;
	vec3 emission = vec3(0.0);
	float roughness = 1.0;
	float rim = 0.0;
	float rim_tint = 0.0;
	float clearcoat = 0.0;
	float clearcoat_roughness = 0.0;
	float anisotropy = 0.0;
	vec2 anisotropy_flow = vec2(1.0, 0.0);
	vec4 fog = vec4(0.0);
#if defined(CUSTOM_RADIANCE_USED)
	vec4 custom_radiance = vec4(0.0);
#endif
#if defined(CUSTOM_IRRADIANCE_USED)
	vec4 custom_irradiance = vec4(0.0);
#endif

	float ao = 1.0;
	float ao_light_affect = 0.0;

	float alpha = 1.0;

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	vec3 binormal = normalize(binormal_interp);
	vec3 tangent = normalize(tangent_interp);
#else
	vec3 binormal = vec3(0.0);
	vec3 tangent = vec3(0.0);
#endif

#ifdef NORMAL_USED
	vec3 normal = normalize(normal_interp);

#if defined(DO_SIDE_CHECK)
	if (!gl_FrontFacing) {
		normal = -normal;
	}
#endif

#endif //NORMAL_USED

#ifdef UV_USED
	vec2 uv = uv_interp;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
	vec2 uv2 = uv2_interp;
#endif

#if defined(COLOR_USED)
	vec4 color = color_interp;
#endif

#if defined(NORMAL_MAP_USED)

	vec3 normal_map = vec3(0.5);
#endif

	float normal_map_depth = 1.0;

	vec2 screen_uv = gl_FragCoord.xy * scene_data.screen_pixel_size + scene_data.screen_pixel_size * 0.5; //account for center

	float sss_strength = 0.0;

#ifdef ALPHA_SCISSOR_USED
	float alpha_scissor_threshold = 1.0;
#endif // ALPHA_SCISSOR_USED

#ifdef ALPHA_HASH_USED
	float alpha_hash_scale = 1.0;
#endif // ALPHA_HASH_USED

#ifdef ALPHA_ANTIALIASING_EDGE_USED
	float alpha_antialiasing_edge = 0.0;
	vec2 alpha_texture_coordinate = vec2(0.0, 0.0);
#endif // ALPHA_ANTIALIASING_EDGE_USED
	{
#CODE : FRAGMENT
	}

#ifndef USE_SHADOW_TO_OPACITY

#if defined(ALPHA_SCISSOR_USED)
	if (alpha < alpha_scissor_threshold) {
		discard;
	}
#endif // ALPHA_SCISSOR_USED

#ifdef USE_OPAQUE_PREPASS
#if !defined(ALPHA_SCISSOR_USED)

	if (alpha < scene_data.opaque_prepass_threshold) {
		discard;
	}

#endif // not ALPHA_SCISSOR_USED
#endif // USE_OPAQUE_PREPASS

#endif // !USE_SHADOW_TO_OPACITY

#ifdef NORMAL_MAP_USED

	normal_map.xy = normal_map.xy * 2.0 - 1.0;
	normal_map.z = sqrt(max(0.0, 1.0 - dot(normal_map.xy, normal_map.xy))); //always ignore Z, as it can be RG packed, Z may be pos/neg, etc.

	normal = normalize(mix(normal, tangent * normal_map.x + binormal * normal_map.y + normal * normal_map.z, normal_map_depth));

#endif

#ifdef LIGHT_ANISOTROPY_USED

	if (anisotropy > 0.01) {
		//rotation matrix
		mat3 rot = mat3(tangent, binormal, normal);
		//make local to space
		tangent = normalize(rot * vec3(anisotropy_flow.x, anisotropy_flow.y, 0.0));
		binormal = normalize(rot * vec3(-anisotropy_flow.y, anisotropy_flow.x, 0.0));
	}

#endif

#ifndef MODE_RENDER_DEPTH
	vec3 f0 = F0(metallic, specular, albedo);
	// Convert albedo to linear. Approximation from: http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
	albedo = albedo * (albedo * (albedo * 0.305306011 + 0.682171111) + 0.012522878);
	vec3 specular_light = vec3(0.0, 0.0, 0.0);
	vec3 diffuse_light = vec3(0.0, 0.0, 0.0);
	vec3 ambient_light = vec3(0.0, 0.0, 0.0);

#ifdef BASE_PASS
	/////////////////////// LIGHTING //////////////////////////////

	// IBL precalculations
	float ndotv = clamp(dot(normal, view), 0.0, 1.0);
	vec3 F = f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(1.0 - ndotv, 5.0);

	// Calculate IBL
	// Calculate Reflection probes
	// Caclculate Lightmaps

	float specular_blob_intensity = 1.0;

#if defined(SPECULAR_TOON)
	specular_blob_intensity *= specular * 2.0;
#endif

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
		specular_light *= env.x * f0 + env.y;
#endif
	}

#endif // BASE_PASS

	//this saves some VGPRs
	uint orms = packUnorm4x8(vec4(ao, roughness, metallic, specular));

#ifdef USE_LIGHT_DIRECTIONAL

	float size_A = directional_lights[i].size;

	light_compute(normal, directional_lights[i].direction, normalize(view), size_A, directional_lights[i].color * directional_lights[i].energy, shadow, f0, orms, 1.0, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_RIM_USED
			rim, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, normalize(normal_interp),
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal,
			tangent, anisotropy,
#endif
			diffuse_light,
			specular_light);

#endif //#USE_LIGHT_DIRECTIONAL

#ifdef USE_LIGHT_POSITIONAL
	float shadow = 0.0;
	for (int i = 0; i < omni_light_count; i++) {
		light_process_omni(omni_light_indices[i], vertex, view, normal, f0, orms, shadow, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
				backlight,
#endif
#ifdef LIGHT_RIM_USED
				rim,
				rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
				clearcoat, clearcoat_roughness, normalize(normal_interp),
#endif
#ifdef LIGHT_ANISOTROPY_USED
				tangent, binormal, anisotropy,
#endif
				diffuse_light, specular_light);
	}

	for (int i = 0; i < spot_light_count; i++) {
		light_process_spot(spot_light_indices[i], vertex, view, normal, f0, orms, shadow, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
				backlight,
#endif
#ifdef LIGHT_RIM_USED
				rim,
				rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
				clearcoat, clearcoat_roughness, normalize(normal_interp),
#endif
#ifdef LIGHT_ANISOTROPY_USED
				tangent,
				binormal, anisotropy,
#endif
				diffuse_light, specular_light);
	}

#endif // USE_LIGHT_POSITIONAL
#endif //!MODE_RENDER_DEPTH

#if defined(USE_SHADOW_TO_OPACITY)
	alpha = min(alpha, clamp(length(ambient_light), 0.0, 1.0));

#if defined(ALPHA_SCISSOR_USED)
	if (alpha < alpha_scissor) {
		discard;
	}
#endif // ALPHA_SCISSOR_USED

#ifdef USE_OPAQUE_PREPASS
#if !defined(ALPHA_SCISSOR_USED)

	if (alpha < opaque_prepass_threshold) {
		discard;
	}

#endif // not ALPHA_SCISSOR_USED
#endif // USE_OPAQUE_PREPASS

#endif // USE_SHADOW_TO_OPACITY

#ifdef MODE_RENDER_DEPTH
//nothing happens, so a tree-ssa optimizer will result in no fragment shader :)
#else // !MODE_RENDER_DEPTH

	specular_light *= scene_data.reflection_multiplier;
	ambient_light *= albedo; //ambient must be multiplied by albedo at the end

	// base color remapping
	diffuse_light *= 1.0 - metallic;
	ambient_light *= 1.0 - metallic;

#ifdef MODE_UNSHADED
	frag_color = vec4(albedo, alpha);
#else
	frag_color = vec4(ambient_light + diffuse_light + specular_light, alpha);
#ifdef BASE_PASS
	frag_color.rgb += emission;
#endif
#endif //MODE_UNSHADED

	// Tonemap before writing as we are writing to an sRGB framebuffer
	frag_color.rgb *= exposure;
	frag_color.rgb = apply_tonemapping(frag_color.rgb, white);
	frag_color.rgb = linear_to_srgb(frag_color.rgb);

#ifdef USE_BCS
	frag_color.rgb = apply_bcs(frag_color.rgb, bcs);
#endif

#ifdef USE_COLOR_CORRECTION
	frag_color.rgb = apply_color_correction(frag_color.rgb, color_correction);
#endif

#endif //!MODE_RENDER_DEPTH
}
