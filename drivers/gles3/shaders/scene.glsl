/* clang-format off */
#[modes]

mode_color = 
mode_color_instancing = \n#define USE_INSTANCING
mode_depth = #define MODE_RENDER_DEPTH
mode_depth_instancing = #define MODE_RENDER_DEPTH \n#define USE_INSTANCING

#[specializations]

DISABLE_LIGHTMAP = false
DISABLE_LIGHT_DIRECTIONAL = false
DISABLE_LIGHT_OMNI = false
DISABLE_LIGHT_SPOT = false
DISABLE_FOG = false
USE_RADIANCE_MAP = true
USE_MULTIVIEW = false
RENDER_SHADOWS = false
RENDER_SHADOWS_LINEAR = false
SHADOW_MODE_PCF_5 = false
SHADOW_MODE_PCF_13 = false
LIGHT_USE_PSSM2 = false
LIGHT_USE_PSSM4 = false
LIGHT_USE_PSSM_BLEND = false
BASE_PASS = true
USE_ADDITIVE_LIGHTING = false
// We can only use one type of light per additive pass. This means that if USE_ADDITIVE_LIGHTING is defined, and
// these are false, we are doing a directional light pass.
ADDITIVE_OMNI = false
ADDITIVE_SPOT = false


#[vertex]

#define M_PI 3.14159265359

#define SHADER_IS_SRGB true

#include "stdlib_inc.glsl"

#if !defined(MODE_RENDER_DEPTH) || defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED) ||defined(LIGHT_CLEARCOAT_USED)
#ifndef NORMAL_USED
#define NORMAL_USED
#endif
#endif

#ifdef MODE_UNSHADED
#ifdef USE_ADDITIVE_LIGHTING
#undef USE_ADDITIVE_LIGHTING
#endif
#endif // MODE_UNSHADED

/*
from RenderingServer:
ARRAY_VERTEX = 0, // RGB32F or RGBA16
ARRAY_NORMAL = 1, // RG16 octahedral compression or RGBA16 normal + angle
ARRAY_TANGENT = 2, // RG16 octahedral compression, sign stored in sign of G
ARRAY_COLOR = 3, // RGBA8
ARRAY_TEX_UV = 4, // RG32F
ARRAY_TEX_UV2 = 5, // RG32F
ARRAY_CUSTOM0 = 6, // Depends on ArrayCustomFormat.
ARRAY_CUSTOM1 = 7,
ARRAY_CUSTOM2 = 8,
ARRAY_CUSTOM3 = 9,
ARRAY_BONES = 10, // RGBA16UI (x2 if 8 weights)
ARRAY_WEIGHTS = 11, // RGBA16UNORM (x2 if 8 weights)
*/

/* INPUT ATTRIBS */

// Always contains vertex position in XYZ, can contain tangent angle in W.
layout(location = 0) in highp vec4 vertex_angle_attrib;
/* clang-format on */

#ifdef NORMAL_USED
// Contains Normal/Axis in RG, can contain tangent in BA.
layout(location = 1) in vec4 axis_tangent_attrib;
#endif

// location 2 is unused.

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

vec3 oct_to_vec3(vec2 e) {
	vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
	float t = max(-v.z, 0.0);
	v.xy += t * -sign(v.xy);
	return normalize(v);
}

void axis_angle_to_tbn(vec3 axis, float angle, out vec3 tangent, out vec3 binormal, out vec3 normal) {
	float c = cos(angle);
	float s = sin(angle);
	vec3 omc_axis = (1.0 - c) * axis;
	vec3 s_axis = s * axis;
	tangent = omc_axis.xxx * axis + vec3(c, -s_axis.z, s_axis.y);
	binormal = omc_axis.yyy * axis + vec3(s_axis.z, c, -s_axis.x);
	normal = omc_axis.zzz * axis + vec3(-s_axis.y, s_axis.x, c);
}

#ifdef USE_INSTANCING
layout(location = 12) in highp vec4 instance_xform0;
layout(location = 13) in highp vec4 instance_xform1;
layout(location = 14) in highp vec4 instance_xform2;
layout(location = 15) in highp uvec4 instance_color_custom_data; // Color packed into xy, Custom data into zw.
#endif

layout(std140) uniform GlobalShaderUniformData { //ubo:1
	vec4 global_shader_uniforms[MAX_GLOBAL_SHADER_UNIFORMS];
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
	bool material_uv2_mode;
	float emissive_exposure_normalization;
	bool use_ambient_light;
	bool use_ambient_cubemap;
	bool use_reflection_cubemap;

	float fog_aerial_perspective;
	float time;

	mat3 radiance_inverse_xform;

	uint directional_light_count;
	float z_far;
	float z_near;
	float IBL_exposure_normalization;

	bool fog_enabled;
	float fog_density;
	float fog_height;
	float fog_height_density;

	vec3 fog_light_color;
	float fog_sun_scatter;

	float shadow_bias;
	float pad;
	uint camera_visible_layers;
	bool pancake_shadows;
}
scene_data;

#ifdef USE_ADDITIVE_LIGHTING

#if defined(ADDITIVE_OMNI) || defined(ADDITIVE_SPOT)
struct PositionalShadowData {
	highp mat4 shadow_matrix;
	highp vec3 light_position;
	highp float shadow_normal_bias;
	vec3 pad;
	highp float shadow_atlas_pixel_size;
};

layout(std140) uniform PositionalShadows { // ubo:9
	PositionalShadowData positional_shadows[MAX_LIGHT_DATA_STRUCTS];
};

uniform lowp uint positional_shadow_index;

#else // ADDITIVE_DIRECTIONAL

struct DirectionalShadowData {
	highp vec3 direction;
	highp float shadow_atlas_pixel_size;
	highp vec4 shadow_normal_bias;
	highp vec4 shadow_split_offsets;
	highp mat4 shadow_matrix1;
	highp mat4 shadow_matrix2;
	highp mat4 shadow_matrix3;
	highp mat4 shadow_matrix4;
	mediump float fade_from;
	mediump float fade_to;
	mediump vec2 pad;
};

layout(std140) uniform DirectionalShadows { // ubo:10
	DirectionalShadowData directional_shadows[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
};

uniform lowp uint directional_shadow_index;

#endif // !(defined(ADDITIVE_OMNI) || defined(ADDITIVE_SPOT))
#endif // USE_ADDITIVE_LIGHTING

#ifdef USE_MULTIVIEW
layout(std140) uniform MultiviewData { // ubo:8
	highp mat4 projection_matrix_view[MAX_VIEWS];
	highp mat4 inv_projection_matrix_view[MAX_VIEWS];
	highp vec4 eye_offset[MAX_VIEWS];
}
multiview_data;
#endif

uniform highp mat4 world_transform;
uniform highp vec3 compressed_aabb_position;
uniform highp vec3 compressed_aabb_size;
uniform highp vec4 uv_scale;

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

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
out vec2 uv2_interp;
#endif

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
out vec3 tangent_interp;
out vec3 binormal_interp;
#endif

#ifdef USE_ADDITIVE_LIGHTING
out highp vec4 shadow_coord;

#if defined(LIGHT_USE_PSSM2) || defined(LIGHT_USE_PSSM4)
out highp vec4 shadow_coord2;
#endif

#ifdef LIGHT_USE_PSSM4
out highp vec4 shadow_coord3;
out highp vec4 shadow_coord4;
#endif //LIGHT_USE_PSSM4
#endif

#ifdef MATERIAL_UNIFORMS_USED

/* clang-format off */
layout(std140) uniform MaterialUniforms { // ubo:3

#MATERIAL_UNIFORMS

};
/* clang-format on */

#endif

/* clang-format off */

#GLOBALS

/* clang-format on */
invariant gl_Position;

void main() {
	highp vec3 vertex = vertex_angle_attrib.xyz * compressed_aabb_size + compressed_aabb_position;

	highp mat4 model_matrix = world_transform;
#ifdef USE_INSTANCING
	highp mat4 m = mat4(instance_xform0, instance_xform1, instance_xform2, vec4(0.0, 0.0, 0.0, 1.0));
	model_matrix = model_matrix * transpose(m);
#endif

#ifdef NORMAL_USED
	vec3 normal = oct_to_vec3(axis_tangent_attrib.xy * 2.0 - 1.0);
#endif
	highp mat3 model_normal_matrix = mat3(model_matrix);

#if defined(NORMAL_USED) || defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	vec3 binormal;
	float binormal_sign;
	vec3 tangent;
	if (axis_tangent_attrib.z > 0.0 || axis_tangent_attrib.w < 1.0) {
		// Uncompressed format.
		vec2 signed_tangent_attrib = axis_tangent_attrib.zw * 2.0 - 1.0;
		tangent = oct_to_vec3(vec2(signed_tangent_attrib.x, abs(signed_tangent_attrib.y) * 2.0 - 1.0));
		binormal_sign = sign(signed_tangent_attrib.y);
		binormal = normalize(cross(normal, tangent) * binormal_sign);
	} else {
		// Compressed format.
		float angle = vertex_angle_attrib.w;
		binormal_sign = angle > 0.5 ? 1.0 : -1.0; // 0.5 does not exist in UNORM16, so values are either greater or smaller.
		angle = abs(angle * 2.0 - 1.0) * M_PI; // 0.5 is basically zero, allowing to encode both signs reliably.
		vec3 axis = normal;
		axis_angle_to_tbn(axis, angle, tangent, binormal, normal);
		binormal *= binormal_sign;
	}
#endif

#if defined(COLOR_USED)
	color_interp = color_attrib;
#ifdef USE_INSTANCING
	vec4 instance_color = vec4(unpackHalf2x16(instance_color_custom_data.x), unpackHalf2x16(instance_color_custom_data.y));
	color_interp *= instance_color;
#endif
#endif

#if defined(UV_USED)
	uv_interp = uv_attrib;
#endif

#if defined(UV2_USED) || defined(USE_LIGHTMAP)
	uv2_interp = uv2_attrib;
#endif

	if (uv_scale != vec4(0.0)) { // Compression enabled
#ifdef UV_USED
		uv_interp = (uv_interp - 0.5) * uv_scale.xy;
#endif
#if defined(UV2_USED) || defined(USE_LIGHTMAP)
		uv2_interp = (uv2_interp - 0.5) * uv_scale.zw;
#endif
	}

#if defined(OVERRIDE_POSITION)
	highp vec4 position;
#endif

#ifdef USE_MULTIVIEW
	mat4 projection_matrix = multiview_data.projection_matrix_view[ViewIndex];
	mat4 inv_projection_matrix = multiview_data.inv_projection_matrix_view[ViewIndex];
	vec3 eye_offset = multiview_data.eye_offset[ViewIndex].xyz;
#else
	mat4 projection_matrix = scene_data.projection_matrix;
	mat4 inv_projection_matrix = scene_data.inv_projection_matrix;
	vec3 eye_offset = vec3(0.0, 0.0, 0.0);
#endif //USE_MULTIVIEW

#ifdef USE_INSTANCING
	vec4 instance_custom = vec4(unpackHalf2x16(instance_color_custom_data.z), unpackHalf2x16(instance_color_custom_data.w));
#else
	vec4 instance_custom = vec4(0.0);
#endif

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

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)

	binormal = modelview_normal * binormal;
	tangent = modelview_normal * tangent;
#endif
#endif // !defined(SKIP_TRANSFORM_USED) && !defined(VERTEX_WORLD_COORDS_USED)

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

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
	tangent_interp = tangent;
	binormal_interp = binormal;
#endif

	// Calculate shadows.
#ifdef USE_ADDITIVE_LIGHTING
#if defined(ADDITIVE_OMNI) || defined(ADDITIVE_SPOT)
	// Apply normal bias at draw time to avoid issues with scaling non-fused geometry.
	vec3 light_rel_vec = positional_shadows[positional_shadow_index].light_position - vertex_interp;
	float light_length = length(light_rel_vec);
	float aNdotL = abs(dot(normalize(normal_interp), normalize(light_rel_vec)));
	vec3 normal_offset = (1.0 - aNdotL) * positional_shadows[positional_shadow_index].shadow_normal_bias * light_length * normal_interp;

#ifdef ADDITIVE_SPOT
	// Calculate coord here so we can take advantage of prefetch.
	shadow_coord = positional_shadows[positional_shadow_index].shadow_matrix * vec4(vertex_interp + normal_offset, 1.0);
#endif

#ifdef ADDITIVE_OMNI
	// Can't interpolate unit direction nicely, so forget about prefetch.
	shadow_coord = vec4(vertex_interp + normal_offset, 1.0);
#endif
#else // ADDITIVE_DIRECTIONAL
	vec3 base_normal_bias = normalize(normal_interp) * (1.0 - max(0.0, dot(directional_shadows[directional_shadow_index].direction, -normalize(normal_interp))));
	vec3 normal_offset = base_normal_bias * directional_shadows[directional_shadow_index].shadow_normal_bias.x;
	shadow_coord = directional_shadows[directional_shadow_index].shadow_matrix1 * vec4(vertex_interp + normal_offset, 1.0);

#if defined(LIGHT_USE_PSSM2) || defined(LIGHT_USE_PSSM4)
	normal_offset = base_normal_bias * directional_shadows[directional_shadow_index].shadow_normal_bias.y;
	shadow_coord2 = directional_shadows[directional_shadow_index].shadow_matrix2 * vec4(vertex_interp + normal_offset, 1.0);
#endif

#ifdef LIGHT_USE_PSSM4
	normal_offset = base_normal_bias * directional_shadows[directional_shadow_index].shadow_normal_bias.z;
	shadow_coord3 = directional_shadows[directional_shadow_index].shadow_matrix3 * vec4(vertex_interp + normal_offset, 1.0);
	normal_offset = base_normal_bias * directional_shadows[directional_shadow_index].shadow_normal_bias.w;
	shadow_coord4 = directional_shadows[directional_shadow_index].shadow_matrix4 * vec4(vertex_interp + normal_offset, 1.0);
#endif //LIGHT_USE_PSSM4

#endif // !(defined(ADDITIVE_OMNI) || defined(ADDITIVE_SPOT))
#endif // USE_ADDITIVE_LIGHTING

#if defined(RENDER_SHADOWS) && !defined(RENDER_SHADOWS_LINEAR)
	// This is an optimized version of normalize(vertex_interp) * scene_data.shadow_bias / length(vertex_interp).
	float light_length_sq = dot(vertex_interp, vertex_interp);
	vertex_interp += vertex_interp * scene_data.shadow_bias / light_length_sq;
#endif

#if defined(OVERRIDE_POSITION)
	gl_Position = position;
#else
	gl_Position = projection_matrix * vec4(vertex_interp, 1.0);
#endif
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

#ifdef MODE_UNSHADED
#ifdef USE_ADDITIVE_LIGHTING
#undef USE_ADDITIVE_LIGHTING
#endif
#endif // MODE_UNSHADED

#ifndef MODE_RENDER_DEPTH
#include "tonemap_inc.glsl"
#endif
#include "stdlib_inc.glsl"

/* texture unit usage, N is max_texture_unit-N

1-color correction // In tonemap_inc.glsl
2-radiance
3-shadow
5-screen
6-depth

*/

#define M_PI 3.14159265359
/* clang-format on */

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

#if defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
in vec3 tangent_interp;
in vec3 binormal_interp;
#endif

#ifdef NORMAL_USED
in vec3 normal_interp;
#endif

in highp vec3 vertex_interp;

#ifdef USE_ADDITIVE_LIGHTING
in highp vec4 shadow_coord;

#if defined(LIGHT_USE_PSSM2) || defined(LIGHT_USE_PSSM4)
in highp vec4 shadow_coord2;
#endif

#ifdef LIGHT_USE_PSSM4
in highp vec4 shadow_coord3;
in highp vec4 shadow_coord4;
#endif //LIGHT_USE_PSSM4
#endif

#ifdef USE_RADIANCE_MAP

#define RADIANCE_MAX_LOD 5.0

uniform samplerCube radiance_map; // texunit:-2

#endif

layout(std140) uniform GlobalShaderUniformData { //ubo:1
	vec4 global_shader_uniforms[MAX_GLOBAL_SHADER_UNIFORMS];
};

	/* Material Uniforms */

#ifdef MATERIAL_UNIFORMS_USED

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
	bool material_uv2_mode;
	float emissive_exposure_normalization;
	bool use_ambient_light;
	bool use_ambient_cubemap;
	bool use_reflection_cubemap;

	float fog_aerial_perspective;
	float time;

	mat3 radiance_inverse_xform;

	uint directional_light_count;
	float z_far;
	float z_near;
	float IBL_exposure_normalization;

	bool fog_enabled;
	float fog_density;
	float fog_height;
	float fog_height_density;

	vec3 fog_light_color;
	float fog_sun_scatter;

	float shadow_bias;
	float pad;
	uint camera_visible_layers;
	bool pancake_shadows;
}
scene_data;

#ifdef USE_MULTIVIEW
layout(std140) uniform MultiviewData { // ubo:8
	highp mat4 projection_matrix_view[MAX_VIEWS];
	highp mat4 inv_projection_matrix_view[MAX_VIEWS];
	highp vec4 eye_offset[MAX_VIEWS];
}
multiview_data;
#endif

/* clang-format off */

#GLOBALS

/* clang-format on */

#ifndef MODE_RENDER_DEPTH
// Directional light data.
#if !defined(DISABLE_LIGHT_DIRECTIONAL) || (!defined(ADDITIVE_OMNI) && !defined(ADDITIVE_SPOT))

struct DirectionalLightData {
	mediump vec3 direction;
	mediump float energy;
	mediump vec3 color;
	mediump float size;
	mediump vec2 pad;
	mediump float shadow_opacity;
	mediump float specular;
};

layout(std140) uniform DirectionalLights { // ubo:7
	DirectionalLightData directional_lights[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
};

#if defined(USE_ADDITIVE_LIGHTING) && (!defined(ADDITIVE_OMNI) && !defined(ADDITIVE_SPOT))
// Directional shadows can be in the base pass or in the additive passes
uniform highp sampler2DShadow directional_shadow_atlas; // texunit:-3
#endif // defined(USE_ADDITIVE_LIGHTING) && (!defined(ADDITIVE_OMNI) && !defined(ADDITIVE_SPOT))

#endif // !DISABLE_LIGHT_DIRECTIONAL

// Omni and spot light data.
#if !defined(DISABLE_LIGHT_OMNI) || !defined(DISABLE_LIGHT_SPOT) || defined(ADDITIVE_OMNI) || defined(ADDITIVE_SPOT)

struct LightData { // This structure needs to be as packed as possible.
	highp vec3 position;
	highp float inv_radius;

	mediump vec3 direction;
	highp float size;

	mediump vec3 color;
	mediump float attenuation;

	mediump float cone_attenuation;
	mediump float cone_angle;
	mediump float specular_amount;
	mediump float shadow_opacity;
};

#if !defined(DISABLE_LIGHT_OMNI) || defined(ADDITIVE_OMNI)
layout(std140) uniform OmniLightData { // ubo:5
	LightData omni_lights[MAX_LIGHT_DATA_STRUCTS];
};
#ifdef BASE_PASS
uniform uint omni_light_indices[MAX_FORWARD_LIGHTS];
uniform uint omni_light_count;
#endif // BASE_PASS
#endif // DISABLE_LIGHT_OMNI

#if !defined(DISABLE_LIGHT_SPOT) || defined(ADDITIVE_SPOT)
layout(std140) uniform SpotLightData { // ubo:6
	LightData spot_lights[MAX_LIGHT_DATA_STRUCTS];
};
#ifdef BASE_PASS
uniform uint spot_light_indices[MAX_FORWARD_LIGHTS];
uniform uint spot_light_count;
#endif // BASE_PASS
#endif // DISABLE_LIGHT_SPOT
#endif // !defined(DISABLE_LIGHT_OMNI) || !defined(DISABLE_LIGHT_SPOT)

#ifdef USE_ADDITIVE_LIGHTING
#ifdef ADDITIVE_OMNI
uniform highp samplerCubeShadow omni_shadow_texture; // texunit:-3
uniform lowp uint omni_light_index;
#endif
#ifdef ADDITIVE_SPOT
uniform highp sampler2DShadow spot_shadow_texture; // texunit:-3
uniform lowp uint spot_light_index;
#endif

#if defined(ADDITIVE_OMNI) || defined(ADDITIVE_SPOT)
struct PositionalShadowData {
	highp mat4 shadow_matrix;
	highp vec3 light_position;
	highp float shadow_normal_bias;
	vec3 pad;
	highp float shadow_atlas_pixel_size;
};

layout(std140) uniform PositionalShadows { // ubo:9
	PositionalShadowData positional_shadows[MAX_LIGHT_DATA_STRUCTS];
};

uniform lowp uint positional_shadow_index;
#else // ADDITIVE_DIRECTIONAL
struct DirectionalShadowData {
	highp vec3 direction;
	highp float shadow_atlas_pixel_size;
	highp vec4 shadow_normal_bias;
	highp vec4 shadow_split_offsets;
	highp mat4 shadow_matrix1;
	highp mat4 shadow_matrix2;
	highp mat4 shadow_matrix3;
	highp mat4 shadow_matrix4;
	mediump float fade_from;
	mediump float fade_to;
	mediump vec2 pad;
};

layout(std140) uniform DirectionalShadows { // ubo:10
	DirectionalShadowData directional_shadows[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
};

uniform lowp uint directional_shadow_index;
#endif // !(defined(ADDITIVE_OMNI) || defined(ADDITIVE_SPOT))

#if !defined(ADDITIVE_OMNI)
float sample_shadow(highp sampler2DShadow shadow, float shadow_pixel_size, vec4 pos) {
	float avg = textureProj(shadow, pos);
#ifdef SHADOW_MODE_PCF_13
	pos /= pos.w;
	avg += textureProj(shadow, vec4(pos.xy + vec2(shadow_pixel_size * 2.0, 0.0), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(-shadow_pixel_size * 2.0, 0.0), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(0.0, shadow_pixel_size * 2.0), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(0.0, -shadow_pixel_size * 2.0), pos.zw));

	// Early bail if distant samples are fully shaded (or none are shaded) to improve performance.
	if (avg <= 0.000001) {
		// None shaded at all.
		return 0.0;
	} else if (avg >= 4.999999) {
		// All fully shaded.
		return 1.0;
	}

	avg += textureProj(shadow, vec4(pos.xy + vec2(shadow_pixel_size, 0.0), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(-shadow_pixel_size, 0.0), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(0.0, shadow_pixel_size), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(0.0, -shadow_pixel_size), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(shadow_pixel_size, shadow_pixel_size), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(-shadow_pixel_size, shadow_pixel_size), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(shadow_pixel_size, -shadow_pixel_size), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(-shadow_pixel_size, -shadow_pixel_size), pos.zw));
	return avg * (1.0 / 13.0);
#endif

#ifdef SHADOW_MODE_PCF_5
	pos /= pos.w;
	avg += textureProj(shadow, vec4(pos.xy + vec2(shadow_pixel_size, 0.0), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(-shadow_pixel_size, 0.0), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(0.0, shadow_pixel_size), pos.zw));
	avg += textureProj(shadow, vec4(pos.xy + vec2(0.0, -shadow_pixel_size), pos.zw));
	return avg * (1.0 / 5.0);

#endif

	return avg;
}
#endif //!defined(ADDITIVE_OMNI)
#endif // USE_ADDITIVE_LIGHTING

#endif // !MODE_RENDER_DEPTH

#ifdef USE_MULTIVIEW
uniform highp sampler2DArray depth_buffer; // texunit:-6
uniform highp sampler2DArray color_buffer; // texunit:-5
vec3 multiview_uv(vec2 uv) {
	return vec3(uv, ViewIndex);
}
#else
uniform highp sampler2D depth_buffer; // texunit:-6
uniform highp sampler2D color_buffer; // texunit:-5
vec2 multiview_uv(vec2 uv) {
	return uv;
}
#endif

uniform highp mat4 world_transform;
uniform mediump float opaque_prepass_threshold;

layout(location = 0) out vec4 frag_color;

vec3 F0(float metallic, float specular, vec3 albedo) {
	float dielectric = 0.16 * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(vec3(dielectric), albedo, vec3(metallic));
}
#ifndef MODE_RENDER_DEPTH
#if !defined(DISABLE_LIGHT_DIRECTIONAL) || !defined(DISABLE_LIGHT_OMNI) || !defined(DISABLE_LIGHT_SPOT) || defined(USE_ADDITIVE_LIGHTING)

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

void light_compute(vec3 N, vec3 L, vec3 V, float A, vec3 light_color, bool is_directional, float attenuation, vec3 f0, float roughness, float metallic, float specular_amount, vec3 albedo, inout float alpha,
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

#if defined(USE_LIGHT_SHADER_CODE)
	// light is written by the light shader

	highp mat4 model_matrix = world_transform;
	mat4 projection_matrix = scene_data.projection_matrix;
	mat4 inv_projection_matrix = scene_data.inv_projection_matrix;

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
		}
#else
		// Lambert
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
#else
		float D = D_GGX(cNdotH, alpha_ggx);
		float G = V_GGX(cNdotL, cNdotV, alpha_ggx);
#endif // LIGHT_ANISOTROPY_USED
	   // F
		float cLdotH5 = SchlickFresnel(cLdotH);
		// Calculate Fresnel using cheap approximate specular occlusion term from Filament:
		// https://google.github.io/filament/Filament.html#lighting/occlusion/specularocclusion
		float f90 = clamp(50.0 * f0.g, 0.0, 1.0);
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

#endif // USE_LIGHT_SHADER_CODE
}

float get_omni_spot_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

#if !defined(DISABLE_LIGHT_OMNI) || defined(ADDITIVE_OMNI)
void light_process_omni(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 f0, float roughness, float metallic, float shadow, vec3 albedo, inout float alpha,
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
		inout vec3 diffuse_light, inout vec3 specular_light) {
	vec3 light_rel_vec = omni_lights[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float omni_attenuation = get_omni_spot_attenuation(light_length, omni_lights[idx].inv_radius, omni_lights[idx].attenuation);
	vec3 color = omni_lights[idx].color;
	float size_A = 0.0;

	if (omni_lights[idx].size > 0.0) {
		float t = omni_lights[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1.0 / sqrt(1.0 + t * t));
	}

	omni_attenuation *= shadow;

	light_compute(normal, normalize(light_rel_vec), eye_vec, size_A, color, false, omni_attenuation, f0, roughness, metallic, omni_lights[idx].specular_amount, albedo, alpha,
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
#endif // !DISABLE_LIGHT_OMNI

#if !defined(DISABLE_LIGHT_SPOT) || defined(ADDITIVE_SPOT)
void light_process_spot(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 f0, float roughness, float metallic, float shadow, vec3 albedo, inout float alpha,
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
	float spot_attenuation = get_omni_spot_attenuation(light_length, spot_lights[idx].inv_radius, spot_lights[idx].attenuation);
	vec3 spot_dir = spot_lights[idx].direction;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir), spot_lights[idx].cone_angle);
	float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - spot_lights[idx].cone_angle));
	spot_attenuation *= 1.0 - pow(spot_rim, spot_lights[idx].cone_attenuation);
	vec3 color = spot_lights[idx].color;

	float size_A = 0.0;

	if (spot_lights[idx].size > 0.0) {
		float t = spot_lights[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1.0 / sqrt(1.0 + t * t));
	}

	spot_attenuation *= shadow;

	light_compute(normal, normalize(light_rel_vec), eye_vec, size_A, color, false, spot_attenuation, f0, roughness, metallic, spot_lights[idx].specular_amount, albedo, alpha,
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
#endif // !defined(DISABLE_LIGHT_SPOT) || defined(ADDITIVE_SPOT)

#endif // !defined(DISABLE_LIGHT_DIRECTIONAL) || !defined(DISABLE_LIGHT_OMNI) || !defined(DISABLE_LIGHT_SPOT)

vec4 fog_process(vec3 vertex) {
	vec3 fog_color = scene_data.fog_light_color;

#ifdef USE_RADIANCE_MAP
/*
		if (scene_data.fog_aerial_perspective > 0.0) {
		vec3 sky_fog_color = vec3(0.0);
		vec3 cube_view = scene_data.radiance_inverse_xform * vertex;
		// mip_level always reads from the second mipmap and higher so the fog is always slightly blurred
		float mip_level = mix(1.0 / MAX_ROUGHNESS_LOD, 1.0, 1.0 - (abs(vertex.z) - scene_data.z_near) / (scene_data.z_far - scene_data.z_near));

		sky_fog_color = textureLod(radiance_map, cube_view, mip_level * RADIANCE_MAX_LOD).rgb;

		fog_color = mix(fog_color, sky_fog_color, scene_data.fog_aerial_perspective);
	}
	*/
#endif

#ifndef DISABLE_LIGHT_DIRECTIONAL
	if (scene_data.fog_sun_scatter > 0.001) {
		vec4 sun_scatter = vec4(0.0);
		float sun_total = 0.0;
		vec3 view = normalize(vertex);
		for (uint i = uint(0); i < scene_data.directional_light_count; i++) {
			vec3 light_color = directional_lights[i].color * directional_lights[i].energy;
			float light_amount = pow(max(dot(view, directional_lights[i].direction), 0.0), 8.0);
			fog_color += light_color * light_amount * scene_data.fog_sun_scatter;
		}
	}
#endif // !DISABLE_LIGHT_DIRECTIONAL

	float fog_amount = 1.0 - exp(min(0.0, -length(vertex) * scene_data.fog_density));

	if (abs(scene_data.fog_height_density) >= 0.0001) {
		float y = (scene_data.inv_view_matrix * vec4(vertex, 1.0)).y;

		float y_dist = y - scene_data.fog_height;

		float vfog_amount = 1.0 - exp(min(0.0, y_dist * scene_data.fog_height_density));

		fog_amount = max(vfog_amount, fog_amount);
	}

	return vec4(fog_color, fog_amount);
}

#endif // !MODE_RENDER_DEPTH

void main() {
	//lay out everything, whatever is unused is optimized away anyway
	vec3 vertex = vertex_interp;
#ifdef USE_MULTIVIEW
	vec3 eye_offset = multiview_data.eye_offset[ViewIndex].xyz;
	vec3 view = -normalize(vertex_interp - eye_offset);
	mat4 projection_matrix = multiview_data.projection_matrix_view[ViewIndex];
	mat4 inv_projection_matrix = multiview_data.inv_projection_matrix_view[ViewIndex];
#else
	vec3 eye_offset = vec3(0.0, 0.0, 0.0);
	vec3 view = -normalize(vertex_interp);
	mat4 projection_matrix = scene_data.projection_matrix;
	mat4 inv_projection_matrix = scene_data.inv_projection_matrix;
#endif
	highp mat4 model_matrix = world_transform;
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
#ifndef FOG_DISABLED
	vec4 fog = vec4(0.0);
#endif // !FOG_DISABLED
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

	vec2 screen_uv = gl_FragCoord.xy * scene_data.screen_pixel_size;

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
#else
#ifdef MODE_RENDER_DEPTH
#ifdef USE_OPAQUE_PREPASS

	if (alpha < opaque_prepass_threshold) {
		discard;
	}
#endif // USE_OPAQUE_PREPASS
#endif // MODE_RENDER_DEPTH
#endif // !ALPHA_SCISSOR_USED

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

#ifndef FOG_DISABLED
#ifndef CUSTOM_FOG_USED
#ifndef DISABLE_FOG
	// fog must be processed as early as possible and then packed.
	// to maximize VGPR usage

	if (scene_data.fog_enabled) {
		fog = fog_process(vertex);
	}
#endif // !DISABLE_FOG
#endif // !CUSTOM_FOG_USED

	uint fog_rg = packHalf2x16(fog.rg);
	uint fog_ba = packHalf2x16(fog.ba);
#endif // !FOG_DISABLED

	// Convert colors to linear
	albedo = srgb_to_linear(albedo);
	emission = srgb_to_linear(emission);
	// TODO Backlight and transmittance when used
#ifndef MODE_UNSHADED
	vec3 f0 = F0(metallic, specular, albedo);
	vec3 specular_light = vec3(0.0, 0.0, 0.0);
	vec3 diffuse_light = vec3(0.0, 0.0, 0.0);
	vec3 ambient_light = vec3(0.0, 0.0, 0.0);

#ifdef BASE_PASS
	/////////////////////// LIGHTING //////////////////////////////

	// IBL precalculations
	float ndotv = clamp(dot(normal, view), 0.0, 1.0);
	vec3 F = f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(1.0 - ndotv, 5.0);

#ifdef USE_RADIANCE_MAP
	if (scene_data.use_reflection_cubemap) {
#ifdef LIGHT_ANISOTROPY_USED
		// https://google.github.io/filament/Filament.html#lighting/imagebasedlights/anisotropy
		vec3 anisotropic_direction = anisotropy >= 0.0 ? binormal : tangent;
		vec3 anisotropic_tangent = cross(anisotropic_direction, view);
		vec3 anisotropic_normal = cross(anisotropic_tangent, anisotropic_direction);
		vec3 bent_normal = normalize(mix(normal, anisotropic_normal, abs(anisotropy) * clamp(5.0 * roughness, 0.0, 1.0)));
		vec3 ref_vec = reflect(-view, bent_normal);
#else
		vec3 ref_vec = reflect(-view, normal);
#endif
		ref_vec = mix(ref_vec, normal, roughness * roughness);
		float horizon = min(1.0 + dot(ref_vec, normal), 1.0);
		ref_vec = scene_data.radiance_inverse_xform * ref_vec;
		specular_light = textureLod(radiance_map, ref_vec, sqrt(roughness) * RADIANCE_MAX_LOD).rgb;
		specular_light = srgb_to_linear(specular_light);
		specular_light *= horizon * horizon;
		specular_light *= scene_data.ambient_light_color_energy.a;
	}
#endif

	// Calculate Reflection probes
	// Calculate Lightmaps

#if defined(CUSTOM_RADIANCE_USED)
	specular_light = mix(specular_light, custom_radiance.rgb, custom_radiance.a);
#endif // CUSTOM_RADIANCE_USED

#ifndef USE_LIGHTMAP
	//lightmap overrides everything
	if (scene_data.use_ambient_light) {
		ambient_light = scene_data.ambient_light_color_energy.rgb;
#ifdef USE_RADIANCE_MAP
		if (scene_data.use_ambient_cubemap) {
			vec3 ambient_dir = scene_data.radiance_inverse_xform * normal;
			vec3 cubemap_ambient = textureLod(radiance_map, ambient_dir, RADIANCE_MAX_LOD).rgb;
			cubemap_ambient = srgb_to_linear(cubemap_ambient);
			ambient_light = mix(ambient_light, cubemap_ambient * scene_data.ambient_light_color_energy.a, scene_data.ambient_color_sky_mix);
		}
#endif
	}
#endif // USE_LIGHTMAP

#if defined(CUSTOM_IRRADIANCE_USED)
	ambient_light = mix(ambient_light, custom_irradiance.rgb, custom_irradiance.a);
#endif // CUSTOM_IRRADIANCE_USED

	{
#if defined(AMBIENT_LIGHT_DISABLED)
		ambient_light = vec3(0.0, 0.0, 0.0);
#else
		ambient_light *= albedo.rgb;
		ambient_light *= ao;
#endif // AMBIENT_LIGHT_DISABLED
	}

	// convert ao to direct light ao
	ao = mix(1.0, ao, ao_light_affect);

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
		specular_light *= env.x * f0 + env.y * clamp(50.0 * f0.g, metallic, 1.0);
#endif
	}

#ifndef DISABLE_LIGHT_DIRECTIONAL
	for (uint i = uint(0); i < scene_data.directional_light_count; i++) {
		light_compute(normal, normalize(directional_lights[i].direction), normalize(view), directional_lights[i].size, directional_lights[i].color * directional_lights[i].energy, true, 1.0, f0, roughness, metallic, 1.0, albedo, alpha,
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
	}
#endif // !DISABLE_LIGHT_DIRECTIONAL

#ifndef DISABLE_LIGHT_OMNI
	for (uint i = 0u; i < MAX_FORWARD_LIGHTS; i++) {
		if (i >= omni_light_count) {
			break;
		}
		light_process_omni(omni_light_indices[i], vertex, view, normal, f0, roughness, metallic, 1.0, albedo, alpha,
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
				binormal, tangent, anisotropy,
#endif
				diffuse_light, specular_light);
	}
#endif // !DISABLE_LIGHT_OMNI

#ifndef DISABLE_LIGHT_SPOT
	for (uint i = 0u; i < MAX_FORWARD_LIGHTS; i++) {
		if (i >= spot_light_count) {
			break;
		}
		light_process_spot(spot_light_indices[i], vertex, view, normal, f0, roughness, metallic, 1.0, albedo, alpha,
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
#endif // !DISABLE_LIGHT_SPOT
#endif // BASE_PASS
#endif // !MODE_UNSHADED

#endif // !MODE_RENDER_DEPTH

#if defined(USE_SHADOW_TO_OPACITY)
	alpha = min(alpha, clamp(length(ambient_light), 0.0, 1.0));

#if defined(ALPHA_SCISSOR_USED)
	if (alpha < alpha_scissor) {
		discard;
	}
#else
#ifdef MODE_RENDER_DEPTH
#ifdef USE_OPAQUE_PREPASS

	if (alpha < opaque_prepass_threshold) {
		discard;
	}
#endif // USE_OPAQUE_PREPASS
#endif // MODE_RENDER_DEPTH
#endif // !ALPHA_SCISSOR_USED

#endif // USE_SHADOW_TO_OPACITY

#ifdef MODE_RENDER_DEPTH
#ifdef RENDER_SHADOWS_LINEAR
	// Linearize the depth buffer if rendering cubemap shadows.
	gl_FragDepth = (length(vertex) + scene_data.shadow_bias) / scene_data.z_far;
#endif

// Nothing happens, so a tree-ssa optimizer will result in no fragment shader :)
#else // !MODE_RENDER_DEPTH
#ifdef BASE_PASS
#ifdef MODE_UNSHADED
	frag_color = vec4(albedo, alpha);
#else

	diffuse_light *= albedo;

	diffuse_light *= 1.0 - metallic;
	ambient_light *= 1.0 - metallic;

	frag_color = vec4(diffuse_light + specular_light, alpha);
	frag_color.rgb += emission + ambient_light;
#endif //!MODE_UNSHADED

#ifndef FOG_DISABLED
	fog = vec4(unpackHalf2x16(fog_rg), unpackHalf2x16(fog_ba));

#ifndef DISABLE_FOG
	if (scene_data.fog_enabled) {
		frag_color.rgb = mix(frag_color.rgb, fog.rgb, fog.a);
	}
#endif // !DISABLE_FOG
#endif // !FOG_DISABLED

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
#else // !BASE_PASS
	frag_color = vec4(0.0, 0.0, 0.0, alpha);
#endif // !BASE_PASS

/* ADDITIVE LIGHTING PASS */
#ifdef USE_ADDITIVE_LIGHTING
	diffuse_light = vec3(0.0);
	specular_light = vec3(0.0);

#if !defined(ADDITIVE_OMNI) && !defined(ADDITIVE_SPOT)

// Orthogonal shadows
#if !defined(LIGHT_USE_PSSM2) && !defined(LIGHT_USE_PSSM4)
	float directional_shadow = sample_shadow(directional_shadow_atlas, directional_shadows[directional_shadow_index].shadow_atlas_pixel_size, shadow_coord);
#endif // !defined(LIGHT_USE_PSSM2) && !defined(LIGHT_USE_PSSM4)

// PSSM2 shadows
#ifdef LIGHT_USE_PSSM2
	float depth_z = -vertex.z;
	vec4 light_split_offsets = directional_shadows[directional_shadow_index].shadow_split_offsets;
	//take advantage of prefetch
	float shadow1 = sample_shadow(directional_shadow_atlas, directional_shadows[directional_shadow_index].shadow_atlas_pixel_size, shadow_coord);
	float shadow2 = sample_shadow(directional_shadow_atlas, directional_shadows[directional_shadow_index].shadow_atlas_pixel_size, shadow_coord2);
	float directional_shadow = 1.0;

	if (depth_z < light_split_offsets.y) {

#ifdef LIGHT_USE_PSSM_BLEND
		float directional_shadow2 = 1.0;
		float pssm_blend = 0.0;
		bool use_blend = true;
#endif
		if (depth_z < light_split_offsets.x) {
			directional_shadow = shadow1;

#ifdef LIGHT_USE_PSSM_BLEND
			directional_shadow2 = shadow2;
			pssm_blend = smoothstep(0.0, light_split_offsets.x, depth_z);
#endif
		} else {
			directional_shadow = shadow2;
#ifdef LIGHT_USE_PSSM_BLEND
			use_blend = false;
#endif
		}
#ifdef LIGHT_USE_PSSM_BLEND
		if (use_blend) {
			directional_shadow = mix(directional_shadow, directional_shadow2, pssm_blend);
		}
#endif
	}

#endif //LIGHT_USE_PSSM2
// PSSM4 shadows
#ifdef LIGHT_USE_PSSM4
	float depth_z = -vertex.z;
	vec4 light_split_offsets = directional_shadows[directional_shadow_index].shadow_split_offsets;

	float shadow1 = sample_shadow(directional_shadow_atlas, directional_shadows[directional_shadow_index].shadow_atlas_pixel_size, shadow_coord);
	float shadow2 = sample_shadow(directional_shadow_atlas, directional_shadows[directional_shadow_index].shadow_atlas_pixel_size, shadow_coord2);
	float shadow3 = sample_shadow(directional_shadow_atlas, directional_shadows[directional_shadow_index].shadow_atlas_pixel_size, shadow_coord3);
	float shadow4 = sample_shadow(directional_shadow_atlas, directional_shadows[directional_shadow_index].shadow_atlas_pixel_size, shadow_coord4);
	float directional_shadow = 1.0;

	if (depth_z < light_split_offsets.w) {

#ifdef LIGHT_USE_PSSM_BLEND
		float directional_shadow2 = 1.0;
		float pssm_blend = 0.0;
		bool use_blend = true;
#endif
		if (depth_z < light_split_offsets.y) {
			if (depth_z < light_split_offsets.x) {
				directional_shadow = shadow1;

#ifdef LIGHT_USE_PSSM_BLEND
				directional_shadow2 = shadow2;

				pssm_blend = smoothstep(0.0, light_split_offsets.x, depth_z);
#endif
			} else {
				directional_shadow = shadow2;

#ifdef LIGHT_USE_PSSM_BLEND
				directional_shadow2 = shadow3;

				pssm_blend = smoothstep(light_split_offsets.x, light_split_offsets.y, depth_z);
#endif
			}
		} else {
			if (depth_z < light_split_offsets.z) {
				directional_shadow = shadow3;

#if defined(LIGHT_USE_PSSM_BLEND)
				directional_shadow2 = shadow4;
				pssm_blend = smoothstep(light_split_offsets.y, light_split_offsets.z, depth_z);
#endif

			} else {
				directional_shadow = shadow4;

#if defined(LIGHT_USE_PSSM_BLEND)
				use_blend = false;
#endif
			}
		}
#if defined(LIGHT_USE_PSSM_BLEND)
		if (use_blend) {
			directional_shadow = mix(directional_shadow, directional_shadow2, pssm_blend);
		}
#endif
	}

#endif //LIGHT_USE_PSSM4
	directional_shadow = mix(directional_shadow, 1.0, smoothstep(directional_shadows[directional_shadow_index].fade_from, directional_shadows[directional_shadow_index].fade_to, vertex.z));
	directional_shadow = mix(1.0, directional_shadow, directional_lights[directional_shadow_index].shadow_opacity);

	light_compute(normal, normalize(directional_lights[directional_shadow_index].direction), normalize(view), directional_lights[directional_shadow_index].size, directional_lights[directional_shadow_index].color * directional_lights[directional_shadow_index].energy, true, directional_shadow, f0, roughness, metallic, 1.0, albedo, alpha,
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
#endif // !defined(ADDITIVE_OMNI) && !defined(ADDITIVE_SPOT)

#ifdef ADDITIVE_OMNI
	vec3 light_ray = ((positional_shadows[positional_shadow_index].shadow_matrix * vec4(shadow_coord.xyz, 1.0))).xyz;

	float omni_shadow = texture(omni_shadow_texture, vec4(light_ray, length(light_ray) * omni_lights[omni_light_index].inv_radius));
	omni_shadow = mix(1.0, omni_shadow, omni_lights[omni_light_index].shadow_opacity);

	light_process_omni(omni_light_index, vertex, view, normal, f0, roughness, metallic, omni_shadow, albedo, alpha,
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
			binormal, tangent, anisotropy,
#endif
			diffuse_light, specular_light);
#endif // ADDITIVE_OMNI

#ifdef ADDITIVE_SPOT
	float spot_shadow = sample_shadow(spot_shadow_texture, positional_shadows[positional_shadow_index].shadow_atlas_pixel_size, shadow_coord);
	spot_shadow = mix(1.0, spot_shadow, spot_lights[spot_light_index].shadow_opacity);

	light_process_spot(spot_light_index, vertex, view, normal, f0, roughness, metallic, spot_shadow, albedo, alpha,
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

#endif // ADDITIVE_SPOT

	diffuse_light *= albedo;
	diffuse_light *= 1.0 - metallic;
	vec3 additive_light_color = diffuse_light + specular_light;

#ifndef FOG_DISABLED
	fog = vec4(unpackHalf2x16(fog_rg), unpackHalf2x16(fog_ba));

#ifndef DISABLE_FOG
	if (scene_data.fog_enabled) {
		additive_light_color *= (1.0 - fog.a);
	}
#endif // !DISABLE_FOG
#endif // !FOG_DISABLED

	// Tonemap before writing as we are writing to an sRGB framebuffer
	additive_light_color *= exposure;
	additive_light_color = apply_tonemapping(additive_light_color, white);
	additive_light_color = linear_to_srgb(additive_light_color);

#ifdef USE_BCS
	additive_light_color = apply_bcs(additive_light_color, bcs);
#endif

#ifdef USE_COLOR_CORRECTION
	additive_light_color = apply_color_correction(additive_light_color, color_correction);
#endif

	frag_color.rgb += additive_light_color;
#endif // USE_ADDITIVE_LIGHTING

#endif //!MODE_RENDER_DEPTH
}
