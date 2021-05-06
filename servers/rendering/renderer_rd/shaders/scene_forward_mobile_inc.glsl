#define M_PI 3.14159265359

#include "decal_data_inc.glsl"

#if !defined(MODE_RENDER_DEPTH) || defined(MODE_RENDER_MATERIAL) || defined(TANGENT_USED) || defined(NORMAL_MAP_USED)
#ifndef NORMAL_USED
#define NORMAL_USED
#endif
#endif

/* don't exceed 128 bytes!! */
/* put instance data into our push content, not a array */
layout(push_constant, binding = 0, std430) uniform DrawCall {
	mat4 transform; // 64 - 64
	uint flags; // 04 - 68
	uint instance_uniforms_ofs; //base offset in global buffer for instance variables	// 04 - 72
	uint gi_offset; //GI information when using lightmapping (VCT or lightmap index)    // 04 - 76
	uint layer_mask; // 04 - 80
	vec4 lightmap_uv_scale; // 16 - 96 doubles as uv_offset when needed

	uvec2 reflection_probes; // 08 - 104
	uvec2 omni_lights; // 08 - 112
	uvec2 spot_lights; // 08 - 120
	uvec2 decals; // 08 - 128
}
draw_call;

/* Set 0: Base Pass (never changes) */

#include "light_data_inc.glsl"

#define SAMPLER_NEAREST_CLAMP 0
#define SAMPLER_LINEAR_CLAMP 1
#define SAMPLER_NEAREST_WITH_MIPMAPS_CLAMP 2
#define SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP 3
#define SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_CLAMP 4
#define SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_CLAMP 5
#define SAMPLER_NEAREST_REPEAT 6
#define SAMPLER_LINEAR_REPEAT 7
#define SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT 8
#define SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT 9
#define SAMPLER_NEAREST_WITH_MIPMAPS_ANISOTROPIC_REPEAT 10
#define SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT 11

layout(set = 0, binding = 1) uniform sampler material_samplers[12];

layout(set = 0, binding = 2) uniform sampler shadow_sampler;

#define INSTANCE_FLAGS_USE_GI_BUFFERS (1 << 6)
#define INSTANCE_FLAGS_USE_SDFGI (1 << 7)
#define INSTANCE_FLAGS_USE_LIGHTMAP_CAPTURE (1 << 8)
#define INSTANCE_FLAGS_USE_LIGHTMAP (1 << 9)
#define INSTANCE_FLAGS_USE_SH_LIGHTMAP (1 << 10)
#define INSTANCE_FLAGS_USE_GIPROBE (1 << 11)
#define INSTANCE_FLAGS_MULTIMESH (1 << 12)
#define INSTANCE_FLAGS_MULTIMESH_FORMAT_2D (1 << 13)
#define INSTANCE_FLAGS_MULTIMESH_HAS_COLOR (1 << 14)
#define INSTANCE_FLAGS_MULTIMESH_HAS_CUSTOM_DATA (1 << 15)
#define INSTANCE_FLAGS_PARTICLE_TRAIL_SHIFT 16
//3 bits of stride
#define INSTANCE_FLAGS_PARTICLE_TRAIL_MASK 0xFF

#define INSTANCE_FLAGS_NON_UNIFORM_SCALE (1 << 24)

layout(set = 0, binding = 3, std430) restrict readonly buffer OmniLights {
	LightData data[];
}
omni_lights;

layout(set = 0, binding = 4, std430) restrict readonly buffer SpotLights {
	LightData data[];
}
spot_lights;

layout(set = 0, binding = 5, std430) restrict readonly buffer ReflectionProbeData {
	ReflectionData data[];
}
reflections;

layout(set = 0, binding = 6, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

#define LIGHTMAP_FLAG_USE_DIRECTION 1
#define LIGHTMAP_FLAG_USE_SPECULAR_DIRECTION 2

struct Lightmap {
	mat3 normal_xform;
};

layout(set = 0, binding = 7, std140) restrict readonly buffer Lightmaps {
	Lightmap data[];
}
lightmaps;

struct LightmapCapture {
	vec4 sh[9];
};

layout(set = 0, binding = 8, std140) restrict readonly buffer LightmapCaptures {
	LightmapCapture data[];
}
lightmap_captures;

layout(set = 0, binding = 9) uniform texture2D decal_atlas;
layout(set = 0, binding = 10) uniform texture2D decal_atlas_srgb;

layout(set = 0, binding = 11, std430) restrict readonly buffer Decals {
	DecalData data[];
}
decals;

layout(set = 0, binding = 12, std430) restrict readonly buffer GlobalVariableData {
	vec4 data[];
}
global_variables;

/* Set 1: Render Pass (changes per render pass) */

layout(set = 1, binding = 0, std140) uniform SceneData {
	mat4 projection_matrix;
	mat4 inv_projection_matrix;

	mat4 camera_matrix;
	mat4 inv_camera_matrix;

	vec2 viewport_size;
	vec2 screen_pixel_size;

	//use vec4s because std140 doesnt play nice with vec2s, z and w are wasted
	vec4 directional_penumbra_shadow_kernel[32];
	vec4 directional_soft_shadow_kernel[32];
	vec4 penumbra_shadow_kernel[32];
	vec4 soft_shadow_kernel[32];

	uint directional_penumbra_shadow_samples;
	uint directional_soft_shadow_samples;
	uint penumbra_shadow_samples;
	uint soft_shadow_samples;

	vec4 ambient_light_color_energy;

	float ambient_color_sky_mix;
	bool use_ambient_light;
	bool use_ambient_cubemap;
	bool use_reflection_cubemap;

	mat3 radiance_inverse_xform;

	vec2 shadow_atlas_pixel_size;
	vec2 directional_shadow_pixel_size;

	uint directional_light_count;
	float dual_paraboloid_side;
	float z_far;
	float z_near;

	bool ssao_enabled;
	float ssao_light_affect;
	float ssao_ao_affect;
	bool roughness_limiter_enabled;

	float roughness_limiter_amount;
	float roughness_limiter_limit;
	uvec2 roughness_limiter_pad;

	vec4 ao_color;

	bool fog_enabled;
	float fog_density;
	float fog_height;
	float fog_height_density;

	vec3 fog_light_color;
	float fog_sun_scatter;

	float fog_aerial_perspective;
	bool material_uv2_mode;

	float time;
	float reflection_multiplier; // one normally, zero when rendering reflections

	bool pancake_shadows;
	uint pad1;
	uint pad2;
	uint pad3;
}
scene_data;

#ifdef USE_RADIANCE_CUBEMAP_ARRAY

layout(set = 1, binding = 2) uniform textureCubeArray radiance_cubemap;

#else

layout(set = 1, binding = 2) uniform textureCube radiance_cubemap;

#endif

layout(set = 1, binding = 3) uniform textureCubeArray reflection_atlas;

layout(set = 1, binding = 4) uniform texture2D shadow_atlas;

layout(set = 1, binding = 5) uniform texture2D directional_shadow_atlas;

// this needs to change to providing just the lightmap we're using..
layout(set = 1, binding = 6) uniform texture2DArray lightmap_textures[MAX_LIGHTMAP_TEXTURES];

layout(set = 1, binding = 9) uniform texture2D depth_buffer;
layout(set = 1, binding = 10) uniform texture2D color_buffer;

/* Set 2 Skeleton & Instancing (can change per item) */

layout(set = 2, binding = 0, std430) restrict readonly buffer Transforms {
	vec4 data[];
}
transforms;

/* Set 3 User Material */
