#define M_PI 3.14159265359
#define MAX_VIEWS 2

#if defined(USE_MULTIVIEW) && defined(has_VK_KHR_multiview)
#extension GL_EXT_multiview : enable
#endif

#include "../decal_data_inc.glsl"
#include "../scene_data_inc.glsl"

#if !defined(MODE_RENDER_DEPTH) || defined(MODE_RENDER_MATERIAL) || defined(TANGENT_USED) || defined(NORMAL_MAP_USED) || defined(LIGHT_ANISOTROPY_USED)
#ifndef NORMAL_USED
#define NORMAL_USED
#endif
#endif

#define USING_MOBILE_RENDERER

layout(push_constant, std430) uniform DrawCall {
	vec2 uv_offset;
	uint instance_index;
	uint pad;
}
draw_call;

/* Set 0: Base Pass (never changes) */

#include "../light_data_inc.glsl"

#include "../samplers_inc.glsl"

layout(set = 0, binding = 2) uniform sampler shadow_sampler;

layout(set = 0, binding = 3) uniform sampler decal_sampler;
layout(set = 0, binding = 4) uniform sampler light_projector_sampler;

#define INSTANCE_FLAGS_NON_UNIFORM_SCALE (1 << 4)
#define INSTANCE_FLAGS_USE_GI_BUFFERS (1 << 5)
#define INSTANCE_FLAGS_USE_SDFGI (1 << 6)
#define INSTANCE_FLAGS_USE_LIGHTMAP_CAPTURE (1 << 7)
#define INSTANCE_FLAGS_USE_LIGHTMAP (1 << 8)
#define INSTANCE_FLAGS_USE_SH_LIGHTMAP (1 << 9)
#define INSTANCE_FLAGS_USE_VOXEL_GI (1 << 10)
#define INSTANCE_FLAGS_PARTICLES (1 << 11)
#define INSTANCE_FLAGS_MULTIMESH (1 << 12)
#define INSTANCE_FLAGS_MULTIMESH_FORMAT_2D (1 << 13)
#define INSTANCE_FLAGS_MULTIMESH_HAS_COLOR (1 << 14)
#define INSTANCE_FLAGS_MULTIMESH_HAS_CUSTOM_DATA (1 << 15)
#define INSTANCE_FLAGS_PARTICLE_TRAIL_SHIFT 16
//3 bits of stride
#define INSTANCE_FLAGS_PARTICLE_TRAIL_MASK 0xFF

layout(set = 0, binding = 5, std430) restrict readonly buffer OmniLights {
	LightData data[];
}
omni_lights;

layout(set = 0, binding = 6, std430) restrict readonly buffer SpotLights {
	LightData data[];
}
spot_lights;

layout(set = 0, binding = 7, std430) restrict readonly buffer ReflectionProbeData {
	ReflectionData data[];
}
reflections;

layout(set = 0, binding = 8, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

#define LIGHTMAP_FLAG_USE_DIRECTION 1
#define LIGHTMAP_FLAG_USE_SPECULAR_DIRECTION 2

struct Lightmap {
	mediump mat3 normal_xform;
	vec3 pad;
	float exposure_normalization;
};

layout(set = 0, binding = 9, std140) restrict readonly buffer Lightmaps {
	Lightmap data[];
}
lightmaps;

struct LightmapCapture {
	mediump vec4 sh[9];
};

layout(set = 0, binding = 10, std140) restrict readonly buffer LightmapCaptures {
	LightmapCapture data[];
}
lightmap_captures;

layout(set = 0, binding = 11) uniform mediump texture2D decal_atlas;
layout(set = 0, binding = 12) uniform mediump texture2D decal_atlas_srgb;

layout(set = 0, binding = 13, std430) restrict readonly buffer Decals {
	DecalData data[];
}
decals;

layout(set = 0, binding = 14, std430) restrict readonly buffer GlobalShaderUniformData {
	highp vec4 data[];
}
global_shader_uniforms;

/* Set 1: Render Pass (changes per render pass) */

layout(set = 1, binding = 0, std140) uniform SceneDataBlock {
	SceneData data;
	SceneData prev_data;
}
scene_data_block;

struct InstanceData {
	highp mat4 transform; // 64 - 64
	uint flags; // 04 - 68
	uint instance_uniforms_ofs; // Base offset in global buffer for instance variables.	// 04 - 72
	uint gi_offset; // GI information when using lightmapping (VCT or lightmap index).    // 04 - 76
	uint layer_mask; // 04 - 80
	highp vec4 lightmap_uv_scale; // 16 - 96 Doubles as uv_offset when needed.

	uvec2 reflection_probes; // 08 - 104
	uvec2 omni_lights; // 08 - 112
	uvec2 spot_lights; // 08 - 120
	uvec2 decals; // 08 - 128

	vec4 compressed_aabb_position_pad; // 16 - 144 // Only .xyz is used. .w is padding.
	vec4 compressed_aabb_size_pad; // 16 - 160 // Only .xyz is used. .w is padding.
	vec4 uv_scale; // 16 - 176
};

layout(set = 1, binding = 1, std430) buffer restrict readonly InstanceDataBuffer {
	InstanceData data[];
}
instances;

#ifdef USE_RADIANCE_CUBEMAP_ARRAY

layout(set = 1, binding = 2) uniform mediump textureCubeArray radiance_cubemap;

#else

layout(set = 1, binding = 2) uniform mediump textureCube radiance_cubemap;

#endif

layout(set = 1, binding = 3) uniform mediump textureCubeArray reflection_atlas;

layout(set = 1, binding = 4) uniform highp texture2D shadow_atlas;

layout(set = 1, binding = 5) uniform highp texture2D directional_shadow_atlas;

// this needs to change to providing just the lightmap we're using..
layout(set = 1, binding = 6) uniform texture2DArray lightmap_textures[MAX_LIGHTMAP_TEXTURES];

#ifdef USE_MULTIVIEW
layout(set = 1, binding = 9) uniform highp texture2DArray depth_buffer;
layout(set = 1, binding = 10) uniform mediump texture2DArray color_buffer;
#define multiviewSampler sampler2DArray
#else
layout(set = 1, binding = 9) uniform highp texture2D depth_buffer;
layout(set = 1, binding = 10) uniform mediump texture2D color_buffer;
#define multiviewSampler sampler2D
#endif // USE_MULTIVIEW

/* Set 2 Skeleton & Instancing (can change per item) */

layout(set = 2, binding = 0, std430) restrict readonly buffer Transforms {
	highp vec4 data[];
}
transforms;

/* Set 3 User Material */
