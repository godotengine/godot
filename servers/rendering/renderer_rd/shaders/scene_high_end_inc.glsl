#define M_PI 3.14159265359
#define ROUGHNESS_MAX_LOD 5

#define MAX_GI_PROBES 8

#include "cluster_data_inc.glsl"

layout(push_constant, binding = 0, std430) uniform DrawCall {
	uint instance_index;
	uint pad; //16 bits minimum size
	vec2 bake_uv2_offset; //used for bake to uv2, ignored otherwise
}
draw_call;

/* Set 0 Scene data that never changes, ever */

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

#define SDFGI_MAX_CASCADES 8

layout(set = 0, binding = 3, std140) uniform SceneData {
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

	mat4 sdf_to_bounds;

	ivec3 sdf_offset;
	bool material_uv2_mode;

	ivec3 sdf_size;
	bool gi_upscale_for_msaa;

	bool volumetric_fog_enabled;
	float volumetric_fog_inv_length;
	float volumetric_fog_detail_spread;
	uint volumetric_fog_pad;

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
#define INSTANCE_FLAGS_MULTIMESH_STRIDE_SHIFT 16
//3 bits of stride
#define INSTANCE_FLAGS_MULTIMESH_STRIDE_MASK 0x7

#define INSTANCE_FLAGS_SKELETON (1 << 19)

struct InstanceData {
	mat4 transform;
	mat4 normal_transform;
	uint flags;
	uint instance_uniforms_ofs; //base offset in global buffer for instance variables
	uint gi_offset; //GI information when using lightmapping (VCT or lightmap index)
	uint layer_mask;
	vec4 lightmap_uv_scale;
};

layout(set = 0, binding = 4, std430) restrict readonly buffer Instances {
	InstanceData data[];
}
instances;

layout(set = 0, binding = 5, std430) restrict readonly buffer Lights {
	LightData data[];
}
lights;

layout(set = 0, binding = 6) buffer restrict readonly ReflectionProbeData {
	ReflectionData data[];
}
reflections;

layout(set = 0, binding = 7, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
}
directional_lights;

#define LIGHTMAP_FLAG_USE_DIRECTION 1
#define LIGHTMAP_FLAG_USE_SPECULAR_DIRECTION 2

struct Lightmap {
	mat3 normal_xform;
};

layout(set = 0, binding = 10, std140) restrict readonly buffer Lightmaps {
	Lightmap data[];
}
lightmaps;

layout(set = 0, binding = 11) uniform texture2DArray lightmap_textures[MAX_LIGHTMAP_TEXTURES];

struct LightmapCapture {
	vec4 sh[9];
};

layout(set = 0, binding = 12, std140) restrict readonly buffer LightmapCaptures {
	LightmapCapture data[];
}
lightmap_captures;

layout(set = 0, binding = 13) uniform texture2D decal_atlas;
layout(set = 0, binding = 14) uniform texture2D decal_atlas_srgb;

layout(set = 0, binding = 15, std430) restrict readonly buffer Decals {
	DecalData data[];
}
decals;

layout(set = 0, binding = 16) uniform utexture3D cluster_texture;

layout(set = 0, binding = 17, std430) restrict readonly buffer ClusterData {
	uint indices[];
}
cluster_data;

layout(set = 0, binding = 18) uniform texture2D directional_shadow_atlas;

layout(set = 0, binding = 19, std430) restrict readonly buffer GlobalVariableData {
	vec4 data[];
}
global_variables;

struct SDFGIProbeCascadeData {
	vec3 position;
	float to_probe;
	ivec3 probe_world_offset;
	float to_cell; // 1/bounds * grid_size
};

layout(set = 0, binding = 20, std140) uniform SDFGI {
	vec3 grid_size;
	uint max_cascades;

	bool use_occlusion;
	int probe_axis_size;
	float probe_to_uvw;
	float normal_bias;

	vec3 lightprobe_tex_pixel_size;
	float energy;

	vec3 lightprobe_uv_offset;
	float y_mult;

	vec3 occlusion_clamp;
	uint pad3;

	vec3 occlusion_renormalize;
	uint pad4;

	vec3 cascade_probe_size;
	uint pad5;

	SDFGIProbeCascadeData cascades[SDFGI_MAX_CASCADES];
}
sdfgi;

// decal atlas

/* Set 1, Radiance */

#ifdef USE_RADIANCE_CUBEMAP_ARRAY

layout(set = 1, binding = 0) uniform textureCubeArray radiance_cubemap;

#else

layout(set = 1, binding = 0) uniform textureCube radiance_cubemap;

#endif

/* Set 2, Reflection and Shadow Atlases (view dependent) */

layout(set = 2, binding = 0) uniform textureCubeArray reflection_atlas;

layout(set = 2, binding = 1) uniform texture2D shadow_atlas;

layout(set = 2, binding = 2) uniform texture3D gi_probe_textures[MAX_GI_PROBES];

/* Set 3, Render Buffers */

#ifdef MODE_RENDER_SDF

layout(r16ui, set = 3, binding = 0) uniform restrict writeonly uimage3D albedo_volume_grid;
layout(r32ui, set = 3, binding = 1) uniform restrict writeonly uimage3D emission_grid;
layout(r32ui, set = 3, binding = 2) uniform restrict writeonly uimage3D emission_aniso_grid;
layout(r32ui, set = 3, binding = 3) uniform restrict uimage3D geom_facing_grid;

//still need to be present for shaders that use it, so remap them to something
#define depth_buffer shadow_atlas
#define color_buffer shadow_atlas
#define normal_roughness_buffer shadow_atlas

#else

layout(set = 3, binding = 0) uniform texture2D depth_buffer;
layout(set = 3, binding = 1) uniform texture2D color_buffer;
layout(set = 3, binding = 2) uniform texture2D normal_roughness_buffer;
layout(set = 3, binding = 4) uniform texture2D ao_buffer;
layout(set = 3, binding = 5) uniform texture2D ambient_buffer;
layout(set = 3, binding = 6) uniform texture2D reflection_buffer;
layout(set = 3, binding = 7) uniform texture2DArray sdfgi_lightprobe_texture;
layout(set = 3, binding = 8) uniform texture3D sdfgi_occlusion_cascades;

struct GIProbeData {
	mat4 xform;
	vec3 bounds;
	float dynamic_range;

	float bias;
	float normal_bias;
	bool blend_ambient;
	uint texture_slot;

	float anisotropy_strength;
	float ambient_occlusion;
	float ambient_occlusion_size;
	uint mipmaps;
};

layout(set = 3, binding = 9, std140) uniform GIProbes {
	GIProbeData data[MAX_GI_PROBES];
}
gi_probes;

layout(set = 3, binding = 10) uniform texture3D volumetric_fog_texture;

#endif

/* Set 4 Skeleton & Instancing (Multimesh) */

layout(set = 4, binding = 0, std430) restrict readonly buffer Transforms {
	vec4 data[];
}
transforms;

/* Set 5 User Material */
