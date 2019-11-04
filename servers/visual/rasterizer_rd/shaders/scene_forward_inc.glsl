
#define M_PI 3.14159265359
#define ROUGHNESS_MAX_LOD 5

layout(push_constant, binding = 0, std430) uniform DrawCall {
	uint instance_index;
	uint pad[3]; //16 bits minimum size
} draw_call;



/* Set 0 Scene data that never changes, ever */


#define SAMPLER_NEAREST_CLAMP 0
#define SAMPLER_LINEAR_CLAMP 1
#define SAMPLER_NEAREST_WITH_MIMPAMPS_CLAMP 2
#define SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP 3
#define SAMPLER_NEAREST_WITH_MIMPAMPS_ANISOTROPIC_CLAMP 4
#define SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_CLAMP 5
#define SAMPLER_NEAREST_REPEAT 6
#define SAMPLER_LINEAR_REPEAT 7
#define SAMPLER_NEAREST_WITH_MIMPAMPS_REPEAT 8
#define SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT 9
#define SAMPLER_NEAREST_WITH_MIMPAMPS_ANISOTROPIC_REPEAT 10
#define SAMPLER_LINEAR_WITH_MIPMAPS_ANISOTROPIC_REPEAT 11

layout(set = 0, binding = 1) uniform sampler material_samplers[12];

layout(set = 0, binding = 2) uniform sampler shadow_sampler;

layout(set=0,binding=3,std140) uniform SceneData {

	mat4 projection_matrix;
	mat4 inv_projection_matrix;

	mat4 camera_matrix;
	mat4 inv_camera_matrix;

	vec2 viewport_size;
	vec2 screen_pixel_size;

	//used for shadow mapping only
	float z_offset;
	float z_slope_scale;


	float time;
	float reflection_multiplier; // one normally, zero when rendering reflections

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
	uint pad0;

#if 0
	vec4 ambient_light_color;
	vec4 bg_color;

	vec4 fog_color_enabled;
	vec4 fog_sun_color_amount;

	float ambient_energy;
	float bg_energy;

#endif

#if 0
	vec2 shadow_atlas_pixel_size;
	vec2 directional_shadow_pixel_size;


	float z_far;

	float subsurface_scatter_width;
	float ambient_occlusion_affect_light;
	float ambient_occlusion_affect_ao_channel;
	float opaque_prepass_threshold;

	bool fog_depth_enabled;
	float fog_depth_begin;
	float fog_depth_end;
	float fog_density;
	float fog_depth_curve;
	bool fog_transmit_enabled;
	float fog_transmit_curve;
	bool fog_height_enabled;
	float fog_height_min;
	float fog_height_max;
	float fog_height_curve;
#endif
} scene_data;

#define INSTANCE_FLAGS_FORWARD_MASK 0x7
#define INSTANCE_FLAGS_FORWARD_OMNI_LIGHT_SHIFT 3
#define INSTANCE_FLAGS_FORWARD_SPOT_LIGHT_SHIFT 6
#define INSTANCE_FLAGS_FORWARD_DECAL_SHIFT 9

#define INSTANCE_FLAGS_MULTIMESH  (1 << 12)
#define INSTANCE_FLAGS_MULTIMESH_FORMAT_2D  (1 << 13)
#define INSTANCE_FLAGS_MULTIMESH_HAS_COLOR  (1 << 14)
#define INSTANCE_FLAGS_MULTIMESH_HAS_CUSTOM_DATA  (1 << 15)
#define INSTANCE_FLAGS_MULTIMESH_STRIDE_SHIFT 16
//3 bits of stride
#define INSTANCE_FLAGS_MULTIMESH_STRIDE_MASK 0x7

#define INSTANCE_FLAGS_SKELETON (1 << 19)


struct InstanceData {
	mat4 transform;
	mat4 normal_transform;
	uint flags;
	uint instance_ofs; //instance_offset in instancing/skeleton buffer
	uint gi_offset; //GI information when using lightmapping (VCT or lightmap)
	uint layer_mask;

	uint reflection_probe_indices[4];
	uint omni_light_indices[4];
	uint spot_light_indices[4];
	uint decal_indices[4];
};


layout(set=0,binding=4,std430)  buffer Instances {
    InstanceData data[];
} instances;

struct LightData { //this structure needs to be 128 bits

	vec3 position;
	float inv_radius;
	vec3 direction;
	uint attenuation_energy; //attenuation
	uint color_specular; //rgb color, a specular (8 bit unorm)
	uint cone_attenuation_angle; // attenuation and angle, (16bit float)
	uint mask;
	uint shadow_color_enabled; //shadow rgb color, a>0.5 enabled (8bit unorm)
	vec4 atlas_rect; //used for shadow atlas uv on omni, and for projection atlas on spot
	mat4 shadow_matrix;
};

layout(set=0,binding=5,std140) uniform Lights {
	LightData data[MAX_LIGHT_DATA_STRUCTS];
} lights;

struct ReflectionData {

	vec3 box_extents;
	float index;
	vec3 box_offset;
	uint mask;
	vec4 params; // intensity, 0, interior , boxproject
	vec4 ambient; // ambient color, energy
	mat4 local_matrix; // up to here for spot and omni, rest is for directional
	// notes: for ambientblend, use distance to edge to blend between already existing global environment
};

layout(set=0,binding=6,std140) uniform ReflectionProbeData {
	ReflectionData data[MAX_REFLECTION_DATA_STRUCTS];
} reflections;

struct DirectionalLightData {

	vec3 direction;
	float energy;
	vec3 color;
	float specular;
	vec3 shadow_color;
	uint mask;
	bool blend_splits;
	bool shadow_enabled;
	float fade_from;
	float fade_to;
	vec4 shadow_split_offsets;
	mat4 shadow_matrix1;
	mat4 shadow_matrix2;
	mat4 shadow_matrix3;
	mat4 shadow_matrix4;

};

layout(set=0,binding=7,std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS];
} directional_lights;

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
	uint pad2;
};

layout(set=0,binding=8,std140) uniform GIProbes {
	GIProbeData data[MAX_GI_PROBES];
} gi_probes;

layout(set=0,binding=9) uniform texture3D gi_probe_textures[MAX_GI_PROBE_TEXTURES];


/* Set 1, Scene data that changes per render pass */


layout(set=1,binding=0) uniform texture2D depth_buffer;
layout(set=1,binding=1) uniform texture2D color_buffer;
layout(set=1,binding=2) uniform texture2D normal_buffer;
layout(set=1,binding=3) uniform texture2D roughness_limit;

#ifdef USE_RADIANCE_CUBEMAP_ARRAY

layout(set = 1, binding = 4) uniform textureCubeArray radiance_cubemap;

#else

layout(set = 1, binding = 4) uniform textureCube radiance_cubemap;

#endif


layout(set=1,binding=5) uniform textureCubeArray reflection_atlas;

layout(set=1,binding=6) uniform texture2D shadow_atlas;

layout(set=1,binding=7) uniform texture2D directional_shadow_atlas;


/* Set 2 Skeleton & Instancing (Multimesh) */

layout(set=2,binding=0,std430) buffer Transforms {
    vec4 data[];
} transforms;

/* Set 3 User Material */


