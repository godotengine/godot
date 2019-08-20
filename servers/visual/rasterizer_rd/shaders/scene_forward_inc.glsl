
#define M_PI 3.14159265359
#define ROUGHNESS_MAX_LOD 5

/* Set 0 Scene data, screen and sources (changes the least) */

layout(set=0,binding=1) uniform texture2D depth_buffer;
layout(set=0,binding=2) uniform texture2D color_buffer;
layout(set=0,binding=3) uniform texture2D normal_buffer;

layout(set=0,binding=4,std140) uniform SceneData {

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

layout(set = 0, binding = 5) uniform sampler material_samplers[12];

#if 0
struct DirectionalLightData {

	vec4 light_pos_inv_radius;
	vec4 light_direction_attenuation;
	vec4 light_color_energy;
	vec4 light_params; // cone attenuation, angle, specular, shadow enabled,
	vec4 light_clamp;
	vec4 shadow_color_contact;
	mat4 shadow_matrix1;
	mat4 shadow_matrix2;
	mat4 shadow_matrix3;
	mat4 shadow_matrix4;
	vec4 shadow_split_offsets;
};
#endif

/* Set 1 Skeleton Data (most objects lack it, so it changes little */

#if 0
layout(set = 1 binding = 0, std140) uniform SkeletonData {
	mat4 transform;
	bool use_skeleton;
	bool use_world_coords;
	bool pad1;
	bool pad2;
} skeleton;

layout(set = 1, binding = 1) uniform textureBuffer skeleton_bones;
#endif

/* Set 2 Custom Material Data (changess less than instance) */


/* Set 3 Instance Data (Set on every draw call) */

layout(push_constant, binding = 0, std430) uniform DrawData {
	//used in forward rendering, 16 bits indices, max 8
	uint reflection_probe_count;
	uint omni_light_count;
	uint spot_light_count;
	uint decal_count;
	uvec4 reflection_probe_indices;
	uvec4 omni_light_indices;
	uvec4 spot_light_indices;
	uvec4 decal_indices;
} draw_data;

layout(set = 3, binding = 0, std140) uniform InstanceData {
	mat4 transform;
	mat3 normal_transform;
	uint flags;
	uint pad0;
	uint pad1;
	uint pad2;
} instance_data;

layout(set = 3, binding = 1) uniform textureBuffer multimesh_transforms;

#ifdef USE_LIGHTMAP

layout(set = 3, binding = 2) uniform texture2D lightmap;

#endif

#ifdef USE_VOXEL_CONE_TRACING

layout(set = 3, binding = 3) uniform texture3D gi_probe[2];

#ifdef USE_ANISOTROPIC_VOXEL_CONE_TRACING
layout(set = 3, binding = 4) uniform texture3D gi_probe_aniso_pos[2];
layout(set = 3, binding = 5) uniform texture3D gi_probe_aniso_neg[2];
#endif


#endif
