// Scene data stores all our 3D rendering globals for a frame such as our matrices
// where this information is independent of the different RD implementations.
// This enables us to use this UBO in our main scene render shaders but also in
// effects that need access to this data.

#define SCENE_DATA_FLAGS_USE_AMBIENT_LIGHT (1 << 0)
#define SCENE_DATA_FLAGS_USE_AMBIENT_CUBEMAP (1 << 1)
#define SCENE_DATA_FLAGS_USE_REFLECTION_CUBEMAP (1 << 2)
#define SCENE_DATA_FLAGS_USE_ROUGHNESS_LIMITER (1 << 3)
#define SCENE_DATA_FLAGS_USE_FOG (1 << 4)
#define SCENE_DATA_FLAGS_USE_UV2_MATERIAL (1 << 5)
#define SCENE_DATA_FLAGS_USE_PANCAKE_SHADOWS (1 << 6)
#define SCENE_DATA_FLAGS_IN_SHADOW_PASS (1 << 7)

struct SceneData {
	mat4 projection_matrix;
	mat4 inv_projection_matrix;
	mat3x4 inv_view_matrix;
	mat3x4 view_matrix;

#ifdef USE_DOUBLE_PRECISION
	vec4 inv_view_precision;
#endif

	// only used for multiview
	mat4 projection_matrix_view[MAX_VIEWS];
	mat4 inv_projection_matrix_view[MAX_VIEWS];
	vec4 eye_offset[MAX_VIEWS];

	// Used for billboards to cast correct shadows.
	mat4 main_cam_inv_view_matrix;

	vec2 viewport_size;
	vec2 screen_pixel_size;

	// Use vec4s because std140 doesn't play nice with vec2s, z and w are wasted.
	vec4 directional_penumbra_shadow_kernel[32];
	vec4 directional_soft_shadow_kernel[32];
	vec4 penumbra_shadow_kernel[32];
	vec4 soft_shadow_kernel[32];

	vec2 shadow_atlas_pixel_size;
	vec2 directional_shadow_pixel_size;

	float radiance_pixel_size;
	float radiance_border_size;
	vec2 reflection_atlas_border_size;

	uint directional_light_count;
	float dual_paraboloid_side;
	float z_far;
	float z_near;

	float roughness_limiter_amount;
	float roughness_limiter_limit;
	float opaque_prepass_threshold;
	uint flags;

	mat3 radiance_inverse_xform;

	vec4 ambient_light_color_energy;

	float ambient_color_sky_mix;
	float fog_density;
	float fog_height;
	float fog_height_density;

	float fog_depth_curve;
	float fog_depth_begin;
	float fog_depth_end;
	float fog_sun_scatter;

	vec3 fog_light_color;
	float fog_aerial_perspective;

	float time;
	float taa_frame_count;
	vec2 taa_jitter;

	float emissive_exposure_normalization;
	float IBL_exposure_normalization;
	uint camera_visible_layers;
	float pass_alpha_multiplier;
};
