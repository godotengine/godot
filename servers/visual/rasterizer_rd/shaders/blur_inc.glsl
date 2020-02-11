#define FLAG_HORIZONTAL (1 << 0)
#define FLAG_USE_BLUR_SECTION (1 << 1)
#define FLAG_USE_ORTHOGONAL_PROJECTION (1 << 2)
#define FLAG_DOF_NEAR_FIRST_TAP (1 << 3)
#define FLAG_GLOW_FIRST_PASS (1 << 4)
#define FLAG_FLIP_Y (1 << 5)
#define FLAG_COPY_FORCE_LUMINANCE (1 << 6)

layout(push_constant, binding = 1, std430) uniform Blur {
	vec4 section;
	vec2 pixel_size;
	uint flags;
	uint pad;
	// Glow.
	float glow_strength;
	float glow_bloom;
	float glow_hdr_threshold;
	float glow_hdr_scale;
	float glow_exposure;
	float glow_white;
	float glow_luminance_cap;
	float glow_auto_exposure_grey;
	// DOF.
	float dof_begin;
	float dof_end;
	float dof_radius;
	float dof_pad;

	vec2 dof_dir;
	float camera_z_far;
	float camera_z_near;

	vec4 ssao_color;
}
blur;
