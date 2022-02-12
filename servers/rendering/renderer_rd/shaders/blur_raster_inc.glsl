#define FLAG_HORIZONTAL (1 << 0)
#define FLAG_USE_ORTHOGONAL_PROJECTION (1 << 1)
#define FLAG_GLOW_FIRST_PASS (1 << 2)

layout(push_constant, binding = 1, std430) uniform Blur {
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
}
blur;
