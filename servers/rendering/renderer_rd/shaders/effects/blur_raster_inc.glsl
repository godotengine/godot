#define FLAG_HORIZONTAL (1 << 0)
#define FLAG_USE_ORTHOGONAL_PROJECTION (1 << 1)
#define FLAG_GLOW_FIRST_PASS (1 << 2)

layout(push_constant, std430) uniform Blur {
	vec2 pixel_size; // 08 - 08
	uint flags; // 04 - 12
	uint pad; // 04 - 16

	// Glow.
	float glow_strength; // 04 - 20
	float glow_bloom; // 04 - 24
	float glow_hdr_threshold; // 04 - 28
	float glow_hdr_scale; // 04 - 32

	float glow_exposure; // 04 - 36
	float glow_white; // 04 - 40
	float glow_luminance_cap; // 04 - 44
	float glow_auto_exposure_scale; // 04 - 48

	float luminance_multiplier; // 04 - 52
	float res1; // 04 - 56
	float res2; // 04 - 60
	float res3; // 04 - 64
}
blur;
