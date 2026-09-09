#define FLAG_HORIZONTAL (1 << 0)
#define FLAG_USE_ORTHOGONAL_PROJECTION (1 << 1)
#define FLAG_GLOW_FIRST_PASS (1 << 2)

layout(push_constant, std430) uniform Blur {
	vec2 dest_pixel_size; // 08 - 08
	vec2 source_pixel_size; // 08 - 16

	vec2 pad; // 08 - 24
	uint flags; // 04 - 28
	float glow_level; // 04 - 32

	// Glow.
	float glow_strength; // 04 - 36
	float glow_bloom; // 04 - 40
	float glow_hdr_threshold; // 04 - 44
	float glow_hdr_scale; // 04 - 48

	float glow_exposure; // 04 - 52
	float glow_white; // 04 - 56
	float glow_luminance_cap; // 04 - 60
	float luminance_multiplier; // 04 - 64
}
blur;
