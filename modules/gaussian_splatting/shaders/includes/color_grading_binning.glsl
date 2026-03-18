#ifndef COLOR_GRADING_BINNING_GLSL
#define COLOR_GRADING_BINNING_GLSL

// ============================================================================
// Binning Stage Color Grading
// ============================================================================
// Applied after SH evaluation, before packing into ProjectedGaussian
// Cost: ~20 ALU operations per splat = ~0.02ms for 1M splats
// ============================================================================

// RGB to HSV (copy from tonemap.glsl)
vec3 rgb_to_hsv(vec3 c) {
	vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// HSV to RGB (copy from tonemap.glsl)
vec3 hsv_to_rgb(vec3 c) {
	vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Apply color grading to splat color (binning stage)
// Operates in linear color space, before R11G11B10 packing
vec3 apply_color_grading_binning(vec3 color) {
	// Check if enabled
	if (params.color_grading_primary.x < 0.5) {
		return color;
	}

	// Unpack parameters from RenderParams
	float exposure = params.color_grading_primary.y;
	float contrast = params.color_grading_primary.z;
	float saturation = params.color_grading_primary.w;

	float temperature = params.color_grading_secondary.x;
	float tint = params.color_grading_secondary.y;
	float hue_shift_deg = params.color_grading_secondary.z;

	// 1. Exposure (multiply by 2^EV)
	color *= exp2(exposure);

	// 2. Temperature & Tint (simple color shift)
	// Temperature: shift toward orange (+) or blue (-)
	// Tint: shift toward green (+) or magenta (-)
	float temp_factor = temperature * 0.01; // Normalize to -1..1
	float tint_factor = tint * 0.01;

	color.r += temp_factor * 0.5;
	color.b -= temp_factor * 0.5;

	color.g += tint_factor * 0.5;
	color.r -= tint_factor * 0.25;
	color.b -= tint_factor * 0.25;

	// Clamp after temperature/tint
	color = max(color, vec3(0.0));

	// 3. Contrast (around 0.5 midpoint)
	color = (color - 0.5) * contrast + 0.5;

	// 4. HSV adjustments (hue shift + saturation)
	vec3 hsv = rgb_to_hsv(max(color, vec3(0.0)));

	// Hue shift (convert degrees to 0-1 range)
	hsv.x += (hue_shift_deg / 360.0);
	hsv.x = fract(hsv.x); // Wrap around

	// Saturation
	hsv.y *= saturation;
	hsv.y = clamp(hsv.y, 0.0, 1.0);

	color = hsv_to_rgb(hsv);

	// Final clamp to prevent negative values
	return max(color, vec3(0.0));
}

#endif // COLOR_GRADING_BINNING_GLSL
