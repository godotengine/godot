layout(std140) uniform TonemapData { //ubo:0
	float exposure;
	float white;
	int tonemapper;
	int pad;

	int pad2;
	float brightness;
	float contrast;
	float saturation;
};

// This expects 0-1 range input.
vec3 linear_to_srgb(vec3 color) {
	//color = clamp(color, vec3(0.0), vec3(1.0));
	//const vec3 a = vec3(0.055f);
	//return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
	// Approximation from http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
	return max(vec3(1.055) * pow(color, vec3(0.416666667)) - vec3(0.055), vec3(0.0));
}

// This expects 0-1 range input, outside that range it behaves poorly.
vec3 srgb_to_linear(vec3 color) {
	// Approximation from http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
	return color * (color * (color * 0.305306011 + 0.682171111) + 0.012522878);
}

#ifdef APPLY_TONEMAPPING

// Based on Reinhard's extended formula, see equation 4 in https://doi.org/cjbgrt
vec3 tonemap_reinhard(vec3 color, float p_white) {
	float white_squared = p_white * p_white;
	vec3 white_squared_color = white_squared * color;
	// Equivalent to color * (1 + color / white_squared) / (1 + color)
	return (white_squared_color + color * color) / (white_squared_color + white_squared);
}

vec3 tonemap_filmic(vec3 color, float p_white) {
	// exposure bias: input scale (color *= bias, white *= bias) to make the brightness consistent with other tonemappers
	// also useful to scale the input to the range that the tonemapper is designed for (some require very high input values)
	// has no effect on the curve's general shape or visual properties
	const float exposure_bias = 2.0f;
	const float A = 0.22f * exposure_bias * exposure_bias; // bias baked into constants for performance
	const float B = 0.30f * exposure_bias;
	const float C = 0.10f;
	const float D = 0.20f;
	const float E = 0.01f;
	const float F = 0.30f;

	vec3 color_tonemapped = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
	float p_white_tonemapped = ((p_white * (A * p_white + C * B) + D * E) / (p_white * (A * p_white + B) + D * F)) - E / F;

	return color_tonemapped / p_white_tonemapped;
}

// Adapted from https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// (MIT License).
vec3 tonemap_aces(vec3 color, float p_white) {
	const float exposure_bias = 1.8f;
	const float A = 0.0245786f;
	const float B = 0.000090537f;
	const float C = 0.983729f;
	const float D = 0.432951f;
	const float E = 0.238081f;

	// Exposure bias baked into transform to save shader instructions. Equivalent to `color *= exposure_bias`
	const mat3 rgb_to_rrt = mat3(
			vec3(0.59719f * exposure_bias, 0.35458f * exposure_bias, 0.04823f * exposure_bias),
			vec3(0.07600f * exposure_bias, 0.90834f * exposure_bias, 0.01566f * exposure_bias),
			vec3(0.02840f * exposure_bias, 0.13383f * exposure_bias, 0.83777f * exposure_bias));

	const mat3 odt_to_rgb = mat3(
			vec3(1.60475f, -0.53108f, -0.07367f),
			vec3(-0.10208f, 1.10813f, -0.00605f),
			vec3(-0.00327f, -0.07276f, 1.07602f));

	color *= rgb_to_rrt;
	vec3 color_tonemapped = (color * (color + A) - B) / (color * (C * color + D) + E);
	color_tonemapped *= odt_to_rgb;

	p_white *= exposure_bias;
	float p_white_tonemapped = (p_white * (p_white + A) - B) / (p_white * (C * p_white + D) + E);

	return color_tonemapped / p_white_tonemapped;
}

// Mean error^2: 3.6705141e-06
vec3 agx_default_contrast_approx(vec3 x) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;

	return +15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 agx(vec3 val) {
	const mat3 agx_mat = transpose(mat3(
			0.544813, 0.37379614, 0.08139087,
			0.14041554, 0.75414325, 0.10544122,
			0.0888119, 0.17888511, 0.73230299));

	// AgX does not provide whitepoint adjustments, so hardcode it to a value that matches the Blender appearance closely.
	const float white = 16.016004;

	const float min_ev = -12.47393;
	const float max_ev = log2(white);

	// Input transform (inset).
	val = agx_mat * val;

	// Log2 space encoding.
	val = clamp(log2(val), min_ev, max_ev);
	val = (val - min_ev) / (max_ev - min_ev);

	// Apply sigmoid function approximation.
	val = agx_default_contrast_approx(val);

	return val;
}

vec3 agx_eotf(vec3 val) {
	const mat3 agx_mat_inv = transpose(mat3(
			1.96489403, -0.85600791, -0.10888612,
			-0.29930908, 1.32639189, -0.02708281,
			-0.16435644, -0.2382074, 1.40256385));

	// Inverse input transform (outset).
	val = agx_mat_inv * val;

	// sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
	// NOTE: We're linearizing the output here. Comment/adjust when
	// *not* using a sRGB render target.
	val = pow(val, vec3(2.2));

	return val;
}

vec3 agx_look_punchy(vec3 val) {
	const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
	float luma = dot(val, lw);

	vec3 offset = vec3(0.0);
	vec3 slope = vec3(1.0);
	vec3 power = vec3(1.35, 1.35, 1.35);
	float sat = 1.4;

	// ASC CDL.
	val = pow(val * slope + offset, power);
	return luma + sat * (val - luma);
}

// Adapted from https://iolite-engine.com/blog_posts/minimal_agx_implementation
vec3 tonemap_agx(vec3 color, bool punchy) {
	color = agx(color);
	if (punchy) {
		color = agx_look_punchy(color);
	}
	color = agx_eotf(color);
	return color;
}

#define TONEMAPPER_LINEAR 0
#define TONEMAPPER_REINHARD 1
#define TONEMAPPER_FILMIC 2
#define TONEMAPPER_ACES 3
#define TONEMAPPER_AGX 4
#define TONEMAPPER_AGX_PUNCHY 5

vec3 apply_tonemapping(vec3 color, float p_white) { // inputs are LINEAR
	// Ensure color values passed to tonemappers are positive.
	// They can be negative in the case of negative lights, which leads to undesired behavior.
	if (tonemapper == TONEMAPPER_LINEAR) {
		return color;
	} else if (tonemapper == TONEMAPPER_REINHARD) {
		return tonemap_reinhard(max(vec3(0.0f), color), p_white);
	} else if (tonemapper == TONEMAPPER_FILMIC) {
		return tonemap_filmic(max(vec3(0.0f), color), p_white);
	} else if (tonemapper == TONEMAPPER_ACES) {
		return tonemap_aces(max(vec3(0.0f), color), p_white);
	} else if (tonemapper == TONEMAPPER_AGX) {
		return tonemap_agx(max(vec3(0.0f), color), false);
	} else { // TONEMAPPER_AGX_PUNCHY
		return tonemap_agx(max(vec3(0.0f), color), true);
	}
}

#endif // APPLY_TONEMAPPING
