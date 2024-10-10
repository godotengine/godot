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

uniform sampler3D tony_mc_mapface_lut; //texunit:3

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

// Based on Reinhard's extended formula, see equation 4 in https://doi.org/cjbgrt
vec3 tonemap_reinhard(vec3 color, float p_white) {
	float white_squared = p_white * p_white;
	vec3 white_squared_color = white_squared * color;
	// Equivalent to color * (1 + color / white_squared) / (1 + color)
	return (white_squared_color + color * color) / (white_squared_color + white_squared);
}

// https://github.com/h3r2tic/tony-mc-mapface/blob/main/shader/tony_mc_mapface.hlsl
vec3 tonemap_tony_mc_mapface(vec3 stimulus) {
	vec3 encoded = stimulus / (stimulus + 1.0f);

	const float LUT_DIMS = 48.0f;
	vec3 uv = encoded * ((LUT_DIMS - 1.0f) / LUT_DIMS) + 0.5f / LUT_DIMS;

	return texture(tony_mc_mapface_lut, uv).rgb;
}

#define TONEMAPPER_LINEAR 0
#define TONEMAPPER_REINHARD 1
#define TONEMAPPER_FILMIC 2
#define TONEMAPPER_ACES 3
#define TONEMAPPER_TONY_MC_MAPFACE 4

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
	} else if (tonemapper == TONEMAPPER_TONY_MC_MAPFACE) {
#ifdef SKY_SHADER
		// Sampling the Tony McMapface LUT in the sky shader leads to pitch black shadows if the "Sky" background
		// mode is enabled for the environment. Avoid this by returning the color as is.
		return color;
#else
		return tonemap_tony_mc_mapface(max(vec3(0.0f), color));
#endif
	} else {
		return color;
	}
}

#endif // APPLY_TONEMAPPING
