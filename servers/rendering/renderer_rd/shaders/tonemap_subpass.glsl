#[vertex]

#version 450

#VERSION_DEFINES

#ifdef MULTIVIEW
#ifdef has_VK_KHR_multiview
#extension GL_EXT_multiview : enable
#endif
#endif

void main() {
	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	gl_Position = vec4(base_arr[gl_VertexIndex] * 2.0 - 1.0, 0.0, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#ifdef MULTIVIEW
#ifdef has_VK_KHR_multiview
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#else // has_VK_KHR_multiview
#define ViewIndex 0
#endif // has_VK_KHR_multiview
#endif //MULTIVIEW

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput source_color;

layout(set = 1, binding = 0) uniform sampler2D source_auto_exposure;

#ifdef USE_1D_LUT
layout(set = 2, binding = 0) uniform sampler2D source_color_correction;
#else
layout(set = 2, binding = 0) uniform sampler3D source_color_correction;
#endif

layout(push_constant, binding = 1, std430) uniform Params {
	vec3 bcs;
	bool use_bcs;

	bool use_auto_exposure;
	bool use_color_correction;
	uint tonemapper;
	uint pad1;

	float exposure;
	float white;
	float auto_exposure_grey;
	bool use_debanding;
}
params;

layout(location = 0) out vec4 frag_color;

vec3 tonemap_filmic(vec3 color, float white) {
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
	float white_tonemapped = ((white * (A * white + C * B) + D * E) / (white * (A * white + B) + D * F)) - E / F;

	return color_tonemapped / white_tonemapped;
}

vec3 tonemap_aces(vec3 color, float white) {
	const float exposure_bias = 0.85f;
	const float A = 2.51f * exposure_bias * exposure_bias;
	const float B = 0.03f * exposure_bias;
	const float C = 2.43f * exposure_bias * exposure_bias;
	const float D = 0.59f * exposure_bias;
	const float E = 0.14f;

	vec3 color_tonemapped = (color * (A * color + B)) / (color * (C * color + D) + E);
	float white_tonemapped = (white * (A * white + B)) / (white * (C * white + D) + E);

	return color_tonemapped / white_tonemapped;
}

vec3 tonemap_reinhard(vec3 color, float white) {
	// Ensure color values are positive.
	// They can be negative in the case of negative lights, which leads to undesired behavior.
	color = max(vec3(0.0), color);

	return (white * color + color) / (color * white + white);
}

vec3 linear_to_srgb(vec3 color) {
	//if going to srgb, clamp from 0 to 1.
	color = clamp(color, vec3(0.0), vec3(1.0));
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

#define TONEMAPPER_LINEAR 0
#define TONEMAPPER_REINHARD 1
#define TONEMAPPER_FILMIC 2
#define TONEMAPPER_ACES 3

vec3 apply_tonemapping(vec3 color, float white) { // inputs are LINEAR, always outputs clamped [0;1] color

	if (params.tonemapper == TONEMAPPER_LINEAR) {
		return color;
	} else if (params.tonemapper == TONEMAPPER_REINHARD) {
		return tonemap_reinhard(color, white);
	} else if (params.tonemapper == TONEMAPPER_FILMIC) {
		return tonemap_filmic(color, white);
	} else { //aces
		return tonemap_aces(color, white);
	}
}

vec3 apply_bcs(vec3 color, vec3 bcs) {
	color = mix(vec3(0.0f), color, bcs.x);
	color = mix(vec3(0.5f), color, bcs.y);
	color = mix(vec3(dot(vec3(1.0f), color) * 0.33333f), color, bcs.z);

	return color;
}

#ifdef USE_1D_LUT
vec3 apply_color_correction(vec3 color) {
	color.r = texture(source_color_correction, vec2(color.r, 0.0f)).r;
	color.g = texture(source_color_correction, vec2(color.g, 0.0f)).g;
	color.b = texture(source_color_correction, vec2(color.b, 0.0f)).b;
	return color;
}
#else
vec3 apply_color_correction(vec3 color) {
	return textureLod(source_color_correction, color, 0.0).rgb;
}
#endif

// From http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
// and https://www.shadertoy.com/view/MslGR8 (5th one starting from the bottom)
// NOTE: `frag_coord` is in pixels (i.e. not normalized UV).
vec3 screen_space_dither(vec2 frag_coord) {
	// Iestyn's RGB dither (7 asm instructions) from Portal 2 X360, slightly modified for VR.
	vec3 dither = vec3(dot(vec2(171.0, 231.0), frag_coord));
	dither.rgb = fract(dither.rgb / vec3(103.0, 71.0, 97.0));

	// Subtract 0.5 to avoid slightly brightening the whole viewport.
	return (dither.rgb - 0.5) / 255.0;
}

void main() {
	vec3 color = subpassLoad(source_color).rgb;

	// Exposure

	float exposure = params.exposure;

	if (params.use_auto_exposure) {
		exposure *= 1.0 / (texelFetch(source_auto_exposure, ivec2(0, 0), 0).r / params.auto_exposure_grey);
	}

	color *= exposure;

	// Tonemap & SRGB Conversion

	if (params.use_debanding) {
		// For best results, debanding should be done before tonemapping.
		// Otherwise, we're adding noise to an already-quantized image.
		color += screen_space_dither(gl_FragCoord.xy);
	}

	color = apply_tonemapping(color, params.white);

	color = linear_to_srgb(color); // regular linear -> SRGB conversion

	// Additional effects

	if (params.use_bcs) {
		color = apply_bcs(color, params.bcs);
	}

	if (params.use_color_correction) {
		color = apply_color_correction(color);
	}

	frag_color = vec4(color, 1.0f);
}
