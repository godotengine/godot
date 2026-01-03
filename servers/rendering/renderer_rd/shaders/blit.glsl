#[vertex]

#version 450

#VERSION_DEFINES

layout(push_constant, std430) uniform Pos {
	vec4 src_rect;
	vec4 dst_rect;

	float rotation_sin;
	float rotation_cos;
	vec2 eye_center;

	float k1;
	float k2;
	float upscale;
	float aspect_ratio;

	uint layer;
	bool source_is_srgb;
	bool use_debanding;
	uint target_color_space;

	float reference_multiplier;
	float output_max_value;
	uint pad[2];
}
data;

layout(location = 0) out vec2 uv;

void main() {
	mat4 swapchain_transform = mat4(1.0);
	swapchain_transform[0][0] = data.rotation_cos;
	swapchain_transform[0][1] = -data.rotation_sin;
	swapchain_transform[1][0] = data.rotation_sin;
	swapchain_transform[1][1] = data.rotation_cos;

	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv = data.src_rect.xy + base_arr[gl_VertexIndex] * data.src_rect.zw;
	vec2 vtx = data.dst_rect.xy + base_arr[gl_VertexIndex] * data.dst_rect.zw;
	gl_Position = swapchain_transform * vec4(vtx * 2.0 - 1.0, 0.0, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(push_constant, std430) uniform Pos {
	vec4 src_rect;
	vec4 dst_rect;

	float rotation_sin;
	float rotation_cos;
	vec2 eye_center;

	float k1;
	float k2;
	float upscale;
	float aspect_ratio;

	uint layer;
	bool source_is_srgb;
	bool use_debanding;
	uint target_color_space;

	float reference_multiplier;
	float output_max_value;
	uint pad[2];
}
data;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 color;

#ifdef USE_LAYER
layout(binding = 0) uniform sampler2DArray src_rt;
#else
layout(binding = 0) uniform sampler2D src_rt;
#endif

// Keep in sync with RenderingDeviceCommons::ColorSpace
#define COLOR_SPACE_REC709_LINEAR 0
#define COLOR_SPACE_REC709_NONLINEAR_SRGB 1
#define COLOR_SPACE_REC2020_NONLINEAR_ST2084 2

vec3 srgb_to_linear(vec3 color) {
	return mix(pow((color.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)), color.rgb * (1.0 / 12.92), lessThan(color.rgb, vec3(0.04045)));
}

vec3 linear_to_srgb(vec3 color) {
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

vec3 rec709_to_rec2020(vec3 color) {
	const mat3 conversion = mat3(
			0.627403895934699, 0.069097289358232, 0.016391438875150,
			0.329283038377884, 0.919540395075458, 0.088013307877226,
			0.043313065687417, 0.011362315566309, 0.895595253247624);
	return conversion * color;
}

// Linear color must be non-negative. 1.0 represents 10,000 nits.
vec3 linear_to_st2084(vec3 color) {
	// Apply ST2084 curve
	const float c1 = 0.8359375;
	const float c2 = 18.8515625;
	const float c3 = 18.6875;
	const float m1 = 0.1593017578125;
	const float m2 = 78.84375;
	vec3 cp = pow(color, vec3(m1));

	return pow((c1 + c2 * cp) / (1 + c3 * cp), vec3(m2));
}

// From https://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
// and https://www.shadertoy.com/view/MslGR8 (5th one starting from the bottom)
// NOTE: `frag_coord` is in pixels (i.e. not normalized UV).
// This dithering must be applied after encoding changes (linear/nonlinear) have been applied
// as the final step before quantization from floating point to integer values.
vec3 screen_space_dither(vec2 frag_coord) {
	// Iestyn's RGB dither (7 asm instructions) from Portal 2 X360, slightly modified for VR.
	// Removed the time component to avoid passing time into this shader.
	vec3 dither = vec3(dot(vec2(171.0, 231.0), frag_coord));
	dither.rgb = fract(dither.rgb / vec3(103.0, 71.0, 97.0));

	// Subtract 0.5 to avoid slightly brightening the whole viewport.
	// Use a dither strength of 100% rather than the 37.5% suggested by the original source.
	// Divide by 255 to align to 8-bit quantization.
	return (dither.rgb - 0.5) / 255.0;
}

void main() {
#ifdef APPLY_LENS_DISTORTION
	vec2 coords = uv * 2.0 - 1.0;
	vec2 offset = coords - data.eye_center;

	// take aspect ratio into account
	offset.y /= data.aspect_ratio;

	// distort
	vec2 offset_sq = offset * offset;
	float radius_sq = offset_sq.x + offset_sq.y;
	float radius_s4 = radius_sq * radius_sq;
	float distortion_scale = 1.0 + (data.k1 * radius_sq) + (data.k2 * radius_s4);
	offset *= distortion_scale;

	// reapply aspect ratio
	offset.y *= data.aspect_ratio;

	// add our eye center back in
	coords = offset + data.eye_center;
	coords /= data.upscale;

	// and check our color
	if (coords.x < -1.0 || coords.y < -1.0 || coords.x > 1.0 || coords.y > 1.0) {
		color = vec4(0.0, 0.0, 0.0, 1.0);
	} else {
		// layer is always used here
		coords = (coords + vec2(1.0)) / vec2(2.0);
		color = texture(src_rt, vec3(coords, data.layer));
	}
#elif defined(USE_LAYER)
	color = texture(src_rt, vec3(uv, data.layer));
#else
	color = texture(src_rt, uv);
#endif

	// Colorspace conversion for final blit
	if (data.target_color_space == COLOR_SPACE_REC709_LINEAR) {
		if (data.source_is_srgb == true) {
			// sRGB -> linear conversion
			color.rgb = srgb_to_linear(color.rgb);
		}

		// Negative values may be interpreted as scRGB colors,
		// so clip them to the intended Rec. 709 colors.
		// Additionally, it is important that the game developer can trust that
		// Window.output_max_linear_value is truly the max output value, even if
		// the max luminance has been misconfigured by the player. This ensures that
		// the resulting image will always be as the game developer expects when they
		// use Window.output_max_linear_value and tonemapping functions will behave
		// as expected.
		color.rgb = clamp(color.rgb, vec3(0.0), vec3(data.output_max_value));

		// Adjust brightness of SDR content to reference luminance
		color.rgb *= data.reference_multiplier;
	} else if (data.target_color_space == COLOR_SPACE_REC709_NONLINEAR_SRGB) {
		// Negative values and values above 1.0 will be clipped by the target,
		// so no need to clip them here.
		if (data.source_is_srgb == false) {
			// linear -> sRGB conversion
			color.rgb = linear_to_srgb(color.rgb);

			// Even if debanding was applied earlier in the rendering process, it must
			// be reapplied after the linear_to_srgb floating point operations.
			// When the linear_to_srgb operation was not performed, the source is
			// already an 8-bit format and debanding cannot be effective. In this
			// case, GPU driver rounding error can add noise so debanding should be
			// skipped entirely.
			if (data.use_debanding) {
				color.rgb += screen_space_dither(gl_FragCoord.xy);
			}
		}
	} else if (data.target_color_space == COLOR_SPACE_REC2020_NONLINEAR_ST2084) {
		// Negative values may be interpreted as colors outside of sRGB,
		// so clip them to the intended sRGB colors.
		color.rgb = max(vec3(0.0), color.rgb);

		if (data.source_is_srgb == true) {
			// sRGB -> linear conversion
			color.rgb = srgb_to_linear(color.rgb);
		}

		// Convert to Rec.2020 primaries
		color.rgb = rec709_to_rec2020(color.rgb);

		// Adjust brightness of SDR content to reference luminance
		color.rgb *= data.reference_multiplier;

		// Apply the ST2084 curve
		color.rgb = linear_to_st2084(color.rgb);
	}
}
