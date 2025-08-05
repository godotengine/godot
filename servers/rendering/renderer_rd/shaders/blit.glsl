#[vertex]

#version 450

#VERSION_DEFINES

layout(push_constant, std140) uniform Pos {
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
	bool convert_to_srgb;
	bool use_debanding;
	float pad;
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

layout(push_constant, std140) uniform Pos {
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
	bool convert_to_srgb;
	bool use_debanding;
	float pad;
}
data;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 color;

#ifdef USE_LAYER
layout(binding = 0) uniform sampler2DArray src_rt;
#else
layout(binding = 0) uniform sampler2D src_rt;
#endif

vec3 linear_to_srgb(vec3 color) {
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
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

	if (data.convert_to_srgb) {
		color.rgb = linear_to_srgb(color.rgb); // Regular linear -> SRGB conversion.

		// Even if debanding was applied earlier in the rendering process, it must
		// be reapplied after the linear_to_srgb floating point operations.
		// When the linear_to_srgb operation was not performed, the source is
		// already an 8-bit format and debanding cannot be effective. In this
		// case, GPU driver rounding error can add noise so debanding should be
		// skipped entirely.
		if (data.use_debanding) {
			color.rgb += screen_space_dither(gl_FragCoord.xy);
		}

		color.rgb = clamp(color.rgb, vec3(0.0), vec3(1.0));
	}
}
