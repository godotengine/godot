#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

void main() {
	// old code, ARM driver bug on Mali-GXXx GPUs and Vulkan API 1.3.xxx
	// https://github.com/godotengine/godot/pull/92817#issuecomment-2168625982
	//vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	//gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	//uv_interp = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0

	vec2 vertex_base;
	if (gl_VertexIndex == 0) {
		vertex_base = vec2(-1.0, -1.0);
	} else if (gl_VertexIndex == 1) {
		vertex_base = vec2(-1.0, 3.0);
	} else {
		vertex_base = vec2(3.0, -1.0);
	}
	gl_Position = vec4(vertex_base, 0.0, 1.0);
	uv_interp = clamp(vertex_base, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
}

#[fragment]

#version 450

#VERSION_DEFINES

#ifndef SUBPASS
#include "fxaa_inc.glsl"
#endif // !SUBPASS

#include "../tonemapper_inc.glsl"

#ifdef USE_MULTIVIEW
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#endif //USE_MULTIVIEW

layout(location = 0) in vec2 uv_interp;

#ifdef USE_MULTIVIEW
#define SAMPLER_FORMAT sampler2DArray
#else
#define SAMPLER_FORMAT sampler2D
#endif

#ifdef SUBPASS
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput input_color;
#else
layout(set = 0, binding = 0) uniform SAMPLER_FORMAT source_color;
#endif

layout(set = 1, binding = 0) uniform SAMPLER_FORMAT source_glow;
layout(set = 1, binding = 1) uniform sampler2D glow_map;

layout(set = 2, binding = 0) uniform sampler2D source_color_correction_1d;
layout(set = 2, binding = 1) uniform sampler3D source_color_correction_3d;

layout(constant_id = 0) const bool use_bcs = false;
layout(constant_id = 1) const bool use_glow = false;
layout(constant_id = 2) const bool use_glow_map = false;
layout(constant_id = 3) const bool use_color_correction = false;
layout(constant_id = 4) const bool use_color_correction_lut_1d = false;
layout(constant_id = 5) const bool use_fxaa = false;
layout(constant_id = 6) const bool deband_8_bit = false;
layout(constant_id = 7) const bool deband_10_bit = false;
layout(constant_id = 8) const bool convert_to_srgb = false;
layout(constant_id = 9) const bool tonemapper_linear = false;
layout(constant_id = 10) const bool tonemapper_reinhard = false;
layout(constant_id = 11) const bool tonemapper_filmic = false;
layout(constant_id = 12) const bool tonemapper_aces = false;
layout(constant_id = 13) const bool tonemapper_agx = false;
layout(constant_id = 14) const bool glow_mode_add = false;
layout(constant_id = 15) const bool glow_mode_screen = false;
layout(constant_id = 16) const bool glow_mode_softlight = false;
layout(constant_id = 17) const bool glow_mode_replace = false;
layout(constant_id = 18) const bool glow_mode_mix = false;

layout(push_constant, std430) uniform Params {
	vec3 bcs;
	float luminance_multiplier;

	vec2 src_pixel_size;
	vec2 dest_pixel_size;

	float glow_intensity;
	float glow_map_strength;
	float exposure;
	float white;

	vec4 tonemapper_params;
}
params;

layout(location = 0) out vec4 frag_color;

#ifdef USE_MULTIVIEW
vec3 gather_glow() {
	vec2 uv = gl_FragCoord.xy * params.dest_pixel_size;
	return textureLod(source_glow, vec3(uv, ViewIndex), 0.0).rgb * params.luminance_multiplier;
}
#else
vec3 gather_glow() {
	vec2 uv = gl_FragCoord.xy * params.dest_pixel_size;
	return textureLod(source_glow, uv, 0.0).rgb * params.luminance_multiplier;
}
#endif // !USE_MULTIVIEW

// Applies glow using the selected blending mode. Does not handle the mix blend mode.
vec3 apply_glow(vec3 color, vec3 glow, float white) {
	if (glow_mode_add) {
		return color + glow;
	} else if (glow_mode_screen) {
		// Glow cannot be above 1.0 after normalizing and should be non-negative
		// to produce expected results. It is possible that glow can be negative
		// if negative lights were used in the scene.
		// We clamp to white because glow will be normalized to this range.
		// Note: white cannot be smaller than the maximum output value.
		glow.rgb = clamp(glow.rgb, 0.0, white);

		// Normalize to white range.
		//glow.rgb /= white;
		//color.rgb /= white;
		//color.rgb = (color.rgb + glow.rgb) - (color.rgb * glow.rgb);
		// Expand back to original range.
		//color.rgb *= white;

		// The following is a mathematically simplified version of the above.
		color.rgb = color.rgb + glow.rgb - (color.rgb * glow.rgb / white);

		return color;
	} else if (glow_mode_softlight) {
		// Glow cannot be above 1.0 should be non-negative to produce
		// expected results. It is possible that glow can be negative
		// if negative lights were used in the scene.
		// Note: This approach causes a discontinuity with scene values
		// at 1.0, but because this glow should have its strongest influence
		// anchored at 0.25 there is no way around this.
		glow.rgb = clamp(glow.rgb, 0.0, 1.0);

		color.r = color.r > 1.0 ? color.r : color.r + glow.r * ((color.r <= 0.25f ? ((16.0f * color.r - 12.0f) * color.r + 4.0f) * color.r : sqrt(color.r)) - color.r);
		color.g = color.g > 1.0 ? color.g : color.g + glow.g * ((color.g <= 0.25f ? ((16.0f * color.g - 12.0f) * color.g + 4.0f) * color.g : sqrt(color.g)) - color.g);
		color.b = color.b > 1.0 ? color.b : color.b + glow.b * ((color.b <= 0.25f ? ((16.0f * color.b - 12.0f) * color.b + 4.0f) * color.b : sqrt(color.b)) - color.b);

		return color;
	} else { //replace
		return glow;
	}
}

void main() {
#ifdef SUBPASS
	// SUBPASS and USE_MULTIVIEW can be combined but in that case we're already reading from the correct layer
#ifdef USE_MULTIVIEW
	// In order to ensure the `SpvCapabilityMultiView` is included in the SPIR-V capabilities, gl_ViewIndex must
	// be read in the shader. Without this, transpilation to Metal fails to include the multi-view variant.
	uint vi = ViewIndex;
#endif
	vec4 color = subpassLoad(input_color);
#elif defined(USE_MULTIVIEW)
	vec4 color = textureLod(source_color, vec3(uv_interp, ViewIndex), 0.0f);
#else
	vec4 color = textureLod(source_color, uv_interp, 0.0f);
#endif
	color.rgb *= params.luminance_multiplier;

	// Exposure

	color.rgb *= params.exposure;

#ifndef SUBPASS
	// Single-pass FXAA and pre-tonemap glow.
	if (use_fxaa) {
		// FXAA must be performed before glow to preserve the "bleed" effect of glow.
		color.rgb = do_fxaa(color.rgb, params.exposure, uv_interp, params.luminance_multiplier, params.src_pixel_size, source_color
#ifdef USE_MULTIVIEW
				,
				ViewIndex
#endif
		);
	}

	if (use_glow && !glow_mode_softlight) {
		vec3 glow = gather_glow() * params.glow_intensity;
		if (use_glow_map) {
			glow = mix(glow, texture(glow_map, uv_interp).rgb * glow, params.glow_map_strength);
		}

		if (glow_mode_mix) {
			color.rgb = color.rgb * (1.0 - params.glow_intensity) + glow;
		} else {
			color.rgb = apply_glow(color.rgb, glow, params.white);
		}
	}
#endif

	// Tonemap to lower dynamic range.

	uint tonemapper_mode = tonemapper_mode_from_booleans(tonemapper_linear, tonemapper_reinhard, tonemapper_filmic, tonemapper_aces, tonemapper_agx);
	color.rgb = apply_tonemapping(color.rgb, tonemapper_mode, params.tonemapper_params);

#ifndef SUBPASS
	// Post-tonemap glow.

	if (use_glow && glow_mode_softlight) {
		// Apply soft light after tonemapping to mitigate the issue of discontinuity
		// at 1.0 and higher. This makes the issue only appear with HDR output that
		// can exceed a 1.0 output value.
		vec3 glow = gather_glow() * params.glow_intensity;
		if (use_glow_map) {
			glow = mix(glow, texture(glow_map, uv_interp).rgb * glow, params.glow_map_strength);
		}
		glow = apply_tonemapping(glow, tonemapper_mode, params.tonemapper_params);
		color.rgb = apply_glow(color.rgb, glow, params.white);
	}
#endif

	if (use_bcs) {
		// Additional effects.
		color.rgb = apply_bcs(color.rgb, params.bcs, source_color_correction_1d, source_color_correction_3d, use_color_correction, use_color_correction_lut_1d, convert_to_srgb);
	} else if (convert_to_srgb) {
		// Regular linear -> SRGB conversion.
		color.rgb = linear_to_srgb(color.rgb);
	}

	// Debanding should be done at the end of tonemapping, but before writing to the LDR buffer.
	// Otherwise, we're adding noise to an already-quantized image.
	if (deband_8_bit) {
		// Divide by 255 to align to 8-bit quantization.
		color.rgb += screen_space_dither(gl_FragCoord.xy, 255.0);
	} else if (deband_10_bit) {
		// Divide by 1023 to align to 10-bit quantization.
		color.rgb += screen_space_dither(gl_FragCoord.xy, 1023.0);
	}

	frag_color = color;
}
