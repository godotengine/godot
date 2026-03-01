/* clang-format off */
#[modes]
mode_default =

#[specializations]

USE_MULTIVIEW = false
USE_GLOW = false
USE_LUMINANCE_MULTIPLIER = false
USE_BCS = false
USE_COLOR_CORRECTION = false
USE_1D_LUT = false
USE_SSAO_ABYSS = false
USE_SSAO_LOW = false
USE_SSAO_MED = false
USE_SSAO_HIGH = false
USE_SSAO_MEGA = false

#[vertex]
layout(location = 0) in vec2 vertex_attrib;

/* clang-format on */

out vec2 uv_interp;

void main() {
	uv_interp = vertex_attrib * 0.5 + 0.5;
	gl_Position = vec4(vertex_attrib, 1.0, 1.0);
}

/* clang-format off */
#[fragment]
/* clang-format on */

// If we reach this code, we always tonemap.
#define APPLY_TONEMAPPING

#include "../tonemap_inc.glsl"

#ifdef USE_MULTIVIEW
uniform sampler2DArray source_color; // texunit:0
#else
uniform sampler2D source_color; // texunit:0
#endif // USE_MULTIVIEW

uniform float view;
uniform float luminance_multiplier;

#ifdef USE_GLOW
uniform sampler2D glow_color; // texunit:1
uniform vec2 pixel_size;
uniform float glow_intensity;
uniform float srgb_white;

vec4 get_glow_color(vec2 uv) {
	vec2 half_pixel = pixel_size * 0.5;

	vec4 color = textureLod(glow_color, uv + vec2(-half_pixel.x * 2.0, 0.0), 0.0);
	color += textureLod(glow_color, uv + vec2(-half_pixel.x, half_pixel.y), 0.0) * 2.0;
	color += textureLod(glow_color, uv + vec2(0.0, half_pixel.y * 2.0), 0.0);
	color += textureLod(glow_color, uv + vec2(half_pixel.x, half_pixel.y), 0.0) * 2.0;
	color += textureLod(glow_color, uv + vec2(half_pixel.x * 2.0, 0.0), 0.0);
	color += textureLod(glow_color, uv + vec2(half_pixel.x, -half_pixel.y), 0.0) * 2.0;
	color += textureLod(glow_color, uv + vec2(0.0, -half_pixel.y * 2.0), 0.0);
	color += textureLod(glow_color, uv + vec2(-half_pixel.x, -half_pixel.y), 0.0) * 2.0;

#ifdef USE_LUMINANCE_MULTIPLIER
	color = color / luminance_multiplier;
#endif

	return color / 12.0;
}
#endif // USE_GLOW

#ifdef USE_COLOR_CORRECTION
#ifdef USE_1D_LUT
uniform sampler2D source_color_correction; //texunit:2

vec3 apply_color_correction(vec3 color) {
	color.r = texture(source_color_correction, vec2(color.r, 0.0f)).r;
	color.g = texture(source_color_correction, vec2(color.g, 0.0f)).g;
	color.b = texture(source_color_correction, vec2(color.b, 0.0f)).b;
	return color;
}
#else
uniform sampler3D source_color_correction; //texunit:2

vec3 apply_color_correction(vec3 color) {
	return textureLod(source_color_correction, color, 0.0).rgb;
}
#endif // USE_1D_LUT
#endif // USE_COLOR_CORRECTION

#if defined(USE_SSAO_ABYSS) || defined(USE_SSAO_LOW) || defined(USE_SSAO_MED) || defined(USE_SSAO_HIGH) || defined(USE_SSAO_MEGA)
#define USE_SOME_SSAO
uniform float ssao_intensity;
uniform float ssao_radius_frac;
uniform vec2 ssao_prn_UV;
#ifdef USE_MULTIVIEW
// VR will have 2 depth buffers.
uniform sampler2DArray depth_buffer_array; // texunit:3
#else
uniform sampler2D depth_buffer; // texunit:3
#endif
#if defined(USE_SSAO_ABYSS)
// Use the tiny 2-sample version.
#include "../s4ao_micro_inc.glsl"
#elif defined(USE_SSAO_HIGH) || defined(USE_SSAO_MEGA)
// Use the rings version for the higher qualities.
#include "../s4ao_mega_inc.glsl"
#else
// Use the more generic NxN grid version.
#include "../s4ao_inc.glsl"
#endif
#endif

in vec2 uv_interp;

layout(location = 0) out vec4 frag_color;

void main() {
#ifdef USE_MULTIVIEW
	vec4 color = texture(source_color, vec3(uv_interp, view));
#else
	vec4 color = texture(source_color, uv_interp);
#endif

#ifdef USE_LUMINANCE_MULTIPLIER
	color = color / luminance_multiplier;
#endif

#ifdef USE_GLOW
	// Glow blending is performed before srgb_to_linear because
	// the glow texture was created from a nonlinear sRGB-encoded
	// scene, so it only makes sense to add this glow to an equally
	// nonlinear sRGB-encoded scene.

	vec4 glow = get_glow_color(uv_interp) * glow_intensity;

	// Glow always uses the screen blend mode in the Compatibility renderer:

	// Glow cannot be above 1.0 after normalizing and should be non-negative
	// to produce expected results. It is possible that glow can be negative
	// if negative lights were used in the scene.
	// We clamp to srgb_white because glow will be normalized to this range.
	// Note: srgb_white cannot be smaller than the maximum output value (1.0).
	glow.rgb = clamp(glow.rgb, 0.0, srgb_white);

	// Normalize to srgb_white range.
	//glow.rgb /= srgb_white;
	//color.rgb /= srgb_white;
	//color.rgb = (color.rgb + glow.rgb) - (color.rgb * glow.rgb);
	// Expand back to original range.
	//color.rgb *= srgb_white;

	// The following is a mathematically simplified version of the above.
	color.rgb = color.rgb + glow.rgb - (color.rgb * glow.rgb / srgb_white);
#endif // USE_GLOW

	color.rgb = srgb_to_linear(color.rgb);

#if defined(USE_SOME_SSAO)
	// Putting SSAO after the conversion to linear color, though it might be better before the glow.
	color.rgb *= s4ao(uv_interp); // The USE_SSAO_X controls the number of samples.
#endif

	color.rgb = apply_tonemapping(color.rgb);

#ifdef USE_BCS
	// Apply brightness:
	// Apply to relative luminance. This ensures that the hue and saturation of
	// colors is not affected by the adjustment, but requires the multiplication
	// to be performed on linear-encoded values.
	color.rgb = color.rgb * brightness;

	color.rgb = linear_to_srgb(color.rgb);

	// Apply contrast:
	// By applying contrast to RGB values that are perceptually uniform (nonlinear),
	// the darkest values are not hard-clipped as badly, which produces a
	// higher quality contrast adjustment and maintains compatibility with
	// existing projects.
	color.rgb = mix(vec3(0.5), color.rgb, contrast);

	// Apply saturation:
	// By applying saturation adjustment to nonlinear sRGB-encoded values with
	// even weights the preceived brightness of blues are affected, but this
	// maintains compatibility with existing projects.
	color.rgb = mix(vec3(dot(vec3(1.0), color.rgb) * (1.0 / 3.0)), color.rgb, saturation);
#else
	color.rgb = linear_to_srgb(color.rgb);
#endif // USE_BCS

#ifdef USE_COLOR_CORRECTION
	color.rgb = apply_color_correction(color.rgb);
#endif

	frag_color = color;
}
