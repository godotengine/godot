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

#ifdef USE_BCS
vec3 apply_bcs(vec3 color) {
	color = mix(vec3(0.0), color, brightness);
	color = mix(vec3(0.5), color, contrast);
	color = mix(vec3(dot(vec3(1.0), color) * 0.33333), color, saturation);

	return color;
}
#endif

in vec2 uv_interp;

layout(location = 0) out vec4 frag_color;

void main() {
#ifdef USE_MULTIVIEW
	vec4 color = texture(source_color, vec3(uv_interp, view));
#else
	vec4 color = texture(source_color, uv_interp);
#endif

#ifdef USE_GLOW
	vec4 glow = get_glow_color(uv_interp) * glow_intensity;

	// Just use softlight...
	glow.rgb = clamp(glow.rgb, vec3(0.0f), vec3(1.0f));
	color.rgb = max((color.rgb + glow.rgb) - (color.rgb * glow.rgb), vec3(0.0));
#endif // USE_GLOW

#ifdef USE_LUMINANCE_MULTIPLIER
	color = color / luminance_multiplier;
#endif

	color.rgb = srgb_to_linear(color.rgb);
	color.rgb = apply_tonemapping(color.rgb, white);
	color.rgb = linear_to_srgb(color.rgb);

#ifdef USE_BCS
	color.rgb = apply_bcs(color.rgb);
#endif

#ifdef USE_COLOR_CORRECTION
	color.rgb = apply_color_correction(color.rgb);
#endif

	frag_color = color;
}
