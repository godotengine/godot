/* clang-format off */
[vertex]

layout(location = 0) in highp vec4 vertex_attrib;
/* clang-format on */
layout(location = 4) in vec2 uv_in;

out vec2 uv_interp;

void main() {
	gl_Position = vertex_attrib;

	uv_interp = uv_in;

#ifdef V_FLIP
	uv_interp.y = 1.0f - uv_interp.y;
#endif
}

/* clang-format off */
[fragment]

#if !defined(GLES_OVER_GL)
precision mediump float;
#endif
/* clang-format on */

in vec2 uv_interp;

uniform highp sampler2D source; //texunit:0

uniform float exposure;
uniform float white;

#ifdef USE_AUTO_EXPOSURE
uniform highp sampler2D source_auto_exposure; //texunit:1
uniform highp float auto_exposure_grey;
#endif

#if defined(USE_GLOW_LEVEL1) || defined(USE_GLOW_LEVEL2) || defined(USE_GLOW_LEVEL3) || defined(USE_GLOW_LEVEL4) || defined(USE_GLOW_LEVEL5) || defined(USE_GLOW_LEVEL6) || defined(USE_GLOW_LEVEL7)
#define USING_GLOW // only use glow when at least one glow level is selected

uniform highp sampler2D source_glow; //texunit:2
uniform highp float glow_intensity;
#endif

#ifdef USE_BCS
uniform vec3 bcs;
#endif

#ifdef USE_FXAA
uniform vec2 pixel_size;
#endif

#ifdef USE_COLOR_CORRECTION
uniform sampler2D color_correction; //texunit:3
#endif

layout(location = 0) out vec4 frag_color;

#ifdef USE_GLOW_FILTER_BICUBIC
// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float w0(float a) {
	return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

float w1(float a) {
	return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

float w2(float a) {
	return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

float w3(float a) {
	return (1.0f / 6.0f) * (a * a * a);
}

// g0 and g1 are the two amplitude functions
float g0(float a) {
	return w0(a) + w1(a);
}

float g1(float a) {
	return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
float h0(float a) {
	return -1.0f + w1(a) / (w0(a) + w1(a));
}

float h1(float a) {
	return 1.0f + w3(a) / (w2(a) + w3(a));
}

uniform ivec2 glow_texture_size;

vec4 texture2D_bicubic(sampler2D tex, vec2 uv, int p_lod) {
	float lod = float(p_lod);
	vec2 tex_size = vec2(glow_texture_size >> p_lod);
	vec2 texel_size = vec2(1.0f) / tex_size;

	uv = uv * tex_size + vec2(0.5f);

	vec2 iuv = floor(uv);
	vec2 fuv = fract(uv);

	float g0x = g0(fuv.x);
	float g1x = g1(fuv.x);
	float h0x = h0(fuv.x);
	float h1x = h1(fuv.x);
	float h0y = h0(fuv.y);
	float h1y = h1(fuv.y);

	vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - vec2(0.5f)) * texel_size;
	vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - vec2(0.5f)) * texel_size;
	vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - vec2(0.5f)) * texel_size;
	vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - vec2(0.5f)) * texel_size;

	return (g0(fuv.y) * (g0x * textureLod(tex, p0, lod) + g1x * textureLod(tex, p1, lod))) +
		   (g1(fuv.y) * (g0x * textureLod(tex, p2, lod) + g1x * textureLod(tex, p3, lod)));
}

#define GLOW_TEXTURE_SAMPLE(m_tex, m_uv, m_lod) texture2D_bicubic(m_tex, m_uv, m_lod)
#else
#define GLOW_TEXTURE_SAMPLE(m_tex, m_uv, m_lod) textureLod(m_tex, m_uv, float(m_lod))
#endif

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

	return clamp(color_tonemapped / white_tonemapped, vec3(0.0f), vec3(1.0f));
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

	return clamp(color_tonemapped / white_tonemapped, vec3(0.0f), vec3(1.0f));
}

vec3 tonemap_reinhard(vec3 color, float white) {
	// Ensure color values are positive.
	// They can be negative in the case of negative lights, which leads to undesired behavior.
	color = max(vec3(0.0), color);

	return clamp((white * color + color) / (color * white + white), vec3(0.0f), vec3(1.0f));
}

vec3 linear_to_srgb(vec3 color) { // convert linear rgb to srgb, assumes clamped input in range [0;1]
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

// inputs are LINEAR, If Linear tonemapping is selected no transform is performed else outputs are clamped [0, 1] color
vec3 apply_tonemapping(vec3 color, float white) {
#ifdef USE_REINHARD_TONEMAPPER
	return tonemap_reinhard(color, white);
#endif

#ifdef USE_FILMIC_TONEMAPPER
	return tonemap_filmic(color, white);
#endif

#ifdef USE_ACES_TONEMAPPER
	return tonemap_aces(color, white);
#endif

	return color; // no other selected -> linear: no color transform applied
}

vec3 gather_glow(sampler2D tex, vec2 uv) { // sample all selected glow levels
	vec3 glow = vec3(0.0f);

#ifdef USE_GLOW_LEVEL1
	glow += GLOW_TEXTURE_SAMPLE(tex, uv, 1).rgb;
#endif

#ifdef USE_GLOW_LEVEL2
	glow += GLOW_TEXTURE_SAMPLE(tex, uv, 2).rgb;
#endif

#ifdef USE_GLOW_LEVEL3
	glow += GLOW_TEXTURE_SAMPLE(tex, uv, 3).rgb;
#endif

#ifdef USE_GLOW_LEVEL4
	glow += GLOW_TEXTURE_SAMPLE(tex, uv, 4).rgb;
#endif

#ifdef USE_GLOW_LEVEL5
	glow += GLOW_TEXTURE_SAMPLE(tex, uv, 5).rgb;
#endif

#ifdef USE_GLOW_LEVEL6
	glow += GLOW_TEXTURE_SAMPLE(tex, uv, 6).rgb;
#endif

#ifdef USE_GLOW_LEVEL7
	glow += GLOW_TEXTURE_SAMPLE(tex, uv, 7).rgb;
#endif

	return glow;
}

vec3 apply_glow(vec3 color, vec3 glow) { // apply glow using the selected blending mode
#ifdef USE_GLOW_REPLACE
	color = glow;
#endif

#ifdef USE_GLOW_SCREEN
	//need color clamping
	color = clamp(color, vec3(0.0f), vec3(1.0f));
	color = max((color + glow) - (color * glow), vec3(0.0));
#endif

#ifdef USE_GLOW_SOFTLIGHT
	//need color clamping
	color = clamp(color, vec3(0.0f), vec3(1.0));
	glow = glow * vec3(0.5f) + vec3(0.5f);

	color.r = (glow.r <= 0.5f) ? (color.r - (1.0f - 2.0f * glow.r) * color.r * (1.0f - color.r)) : (((glow.r > 0.5f) && (color.r <= 0.25f)) ? (color.r + (2.0f * glow.r - 1.0f) * (4.0f * color.r * (4.0f * color.r + 1.0f) * (color.r - 1.0f) + 7.0f * color.r)) : (color.r + (2.0f * glow.r - 1.0f) * (sqrt(color.r) - color.r)));
	color.g = (glow.g <= 0.5f) ? (color.g - (1.0f - 2.0f * glow.g) * color.g * (1.0f - color.g)) : (((glow.g > 0.5f) && (color.g <= 0.25f)) ? (color.g + (2.0f * glow.g - 1.0f) * (4.0f * color.g * (4.0f * color.g + 1.0f) * (color.g - 1.0f) + 7.0f * color.g)) : (color.g + (2.0f * glow.g - 1.0f) * (sqrt(color.g) - color.g)));
	color.b = (glow.b <= 0.5f) ? (color.b - (1.0f - 2.0f * glow.b) * color.b * (1.0f - color.b)) : (((glow.b > 0.5f) && (color.b <= 0.25f)) ? (color.b + (2.0f * glow.b - 1.0f) * (4.0f * color.b * (4.0f * color.b + 1.0f) * (color.b - 1.0f) + 7.0f * color.b)) : (color.b + (2.0f * glow.b - 1.0f) * (sqrt(color.b) - color.b)));
#endif

#if !defined(USE_GLOW_SCREEN) && !defined(USE_GLOW_SOFTLIGHT) && !defined(USE_GLOW_REPLACE) // no other selected -> additive
	color += glow;
#endif

	return color;
}

vec3 apply_bcs(vec3 color, vec3 bcs) {
	color = mix(vec3(0.0f), color, bcs.x);
	color = mix(vec3(0.5f), color, bcs.y);
	color = mix(vec3(dot(vec3(1.0f), color) * 0.33333f), color, bcs.z);

	return color;
}

vec3 apply_color_correction(vec3 color, sampler2D correction_tex) {
	color.r = texture(correction_tex, vec2(color.r, 0.0f)).r;
	color.g = texture(correction_tex, vec2(color.g, 0.0f)).g;
	color.b = texture(correction_tex, vec2(color.b, 0.0f)).b;

	return color;
}

vec3 apply_fxaa(vec3 color, float exposure, vec2 uv_interp, vec2 pixel_size) {
	const float FXAA_REDUCE_MIN = (1.0 / 128.0);
	const float FXAA_REDUCE_MUL = (1.0 / 8.0);
	const float FXAA_SPAN_MAX = 8.0;

	vec3 rgbNW = textureLod(source, uv_interp + vec2(-1.0, -1.0) * pixel_size, 0.0).xyz * exposure;
	vec3 rgbNE = textureLod(source, uv_interp + vec2(1.0, -1.0) * pixel_size, 0.0).xyz * exposure;
	vec3 rgbSW = textureLod(source, uv_interp + vec2(-1.0, 1.0) * pixel_size, 0.0).xyz * exposure;
	vec3 rgbSE = textureLod(source, uv_interp + vec2(1.0, 1.0) * pixel_size, 0.0).xyz * exposure;
	vec3 rgbM = color;
	vec3 luma = vec3(0.299, 0.587, 0.114);
	float lumaNW = dot(rgbNW, luma);
	float lumaNE = dot(rgbNE, luma);
	float lumaSW = dot(rgbSW, luma);
	float lumaSE = dot(rgbSE, luma);
	float lumaM = dot(rgbM, luma);
	float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
	float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

	vec2 dir;
	dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
	dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

	float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
								  (0.25 * FXAA_REDUCE_MUL),
			FXAA_REDUCE_MIN);

	float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
	dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
				  max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
						  dir * rcpDirMin)) *
		  pixel_size;

	vec3 rgbA = 0.5 * exposure * (textureLod(source, uv_interp + dir * (1.0 / 3.0 - 0.5), 0.0).xyz + textureLod(source, uv_interp + dir * (2.0 / 3.0 - 0.5), 0.0).xyz);
	vec3 rgbB = rgbA * 0.5 + 0.25 * exposure * (textureLod(source, uv_interp + dir * -0.5, 0.0).xyz + textureLod(source, uv_interp + dir * 0.5, 0.0).xyz);

	float lumaB = dot(rgbB, luma);
	if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
		return rgbA;
	} else {
		return rgbB;
	}
}

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
	vec3 color = textureLod(source, uv_interp, 0.0f).rgb;

	// Exposure

#ifdef USE_AUTO_EXPOSURE
	color /= texelFetch(source_auto_exposure, ivec2(0, 0), 0).r / auto_exposure_grey;
#endif

	color *= exposure;

#ifdef USE_FXAA
	// FXAA must be applied before tonemapping.
	color = apply_fxaa(color, exposure, uv_interp, pixel_size);
#endif

#ifdef USE_DEBANDING
	// For best results, debanding should be done before tonemapping.
	// Otherwise, we're adding noise to an already-quantized image.
	color += screen_space_dither(gl_FragCoord.xy);
#endif

	// Early Tonemap & SRGB Conversion; note that Linear tonemapping does not clamp to [0, 1]; some operations below expect a [0, 1] range and will clamp

	color = apply_tonemapping(color, white);

#ifdef KEEP_3D_LINEAR
	// leave color as is (-> don't convert to SRGB)
#else
	//need color clamping
	color = clamp(color, vec3(0.0f), vec3(1.0f));
	color = linear_to_srgb(color); // regular linear -> SRGB conversion (needs clamped values)
#endif

	// Glow

#ifdef USING_GLOW
	vec3 glow = gather_glow(source_glow, uv_interp) * glow_intensity;

	// high dynamic range -> SRGB
	glow = apply_tonemapping(glow, white);
	glow = clamp(glow, vec3(0.0f), vec3(1.0f));
	glow = linear_to_srgb(glow);

	color = apply_glow(color, glow);
#endif

	// Additional effects

#ifdef USE_BCS
	color = apply_bcs(color, bcs);
#endif

#ifdef USE_COLOR_CORRECTION
	color = apply_color_correction(color, color_correction);
#endif

	frag_color = vec4(color, 1.0f);
}
