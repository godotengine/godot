/* clang-format off */
[vertex]

#ifdef USE_GLES_OVER_GL
#define lowp
#define mediump
#define highp
#else
precision highp float;
precision highp int;
#endif

attribute vec2 vertex_attrib; // attrib:0
/* clang-format on */
attribute vec2 uv_in; // attrib:4

varying vec2 uv_interp;

void main() {
	gl_Position = vec4(vertex_attrib, 0.0, 1.0);

	uv_interp = uv_in;
}

/* clang-format off */
[fragment]


// texture2DLodEXT and textureCubeLodEXT are fragment shader specific.
// Do not copy these defines in the vertex section.
#ifndef USE_GLES_OVER_GL
#ifdef GL_EXT_shader_texture_lod
#extension GL_EXT_shader_texture_lod : enable
#define texture2DLod(img, coord, lod) texture2DLodEXT(img, coord, lod)
#define textureCubeLod(img, coord, lod) textureCubeLodEXT(img, coord, lod)
#endif
#endif // !USE_GLES_OVER_GL

#ifdef GL_ARB_shader_texture_lod
#extension GL_ARB_shader_texture_lod : enable
#endif

#if !defined(GL_EXT_shader_texture_lod) && !defined(GL_ARB_shader_texture_lod)
#define texture2DLod(img, coord, lod) texture2D(img, coord, lod)
#define textureCubeLod(img, coord, lod) textureCube(img, coord, lod)
#endif

// Allows the use of bitshift operators for bicubic upscale
#ifdef GL_EXT_gpu_shader4
#extension GL_EXT_gpu_shader4 : enable
#endif

#ifdef USE_GLES_OVER_GL
#define lowp
#define mediump
#define highp
#else
#if defined(USE_HIGHP_PRECISION)
precision highp float;
precision highp int;
#else
precision mediump float;
precision mediump int;
#endif
#endif

#include "stdlib.glsl"

varying vec2 uv_interp;
/* clang-format on */

uniform highp sampler2D source; //texunit:0

#if defined(USE_GLOW_LEVEL1) || defined(USE_GLOW_LEVEL2) || defined(USE_GLOW_LEVEL3) || defined(USE_GLOW_LEVEL4) || defined(USE_GLOW_LEVEL5) || defined(USE_GLOW_LEVEL6) || defined(USE_GLOW_LEVEL7)
#define USING_GLOW // only use glow when at least one glow level is selected

#ifdef USE_MULTI_TEXTURE_GLOW
uniform highp sampler2D source_glow1; //texunit:2
uniform highp sampler2D source_glow2; //texunit:3
uniform highp sampler2D source_glow3; //texunit:4
uniform highp sampler2D source_glow4; //texunit:5
uniform highp sampler2D source_glow5; //texunit:6
uniform highp sampler2D source_glow6; //texunit:7
#ifdef USE_GLOW_LEVEL7
uniform highp sampler2D source_glow7; //texunit:8
#endif
#else
uniform highp sampler2D source_glow; //texunit:2
#endif
uniform highp float glow_intensity;
#endif

#ifdef USE_BCS
uniform vec3 bcs;
#endif

#ifdef USE_FXAA
uniform vec2 pixel_size;
#endif

#ifdef USE_SHARPENING
uniform float sharpen_intensity;
#endif

#ifdef USE_COLOR_CORRECTION
uniform sampler2D color_correction; //texunit:1
#endif

#ifdef GL_EXT_gpu_shader4
#ifdef USE_GLOW_FILTER_BICUBIC
// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float w0(float a) {
	return (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0);
}

float w1(float a) {
	return (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0);
}

float w2(float a) {
	return (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0);
}

float w3(float a) {
	return (1.0 / 6.0) * (a * a * a);
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
	return -1.0 + w1(a) / (w0(a) + w1(a));
}

float h1(float a) {
	return 1.0 + w3(a) / (w2(a) + w3(a));
}

uniform ivec2 glow_texture_size;

vec4 texture2D_bicubic(sampler2D tex, vec2 uv, int p_lod) {
	float lod = float(p_lod);
	vec2 tex_size = vec2(glow_texture_size >> p_lod);
	vec2 texel_size = vec2(1.0) / tex_size;

	uv = uv * tex_size + vec2(0.5);

	vec2 iuv = floor(uv);
	vec2 fuv = fract(uv);

	float g0x = g0(fuv.x);
	float g1x = g1(fuv.x);
	float h0x = h0(fuv.x);
	float h1x = h1(fuv.x);
	float h0y = h0(fuv.y);
	float h1y = h1(fuv.y);

	vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - vec2(0.5)) * texel_size;
	vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - vec2(0.5)) * texel_size;
	vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - vec2(0.5)) * texel_size;
	vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - vec2(0.5)) * texel_size;

	return (g0(fuv.y) * (g0x * texture2DLod(tex, p0, lod) + g1x * texture2DLod(tex, p1, lod))) +
			(g1(fuv.y) * (g0x * texture2DLod(tex, p2, lod) + g1x * texture2DLod(tex, p3, lod)));
}

#define GLOW_TEXTURE_SAMPLE(m_tex, m_uv, m_lod) texture2D_bicubic(m_tex, m_uv, m_lod)
#else //!USE_GLOW_FILTER_BICUBIC
#define GLOW_TEXTURE_SAMPLE(m_tex, m_uv, m_lod) texture2DLod(m_tex, m_uv, float(m_lod))
#endif //USE_GLOW_FILTER_BICUBIC

#else //!GL_EXT_gpu_shader4
#define GLOW_TEXTURE_SAMPLE(m_tex, m_uv, m_lod) texture2DLod(m_tex, m_uv, float(m_lod))
#endif //GL_EXT_gpu_shader4

vec4 apply_glow(vec4 color, vec3 glow) { // apply glow using the selected blending mode
#ifdef USE_GLOW_REPLACE
	color.rgb = glow;
#endif

#ifdef USE_GLOW_SCREEN
	color.rgb = max((color.rgb + glow) - (color.rgb * glow), vec3(0.0));
#endif

#ifdef USE_GLOW_SOFTLIGHT
	glow = glow * vec3(0.5) + vec3(0.5);

	color.r = (glow.r <= 0.5) ? (color.r - (1.0 - 2.0 * glow.r) * color.r * (1.0 - color.r)) : (((glow.r > 0.5) && (color.r <= 0.25)) ? (color.r + (2.0 * glow.r - 1.0) * (4.0 * color.r * (4.0 * color.r + 1.0) * (color.r - 1.0) + 7.0 * color.r)) : (color.r + (2.0 * glow.r - 1.0) * (sqrt(color.r) - color.r)));
	color.g = (glow.g <= 0.5) ? (color.g - (1.0 - 2.0 * glow.g) * color.g * (1.0 - color.g)) : (((glow.g > 0.5) && (color.g <= 0.25)) ? (color.g + (2.0 * glow.g - 1.0) * (4.0 * color.g * (4.0 * color.g + 1.0) * (color.g - 1.0) + 7.0 * color.g)) : (color.g + (2.0 * glow.g - 1.0) * (sqrt(color.g) - color.g)));
	color.b = (glow.b <= 0.5) ? (color.b - (1.0 - 2.0 * glow.b) * color.b * (1.0 - color.b)) : (((glow.b > 0.5) && (color.b <= 0.25)) ? (color.b + (2.0 * glow.b - 1.0) * (4.0 * color.b * (4.0 * color.b + 1.0) * (color.b - 1.0) + 7.0 * color.b)) : (color.b + (2.0 * glow.b - 1.0) * (sqrt(color.b) - color.b)));
#endif

#if !defined(USE_GLOW_SCREEN) && !defined(USE_GLOW_SOFTLIGHT) && !defined(USE_GLOW_REPLACE) // no other selected -> additive
	color.rgb += glow;
#endif

#ifndef USE_GLOW_SOFTLIGHT // softlight has no effect on black color
	// compute the alpha from glow
	float a = max(max(glow.r, glow.g), glow.b);
	color.a = a + color.a * (1.0 - a);
	if (color.a == 0.0) {
		color.rgb = vec3(0.0);
	} else if (color.a < 1.0) {
		color.rgb /= color.a;
	}
#endif

	return color;
}

vec3 apply_bcs(vec3 color, vec3 bcs) {
	color = mix(vec3(0.0), color, bcs.x);
	color = mix(vec3(0.5), color, bcs.y);
	color = mix(vec3(dot(vec3(1.0), color) * 0.33333), color, bcs.z);

	return color;
}

vec3 apply_color_correction(vec3 color, sampler2D correction_tex) {
	color.r = texture2D(correction_tex, vec2(color.r, 0.0)).r;
	color.g = texture2D(correction_tex, vec2(color.g, 0.0)).g;
	color.b = texture2D(correction_tex, vec2(color.b, 0.0)).b;

	return color;
}

vec4 apply_fxaa(vec4 color, vec2 uv_interp, vec2 pixel_size) {
	const float FXAA_REDUCE_MIN = (1.0 / 128.0);
	const float FXAA_REDUCE_MUL = (1.0 / 8.0);
	const float FXAA_SPAN_MAX = 8.0;
	const vec3 luma = vec3(0.299, 0.587, 0.114);

	vec4 rgbNW = texture2DLod(source, uv_interp + vec2(-0.5, -0.5) * pixel_size, 0.0);
	vec4 rgbNE = texture2DLod(source, uv_interp + vec2(0.5, -0.5) * pixel_size, 0.0);
	vec4 rgbSW = texture2DLod(source, uv_interp + vec2(-0.5, 0.5) * pixel_size, 0.0);
	vec4 rgbSE = texture2DLod(source, uv_interp + vec2(0.5, 0.5) * pixel_size, 0.0);
	vec3 rgbM = color.rgb;

#ifdef DISABLE_ALPHA
	float lumaNW = dot(rgbNW.rgb, luma);
	float lumaNE = dot(rgbNE.rgb, luma);
	float lumaSW = dot(rgbSW.rgb, luma);
	float lumaSE = dot(rgbSE.rgb, luma);
	float lumaM = dot(rgbM, luma);
#else
	float lumaNW = dot(rgbNW.rgb, luma) - ((1.0 - rgbNW.a) / 8.0);
	float lumaNE = dot(rgbNE.rgb, luma) - ((1.0 - rgbNE.a) / 8.0);
	float lumaSW = dot(rgbSW.rgb, luma) - ((1.0 - rgbSW.a) / 8.0);
	float lumaSE = dot(rgbSE.rgb, luma) - ((1.0 - rgbSE.a) / 8.0);
	float lumaM = dot(rgbM, luma) - (color.a / 8.0);
#endif

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

	vec4 rgbA = 0.5 * (texture2DLod(source, uv_interp + dir * (1.0 / 3.0 - 0.5), 0.0) + texture2DLod(source, uv_interp + dir * (2.0 / 3.0 - 0.5), 0.0));
	vec4 rgbB = rgbA * 0.5 + 0.25 * (texture2DLod(source, uv_interp + dir * -0.5, 0.0) + texture2DLod(source, uv_interp + dir * 0.5, 0.0));

#ifdef DISABLE_ALPHA
	float lumaB = dot(rgbB.rgb, luma);
	vec4 color_output = ((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB;
	return vec4(color_output.rgb, 1.0);
#else
	float lumaB = dot(rgbB.rgb, luma) - ((1.0 - rgbB.a) / 8.0);
	vec4 color_output = ((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB;
	if (color_output.a == 0.0) {
		color_output.rgb = vec3(0.0);
	} else if (color_output.a < 1.0) {
		color_output.rgb /= color_output.a;
	}
	return color_output;
#endif
}

void main() {
	vec4 color = texture2DLod(source, uv_interp, 0.0);

#ifdef DISABLE_ALPHA
	color.a = 1.0;
#endif

#ifdef USE_FXAA
	color = apply_fxaa(color, uv_interp, pixel_size);
#endif

	// Glow

#ifdef USING_GLOW
	vec3 glow = vec3(0.0);
#ifdef USE_MULTI_TEXTURE_GLOW
#ifdef USE_GLOW_LEVEL1
	glow += GLOW_TEXTURE_SAMPLE(source_glow1, uv_interp, 0).rgb;
#ifdef USE_GLOW_LEVEL2
	glow += GLOW_TEXTURE_SAMPLE(source_glow2, uv_interp, 0).rgb;
#ifdef USE_GLOW_LEVEL3
	glow += GLOW_TEXTURE_SAMPLE(source_glow3, uv_interp, 0).rgb;
#ifdef USE_GLOW_LEVEL4
	glow += GLOW_TEXTURE_SAMPLE(source_glow4, uv_interp, 0).rgb;
#ifdef USE_GLOW_LEVEL5
	glow += GLOW_TEXTURE_SAMPLE(source_glow5, uv_interp, 0).rgb;
#ifdef USE_GLOW_LEVEL6
	glow += GLOW_TEXTURE_SAMPLE(source_glow6, uv_interp, 0).rgb;
#ifdef USE_GLOW_LEVEL7
	glow += GLOW_TEXTURE_SAMPLE(source_glow7, uv_interp, 0).rgb;
#endif
#endif
#endif
#endif
#endif
#endif
#endif

#else

#ifdef USE_GLOW_LEVEL1
	glow += GLOW_TEXTURE_SAMPLE(source_glow, uv_interp, 1).rgb;
#endif

#ifdef USE_GLOW_LEVEL2
	glow += GLOW_TEXTURE_SAMPLE(source_glow, uv_interp, 2).rgb;
#endif

#ifdef USE_GLOW_LEVEL3
	glow += GLOW_TEXTURE_SAMPLE(source_glow, uv_interp, 3).rgb;
#endif

#ifdef USE_GLOW_LEVEL4
	glow += GLOW_TEXTURE_SAMPLE(source_glow, uv_interp, 4).rgb;
#endif

#ifdef USE_GLOW_LEVEL5
	glow += GLOW_TEXTURE_SAMPLE(source_glow, uv_interp, 5).rgb;
#endif

#ifdef USE_GLOW_LEVEL6
	glow += GLOW_TEXTURE_SAMPLE(source_glow, uv_interp, 6).rgb;
#endif

#ifdef USE_GLOW_LEVEL7
	glow += GLOW_TEXTURE_SAMPLE(source_glow, uv_interp, 7).rgb;
#endif
#endif //USE_MULTI_TEXTURE_GLOW

	glow *= glow_intensity;
	color = apply_glow(color, glow);
#endif

	// Additional effects

#ifdef USE_BCS
	color.rgb = apply_bcs(color.rgb, bcs);
#endif

#ifdef USE_COLOR_CORRECTION
	color.rgb = apply_color_correction(color.rgb, color_correction);
#endif

#ifdef DISABLE_ALPHA
	color.a = 1.0;
#endif

	gl_FragColor = color;
}
