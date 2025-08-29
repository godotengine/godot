/* clang-format off */
#[vertex]

#version 450

#VERSION_DEFINES

#include "blur_raster_inc.glsl"

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

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

/* clang-format off */
#[fragment]

#version 450

#VERSION_DEFINES

#include "blur_raster_inc.glsl"

layout(location = 0) in vec2 uv_interp;
/* clang-format on */

layout(set = 0, binding = 0) uniform sampler2D source_color;

#ifdef MODE_GLOW_UPSAMPLE
// When upsampling this is original downsampled texture, not the blended upsampled texture.
layout(set = 1, binding = 0) uniform sampler2D blend_color;
layout(constant_id = 0) const bool use_blend_color = false;
#endif

layout(location = 0) out vec4 frag_color;

#ifdef MODE_GLOW_DOWNSAMPLE

// https://www.shadertoy.com/view/mdsyDf
vec4 BloomDownKernel4(sampler2D Tex, vec2 uv0) {
	vec2 RcpSrcTexRes = blur.pixel_size;

	vec2 tc = (uv0 * 2.0 + 1.0) * RcpSrcTexRes;

	float la = 1.0 / 4.0;

	vec2 o = (0.5 + la) * RcpSrcTexRes;

	vec4 c = vec4(0.0);
	c += textureLod(Tex, tc + vec2(-1.0, -1.0) * o, 0.0) * 0.25;
	c += textureLod(Tex, tc + vec2(1.0, -1.0) * o, 0.0) * 0.25;
	c += textureLod(Tex, tc + vec2(-1.0, 1.0) * o, 0.0) * 0.25;
	c += textureLod(Tex, tc + vec2(1.0, 1.0) * o, 0.0) * 0.25;

	return c;
}

#endif

#ifdef MODE_GLOW_UPSAMPLE

vec4 BloomUpKernel4B(sampler2D Tex, vec2 uv0) {
	vec2 RcpSrcTexRes = blur.pixel_size;

	vec2 uv = uv0;
	//uv += 0.5;

	vec2 uvI = floor(uv);
	vec2 uvF = uv - uvI;

	vec2 tc = uvI * RcpSrcTexRes.xy;

#if 1
	vec2 l00 = vec2(11.0 / 32.0, 17.0 / 32.0); // 0.34375  ~ 0.347209
	vec2 l10 = vec2(7.0 / 64.0, 11.0 / 32.0); // 0.109375 ~ 0.109840
	vec2 l01 = vec2(11.0 / 32.0, 7.0 / 64.0); // 0.34375  ~ 0.334045
	vec2 l11 = vec2(17.0 / 32.0, 11.0 / 32.0); // 0.53125  ~ 0.526425
#else
	vec2 l00 = vec2(0.347209, 0.526425);
	vec2 l10 = vec2(0.109840, 0.334045);
	vec2 l01 = vec2(0.334045, 0.109840);
	vec2 l11 = vec2(0.526425, 0.347209);
#endif

	vec4 w = vec4(0.288971, 0.211029, 0.211029, 0.288971);

	bool flipX = uvF.x != 0.0;
	bool flipY = uvF.y != 0.0;

	if (flipX) {
		vec2 tmp = l11;
		l11 = l10;
		l10 = tmp;

		l00.x = 1.0 - l00.x;
		l10.x = 1.0 - l10.x;
		l01.x = 1.0 - l01.x;
		l11.x = 1.0 - l11.x;

		w = vec4(w.x, w.w, w.z, w.y);
	}

	if (flipY) {
		vec2 tmp = l00;
		l00 = l01;
		l01 = tmp;

		l00.y = 1.0 - l00.y;
		l10.y = 1.0 - l10.y;
		l01.y = 1.0 - l01.y;
		l11.y = 1.0 - l11.y;

		w = vec4(w.z, w.y, w.x, w.w);
	}

	vec4 col = vec4(0.0);

	col += textureLod(Tex, tc + (vec2(-0.5, -1.5) + l00) * RcpSrcTexRes, 0.0) * w.x;
	col += textureLod(Tex, tc + (vec2(0.5, -0.5) + l10) * RcpSrcTexRes, 0.0) * w.y;
	col += textureLod(Tex, tc + (vec2(-0.5, 0.5) + l01) * RcpSrcTexRes, 0.0) * w.z;
	col += textureLod(Tex, tc + (vec2(-1.5, -0.5) + l11) * RcpSrcTexRes, 0.0) * w.w;

	return col;
}

vec4 BloomUpKernel4(sampler2D Tex, vec2 uv0) {
	vec2 RcpSrcTexRes = blur.pixel_size * 2.0;

	vec2 uv = uv0 * 0.5 + 0.5;

	vec2 uvI = floor(uv);
	vec2 uvF = uv - uvI;

	vec2 tc = uvI * RcpSrcTexRes.xy;

	// optimal stop-band
	float lw = 0.357386;
	float la = 25.0 / 32.0; // 0.78125  ~ 0.779627;
	float lb = 3.0 / 64.0; // 0.046875 ~ 0.0493871;

	vec2 l = vec2(-1.5 + la, 0.5 + lb);

	vec2 lx = uvF.x == 0.0 ? l.xy : -l.yx;
	vec2 ly = uvF.y == 0.0 ? l.xy : -l.yx;

	lx *= RcpSrcTexRes.xx;
	ly *= RcpSrcTexRes.yy;

	vec4 c00 = textureLod(Tex, tc + vec2(lx.x, ly.x), 0.0);
	vec4 c10 = textureLod(Tex, tc + vec2(lx.y, ly.x), 0.0);
	vec4 c01 = textureLod(Tex, tc + vec2(lx.x, ly.y), 0.0);
	vec4 c11 = textureLod(Tex, tc + vec2(lx.y, ly.y), 0.0);

	vec2 w = abs(uvF * 2.0 - lw);

	vec4 cx0 = c00 * (1.0 - w.x) + (c10 * w.x);
	vec4 cx1 = c01 * (1.0 - w.x) + (c11 * w.x);

	vec4 cxy = cx0 * (1.0 - w.y) + (cx1 * w.y);

	return cxy;
}

#endif // MODE_GLOW_UPSAMPLE

void main() {
	// We do not apply our color scale for our mobile renderer here, we'll leave our colors at half brightness and apply scale in the tonemap raster.

#ifdef MODE_MIPMAP
	frag_color = texture(source_color, uv_interp);
#endif

#ifdef MODE_GAUSSIAN_BLUR

	// For Gaussian Blur we use 13 taps in a single pass instead of 12 taps over 2 passes.
	// This minimizes the number of times we change framebuffers which is very important for mobile.
	// Source: http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
	vec4 A = texture(source_color, uv_interp + blur.pixel_size * vec2(-1.0, -1.0));
	vec4 B = texture(source_color, uv_interp + blur.pixel_size * vec2(0.0, -1.0));
	vec4 C = texture(source_color, uv_interp + blur.pixel_size * vec2(1.0, -1.0));
	vec4 D = texture(source_color, uv_interp + blur.pixel_size * vec2(-0.5, -0.5));
	vec4 E = texture(source_color, uv_interp + blur.pixel_size * vec2(0.5, -0.5));
	vec4 F = texture(source_color, uv_interp + blur.pixel_size * vec2(-1.0, 0.0));
	vec4 G = texture(source_color, uv_interp);
	vec4 H = texture(source_color, uv_interp + blur.pixel_size * vec2(1.0, 0.0));
	vec4 I = texture(source_color, uv_interp + blur.pixel_size * vec2(-0.5, 0.5));
	vec4 J = texture(source_color, uv_interp + blur.pixel_size * vec2(0.5, 0.5));
	vec4 K = texture(source_color, uv_interp + blur.pixel_size * vec2(-1.0, 1.0));
	vec4 L = texture(source_color, uv_interp + blur.pixel_size * vec2(0.0, 1.0));
	vec4 M = texture(source_color, uv_interp + blur.pixel_size * vec2(1.0, 1.0));

	float base_weight = 0.5 / 4.0;
	float lesser_weight = 0.125 / 4.0;

	frag_color = (D + E + I + J) * base_weight;
	frag_color += (A + B + G + F) * lesser_weight;
	frag_color += (B + C + H + G) * lesser_weight;
	frag_color += (F + G + L + K) * lesser_weight;
	frag_color += (G + H + M + L) * lesser_weight;
#endif

#ifdef MODE_GLOW_DOWNSAMPLE

#ifdef MODE_GLOW_GATHER
	// Simple mip-like gather, just average 4 closest samples.
	// Source size in texels (mip 0)
	ivec2 srcSize_i = textureSize(source_color, 0);
	vec2 srcSize = vec2(srcSize_i);

	// Destination pixel index computed from gl_FragCoord.
	// gl_FragCoord is at pixel center (0.5, 0.5) for the first pixel, so floor gives integer pixel indices 0..w-1
	ivec2 dstPixel = ivec2(floor(gl_FragCoord.xy));

	// Map destination pixel to the top-left texel index of the corresponding 2x2 block in source:
	// each dest pixel covers [dstPixel*2 .. dstPixel*2 + 1] in src texel indices
	vec2 blockStart = vec2(dstPixel) * 2.0;

	// Clamp blockStart so the 2x2 block stays inside the source texture.
	// We clamp to [0, srcSize - 2], so blockStart + (0..1) is valid.
	vec2 maxStart = max(srcSize - vec2(2.0), vec2(0.0));
	blockStart = clamp(blockStart, vec2(0.0), maxStart);

	// The exact center of the 2x2 texel block (in texel coordinates)
	// Texel centers are at i + 0.5; the four centers are at (s+0.5, s+1.5) etc.
	// The block center lies halfway between them: s + 1.0
	vec2 sampleTexel = blockStart + vec2(1.0, 1.0);

	// Convert to normalized UV for texture()
	vec2 sampleUV = sampleTexel / srcSize;
	frag_color = textureLod(source_color, sampleUV, 0.0);

#else
#ifdef MODE_GLOW_GATHER_WIDE
	// We skipped a level, so gather 16 closest samples now.
	vec4 color = texture(source_color, uv_interp + vec2(-1.0, -1.0) * blur.pixel_size);
	color += texture(source_color, uv_interp + vec2(1.0, -1.0) * blur.pixel_size);
	color += texture(source_color, uv_interp + vec2(1.0, 1.0) * blur.pixel_size);
	color += texture(source_color, uv_interp + vec2(-1.0, 1.0) * blur.pixel_size);
	frag_color = color * 0.25;

	// Apply strength a second time since it usually gets added at each level.
	frag_color *= blur.glow_strength;
#else
	// Regular downsample.
	frag_color = BloomDownKernel4(source_color, floor(gl_FragCoord.xy));
#endif // MODE_GLOW_GATHER_WIDE

#endif // MODE_GLOW_GATHER

	frag_color *= blur.glow_strength;

	if (bool(blur.flags & FLAG_GLOW_FIRST_PASS)) {
		// In the first pass bring back to correct color range else we're applying the wrong threshold
		// in subsequent passes we can use it as is as we'd just be undoing it right after.
		frag_color *= blur.luminance_multiplier;
		frag_color *= blur.glow_exposure;

		float luminance = max(frag_color.r, max(frag_color.g, frag_color.b));
		float feedback = max(smoothstep(blur.glow_hdr_threshold, blur.glow_hdr_threshold + blur.glow_hdr_scale, luminance), blur.glow_bloom);

		frag_color = min(frag_color * feedback, vec4(blur.glow_luminance_cap)) / blur.luminance_multiplier;
	}
#endif // MODE_GLOW_DOWNSAMPLE

#ifdef MODE_GLOW_UPSAMPLE

	frag_color = BloomUpKernel4(source_color, floor(gl_FragCoord.xy)) * blur.glow_strength; // "glow_strength" here is actually the glow level. It is always 1.0, except for the first upsample where we need to apply the level to two textures at once.
	if (use_blend_color) {
		vec2 uv = floor(gl_FragCoord.xy) + 0.5;
		frag_color += textureLod(blend_color, uv * blur.pixel_size, 0.0) * blur.glow_level;
	}

#endif // MODE_GLOW_UPSAMPLE

#ifdef MODE_COPY
	vec4 color = textureLod(source_color, uv_interp, 0.0);
	frag_color = color;
#endif
}
