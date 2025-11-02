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
layout(constant_id = 0) const bool use_debanding = false;
layout(constant_id = 1) const bool use_blend_color = false;

// From https://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
// and https://www.shadertoy.com/view/MslGR8 (5th one starting from the bottom)
// NOTE: `frag_coord` is in pixels (i.e. not normalized UV).
// This dithering must be applied after encoding changes (linear/nonlinear) have been applied
// as the final step before quantization from floating point to integer values.
vec3 screen_space_dither(vec2 frag_coord, float bit_alignment_diviser) {
	// Iestyn's RGB dither (7 asm instructions) from Portal 2 X360, slightly modified for VR.
	// Removed the time component to avoid passing time into this shader.
	vec3 dither = vec3(dot(vec2(171.0, 231.0), frag_coord));
	dither.rgb = fract(dither.rgb / vec3(103.0, 71.0, 97.0));

	// Subtract 0.5 to avoid slightly brightening the whole viewport.
	// Use a dither strength of 100% rather than the 37.5% suggested by the original source.
	return (dither.rgb - 0.5) / bit_alignment_diviser;
}
#endif

layout(location = 0) out vec4 frag_color;

#ifdef MODE_GLOW_DOWNSAMPLE

// https://www.shadertoy.com/view/mdsyDf
vec4 BloomDownKernel4(sampler2D Tex, vec2 uv0) {
	vec2 RcpSrcTexRes = blur.source_pixel_size;

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

// https://www.shadertoy.com/view/mdsyDf
vec4 BloomUpKernel4(sampler2D Tex, vec2 uv0) {
	vec2 RcpSrcTexRes = blur.source_pixel_size;

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

	vec2 pix_size = blur.dest_pixel_size;
	vec4 color = texture(source_color, uv_interp + vec2(-0.5, -0.5) * pix_size);
	color += texture(source_color, uv_interp + vec2(0.5, -0.5) * pix_size);
	color += texture(source_color, uv_interp + vec2(0.5, 0.5) * pix_size);
	color += texture(source_color, uv_interp + vec2(-0.5, 0.5) * pix_size);
	frag_color = color / 4.0;

#endif

#ifdef MODE_GAUSSIAN_BLUR

	// For Gaussian Blur we use 13 taps in a single pass instead of 12 taps over 2 passes.
	// This minimizes the number of times we change framebuffers which is very important for mobile.
	// Source: http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
	vec4 A = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(-1.0, -1.0));
	vec4 B = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(0.0, -1.0));
	vec4 C = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(1.0, -1.0));
	vec4 D = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(-0.5, -0.5));
	vec4 E = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(0.5, -0.5));
	vec4 F = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(-1.0, 0.0));
	vec4 G = texture(source_color, uv_interp);
	vec4 H = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(1.0, 0.0));
	vec4 I = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(-0.5, 0.5));
	vec4 J = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(0.5, 0.5));
	vec4 K = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(-1.0, 1.0));
	vec4 L = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(0.0, 1.0));
	vec4 M = texture(source_color, uv_interp + blur.dest_pixel_size * vec2(1.0, 1.0));

	float base_weight = 0.5 / 4.0;
	float lesser_weight = 0.125 / 4.0;

	frag_color = (D + E + I + J) * base_weight;
	frag_color += (A + B + G + F) * lesser_weight;
	frag_color += (B + C + H + G) * lesser_weight;
	frag_color += (F + G + L + K) * lesser_weight;
	frag_color += (G + H + M + L) * lesser_weight;
#endif

#ifdef MODE_GLOW_GATHER
	// First step, go straight to quarter resolution.
	// Don't apply blur, but include thresholding.

	vec2 block_pos = floor(gl_FragCoord.xy) * 4.0;
	vec2 end = max(1.0 / blur.source_pixel_size - vec2(4.0), vec2(0.0));
	block_pos = clamp(block_pos, vec2(0.0), end);

	// We skipped a level, so gather 16 closest samples now.

	vec4 color = textureLod(source_color, (block_pos + vec2(1.0, 1.0)) * blur.source_pixel_size, 0.0);
	color += textureLod(source_color, (block_pos + vec2(1.0, 3.0)) * blur.source_pixel_size, 0.0);
	color += textureLod(source_color, (block_pos + vec2(3.0, 1.0)) * blur.source_pixel_size, 0.0);
	color += textureLod(source_color, (block_pos + vec2(3.0, 3.0)) * blur.source_pixel_size, 0.0);
	frag_color = color * 0.25;

	// Apply strength a second time since it usually gets added at each level.
	frag_color *= blur.glow_strength;
	frag_color *= blur.glow_strength;

	// In the first pass bring back to correct color range else we're applying the wrong threshold
	// in subsequent passes we can use it as is as we'd just be undoing it right after.
	frag_color *= blur.luminance_multiplier;
	frag_color *= blur.glow_exposure;

	float luminance = max(frag_color.r, max(frag_color.g, frag_color.b));
	float feedback = max(smoothstep(blur.glow_hdr_threshold, blur.glow_hdr_threshold + blur.glow_hdr_scale, luminance), blur.glow_bloom);

	frag_color = min(frag_color * feedback, vec4(blur.glow_luminance_cap)) / blur.luminance_multiplier;
#endif // MODE_GLOW_GATHER_WIDE

#ifdef MODE_GLOW_DOWNSAMPLE
	// Regular downsample, apply a simple blur.
	frag_color = BloomDownKernel4(source_color, floor(gl_FragCoord.xy));
	frag_color *= blur.glow_strength;
#endif // MODE_GLOW_DOWNSAMPLE

#ifdef MODE_GLOW_UPSAMPLE

	frag_color = BloomUpKernel4(source_color, floor(gl_FragCoord.xy)) * blur.glow_strength; // "glow_strength" here is actually the glow level. It is always 1.0, except for the first upsample where we need to apply the level to two textures at once.
	if (use_blend_color) {
		vec2 uv = floor(gl_FragCoord.xy) + 0.5;
		frag_color += textureLod(blend_color, uv * blur.dest_pixel_size, 0.0) * blur.glow_level;
	}

	if (use_debanding) {
		frag_color.rgb += screen_space_dither(gl_FragCoord.xy, 1023.0);
	}
#endif // MODE_GLOW_UPSAMPLE

#ifdef MODE_COPY
	vec4 color = textureLod(source_color, uv_interp, 0.0);
	frag_color = color;
#endif
}
