/**
 * Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
 * Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
 * Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
 * Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
 * Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to
 * do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software. As clarification, there
 * is no requirement that the copyright notice and permission be included in
 * binary distributions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#[vertex]
#version 450

layout(location = 0) out vec2 tex_coord;
layout(location = 1) out vec4 offset;

layout(push_constant, std430) uniform Params {
	vec2 inv_size;
	vec2 pad;
}
params;

void main() {
	vec2 vertex_base;
	if (gl_VertexIndex == 0) {
		vertex_base = vec2(-1.0, -1.0);
	} else if (gl_VertexIndex == 1) {
		vertex_base = vec2(-1.0, 3.0);
	} else {
		vertex_base = vec2(3.0, -1.0);
	}
	gl_Position = vec4(vertex_base, 0.0, 1.0);
	tex_coord = clamp(vertex_base, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
	offset = fma(params.inv_size.xyxy, vec4(1.0, 0.0, 0.0, 1.0), tex_coord.xyxy);
}

#[fragment]
#version 450

layout(location = 0) in vec2 tex_coord;
layout(location = 1) in vec4 offset;
layout(set = 0, binding = 0) uniform sampler2D color_tex;
layout(set = 1, binding = 0) uniform sampler2D blend_tex;

layout(location = 0) out vec4 out_color;

#define FLAG_USE_8_BIT_DEBANDING (1 << 0)
#define FLAG_USE_10_BIT_DEBANDING (1 << 1)

layout(push_constant, std430) uniform Params {
	vec2 inv_size;
	uint use_debanding;
	float pad;
}
params;

#define textureLinear(tex, uv) srgb_to_linear(textureLod(tex, uv, 0.0).rgb)

vec3 linear_to_srgb(vec3 color) {
	// If going to srgb, clamp from 0 to 1.
	color = clamp(color, vec3(0.0), vec3(1.0));
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

vec3 srgb_to_linear(vec3 color) {
	return mix(pow((color.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)), color.rgb * (1.0 / 12.92), lessThan(color.rgb, vec3(0.04045)));
}

void SMAAMovc(bvec2 cond, inout vec2 variable, vec2 value) {
	if (cond.x) {
		variable.x = value.x;
	}
	if (cond.y) {
		variable.y = value.y;
	}
}

void SMAAMovc(bvec4 cond, inout vec4 variable, vec4 value) {
	SMAAMovc(cond.xy, variable.xy, value.xy);
	SMAAMovc(cond.zw, variable.zw, value.zw);
}

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

void main() {
	vec4 a;
	a.x = texture(blend_tex, offset.xy).a;
	a.y = texture(blend_tex, offset.zw).g;
	a.wz = texture(blend_tex, tex_coord).xz;

	if (dot(a, vec4(1.0, 1.0, 1.0, 1.0)) < 1e-5) {
		out_color = textureLod(color_tex, tex_coord, 0.0);
	} else {
		bool h = max(a.x, a.z) > max(a.y, a.w);

		vec4 blending_offset = vec4(0.0, a.y, 0.0, a.w);
		vec2 blending_weight = a.yw;

		SMAAMovc(bvec4(h, h, h, h), blending_offset, vec4(a.x, 0.0, a.z, 0.0));
		SMAAMovc(bvec2(h, h), blending_weight, a.xz);
		blending_weight /= dot(blending_weight, vec2(1.0, 1.0));

		vec4 blending_coord = fma(blending_offset, vec4(params.inv_size.xy, -params.inv_size.xy), tex_coord.xyxy);

		out_color.rgb = blending_weight.x * textureLinear(color_tex, blending_coord.xy);
		out_color.rgb += blending_weight.y * textureLinear(color_tex, blending_coord.zw);
		out_color.rgb = linear_to_srgb(out_color.rgb);
		out_color.a = texture(color_tex, tex_coord).a;
	}
	if (bool(params.use_debanding)) {
		// Divide by 255 to align to 8-bit quantization.
		out_color.rgb += screen_space_dither(gl_FragCoord.xy, 255.0);
	}
}
