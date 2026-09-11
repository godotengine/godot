///////////////////////////////////////////////////////////////////////////////////
// Copyright(c) 2016-2022 Panos Karabelas
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////
// File changes (yyyy-mm-dd)
// 2026-07-14: Rene Prašnikar: Total shader rewrite, simplified the shader, new anti-ghosting strategy introduced
// 2025-11-05: Jakub Brzyski: Added dynamic variance, base variance value adjusted to reduce ghosting
// 2022-05-06: Panos Karabelas: first commit
// 2020-12-05: Joan Fons: convert to Vulkan and Godot
///////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

#define FLT_MIN 0.00000001
#define FLT_MAX 32767.0
#define RPC_9 0.11111111111
#define RPC_16 0.0625

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D color_buffer;
layout(set = 0, binding = 1) uniform sampler2D depth_buffer;
layout(set = 0, binding = 2) uniform sampler2D velocity_buffer;
layout(set = 0, binding = 3) uniform sampler2D history_buffer;
layout(rgba16f, set = 0, binding = 4) uniform restrict writeonly image2D output_buffer;

layout(push_constant, std430) uniform Params {
	vec2 resolution;
	vec2 jitter;
}
params;

const ivec2 numpad[10] = {
	ivec2(0, 0),
	ivec2(-1, -1),
	ivec2(0, -1),
	ivec2(1, -1),
	ivec2(-1, 0),
	ivec2(0, 0),
	ivec2(1, 0),
	ivec2(-1, 1),
	ivec2(0, 1),
	ivec2(1, 1),
};

vec3 sample_catmull_rom_9(sampler2D stex, vec2 uv, vec2 resolution) {
	// Source: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
	// License: https://gist.github.com/TheRealMJP/bc503b0b87b643d3505d41eab8b332ae

	// We're going to sample a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
	// down the sample location to get the exact center of our "starting" texel. The starting texel will be at
	// location [1, 1] in the grid, where [0, 0] is the top left corner.
	vec2 sample_pos = uv * resolution;
	vec2 texPos1 = floor(sample_pos - 0.5f) + 0.5f;

	// Compute the fractional offset from our starting texel to our original sample location, which we'll
	// feed into the Catmull-Rom spline function to get our filter weights.
	vec2 f = sample_pos - texPos1;

	// Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
	// These equations are pre-expanded based on our knowledge of where the texels will be located,
	// which lets us avoid having to evaluate a piece-wise function.
	vec2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
	vec2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
	vec2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
	vec2 w3 = f * f * (-0.5f + 0.5f * f);

	// Work out weighting factors and sampling offsets that will let us use bilinear filtering to
	// simultaneously evaluate the middle 2 samples from the 4x4 grid.
	vec2 w12 = w1 + w2;
	vec2 offset12 = w2 / (w1 + w2);

	// Compute the final UV coordinates we'll use for sampling the texture
	vec2 texPos0 = texPos1 - 1.0f;
	vec2 texPos3 = texPos1 + 2.0f;
	vec2 texPos12 = texPos1 + offset12;

	texPos0 /= resolution;
	texPos3 /= resolution;
	texPos12 /= resolution;

	vec3 result = vec3(0.0f, 0.0f, 0.0f);

	result += textureLod(stex, vec2(texPos0.x, texPos0.y), 0.0).xyz * w0.x * w0.y;
	result += textureLod(stex, vec2(texPos12.x, texPos0.y), 0.0).xyz * w12.x * w0.y;
	result += textureLod(stex, vec2(texPos3.x, texPos0.y), 0.0).xyz * w3.x * w0.y;

	result += textureLod(stex, vec2(texPos0.x, texPos12.y), 0.0).xyz * w0.x * w12.y;
	result += textureLod(stex, vec2(texPos12.x, texPos12.y), 0.0).xyz * w12.x * w12.y;
	result += textureLod(stex, vec2(texPos3.x, texPos12.y), 0.0).xyz * w3.x * w12.y;

	result += textureLod(stex, vec2(texPos0.x, texPos3.y), 0.0).xyz * w0.x * w3.y;
	result += textureLod(stex, vec2(texPos12.x, texPos3.y), 0.0).xyz * w12.x * w3.y;
	result += textureLod(stex, vec2(texPos3.x, texPos3.y), 0.0).xyz * w3.x * w3.y;

	return max(result, 0.0f);
}

vec4 temporal_antialiasing(vec2 uv) {
	vec2 jitter = params.jitter / params.resolution;
	vec2 velocity = textureLod(velocity_buffer, uv, 0.0).xy;

	vec2 uv_reprojected = uv + velocity;

	vec3 s = textureLod(color_buffer, uv + jitter, 0.0).rgb;

	vec3 history = sample_catmull_rom_9(history_buffer, uv_reprojected, params.resolution).rgb;

	history = mix(history, s, RPC_16);

	// Sample pattern taken from https://stackoverflow.com/questions/74541193/what-algorithm-8xmsaa-16xmsaa-use-to-generate-the-position-of-8-points-16-poi
	vec3 s1 = textureLod(color_buffer, uv + jitter + (vec2(0.5625, 0.5625) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s2 = textureLod(color_buffer, uv + jitter + (vec2(0.4375, 0.3125) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s3 = textureLod(color_buffer, uv + jitter + (vec2(0.3125, 0.625) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s4 = textureLod(color_buffer, uv + jitter + (vec2(0.75, 0.4375) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s5 = textureLod(color_buffer, uv + jitter + (vec2(0.1875, 0.375) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s6 = textureLod(color_buffer, uv + jitter + (vec2(0.625, 0.8125) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s7 = textureLod(color_buffer, uv + jitter + (vec2(0.8125, 0.6875) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s8 = textureLod(color_buffer, uv + jitter + (vec2(0.6875, 0.1875) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s9 = textureLod(color_buffer, uv + jitter + (vec2(0.375, 0.875) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s10 = textureLod(color_buffer, uv + jitter + (vec2(0.5, 0.0625) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s11 = textureLod(color_buffer, uv + jitter + (vec2(0.25, 0.125) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s12 = textureLod(color_buffer, uv + jitter + (vec2(0.125, 0.75) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s13 = textureLod(color_buffer, uv + jitter + (vec2(0.0, 0.5) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s14 = textureLod(color_buffer, uv + jitter + (vec2(0.9375, 0.25) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s15 = textureLod(color_buffer, uv + jitter + (vec2(0.875, 0.9375) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;
	vec3 s16 = textureLod(color_buffer, uv + jitter + (vec2(0.0625, 0.0) - vec2(0.5, 0.5)) / params.resolution, 0.0).rgb;

	// Pick Sample blend closest to History
	float min_delta = length(history - s);
	if (min_delta > length(history - s1)) {
		s = s1;
		min_delta = length(history - s1);
	}
	if (min_delta > length(history - s2)) {
		s = s2;
		min_delta = length(history - s2);
	}
	if (min_delta > length(history - s3)) {
		s = s3;
		min_delta = length(history - s3);
	}
	if (min_delta > length(history - s4)) {
		s = s4;
		min_delta = length(history - s4);
	}
	if (min_delta > length(history - s5)) {
		s = s5;
		min_delta = length(history - s5);
	}
	if (min_delta > length(history - s6)) {
		s = s6;
		min_delta = length(history - s6);
	}
	if (min_delta > length(history - s7)) {
		s = s7;
		min_delta = length(history - s7);
	}
	if (min_delta > length(history - s8)) {
		s = s8;
		min_delta = length(history - s8);
	}
	if (min_delta > length(history - s9)) {
		s = s9;
		min_delta = length(history - s9);
	}
	if (min_delta > length(history - s10)) {
		s = s10;
		min_delta = length(history - s10);
	}
	if (min_delta > length(history - s11)) {
		s = s11;
		min_delta = length(history - s11);
	}
	if (min_delta > length(history - s12)) {
		s = s12;
		min_delta = length(history - s12);
	}
	if (min_delta > length(history - s13)) {
		s = s13;
		min_delta = length(history - s13);
	}
	if (min_delta > length(history - s14)) {
		s = s14;
		min_delta = length(history - s14);
	}
	if (min_delta > length(history - s15)) {
		s = s15;
		min_delta = length(history - s15);
	}
	if (min_delta > length(history - s16)) {
		s = s16;
		min_delta = length(history - s16);
	}

	float d1 = textureLodOffset(depth_buffer, uv + jitter, 0.0, numpad[1]).r;
	float d2 = textureLodOffset(depth_buffer, uv + jitter, 0.0, numpad[2]).r;
	float d3 = textureLodOffset(depth_buffer, uv + jitter, 0.0, numpad[3]).r;
	float d4 = textureLodOffset(depth_buffer, uv + jitter, 0.0, numpad[4]).r;
	float d5 = textureLodOffset(depth_buffer, uv + jitter, 0.0, numpad[5]).r;
	float d6 = textureLodOffset(depth_buffer, uv + jitter, 0.0, numpad[6]).r;
	float d7 = textureLodOffset(depth_buffer, uv + jitter, 0.0, numpad[7]).r;
	float d8 = textureLodOffset(depth_buffer, uv + jitter, 0.0, numpad[8]).r;
	float d9 = textureLodOffset(depth_buffer, uv + jitter, 0.0, numpad[9]).r;

	float depth_avg = (d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9) * RPC_9;

	// Reject Background, it has no valid Motion Vectors
	if (depth_avg == 0) {
		s = textureLod(color_buffer, uv + jitter, 0.0).rgb;
	}

	return vec4(s, d5 * 10000.0);
}

void main() {
	// Out of bounds check
	if (any(greaterThanEqual(vec2(gl_GlobalInvocationID.xy), params.resolution))) {
		return;
	}

	const ivec2 screen = ivec2(gl_GlobalInvocationID.xy);
	const vec2 uv = (gl_GlobalInvocationID.xy + 0.5f) / params.resolution;

	vec4 result = temporal_antialiasing(uv);
	imageStore(output_buffer, screen, result);
}
