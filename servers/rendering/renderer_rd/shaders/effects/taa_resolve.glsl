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
// 2026-07-14: Rene Prašnikar: Total shader rewrite, simplified the shader, new "all-in" anti-ghosting strategy introduced
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

// Based on "Temporal Reprojection Anti-Aliasing" - https://github.com/playdeadgames/temporal
vec3 clip_aabb(vec3 aabb_min, vec3 aabb_max, vec3 p, vec3 q) {
	vec3 r = q - p;
	vec3 rmax = (aabb_max - p.xyz);
	vec3 rmin = (aabb_min - p.xyz);

	if (r.x > rmax.x + FLT_MIN) {
		r *= (rmax.x / r.x);
	}
	if (r.y > rmax.y + FLT_MIN) {
		r *= (rmax.y / r.y);
	}
	if (r.z > rmax.z + FLT_MIN) {
		r *= (rmax.z / r.z);
	}
	if (r.x < rmin.x - FLT_MIN) {
		r *= (rmin.x / r.x);
	}
	if (r.y < rmin.y - FLT_MIN) {
		r *= (rmin.y / r.y);
	}
	if (r.z < rmin.z - FLT_MIN) {
		r *= (rmin.z / r.z);
	}

	return p + r;
}

vec4 temporal_antialiasing(vec2 uv) {
	vec2 jitter = params.jitter / params.resolution;
	vec2 v1 = textureLodOffset(velocity_buffer, uv, 0.0, numpad[1]).xy;
	vec2 v2 = textureLodOffset(velocity_buffer, uv, 0.0, numpad[2]).xy;
	vec2 v3 = textureLodOffset(velocity_buffer, uv, 0.0, numpad[3]).xy;
	vec2 v4 = textureLodOffset(velocity_buffer, uv, 0.0, numpad[4]).xy;
	vec2 v5 = textureLodOffset(velocity_buffer, uv, 0.0, numpad[5]).xy;
	vec2 v6 = textureLodOffset(velocity_buffer, uv, 0.0, numpad[6]).xy;
	vec2 v7 = textureLodOffset(velocity_buffer, uv, 0.0, numpad[7]).xy;
	vec2 v8 = textureLodOffset(velocity_buffer, uv, 0.0, numpad[8]).xy;
	vec2 v9 = textureLodOffset(velocity_buffer, uv, 0.0, numpad[9]).xy;

	vec2 v_avg = (v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9) * RPC_9;

	vec2 uv_reprojected = uv + v5;

	vec3 history = sample_catmull_rom_9(history_buffer, uv_reprojected, params.resolution).rgb;

	// Current Samples
	vec3 s1 = textureLodOffset(color_buffer, uv + jitter, 0.0, numpad[1]).rgb;
	vec3 s2 = textureLodOffset(color_buffer, uv + jitter, 0.0, numpad[2]).rgb;
	vec3 s3 = textureLodOffset(color_buffer, uv + jitter, 0.0, numpad[3]).rgb;
	vec3 s4 = textureLodOffset(color_buffer, uv + jitter, 0.0, numpad[4]).rgb;
	vec3 s5 = textureLodOffset(color_buffer, uv + jitter, 0.0, numpad[5]).rgb;
	vec3 s6 = textureLodOffset(color_buffer, uv + jitter, 0.0, numpad[6]).rgb;
	vec3 s7 = textureLodOffset(color_buffer, uv + jitter, 0.0, numpad[7]).rgb;
	vec3 s8 = textureLodOffset(color_buffer, uv + jitter, 0.0, numpad[8]).rgb;
	vec3 s9 = textureLodOffset(color_buffer, uv + jitter, 0.0, numpad[9]).rgb;

	vec3 s_min = min(s1, min(s2, min(s3, min(s4, min(s5, min(s6, min(s7, min(s8, s9))))))));
	vec3 s_max = max(s1, max(s2, max(s3, max(s4, max(s5, max(s6, max(s7, max(s8, s9))))))));
	vec3 s_avg = (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9) * RPC_9;
	vec3 s_range = s_max - s_min;

	float c1 = dot(normalize(history), normalize(s1));
	float c2 = dot(normalize(history), normalize(s2));
	float c3 = dot(normalize(history), normalize(s3));
	float c4 = dot(normalize(history), normalize(s4));
	float c5 = dot(normalize(history), normalize(s5));
	float c6 = dot(normalize(history), normalize(s6));
	float c7 = dot(normalize(history), normalize(s7));
	float c8 = dot(normalize(history), normalize(s8));
	float c9 = dot(normalize(history), normalize(s9));

	vec3 unjittered = textureLodOffset(color_buffer, uv, 0.0, numpad[5]).rgb;

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

	float factor_screen = any(lessThan(uv_reprojected, vec2(0.0))) || any(greaterThan(uv_reprojected, vec2(1.0))) ? 1.0 : 0.0;
	float blend_factor = clamp(RPC_16 + factor_screen, 0.0, 1.0);

	vec3 color_resolved = mix(history, s5, blend_factor);

	// Compute min and max (with an adaptive box size, which greatly reduces ghosting)
	vec3 color_avg = (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9) * RPC_9;
	vec3 color_avg2 = ((s1 * s1) + (s2 * s2) + (s3 * s3) + (s4 * s4) + (s5 * s5) + (s6 * s6) + (s7 * s7) + (s8 * s8) + (s9 * s9)) * RPC_9;
	// Use variance clipping as described in https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
	// Larger multiplier relaxes Clipping
	vec3 dev = sqrt(abs(color_avg2 - (color_avg * color_avg))) * 1.75;
	vec3 color_min = color_avg - dev;
	vec3 color_max = color_avg + dev;

	// Variance clipping
	vec3 color = clip_aabb(color_min, color_max, clamp(color_avg, color_min, color_max), history);

	// Clamp to prevent NaNs
	color = clamp(color, FLT_MIN, FLT_MAX);

	// Higher values stabilize higher distances, but lead to more ghosting
	float distance_relaxation_strength = 0.25;
	float distance_relaxation = (1.0 - distance_relaxation_strength) + (depth_avg * distance_relaxation_strength);
	float velocity_rejection = clamp(1.0 - length(v_avg), 0.0, 1.0);

	float sample_brightness_divergence = length(history - s5);
	float sample_brightness_range = length(s_range);
	bool brightness_rejection_criteria = sample_brightness_divergence * distance_relaxation > sample_brightness_range * velocity_rejection;

	float sample_chroma_divergence = dot(normalize(history), normalize(s5));
	float sample_chroma_range = max(c1, max(c2, max(c3, max(c4, max(c5, max(c6, max(c7, max(c8, c9))))))));
	bool chroma_rejection_criteria = sample_chroma_divergence * distance_relaxation > sample_chroma_range * velocity_rejection;

	bool variance_clipped = length(history - color) > FLT_MIN;

	// Maintains Glow for Glowing Bullets flying around
	bool bullet_detected = length(v_avg - v5) > length(jitter) * 5.0 && length(unjittered) > 100.0;

	// Reject Background, it has no valid Motion Vectors
	if (depth_avg == 0) {
		color_resolved = s5;
	}
	if (brightness_rejection_criteria == true) {
		color_resolved = s5;
	}
	if (chroma_rejection_criteria == true) {
		color_resolved = s5;
	}
	if (variance_clipped == true) {
		color_resolved = s5;
	}
	if (bullet_detected == true) {
		color_resolved = unjittered;
	}

	return vec4(color_resolved, d5 * 10000.0);
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
