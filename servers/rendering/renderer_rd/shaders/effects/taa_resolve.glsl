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

// Based on SMAA's Temporal Reprojection (This is NOT SMAA T2x)
// https://github.com/iryoku/smaa/blob/master/SMAA.hlsl

#define RPC_9 0.11111111111
#define RPC_16 0.0625

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D color_buffer;
layout(set = 0, binding = 1) uniform sampler2D depth_buffer;
layout(rg16f, set = 0, binding = 2) uniform restrict readonly image2D velocity_buffer;
layout(rg16f, set = 0, binding = 3) uniform restrict readonly image2D last_velocity_buffer;
layout(set = 0, binding = 4) uniform sampler2D history_buffer;
layout(rgba16f, set = 0, binding = 5) uniform restrict writeonly image2D output_buffer;

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

vec4 temporal_antialiasing(vec2 uv, ivec2 screen) {
	vec2 velocity = imageLoad(velocity_buffer, screen).xy;
	vec2 uv_reprojected = uv + velocity;
	vec2 velocity_previous = imageLoad(last_velocity_buffer, ivec2(uv_reprojected * params.resolution)).xy;

	vec2 jitter = params.jitter / params.resolution;

	float d1 = textureLodOffset(depth_buffer, uv, 0.0, numpad[1]).r;
	float d2 = textureLodOffset(depth_buffer, uv, 0.0, numpad[2]).r;
	float d3 = textureLodOffset(depth_buffer, uv, 0.0, numpad[3]).r;
	float d4 = textureLodOffset(depth_buffer, uv, 0.0, numpad[4]).r;
	float d5 = textureLodOffset(depth_buffer, uv, 0.0, numpad[5]).r;
	float d6 = textureLodOffset(depth_buffer, uv, 0.0, numpad[6]).r;
	float d7 = textureLodOffset(depth_buffer, uv, 0.0, numpad[7]).r;
	float d8 = textureLodOffset(depth_buffer, uv, 0.0, numpad[8]).r;
	float d9 = textureLodOffset(depth_buffer, uv, 0.0, numpad[9]).r;

	float depth_avg = (d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9) * RPC_9;

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

	vec3 current = textureLod(color_buffer, uv, 0.0).rgb;

	vec3 previous = textureLod(history_buffer, uv_reprojected, 0.0).rgb;

	float delta = length(velocity * velocity - velocity_previous * velocity_previous) / 5.0;
	float weight = 0.5 * clamp(1.0 - sqrt(delta) * 30.0, 0.0, 1.0);

	vec3 color_resolved = mix(current, previous, weight);

	vec3 clip = clamp(color_resolved, s_min, s_max);

	if (length(color_resolved - clip) > 0.0) {
		color_resolved = current;
	}

	if (depth_avg == 0) {
		color_resolved = s5;
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

	vec4 result = temporal_antialiasing(uv, screen);
	imageStore(output_buffer, screen, result);
}
