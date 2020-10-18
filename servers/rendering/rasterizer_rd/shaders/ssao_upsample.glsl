//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author:  James Stanard
//

#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D low_res_depth;
layout(set = 1, binding = 0) uniform sampler2D high_res_depth;
layout(set = 2, binding = 0) uniform sampler2D low_res_ao1;
layout(r8, set = 3, binding = 0) uniform restrict writeonly image2D ao_result;
#ifdef COMBINE_LOWER_RESOLUTIONS
layout(set = 4, binding = 0) uniform sampler2D low_res_ao2;
#endif
#ifdef BLEND_WITH_HIGHER_RESOLUTION
layout(set = 5, binding = 0) uniform sampler2D high_res_ao;
#endif

//SamplerState LinearSampler : register(s0);

layout(push_constant, binding = 1, std430) uniform Params {
	vec2 inv_low_resolution;
	vec2 inv_high_resolution;
	float noise_filter_strength;
	float step_size;
	float blur_tolerance;
	float upsample_tolerance;
}
params;

shared float depth_cache[256];
shared float ao_cache1[256];
shared float ao_cache2[256];

void prefetch_data(uint p_index, vec2 p_uv) {
	vec4 ao1 = textureGather(low_res_ao1, p_uv); // textureGather

#ifdef COMBINE_LOWER_RESOLUTIONS
	ao1 = min(ao1, textureGather(low_res_ao2, p_uv));
#endif

	ao_cache1[p_index] = ao1.w;
	ao_cache1[p_index + 1] = ao1.z;
	ao_cache1[p_index + 16] = ao1.x;
	ao_cache1[p_index + 17] = ao1.y;

	vec4 ID = 1.0 / textureGather(low_res_depth, p_uv);
	depth_cache[p_index] = ID.w;
	depth_cache[p_index + 1] = ID.z;
	depth_cache[p_index + 16] = ID.x;
	depth_cache[p_index + 17] = ID.y;
}

float smart_blur(float p_a, float p_b, float p_c, float p_d, float p_e, bool p_left, bool p_middle, bool p_right) {
	p_b = p_left || p_middle ? p_b : p_c;
	p_a = p_left ? p_a : p_b;
	p_d = p_right || p_middle ? p_d : p_c;
	p_e = p_right ? p_e : p_d;
	return ((p_a + p_e) / 2.0 + p_b + p_c + p_d) / 4.0;
}

bool compare_deltas(float p_d1, float p_d2, float p_l1, float p_l2) {
	float temp = p_d1 * p_d2 + params.step_size;
	return temp * temp > p_l1 * p_l2 * params.blur_tolerance;
}

void blur_horizontally(uint p_left_most_index) {
	float a0 = ao_cache1[p_left_most_index];
	float a1 = ao_cache1[p_left_most_index + 1];
	float a2 = ao_cache1[p_left_most_index + 2];
	float a3 = ao_cache1[p_left_most_index + 3];
	float a4 = ao_cache1[p_left_most_index + 4];
	float a5 = ao_cache1[p_left_most_index + 5];
	float a6 = ao_cache1[p_left_most_index + 6];

	float d0 = depth_cache[p_left_most_index];
	float d1 = depth_cache[p_left_most_index + 1];
	float d2 = depth_cache[p_left_most_index + 2];
	float d3 = depth_cache[p_left_most_index + 3];
	float d4 = depth_cache[p_left_most_index + 4];
	float d5 = depth_cache[p_left_most_index + 5];
	float d6 = depth_cache[p_left_most_index + 6];

	float d01 = d1 - d0;
	float d12 = d2 - d1;
	float d23 = d3 - d2;
	float d34 = d4 - d3;
	float d45 = d5 - d4;
	float d56 = d6 - d5;

	float l01 = d01 * d01 + params.step_size;
	float l12 = d12 * d12 + params.step_size;
	float l23 = d23 * d23 + params.step_size;
	float l34 = d34 * d34 + params.step_size;
	float l45 = d45 * d45 + params.step_size;
	float l56 = d56 * d56 + params.step_size;

	bool c02 = compare_deltas(d01, d12, l01, l12);
	bool c13 = compare_deltas(d12, d23, l12, l23);
	bool c24 = compare_deltas(d23, d34, l23, l34);
	bool c35 = compare_deltas(d34, d45, l34, l45);
	bool c46 = compare_deltas(d45, d56, l45, l56);

	ao_cache2[p_left_most_index] = smart_blur(a0, a1, a2, a3, a4, c02, c13, c24);
	ao_cache2[p_left_most_index + 1] = smart_blur(a1, a2, a3, a4, a5, c13, c24, c35);
	ao_cache2[p_left_most_index + 2] = smart_blur(a2, a3, a4, a5, a6, c24, c35, c46);
}

void blur_vertically(uint p_top_most_index) {
	float a0 = ao_cache2[p_top_most_index];
	float a1 = ao_cache2[p_top_most_index + 16];
	float a2 = ao_cache2[p_top_most_index + 32];
	float a3 = ao_cache2[p_top_most_index + 48];
	float a4 = ao_cache2[p_top_most_index + 64];
	float a5 = ao_cache2[p_top_most_index + 80];

	float d0 = depth_cache[p_top_most_index + 2];
	float d1 = depth_cache[p_top_most_index + 18];
	float d2 = depth_cache[p_top_most_index + 34];
	float d3 = depth_cache[p_top_most_index + 50];
	float d4 = depth_cache[p_top_most_index + 66];
	float d5 = depth_cache[p_top_most_index + 82];

	float d01 = d1 - d0;
	float d12 = d2 - d1;
	float d23 = d3 - d2;
	float d34 = d4 - d3;
	float d45 = d5 - d4;

	float l01 = d01 * d01 + params.step_size;
	float l12 = d12 * d12 + params.step_size;
	float l23 = d23 * d23 + params.step_size;
	float l34 = d34 * d34 + params.step_size;
	float l45 = d45 * d45 + params.step_size;

	bool c02 = compare_deltas(d01, d12, l01, l12);
	bool c13 = compare_deltas(d12, d23, l12, l23);
	bool c24 = compare_deltas(d23, d34, l23, l34);
	bool c35 = compare_deltas(d34, d45, l34, l45);

	float ao_result1 = smart_blur(a0, a1, a2, a3, a4, c02, c13, c24);
	float ao_result2 = smart_blur(a1, a2, a3, a4, a5, c13, c24, c35);

	ao_cache1[p_top_most_index] = ao_result1;
	ao_cache1[p_top_most_index + 16] = ao_result2;
}

// We essentially want 5 weights:  4 for each low-res pixel and 1 to blend in when none of the 4 really
// match.  The filter strength is 1 / DeltaZTolerance.  So a tolerance of 0.01 would yield a strength of 100.
// Note that a perfect match of low to high depths would yield a weight of 10^6, completely superceding any
// noise filtering.  The noise filter is intended to soften the effects of shimmering when the high-res depth
// buffer has a lot of small holes in it causing the low-res depth buffer to inaccurately represent it.
float bilateral_upsample(float p_high_depth, float p_high_ao, vec4 p_low_depths, vec4 p_low_ao) {
	vec4 weights = vec4(9.0, 3.0, 1.0, 3.0) / (abs(p_high_depth - p_low_depths) + params.upsample_tolerance);
	float total_weight = dot(weights, vec4(1.0)) + params.noise_filter_strength;
	float weighted_sum = dot(p_low_ao, weights) + params.noise_filter_strength;
	return p_high_ao * weighted_sum / total_weight;
}

void main() {
	// Load 4 pixels per thread into LDS to fill the 16x16 LDS cache with depth and AO
	prefetch_data(gl_LocalInvocationID.x << 1 | gl_LocalInvocationID.y << 5, vec2(gl_GlobalInvocationID.xy + gl_LocalInvocationID.xy - 2.5) * params.inv_low_resolution);
	groupMemoryBarrier();
	barrier();

	// Goal:  End up with a 9x9 patch that is blurred so we can upsample.  Blur radius is 2 pixels, so start with 13x13 area.

	// Horizontally blur the pixels.    13x13 -> 9x13
	if (gl_LocalInvocationIndex < 39)
		blur_horizontally((gl_LocalInvocationIndex / 3) * 16 + (gl_LocalInvocationIndex % 3) * 3);
	groupMemoryBarrier();
	barrier();

	// Vertically blur the pixels.        9x13 -> 9x9
	if (gl_LocalInvocationIndex < 45)
		blur_vertically((gl_LocalInvocationIndex / 9) * 32 + gl_LocalInvocationIndex % 9);
	groupMemoryBarrier();
	barrier();

	// Bilateral upsample
	uint index = gl_LocalInvocationID.x + gl_LocalInvocationID.y * 16;
	vec4 low_SSAOs = vec4(ao_cache1[index + 16], ao_cache1[index + 17], ao_cache1[index + 1], ao_cache1[index]);

	// We work on a quad of pixels at once because then we can gather 4 each of high and low-res depth values
	vec2 UV0 = (gl_GlobalInvocationID.xy - 0.5) * params.inv_low_resolution;
	vec2 UV1 = (gl_GlobalInvocationID.xy * 2.0 - 0.5) * params.inv_high_resolution;

#ifdef BLEND_WITH_HIGHER_RESOLUTION
	vec4 hi_SSAOs = textureGather(high_res_ao, UV1);
#else
	vec4 hi_SSAOs = vec4(1.0);
#endif
	vec4 Low_depths = textureGather(low_res_depth, UV0);
	vec4 high_depths = textureGather(high_res_depth, UV1);

	ivec2 OutST = ivec2(gl_GlobalInvocationID.xy << 1);

	imageStore(ao_result, OutST + ivec2(-1, 0), vec4(bilateral_upsample(high_depths.x, hi_SSAOs.x, Low_depths.xyzw, low_SSAOs.xyzw)));
	imageStore(ao_result, OutST + ivec2(0, 0), vec4(bilateral_upsample(high_depths.y, hi_SSAOs.y, Low_depths.yzwx, low_SSAOs.yzwx)));
	imageStore(ao_result, OutST + ivec2(0, -1), vec4(bilateral_upsample(high_depths.z, hi_SSAOs.z, Low_depths.zwxy, low_SSAOs.zwxy)));
	imageStore(ao_result, OutST + ivec2(-1, -1), vec4(bilateral_upsample(high_depths.w, hi_SSAOs.w, Low_depths.wxyz, low_SSAOs.wxyz)));
}
