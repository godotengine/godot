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

#ifndef INTERLEAVE_RESULT
#define WIDE_SAMPLING 1
#endif

#if WIDE_SAMPLING
// 32x32 cache size:  the 16x16 in the center forms the area of focus with the 8-pixel perimeter used for wide gathering.
#define TILE_DIM 32
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
#else
// 16x16 cache size:  the 8x8 in the center forms the area of focus with the 4-pixel perimeter used for gathering.
#define TILE_DIM 16
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
#endif

#ifdef INTERLEAVE_RESULT
layout(set = 0, binding = 0) uniform sampler2DArray depth_texture;
#else
layout(set = 0, binding = 0) uniform sampler2D depth_texture;
#endif

layout(r8, set = 1, binding = 0) uniform restrict writeonly image2D occlusion;
//SamplerState LinearBorderSampler : register(s1);

layout(push_constant, binding = 1, std430) uniform Params {
	vec4 inv_thickness_table[3];
	vec4 sample_weight_table[3];
	vec2 texel_size;
	float rejection_fadeoff;
	float intensity;
}
params;

shared float depth_samples[TILE_DIM * TILE_DIM];

float test_sample_pair(float front_depth, float inv_range, uint p_base, uint p_offset) {
	// "Disocclusion" measures the penetration distance of the depth sample within the sphere.
	// Disocclusion < 0 (full occlusion) -> the sample fell in front of the sphere
	// Disocclusion > 1 (no occlusion) -> the sample fell behind the sphere
	float disocclusion1 = depth_samples[p_base + p_offset] * inv_range - front_depth;
	float disocclusion2 = depth_samples[p_base - p_offset] * inv_range - front_depth;

	float pseudo_disocclusion1 = clamp(params.rejection_fadeoff * disocclusion1, 0.0, 1.0);
	float pseudo_disocclusion2 = clamp(params.rejection_fadeoff * disocclusion2, 0.0, 1.0);

	return clamp(disocclusion1, pseudo_disocclusion2, 1.0) +
		   clamp(disocclusion2, pseudo_disocclusion1, 1.0) -
		   pseudo_disocclusion1 * pseudo_disocclusion2;
}

float test_samples(uint p_center_index, uint p_x, uint p_y, float p_inv_depth, float p_inv_thickness) {
#if WIDE_SAMPLING
	p_x <<= 1;
	p_y <<= 1;
#endif

	float inv_range = p_inv_thickness * p_inv_depth;
	float front_depth = p_inv_thickness - 0.5;

	if (p_y == 0) {
		// Axial
		return 0.5 * (test_sample_pair(front_depth, inv_range, p_center_index, p_x) +
							 test_sample_pair(front_depth, inv_range, p_center_index, p_x * TILE_DIM));
	} else if (p_x == p_y) {
		// Diagonal
		return 0.5 * (test_sample_pair(front_depth, inv_range, p_center_index, p_x * TILE_DIM - p_x) +
							 test_sample_pair(front_depth, inv_range, p_center_index, p_x * TILE_DIM + p_x));
	} else {
		// L-Shaped
		return 0.25 * (test_sample_pair(front_depth, inv_range, p_center_index, p_y * TILE_DIM + p_x) +
							  test_sample_pair(front_depth, inv_range, p_center_index, p_y * TILE_DIM - p_x) +
							  test_sample_pair(front_depth, inv_range, p_center_index, p_x * TILE_DIM + p_y) +
							  test_sample_pair(front_depth, inv_range, p_center_index, p_x * TILE_DIM - p_y));
	}
}

void main() {
#if WIDE_SAMPLING
	vec2 quad_center_uv = clamp(vec2(gl_GlobalInvocationID.xy + gl_LocalInvocationID.xy - 7.5) * params.texel_size, vec2(params.texel_size * 0.5), vec2(1.0 - params.texel_size * 0.5));
#else
	vec2 quad_center_uv = clamp(vec2(gl_GlobalInvocationID.xy + gl_LocalInvocationID.xy - 3.5) * params.texel_size, vec2(params.texel_size * 0.5), vec2(1.0 - params.texel_size * 0.5));
#endif

	// Fetch four depths and store them in LDS
#ifdef INTERLEAVE_RESULT
	vec4 depths = textureGather(depth_texture, vec3(quad_center_uv, gl_GlobalInvocationID.z)); // textureGather
#else
	vec4 depths = textureGather(depth_texture, quad_center_uv);
#endif

	uint dest_index = gl_LocalInvocationID.x * 2 + gl_LocalInvocationID.y * 2 * TILE_DIM;
	depth_samples[dest_index] = depths.w;
	depth_samples[dest_index + 1] = depths.z;
	depth_samples[dest_index + TILE_DIM] = depths.x;
	depth_samples[dest_index + TILE_DIM + 1] = depths.y;

	groupMemoryBarrier();
	barrier();

#if WIDE_SAMPLING
	uint index = gl_LocalInvocationID.x + gl_LocalInvocationID.y * TILE_DIM + 8 * TILE_DIM + 8;
#else
	uint index = gl_LocalInvocationID.x + gl_LocalInvocationID.y * TILE_DIM + 4 * TILE_DIM + 4;
#endif
	const float inv_depth = 1.0 / depth_samples[index];

	float ao = 0.0;

	if (params.sample_weight_table[0].x > 0.0) {
		// 68 samples:  sample all cells in *within* a circular radius of 5
		ao += params.sample_weight_table[0].x * test_samples(index, 1, 0, inv_depth, params.inv_thickness_table[0].x);
		ao += params.sample_weight_table[0].y * test_samples(index, 2, 0, inv_depth, params.inv_thickness_table[0].y);
		ao += params.sample_weight_table[0].z * test_samples(index, 3, 0, inv_depth, params.inv_thickness_table[0].z);
		ao += params.sample_weight_table[0].w * test_samples(index, 4, 0, inv_depth, params.inv_thickness_table[0].w);
		ao += params.sample_weight_table[1].x * test_samples(index, 1, 1, inv_depth, params.inv_thickness_table[1].x);
		ao += params.sample_weight_table[2].x * test_samples(index, 2, 2, inv_depth, params.inv_thickness_table[2].x);
		ao += params.sample_weight_table[2].w * test_samples(index, 3, 3, inv_depth, params.inv_thickness_table[2].w);
		ao += params.sample_weight_table[1].y * test_samples(index, 1, 2, inv_depth, params.inv_thickness_table[1].y);
		ao += params.sample_weight_table[1].z * test_samples(index, 1, 3, inv_depth, params.inv_thickness_table[1].z);
		ao += params.sample_weight_table[1].w * test_samples(index, 1, 4, inv_depth, params.inv_thickness_table[1].w);
		ao += params.sample_weight_table[2].y * test_samples(index, 2, 3, inv_depth, params.inv_thickness_table[2].y);
		ao += params.sample_weight_table[2].z * test_samples(index, 2, 4, inv_depth, params.inv_thickness_table[2].z);
	} else {
		// SAMPLE_CHECKER
		// 36 samples:  sample every-other cell in a checker board pattern
		ao += params.sample_weight_table[0].y * test_samples(index, 2, 0, inv_depth, params.inv_thickness_table[0].y);
		ao += params.sample_weight_table[0].w * test_samples(index, 4, 0, inv_depth, params.inv_thickness_table[0].w);
		ao += params.sample_weight_table[1].x * test_samples(index, 1, 1, inv_depth, params.inv_thickness_table[1].x);
		ao += params.sample_weight_table[2].x * test_samples(index, 2, 2, inv_depth, params.inv_thickness_table[2].x);
		ao += params.sample_weight_table[2].w * test_samples(index, 3, 3, inv_depth, params.inv_thickness_table[2].w);
		ao += params.sample_weight_table[1].z * test_samples(index, 1, 3, inv_depth, params.inv_thickness_table[1].z);
		ao += params.sample_weight_table[2].z * test_samples(index, 2, 4, inv_depth, params.inv_thickness_table[2].z);
	}

#ifdef INTERLEAVE_RESULT
	uvec2 out_pixel = gl_GlobalInvocationID.xy << 2 | uvec2(gl_GlobalInvocationID.z & 3, gl_GlobalInvocationID.z >> 2);
#else
	uvec2 out_pixel = gl_GlobalInvocationID.xy;
#endif
	imageStore(occlusion, ivec2(out_pixel), vec4(mix(1.0, ao, params.intensity)));
}
