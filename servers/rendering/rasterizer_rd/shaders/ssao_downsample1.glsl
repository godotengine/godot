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

layout(push_constant, binding = 1, std430) uniform Params {
	float z_far;
	float z_near;
	bool orthogonal;
	uint pad;
}
params;

layout(set = 0, binding = 0) uniform sampler2D source_depth;

layout(r16f, set = 1, binding = 0) uniform restrict writeonly image2D linear_z;
layout(r32f, set = 2, binding = 0) uniform restrict writeonly image2D downsampled2x;
layout(r16f, set = 3, binding = 0) uniform restrict writeonly image2DArray downsampled2x_atlas;
layout(r32f, set = 4, binding = 0) uniform restrict writeonly image2D downsampled4x;
layout(r16f, set = 5, binding = 0) uniform restrict writeonly image2DArray downsampled4x_atlas;

float Linearize(uvec2 p_pos) {
	float depth = texelFetch(source_depth, ivec2(p_pos), 0).r * 2.0 - 1.0;
	if (params.orthogonal) {
		depth = ((depth + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / (2.0 * params.z_far);
	} else {
		depth = 2.0 * params.z_near / (params.z_far + params.z_near - depth * (params.z_far - params.z_near));
	}
	imageStore(linear_z, ivec2(p_pos), vec4(depth));
	return depth;
}

shared float local_cache[256];

void main() {
	uvec2 start = gl_WorkGroupID.xy << 4 | gl_LocalInvocationID.xy;
	uint dest_index = gl_LocalInvocationID.y << 4 | gl_LocalInvocationID.x;
	local_cache[dest_index + 0] = Linearize(start | uvec2(0, 0));
	local_cache[dest_index + 8] = Linearize(start | uvec2(8, 0));
	local_cache[dest_index + 128] = Linearize(start | uvec2(0, 8));
	local_cache[dest_index + 136] = Linearize(start | uvec2(8, 8));

	groupMemoryBarrier();
	barrier();

	uint index = (gl_LocalInvocationID.x << 1) | (gl_LocalInvocationID.y << 5);

	float w1 = local_cache[index];

	uvec2 pos = gl_GlobalInvocationID.xy;
	uint slice = (pos.x & 3) | ((pos.y & 3) << 2);
	imageStore(downsampled2x, ivec2(pos), vec4(w1));
	imageStore(downsampled2x_atlas, ivec3(pos >> 2, slice), vec4(w1));

	if ((gl_LocalInvocationIndex & 011) == 0) {
		pos = gl_GlobalInvocationID.xy >> 1;
		slice = (pos.x & 3) | ((pos.y & 3) << 2);
		imageStore(downsampled4x, ivec2(pos), vec4(w1));
		imageStore(downsampled4x_atlas, ivec3(pos >> 2, slice), vec4(w1));
	}
}
