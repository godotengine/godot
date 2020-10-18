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

layout(r32f, set = 0, binding = 0) uniform restrict readonly image2D downsampled4x;
layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D downsampled8x;
layout(r16f, set = 2, binding = 0) uniform restrict writeonly image2DArray downsampled8x_atlas;
layout(r32f, set = 3, binding = 0) uniform restrict writeonly image2D downsampled16x;
layout(r16f, set = 4, binding = 0) uniform restrict writeonly image2DArray downsampled16x_atlas;

void main() {
	vec4 w1 = imageLoad(downsampled4x, min(ivec2(gl_GlobalInvocationID.xy << 1), imageSize(downsampled4x) - ivec2(2)));

	uvec2 pos = gl_GlobalInvocationID.xy;
	uvec2 pos_atlas = pos >> 2;
	uint pos_slice = (pos.x & 3) | ((pos.y & 3) << 2);
	ivec2 ds8s = imageSize(downsampled8x);

	if (pos.x < ds8s.x && pos.y < ds8s.y) {
		imageStore(downsampled8x, ivec2(pos), w1);
	}

	imageStore(downsampled8x_atlas, ivec3(pos_atlas, pos_slice), w1);

	if ((gl_LocalInvocationIndex & 011) == 0) {
		uvec2 pos = gl_GlobalInvocationID.xy >> 1;
		uvec2 pos_atlas = pos >> 2;
		uint pos_slice = (pos.x & 3) | ((pos.y & 3) << 2);
		imageStore(downsampled16x, ivec2(pos), w1);
		imageStore(downsampled16x_atlas, ivec3(pos_atlas, pos_slice), w1);
	}
}
