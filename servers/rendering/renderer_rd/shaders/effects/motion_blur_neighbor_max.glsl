///////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025 sphynx-owner

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////
// File changes (yyyy-mm-dd)
// 2025-01-11: sphynx: first commit
// 2026-01-16: HydrogenC: make tile size specification constant and simplify push constant
///////////////////////////////////////////////////////////////////////////////////
// Original file link: https://github.com/sphynx-owner/Godot-Motion-Blur-Addon/blob/main/addons/godot-motion-blur/guertin/shader_stages/guertin_neighbor_max.glsl

#[compute]
#version 450

#VERSION_DEFINES

#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define COS_45 0.70710678118 // sqrt(2.0)/2.0
#define ONE_OVER_SQRT_2 0.70710678118 // 1/sqrt(2.0) = sqrt(2.0)/2.0 = COS_45

layout(set = 0, binding = 0) uniform isampler2D tile_max;
layout(rg8i, set = 0, binding = 1) uniform writeonly iimage2D neighbor_max;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
	ivec2 render_size = ivec2(textureSize(tile_max, 0));

	ivec2 uvi = ivec2(gl_GlobalInvocationID.xy);

	if ((uvi.x >= render_size.x) || (uvi.y >= render_size.y)) {
		return;
	}

	vec2 max_neighbor_velocity = vec2(0);

	float max_neighbor_velocity_length = 0;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			ivec2 current_offset = ivec2(i, j);

			ivec2 current_uvi = uvi + current_offset;

			if (current_uvi.x < 0 || current_uvi.x >= render_size.x || current_uvi.y < 0 || current_uvi.y >= render_size.y) {
				continue;
			}

			bool is_diagonal = i != 0 && j != 0;

			vec2 current_neighbor_velocity = texelFetch(tile_max, current_uvi, 0).xy;

			float current_neighbor_velocity_length = length(current_neighbor_velocity);

			bool can_reach_tile = abs(dot(current_neighbor_velocity / max(1e-6, current_neighbor_velocity_length), current_offset * ONE_OVER_SQRT_2)) > COS_45;

			if (is_diagonal && !can_reach_tile) {
				continue;
			}

			if (current_neighbor_velocity_length > max_neighbor_velocity_length) {
				max_neighbor_velocity_length = current_neighbor_velocity_length;

				max_neighbor_velocity = current_neighbor_velocity;
			}
		}
	}

	imageStore(neighbor_max, uvi, ivec4(max_neighbor_velocity, 0, 0));
}
