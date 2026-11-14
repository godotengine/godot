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
// Original file link: https://github.com/sphynx-owner/Godot-Motion-Blur-Addon/blob/main/addons/godot-motion-blur/guertin/shader_stages/guertin_tile_max_x.glsl

#[compute]
#version 450

#VERSION_DEFINES

#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38

layout(set = 0, binding = 0) uniform sampler2D velocity_sampler;
layout(rg8i, set = 0, binding = 1) uniform writeonly iimage2D tile_max_x;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
	ivec2 render_size = ivec2(textureSize(velocity_sampler, 0));
	ivec2 output_size = imageSize(tile_max_x);
	ivec2 uvi = ivec2(gl_GlobalInvocationID.xy);

	ivec2 global_uvi = uvi * ivec2(TILE_SIZE, 1);
	if ((uvi.x >= output_size.x) || (uvi.y >= output_size.y) || (global_uvi.x >= render_size.x) || (global_uvi.y >= render_size.y)) {
		return;
	}

	ivec4 max_velocity = ivec4(0);

	float max_velocity_length_squared = -1;

	for (int i = 0; i < TILE_SIZE; i++) {
		ivec2 current_uvi = global_uvi + ivec2(i, 0);

		vec4 velocity_sample = texelFetch(velocity_sampler, current_uvi, 0);

		// If the depth at the potential dominant velocity is 0 (background or skybox)
		// then it will never go in front of other geometry, and can be skipped.
		if (velocity_sample.w == 0) {
			continue;
		}

		float current_velocity_length_squared = dot(velocity_sample.xy, velocity_sample.xy);

		if (current_velocity_length_squared > max_velocity_length_squared) {
			max_velocity_length_squared = current_velocity_length_squared;

			max_velocity = ivec4(velocity_sample.xy, 0, 0);
		}
	}

	imageStore(tile_max_x, uvi, max_velocity);
}
