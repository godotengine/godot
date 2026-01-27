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
// Original file link: https://github.com/sphynx-owner/godot-motion-blur-addon-simplified/blob/master/addons/sphynx_motion_blur_toolkit/guertin/shader_stages/shader_files/guertin_tile_max_x.glsl

#[compute]
#version 450

#VERSION_DEFINES

#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38

layout(set = 0, binding = 0) uniform sampler2D velocity_sampler;
layout(set = 0, binding = 1) uniform sampler2D depth_sampler;
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D tile_max_x;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
	ivec2 render_size = ivec2(textureSize(velocity_sampler, 0));
	ivec2 output_size = imageSize(tile_max_x);
	ivec2 uvi = ivec2(gl_GlobalInvocationID.xy);

	ivec2 global_uvi = uvi * ivec2(TILE_SIZE, 1);
	if ((uvi.x >= output_size.x) || (uvi.y >= output_size.y) || (global_uvi.x >= render_size.x) || (global_uvi.y >= render_size.y)) {
		return;
	}

	vec2 uvn = (vec2(global_uvi) + vec2(0.5)) / render_size;

	vec4 max_velocity = vec4(0);

	float max_velocity_length = -1;

	for (int i = 0; i < TILE_SIZE; i++) {
		vec2 current_uv = uvn + vec2(float(i) / render_size.x, 0);
		vec4 velocity_sample = textureLod(velocity_sampler, current_uv, 0.0);

		// If the depth at the potential dominant velocity is infinity (background or skybox)
		// then it will never go in front of other geometry, and can be skipped.
		// TODO @sphynx-owner: enable when considering ignoring skybox for dominant velocity
		// if(velocity_sample.w == (-1.0 / 0.0))
		// {
		// 	continue;
		// }

		float current_velocity_length = dot(velocity_sample.xy, velocity_sample.xy);
		if (current_velocity_length > max_velocity_length) {
			max_velocity_length = current_velocity_length;
			max_velocity = vec4(velocity_sample.xy, 0, 0);
		}
	}

	imageStore(tile_max_x, uvi, max_velocity);
}
