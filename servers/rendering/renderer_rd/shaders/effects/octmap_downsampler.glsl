// Copyright 2016 Activision Publishing, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#[compute]

#version 450

#VERSION_DEFINES

#define BLOCK_SIZE 8

#include "../oct_inc.glsl"

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_octmap;

layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly image2D dest_octmap;

layout(push_constant, std430) uniform Params {
	uint size;
}
params;

float calcWeight(float u, float v) {
	float val = u * u + v * v + 1.0;
	return val * sqrt(val);
}

void main() {
	uvec2 id = gl_GlobalInvocationID.xy;
	if (id.x < params.size && id.y < params.size) {
		float inv_size = 1.0 / float(params.size);

		// Find the right sampling point if the pixel is on the border.
		uvec2 sample_id = id;
		if (sample_id.x < 1 || sample_id.x > (params.size - 2)) {
			// Flip vertically if it's the left or right border.
			sample_id.y = params.size - sample_id.y - 1;
		}

		if (sample_id.y < 1 || sample_id.y > (params.size - 2)) {
			// Flip horizontally if it's the top or bottom border.
			sample_id.x = params.size - sample_id.x - 1;
		}

		sample_id = clamp(sample_id, uvec2(1), uvec2(params.size - 2));

#ifdef USE_HIGH_QUALITY
		float u0 = (float(sample_id.x) * 2.0f + 1.0f - 0.75f) * inv_size - 1.0f;
		float u1 = (float(sample_id.x) * 2.0f + 1.0f + 0.75f) * inv_size - 1.0f;
		float v0 = (float(sample_id.y) * 2.0f + 1.0f - 0.75f) * inv_size - 1.0f;
		float v1 = (float(sample_id.y) * 2.0f + 1.0f + 0.75f) * inv_size - 1.0f;
		float weights[4];
		weights[0] = calcWeight(u0, v0);
		weights[1] = calcWeight(u1, v0);
		weights[2] = calcWeight(u0, v1);
		weights[3] = calcWeight(u1, v1);

		const float wsum = 0.5 / (weights[0] + weights[1] + weights[2] + weights[3]);
		for (int i = 0; i < 4; i++) {
			weights[i] = weights[i] * wsum + 0.125;
		}

		vec4 color = textureLod(source_octmap, vec2(u0, v0) * 0.5f + 0.5f, 0.0) * weights[0];
		color += textureLod(source_octmap, vec2(u1, v0) * 0.5f + 0.5f, 0.0) * weights[1];
		color += textureLod(source_octmap, vec2(u0, v1) * 0.5f + 0.5f, 0.0) * weights[2];
		color += textureLod(source_octmap, vec2(u1, v1) * 0.5f + 0.5f, 0.0) * weights[3];
		imageStore(dest_octmap, ivec2(id), color);
#else
		vec2 uv = (vec2(sample_id.xy) + 0.5) * inv_size;
		imageStore(dest_octmap, ivec2(id), textureLod(source_octmap, uv, 0.0));
#endif
	}
}
