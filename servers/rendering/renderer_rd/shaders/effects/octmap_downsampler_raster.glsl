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

/* clang-format off */
#[vertex]

#version 450

#VERSION_DEFINES

#include "../oct_inc.glsl"

layout(push_constant, std430) uniform Params {
	float border_size;
	uint size;
	uint pad1;
	uint pad2;
}
params;

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {
	// old code, ARM driver bug on Mali-GXXx GPUs and Vulkan API 1.3.xxx
	// https://github.com/godotengine/godot/pull/92817#issuecomment-2168625982
	//vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	//gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	//uv_interp = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0

	vec2 vertex_base;
	if (gl_VertexIndex == 0) {
		vertex_base = vec2(-1.0, -1.0);
	} else if (gl_VertexIndex == 1) {
		vertex_base = vec2(-1.0, 3.0);
	} else {
		vertex_base = vec2(3.0, -1.0);
	}
	gl_Position = vec4(vertex_base, 0.0, 1.0);
	uv_interp = clamp(vertex_base, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
}

/* clang-format off */
#[fragment]

#version 450

#VERSION_DEFINES

#include "../oct_inc.glsl"

layout(push_constant, std430) uniform Params {
	uint size;
}
params;

layout(set = 0, binding = 0) uniform sampler2D source_octmap;

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;
/* clang-format on */

float calcWeight(float u, float v) {
	float val = u * u + v * v + 1.0;
	return val * sqrt(val);
}

void main() {
#ifdef USE_HIGH_QUALITY
	float inv_size = 1.0 / float(params.size);
	float u0 = (uv_interp.x * 2.0f) + (1.0f - 0.75f) * inv_size - 1.0f;
	float u1 = (uv_interp.x * 2.0f) + (1.0f + 0.75f) * inv_size - 1.0f;
	float v0 = (uv_interp.y * 2.0f) + (1.0f - 0.75f) * inv_size - 1.0f;
	float v1 = (uv_interp.y * 2.0f) + (1.0f + 0.75f) * inv_size - 1.0f;
	float weights[4];
	weights[0] = calcWeight(u0, v0);
	weights[1] = calcWeight(u1, v0);
	weights[2] = calcWeight(u0, v1);
	weights[3] = calcWeight(u1, v1);

	const float wsum = 0.5 / (weights[0] + weights[1] + weights[2] + weights[3]);
	for (int i = 0; i < 4; i++) {
		weights[i] = weights[i] * wsum + 0.125;
	}

	frag_color = textureLod(source_octmap, vec2(u0, v0) * 0.5f + 0.5f, 0.0) * weights[0];
	frag_color += textureLod(source_octmap, vec2(u1, v0) * 0.5f + 0.5f, 0.0) * weights[1];
	frag_color += textureLod(source_octmap, vec2(u0, v1) * 0.5f + 0.5f, 0.0) * weights[2];
	frag_color += textureLod(source_octmap, vec2(u1, v1) * 0.5f + 0.5f, 0.0) * weights[3];
#else
	frag_color = textureLod(source_octmap, uv_interp, 0.0);
#endif
}
