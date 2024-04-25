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

#include "cubemap_downsampler_inc.glsl"

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {
	vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	uv_interp = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0 * float(params.face_size); // saturate(x) * 2.0
}

/* clang-format off */
#[fragment]

#version 450

#VERSION_DEFINES

#include "cubemap_downsampler_inc.glsl"

layout(set = 0, binding = 0) uniform samplerCube source_cubemap;

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;
/* clang-format on */

void main() {
	// Converted from compute shader which uses absolute coordinates.
	// Could possibly simplify this
	float face_size = float(params.face_size);
	float inv_face_size = 1.0 / face_size;
	vec2 id = floor(uv_interp);

	float u1 = (id.x * 2.0 + 1.0 + 0.75) * inv_face_size - 1.0;
	float u0 = (id.x * 2.0 + 1.0 - 0.75) * inv_face_size - 1.0;

	float v0 = (id.y * 2.0 + 1.0 - 0.75) * -inv_face_size + 1.0;
	float v1 = (id.y * 2.0 + 1.0 + 0.75) * -inv_face_size + 1.0;

	float weights[4];
	weights[0] = calcWeight(u0, v0);
	weights[1] = calcWeight(u1, v0);
	weights[2] = calcWeight(u0, v1);
	weights[3] = calcWeight(u1, v1);

	const float wsum = 0.5 / (weights[0] + weights[1] + weights[2] + weights[3]);
	for (int i = 0; i < 4; i++) {
		weights[i] = weights[i] * wsum + .125;
	}

	vec3 dir;
	vec4 color;
	switch (params.face_id) {
		case 0:
			get_dir_0(dir, u0, v0);
			color = textureLod(source_cubemap, normalize(dir), 0.0) * weights[0];

			get_dir_0(dir, u1, v0);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[1];

			get_dir_0(dir, u0, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[2];

			get_dir_0(dir, u1, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[3];
			break;
		case 1:
			get_dir_1(dir, u0, v0);
			color = textureLod(source_cubemap, normalize(dir), 0.0) * weights[0];

			get_dir_1(dir, u1, v0);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[1];

			get_dir_1(dir, u0, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[2];

			get_dir_1(dir, u1, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[3];
			break;
		case 2:
			get_dir_2(dir, u0, v0);
			color = textureLod(source_cubemap, normalize(dir), 0.0) * weights[0];

			get_dir_2(dir, u1, v0);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[1];

			get_dir_2(dir, u0, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[2];

			get_dir_2(dir, u1, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[3];
			break;
		case 3:
			get_dir_3(dir, u0, v0);
			color = textureLod(source_cubemap, normalize(dir), 0.0) * weights[0];

			get_dir_3(dir, u1, v0);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[1];

			get_dir_3(dir, u0, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[2];

			get_dir_3(dir, u1, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[3];
			break;
		case 4:
			get_dir_4(dir, u0, v0);
			color = textureLod(source_cubemap, normalize(dir), 0.0) * weights[0];

			get_dir_4(dir, u1, v0);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[1];

			get_dir_4(dir, u0, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[2];

			get_dir_4(dir, u1, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[3];
			break;
		default:
			get_dir_5(dir, u0, v0);
			color = textureLod(source_cubemap, normalize(dir), 0.0) * weights[0];

			get_dir_5(dir, u1, v0);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[1];

			get_dir_5(dir, u0, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[2];

			get_dir_5(dir, u1, v1);
			color += textureLod(source_cubemap, normalize(dir), 0.0) * weights[3];
			break;
	}
	frag_color = color;
}
