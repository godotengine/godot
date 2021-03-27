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

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform samplerCube source_cubemap;

layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly imageCube dest_cubemap;

#include "cubemap_downsampler_inc.glsl"

void main() {
	uvec3 id = gl_GlobalInvocationID;
	uint face_size = params.face_size;

	if (id.x < face_size && id.y < face_size) {
		float inv_face_size = 1.0 / float(face_size);

		float u0 = (float(id.x) * 2.0 + 1.0 - 0.75) * inv_face_size - 1.0;
		float u1 = (float(id.x) * 2.0 + 1.0 + 0.75) * inv_face_size - 1.0;

		float v0 = (float(id.y) * 2.0 + 1.0 - 0.75) * -inv_face_size + 1.0;
		float v1 = (float(id.y) * 2.0 + 1.0 + 0.75) * -inv_face_size + 1.0;

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
		switch (id.z) {
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
		imageStore(dest_cubemap, ivec3(id), color);
	}
}
