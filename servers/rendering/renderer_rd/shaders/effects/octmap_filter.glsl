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

#define GROUP_SIZE 64

#include "../oct_inc.glsl"

layout(local_size_x = GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_octmap;
layout(OCTMAP_FORMAT, set = 2, binding = 0) uniform restrict writeonly image2D dest_octmap0;
layout(OCTMAP_FORMAT, set = 2, binding = 1) uniform restrict writeonly image2D dest_octmap1;
layout(OCTMAP_FORMAT, set = 2, binding = 2) uniform restrict writeonly image2D dest_octmap2;
layout(OCTMAP_FORMAT, set = 2, binding = 3) uniform restrict writeonly image2D dest_octmap3;
layout(OCTMAP_FORMAT, set = 2, binding = 4) uniform restrict writeonly image2D dest_octmap4;
layout(OCTMAP_FORMAT, set = 2, binding = 5) uniform restrict writeonly image2D dest_octmap5;

#ifdef USE_HIGH_QUALITY
#define NUM_TAPS 32
#else
#define NUM_TAPS 8
#endif

layout(push_constant, std430) uniform Params {
	vec2 border_size;
	vec2 pad;
}
params;

#define BASE_RESOLUTION 320

#ifdef USE_HIGH_QUALITY
layout(set = 1, binding = 0, std430) buffer restrict readonly Data {
	vec4[7][5][3][24] coeffs;
}
data;
#else
layout(set = 1, binding = 0, std430) buffer restrict readonly Data {
	vec4[7][5][6] coeffs;
}
data;
#endif

void main() {
	// NOTE (macOS/MoltenVK): Do not rename, "level" variable name conflicts with the Metal "level(float lod)" mipmap sampling function name.
	uvec2 id = gl_GlobalInvocationID.xy;
	uint mip_level = 0;
#ifndef USE_TEXTURE_ARRAY
	uint res = BASE_RESOLUTION;
	while ((id.x >= (res * res)) && (res > 1)) {
		id.x -= res * res;
		res = res >> 1;
		mip_level++;
	}
#else // Using Texture Arrays so all levels are the same resolution
	uint res = BASE_RESOLUTION;
	mip_level = id.x / (BASE_RESOLUTION * BASE_RESOLUTION);
	id.x -= mip_level * BASE_RESOLUTION * BASE_RESOLUTION;
#endif
	// Determine the direction from the texel's position.
	id.y = id.x / res;
	id.x -= id.y * res;

	vec2 inv_res = 1.0 / vec2(res);
	vec3 dir = oct_to_vec3_with_border((vec2(id.xy) + vec2(0.5)) * inv_res, params.border_size.y);
	vec3 adir = abs(dir);
	vec3 frameZ = dir;

	// Gather colors using GGX.
	vec4 color = vec4(0.0);
	for (int axis = 0; axis < 3; axis++) {
		const int otherAxis0 = 1 - (axis & 1) - (axis >> 1);
		const int otherAxis1 = 2 - (axis >> 1);
		const float lowerBound = 0.57735; // 1 / sqrt(3), magnitude for each component on a vector where all the components are equal.
		float frameweight = (max(adir[otherAxis0], adir[otherAxis1]) - lowerBound) / (1.0 - lowerBound);
		if (frameweight > 0.0) {
			// determine frame
			vec3 UpVector;
			switch (axis) {
				case 0:
					UpVector = vec3(1, 0, 0);
					break;
				case 1:
					UpVector = vec3(0, 1, 0);
					break;
				default:
					UpVector = vec3(0, 0, 1);
					break;
			}

			vec3 frameX = normalize(cross(UpVector, frameZ));
			vec3 frameY = cross(frameZ, frameX);

			// Calculate parametrization for polynomial.
			float Nx = dir[otherAxis0];
			float Ny = dir[otherAxis1];
			float Nz = adir[axis];

			float NmaxXY = max(abs(Ny), abs(Nx));
			Nx /= NmaxXY;
			Ny /= NmaxXY;

			float theta;
			if (Ny < Nx) {
				if (Ny <= -0.999) {
					theta = Nx;
				} else {
					theta = Ny;
				}
			} else {
				if (Ny >= 0.999) {
					theta = -Nx;
				} else {
					theta = -Ny;
				}
			}

			float phi;
			if (Nz <= -0.999) {
				phi = -NmaxXY;
			} else if (Nz >= 0.999) {
				phi = NmaxXY;
			} else {
				phi = Nz;
			}

			float theta2 = theta * theta;
			float phi2 = phi * phi;

			// Sample. The coefficient table was computed with less mip levels than required, so we clamp the maximum level.
			uint coeff_mip_level = min(mip_level, 5);
			for (int iSuperTap = 0; iSuperTap < NUM_TAPS / 4; iSuperTap++) {
				const int index = (NUM_TAPS / 4) * axis + iSuperTap;

#ifdef USE_HIGH_QUALITY
				vec4 coeffsDir0[3];
				vec4 coeffsDir1[3];
				vec4 coeffsDir2[3];
				vec4 coeffsLevel[3];
				vec4 coeffsWeight[3];

				for (int iCoeff = 0; iCoeff < 3; iCoeff++) {
					coeffsDir0[iCoeff] = data.coeffs[coeff_mip_level][0][iCoeff][index];
					coeffsDir1[iCoeff] = data.coeffs[coeff_mip_level][1][iCoeff][index];
					coeffsDir2[iCoeff] = data.coeffs[coeff_mip_level][2][iCoeff][index];
					coeffsLevel[iCoeff] = data.coeffs[coeff_mip_level][3][iCoeff][index];
					coeffsWeight[iCoeff] = data.coeffs[coeff_mip_level][4][iCoeff][index];
				}

				for (int iSubTap = 0; iSubTap < 4; iSubTap++) {
					// Determine sample attributes (dir, weight, coeff_mip_level)
					vec3 sample_dir = frameX * (coeffsDir0[0][iSubTap] + coeffsDir0[1][iSubTap] * theta2 + coeffsDir0[2][iSubTap] * phi2) + frameY * (coeffsDir1[0][iSubTap] + coeffsDir1[1][iSubTap] * theta2 + coeffsDir1[2][iSubTap] * phi2) + frameZ * (coeffsDir2[0][iSubTap] + coeffsDir2[1][iSubTap] * theta2 + coeffsDir2[2][iSubTap] * phi2);

					float sample_level = coeffsLevel[0][iSubTap] + coeffsLevel[1][iSubTap] * theta2 + coeffsLevel[2][iSubTap] * phi2;

					float sample_weight = coeffsWeight[0][iSubTap] + coeffsWeight[1][iSubTap] * theta2 + coeffsWeight[2][iSubTap] * phi2;
#else
				vec4 coeffsDir0 = data.coeffs[coeff_mip_level][0][index];
				vec4 coeffsDir1 = data.coeffs[coeff_mip_level][1][index];
				vec4 coeffsDir2 = data.coeffs[coeff_mip_level][2][index];
				vec4 coeffsLevel = data.coeffs[coeff_mip_level][3][index];
				vec4 coeffsWeight = data.coeffs[coeff_mip_level][4][index];

				for (int iSubTap = 0; iSubTap < 4; iSubTap++) {
					// determine sample attributes (dir, weight, coeff_mip_level)
					vec3 sample_dir = frameX * coeffsDir0[iSubTap] + frameY * coeffsDir1[iSubTap] + frameZ * coeffsDir2[iSubTap];

					float sample_level = coeffsLevel[iSubTap];

					float sample_weight = coeffsWeight[iSubTap];
#endif

					sample_weight *= frameweight;

#ifdef USE_HIGH_QUALITY
					// Adjust for Jacobian.
					sample_dir /= max(abs(sample_dir[0]), max(abs(sample_dir[1]), abs(sample_dir[2])));
					sample_level += 0.75 * log2(dot(sample_dir, sample_dir));
#endif

#ifndef USE_TEXTURE_ARRAY
					sample_level += float(mip_level) / 5.0; // Hack to increase the perceived roughness and reduce upscaling artifacts
#endif
					// Sample Octmap.
					vec2 sample_uv = vec3_to_oct_with_border(normalize(sample_dir), params.border_size);
					color.rgb += textureLod(source_octmap, sample_uv, sample_level).rgb * sample_weight;
					color.a += sample_weight;
				}
			}
		}
	}

	// Write out the result.
	color = vec4(max(vec3(0.0), color.rgb / color.a), 1.0);

#ifdef USE_TEXTURE_ARRAY
	id.xy *= uvec2(2, 2);
#endif

	if (mip_level > 5) {
		return;
	}

#ifdef USE_TEXTURE_ARRAY
#define IMAGE_STORE(x)                             \
	imageStore(x, ivec2(id), color);               \
	imageStore(x, ivec2(id) + ivec2(1, 0), color); \
	imageStore(x, ivec2(id) + ivec2(0, 1), color); \
	imageStore(x, ivec2(id) + ivec2(1, 1), color)
#else
#define IMAGE_STORE(x) imageStore(x, ivec2(id), color)
#endif

	switch (mip_level) {
		case 0:
			IMAGE_STORE(dest_octmap0);
			break;
		case 1:
			IMAGE_STORE(dest_octmap1);
			break;
		case 2:
			IMAGE_STORE(dest_octmap2);
			break;
		case 3:
			IMAGE_STORE(dest_octmap3);
			break;
		case 4:
			IMAGE_STORE(dest_octmap4);
			break;
		case 5:
		default:
			IMAGE_STORE(dest_octmap5);
			break;
	}
}
