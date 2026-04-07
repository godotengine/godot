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

layout(push_constant, std430) uniform Params {
	vec2 border_size;
	int mip_level;
	int pad;
}
params;

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {
	vec2 base_arr[3] = vec2[](vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
	uv_interp = base_arr[gl_VertexIndex];
	gl_Position = vec4(uv_interp, 0.0, 1.0);
}

/* clang-format off */
#[fragment]

#version 450

#VERSION_DEFINES

#include "../oct_inc.glsl"

layout(push_constant, std430) uniform Params {
	vec2 border_size;
	int mip_level;
	int pad;
}
params;

layout(set = 0, binding = 0) uniform sampler2D source_octmap;

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

/* clang-format on */

#ifdef USE_HIGH_QUALITY
#define NUM_TAPS 32
#else
#define NUM_TAPS 8
#endif

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
	int mip_level = 0;
	vec2 uv = uv_interp * 0.5 + 0.5;
	if (params.mip_level < 0) {
		// Just copy the octmap directly.
		frag_color.rgb = textureLod(source_octmap, uv, 0.0).rgb;
		frag_color.a = 1.0;
		return;
	} else if (params.mip_level > 5) {
		// Limit the level.
		mip_level = 5;
	} else {
		// Use the level from the push constant.
		mip_level = params.mip_level;
	}

	// Determine the direction from the texel's position.
	vec3 dir = oct_to_vec3_with_border(uv, params.border_size.y);
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

			// calculate parametrization for polynomial
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

			// Sample.
			for (int iSuperTap = 0; iSuperTap < NUM_TAPS / 4; iSuperTap++) {
				const int index = (NUM_TAPS / 4) * axis + iSuperTap;

#ifdef USE_HIGH_QUALITY
				vec4 coeffsDir0[3];
				vec4 coeffsDir1[3];
				vec4 coeffsDir2[3];
				vec4 coeffsLevel[3];
				vec4 coeffsWeight[3];

				for (int iCoeff = 0; iCoeff < 3; iCoeff++) {
					coeffsDir0[iCoeff] = data.coeffs[mip_level][0][iCoeff][index];
					coeffsDir1[iCoeff] = data.coeffs[mip_level][1][iCoeff][index];
					coeffsDir2[iCoeff] = data.coeffs[mip_level][2][iCoeff][index];
					coeffsLevel[iCoeff] = data.coeffs[mip_level][3][iCoeff][index];
					coeffsWeight[iCoeff] = data.coeffs[mip_level][4][iCoeff][index];
				}

				for (int iSubTap = 0; iSubTap < 4; iSubTap++) {
					// determine sample attributes (dir, weight, mip_level)
					vec3 sample_dir = frameX * (coeffsDir0[0][iSubTap] + coeffsDir0[1][iSubTap] * theta2 + coeffsDir0[2][iSubTap] * phi2) + frameY * (coeffsDir1[0][iSubTap] + coeffsDir1[1][iSubTap] * theta2 + coeffsDir1[2][iSubTap] * phi2) + frameZ * (coeffsDir2[0][iSubTap] + coeffsDir2[1][iSubTap] * theta2 + coeffsDir2[2][iSubTap] * phi2);

					float sample_level = coeffsLevel[0][iSubTap] + coeffsLevel[1][iSubTap] * theta2 + coeffsLevel[2][iSubTap] * phi2;

					float sample_weight = coeffsWeight[0][iSubTap] + coeffsWeight[1][iSubTap] * theta2 + coeffsWeight[2][iSubTap] * phi2;
#else
				vec4 coeffsDir0 = data.coeffs[mip_level][0][index];
				vec4 coeffsDir1 = data.coeffs[mip_level][1][index];
				vec4 coeffsDir2 = data.coeffs[mip_level][2][index];
				vec4 coeffsLevel = data.coeffs[mip_level][3][index];
				vec4 coeffsWeight = data.coeffs[mip_level][4][index];

				for (int iSubTap = 0; iSubTap < 4; iSubTap++) {
					// determine sample attributes (dir, weight, mip_level)
					vec3 sample_dir = frameX * coeffsDir0[iSubTap] + frameY * coeffsDir1[iSubTap] + frameZ * coeffsDir2[iSubTap];

					float sample_level = coeffsLevel[iSubTap];

					float sample_weight = coeffsWeight[iSubTap];
#endif

					sample_weight *= frameweight;

					// Sample Octmap.
					vec2 sample_uv = vec3_to_oct_with_border(normalize(sample_dir), params.border_size);
					color.rgb += textureLod(source_octmap, sample_uv, sample_level).rgb * sample_weight;
					color.a += sample_weight;
				}
			}
		}
	}

	// Write out the result.
	frag_color = vec4(max(vec3(0.0), color.rgb / color.a), 1.0);
}
