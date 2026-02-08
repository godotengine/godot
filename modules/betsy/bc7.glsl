// BC7 (BPTC) Compute Shader Encoder
// Based on ARB_texture_compression_bptc specification
//
// Implements a subset of BC7's 8 modes:
// - Mode 6 for uniform/smooth blocks
// - Mode 1 for blocks with two distinct color regions

#[versions]

default = "";

#[compute]
#version 450

#VERSION_DEFINES

#define FLT_MAX 3.402823466e+38

layout(binding = 0) uniform sampler2D srcTex;
layout(binding = 1, rgba32ui) uniform restrict writeonly uimage2D dstTexture;

layout(push_constant, std430) uniform Params {
	vec2 p_textureSizeRcp;
	uint p_qualityLevel; // 0 = fast, 1 = balanced, 2 = high quality
	uint padding;
}
params;

// ============================================================================
// BC7 Mode Table (from ARB_texture_compression_bptc spec)
// Mode NS PB RB ISB CB AB EPB SPB IB IB2
// ---- -- -- -- --- -- -- --- --- -- ---
// 0    3  4  0  0   4  0  1   0   3  0
// 1    2  6  0  0   6  0  0   1   3  0
// 2    3  6  0  0   5  0  0   0   2  0
// 3    2  6  0  0   7  0  1   0   2  0
// 4    1  0  2  1   5  6  0   0   2  3
// 5    1  0  2  0   7  8  0   0   2  2
// 6    1  0  0  0   7  7  1   0   4  0
// 7    2  6  0  0   5  5  1   0   2  0
// ============================================================================

// Partition tables for 2 and 3 subsets
// Table.P2 - 64 partitions for 2 subsets
const uint g_bc7_partition2[64] = uint[64](
		0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80,
		0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
		0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
		0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
		0xAAAA, 0xF0F0, 0x5A5A, 0x33CC, 0x3C3C, 0x55AA, 0x9696, 0xA55A,
		0x73CE, 0x13C8, 0x324C, 0x3BDC, 0x6996, 0xC33C, 0x9966, 0x0660,
		0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C,
		0x9336, 0x9CC6, 0x817E, 0xE718, 0xCCF0, 0x0FCC, 0x7744, 0xEE22);

// Table.P3 - 64 partitions for 3 subsets (encoded as 2 bits per texel, packed into 32-bit words)
// Note: Currently unused since we only implement 2-subset modes, but kept for future expansion
const uint g_bc7_partition3[64] = uint[64](
		0xAA685050, 0x6A5A5040, 0x5A5A4200, 0x5450A0A8,
		0xA5A50000, 0xA0A05050, 0x5555A0A0, 0x5A5A5050,
		0xAA550000, 0xAA555500, 0xAAAA5500, 0x90909090,
		0x94949494, 0xA4A4A4A4, 0xA9A59450, 0x2A0A4250,
		0xA5945040, 0x0A425054, 0xA5A5A500, 0x55A0A0A0,
		0xA8A85454, 0x6A6A4040, 0xA4A45000, 0x1A1A0500,
		0x0050A4A4, 0xAAA59090, 0x14696914, 0x69691400,
		0xA08585A0, 0xAA821414, 0x50A4A450, 0x6A5A0200,
		0xA9A58000, 0x5090A0A8, 0xA8A09050, 0x24242424,
		0x00AA5500, 0x24924924, 0x24499224, 0x50A50A50,
		0x500AA550, 0xAAAA4444, 0x66660000, 0xA5A0A5A0,
		0x50A050A0, 0x69286928, 0x44AAAA44, 0x66666600,
		0xAA444444, 0x54A854A8, 0x95809580, 0x96969600,
		0xA85454A8, 0x80959580, 0xAA141414, 0x96960000,
		0xAAAA1414, 0xA05050A0, 0xA5050A00, 0x96000000,
		0x40804080, 0xA9A8A9A8, 0xAAAAAA44, 0x2A4A5254);

// Anchor indices for second subset in 2-subset partitioning (Table.A2)
const uint g_bc7_anchor2[64] = uint[64](
		15, 15, 15, 15, 15, 15, 15, 15,
		15, 15, 15, 15, 15, 15, 15, 15,
		15, 2, 8, 2, 2, 8, 8, 15,
		2, 8, 2, 2, 8, 8, 2, 2,
		15, 15, 6, 8, 2, 8, 15, 15,
		2, 8, 2, 2, 2, 15, 15, 6,
		6, 2, 6, 8, 15, 15, 2, 2,
		15, 15, 15, 15, 15, 2, 2, 15);

// Interpolation weights for different index bit counts
const uint g_bc7_weights2[4] = uint[4](0, 21, 43, 64);
const uint g_bc7_weights3[8] = uint[8](0, 9, 18, 27, 37, 46, 55, 64);
const uint g_bc7_weights4[16] = uint[16](0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64);

uint getPartition2(uint partitionIndex, uint texelIndex) {
	return (g_bc7_partition2[partitionIndex] >> texelIndex) & 1u;
}

uint getPartition3(uint partitionIndex, uint texelIndex) {
	uint bits = g_bc7_partition3[partitionIndex];
	return (bits >> (texelIndex * 2u)) & 3u;
}

float colorError(vec4 a, vec4 b) {
	vec4 diff = a - b;
	return dot(diff, diff);
}

// Matches the BC7 decoder: result = (e0 * (64 - w) + e1 * w + 32) >> 6
vec4 interpolateColor(vec4 e0, vec4 e1, uint weight) {
	ivec4 ie0 = ivec4(e0 * 255.0 + 0.5);
	ivec4 ie1 = ivec4(e1 * 255.0 + 0.5);
	int w = int(weight);
	ivec4 result = (ie0 * (64 - w) + ie1 * w + 32) >> 6;
	return vec4(result) / 255.0;
}

uint quantize(float value, uint bits) {
	float maxVal = float((1u << bits) - 1u);
	return uint(clamp(value * maxVal + 0.5, 0.0, maxVal));
}

float dequantize(uint value, uint bits) {
	// Replicate top bits to fill 8 bits
	uint shift = 8u - bits;
	uint expanded = (value << shift) | (value >> (bits - shift));
	return float(expanded) / 255.0;
}

void findEndpoints(vec4 texels[16], uint subset, uint partitionIndex, uint numSubsets,
		out vec4 minEndpoint, out vec4 maxEndpoint) {
	vec4 minColor = vec4(1.0);
	vec4 maxColor = vec4(0.0);
	vec4 avgColor = vec4(0.0);
	float count = 0.0;

	for (uint i = 0u; i < 16u; i++) {
		uint texelSubset;
		if (numSubsets == 1u) {
			texelSubset = 0u;
		} else if (numSubsets == 2u) {
			texelSubset = getPartition2(partitionIndex, i);
		} else {
			texelSubset = getPartition3(partitionIndex, i);
		}

		if (texelSubset == subset) {
			minColor = min(minColor, texels[i]);
			maxColor = max(maxColor, texels[i]);
			avgColor += texels[i];
			count += 1.0;
		}
	}

	if (count < 1.0) {
		minEndpoint = vec4(0.0);
		maxEndpoint = vec4(0.0);
		return;
	}

	avgColor /= count;

	// Use PCA to find the principal axis
	vec4 cov_rr = vec4(0.0);
	vec4 axis = maxColor - minColor;

	// Power iteration
	if (dot(axis, axis) > 0.0001) {
		for (uint iter = 0u; iter < 4u; iter++) {
			vec4 newAxis = vec4(0.0);
			for (uint i = 0u; i < 16u; i++) {
				uint texelSubset;
				if (numSubsets == 1u) {
					texelSubset = 0u;
				} else if (numSubsets == 2u) {
					texelSubset = getPartition2(partitionIndex, i);
				} else {
					texelSubset = getPartition3(partitionIndex, i);
				}

				if (texelSubset == subset) {
					vec4 diff = texels[i] - avgColor;
					float d = dot(diff, axis);
					newAxis += diff * d;
				}
			}
			float len = length(newAxis);
			if (len > 0.0001) {
				axis = newAxis / len;
			}
		}

		// Project all texels onto the axis and find min/max
		float minProj = FLT_MAX;
		float maxProj = -FLT_MAX;
		for (uint i = 0u; i < 16u; i++) {
			uint texelSubset;
			if (numSubsets == 1u) {
				texelSubset = 0u;
			} else if (numSubsets == 2u) {
				texelSubset = getPartition2(partitionIndex, i);
			} else {
				texelSubset = getPartition3(partitionIndex, i);
			}

			if (texelSubset == subset) {
				float proj = dot(texels[i] - avgColor, axis);
				minProj = min(minProj, proj);
				maxProj = max(maxProj, proj);
			}
		}

		minEndpoint = clamp(avgColor + axis * minProj, vec4(0.0), vec4(1.0));
		maxEndpoint = clamp(avgColor + axis * maxProj, vec4(0.0), vec4(1.0));
	} else {
		minEndpoint = avgColor;
		maxEndpoint = avgColor;
	}
}

void refineEndpoints(vec4 texels[16], uint subset, uint partitionIndex, uint numSubsets,
		uint indexBits, inout vec4 e0, inout vec4 e1) {
	uint numWeights = 1u << indexBits;

	for (uint refineIter = 0u; refineIter < 2u; refineIter++) {
		// Accumulate for least squares refinement
		vec4 atb0 = vec4(0.0);
		vec4 atb1 = vec4(0.0);
		float ata00 = 0.0;
		float ata01 = 0.0;
		float ata11 = 0.0;

		for (uint i = 0u; i < 16u; i++) {
			uint texelSubset;
			if (numSubsets == 1u) {
				texelSubset = 0u;
			} else if (numSubsets == 2u) {
				texelSubset = getPartition2(partitionIndex, i);
			} else {
				texelSubset = getPartition3(partitionIndex, i);
			}

			if (texelSubset == subset) {
				float bestError = FLT_MAX;
				uint bestIndex = 0u;

				for (uint j = 0u; j < numWeights; j++) {
					uint weight;
					if (indexBits == 2u) {
						weight = g_bc7_weights2[j];
					} else if (indexBits == 3u) {
						weight = g_bc7_weights3[j];
					} else {
						weight = g_bc7_weights4[j];
					}

					vec4 interpolated = interpolateColor(e0, e1, weight);
					float err = colorError(texels[i], interpolated);
					if (err < bestError) {
						bestError = err;
						bestIndex = j;
					}
				}

				// Use best index weight for refinement
				uint weight;
				if (indexBits == 2u) {
					weight = g_bc7_weights2[bestIndex];
				} else if (indexBits == 3u) {
					weight = g_bc7_weights3[bestIndex];
				} else {
					weight = g_bc7_weights4[bestIndex];
				}

				float w = float(weight) / 64.0;
				float w0 = 1.0 - w;
				float w1 = w;

				atb0 += texels[i] * w0;
				atb1 += texels[i] * w1;
				ata00 += w0 * w0;
				ata01 += w0 * w1;
				ata11 += w1 * w1;
			}
		}

		// Solve 2x2 system for each channel
		float det = ata00 * ata11 - ata01 * ata01;
		if (abs(det) > 1e-6) {
			float invDet = 1.0 / det;
			e0 = clamp((atb0 * ata11 - atb1 * ata01) * invDet, vec4(0.0), vec4(1.0));
			e1 = clamp((atb1 * ata00 - atb0 * ata01) * invDet, vec4(0.0), vec4(1.0));
		}
	}
}

// Assign optimal indices for a given endpoint pair
void assignIndices(vec4 texels[16], uint subset, uint partitionIndex, uint numSubsets,
		vec4 e0, vec4 e1, uint indexBits, out uint indices[16], out float totalError) {
	uint numWeights = 1u << indexBits;
	totalError = 0.0;

	for (uint i = 0u; i < 16u; i++) {
		uint texelSubset;
		if (numSubsets == 1u) {
			texelSubset = 0u;
		} else if (numSubsets == 2u) {
			texelSubset = getPartition2(partitionIndex, i);
		} else {
			texelSubset = getPartition3(partitionIndex, i);
		}

		if (texelSubset == subset) {
			float bestError = FLT_MAX;
			uint bestIndex = 0u;

			for (uint j = 0u; j < numWeights; j++) {
				uint weight;
				if (indexBits == 2u) {
					weight = g_bc7_weights2[j];
				} else if (indexBits == 3u) {
					weight = g_bc7_weights3[j];
				} else {
					weight = g_bc7_weights4[j];
				}

				vec4 interpolated = interpolateColor(e0, e1, weight);
				float err = colorError(texels[i], interpolated);
				if (err < bestError) {
					bestError = err;
					bestIndex = j;
				}
			}

			indices[i] = bestIndex;
			totalError += bestError;
		} else {
			indices[i] = 0u;
		}
	}
}

// ============================================================================
// Mode Encoding Functions
// ============================================================================

// Expand 7-bit + p-bit endpoint to full 8-bit color (matches BC7 decoder)
vec4 expandEndpoint(uint r, uint g, uint b, uint a, uint p) {
	// Value = (val << 1) | p
	return vec4(
			float((r << 1u) | p) / 255.0,
			float((g << 1u) | p) / 255.0,
			float((b << 1u) | p) / 255.0,
			float((a << 1u) | p) / 255.0);
}

float computeBlockError(vec4 texels[16], uint r0, uint g0, uint b0, uint a0, uint p0,
		uint r1, uint g1, uint b1, uint a1, uint p1) {
	vec4 qe0 = expandEndpoint(r0, g0, b0, a0, p0);
	vec4 qe1 = expandEndpoint(r1, g1, b1, a1, p1);

	float totalError = 0.0;
	for (uint i = 0u; i < 16u; i++) {
		float bestErr = FLT_MAX;
		for (uint j = 0u; j < 16u; j++) {
			vec4 interp = interpolateColor(qe0, qe1, g_bc7_weights4[j]);
			float err = colorError(texels[i], interp);
			bestErr = min(bestErr, err);
		}
		totalError += bestErr;
	}
	return totalError;
}

// Compute optimal endpoints from indices using least squares (in quantized space)
void computeOptimalEndpoints(vec4 texels[16], uint indices[16],
		out vec4 optE0, out vec4 optE1) {
	vec4 atb0 = vec4(0.0);
	vec4 atb1 = vec4(0.0);
	float ata00 = 0.0;
	float ata01 = 0.0;
	float ata11 = 0.0;

	for (uint i = 0u; i < 16u; i++) {
		float w = float(g_bc7_weights4[indices[i]]) / 64.0;
		float w0 = 1.0 - w;
		float w1 = w;

		atb0 += texels[i] * w0;
		atb1 += texels[i] * w1;
		ata00 += w0 * w0;
		ata01 += w0 * w1;
		ata11 += w1 * w1;
	}

	float det = ata00 * ata11 - ata01 * ata01;
	if (abs(det) > 1e-8) {
		float invDet = 1.0 / det;
		optE0 = clamp((atb0 * ata11 - atb1 * ata01) * invDet, vec4(0.0), vec4(1.0));
		optE1 = clamp((atb1 * ata00 - atb0 * ata01) * invDet, vec4(0.0), vec4(1.0));
	} else {
		// Degenerate case - all same index
		vec4 avg = vec4(0.0);
		for (uint i = 0u; i < 16u; i++) {
			avg += texels[i];
		}
		optE0 = avg / 16.0;
		optE1 = optE0;
	}
}

// Encode Mode 6: 1 subset, 7-bit RGBA endpoints, 1 P-bit per endpoint, 4-bit indices
uvec4 encodeMode6(vec4 texels[16], out float outError) {
	vec4 e0, e1;
	findEndpoints(texels, 0u, 0u, 1u, e0, e1);

	// Initial quantization
	uint r0 = quantize(e0.r, 7u);
	uint g0 = quantize(e0.g, 7u);
	uint b0 = quantize(e0.b, 7u);
	uint a0 = quantize(e0.a, 7u);
	uint r1 = quantize(e1.r, 7u);
	uint g1 = quantize(e1.g, 7u);
	uint b1 = quantize(e1.b, 7u);
	uint a1 = quantize(e1.a, 7u);

	uint bestR0 = r0, bestG0 = g0, bestB0 = b0, bestA0 = a0;
	uint bestR1 = r1, bestG1 = g1, bestB1 = b1, bestA1 = a1;
	uint bestP0 = 0u, bestP1 = 0u;
	float bestError = FLT_MAX;

	// Iterative refinement in quantized space
	for (uint iter = 0u; iter < 4u; iter++) {
		// Try all p-bit combinations for current endpoints
		for (uint p0Trial = 0u; p0Trial < 2u; p0Trial++) {
			for (uint p1Trial = 0u; p1Trial < 2u; p1Trial++) {
				vec4 qe0 = expandEndpoint(r0, g0, b0, a0, p0Trial);
				vec4 qe1 = expandEndpoint(r1, g1, b1, a1, p1Trial);

				// Assign indices
				uint indices[16];
				float totalError = 0.0;
				for (uint i = 0u; i < 16u; i++) {
					float bestErr = FLT_MAX;
					uint bestIdx = 0u;
					for (uint j = 0u; j < 16u; j++) {
						vec4 interp = interpolateColor(qe0, qe1, g_bc7_weights4[j]);
						float err = colorError(texels[i], interp);
						if (err < bestErr) {
							bestErr = err;
							bestIdx = j;
						}
					}
					indices[i] = bestIdx;
					totalError += bestErr;
				}

				if (totalError < bestError) {
					bestError = totalError;
					bestR0 = r0;
					bestG0 = g0;
					bestB0 = b0;
					bestA0 = a0;
					bestR1 = r1;
					bestG1 = g1;
					bestB1 = b1;
					bestA1 = a1;
					bestP0 = p0Trial;
					bestP1 = p1Trial;
				}

				// Compute optimal float endpoints from current indices
				if (iter < 3u) {
					vec4 optE0, optE1;
					computeOptimalEndpoints(texels, indices, optE0, optE1);

					// Re-quantize for next iteration
					r0 = quantize(optE0.r, 7u);
					g0 = quantize(optE0.g, 7u);
					b0 = quantize(optE0.b, 7u);
					a0 = quantize(optE0.a, 7u);
					r1 = quantize(optE1.r, 7u);
					g1 = quantize(optE1.g, 7u);
					b1 = quantize(optE1.b, 7u);
					a1 = quantize(optE1.a, 7u);
				}
			}
		}
	}

	// Use best found endpoints
	r0 = bestR0;
	g0 = bestG0;
	b0 = bestB0;
	a0 = bestA0;
	r1 = bestR1;
	g1 = bestG1;
	b1 = bestB1;
	a1 = bestA1;
	uint p0 = bestP0;
	uint p1 = bestP1;

	// Final index assignment with best endpoints
	vec4 qe0 = expandEndpoint(r0, g0, b0, a0, p0);
	vec4 qe1 = expandEndpoint(r1, g1, b1, a1, p1);

	uint indices[16];
	assignIndices(texels, 0u, 0u, 1u, qe0, qe1, 4u, indices, outError);

	// Ensure anchor index (index 0) has MSB = 0 (i.e., < 8)
	if (indices[0] >= 8u) {
		// Swap endpoints and invert indices
		uint tmp;
		tmp = r0;
		r0 = r1;
		r1 = tmp;
		tmp = g0;
		g0 = g1;
		g1 = tmp;
		tmp = b0;
		b0 = b1;
		b1 = tmp;
		tmp = a0;
		a0 = a1;
		a1 = tmp;
		tmp = p0;
		p0 = p1;
		p1 = tmp;
		for (uint i = 0u; i < 16u; i++) {
			indices[i] = 15u - indices[i];
		}
	}

	// Pack into 128-bit block (4 x 32-bit words)
	uvec4 block = uvec4(0u);

	// Word 0 (bits 0-31):
	//   bits 0-6: mode = 0b1000000 (64)
	//   bits 7-13: r0
	//   bits 14-20: r1
	//   bits 21-27: g0
	//   bits 28-31: g1[0:3]
	block.x = 64u; // Mode 6 = 0b1000000
	block.x |= r0 << 7u; // bits 7-13
	block.x |= r1 << 14u; // bits 14-20
	block.x |= g0 << 21u; // bits 21-27
	block.x |= (g1 & 0xFu) << 28u; // bits 28-31 (low 4 bits of g1)

	// Word 1 (bits 32-63):
	//   bits 0-2: g1[4:6]
	//   bits 3-9: b0
	//   bits 10-16: b1
	//   bits 17-23: a0
	//   bits 24-30: a1
	//   bit 31: p0
	block.y = (g1 >> 4u); // bits 0-2 (high 3 bits of g1)
	block.y |= b0 << 3u; // bits 3-9
	block.y |= b1 << 10u; // bits 10-16
	block.y |= a0 << 17u; // bits 17-23
	block.y |= a1 << 24u; // bits 24-30
	block.y |= p0 << 31u; // bit 31

	// Word 2 (bits 64-95):
	//   bit 0: p1
	//   bits 1-3: index[0] (anchor, 3 bits)
	//   bits 4-7: index[1]
	//   bits 8-11: index[2]
	//   bits 12-15: index[3]
	//   bits 16-19: index[4]
	//   bits 20-23: index[5]
	//   bits 24-27: index[6]
	//   bits 28-31: index[7]
	block.z = p1; // bit 0
	block.z |= (indices[0] & 0x7u) << 1u; // bits 1-3 (anchor, 3 bits)
	block.z |= indices[1] << 4u; // bits 4-7
	block.z |= indices[2] << 8u; // bits 8-11
	block.z |= indices[3] << 12u; // bits 12-15
	block.z |= indices[4] << 16u; // bits 16-19
	block.z |= indices[5] << 20u; // bits 20-23
	block.z |= indices[6] << 24u; // bits 24-27
	block.z |= indices[7] << 28u; // bits 28-31

	// Word 3 (bits 96-127):
	//   bits 0-3: index[8]
	//   bits 4-7: index[9]
	//   bits 8-11: index[10]
	//   bits 12-15: index[11]
	//   bits 16-19: index[12]
	//   bits 20-23: index[13]
	//   bits 24-27: index[14]
	//   bits 28-31: index[15]
	block.w = indices[8];
	block.w |= indices[9] << 4u;
	block.w |= indices[10] << 8u;
	block.w |= indices[11] << 12u;
	block.w |= indices[12] << 16u;
	block.w |= indices[13] << 20u;
	block.w |= indices[14] << 24u;
	block.w |= indices[15] << 28u;

	return block;
}

// Encode Mode 1: 2 subsets, 6-bit RGB, shared P-bit, 3-bit indices
uvec4 encodeMode1(vec4 texels[16], uint bestPartition, out float outError) {
	// Find endpoints for each subset
	vec4 e0[2], e1[2];
	findEndpoints(texels, 0u, bestPartition, 2u, e0[0], e1[0]);
	findEndpoints(texels, 1u, bestPartition, 2u, e0[1], e1[1]);

	// Refine endpoints
	refineEndpoints(texels, 0u, bestPartition, 2u, 3u, e0[0], e1[0]);
	refineEndpoints(texels, 1u, bestPartition, 2u, 3u, e0[1], e1[1]);

	uint r0_0 = quantize(e0[0].r, 6u);
	uint g0_0 = quantize(e0[0].g, 6u);
	uint b0_0 = quantize(e0[0].b, 6u);
	uint r1_0 = quantize(e1[0].r, 6u);
	uint g1_0 = quantize(e1[0].g, 6u);
	uint b1_0 = quantize(e1[0].b, 6u);

	uint r0_1 = quantize(e0[1].r, 6u);
	uint g0_1 = quantize(e0[1].g, 6u);
	uint b0_1 = quantize(e0[1].b, 6u);
	uint r1_1 = quantize(e1[1].r, 6u);
	uint g1_1 = quantize(e1[1].g, 6u);
	uint b1_1 = quantize(e1[1].b, 6u);

	// Try all shared P-bit combinations and keep the best
	uint p0 = 0u;
	uint p1 = 0u;
	uint indices[16];
	float totalError = FLT_MAX;

	for (uint p0Trial = 0u; p0Trial < 2u; p0Trial++) {
		for (uint p1Trial = 0u; p1Trial < 2u; p1Trial++) {
			// Reconstruct quantized endpoints with trial p-bits
			vec4 qe0[2], qe1[2];
			qe0[0] = vec4(dequantize((r0_0 << 1u) | p0Trial, 7u),
					dequantize((g0_0 << 1u) | p0Trial, 7u),
					dequantize((b0_0 << 1u) | p0Trial, 7u), 1.0);
			qe1[0] = vec4(dequantize((r1_0 << 1u) | p0Trial, 7u),
					dequantize((g1_0 << 1u) | p0Trial, 7u),
					dequantize((b1_0 << 1u) | p0Trial, 7u), 1.0);
			qe0[1] = vec4(dequantize((r0_1 << 1u) | p1Trial, 7u),
					dequantize((g0_1 << 1u) | p1Trial, 7u),
					dequantize((b0_1 << 1u) | p1Trial, 7u), 1.0);
			qe1[1] = vec4(dequantize((r1_1 << 1u) | p1Trial, 7u),
					dequantize((g1_1 << 1u) | p1Trial, 7u),
					dequantize((b1_1 << 1u) | p1Trial, 7u), 1.0);

			// Assign indices for each subset and compute error
			uint trialIndices[16];
			float trialError = 0.0;
			for (uint s = 0u; s < 2u; s++) {
				float subsetError;
				uint subsetIndices[16];
				assignIndices(texels, s, bestPartition, 2u, qe0[s], qe1[s], 3u, subsetIndices, subsetError);
				trialError += subsetError;

				for (uint i = 0u; i < 16u; i++) {
					if (getPartition2(bestPartition, i) == s) {
						trialIndices[i] = subsetIndices[i];
					}
				}
			}

			if (trialError < totalError) {
				totalError = trialError;
				p0 = p0Trial;
				p1 = p1Trial;
				for (uint i = 0u; i < 16u; i++) {
					indices[i] = trialIndices[i];
				}
			}
		}
	}

	// Ensure anchor indices have MSB = 0
	// Anchor for subset 0 is always index 0
	// Anchor for subset 1 is from Table.A2
	uint anchor1 = g_bc7_anchor2[bestPartition];

	if (indices[0] >= 4u) {
		// Swap endpoints for subset 0 and invert indices
		uint tmp;
		tmp = r0_0;
		r0_0 = r1_0;
		r1_0 = tmp;
		tmp = g0_0;
		g0_0 = g1_0;
		g1_0 = tmp;
		tmp = b0_0;
		b0_0 = b1_0;
		b1_0 = tmp;
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition2(bestPartition, i) == 0u) {
				indices[i] = 7u - indices[i];
			}
		}
	}

	if (indices[anchor1] >= 4u) {
		// Swap endpoints for subset 1 and invert indices
		uint tmp;
		tmp = r0_1;
		r0_1 = r1_1;
		r1_1 = tmp;
		tmp = g0_1;
		g0_1 = g1_1;
		g1_1 = tmp;
		tmp = b0_1;
		b0_1 = b1_1;
		b1_1 = tmp;
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition2(bestPartition, i) == 1u) {
				indices[i] = 7u - indices[i];
			}
		}
	}

	outError = totalError;

	// Pack into 128-bit block
	// Mode 1: bits 0-1 = 01 (binary)
	uvec4 block = uvec4(0u);

	// Mode (2 bits) + partition (6 bits) + endpoints...
	block.x = 0x2u; // Mode 1 = 0b10
	block.x |= bestPartition << 2u;

	// Pack RGB endpoints (6 bits each): r0, r1, r2, r3, g0, g1, g2, g3, b0...
	block.x |= r0_0 << 8u;
	block.x |= r1_0 << 14u;
	block.x |= r0_1 << 20u;
	block.x |= r1_1 << 26u;

	block.y = g0_0;
	block.y |= g1_0 << 6u;
	block.y |= g0_1 << 12u;
	block.y |= g1_1 << 18u;
	block.y |= b0_0 << 24u;
	block.y |= (b1_0 & 0x3u) << 30u;

	block.z = (b1_0 >> 2u);
	block.z |= b0_1 << 4u;
	block.z |= b1_1 << 10u;
	block.z |= p0 << 16u;
	block.z |= p1 << 17u;

	// Pack indices (3 bits each, anchors get 2 bits)
	uint bitPos = 18u;
	for (uint i = 0u; i < 16u; i++) {
		uint numBits = (i == 0u || i == anchor1) ? 2u : 3u;
		uint idx = indices[i] & ((1u << numBits) - 1u);

		if (bitPos < 32u) {
			block.z |= idx << bitPos;
		}
		if (bitPos + numBits > 32u) {
			if (bitPos < 32u) {
				block.w |= idx >> (32u - bitPos);
			} else {
				block.w |= idx << (bitPos - 32u);
			}
		}
		bitPos += numBits;
	}

	return block;
}

// ============================================================================
// Partition Selection
// ============================================================================

// Evaluate error for a 2-subset partition
float evaluatePartition2(vec4 texels[16], uint partitionIndex) {
	float totalError = 0.0;

	for (uint subset = 0u; subset < 2u; subset++) {
		vec4 minColor = vec4(1.0);
		vec4 maxColor = vec4(0.0);

		for (uint i = 0u; i < 16u; i++) {
			if (getPartition2(partitionIndex, i) == subset) {
				minColor = min(minColor, texels[i]);
				maxColor = max(maxColor, texels[i]);
			}
		}

		// Simple error metric: variance within subset
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition2(partitionIndex, i) == subset) {
				vec4 center = (minColor + maxColor) * 0.5;
				totalError += colorError(texels[i], center);
			}
		}
	}

	return totalError;
}

uint findBestPartition2(vec4 texels[16]) {
	float bestError = FLT_MAX;
	uint bestPartition = 0u;

	// For fast mode, only test a subset of partitions
	uint numPartitions = (params.p_qualityLevel == 0u) ? 16u : 64u;

	for (uint p = 0u; p < numPartitions; p++) {
		float error = evaluatePartition2(texels, p);
		if (error < bestError) {
			bestError = error;
			bestPartition = p;
		}
	}

	return bestPartition;
}

uvec4 compressBlock(vec4 texels[16]) {
	// Quality 0 (fast): Mode 6 only, minimal refinement
	// Quality 1 (balanced): Mode 6 with full refinement
	// Quality 2 (high): Try Mode 6 and Mode 1, pick best

	if (params.p_qualityLevel == 0u) {
		// Fast mode: just use Mode 6 with basic encoding
		float mode6Error;
		return encodeMode6(texels, mode6Error);
	}

	// Encode with Mode 6
	float mode6Error;
	uvec4 mode6Block = encodeMode6(texels, mode6Error);

	if (params.p_qualityLevel == 1u) {
		// Balanced: just Mode 6 with full refinement (already done)
		return mode6Block;
	}

	// High quality: also try Mode 1 with partitions and pick best
	uint bestPartition = findBestPartition2(texels);
	float mode1Error;
	uvec4 mode1Block = encodeMode1(texels, bestPartition, mode1Error);

	if (mode1Error < mode6Error) {
		return mode1Block;
	}

	return mode6Block;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
	vec2 uv = (vec2(gl_GlobalInvocationID.xy) * 4.0 + 0.5) * params.p_textureSizeRcp;

	vec4 texels[16];
	for (uint y = 0u; y < 4u; y++) {
		for (uint x = 0u; x < 4u; x++) {
			vec2 texelUV = uv + vec2(float(x), float(y)) * params.p_textureSizeRcp;
			texels[y * 4u + x] = texture(srcTex, texelUV);
		}
	}

	uvec4 block = compressBlock(texels);

	// Write the compressed block
	imageStore(dstTexture, ivec2(gl_GlobalInvocationID.xy), block);
}
