// BC7 (BPTC) compute shader encoder.
// Based on ARB_texture_compression_bptc specification.

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
		0xAAAA1414, 0xA05050A0, 0xA0A5A5A0, 0x96000000,
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

// Anchor indices for subset 1 and subset 2 in 3-subset partitioning (Table.A3)
const uvec2 g_bc7_anchor3[64] = uvec2[64](
		uvec2(3, 15), uvec2(3, 8), uvec2(15, 8), uvec2(15, 3),
		uvec2(8, 15), uvec2(3, 15), uvec2(15, 3), uvec2(15, 8),
		uvec2(8, 15), uvec2(8, 15), uvec2(6, 15), uvec2(6, 15),
		uvec2(6, 15), uvec2(5, 15), uvec2(3, 15), uvec2(3, 8),
		uvec2(3, 15), uvec2(3, 8), uvec2(8, 15), uvec2(15, 3),
		uvec2(3, 15), uvec2(3, 8), uvec2(6, 15), uvec2(10, 8),
		uvec2(5, 3), uvec2(8, 15), uvec2(8, 6), uvec2(6, 10),
		uvec2(8, 15), uvec2(5, 15), uvec2(15, 10), uvec2(15, 8),

		uvec2(8, 15), uvec2(15, 3), uvec2(3, 15), uvec2(5, 10),
		uvec2(6, 10), uvec2(10, 8), uvec2(8, 9), uvec2(15, 10),
		uvec2(15, 6), uvec2(3, 15), uvec2(15, 8), uvec2(5, 15),
		uvec2(15, 3), uvec2(15, 6), uvec2(15, 6), uvec2(15, 8),
		uvec2(3, 15), uvec2(15, 3), uvec2(5, 15), uvec2(5, 15),
		uvec2(5, 15), uvec2(8, 15), uvec2(5, 15), uvec2(10, 15),
		uvec2(5, 15), uvec2(10, 15), uvec2(8, 15), uvec2(13, 15),
		uvec2(15, 3), uvec2(12, 15), uvec2(3, 15), uvec2(3, 8));

// Interpolation weights for different index bit counts.
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

float colorErrorRGB(vec4 a, vec4 b) {
	vec3 diff = a.rgb - b.rgb;
	return dot(diff, diff);
}

float alphaError(float a, float b) {
	float d = a - b;
	return d * d;
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

void bc7WriteBits(inout uvec4 block, inout uint bitPos, uint value, uint bitCount) {
	uint wordIdx = bitPos >> 5u;
	uint bitIdx = bitPos & 31u;
	// Write value into current word.
	if (wordIdx == 0u) {
		block.x |= value << bitIdx;
	} else if (wordIdx == 1u) {
		block.y |= value << bitIdx;
	} else if (wordIdx == 2u) {
		block.z |= value << bitIdx;
	} else {
		block.w |= value << bitIdx;
	}
	// Handle overflow into the next word.
	if (bitIdx + bitCount > 32u) {
		uint overflow = value >> (32u - bitIdx);
		if (wordIdx == 0u) {
			block.y |= overflow;
		} else if (wordIdx == 1u) {
			block.z |= overflow;
		} else if (wordIdx == 2u) {
			block.w |= overflow;
		}
	}
	bitPos += bitCount;
}

float dequantize(uint value, uint bits) {
	if (bits == 8u) {
		return float(value) / 255.0;
	}
	// Replicate top bits to fill 8 bits.
	uint shift = 8u - bits;
	uint expanded = (value << shift) | (value >> (bits - shift));
	return float(expanded) / 255.0;
}

void findEndpoints(vec4 texels[16], uint subset, uint partitionIndex, uint numSubsets,
		out vec4 minEndpoint, out vec4 maxEndpoint) {
	// Pre-compute subset membership to avoid repeated partition lookups.
	bool inSubset[16];
	for (uint i = 0u; i < 16u; i++) {
		uint texelSubset;
		if (numSubsets == 1u) {
			texelSubset = 0u;
		} else if (numSubsets == 2u) {
			texelSubset = getPartition2(partitionIndex, i);
		} else {
			texelSubset = getPartition3(partitionIndex, i);
		}
		inSubset[i] = (texelSubset == subset);
	}

	vec4 minColor = vec4(1.0);
	vec4 maxColor = vec4(0.0);
	vec4 avgColor = vec4(0.0);
	float count = 0.0;

	for (uint i = 0u; i < 16u; i++) {
		if (inSubset[i]) {
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

	// Use PCA to find the principal axis.
	vec4 axis = maxColor - minColor;

	// Power iteration
	if (dot(axis, axis) > 0.0001) {
		for (uint iter = 0u; iter < 4u; iter++) {
			vec4 newAxis = vec4(0.0);
			for (uint i = 0u; i < 16u; i++) {
				if (inSubset[i]) {
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
			if (inSubset[i]) {
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

	// Pre-compute subset membership.
	bool inSubset[16];
	for (uint i = 0u; i < 16u; i++) {
		uint texelSubset;
		if (numSubsets == 1u) {
			texelSubset = 0u;
		} else if (numSubsets == 2u) {
			texelSubset = getPartition2(partitionIndex, i);
		} else {
			texelSubset = getPartition3(partitionIndex, i);
		}
		inSubset[i] = (texelSubset == subset);
	}

	for (uint refineIter = 0u; refineIter < 2u; refineIter++) {
		// Accumulate for least squares refinement
		vec4 atb0 = vec4(0.0);
		vec4 atb1 = vec4(0.0);
		float ata00 = 0.0;
		float ata01 = 0.0;
		float ata11 = 0.0;

		for (uint i = 0u; i < 16u; i++) {
			if (inSubset[i]) {
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

// Expand 7-bit + p-bit endpoint to full 8-bit color (Mode 6 specific).
vec4 expandEndpointMode6(uint r, uint g, uint b, uint a, uint p) {
	// Value = (val << 1) | p
	return vec4(
			float((r << 1u) | p) / 255.0,
			float((g << 1u) | p) / 255.0,
			float((b << 1u) | p) / 255.0,
			float((a << 1u) | p) / 255.0);
}

float computeBlockError(vec4 texels[16], uint r0, uint g0, uint b0, uint a0, uint p0,
		uint r1, uint g1, uint b1, uint a1, uint p1) {
	vec4 qe0 = expandEndpointMode6(r0, g0, b0, a0, p0);
	vec4 qe1 = expandEndpointMode6(r1, g1, b1, a1, p1);

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
		uint iterBestIndices[16];
		float iterBestError = FLT_MAX;

		for (uint p0Trial = 0u; p0Trial < 2u; p0Trial++) {
			for (uint p1Trial = 0u; p1Trial < 2u; p1Trial++) {
				vec4 qe0 = expandEndpointMode6(r0, g0, b0, a0, p0Trial);
				vec4 qe1 = expandEndpointMode6(r1, g1, b1, a1, p1Trial);

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

				if (totalError < iterBestError) {
					iterBestError = totalError;
					for (uint i = 0u; i < 16u; i++) {
						iterBestIndices[i] = indices[i];
					}
				}
			}
		}

		// Refine endpoints from the best indices found this iteration.
		if (iter < 3u) {
			vec4 optE0, optE1;
			computeOptimalEndpoints(texels, iterBestIndices, optE0, optE1);

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
	vec4 qe0 = expandEndpointMode6(r0, g0, b0, a0, p0);
	vec4 qe1 = expandEndpointMode6(r1, g1, b1, a1, p1);

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

	uvec4 block = uvec4(0u);
	uint bitPos = 0u;

	bc7WriteBits(block, bitPos, 64u, 7u);
	bc7WriteBits(block, bitPos, r0, 7u);
	bc7WriteBits(block, bitPos, r1, 7u);
	bc7WriteBits(block, bitPos, g0, 7u);
	bc7WriteBits(block, bitPos, g1, 7u);
	bc7WriteBits(block, bitPos, b0, 7u);
	bc7WriteBits(block, bitPos, b1, 7u);
	bc7WriteBits(block, bitPos, a0, 7u);
	bc7WriteBits(block, bitPos, a1, 7u);
	bc7WriteBits(block, bitPos, p0, 1u);
	bc7WriteBits(block, bitPos, p1, 1u);

	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u) ? 3u : 4u;
		bc7WriteBits(block, bitPos, indices[i], bits);
	}

	return block;
}

// Encode Mode 5: 1 subset, 7-bit RGB + 8-bit A endpoints, separate 2-bit color/alpha indices
uvec4 encodeMode5(vec4 texels[16], out float outError) {
	vec4 e0, e1;
	findEndpoints(texels, 0u, 0u, 1u, e0, e1);

	uint r0 = quantize(e0.r, 7u);
	uint g0 = quantize(e0.g, 7u);
	uint b0 = quantize(e0.b, 7u);
	uint a0 = quantize(e0.a, 8u);
	uint r1 = quantize(e1.r, 7u);
	uint g1 = quantize(e1.g, 7u);
	uint b1 = quantize(e1.b, 7u);
	uint a1 = quantize(e1.a, 8u);

	vec4 qe0 = vec4(dequantize(r0, 7u), dequantize(g0, 7u), dequantize(b0, 7u), dequantize(a0, 8u));
	vec4 qe1 = vec4(dequantize(r1, 7u), dequantize(g1, 7u), dequantize(b1, 7u), dequantize(a1, 8u));

	uint colorIdx[16];
	uint alphaIdx[16];
	float totalError = 0.0;

	for (uint i = 0u; i < 16u; i++) {
		float bestColorErr = FLT_MAX;
		float bestAlphaErr = FLT_MAX;
		uint bestC = 0u;
		uint bestA = 0u;

		for (uint j = 0u; j < 4u; j++) {
			uint w = g_bc7_weights2[j];
			vec4 interp = interpolateColor(qe0, qe1, w);

			float cErr = colorErrorRGB(texels[i], interp);
			if (cErr < bestColorErr) {
				bestColorErr = cErr;
				bestC = j;
			}

			float aErr = alphaError(texels[i].a, interp.a);
			if (aErr < bestAlphaErr) {
				bestAlphaErr = aErr;
				bestA = j;
			}
		}

		colorIdx[i] = bestC;
		alphaIdx[i] = bestA;
		totalError += bestColorErr + bestAlphaErr;
	}

	// Anchor texel (0) omits MSB in both index planes.
	if (colorIdx[0] >= 2u) {
		uint t;
		t = r0;
		r0 = r1;
		r1 = t;
		t = g0;
		g0 = g1;
		g1 = t;
		t = b0;
		b0 = b1;
		b1 = t;

		for (uint i = 0u; i < 16u; i++) {
			colorIdx[i] = 3u - colorIdx[i];
		}
	}

	if (alphaIdx[0] >= 2u) {
		uint t;
		t = a0;
		a0 = a1;
		a1 = t;

		for (uint i = 0u; i < 16u; i++) {
			alphaIdx[i] = 3u - alphaIdx[i];
		}
	}

	outError = totalError;

	// Generic BC7 LSB-first pack.
	uvec4 block = uvec4(0u);
	uint bitPos = 0u;

	// Mode 5 selector: 0b100000 in 6 bits.
	bc7WriteBits(block, bitPos, 32u, 6u);

	// Rotation (2 bits). Keep alpha in A channel.
	bc7WriteBits(block, bitPos, 0u, 2u);

	// Endpoints: RGB (7 bits), then A (8 bits), ep0 then ep1.
	bc7WriteBits(block, bitPos, r0, 7u);
	bc7WriteBits(block, bitPos, r1, 7u);
	bc7WriteBits(block, bitPos, g0, 7u);
	bc7WriteBits(block, bitPos, g1, 7u);
	bc7WriteBits(block, bitPos, b0, 7u);
	bc7WriteBits(block, bitPos, b1, 7u);
	bc7WriteBits(block, bitPos, a0, 8u);
	bc7WriteBits(block, bitPos, a1, 8u);

	// Color indices: texel 0 anchor has 1 bit, others 2 bits.
	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u) ? 1u : 2u;
		bc7WriteBits(block, bitPos, colorIdx[i], bits);
	}

	// Alpha indices: texel 0 anchor has 1 bit, others 2 bits.
	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u) ? 1u : 2u;
		bc7WriteBits(block, bitPos, alphaIdx[i], bits);
	}

	return block;
}

// Encode Mode 4: 1 subset, 5-bit RGB + 6-bit A endpoints, separate indices (2/3 bits) + selector
uvec4 encodeMode4(vec4 texels[16], out float outError) {
	vec4 e0, e1;
	findEndpoints(texels, 0u, 0u, 1u, e0, e1);

	uint r0 = quantize(e0.r, 5u);
	uint g0 = quantize(e0.g, 5u);
	uint b0 = quantize(e0.b, 5u);
	uint a0 = quantize(e0.a, 6u);
	uint r1 = quantize(e1.r, 5u);
	uint g1 = quantize(e1.g, 5u);
	uint b1 = quantize(e1.b, 5u);
	uint a1 = quantize(e1.a, 6u);

	vec4 qe0 = vec4(dequantize(r0, 5u), dequantize(g0, 5u), dequantize(b0, 5u), dequantize(a0, 6u));
	vec4 qe1 = vec4(dequantize(r1, 5u), dequantize(g1, 5u), dequantize(b1, 5u), dequantize(a1, 6u));

	uint bestSelector = 0u; // 0: primary->RGB(2), secondary->A(3). 1: swapped.
	uint bestIdx0[16];
	uint bestIdx1[16];
	float bestError = FLT_MAX;

	for (uint selector = 0u; selector < 2u; selector++) {
		uint trialIdx0[16];
		uint trialIdx1[16];
		float trialError = 0.0;

		for (uint i = 0u; i < 16u; i++) {
			float bestRgbErr2 = FLT_MAX;
			float bestRgbErr3 = FLT_MAX;
			float bestAlphaErr2 = FLT_MAX;
			float bestAlphaErr3 = FLT_MAX;
			uint bestRgb2 = 0u;
			uint bestRgb3 = 0u;
			uint bestA2 = 0u;
			uint bestA3 = 0u;

			for (uint j = 0u; j < 4u; j++) {
				vec4 interp2 = interpolateColor(qe0, qe1, g_bc7_weights2[j]);
				float rgbErr2 = colorErrorRGB(texels[i], interp2);
				if (rgbErr2 < bestRgbErr2) {
					bestRgbErr2 = rgbErr2;
					bestRgb2 = j;
				}
				float aErr2 = alphaError(texels[i].a, interp2.a);
				if (aErr2 < bestAlphaErr2) {
					bestAlphaErr2 = aErr2;
					bestA2 = j;
				}
			}

			for (uint j = 0u; j < 8u; j++) {
				vec4 interp3 = interpolateColor(qe0, qe1, g_bc7_weights3[j]);
				float rgbErr3 = colorErrorRGB(texels[i], interp3);
				if (rgbErr3 < bestRgbErr3) {
					bestRgbErr3 = rgbErr3;
					bestRgb3 = j;
				}
				float aErr3 = alphaError(texels[i].a, interp3.a);
				if (aErr3 < bestAlphaErr3) {
					bestAlphaErr3 = aErr3;
					bestA3 = j;
				}
			}

			if (selector == 0u) {
				trialIdx0[i] = bestRgb2;
				trialIdx1[i] = bestA3;
				trialError += bestRgbErr2 + bestAlphaErr3;
			} else {
				trialIdx0[i] = bestA2;
				trialIdx1[i] = bestRgb3;
				trialError += bestRgbErr3 + bestAlphaErr2;
			}
		}

		if (trialError < bestError) {
			bestError = trialError;
			bestSelector = selector;
			for (uint i = 0u; i < 16u; i++) {
				bestIdx0[i] = trialIdx0[i];
				bestIdx1[i] = trialIdx1[i];
			}
		}
	}

	// Anchor texel 0: primary index stores 1 bit (of 2), secondary stores 2 bits (of 3).
	// Flip endpoint direction for the channel space that each index plane controls.
	if (bestIdx0[0] >= 2u) {
		if (bestSelector == 0u) {
			uint t;
			t = r0;
			r0 = r1;
			r1 = t;
			t = g0;
			g0 = g1;
			g1 = t;
			t = b0;
			b0 = b1;
			b1 = t;
		} else {
			uint t = a0;
			a0 = a1;
			a1 = t;
		}

		for (uint i = 0u; i < 16u; i++) {
			bestIdx0[i] = 3u - bestIdx0[i];
		}
	}

	if (bestIdx1[0] >= 4u) {
		if (bestSelector == 0u) {
			uint t = a0;
			a0 = a1;
			a1 = t;
		} else {
			uint t;
			t = r0;
			r0 = r1;
			r1 = t;
			t = g0;
			g0 = g1;
			g1 = t;
			t = b0;
			b0 = b1;
			b1 = t;
		}

		for (uint i = 0u; i < 16u; i++) {
			bestIdx1[i] = 7u - bestIdx1[i];
		}
	}

	outError = bestError;

	uvec4 block = uvec4(0u);
	uint bitPos = 0u;

	// Mode 4 selector: 0b10000 in 5 bits.
	bc7WriteBits(block, bitPos, 16u, 5u);

	// Rotation (2 bits) and index selector (1 bit).
	bc7WriteBits(block, bitPos, 0u, 2u);
	bc7WriteBits(block, bitPos, bestSelector, 1u);

	// RGB endpoints (5 bits), channel-major, subset=0, ep0 then ep1.
	bc7WriteBits(block, bitPos, r0, 5u);
	bc7WriteBits(block, bitPos, r1, 5u);
	bc7WriteBits(block, bitPos, g0, 5u);
	bc7WriteBits(block, bitPos, g1, 5u);
	bc7WriteBits(block, bitPos, b0, 5u);
	bc7WriteBits(block, bitPos, b1, 5u);

	// Alpha endpoints (6 bits).
	bc7WriteBits(block, bitPos, a0, 6u);
	bc7WriteBits(block, bitPos, a1, 6u);

	// Primary indices (2 bits, texel 0 uses 1 bit).
	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u) ? 1u : 2u;
		bc7WriteBits(block, bitPos, bestIdx0[i], bits);
	}

	// Secondary indices (3 bits, texel 0 uses 2 bits).
	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u) ? 2u : 3u;
		bc7WriteBits(block, bitPos, bestIdx1[i], bits);
	}

	return block;
}

// Encode Mode 7: 2 subsets, 5-bit RGBA endpoints + per-endpoint p-bit, 2-bit indices
uvec4 encodeMode7(vec4 texels[16], uint bestPartition, out float outError) {
	vec4 e0[2], e1[2];
	findEndpoints(texels, 0u, bestPartition, 2u, e0[0], e1[0]);
	findEndpoints(texels, 1u, bestPartition, 2u, e0[1], e1[1]);

	refineEndpoints(texels, 0u, bestPartition, 2u, 2u, e0[0], e1[0]);
	refineEndpoints(texels, 1u, bestPartition, 2u, 2u, e0[1], e1[1]);

	uint r0_0 = quantize(e0[0].r, 5u);
	uint g0_0 = quantize(e0[0].g, 5u);
	uint b0_0 = quantize(e0[0].b, 5u);
	uint a0_0 = quantize(e0[0].a, 5u);
	uint r1_0 = quantize(e1[0].r, 5u);
	uint g1_0 = quantize(e1[0].g, 5u);
	uint b1_0 = quantize(e1[0].b, 5u);
	uint a1_0 = quantize(e1[0].a, 5u);

	uint r0_1 = quantize(e0[1].r, 5u);
	uint g0_1 = quantize(e0[1].g, 5u);
	uint b0_1 = quantize(e0[1].b, 5u);
	uint a0_1 = quantize(e0[1].a, 5u);
	uint r1_1 = quantize(e1[1].r, 5u);
	uint g1_1 = quantize(e1[1].g, 5u);
	uint b1_1 = quantize(e1[1].b, 5u);
	uint a1_1 = quantize(e1[1].a, 5u);

	uint bestP00 = 0u;
	uint bestP10 = 0u;
	uint bestP01 = 0u;
	uint bestP11 = 0u;
	uint indices[16];
	float totalError = FLT_MAX;

	for (uint p00 = 0u; p00 < 2u; p00++) {
		for (uint p10 = 0u; p10 < 2u; p10++) {
			for (uint p01 = 0u; p01 < 2u; p01++) {
				for (uint p11 = 0u; p11 < 2u; p11++) {
					vec4 qe0[2], qe1[2];
					qe0[0] = vec4(dequantize((r0_0 << 1u) | p00, 6u),
							dequantize((g0_0 << 1u) | p00, 6u),
							dequantize((b0_0 << 1u) | p00, 6u),
							dequantize((a0_0 << 1u) | p00, 6u));
					qe1[0] = vec4(dequantize((r1_0 << 1u) | p10, 6u),
							dequantize((g1_0 << 1u) | p10, 6u),
							dequantize((b1_0 << 1u) | p10, 6u),
							dequantize((a1_0 << 1u) | p10, 6u));
					qe0[1] = vec4(dequantize((r0_1 << 1u) | p01, 6u),
							dequantize((g0_1 << 1u) | p01, 6u),
							dequantize((b0_1 << 1u) | p01, 6u),
							dequantize((a0_1 << 1u) | p01, 6u));
					qe1[1] = vec4(dequantize((r1_1 << 1u) | p11, 6u),
							dequantize((g1_1 << 1u) | p11, 6u),
							dequantize((b1_1 << 1u) | p11, 6u),
							dequantize((a1_1 << 1u) | p11, 6u));

					uint trialIdx[16];
					float trialError = 0.0;
					for (uint s = 0u; s < 2u; s++) {
						uint subsetIdx[16];
						float subsetErr;
						assignIndices(texels, s, bestPartition, 2u, qe0[s], qe1[s], 2u, subsetIdx, subsetErr);
						trialError += subsetErr;

						for (uint i = 0u; i < 16u; i++) {
							if (getPartition2(bestPartition, i) == s) {
								trialIdx[i] = subsetIdx[i];
							}
						}
					}

					if (trialError < totalError) {
						totalError = trialError;
						bestP00 = p00;
						bestP10 = p10;
						bestP01 = p01;
						bestP11 = p11;
						for (uint i = 0u; i < 16u; i++) {
							indices[i] = trialIdx[i];
						}
					}
				}
			}
		}
	}

	uint anchor1 = g_bc7_anchor2[bestPartition];

	if (indices[0] >= 2u) {
		uint t;
		t = r0_0;
		r0_0 = r1_0;
		r1_0 = t;
		t = g0_0;
		g0_0 = g1_0;
		g1_0 = t;
		t = b0_0;
		b0_0 = b1_0;
		b1_0 = t;
		t = a0_0;
		a0_0 = a1_0;
		a1_0 = t;
		t = bestP00;
		bestP00 = bestP10;
		bestP10 = t;

		for (uint i = 0u; i < 16u; i++) {
			if (getPartition2(bestPartition, i) == 0u) {
				indices[i] = 3u - indices[i];
			}
		}
	}

	if (indices[anchor1] >= 2u) {
		uint t;
		t = r0_1;
		r0_1 = r1_1;
		r1_1 = t;
		t = g0_1;
		g0_1 = g1_1;
		g1_1 = t;
		t = b0_1;
		b0_1 = b1_1;
		b1_1 = t;
		t = a0_1;
		a0_1 = a1_1;
		a1_1 = t;
		t = bestP01;
		bestP01 = bestP11;
		bestP11 = t;

		for (uint i = 0u; i < 16u; i++) {
			if (getPartition2(bestPartition, i) == 1u) {
				indices[i] = 3u - indices[i];
			}
		}
	}

	outError = totalError;

	uvec4 block = uvec4(0u);
	uint bitPos = 0u;

	// Mode 7 selector: 0b10000000 in 8 bits.
	bc7WriteBits(block, bitPos, 128u, 8u);
	bc7WriteBits(block, bitPos, bestPartition, 6u);

	// RGB endpoints (5 bits), channel-major, subset-major, endpoint-minor.
	bc7WriteBits(block, bitPos, r0_0, 5u);
	bc7WriteBits(block, bitPos, r1_0, 5u);
	bc7WriteBits(block, bitPos, r0_1, 5u);
	bc7WriteBits(block, bitPos, r1_1, 5u);

	bc7WriteBits(block, bitPos, g0_0, 5u);
	bc7WriteBits(block, bitPos, g1_0, 5u);
	bc7WriteBits(block, bitPos, g0_1, 5u);
	bc7WriteBits(block, bitPos, g1_1, 5u);

	bc7WriteBits(block, bitPos, b0_0, 5u);
	bc7WriteBits(block, bitPos, b1_0, 5u);
	bc7WriteBits(block, bitPos, b0_1, 5u);
	bc7WriteBits(block, bitPos, b1_1, 5u);

	// Alpha endpoints (5 bits), subset-major, endpoint-minor.
	bc7WriteBits(block, bitPos, a0_0, 5u);
	bc7WriteBits(block, bitPos, a1_0, 5u);
	bc7WriteBits(block, bitPos, a0_1, 5u);
	bc7WriteBits(block, bitPos, a1_1, 5u);

	// Per-endpoint p-bits.
	bc7WriteBits(block, bitPos, bestP00, 1u);
	bc7WriteBits(block, bitPos, bestP10, 1u);
	bc7WriteBits(block, bitPos, bestP01, 1u);
	bc7WriteBits(block, bitPos, bestP11, 1u);

	// Indices (2 bits), with anchor texels using 1 bit.
	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u || i == anchor1) ? 1u : 2u;
		bc7WriteBits(block, bitPos, indices[i], bits);
	}

	return block;
}

// Encode Mode 3: 2 subsets, 7-bit RGB endpoints + per-endpoint p-bit, 2-bit indices (alpha fixed to 1)
uvec4 encodeMode3(vec4 texels[16], uint bestPartition, out float outError) {
	vec4 e0[2], e1[2];
	findEndpoints(texels, 0u, bestPartition, 2u, e0[0], e1[0]);
	findEndpoints(texels, 1u, bestPartition, 2u, e0[1], e1[1]);

	refineEndpoints(texels, 0u, bestPartition, 2u, 2u, e0[0], e1[0]);
	refineEndpoints(texels, 1u, bestPartition, 2u, 2u, e0[1], e1[1]);

	uint r0_0 = quantize(e0[0].r, 7u);
	uint g0_0 = quantize(e0[0].g, 7u);
	uint b0_0 = quantize(e0[0].b, 7u);
	uint r1_0 = quantize(e1[0].r, 7u);
	uint g1_0 = quantize(e1[0].g, 7u);
	uint b1_0 = quantize(e1[0].b, 7u);

	uint r0_1 = quantize(e0[1].r, 7u);
	uint g0_1 = quantize(e0[1].g, 7u);
	uint b0_1 = quantize(e0[1].b, 7u);
	uint r1_1 = quantize(e1[1].r, 7u);
	uint g1_1 = quantize(e1[1].g, 7u);
	uint b1_1 = quantize(e1[1].b, 7u);

	uint bestP00 = 0u;
	uint bestP10 = 0u;
	uint bestP01 = 0u;
	uint bestP11 = 0u;
	uint indices[16];
	float totalError = FLT_MAX;

	for (uint p00 = 0u; p00 < 2u; p00++) {
		for (uint p10 = 0u; p10 < 2u; p10++) {
			for (uint p01 = 0u; p01 < 2u; p01++) {
				for (uint p11 = 0u; p11 < 2u; p11++) {
					vec4 qe0[2], qe1[2];
					qe0[0] = vec4(dequantize((r0_0 << 1u) | p00, 8u),
							dequantize((g0_0 << 1u) | p00, 8u),
							dequantize((b0_0 << 1u) | p00, 8u), 1.0);
					qe1[0] = vec4(dequantize((r1_0 << 1u) | p10, 8u),
							dequantize((g1_0 << 1u) | p10, 8u),
							dequantize((b1_0 << 1u) | p10, 8u), 1.0);
					qe0[1] = vec4(dequantize((r0_1 << 1u) | p01, 8u),
							dequantize((g0_1 << 1u) | p01, 8u),
							dequantize((b0_1 << 1u) | p01, 8u), 1.0);
					qe1[1] = vec4(dequantize((r1_1 << 1u) | p11, 8u),
							dequantize((g1_1 << 1u) | p11, 8u),
							dequantize((b1_1 << 1u) | p11, 8u), 1.0);

					uint trialIdx[16];
					float trialError = 0.0;
					for (uint s = 0u; s < 2u; s++) {
						uint subsetIdx[16];
						float subsetErr;
						assignIndices(texels, s, bestPartition, 2u, qe0[s], qe1[s], 2u, subsetIdx, subsetErr);
						trialError += subsetErr;

						for (uint i = 0u; i < 16u; i++) {
							if (getPartition2(bestPartition, i) == s) {
								trialIdx[i] = subsetIdx[i];
							}
						}
					}

					// Mode 3 has no alpha; decoded alpha is 1.0.
					for (uint i = 0u; i < 16u; i++) {
						trialError += alphaError(texels[i].a, 1.0);
					}

					if (trialError < totalError) {
						totalError = trialError;
						bestP00 = p00;
						bestP10 = p10;
						bestP01 = p01;
						bestP11 = p11;
						for (uint i = 0u; i < 16u; i++) {
							indices[i] = trialIdx[i];
						}
					}
				}
			}
		}
	}

	uint anchor1 = g_bc7_anchor2[bestPartition];

	if (indices[0] >= 2u) {
		uint t;
		t = r0_0;
		r0_0 = r1_0;
		r1_0 = t;
		t = g0_0;
		g0_0 = g1_0;
		g1_0 = t;
		t = b0_0;
		b0_0 = b1_0;
		b1_0 = t;
		t = bestP00;
		bestP00 = bestP10;
		bestP10 = t;

		for (uint i = 0u; i < 16u; i++) {
			if (getPartition2(bestPartition, i) == 0u) {
				indices[i] = 3u - indices[i];
			}
		}
	}

	if (indices[anchor1] >= 2u) {
		uint t;
		t = r0_1;
		r0_1 = r1_1;
		r1_1 = t;
		t = g0_1;
		g0_1 = g1_1;
		g1_1 = t;
		t = b0_1;
		b0_1 = b1_1;
		b1_1 = t;
		t = bestP01;
		bestP01 = bestP11;
		bestP11 = t;

		for (uint i = 0u; i < 16u; i++) {
			if (getPartition2(bestPartition, i) == 1u) {
				indices[i] = 3u - indices[i];
			}
		}
	}

	outError = totalError;

	uvec4 block = uvec4(0u);
	uint bitPos = 0u;

	// Mode 3 selector: 0b1000 in 4 bits.
	bc7WriteBits(block, bitPos, 8u, 4u);
	bc7WriteBits(block, bitPos, bestPartition, 6u);

	// RGB endpoints (7 bits), channel-major.
	bc7WriteBits(block, bitPos, r0_0, 7u);
	bc7WriteBits(block, bitPos, r1_0, 7u);
	bc7WriteBits(block, bitPos, r0_1, 7u);
	bc7WriteBits(block, bitPos, r1_1, 7u);

	bc7WriteBits(block, bitPos, g0_0, 7u);
	bc7WriteBits(block, bitPos, g1_0, 7u);
	bc7WriteBits(block, bitPos, g0_1, 7u);
	bc7WriteBits(block, bitPos, g1_1, 7u);

	bc7WriteBits(block, bitPos, b0_0, 7u);
	bc7WriteBits(block, bitPos, b1_0, 7u);
	bc7WriteBits(block, bitPos, b0_1, 7u);
	bc7WriteBits(block, bitPos, b1_1, 7u);

	// Per-endpoint p-bits.
	bc7WriteBits(block, bitPos, bestP00, 1u);
	bc7WriteBits(block, bitPos, bestP10, 1u);
	bc7WriteBits(block, bitPos, bestP01, 1u);
	bc7WriteBits(block, bitPos, bestP11, 1u);

	// Indices (2 bits), anchors use 1 bit.
	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u || i == anchor1) ? 1u : 2u;
		bc7WriteBits(block, bitPos, indices[i], bits);
	}

	return block;
}

// Encode Mode 2: 3 subsets, 5-bit RGB endpoints, 2-bit indices (alpha fixed to 1)
uvec4 encodeMode2(vec4 texels[16], uint bestPartition, out float outError) {
	vec4 e0[3], e1[3];
	for (uint s = 0u; s < 3u; s++) {
		findEndpoints(texels, s, bestPartition, 3u, e0[s], e1[s]);
		refineEndpoints(texels, s, bestPartition, 3u, 2u, e0[s], e1[s]);
	}

	uint r0[3], g0[3], b0[3], r1[3], g1[3], b1[3];
	for (uint s = 0u; s < 3u; s++) {
		r0[s] = quantize(e0[s].r, 5u);
		g0[s] = quantize(e0[s].g, 5u);
		b0[s] = quantize(e0[s].b, 5u);
		r1[s] = quantize(e1[s].r, 5u);
		g1[s] = quantize(e1[s].g, 5u);
		b1[s] = quantize(e1[s].b, 5u);
	}

	vec4 qe0[3], qe1[3];
	for (uint s = 0u; s < 3u; s++) {
		qe0[s] = vec4(dequantize(r0[s], 5u), dequantize(g0[s], 5u), dequantize(b0[s], 5u), 1.0);
		qe1[s] = vec4(dequantize(r1[s], 5u), dequantize(g1[s], 5u), dequantize(b1[s], 5u), 1.0);
	}

	uint indices[16];
	float totalError = 0.0;
	for (uint s = 0u; s < 3u; s++) {
		uint subsetIdx[16];
		float subsetErr;
		assignIndices(texels, s, bestPartition, 3u, qe0[s], qe1[s], 2u, subsetIdx, subsetErr);
		totalError += subsetErr;
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition3(bestPartition, i) == s) {
				indices[i] = subsetIdx[i];
			}
		}
	}

	// Mode 2 has no alpha; decoded alpha is 1.0.
	for (uint i = 0u; i < 16u; i++) {
		totalError += alphaError(texels[i].a, 1.0);
	}

	uint anchor1 = g_bc7_anchor3[bestPartition].x;
	uint anchor2 = g_bc7_anchor3[bestPartition].y;

	if (indices[0] >= 2u) {
		uint t;
		t = r0[0];
		r0[0] = r1[0];
		r1[0] = t;
		t = g0[0];
		g0[0] = g1[0];
		g1[0] = t;
		t = b0[0];
		b0[0] = b1[0];
		b1[0] = t;
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition3(bestPartition, i) == 0u) {
				indices[i] = 3u - indices[i];
			}
		}
	}

	if (indices[anchor1] >= 2u) {
		uint t;
		t = r0[1];
		r0[1] = r1[1];
		r1[1] = t;
		t = g0[1];
		g0[1] = g1[1];
		g1[1] = t;
		t = b0[1];
		b0[1] = b1[1];
		b1[1] = t;
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition3(bestPartition, i) == 1u) {
				indices[i] = 3u - indices[i];
			}
		}
	}

	if (indices[anchor2] >= 2u) {
		uint t;
		t = r0[2];
		r0[2] = r1[2];
		r1[2] = t;
		t = g0[2];
		g0[2] = g1[2];
		g1[2] = t;
		t = b0[2];
		b0[2] = b1[2];
		b1[2] = t;
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition3(bestPartition, i) == 2u) {
				indices[i] = 3u - indices[i];
			}
		}
	}

	outError = totalError;

	uvec4 block = uvec4(0u);
	uint bitPos = 0u;

	// Mode 2 selector: 0b100 in 3 bits.
	bc7WriteBits(block, bitPos, 4u, 3u);
	bc7WriteBits(block, bitPos, bestPartition, 6u);

	// RGB endpoints (5 bits), channel-major, subset-major, endpoint-minor.
	for (uint ch = 0u; ch < 3u; ch++) {
		for (uint s = 0u; s < 3u; s++) {
			if (ch == 0u) {
				bc7WriteBits(block, bitPos, r0[s], 5u);
				bc7WriteBits(block, bitPos, r1[s], 5u);
			} else if (ch == 1u) {
				bc7WriteBits(block, bitPos, g0[s], 5u);
				bc7WriteBits(block, bitPos, g1[s], 5u);
			} else {
				bc7WriteBits(block, bitPos, b0[s], 5u);
				bc7WriteBits(block, bitPos, b1[s], 5u);
			}
		}
	}

	// Indices (2 bits), anchors use 1 bit.
	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u || i == anchor1 || i == anchor2) ? 1u : 2u;
		bc7WriteBits(block, bitPos, indices[i], bits);
	}

	return block;
}

// Encode Mode 0: 3 subsets, 4-bit RGB endpoints + per-endpoint p-bit, 3-bit indices (alpha fixed to 1)
uvec4 encodeMode0(vec4 texels[16], uint bestPartition, out float outError) {
	vec4 e0[3], e1[3];
	for (uint s = 0u; s < 3u; s++) {
		findEndpoints(texels, s, bestPartition, 3u, e0[s], e1[s]);
		refineEndpoints(texels, s, bestPartition, 3u, 3u, e0[s], e1[s]);
	}

	uint r0[3], g0[3], b0[3], r1[3], g1[3], b1[3];
	for (uint s = 0u; s < 3u; s++) {
		r0[s] = quantize(e0[s].r, 4u);
		g0[s] = quantize(e0[s].g, 4u);
		b0[s] = quantize(e0[s].b, 4u);
		r1[s] = quantize(e1[s].r, 4u);
		g1[s] = quantize(e1[s].g, 4u);
		b1[s] = quantize(e1[s].b, 4u);
	}

	uint bestP[3][2];
	bestP[0][0] = 0u;
	bestP[0][1] = 0u;
	bestP[1][0] = 0u;
	bestP[1][1] = 0u;
	bestP[2][0] = 0u;
	bestP[2][1] = 0u;

	uint indices[16];
	float totalError = FLT_MAX;

	// Optimize: find best p-bits per subset independently (3 * 4 = 12 trials
	// instead of 2^6 = 64), since subsets don't interact through p-bits.
	uint subsetMap[16];
	for (uint i = 0u; i < 16u; i++) {
		subsetMap[i] = getPartition3(bestPartition, i);
	}

	float alphaErr = 0.0;
	for (uint i = 0u; i < 16u; i++) {
		alphaErr += alphaError(texels[i].a, 1.0);
	}

	for (uint s = 0u; s < 3u; s++) {
		float bestSubsetErr = FLT_MAX;

		for (uint pe0 = 0u; pe0 < 2u; pe0++) {
			for (uint pe1 = 0u; pe1 < 2u; pe1++) {
				vec4 qe0s = vec4(dequantize((r0[s] << 1u) | pe0, 5u),
						dequantize((g0[s] << 1u) | pe0, 5u),
						dequantize((b0[s] << 1u) | pe0, 5u), 1.0);
				vec4 qe1s = vec4(dequantize((r1[s] << 1u) | pe1, 5u),
						dequantize((g1[s] << 1u) | pe1, 5u),
						dequantize((b1[s] << 1u) | pe1, 5u), 1.0);

				uint subsetIdx[16];
				float subsetErr;
				assignIndices(texels, s, bestPartition, 3u, qe0s, qe1s, 3u, subsetIdx, subsetErr);

				if (subsetErr < bestSubsetErr) {
					bestSubsetErr = subsetErr;
					bestP[s][0] = pe0;
					bestP[s][1] = pe1;
					for (uint i = 0u; i < 16u; i++) {
						if (subsetMap[i] == s) {
							indices[i] = subsetIdx[i];
						}
					}
				}
			}
		}
	}

	totalError = alphaErr;
	for (uint s = 0u; s < 3u; s++) {
		vec4 qe0s = vec4(dequantize((r0[s] << 1u) | bestP[s][0], 5u),
				dequantize((g0[s] << 1u) | bestP[s][0], 5u),
				dequantize((b0[s] << 1u) | bestP[s][0], 5u), 1.0);
		vec4 qe1s = vec4(dequantize((r1[s] << 1u) | bestP[s][1], 5u),
				dequantize((g1[s] << 1u) | bestP[s][1], 5u),
				dequantize((b1[s] << 1u) | bestP[s][1], 5u), 1.0);
		uint tmpIdx[16];
		float subsetErr;
		assignIndices(texels, s, bestPartition, 3u, qe0s, qe1s, 3u, tmpIdx, subsetErr);
		totalError += subsetErr;
	}

	uint anchor1 = g_bc7_anchor3[bestPartition].x;
	uint anchor2 = g_bc7_anchor3[bestPartition].y;

	if (indices[0] >= 4u) {
		uint t;
		t = r0[0];
		r0[0] = r1[0];
		r1[0] = t;
		t = g0[0];
		g0[0] = g1[0];
		g1[0] = t;
		t = b0[0];
		b0[0] = b1[0];
		b1[0] = t;
		t = bestP[0][0];
		bestP[0][0] = bestP[0][1];
		bestP[0][1] = t;
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition3(bestPartition, i) == 0u) {
				indices[i] = 7u - indices[i];
			}
		}
	}

	if (indices[anchor1] >= 4u) {
		uint t;
		t = r0[1];
		r0[1] = r1[1];
		r1[1] = t;
		t = g0[1];
		g0[1] = g1[1];
		g1[1] = t;
		t = b0[1];
		b0[1] = b1[1];
		b1[1] = t;
		t = bestP[1][0];
		bestP[1][0] = bestP[1][1];
		bestP[1][1] = t;
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition3(bestPartition, i) == 1u) {
				indices[i] = 7u - indices[i];
			}
		}
	}

	if (indices[anchor2] >= 4u) {
		uint t;
		t = r0[2];
		r0[2] = r1[2];
		r1[2] = t;
		t = g0[2];
		g0[2] = g1[2];
		g1[2] = t;
		t = b0[2];
		b0[2] = b1[2];
		b1[2] = t;
		t = bestP[2][0];
		bestP[2][0] = bestP[2][1];
		bestP[2][1] = t;
		for (uint i = 0u; i < 16u; i++) {
			if (getPartition3(bestPartition, i) == 2u) {
				indices[i] = 7u - indices[i];
			}
		}
	}

	outError = totalError;

	uvec4 block = uvec4(0u);
	uint bitPos = 0u;

	// Mode 0 selector: 0b1 in 1 bit.
	bc7WriteBits(block, bitPos, 1u, 1u);
	bc7WriteBits(block, bitPos, bestPartition, 4u);

	// RGB endpoints (4 bits), channel-major, subset-major, endpoint-minor.
	for (uint ch = 0u; ch < 3u; ch++) {
		for (uint s = 0u; s < 3u; s++) {
			if (ch == 0u) {
				bc7WriteBits(block, bitPos, r0[s], 4u);
				bc7WriteBits(block, bitPos, r1[s], 4u);
			} else if (ch == 1u) {
				bc7WriteBits(block, bitPos, g0[s], 4u);
				bc7WriteBits(block, bitPos, g1[s], 4u);
			} else {
				bc7WriteBits(block, bitPos, b0[s], 4u);
				bc7WriteBits(block, bitPos, b1[s], 4u);
			}
		}
	}

	// Per-endpoint p-bits (subset-major, endpoint-minor).
	for (uint s = 0u; s < 3u; s++) {
		bc7WriteBits(block, bitPos, bestP[s][0], 1u);
		bc7WriteBits(block, bitPos, bestP[s][1], 1u);
	}

	// Indices (3 bits), anchors use 2 bits.
	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u || i == anchor1 || i == anchor2) ? 2u : 3u;
		bc7WriteBits(block, bitPos, indices[i], bits);
	}

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

	uvec4 block = uvec4(0u);
	uint bitPos = 0u;

	// Mode 1 selector: 0b10 in 2 bits.
	bc7WriteBits(block, bitPos, 2u, 2u);
	bc7WriteBits(block, bitPos, bestPartition, 6u);

	// RGB endpoints (6 bits), channel-major, subset-major, endpoint-minor.
	bc7WriteBits(block, bitPos, r0_0, 6u);
	bc7WriteBits(block, bitPos, r1_0, 6u);
	bc7WriteBits(block, bitPos, r0_1, 6u);
	bc7WriteBits(block, bitPos, r1_1, 6u);

	bc7WriteBits(block, bitPos, g0_0, 6u);
	bc7WriteBits(block, bitPos, g1_0, 6u);
	bc7WriteBits(block, bitPos, g0_1, 6u);
	bc7WriteBits(block, bitPos, g1_1, 6u);

	bc7WriteBits(block, bitPos, b0_0, 6u);
	bc7WriteBits(block, bitPos, b1_0, 6u);
	bc7WriteBits(block, bitPos, b0_1, 6u);
	bc7WriteBits(block, bitPos, b1_1, 6u);

	// Shared p-bits.
	bc7WriteBits(block, bitPos, p0, 1u);
	bc7WriteBits(block, bitPos, p1, 1u);

	// Indices (3 bits), anchors use 2 bits.
	for (uint i = 0u; i < 16u; i++) {
		uint bits = (i == 0u || i == anchor1) ? 2u : 3u;
		bc7WriteBits(block, bitPos, indices[i], bits);
	}

	return block;
}

// ============================================================================
// Partition Selection
// ============================================================================

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

		for (uint i = 0u; i < 16u; i++) {
			if (getPartition2(partitionIndex, i) == subset) {
				vec4 center = (minColor + maxColor) * 0.5;
				totalError += colorError(texels[i], center);
			}
		}
	}

	return totalError;
}

float evaluatePartition3(vec4 texels[16], uint partitionIndex) {
	float totalError = 0.0;

	for (uint subset = 0u; subset < 3u; subset++) {
		vec4 minColor = vec4(1.0);
		vec4 maxColor = vec4(0.0);

		for (uint i = 0u; i < 16u; i++) {
			if (getPartition3(partitionIndex, i) == subset) {
				minColor = min(minColor, texels[i]);
				maxColor = max(maxColor, texels[i]);
			}
		}

		for (uint i = 0u; i < 16u; i++) {
			if (getPartition3(partitionIndex, i) == subset) {
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

void findBestPartition3Both(vec4 texels[16], out uint bestFull, out uint bestFirst16) {
	float bestErrorFull = FLT_MAX;
	float bestError16 = FLT_MAX;
	bestFull = 0u;
	bestFirst16 = 0u;

	for (uint p = 0u; p < 64u; p++) {
		float error = evaluatePartition3(texels, p);
		if (error < bestErrorFull) {
			bestErrorFull = error;
			bestFull = p;
		}
		if (p < 16u && error < bestError16) {
			bestError16 = error;
			bestFirst16 = p;
		}
	}
}

void analyzeBlock(vec4 texels[16], out float outAlphaRange, out float outMaxRgbRange) {
	vec3 minRgb = vec3(1.0);
	vec3 maxRgb = vec3(0.0);
	float minA = 1.0;
	float maxA = 0.0;

	for (uint i = 0u; i < 16u; i++) {
		minRgb = min(minRgb, texels[i].rgb);
		maxRgb = max(maxRgb, texels[i].rgb);
		minA = min(minA, texels[i].a);
		maxA = max(maxA, texels[i].a);
	}

	outAlphaRange = maxA - minA;
	vec3 rgbRange = maxRgb - minRgb;
	outMaxRgbRange = max(rgbRange.r, max(rgbRange.g, rgbRange.b));
}

uvec4 compressBlock(vec4 texels[16]) {
	// Quality 0 (fast): Mode 6 only, minimal refinement
	// Quality 1 (balanced): Try Mode 6, Mode 5 and Mode 4
	// Quality 2 (high): Try all modes, pick best

	float bestError;
	uvec4 bestBlock = encodeMode6(texels, bestError);

	if (params.p_qualityLevel == 0u) {
		return bestBlock;
	}

	{
		float err;
		uvec4 blk = encodeMode5(texels, err);
		if (err < bestError) {
			bestError = err;
			bestBlock = blk;
		}
	}

	{
		float err;
		uvec4 blk = encodeMode4(texels, err);
		if (err < bestError) {
			bestError = err;
			bestBlock = blk;
		}
	}

	if (params.p_qualityLevel == 1u) {
		return bestBlock;
	}

	float alphaRange;
	float maxRgbRange;
	analyzeBlock(texels, alphaRange, maxRgbRange);

	bool alphaIsConstant = alphaRange < (2.0 / 255.0);
	bool alphaHasVariation = alphaRange > (12.0 / 255.0);
	bool colorIsFlat = maxRgbRange < 0.075;

	bool allowPartitioned = !colorIsFlat;
	bool allowAlphaLessModes = !alphaHasVariation;

	if (allowPartitioned) {
		uint bestPartition2 = findBestPartition2(texels);

		if (allowAlphaLessModes) {
			{
				float err;
				uvec4 blk = encodeMode1(texels, bestPartition2, err);
				if (err < bestError) {
					bestError = err;
					bestBlock = blk;
				}
			}
			{
				float err;
				uvec4 blk = encodeMode3(texels, bestPartition2, err);
				if (err < bestError) {
					bestError = err;
					bestBlock = blk;
				}
			}
		}

		if (!alphaIsConstant) {
			float err;
			uvec4 blk = encodeMode7(texels, bestPartition2, err);
			if (err < bestError) {
				bestError = err;
				bestBlock = blk;
			}
		}

		if (allowAlphaLessModes) {
			uint bestPartition3, bestPartition0;
			findBestPartition3Both(texels, bestPartition3, bestPartition0);
			{
				float err;
				uvec4 blk = encodeMode2(texels, bestPartition3, err);
				if (err < bestError) {
					bestError = err;
					bestBlock = blk;
				}
			}
			{
				float err;
				uvec4 blk = encodeMode0(texels, bestPartition0, err);
				if (err < bestError) {
					bestError = err;
					bestBlock = blk;
				}
			}
		}
	}

	return bestBlock;
}

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

void main() {
	// Gather texels for the current 4x4 block.
	vec2 uv = (vec2(gl_GlobalInvocationID.xy) * 4.0 + 1.0) * params.p_textureSizeRcp;
	vec2 block0UV = uv;
	vec2 block1UV = uv + vec2(2.0 * params.p_textureSizeRcp.x, 0.0);
	vec2 block2UV = uv + vec2(0.0, 2.0 * params.p_textureSizeRcp.y);
	vec2 block3UV = uv + vec2(2.0 * params.p_textureSizeRcp.x, 2.0 * params.p_textureSizeRcp.y);

	vec4 block0R = textureGather(srcTex, block0UV, 0);
	vec4 block1R = textureGather(srcTex, block1UV, 0);
	vec4 block2R = textureGather(srcTex, block2UV, 0);
	vec4 block3R = textureGather(srcTex, block3UV, 0);
	vec4 block0G = textureGather(srcTex, block0UV, 1);
	vec4 block1G = textureGather(srcTex, block1UV, 1);
	vec4 block2G = textureGather(srcTex, block2UV, 1);
	vec4 block3G = textureGather(srcTex, block3UV, 1);
	vec4 block0B = textureGather(srcTex, block0UV, 2);
	vec4 block1B = textureGather(srcTex, block1UV, 2);
	vec4 block2B = textureGather(srcTex, block2UV, 2);
	vec4 block3B = textureGather(srcTex, block3UV, 2);
	vec4 block0A = textureGather(srcTex, block0UV, 3);
	vec4 block1A = textureGather(srcTex, block1UV, 3);
	vec4 block2A = textureGather(srcTex, block2UV, 3);
	vec4 block3A = textureGather(srcTex, block3UV, 3);

	vec4 texels[16];
	texels[0] = vec4(block0R.w, block0G.w, block0B.w, block0A.w);
	texels[1] = vec4(block0R.z, block0G.z, block0B.z, block0A.z);
	texels[2] = vec4(block1R.w, block1G.w, block1B.w, block1A.w);
	texels[3] = vec4(block1R.z, block1G.z, block1B.z, block1A.z);
	texels[4] = vec4(block0R.x, block0G.x, block0B.x, block0A.x);
	texels[5] = vec4(block0R.y, block0G.y, block0B.y, block0A.y);
	texels[6] = vec4(block1R.x, block1G.x, block1B.x, block1A.x);
	texels[7] = vec4(block1R.y, block1G.y, block1B.y, block1A.y);
	texels[8] = vec4(block2R.w, block2G.w, block2B.w, block2A.w);
	texels[9] = vec4(block2R.z, block2G.z, block2B.z, block2A.z);
	texels[10] = vec4(block3R.w, block3G.w, block3B.w, block3A.w);
	texels[11] = vec4(block3R.z, block3G.z, block3B.z, block3A.z);
	texels[12] = vec4(block2R.x, block2G.x, block2B.x, block2A.x);
	texels[13] = vec4(block2R.y, block2G.y, block2B.y, block2A.y);
	texels[14] = vec4(block3R.x, block3G.x, block3B.x, block3A.x);
	texels[15] = vec4(block3R.y, block3G.y, block3B.y, block3A.y);

	uvec4 block = compressBlock(texels);
	imageStore(dstTexture, ivec2(gl_GlobalInvocationID.xy), block);
}
