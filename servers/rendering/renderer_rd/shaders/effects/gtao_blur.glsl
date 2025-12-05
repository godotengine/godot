///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// See `gtao.glsl` for details
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[compute]

#version 450

#VERSION_DEFINES

// This push_constant is full - 128 bytes - if you need to add more data, consider adding to the uniform buffer instead
layout(push_constant, std430) uniform Params {
	vec2 viewport_pixel_size;
	float blur_beta;
	int pad;
}
params;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_working_term;

layout(set = 0, binding = 1) uniform sampler2D source_edges;

layout(r8, set = 0, binding = 2) uniform restrict writeonly image2D dest_working_term;

vec4 unpack_edges(float p_packed_val) {
	uint packed_val = uint(p_packed_val * 255.5);
	vec4 edgesLRTB;
	edgesLRTB.x = float((packed_val >> 6) & 0x03) / 3.0;
	edgesLRTB.y = float((packed_val >> 4) & 0x03) / 3.0;
	edgesLRTB.z = float((packed_val >> 2) & 0x03) / 3.0;
	edgesLRTB.w = float((packed_val >> 0) & 0x03) / 3.0;

	return clamp(edgesLRTB, 0.0, 1.0);
}

void add_sample(float p_ao_term, float p_weight, inout float r_sum, inout float r_sum_weight) {
	r_sum += p_weight * p_ao_term;
	r_sum_weight += p_weight;
}

void main() {
	ivec2 pix_coord_base = ivec2(gl_GlobalInvocationID.xy) * ivec2(2, 1);

#ifdef GTAO_BLUR_FINAL_APPLY
	const float blur_amount = params.blur_beta;
#else
	const float blur_amount = params.blur_beta / 5.0;
#endif

	const float diag_weight = 0.85 * 0.5;

	// We calculate 2 horizontal pixels at a time for performance reasons
	// pixel pixCoordBase and pixel pixCoordBase + int2(1, 0)
	float aoTerm[2];
	vec4 edgesC_LRTB[2];
	float weightTL[2];
	float weightTR[2];
	float weightBL[2];
	float weightBR[2];

	const vec2 gather_center = pix_coord_base * params.viewport_pixel_size;
	const vec4 edgesQ0 = textureGatherOffset(source_edges, gather_center, ivec2(0, 0), 0);
	const vec4 edgesQ1 = textureGatherOffset(source_edges, gather_center, ivec2(2, 0), 0);
	const vec4 edgesQ2 = textureGatherOffset(source_edges, gather_center, ivec2(1, 2), 0);

	const vec4 visQ0 = textureGatherOffset(source_working_term, gather_center, ivec2(0, 0), 0);
	const vec4 visQ1 = textureGatherOffset(source_working_term, gather_center, ivec2(2, 0), 0);
	const vec4 visQ2 = textureGatherOffset(source_working_term, gather_center, ivec2(0, 2), 0);
	const vec4 visQ3 = textureGatherOffset(source_working_term, gather_center, ivec2(2, 2), 0);

	for (int side = 0; side < 2; side++) {
		const ivec2 pix_coord = ivec2(pix_coord_base.x + side, pix_coord_base.y);

		vec4 edgesL_LRTB = unpack_edges((side == 0) ? (edgesQ0.x) : (edgesQ0.y));
		vec4 edgesT_LRTB = unpack_edges((side == 0) ? (edgesQ0.z) : (edgesQ1.w));
		vec4 edgesR_LRTB = unpack_edges((side == 0) ? (edgesQ1.x) : (edgesQ1.y));
		vec4 edgesB_LRTB = unpack_edges((side == 0) ? (edgesQ2.w) : (edgesQ2.z));

		edgesC_LRTB[side] = unpack_edges((side == 0) ? (edgesQ0.y) : (edgesQ1.x));

		// Edges aren't perfectly symmetrical: edge detection algorithm does not guarantee that a left edge on the right pixel will match the right edge on the left pixel (although
		// they will match in majority of cases). This line further enforces the symmetricity, creating a slightly sharper blur. Works real nice with TAA.
		edgesC_LRTB[side] *= vec4(edgesL_LRTB.y, edgesR_LRTB.x, edgesT_LRTB.w, edgesB_LRTB.z);

		// this allows some small amount of AO leaking from neighbors if there are 3 or 4 edges; this reduces both spatial and temporal aliasing
		const float leak_threshold = 2.5;
		const float leak_strength = 0.5;
		float edginess = (clamp(4.0 - leak_threshold - dot(edgesC_LRTB[side], vec4(1.0)), 0.0, 1.0) / (4 - leak_threshold)) * leak_strength;
		edgesC_LRTB[side] = clamp(edgesC_LRTB[side] + edginess, 0.0, 1.0);

		// for diagonals; used by first and second pass
		weightTL[side] = diag_weight * (edgesC_LRTB[side].x * edgesL_LRTB.z + edgesC_LRTB[side].z * edgesT_LRTB.x);
		weightTR[side] = diag_weight * (edgesC_LRTB[side].z * edgesT_LRTB.y + edgesC_LRTB[side].y * edgesR_LRTB.z);
		weightBL[side] = diag_weight * (edgesC_LRTB[side].w * edgesB_LRTB.x + edgesC_LRTB[side].x * edgesL_LRTB.w);
		weightBR[side] = diag_weight * (edgesC_LRTB[side].y * edgesR_LRTB.w + edgesC_LRTB[side].w * edgesB_LRTB.y);

		// first pass
		float ssaoValue = (side == 0) ? (visQ0[1]) : (visQ1[0]);
		float ssaoValueL = (side == 0) ? (visQ0[0]) : (visQ0[1]);
		float ssaoValueT = (side == 0) ? (visQ0[2]) : (visQ1[3]);
		float ssaoValueR = (side == 0) ? (visQ1[0]) : (visQ1[1]);
		float ssaoValueB = (side == 0) ? (visQ2[2]) : (visQ3[3]);
		float ssaoValueTL = (side == 0) ? (visQ0[3]) : (visQ0[2]);
		float ssaoValueBR = (side == 0) ? (visQ3[3]) : (visQ3[2]);
		float ssaoValueTR = (side == 0) ? (visQ1[3]) : (visQ1[2]);
		float ssaoValueBL = (side == 0) ? (visQ2[3]) : (visQ2[2]);

		float sum_weight = blur_amount;
		float sum = ssaoValue * sum_weight;

		add_sample(ssaoValueL, edgesC_LRTB[side].x, sum, sum_weight);
		add_sample(ssaoValueR, edgesC_LRTB[side].y, sum, sum_weight);
		add_sample(ssaoValueT, edgesC_LRTB[side].z, sum, sum_weight);
		add_sample(ssaoValueB, edgesC_LRTB[side].w, sum, sum_weight);

		add_sample(ssaoValueTL, weightTL[side], sum, sum_weight);
		add_sample(ssaoValueTR, weightTR[side], sum, sum_weight);
		add_sample(ssaoValueBL, weightBL[side], sum, sum_weight);
		add_sample(ssaoValueBR, weightBR[side], sum, sum_weight);

		aoTerm[side] = sum / sum_weight;
		imageStore(dest_working_term, pix_coord, vec4(aoTerm[side], 0.0, 0.0, 0.0));
	}
}
