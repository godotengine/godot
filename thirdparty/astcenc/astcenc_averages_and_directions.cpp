// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2023 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

/**
 * @brief Functions for finding dominant direction of a set of colors.
 */
#if !defined(ASTCENC_DECOMPRESS_ONLY)

#include "astcenc_internal.h"

#include <cassert>

/**
 * @brief Compute the average RGB color of each partition.
 *
 * The algorithm here uses a vectorized sequential scan and per-partition
 * color accumulators, using select() to mask texel lanes in other partitions.
 *
 * We only accumulate sums for N-1 partitions during the scan; the value for
 * the last partition can be computed given that we know the block-wide average
 * already.
 *
 * Because of this we could reduce the loop iteration count so it "just" spans
 * the max texel index needed for the N-1 partitions, which could need fewer
 * iterations than the full block texel count. However, this makes the loop
 * count erratic and causes more branch mispredictions so is a net loss.
 *
 * @param      pi         The partitioning to use.
 * @param      blk        The block data to process.
 * @param[out] averages   The output averages. Unused partition indices will
 *                        not be initialized, and lane<3> will be zero.
 */
static void compute_partition_averages_rgb(
	const partition_info& pi,
	const image_block& blk,
	vfloat4 averages[BLOCK_MAX_PARTITIONS]
) {
	unsigned int partition_count = pi.partition_count;
	unsigned int texel_count = blk.texel_count;
	promise(texel_count > 0);

	// For 1 partition just use the precomputed mean
	if (partition_count == 1)
	{
		averages[0] = blk.data_mean.swz<0, 1, 2>();
	}
	// For 2 partitions scan results for partition 0, compute partition 1
	else if (partition_count == 2)
	{
		vfloatacc pp_avg_rgb[3] {};

		vint lane_id = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vint texel_partition(pi.partition_of_texel + i);

			vmask lane_mask = lane_id < vint(texel_count);
			lane_id += vint(ASTCENC_SIMD_WIDTH);

			vmask p0_mask = lane_mask & (texel_partition == vint(0));

			vfloat data_r = loada(blk.data_r + i);
			haccumulate(pp_avg_rgb[0], data_r, p0_mask);

			vfloat data_g = loada(blk.data_g + i);
			haccumulate(pp_avg_rgb[1], data_g, p0_mask);

			vfloat data_b = loada(blk.data_b + i);
			haccumulate(pp_avg_rgb[2], data_b, p0_mask);
		}

		vfloat4 block_total = blk.data_mean.swz<0, 1, 2>() * static_cast<float>(blk.texel_count);

		vfloat4 p0_total = vfloat3(hadd_s(pp_avg_rgb[0]),
		                           hadd_s(pp_avg_rgb[1]),
		                           hadd_s(pp_avg_rgb[2]));

		vfloat4 p1_total = block_total - p0_total;

		averages[0] = p0_total / static_cast<float>(pi.partition_texel_count[0]);
		averages[1] = p1_total / static_cast<float>(pi.partition_texel_count[1]);
	}
	// For 3 partitions scan results for partition 0/1, compute partition 2
	else if (partition_count == 3)
	{
		vfloatacc pp_avg_rgb[2][3] {};

		vint lane_id = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vint texel_partition(pi.partition_of_texel + i);

			vmask lane_mask = lane_id < vint(texel_count);
			lane_id += vint(ASTCENC_SIMD_WIDTH);

			vmask p0_mask = lane_mask & (texel_partition == vint(0));
			vmask p1_mask = lane_mask & (texel_partition == vint(1));

			vfloat data_r = loada(blk.data_r + i);
			haccumulate(pp_avg_rgb[0][0], data_r, p0_mask);
			haccumulate(pp_avg_rgb[1][0], data_r, p1_mask);

			vfloat data_g = loada(blk.data_g + i);
			haccumulate(pp_avg_rgb[0][1], data_g, p0_mask);
			haccumulate(pp_avg_rgb[1][1], data_g, p1_mask);

			vfloat data_b = loada(blk.data_b + i);
			haccumulate(pp_avg_rgb[0][2], data_b, p0_mask);
			haccumulate(pp_avg_rgb[1][2], data_b, p1_mask);
		}

		vfloat4 block_total = blk.data_mean.swz<0, 1, 2>() * static_cast<float>(blk.texel_count);

		vfloat4 p0_total = vfloat3(hadd_s(pp_avg_rgb[0][0]),
		                           hadd_s(pp_avg_rgb[0][1]),
		                           hadd_s(pp_avg_rgb[0][2]));

		vfloat4 p1_total = vfloat3(hadd_s(pp_avg_rgb[1][0]),
		                           hadd_s(pp_avg_rgb[1][1]),
		                           hadd_s(pp_avg_rgb[1][2]));

		vfloat4 p2_total = block_total - p0_total - p1_total;

		averages[0] = p0_total / static_cast<float>(pi.partition_texel_count[0]);
		averages[1] = p1_total / static_cast<float>(pi.partition_texel_count[1]);
		averages[2] = p2_total / static_cast<float>(pi.partition_texel_count[2]);
	}
	else
	{
		// For 4 partitions scan results for partition 0/1/2, compute partition 3
		vfloatacc pp_avg_rgb[3][3] {};

		vint lane_id = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vint texel_partition(pi.partition_of_texel + i);

			vmask lane_mask = lane_id < vint(texel_count);
			lane_id += vint(ASTCENC_SIMD_WIDTH);

			vmask p0_mask = lane_mask & (texel_partition == vint(0));
			vmask p1_mask = lane_mask & (texel_partition == vint(1));
			vmask p2_mask = lane_mask & (texel_partition == vint(2));

			vfloat data_r = loada(blk.data_r + i);
			haccumulate(pp_avg_rgb[0][0], data_r, p0_mask);
			haccumulate(pp_avg_rgb[1][0], data_r, p1_mask);
			haccumulate(pp_avg_rgb[2][0], data_r, p2_mask);

			vfloat data_g = loada(blk.data_g + i);
			haccumulate(pp_avg_rgb[0][1], data_g, p0_mask);
			haccumulate(pp_avg_rgb[1][1], data_g, p1_mask);
			haccumulate(pp_avg_rgb[2][1], data_g, p2_mask);

			vfloat data_b = loada(blk.data_b + i);
			haccumulate(pp_avg_rgb[0][2], data_b, p0_mask);
			haccumulate(pp_avg_rgb[1][2], data_b, p1_mask);
			haccumulate(pp_avg_rgb[2][2], data_b, p2_mask);
		}

		vfloat4 block_total = blk.data_mean.swz<0, 1, 2>() * static_cast<float>(blk.texel_count);

		vfloat4 p0_total = vfloat3(hadd_s(pp_avg_rgb[0][0]),
		                           hadd_s(pp_avg_rgb[0][1]),
		                           hadd_s(pp_avg_rgb[0][2]));

		vfloat4 p1_total = vfloat3(hadd_s(pp_avg_rgb[1][0]),
		                           hadd_s(pp_avg_rgb[1][1]),
		                           hadd_s(pp_avg_rgb[1][2]));

		vfloat4 p2_total = vfloat3(hadd_s(pp_avg_rgb[2][0]),
		                           hadd_s(pp_avg_rgb[2][1]),
		                           hadd_s(pp_avg_rgb[2][2]));

		vfloat4 p3_total = block_total - p0_total - p1_total- p2_total;

		averages[0] = p0_total / static_cast<float>(pi.partition_texel_count[0]);
		averages[1] = p1_total / static_cast<float>(pi.partition_texel_count[1]);
		averages[2] = p2_total / static_cast<float>(pi.partition_texel_count[2]);
		averages[3] = p3_total / static_cast<float>(pi.partition_texel_count[3]);
	}
}

/**
 * @brief Compute the average RGBA color of each partition.
 *
 * The algorithm here uses a vectorized sequential scan and per-partition
 * color accumulators, using select() to mask texel lanes in other partitions.
 *
 * We only accumulate sums for N-1 partitions during the scan; the value for
 * the last partition can be computed given that we know the block-wide average
 * already.
 *
 * Because of this we could reduce the loop iteration count so it "just" spans
 * the max texel index needed for the N-1 partitions, which could need fewer
 * iterations than the full block texel count. However, this makes the loop
 * count erratic and causes more branch mispredictions so is a net loss.
 *
 * @param      pi         The partitioning to use.
 * @param      blk        The block data to process.
 * @param[out] averages   The output averages. Unused partition indices will
 *                        not be initialized.
 */
static void compute_partition_averages_rgba(
	const partition_info& pi,
	const image_block& blk,
	vfloat4 averages[BLOCK_MAX_PARTITIONS]
) {
	unsigned int partition_count = pi.partition_count;
	unsigned int texel_count = blk.texel_count;
	promise(texel_count > 0);

	// For 1 partition just use the precomputed mean
	if (partition_count == 1)
	{
		averages[0] = blk.data_mean;
	}
	// For 2 partitions scan results for partition 0, compute partition 1
	else if (partition_count == 2)
	{
		vfloat4 pp_avg_rgba[4] {};

		vint lane_id = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vint texel_partition(pi.partition_of_texel + i);

			vmask lane_mask = lane_id < vint(texel_count);
			lane_id += vint(ASTCENC_SIMD_WIDTH);

			vmask p0_mask = lane_mask & (texel_partition == vint(0));

			vfloat data_r = loada(blk.data_r + i);
			haccumulate(pp_avg_rgba[0], data_r, p0_mask);

			vfloat data_g = loada(blk.data_g + i);
			haccumulate(pp_avg_rgba[1], data_g, p0_mask);

			vfloat data_b = loada(blk.data_b + i);
			haccumulate(pp_avg_rgba[2], data_b, p0_mask);

			vfloat data_a = loada(blk.data_a + i);
			haccumulate(pp_avg_rgba[3], data_a, p0_mask);
		}

		vfloat4 block_total = blk.data_mean * static_cast<float>(blk.texel_count);

		vfloat4 p0_total = vfloat4(hadd_s(pp_avg_rgba[0]),
		                           hadd_s(pp_avg_rgba[1]),
		                           hadd_s(pp_avg_rgba[2]),
		                           hadd_s(pp_avg_rgba[3]));

		vfloat4 p1_total = block_total - p0_total;

		averages[0] = p0_total / static_cast<float>(pi.partition_texel_count[0]);
		averages[1] = p1_total / static_cast<float>(pi.partition_texel_count[1]);
	}
	// For 3 partitions scan results for partition 0/1, compute partition 2
	else if (partition_count == 3)
	{
		vfloat4 pp_avg_rgba[2][4] {};

		vint lane_id = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vint texel_partition(pi.partition_of_texel + i);

			vmask lane_mask = lane_id < vint(texel_count);
			lane_id += vint(ASTCENC_SIMD_WIDTH);

			vmask p0_mask = lane_mask & (texel_partition == vint(0));
			vmask p1_mask = lane_mask & (texel_partition == vint(1));

			vfloat data_r = loada(blk.data_r + i);
			haccumulate(pp_avg_rgba[0][0], data_r, p0_mask);
			haccumulate(pp_avg_rgba[1][0], data_r, p1_mask);

			vfloat data_g = loada(blk.data_g + i);
			haccumulate(pp_avg_rgba[0][1], data_g, p0_mask);
			haccumulate(pp_avg_rgba[1][1], data_g, p1_mask);

			vfloat data_b = loada(blk.data_b + i);
			haccumulate(pp_avg_rgba[0][2], data_b, p0_mask);
			haccumulate(pp_avg_rgba[1][2], data_b, p1_mask);

			vfloat data_a = loada(blk.data_a + i);
			haccumulate(pp_avg_rgba[0][3], data_a, p0_mask);
			haccumulate(pp_avg_rgba[1][3], data_a, p1_mask);
		}

		vfloat4 block_total = blk.data_mean * static_cast<float>(blk.texel_count);

		vfloat4 p0_total = vfloat4(hadd_s(pp_avg_rgba[0][0]),
		                           hadd_s(pp_avg_rgba[0][1]),
		                           hadd_s(pp_avg_rgba[0][2]),
		                           hadd_s(pp_avg_rgba[0][3]));

		vfloat4 p1_total = vfloat4(hadd_s(pp_avg_rgba[1][0]),
		                           hadd_s(pp_avg_rgba[1][1]),
		                           hadd_s(pp_avg_rgba[1][2]),
		                           hadd_s(pp_avg_rgba[1][3]));

		vfloat4 p2_total = block_total - p0_total - p1_total;

		averages[0] = p0_total / static_cast<float>(pi.partition_texel_count[0]);
		averages[1] = p1_total / static_cast<float>(pi.partition_texel_count[1]);
		averages[2] = p2_total / static_cast<float>(pi.partition_texel_count[2]);
	}
	else
	{
		// For 4 partitions scan results for partition 0/1/2, compute partition 3
		vfloat4 pp_avg_rgba[3][4] {};

		vint lane_id = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vint texel_partition(pi.partition_of_texel + i);

			vmask lane_mask = lane_id < vint(texel_count);
			lane_id += vint(ASTCENC_SIMD_WIDTH);

			vmask p0_mask = lane_mask & (texel_partition == vint(0));
			vmask p1_mask = lane_mask & (texel_partition == vint(1));
			vmask p2_mask = lane_mask & (texel_partition == vint(2));

			vfloat data_r = loada(blk.data_r + i);
			haccumulate(pp_avg_rgba[0][0], data_r, p0_mask);
			haccumulate(pp_avg_rgba[1][0], data_r, p1_mask);
			haccumulate(pp_avg_rgba[2][0], data_r, p2_mask);

			vfloat data_g = loada(blk.data_g + i);
			haccumulate(pp_avg_rgba[0][1], data_g, p0_mask);
			haccumulate(pp_avg_rgba[1][1], data_g, p1_mask);
			haccumulate(pp_avg_rgba[2][1], data_g, p2_mask);

			vfloat data_b = loada(blk.data_b + i);
			haccumulate(pp_avg_rgba[0][2], data_b, p0_mask);
			haccumulate(pp_avg_rgba[1][2], data_b, p1_mask);
			haccumulate(pp_avg_rgba[2][2], data_b, p2_mask);

			vfloat data_a = loada(blk.data_a + i);
			haccumulate(pp_avg_rgba[0][3], data_a, p0_mask);
			haccumulate(pp_avg_rgba[1][3], data_a, p1_mask);
			haccumulate(pp_avg_rgba[2][3], data_a, p2_mask);
		}

		vfloat4 block_total = blk.data_mean * static_cast<float>(blk.texel_count);

		vfloat4 p0_total = vfloat4(hadd_s(pp_avg_rgba[0][0]),
		                           hadd_s(pp_avg_rgba[0][1]),
		                           hadd_s(pp_avg_rgba[0][2]),
		                           hadd_s(pp_avg_rgba[0][3]));

		vfloat4 p1_total = vfloat4(hadd_s(pp_avg_rgba[1][0]),
		                           hadd_s(pp_avg_rgba[1][1]),
		                           hadd_s(pp_avg_rgba[1][2]),
		                           hadd_s(pp_avg_rgba[1][3]));

		vfloat4 p2_total = vfloat4(hadd_s(pp_avg_rgba[2][0]),
		                           hadd_s(pp_avg_rgba[2][1]),
		                           hadd_s(pp_avg_rgba[2][2]),
		                           hadd_s(pp_avg_rgba[2][3]));

		vfloat4 p3_total = block_total - p0_total - p1_total- p2_total;

		averages[0] = p0_total / static_cast<float>(pi.partition_texel_count[0]);
		averages[1] = p1_total / static_cast<float>(pi.partition_texel_count[1]);
		averages[2] = p2_total / static_cast<float>(pi.partition_texel_count[2]);
		averages[3] = p3_total / static_cast<float>(pi.partition_texel_count[3]);
	}
}

/* See header for documentation. */
void compute_avgs_and_dirs_4_comp(
	const partition_info& pi,
	const image_block& blk,
	partition_metrics pm[BLOCK_MAX_PARTITIONS]
) {
	int partition_count = pi.partition_count;
	promise(partition_count > 0);

	// Pre-compute partition_averages
	vfloat4 partition_averages[BLOCK_MAX_PARTITIONS];
	compute_partition_averages_rgba(pi, blk, partition_averages);

	for (int partition = 0; partition < partition_count; partition++)
	{
		const uint8_t *texel_indexes = pi.texels_of_partition[partition];
		unsigned int texel_count = pi.partition_texel_count[partition];
		promise(texel_count > 0);

		vfloat4 average = partition_averages[partition];
		pm[partition].avg = average;

		vfloat4 sum_xp = vfloat4::zero();
		vfloat4 sum_yp = vfloat4::zero();
		vfloat4 sum_zp = vfloat4::zero();
		vfloat4 sum_wp = vfloat4::zero();

		for (unsigned int i = 0; i < texel_count; i++)
		{
			unsigned int iwt = texel_indexes[i];
			vfloat4 texel_datum = blk.texel(iwt);
			texel_datum = texel_datum - average;

			vfloat4 zero = vfloat4::zero();

			vmask4 tdm0 = texel_datum.swz<0,0,0,0>() > zero;
			sum_xp += select(zero, texel_datum, tdm0);

			vmask4 tdm1 = texel_datum.swz<1,1,1,1>() > zero;
			sum_yp += select(zero, texel_datum, tdm1);

			vmask4 tdm2 = texel_datum.swz<2,2,2,2>() > zero;
			sum_zp += select(zero, texel_datum, tdm2);

			vmask4 tdm3 = texel_datum.swz<3,3,3,3>() > zero;
			sum_wp += select(zero, texel_datum, tdm3);
		}

		vfloat4 prod_xp = dot(sum_xp, sum_xp);
		vfloat4 prod_yp = dot(sum_yp, sum_yp);
		vfloat4 prod_zp = dot(sum_zp, sum_zp);
		vfloat4 prod_wp = dot(sum_wp, sum_wp);

		vfloat4 best_vector = sum_xp;
		vfloat4 best_sum = prod_xp;

		vmask4 mask = prod_yp > best_sum;
		best_vector = select(best_vector, sum_yp, mask);
		best_sum = select(best_sum, prod_yp, mask);

		mask = prod_zp > best_sum;
		best_vector = select(best_vector, sum_zp, mask);
		best_sum = select(best_sum, prod_zp, mask);

		mask = prod_wp > best_sum;
		best_vector = select(best_vector, sum_wp, mask);

		pm[partition].dir = best_vector;
	}
}

/* See header for documentation. */
void compute_avgs_and_dirs_3_comp(
	const partition_info& pi,
	const image_block& blk,
	unsigned int omitted_component,
	partition_metrics pm[BLOCK_MAX_PARTITIONS]
) {
	// Pre-compute partition_averages
	vfloat4 partition_averages[BLOCK_MAX_PARTITIONS];
	compute_partition_averages_rgba(pi, blk, partition_averages);

	const float* data_vr = blk.data_r;
	const float* data_vg = blk.data_g;
	const float* data_vb = blk.data_b;

	// TODO: Data-driven permute would be useful to avoid this ...
	if (omitted_component == 0)
	{
		partition_averages[0] = partition_averages[0].swz<1, 2, 3>();
		partition_averages[1] = partition_averages[1].swz<1, 2, 3>();
		partition_averages[2] = partition_averages[2].swz<1, 2, 3>();
		partition_averages[3] = partition_averages[3].swz<1, 2, 3>();

		data_vr = blk.data_g;
		data_vg = blk.data_b;
		data_vb = blk.data_a;
	}
	else if (omitted_component == 1)
	{
		partition_averages[0] = partition_averages[0].swz<0, 2, 3>();
		partition_averages[1] = partition_averages[1].swz<0, 2, 3>();
		partition_averages[2] = partition_averages[2].swz<0, 2, 3>();
		partition_averages[3] = partition_averages[3].swz<0, 2, 3>();

		data_vg = blk.data_b;
		data_vb = blk.data_a;
	}
	else if (omitted_component == 2)
	{
		partition_averages[0] = partition_averages[0].swz<0, 1, 3>();
		partition_averages[1] = partition_averages[1].swz<0, 1, 3>();
		partition_averages[2] = partition_averages[2].swz<0, 1, 3>();
		partition_averages[3] = partition_averages[3].swz<0, 1, 3>();

		data_vb = blk.data_a;
	}
	else
	{
		partition_averages[0] = partition_averages[0].swz<0, 1, 2>();
		partition_averages[1] = partition_averages[1].swz<0, 1, 2>();
		partition_averages[2] = partition_averages[2].swz<0, 1, 2>();
		partition_averages[3] = partition_averages[3].swz<0, 1, 2>();
	}

	unsigned int partition_count = pi.partition_count;
	promise(partition_count > 0);

	for (unsigned int partition = 0; partition < partition_count; partition++)
	{
		const uint8_t *texel_indexes = pi.texels_of_partition[partition];
		unsigned int texel_count = pi.partition_texel_count[partition];
		promise(texel_count > 0);

		vfloat4 average = partition_averages[partition];
		pm[partition].avg = average;

		vfloat4 sum_xp = vfloat4::zero();
		vfloat4 sum_yp = vfloat4::zero();
		vfloat4 sum_zp = vfloat4::zero();

		for (unsigned int i = 0; i < texel_count; i++)
		{
			unsigned int iwt = texel_indexes[i];

			vfloat4 texel_datum = vfloat3(data_vr[iwt],
			                              data_vg[iwt],
			                              data_vb[iwt]);
			texel_datum = texel_datum - average;

			vfloat4 zero = vfloat4::zero();

			vmask4 tdm0 = texel_datum.swz<0,0,0,0>() > zero;
			sum_xp += select(zero, texel_datum, tdm0);

			vmask4 tdm1 = texel_datum.swz<1,1,1,1>() > zero;
			sum_yp += select(zero, texel_datum, tdm1);

			vmask4 tdm2 = texel_datum.swz<2,2,2,2>() > zero;
			sum_zp += select(zero, texel_datum, tdm2);
		}

		vfloat4 prod_xp = dot(sum_xp, sum_xp);
		vfloat4 prod_yp = dot(sum_yp, sum_yp);
		vfloat4 prod_zp = dot(sum_zp, sum_zp);

		vfloat4 best_vector = sum_xp;
		vfloat4 best_sum = prod_xp;

		vmask4 mask = prod_yp > best_sum;
		best_vector = select(best_vector, sum_yp, mask);
		best_sum = select(best_sum, prod_yp, mask);

		mask = prod_zp > best_sum;
		best_vector = select(best_vector, sum_zp, mask);

		pm[partition].dir = best_vector;
	}
}

/* See header for documentation. */
void compute_avgs_and_dirs_3_comp_rgb(
	const partition_info& pi,
	const image_block& blk,
	partition_metrics pm[BLOCK_MAX_PARTITIONS]
) {
	unsigned int partition_count = pi.partition_count;
	promise(partition_count > 0);

	// Pre-compute partition_averages
	vfloat4 partition_averages[BLOCK_MAX_PARTITIONS];
	compute_partition_averages_rgb(pi, blk, partition_averages);

	for (unsigned int partition = 0; partition < partition_count; partition++)
	{
		const uint8_t *texel_indexes = pi.texels_of_partition[partition];
		unsigned int texel_count = pi.partition_texel_count[partition];
		promise(texel_count > 0);

		vfloat4 average = partition_averages[partition];
		pm[partition].avg = average;

		vfloat4 sum_xp = vfloat4::zero();
		vfloat4 sum_yp = vfloat4::zero();
		vfloat4 sum_zp = vfloat4::zero();

		for (unsigned int i = 0; i < texel_count; i++)
		{
			unsigned int iwt = texel_indexes[i];

			vfloat4 texel_datum = blk.texel3(iwt);
			texel_datum = texel_datum - average;

			vfloat4 zero = vfloat4::zero();

			vmask4 tdm0 = texel_datum.swz<0,0,0,0>() > zero;
			sum_xp += select(zero, texel_datum, tdm0);

			vmask4 tdm1 = texel_datum.swz<1,1,1,1>() > zero;
			sum_yp += select(zero, texel_datum, tdm1);

			vmask4 tdm2 = texel_datum.swz<2,2,2,2>() > zero;
			sum_zp += select(zero, texel_datum, tdm2);
		}

		vfloat4 prod_xp = dot(sum_xp, sum_xp);
		vfloat4 prod_yp = dot(sum_yp, sum_yp);
		vfloat4 prod_zp = dot(sum_zp, sum_zp);

		vfloat4 best_vector = sum_xp;
		vfloat4 best_sum = prod_xp;

		vmask4 mask = prod_yp > best_sum;
		best_vector = select(best_vector, sum_yp, mask);
		best_sum = select(best_sum, prod_yp, mask);

		mask = prod_zp > best_sum;
		best_vector = select(best_vector, sum_zp, mask);

		pm[partition].dir = best_vector;
	}
}

/* See header for documentation. */
void compute_avgs_and_dirs_2_comp(
	const partition_info& pt,
	const image_block& blk,
	unsigned int component1,
	unsigned int component2,
	partition_metrics pm[BLOCK_MAX_PARTITIONS]
) {
	vfloat4 average;

	const float* data_vr = nullptr;
	const float* data_vg = nullptr;

	if (component1 == 0 && component2 == 1)
	{
		average = blk.data_mean.swz<0, 1>();

		data_vr = blk.data_r;
		data_vg = blk.data_g;
	}
	else if (component1 == 0 && component2 == 2)
	{
		average = blk.data_mean.swz<0, 2>();

		data_vr = blk.data_r;
		data_vg = blk.data_b;
	}
	else // (component1 == 1 && component2 == 2)
	{
		assert(component1 == 1 && component2 == 2);

		average = blk.data_mean.swz<1, 2>();

		data_vr = blk.data_g;
		data_vg = blk.data_b;
	}

	unsigned int partition_count = pt.partition_count;
	promise(partition_count > 0);

	for (unsigned int partition = 0; partition < partition_count; partition++)
	{
		const uint8_t *texel_indexes = pt.texels_of_partition[partition];
		unsigned int texel_count = pt.partition_texel_count[partition];
		promise(texel_count > 0);

		// Only compute a partition mean if more than one partition
		if (partition_count > 1)
		{
			average = vfloat4::zero();
			for (unsigned int i = 0; i < texel_count; i++)
			{
				unsigned int iwt = texel_indexes[i];
				average += vfloat2(data_vr[iwt], data_vg[iwt]);
			}

			average = average / static_cast<float>(texel_count);
		}

		pm[partition].avg = average;

		vfloat4 sum_xp = vfloat4::zero();
		vfloat4 sum_yp = vfloat4::zero();

		for (unsigned int i = 0; i < texel_count; i++)
		{
			unsigned int iwt = texel_indexes[i];
			vfloat4 texel_datum = vfloat2(data_vr[iwt], data_vg[iwt]);
			texel_datum = texel_datum - average;

			vfloat4 zero = vfloat4::zero();

			vmask4 tdm0 = texel_datum.swz<0,0,0,0>() > zero;
			sum_xp += select(zero, texel_datum, tdm0);

			vmask4 tdm1 = texel_datum.swz<1,1,1,1>() > zero;
			sum_yp += select(zero, texel_datum, tdm1);
		}

		vfloat4 prod_xp = dot(sum_xp, sum_xp);
		vfloat4 prod_yp = dot(sum_yp, sum_yp);

		vfloat4 best_vector = sum_xp;
		vfloat4 best_sum = prod_xp;

		vmask4 mask = prod_yp > best_sum;
		best_vector = select(best_vector, sum_yp, mask);

		pm[partition].dir = best_vector;
	}
}

/* See header for documentation. */
void compute_error_squared_rgba(
	const partition_info& pi,
	const image_block& blk,
	const processed_line4 uncor_plines[BLOCK_MAX_PARTITIONS],
	const processed_line4 samec_plines[BLOCK_MAX_PARTITIONS],
	float line_lengths[BLOCK_MAX_PARTITIONS],
	float& uncor_error,
	float& samec_error
) {
	unsigned int partition_count = pi.partition_count;
	promise(partition_count > 0);

	vfloatacc uncor_errorsumv = vfloatacc::zero();
	vfloatacc samec_errorsumv = vfloatacc::zero();

	for (unsigned int partition = 0; partition < partition_count; partition++)
	{
		const uint8_t *texel_indexes = pi.texels_of_partition[partition];

		processed_line4 l_uncor = uncor_plines[partition];
		processed_line4 l_samec = samec_plines[partition];

		unsigned int texel_count = pi.partition_texel_count[partition];
		promise(texel_count > 0);

		// Vectorize some useful scalar inputs
		vfloat l_uncor_bs0(l_uncor.bs.lane<0>());
		vfloat l_uncor_bs1(l_uncor.bs.lane<1>());
		vfloat l_uncor_bs2(l_uncor.bs.lane<2>());
		vfloat l_uncor_bs3(l_uncor.bs.lane<3>());

		vfloat l_uncor_amod0(l_uncor.amod.lane<0>());
		vfloat l_uncor_amod1(l_uncor.amod.lane<1>());
		vfloat l_uncor_amod2(l_uncor.amod.lane<2>());
		vfloat l_uncor_amod3(l_uncor.amod.lane<3>());

		vfloat l_samec_bs0(l_samec.bs.lane<0>());
		vfloat l_samec_bs1(l_samec.bs.lane<1>());
		vfloat l_samec_bs2(l_samec.bs.lane<2>());
		vfloat l_samec_bs3(l_samec.bs.lane<3>());

		assert(all(l_samec.amod == vfloat4(0.0f)));

		vfloat uncor_loparamv(1e10f);
		vfloat uncor_hiparamv(-1e10f);

		vfloat ew_r(blk.channel_weight.lane<0>());
		vfloat ew_g(blk.channel_weight.lane<1>());
		vfloat ew_b(blk.channel_weight.lane<2>());
		vfloat ew_a(blk.channel_weight.lane<3>());

		// This implementation over-shoots, but this is safe as we initialize the texel_indexes
		// array to extend the last value. This means min/max are not impacted, but we need to mask
		// out the dummy values when we compute the line weighting.
		vint lane_ids = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vmask mask = lane_ids < vint(texel_count);
			vint texel_idxs(texel_indexes + i);

			vfloat data_r = gatherf(blk.data_r, texel_idxs);
			vfloat data_g = gatherf(blk.data_g, texel_idxs);
			vfloat data_b = gatherf(blk.data_b, texel_idxs);
			vfloat data_a = gatherf(blk.data_a, texel_idxs);

			vfloat uncor_param = (data_r * l_uncor_bs0)
			                   + (data_g * l_uncor_bs1)
			                   + (data_b * l_uncor_bs2)
			                   + (data_a * l_uncor_bs3);

			uncor_loparamv = min(uncor_param, uncor_loparamv);
			uncor_hiparamv = max(uncor_param, uncor_hiparamv);

			vfloat uncor_dist0 = (l_uncor_amod0 - data_r)
			                   + (uncor_param * l_uncor_bs0);
			vfloat uncor_dist1 = (l_uncor_amod1 - data_g)
			                   + (uncor_param * l_uncor_bs1);
			vfloat uncor_dist2 = (l_uncor_amod2 - data_b)
			                   + (uncor_param * l_uncor_bs2);
			vfloat uncor_dist3 = (l_uncor_amod3 - data_a)
			                   + (uncor_param * l_uncor_bs3);

			vfloat uncor_err = (ew_r * uncor_dist0 * uncor_dist0)
			                 + (ew_g * uncor_dist1 * uncor_dist1)
			                 + (ew_b * uncor_dist2 * uncor_dist2)
			                 + (ew_a * uncor_dist3 * uncor_dist3);

			haccumulate(uncor_errorsumv, uncor_err, mask);

			// Process samechroma data
			vfloat samec_param = (data_r * l_samec_bs0)
			                   + (data_g * l_samec_bs1)
			                   + (data_b * l_samec_bs2)
			                   + (data_a * l_samec_bs3);

			vfloat samec_dist0 = samec_param * l_samec_bs0 - data_r;
			vfloat samec_dist1 = samec_param * l_samec_bs1 - data_g;
			vfloat samec_dist2 = samec_param * l_samec_bs2 - data_b;
			vfloat samec_dist3 = samec_param * l_samec_bs3 - data_a;

			vfloat samec_err = (ew_r * samec_dist0 * samec_dist0)
			                 + (ew_g * samec_dist1 * samec_dist1)
			                 + (ew_b * samec_dist2 * samec_dist2)
			                 + (ew_a * samec_dist3 * samec_dist3);

			haccumulate(samec_errorsumv, samec_err, mask);

			lane_ids += vint(ASTCENC_SIMD_WIDTH);
		}

		// Turn very small numbers and NaNs into a small number
		float uncor_linelen = hmax_s(uncor_hiparamv) - hmin_s(uncor_loparamv);
		line_lengths[partition] = astc::max(uncor_linelen, 1e-7f);
	}

	uncor_error = hadd_s(uncor_errorsumv);
	samec_error = hadd_s(samec_errorsumv);
}

/* See header for documentation. */
void compute_error_squared_rgb(
	const partition_info& pi,
	const image_block& blk,
	partition_lines3 plines[BLOCK_MAX_PARTITIONS],
	float& uncor_error,
	float& samec_error
) {
	unsigned int partition_count = pi.partition_count;
	promise(partition_count > 0);

	vfloatacc uncor_errorsumv = vfloatacc::zero();
	vfloatacc samec_errorsumv = vfloatacc::zero();

	for (unsigned int partition = 0; partition < partition_count; partition++)
	{
		partition_lines3& pl = plines[partition];
		const uint8_t *texel_indexes = pi.texels_of_partition[partition];
		unsigned int texel_count = pi.partition_texel_count[partition];
		promise(texel_count > 0);

		processed_line3 l_uncor = pl.uncor_pline;
		processed_line3 l_samec = pl.samec_pline;

		// Vectorize some useful scalar inputs
		vfloat l_uncor_bs0(l_uncor.bs.lane<0>());
		vfloat l_uncor_bs1(l_uncor.bs.lane<1>());
		vfloat l_uncor_bs2(l_uncor.bs.lane<2>());

		vfloat l_uncor_amod0(l_uncor.amod.lane<0>());
		vfloat l_uncor_amod1(l_uncor.amod.lane<1>());
		vfloat l_uncor_amod2(l_uncor.amod.lane<2>());

		vfloat l_samec_bs0(l_samec.bs.lane<0>());
		vfloat l_samec_bs1(l_samec.bs.lane<1>());
		vfloat l_samec_bs2(l_samec.bs.lane<2>());

		assert(all(l_samec.amod == vfloat4(0.0f)));

		vfloat uncor_loparamv(1e10f);
		vfloat uncor_hiparamv(-1e10f);

		vfloat ew_r(blk.channel_weight.lane<0>());
		vfloat ew_g(blk.channel_weight.lane<1>());
		vfloat ew_b(blk.channel_weight.lane<2>());

		// This implementation over-shoots, but this is safe as we initialize the weights array
		// to extend the last value. This means min/max are not impacted, but we need to mask
		// out the dummy values when we compute the line weighting.
		vint lane_ids = vint::lane_id();
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vmask mask = lane_ids < vint(texel_count);
			vint texel_idxs(texel_indexes + i);

			vfloat data_r = gatherf(blk.data_r, texel_idxs);
			vfloat data_g = gatherf(blk.data_g, texel_idxs);
			vfloat data_b = gatherf(blk.data_b, texel_idxs);

			vfloat uncor_param = (data_r * l_uncor_bs0)
			                   + (data_g * l_uncor_bs1)
			                   + (data_b * l_uncor_bs2);

			uncor_loparamv = min(uncor_param, uncor_loparamv);
			uncor_hiparamv = max(uncor_param, uncor_hiparamv);

			vfloat uncor_dist0 = (l_uncor_amod0 - data_r)
			                   + (uncor_param * l_uncor_bs0);
			vfloat uncor_dist1 = (l_uncor_amod1 - data_g)
			                   + (uncor_param * l_uncor_bs1);
			vfloat uncor_dist2 = (l_uncor_amod2 - data_b)
			                   + (uncor_param * l_uncor_bs2);

			vfloat uncor_err = (ew_r * uncor_dist0 * uncor_dist0)
			                 + (ew_g * uncor_dist1 * uncor_dist1)
			                 + (ew_b * uncor_dist2 * uncor_dist2);

			haccumulate(uncor_errorsumv, uncor_err, mask);

			// Process samechroma data
			vfloat samec_param = (data_r * l_samec_bs0)
			                   + (data_g * l_samec_bs1)
			                   + (data_b * l_samec_bs2);

			vfloat samec_dist0 = samec_param * l_samec_bs0 - data_r;
			vfloat samec_dist1 = samec_param * l_samec_bs1 - data_g;
			vfloat samec_dist2 = samec_param * l_samec_bs2 - data_b;

			vfloat samec_err = (ew_r * samec_dist0 * samec_dist0)
			                 + (ew_g * samec_dist1 * samec_dist1)
			                 + (ew_b * samec_dist2 * samec_dist2);

			haccumulate(samec_errorsumv, samec_err, mask);

			lane_ids += vint(ASTCENC_SIMD_WIDTH);
		}

		// Turn very small numbers and NaNs into a small number
		float uncor_linelen = hmax_s(uncor_hiparamv) - hmin_s(uncor_loparamv);
		pl.line_length = astc::max(uncor_linelen, 1e-7f);
	}

	uncor_error = hadd_s(uncor_errorsumv);
	samec_error = hadd_s(samec_errorsumv);
}

#endif
