// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2024 Arm Limited
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

#if !defined(ASTCENC_DECOMPRESS_ONLY)

/**
 * @brief Functions for computing color endpoints and texel weights.
 */

#include <cassert>

#include "astcenc_internal.h"
#include "astcenc_vecmathlib.h"

/**
 * @brief Compute the infilled weight for N texel indices in a decimated grid.
 *
 * @param di        The weight grid decimation to use.
 * @param weights   The decimated weight values to use.
 * @param index     The first texel index to interpolate.
 *
 * @return The interpolated weight for the given set of SIMD_WIDTH texels.
 */
static vfloat bilinear_infill_vla(
	const decimation_info& di,
	const float* weights,
	unsigned int index
) {
	// Load the bilinear filter texel weight indexes in the decimated grid
	const uint8_t* weight_idx0 = di.texel_weights_tr[0] + index;
	const uint8_t* weight_idx1 = di.texel_weights_tr[1] + index;
	const uint8_t* weight_idx2 = di.texel_weights_tr[2] + index;
	const uint8_t* weight_idx3 = di.texel_weights_tr[3] + index;

	// Load the bilinear filter weights from the decimated grid
	vfloat weight_val0 = gatherf_byte_inds<vfloat>(weights, weight_idx0);
	vfloat weight_val1 = gatherf_byte_inds<vfloat>(weights, weight_idx1);
	vfloat weight_val2 = gatherf_byte_inds<vfloat>(weights, weight_idx2);
	vfloat weight_val3 = gatherf_byte_inds<vfloat>(weights, weight_idx3);

	// Load the weight contribution factors for each decimated weight
	vfloat tex_weight_float0 = loada(di.texel_weight_contribs_float_tr[0] + index);
	vfloat tex_weight_float1 = loada(di.texel_weight_contribs_float_tr[1] + index);
	vfloat tex_weight_float2 = loada(di.texel_weight_contribs_float_tr[2] + index);
	vfloat tex_weight_float3 = loada(di.texel_weight_contribs_float_tr[3] + index);

	// Compute the bilinear interpolation to generate the per-texel weight
	return (weight_val0 * tex_weight_float0 + weight_val1 * tex_weight_float1) +
	       (weight_val2 * tex_weight_float2 + weight_val3 * tex_weight_float3);
}

/**
 * @brief Compute the infilled weight for N texel indices in a decimated grid.
 *
 * This is specialized version which computes only two weights per texel for
 * encodings that are only decimated in a single axis.
 *
 * @param di        The weight grid decimation to use.
 * @param weights   The decimated weight values to use.
 * @param index     The first texel index to interpolate.
 *
 * @return The interpolated weight for the given set of SIMD_WIDTH texels.
 */
static vfloat bilinear_infill_vla_2(
	const decimation_info& di,
	const float* weights,
	unsigned int index
) {
	// Load the bilinear filter texel weight indexes in the decimated grid
	const uint8_t* weight_idx0 = di.texel_weights_tr[0] + index;
	const uint8_t* weight_idx1 = di.texel_weights_tr[1] + index;

	// Load the bilinear filter weights from the decimated grid
	vfloat weight_val0 = gatherf_byte_inds<vfloat>(weights, weight_idx0);
	vfloat weight_val1 = gatherf_byte_inds<vfloat>(weights, weight_idx1);

	// Load the weight contribution factors for each decimated weight
	vfloat tex_weight_float0 = loada(di.texel_weight_contribs_float_tr[0] + index);
	vfloat tex_weight_float1 = loada(di.texel_weight_contribs_float_tr[1] + index);

	// Compute the bilinear interpolation to generate the per-texel weight
	return (weight_val0 * tex_weight_float0 + weight_val1 * tex_weight_float1);
}

/**
 * @brief Compute the ideal endpoints and weights for 1 color component.
 *
 * @param      blk         The image block color data to compress.
 * @param      pi          The partition info for the current trial.
 * @param[out] ei          The computed ideal endpoints and weights.
 * @param      component   The color component to compute.
 */
static void compute_ideal_colors_and_weights_1_comp(
	const image_block& blk,
	const partition_info& pi,
	endpoints_and_weights& ei,
	unsigned int component
) {
	unsigned int partition_count = pi.partition_count;
	ei.ep.partition_count = partition_count;
	promise(partition_count > 0);

	unsigned int texel_count = blk.texel_count;
	promise(texel_count > 0);

	float error_weight;
	const float* data_vr = nullptr;

	assert(component < BLOCK_MAX_COMPONENTS);
	switch (component)
	{
	case 0:
		error_weight = blk.channel_weight.lane<0>();
		data_vr = blk.data_r;
		break;
	case 1:
		error_weight = blk.channel_weight.lane<1>();
		data_vr = blk.data_g;
		break;
	case 2:
		error_weight = blk.channel_weight.lane<2>();
		data_vr = blk.data_b;
		break;
	default:
		assert(component == 3);
		error_weight = blk.channel_weight.lane<3>();
		data_vr = blk.data_a;
		break;
	}

	vmask4 sep_mask = vint4::lane_id() == vint4(component);
	bool is_constant_wes { true };
	float partition0_len_sq { 0.0f };

	for (unsigned int i = 0; i < partition_count; i++)
	{
		float lowvalue { 1e10f };
		float highvalue { -1e10f };

		unsigned int partition_texel_count = pi.partition_texel_count[i];
		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			float value = data_vr[tix];
			lowvalue = astc::min(value, lowvalue);
			highvalue = astc::max(value, highvalue);
		}

		if (highvalue <= lowvalue)
		{
			lowvalue = 0.0f;
			highvalue = 1e-7f;
		}

		float length = highvalue - lowvalue;
		float length_squared = length * length;
		float scale = 1.0f / length;

		if (i == 0)
		{
			partition0_len_sq = length_squared;
		}
		else
		{
			is_constant_wes = is_constant_wes && length_squared == partition0_len_sq;
		}

		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			float value = (data_vr[tix] - lowvalue) * scale;
			value = astc::clamp1f(value);

			ei.weights[tix] = value;
			ei.weight_error_scale[tix] = length_squared * error_weight;
			assert(!astc::isnan(ei.weight_error_scale[tix]));
		}

		ei.ep.endpt0[i] = select(blk.data_min, vfloat4(lowvalue), sep_mask);
		ei.ep.endpt1[i] = select(blk.data_max, vfloat4(highvalue), sep_mask);
	}

	// Zero initialize any SIMD over-fetch
	size_t texel_count_simd = round_up_to_simd_multiple_vla(texel_count);
	for (size_t i = texel_count; i < texel_count_simd; i++)
	{
		ei.weights[i] = 0.0f;
		ei.weight_error_scale[i] = 0.0f;
	}

	ei.is_constant_weight_error_scale = is_constant_wes;
}

/**
 * @brief Compute the ideal endpoints and weights for 2 color components.
 *
 * @param      blk          The image block color data to compress.
 * @param      pi           The partition info for the current trial.
 * @param[out] ei           The computed ideal endpoints and weights.
 * @param      component1   The first color component to compute.
 * @param      component2   The second color component to compute.
 */
static void compute_ideal_colors_and_weights_2_comp(
	const image_block& blk,
	const partition_info& pi,
	endpoints_and_weights& ei,
	int component1,
	int component2
) {
	unsigned int partition_count = pi.partition_count;
	ei.ep.partition_count = partition_count;
	promise(partition_count > 0);

	unsigned int texel_count = blk.texel_count;
	promise(texel_count > 0);

	partition_metrics pms[BLOCK_MAX_PARTITIONS];

	float error_weight;
	const float* data_vr = nullptr;
	const float* data_vg = nullptr;

	if (component1 == 0 && component2 == 1)
	{
		error_weight = hadd_s(blk.channel_weight.swz<0, 1>()) / 2.0f;

		data_vr = blk.data_r;
		data_vg = blk.data_g;
	}
	else if (component1 == 0 && component2 == 2)
	{
		error_weight = hadd_s(blk.channel_weight.swz<0, 2>()) / 2.0f;

		data_vr = blk.data_r;
		data_vg = blk.data_b;
	}
	else // (component1 == 1 && component2 == 2)
	{
		assert(component1 == 1 && component2 == 2);

		error_weight = hadd_s(blk.channel_weight.swz<1, 2>()) / 2.0f;

		data_vr = blk.data_g;
		data_vg = blk.data_b;
	}

	compute_avgs_and_dirs_2_comp(pi, blk, component1, component2, pms);

	bool is_constant_wes { true };
	float partition0_len_sq { 0.0f };

	vmask4 comp1_mask = vint4::lane_id() == vint4(component1);
	vmask4 comp2_mask = vint4::lane_id() == vint4(component2);

	for (unsigned int i = 0; i < partition_count; i++)
	{
		vfloat4 dir = pms[i].dir;
		if (hadd_s(dir) < 0.0f)
		{
			dir = vfloat4::zero() - dir;
		}

		line2 line { pms[i].avg, normalize_safe(dir, unit2()) };
		float lowparam { 1e10f };
		float highparam { -1e10f };

		unsigned int partition_texel_count = pi.partition_texel_count[i];
		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			vfloat4 point = vfloat2(data_vr[tix], data_vg[tix]);
			float param = dot_s(point - line.a, line.b);
			ei.weights[tix] = param;

			lowparam = astc::min(param, lowparam);
			highparam = astc::max(param, highparam);
		}

		// It is possible for a uniform-color partition to produce length=0;
		// this causes NaN issues so set to small value to avoid this problem
		if (highparam <= lowparam)
		{
			lowparam = 0.0f;
			highparam = 1e-7f;
		}

		float length = highparam - lowparam;
		float length_squared = length * length;
		float scale = 1.0f / length;

		if (i == 0)
		{
			partition0_len_sq = length_squared;
		}
		else
		{
			is_constant_wes = is_constant_wes && length_squared == partition0_len_sq;
		}

		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			float idx = (ei.weights[tix] - lowparam) * scale;
			idx = astc::clamp1f(idx);

			ei.weights[tix] = idx;
			ei.weight_error_scale[tix] = length_squared * error_weight;
			assert(!astc::isnan(ei.weight_error_scale[tix]));
		}

		vfloat4 lowvalue = line.a + line.b * lowparam;
		vfloat4 highvalue = line.a + line.b * highparam;

		vfloat4 ep0 = select(blk.data_min, vfloat4(lowvalue.lane<0>()), comp1_mask);
		vfloat4 ep1 = select(blk.data_max, vfloat4(highvalue.lane<0>()), comp1_mask);

		ei.ep.endpt0[i] = select(ep0, vfloat4(lowvalue.lane<1>()), comp2_mask);
		ei.ep.endpt1[i] = select(ep1, vfloat4(highvalue.lane<1>()), comp2_mask);
	}

	// Zero initialize any SIMD over-fetch
	size_t texel_count_simd = round_up_to_simd_multiple_vla(texel_count);
	for (size_t i = texel_count; i < texel_count_simd; i++)
	{
		ei.weights[i] = 0.0f;
		ei.weight_error_scale[i] = 0.0f;
	}

	ei.is_constant_weight_error_scale = is_constant_wes;
}

/**
 * @brief Compute the ideal endpoints and weights for 3 color components.
 *
 * @param      blk                 The image block color data to compress.
 * @param      pi                  The partition info for the current trial.
 * @param[out] ei                  The computed ideal endpoints and weights.
 * @param      omitted_component   The color component excluded from the calculation.
 */
static void compute_ideal_colors_and_weights_3_comp(
	const image_block& blk,
	const partition_info& pi,
	endpoints_and_weights& ei,
	unsigned int omitted_component
) {
	unsigned int partition_count = pi.partition_count;
	ei.ep.partition_count = partition_count;
	promise(partition_count > 0);

	unsigned int texel_count = blk.texel_count;
	promise(texel_count > 0);

	partition_metrics pms[BLOCK_MAX_PARTITIONS];

	float error_weight;
	const float* data_vr = nullptr;
	const float* data_vg = nullptr;
	const float* data_vb = nullptr;
	if (omitted_component == 0)
	{
		error_weight = hadd_s(blk.channel_weight.swz<0, 1, 2>());
		data_vr = blk.data_g;
		data_vg = blk.data_b;
		data_vb = blk.data_a;
	}
	else if (omitted_component == 1)
	{
		error_weight = hadd_s(blk.channel_weight.swz<0, 2, 3>());
		data_vr = blk.data_r;
		data_vg = blk.data_b;
		data_vb = blk.data_a;
	}
	else if (omitted_component == 2)
	{
		error_weight = hadd_s(blk.channel_weight.swz<0, 1, 3>());
		data_vr = blk.data_r;
		data_vg = blk.data_g;
		data_vb = blk.data_a;
	}
	else
	{
		assert(omitted_component == 3);

		error_weight = hadd_s(blk.channel_weight.swz<0, 1, 2>());
		data_vr = blk.data_r;
		data_vg = blk.data_g;
		data_vb = blk.data_b;
	}

	error_weight = error_weight * (1.0f / 3.0f);

	if (omitted_component == 3)
	{
		compute_avgs_and_dirs_3_comp_rgb(pi, blk, pms);
	}
	else
	{
		compute_avgs_and_dirs_3_comp(pi, blk, omitted_component, pms);
	}

	bool is_constant_wes { true };
	float partition0_len_sq { 0.0f };

	for (unsigned int i = 0; i < partition_count; i++)
	{
		vfloat4 dir = pms[i].dir;
		if (hadd_rgb_s(dir) < 0.0f)
		{
			dir = vfloat4::zero() - dir;
		}

		line3 line { pms[i].avg, normalize_safe(dir, unit3()) };
		float lowparam { 1e10f };
		float highparam { -1e10f };

		unsigned int partition_texel_count = pi.partition_texel_count[i];
		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			vfloat4 point = vfloat3(data_vr[tix], data_vg[tix], data_vb[tix]);
			float param = dot3_s(point - line.a, line.b);
			ei.weights[tix] = param;

			lowparam = astc::min(param, lowparam);
			highparam = astc::max(param, highparam);
		}

		// It is possible for a uniform-color partition to produce length=0;
		// this causes NaN issues so set to small value to avoid this problem
		if (highparam <= lowparam)
		{
			lowparam = 0.0f;
			highparam = 1e-7f;
		}

		float length = highparam - lowparam;
		float length_squared = length * length;
		float scale = 1.0f / length;

		if (i == 0)
		{
			partition0_len_sq = length_squared;
		}
		else
		{
			is_constant_wes = is_constant_wes && length_squared == partition0_len_sq;
		}

		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			float idx = (ei.weights[tix] - lowparam) * scale;
			idx = astc::clamp1f(idx);

			ei.weights[tix] = idx;
			ei.weight_error_scale[tix] = length_squared * error_weight;
			assert(!astc::isnan(ei.weight_error_scale[tix]));
		}

		vfloat4 ep0 = line.a + line.b * lowparam;
		vfloat4 ep1 = line.a + line.b * highparam;

		vfloat4 bmin = blk.data_min;
		vfloat4 bmax = blk.data_max;

		assert(omitted_component < BLOCK_MAX_COMPONENTS);
		switch (omitted_component)
		{
			case 0:
				ei.ep.endpt0[i] = vfloat4(bmin.lane<0>(), ep0.lane<0>(), ep0.lane<1>(), ep0.lane<2>());
				ei.ep.endpt1[i] = vfloat4(bmax.lane<0>(), ep1.lane<0>(), ep1.lane<1>(), ep1.lane<2>());
				break;
			case 1:
				ei.ep.endpt0[i] = vfloat4(ep0.lane<0>(), bmin.lane<1>(), ep0.lane<1>(), ep0.lane<2>());
				ei.ep.endpt1[i] = vfloat4(ep1.lane<0>(), bmax.lane<1>(), ep1.lane<1>(), ep1.lane<2>());
				break;
			case 2:
				ei.ep.endpt0[i] = vfloat4(ep0.lane<0>(), ep0.lane<1>(), bmin.lane<2>(), ep0.lane<2>());
				ei.ep.endpt1[i] = vfloat4(ep1.lane<0>(), ep1.lane<1>(), bmax.lane<2>(), ep1.lane<2>());
				break;
			default:
				ei.ep.endpt0[i] = vfloat4(ep0.lane<0>(), ep0.lane<1>(), ep0.lane<2>(), bmin.lane<3>());
				ei.ep.endpt1[i] = vfloat4(ep1.lane<0>(), ep1.lane<1>(), ep1.lane<2>(), bmax.lane<3>());
				break;
		}
	}

	// Zero initialize any SIMD over-fetch
	size_t texel_count_simd = round_up_to_simd_multiple_vla(texel_count);
	for (size_t i = texel_count; i < texel_count_simd; i++)
	{
		ei.weights[i] = 0.0f;
		ei.weight_error_scale[i] = 0.0f;
	}

	ei.is_constant_weight_error_scale = is_constant_wes;
}

/**
 * @brief Compute the ideal endpoints and weights for 4 color components.
 *
 * @param      blk   The image block color data to compress.
 * @param      pi    The partition info for the current trial.
 * @param[out] ei    The computed ideal endpoints and weights.
 */
static void compute_ideal_colors_and_weights_4_comp(
	const image_block& blk,
	const partition_info& pi,
	endpoints_and_weights& ei
) {
	const float error_weight = hadd_s(blk.channel_weight) / 4.0f;

	unsigned int partition_count = pi.partition_count;

	unsigned int texel_count = blk.texel_count;
	promise(texel_count > 0);
	promise(partition_count > 0);

	partition_metrics pms[BLOCK_MAX_PARTITIONS];

	compute_avgs_and_dirs_4_comp(pi, blk, pms);

	bool is_constant_wes { true };
	float partition0_len_sq { 0.0f };

	for (unsigned int i = 0; i < partition_count; i++)
	{
		vfloat4 dir = pms[i].dir;
		if (hadd_rgb_s(dir) < 0.0f)
		{
			dir = vfloat4::zero() - dir;
		}

		line4 line { pms[i].avg, normalize_safe(dir, unit4()) };
		float lowparam { 1e10f };
		float highparam { -1e10f };

		unsigned int partition_texel_count = pi.partition_texel_count[i];
		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			vfloat4 point = blk.texel(tix);
			float param = dot_s(point - line.a, line.b);
			ei.weights[tix] = param;

			lowparam = astc::min(param, lowparam);
			highparam = astc::max(param, highparam);
		}

		// It is possible for a uniform-color partition to produce length=0;
		// this causes NaN issues so set to small value to avoid this problem
		if (highparam <= lowparam)
		{
			lowparam = 0.0f;
			highparam = 1e-7f;
		}

		float length = highparam - lowparam;
		float length_squared = length * length;
		float scale = 1.0f / length;

		if (i == 0)
		{
			partition0_len_sq = length_squared;
		}
		else
		{
			is_constant_wes = is_constant_wes && length_squared == partition0_len_sq;
		}

		ei.ep.endpt0[i] = line.a + line.b * lowparam;
		ei.ep.endpt1[i] = line.a + line.b * highparam;

		for (unsigned int j = 0; j < partition_texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			float idx = (ei.weights[tix] - lowparam) * scale;
			idx = astc::clamp1f(idx);

			ei.weights[tix] = idx;
			ei.weight_error_scale[tix] = length_squared * error_weight;
			assert(!astc::isnan(ei.weight_error_scale[tix]));
		}
	}

	// Zero initialize any SIMD over-fetch
	size_t texel_count_simd = round_up_to_simd_multiple_vla(texel_count);
	for (size_t i = texel_count; i < texel_count_simd; i++)
	{
		ei.weights[i] = 0.0f;
		ei.weight_error_scale[i] = 0.0f;
	}

	ei.is_constant_weight_error_scale = is_constant_wes;
}

/* See header for documentation. */
void compute_ideal_colors_and_weights_1plane(
	const image_block& blk,
	const partition_info& pi,
	endpoints_and_weights& ei
) {
	bool uses_alpha = !blk.is_constant_channel(3);

	if (uses_alpha)
	{
		compute_ideal_colors_and_weights_4_comp(blk, pi, ei);
	}
	else
	{
		compute_ideal_colors_and_weights_3_comp(blk, pi, ei, 3);
	}
}

/* See header for documentation. */
void compute_ideal_colors_and_weights_2planes(
	const block_size_descriptor& bsd,
	const image_block& blk,
	unsigned int plane2_component,
	endpoints_and_weights& ei1,
	endpoints_and_weights& ei2
) {
	const auto& pi = bsd.get_partition_info(1, 0);
	bool uses_alpha = !blk.is_constant_channel(3);

	assert(plane2_component < BLOCK_MAX_COMPONENTS);
	switch (plane2_component)
	{
	case 0: // Separate weights for red
		if (uses_alpha)
		{
			compute_ideal_colors_and_weights_3_comp(blk, pi, ei1, 0);
		}
		else
		{
			compute_ideal_colors_and_weights_2_comp(blk, pi, ei1, 1, 2);
		}
		compute_ideal_colors_and_weights_1_comp(blk, pi, ei2, 0);
		break;

	case 1: // Separate weights for green
		if (uses_alpha)
		{
			compute_ideal_colors_and_weights_3_comp(blk, pi, ei1, 1);
		}
		else
		{
			compute_ideal_colors_and_weights_2_comp(blk, pi, ei1, 0, 2);
		}
		compute_ideal_colors_and_weights_1_comp(blk, pi, ei2, 1);
		break;

	case 2: // Separate weights for blue
		if (uses_alpha)
		{
			compute_ideal_colors_and_weights_3_comp(blk, pi, ei1, 2);
		}
		else
		{
			compute_ideal_colors_and_weights_2_comp(blk, pi, ei1, 0, 1);
		}
		compute_ideal_colors_and_weights_1_comp(blk, pi, ei2, 2);
		break;

	default: // Separate weights for alpha
		assert(uses_alpha);
		compute_ideal_colors_and_weights_3_comp(blk, pi, ei1, 3);
		compute_ideal_colors_and_weights_1_comp(blk, pi, ei2, 3);
		break;
	}
}

/* See header for documentation. */
float compute_error_of_weight_set_1plane(
	const endpoints_and_weights& eai,
	const decimation_info& di,
	const float* dec_weight_quant_uvalue
) {
	vfloatacc error_summav = vfloatacc::zero();
	unsigned int texel_count = di.texel_count;
	promise(texel_count > 0);

	// Process SIMD-width chunks, safe to over-fetch - the extra space is zero initialized
	if (di.max_texel_weight_count > 2)
	{
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			// Compute the bilinear interpolation of the decimated weight grid
			vfloat current_values = bilinear_infill_vla(di, dec_weight_quant_uvalue, i);

			// Compute the error between the computed value and the ideal weight
			vfloat actual_values = loada(eai.weights + i);
			vfloat diff = current_values - actual_values;
			vfloat significance = loada(eai.weight_error_scale + i);
			vfloat error = diff * diff * significance;

			haccumulate(error_summav, error);
		}
	}
	else if (di.max_texel_weight_count > 1)
	{
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			// Compute the bilinear interpolation of the decimated weight grid
			vfloat current_values = bilinear_infill_vla_2(di, dec_weight_quant_uvalue, i);

			// Compute the error between the computed value and the ideal weight
			vfloat actual_values = loada(eai.weights + i);
			vfloat diff = current_values - actual_values;
			vfloat significance = loada(eai.weight_error_scale + i);
			vfloat error = diff * diff * significance;

			haccumulate(error_summav, error);
		}
	}
	else
	{
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			// Load the weight set directly, without interpolation
			vfloat current_values = loada(dec_weight_quant_uvalue + i);

			// Compute the error between the computed value and the ideal weight
			vfloat actual_values = loada(eai.weights + i);
			vfloat diff = current_values - actual_values;
			vfloat significance = loada(eai.weight_error_scale + i);
			vfloat error = diff * diff * significance;

			haccumulate(error_summav, error);
		}
	}

	// Resolve the final scalar accumulator sum
	return hadd_s(error_summav);
}

/* See header for documentation. */
float compute_error_of_weight_set_2planes(
	const endpoints_and_weights& eai1,
	const endpoints_and_weights& eai2,
	const decimation_info& di,
	const float* dec_weight_quant_uvalue_plane1,
	const float* dec_weight_quant_uvalue_plane2
) {
	vfloatacc error_summav = vfloatacc::zero();
	unsigned int texel_count = di.texel_count;
	promise(texel_count > 0);

	// Process SIMD-width chunks, safe to over-fetch - the extra space is zero initialized
	if (di.max_texel_weight_count > 2)
	{
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			// Plane 1
			// Compute the bilinear interpolation of the decimated weight grid
			vfloat current_values1 = bilinear_infill_vla(di, dec_weight_quant_uvalue_plane1, i);

			// Compute the error between the computed value and the ideal weight
			vfloat actual_values1 = loada(eai1.weights + i);
			vfloat diff = current_values1 - actual_values1;
			vfloat error1 = diff * diff * loada(eai1.weight_error_scale + i);

			// Plane 2
			// Compute the bilinear interpolation of the decimated weight grid
			vfloat current_values2 = bilinear_infill_vla(di, dec_weight_quant_uvalue_plane2, i);

			// Compute the error between the computed value and the ideal weight
			vfloat actual_values2 = loada(eai2.weights + i);
			diff = current_values2 - actual_values2;
			vfloat error2 = diff * diff * loada(eai2.weight_error_scale + i);

			haccumulate(error_summav, error1 + error2);
		}
	}
	else if (di.max_texel_weight_count > 1)
	{
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			// Plane 1
			// Compute the bilinear interpolation of the decimated weight grid
			vfloat current_values1 = bilinear_infill_vla_2(di, dec_weight_quant_uvalue_plane1, i);

			// Compute the error between the computed value and the ideal weight
			vfloat actual_values1 = loada(eai1.weights + i);
			vfloat diff = current_values1 - actual_values1;
			vfloat error1 = diff * diff * loada(eai1.weight_error_scale + i);

			// Plane 2
			// Compute the bilinear interpolation of the decimated weight grid
			vfloat current_values2 = bilinear_infill_vla_2(di, dec_weight_quant_uvalue_plane2, i);

			// Compute the error between the computed value and the ideal weight
			vfloat actual_values2 = loada(eai2.weights + i);
			diff = current_values2 - actual_values2;
			vfloat error2 = diff * diff * loada(eai2.weight_error_scale + i);

			haccumulate(error_summav, error1 + error2);
		}
	}
	else
	{
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			// Plane 1
			// Load the weight set directly, without interpolation
			vfloat current_values1 = loada(dec_weight_quant_uvalue_plane1 + i);

			// Compute the error between the computed value and the ideal weight
			vfloat actual_values1 = loada(eai1.weights + i);
			vfloat diff = current_values1 - actual_values1;
			vfloat error1 = diff * diff * loada(eai1.weight_error_scale + i);

			// Plane 2
			// Load the weight set directly, without interpolation
			vfloat current_values2 = loada(dec_weight_quant_uvalue_plane2 + i);

			// Compute the error between the computed value and the ideal weight
			vfloat actual_values2 = loada(eai2.weights + i);
			diff = current_values2 - actual_values2;
			vfloat error2 = diff * diff * loada(eai2.weight_error_scale + i);

			haccumulate(error_summav, error1 + error2);
		}
	}

	// Resolve the final scalar accumulator sum
	return hadd_s(error_summav);
}

/* See header for documentation. */
void compute_ideal_weights_for_decimation(
	const endpoints_and_weights& ei,
	const decimation_info& di,
	float* dec_weight_ideal_value
) {
	unsigned int texel_count = di.texel_count;
	unsigned int weight_count = di.weight_count;
	bool is_direct = texel_count == weight_count;
	promise(texel_count > 0);
	promise(weight_count > 0);

	// If we have a 1:1 mapping just shortcut the computation. Transfer enough to also copy the
	// zero-initialized SIMD over-fetch region
	if (is_direct)
	{
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vfloat weight(ei.weights + i);
			storea(weight, dec_weight_ideal_value + i);
		}

		return;
	}

	// Otherwise compute an estimate and perform single refinement iteration

	// Compute an initial average for each decimated weight
	bool constant_wes = ei.is_constant_weight_error_scale;
	vfloat weight_error_scale(ei.weight_error_scale[0]);

	// This overshoots - this is OK as we initialize the array tails in the
	// decimation table structures to safe values ...
	for (unsigned int i = 0; i < weight_count; i += ASTCENC_SIMD_WIDTH)
	{
		// Start with a small value to avoid div-by-zero later
		vfloat weight_weight(1e-10f);
		vfloat initial_weight = vfloat::zero();

		// Accumulate error weighting of all the texels using this weight
		vint weight_texel_count(di.weight_texel_count + i);
		unsigned int max_texel_count = hmax_s(weight_texel_count);
		promise(max_texel_count > 0);

		for (unsigned int j = 0; j < max_texel_count; j++)
		{
			const uint8_t* texel = di.weight_texels_tr[j] + i;
			vfloat weight = loada(di.weights_texel_contribs_tr[j] + i);

			if (!constant_wes)
			{
				weight_error_scale = gatherf_byte_inds<vfloat>(ei.weight_error_scale, texel);
			}

			vfloat contrib_weight = weight * weight_error_scale;

			weight_weight += contrib_weight;
			initial_weight += gatherf_byte_inds<vfloat>(ei.weights, texel) * contrib_weight;
		}

		storea(initial_weight / weight_weight, dec_weight_ideal_value + i);
	}

	// Populate the interpolated weight grid based on the initial average
	// Process SIMD-width texel coordinates at at time while we can. Safe to
	// over-process full SIMD vectors - the tail is zeroed.
	ASTCENC_ALIGNAS float infilled_weights[BLOCK_MAX_TEXELS];
	if (di.max_texel_weight_count <= 2)
	{
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vfloat weight = bilinear_infill_vla_2(di, dec_weight_ideal_value, i);
			storea(weight, infilled_weights + i);
		}
	}
	else
	{
		for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vfloat weight = bilinear_infill_vla(di, dec_weight_ideal_value, i);
			storea(weight, infilled_weights + i);
		}
	}

	// Perform a single iteration of refinement
	// Empirically determined step size; larger values don't help but smaller drops image quality
	constexpr float stepsize = 0.25f;
	constexpr float chd_scale = -WEIGHTS_TEXEL_SUM;

	for (unsigned int i = 0; i < weight_count; i += ASTCENC_SIMD_WIDTH)
	{
		vfloat weight_val = loada(dec_weight_ideal_value + i);

		// Accumulate error weighting of all the texels using this weight
		// Start with a small value to avoid div-by-zero later
		vfloat error_change0(1e-10f);
		vfloat error_change1(0.0f);

		// Accumulate error weighting of all the texels using this weight
		vint weight_texel_count(di.weight_texel_count + i);
		unsigned int max_texel_count = hmax_s(weight_texel_count);
		promise(max_texel_count > 0);

		for (unsigned int j = 0; j < max_texel_count; j++)
		{
			const uint8_t* texel = di.weight_texels_tr[j] + i;
			vfloat contrib_weight = loada(di.weights_texel_contribs_tr[j] + i);

			if (!constant_wes)
			{
				weight_error_scale = gatherf_byte_inds<vfloat>(ei.weight_error_scale, texel);
			}

			vfloat scale = weight_error_scale * contrib_weight;
			vfloat old_weight = gatherf_byte_inds<vfloat>(infilled_weights, texel);
			vfloat ideal_weight = gatherf_byte_inds<vfloat>(ei.weights, texel);

			error_change0 += contrib_weight * scale;
			error_change1 += (old_weight - ideal_weight) * scale;
		}

		vfloat step = (error_change1 * chd_scale) / error_change0;
		step = clamp(-stepsize, stepsize, step);

		// Update the weight; note this can store negative values
		storea(weight_val + step, dec_weight_ideal_value + i);
	}
}

/* See header for documentation. */
void compute_quantized_weights_for_decimation(
	const decimation_info& di,
	float low_bound,
	float high_bound,
	const float* dec_weight_ideal_value,
	float* weight_set_out,
	uint8_t* quantized_weight_set,
	quant_method quant_level
) {
	int weight_count = di.weight_count;
	promise(weight_count > 0);
	const quant_and_transfer_table& qat = quant_and_xfer_tables[quant_level];

	// The available quant levels, stored with a minus 1 bias
	static const float quant_levels_m1[12] {
		1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, 9.0f, 11.0f, 15.0f, 19.0f, 23.0f, 31.0f
	};

	vint steps_m1(get_quant_level(quant_level) - 1);
	float quant_level_m1 = quant_levels_m1[quant_level];

	// Quantize the weight set using both the specified low/high bounds and standard 0..1 bounds

	// TODO: Oddity to investigate; triggered by test in issue #265.
	if (high_bound <= low_bound)
	{
		low_bound = 0.0f;
		high_bound = 1.0f;
	}

	float rscale = high_bound - low_bound;
	float scale = 1.0f / rscale;

	float scaled_low_bound = low_bound * scale;
	rscale *= 1.0f / 64.0f;

	vfloat scalev(scale);
	vfloat scaled_low_boundv(scaled_low_bound);
	vfloat quant_level_m1v(quant_level_m1);
	vfloat rscalev(rscale);
	vfloat low_boundv(low_bound);

	// This runs to the rounded-up SIMD size, which is safe as the loop tail is filled with known
	// safe data in compute_ideal_weights_for_decimation and arrays are always 64 elements
	if (get_quant_level(quant_level) <= 16)
	{
		vtable_16x8 table;
		vtable_prepare(table, qat.quant_to_unquant);

		for (int i = 0; i < weight_count; i += ASTCENC_SIMD_WIDTH)
		{
			vfloat ix = loada(dec_weight_ideal_value + i) * scalev - scaled_low_boundv;
			ix = clampzo(ix);

			// Look up the two closest indexes and return the one that was closest
			vfloat ix1 = ix * quant_level_m1v;

			vint weightl = float_to_int(ix1);
			vint weighth = min(weightl + vint(1), steps_m1);

			vint ixli = vtable_lookup_32bit(table, weightl);
			vint ixhi = vtable_lookup_32bit(table, weighth);

			vfloat ixl = int_to_float(ixli);
			vfloat ixh = int_to_float(ixhi);

			vmask mask = (ixl + ixh) < (vfloat(128.0f) * ix);
			vint weight = select(ixli, ixhi, mask);
			ixl = select(ixl, ixh, mask);

			// Invert the weight-scaling that was done initially
			storea(ixl * rscalev + low_boundv, weight_set_out + i);
			pack_and_store_low_bytes(weight, quantized_weight_set + i);
		}
	}
	else
	{
		vtable_32x8 table;
		vtable_prepare(table, qat.quant_to_unquant);

		for (int i = 0; i < weight_count; i += ASTCENC_SIMD_WIDTH)
		{
			vfloat ix = loada(dec_weight_ideal_value + i) * scalev - scaled_low_boundv;
			ix = clampzo(ix);

			// Look up the two closest indexes and return the one that was closest
			vfloat ix1 = ix * quant_level_m1v;

			vint weightl = float_to_int(ix1);
			vint weighth = min(weightl + vint(1), steps_m1);

			vint ixli = vtable_lookup_32bit(table, weightl);
			vint ixhi = vtable_lookup_32bit(table, weighth);

			vfloat ixl = int_to_float(ixli);
			vfloat ixh = int_to_float(ixhi);

			vmask mask = (ixl + ixh) < (vfloat(128.0f) * ix);
			vint weight = select(ixli, ixhi, mask);
			ixl = select(ixl, ixh, mask);

			// Invert the weight-scaling that was done initially
			storea(ixl * rscalev + low_boundv, weight_set_out + i);
			pack_and_store_low_bytes(weight, quantized_weight_set + i);
		}
	}
}

/**
 * @brief Compute the RGB + offset for a HDR endpoint mode #7.
 *
 * Since the matrix needed has a regular structure we can simplify the inverse calculation. This
 * gives us ~24 multiplications vs. 96 for a generic inverse.
 *
 *  mat[0] = vfloat4(rgba_ws.x,      0.0f,      0.0f, wght_ws.x);
 *  mat[1] = vfloat4(     0.0f, rgba_ws.y,      0.0f, wght_ws.y);
 *  mat[2] = vfloat4(     0.0f,      0.0f, rgba_ws.z, wght_ws.z);
 *  mat[3] = vfloat4(wght_ws.x, wght_ws.y, wght_ws.z,      psum);
 *  mat = invert(mat);
 *
 * @param rgba_weight_sum     Sum of partition component error weights.
 * @param weight_weight_sum   Sum of partition component error weights * texel weight.
 * @param rgbq_sum            Sum of partition component error weights * texel weight * color data.
 * @param psum                Sum of RGB color weights * texel weight^2.
 */
static inline vfloat4 compute_rgbo_vector(
	vfloat4 rgba_weight_sum,
	vfloat4 weight_weight_sum,
	vfloat4 rgbq_sum,
	float psum
) {
	float X = rgba_weight_sum.lane<0>();
	float Y = rgba_weight_sum.lane<1>();
	float Z = rgba_weight_sum.lane<2>();
	float P = weight_weight_sum.lane<0>();
	float Q = weight_weight_sum.lane<1>();
	float R = weight_weight_sum.lane<2>();
	float S = psum;

	float PP = P * P;
	float QQ = Q * Q;
	float RR = R * R;

	float SZmRR = S * Z - RR;
	float DT = SZmRR * Y - Z * QQ;
	float YP = Y * P;
	float QX = Q * X;
	float YX = Y * X;
	float mZYP = -Z * YP;
	float mZQX = -Z * QX;
	float mRYX = -R * YX;
	float ZQP = Z * Q * P;
	float RYP = R * YP;
	float RQX = R * QX;

	// Compute the reciprocal of matrix determinant
	float rdet = 1.0f / (DT * X + mZYP * P);

	// Actually compute the adjugate, and then apply 1/det separately
	vfloat4 mat0(DT, ZQP, RYP, mZYP);
	vfloat4 mat1(ZQP, SZmRR * X - Z * PP, RQX, mZQX);
	vfloat4 mat2(RYP, RQX, (S * Y - QQ) * X - Y * PP, mRYX);
	vfloat4 mat3(mZYP, mZQX, mRYX, Z * YX);
	vfloat4 vect = rgbq_sum * rdet;

	return vfloat4(dot_s(mat0, vect),
	               dot_s(mat1, vect),
	               dot_s(mat2, vect),
	               dot_s(mat3, vect));
}

/* See header for documentation. */
void recompute_ideal_colors_1plane(
	const image_block& blk,
	const partition_info& pi,
	const decimation_info& di,
	const uint8_t* dec_weights_uquant,
	endpoints& ep,
	vfloat4 rgbs_vectors[BLOCK_MAX_PARTITIONS],
	vfloat4 rgbo_vectors[BLOCK_MAX_PARTITIONS]
) {
	unsigned int weight_count = di.weight_count;
	unsigned int total_texel_count = blk.texel_count;
	unsigned int partition_count = pi.partition_count;

	promise(weight_count > 0);
	promise(total_texel_count > 0);
	promise(partition_count > 0);

	ASTCENC_ALIGNAS float dec_weight[BLOCK_MAX_WEIGHTS];
	for (unsigned int i = 0; i < weight_count; i += ASTCENC_SIMD_WIDTH)
	{
		vint unquant_value(dec_weights_uquant + i);
		vfloat unquant_valuef = int_to_float(unquant_value) * vfloat(1.0f / 64.0f);
		storea(unquant_valuef, dec_weight + i);
	}

	ASTCENC_ALIGNAS float undec_weight[BLOCK_MAX_TEXELS];
	float* undec_weight_ref;
	if (di.max_texel_weight_count == 1)
	{
		undec_weight_ref = dec_weight;
	}
	else if (di.max_texel_weight_count <= 2)
	{
		for (unsigned int i = 0; i < total_texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vfloat weight = bilinear_infill_vla_2(di, dec_weight, i);
			storea(weight, undec_weight + i);
		}

		undec_weight_ref = undec_weight;
	}
	else
	{
		for (unsigned int i = 0; i < total_texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vfloat weight = bilinear_infill_vla(di, dec_weight, i);
			storea(weight, undec_weight + i);
		}

		undec_weight_ref = undec_weight;
	}

	vfloat4 rgba_sum(blk.data_mean * static_cast<float>(blk.texel_count));

	for (unsigned int i = 0; i < partition_count; i++)
	{
		unsigned int texel_count = pi.partition_texel_count[i];
		const uint8_t *texel_indexes = pi.texels_of_partition[i];

		// Only compute a partition mean if more than one partition
		if (partition_count > 1)
		{
			rgba_sum = vfloat4::zero();
			promise(texel_count > 0);
			for (unsigned int j = 0; j < texel_count; j++)
			{
				unsigned int tix = texel_indexes[j];
				rgba_sum += blk.texel(tix);
			}
		}

		rgba_sum = rgba_sum * blk.channel_weight;
		vfloat4 rgba_weight_sum = max(blk.channel_weight * static_cast<float>(texel_count), 1e-17f);
		vfloat4 scale_dir = normalize((rgba_sum / rgba_weight_sum).swz<0, 1, 2>());

		float scale_max = 0.0f;
		float scale_min = 1e10f;

		float wmin1 = 1.0f;
		float wmax1 = 0.0f;

		float left_sum_s = 0.0f;
		float middle_sum_s = 0.0f;
		float right_sum_s = 0.0f;

		vfloat4 color_vec_x = vfloat4::zero();
		vfloat4 color_vec_y = vfloat4::zero();

		vfloat4 scale_vec = vfloat4::zero();

		float weight_weight_sum_s = 1e-17f;

		vfloat4 color_weight = blk.channel_weight;
		float ls_weight = hadd_rgb_s(color_weight);

		for (unsigned int j = 0; j < texel_count; j++)
		{
			unsigned int tix = texel_indexes[j];
			vfloat4 rgba = blk.texel(tix);

			float idx0 = undec_weight_ref[tix];

			float om_idx0 = 1.0f - idx0;
			wmin1 = astc::min(idx0, wmin1);
			wmax1 = astc::max(idx0, wmax1);

			float scale = dot3_s(scale_dir, rgba);
			scale_min = astc::min(scale, scale_min);
			scale_max = astc::max(scale, scale_max);

			left_sum_s   += om_idx0 * om_idx0;
			middle_sum_s += om_idx0 * idx0;
			right_sum_s  += idx0 * idx0;
			weight_weight_sum_s += idx0;

			vfloat4 color_idx(idx0);
			vfloat4 cwprod = rgba;
			vfloat4 cwiprod = cwprod * color_idx;

			color_vec_y += cwiprod;
			color_vec_x += cwprod - cwiprod;

			scale_vec += vfloat2(om_idx0, idx0) * (scale * ls_weight);
		}

		vfloat4 left_sum   = vfloat4(left_sum_s) * color_weight;
		vfloat4 middle_sum = vfloat4(middle_sum_s) * color_weight;
		vfloat4 right_sum  = vfloat4(right_sum_s) * color_weight;
		vfloat4 lmrs_sum   = vfloat3(left_sum_s, middle_sum_s, right_sum_s) * ls_weight;

		color_vec_x = color_vec_x * color_weight;
		color_vec_y = color_vec_y * color_weight;

		// Initialize the luminance and scale vectors with a reasonable default
		float scalediv = scale_min / astc::max(scale_max, 1e-10f);
		scalediv = astc::clamp1f(scalediv);

		vfloat4 sds = scale_dir * scale_max;

		rgbs_vectors[i] = vfloat4(sds.lane<0>(), sds.lane<1>(), sds.lane<2>(), scalediv);

		if (wmin1 >= wmax1 * 0.999f)
		{
			// If all weights in the partition were equal, then just take average of all colors in
			// the partition and use that as both endpoint colors
			vfloat4 avg = (color_vec_x + color_vec_y) / rgba_weight_sum;

			vmask4 notnan_mask = avg == avg;
			ep.endpt0[i] = select(ep.endpt0[i], avg, notnan_mask);
			ep.endpt1[i] = select(ep.endpt1[i], avg, notnan_mask);

			rgbs_vectors[i] = vfloat4(sds.lane<0>(), sds.lane<1>(), sds.lane<2>(), 1.0f);
		}
		else
		{
			// Otherwise, complete the analytic calculation of ideal-endpoint-values for the given
			// set of texel weights and pixel colors
			vfloat4 color_det1 = (left_sum * right_sum) - (middle_sum * middle_sum);
			vfloat4 color_rdet1 = 1.0f / color_det1;

			float ls_det1  = (lmrs_sum.lane<0>() * lmrs_sum.lane<2>()) - (lmrs_sum.lane<1>() * lmrs_sum.lane<1>());
			float ls_rdet1 = 1.0f / ls_det1;

			vfloat4 color_mss1 = (left_sum * left_sum)
			                   + (2.0f * middle_sum * middle_sum)
			                   + (right_sum * right_sum);

			float ls_mss1 = (lmrs_sum.lane<0>() * lmrs_sum.lane<0>())
			              + (2.0f * lmrs_sum.lane<1>() * lmrs_sum.lane<1>())
			              + (lmrs_sum.lane<2>() * lmrs_sum.lane<2>());

			vfloat4 ep0 = (right_sum * color_vec_x - middle_sum * color_vec_y) * color_rdet1;
			vfloat4 ep1 = (left_sum * color_vec_y - middle_sum * color_vec_x) * color_rdet1;

			vmask4 det_mask = abs(color_det1) > (color_mss1 * 1e-4f);
			vmask4 notnan_mask = (ep0 == ep0) & (ep1 == ep1);
			vmask4 full_mask = det_mask & notnan_mask;

			ep.endpt0[i] = select(ep.endpt0[i], ep0, full_mask);
			ep.endpt1[i] = select(ep.endpt1[i], ep1, full_mask);

			float scale_ep0 = (lmrs_sum.lane<2>() * scale_vec.lane<0>() - lmrs_sum.lane<1>() * scale_vec.lane<1>()) * ls_rdet1;
			float scale_ep1 = (lmrs_sum.lane<0>() * scale_vec.lane<1>() - lmrs_sum.lane<1>() * scale_vec.lane<0>()) * ls_rdet1;

			if (fabsf(ls_det1) > (ls_mss1 * 1e-4f) && scale_ep0 == scale_ep0 && scale_ep1 == scale_ep1 && scale_ep0 < scale_ep1)
			{
				float scalediv2 = scale_ep0 / scale_ep1;
				vfloat4 sdsm = scale_dir * scale_ep1;
				rgbs_vectors[i] = vfloat4(sdsm.lane<0>(), sdsm.lane<1>(), sdsm.lane<2>(), scalediv2);
			}
		}

		// Calculations specific to mode #7, the HDR RGB-scale mode - skip if known LDR
		if (blk.rgb_lns[0] || blk.alpha_lns[0])
		{
			vfloat4 weight_weight_sum = vfloat4(weight_weight_sum_s) * color_weight;
			float psum = right_sum_s * hadd_rgb_s(color_weight);

			vfloat4 rgbq_sum = color_vec_x + color_vec_y;
			rgbq_sum.set_lane<3>(hadd_rgb_s(color_vec_y));

			vfloat4 rgbovec = compute_rgbo_vector(rgba_weight_sum, weight_weight_sum, rgbq_sum, psum);
			rgbo_vectors[i] = rgbovec;

			// We can get a failure due to the use of a singular (non-invertible) matrix
			// If it failed, compute rgbo_vectors[] with a different method ...
			if (astc::isnan(dot_s(rgbovec, rgbovec)))
			{
				vfloat4 v0 = ep.endpt0[i];
				vfloat4 v1 = ep.endpt1[i];

				float avgdif = hadd_rgb_s(v1 - v0) * (1.0f / 3.0f);
				avgdif = astc::max(avgdif, 0.0f);

				vfloat4 avg = (v0 + v1) * 0.5f;
				vfloat4 ep0 = avg - vfloat4(avgdif) * 0.5f;
				rgbo_vectors[i] = vfloat4(ep0.lane<0>(), ep0.lane<1>(), ep0.lane<2>(), avgdif);
			}
		}
	}
}

/* See header for documentation. */
void recompute_ideal_colors_2planes(
	const image_block& blk,
	const block_size_descriptor& bsd,
	const decimation_info& di,
	const uint8_t* dec_weights_uquant_plane1,
	const uint8_t* dec_weights_uquant_plane2,
	endpoints& ep,
	vfloat4& rgbs_vector,
	vfloat4& rgbo_vector,
	int plane2_component
) {
	unsigned int weight_count = di.weight_count;
	unsigned int total_texel_count = blk.texel_count;

	promise(total_texel_count > 0);
	promise(weight_count > 0);

	ASTCENC_ALIGNAS float dec_weight_plane1[BLOCK_MAX_WEIGHTS_2PLANE];
	ASTCENC_ALIGNAS float dec_weight_plane2[BLOCK_MAX_WEIGHTS_2PLANE];

	assert(weight_count <= BLOCK_MAX_WEIGHTS_2PLANE);

	for (unsigned int i = 0; i < weight_count; i += ASTCENC_SIMD_WIDTH)
	{
		vint unquant_value1(dec_weights_uquant_plane1 + i);
		vfloat unquant_value1f = int_to_float(unquant_value1) * vfloat(1.0f / 64.0f);
		storea(unquant_value1f, dec_weight_plane1 + i);

		vint unquant_value2(dec_weights_uquant_plane2 + i);
		vfloat unquant_value2f = int_to_float(unquant_value2) * vfloat(1.0f / 64.0f);
		storea(unquant_value2f, dec_weight_plane2 + i);
	}

	ASTCENC_ALIGNAS float undec_weight_plane1[BLOCK_MAX_TEXELS];
	ASTCENC_ALIGNAS float undec_weight_plane2[BLOCK_MAX_TEXELS];

	float* undec_weight_plane1_ref;
	float* undec_weight_plane2_ref;

	if (di.max_texel_weight_count == 1)
	{
		undec_weight_plane1_ref = dec_weight_plane1;
		undec_weight_plane2_ref = dec_weight_plane2;
	}
	else if (di.max_texel_weight_count <= 2)
	{
		for (unsigned int i = 0; i < total_texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vfloat weight = bilinear_infill_vla_2(di, dec_weight_plane1, i);
			storea(weight, undec_weight_plane1 + i);

			weight = bilinear_infill_vla_2(di, dec_weight_plane2, i);
			storea(weight, undec_weight_plane2 + i);
		}

		undec_weight_plane1_ref = undec_weight_plane1;
		undec_weight_plane2_ref = undec_weight_plane2;
	}
	else
	{
		for (unsigned int i = 0; i < total_texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vfloat weight = bilinear_infill_vla(di, dec_weight_plane1, i);
			storea(weight, undec_weight_plane1 + i);

			weight = bilinear_infill_vla(di, dec_weight_plane2, i);
			storea(weight, undec_weight_plane2 + i);
		}

		undec_weight_plane1_ref = undec_weight_plane1;
		undec_weight_plane2_ref = undec_weight_plane2;
	}

	unsigned int texel_count = bsd.texel_count;
	vfloat4 rgba_weight_sum = max(blk.channel_weight * static_cast<float>(texel_count), 1e-17f);
	vfloat4 scale_dir = normalize(blk.data_mean.swz<0, 1, 2>());

	float scale_max = 0.0f;
	float scale_min = 1e10f;

	float wmin1 = 1.0f;
	float wmax1 = 0.0f;

	float wmin2 = 1.0f;
	float wmax2 = 0.0f;

	float left1_sum_s = 0.0f;
	float middle1_sum_s = 0.0f;
	float right1_sum_s = 0.0f;

	float left2_sum_s = 0.0f;
	float middle2_sum_s = 0.0f;
	float right2_sum_s = 0.0f;

	vfloat4 color_vec_x = vfloat4::zero();
	vfloat4 color_vec_y = vfloat4::zero();

	vfloat4 scale_vec = vfloat4::zero();

	vfloat4 weight_weight_sum = vfloat4(1e-17f);

	vmask4 p2_mask = vint4::lane_id() == vint4(plane2_component);
	vfloat4 color_weight = blk.channel_weight;
	float ls_weight = hadd_rgb_s(color_weight);

	for (unsigned int j = 0; j < texel_count; j++)
	{
		vfloat4 rgba = blk.texel(j);

		float idx0 = undec_weight_plane1_ref[j];

		float om_idx0 = 1.0f - idx0;
		wmin1 = astc::min(idx0, wmin1);
		wmax1 = astc::max(idx0, wmax1);

		float scale = dot3_s(scale_dir, rgba);
		scale_min = astc::min(scale, scale_min);
		scale_max = astc::max(scale, scale_max);

		left1_sum_s   += om_idx0 * om_idx0;
		middle1_sum_s += om_idx0 * idx0;
		right1_sum_s  += idx0 * idx0;

		float idx1 = undec_weight_plane2_ref[j];

		float om_idx1 = 1.0f - idx1;
		wmin2 = astc::min(idx1, wmin2);
		wmax2 = astc::max(idx1, wmax2);

		left2_sum_s   += om_idx1 * om_idx1;
		middle2_sum_s += om_idx1 * idx1;
		right2_sum_s  += idx1 * idx1;

		vfloat4 color_idx = select(vfloat4(idx0), vfloat4(idx1), p2_mask);

		vfloat4 cwprod = rgba;
		vfloat4 cwiprod = cwprod * color_idx;

		color_vec_y += cwiprod;
		color_vec_x += cwprod - cwiprod;

		scale_vec += vfloat2(om_idx0, idx0) * (ls_weight * scale);
		weight_weight_sum += color_idx;
	}

	vfloat4 left1_sum   = vfloat4(left1_sum_s) * color_weight;
	vfloat4 middle1_sum = vfloat4(middle1_sum_s) * color_weight;
	vfloat4 right1_sum  = vfloat4(right1_sum_s) * color_weight;
	vfloat4 lmrs_sum    = vfloat3(left1_sum_s, middle1_sum_s, right1_sum_s) * ls_weight;

	vfloat4 left2_sum   = vfloat4(left2_sum_s) * color_weight;
	vfloat4 middle2_sum = vfloat4(middle2_sum_s) * color_weight;
	vfloat4 right2_sum  = vfloat4(right2_sum_s) * color_weight;

	color_vec_x = color_vec_x * color_weight;
	color_vec_y = color_vec_y * color_weight;

	// Initialize the luminance and scale vectors with a reasonable default
	float scalediv = scale_min / astc::max(scale_max, 1e-10f);
	scalediv = astc::clamp1f(scalediv);

	vfloat4 sds = scale_dir * scale_max;

	rgbs_vector = vfloat4(sds.lane<0>(), sds.lane<1>(), sds.lane<2>(), scalediv);

	if (wmin1 >= wmax1 * 0.999f)
	{
		// If all weights in the partition were equal, then just take average of all colors in
		// the partition and use that as both endpoint colors
		vfloat4 avg = (color_vec_x + color_vec_y) / rgba_weight_sum;

		vmask4 p1_mask = vint4::lane_id() != vint4(plane2_component);
		vmask4 notnan_mask = avg == avg;
		vmask4 full_mask = p1_mask & notnan_mask;

		ep.endpt0[0] = select(ep.endpt0[0], avg, full_mask);
		ep.endpt1[0] = select(ep.endpt1[0], avg, full_mask);

		rgbs_vector = vfloat4(sds.lane<0>(), sds.lane<1>(), sds.lane<2>(), 1.0f);
	}
	else
	{
		// Otherwise, complete the analytic calculation of ideal-endpoint-values for the given
		// set of texel weights and pixel colors
		vfloat4 color_det1 = (left1_sum * right1_sum) - (middle1_sum * middle1_sum);
		vfloat4 color_rdet1 = 1.0f / color_det1;

		float ls_det1  = (lmrs_sum.lane<0>() * lmrs_sum.lane<2>()) - (lmrs_sum.lane<1>() * lmrs_sum.lane<1>());
		float ls_rdet1 = 1.0f / ls_det1;

		vfloat4 color_mss1 = (left1_sum * left1_sum)
		                   + (2.0f * middle1_sum * middle1_sum)
		                   + (right1_sum * right1_sum);

		float ls_mss1 = (lmrs_sum.lane<0>() * lmrs_sum.lane<0>())
		              + (2.0f * lmrs_sum.lane<1>() * lmrs_sum.lane<1>())
		              + (lmrs_sum.lane<2>() * lmrs_sum.lane<2>());

		vfloat4 ep0 = (right1_sum * color_vec_x - middle1_sum * color_vec_y) * color_rdet1;
		vfloat4 ep1 = (left1_sum * color_vec_y - middle1_sum * color_vec_x) * color_rdet1;

		float scale_ep0 = (lmrs_sum.lane<2>() * scale_vec.lane<0>() - lmrs_sum.lane<1>() * scale_vec.lane<1>()) * ls_rdet1;
		float scale_ep1 = (lmrs_sum.lane<0>() * scale_vec.lane<1>() - lmrs_sum.lane<1>() * scale_vec.lane<0>()) * ls_rdet1;

		vmask4 p1_mask = vint4::lane_id() != vint4(plane2_component);
		vmask4 det_mask = abs(color_det1) > (color_mss1 * 1e-4f);
		vmask4 notnan_mask = (ep0 == ep0) & (ep1 == ep1);
		vmask4 full_mask = p1_mask & det_mask & notnan_mask;

		ep.endpt0[0] = select(ep.endpt0[0], ep0, full_mask);
		ep.endpt1[0] = select(ep.endpt1[0], ep1, full_mask);

		if (fabsf(ls_det1) > (ls_mss1 * 1e-4f) && scale_ep0 == scale_ep0 && scale_ep1 == scale_ep1 && scale_ep0 < scale_ep1)
		{
			float scalediv2 = scale_ep0 / scale_ep1;
			vfloat4 sdsm = scale_dir * scale_ep1;
			rgbs_vector = vfloat4(sdsm.lane<0>(), sdsm.lane<1>(), sdsm.lane<2>(), scalediv2);
		}
	}

	if (wmin2 >= wmax2 * 0.999f)
	{
		// If all weights in the partition were equal, then just take average of all colors in
		// the partition and use that as both endpoint colors
		vfloat4 avg = (color_vec_x + color_vec_y) / rgba_weight_sum;

		vmask4 notnan_mask = avg == avg;
		vmask4 full_mask = p2_mask & notnan_mask;

		ep.endpt0[0] = select(ep.endpt0[0], avg, full_mask);
		ep.endpt1[0] = select(ep.endpt1[0], avg, full_mask);
	}
	else
	{
		// Otherwise, complete the analytic calculation of ideal-endpoint-values for the given
		// set of texel weights and pixel colors
		vfloat4 color_det2 = (left2_sum * right2_sum) - (middle2_sum * middle2_sum);
		vfloat4 color_rdet2 = 1.0f / color_det2;

		vfloat4 color_mss2 = (left2_sum * left2_sum)
		                   + (2.0f * middle2_sum * middle2_sum)
		                   + (right2_sum * right2_sum);

		vfloat4 ep0 = (right2_sum * color_vec_x - middle2_sum * color_vec_y) * color_rdet2;
		vfloat4 ep1 = (left2_sum * color_vec_y - middle2_sum * color_vec_x) * color_rdet2;

		vmask4 det_mask = abs(color_det2) > (color_mss2 * 1e-4f);
		vmask4 notnan_mask = (ep0 == ep0) & (ep1 == ep1);
		vmask4 full_mask = p2_mask & det_mask & notnan_mask;

		ep.endpt0[0] = select(ep.endpt0[0], ep0, full_mask);
		ep.endpt1[0] = select(ep.endpt1[0], ep1, full_mask);
	}

	// Calculations specific to mode #7, the HDR RGB-scale mode - skip if known LDR
	if (blk.rgb_lns[0] || blk.alpha_lns[0])
	{
		weight_weight_sum = weight_weight_sum * color_weight;
		float psum = dot3_s(select(right1_sum, right2_sum, p2_mask), color_weight);

		vfloat4 rgbq_sum = color_vec_x + color_vec_y;
		rgbq_sum.set_lane<3>(hadd_rgb_s(color_vec_y));

		rgbo_vector = compute_rgbo_vector(rgba_weight_sum, weight_weight_sum, rgbq_sum, psum);

		// We can get a failure due to the use of a singular (non-invertible) matrix
		// If it failed, compute rgbo_vectors[] with a different method ...
		if (astc::isnan(dot_s(rgbo_vector, rgbo_vector)))
		{
			vfloat4 v0 = ep.endpt0[0];
			vfloat4 v1 = ep.endpt1[0];

			float avgdif = hadd_rgb_s(v1 - v0) * (1.0f / 3.0f);
			avgdif = astc::max(avgdif, 0.0f);

			vfloat4 avg = (v0 + v1) * 0.5f;
			vfloat4 ep0 = avg - vfloat4(avgdif) * 0.5f;

			rgbo_vector = vfloat4(ep0.lane<0>(), ep0.lane<1>(), ep0.lane<2>(), avgdif);
		}
	}
}

#endif
