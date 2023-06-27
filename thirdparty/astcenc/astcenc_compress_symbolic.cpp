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

#if !defined(ASTCENC_DECOMPRESS_ONLY)

/**
 * @brief Functions to compress a symbolic block.
 */

#include "astcenc_internal.h"
#include "astcenc_diagnostic_trace.h"

#include <cassert>

/**
 * @brief Merge two planes of endpoints into a single vector.
 *
 * @param      ep_plane1          The endpoints for plane 1.
 * @param      ep_plane2          The endpoints for plane 2.
 * @param      component_plane2   The color component for plane 2.
 * @param[out] result             The merged output.
 */
static void merge_endpoints(
	const endpoints& ep_plane1,
	const endpoints& ep_plane2,
	unsigned int component_plane2,
	endpoints& result
) {
	unsigned int partition_count = ep_plane1.partition_count;
	assert(partition_count == 1);

	vmask4 sep_mask = vint4::lane_id() == vint4(component_plane2);

	result.partition_count = partition_count;
	result.endpt0[0] = select(ep_plane1.endpt0[0], ep_plane2.endpt0[0], sep_mask);
	result.endpt1[0] = select(ep_plane1.endpt1[0], ep_plane2.endpt1[0], sep_mask);
}

/**
 * @brief Attempt to improve weights given a chosen configuration.
 *
 * Given a fixed weight grid decimation and weight value quantization, iterate over all weights (per
 * partition and per plane) and attempt to improve image quality by moving each weight up by one or
 * down by one quantization step.
 *
 * This is a specialized function which only supports operating on undecimated weight grids,
 * therefore primarily improving the performance of 4x4 and 5x5 blocks where grid decimation
 * is needed less often.
 *
 * @param      decode_mode   The decode mode (LDR, HDR).
 * @param      bsd           The block size information.
 * @param      blk           The image block color data to compress.
 * @param[out] scb           The symbolic compressed block output.
 */
static bool realign_weights_undecimated(
	astcenc_profile decode_mode,
	const block_size_descriptor& bsd,
	const image_block& blk,
	symbolic_compressed_block& scb
) {
	// Get the partition descriptor
	unsigned int partition_count = scb.partition_count;
	const auto& pi = bsd.get_partition_info(partition_count, scb.partition_index);

	// Get the quantization table
	const block_mode& bm = bsd.get_block_mode(scb.block_mode);
	unsigned int weight_quant_level = bm.quant_mode;
	const quant_and_transfer_table& qat = quant_and_xfer_tables[weight_quant_level];

	unsigned int max_plane = bm.is_dual_plane;
	int plane2_component = scb.plane2_component;
	vmask4 plane_mask = vint4::lane_id() == vint4(plane2_component);

	// Decode the color endpoints
	bool rgb_hdr;
	bool alpha_hdr;
	vint4 endpnt0[BLOCK_MAX_PARTITIONS];
	vint4 endpnt1[BLOCK_MAX_PARTITIONS];
	vfloat4 endpnt0f[BLOCK_MAX_PARTITIONS];
	vfloat4 offset[BLOCK_MAX_PARTITIONS];

	promise(partition_count > 0);

	for (unsigned int pa_idx = 0; pa_idx < partition_count; pa_idx++)
	{
		unpack_color_endpoints(decode_mode,
		                       scb.color_formats[pa_idx],
		                       scb.color_values[pa_idx],
		                       rgb_hdr, alpha_hdr,
		                       endpnt0[pa_idx],
		                       endpnt1[pa_idx]);
	}

	uint8_t* dec_weights_uquant = scb.weights;
	bool adjustments = false;

	// For each plane and partition ...
	for (unsigned int pl_idx = 0; pl_idx <= max_plane; pl_idx++)
	{
		for (unsigned int pa_idx = 0; pa_idx < partition_count; pa_idx++)
		{
			// Compute the endpoint delta for all components in current plane
			vint4 epd = endpnt1[pa_idx] - endpnt0[pa_idx];
			epd = select(epd, vint4::zero(), plane_mask);

			endpnt0f[pa_idx] = int_to_float(endpnt0[pa_idx]);
			offset[pa_idx] = int_to_float(epd) * (1.0f / 64.0f);
		}

		// For each weight compute previous, current, and next errors
		promise(bsd.texel_count > 0);
		for (unsigned int texel = 0; texel < bsd.texel_count; texel++)
		{
			int uqw = dec_weights_uquant[texel];

			uint32_t prev_and_next = qat.prev_next_values[uqw];
			int uqw_down = prev_and_next & 0xFF;
			int uqw_up = (prev_and_next >> 8) & 0xFF;

			// Interpolate the colors to create the diffs
			float weight_base = static_cast<float>(uqw);
			float weight_down = static_cast<float>(uqw_down - uqw);
			float weight_up = static_cast<float>(uqw_up - uqw);

			unsigned int partition = pi.partition_of_texel[texel];
			vfloat4 color_offset = offset[partition];
			vfloat4 color_base   = endpnt0f[partition];

			vfloat4 color = color_base + color_offset * weight_base;
			vfloat4 orig_color   = blk.texel(texel);
			vfloat4 error_weight = blk.channel_weight;

			vfloat4 color_diff      = color - orig_color;
			vfloat4 color_diff_down = color_diff + color_offset * weight_down;
			vfloat4 color_diff_up   = color_diff + color_offset * weight_up;

			float error_base = dot_s(color_diff      * color_diff,      error_weight);
			float error_down = dot_s(color_diff_down * color_diff_down, error_weight);
			float error_up   = dot_s(color_diff_up   * color_diff_up,   error_weight);

			// Check if the prev or next error is better, and if so use it
			if ((error_up < error_base) && (error_up < error_down) && (uqw < 64))
			{
				dec_weights_uquant[texel] = static_cast<uint8_t>(uqw_up);
				adjustments = true;
			}
			else if ((error_down < error_base) && (uqw > 0))
			{
				dec_weights_uquant[texel] = static_cast<uint8_t>(uqw_down);
				adjustments = true;
			}
		}

		// Prepare iteration for plane 2
		dec_weights_uquant += WEIGHTS_PLANE2_OFFSET;
		plane_mask = ~plane_mask;
	}

	return adjustments;
}

/**
 * @brief Attempt to improve weights given a chosen configuration.
 *
 * Given a fixed weight grid decimation and weight value quantization, iterate over all weights (per
 * partition and per plane) and attempt to improve image quality by moving each weight up by one or
 * down by one quantization step.
 *
 * @param      decode_mode   The decode mode (LDR, HDR).
 * @param      bsd           The block size information.
 * @param      blk           The image block color data to compress.
 * @param[out] scb           The symbolic compressed block output.
 */
static bool realign_weights_decimated(
	astcenc_profile decode_mode,
	const block_size_descriptor& bsd,
	const image_block& blk,
	symbolic_compressed_block& scb
) {
	// Get the partition descriptor
	unsigned int partition_count = scb.partition_count;
	const auto& pi = bsd.get_partition_info(partition_count, scb.partition_index);

	// Get the quantization table
	const block_mode& bm = bsd.get_block_mode(scb.block_mode);
	unsigned int weight_quant_level = bm.quant_mode;
	const quant_and_transfer_table& qat = quant_and_xfer_tables[weight_quant_level];

	// Get the decimation table
	const decimation_info& di = bsd.get_decimation_info(bm.decimation_mode);
	unsigned int weight_count = di.weight_count;
	assert(weight_count != bsd.texel_count);

	unsigned int max_plane = bm.is_dual_plane;
	int plane2_component = scb.plane2_component;
	vmask4 plane_mask = vint4::lane_id() == vint4(plane2_component);

	// Decode the color endpoints
	bool rgb_hdr;
	bool alpha_hdr;
	vint4 endpnt0[BLOCK_MAX_PARTITIONS];
	vint4 endpnt1[BLOCK_MAX_PARTITIONS];
	vfloat4 endpnt0f[BLOCK_MAX_PARTITIONS];
	vfloat4 offset[BLOCK_MAX_PARTITIONS];

	promise(partition_count > 0);
	promise(weight_count > 0);

	for (unsigned int pa_idx = 0; pa_idx < partition_count; pa_idx++)
	{
		unpack_color_endpoints(decode_mode,
		                       scb.color_formats[pa_idx],
		                       scb.color_values[pa_idx],
		                       rgb_hdr, alpha_hdr,
		                       endpnt0[pa_idx],
		                       endpnt1[pa_idx]);
	}

	uint8_t* dec_weights_uquant = scb.weights;
	bool adjustments = false;

	// For each plane and partition ...
	for (unsigned int pl_idx = 0; pl_idx <= max_plane; pl_idx++)
	{
		for (unsigned int pa_idx = 0; pa_idx < partition_count; pa_idx++)
		{
			// Compute the endpoint delta for all components in current plane
			vint4 epd = endpnt1[pa_idx] - endpnt0[pa_idx];
			epd = select(epd, vint4::zero(), plane_mask);

			endpnt0f[pa_idx] = int_to_float(endpnt0[pa_idx]);
			offset[pa_idx] = int_to_float(epd) * (1.0f / 64.0f);
		}

		// Create an unquantized weight grid for this decimation level
		alignas(ASTCENC_VECALIGN) float uq_weightsf[BLOCK_MAX_WEIGHTS];
		for (unsigned int we_idx = 0; we_idx < weight_count; we_idx += ASTCENC_SIMD_WIDTH)
		{
			vint unquant_value(dec_weights_uquant + we_idx);
			vfloat unquant_valuef = int_to_float(unquant_value);
			storea(unquant_valuef, uq_weightsf + we_idx);
		}

		// For each weight compute previous, current, and next errors
		for (unsigned int we_idx = 0; we_idx < weight_count; we_idx++)
		{
			int uqw = dec_weights_uquant[we_idx];
			uint32_t prev_and_next = qat.prev_next_values[uqw];

			float uqw_base = uq_weightsf[we_idx];
			float uqw_down = static_cast<float>(prev_and_next & 0xFF);
			float uqw_up = static_cast<float>((prev_and_next >> 8) & 0xFF);

			float uqw_diff_down = uqw_down - uqw_base;
			float uqw_diff_up = uqw_up - uqw_base;

			vfloat4 error_basev = vfloat4::zero();
			vfloat4 error_downv = vfloat4::zero();
			vfloat4 error_upv = vfloat4::zero();

			// Interpolate the colors to create the diffs
			unsigned int texels_to_evaluate = di.weight_texel_count[we_idx];
			promise(texels_to_evaluate > 0);
			for (unsigned int te_idx = 0; te_idx < texels_to_evaluate; te_idx++)
			{
				unsigned int texel = di.weight_texels_tr[te_idx][we_idx];

				float tw_base = di.texel_contrib_for_weight[te_idx][we_idx];

				float weight_base = (uq_weightsf[di.texel_weights_tr[0][texel]] * di.texel_weight_contribs_float_tr[0][texel]
				                   + uq_weightsf[di.texel_weights_tr[1][texel]] * di.texel_weight_contribs_float_tr[1][texel])
					              + (uq_weightsf[di.texel_weights_tr[2][texel]] * di.texel_weight_contribs_float_tr[2][texel]
				                   + uq_weightsf[di.texel_weights_tr[3][texel]] * di.texel_weight_contribs_float_tr[3][texel]);

				// Ideally this is integer rounded, but IQ gain it isn't worth the overhead
				// float weight = astc::flt_rd(weight_base + 0.5f);
				// float weight_down = astc::flt_rd(weight_base + 0.5f + uqw_diff_down * tw_base) - weight;
				// float weight_up = astc::flt_rd(weight_base + 0.5f + uqw_diff_up * tw_base) - weight;
				float weight_down = weight_base + uqw_diff_down * tw_base - weight_base;
				float weight_up = weight_base + uqw_diff_up * tw_base - weight_base;

				unsigned int partition = pi.partition_of_texel[texel];
				vfloat4 color_offset = offset[partition];
				vfloat4 color_base   = endpnt0f[partition];

				vfloat4 color = color_base + color_offset * weight_base;
				vfloat4 orig_color = blk.texel(texel);

				vfloat4 color_diff      = color - orig_color;
				vfloat4 color_down_diff = color_diff + color_offset * weight_down;
				vfloat4 color_up_diff   = color_diff + color_offset * weight_up;

				error_basev += color_diff * color_diff;
				error_downv += color_down_diff * color_down_diff;
				error_upv   += color_up_diff * color_up_diff;
			}

			vfloat4 error_weight = blk.channel_weight;
			float error_base = hadd_s(error_basev * error_weight);
			float error_down = hadd_s(error_downv * error_weight);
			float error_up   = hadd_s(error_upv   * error_weight);

			// Check if the prev or next error is better, and if so use it
			if ((error_up < error_base) && (error_up < error_down) && (uqw < 64))
			{
				uq_weightsf[we_idx] = uqw_up;
				dec_weights_uquant[we_idx] = static_cast<uint8_t>(uqw_up);
				adjustments = true;
			}
			else if ((error_down < error_base) && (uqw > 0))
			{
				uq_weightsf[we_idx] = uqw_down;
				dec_weights_uquant[we_idx] = static_cast<uint8_t>(uqw_down);
				adjustments = true;
			}
		}

		// Prepare iteration for plane 2
		dec_weights_uquant += WEIGHTS_PLANE2_OFFSET;
		plane_mask = ~plane_mask;
	}

	return adjustments;
}

/**
 * @brief Compress a block using a chosen partitioning and 1 plane of weights.
 *
 * @param      config                    The compressor configuration.
 * @param      bsd                       The block size information.
 * @param      blk                       The image block color data to compress.
 * @param      only_always               True if we only use "always" percentile block modes.
 * @param      tune_errorval_threshold   The error value threshold.
 * @param      partition_count           The partition count.
 * @param      partition_index           The partition index if @c partition_count is 2-4.
 * @param[out] scb                       The symbolic compressed block output.
 * @param[out] tmpbuf                    The quantized weights for plane 1.
 */
static float compress_symbolic_block_for_partition_1plane(
	const astcenc_config& config,
	const block_size_descriptor& bsd,
	const image_block& blk,
	bool only_always,
	float tune_errorval_threshold,
	unsigned int partition_count,
	unsigned int partition_index,
	symbolic_compressed_block& scb,
	compression_working_buffers& tmpbuf,
	int quant_limit
) {
	promise(partition_count > 0);
	promise(config.tune_candidate_limit > 0);
	promise(config.tune_refinement_limit > 0);

	int max_weight_quant = astc::min(static_cast<int>(QUANT_32), quant_limit);

	auto compute_difference = &compute_symbolic_block_difference_1plane;
	if ((partition_count == 1) && !(config.flags & ASTCENC_FLG_MAP_RGBM))
	{
		compute_difference = &compute_symbolic_block_difference_1plane_1partition;
	}

	const auto& pi = bsd.get_partition_info(partition_count, partition_index);

	// Compute ideal weights and endpoint colors, with no quantization or decimation
	endpoints_and_weights& ei = tmpbuf.ei1;
	compute_ideal_colors_and_weights_1plane(blk, pi, ei);

	// Compute ideal weights and endpoint colors for every decimation
	float* dec_weights_ideal = tmpbuf.dec_weights_ideal;
	uint8_t* dec_weights_uquant = tmpbuf.dec_weights_uquant;

	// For each decimation mode, compute an ideal set of weights with no quantization
	unsigned int max_decimation_modes = only_always ? bsd.decimation_mode_count_always
	                                                : bsd.decimation_mode_count_selected;
	promise(max_decimation_modes > 0);
	for (unsigned int i = 0; i < max_decimation_modes; i++)
	{
		const auto& dm = bsd.get_decimation_mode(i);
		if (!dm.is_ref_1plane(static_cast<quant_method>(max_weight_quant)))
		{
			continue;
		}

		const auto& di = bsd.get_decimation_info(i);

		compute_ideal_weights_for_decimation(
		    ei,
		    di,
		    dec_weights_ideal + i * BLOCK_MAX_WEIGHTS);
	}

	// Compute maximum colors for the endpoints and ideal weights, then for each endpoint and ideal
	// weight pair, compute the smallest weight that will result in a color value greater than 1
	vfloat4 min_ep(10.0f);
	for (unsigned int i = 0; i < partition_count; i++)
	{
		vfloat4 ep = (vfloat4(1.0f) - ei.ep.endpt0[i]) / (ei.ep.endpt1[i] - ei.ep.endpt0[i]);

		vmask4 use_ep = (ep > vfloat4(0.5f)) & (ep < min_ep);
		min_ep = select(min_ep, ep, use_ep);
	}

	float min_wt_cutoff = hmin_s(min_ep);

	// For each mode, use the angular method to compute a shift
	compute_angular_endpoints_1plane(
	    only_always, bsd, dec_weights_ideal, max_weight_quant, tmpbuf);

	float* weight_low_value = tmpbuf.weight_low_value1;
	float* weight_high_value = tmpbuf.weight_high_value1;
	int8_t* qwt_bitcounts = tmpbuf.qwt_bitcounts;
	float* qwt_errors = tmpbuf.qwt_errors;

	// For each mode (which specifies a decimation and a quantization):
	//     * Compute number of bits needed for the quantized weights
	//     * Generate an optimized set of quantized weights
	//     * Compute quantization errors for the mode


	static const int8_t free_bits_for_partition_count[4] {
		115 - 4, 111 - 4 - PARTITION_INDEX_BITS, 108 - 4 - PARTITION_INDEX_BITS, 105 - 4 - PARTITION_INDEX_BITS
	};

	unsigned int max_block_modes = only_always ? bsd.block_mode_count_1plane_always
	                                           : bsd.block_mode_count_1plane_selected;
	promise(max_block_modes > 0);
	for (unsigned int i = 0; i < max_block_modes; i++)
	{
		const block_mode& bm = bsd.block_modes[i];

		if (bm.quant_mode > max_weight_quant)
		{
			qwt_errors[i] = 1e38f;
			continue;
		}

		assert(!bm.is_dual_plane);
		int bitcount = free_bits_for_partition_count[partition_count - 1] - bm.weight_bits;
		if (bitcount <= 0)
		{
			qwt_errors[i] = 1e38f;
			continue;
		}

		if (weight_high_value[i] > 1.02f * min_wt_cutoff)
		{
			weight_high_value[i] = 1.0f;
		}

		int decimation_mode = bm.decimation_mode;
		const auto& di = bsd.get_decimation_info(decimation_mode);

		qwt_bitcounts[i] = static_cast<int8_t>(bitcount);

		alignas(ASTCENC_VECALIGN) float dec_weights_uquantf[BLOCK_MAX_WEIGHTS];

		// Generate the optimized set of weights for the weight mode
		compute_quantized_weights_for_decimation(
		    di,
		    weight_low_value[i], weight_high_value[i],
		    dec_weights_ideal + BLOCK_MAX_WEIGHTS * decimation_mode,
		    dec_weights_uquantf,
		    dec_weights_uquant + BLOCK_MAX_WEIGHTS * i,
		    bm.get_weight_quant_mode());

		// Compute weight quantization errors for the block mode
		qwt_errors[i] = compute_error_of_weight_set_1plane(
		    ei,
		    di,
		    dec_weights_uquantf);
	}

	// Decide the optimal combination of color endpoint encodings and weight encodings
	uint8_t partition_format_specifiers[TUNE_MAX_TRIAL_CANDIDATES][BLOCK_MAX_PARTITIONS];
	int block_mode_index[TUNE_MAX_TRIAL_CANDIDATES];

	quant_method color_quant_level[TUNE_MAX_TRIAL_CANDIDATES];
	quant_method color_quant_level_mod[TUNE_MAX_TRIAL_CANDIDATES];

	unsigned int candidate_count = compute_ideal_endpoint_formats(
	    pi, blk, ei.ep, qwt_bitcounts, qwt_errors,
	    config.tune_candidate_limit, 0, max_block_modes,
	    partition_format_specifiers, block_mode_index,
	    color_quant_level, color_quant_level_mod, tmpbuf);

	// Iterate over the N believed-to-be-best modes to find out which one is actually best
	float best_errorval_in_mode = ERROR_CALC_DEFAULT;
	float best_errorval_in_scb = scb.errorval;

	for (unsigned int i = 0; i < candidate_count; i++)
	{
		TRACE_NODE(node0, "candidate");

		const int bm_packed_index = block_mode_index[i];
		assert(bm_packed_index >= 0 && bm_packed_index < static_cast<int>(bsd.block_mode_count_1plane_selected));
		const block_mode& qw_bm = bsd.block_modes[bm_packed_index];

		int decimation_mode = qw_bm.decimation_mode;
		const auto& di = bsd.get_decimation_info(decimation_mode);
		promise(di.weight_count > 0);

		trace_add_data("weight_x", di.weight_x);
		trace_add_data("weight_y", di.weight_y);
		trace_add_data("weight_z", di.weight_z);
		trace_add_data("weight_quant", qw_bm.quant_mode);

		// Recompute the ideal color endpoints before storing them
		vfloat4 rgbs_colors[BLOCK_MAX_PARTITIONS];
		vfloat4 rgbo_colors[BLOCK_MAX_PARTITIONS];

		symbolic_compressed_block workscb;
		endpoints workep = ei.ep;

		uint8_t* u8_weight_src = dec_weights_uquant + BLOCK_MAX_WEIGHTS * bm_packed_index;

		for (unsigned int j = 0; j < di.weight_count; j++)
		{
			workscb.weights[j] = u8_weight_src[j];
		}

		for (unsigned int l = 0; l < config.tune_refinement_limit; l++)
		{
			recompute_ideal_colors_1plane(
			    blk, pi, di, workscb.weights,
			    workep, rgbs_colors, rgbo_colors);

			// Quantize the chosen color, tracking if worth trying the mod value
			bool all_same = color_quant_level[i] != color_quant_level_mod[i];
			for (unsigned int j = 0; j < partition_count; j++)
			{
				workscb.color_formats[j] = pack_color_endpoints(
				    workep.endpt0[j],
				    workep.endpt1[j],
				    rgbs_colors[j],
				    rgbo_colors[j],
				    partition_format_specifiers[i][j],
				    workscb.color_values[j],
				    color_quant_level[i]);

				all_same = all_same && workscb.color_formats[j] == workscb.color_formats[0];
			}

			// If all the color endpoint modes are the same, we get a few more bits to store colors;
			// let's see if we can take advantage of this: requantize all the colors and see if the
			// endpoint modes remain the same.
			workscb.color_formats_matched = 0;
			if (partition_count >= 2 && all_same)
			{
				uint8_t colorvals[BLOCK_MAX_PARTITIONS][8];
				uint8_t color_formats_mod[BLOCK_MAX_PARTITIONS] { 0 };
				bool all_same_mod = true;
				for (unsigned int j = 0; j < partition_count; j++)
				{
					color_formats_mod[j] = pack_color_endpoints(
					    workep.endpt0[j],
					    workep.endpt1[j],
					    rgbs_colors[j],
					    rgbo_colors[j],
					    partition_format_specifiers[i][j],
					    colorvals[j],
					    color_quant_level_mod[i]);

					// Early out as soon as it's no longer possible to use mod
					if (color_formats_mod[j] != color_formats_mod[0])
					{
						all_same_mod = false;
						break;
					}
				}

				if (all_same_mod)
				{
					workscb.color_formats_matched = 1;
					for (unsigned int j = 0; j < BLOCK_MAX_PARTITIONS; j++)
					{
						for (unsigned int k = 0; k < 8; k++)
						{
							workscb.color_values[j][k] = colorvals[j][k];
						}

						workscb.color_formats[j] = color_formats_mod[j];
					}
				}
			}

			// Store header fields
			workscb.partition_count = static_cast<uint8_t>(partition_count);
			workscb.partition_index = static_cast<uint16_t>(partition_index);
			workscb.plane2_component = -1;
			workscb.quant_mode = workscb.color_formats_matched ? color_quant_level_mod[i] : color_quant_level[i];
			workscb.block_mode = qw_bm.mode_index;
			workscb.block_type = SYM_BTYPE_NONCONST;

			// Pre-realign test
			if (l == 0)
			{
				float errorval = compute_difference(config, bsd, workscb, blk);
				if (errorval == -ERROR_CALC_DEFAULT)
				{
					errorval = -errorval;
					workscb.block_type = SYM_BTYPE_ERROR;
				}

				trace_add_data("error_prerealign", errorval);
				best_errorval_in_mode = astc::min(errorval, best_errorval_in_mode);

				// Average refinement improvement is 3.5% per iteration (allow 4.5%), but the first
				// iteration can help more so we give it a extra 8% leeway. Use this knowledge to
				// drive a heuristic to skip blocks that are unlikely to catch up with the best
				// block we have already.
				unsigned int iters_remaining = config.tune_refinement_limit - l;
				float threshold = (0.045f * static_cast<float>(iters_remaining)) + 1.08f;
				if (errorval > (threshold * best_errorval_in_scb))
				{
					break;
				}

				if (errorval < best_errorval_in_scb)
				{
					best_errorval_in_scb = errorval;
					workscb.errorval = errorval;
					scb = workscb;

					if (errorval < tune_errorval_threshold)
					{
						// Skip remaining candidates - this is "good enough"
						i = candidate_count;
						break;
					}
				}
			}

			bool adjustments;
			if (di.weight_count != bsd.texel_count)
			{
				adjustments = realign_weights_decimated(
					config.profile, bsd, blk, workscb);
			}
			else
			{
				adjustments = realign_weights_undecimated(
					config.profile, bsd, blk, workscb);
			}

			// Post-realign test
			float errorval = compute_difference(config, bsd, workscb, blk);
			if (errorval == -ERROR_CALC_DEFAULT)
			{
				errorval = -errorval;
				workscb.block_type = SYM_BTYPE_ERROR;
			}

			trace_add_data("error_postrealign", errorval);
			best_errorval_in_mode = astc::min(errorval, best_errorval_in_mode);

			// Average refinement improvement is 3.5% per iteration, so skip blocks that are
			// unlikely to catch up with the best block we have already. Assume a 4.5% per step to
			// give benefit of the doubt ...
			unsigned int iters_remaining = config.tune_refinement_limit - 1 - l;
			float threshold = (0.045f * static_cast<float>(iters_remaining)) + 1.0f;
			if (errorval > (threshold * best_errorval_in_scb))
			{
				break;
			}

			if (errorval < best_errorval_in_scb)
			{
				best_errorval_in_scb = errorval;
				workscb.errorval = errorval;
				scb = workscb;

				if (errorval < tune_errorval_threshold)
				{
					// Skip remaining candidates - this is "good enough"
					i = candidate_count;
					break;
				}
			}

			if (!adjustments)
			{
				break;
			}
		}
	}

	return best_errorval_in_mode;
}

/**
 * @brief Compress a block using a chosen partitioning and 2 planes of weights.
 *
 * @param      config                    The compressor configuration.
 * @param      bsd                       The block size information.
 * @param      blk                       The image block color data to compress.
 * @param      tune_errorval_threshold   The error value threshold.
 * @param      plane2_component          The component index for the second plane of weights.
 * @param[out] scb                       The symbolic compressed block output.
 * @param[out] tmpbuf                    The quantized weights for plane 1.
 */
static float compress_symbolic_block_for_partition_2planes(
	const astcenc_config& config,
	const block_size_descriptor& bsd,
	const image_block& blk,
	float tune_errorval_threshold,
	unsigned int plane2_component,
	symbolic_compressed_block& scb,
	compression_working_buffers& tmpbuf,
	int quant_limit
) {
	promise(config.tune_candidate_limit > 0);
	promise(config.tune_refinement_limit > 0);
	promise(bsd.decimation_mode_count_selected > 0);

	int max_weight_quant = astc::min(static_cast<int>(QUANT_32), quant_limit);

	// Compute ideal weights and endpoint colors, with no quantization or decimation
	endpoints_and_weights& ei1 = tmpbuf.ei1;
	endpoints_and_weights& ei2 = tmpbuf.ei2;

	compute_ideal_colors_and_weights_2planes(bsd, blk, plane2_component, ei1, ei2);

	// Compute ideal weights and endpoint colors for every decimation
	float* dec_weights_ideal = tmpbuf.dec_weights_ideal;
	uint8_t* dec_weights_uquant = tmpbuf.dec_weights_uquant;

	// For each decimation mode, compute an ideal set of weights with no quantization
	for (unsigned int i = 0; i < bsd.decimation_mode_count_selected; i++)
	{
		const auto& dm = bsd.get_decimation_mode(i);
		if (!dm.is_ref_2plane(static_cast<quant_method>(max_weight_quant)))
		{
			continue;
		}

		const auto& di = bsd.get_decimation_info(i);

		compute_ideal_weights_for_decimation(
		    ei1,
		    di,
		    dec_weights_ideal + i * BLOCK_MAX_WEIGHTS);

		compute_ideal_weights_for_decimation(
		    ei2,
		    di,
		    dec_weights_ideal + i * BLOCK_MAX_WEIGHTS + WEIGHTS_PLANE2_OFFSET);
	}

	// Compute maximum colors for the endpoints and ideal weights, then for each endpoint and ideal
	// weight pair, compute the smallest weight that will result in a color value greater than 1
	vfloat4 min_ep1(10.0f);
	vfloat4 min_ep2(10.0f);

	vfloat4 ep1 = (vfloat4(1.0f) - ei1.ep.endpt0[0]) / (ei1.ep.endpt1[0] - ei1.ep.endpt0[0]);
	vmask4 use_ep1 = (ep1 > vfloat4(0.5f)) & (ep1 < min_ep1);
	min_ep1 = select(min_ep1, ep1, use_ep1);

	vfloat4 ep2 = (vfloat4(1.0f) - ei2.ep.endpt0[0]) / (ei2.ep.endpt1[0] - ei2.ep.endpt0[0]);
	vmask4 use_ep2 = (ep2 > vfloat4(0.5f)) & (ep2 < min_ep2);
	min_ep2 = select(min_ep2, ep2, use_ep2);

	vfloat4 err_max(ERROR_CALC_DEFAULT);
	vmask4 err_mask = vint4::lane_id() == vint4(plane2_component);

	// Set the plane2 component to max error in ep1
	min_ep1 = select(min_ep1, err_max, err_mask);

	float min_wt_cutoff1 = hmin_s(min_ep1);

	// Set the minwt2 to the plane2 component min in ep2
	float min_wt_cutoff2 = hmin_s(select(err_max, min_ep2, err_mask));

	compute_angular_endpoints_2planes(
	    bsd, dec_weights_ideal, max_weight_quant, tmpbuf);

	// For each mode (which specifies a decimation and a quantization):
	//     * Compute number of bits needed for the quantized weights
	//     * Generate an optimized set of quantized weights
	//     * Compute quantization errors for the mode

	float* weight_low_value1 = tmpbuf.weight_low_value1;
	float* weight_high_value1 = tmpbuf.weight_high_value1;
	float* weight_low_value2 = tmpbuf.weight_low_value2;
	float* weight_high_value2 = tmpbuf.weight_high_value2;

	int8_t* qwt_bitcounts = tmpbuf.qwt_bitcounts;
	float* qwt_errors = tmpbuf.qwt_errors;

	unsigned int start_2plane = bsd.block_mode_count_1plane_selected;
	unsigned int end_2plane = bsd.block_mode_count_1plane_2plane_selected;

	for (unsigned int i = start_2plane; i < end_2plane; i++)
	{
		const block_mode& bm = bsd.block_modes[i];
		assert(bm.is_dual_plane);

		if (bm.quant_mode > max_weight_quant)
		{
			qwt_errors[i] = 1e38f;
			continue;
		}

		qwt_bitcounts[i] = static_cast<int8_t>(109 - bm.weight_bits);

		if (weight_high_value1[i] > 1.02f * min_wt_cutoff1)
		{
			weight_high_value1[i] = 1.0f;
		}

		if (weight_high_value2[i] > 1.02f * min_wt_cutoff2)
		{
			weight_high_value2[i] = 1.0f;
		}

		unsigned int decimation_mode = bm.decimation_mode;
		const auto& di = bsd.get_decimation_info(decimation_mode);

		alignas(ASTCENC_VECALIGN) float dec_weights_uquantf[BLOCK_MAX_WEIGHTS];

		// Generate the optimized set of weights for the mode
		compute_quantized_weights_for_decimation(
		    di,
		    weight_low_value1[i],
		    weight_high_value1[i],
		    dec_weights_ideal + BLOCK_MAX_WEIGHTS * decimation_mode,
		    dec_weights_uquantf,
		    dec_weights_uquant + BLOCK_MAX_WEIGHTS * i,
		    bm.get_weight_quant_mode());

		compute_quantized_weights_for_decimation(
		    di,
		    weight_low_value2[i],
		    weight_high_value2[i],
		    dec_weights_ideal + BLOCK_MAX_WEIGHTS * decimation_mode + WEIGHTS_PLANE2_OFFSET,
		    dec_weights_uquantf + WEIGHTS_PLANE2_OFFSET,
		    dec_weights_uquant + BLOCK_MAX_WEIGHTS * i + WEIGHTS_PLANE2_OFFSET,
		    bm.get_weight_quant_mode());

		// Compute weight quantization errors for the block mode
		qwt_errors[i] = compute_error_of_weight_set_2planes(
		    ei1,
		    ei2,
		    di,
		    dec_weights_uquantf,
		    dec_weights_uquantf + WEIGHTS_PLANE2_OFFSET);
	}

	// Decide the optimal combination of color endpoint encodings and weight encodings
	uint8_t partition_format_specifiers[TUNE_MAX_TRIAL_CANDIDATES][BLOCK_MAX_PARTITIONS];
	int block_mode_index[TUNE_MAX_TRIAL_CANDIDATES];

	quant_method color_quant_level[TUNE_MAX_TRIAL_CANDIDATES];
	quant_method color_quant_level_mod[TUNE_MAX_TRIAL_CANDIDATES];

	endpoints epm;
	merge_endpoints(ei1.ep, ei2.ep, plane2_component, epm);

	const auto& pi = bsd.get_partition_info(1, 0);
	unsigned int candidate_count = compute_ideal_endpoint_formats(
	    pi, blk, epm, qwt_bitcounts, qwt_errors,
	    config.tune_candidate_limit,
		bsd.block_mode_count_1plane_selected, bsd.block_mode_count_1plane_2plane_selected,
	    partition_format_specifiers, block_mode_index,
	    color_quant_level, color_quant_level_mod, tmpbuf);

	// Iterate over the N believed-to-be-best modes to find out which one is actually best
	float best_errorval_in_mode = ERROR_CALC_DEFAULT;
	float best_errorval_in_scb = scb.errorval;

	for (unsigned int i = 0; i < candidate_count; i++)
	{
		TRACE_NODE(node0, "candidate");

		const int bm_packed_index = block_mode_index[i];
		assert(bm_packed_index >= static_cast<int>(bsd.block_mode_count_1plane_selected) &&
		       bm_packed_index < static_cast<int>(bsd.block_mode_count_1plane_2plane_selected));
		const block_mode& qw_bm = bsd.block_modes[bm_packed_index];

		int decimation_mode = qw_bm.decimation_mode;
		const auto& di = bsd.get_decimation_info(decimation_mode);
		promise(di.weight_count > 0);

		trace_add_data("weight_x", di.weight_x);
		trace_add_data("weight_y", di.weight_y);
		trace_add_data("weight_z", di.weight_z);
		trace_add_data("weight_quant", qw_bm.quant_mode);

		vfloat4 rgbs_color;
		vfloat4 rgbo_color;

		symbolic_compressed_block workscb;
		endpoints workep = epm;

		uint8_t* u8_weight1_src = dec_weights_uquant + BLOCK_MAX_WEIGHTS * bm_packed_index;
		uint8_t* u8_weight2_src = dec_weights_uquant + BLOCK_MAX_WEIGHTS * bm_packed_index + WEIGHTS_PLANE2_OFFSET;

		for (int j = 0; j < di.weight_count; j++)
		{
			workscb.weights[j] = u8_weight1_src[j];
			workscb.weights[j + WEIGHTS_PLANE2_OFFSET] = u8_weight2_src[j];
		}

		for (unsigned int l = 0; l < config.tune_refinement_limit; l++)
		{
			recompute_ideal_colors_2planes(
			    blk, bsd, di,
			    workscb.weights, workscb.weights + WEIGHTS_PLANE2_OFFSET,
			    workep, rgbs_color, rgbo_color, plane2_component);

			// Quantize the chosen color
			workscb.color_formats[0] = pack_color_endpoints(
			                               workep.endpt0[0],
			                               workep.endpt1[0],
			                               rgbs_color, rgbo_color,
			                               partition_format_specifiers[i][0],
			                               workscb.color_values[0],
			                               color_quant_level[i]);

			// Store header fields
			workscb.partition_count = 1;
			workscb.partition_index = 0;
			workscb.quant_mode = color_quant_level[i];
			workscb.color_formats_matched = 0;
			workscb.block_mode = qw_bm.mode_index;
			workscb.plane2_component = static_cast<int8_t>(plane2_component);
			workscb.block_type = SYM_BTYPE_NONCONST;

			// Pre-realign test
			if (l == 0)
			{
				float errorval = compute_symbolic_block_difference_2plane(config, bsd, workscb, blk);
				if (errorval == -ERROR_CALC_DEFAULT)
				{
					errorval = -errorval;
					workscb.block_type = SYM_BTYPE_ERROR;
				}

				trace_add_data("error_prerealign", errorval);
				best_errorval_in_mode = astc::min(errorval, best_errorval_in_mode);

				// Average refinement improvement is 3.5% per iteration (allow 4.5%), but the first
				// iteration can help more so we give it a extra 8% leeway. Use this knowledge to
				// drive a heuristic to skip blocks that are unlikely to catch up with the best
				// block we have already.
				unsigned int iters_remaining = config.tune_refinement_limit - l;
				float threshold = (0.045f * static_cast<float>(iters_remaining)) + 1.08f;
				if (errorval > (threshold * best_errorval_in_scb))
				{
					break;
				}

				if (errorval < best_errorval_in_scb)
				{
					best_errorval_in_scb = errorval;
					workscb.errorval = errorval;
					scb = workscb;

					if (errorval < tune_errorval_threshold)
					{
						// Skip remaining candidates - this is "good enough"
						i = candidate_count;
						break;
					}
				}
			}

			// Perform a final pass over the weights to try to improve them.
			bool adjustments;
			if (di.weight_count != bsd.texel_count)
			{
				adjustments = realign_weights_decimated(
					config.profile, bsd, blk, workscb);
			}
			else
			{
				adjustments = realign_weights_undecimated(
					config.profile, bsd, blk, workscb);
			}

			// Post-realign test
			float errorval = compute_symbolic_block_difference_2plane(config, bsd, workscb, blk);
			if (errorval == -ERROR_CALC_DEFAULT)
			{
				errorval = -errorval;
				workscb.block_type = SYM_BTYPE_ERROR;
			}

			trace_add_data("error_postrealign", errorval);
			best_errorval_in_mode = astc::min(errorval, best_errorval_in_mode);

			// Average refinement improvement is 3.5% per iteration, so skip blocks that are
			// unlikely to catch up with the best block we have already. Assume a 4.5% per step to
			// give benefit of the doubt ...
			unsigned int iters_remaining = config.tune_refinement_limit - 1 - l;
			float threshold = (0.045f * static_cast<float>(iters_remaining)) + 1.0f;
			if (errorval > (threshold * best_errorval_in_scb))
			{
				break;
			}

			if (errorval < best_errorval_in_scb)
			{
				best_errorval_in_scb = errorval;
				workscb.errorval = errorval;
				scb = workscb;

				if (errorval < tune_errorval_threshold)
				{
					// Skip remaining candidates - this is "good enough"
					i = candidate_count;
					break;
				}
			}

			if (!adjustments)
			{
				break;
			}
		}
	}

	return best_errorval_in_mode;
}

/**
 * @brief Determine the lowest cross-channel correlation factor.
 *
 * @param texels_per_block   The number of texels in a block.
 * @param blk                The image block color data to compress.
 *
 * @return Return the lowest correlation factor.
 */
static float prepare_block_statistics(
	int texels_per_block,
	const image_block& blk
) {
	// Compute covariance matrix, as a collection of 10 scalars that form the upper-triangular row
	// of the matrix. The matrix is symmetric, so this is all we need for this use case.
	float rs = 0.0f;
	float gs = 0.0f;
	float bs = 0.0f;
	float as = 0.0f;
	float rr_var = 0.0f;
	float gg_var = 0.0f;
	float bb_var = 0.0f;
	float aa_var = 0.0f;
	float rg_cov = 0.0f;
	float rb_cov = 0.0f;
	float ra_cov = 0.0f;
	float gb_cov = 0.0f;
	float ga_cov = 0.0f;
	float ba_cov = 0.0f;

	float weight_sum = 0.0f;

	promise(texels_per_block > 0);
	for (int i = 0; i < texels_per_block; i++)
	{
		float weight = hadd_s(blk.channel_weight) / 4.0f;
		assert(weight >= 0.0f);
		weight_sum += weight;

		float r = blk.data_r[i];
		float g = blk.data_g[i];
		float b = blk.data_b[i];
		float a = blk.data_a[i];

		float rw = r * weight;
		rs += rw;
		rr_var += r * rw;
		rg_cov += g * rw;
		rb_cov += b * rw;
		ra_cov += a * rw;

		float gw = g * weight;
		gs += gw;
		gg_var += g * gw;
		gb_cov += b * gw;
		ga_cov += a * gw;

		float bw = b * weight;
		bs += bw;
		bb_var += b * bw;
		ba_cov += a * bw;

		float aw = a * weight;
		as += aw;
		aa_var += a * aw;
	}

	float rpt = 1.0f / astc::max(weight_sum, 1e-7f);

	rr_var -= rs * (rs * rpt);
	rg_cov -= gs * (rs * rpt);
	rb_cov -= bs * (rs * rpt);
	ra_cov -= as * (rs * rpt);

	gg_var -= gs * (gs * rpt);
	gb_cov -= bs * (gs * rpt);
	ga_cov -= as * (gs * rpt);

	bb_var -= bs * (bs * rpt);
	ba_cov -= as * (bs * rpt);

	aa_var -= as * (as * rpt);

	// These will give a NaN if a channel is constant - these are fixed up in the next step
	rg_cov *= astc::rsqrt(rr_var * gg_var);
	rb_cov *= astc::rsqrt(rr_var * bb_var);
	ra_cov *= astc::rsqrt(rr_var * aa_var);
	gb_cov *= astc::rsqrt(gg_var * bb_var);
	ga_cov *= astc::rsqrt(gg_var * aa_var);
	ba_cov *= astc::rsqrt(bb_var * aa_var);

	if (astc::isnan(rg_cov)) rg_cov = 1.0f;
	if (astc::isnan(rb_cov)) rb_cov = 1.0f;
	if (astc::isnan(ra_cov)) ra_cov = 1.0f;
	if (astc::isnan(gb_cov)) gb_cov = 1.0f;
	if (astc::isnan(ga_cov)) ga_cov = 1.0f;
	if (astc::isnan(ba_cov)) ba_cov = 1.0f;

	float lowest_correlation = astc::min(fabsf(rg_cov),      fabsf(rb_cov));
	lowest_correlation       = astc::min(lowest_correlation, fabsf(ra_cov));
	lowest_correlation       = astc::min(lowest_correlation, fabsf(gb_cov));
	lowest_correlation       = astc::min(lowest_correlation, fabsf(ga_cov));
	lowest_correlation       = astc::min(lowest_correlation, fabsf(ba_cov));

	// Diagnostic trace points
	trace_add_data("min_r", blk.data_min.lane<0>());
	trace_add_data("max_r", blk.data_max.lane<0>());
	trace_add_data("min_g", blk.data_min.lane<1>());
	trace_add_data("max_g", blk.data_max.lane<1>());
	trace_add_data("min_b", blk.data_min.lane<2>());
	trace_add_data("max_b", blk.data_max.lane<2>());
	trace_add_data("min_a", blk.data_min.lane<3>());
	trace_add_data("max_a", blk.data_max.lane<3>());
	trace_add_data("cov_rg", fabsf(rg_cov));
	trace_add_data("cov_rb", fabsf(rb_cov));
	trace_add_data("cov_ra", fabsf(ra_cov));
	trace_add_data("cov_gb", fabsf(gb_cov));
	trace_add_data("cov_ga", fabsf(ga_cov));
	trace_add_data("cov_ba", fabsf(ba_cov));

	return lowest_correlation;
}

/* See header for documentation. */
void compress_block(
	const astcenc_contexti& ctx,
	const image_block& blk,
	physical_compressed_block& pcb,
	compression_working_buffers& tmpbuf)
{
	astcenc_profile decode_mode = ctx.config.profile;
	symbolic_compressed_block scb;
	const block_size_descriptor& bsd = *ctx.bsd;
	float lowest_correl;

	TRACE_NODE(node0, "block");
	trace_add_data("pos_x", blk.xpos);
	trace_add_data("pos_y", blk.ypos);
	trace_add_data("pos_z", blk.zpos);

	// Set stricter block targets for luminance data as we have more bits to play with
	bool block_is_l = blk.is_luminance();
	float block_is_l_scale = block_is_l ? 1.0f / 1.5f : 1.0f;

	// Set slightly stricter block targets for lumalpha data as we have more bits to play with
	bool block_is_la = blk.is_luminancealpha();
	float block_is_la_scale = block_is_la ? 1.0f / 1.05f : 1.0f;

	bool block_skip_two_plane = false;
	int max_partitions = ctx.config.tune_partition_count_limit;

	unsigned int requested_partition_indices[3] {
		ctx.config.tune_2partition_index_limit,
		ctx.config.tune_3partition_index_limit,
		ctx.config.tune_4partition_index_limit
	};

	unsigned int requested_partition_trials[3] {
		ctx.config.tune_2partitioning_candidate_limit,
		ctx.config.tune_3partitioning_candidate_limit,
		ctx.config.tune_4partitioning_candidate_limit
	};

#if defined(ASTCENC_DIAGNOSTICS)
	// Do this early in diagnostic builds so we can dump uniform metrics
	// for every block. Do it later in release builds to avoid redundant work!
	float error_weight_sum = hadd_s(blk.channel_weight) * bsd.texel_count;
	float error_threshold = ctx.config.tune_db_limit
	                      * error_weight_sum
	                      * block_is_l_scale
	                      * block_is_la_scale;

	lowest_correl = prepare_block_statistics(bsd.texel_count, blk);
	trace_add_data("lowest_correl", lowest_correl);
	trace_add_data("tune_error_threshold", error_threshold);
#endif

	// Detected a constant-color block
	if (all(blk.data_min == blk.data_max))
	{
		TRACE_NODE(node1, "pass");
		trace_add_data("partition_count", 0);
		trace_add_data("plane_count", 1);

		scb.partition_count = 0;

		// Encode as FP16 if using HDR
		if ((decode_mode == ASTCENC_PRF_HDR) ||
		    (decode_mode == ASTCENC_PRF_HDR_RGB_LDR_A))
		{
			scb.block_type = SYM_BTYPE_CONST_F16;
			vint4 color_f16 = float_to_float16(blk.origin_texel);
			store(color_f16, scb.constant_color);
		}
		// Encode as UNORM16 if NOT using HDR
		else
		{
			scb.block_type = SYM_BTYPE_CONST_U16;
			vfloat4 color_f32 = clamp(0.0f, 1.0f, blk.origin_texel) * 65535.0f;
			vint4 color_u16 = float_to_int_rtn(color_f32);
			store(color_u16, scb.constant_color);
		}

		trace_add_data("exit", "quality hit");

		symbolic_to_physical(bsd, scb, pcb);
		return;
	}

#if !defined(ASTCENC_DIAGNOSTICS)
	float error_weight_sum = hadd_s(blk.channel_weight) * bsd.texel_count;
	float error_threshold = ctx.config.tune_db_limit
	                      * error_weight_sum
	                      * block_is_l_scale
	                      * block_is_la_scale;
#endif

	// Set SCB and mode errors to a very high error value
	scb.errorval = ERROR_CALC_DEFAULT;
	scb.block_type = SYM_BTYPE_ERROR;

	float best_errorvals_for_pcount[BLOCK_MAX_PARTITIONS] {
		ERROR_CALC_DEFAULT, ERROR_CALC_DEFAULT, ERROR_CALC_DEFAULT, ERROR_CALC_DEFAULT
	};

	float exit_thresholds_for_pcount[BLOCK_MAX_PARTITIONS] {
		0.0f,
		ctx.config.tune_2partition_early_out_limit_factor,
		ctx.config.tune_3partition_early_out_limit_factor,
		0.0f
	};

	// Trial using 1 plane of weights and 1 partition.

	// Most of the time we test it twice, first with a mode cutoff of 0 and then with the specified
	// mode cutoff. This causes an early-out that speeds up encoding of easy blocks. However, this
	// optimization is disabled for 4x4 and 5x4 blocks where it nearly always slows down the
	// compression and slightly reduces image quality.

	float errorval_mult[2] {
		1.0f / ctx.config.tune_mse_overshoot,
		1.0f
	};

	static const float errorval_overshoot = 1.0f / ctx.config.tune_mse_overshoot;

	// Only enable MODE0 fast path (trial 0) if 2D, and more than 25 texels
	int start_trial = 1;
	if ((bsd.texel_count >= TUNE_MIN_TEXELS_MODE0_FASTPATH) && (bsd.zdim == 1))
	{
		start_trial = 0;
	}

	int quant_limit = QUANT_32;
	for (int i = start_trial; i < 2; i++)
	{
		TRACE_NODE(node1, "pass");
		trace_add_data("partition_count", 1);
		trace_add_data("plane_count", 1);
		trace_add_data("search_mode", i);

		float errorval = compress_symbolic_block_for_partition_1plane(
		    ctx.config, bsd, blk, i == 0,
		    error_threshold * errorval_mult[i] * errorval_overshoot,
		    1, 0,  scb, tmpbuf, QUANT_32);

		// Record the quant level so we can use the filter later searches
		const auto& bm = bsd.get_block_mode(scb.block_mode);
		quant_limit = bm.get_weight_quant_mode();

		best_errorvals_for_pcount[0] = astc::min(best_errorvals_for_pcount[0], errorval);
		if (errorval < (error_threshold * errorval_mult[i]))
		{
			trace_add_data("exit", "quality hit");
			goto END_OF_TESTS;
		}
	}

#if !defined(ASTCENC_DIAGNOSTICS)
	lowest_correl = prepare_block_statistics(bsd.texel_count, blk);
#endif

	block_skip_two_plane = lowest_correl > ctx.config.tune_2plane_early_out_limit_correlation;

	// Test the four possible 1-partition, 2-planes modes. Do this in reverse, as
	// alpha is the most likely to be non-correlated if it is present in the data.
	for (int i = BLOCK_MAX_COMPONENTS - 1; i >= 0; i--)
	{
		TRACE_NODE(node1, "pass");
		trace_add_data("partition_count", 1);
		trace_add_data("plane_count", 2);
		trace_add_data("plane_component", i);

		if (block_skip_two_plane)
		{
			trace_add_data("skip", "tune_2plane_early_out_limit_correlation");
			continue;
		}

		if (blk.grayscale && i != 3)
		{
			trace_add_data("skip", "grayscale block");
			continue;
		}

		if (blk.is_constant_channel(i))
		{
			trace_add_data("skip", "constant component");
			continue;
		}

		float errorval = compress_symbolic_block_for_partition_2planes(
		    ctx.config, bsd, blk, error_threshold * errorval_overshoot,
		    i, scb, tmpbuf, quant_limit);

		// If attempting two planes is much worse than the best one plane result
		// then further two plane searches are unlikely to help so move on ...
		if (errorval > (best_errorvals_for_pcount[0] * 1.85f))
		{
			break;
		}

		if (errorval < error_threshold)
		{
			trace_add_data("exit", "quality hit");
			goto END_OF_TESTS;
		}
	}

	// Find best blocks for 2, 3 and 4 partitions
	for (int partition_count = 2; partition_count <= max_partitions; partition_count++)
	{
		unsigned int partition_indices[TUNE_MAX_PARTITIONING_CANDIDATES];

		unsigned int requested_indices = requested_partition_indices[partition_count - 2];

		unsigned int requested_trials = requested_partition_trials[partition_count - 2];
		requested_trials = astc::min(requested_trials, requested_indices);

		unsigned int actual_trials = find_best_partition_candidates(
		    bsd, blk, partition_count, requested_indices, partition_indices, requested_trials);

		float best_error_in_prev = best_errorvals_for_pcount[partition_count - 2];

		for (unsigned int i = 0; i < actual_trials; i++)
		{
			TRACE_NODE(node1, "pass");
			trace_add_data("partition_count", partition_count);
			trace_add_data("partition_index", partition_indices[i]);
			trace_add_data("plane_count", 1);
			trace_add_data("search_mode", i);

			float errorval = compress_symbolic_block_for_partition_1plane(
			    ctx.config, bsd, blk, false,
			    error_threshold * errorval_overshoot,
			    partition_count, partition_indices[i],
			    scb, tmpbuf, quant_limit);

			best_errorvals_for_pcount[partition_count - 1] = astc::min(best_errorvals_for_pcount[partition_count - 1], errorval);

			// If using N partitions doesn't improve much over using N-1 partitions then skip trying
			// N+1. Error can dramatically improve if the data is correlated or non-correlated and
			// aligns with a partitioning that suits that encoding, so for this inner loop check add
			// a large error scale because the "other" trial could be a lot better.
			float best_error = best_errorvals_for_pcount[partition_count - 1];
			float best_error_scale = exit_thresholds_for_pcount[partition_count - 1] * 1.85f;
			if (best_error > (best_error_in_prev * best_error_scale))
			{
				trace_add_data("skip", "tune_partition_early_out_limit_factor");
				goto END_OF_TESTS;
			}

			if (errorval < error_threshold)
			{
				trace_add_data("exit", "quality hit");
				goto END_OF_TESTS;
			}
		}

		// If using N partitions doesn't improve much over using N-1 partitions then skip trying N+1
		float best_error = best_errorvals_for_pcount[partition_count - 1];
		float best_error_scale = exit_thresholds_for_pcount[partition_count - 1];
		if (best_error > (best_error_in_prev * best_error_scale))
		{
			trace_add_data("skip", "tune_partition_early_out_limit_factor");
			goto END_OF_TESTS;
		}
	}

	trace_add_data("exit", "quality not hit");

END_OF_TESTS:
	// If we still have an error block then convert to something we can encode
	// TODO: Do something more sensible here, such as average color block
	if (scb.block_type == SYM_BTYPE_ERROR)
	{
#if defined(ASTCENC_DIAGNOSTICS)
		static bool printed_once = false;
		if (!printed_once)
		{
			printed_once = true;
			printf("WARN: At least one block failed to find a valid encoding.\n"
			       "      Try increasing compression quality settings.\n\n");
		}
#endif

		scb.block_type = SYM_BTYPE_CONST_U16;
		vfloat4 color_f32 = clamp(0.0f, 1.0f, blk.origin_texel) * 65535.0f;
		vint4 color_u16 = float_to_int_rtn(color_f32);
		store(color_u16, scb.constant_color);
	}

	// Compress to a physical block
	symbolic_to_physical(bsd, scb, pcb);
}

#endif
