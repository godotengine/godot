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

/**
 * @brief Functions to decompress a symbolic block.
 */

#include "astcenc_internal.h"

#include <stdio.h>
#include <assert.h>

/**
 * @brief Compute the integer linear interpolation of two color endpoints.
 *
 * @param u8_mask       The mask for lanes using decode_unorm8 rather than decode_f16.
 * @param color0        The endpoint0 color.
 * @param color1        The endpoint1 color.
 * @param weights       The interpolation weight (between 0 and 64).
 *
 * @return The interpolated color.
 */
static vint4 lerp_color_int(
	vmask4 u8_mask,
	vint4 color0,
	vint4 color1,
	vint4 weights
) {
	vint4 weight1 = weights;
	vint4 weight0 = vint4(64) - weight1;

	vint4 color = (color0 * weight0) + (color1 * weight1) + vint4(32);
	color = asr<6>(color);

	// For decode_unorm8 values force the codec to bit replicate. This allows the
	// rest of the codec to assume the full 0xFFFF range for everything and ignore
	// the decode_mode setting
	vint4 color_u8 = asr<8>(color) * vint4(257);
	color = select(color, color_u8, u8_mask);

	return color;
}

/**
 * @brief Convert integer color value into a float value for the decoder.
 *
 * @param data       The integer color value post-interpolation.
 * @param lns_mask   If set treat lane as HDR (LNS) else LDR (unorm16).
 *
 * @return The float color value.
 */
static inline vfloat4 decode_texel(
	vint4 data,
	vmask4 lns_mask
) {
	vint4 color_lns = vint4::zero();
	vint4 color_unorm = vint4::zero();

	if (any(lns_mask))
	{
		color_lns = lns_to_sf16(data);
	}

	if (!all(lns_mask))
	{
		color_unorm = unorm16_to_sf16(data);
	}

	// Pick components and then convert to FP16
	vint4 datai = select(color_unorm, color_lns, lns_mask);
	return float16_to_float(datai);
}

/* See header for documentation. */
void unpack_weights(
	const block_size_descriptor& bsd,
	const symbolic_compressed_block& scb,
	const decimation_info& di,
	bool is_dual_plane,
	int weights_plane1[BLOCK_MAX_TEXELS],
	int weights_plane2[BLOCK_MAX_TEXELS]
) {
	// Safe to overshoot as all arrays are allocated to full size
	if (!is_dual_plane)
	{
		// Build full 64-entry weight lookup table
		vint4 tab0 = vint4::load(scb.weights +  0);
		vint4 tab1 = vint4::load(scb.weights + 16);
		vint4 tab2 = vint4::load(scb.weights + 32);
		vint4 tab3 = vint4::load(scb.weights + 48);

		vint tab0p, tab1p, tab2p, tab3p;
		vtable_prepare(tab0, tab1, tab2, tab3, tab0p, tab1p, tab2p, tab3p);

		for (unsigned int i = 0; i < bsd.texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vint summed_value(8);
			vint weight_count(di.texel_weight_count + i);
			int max_weight_count = hmax(weight_count).lane<0>();

			promise(max_weight_count > 0);
			for (int j = 0; j < max_weight_count; j++)
			{
				vint texel_weights(di.texel_weights_tr[j] + i);
				vint texel_weights_int(di.texel_weight_contribs_int_tr[j] + i);

				summed_value += vtable_8bt_32bi(tab0p, tab1p, tab2p, tab3p, texel_weights) * texel_weights_int;
			}

			store(lsr<4>(summed_value), weights_plane1 + i);
		}
	}
	else
	{
		// Build a 32-entry weight lookup table per plane
		// Plane 1
		vint4 tab0_plane1 = vint4::load(scb.weights +  0);
		vint4 tab1_plane1 = vint4::load(scb.weights + 16);
		vint tab0_plane1p, tab1_plane1p;
		vtable_prepare(tab0_plane1, tab1_plane1, tab0_plane1p, tab1_plane1p);

		// Plane 2
		vint4 tab0_plane2 = vint4::load(scb.weights + 32);
		vint4 tab1_plane2 = vint4::load(scb.weights + 48);
		vint tab0_plane2p, tab1_plane2p;
		vtable_prepare(tab0_plane2, tab1_plane2, tab0_plane2p, tab1_plane2p);

		for (unsigned int i = 0; i < bsd.texel_count; i += ASTCENC_SIMD_WIDTH)
		{
			vint sum_plane1(8);
			vint sum_plane2(8);

			vint weight_count(di.texel_weight_count + i);
			int max_weight_count = hmax(weight_count).lane<0>();

			promise(max_weight_count > 0);
			for (int j = 0; j < max_weight_count; j++)
			{
				vint texel_weights(di.texel_weights_tr[j] + i);
				vint texel_weights_int(di.texel_weight_contribs_int_tr[j] + i);

				sum_plane1 += vtable_8bt_32bi(tab0_plane1p, tab1_plane1p, texel_weights) * texel_weights_int;
				sum_plane2 += vtable_8bt_32bi(tab0_plane2p, tab1_plane2p, texel_weights) * texel_weights_int;
			}

			store(lsr<4>(sum_plane1), weights_plane1 + i);
			store(lsr<4>(sum_plane2), weights_plane2 + i);
		}
	}
}

/**
 * @brief Return an FP32 NaN value for use in error colors.
 *
 * This NaN encoding will turn into 0xFFFF when converted to an FP16 NaN.
 *
 * @return The float color value.
 */
static float error_color_nan()
{
	if32 v;
	v.u = 0xFFFFE000U;
	return v.f;
}

/* See header for documentation. */
void decompress_symbolic_block(
	astcenc_profile decode_mode,
	const block_size_descriptor& bsd,
	int xpos,
	int ypos,
	int zpos,
	const symbolic_compressed_block& scb,
	image_block& blk
) {
	blk.xpos = xpos;
	blk.ypos = ypos;
	blk.zpos = zpos;

	blk.data_min = vfloat4::zero();
	blk.data_mean = vfloat4::zero();
	blk.data_max = vfloat4::zero();
	blk.grayscale = false;

	// If we detected an error-block, blow up immediately.
	if (scb.block_type == SYM_BTYPE_ERROR)
	{
		for (unsigned int i = 0; i < bsd.texel_count; i++)
		{
			blk.data_r[i] = error_color_nan();
			blk.data_g[i] = error_color_nan();
			blk.data_b[i] = error_color_nan();
			blk.data_a[i] = error_color_nan();
			blk.rgb_lns[i] = 0;
			blk.alpha_lns[i] = 0;
		}

		return;
	}

	if ((scb.block_type == SYM_BTYPE_CONST_F16) ||
	    (scb.block_type == SYM_BTYPE_CONST_U16))
	{
		vfloat4 color;
		uint8_t use_lns = 0;

		// UNORM16 constant color block
		if (scb.block_type == SYM_BTYPE_CONST_U16)
		{
			vint4 colori(scb.constant_color);

			// Determine the UNORM8 rounding on the decode
			vmask4 u8_mask = get_u8_component_mask(decode_mode, blk);

			// The real decoder would just use the top 8 bits, but we rescale
			// in to a 16-bit value that rounds correctly.
			vint4 colori_u8 = asr<8>(colori) * 257;
			colori = select(colori, colori_u8, u8_mask);

			vint4 colorf16 = unorm16_to_sf16(colori);
			color = float16_to_float(colorf16);
		}
		// FLOAT16 constant color block
		else
		{
			switch (decode_mode)
			{
			case ASTCENC_PRF_LDR_SRGB:
			case ASTCENC_PRF_LDR:
				color = vfloat4(error_color_nan());
				break;
			case ASTCENC_PRF_HDR_RGB_LDR_A:
			case ASTCENC_PRF_HDR:
				// Constant-color block; unpack from FP16 to FP32.
				color = float16_to_float(vint4(scb.constant_color));
				use_lns = 1;
				break;
			}
		}

		for (unsigned int i = 0; i < bsd.texel_count; i++)
		{
			blk.data_r[i] = color.lane<0>();
			blk.data_g[i] = color.lane<1>();
			blk.data_b[i] = color.lane<2>();
			blk.data_a[i] = color.lane<3>();
			blk.rgb_lns[i] = use_lns;
			blk.alpha_lns[i] = use_lns;
		}

		return;
	}

	// Get the appropriate partition-table entry
	int partition_count = scb.partition_count;
	const auto& pi = bsd.get_partition_info(partition_count, scb.partition_index);

	// Get the appropriate block descriptors
	const auto& bm = bsd.get_block_mode(scb.block_mode);
	const auto& di = bsd.get_decimation_info(bm.decimation_mode);

	bool is_dual_plane = static_cast<bool>(bm.is_dual_plane);

	// Unquantize and undecimate the weights
	int plane1_weights[BLOCK_MAX_TEXELS];
	int plane2_weights[BLOCK_MAX_TEXELS];
	unpack_weights(bsd, scb, di, is_dual_plane, plane1_weights, plane2_weights);

	// Now that we have endpoint colors and weights, we can unpack texel colors
	int plane2_component = scb.plane2_component;
	vmask4 plane2_mask = vint4::lane_id() == vint4(plane2_component);

	vmask4 u8_mask = get_u8_component_mask(decode_mode, blk);

	for (int i = 0; i < partition_count; i++)
	{
		// Decode the color endpoints for this partition
		vint4 ep0;
		vint4 ep1;
		bool rgb_lns;
		bool a_lns;

		unpack_color_endpoints(decode_mode,
		                       scb.color_formats[i],
		                       scb.color_values[i],
		                       rgb_lns, a_lns,
		                       ep0, ep1);

		vmask4 lns_mask(rgb_lns, rgb_lns, rgb_lns, a_lns);

		int texel_count = pi.partition_texel_count[i];
		for (int j = 0; j < texel_count; j++)
		{
			int tix = pi.texels_of_partition[i][j];
			vint4 weight = select(vint4(plane1_weights[tix]), vint4(plane2_weights[tix]), plane2_mask);
			vint4 color = lerp_color_int(u8_mask, ep0, ep1, weight);
			vfloat4 colorf = decode_texel(color, lns_mask);

			blk.data_r[tix] = colorf.lane<0>();
			blk.data_g[tix] = colorf.lane<1>();
			blk.data_b[tix] = colorf.lane<2>();
			blk.data_a[tix] = colorf.lane<3>();
		}
	}
}

#if !defined(ASTCENC_DECOMPRESS_ONLY)

/* See header for documentation. */
float compute_symbolic_block_difference_2plane(
	const astcenc_config& config,
	const block_size_descriptor& bsd,
	const symbolic_compressed_block& scb,
	const image_block& blk
) {
	// If we detected an error-block, blow up immediately.
	if (scb.block_type == SYM_BTYPE_ERROR)
	{
		return ERROR_CALC_DEFAULT;
	}

	assert(scb.block_mode >= 0);
	assert(scb.partition_count == 1);
	assert(bsd.get_block_mode(scb.block_mode).is_dual_plane == 1);

	// Get the appropriate block descriptor
	const block_mode& bm = bsd.get_block_mode(scb.block_mode);
	const decimation_info& di = bsd.get_decimation_info(bm.decimation_mode);

	// Unquantize and undecimate the weights
	int plane1_weights[BLOCK_MAX_TEXELS];
	int plane2_weights[BLOCK_MAX_TEXELS];
	unpack_weights(bsd, scb, di, true, plane1_weights, plane2_weights);

	vmask4 plane2_mask = vint4::lane_id() == vint4(scb.plane2_component);

	vfloat4 summa = vfloat4::zero();

	// Decode the color endpoints for this partition
	vint4 ep0;
	vint4 ep1;
	bool rgb_lns;
	bool a_lns;

	unpack_color_endpoints(config.profile,
	                       scb.color_formats[0],
	                       scb.color_values[0],
	                       rgb_lns, a_lns,
	                       ep0, ep1);

	vmask4 u8_mask = get_u8_component_mask(config.profile, blk);

	// Unpack and compute error for each texel in the partition
	unsigned int texel_count = bsd.texel_count;
	for (unsigned int i = 0; i < texel_count; i++)
	{
		vint4 weight = select(vint4(plane1_weights[i]), vint4(plane2_weights[i]), plane2_mask);
		vint4 colori = lerp_color_int(u8_mask, ep0, ep1, weight);

		vfloat4 color = int_to_float(colori);
		vfloat4 oldColor = blk.texel(i);

		// Compare error using a perceptual decode metric for RGBM textures
		if (config.flags & ASTCENC_FLG_MAP_RGBM)
		{
			// Fail encodings that result in zero weight M pixels. Note that this can cause
			// "interesting" artifacts if we reject all useful encodings - we typically get max
			// brightness encodings instead which look just as bad. We recommend users apply a
			// bias to their stored M value, limiting the lower value to 16 or 32 to avoid
			// getting small M values post-quantization, but we can't prove it would never
			// happen, especially at low bit rates ...
			if (color.lane<3>() == 0.0f)
			{
				return -ERROR_CALC_DEFAULT;
			}

			// Compute error based on decoded RGBM color
			color = vfloat4(
				color.lane<0>() * color.lane<3>() * config.rgbm_m_scale,
				color.lane<1>() * color.lane<3>() * config.rgbm_m_scale,
				color.lane<2>() * color.lane<3>() * config.rgbm_m_scale,
				1.0f
			);

			oldColor = vfloat4(
				oldColor.lane<0>() * oldColor.lane<3>() * config.rgbm_m_scale,
				oldColor.lane<1>() * oldColor.lane<3>() * config.rgbm_m_scale,
				oldColor.lane<2>() * oldColor.lane<3>() * config.rgbm_m_scale,
				1.0f
			);
		}

		vfloat4 error = oldColor - color;
		error = min(abs(error), 1e15f);
		error = error * error;

		summa += min(dot(error, blk.channel_weight), ERROR_CALC_DEFAULT);
	}

	return summa.lane<0>();
}

/* See header for documentation. */
float compute_symbolic_block_difference_1plane(
	const astcenc_config& config,
	const block_size_descriptor& bsd,
	const symbolic_compressed_block& scb,
	const image_block& blk
) {
	assert(bsd.get_block_mode(scb.block_mode).is_dual_plane == 0);

	// If we detected an error-block, blow up immediately.
	if (scb.block_type == SYM_BTYPE_ERROR)
	{
		return ERROR_CALC_DEFAULT;
	}

	assert(scb.block_mode >= 0);

	// Get the appropriate partition-table entry
	unsigned int partition_count = scb.partition_count;
	const auto& pi = bsd.get_partition_info(partition_count, scb.partition_index);

	// Get the appropriate block descriptor
	const block_mode& bm = bsd.get_block_mode(scb.block_mode);
	const decimation_info& di = bsd.get_decimation_info(bm.decimation_mode);

	// Unquantize and undecimate the weights
	int plane1_weights[BLOCK_MAX_TEXELS];
	unpack_weights(bsd, scb, di, false, plane1_weights, nullptr);

	vmask4 u8_mask = get_u8_component_mask(config.profile, blk);

	vfloat4 summa = vfloat4::zero();
	for (unsigned int i = 0; i < partition_count; i++)
	{
		// Decode the color endpoints for this partition
		vint4 ep0;
		vint4 ep1;
		bool rgb_lns;
		bool a_lns;

		unpack_color_endpoints(config.profile,
		                       scb.color_formats[i],
		                       scb.color_values[i],
		                       rgb_lns, a_lns,
		                       ep0, ep1);

		// Unpack and compute error for each texel in the partition
		unsigned int texel_count = pi.partition_texel_count[i];
		for (unsigned int j = 0; j < texel_count; j++)
		{
			unsigned int tix = pi.texels_of_partition[i][j];
			vint4 colori = lerp_color_int(u8_mask, ep0, ep1,
			                              vint4(plane1_weights[tix]));

			vfloat4 color = int_to_float(colori);
			vfloat4 oldColor = blk.texel(tix);

			// Compare error using a perceptual decode metric for RGBM textures
			if (config.flags & ASTCENC_FLG_MAP_RGBM)
			{
				// Fail encodings that result in zero weight M pixels. Note that this can cause
				// "interesting" artifacts if we reject all useful encodings - we typically get max
				// brightness encodings instead which look just as bad. We recommend users apply a
				// bias to their stored M value, limiting the lower value to 16 or 32 to avoid
				// getting small M values post-quantization, but we can't prove it would never
				// happen, especially at low bit rates ...
				if (color.lane<3>() == 0.0f)
				{
					return -ERROR_CALC_DEFAULT;
				}

				// Compute error based on decoded RGBM color
				color = vfloat4(
					color.lane<0>() * color.lane<3>() * config.rgbm_m_scale,
					color.lane<1>() * color.lane<3>() * config.rgbm_m_scale,
					color.lane<2>() * color.lane<3>() * config.rgbm_m_scale,
					1.0f
				);

				oldColor = vfloat4(
					oldColor.lane<0>() * oldColor.lane<3>() * config.rgbm_m_scale,
					oldColor.lane<1>() * oldColor.lane<3>() * config.rgbm_m_scale,
					oldColor.lane<2>() * oldColor.lane<3>() * config.rgbm_m_scale,
					1.0f
				);
			}

			vfloat4 error = oldColor - color;
			error = min(abs(error), 1e15f);
			error = error * error;

			summa += min(dot(error, blk.channel_weight), ERROR_CALC_DEFAULT);
		}
	}

	return summa.lane<0>();
}

/* See header for documentation. */
float compute_symbolic_block_difference_1plane_1partition(
	const astcenc_config& config,
	const block_size_descriptor& bsd,
	const symbolic_compressed_block& scb,
	const image_block& blk
) {
	// If we detected an error-block, blow up immediately.
	if (scb.block_type == SYM_BTYPE_ERROR)
	{
		return ERROR_CALC_DEFAULT;
	}

	assert(scb.block_mode >= 0);
	assert(bsd.get_partition_info(scb.partition_count, scb.partition_index).partition_count == 1);

	// Get the appropriate block descriptor
	const block_mode& bm = bsd.get_block_mode(scb.block_mode);
	const decimation_info& di = bsd.get_decimation_info(bm.decimation_mode);

	// Unquantize and undecimate the weights
	ASTCENC_ALIGNAS int plane1_weights[BLOCK_MAX_TEXELS];
	unpack_weights(bsd, scb, di, false, plane1_weights, nullptr);

	// Decode the color endpoints for this partition
	vint4 ep0;
	vint4 ep1;
	bool rgb_lns;
	bool a_lns;

	unpack_color_endpoints(config.profile,
	                       scb.color_formats[0],
	                       scb.color_values[0],
	                       rgb_lns, a_lns,
	                       ep0, ep1);

	vmask4 u8_mask = get_u8_component_mask(config.profile, blk);

	// Unpack and compute error for each texel in the partition
	vfloatacc summav = vfloatacc::zero();

	vint lane_id = vint::lane_id();

	unsigned int texel_count = bsd.texel_count;
	for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
	{
		// Compute EP1 contribution
		vint weight1 = vint::loada(plane1_weights + i);
		vint ep1_r = vint(ep1.lane<0>()) * weight1;
		vint ep1_g = vint(ep1.lane<1>()) * weight1;
		vint ep1_b = vint(ep1.lane<2>()) * weight1;
		vint ep1_a = vint(ep1.lane<3>()) * weight1;

		// Compute EP0 contribution
		vint weight0 = vint(64) - weight1;
		vint ep0_r = vint(ep0.lane<0>()) * weight0;
		vint ep0_g = vint(ep0.lane<1>()) * weight0;
		vint ep0_b = vint(ep0.lane<2>()) * weight0;
		vint ep0_a = vint(ep0.lane<3>()) * weight0;

		// Combine contributions
		vint colori_r = asr<6>(ep0_r + ep1_r + vint(32));
		vint colori_g = asr<6>(ep0_g + ep1_g + vint(32));
		vint colori_b = asr<6>(ep0_b + ep1_b + vint(32));
		vint colori_a = asr<6>(ep0_a + ep1_a + vint(32));

		// If using a U8 decode mode bit replicate top 8 bits
		// so rest of codec can assume 0xFFFF max range everywhere
		vint colori_r8 = asr<8>(colori_r) * vint(257);
		colori_r = select(colori_r, colori_r8, vmask(u8_mask.lane<0>()));

		vint colori_g8 = asr<8>(colori_g) * vint(257);
		colori_g = select(colori_g, colori_g8, vmask(u8_mask.lane<1>()));

		vint colori_b8 = asr<8>(colori_b) * vint(257);
		colori_b = select(colori_b, colori_b8, vmask(u8_mask.lane<2>()));

		vint colori_a8 = asr<8>(colori_a) * vint(257);
		colori_a = select(colori_a, colori_a8, vmask(u8_mask.lane<3>()));

		// Compute color diff
		vfloat color_r = int_to_float(colori_r);
		vfloat color_g = int_to_float(colori_g);
		vfloat color_b = int_to_float(colori_b);
		vfloat color_a = int_to_float(colori_a);

		vfloat color_orig_r = loada(blk.data_r + i);
		vfloat color_orig_g = loada(blk.data_g + i);
		vfloat color_orig_b = loada(blk.data_b + i);
		vfloat color_orig_a = loada(blk.data_a + i);

		vfloat color_error_r = min(abs(color_orig_r - color_r), vfloat(1e15f));
		vfloat color_error_g = min(abs(color_orig_g - color_g), vfloat(1e15f));
		vfloat color_error_b = min(abs(color_orig_b - color_b), vfloat(1e15f));
		vfloat color_error_a = min(abs(color_orig_a - color_a), vfloat(1e15f));

		// Compute squared error metric
		color_error_r = color_error_r * color_error_r;
		color_error_g = color_error_g * color_error_g;
		color_error_b = color_error_b * color_error_b;
		color_error_a = color_error_a * color_error_a;

		vfloat metric = color_error_r * blk.channel_weight.lane<0>()
		              + color_error_g * blk.channel_weight.lane<1>()
		              + color_error_b * blk.channel_weight.lane<2>()
		              + color_error_a * blk.channel_weight.lane<3>();

		// Mask off bad lanes
		vmask mask = lane_id < vint(texel_count);
		lane_id += vint(ASTCENC_SIMD_WIDTH);
		haccumulate(summav, metric, mask);
	}

	return hadd_s(summav);
}

#endif
