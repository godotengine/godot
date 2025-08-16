// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2025 Arm Limited
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
 * @brief Functions for finding best endpoint format.
 *
 * We assume there are two independent sources of error in any given partition:
 *
 *   - Encoding choice errors
 *   - Quantization errors
 *
 * Encoding choice errors are caused by encoder decisions. For example:
 *
 *   - Using luminance instead of separate RGB components.
 *   - Using a constant 1.0 alpha instead of storing an alpha component.
 *   - Using RGB+scale instead of storing two full RGB endpoints.
 *
 * Quantization errors occur due to the limited precision we use for storage. These errors generally
 * scale with quantization level, but are not actually independent of color encoding. In particular:
 *
 *   - If we can use offset encoding then quantization error is halved.
 *   - If we can use blue-contraction then quantization error for RG is halved.
 *   - If we use HDR endpoints the quantization error is higher.
 *
 * Apart from these effects, we assume the error is proportional to the quantization step size.
 */


#include "astcenc_internal.h"
#include "astcenc_vecmathlib.h"

#include <assert.h>

/**
 * @brief Compute the errors of the endpoint line options for one partition.
 *
 * Uncorrelated data assumes storing completely independent RGBA channels for each endpoint. Same
 * chroma data assumes storing RGBA endpoints which pass though the origin (LDR only). RGBL data
 * assumes storing RGB + lumashift (HDR only). Luminance error assumes storing RGB channels as a
 * single value.
 *
 *
 * @param      pi                The partition info data.
 * @param      partition_index   The partition index to compule the error for.
 * @param      blk               The image block.
 * @param      uncor_pline       The endpoint line assuming uncorrelated endpoints.
 * @param[out] uncor_err         The computed error for the uncorrelated endpoint line.
 * @param      samec_pline       The endpoint line assuming the same chroma for both endpoints.
 * @param[out] samec_err         The computed error for the uncorrelated endpoint line.
 * @param      rgbl_pline        The endpoint line assuming RGB + lumashift data.
 * @param[out] rgbl_err          The computed error for the RGB + lumashift endpoint line.
 * @param      l_pline           The endpoint line assuming luminance data.
 * @param[out] l_err             The computed error for the luminance endpoint line.
 * @param[out] a_drop_err        The computed error for dropping the alpha component.
 */
static void compute_error_squared_rgb_single_partition(
	const partition_info& pi,
	int partition_index,
	const image_block& blk,
	const processed_line3& uncor_pline,
	float& uncor_err,
	const processed_line3& samec_pline,
	float& samec_err,
	const processed_line3& rgbl_pline,
	float& rgbl_err,
	const processed_line3& l_pline,
	float& l_err,
	float& a_drop_err
) {
	vfloat4 ews = blk.channel_weight;

	unsigned int texel_count = pi.partition_texel_count[partition_index];
	const uint8_t* texel_indexes = pi.texels_of_partition[partition_index];
	promise(texel_count > 0);

	vfloatacc a_drop_errv = vfloatacc::zero();
	vfloat default_a(blk.get_default_alpha());

	vfloatacc uncor_errv = vfloatacc::zero();
	vfloat uncor_bs0(uncor_pline.bs.lane<0>());
	vfloat uncor_bs1(uncor_pline.bs.lane<1>());
	vfloat uncor_bs2(uncor_pline.bs.lane<2>());

	vfloat uncor_amod0(uncor_pline.amod.lane<0>());
	vfloat uncor_amod1(uncor_pline.amod.lane<1>());
	vfloat uncor_amod2(uncor_pline.amod.lane<2>());

	vfloatacc samec_errv = vfloatacc::zero();
	vfloat samec_bs0(samec_pline.bs.lane<0>());
	vfloat samec_bs1(samec_pline.bs.lane<1>());
	vfloat samec_bs2(samec_pline.bs.lane<2>());

	vfloatacc rgbl_errv = vfloatacc::zero();
	vfloat rgbl_bs0(rgbl_pline.bs.lane<0>());
	vfloat rgbl_bs1(rgbl_pline.bs.lane<1>());
	vfloat rgbl_bs2(rgbl_pline.bs.lane<2>());

	vfloat rgbl_amod0(rgbl_pline.amod.lane<0>());
	vfloat rgbl_amod1(rgbl_pline.amod.lane<1>());
	vfloat rgbl_amod2(rgbl_pline.amod.lane<2>());

	vfloatacc l_errv = vfloatacc::zero();
	vfloat l_bs0(l_pline.bs.lane<0>());
	vfloat l_bs1(l_pline.bs.lane<1>());
	vfloat l_bs2(l_pline.bs.lane<2>());

	vint lane_ids = vint::lane_id();
	for (unsigned int i = 0; i < texel_count; i += ASTCENC_SIMD_WIDTH)
	{
		const uint8_t* tix = texel_indexes + i;

		vmask mask = lane_ids < vint(texel_count);
		lane_ids += vint(ASTCENC_SIMD_WIDTH);

		// Compute the error that arises from just ditching alpha
		vfloat data_a = gatherf_byte_inds<vfloat>(blk.data_a, tix);
		vfloat alpha_diff = data_a - default_a;
		alpha_diff = alpha_diff * alpha_diff;

		haccumulate(a_drop_errv, alpha_diff, mask);

		vfloat data_r = gatherf_byte_inds<vfloat>(blk.data_r, tix);
		vfloat data_g = gatherf_byte_inds<vfloat>(blk.data_g, tix);
		vfloat data_b = gatherf_byte_inds<vfloat>(blk.data_b, tix);

		// Compute uncorrelated error
		vfloat param = data_r * uncor_bs0
		             + data_g * uncor_bs1
		             + data_b * uncor_bs2;

		vfloat dist0 = (uncor_amod0 + param * uncor_bs0) - data_r;
		vfloat dist1 = (uncor_amod1 + param * uncor_bs1) - data_g;
		vfloat dist2 = (uncor_amod2 + param * uncor_bs2) - data_b;

		vfloat error = dist0 * dist0 * ews.lane<0>()
		             + dist1 * dist1 * ews.lane<1>()
		             + dist2 * dist2 * ews.lane<2>();

		haccumulate(uncor_errv, error, mask);

		// Compute same chroma error - no "amod", its always zero
		param = data_r * samec_bs0
		      + data_g * samec_bs1
		      + data_b * samec_bs2;

		dist0 = (param * samec_bs0) - data_r;
		dist1 = (param * samec_bs1) - data_g;
		dist2 = (param * samec_bs2) - data_b;

		error = dist0 * dist0 * ews.lane<0>()
		      + dist1 * dist1 * ews.lane<1>()
		      + dist2 * dist2 * ews.lane<2>();

		haccumulate(samec_errv, error, mask);

		// Compute rgbl error
		param = data_r * rgbl_bs0
		      + data_g * rgbl_bs1
		      + data_b * rgbl_bs2;

		dist0 = (rgbl_amod0 + param * rgbl_bs0) - data_r;
		dist1 = (rgbl_amod1 + param * rgbl_bs1) - data_g;
		dist2 = (rgbl_amod2 + param * rgbl_bs2) - data_b;

		error = dist0 * dist0 * ews.lane<0>()
		      + dist1 * dist1 * ews.lane<1>()
		      + dist2 * dist2 * ews.lane<2>();

		haccumulate(rgbl_errv, error, mask);

		// Compute luma error - no "amod", its always zero
		param = data_r * l_bs0
		      + data_g * l_bs1
		      + data_b * l_bs2;

		dist0 = (param * l_bs0) - data_r;
		dist1 = (param * l_bs1) - data_g;
		dist2 = (param * l_bs2) - data_b;

		error = dist0 * dist0 * ews.lane<0>()
		      + dist1 * dist1 * ews.lane<1>()
		      + dist2 * dist2 * ews.lane<2>();

		haccumulate(l_errv, error, mask);
	}

	a_drop_err = hadd_s(a_drop_errv) * ews.lane<3>();
	uncor_err = hadd_s(uncor_errv);
	samec_err = hadd_s(samec_errv);
	rgbl_err = hadd_s(rgbl_errv);
	l_err = hadd_s(l_errv);
}

/**
 * @brief For a given set of input colors and partitioning determine endpoint encode errors.
 *
 * This function determines the color error that results from RGB-scale encoding (LDR only),
 * RGB-lumashift encoding (HDR only), luminance-encoding, and alpha drop. Also determines whether
 * the endpoints are eligible for offset encoding or blue-contraction
 *
 * @param      blk   The image block.
 * @param      pi    The partition info data.
 * @param      ep    The idealized endpoints.
 * @param[out] eci   The resulting encoding choice error metrics.
  */
static void compute_encoding_choice_errors(
	const image_block& blk,
	const partition_info& pi,
	const endpoints& ep,
	encoding_choice_errors eci[BLOCK_MAX_PARTITIONS])
{
	int partition_count = pi.partition_count;
	promise(partition_count > 0);

	partition_metrics pms[BLOCK_MAX_PARTITIONS];

	compute_avgs_and_dirs_3_comp_rgb(pi, blk, pms);

	for (int i = 0; i < partition_count; i++)
	{
		partition_metrics& pm = pms[i];

		line3 uncor_rgb_lines;
		line3 samec_rgb_lines;  // for LDR-RGB-scale
		line3 rgb_luma_lines;   // for HDR-RGB-scale

		processed_line3 uncor_rgb_plines;
		processed_line3 samec_rgb_plines;
		processed_line3 rgb_luma_plines;
		processed_line3 luminance_plines;

		float uncorr_rgb_error;
		float samechroma_rgb_error;
		float rgb_luma_error;
		float luminance_rgb_error;
		float alpha_drop_error;

		uncor_rgb_lines.a = pm.avg;
		uncor_rgb_lines.b = normalize_safe(pm.dir, unit3());

		samec_rgb_lines.a = vfloat4::zero();
		samec_rgb_lines.b = normalize_safe(pm.avg, unit3());

		rgb_luma_lines.a = pm.avg;
		rgb_luma_lines.b = unit3();

		uncor_rgb_plines.amod = uncor_rgb_lines.a - uncor_rgb_lines.b * dot3(uncor_rgb_lines.a, uncor_rgb_lines.b);
		uncor_rgb_plines.bs   = uncor_rgb_lines.b;

		// Same chroma always goes though zero, so this is simpler than the others
		samec_rgb_plines.amod = vfloat4::zero();
		samec_rgb_plines.bs   = samec_rgb_lines.b;

		rgb_luma_plines.amod = rgb_luma_lines.a - rgb_luma_lines.b * dot3(rgb_luma_lines.a, rgb_luma_lines.b);
		rgb_luma_plines.bs   = rgb_luma_lines.b;

		// Luminance always goes though zero, so this is simpler than the others
		luminance_plines.amod = vfloat4::zero();
		luminance_plines.bs   = unit3();

		compute_error_squared_rgb_single_partition(
		    pi, i, blk,
		    uncor_rgb_plines, uncorr_rgb_error,
		    samec_rgb_plines, samechroma_rgb_error,
		    rgb_luma_plines,  rgb_luma_error,
		    luminance_plines, luminance_rgb_error,
		                      alpha_drop_error);

		// Determine if we can offset encode RGB lanes
		vfloat4 endpt0 = ep.endpt0[i];
		vfloat4 endpt1 = ep.endpt1[i];
		vfloat4 endpt_diff = abs(endpt1 - endpt0);
		vmask4 endpt_can_offset = endpt_diff < vfloat4(0.12f * 65535.0f);
		bool can_offset_encode = (mask(endpt_can_offset) & 0x7) == 0x7;

		// Store out the settings
		eci[i].rgb_scale_error = (samechroma_rgb_error - uncorr_rgb_error) * 0.7f;  // empirical
		eci[i].rgb_luma_error  = (rgb_luma_error - uncorr_rgb_error) * 1.5f;        // wild guess
		eci[i].luminance_error = (luminance_rgb_error - uncorr_rgb_error) * 3.0f;   // empirical
		eci[i].alpha_drop_error = alpha_drop_error * 3.0f;
		eci[i].can_offset_encode = can_offset_encode;
		eci[i].can_blue_contract = !blk.is_luminance();
	}
}

/**
 * @brief For a given partition compute the error for every endpoint integer count and quant level.
 *
 * @param      encode_hdr_rgb     @c true if using HDR for RGB, @c false for LDR.
 * @param      encode_hdr_alpha   @c true if using HDR for alpha, @c false for LDR.
 * @param      partition_index    The partition index.
 * @param      pi                 The partition info.
 * @param      eci                The encoding choice error metrics.
 * @param      ep                 The idealized endpoints.
 * @param      error_weight       The resulting encoding choice error metrics.
 * @param[out] best_error         The best error for each integer count and quant level.
 * @param[out] format_of_choice   The preferred endpoint format for each integer count and quant level.
 */
static void compute_color_error_for_every_integer_count_and_quant_level(
	bool encode_hdr_rgb,
	bool encode_hdr_alpha,
	int partition_index,
	const partition_info& pi,
	const encoding_choice_errors& eci,
	const endpoints& ep,
	vfloat4 error_weight,
	float best_error[21][4],
	uint8_t format_of_choice[21][4]
) {
	int partition_size = pi.partition_texel_count[partition_index];

	static const float baseline_quant_error[21 - QUANT_6] {
		(65536.0f * 65536.0f / 18.0f) / (5 * 5),
		(65536.0f * 65536.0f / 18.0f) / (7 * 7),
		(65536.0f * 65536.0f / 18.0f) / (9 * 9),
		(65536.0f * 65536.0f / 18.0f) / (11 * 11),
		(65536.0f * 65536.0f / 18.0f) / (15 * 15),
		(65536.0f * 65536.0f / 18.0f) / (19 * 19),
		(65536.0f * 65536.0f / 18.0f) / (23 * 23),
		(65536.0f * 65536.0f / 18.0f) / (31 * 31),
		(65536.0f * 65536.0f / 18.0f) / (39 * 39),
		(65536.0f * 65536.0f / 18.0f) / (47 * 47),
		(65536.0f * 65536.0f / 18.0f) / (63 * 63),
		(65536.0f * 65536.0f / 18.0f) / (79 * 79),
		(65536.0f * 65536.0f / 18.0f) / (95 * 95),
		(65536.0f * 65536.0f / 18.0f) / (127 * 127),
		(65536.0f * 65536.0f / 18.0f) / (159 * 159),
		(65536.0f * 65536.0f / 18.0f) / (191 * 191),
		(65536.0f * 65536.0f / 18.0f) / (255 * 255)
	};

	vfloat4 ep0 = ep.endpt0[partition_index];
	vfloat4 ep1 = ep.endpt1[partition_index];

	float ep1_min = hmin_rgb_s(ep1);
	ep1_min = astc::max(ep1_min, 0.0f);

	float error_weight_rgbsum = hadd_rgb_s(error_weight);

	float range_upper_limit_rgb = encode_hdr_rgb ? 61440.0f : 65535.0f;
	float range_upper_limit_alpha = encode_hdr_alpha ? 61440.0f : 65535.0f;

	// It is possible to get endpoint colors significantly outside [0,upper-limit] even if the
	// input data are safely contained in [0,upper-limit]; we need to add an error term for this
	vfloat4 offset(range_upper_limit_rgb, range_upper_limit_rgb, range_upper_limit_rgb, range_upper_limit_alpha);
	vfloat4 ep0_range_error_high = max(ep0 - offset, 0.0f);
	vfloat4 ep1_range_error_high = max(ep1 - offset, 0.0f);

	vfloat4 ep0_range_error_low = min(ep0, 0.0f);
	vfloat4 ep1_range_error_low = min(ep1, 0.0f);

	vfloat4 sum_range_error =
		(ep0_range_error_low * ep0_range_error_low) +
		(ep1_range_error_low * ep1_range_error_low) +
		(ep0_range_error_high * ep0_range_error_high) +
		(ep1_range_error_high * ep1_range_error_high);

	float rgb_range_error = dot3_s(sum_range_error, error_weight)
	                      * 0.5f * static_cast<float>(partition_size);
	float alpha_range_error = sum_range_error.lane<3>() * error_weight.lane<3>()
	                        * 0.5f * static_cast<float>(partition_size);

	if (encode_hdr_rgb)
	{

		// Collect some statistics
		float af, cf;
		if (ep1.lane<0>() > ep1.lane<1>() && ep1.lane<0>() > ep1.lane<2>())
		{
			af = ep1.lane<0>();
			cf = ep1.lane<0>() - ep0.lane<0>();
		}
		else if (ep1.lane<1>() > ep1.lane<2>())
		{
			af = ep1.lane<1>();
			cf = ep1.lane<1>() - ep0.lane<1>();
		}
		else
		{
			af = ep1.lane<2>();
			cf = ep1.lane<2>() - ep0.lane<2>();
		}

		// Estimate of color-component spread in high endpoint color
		float bf = af - ep1_min;
		vfloat4 prd = (ep1 - vfloat4(cf)).swz<0, 1, 2>();
		vfloat4 pdif = prd - ep0.swz<0, 1, 2>();
		// Estimate of color-component spread in low endpoint color
		float df = hmax_s(abs(pdif));

		int b = static_cast<int>(bf);
		int c = static_cast<int>(cf);
		int d = static_cast<int>(df);

		// Determine which one of the 6 submodes is likely to be used in case of an RGBO-mode
		int rgbo_mode = 5;		// 7 bits per component
		// mode 4: 8 7 6
		if (b < 32768 && c < 16384)
		{
			rgbo_mode = 4;
		}

		// mode 3: 9 6 7
		if (b < 8192 && c < 16384)
		{
			rgbo_mode = 3;
		}

		// mode 2: 10 5 8
		if (b < 2048 && c < 16384)
		{
			rgbo_mode = 2;
		}

		// mode 1: 11 6 5
		if (b < 2048 && c < 1024)
		{
			rgbo_mode = 1;
		}

		// mode 0: 11 5 7
		if (b < 1024 && c < 4096)
		{
			rgbo_mode = 0;
		}

		// Determine which one of the 9 submodes is likely to be used in case of an RGB-mode.
		int rgb_mode = 8;		// 8 bits per component, except 7 bits for blue

		// mode 0: 9 7 6 7
		if (b < 16384 && c < 8192 && d < 8192)
		{
			rgb_mode = 0;
		}

		// mode 1: 9 8 6 6
		if (b < 32768 && c < 8192 && d < 4096)
		{
			rgb_mode = 1;
		}

		// mode 2: 10 6 7 7
		if (b < 4096 && c < 8192 && d < 4096)
		{
			rgb_mode = 2;
		}

		// mode 3: 10 7 7 6
		if (b < 8192 && c < 8192 && d < 2048)
		{
			rgb_mode = 3;
		}

		// mode 4: 11 8 6 5
		if (b < 8192 && c < 2048 && d < 512)
		{
			rgb_mode = 4;
		}

		// mode 5: 11 6 8 6
		if (b < 2048 && c < 8192 && d < 1024)
		{
			rgb_mode = 5;
		}

		// mode 6: 12 7 7 5
		if (b < 2048 && c < 2048 && d < 256)
		{
			rgb_mode = 6;
		}

		// mode 7: 12 6 7 6
		if (b < 1024 && c < 2048 && d < 512)
		{
			rgb_mode = 7;
		}

		static const float rgbo_error_scales[6] { 4.0f, 4.0f, 16.0f, 64.0f, 256.0f, 1024.0f };
		static const float rgb_error_scales[9] { 64.0f, 64.0f, 16.0f, 16.0f, 4.0f, 4.0f, 1.0f, 1.0f, 384.0f };

		float mode7mult = rgbo_error_scales[rgbo_mode] * 0.0015f;  // Empirically determined ....
		float mode11mult = rgb_error_scales[rgb_mode] * 0.010f;    // Empirically determined ....


		float lum_high = hadd_rgb_s(ep1) * (1.0f / 3.0f);
		float lum_low = hadd_rgb_s(ep0) * (1.0f / 3.0f);
		float lumdif = lum_high - lum_low;
		float mode23mult = lumdif < 960 ? 4.0f : lumdif < 3968 ? 16.0f : 128.0f;

		mode23mult *= 0.0005f;  // Empirically determined ....

		// Pick among the available HDR endpoint modes
		for (int i = QUANT_2; i < QUANT_16; i++)
		{
			best_error[i][3] = ERROR_CALC_DEFAULT;
			best_error[i][2] = ERROR_CALC_DEFAULT;
			best_error[i][1] = ERROR_CALC_DEFAULT;
			best_error[i][0] = ERROR_CALC_DEFAULT;

			format_of_choice[i][3] = static_cast<uint8_t>(encode_hdr_alpha ? FMT_HDR_RGBA : FMT_HDR_RGB_LDR_ALPHA);
			format_of_choice[i][2] = FMT_HDR_RGB;
			format_of_choice[i][1] = FMT_HDR_RGB_SCALE;
			format_of_choice[i][0] = FMT_HDR_LUMINANCE_LARGE_RANGE;
		}

		for (int i = QUANT_16; i <= QUANT_256; i++)
		{
			// The base_quant_error should depend on the scale-factor that would be used during
			// actual encode of the color value

			float base_quant_error = baseline_quant_error[i - QUANT_6] * static_cast<float>(partition_size);
			float rgb_quantization_error = error_weight_rgbsum * base_quant_error * 2.0f;
			float alpha_quantization_error = error_weight.lane<3>() * base_quant_error * 2.0f;
			float rgba_quantization_error = rgb_quantization_error + alpha_quantization_error;

			// For 8 integers, we have two encodings: one with HDR A and another one with LDR A

			float full_hdr_rgba_error = rgba_quantization_error + rgb_range_error + alpha_range_error;
			best_error[i][3] = full_hdr_rgba_error;
			format_of_choice[i][3] = static_cast<uint8_t>(encode_hdr_alpha ? FMT_HDR_RGBA : FMT_HDR_RGB_LDR_ALPHA);

			// For 6 integers, we have one HDR-RGB encoding
			float full_hdr_rgb_error = (rgb_quantization_error * mode11mult) + rgb_range_error + eci.alpha_drop_error;
			best_error[i][2] = full_hdr_rgb_error;
			format_of_choice[i][2] = FMT_HDR_RGB;

			// For 4 integers, we have one HDR-RGB-Scale encoding
			float hdr_rgb_scale_error = (rgb_quantization_error * mode7mult) + rgb_range_error + eci.alpha_drop_error + eci.rgb_luma_error;

			best_error[i][1] = hdr_rgb_scale_error;
			format_of_choice[i][1] = FMT_HDR_RGB_SCALE;

			// For 2 integers, we assume luminance-with-large-range
			float hdr_luminance_error = (rgb_quantization_error * mode23mult) + rgb_range_error + eci.alpha_drop_error + eci.luminance_error;
			best_error[i][0] = hdr_luminance_error;
			format_of_choice[i][0] = FMT_HDR_LUMINANCE_LARGE_RANGE;
		}
	}
	else
	{
		for (int i = QUANT_2; i < QUANT_6; i++)
		{
			best_error[i][3] = ERROR_CALC_DEFAULT;
			best_error[i][2] = ERROR_CALC_DEFAULT;
			best_error[i][1] = ERROR_CALC_DEFAULT;
			best_error[i][0] = ERROR_CALC_DEFAULT;

			format_of_choice[i][3] = FMT_RGBA;
			format_of_choice[i][2] = FMT_RGB;
			format_of_choice[i][1] = FMT_RGB_SCALE;
			format_of_choice[i][0] = FMT_LUMINANCE;
		}

		float base_quant_error_rgb = error_weight_rgbsum * static_cast<float>(partition_size);
		float base_quant_error_a = error_weight.lane<3>() * static_cast<float>(partition_size);
		float base_quant_error_rgba = base_quant_error_rgb + base_quant_error_a;

		float error_scale_bc_rgba = eci.can_blue_contract ? 0.625f : 1.0f;
		float error_scale_oe_rgba = eci.can_offset_encode ? 0.5f : 1.0f;

		float error_scale_bc_rgb = eci.can_blue_contract ? 0.5f : 1.0f;
		float error_scale_oe_rgb = eci.can_offset_encode ? 0.25f : 1.0f;

		// Pick among the available LDR endpoint modes
		for (int i = QUANT_6; i <= QUANT_256; i++)
		{
			// Offset encoding not possible at higher quant levels
			if (i >= QUANT_192)
			{
				error_scale_oe_rgba = 1.0f;
				error_scale_oe_rgb = 1.0f;
			}

			float base_quant_error = baseline_quant_error[i - QUANT_6];
			float quant_error_rgb  = base_quant_error_rgb * base_quant_error;
			float quant_error_rgba = base_quant_error_rgba * base_quant_error;

			// 8 integers can encode as RGBA+RGBA
			float full_ldr_rgba_error = quant_error_rgba
			                          * error_scale_bc_rgba
			                          * error_scale_oe_rgba
			                          + rgb_range_error
			                          + alpha_range_error;

			best_error[i][3] = full_ldr_rgba_error;
			format_of_choice[i][3] = FMT_RGBA;

			// 6 integers can encode as RGB+RGB or RGBS+AA
			float full_ldr_rgb_error = quant_error_rgb
			                         * error_scale_bc_rgb
			                         * error_scale_oe_rgb
			                         + rgb_range_error
			                         + eci.alpha_drop_error;

			float rgbs_alpha_error = quant_error_rgba
			                       + eci.rgb_scale_error
			                       + rgb_range_error
			                       + alpha_range_error;

			if (rgbs_alpha_error < full_ldr_rgb_error)
			{
				best_error[i][2] = rgbs_alpha_error;
				format_of_choice[i][2] = FMT_RGB_SCALE_ALPHA;
			}
			else
			{
				best_error[i][2] = full_ldr_rgb_error;
				format_of_choice[i][2] = FMT_RGB;
			}

			// 4 integers can encode as RGBS or LA+LA
			float ldr_rgbs_error = quant_error_rgb
			                     + rgb_range_error
			                     + eci.alpha_drop_error
			                     + eci.rgb_scale_error;

			float lum_alpha_error = quant_error_rgba
			                      + rgb_range_error
			                      + alpha_range_error
			                      + eci.luminance_error;

			if (ldr_rgbs_error < lum_alpha_error)
			{
				best_error[i][1] = ldr_rgbs_error;
				format_of_choice[i][1] = FMT_RGB_SCALE;
			}
			else
			{
				best_error[i][1] = lum_alpha_error;
				format_of_choice[i][1] = FMT_LUMINANCE_ALPHA;
			}

			// 2 integers can encode as L+L
			float luminance_error = quant_error_rgb
			                      + rgb_range_error
			                      + eci.alpha_drop_error
			                      + eci.luminance_error;

			best_error[i][0] = luminance_error;
			format_of_choice[i][0] = FMT_LUMINANCE;
		}
	}
}

/**
 * @brief For one partition compute the best format and quantization for a given bit count.
 *
 * @param      best_combined_error    The best error for each quant level and integer count.
 * @param      best_combined_format   The best format for each quant level and integer count.
 * @param      bits_available         The number of bits available for encoding.
 * @param[out] best_quant_level       The output best color quant level.
 * @param[out] best_format            The output best color format.
 *
 * @return The output error for the best pairing.
 */
static float one_partition_find_best_combination_for_bitcount(
	const float best_combined_error[21][4],
	const uint8_t best_combined_format[21][4],
	int bits_available,
	uint8_t& best_quant_level,
	uint8_t& best_format
) {
	int best_integer_count = 0;
	float best_integer_count_error = ERROR_CALC_DEFAULT;

	for (int integer_count = 1; integer_count <= 4;  integer_count++)
	{
		// Compute the quantization level for a given number of integers and a given number of bits
		int quant_level = quant_mode_table[integer_count][bits_available];

		// Don't have enough bits to represent a given endpoint format at all!
		if (quant_level < QUANT_6)
		{
			continue;
		}

		float integer_count_error = best_combined_error[quant_level][integer_count - 1];
		if (integer_count_error < best_integer_count_error)
		{
			best_integer_count_error = integer_count_error;
			best_integer_count = integer_count - 1;
		}
	}

	int ql = quant_mode_table[best_integer_count + 1][bits_available];

	best_quant_level = static_cast<uint8_t>(ql);
	best_format = FMT_LUMINANCE;

	if (ql >= QUANT_6)
	{
		best_format = best_combined_format[ql][best_integer_count];
	}

	return best_integer_count_error;
}

/**
 * @brief For 2 partitions compute the best format combinations for every pair of quant mode and integer count.
 *
 * @param      best_error             The best error for a single endpoint quant level and integer count.
 * @param      best_format            The best format for a single endpoint quant level and integer count.
 * @param[out] best_combined_error    The best combined error pairings for the 2 partitions.
 * @param[out] best_combined_format   The best combined format pairings for the 2 partitions.
 */
static void two_partitions_find_best_combination_for_every_quantization_and_integer_count(
	const float best_error[2][21][4],	// indexed by (partition, quant-level, integer-pair-count-minus-1)
	const uint8_t best_format[2][21][4],
	float best_combined_error[21][7],	// indexed by (quant-level, integer-pair-count-minus-2)
	uint8_t best_combined_format[21][7][2]
) {
	for (int i = QUANT_2; i <= QUANT_256; i++)
	{
		for (int j = 0; j < 7; j++)
		{
			best_combined_error[i][j] = ERROR_CALC_DEFAULT;
		}
	}

	for (int quant = QUANT_6; quant <= QUANT_256; quant++)
	{
		for (int i = 0; i < 4; i++)	// integer-count for first endpoint-pair
		{
			for (int j = 0; j < 4; j++)	// integer-count for second endpoint-pair
			{
				int low2 = astc::min(i, j);
				int high2 = astc::max(i, j);
				if ((high2 - low2) > 1)
				{
					continue;
				}

				int intcnt = i + j;
				float errorterm = astc::min(best_error[0][quant][i] + best_error[1][quant][j], 1e10f);
				if (errorterm <= best_combined_error[quant][intcnt])
				{
					best_combined_error[quant][intcnt] = errorterm;
					best_combined_format[quant][intcnt][0] = best_format[0][quant][i];
					best_combined_format[quant][intcnt][1] = best_format[1][quant][j];
				}
			}
		}
	}
}

/**
 * @brief For 2 partitions compute the best format and quantization for a given bit count.
 *
 * @param      best_combined_error    The best error for each quant level and integer count.
 * @param      best_combined_format   The best format for each quant level and integer count.
 * @param      bits_available         The number of bits available for encoding.
 * @param[out] best_quant_level       The output best color quant level.
 * @param[out] best_quant_level_mod   The output best color quant level assuming two more bits are available.
 * @param[out] best_formats           The output best color formats.
 *
 * @return The output error for the best pairing.
 */
static float two_partitions_find_best_combination_for_bitcount(
	float best_combined_error[21][7],
	uint8_t best_combined_format[21][7][2],
	int bits_available,
	uint8_t& best_quant_level,
	uint8_t& best_quant_level_mod,
	uint8_t* best_formats
) {
	int best_integer_count = 0;
	float best_integer_count_error = ERROR_CALC_DEFAULT;

	for (int integer_count = 2; integer_count <= 8; integer_count++)
	{
		// Compute the quantization level for a given number of integers and a given number of bits
		int quant_level = quant_mode_table[integer_count][bits_available];

		// Don't have enough bits to represent a given endpoint format at all!
		if (quant_level < QUANT_6)
		{
			break;
		}

		float integer_count_error = best_combined_error[quant_level][integer_count - 2];
		if (integer_count_error < best_integer_count_error)
		{
			best_integer_count_error = integer_count_error;
			best_integer_count = integer_count;
		}
	}

	int ql = quant_mode_table[best_integer_count][bits_available];
	int ql_mod = quant_mode_table[best_integer_count][bits_available + 2];

	best_quant_level = static_cast<uint8_t>(ql);
	best_quant_level_mod = static_cast<uint8_t>(ql_mod);

	if (ql >= QUANT_6)
	{
		for (int i = 0; i < 2; i++)
		{
			best_formats[i] = best_combined_format[ql][best_integer_count - 2][i];
		}
	}
	else
	{
		for (int i = 0; i < 2; i++)
		{
			best_formats[i] = FMT_LUMINANCE;
		}
	}

	return best_integer_count_error;
}

/**
 * @brief For 3 partitions compute the best format combinations for every pair of quant mode and integer count.
 *
 * @param      best_error             The best error for a single endpoint quant level and integer count.
 * @param      best_format            The best format for a single endpoint quant level and integer count.
 * @param[out] best_combined_error    The best combined error pairings for the 3 partitions.
 * @param[out] best_combined_format   The best combined format pairings for the 3 partitions.
 */
static void three_partitions_find_best_combination_for_every_quantization_and_integer_count(
	const float best_error[3][21][4],	// indexed by (partition, quant-level, integer-count)
	const uint8_t best_format[3][21][4],
	float best_combined_error[21][10],
	uint8_t best_combined_format[21][10][3]
) {
	for (int i = QUANT_2; i <= QUANT_256; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			best_combined_error[i][j] = ERROR_CALC_DEFAULT;
		}
	}

	for (int quant = QUANT_6; quant <= QUANT_256; quant++)
	{
		for (int i = 0; i < 4; i++)	// integer-count for first endpoint-pair
		{
			for (int j = 0; j < 4; j++)	// integer-count for second endpoint-pair
			{
				int low2 = astc::min(i, j);
				int high2 = astc::max(i, j);
				if ((high2 - low2) > 1)
				{
					continue;
				}

				for (int k = 0; k < 4; k++)	// integer-count for third endpoint-pair
				{
					int low3 = astc::min(k, low2);
					int high3 = astc::max(k, high2);
					if ((high3 - low3) > 1)
					{
						continue;
					}

					int intcnt = i + j + k;
					float errorterm = astc::min(best_error[0][quant][i] + best_error[1][quant][j] + best_error[2][quant][k], 1e10f);
					if (errorterm <= best_combined_error[quant][intcnt])
					{
						best_combined_error[quant][intcnt] = errorterm;
						best_combined_format[quant][intcnt][0] = best_format[0][quant][i];
						best_combined_format[quant][intcnt][1] = best_format[1][quant][j];
						best_combined_format[quant][intcnt][2] = best_format[2][quant][k];
					}
				}
			}
		}
	}
}

/**
 * @brief For 3 partitions compute the best format and quantization for a given bit count.
 *
 * @param      best_combined_error    The best error for each quant level and integer count.
 * @param      best_combined_format   The best format for each quant level and integer count.
 * @param      bits_available         The number of bits available for encoding.
 * @param[out] best_quant_level       The output best color quant level.
 * @param[out] best_quant_level_mod   The output best color quant level assuming two more bits are available.
 * @param[out] best_formats           The output best color formats.
 *
 * @return The output error for the best pairing.
 */
static float three_partitions_find_best_combination_for_bitcount(
	const float best_combined_error[21][10],
	const uint8_t best_combined_format[21][10][3],
	int bits_available,
	uint8_t& best_quant_level,
	uint8_t& best_quant_level_mod,
	uint8_t* best_formats
) {
	int best_integer_count = 0;
	float best_integer_count_error = ERROR_CALC_DEFAULT;

	for (int integer_count = 3; integer_count <= 9; integer_count++)
	{
		// Compute the quantization level for a given number of integers and a given number of bits
		int quant_level = quant_mode_table[integer_count][bits_available];

		// Don't have enough bits to represent a given endpoint format at all!
		if (quant_level < QUANT_6)
		{
			break;
		}

		float integer_count_error = best_combined_error[quant_level][integer_count - 3];
		if (integer_count_error < best_integer_count_error)
		{
			best_integer_count_error = integer_count_error;
			best_integer_count = integer_count;
		}
	}

	int ql = quant_mode_table[best_integer_count][bits_available];
	int ql_mod = quant_mode_table[best_integer_count][bits_available + 5];

	best_quant_level = static_cast<uint8_t>(ql);
	best_quant_level_mod = static_cast<uint8_t>(ql_mod);

	if (ql >= QUANT_6)
	{
		for (int i = 0; i < 3; i++)
		{
			best_formats[i] = best_combined_format[ql][best_integer_count - 3][i];
		}
	}
	else
	{
		for (int i = 0; i < 3; i++)
		{
			best_formats[i] = FMT_LUMINANCE;
		}
	}

	return best_integer_count_error;
}

/**
 * @brief For 4 partitions compute the best format combinations for every pair of quant mode and integer count.
 *
 * @param      best_error             The best error for a single endpoint quant level and integer count.
 * @param      best_format            The best format for a single endpoint quant level and integer count.
 * @param[out] best_combined_error    The best combined error pairings for the 4 partitions.
 * @param[out] best_combined_format   The best combined format pairings for the 4 partitions.
 */
static void four_partitions_find_best_combination_for_every_quantization_and_integer_count(
	const float best_error[4][21][4],	// indexed by (partition, quant-level, integer-count)
	const uint8_t best_format[4][21][4],
	float best_combined_error[21][13],
	uint8_t best_combined_format[21][13][4]
) {
	for (int i = QUANT_2; i <= QUANT_256; i++)
	{
		for (int j = 0; j < 13; j++)
		{
			best_combined_error[i][j] = ERROR_CALC_DEFAULT;
		}
	}

	for (int quant = QUANT_6; quant <= QUANT_256; quant++)
	{
		for (int i = 0; i < 4; i++)	// integer-count for first endpoint-pair
		{
			for (int j = 0; j < 4; j++)	// integer-count for second endpoint-pair
			{
				int low2 = astc::min(i, j);
				int high2 = astc::max(i, j);
				if ((high2 - low2) > 1)
				{
					continue;
				}

				for (int k = 0; k < 4; k++)	// integer-count for third endpoint-pair
				{
					int low3 = astc::min(k, low2);
					int high3 = astc::max(k, high2);
					if ((high3 - low3) > 1)
					{
						continue;
					}

					for (int l = 0; l < 4; l++)	// integer-count for fourth endpoint-pair
					{
						int low4 = astc::min(l, low3);
						int high4 = astc::max(l, high3);
						if ((high4 - low4) > 1)
						{
							continue;
						}

						int intcnt = i + j + k + l;
						float errorterm = astc::min(best_error[0][quant][i] + best_error[1][quant][j] + best_error[2][quant][k] + best_error[3][quant][l], 1e10f);
						if (errorterm <= best_combined_error[quant][intcnt])
						{
							best_combined_error[quant][intcnt] = errorterm;
							best_combined_format[quant][intcnt][0] = best_format[0][quant][i];
							best_combined_format[quant][intcnt][1] = best_format[1][quant][j];
							best_combined_format[quant][intcnt][2] = best_format[2][quant][k];
							best_combined_format[quant][intcnt][3] = best_format[3][quant][l];
						}
					}
				}
			}
		}
	}
}

/**
 * @brief For 4 partitions compute the best format and quantization for a given bit count.
 *
 * @param      best_combined_error    The best error for each quant level and integer count.
 * @param      best_combined_format   The best format for each quant level and integer count.
 * @param      bits_available         The number of bits available for encoding.
 * @param[out] best_quant_level       The output best color quant level.
 * @param[out] best_quant_level_mod   The output best color quant level assuming two more bits are available.
 * @param[out] best_formats           The output best color formats.
 *
 * @return best_error The output error for the best pairing.
 */
static float four_partitions_find_best_combination_for_bitcount(
	const float best_combined_error[21][13],
	const uint8_t best_combined_format[21][13][4],
	int bits_available,
	uint8_t& best_quant_level,
	uint8_t& best_quant_level_mod,
	uint8_t* best_formats
) {
	int best_integer_count = 0;
	float best_integer_count_error = ERROR_CALC_DEFAULT;

	for (int integer_count = 4; integer_count <= 9; integer_count++)
	{
		// Compute the quantization level for a given number of integers and a given number of bits
		int quant_level = quant_mode_table[integer_count][bits_available];

		// Don't have enough bits to represent a given endpoint format at all!
		if (quant_level < QUANT_6)
		{
			break;
		}

		float integer_count_error = best_combined_error[quant_level][integer_count - 4];
		if (integer_count_error < best_integer_count_error)
		{
			best_integer_count_error = integer_count_error;
			best_integer_count = integer_count;
		}
	}

	int ql = quant_mode_table[best_integer_count][bits_available];
	int ql_mod = quant_mode_table[best_integer_count][bits_available + 8];

	best_quant_level = static_cast<uint8_t>(ql);
	best_quant_level_mod = static_cast<uint8_t>(ql_mod);

	if (ql >= QUANT_6)
	{
		for (int i = 0; i < 4; i++)
		{
			best_formats[i] = best_combined_format[ql][best_integer_count - 4][i];
		}
	}
	else
	{
		for (int i = 0; i < 4; i++)
		{
			best_formats[i] = FMT_LUMINANCE;
		}
	}

	return best_integer_count_error;
}

/* See header for documentation. */
unsigned int compute_ideal_endpoint_formats(
	const partition_info& pi,
	const image_block& blk,
	const endpoints& ep,
	 // bitcounts and errors computed for the various quantization methods
	const int8_t* qwt_bitcounts,
	const float* qwt_errors,
	unsigned int tune_candidate_limit,
	unsigned int start_block_mode,
	unsigned int end_block_mode,
	// output data
	uint8_t partition_format_specifiers[TUNE_MAX_TRIAL_CANDIDATES][BLOCK_MAX_PARTITIONS],
	int block_mode[TUNE_MAX_TRIAL_CANDIDATES],
	quant_method quant_level[TUNE_MAX_TRIAL_CANDIDATES],
	quant_method quant_level_mod[TUNE_MAX_TRIAL_CANDIDATES],
	compression_working_buffers& tmpbuf
) {
	int partition_count = pi.partition_count;

	promise(partition_count > 0);

	bool encode_hdr_rgb = static_cast<bool>(blk.rgb_lns[0]);
	bool encode_hdr_alpha = static_cast<bool>(blk.alpha_lns[0]);

	// Compute the errors that result from various encoding choices (such as using luminance instead
	// of RGB, discarding Alpha, using RGB-scale in place of two separate RGB endpoints and so on)
	encoding_choice_errors eci[BLOCK_MAX_PARTITIONS];
	compute_encoding_choice_errors(blk, pi, ep, eci);

	float best_error[BLOCK_MAX_PARTITIONS][21][4];
	uint8_t format_of_choice[BLOCK_MAX_PARTITIONS][21][4];
	for (int i = 0; i < partition_count; i++)
	{
		compute_color_error_for_every_integer_count_and_quant_level(
		    encode_hdr_rgb, encode_hdr_alpha, i,
		    pi, eci[i], ep, blk.channel_weight, best_error[i],
		    format_of_choice[i]);
	}

	float* errors_of_best_combination = tmpbuf.errors_of_best_combination;
	uint8_t* best_quant_levels = tmpbuf.best_quant_levels;
	uint8_t* best_quant_levels_mod = tmpbuf.best_quant_levels_mod;
	uint8_t (&best_ep_formats)[WEIGHTS_MAX_BLOCK_MODES][BLOCK_MAX_PARTITIONS] = tmpbuf.best_ep_formats;

	// Ensure that the first iteration understep contains data that will never be picked
	vfloat clear_error(ERROR_CALC_DEFAULT);
	vint clear_quant(0);

	size_t packed_start_block_mode = round_down_to_simd_multiple_vla(start_block_mode);
	storea(clear_error, errors_of_best_combination + packed_start_block_mode);
	store_nbytes(clear_quant, best_quant_levels + packed_start_block_mode);
	store_nbytes(clear_quant, best_quant_levels_mod + packed_start_block_mode);

	// Ensure that last iteration overstep contains data that will never be picked
	size_t packed_end_block_mode = round_down_to_simd_multiple_vla(end_block_mode - 1);
	storea(clear_error, errors_of_best_combination + packed_end_block_mode);
	store_nbytes(clear_quant, best_quant_levels + packed_end_block_mode);
	store_nbytes(clear_quant, best_quant_levels_mod + packed_end_block_mode);

	// Track a scalar best to avoid expensive search at least once ...
	float error_of_best_combination = ERROR_CALC_DEFAULT;
	int index_of_best_combination = -1;

	// The block contains 1 partition
	if (partition_count == 1)
	{
		for (unsigned int i = start_block_mode; i < end_block_mode; i++)
		{
			if (qwt_errors[i] >= ERROR_CALC_DEFAULT)
			{
				errors_of_best_combination[i] = ERROR_CALC_DEFAULT;
				continue;
			}

			float error_of_best = one_partition_find_best_combination_for_bitcount(
			    best_error[0], format_of_choice[0], qwt_bitcounts[i],
			    best_quant_levels[i], best_ep_formats[i][0]);

			float total_error = error_of_best + qwt_errors[i];
			errors_of_best_combination[i] = total_error;
			best_quant_levels_mod[i] = best_quant_levels[i];

			if (total_error < error_of_best_combination)
			{
				error_of_best_combination = total_error;
				index_of_best_combination = i;
			}
		}
	}
	// The block contains 2 partitions
	else if (partition_count == 2)
	{
		float combined_best_error[21][7];
		uint8_t formats_of_choice[21][7][2];

		two_partitions_find_best_combination_for_every_quantization_and_integer_count(
		    best_error, format_of_choice, combined_best_error, formats_of_choice);

		assert(start_block_mode == 0);
		for (unsigned int i = 0; i < end_block_mode; i++)
		{
			if (qwt_errors[i] >= ERROR_CALC_DEFAULT)
			{
				errors_of_best_combination[i] = ERROR_CALC_DEFAULT;
				continue;
			}

			float error_of_best = two_partitions_find_best_combination_for_bitcount(
			    combined_best_error, formats_of_choice, qwt_bitcounts[i],
			    best_quant_levels[i], best_quant_levels_mod[i],
			    best_ep_formats[i]);

			float total_error = error_of_best + qwt_errors[i];
			errors_of_best_combination[i] = total_error;

			if (total_error < error_of_best_combination)
			{
				error_of_best_combination = total_error;
				index_of_best_combination = i;
			}
		}
	}
	// The block contains 3 partitions
	else if (partition_count == 3)
	{
		float combined_best_error[21][10];
		uint8_t formats_of_choice[21][10][3];

		three_partitions_find_best_combination_for_every_quantization_and_integer_count(
		    best_error, format_of_choice, combined_best_error, formats_of_choice);

		assert(start_block_mode == 0);
		for (unsigned int i = 0; i < end_block_mode; i++)
		{
			if (qwt_errors[i] >= ERROR_CALC_DEFAULT)
			{
				errors_of_best_combination[i] = ERROR_CALC_DEFAULT;
				continue;
			}

			float error_of_best = three_partitions_find_best_combination_for_bitcount(
			    combined_best_error, formats_of_choice, qwt_bitcounts[i],
			    best_quant_levels[i], best_quant_levels_mod[i],
			    best_ep_formats[i]);

			float total_error = error_of_best + qwt_errors[i];
			errors_of_best_combination[i] = total_error;

			if (total_error < error_of_best_combination)
			{
				error_of_best_combination = total_error;
				index_of_best_combination = i;
			}
		}
	}
	// The block contains 4 partitions
	else // if (partition_count == 4)
	{
		assert(partition_count == 4);
		float combined_best_error[21][13];
		uint8_t formats_of_choice[21][13][4];

		four_partitions_find_best_combination_for_every_quantization_and_integer_count(
		    best_error, format_of_choice, combined_best_error, formats_of_choice);

		assert(start_block_mode == 0);
		for (unsigned int i = 0; i < end_block_mode; i++)
		{
			if (qwt_errors[i] >= ERROR_CALC_DEFAULT)
			{
				errors_of_best_combination[i] = ERROR_CALC_DEFAULT;
				continue;
			}

			float error_of_best = four_partitions_find_best_combination_for_bitcount(
			    combined_best_error, formats_of_choice, qwt_bitcounts[i],
			    best_quant_levels[i], best_quant_levels_mod[i],
			    best_ep_formats[i]);

			float total_error = error_of_best + qwt_errors[i];
			errors_of_best_combination[i] = total_error;

			if (total_error < error_of_best_combination)
			{
				error_of_best_combination = total_error;
				index_of_best_combination = i;
			}
		}
	}

	int best_error_weights[TUNE_MAX_TRIAL_CANDIDATES];

	// Fast path the first result and avoid the list search for trial 0
	best_error_weights[0] = index_of_best_combination;
	if (index_of_best_combination >= 0)
	{
		errors_of_best_combination[index_of_best_combination] = ERROR_CALC_DEFAULT;
	}

	// Search the remaining results and pick the best candidate modes for trial 1+
	for (unsigned int i = 1; i < tune_candidate_limit; i++)
	{
		vint vbest_error_index(-1);
		vfloat vbest_ep_error(ERROR_CALC_DEFAULT);

		// TODO: This should use size_t for the inputs of start/end_block_mode
		// to avoid some of this type conversion, but that propagates and will
		// need a bigger PR to fix
		size_t start_mode = round_down_to_simd_multiple_vla(start_block_mode);
		vint lane_ids = vint::lane_id() + vint_from_size(start_mode);
		for (size_t j = start_mode; j < end_block_mode; j += ASTCENC_SIMD_WIDTH)
		{
			vfloat err = vfloat(errors_of_best_combination + j);
			vmask mask = err < vbest_ep_error;
			vbest_ep_error = select(vbest_ep_error, err, mask);
			vbest_error_index = select(vbest_error_index, lane_ids, mask);
			lane_ids += vint(ASTCENC_SIMD_WIDTH);
		}

		// Pick best mode from the SIMD result, using lowest matching index to ensure invariance
		vmask lanes_min_error = vbest_ep_error == hmin(vbest_ep_error);
		vbest_error_index = select(vint(0x7FFFFFFF), vbest_error_index, lanes_min_error);

		int best_error_index = hmin_s(vbest_error_index);

		best_error_weights[i] = best_error_index;

		// Max the error for this candidate so we don't pick it again
		if (best_error_index >= 0)
		{
			errors_of_best_combination[best_error_index] = ERROR_CALC_DEFAULT;
		}
		// Early-out if no more candidates are valid
		else
		{
			break;
		}
	}

	for (unsigned int i = 0; i < tune_candidate_limit; i++)
	{
		if (best_error_weights[i] < 0)
		{
			return i;
		}

		block_mode[i] = best_error_weights[i];

		quant_level[i] = static_cast<quant_method>(best_quant_levels[best_error_weights[i]]);
		quant_level_mod[i] = static_cast<quant_method>(best_quant_levels_mod[best_error_weights[i]]);

		assert(quant_level[i] >= QUANT_6 && quant_level[i] <= QUANT_256);
		assert(quant_level_mod[i] >= QUANT_6 && quant_level_mod[i] <= QUANT_256);

		for (int j = 0; j < partition_count; j++)
		{
			partition_format_specifiers[i][j] = best_ep_formats[best_error_weights[i]][j];
		}
	}

	return tune_candidate_limit;
}

#endif
