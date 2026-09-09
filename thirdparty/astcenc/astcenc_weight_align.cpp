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
 * @brief Functions for angular-sum algorithm for weight alignment.
 *
 * This algorithm works as follows:
 * - we compute a complex number P as (cos s*i, sin s*i) for each weight,
 *   where i is the input value and s is a scaling factor based on the spacing between the weights.
 * - we then add together complex numbers for all the weights.
 * - we then compute the length and angle of the resulting sum.
 *
 * This should produce the following results:
 * - perfect alignment results in a vector whose length is equal to the sum of lengths of all inputs
 * - even distribution results in a vector of length 0.
 * - all samples identical results in perfect alignment for every scaling.
 *
 * For each scaling factor within a given set, we compute an alignment factor from 0 to 1. This
 * should then result in some scalings standing out as having particularly good alignment factors;
 * we can use this to produce a set of candidate scale/shift values for various quantization levels;
 * we should then actually try them and see what happens.
 */

#include "astcenc_internal.h"
#include "astcenc_vecmathlib.h"

#include <stdio.h>
#include <cassert>
#include <cstring>
#include <cfloat>

static constexpr unsigned int ANGULAR_STEPS { 32 };

static_assert((ANGULAR_STEPS % ASTCENC_SIMD_WIDTH) == 0,
              "ANGULAR_STEPS must be multiple of ASTCENC_SIMD_WIDTH");

static_assert(ANGULAR_STEPS >= 32,
              "ANGULAR_STEPS must be at least max(steps_for_quant_level)");

// Store a reduced sin/cos table for 64 possible weight values; this causes
// slight quality loss compared to using sin() and cos() directly. Must be 2^N.
static constexpr unsigned int SINCOS_STEPS { 64 };

static const uint8_t steps_for_quant_level[12] {
	2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32
};

ASTCENC_ALIGNAS static float sin_table[SINCOS_STEPS][ANGULAR_STEPS];
ASTCENC_ALIGNAS static float cos_table[SINCOS_STEPS][ANGULAR_STEPS];

#if defined(ASTCENC_DIAGNOSTICS)
	static bool print_once { true };
#endif

/* See header for documentation. */
void prepare_angular_tables()
{
	for (unsigned int i = 0; i < ANGULAR_STEPS; i++)
	{
		float angle_step = static_cast<float>(i + 1);

		for (unsigned int j = 0; j < SINCOS_STEPS; j++)
		{
			sin_table[j][i] = static_cast<float>(sinf((2.0f * astc::PI / (SINCOS_STEPS - 1.0f)) * angle_step * static_cast<float>(j)));
			cos_table[j][i] = static_cast<float>(cosf((2.0f * astc::PI / (SINCOS_STEPS - 1.0f)) * angle_step * static_cast<float>(j)));
		}
	}
}

/**
 * @brief Compute the angular alignment factors and offsets.
 *
 * @param      weight_count              The number of (decimated) weights.
 * @param      dec_weight_ideal_value    The ideal decimated unquantized weight values.
 * @param      max_angular_steps         The maximum number of steps to be tested.
 * @param[out] offsets                   The output angular offsets array.
 */
static void compute_angular_offsets(
	unsigned int weight_count,
	const float* dec_weight_ideal_value,
	unsigned int max_angular_steps,
	float* offsets
) {
	promise(weight_count > 0);
	promise(max_angular_steps > 0);

	ASTCENC_ALIGNAS int isamplev[BLOCK_MAX_WEIGHTS];

	// Precompute isample; arrays are always allocated 64 elements long
	for (unsigned int i = 0; i < weight_count; i += ASTCENC_SIMD_WIDTH)
	{
		// Ideal weight can be outside [0, 1] range, so clamp to fit table
		vfloat ideal_weight = clampzo(loada(dec_weight_ideal_value + i));

		// Convert a weight to a sincos table index
		vfloat sample = ideal_weight * (SINCOS_STEPS - 1.0f);
		vint isample = float_to_int_rtn(sample);
		storea(isample, isamplev + i);
	}

	// Arrays are multiple of SIMD width (ANGULAR_STEPS), safe to overshoot max
	vfloat mult(1.0f / (2.0f * astc::PI));

	for (unsigned int i = 0; i < max_angular_steps; i += ASTCENC_SIMD_WIDTH)
	{
		vfloat anglesum_x = vfloat::zero();
		vfloat anglesum_y = vfloat::zero();

		for (unsigned int j = 0; j < weight_count; j++)
		{
			int isample = isamplev[j];
			anglesum_x += loada(cos_table[isample] + i);
			anglesum_y += loada(sin_table[isample] + i);
		}

		vfloat angle = atan2(anglesum_y, anglesum_x);
		vfloat ofs = angle * mult;
		storea(ofs, offsets + i);
	}
}

/**
 * @brief For a given step size compute the lowest and highest weight.
 *
 * Compute the lowest and highest weight that results from quantizing using the given stepsize and
 * offset, and then compute the resulting error. The cut errors indicate the error that results from
 * forcing samples that should have had one weight value one step up or down.
 *
 * @param      weight_count              The number of (decimated) weights.
 * @param      dec_weight_ideal_value    The ideal decimated unquantized weight values.
 * @param      max_angular_steps         The maximum number of steps to be tested.
 * @param      max_quant_steps           The maximum quantization level to be tested.
 * @param      offsets                   The angular offsets array.
 * @param[out] lowest_weight             Per angular step, the lowest weight.
 * @param[out] weight_span               Per angular step, the span between lowest and highest weight.
 * @param[out] error                     Per angular step, the error.
 * @param[out] cut_low_weight_error      Per angular step, the low weight cut error.
 * @param[out] cut_high_weight_error     Per angular step, the high weight cut error.
 */
static void compute_lowest_and_highest_weight(
	unsigned int weight_count,
	const float* dec_weight_ideal_value,
	unsigned int max_angular_steps,
	unsigned int max_quant_steps,
	const float* offsets,
	float* lowest_weight,
	int* weight_span,
	float* error,
	float* cut_low_weight_error,
	float* cut_high_weight_error
) {
	promise(weight_count > 0);
	promise(max_angular_steps > 0);

	vfloat rcp_stepsize = int_to_float(vint::lane_id()) + vfloat(1.0f);

	// Compute minimum/maximum weights in the weight array. Our remapping
	// is monotonic, so the min/max rounded weights relate to the min/max
	// unrounded weights in a straightforward way.
	vfloat min_weight(FLT_MAX);
	vfloat max_weight(-FLT_MAX);

	vint lane_id = vint::lane_id();
	for (unsigned int i = 0; i < weight_count; i += ASTCENC_SIMD_WIDTH)
	{
		vmask active = lane_id < vint(weight_count);
		lane_id += vint(ASTCENC_SIMD_WIDTH);

		vfloat weights = loada(dec_weight_ideal_value + i);
		min_weight = min(min_weight, select(min_weight, weights, active));
		max_weight = max(max_weight, select(max_weight, weights, active));
	}

	min_weight = hmin(min_weight);
	max_weight = hmax(max_weight);

	// Arrays are ANGULAR_STEPS long, so always safe to run full vectors
	for (unsigned int sp = 0; sp < max_angular_steps; sp += ASTCENC_SIMD_WIDTH)
	{
		vfloat errval = vfloat::zero();
		vfloat cut_low_weight_err = vfloat::zero();
		vfloat cut_high_weight_err = vfloat::zero();
		vfloat offset = loada(offsets + sp);

		// We know the min and max weight values, so we can figure out
		// the corresponding indices before we enter the loop.
		vfloat minidx = round(min_weight * rcp_stepsize - offset);
		vfloat maxidx = round(max_weight * rcp_stepsize - offset);

		for (unsigned int j = 0; j < weight_count; j++)
		{
			vfloat sval = load1(dec_weight_ideal_value + j) * rcp_stepsize - offset;
			vfloat svalrte = round(sval);
			vfloat diff = sval - svalrte;
			errval += diff * diff;

			// Accumulate errors for minimum index
			vmask mask = svalrte == minidx;
			vfloat accum = cut_low_weight_err + vfloat(1.0f) - vfloat(2.0f) * diff;
			cut_low_weight_err = select(cut_low_weight_err, accum, mask);

			// Accumulate errors for maximum index
			mask = svalrte == maxidx;
			accum = cut_high_weight_err + vfloat(1.0f) + vfloat(2.0f) * diff;
			cut_high_weight_err = select(cut_high_weight_err, accum, mask);
		}

		// Write out min weight and weight span; clamp span to a usable range
		vint span = float_to_int(maxidx - minidx + vfloat(1));
		span = min(span, vint(max_quant_steps + 3));
		span = max(span, vint(2));
		storea(minidx, lowest_weight + sp);
		storea(span, weight_span + sp);

		// The cut_(lowest/highest)_weight_error indicate the error that results from  forcing
		// samples that should have had the weight value one step (up/down).
		vfloat ssize = 1.0f / rcp_stepsize;
		vfloat errscale = ssize * ssize;
		storea(errval * errscale, error + sp);
		storea(cut_low_weight_err * errscale, cut_low_weight_error + sp);
		storea(cut_high_weight_err * errscale, cut_high_weight_error + sp);

		rcp_stepsize = rcp_stepsize + vfloat(ASTCENC_SIMD_WIDTH);
	}
}

/**
 * @brief The main function for the angular algorithm.
 *
 * @param      weight_count              The number of (decimated) weights.
 * @param      dec_weight_ideal_value    The ideal decimated unquantized weight values.
 * @param      max_quant_level           The maximum quantization level to be tested.
 * @param[out] low_value                 Per angular step, the lowest weight value.
 * @param[out] high_value                Per angular step, the highest weight value.
 */
static void compute_angular_endpoints_for_quant_levels(
	unsigned int weight_count,
	const float* dec_weight_ideal_value,
	unsigned int max_quant_level,
	float low_value[TUNE_MAX_ANGULAR_QUANT + 1],
	float high_value[TUNE_MAX_ANGULAR_QUANT + 1]
) {
	unsigned int max_quant_steps = steps_for_quant_level[max_quant_level];
	unsigned int max_angular_steps = steps_for_quant_level[max_quant_level];

	ASTCENC_ALIGNAS float angular_offsets[ANGULAR_STEPS];

	compute_angular_offsets(weight_count, dec_weight_ideal_value,
	                        max_angular_steps, angular_offsets);

	ASTCENC_ALIGNAS float lowest_weight[ANGULAR_STEPS];
	ASTCENC_ALIGNAS int32_t weight_span[ANGULAR_STEPS];
	ASTCENC_ALIGNAS float error[ANGULAR_STEPS];
	ASTCENC_ALIGNAS float cut_low_weight_error[ANGULAR_STEPS];
	ASTCENC_ALIGNAS float cut_high_weight_error[ANGULAR_STEPS];

	compute_lowest_and_highest_weight(weight_count, dec_weight_ideal_value,
	                                  max_angular_steps, max_quant_steps,
	                                  angular_offsets, lowest_weight, weight_span, error,
	                                  cut_low_weight_error, cut_high_weight_error);

	// For each quantization level, find the best error terms. Use packed vectors so data-dependent
	// branches can become selects. This involves some integer to float casts, but the values are
	// small enough so they never round the wrong way.
	vfloat4 best_results[36];

	// Initialize the array to some safe defaults
	promise(max_quant_steps > 0);
	for (unsigned int i = 0; i < (max_quant_steps + 4); i++)
	{
		// Lane<0> = Best error
		// Lane<1> = Best scale; -1 indicates no solution found
		// Lane<2> = Cut low weight
		best_results[i] = vfloat4(ERROR_CALC_DEFAULT, -1.0f, 0.0f, 0.0f);
	}

	promise(max_angular_steps > 0);
	for (unsigned int i = 0; i < max_angular_steps; i++)
	{
		float i_flt = static_cast<float>(i);

		int idx_span = weight_span[i];

		float error_cut_low = error[i] + cut_low_weight_error[i];
		float error_cut_high = error[i] + cut_high_weight_error[i];
		float error_cut_low_high = error[i] + cut_low_weight_error[i] + cut_high_weight_error[i];

		// Check best error against record N
		vfloat4 best_result = best_results[idx_span];
		vfloat4 new_result = vfloat4(error[i], i_flt, 0.0f, 0.0f);
		vmask4 mask = vfloat4(best_result.lane<0>()) > vfloat4(error[i]);
		best_results[idx_span] = select(best_result, new_result, mask);

		// Check best error against record N-1 with either cut low or cut high
		best_result = best_results[idx_span - 1];

		new_result = vfloat4(error_cut_low, i_flt, 1.0f, 0.0f);
		mask = vfloat4(best_result.lane<0>()) > vfloat4(error_cut_low);
		best_result = select(best_result, new_result, mask);

		new_result = vfloat4(error_cut_high, i_flt, 0.0f, 0.0f);
		mask = vfloat4(best_result.lane<0>()) > vfloat4(error_cut_high);
		best_results[idx_span - 1] = select(best_result, new_result, mask);

		// Check best error against record N-2 with both cut low and high
		best_result = best_results[idx_span - 2];
		new_result = vfloat4(error_cut_low_high, i_flt, 1.0f, 0.0f);
		mask = vfloat4(best_result.lane<0>()) > vfloat4(error_cut_low_high);
		best_results[idx_span - 2] = select(best_result, new_result, mask);
	}

	for (unsigned int i = 0; i <= max_quant_level; i++)
	{
		unsigned int q = steps_for_quant_level[i];
		int bsi = static_cast<int>(best_results[q].lane<1>());

		// Did we find anything?
#if defined(ASTCENC_DIAGNOSTICS)
		if ((bsi < 0) && print_once)
		{
			print_once = false;
			printf("INFO: Unable to find full encoding within search error limit.\n\n");
		}
#endif

		bsi = astc::max(0, bsi);

		float lwi = lowest_weight[bsi] + best_results[q].lane<2>();
		float hwi = lwi + static_cast<float>(q) - 1.0f;

		float stepsize = 1.0f / (1.0f + static_cast<float>(bsi));
		low_value[i]  = (angular_offsets[bsi] + lwi) * stepsize;
		high_value[i] = (angular_offsets[bsi] + hwi) * stepsize;
	}
}

/* See header for documentation. */
void compute_angular_endpoints_1plane(
	bool only_always,
	const block_size_descriptor& bsd,
	const float* dec_weight_ideal_value,
	unsigned int max_weight_quant,
	compression_working_buffers& tmpbuf
) {
	float (&low_value)[WEIGHTS_MAX_BLOCK_MODES] = tmpbuf.weight_low_value1;
	float (&high_value)[WEIGHTS_MAX_BLOCK_MODES] = tmpbuf.weight_high_value1;

	float (&low_values)[WEIGHTS_MAX_DECIMATION_MODES][TUNE_MAX_ANGULAR_QUANT + 1] = tmpbuf.weight_low_values1;
	float (&high_values)[WEIGHTS_MAX_DECIMATION_MODES][TUNE_MAX_ANGULAR_QUANT + 1] = tmpbuf.weight_high_values1;

	unsigned int max_decimation_modes = only_always ? bsd.decimation_mode_count_always
	                                                : bsd.decimation_mode_count_selected;
	promise(max_decimation_modes > 0);
	for (unsigned int i = 0; i < max_decimation_modes; i++)
	{
		const decimation_mode& dm = bsd.decimation_modes[i];
		if (!dm.is_ref_1plane(static_cast<quant_method>(max_weight_quant)))
		{
			continue;
		}

		unsigned int weight_count = bsd.get_decimation_info(i).weight_count;

		unsigned int max_precision = dm.maxprec_1plane;
		if (max_precision > TUNE_MAX_ANGULAR_QUANT)
		{
			max_precision = TUNE_MAX_ANGULAR_QUANT;
		}

		if (max_precision > max_weight_quant)
		{
			max_precision = max_weight_quant;
		}

		compute_angular_endpoints_for_quant_levels(
		    weight_count,
		    dec_weight_ideal_value + i * BLOCK_MAX_WEIGHTS,
		    max_precision, low_values[i], high_values[i]);
	}

	unsigned int max_block_modes = only_always ? bsd.block_mode_count_1plane_always
	                                           : bsd.block_mode_count_1plane_selected;
	promise(max_block_modes > 0);
	for (unsigned int i = 0; i < max_block_modes; i++)
	{
		const block_mode& bm = bsd.block_modes[i];
		assert(!bm.is_dual_plane);

		unsigned int quant_mode = bm.quant_mode;
		unsigned int decim_mode = bm.decimation_mode;

		if (quant_mode <= TUNE_MAX_ANGULAR_QUANT)
		{
			low_value[i] = low_values[decim_mode][quant_mode];
			high_value[i] = high_values[decim_mode][quant_mode];
		}
		else
		{
			low_value[i] = 0.0f;
			high_value[i] = 1.0f;
		}
	}
}

/* See header for documentation. */
void compute_angular_endpoints_2planes(
	const block_size_descriptor& bsd,
	const float* dec_weight_ideal_value,
	unsigned int max_weight_quant,
	compression_working_buffers& tmpbuf
) {
	float (&low_value1)[WEIGHTS_MAX_BLOCK_MODES] = tmpbuf.weight_low_value1;
	float (&high_value1)[WEIGHTS_MAX_BLOCK_MODES] = tmpbuf.weight_high_value1;
	float (&low_value2)[WEIGHTS_MAX_BLOCK_MODES] = tmpbuf.weight_low_value2;
	float (&high_value2)[WEIGHTS_MAX_BLOCK_MODES] = tmpbuf.weight_high_value2;

	float (&low_values1)[WEIGHTS_MAX_DECIMATION_MODES][TUNE_MAX_ANGULAR_QUANT + 1] = tmpbuf.weight_low_values1;
	float (&high_values1)[WEIGHTS_MAX_DECIMATION_MODES][TUNE_MAX_ANGULAR_QUANT + 1] = tmpbuf.weight_high_values1;
	float (&low_values2)[WEIGHTS_MAX_DECIMATION_MODES][TUNE_MAX_ANGULAR_QUANT + 1] = tmpbuf.weight_low_values2;
	float (&high_values2)[WEIGHTS_MAX_DECIMATION_MODES][TUNE_MAX_ANGULAR_QUANT + 1] = tmpbuf.weight_high_values2;

	promise(bsd.decimation_mode_count_selected > 0);
	for (unsigned int i = 0; i < bsd.decimation_mode_count_selected; i++)
	{
		const decimation_mode& dm = bsd.decimation_modes[i];
		if (!dm.is_ref_2plane(static_cast<quant_method>(max_weight_quant)))
		{
			continue;
		}

		unsigned int weight_count = bsd.get_decimation_info(i).weight_count;

		unsigned int max_precision = dm.maxprec_2planes;
		if (max_precision > TUNE_MAX_ANGULAR_QUANT)
		{
			max_precision = TUNE_MAX_ANGULAR_QUANT;
		}

		if (max_precision > max_weight_quant)
		{
			max_precision = max_weight_quant;
		}

		compute_angular_endpoints_for_quant_levels(
		    weight_count,
		    dec_weight_ideal_value + i * BLOCK_MAX_WEIGHTS,
		    max_precision, low_values1[i], high_values1[i]);

		compute_angular_endpoints_for_quant_levels(
		    weight_count,
		    dec_weight_ideal_value + i * BLOCK_MAX_WEIGHTS + WEIGHTS_PLANE2_OFFSET,
		    max_precision, low_values2[i], high_values2[i]);
	}

	unsigned int start = bsd.block_mode_count_1plane_selected;
	unsigned int end = bsd.block_mode_count_1plane_2plane_selected;
	for (unsigned int i = start; i < end; i++)
	{
		const block_mode& bm = bsd.block_modes[i];
		unsigned int quant_mode = bm.quant_mode;
		unsigned int decim_mode = bm.decimation_mode;

		if (quant_mode <= TUNE_MAX_ANGULAR_QUANT)
		{
			low_value1[i] = low_values1[decim_mode][quant_mode];
			high_value1[i] = high_values1[decim_mode][quant_mode];
			low_value2[i] = low_values2[decim_mode][quant_mode];
			high_value2[i] = high_values2[decim_mode][quant_mode];
		}
		else
		{
			low_value1[i] = 0.0f;
			high_value1[i] = 1.0f;
			low_value2[i] = 0.0f;
			high_value2[i] = 1.0f;
		}
	}
}

#endif
