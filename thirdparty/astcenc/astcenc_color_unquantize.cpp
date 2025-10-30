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

#include <utility>

/**
 * @brief Functions for color unquantization.
 */

#include "astcenc_internal.h"

/**
 * @brief Un-blue-contract a color.
 *
 * This function reverses any applied blue contraction.
 *
 * @param input   The input color that has been blue-contracted.
 *
 * @return The uncontracted color.
 */
static ASTCENC_SIMD_INLINE vint4 uncontract_color(
	vint4 input
) {
	vmask4 mask(true, true, false, false);
	vint4 bc0 = asr<1>(input + input.lane<2>());
	return select(input, bc0, mask);
}

void rgba_delta_unpack(
	vint4 input0,
	vint4 input1,
	vint4& output0,
	vint4& output1
) {
	// Apply bit transfer
	bit_transfer_signed(input1, input0);

	// Apply blue-uncontraction if needed
	int rgb_sum = hadd_rgb_s(input1);
	input1 = input1 + input0;
	if (rgb_sum < 0)
	{
		input0 = uncontract_color(input0);
		input1 = uncontract_color(input1);
		std::swap(input0, input1);
	}

	output0 = clamp(0, 255, input0);
	output1 = clamp(0, 255, input1);
}

/**
 * @brief Unpack an LDR RGB color that uses delta encoding.
 *
 * Output alpha set to 255.
 *
 * @param      input0    The packed endpoint 0 color.
 * @param      input1    The packed endpoint 1 color deltas.
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void rgb_delta_unpack(
	vint4 input0,
	vint4 input1,
	vint4& output0,
	vint4& output1
) {
	rgba_delta_unpack(input0, input1, output0, output1);
	output0.set_lane<3>(255);
	output1.set_lane<3>(255);
}

void rgba_unpack(
	vint4 input0,
	vint4 input1,
	vint4& output0,
	vint4& output1
) {
	// Apply blue-uncontraction if needed
	if (hadd_rgb_s(input0) > hadd_rgb_s(input1))
	{
		input0 = uncontract_color(input0);
		input1 = uncontract_color(input1);
		std::swap(input0, input1);
	}

	output0 = input0;
	output1 = input1;
}

/**
 * @brief Unpack an LDR RGB color that uses direct encoding.
 *
 * Output alpha set to 255.
 *
 * @param      input0    The packed endpoint 0 color.
 * @param      input1    The packed endpoint 1 color.
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void rgb_unpack(
	vint4 input0,
	vint4 input1,
	vint4& output0,
	vint4& output1
) {
	rgba_unpack(input0, input1, output0, output1);
	output0.set_lane<3>(255);
	output1.set_lane<3>(255);
}

/**
 * @brief Unpack an LDR RGBA color that uses scaled encoding.
 *
 * Note only the RGB channels use the scaled encoding, alpha uses direct.
 *
 * @param      input0    The packed endpoint 0 color.
 * @param      alpha1    The packed endpoint 1 alpha value.
 * @param      scale     The packed quantized scale.
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void rgb_scale_alpha_unpack(
	vint4 input0,
	uint8_t alpha1,
	uint8_t scale,
	vint4& output0,
	vint4& output1
) {
	output1 = input0;
	output1.set_lane<3>(alpha1);

	output0 = asr<8>(input0 * scale);
	output0.set_lane<3>(input0.lane<3>());
}

/**
 * @brief Unpack an LDR RGB color that uses scaled encoding.
 *
 * Output alpha is 255.
 *
 * @param      input0    The packed endpoint 0 color.
 * @param      scale     The packed scale.
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void rgb_scale_unpack(
	vint4 input0,
	int scale,
	vint4& output0,
	vint4& output1
) {
	output1 = input0;
	output1.set_lane<3>(255);

	output0 = asr<8>(input0 * scale);
	output0.set_lane<3>(255);
}

/**
 * @brief Unpack an LDR L color that uses direct encoding.
 *
 * Output alpha is 255.
 *
 * @param      input     The packed endpoints.
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void luminance_unpack(
	const uint8_t input[2],
	vint4& output0,
	vint4& output1
) {
	int lum0 = input[0];
	int lum1 = input[1];
	output0 = vint4(lum0, lum0, lum0, 255);
	output1 = vint4(lum1, lum1, lum1, 255);
}

/**
 * @brief Unpack an LDR L color that uses delta encoding.
 *
 * Output alpha is 255.
 *
 * @param      input     The packed endpoints (L0, L1).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void luminance_delta_unpack(
	const uint8_t input[2],
	vint4& output0,
	vint4& output1
) {
	int v0 = input[0];
	int v1 = input[1];
	int l0 = (v0 >> 2) | (v1 & 0xC0);
	int l1 = l0 + (v1 & 0x3F);

	l1 = astc::min(l1, 255);

	output0 = vint4(l0, l0, l0, 255);
	output1 = vint4(l1, l1, l1, 255);
}

/**
 * @brief Unpack an LDR LA color that uses direct encoding.
 *
 * @param      input     The packed endpoints (L0, L1, A0, A1).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void luminance_alpha_unpack(
	const uint8_t input[4],
	vint4& output0,
	vint4& output1
) {
	int lum0 = input[0];
	int lum1 = input[1];
	int alpha0 = input[2];
	int alpha1 = input[3];
	output0 = vint4(lum0, lum0, lum0, alpha0);
	output1 = vint4(lum1, lum1, lum1, alpha1);
}

/**
 * @brief Unpack an LDR LA color that uses delta encoding.
 *
 * @param      input     The packed endpoints (L0, L1, A0, A1).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void luminance_alpha_delta_unpack(
	const uint8_t input[4],
	vint4& output0,
	vint4& output1
) {
	int lum0 = input[0];
	int lum1 = input[1];
	int alpha0 = input[2];
	int alpha1 = input[3];

	lum0 |= (lum1 & 0x80) << 1;
	alpha0 |= (alpha1 & 0x80) << 1;
	lum1 &= 0x7F;
	alpha1 &= 0x7F;

	if (lum1 & 0x40)
	{
		lum1 -= 0x80;
	}

	if (alpha1 & 0x40)
	{
		alpha1 -= 0x80;
	}

	lum0 >>= 1;
	lum1 >>= 1;
	alpha0 >>= 1;
	alpha1 >>= 1;
	lum1 += lum0;
	alpha1 += alpha0;

	lum1 = astc::clamp(lum1, 0, 255);
	alpha1 = astc::clamp(alpha1, 0, 255);

	output0 = vint4(lum0, lum0, lum0, alpha0);
	output1 = vint4(lum1, lum1, lum1, alpha1);
}

/**
 * @brief Unpack an HDR RGB + offset encoding.
 *
 * @param      input     The packed endpoints (packed and modal).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void hdr_rgbo_unpack(
	const uint8_t input[4],
	vint4& output0,
	vint4& output1
) {
	int v0 = input[0];
	int v1 = input[1];
	int v2 = input[2];
	int v3 = input[3];

	int modeval = ((v0 & 0xC0) >> 6) | (((v1 & 0x80) >> 7) << 2) | (((v2 & 0x80) >> 7) << 3);

	int majcomp;
	int mode;
	if ((modeval & 0xC) != 0xC)
	{
		majcomp = modeval >> 2;
		mode = modeval & 3;
	}
	else if (modeval != 0xF)
	{
		majcomp = modeval & 3;
		mode = 4;
	}
	else
	{
		majcomp = 0;
		mode = 5;
	}

	int red = v0 & 0x3F;
	int green = v1 & 0x1F;
	int blue = v2 & 0x1F;
	int scale = v3 & 0x1F;

	int bit0 = (v1 >> 6) & 1;
	int bit1 = (v1 >> 5) & 1;
	int bit2 = (v2 >> 6) & 1;
	int bit3 = (v2 >> 5) & 1;
	int bit4 = (v3 >> 7) & 1;
	int bit5 = (v3 >> 6) & 1;
	int bit6 = (v3 >> 5) & 1;

	int ohcomp = 1 << mode;

	if (ohcomp & 0x30)
		green |= bit0 << 6;
	if (ohcomp & 0x3A)
		green |= bit1 << 5;
	if (ohcomp & 0x30)
		blue |= bit2 << 6;
	if (ohcomp & 0x3A)
		blue |= bit3 << 5;

	if (ohcomp & 0x3D)
		scale |= bit6 << 5;
	if (ohcomp & 0x2D)
		scale |= bit5 << 6;
	if (ohcomp & 0x04)
		scale |= bit4 << 7;

	if (ohcomp & 0x3B)
		red |= bit4 << 6;
	if (ohcomp & 0x04)
		red |= bit3 << 6;

	if (ohcomp & 0x10)
		red |= bit5 << 7;
	if (ohcomp & 0x0F)
		red |= bit2 << 7;

	if (ohcomp & 0x05)
		red |= bit1 << 8;
	if (ohcomp & 0x0A)
		red |= bit0 << 8;

	if (ohcomp & 0x05)
		red |= bit0 << 9;
	if (ohcomp & 0x02)
		red |= bit6 << 9;

	if (ohcomp & 0x01)
		red |= bit3 << 10;
	if (ohcomp & 0x02)
		red |= bit5 << 10;

	// expand to 12 bits.
	static const int shamts[6] { 1, 1, 2, 3, 4, 5 };
	int shamt = shamts[mode];
	red <<= shamt;
	green <<= shamt;
	blue <<= shamt;
	scale <<= shamt;

	// on modes 0 to 4, the values stored for "green" and "blue" are differentials,
	// not absolute values.
	if (mode != 5)
	{
		green = red - green;
		blue = red - blue;
	}

	// switch around components.
	int temp;
	switch (majcomp)
	{
	case 1:
		temp = red;
		red = green;
		green = temp;
		break;
	case 2:
		temp = red;
		red = blue;
		blue = temp;
		break;
	default:
		break;
	}

	int red0 = red - scale;
	int green0 = green - scale;
	int blue0 = blue - scale;

	// clamp to [0,0xFFF].
	if (red < 0)
		red = 0;
	if (green < 0)
		green = 0;
	if (blue < 0)
		blue = 0;

	if (red0 < 0)
		red0 = 0;
	if (green0 < 0)
		green0 = 0;
	if (blue0 < 0)
		blue0 = 0;

	output0 = vint4(red0 << 4, green0 << 4, blue0 << 4, 0x7800);
	output1 = vint4(red << 4, green << 4, blue << 4, 0x7800);
}

/**
 * @brief Unpack an HDR RGB direct encoding.
 *
 * @param      input     The packed endpoints (packed and modal).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void hdr_rgb_unpack(
	const uint8_t input[6],
	vint4& output0,
	vint4& output1
) {

	int v0 = input[0];
	int v1 = input[1];
	int v2 = input[2];
	int v3 = input[3];
	int v4 = input[4];
	int v5 = input[5];

	// extract all the fixed-placement bitfields
	int modeval = ((v1 & 0x80) >> 7) | (((v2 & 0x80) >> 7) << 1) | (((v3 & 0x80) >> 7) << 2);

	int majcomp = ((v4 & 0x80) >> 7) | (((v5 & 0x80) >> 7) << 1);

	if (majcomp == 3)
	{
		output0 = vint4(v0 << 8, v2 << 8, (v4 & 0x7F) << 9, 0x7800);
		output1 = vint4(v1 << 8, v3 << 8, (v5 & 0x7F) << 9, 0x7800);
		return;
	}

	int a = v0 | ((v1 & 0x40) << 2);
	int b0 = v2 & 0x3f;
	int b1 = v3 & 0x3f;
	int c = v1 & 0x3f;
	int d0 = v4 & 0x7f;
	int d1 = v5 & 0x7f;

	// get hold of the number of bits in 'd0' and 'd1'
	static const int dbits_tab[8] { 7, 6, 7, 6, 5, 6, 5, 6 };
	int dbits = dbits_tab[modeval];

	// extract six variable-placement bits
	int bit0 = (v2 >> 6) & 1;
	int bit1 = (v3 >> 6) & 1;
	int bit2 = (v4 >> 6) & 1;
	int bit3 = (v5 >> 6) & 1;
	int bit4 = (v4 >> 5) & 1;
	int bit5 = (v5 >> 5) & 1;

	// and prepend the variable-placement bits depending on mode.
	int ohmod = 1 << modeval;	// one-hot-mode
	if (ohmod & 0xA4)
		a |= bit0 << 9;
	if (ohmod & 0x8)
		a |= bit2 << 9;
	if (ohmod & 0x50)
		a |= bit4 << 9;

	if (ohmod & 0x50)
		a |= bit5 << 10;
	if (ohmod & 0xA0)
		a |= bit1 << 10;

	if (ohmod & 0xC0)
		a |= bit2 << 11;

	if (ohmod & 0x4)
		c |= bit1 << 6;
	if (ohmod & 0xE8)
		c |= bit3 << 6;

	if (ohmod & 0x20)
		c |= bit2 << 7;

	if (ohmod & 0x5B)
	{
		b0 |= bit0 << 6;
		b1 |= bit1 << 6;
	}

	if (ohmod & 0x12)
	{
		b0 |= bit2 << 7;
		b1 |= bit3 << 7;
	}

	if (ohmod & 0xAF)
	{
		d0 |= bit4 << 5;
		d1 |= bit5 << 5;
	}

	if (ohmod & 0x5)
	{
		d0 |= bit2 << 6;
		d1 |= bit3 << 6;
	}

	// sign-extend 'd0' and 'd1'
	// note: this code assumes that signed right-shift actually sign-fills, not zero-fills.
	int32_t d0x = d0;
	int32_t d1x = d1;
	int sx_shamt = 32 - dbits;
	d0x <<= sx_shamt;
	d0x >>= sx_shamt;
	d1x <<= sx_shamt;
	d1x >>= sx_shamt;
	d0 = d0x;
	d1 = d1x;

	// expand all values to 12 bits, with left-shift as needed.
	int val_shamt = (modeval >> 1) ^ 3;
	a <<= val_shamt;
	b0 <<= val_shamt;
	b1 <<= val_shamt;
	c <<= val_shamt;
	d0 <<= val_shamt;
	d1 <<= val_shamt;

	// then compute the actual color values.
	int red1 = a;
	int green1 = a - b0;
	int blue1 = a - b1;
	int red0 = a - c;
	int green0 = a - b0 - c - d0;
	int blue0 = a - b1 - c - d1;

	// clamp the color components to [0,2^12 - 1]
	red0 = astc::clamp(red0, 0, 4095);
	green0 = astc::clamp(green0, 0, 4095);
	blue0 = astc::clamp(blue0, 0, 4095);

	red1 = astc::clamp(red1, 0, 4095);
	green1 = astc::clamp(green1, 0, 4095);
	blue1 = astc::clamp(blue1, 0, 4095);

	// switch around the color components
	int temp0, temp1;
	switch (majcomp)
	{
	case 1:					// switch around red and green
		temp0 = red0;
		temp1 = red1;
		red0 = green0;
		red1 = green1;
		green0 = temp0;
		green1 = temp1;
		break;
	case 2:					// switch around red and blue
		temp0 = red0;
		temp1 = red1;
		red0 = blue0;
		red1 = blue1;
		blue0 = temp0;
		blue1 = temp1;
		break;
	case 0:					// no switch
		break;
	}

	output0 = vint4(red0 << 4, green0 << 4, blue0 << 4, 0x7800);
	output1 = vint4(red1 << 4, green1 << 4, blue1 << 4, 0x7800);
}

/**
 * @brief Unpack an HDR RGB + LDR A direct encoding.
 *
 * @param      input     The packed endpoints (packed and modal).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void hdr_rgb_ldr_alpha_unpack(
	const uint8_t input[8],
	vint4& output0,
	vint4& output1
) {
	hdr_rgb_unpack(input, output0, output1);

	int v6 = input[6];
	int v7 = input[7];
	output0.set_lane<3>(v6);
	output1.set_lane<3>(v7);
}

/**
 * @brief Unpack an HDR L (small range) direct encoding.
 *
 * @param      input     The packed endpoints (packed and modal).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void hdr_luminance_small_range_unpack(
	const uint8_t input[2],
	vint4& output0,
	vint4& output1
) {
	int v0 = input[0];
	int v1 = input[1];

	int y0, y1;
	if (v0 & 0x80)
	{
		y0 = ((v1 & 0xE0) << 4) | ((v0 & 0x7F) << 2);
		y1 = (v1 & 0x1F) << 2;
	}
	else
	{
		y0 = ((v1 & 0xF0) << 4) | ((v0 & 0x7F) << 1);
		y1 = (v1 & 0xF) << 1;
	}

	y1 += y0;
	if (y1 > 0xFFF)
	{
		y1 = 0xFFF;
	}

	output0 = vint4(y0 << 4, y0 << 4, y0 << 4, 0x7800);
	output1 = vint4(y1 << 4, y1 << 4, y1 << 4, 0x7800);
}

/**
 * @brief Unpack an HDR L (large range) direct encoding.
 *
 * @param      input     The packed endpoints (packed and modal).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void hdr_luminance_large_range_unpack(
	const uint8_t input[2],
	vint4& output0,
	vint4& output1
) {
	int v0 = input[0];
	int v1 = input[1];

	int y0, y1;
	if (v1 >= v0)
	{
		y0 = v0 << 4;
		y1 = v1 << 4;
	}
	else
	{
		y0 = (v1 << 4) + 8;
		y1 = (v0 << 4) - 8;
	}

	output0 = vint4(y0 << 4, y0 << 4, y0 << 4, 0x7800);
	output1 = vint4(y1 << 4, y1 << 4, y1 << 4, 0x7800);
}

/**
 * @brief Unpack an HDR A direct encoding.
 *
 * @param      input     The packed endpoints (packed and modal).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void hdr_alpha_unpack(
	const uint8_t input[2],
	int& output0,
	int& output1
) {

	int v6 = input[0];
	int v7 = input[1];

	int selector = ((v6 >> 7) & 1) | ((v7 >> 6) & 2);
	v6 &= 0x7F;
	v7 &= 0x7F;
	if (selector == 3)
	{
		output0 = v6 << 5;
		output1 = v7 << 5;
	}
	else
	{
		v6 |= (v7 << (selector + 1)) & 0x780;
		v7 &= (0x3f >> selector);
		v7 ^= 32 >> selector;
		v7 -= 32 >> selector;
		v6 <<= (4 - selector);
		v7 <<= (4 - selector);
		v7 += v6;

		if (v7 < 0)
		{
			v7 = 0;
		}
		else if (v7 > 0xFFF)
		{
			v7 = 0xFFF;
		}

		output0 = v6;
		output1 = v7;
	}

	output0 <<= 4;
	output1 <<= 4;
}

/**
 * @brief Unpack an HDR RGBA direct encoding.
 *
 * @param      input     The packed endpoints (packed and modal).
 * @param[out] output0   The unpacked endpoint 0 color.
 * @param[out] output1   The unpacked endpoint 1 color.
 */
static void hdr_rgb_hdr_alpha_unpack(
	const uint8_t input[8],
	vint4& output0,
	vint4& output1
) {
	hdr_rgb_unpack(input, output0, output1);

	int alpha0, alpha1;
	hdr_alpha_unpack(input + 6, alpha0, alpha1);

	output0.set_lane<3>(alpha0);
	output1.set_lane<3>(alpha1);
}

/* See header for documentation. */
void unpack_color_endpoints(
	astcenc_profile decode_mode,
	int format,
	const uint8_t* input,
	bool& rgb_hdr,
	bool& alpha_hdr,
	vint4& output0,
	vint4& output1
) {
	// Assume no NaNs and LDR endpoints unless set later
	rgb_hdr = false;
	alpha_hdr = false;

	bool alpha_hdr_default = false;

	switch (format)
	{
	case FMT_LUMINANCE:
		luminance_unpack(input, output0, output1);
		break;

	case FMT_LUMINANCE_DELTA:
		luminance_delta_unpack(input, output0, output1);
		break;

	case FMT_HDR_LUMINANCE_SMALL_RANGE:
		rgb_hdr = true;
		alpha_hdr_default = true;
		hdr_luminance_small_range_unpack(input, output0, output1);
		break;

	case FMT_HDR_LUMINANCE_LARGE_RANGE:
		rgb_hdr = true;
		alpha_hdr_default = true;
		hdr_luminance_large_range_unpack(input, output0, output1);
		break;

	case FMT_LUMINANCE_ALPHA:
		luminance_alpha_unpack(input, output0, output1);
		break;

	case FMT_LUMINANCE_ALPHA_DELTA:
		luminance_alpha_delta_unpack(input, output0, output1);
		break;

	case FMT_RGB_SCALE:
		{
			vint4 input0q(input[0], input[1], input[2], 0);
			uint8_t scale = input[3];
			rgb_scale_unpack(input0q, scale, output0, output1);
		}
		break;

	case FMT_RGB_SCALE_ALPHA:
		{
			vint4 input0q(input[0], input[1], input[2], input[4]);
			uint8_t alpha1q = input[5];
			uint8_t scaleq = input[3];
			rgb_scale_alpha_unpack(input0q, alpha1q, scaleq, output0, output1);
		}
		break;

	case FMT_HDR_RGB_SCALE:
		rgb_hdr = true;
		alpha_hdr_default = true;
		hdr_rgbo_unpack(input, output0, output1);
		break;

	case FMT_RGB:
		{
			vint4 input0q(input[0], input[2], input[4], 0);
			vint4 input1q(input[1], input[3], input[5], 0);
			rgb_unpack(input0q, input1q, output0, output1);
		}
		break;

	case FMT_RGB_DELTA:
		{
			vint4 input0q(input[0], input[2], input[4], 0);
			vint4 input1q(input[1], input[3], input[5], 0);
			rgb_delta_unpack(input0q, input1q, output0, output1);
		}
		break;

	case FMT_HDR_RGB:
		rgb_hdr = true;
		alpha_hdr_default = true;
		hdr_rgb_unpack(input, output0, output1);
		break;

	case FMT_RGBA:
		{
			vint4 input0q(input[0], input[2], input[4], input[6]);
			vint4 input1q(input[1], input[3], input[5], input[7]);
			rgba_unpack(input0q, input1q, output0, output1);
		}
		break;

	case FMT_RGBA_DELTA:
		{
			vint4 input0q(input[0], input[2], input[4], input[6]);
			vint4 input1q(input[1], input[3], input[5], input[7]);
			rgba_delta_unpack(input0q, input1q, output0, output1);
		}
		break;

	case FMT_HDR_RGB_LDR_ALPHA:
		rgb_hdr = true;
		hdr_rgb_ldr_alpha_unpack(input, output0, output1);
		break;

	case FMT_HDR_RGBA:
		rgb_hdr = true;
		alpha_hdr = true;
		hdr_rgb_hdr_alpha_unpack(input, output0, output1);
		break;
	}

	// Assign a correct default alpha
	if (alpha_hdr_default)
	{
		if (decode_mode == ASTCENC_PRF_HDR)
		{
			output0.set_lane<3>(0x7800);
			output1.set_lane<3>(0x7800);
			alpha_hdr = true;
		}
		else
		{
			output0.set_lane<3>(0x00FF);
			output1.set_lane<3>(0x00FF);
			alpha_hdr = false;
		}
	}

	// Handle endpoint errors and expansion

	// Linear LDR 8-bit endpoints are expanded to 16-bit by replication
	if (decode_mode == ASTCENC_PRF_LDR)
	{
		// Error color - HDR endpoint in an LDR encoding
		if (rgb_hdr || alpha_hdr)
		{
			output0 = vint4(0xFF, 0x00, 0xFF, 0xFF);
			output1 = vint4(0xFF, 0x00, 0xFF, 0xFF);
			rgb_hdr = false;
			alpha_hdr = false;
		}

		output0 = output0 * 257;
		output1 = output1 * 257;
	}
	// sRGB LDR 8-bit endpoints are expanded to 16 bit by:
	//  - RGB = shift left by 8 bits and OR with 0x80
	//  - A = replication
	else if (decode_mode == ASTCENC_PRF_LDR_SRGB)
	{
		// Error color - HDR endpoint in an LDR encoding
		if (rgb_hdr || alpha_hdr)
		{
			output0 = vint4(0xFF, 0x00, 0xFF, 0xFF);
			output1 = vint4(0xFF, 0x00, 0xFF, 0xFF);
			rgb_hdr = false;
			alpha_hdr = false;
		}

		output0 = lsl<8>(output0) | vint4(0x80);
		output1 = lsl<8>(output1) | vint4(0x80);
	}
	// An HDR profile decode, but may be using linear LDR endpoints
	// Linear LDR 8-bit endpoints are expanded to 16-bit by replication
	// HDR endpoints are already 16-bit
	else
	{
		vmask4 hdr_lanes(rgb_hdr, rgb_hdr, rgb_hdr, alpha_hdr);
		vint4 output_scale = select(vint4(257), vint4(1), hdr_lanes);
		output0 = output0 * output_scale;
		output1 = output1 * output_scale;
	}
}
