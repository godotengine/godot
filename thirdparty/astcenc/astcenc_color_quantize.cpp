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
 * @brief Functions for color quantization.
 *
 * The design of the color quantization functionality requires the caller to use higher level error
 * analysis to determine the base encoding that should be used. This earlier analysis will select
 * the basic type of the endpoint that should be used:
 *
 *     * Mode: LDR or HDR
 *     * Quantization level
 *     * Channel count: L, LA, RGB, or RGBA
 *     * Endpoint 2 type: Direct color endcode, or scaled from endpoint 1.
 *
 * However, this leaves a number of decisions about exactly how to pack the endpoints open. In
 * particular we need to determine if blue contraction can be used, or/and if delta encoding can be
 * used. If they can be applied these will allow us to maintain higher precision in the endpoints
 * without needing additional storage.
 */

#include <stdio.h>
#include <assert.h>

#include "astcenc_internal.h"

/**
 * @brief Determine the quantized value given a quantization level.
 *
 * @param quant_level   The quantization level to use.
 * @param value         The value to convert. This must be in the 0-255 range.
 *
 * @return The unpacked quantized value, returned in 0-255 range.
 */
static inline uint8_t quant_color(
	quant_method quant_level,
	int value
) {
	int index = value * 2 + 1;
	return color_unquant_to_uquant_tables[quant_level - QUANT_6][index];
}

/**
 * @brief Determine the quantized value given a quantization level and residual.
 *
 * @param quant_level   The quantization level to use.
 * @param value         The value to convert. This must be in the 0-255 range.
 * @param valuef        The original value before rounding, used to compute a residual.
 *
 * @return The unpacked quantized value, returned in 0-255 range.
 */
static inline uint8_t quant_color(
	quant_method quant_level,
	int value,
	float valuef
) {
	int index = value * 2;

	// Compute the residual to determine if we should round down or up ties.
	// Test should be residual >= 0, but empirical testing shows small bias helps.
	float residual = valuef - static_cast<float>(value);
	if (residual >= -0.1f)
	{
		index++;
	}

	return color_unquant_to_uquant_tables[quant_level - QUANT_6][index];
}

/**
 * @brief Quantize an LDR RGB color.
 *
 * Since this is a fall-back encoding, we cannot actually fail but must produce a sensible result.
 * For this encoding @c color0 cannot be larger than @c color1. If @c color0 is actually larger
 * than @c color1, @c color0 is reduced and @c color1 is increased until the constraint is met.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (r0, r1, g0, g1, b0, b1).
 * @param      quant_level   The quantization level to use.
 */
static void quantize_rgb(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[6],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float r0 = astc::clamp255f(color0.lane<0>() * scale);
	float g0 = astc::clamp255f(color0.lane<1>() * scale);
	float b0 = astc::clamp255f(color0.lane<2>() * scale);

	float r1 = astc::clamp255f(color1.lane<0>() * scale);
	float g1 = astc::clamp255f(color1.lane<1>() * scale);
	float b1 = astc::clamp255f(color1.lane<2>() * scale);

	int ri0, gi0, bi0, ri1, gi1, bi1;
	float rgb0_addon = 0.0f;
	float rgb1_addon = 0.0f;
	do
	{
		ri0 = quant_color(quant_level, astc::max(astc::flt2int_rtn(r0 + rgb0_addon), 0), r0 + rgb0_addon);
		gi0 = quant_color(quant_level, astc::max(astc::flt2int_rtn(g0 + rgb0_addon), 0), g0 + rgb0_addon);
		bi0 = quant_color(quant_level, astc::max(astc::flt2int_rtn(b0 + rgb0_addon), 0), b0 + rgb0_addon);
		ri1 = quant_color(quant_level, astc::min(astc::flt2int_rtn(r1 + rgb1_addon), 255), r1 + rgb1_addon);
		gi1 = quant_color(quant_level, astc::min(astc::flt2int_rtn(g1 + rgb1_addon), 255), g1 + rgb1_addon);
		bi1 = quant_color(quant_level, astc::min(astc::flt2int_rtn(b1 + rgb1_addon), 255), b1 + rgb1_addon);

		rgb0_addon -= 0.2f;
		rgb1_addon += 0.2f;
	} while (ri0 + gi0 + bi0 > ri1 + gi1 + bi1);

	output[0] = static_cast<uint8_t>(ri0);
	output[1] = static_cast<uint8_t>(ri1);
	output[2] = static_cast<uint8_t>(gi0);
	output[3] = static_cast<uint8_t>(gi1);
	output[4] = static_cast<uint8_t>(bi0);
	output[5] = static_cast<uint8_t>(bi1);
}

/**
 * @brief Quantize an LDR RGBA color.
 *
 * Since this is a fall-back encoding, we cannot actually fail but must produce a sensible result.
 * For this encoding @c color0.rgb cannot be larger than @c color1.rgb (this indicates blue
 * contraction). If @c color0.rgb is actually larger than @c color1.rgb, @c color0.rgb is reduced
 * and @c color1.rgb is increased until the constraint is met.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (r0, r1, g0, g1, b0, b1, a0, a1).
 * @param      quant_level   The quantization level to use.
 */
static void quantize_rgba(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[8],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float a0 = astc::clamp255f(color0.lane<3>() * scale);
	float a1 = astc::clamp255f(color1.lane<3>() * scale);

	output[6] = quant_color(quant_level, astc::flt2int_rtn(a0), a0);
	output[7] = quant_color(quant_level, astc::flt2int_rtn(a1), a1);

	quantize_rgb(color0, color1, output, quant_level);
}

/**
 * @brief Try to quantize an LDR RGB color using blue-contraction.
 *
 * Blue-contraction is only usable if encoded color 1 is larger than color 0.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (r1, r0, g1, g0, b1, b0).
 * @param      quant_level   The quantization level to use.
 *
 * @return Returns @c false on failure, @c true on success.
 */
static bool try_quantize_rgb_blue_contract(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[6],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float r0 = color0.lane<0>() * scale;
	float g0 = color0.lane<1>() * scale;
	float b0 = color0.lane<2>() * scale;

	float r1 = color1.lane<0>() * scale;
	float g1 = color1.lane<1>() * scale;
	float b1 = color1.lane<2>() * scale;

	// Apply inverse blue-contraction. This can produce an overflow; which means BC cannot be used.
	r0 += (r0 - b0);
	g0 += (g0 - b0);
	r1 += (r1 - b1);
	g1 += (g1 - b1);

	if (r0 < 0.0f || r0 > 255.0f || g0 < 0.0f || g0 > 255.0f || b0 < 0.0f || b0 > 255.0f ||
		r1 < 0.0f || r1 > 255.0f || g1 < 0.0f || g1 > 255.0f || b1 < 0.0f || b1 > 255.0f)
	{
		return false;
	}

	// Quantize the inverse-blue-contracted color
	int ri0 = quant_color(quant_level, astc::flt2int_rtn(r0), r0);
	int gi0 = quant_color(quant_level, astc::flt2int_rtn(g0), g0);
	int bi0 = quant_color(quant_level, astc::flt2int_rtn(b0), b0);

	int ri1 = quant_color(quant_level, astc::flt2int_rtn(r1), r1);
	int gi1 = quant_color(quant_level, astc::flt2int_rtn(g1), g1);
	int bi1 = quant_color(quant_level, astc::flt2int_rtn(b1), b1);

	// If color #1 is not larger than color #0 then blue-contraction cannot be used. Note that
	// blue-contraction and quantization change this order, which is why we must test afterwards.
	if (ri1 + gi1 + bi1 <= ri0 + gi0 + bi0)
	{
		return false;
	}

	output[0] = static_cast<uint8_t>(ri1);
	output[1] = static_cast<uint8_t>(ri0);
	output[2] = static_cast<uint8_t>(gi1);
	output[3] = static_cast<uint8_t>(gi0);
	output[4] = static_cast<uint8_t>(bi1);
	output[5] = static_cast<uint8_t>(bi0);

	return true;
}

/**
 * @brief Try to quantize an LDR RGBA color using blue-contraction.
 *
 * Blue-contraction is only usable if encoded color 1 RGB is larger than color 0 RGB.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (r1, r0, g1, g0, b1, b0, a1, a0).
 * @param      quant_level   The quantization level to use.
 *
 * @return Returns @c false on failure, @c true on success.
 */
static bool try_quantize_rgba_blue_contract(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[8],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float a0 = astc::clamp255f(color0.lane<3>() * scale);
	float a1 = astc::clamp255f(color1.lane<3>() * scale);

	output[6] = quant_color(quant_level, astc::flt2int_rtn(a1), a1);
	output[7] = quant_color(quant_level, astc::flt2int_rtn(a0), a0);

	return try_quantize_rgb_blue_contract(color0, color1, output, quant_level);
}

/**
 * @brief Try to quantize an LDR RGB color using delta encoding.
 *
 * At decode time we move one bit from the offset to the base and seize another bit as a sign bit;
 * we then unquantize both values as if they contain one extra bit. If the sum of the offsets is
 * non-negative, then we encode a regular delta.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (r0, r1, g0, g1, b0, b1).
 * @param      quant_level   The quantization level to use.
 *
 * @return Returns @c false on failure, @c true on success.
 */
static bool try_quantize_rgb_delta(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[6],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float r0 = astc::clamp255f(color0.lane<0>() * scale);
	float g0 = astc::clamp255f(color0.lane<1>() * scale);
	float b0 = astc::clamp255f(color0.lane<2>() * scale);

	float r1 = astc::clamp255f(color1.lane<0>() * scale);
	float g1 = astc::clamp255f(color1.lane<1>() * scale);
	float b1 = astc::clamp255f(color1.lane<2>() * scale);

	// Transform r0 to unorm9
	int r0a = astc::flt2int_rtn(r0);
	int g0a = astc::flt2int_rtn(g0);
	int b0a = astc::flt2int_rtn(b0);

	r0a <<= 1;
	g0a <<= 1;
	b0a <<= 1;

	// Mask off the top bit
	int r0b = r0a & 0xFF;
	int g0b = g0a & 0xFF;
	int b0b = b0a & 0xFF;

	// Quantize then unquantize in order to get a value that we take differences against
	int r0be = quant_color(quant_level, r0b);
	int g0be = quant_color(quant_level, g0b);
	int b0be = quant_color(quant_level, b0b);

	r0b = r0be | (r0a & 0x100);
	g0b = g0be | (g0a & 0x100);
	b0b = b0be | (b0a & 0x100);

	// Get hold of the second value
	int r1d = astc::flt2int_rtn(r1);
	int g1d = astc::flt2int_rtn(g1);
	int b1d = astc::flt2int_rtn(b1);

	r1d <<= 1;
	g1d <<= 1;
	b1d <<= 1;

	// ... and take differences
	r1d -= r0b;
	g1d -= g0b;
	b1d -= b0b;

	// Check if the difference is too large to be encodable
	if (r1d > 63 || g1d > 63 || b1d > 63 || r1d < -64 || g1d < -64 || b1d < -64)
	{
		return false;
	}

	// Insert top bit of the base into the offset
	r1d &= 0x7F;
	g1d &= 0x7F;
	b1d &= 0x7F;

	r1d |= (r0b & 0x100) >> 1;
	g1d |= (g0b & 0x100) >> 1;
	b1d |= (b0b & 0x100) >> 1;

	// Then quantize and unquantize; if this causes either top two bits to flip, then encoding fails
	// since we have then corrupted either the top bit of the base or the sign bit of the offset
	int r1de = quant_color(quant_level, r1d);
	int g1de = quant_color(quant_level, g1d);
	int b1de = quant_color(quant_level, b1d);

	if (((r1d ^ r1de) | (g1d ^ g1de) | (b1d ^ b1de)) & 0xC0)
	{
		return false;
	}

	// If the sum of offsets triggers blue-contraction then encoding fails
	vint4 ep0(r0be, g0be, b0be, 0);
	vint4 ep1(r1de, g1de, b1de, 0);
	bit_transfer_signed(ep1, ep0);
	if (hadd_rgb_s(ep1) < 0)
	{
		return false;
	}

	// Check that the offsets produce legitimate sums as well
	ep0 = ep0 + ep1;
	if (any((ep0 < vint4(0)) | (ep0 > vint4(0xFF))))
	{
		return false;
	}

	output[0] = static_cast<uint8_t>(r0be);
	output[1] = static_cast<uint8_t>(r1de);
	output[2] = static_cast<uint8_t>(g0be);
	output[3] = static_cast<uint8_t>(g1de);
	output[4] = static_cast<uint8_t>(b0be);
	output[5] = static_cast<uint8_t>(b1de);

	return true;
}

static bool try_quantize_rgb_delta_blue_contract(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[6],
	quant_method quant_level
) {
	// Note: Switch around endpoint colors already at start
	float scale = 1.0f / 257.0f;

	float r1 = color0.lane<0>() * scale;
	float g1 = color0.lane<1>() * scale;
	float b1 = color0.lane<2>() * scale;

	float r0 = color1.lane<0>() * scale;
	float g0 = color1.lane<1>() * scale;
	float b0 = color1.lane<2>() * scale;

	// Apply inverse blue-contraction. This can produce an overflow; which means BC cannot be used.
	r0 += (r0 - b0);
	g0 += (g0 - b0);
	r1 += (r1 - b1);
	g1 += (g1 - b1);

	if (r0 < 0.0f || r0 > 255.0f || g0 < 0.0f || g0 > 255.0f || b0 < 0.0f || b0 > 255.0f ||
	    r1 < 0.0f || r1 > 255.0f || g1 < 0.0f || g1 > 255.0f || b1 < 0.0f || b1 > 255.0f)
	{
		return false;
	}

	// Transform r0 to unorm9
	int r0a = astc::flt2int_rtn(r0);
	int g0a = astc::flt2int_rtn(g0);
	int b0a = astc::flt2int_rtn(b0);
	r0a <<= 1;
	g0a <<= 1;
	b0a <<= 1;

	// Mask off the top bit
	int r0b = r0a & 0xFF;
	int g0b = g0a & 0xFF;
	int b0b = b0a & 0xFF;

	// Quantize, then unquantize in order to get a value that we take differences against.
	int r0be = quant_color(quant_level, r0b);
	int g0be = quant_color(quant_level, g0b);
	int b0be = quant_color(quant_level, b0b);

	r0b = r0be | (r0a & 0x100);
	g0b = g0be | (g0a & 0x100);
	b0b = b0be | (b0a & 0x100);

	// Get hold of the second value
	int r1d = astc::flt2int_rtn(r1);
	int g1d = astc::flt2int_rtn(g1);
	int b1d = astc::flt2int_rtn(b1);

	r1d <<= 1;
	g1d <<= 1;
	b1d <<= 1;

	// .. and take differences!
	r1d -= r0b;
	g1d -= g0b;
	b1d -= b0b;

	// Check if the difference is too large to be encodable
	if (r1d > 63 || g1d > 63 || b1d > 63 || r1d < -64 || g1d < -64 || b1d < -64)
	{
		return false;
	}

	// Insert top bit of the base into the offset
	r1d &= 0x7F;
	g1d &= 0x7F;
	b1d &= 0x7F;

	r1d |= (r0b & 0x100) >> 1;
	g1d |= (g0b & 0x100) >> 1;
	b1d |= (b0b & 0x100) >> 1;

	// Then quantize and unquantize; if this causes any of the top two bits to flip,
	// then encoding fails, since we have then corrupted either the top bit of the base
	// or the sign bit of the offset.
	int r1de = quant_color(quant_level, r1d);
	int g1de = quant_color(quant_level, g1d);
	int b1de = quant_color(quant_level, b1d);

	if (((r1d ^ r1de) | (g1d ^ g1de) | (b1d ^ b1de)) & 0xC0)
	{
		return false;
	}

	// If the sum of offsets does not trigger blue-contraction then encoding fails
	vint4 ep0(r0be, g0be, b0be, 0);
	vint4 ep1(r1de, g1de, b1de, 0);
	bit_transfer_signed(ep1, ep0);
	if (hadd_rgb_s(ep1) >= 0)
	{
		return false;
	}

	// Check that the offsets produce legitimate sums as well
	ep0 = ep0 + ep1;
	if (any((ep0 < vint4(0)) | (ep0 > vint4(0xFF))))
	{
		return false;
	}

	output[0] = static_cast<uint8_t>(r0be);
	output[1] = static_cast<uint8_t>(r1de);
	output[2] = static_cast<uint8_t>(g0be);
	output[3] = static_cast<uint8_t>(g1de);
	output[4] = static_cast<uint8_t>(b0be);
	output[5] = static_cast<uint8_t>(b1de);

	return true;
}

/**
 * @brief Try to quantize an LDR A color using delta encoding.
 *
 * At decode time we move one bit from the offset to the base and seize another bit as a sign bit;
 * we then unquantize both values as if they contain one extra bit. If the sum of the offsets is
 * non-negative, then we encode a regular delta.
 *
 * This function only compressed the alpha - the other elements in the output array are not touched.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (x, x, x, x, x, x, a0, a1).
 * @param      quant_level   The quantization level to use.
 *
 * @return Returns @c false on failure, @c true on success.
 */
static bool try_quantize_alpha_delta(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[8],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float a0 = astc::clamp255f(color0.lane<3>() * scale);
	float a1 = astc::clamp255f(color1.lane<3>() * scale);

	int a0a = astc::flt2int_rtn(a0);
	a0a <<= 1;
	int a0b = a0a & 0xFF;
	int a0be = quant_color(quant_level, a0b);
	a0b = a0be;
	a0b |= a0a & 0x100;
	int a1d = astc::flt2int_rtn(a1);
	a1d <<= 1;
	a1d -= a0b;

	if (a1d > 63 || a1d < -64)
	{
		return false;
	}

	a1d &= 0x7F;
	a1d |= (a0b & 0x100) >> 1;

	int a1de = quant_color(quant_level, a1d);
	int a1du = a1de;
	if ((a1d ^ a1du) & 0xC0)
	{
		return false;
	}

	a1du &= 0x7F;
	if (a1du & 0x40)
	{
		a1du -= 0x80;
	}

	a1du += a0b;
	if (a1du < 0 || a1du > 0x1FF)
	{
		return false;
	}

	output[6] = static_cast<uint8_t>(a0be);
	output[7] = static_cast<uint8_t>(a1de);

	return true;
}

/**
 * @brief Try to quantize an LDR LA color using delta encoding.
 *
 * At decode time we move one bit from the offset to the base and seize another bit as a sign bit;
 * we then unquantize both values as if they contain one extra bit. If the sum of the offsets is
 * non-negative, then we encode a regular delta.
 *
 * This function only compressed the alpha - the other elements in the output array are not touched.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (l0, l1, a0, a1).
 * @param      quant_level   The quantization level to use.
 *
 * @return Returns @c false on failure, @c true on success.
 */
static bool try_quantize_luminance_alpha_delta(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[4],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float l0 = astc::clamp255f(hadd_rgb_s(color0) * ((1.0f / 3.0f) * scale));
	float l1 = astc::clamp255f(hadd_rgb_s(color1) * ((1.0f / 3.0f) * scale));

	float a0 = astc::clamp255f(color0.lane<3>() * scale);
	float a1 = astc::clamp255f(color1.lane<3>() * scale);

	int l0a = astc::flt2int_rtn(l0);
	int a0a = astc::flt2int_rtn(a0);
	l0a <<= 1;
	a0a <<= 1;

	int l0b = l0a & 0xFF;
	int a0b = a0a & 0xFF;
	int l0be = quant_color(quant_level, l0b);
	int a0be = quant_color(quant_level, a0b);
	l0b = l0be;
	a0b = a0be;
	l0b |= l0a & 0x100;
	a0b |= a0a & 0x100;

	int l1d = astc::flt2int_rtn(l1);
	int a1d = astc::flt2int_rtn(a1);
	l1d <<= 1;
	a1d <<= 1;
	l1d -= l0b;
	a1d -= a0b;

	if (l1d > 63 || l1d < -64)
	{
		return false;
	}

	if (a1d > 63 || a1d < -64)
	{
		return false;
	}

	l1d &= 0x7F;
	a1d &= 0x7F;
	l1d |= (l0b & 0x100) >> 1;
	a1d |= (a0b & 0x100) >> 1;

	int l1de = quant_color(quant_level, l1d);
	int a1de = quant_color(quant_level, a1d);
	int l1du = l1de;
	int a1du = a1de;

	if ((l1d ^ l1du) & 0xC0)
	{
		return false;
	}

	if ((a1d ^ a1du) & 0xC0)
	{
		return false;
	}

	l1du &= 0x7F;
	a1du &= 0x7F;

	if (l1du & 0x40)
	{
		l1du -= 0x80;
	}

	if (a1du & 0x40)
	{
		a1du -= 0x80;
	}

	l1du += l0b;
	a1du += a0b;

	if (l1du < 0 || l1du > 0x1FF)
	{
		return false;
	}

	if (a1du < 0 || a1du > 0x1FF)
	{
		return false;
	}

	output[0] = static_cast<uint8_t>(l0be);
	output[1] = static_cast<uint8_t>(l1de);
	output[2] = static_cast<uint8_t>(a0be);
	output[3] = static_cast<uint8_t>(a1de);

	return true;
}

/**
 * @brief Try to quantize an LDR RGBA color using delta encoding.
 *
 * At decode time we move one bit from the offset to the base and seize another bit as a sign bit;
 * we then unquantize both values as if they contain one extra bit. If the sum of the offsets is
 * non-negative, then we encode a regular delta.
 *
 * This function only compressed the alpha - the other elements in the output array are not touched.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (r0, r1, b0, b1, g0, g1, a0, a1).
 * @param      quant_level   The quantization level to use.
 *
 * @return Returns @c false on failure, @c true on success.
 */
static bool try_quantize_rgba_delta(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[8],
	quant_method quant_level
) {
	return try_quantize_rgb_delta(color0, color1, output, quant_level) &&
	       try_quantize_alpha_delta(color0, color1, output, quant_level);
}


/**
 * @brief Try to quantize an LDR RGBA color using delta and blue contract encoding.
 *
 * At decode time we move one bit from the offset to the base and seize another bit as a sign bit;
 * we then unquantize both values as if they contain one extra bit. If the sum of the offsets is
 * non-negative, then we encode a regular delta.
 *
 * This function only compressed the alpha - the other elements in the output array are not touched.
 *
 * @param      color0       The input unquantized color0 endpoint.
 * @param      color1       The input unquantized color1 endpoint.
 * @param[out] output       The output endpoints, returned as (r0, r1, b0, b1, g0, g1, a0, a1).
 * @param      quant_level  The quantization level to use.
 *
 * @return Returns @c false on failure, @c true on success.
 */
static bool try_quantize_rgba_delta_blue_contract(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[8],
	quant_method quant_level
) {
	// Note that we swap the color0 and color1 ordering for alpha to match RGB blue-contract
	return try_quantize_rgb_delta_blue_contract(color0, color1, output, quant_level) &&
	       try_quantize_alpha_delta(color1, color0, output, quant_level);
}

/**
 * @brief Quantize an LDR RGB color using scale encoding.
 *
 * @param      color         The input unquantized color endpoint and scale factor.
 * @param[out] output        The output endpoints, returned as (r0, g0, b0, s).
 * @param      quant_level   The quantization level to use.
 */
static void quantize_rgbs(
	vfloat4 color,
	uint8_t output[4],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float r = astc::clamp255f(color.lane<0>() * scale);
	float g = astc::clamp255f(color.lane<1>() * scale);
	float b = astc::clamp255f(color.lane<2>() * scale);

	int ri = quant_color(quant_level, astc::flt2int_rtn(r), r);
	int gi = quant_color(quant_level, astc::flt2int_rtn(g), g);
	int bi = quant_color(quant_level, astc::flt2int_rtn(b), b);

	float oldcolorsum = hadd_rgb_s(color) * scale;
	float newcolorsum = static_cast<float>(ri + gi + bi);

	float scalea = astc::clamp1f(color.lane<3>() * (oldcolorsum + 1e-10f) / (newcolorsum + 1e-10f));
	int scale_idx = astc::flt2int_rtn(scalea * 256.0f);
	scale_idx = astc::clamp(scale_idx, 0, 255);

	output[0] = static_cast<uint8_t>(ri);
	output[1] = static_cast<uint8_t>(gi);
	output[2] = static_cast<uint8_t>(bi);
	output[3] = quant_color(quant_level, scale_idx);
}

/**
 * @brief Quantize an LDR RGBA color using scale encoding.
 *
 * @param      color        The input unquantized color endpoint and scale factor.
 * @param[out] output       The output endpoints, returned as (r0, g0, b0, s, a0, a1).
 * @param      quant_level  The quantization level to use.
 */
static void quantize_rgbs_alpha(
	vfloat4 color0,
	vfloat4 color1,
	vfloat4 color,
	uint8_t output[6],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float a0 = astc::clamp255f(color0.lane<3>() * scale);
	float a1 = astc::clamp255f(color1.lane<3>() * scale);

	output[4] = quant_color(quant_level, astc::flt2int_rtn(a0), a0);
	output[5] = quant_color(quant_level, astc::flt2int_rtn(a1), a1);

	quantize_rgbs(color, output, quant_level);
}

/**
 * @brief Quantize a LDR L color.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (l0, l1).
 * @param      quant_level   The quantization level to use.
 */
static void quantize_luminance(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[2],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	color0 = color0 * scale;
	color1 = color1 * scale;

	float lum0 = astc::clamp255f(hadd_rgb_s(color0) * (1.0f / 3.0f));
	float lum1 = astc::clamp255f(hadd_rgb_s(color1) * (1.0f / 3.0f));

	if (lum0 > lum1)
	{
		float avg = (lum0 + lum1) * 0.5f;
		lum0 = avg;
		lum1 = avg;
	}

	output[0] = quant_color(quant_level, astc::flt2int_rtn(lum0), lum0);
	output[1] = quant_color(quant_level, astc::flt2int_rtn(lum1), lum1);
}

/**
 * @brief Quantize a LDR LA color.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (l0, l1, a0, a1).
 * @param      quant_level   The quantization level to use.
 */
static void quantize_luminance_alpha(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[4],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	color0 = color0 * scale;
	color1 = color1 * scale;

	float lum0 = astc::clamp255f(hadd_rgb_s(color0) * (1.0f / 3.0f));
	float lum1 = astc::clamp255f(hadd_rgb_s(color1) * (1.0f / 3.0f));

	float a0 = astc::clamp255f(color0.lane<3>());
	float a1 = astc::clamp255f(color1.lane<3>());

	output[0] = quant_color(quant_level, astc::flt2int_rtn(lum0), lum0);
	output[1] = quant_color(quant_level, astc::flt2int_rtn(lum1), lum1);
	output[2] = quant_color(quant_level, astc::flt2int_rtn(a0), a0);
	output[3] = quant_color(quant_level, astc::flt2int_rtn(a1), a1);
}

/**
 * @brief Quantize and unquantize a value ensuring top two bits are the same.
 *
 * @param      quant_level     The quantization level to use.
 * @param      value           The input unquantized value.
 * @param[out] quant_value     The quantized value.
 */
static inline void quantize_and_unquantize_retain_top_two_bits(
	quant_method quant_level,
	uint8_t value,
	uint8_t& quant_value
) {
	int perform_loop;
	uint8_t quantval;

	do
	{
		quantval = quant_color(quant_level, value);

		// Perform looping if the top two bits were modified by quant/unquant
		perform_loop = (value & 0xC0) != (quantval & 0xC0);

		if ((quantval & 0xC0) > (value & 0xC0))
		{
			// Quant/unquant rounded UP so that the top two bits changed;
			// decrement the input in hopes that this will avoid rounding up.
			value--;
		}
		else if ((quantval & 0xC0) < (value & 0xC0))
		{
			// Quant/unquant rounded DOWN so that the top two bits changed;
			// decrement the input in hopes that this will avoid rounding down.
			value--;
		}
	} while (perform_loop);

	quant_value = quantval;
}

/**
 * @brief Quantize and unquantize a value ensuring top four bits are the same.
 *
 * @param      quant_level     The quantization level to use.
 * @param      value           The input unquantized value.
 * @param[out] quant_value     The quantized value in 0-255 range.
 */
static inline void quantize_and_unquantize_retain_top_four_bits(
	quant_method quant_level,
	uint8_t value,
	uint8_t& quant_value
) {
	uint8_t perform_loop;
	uint8_t quantval;

	do
	{
		quantval = quant_color(quant_level, value);
		// Perform looping if the top four bits were modified by quant/unquant
		perform_loop = (value & 0xF0) != (quantval & 0xF0);

		if ((quantval & 0xF0) > (value & 0xF0))
		{
			// Quant/unquant rounded UP so that the top four bits changed;
			// decrement the input value in hopes that this will avoid rounding up.
			value--;
		}
		else if ((quantval & 0xF0) < (value & 0xF0))
		{
			// Quant/unquant rounded DOWN so that the top four bits changed;
			// decrement the input value in hopes that this will avoid rounding down.
			value--;
		}
	} while (perform_loop);

	quant_value = quantval;
}

/**
 * @brief Quantize a HDR RGB color using RGB + offset.
 *
 * @param      color         The input unquantized color endpoint and offset.
 * @param[out] output        The output endpoints, returned as packed RGBS with some mode bits.
 * @param      quant_level   The quantization level to use.
 */
static void quantize_hdr_rgbo(
	vfloat4 color,
	uint8_t output[4],
	quant_method quant_level
) {
	color.set_lane<0>(color.lane<0>() + color.lane<3>());
	color.set_lane<1>(color.lane<1>() + color.lane<3>());
	color.set_lane<2>(color.lane<2>() + color.lane<3>());

	color = clamp(0.0f, 65535.0f, color);

	vfloat4 color_bak = color;

	int majcomp;
	if (color.lane<0>() > color.lane<1>() && color.lane<0>() > color.lane<2>())
	{
		majcomp = 0;			// red is largest component
	}
	else if (color.lane<1>() > color.lane<2>())
	{
		majcomp = 1;			// green is largest component
	}
	else
	{
		majcomp = 2;			// blue is largest component
	}

	// swap around the red component and the largest component.
	switch (majcomp)
	{
	case 1:
		color = color.swz<1, 0, 2, 3>();
		break;
	case 2:
		color = color.swz<2, 1, 0, 3>();
		break;
	default:
		break;
	}

	static const int mode_bits[5][3] {
		{11, 5, 7},
		{11, 6, 5},
		{10, 5, 8},
		{9, 6, 7},
		{8, 7, 6}
	};

	static const float mode_cutoffs[5][2] {
		{1024, 4096},
		{2048, 1024},
		{2048, 16384},
		{8192, 16384},
		{32768, 16384}
	};

	static const float mode_rscales[5] {
		32.0f,
		32.0f,
		64.0f,
		128.0f,
		256.0f,
	};

	static const float mode_scales[5] {
		1.0f / 32.0f,
		1.0f / 32.0f,
		1.0f / 64.0f,
		1.0f / 128.0f,
		1.0f / 256.0f,
	};

	float r_base = color.lane<0>();
	float g_base = color.lane<0>() - color.lane<1>() ;
	float b_base = color.lane<0>() - color.lane<2>() ;
	float s_base = color.lane<3>() ;

	for (int mode = 0; mode < 5; mode++)
	{
		if (g_base > mode_cutoffs[mode][0] || b_base > mode_cutoffs[mode][0] || s_base > mode_cutoffs[mode][1])
		{
			continue;
		}

		// Encode the mode into a 4-bit vector
		int mode_enc = mode < 4 ? (mode | (majcomp << 2)) : (majcomp | 0xC);

		float mode_scale = mode_scales[mode];
		float mode_rscale = mode_rscales[mode];

		int gb_intcutoff = 1 << mode_bits[mode][1];
		int s_intcutoff = 1 << mode_bits[mode][2];

		// Quantize and unquantize R
		int r_intval = astc::flt2int_rtn(r_base * mode_scale);

		int r_lowbits = r_intval & 0x3f;

		r_lowbits |= (mode_enc & 3) << 6;

		uint8_t r_quantval;
		quantize_and_unquantize_retain_top_two_bits(
		    quant_level, static_cast<uint8_t>(r_lowbits), r_quantval);

		r_intval = (r_intval & ~0x3f) | (r_quantval & 0x3f);
		float r_fval = static_cast<float>(r_intval) * mode_rscale;

		// Recompute G and B, then quantize and unquantize them
		float g_fval = r_fval - color.lane<1>() ;
		float b_fval = r_fval - color.lane<2>() ;

		g_fval = astc::clamp(g_fval, 0.0f, 65535.0f);
		b_fval = astc::clamp(b_fval, 0.0f, 65535.0f);

		int g_intval = astc::flt2int_rtn(g_fval * mode_scale);
		int b_intval = astc::flt2int_rtn(b_fval * mode_scale);

		if (g_intval >= gb_intcutoff || b_intval >= gb_intcutoff)
		{
			continue;
		}

		int g_lowbits = g_intval & 0x1f;
		int b_lowbits = b_intval & 0x1f;

		int bit0 = 0;
		int bit1 = 0;
		int bit2 = 0;
		int bit3 = 0;

		switch (mode)
		{
		case 0:
		case 2:
			bit0 = (r_intval >> 9) & 1;
			break;
		case 1:
		case 3:
			bit0 = (r_intval >> 8) & 1;
			break;
		case 4:
		case 5:
			bit0 = (g_intval >> 6) & 1;
			break;
		}

		switch (mode)
		{
		case 0:
		case 1:
		case 2:
		case 3:
			bit2 = (r_intval >> 7) & 1;
			break;
		case 4:
		case 5:
			bit2 = (b_intval >> 6) & 1;
			break;
		}

		switch (mode)
		{
		case 0:
		case 2:
			bit1 = (r_intval >> 8) & 1;
			break;
		case 1:
		case 3:
		case 4:
		case 5:
			bit1 = (g_intval >> 5) & 1;
			break;
		}

		switch (mode)
		{
		case 0:
			bit3 = (r_intval >> 10) & 1;
			break;
		case 2:
			bit3 = (r_intval >> 6) & 1;
			break;
		case 1:
		case 3:
		case 4:
		case 5:
			bit3 = (b_intval >> 5) & 1;
			break;
		}

		g_lowbits |= (mode_enc & 0x4) << 5;
		b_lowbits |= (mode_enc & 0x8) << 4;

		g_lowbits |= bit0 << 6;
		g_lowbits |= bit1 << 5;
		b_lowbits |= bit2 << 6;
		b_lowbits |= bit3 << 5;

		uint8_t g_quantval;
		uint8_t b_quantval;

		quantize_and_unquantize_retain_top_four_bits(
		    quant_level, static_cast<uint8_t>(g_lowbits), g_quantval);
		quantize_and_unquantize_retain_top_four_bits(
		    quant_level, static_cast<uint8_t>(b_lowbits), b_quantval);

		g_intval = (g_intval & ~0x1f) | (g_quantval & 0x1f);
		b_intval = (b_intval & ~0x1f) | (b_quantval & 0x1f);

		g_fval = static_cast<float>(g_intval) * mode_rscale;
		b_fval = static_cast<float>(b_intval) * mode_rscale;

		// Recompute the scale value, based on the errors introduced to red, green and blue

		// If the error is positive, then the R,G,B errors combined have raised the color
		// value overall; as such, the scale value needs to be increased.
		float rgb_errorsum = (r_fval - color.lane<0>() ) + (r_fval - g_fval - color.lane<1>() ) + (r_fval - b_fval - color.lane<2>() );

		float s_fval = s_base + rgb_errorsum * (1.0f / 3.0f);
		s_fval = astc::clamp(s_fval, 0.0f, 1e9f);

		int s_intval = astc::flt2int_rtn(s_fval * mode_scale);

		if (s_intval >= s_intcutoff)
		{
			continue;
		}

		int s_lowbits = s_intval & 0x1f;

		int bit4;
		int bit5;
		int bit6;
		switch (mode)
		{
		case 1:
			bit6 = (r_intval >> 9) & 1;
			break;
		default:
			bit6 = (s_intval >> 5) & 1;
			break;
		}

		switch (mode)
		{
		case 4:
			bit5 = (r_intval >> 7) & 1;
			break;
		case 1:
			bit5 = (r_intval >> 10) & 1;
			break;
		default:
			bit5 = (s_intval >> 6) & 1;
			break;
		}

		switch (mode)
		{
		case 2:
			bit4 = (s_intval >> 7) & 1;
			break;
		default:
			bit4 = (r_intval >> 6) & 1;
			break;
		}

		s_lowbits |= bit6 << 5;
		s_lowbits |= bit5 << 6;
		s_lowbits |= bit4 << 7;

		uint8_t s_quantval;

		quantize_and_unquantize_retain_top_four_bits(
		    quant_level, static_cast<uint8_t>(s_lowbits), s_quantval);

		output[0] = r_quantval;
		output[1] = g_quantval;
		output[2] = b_quantval;
		output[3] = s_quantval;
		return;
	}

	// Failed to encode any of the modes above? In that case encode using mode #5
	float vals[4];
	vals[0] = color_bak.lane<0>();
	vals[1] = color_bak.lane<1>();
	vals[2] = color_bak.lane<2>();
	vals[3] = color_bak.lane<3>();

	int ivals[4];
	float cvals[3];

	for (int i = 0; i < 3; i++)
	{
		vals[i] = astc::clamp(vals[i], 0.0f, 65020.0f);
		ivals[i] = astc::flt2int_rtn(vals[i] * (1.0f / 512.0f));
		cvals[i] = static_cast<float>(ivals[i]) * 512.0f;
	}

	float rgb_errorsum = (cvals[0] - vals[0]) + (cvals[1] - vals[1]) + (cvals[2] - vals[2]);
	vals[3] += rgb_errorsum * (1.0f / 3.0f);

	vals[3] = astc::clamp(vals[3], 0.0f, 65020.0f);
	ivals[3] = astc::flt2int_rtn(vals[3] * (1.0f / 512.0f));

	int encvals[4];
	encvals[0] = (ivals[0] & 0x3f) | 0xC0;
	encvals[1] = (ivals[1] & 0x7f) | 0x80;
	encvals[2] = (ivals[2] & 0x7f) | 0x80;
	encvals[3] = (ivals[3] & 0x7f) | ((ivals[0] & 0x40) << 1);

	for (uint8_t i = 0; i < 4; i++)
	{
		quantize_and_unquantize_retain_top_four_bits(
		    quant_level, static_cast<uint8_t>(encvals[i]), output[i]);
	}

	return;
}

/**
 * @brief Quantize a HDR RGB color using direct RGB encoding.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as packed RGB+RGB pairs with mode bits.
 * @param      quant_level   The quantization level to use.
 */
static void quantize_hdr_rgb(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[6],
	quant_method quant_level
) {
	// Note: color*.lane<3> is not used so we can ignore it
	color0 = clamp(0.0f, 65535.0f, color0);
	color1 = clamp(0.0f, 65535.0f, color1);

	vfloat4 color0_bak = color0;
	vfloat4 color1_bak = color1;

	int majcomp;
	if (color1.lane<0>() > color1.lane<1>() && color1.lane<0>() > color1.lane<2>())
	{
		majcomp = 0;
	}
	else if (color1.lane<1>() > color1.lane<2>())
	{
		majcomp = 1;
	}
	else
	{
		majcomp = 2;
	}

	// Swizzle the components
	switch (majcomp)
	{
	case 1:  // red-green swap
		color0 = color0.swz<1, 0, 2, 3>();
		color1 = color1.swz<1, 0, 2, 3>();
		break;
	case 2:  // red-blue swap
		color0 = color0.swz<2, 1, 0, 3>();
		color1 = color1.swz<2, 1, 0, 3>();
		break;
	default:
		break;
	}

	float a_base = color1.lane<0>();
	a_base = astc::clamp(a_base, 0.0f, 65535.0f);

	float b0_base = a_base - color1.lane<1>();
	float b1_base = a_base - color1.lane<2>();
	float c_base = a_base - color0.lane<0>();
	float d0_base = a_base - b0_base - c_base - color0.lane<1>();
	float d1_base = a_base - b1_base - c_base - color0.lane<2>();

	// Number of bits in the various fields in the various modes
	static const int mode_bits[8][4] {
		{9, 7, 6, 7},
		{9, 8, 6, 6},
		{10, 6, 7, 7},
		{10, 7, 7, 6},
		{11, 8, 6, 5},
		{11, 6, 8, 6},
		{12, 7, 7, 5},
		{12, 6, 7, 6}
	};

	// Cutoffs to use for the computed values of a,b,c,d, assuming the
	// range 0..65535 are LNS values corresponding to fp16.
	static const float mode_cutoffs[8][4] {
		{16384, 8192, 8192, 8},	// mode 0: 9,7,6,7
		{32768, 8192, 4096, 8},	// mode 1: 9,8,6,6
		{4096, 8192, 4096, 4},	// mode 2: 10,6,7,7
		{8192, 8192, 2048, 4},	// mode 3: 10,7,7,6
		{8192, 2048, 512, 2},	// mode 4: 11,8,6,5
		{2048, 8192, 1024, 2},	// mode 5: 11,6,8,6
		{2048, 2048, 256, 1},	// mode 6: 12,7,7,5
		{1024, 2048, 512, 1},	// mode 7: 12,6,7,6
	};

	static const float mode_scales[8] {
		1.0f / 128.0f,
		1.0f / 128.0f,
		1.0f / 64.0f,
		1.0f / 64.0f,
		1.0f / 32.0f,
		1.0f / 32.0f,
		1.0f / 16.0f,
		1.0f / 16.0f,
	};

	// Scaling factors when going from what was encoded in the mode to 16 bits.
	static const float mode_rscales[8] {
		128.0f,
		128.0f,
		64.0f,
		64.0f,
		32.0f,
		32.0f,
		16.0f,
		16.0f
	};

	// Try modes one by one, with the highest-precision mode first.
	for (int mode = 7; mode >= 0; mode--)
	{
		// For each mode, test if we can in fact accommodate the computed b, c, and d values.
		// If we clearly can't, then we skip to the next mode.

		float b_cutoff = mode_cutoffs[mode][0];
		float c_cutoff = mode_cutoffs[mode][1];
		float d_cutoff = mode_cutoffs[mode][2];

		if (b0_base > b_cutoff || b1_base > b_cutoff || c_base > c_cutoff || fabsf(d0_base) > d_cutoff || fabsf(d1_base) > d_cutoff)
		{
			continue;
		}

		float mode_scale = mode_scales[mode];
		float mode_rscale = mode_rscales[mode];

		int b_intcutoff = 1 << mode_bits[mode][1];
		int c_intcutoff = 1 << mode_bits[mode][2];
		int d_intcutoff = 1 << (mode_bits[mode][3] - 1);

		// Quantize and unquantize A, with the assumption that its high bits can be handled safely.
		int a_intval = astc::flt2int_rtn(a_base * mode_scale);
		int a_lowbits = a_intval & 0xFF;

		int a_quantval = quant_color(quant_level, a_lowbits);
		int a_uquantval = a_quantval;
		a_intval = (a_intval & ~0xFF) | a_uquantval;
		float a_fval = static_cast<float>(a_intval) * mode_rscale;

		// Recompute C, then quantize and unquantize it
		float c_fval = a_fval - color0.lane<0>();
		c_fval = astc::clamp(c_fval, 0.0f, 65535.0f);

		int c_intval = astc::flt2int_rtn(c_fval * mode_scale);

		if (c_intval >= c_intcutoff)
		{
			continue;
		}

		int c_lowbits = c_intval & 0x3f;

		c_lowbits |= (mode & 1) << 7;
		c_lowbits |= (a_intval & 0x100) >> 2;

		uint8_t c_quantval;

		quantize_and_unquantize_retain_top_two_bits(
		    quant_level, static_cast<uint8_t>(c_lowbits), c_quantval);

		c_intval = (c_intval & ~0x3F) | (c_quantval & 0x3F);
		c_fval = static_cast<float>(c_intval) * mode_rscale;

		// Recompute B0 and B1, then quantize and unquantize them
		float b0_fval = a_fval - color1.lane<1>();
		float b1_fval = a_fval - color1.lane<2>();

		b0_fval = astc::clamp(b0_fval, 0.0f, 65535.0f);
		b1_fval = astc::clamp(b1_fval, 0.0f, 65535.0f);
		int b0_intval = astc::flt2int_rtn(b0_fval * mode_scale);
		int b1_intval = astc::flt2int_rtn(b1_fval * mode_scale);

		if (b0_intval >= b_intcutoff || b1_intval >= b_intcutoff)
		{
			continue;
		}

		int b0_lowbits = b0_intval & 0x3f;
		int b1_lowbits = b1_intval & 0x3f;

		int bit0 = 0;
		int bit1 = 0;
		switch (mode)
		{
		case 0:
		case 1:
		case 3:
		case 4:
		case 6:
			bit0 = (b0_intval >> 6) & 1;
			break;
		case 2:
		case 5:
		case 7:
			bit0 = (a_intval >> 9) & 1;
			break;
		}

		switch (mode)
		{
		case 0:
		case 1:
		case 3:
		case 4:
		case 6:
			bit1 = (b1_intval >> 6) & 1;
			break;
		case 2:
			bit1 = (c_intval >> 6) & 1;
			break;
		case 5:
		case 7:
			bit1 = (a_intval >> 10) & 1;
			break;
		}

		b0_lowbits |= bit0 << 6;
		b1_lowbits |= bit1 << 6;

		b0_lowbits |= ((mode >> 1) & 1) << 7;
		b1_lowbits |= ((mode >> 2) & 1) << 7;

		uint8_t b0_quantval;
		uint8_t b1_quantval;

		quantize_and_unquantize_retain_top_two_bits(
		    quant_level, static_cast<uint8_t>(b0_lowbits), b0_quantval);
		quantize_and_unquantize_retain_top_two_bits(
		    quant_level, static_cast<uint8_t>(b1_lowbits), b1_quantval);

		b0_intval = (b0_intval & ~0x3f) | (b0_quantval & 0x3f);
		b1_intval = (b1_intval & ~0x3f) | (b1_quantval & 0x3f);
		b0_fval = static_cast<float>(b0_intval) * mode_rscale;
		b1_fval = static_cast<float>(b1_intval) * mode_rscale;

		// Recompute D0 and D1, then quantize and unquantize them
		float d0_fval = a_fval - b0_fval - c_fval - color0.lane<1>();
		float d1_fval = a_fval - b1_fval - c_fval - color0.lane<2>();

		d0_fval = astc::clamp(d0_fval, -65535.0f, 65535.0f);
		d1_fval = astc::clamp(d1_fval, -65535.0f, 65535.0f);

		int d0_intval = astc::flt2int_rtn(d0_fval * mode_scale);
		int d1_intval = astc::flt2int_rtn(d1_fval * mode_scale);

		if (abs(d0_intval) >= d_intcutoff || abs(d1_intval) >= d_intcutoff)
		{
			continue;
		}

		int d0_lowbits = d0_intval & 0x1f;
		int d1_lowbits = d1_intval & 0x1f;

		int bit2 = 0;
		int bit3 = 0;
		int bit4;
		int bit5;
		switch (mode)
		{
		case 0:
		case 2:
			bit2 = (d0_intval >> 6) & 1;
			break;
		case 1:
		case 4:
			bit2 = (b0_intval >> 7) & 1;
			break;
		case 3:
			bit2 = (a_intval >> 9) & 1;
			break;
		case 5:
			bit2 = (c_intval >> 7) & 1;
			break;
		case 6:
		case 7:
			bit2 = (a_intval >> 11) & 1;
			break;
		}
		switch (mode)
		{
		case 0:
		case 2:
			bit3 = (d1_intval >> 6) & 1;
			break;
		case 1:
		case 4:
			bit3 = (b1_intval >> 7) & 1;
			break;
		case 3:
		case 5:
		case 6:
		case 7:
			bit3 = (c_intval >> 6) & 1;
			break;
		}

		switch (mode)
		{
		case 4:
		case 6:
			bit4 = (a_intval >> 9) & 1;
			bit5 = (a_intval >> 10) & 1;
			break;
		default:
			bit4 = (d0_intval >> 5) & 1;
			bit5 = (d1_intval >> 5) & 1;
			break;
		}

		d0_lowbits |= bit2 << 6;
		d1_lowbits |= bit3 << 6;
		d0_lowbits |= bit4 << 5;
		d1_lowbits |= bit5 << 5;

		d0_lowbits |= (majcomp & 1) << 7;
		d1_lowbits |= ((majcomp >> 1) & 1) << 7;

		uint8_t d0_quantval;
		uint8_t d1_quantval;

		quantize_and_unquantize_retain_top_four_bits(
		    quant_level, static_cast<uint8_t>(d0_lowbits), d0_quantval);
		quantize_and_unquantize_retain_top_four_bits(
		    quant_level, static_cast<uint8_t>(d1_lowbits), d1_quantval);

		output[0] = static_cast<uint8_t>(a_quantval);
		output[1] = c_quantval;
		output[2] = b0_quantval;
		output[3] = b1_quantval;
		output[4] = d0_quantval;
		output[5] = d1_quantval;
		return;
	}

	// If neither of the modes fit we will use a flat representation for storing data, using 8 bits
	// for red and green, and 7 bits for blue. This gives color accuracy roughly similar to LDR
	// 4:4:3 which is not at all great but usable. This representation is used if the light color is
	// more than 4x the color value of the dark color.
	float vals[6];
	vals[0] = color0_bak.lane<0>();
	vals[1] = color1_bak.lane<0>();
	vals[2] = color0_bak.lane<1>();
	vals[3] = color1_bak.lane<1>();
	vals[4] = color0_bak.lane<2>();
	vals[5] = color1_bak.lane<2>();

	for (int i = 0; i < 6; i++)
	{
		vals[i] = astc::clamp(vals[i], 0.0f, 65020.0f);
	}

	for (int i = 0; i < 4; i++)
	{
		int idx = astc::flt2int_rtn(vals[i] * 1.0f / 256.0f);
		output[i] = quant_color(quant_level, idx);
	}

	for (int i = 4; i < 6; i++)
	{
		int idx = astc::flt2int_rtn(vals[i] * 1.0f / 512.0f) + 128;
		quantize_and_unquantize_retain_top_two_bits(
		    quant_level, static_cast<uint8_t>(idx), output[i]);
	}

	return;
}

/**
 * @brief Quantize a HDR RGB + LDR A color using direct RGBA encoding.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as packed RGBA+RGBA pairs with mode bits.
 * @param      quant_level   The quantization level to use.
 */
static void quantize_hdr_rgb_ldr_alpha(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[8],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float a0 = astc::clamp255f(color0.lane<3>() * scale);
	float a1 = astc::clamp255f(color1.lane<3>() * scale);

	output[6] = quant_color(quant_level, astc::flt2int_rtn(a0), a0);
	output[7] = quant_color(quant_level, astc::flt2int_rtn(a1), a1);

	quantize_hdr_rgb(color0, color1, output, quant_level);
}

/**
 * @brief Quantize a HDR L color using the large range encoding.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as packed (l0, l1).
 * @param      quant_level   The quantization level to use.
 */
static void quantize_hdr_luminance_large_range(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[2],
	quant_method quant_level
) {
	float lum0 = hadd_rgb_s(color0) * (1.0f / 3.0f);
	float lum1 = hadd_rgb_s(color1) * (1.0f / 3.0f);

	if (lum1 < lum0)
	{
		float avg = (lum0 + lum1) * 0.5f;
		lum0 = avg;
		lum1 = avg;
	}

	int ilum1 = astc::flt2int_rtn(lum1);
	int ilum0 = astc::flt2int_rtn(lum0);

	// Find the closest encodable point in the upper half of the code-point space
	int upper_v0 = (ilum0 + 128) >> 8;
	int upper_v1 = (ilum1 + 128) >> 8;

	upper_v0 = astc::clamp(upper_v0, 0, 255);
	upper_v1 = astc::clamp(upper_v1, 0, 255);

	// Find the closest encodable point in the lower half of the code-point space
	int lower_v0 = (ilum1 + 256) >> 8;
	int lower_v1 = ilum0 >> 8;

	lower_v0 = astc::clamp(lower_v0, 0, 255);
	lower_v1 = astc::clamp(lower_v1, 0, 255);

	// Determine the distance between the point in code-point space and the input value
	int upper0_dec = upper_v0 << 8;
	int upper1_dec = upper_v1 << 8;
	int lower0_dec = (lower_v1 << 8) + 128;
	int lower1_dec = (lower_v0 << 8) - 128;

	int upper0_diff = upper0_dec - ilum0;
	int upper1_diff = upper1_dec - ilum1;
	int lower0_diff = lower0_dec - ilum0;
	int lower1_diff = lower1_dec - ilum1;

	int upper_error = (upper0_diff * upper0_diff) + (upper1_diff * upper1_diff);
	int lower_error = (lower0_diff * lower0_diff) + (lower1_diff * lower1_diff);

	int v0, v1;
	if (upper_error < lower_error)
	{
		v0 = upper_v0;
		v1 = upper_v1;
	}
	else
	{
		v0 = lower_v0;
		v1 = lower_v1;
	}

	// OK; encode
	output[0] = quant_color(quant_level, v0);
	output[1] = quant_color(quant_level, v1);
}

/**
 * @brief Quantize a HDR L color using the small range encoding.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as packed (l0, l1) with mode bits.
 * @param      quant_level   The quantization level to use.
 *
 * @return Returns @c false on failure, @c true on success.
 */
static bool try_quantize_hdr_luminance_small_range(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[2],
	quant_method quant_level
) {
	float lum0 = hadd_rgb_s(color0) * (1.0f / 3.0f);
	float lum1 = hadd_rgb_s(color1) * (1.0f / 3.0f);

	if (lum1 < lum0)
	{
		float avg = (lum0 + lum1) * 0.5f;
		lum0 = avg;
		lum1 = avg;
	}

	int ilum1 = astc::flt2int_rtn(lum1);
	int ilum0 = astc::flt2int_rtn(lum0);

	// Difference of more than a factor-of-2 results in immediate failure
	if (ilum1 - ilum0 > 2048)
	{
		return false;
	}

	int lowval, highval, diffval;
	int v0, v1;
	int v0e, v1e;
	int v0d, v1d;

	// Try to encode the high-precision submode
	lowval = (ilum0 + 16) >> 5;
	highval = (ilum1 + 16) >> 5;

	lowval = astc::clamp(lowval, 0, 2047);
	highval = astc::clamp(highval, 0, 2047);

	v0 = lowval & 0x7F;
	v0e = quant_color(quant_level, v0);
	v0d = v0e;

	if (v0d < 0x80)
	{
		lowval = (lowval & ~0x7F) | v0d;
		diffval = highval - lowval;
		if (diffval >= 0 && diffval <= 15)
		{
			v1 = ((lowval >> 3) & 0xF0) | diffval;
			v1e = quant_color(quant_level, v1);
			v1d = v1e;
			if ((v1d & 0xF0) == (v1 & 0xF0))
			{
				output[0] = static_cast<uint8_t>(v0e);
				output[1] = static_cast<uint8_t>(v1e);
				return true;
			}
		}
	}

	// Try to encode the low-precision submode
	lowval = (ilum0 + 32) >> 6;
	highval = (ilum1 + 32) >> 6;

	lowval = astc::clamp(lowval, 0, 1023);
	highval = astc::clamp(highval, 0, 1023);

	v0 = (lowval & 0x7F) | 0x80;
	v0e = quant_color(quant_level, v0);
	v0d = v0e;
	if ((v0d & 0x80) == 0)
	{
		return false;
	}

	lowval = (lowval & ~0x7F) | (v0d & 0x7F);
	diffval = highval - lowval;
	if (diffval < 0 || diffval > 31)
	{
		return false;
	}

	v1 = ((lowval >> 2) & 0xE0) | diffval;
	v1e = quant_color(quant_level, v1);
	v1d = v1e;
	if ((v1d & 0xE0) != (v1 & 0xE0))
	{
		return false;
	}

	output[0] = static_cast<uint8_t>(v0e);
	output[1] = static_cast<uint8_t>(v1e);
	return true;
}

/**
 * @brief Quantize a HDR A color using either delta or direct RGBA encoding.
 *
 * @param      alpha0        The input unquantized color0 endpoint.
 * @param      alpha1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as packed RGBA+RGBA pairs with mode bits.
 * @param      quant_level   The quantization level to use.
 */
static void quantize_hdr_alpha(
	float alpha0,
	float alpha1,
	uint8_t output[2],
	quant_method quant_level
) {
	alpha0 = astc::clamp(alpha0, 0.0f, 65280.0f);
	alpha1 = astc::clamp(alpha1, 0.0f, 65280.0f);

	int ialpha0 = astc::flt2int_rtn(alpha0);
	int ialpha1 = astc::flt2int_rtn(alpha1);

	int val0, val1, diffval;
	int v6, v7;
	int v6e, v7e;
	int v6d, v7d;

	// Try to encode one of the delta submodes, in decreasing-precision order
	for (int i = 2; i >= 0; i--)
	{
		val0 = (ialpha0 + (128 >> i)) >> (8 - i);
		val1 = (ialpha1 + (128 >> i)) >> (8 - i);

		v6 = (val0 & 0x7F) | ((i & 1) << 7);
		v6e = quant_color(quant_level, v6);
		v6d = v6e;

		if ((v6 ^ v6d) & 0x80)
		{
			continue;
		}

		val0 = (val0 & ~0x7f) | (v6d & 0x7f);
		diffval = val1 - val0;
		int cutoff = 32 >> i;
		int mask = 2 * cutoff - 1;

		if (diffval < -cutoff || diffval >= cutoff)
		{
			continue;
		}

		v7 = ((i & 2) << 6) | ((val0 >> 7) << (6 - i)) | (diffval & mask);
		v7e = quant_color(quant_level, v7);
		v7d = v7e;

		static const int testbits[3] { 0xE0, 0xF0, 0xF8 };

		if ((v7 ^ v7d) & testbits[i])
		{
			continue;
		}

		output[0] = static_cast<uint8_t>(v6e);
		output[1] = static_cast<uint8_t>(v7e);
		return;
	}

	// Could not encode any of the delta modes; instead encode a flat value
	val0 = (ialpha0 + 256) >> 9;
	val1 = (ialpha1 + 256) >> 9;
	v6 = val0 | 0x80;
	v7 = val1 | 0x80;

	output[0] = quant_color(quant_level, v6);
	output[1] = quant_color(quant_level, v7);

	return;
}

/**
 * @brief Quantize a HDR RGBA color using either delta or direct RGBA encoding.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as packed RGBA+RGBA pairs with mode bits.
 * @param      quant_level   The quantization level to use.
 */
static void quantize_hdr_rgb_alpha(
	vfloat4 color0,
	vfloat4 color1,
	uint8_t output[8],
	quant_method quant_level
) {
	quantize_hdr_rgb(color0, color1, output, quant_level);
	quantize_hdr_alpha(color0.lane<3>(), color1.lane<3>(), output + 6, quant_level);
}

/* See header for documentation. */
uint8_t pack_color_endpoints(
	vfloat4 color0,
	vfloat4 color1,
	vfloat4 rgbs_color,
	vfloat4 rgbo_color,
	int format,
	uint8_t* output,
	quant_method quant_level
) {
	assert(QUANT_6 <= quant_level && quant_level <= QUANT_256);

	// We do not support negative colors
	color0 = max(color0, 0.0f);
	color1 = max(color1, 0.0f);

	uint8_t retval = 0;

	switch (format)
	{
	case FMT_RGB:
		if (quant_level <= QUANT_160)
		{
			if (try_quantize_rgb_delta_blue_contract(color0, color1, output, quant_level))
			{
				retval = FMT_RGB_DELTA;
				break;
			}
			if (try_quantize_rgb_delta(color0, color1, output, quant_level))
			{
				retval = FMT_RGB_DELTA;
				break;
			}
		}
		if (quant_level < QUANT_256 && try_quantize_rgb_blue_contract(color0, color1, output, quant_level))
		{
			retval = FMT_RGB;
			break;
		}
		quantize_rgb(color0, color1, output, quant_level);
		retval = FMT_RGB;
		break;

	case FMT_RGBA:
		if (quant_level <= QUANT_160)
		{
			if (try_quantize_rgba_delta_blue_contract(color0, color1, output, quant_level))
			{
				retval = FMT_RGBA_DELTA;
				break;
			}
			if (try_quantize_rgba_delta(color0, color1, output, quant_level))
			{
				retval = FMT_RGBA_DELTA;
				break;
			}
		}
		if (quant_level < QUANT_256 && try_quantize_rgba_blue_contract(color0, color1, output, quant_level))
		{
			retval = FMT_RGBA;
			break;
		}
		quantize_rgba(color0, color1, output, quant_level);
		retval = FMT_RGBA;
		break;

	case FMT_RGB_SCALE:
		quantize_rgbs(rgbs_color, output, quant_level);
		retval = FMT_RGB_SCALE;
		break;

	case FMT_HDR_RGB_SCALE:
		quantize_hdr_rgbo(rgbo_color, output, quant_level);
		retval = FMT_HDR_RGB_SCALE;
		break;

	case FMT_HDR_RGB:
		quantize_hdr_rgb(color0, color1, output, quant_level);
		retval = FMT_HDR_RGB;
		break;

	case FMT_RGB_SCALE_ALPHA:
		quantize_rgbs_alpha(color0, color1, rgbs_color, output, quant_level);
		retval = FMT_RGB_SCALE_ALPHA;
		break;

	case FMT_HDR_LUMINANCE_SMALL_RANGE:
	case FMT_HDR_LUMINANCE_LARGE_RANGE:
		if (try_quantize_hdr_luminance_small_range(color0, color1, output, quant_level))
		{
			retval = FMT_HDR_LUMINANCE_SMALL_RANGE;
			break;
		}
		quantize_hdr_luminance_large_range(color0, color1, output, quant_level);
		retval = FMT_HDR_LUMINANCE_LARGE_RANGE;
		break;

	case FMT_LUMINANCE:
		quantize_luminance(color0, color1, output, quant_level);
		retval = FMT_LUMINANCE;
		break;

	case FMT_LUMINANCE_ALPHA:
		if (quant_level <= 18)
		{
			if (try_quantize_luminance_alpha_delta(color0, color1, output, quant_level))
			{
				retval = FMT_LUMINANCE_ALPHA_DELTA;
				break;
			}
		}
		quantize_luminance_alpha(color0, color1, output, quant_level);
		retval = FMT_LUMINANCE_ALPHA;
		break;

	case FMT_HDR_RGB_LDR_ALPHA:
		quantize_hdr_rgb_ldr_alpha(color0, color1, output, quant_level);
		retval = FMT_HDR_RGB_LDR_ALPHA;
		break;

	case FMT_HDR_RGBA:
		quantize_hdr_rgb_alpha(color0, color1, output, quant_level);
		retval = FMT_HDR_RGBA;
		break;
	}

	return retval;
}

#endif
