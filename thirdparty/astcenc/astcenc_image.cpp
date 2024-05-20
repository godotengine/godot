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
 * @brief Functions for creating in-memory ASTC image structures.
 */

#include <cassert>
#include <cstring>

#include "astcenc_internal.h"

/**
 * @brief Loader pipeline function type for data fetch from memory.
 */
using pixel_loader = vfloat4(*)(const void*, int);

/**
 * @brief Loader pipeline function type for swizzling data in a vector.
 */
using pixel_swizzler = vfloat4(*)(vfloat4, const astcenc_swizzle&);

/**
 * @brief Loader pipeline function type for converting data in a vector to LNS.
 */
using pixel_converter = vfloat4(*)(vfloat4, vmask4);

/**
 * @brief Load a 8-bit UNORM texel from a data array.
 *
 * @param data          The data pointer.
 * @param base_offset   The index offset to the start of the pixel.
 */
static vfloat4 load_texel_u8(
	const void* data,
	int base_offset
) {
	const uint8_t* data8 = static_cast<const uint8_t*>(data);
	return int_to_float(vint4(data8 + base_offset)) / 255.0f;
}

/**
 * @brief Load a 16-bit fp16 texel from a data array.
 *
 * @param data          The data pointer.
 * @param base_offset   The index offset to the start of the pixel.
 */
static vfloat4 load_texel_f16(
	const void* data,
	int base_offset
) {
	const uint16_t* data16 = static_cast<const uint16_t*>(data);
	int r = data16[base_offset    ];
	int g = data16[base_offset + 1];
	int b = data16[base_offset + 2];
	int a = data16[base_offset + 3];
	return float16_to_float(vint4(r, g, b, a));
}

/**
 * @brief Load a 32-bit float texel from a data array.
 *
 * @param data          The data pointer.
 * @param base_offset   The index offset to the start of the pixel.
 */
static vfloat4 load_texel_f32(
	const void* data,
	int base_offset
) {
	const float* data32 = static_cast<const float*>(data);
	return vfloat4(data32 + base_offset);
}

/**
 * @brief Dummy no-op swizzle function.
 *
 * @param data   The source RGBA vector to swizzle.
 * @param swz    The swizzle to use.
 */
static vfloat4 swz_texel_skip(
	vfloat4 data,
	const astcenc_swizzle& swz
) {
	(void)swz;
	return data;
}

/**
 * @brief Swizzle a texel into a new arrangement.
 *
 * @param data   The source RGBA vector to swizzle.
 * @param swz    The swizzle to use.
 */
static vfloat4 swz_texel(
	vfloat4 data,
	const astcenc_swizzle& swz
) {
	ASTCENC_ALIGNAS float datas[6];

	storea(data, datas);
	datas[ASTCENC_SWZ_0] = 0.0f;
	datas[ASTCENC_SWZ_1] = 1.0f;

	return vfloat4(datas[swz.r], datas[swz.g], datas[swz.b], datas[swz.a]);
}

/**
 * @brief Encode a texel that is entirely LDR linear.
 *
 * @param data       The RGBA data to encode.
 * @param lns_mask   The mask for the HDR channels than need LNS encoding.
 */
static vfloat4 encode_texel_unorm(
	vfloat4 data,
	vmask4 lns_mask
) {
	(void)lns_mask;
	return data * 65535.0f;
}

/**
 * @brief Encode a texel that includes at least some HDR LNS texels.
 *
 * @param data       The RGBA data to encode.
 * @param lns_mask   The mask for the HDR channels than need LNS encoding.
 */
static vfloat4 encode_texel_lns(
	vfloat4 data,
	vmask4 lns_mask
) {
	vfloat4 datav_unorm = data * 65535.0f;
	vfloat4 datav_lns = float_to_lns(data);
	return select(datav_unorm, datav_lns, lns_mask);
}

/* See header for documentation. */
void load_image_block(
	astcenc_profile decode_mode,
	const astcenc_image& img,
	image_block& blk,
	const block_size_descriptor& bsd,
	unsigned int xpos,
	unsigned int ypos,
	unsigned int zpos,
	const astcenc_swizzle& swz
) {
	unsigned int xsize = img.dim_x;
	unsigned int ysize = img.dim_y;
	unsigned int zsize = img.dim_z;

	blk.xpos = xpos;
	blk.ypos = ypos;
	blk.zpos = zpos;

	// True if any non-identity swizzle
	bool needs_swz = (swz.r != ASTCENC_SWZ_R) || (swz.g != ASTCENC_SWZ_G) ||
	                 (swz.b != ASTCENC_SWZ_B) || (swz.a != ASTCENC_SWZ_A);

	int idx = 0;

	vfloat4 data_min(1e38f);
	vfloat4 data_mean(0.0f);
	vfloat4 data_mean_scale(1.0f / static_cast<float>(bsd.texel_count));
	vfloat4 data_max(-1e38f);
	vmask4 grayscalev(true);

	// This works because we impose the same choice everywhere during encode
	uint8_t rgb_lns = (decode_mode == ASTCENC_PRF_HDR) ||
	                  (decode_mode == ASTCENC_PRF_HDR_RGB_LDR_A) ? 1 : 0;
	uint8_t a_lns = decode_mode == ASTCENC_PRF_HDR ? 1 : 0;
	vint4 use_lns(rgb_lns, rgb_lns, rgb_lns, a_lns);
	vmask4 lns_mask = use_lns != vint4::zero();

	// Set up the function pointers for loading pipeline as needed
	pixel_loader loader = load_texel_u8;
	if (img.data_type == ASTCENC_TYPE_F16)
	{
		loader = load_texel_f16;
	}
	else if  (img.data_type == ASTCENC_TYPE_F32)
	{
		loader = load_texel_f32;
	}

	pixel_swizzler swizzler = swz_texel_skip;
	if (needs_swz)
	{
		swizzler = swz_texel;
	}

	pixel_converter converter = encode_texel_unorm;
	if (any(lns_mask))
	{
		converter = encode_texel_lns;
	}

	for (unsigned int z = 0; z < bsd.zdim; z++)
	{
		unsigned int zi = astc::min(zpos + z, zsize - 1);
		void* plane = img.data[zi];

		for (unsigned int y = 0; y < bsd.ydim; y++)
		{
			unsigned int yi = astc::min(ypos + y, ysize - 1);

			for (unsigned int x = 0; x < bsd.xdim; x++)
			{
				unsigned int xi = astc::min(xpos + x, xsize - 1);

				vfloat4 datav = loader(plane, (4 * xsize * yi) + (4 * xi));
				datav = swizzler(datav, swz);
				datav = converter(datav, lns_mask);

				// Compute block metadata
				data_min = min(data_min, datav);
				data_mean += datav * data_mean_scale;
				data_max = max(data_max, datav);

				grayscalev = grayscalev & (datav.swz<0,0,0,0>() == datav.swz<1,1,2,2>());

				blk.data_r[idx] = datav.lane<0>();
				blk.data_g[idx] = datav.lane<1>();
				blk.data_b[idx] = datav.lane<2>();
				blk.data_a[idx] = datav.lane<3>();

				blk.rgb_lns[idx] = rgb_lns;
				blk.alpha_lns[idx] = a_lns;

				idx++;
			}
		}
	}

	// Reverse the encoding so we store origin block in the original format
	vfloat4 data_enc = blk.texel(0);
	vfloat4 data_enc_unorm = data_enc / 65535.0f;
	vfloat4 data_enc_lns = vfloat4::zero();

	if (rgb_lns || a_lns)
	{
		data_enc_lns = float16_to_float(lns_to_sf16(float_to_int(data_enc)));
	}

	blk.origin_texel = select(data_enc_unorm, data_enc_lns, lns_mask);

	// Store block metadata
	blk.data_min = data_min;
	blk.data_mean = data_mean;
	blk.data_max = data_max;
	blk.grayscale = all(grayscalev);
}

/* See header for documentation. */
void load_image_block_fast_ldr(
	astcenc_profile decode_mode,
	const astcenc_image& img,
	image_block& blk,
	const block_size_descriptor& bsd,
	unsigned int xpos,
	unsigned int ypos,
	unsigned int zpos,
	const astcenc_swizzle& swz
) {
	(void)swz;
	(void)decode_mode;

	unsigned int xsize = img.dim_x;
	unsigned int ysize = img.dim_y;

	blk.xpos = xpos;
	blk.ypos = ypos;
	blk.zpos = zpos;

	vfloat4 data_min(1e38f);
	vfloat4 data_mean = vfloat4::zero();
	vfloat4 data_max(-1e38f);
	vmask4 grayscalev(true);
	int idx = 0;

	const uint8_t* plane = static_cast<const uint8_t*>(img.data[0]);
	for (unsigned int y = ypos; y < ypos + bsd.ydim; y++)
	{
		unsigned int yi = astc::min(y, ysize - 1);

		for (unsigned int x = xpos; x < xpos + bsd.xdim; x++)
		{
			unsigned int xi = astc::min(x, xsize - 1);

			vint4 datavi = vint4(plane + (4 * xsize * yi) + (4 * xi));
			vfloat4 datav = int_to_float(datavi) * (65535.0f / 255.0f);

			// Compute block metadata
			data_min = min(data_min, datav);
			data_mean += datav;
			data_max = max(data_max, datav);

			grayscalev = grayscalev & (datav.swz<0,0,0,0>() == datav.swz<1,1,2,2>());

			blk.data_r[idx] = datav.lane<0>();
			blk.data_g[idx] = datav.lane<1>();
			blk.data_b[idx] = datav.lane<2>();
			blk.data_a[idx] = datav.lane<3>();

			idx++;
		}
	}

	// Reverse the encoding so we store origin block in the original format
	blk.origin_texel = blk.texel(0) / 65535.0f;

	// Store block metadata
	blk.rgb_lns[0] = 0;
	blk.alpha_lns[0] = 0;
	blk.data_min = data_min;
	blk.data_mean = data_mean / static_cast<float>(bsd.texel_count);
	blk.data_max = data_max;
	blk.grayscale = all(grayscalev);
}

/* See header for documentation. */
void store_image_block(
	astcenc_image& img,
	const image_block& blk,
	const block_size_descriptor& bsd,
	unsigned int xpos,
	unsigned int ypos,
	unsigned int zpos,
	const astcenc_swizzle& swz
) {
	unsigned int x_size = img.dim_x;
	unsigned int x_start = xpos;
	unsigned int x_end = astc::min(x_size, xpos + bsd.xdim);
	unsigned int x_count = x_end - x_start;
	unsigned int x_nudge = bsd.xdim - x_count;

	unsigned int y_size = img.dim_y;
	unsigned int y_start = ypos;
	unsigned int y_end = astc::min(y_size, ypos + bsd.ydim);
	unsigned int y_count = y_end - y_start;
	unsigned int y_nudge = (bsd.ydim - y_count) * bsd.xdim;

	unsigned int z_size = img.dim_z;
	unsigned int z_start = zpos;
	unsigned int z_end = astc::min(z_size, zpos + bsd.zdim);

	// True if any non-identity swizzle
	bool needs_swz = (swz.r != ASTCENC_SWZ_R) || (swz.g != ASTCENC_SWZ_G) ||
	                 (swz.b != ASTCENC_SWZ_B) || (swz.a != ASTCENC_SWZ_A);

	// True if any swizzle uses Z reconstruct
	bool needs_z = (swz.r == ASTCENC_SWZ_Z) || (swz.g == ASTCENC_SWZ_Z) ||
	               (swz.b == ASTCENC_SWZ_Z) || (swz.a == ASTCENC_SWZ_Z);

	int idx = 0;
	if (img.data_type == ASTCENC_TYPE_U8)
	{
		for (unsigned int z = z_start; z < z_end; z++)
		{
			// Fetch the image plane
			uint8_t* data8 = static_cast<uint8_t*>(img.data[z]);

			for (unsigned int y = y_start; y < y_end; y++)
			{
				uint8_t* data8_row = data8 + (4 * x_size * y) + (4 * x_start);

				for (unsigned int x = 0; x < x_count; x += ASTCENC_SIMD_WIDTH)
				{
					unsigned int max_texels = ASTCENC_SIMD_WIDTH;
					unsigned int used_texels = astc::min(x_count - x, max_texels);

					// Unaligned load as rows are not always SIMD_WIDTH long
					vfloat data_r(blk.data_r + idx);
					vfloat data_g(blk.data_g + idx);
					vfloat data_b(blk.data_b + idx);
					vfloat data_a(blk.data_a + idx);

					vint data_ri = float_to_int_rtn(min(data_r, 1.0f) * 255.0f);
					vint data_gi = float_to_int_rtn(min(data_g, 1.0f) * 255.0f);
					vint data_bi = float_to_int_rtn(min(data_b, 1.0f) * 255.0f);
					vint data_ai = float_to_int_rtn(min(data_a, 1.0f) * 255.0f);

					if (needs_swz)
					{
						vint swizzle_table[7];
						swizzle_table[ASTCENC_SWZ_0] = vint(0);
						swizzle_table[ASTCENC_SWZ_1] = vint(255);
						swizzle_table[ASTCENC_SWZ_R] = data_ri;
						swizzle_table[ASTCENC_SWZ_G] = data_gi;
						swizzle_table[ASTCENC_SWZ_B] = data_bi;
						swizzle_table[ASTCENC_SWZ_A] = data_ai;

						if (needs_z)
						{
							vfloat data_x = (data_r * vfloat(2.0f)) - vfloat(1.0f);
							vfloat data_y = (data_a * vfloat(2.0f)) - vfloat(1.0f);
							vfloat data_z = vfloat(1.0f) - (data_x * data_x) - (data_y * data_y);
							data_z = max(data_z, 0.0f);
							data_z = (sqrt(data_z) * vfloat(0.5f)) + vfloat(0.5f);

							swizzle_table[ASTCENC_SWZ_Z] = float_to_int_rtn(min(data_z, 1.0f) * 255.0f);
						}

						data_ri = swizzle_table[swz.r];
						data_gi = swizzle_table[swz.g];
						data_bi = swizzle_table[swz.b];
						data_ai = swizzle_table[swz.a];
					}

					// Errors are NaN encoded - convert to magenta error color
					// Branch is OK here - it is almost never true so predicts well
					vmask nan_mask = data_r != data_r;
					if (any(nan_mask))
					{
						data_ri = select(data_ri, vint(0xFF), nan_mask);
						data_gi = select(data_gi, vint(0x00), nan_mask);
						data_bi = select(data_bi, vint(0xFF), nan_mask);
						data_ai = select(data_ai, vint(0xFF), nan_mask);
					}

					vint data_rgbai = interleave_rgba8(data_ri, data_gi, data_bi, data_ai);
					vmask store_mask = vint::lane_id() < vint(used_texels);
					store_lanes_masked(data8_row, data_rgbai, store_mask);

					data8_row += ASTCENC_SIMD_WIDTH * 4;
					idx += used_texels;
				}
				idx += x_nudge;
			}
			idx += y_nudge;
		}
	}
	else if (img.data_type == ASTCENC_TYPE_F16)
	{
		for (unsigned int z = z_start; z < z_end; z++)
		{
			// Fetch the image plane
			uint16_t* data16 = static_cast<uint16_t*>(img.data[z]);

			for (unsigned int y = y_start; y < y_end; y++)
			{
				uint16_t* data16_row = data16 + (4 * x_size * y) + (4 * x_start);

				for (unsigned int x = 0; x < x_count; x++)
				{
					vint4 color;

					// NaNs are handled inline - no need to special case
					if (needs_swz)
					{
						float data[7];
						data[ASTCENC_SWZ_0] = 0.0f;
						data[ASTCENC_SWZ_1] = 1.0f;
						data[ASTCENC_SWZ_R] = blk.data_r[idx];
						data[ASTCENC_SWZ_G] = blk.data_g[idx];
						data[ASTCENC_SWZ_B] = blk.data_b[idx];
						data[ASTCENC_SWZ_A] = blk.data_a[idx];

						if (needs_z)
						{
							float xN = (data[0] * 2.0f) - 1.0f;
							float yN = (data[3] * 2.0f) - 1.0f;
							float zN = 1.0f - xN * xN - yN * yN;
							if (zN < 0.0f)
							{
								zN = 0.0f;
							}
							data[ASTCENC_SWZ_Z] = (astc::sqrt(zN) * 0.5f) + 0.5f;
						}

						vfloat4 colorf(data[swz.r], data[swz.g], data[swz.b], data[swz.a]);
						color = float_to_float16(colorf);
					}
					else
					{
						vfloat4 colorf = blk.texel(idx);
						color = float_to_float16(colorf);
					}

					// TODO: Vectorize with store N shorts?
					data16_row[0] = static_cast<uint16_t>(color.lane<0>());
					data16_row[1] = static_cast<uint16_t>(color.lane<1>());
					data16_row[2] = static_cast<uint16_t>(color.lane<2>());
					data16_row[3] = static_cast<uint16_t>(color.lane<3>());
					data16_row += 4;
					idx++;
				}
				idx += x_nudge;
			}
			idx += y_nudge;
		}
	}
	else // if (img.data_type == ASTCENC_TYPE_F32)
	{
		assert(img.data_type == ASTCENC_TYPE_F32);

		for (unsigned int z = z_start; z < z_end; z++)
		{
			// Fetch the image plane
			float* data32 = static_cast<float*>(img.data[z]);

			for (unsigned int y = y_start; y < y_end; y++)
			{
				float* data32_row = data32 + (4 * x_size * y) + (4 * x_start);

				for (unsigned int x = 0; x < x_count; x++)
				{
					vfloat4 color = blk.texel(idx);

					// NaNs are handled inline - no need to special case
					if (needs_swz)
					{
						float data[7];
						data[ASTCENC_SWZ_0] = 0.0f;
						data[ASTCENC_SWZ_1] = 1.0f;
						data[ASTCENC_SWZ_R] = color.lane<0>();
						data[ASTCENC_SWZ_G] = color.lane<1>();
						data[ASTCENC_SWZ_B] = color.lane<2>();
						data[ASTCENC_SWZ_A] = color.lane<3>();

						if (needs_z)
						{
							float xN = (data[0] * 2.0f) - 1.0f;
							float yN = (data[3] * 2.0f) - 1.0f;
							float zN = 1.0f - xN * xN - yN * yN;
							if (zN < 0.0f)
							{
								zN = 0.0f;
							}
							data[ASTCENC_SWZ_Z] = (astc::sqrt(zN) * 0.5f) + 0.5f;
						}

						color = vfloat4(data[swz.r], data[swz.g], data[swz.b], data[swz.a]);
					}

					store(color, data32_row);
					data32_row += 4;
					idx++;
				}
				idx += x_nudge;
			}
			idx += y_nudge;
		}
	}
}
