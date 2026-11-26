// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2026 Arm Limited
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
 * @brief Functions to calculate variance per component in a NxN footprint.
 *
 * We need N to be parametric, so the routine below uses summed area tables in order to execute in
 * O(1) time independent of how big N is.
 *
 * The addition uses a Brent-Kung-based parallel prefix adder. This uses the prefix tree to first
 * perform a binary reduction, and then distributes the results. This method means that there is no
 * serial dependency between a given element and the next one, and also significantly improves
 * numerical stability allowing us to use floats rather than doubles.
 */

#include "astcenc_internal.h"

#include <cassert>

/**
 * @brief Generate a prefix-sum array using the Brent-Kung algorithm.
 *
 * This will take an input array of the form:
 *     v0, v1, v2, ...
 * ... and modify in-place to turn it into a prefix-sum array of the form:
 *     v0, v0+v1, v0+v1+v2, ...
 *
 * @param d      The array to prefix-sum.
 * @param items  The number of items in the array.
 * @param stride The item spacing in the array; i.e. dense arrays should use 1.
 */
static void brent_kung_prefix_sum(
	vfloat4* d,
	size_t items,
	size_t stride
) {
	if (items < 2)
		return;

	size_t lc_stride = 2;
	size_t log2_stride = 1;

	// The reduction-tree loop
	do {
		size_t step = lc_stride >> 1;
		size_t start = lc_stride - 1;
		size_t iters = items >> log2_stride;

		vfloat4 *da = d + (start * stride);
		ptrdiff_t ofs = -static_cast<ptrdiff_t>(step * stride);
		size_t ofs_stride = stride << log2_stride;

		while (iters)
		{
			*da = *da + da[ofs];
			da += ofs_stride;
			iters--;
		}

		log2_stride += 1;
		lc_stride <<= 1;
	} while (lc_stride <= items);

	// The expansion-tree loop
	do {
		log2_stride -= 1;
		lc_stride >>= 1;

		size_t step = lc_stride >> 1;
		size_t start = step + lc_stride - 1;
		size_t iters = (items - step) >> log2_stride;

		vfloat4 *da = d + (start * stride);
		ptrdiff_t ofs = -static_cast<ptrdiff_t>(step * stride);
		size_t ofs_stride = stride << log2_stride;

		while (iters)
		{
			*da = *da + da[ofs];
			da += ofs_stride;
			iters--;
		}
	} while (lc_stride > 2);
}

/* See header for documentation. */
void compute_pixel_region_variance(
	astcenc_contexti& ctx,
	const pixel_region_args& arg
) {
	// Unpack the memory structure into local variables
	const astcenc_image* img = arg.img;
	astcenc_swizzle swz = arg.swz;
	bool have_z = arg.have_z;

	size_t size_x = arg.size_x;
	size_t size_y = arg.size_y;
	size_t size_z = arg.size_z;

	size_t offset_x = arg.offset_x;
	size_t offset_y = arg.offset_y;
	size_t offset_z = arg.offset_z;

	size_t alpha_kernel_radius = arg.alpha_kernel_radius;

	float*   input_alpha_averages = ctx.input_alpha_averages;
	vfloat4* work_memory = arg.work_memory;

	// Compute memory sizes and dimensions that we need
	size_t kernel_radius = alpha_kernel_radius;
	size_t kerneldim = 2 * kernel_radius + 1;
	size_t kernel_radius_xy = kernel_radius;
	size_t kernel_radius_z = have_z ? kernel_radius : 0;

	size_t padsize_x = size_x + kerneldim;
	size_t padsize_y = size_y + kerneldim;
	size_t padsize_z = size_z + (have_z ? kerneldim : 0);
	size_t sizeprod = padsize_x * padsize_y * padsize_z;

	int zd_start = have_z ? 1 : 0;

	vfloat4 *varbuf1 = work_memory;
	vfloat4 *varbuf2 = work_memory + sizeprod;

	// Scaling factors to apply to Y and Z for accesses into the work buffers
	size_t yst = padsize_x;
	size_t zst = padsize_x * padsize_y;

	// Scaling factors to apply to Y and Z for accesses into result buffers
	size_t ydt = img->dim_x;
	size_t zdt = img->dim_x * img->dim_y;

	// Macros to act as accessor functions for the work-memory
	#define VARBUF1(z, y, x) varbuf1[z * zst + y * yst + x]
	#define VARBUF2(z, y, x) varbuf2[z * zst + y * yst + x]

	// Load N and N^2 values into the work buffers
	if (img->data_type == ASTCENC_TYPE_U8)
	{
		// Swizzle data structure 4 = ZERO, 5 = ONE
		uint8_t data[6];
		data[ASTCENC_SWZ_0] = 0;
		data[ASTCENC_SWZ_1] = 255;

		for (size_t z = zd_start; z < padsize_z; z++)
		{
			size_t z_src = (z - zd_start) + offset_z;
			if (z_src <= kernel_radius_z)
			{
				z_src = 0;
			}
			else
			{
				z_src -= kernel_radius_z;
			}

			z_src = astc::min(z_src, static_cast<size_t>(img->dim_z - 1));

			uint8_t* data8 = static_cast<uint8_t*>(img->data[z_src]);

			for (size_t y = 1; y < padsize_y; y++)
			{
				size_t y_src = (y - 1) + offset_y;
				if (y_src <= kernel_radius_xy)
				{
					y_src = 0;
				}
				else
				{
					y_src -= kernel_radius_xy;
				}

				y_src = astc::min(y_src, static_cast<size_t>(img->dim_y - 1));

				for (size_t x = 1; x < padsize_x; x++)
				{
					size_t x_src = (x - 1) + offset_x;
					if (x_src <= kernel_radius_xy)
					{
						x_src = 0;
					}
					else
					{
						x_src -= kernel_radius_xy;
					}

					x_src = astc::min(x_src, static_cast<size_t>(img->dim_x - 1));

					data[0] = data8[(4 * img->dim_x * y_src) + (4 * x_src    )];
					data[1] = data8[(4 * img->dim_x * y_src) + (4 * x_src + 1)];
					data[2] = data8[(4 * img->dim_x * y_src) + (4 * x_src + 2)];
					data[3] = data8[(4 * img->dim_x * y_src) + (4 * x_src + 3)];

					uint8_t r = data[swz.r];
					uint8_t g = data[swz.g];
					uint8_t b = data[swz.b];
					uint8_t a = data[swz.a];

					vfloat4 d = vfloat4 (r * (1.0f / 255.0f),
					                     g * (1.0f / 255.0f),
					                     b * (1.0f / 255.0f),
					                     a * (1.0f / 255.0f));

					VARBUF1(z, y, x) = d;
					VARBUF2(z, y, x) = d * d;
				}
			}
		}
	}
	else if (img->data_type == ASTCENC_TYPE_F16)
	{
		// Swizzle data structure 4 = ZERO, 5 = ONE (in FP16)
		uint16_t data[6];
		data[ASTCENC_SWZ_0] = 0;
		data[ASTCENC_SWZ_1] = 0x3C00;

		for (size_t z = zd_start; z < padsize_z; z++)
		{
			size_t z_src = (z - zd_start) + offset_z;
			if (z_src <= kernel_radius_z)
			{
				z_src = 0;
			}
			else
			{
				z_src -= kernel_radius_z;
			}

			z_src = astc::min(z_src, static_cast<size_t>(img->dim_z - 1));

			uint16_t* data16 = static_cast<uint16_t*>(img->data[z_src]);

			for (size_t y = 1; y < padsize_y; y++)
			{
				size_t y_src = (y - 1) + offset_y;
				if (y_src <= kernel_radius_xy)
				{
					y_src = 0;
				}
				else
				{
					y_src -= kernel_radius_xy;
				}

				y_src = astc::min(y_src, static_cast<size_t>(img->dim_y - 1));

				for (size_t x = 1; x < padsize_x; x++)
				{
					size_t x_src = (x - 1) + offset_x;
					if (x_src <= kernel_radius_xy)
					{
						x_src = 0;
					}
					else
					{
						x_src -= kernel_radius_xy;
					}

					x_src = astc::min(x_src, static_cast<size_t>(img->dim_x - 1));

					data[0] = data16[(4 * img->dim_x * y_src) + (4 * x_src    )];
					data[1] = data16[(4 * img->dim_x * y_src) + (4 * x_src + 1)];
					data[2] = data16[(4 * img->dim_x * y_src) + (4 * x_src + 2)];
					data[3] = data16[(4 * img->dim_x * y_src) + (4 * x_src + 3)];

					vint4 di(data[swz.r], data[swz.g], data[swz.b], data[swz.a]);
					vfloat4 d = float16_to_float(di);

					VARBUF1(z, y, x) = d;
					VARBUF2(z, y, x) = d * d;
				}
			}
		}
	}
	else // if (img->data_type == ASTCENC_TYPE_F32)
	{
		assert(img->data_type == ASTCENC_TYPE_F32);

		// Swizzle data structure 4 = ZERO, 5 = ONE (in FP16)
		float data[6];
		data[ASTCENC_SWZ_0] = 0.0f;
		data[ASTCENC_SWZ_1] = 1.0f;

		for (size_t z = zd_start; z < padsize_z; z++)
		{
			size_t z_src = (z - zd_start) + offset_z;
			if (z_src <= kernel_radius_z)
			{
				z_src = 0;
			}
			else
			{
				z_src -= kernel_radius_z;
			}

			z_src = astc::min(z_src, static_cast<size_t>(img->dim_z - 1));

			float* data32 = static_cast<float*>(img->data[z_src]);

			for (size_t y = 1; y < padsize_y; y++)
			{
				size_t y_src = (y - 1) + offset_y;
				if (y_src <= kernel_radius_xy)
				{
					y_src = 0;
				}
				else
				{
					y_src -= kernel_radius_xy;
				}

				y_src = astc::min(y_src, static_cast<size_t>(img->dim_y - 1));

				for (size_t x = 1; x < padsize_x; x++)
				{
					size_t x_src = (x - 1) + offset_x;
					if (x_src <= kernel_radius_xy)
					{
						x_src = 0;
					}
					else
					{
						x_src -= kernel_radius_xy;
					}

					x_src = astc::min(x_src, static_cast<size_t>(img->dim_x - 1));

					data[0] = data32[(4 * img->dim_x * y_src) + (4 * x_src    )];
					data[1] = data32[(4 * img->dim_x * y_src) + (4 * x_src + 1)];
					data[2] = data32[(4 * img->dim_x * y_src) + (4 * x_src + 2)];
					data[3] = data32[(4 * img->dim_x * y_src) + (4 * x_src + 3)];

					float r = data[swz.r];
					float g = data[swz.g];
					float b = data[swz.b];
					float a = data[swz.a];

					vfloat4 d(r, g, b, a);

					VARBUF1(z, y, x) = d;
					VARBUF2(z, y, x) = d * d;
				}
			}
		}
	}

	// Pad with an extra layer of 0s; this forms the edge of the SAT tables
	vfloat4 vbz = vfloat4::zero();
	for (size_t z = 0; z < padsize_z; z++)
	{
		for (size_t y = 0; y < padsize_y; y++)
		{
			VARBUF1(z, y, 0) = vbz;
			VARBUF2(z, y, 0) = vbz;
		}

		for (size_t x = 0; x < padsize_x; x++)
		{
			VARBUF1(z, 0, x) = vbz;
			VARBUF2(z, 0, x) = vbz;
		}
	}

	if (have_z)
	{
		for (size_t y = 0; y < padsize_y; y++)
		{
			for (size_t x = 0; x < padsize_x; x++)
			{
				VARBUF1(0, y, x) = vbz;
				VARBUF2(0, y, x) = vbz;
			}
		}
	}

	// Generate summed-area tables for N and N^2; this is done in-place, using
	// a Brent-Kung parallel-prefix based algorithm to minimize precision loss
	for (size_t z = zd_start; z < padsize_z; z++)
	{
		for (size_t y = 1; y < padsize_y; y++)
		{
			brent_kung_prefix_sum(&(VARBUF1(z, y, 1)), padsize_x - 1, 1);
			brent_kung_prefix_sum(&(VARBUF2(z, y, 1)), padsize_x - 1, 1);
		}
	}

	for (size_t z = zd_start; z < padsize_z; z++)
	{
		for (size_t x = 1; x < padsize_x; x++)
		{
			brent_kung_prefix_sum(&(VARBUF1(z, 1, x)), padsize_y - 1, yst);
			brent_kung_prefix_sum(&(VARBUF2(z, 1, x)), padsize_y - 1, yst);
		}
	}

	if (have_z)
	{
		for (size_t y = 1; y < padsize_y; y++)
		{
			for (size_t x = 1; x < padsize_x; x++)
			{
				brent_kung_prefix_sum(&(VARBUF1(1, y, x)), padsize_z - 1, zst);
				brent_kung_prefix_sum(&(VARBUF2(1, y, x)), padsize_z - 1, zst);
			}
		}
	}

	// Compute a few constants used in the variance-calculation.
	float alpha_kdim = static_cast<float>(2 * alpha_kernel_radius + 1);
	float alpha_rsamples;

	if (have_z)
	{
		alpha_rsamples = 1.0f / (alpha_kdim * alpha_kdim * alpha_kdim);
	}
	else
	{
		alpha_rsamples = 1.0f / (alpha_kdim * alpha_kdim);
	}

	// Use the summed-area tables to compute variance for each neighborhood
	if (have_z)
	{
		for (size_t z = 0; z < size_z; z++)
		{
			size_t z_src = z + kernel_radius_z;
			size_t z_dst = z + offset_z;
			size_t z_low  = z_src - alpha_kernel_radius;
			size_t z_high = z_src + alpha_kernel_radius + 1;

			for (size_t y = 0; y < size_y; y++)
			{
				size_t y_src = y + kernel_radius_xy;
				size_t y_dst = y + offset_y;
				size_t y_low  = y_src - alpha_kernel_radius;
				size_t y_high = y_src + alpha_kernel_radius + 1;

				for (size_t x = 0; x < size_x; x++)
				{
					size_t x_src = x + kernel_radius_xy;
					size_t x_dst = x + offset_x;
					size_t x_low  = x_src - alpha_kernel_radius;
					size_t x_high = x_src + alpha_kernel_radius + 1;

					// Summed-area table lookups for alpha average
					float vasum = (  VARBUF1(z_high,  y_low,  x_low).lane<3>()
					               - VARBUF1(z_high,  y_low, x_high).lane<3>()
					               - VARBUF1(z_high, y_high,  x_low).lane<3>()
					               + VARBUF1(z_high, y_high, x_high).lane<3>()) -
					              (  VARBUF1(z_low,   y_low,  x_low).lane<3>()
					               - VARBUF1(z_low,   y_low, x_high).lane<3>()
					               - VARBUF1(z_low,  y_high,  x_low).lane<3>()
					               + VARBUF1(z_low,  y_high, x_high).lane<3>());

					size_t out_index = z_dst * zdt + y_dst * ydt + x_dst;
					input_alpha_averages[out_index] = (vasum * alpha_rsamples);
				}
			}
		}
	}
	else
	{
		for (size_t y = 0; y < size_y; y++)
		{
			size_t y_src = y + kernel_radius_xy;
			size_t y_dst = y + offset_y;
			size_t y_low  = y_src - alpha_kernel_radius;
			size_t y_high = y_src + alpha_kernel_radius + 1;

			for (size_t x = 0; x < size_x; x++)
			{
				size_t x_src = x + kernel_radius_xy;
				size_t x_dst = x + offset_x;
				size_t x_low  = x_src - alpha_kernel_radius;
				size_t x_high = x_src + alpha_kernel_radius + 1;

				// Summed-area table lookups for alpha average
				float vasum = VARBUF1(0, y_low,  x_low).lane<3>()
				            - VARBUF1(0, y_low,  x_high).lane<3>()
				            - VARBUF1(0, y_high, x_low).lane<3>()
				            + VARBUF1(0, y_high, x_high).lane<3>();

				size_t out_index = y_dst * ydt + x_dst;
				input_alpha_averages[out_index] = (vasum * alpha_rsamples);
			}
		}
	}
}

/* See header for documentation. */
size_t init_compute_averages(
	const astcenc_image& img,
	size_t alpha_kernel_radius,
	const astcenc_swizzle& swz,
	avg_args& ag
) {
	size_t size_x = img.dim_x;
	size_t size_y = img.dim_y;
	size_t size_z = img.dim_z;

	// Compute maximum block size and from that the working memory buffer size
	size_t kernel_radius = alpha_kernel_radius;
	size_t kerneldim = 2 * kernel_radius + 1;

	bool have_z = size_z > 1;
	size_t max_blk_size_xy = have_z ? 16u : 32u;

	size_t max_blk_size_z_lim = have_z ? 16u : 1u;
	size_t max_blk_size_z = astc::min(size_z, max_blk_size_z_lim);

	size_t max_padsize_xy = max_blk_size_xy + kerneldim;
	size_t max_padsize_z = max_blk_size_z + (have_z ? kerneldim : 0);

	// Perform block-wise averages calculations across the image
	// Initialize fields which are not populated until later
	ag.arg.size_x = 0;
	ag.arg.size_y = 0;
	ag.arg.size_z = 0;
	ag.arg.offset_x = 0;
	ag.arg.offset_y = 0;
	ag.arg.offset_z = 0;
	ag.arg.work_memory = nullptr;

	ag.arg.img = &img;
	ag.arg.swz = swz;
	ag.arg.have_z = have_z;
	ag.arg.alpha_kernel_radius = alpha_kernel_radius;

	ag.img_size_x = size_x;
	ag.img_size_y = size_y;
	ag.img_size_z = size_z;
	ag.blk_size_xy = max_blk_size_xy;
	ag.blk_size_z = max_blk_size_z;
	ag.work_memory_size = 2 * max_padsize_xy * max_padsize_xy * max_padsize_z;

	// The parallel task count
	size_t tasks_z = astc::get_block_count_safe(size_z, max_blk_size_z);
	size_t tasks_y = astc::get_block_count_safe(size_y, max_blk_size_xy);
	return tasks_z * tasks_y;
}

#endif
