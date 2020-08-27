// basisu_pvrtc1_4.cpp
// Copyright (C) 2019 Binomial LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "basisu_pvrtc1_4.h"

namespace basisu
{
	uint32_t pvrtc4_swizzle_uv(uint32_t width, uint32_t height, uint32_t x, uint32_t y)
	{
		assert((x < width) && (y < height) && basisu::is_pow2(height) && basisu::is_pow2(width));
				
		uint32_t min_d = width, max_v = y;
		if (height < width)
		{
			min_d = height;
			max_v = x;
		}

		// Interleave the XY LSB's
		uint32_t shift_ofs = 0, swizzled = 0;
		for (uint32_t s_bit = 1, d_bit = 1; s_bit < min_d; s_bit <<= 1, d_bit <<= 2, ++shift_ofs)
		{
			if (y & s_bit) swizzled |= d_bit;
			if (x & s_bit) swizzled |= (2 * d_bit);
		}

		max_v >>= shift_ofs;
		
		// OR in the rest of the bits from the largest dimension
		swizzled |= (max_v << (2 * shift_ofs));

		return swizzled;
	}

	color_rgba pvrtc4_block::get_endpoint(uint32_t endpoint_index, bool unpack) const
	{
		assert(endpoint_index < 2);
		const uint32_t packed = m_endpoints >> (endpoint_index * 16);

		uint32_t r, g, b, a;
		if (packed & 0x8000)
		{
			// opaque 554 or 555
			if (!endpoint_index)
			{
				r = (packed >> 10) & 31;
				g = (packed >> 5) & 31;
				b = (packed >> 1) & 15;
					
				if (unpack)
				{
					b = (b << 1) | (b >> 3);
				}
			}
			else
			{
				r = (packed >> 10) & 31;
				g = (packed >> 5) & 31;
				b = packed & 31;
			}

			a = unpack ? 255 : 7;
		}
		else
		{
			// translucent 4433 or 4443
			if (!endpoint_index)
			{
				a = (packed >> 12) & 7;
				r = (packed >> 8) & 15;
				g = (packed >> 4) & 15;
				b = (packed >> 1) & 7;

				if (unpack)
				{
					a = (a << 1);
					a = (a << 4) | a;
						
					r = (r << 1) | (r >> 3);
					g = (g << 1) | (g >> 3);
					b = (b << 2) | (b >> 1);
				}
			}
			else
			{
				a = (packed >> 12) & 7;
				r = (packed >> 8) & 15;
				g = (packed >> 4) & 15;
				b = packed & 15;

				if (unpack)
				{
					a = (a << 1);
					a = (a << 4) | a;

					r = (r << 1) | (r >> 3);
					g = (g << 1) | (g >> 3);
					b = (b << 1) | (b >> 3);
				}
			}
		}

		if (unpack)
		{
			r = (r << 3) | (r >> 2);
			g = (g << 3) | (g >> 2);
			b = (b << 3) | (b >> 2);
		}

		assert((r < 256) && (g < 256) && (b < 256) && (a < 256));

		return color_rgba(r, g, b, a);
	}

	color_rgba pvrtc4_block::get_endpoint_5554(uint32_t endpoint_index) const
	{
		assert(endpoint_index < 2);
		const uint32_t packed = m_endpoints >> (endpoint_index * 16);

		uint32_t r, g, b, a;
		if (packed & 0x8000)
		{
			// opaque 554 or 555
			if (!endpoint_index)
			{
				r = (packed >> 10) & 31;
				g = (packed >> 5) & 31;
				b = (packed >> 1) & 15;

				b = (b << 1) | (b >> 3);
			}
			else
			{
				r = (packed >> 10) & 31;
				g = (packed >> 5) & 31;
				b = packed & 31;
			}

			a = 15;
		}
		else
		{
			// translucent 4433 or 4443
			if (!endpoint_index)
			{
				a = (packed >> 12) & 7;
				r = (packed >> 8) & 15;
				g = (packed >> 4) & 15;
				b = (packed >> 1) & 7;

				a = a << 1;
						
				r = (r << 1) | (r >> 3);
				g = (g << 1) | (g >> 3);
				b = (b << 2) | (b >> 1);
			}
			else
			{
				a = (packed >> 12) & 7;
				r = (packed >> 8) & 15;
				g = (packed >> 4) & 15;
				b = packed & 15;

				a = a << 1;
						
				r = (r << 1) | (r >> 3);
				g = (g << 1) | (g >> 3);
				b = (b << 1) | (b >> 3);
			}
		}
						
		assert((r < 32) && (g < 32) && (b < 32) && (a < 16));

		return color_rgba(r, g, b, a);
	}

	bool pvrtc4_image::get_interpolated_colors(uint32_t x, uint32_t y, color_rgba* pColors) const
	{
		assert((x < m_width) && (y < m_height));

		int block_x0 = (static_cast<int>(x) - 2) >> 2;
		int block_x1 = block_x0 + 1;
		int block_y0 = (static_cast<int>(y) - 2) >> 2;
		int block_y1 = block_y0 + 1;
		
		block_x0 = posmod(block_x0, m_block_width);
		block_x1 = posmod(block_x1, m_block_width);
		block_y0 = posmod(block_y0, m_block_height);
		block_y1 = posmod(block_y1, m_block_height);
		
		pColors[0] = interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(0), m_blocks(block_x1, block_y0).get_endpoint_5554(0), m_blocks(block_x0, block_y1).get_endpoint_5554(0), m_blocks(block_x1, block_y1).get_endpoint_5554(0));
		pColors[3] = interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(1), m_blocks(block_x1, block_y0).get_endpoint_5554(1), m_blocks(block_x0, block_y1).get_endpoint_5554(1), m_blocks(block_x1, block_y1).get_endpoint_5554(1));

		if (get_block_uses_transparent_modulation(x >> 2, y >> 2))
		{
			for (uint32_t c = 0; c < 4; c++)
			{
				uint32_t m = (pColors[0][c] + pColors[3][c]) / 2;
				pColors[1][c] = static_cast<uint8_t>(m);
				pColors[2][c] = static_cast<uint8_t>(m);
			}
			pColors[2][3] = 0;
			return true;
		}

		for (uint32_t c = 0; c < 4; c++)
		{
			pColors[1][c] = static_cast<uint8_t>((pColors[0][c] * 5 + pColors[3][c] * 3) / 8);
			pColors[2][c] = static_cast<uint8_t>((pColors[0][c] * 3 + pColors[3][c] * 5) / 8);
		}

		return false;
	}
		
	color_rgba pvrtc4_image::get_pixel(uint32_t x, uint32_t y, uint32_t m) const
	{
		assert((x < m_width) && (y < m_height));

		int block_x0 = (static_cast<int>(x) - 2) >> 2;
		int block_x1 = block_x0 + 1;
		int block_y0 = (static_cast<int>(y) - 2) >> 2;
		int block_y1 = block_y0 + 1;
		
		block_x0 = posmod(block_x0, m_block_width);
		block_x1 = posmod(block_x1, m_block_width);
		block_y0 = posmod(block_y0, m_block_height);
		block_y1 = posmod(block_y1, m_block_height);
		
		if (get_block_uses_transparent_modulation(x >> 2, y >> 2))
		{
			if (m == 0)
				return interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(0), m_blocks(block_x1, block_y0).get_endpoint_5554(0), m_blocks(block_x0, block_y1).get_endpoint_5554(0), m_blocks(block_x1, block_y1).get_endpoint_5554(0));
			else if (m == 3)
				return interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(1), m_blocks(block_x1, block_y0).get_endpoint_5554(1), m_blocks(block_x0, block_y1).get_endpoint_5554(1), m_blocks(block_x1, block_y1).get_endpoint_5554(1));

			color_rgba l(interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(0), m_blocks(block_x1, block_y0).get_endpoint_5554(0), m_blocks(block_x0, block_y1).get_endpoint_5554(0), m_blocks(block_x1, block_y1).get_endpoint_5554(0)));
			color_rgba h(interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(1), m_blocks(block_x1, block_y0).get_endpoint_5554(1), m_blocks(block_x0, block_y1).get_endpoint_5554(1), m_blocks(block_x1, block_y1).get_endpoint_5554(1)));

			return color_rgba((l[0] + h[0]) / 2, (l[1] + h[1]) / 2, (l[2] + h[2]) / 2, (m == 2) ? 0 : (l[3] + h[3]) / 2);
		}
		else
		{
			if (m == 0)
				return interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(0), m_blocks(block_x1, block_y0).get_endpoint_5554(0), m_blocks(block_x0, block_y1).get_endpoint_5554(0), m_blocks(block_x1, block_y1).get_endpoint_5554(0));
			else if (m == 3)
				return interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(1), m_blocks(block_x1, block_y0).get_endpoint_5554(1), m_blocks(block_x0, block_y1).get_endpoint_5554(1), m_blocks(block_x1, block_y1).get_endpoint_5554(1));

			color_rgba l(interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(0), m_blocks(block_x1, block_y0).get_endpoint_5554(0), m_blocks(block_x0, block_y1).get_endpoint_5554(0), m_blocks(block_x1, block_y1).get_endpoint_5554(0)));
			color_rgba h(interpolate(x, y, m_blocks(block_x0, block_y0).get_endpoint_5554(1), m_blocks(block_x1, block_y0).get_endpoint_5554(1), m_blocks(block_x0, block_y1).get_endpoint_5554(1), m_blocks(block_x1, block_y1).get_endpoint_5554(1)));

			if (m == 2)
				return color_rgba((l[0] * 3 + h[0] * 5) / 8, (l[1] * 3 + h[1] * 5) / 8, (l[2] * 3 + h[2] * 5) / 8, (l[3] * 3 + h[3] * 5) / 8);
			else
				return color_rgba((l[0] * 5 + h[0] * 3) / 8, (l[1] * 5 + h[1] * 3) / 8, (l[2] * 5 + h[2] * 3) / 8, (l[3] * 5 + h[3] * 3) / 8);
		}
	}

} // basisu
