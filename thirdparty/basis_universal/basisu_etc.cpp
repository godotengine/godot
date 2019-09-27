// basis_etc.cpp
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
#include "basisu_etc.h"

#define BASISU_DEBUG_ETC_ENCODER 0
#define BASISU_DEBUG_ETC_ENCODER_DEEPER 0

namespace basisu
{
	const uint32_t BASISU_ETC1_CLUSTER_FIT_ORDER_TABLE_SIZE = 165;

	static const struct { uint8_t m_v[4]; } g_cluster_fit_order_tab[BASISU_ETC1_CLUSTER_FIT_ORDER_TABLE_SIZE] =
	{
		{ { 0, 0, 0, 8 } },{ { 0, 5, 2, 1 } },{ { 0, 6, 1, 1 } },{ { 0, 7, 0, 1 } },{ { 0, 7, 1, 0 } },
		{ { 0, 0, 8, 0 } },{ { 0, 0, 3, 5 } },{ { 0, 1, 7, 0 } },{ { 0, 0, 4, 4 } },{ { 0, 0, 2, 6 } },
		{ { 0, 0, 7, 1 } },{ { 0, 0, 1, 7 } },{ { 0, 0, 5, 3 } },{ { 1, 6, 0, 1 } },{ { 0, 0, 6, 2 } },
		{ { 0, 2, 6, 0 } },{ { 2, 4, 2, 0 } },{ { 0, 3, 5, 0 } },{ { 3, 3, 1, 1 } },{ { 4, 2, 0, 2 } },
		{ { 1, 5, 2, 0 } },{ { 0, 5, 3, 0 } },{ { 0, 6, 2, 0 } },{ { 2, 4, 1, 1 } },{ { 5, 1, 0, 2 } },
		{ { 6, 1, 1, 0 } },{ { 3, 3, 0, 2 } },{ { 6, 0, 0, 2 } },{ { 0, 8, 0, 0 } },{ { 6, 1, 0, 1 } },
		{ { 0, 1, 6, 1 } },{ { 1, 6, 1, 0 } },{ { 4, 1, 3, 0 } },{ { 0, 2, 5, 1 } },{ { 5, 0, 3, 0 } },
		{ { 5, 3, 0, 0 } },{ { 0, 1, 5, 2 } },{ { 0, 3, 4, 1 } },{ { 2, 5, 1, 0 } },{ { 1, 7, 0, 0 } },
		{ { 0, 1, 4, 3 } },{ { 6, 0, 2, 0 } },{ { 0, 4, 4, 0 } },{ { 2, 6, 0, 0 } },{ { 0, 2, 4, 2 } },
		{ { 0, 5, 1, 2 } },{ { 0, 6, 0, 2 } },{ { 3, 5, 0, 0 } },{ { 0, 4, 3, 1 } },{ { 3, 4, 1, 0 } },
		{ { 4, 3, 1, 0 } },{ { 1, 5, 0, 2 } },{ { 0, 3, 3, 2 } },{ { 1, 4, 1, 2 } },{ { 0, 4, 2, 2 } },
		{ { 2, 3, 3, 0 } },{ { 4, 4, 0, 0 } },{ { 1, 2, 4, 1 } },{ { 0, 5, 0, 3 } },{ { 0, 1, 3, 4 } },
		{ { 1, 5, 1, 1 } },{ { 1, 4, 2, 1 } },{ { 1, 3, 2, 2 } },{ { 5, 2, 1, 0 } },{ { 1, 3, 3, 1 } },
		{ { 0, 1, 2, 5 } },{ { 1, 1, 5, 1 } },{ { 0, 3, 2, 3 } },{ { 2, 5, 0, 1 } },{ { 3, 2, 2, 1 } },
		{ { 2, 3, 0, 3 } },{ { 1, 4, 3, 0 } },{ { 2, 2, 1, 3 } },{ { 6, 2, 0, 0 } },{ { 1, 0, 6, 1 } },
		{ { 3, 3, 2, 0 } },{ { 7, 1, 0, 0 } },{ { 3, 1, 4, 0 } },{ { 0, 2, 3, 3 } },{ { 0, 4, 1, 3 } },
		{ { 0, 4, 0, 4 } },{ { 0, 1, 0, 7 } },{ { 2, 0, 5, 1 } },{ { 2, 0, 4, 2 } },{ { 3, 0, 2, 3 } },
		{ { 2, 2, 4, 0 } },{ { 2, 2, 3, 1 } },{ { 4, 0, 3, 1 } },{ { 3, 2, 3, 0 } },{ { 2, 3, 2, 1 } },
		{ { 1, 3, 4, 0 } },{ { 7, 0, 1, 0 } },{ { 3, 0, 4, 1 } },{ { 1, 0, 5, 2 } },{ { 8, 0, 0, 0 } },
		{ { 3, 0, 1, 4 } },{ { 4, 1, 1, 2 } },{ { 4, 0, 2, 2 } },{ { 1, 2, 5, 0 } },{ { 4, 2, 1, 1 } },
		{ { 3, 4, 0, 1 } },{ { 2, 0, 3, 3 } },{ { 5, 0, 1, 2 } },{ { 5, 0, 0, 3 } },{ { 2, 4, 0, 2 } },
		{ { 2, 1, 4, 1 } },{ { 4, 0, 1, 3 } },{ { 2, 1, 5, 0 } },{ { 4, 2, 2, 0 } },{ { 4, 0, 4, 0 } },
		{ { 1, 0, 4, 3 } },{ { 1, 4, 0, 3 } },{ { 3, 0, 3, 2 } },{ { 4, 3, 0, 1 } },{ { 0, 1, 1, 6 } },
		{ { 1, 3, 1, 3 } },{ { 0, 2, 2, 4 } },{ { 2, 0, 2, 4 } },{ { 5, 1, 1, 1 } },{ { 3, 0, 5, 0 } },
		{ { 2, 3, 1, 2 } },{ { 3, 0, 0, 5 } },{ { 0, 3, 1, 4 } },{ { 5, 0, 2, 1 } },{ { 2, 1, 3, 2 } },
		{ { 2, 0, 6, 0 } },{ { 3, 1, 3, 1 } },{ { 5, 1, 2, 0 } },{ { 1, 0, 3, 4 } },{ { 1, 1, 6, 0 } },
		{ { 4, 0, 0, 4 } },{ { 2, 0, 1, 5 } },{ { 0, 3, 0, 5 } },{ { 1, 3, 0, 4 } },{ { 4, 1, 2, 1 } },
		{ { 1, 2, 3, 2 } },{ { 3, 1, 0, 4 } },{ { 5, 2, 0, 1 } },{ { 1, 2, 2, 3 } },{ { 3, 2, 1, 2 } },
		{ { 2, 2, 2, 2 } },{ { 6, 0, 1, 1 } },{ { 1, 2, 1, 4 } },{ { 1, 1, 4, 2 } },{ { 3, 2, 0, 3 } },
		{ { 1, 2, 0, 5 } },{ { 1, 0, 7, 0 } },{ { 3, 1, 2, 2 } },{ { 1, 0, 2, 5 } },{ { 2, 0, 0, 6 } },
		{ { 2, 1, 1, 4 } },{ { 2, 2, 0, 4 } },{ { 1, 1, 3, 3 } },{ { 7, 0, 0, 1 } },{ { 1, 0, 0, 7 } },
		{ { 2, 1, 2, 3 } },{ { 4, 1, 0, 3 } },{ { 3, 1, 1, 3 } },{ { 1, 1, 2, 4 } },{ { 2, 1, 0, 5 } },
		{ { 1, 0, 1, 6 } },{ { 0, 2, 1, 5 } },{ { 0, 2, 0, 6 } },{ { 1, 1, 1, 5 } },{ { 1, 1, 0, 6 } }
	};
		
	const int g_etc1_inten_tables[cETC1IntenModifierValues][cETC1SelectorValues] =
	{
		{ -8,  -2,   2,   8 }, { -17,  -5,  5,  17 }, { -29,  -9,   9,  29 }, {  -42, -13, 13,  42 },
		{ -60, -18, 18,  60 }, { -80, -24, 24,  80 }, { -106, -33, 33, 106 }, { -183, -47, 47, 183 }
	};

	const uint8_t g_etc1_to_selector_index[cETC1SelectorValues] = { 2, 3, 1, 0 };
	const uint8_t g_selector_index_to_etc1[cETC1SelectorValues] = { 3, 2, 0, 1 };

	// [flip][subblock][pixel_index]
	const etc_coord2 g_etc1_pixel_coords[2][2][8] =
	{
		{
		  {
			 { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 },
			 { 1, 0 }, { 1, 1 }, { 1, 2 }, { 1, 3 }
		  },
		  {
			 { 2, 0 }, { 2, 1 }, { 2, 2 }, { 2, 3 },
			 { 3, 0 }, { 3, 1 }, { 3, 2 }, { 3, 3 }
		  }
		},
		{
		  {
			 { 0, 0 }, { 1, 0 }, { 2, 0 }, { 3, 0 },
			 { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 }
		  },
		  {
			 { 0, 2 }, { 1, 2 }, { 2, 2 }, { 3, 2 },
			 { 0, 3 }, { 1, 3 }, { 2, 3 }, { 3, 3 }
		  },
		}
	};

	// [flip][subblock][pixel_index]
	const uint32_t g_etc1_pixel_indices[2][2][8] =
	{
		{
			{
				0 + 4 * 0, 0 + 4 * 1, 0 + 4 * 2, 0 + 4 * 3,
				1 + 4 * 0, 1 + 4 * 1, 1 + 4 * 2, 1 + 4 * 3
			},
			{
				2 + 4 * 0, 2 + 4 * 1, 2 + 4 * 2, 2 + 4 * 3,
				3 + 4 * 0, 3 + 4 * 1, 3 + 4 * 2, 3 + 4 * 3
			}
		},
		{
			{
				0 + 4 * 0, 1 + 4 * 0, 2 + 4 * 0, 3 + 4 * 0,
				0 + 4 * 1, 1 + 4 * 1, 2 + 4 * 1, 3 + 4 * 1
			},
			{
				0 + 4 * 2, 1 + 4 * 2, 2 + 4 * 2, 3 + 4 * 2,
				0 + 4 * 3, 1 + 4 * 3, 2 + 4 * 3, 3 + 4 * 3
			},
		}
	};

	uint16_t etc_block::pack_color5(const color_rgba& color, bool scaled, uint32_t bias)
	{
		return pack_color5(color.r, color.g, color.b, scaled, bias);
	}

	uint16_t etc_block::pack_color5(uint32_t r, uint32_t g, uint32_t b, bool scaled, uint32_t bias)
	{
		if (scaled)
		{
			r = (r * 31U + bias) / 255U;
			g = (g * 31U + bias) / 255U;
			b = (b * 31U + bias) / 255U;
		}

		r = minimum(r, 31U);
		g = minimum(g, 31U);
		b = minimum(b, 31U);

		return static_cast<uint16_t>(b | (g << 5U) | (r << 10U));
	}

	color_rgba etc_block::unpack_color5(uint16_t packed_color5, bool scaled, uint32_t alpha)
	{
		uint32_t b = packed_color5 & 31U;
		uint32_t g = (packed_color5 >> 5U) & 31U;
		uint32_t r = (packed_color5 >> 10U) & 31U;

		if (scaled)
		{
			b = (b << 3U) | (b >> 2U);
			g = (g << 3U) | (g >> 2U);
			r = (r << 3U) | (r >> 2U);
		}

		return color_rgba(cNoClamp, r, g, b, minimum(alpha, 255U));
	}

	void etc_block::unpack_color5(color_rgba& result, uint16_t packed_color5, bool scaled)
	{
		result = unpack_color5(packed_color5, scaled, 255);
	}

	void etc_block::unpack_color5(uint32_t& r, uint32_t& g, uint32_t& b, uint16_t packed_color5, bool scaled)
	{
		color_rgba c(unpack_color5(packed_color5, scaled, 0));
		r = c.r;
		g = c.g;
		b = c.b;
	}

	bool etc_block::unpack_color5(color_rgba& result, uint16_t packed_color5, uint16_t packed_delta3, bool scaled, uint32_t alpha)
	{
		color_rgba_i16 dc(unpack_delta3(packed_delta3));

		int b = (packed_color5 & 31U) + dc.b;
		int g = ((packed_color5 >> 5U) & 31U) + dc.g;
		int r = ((packed_color5 >> 10U) & 31U) + dc.r;

		bool success = true;
		if (static_cast<uint32_t>(r | g | b) > 31U)
		{
			success = false;
			r = clamp<int>(r, 0, 31);
			g = clamp<int>(g, 0, 31);
			b = clamp<int>(b, 0, 31);
		}

		if (scaled)
		{
			b = (b << 3U) | (b >> 2U);
			g = (g << 3U) | (g >> 2U);
			r = (r << 3U) | (r >> 2U);
		}

		result.set_noclamp_rgba(r, g, b, minimum(alpha, 255U));
		return success;
	}

	bool etc_block::unpack_color5(uint32_t& r, uint32_t& g, uint32_t& b, uint16_t packed_color5, uint16_t packed_delta3, bool scaled, uint32_t alpha)
	{
		color_rgba result;
		const bool success = unpack_color5(result, packed_color5, packed_delta3, scaled, alpha);
		r = result.r;
		g = result.g;
		b = result.b;
		return success;
	}

	uint16_t etc_block::pack_delta3(const color_rgba_i16& color)
	{
		return pack_delta3(color.r, color.g, color.b);
	}

	uint16_t etc_block::pack_delta3(int r, int g, int b)
	{
		assert((r >= cETC1ColorDeltaMin) && (r <= cETC1ColorDeltaMax));
		assert((g >= cETC1ColorDeltaMin) && (g <= cETC1ColorDeltaMax));
		assert((b >= cETC1ColorDeltaMin) && (b <= cETC1ColorDeltaMax));
		if (r < 0) r += 8;
		if (g < 0) g += 8;
		if (b < 0) b += 8;
		return static_cast<uint16_t>(b | (g << 3) | (r << 6));
	}

	color_rgba_i16 etc_block::unpack_delta3(uint16_t packed_delta3)
	{
		int r = (packed_delta3 >> 6) & 7;
		int g = (packed_delta3 >> 3) & 7;
		int b = packed_delta3 & 7;
		if (r >= 4) r -= 8;
		if (g >= 4) g -= 8;
		if (b >= 4) b -= 8;
		return color_rgba_i16(r, g, b, 255);
	}

	void etc_block::unpack_delta3(int& r, int& g, int& b, uint16_t packed_delta3)
	{
		r = (packed_delta3 >> 6) & 7;
		g = (packed_delta3 >> 3) & 7;
		b = packed_delta3 & 7;
		if (r >= 4) r -= 8;
		if (g >= 4) g -= 8;
		if (b >= 4) b -= 8;
	}

	uint16_t etc_block::pack_color4(const color_rgba& color, bool scaled, uint32_t bias)
	{
		return pack_color4(color.r, color.g, color.b, scaled, bias);
	}

	uint16_t etc_block::pack_color4(uint32_t r, uint32_t g, uint32_t b, bool scaled, uint32_t bias)
	{
		if (scaled)
		{
			r = (r * 15U + bias) / 255U;
			g = (g * 15U + bias) / 255U;
			b = (b * 15U + bias) / 255U;
		}

		r = minimum(r, 15U);
		g = minimum(g, 15U);
		b = minimum(b, 15U);

		return static_cast<uint16_t>(b | (g << 4U) | (r << 8U));
	}

	color_rgba etc_block::unpack_color4(uint16_t packed_color4, bool scaled, uint32_t alpha)
	{
		uint32_t b = packed_color4 & 15U;
		uint32_t g = (packed_color4 >> 4U) & 15U;
		uint32_t r = (packed_color4 >> 8U) & 15U;

		if (scaled)
		{
			b = (b << 4U) | b;
			g = (g << 4U) | g;
			r = (r << 4U) | r;
		}

		return color_rgba(cNoClamp, r, g, b, minimum(alpha, 255U));
	}

	void etc_block::unpack_color4(uint32_t& r, uint32_t& g, uint32_t& b, uint16_t packed_color4, bool scaled)
	{
		color_rgba c(unpack_color4(packed_color4, scaled, 0));
		r = c.r;
		g = c.g;
		b = c.b;
	}

	void etc_block::get_diff_subblock_colors(color_rgba* pDst, uint16_t packed_color5, uint32_t table_idx)
	{
		assert(table_idx < cETC1IntenModifierValues);
		const int *pInten_modifer_table = &g_etc1_inten_tables[table_idx][0];

		uint32_t r, g, b;
		unpack_color5(r, g, b, packed_color5, true);

		const int ir = static_cast<int>(r), ig = static_cast<int>(g), ib = static_cast<int>(b);

		const int y0 = pInten_modifer_table[0];
		pDst[0].set(ir + y0, ig + y0, ib + y0, 255);

		const int y1 = pInten_modifer_table[1];
		pDst[1].set(ir + y1, ig + y1, ib + y1, 255);

		const int y2 = pInten_modifer_table[2];
		pDst[2].set(ir + y2, ig + y2, ib + y2, 255);

		const int y3 = pInten_modifer_table[3];
		pDst[3].set(ir + y3, ig + y3, ib + y3, 255);
	}

	bool etc_block::get_diff_subblock_colors(color_rgba* pDst, uint16_t packed_color5, uint16_t packed_delta3, uint32_t table_idx)
	{
		assert(table_idx < cETC1IntenModifierValues);
		const int *pInten_modifer_table = &g_etc1_inten_tables[table_idx][0];

		uint32_t r, g, b;
		bool success = unpack_color5(r, g, b, packed_color5, packed_delta3, true);

		const int ir = static_cast<int>(r), ig = static_cast<int>(g), ib = static_cast<int>(b);

		const int y0 = pInten_modifer_table[0];
		pDst[0].set(ir + y0, ig + y0, ib + y0, 255);

		const int y1 = pInten_modifer_table[1];
		pDst[1].set(ir + y1, ig + y1, ib + y1, 255);

		const int y2 = pInten_modifer_table[2];
		pDst[2].set(ir + y2, ig + y2, ib + y2, 255);

		const int y3 = pInten_modifer_table[3];
		pDst[3].set(ir + y3, ig + y3, ib + y3, 255);

		return success;
	}

	void etc_block::get_abs_subblock_colors(color_rgba* pDst, uint16_t packed_color4, uint32_t table_idx)
	{
		assert(table_idx < cETC1IntenModifierValues);
		const int *pInten_modifer_table = &g_etc1_inten_tables[table_idx][0];

		uint32_t r, g, b;
		unpack_color4(r, g, b, packed_color4, true);

		const int ir = static_cast<int>(r), ig = static_cast<int>(g), ib = static_cast<int>(b);

		const int y0 = pInten_modifer_table[0];
		pDst[0].set(ir + y0, ig + y0, ib + y0, 255);

		const int y1 = pInten_modifer_table[1];
		pDst[1].set(ir + y1, ig + y1, ib + y1, 255);

		const int y2 = pInten_modifer_table[2];
		pDst[2].set(ir + y2, ig + y2, ib + y2, 255);

		const int y3 = pInten_modifer_table[3];
		pDst[3].set(ir + y3, ig + y3, ib + y3, 255);
	}
		
	bool unpack_etc1(const etc_block& block, color_rgba *pDst, bool preserve_alpha)
	{
		const bool diff_flag = block.get_diff_bit();
		const bool flip_flag = block.get_flip_bit();
		const uint32_t table_index0 = block.get_inten_table(0);
		const uint32_t table_index1 = block.get_inten_table(1);

		color_rgba subblock_colors0[4];
		color_rgba subblock_colors1[4];

		if (diff_flag)
		{
			const uint16_t base_color5 = block.get_base5_color();
			const uint16_t delta_color3 = block.get_delta3_color();
			etc_block::get_diff_subblock_colors(subblock_colors0, base_color5, table_index0);

			if (!etc_block::get_diff_subblock_colors(subblock_colors1, base_color5, delta_color3, table_index1))
				return false;
		}
		else
		{
			const uint16_t base_color4_0 = block.get_base4_color(0);
			etc_block::get_abs_subblock_colors(subblock_colors0, base_color4_0, table_index0);

			const uint16_t base_color4_1 = block.get_base4_color(1);
			etc_block::get_abs_subblock_colors(subblock_colors1, base_color4_1, table_index1);
		}

		if (preserve_alpha)
		{
			if (flip_flag)
			{
				for (uint32_t y = 0; y < 2; y++)
				{
					pDst[0].set_rgb(subblock_colors0[block.get_selector(0, y)]);
					pDst[1].set_rgb(subblock_colors0[block.get_selector(1, y)]);
					pDst[2].set_rgb(subblock_colors0[block.get_selector(2, y)]);
					pDst[3].set_rgb(subblock_colors0[block.get_selector(3, y)]);
					pDst += 4;
				}

				for (uint32_t y = 2; y < 4; y++)
				{
					pDst[0].set_rgb(subblock_colors1[block.get_selector(0, y)]);
					pDst[1].set_rgb(subblock_colors1[block.get_selector(1, y)]);
					pDst[2].set_rgb(subblock_colors1[block.get_selector(2, y)]);
					pDst[3].set_rgb(subblock_colors1[block.get_selector(3, y)]);
					pDst += 4;
				}
			}
			else
			{
				for (uint32_t y = 0; y < 4; y++)
				{
					pDst[0].set_rgb(subblock_colors0[block.get_selector(0, y)]);
					pDst[1].set_rgb(subblock_colors0[block.get_selector(1, y)]);
					pDst[2].set_rgb(subblock_colors1[block.get_selector(2, y)]);
					pDst[3].set_rgb(subblock_colors1[block.get_selector(3, y)]);
					pDst += 4;
				}
			}
		}
		else
		{
			if (flip_flag)
			{
				// 0000
				// 0000
				// 1111
				// 1111
				for (uint32_t y = 0; y < 2; y++)
				{
					pDst[0] = subblock_colors0[block.get_selector(0, y)];
					pDst[1] = subblock_colors0[block.get_selector(1, y)];
					pDst[2] = subblock_colors0[block.get_selector(2, y)];
					pDst[3] = subblock_colors0[block.get_selector(3, y)];
					pDst += 4;
				}

				for (uint32_t y = 2; y < 4; y++)
				{
					pDst[0] = subblock_colors1[block.get_selector(0, y)];
					pDst[1] = subblock_colors1[block.get_selector(1, y)];
					pDst[2] = subblock_colors1[block.get_selector(2, y)];
					pDst[3] = subblock_colors1[block.get_selector(3, y)];
					pDst += 4;
				}
			}
			else
			{
				// 0011
				// 0011
				// 0011
				// 0011
				for (uint32_t y = 0; y < 4; y++)
				{
					pDst[0] = subblock_colors0[block.get_selector(0, y)];
					pDst[1] = subblock_colors0[block.get_selector(1, y)];
					pDst[2] = subblock_colors1[block.get_selector(2, y)];
					pDst[3] = subblock_colors1[block.get_selector(3, y)];
					pDst += 4;
				}
			}
		}

		return true;
	}

	inline int extend_6_to_8(uint32_t n)
	{
		return (n << 2) | (n >> 4);
	}

	inline int extend_7_to_8(uint32_t n)
	{
		return (n << 1) | (n >> 6);
	}

	inline int extend_4_to_8(uint32_t n)
	{
		return (n << 4) | n;
	}
		
	uint64_t etc_block::evaluate_etc1_error(const color_rgba* pBlock_pixels, bool perceptual, int subblock_index) const
	{
		color_rgba unpacked_block[16];

		unpack_etc1(*this, unpacked_block);

		uint64_t total_error = 0;

		if (subblock_index < 0)
		{
			for (uint32_t i = 0; i < 16; i++)
				total_error += color_distance(perceptual, pBlock_pixels[i], unpacked_block[i], false);
		}
		else
		{
			const bool flip_bit = get_flip_bit();

			for (uint32_t i = 0; i < 8; i++)
			{
				const uint32_t idx = g_etc1_pixel_indices[flip_bit][subblock_index][i];

				total_error += color_distance(perceptual, pBlock_pixels[idx], unpacked_block[idx], false);
			}
		}

		return total_error;
	}

	void etc_block::get_subblock_pixels(color_rgba* pPixels, int subblock_index) const
	{
		if (subblock_index < 0)
			unpack_etc1(*this, pPixels);
		else
		{
			color_rgba unpacked_block[16];

			unpack_etc1(*this, unpacked_block);

			const bool flip_bit = get_flip_bit();

			for (uint32_t i = 0; i < 8; i++)
			{
				const uint32_t idx = g_etc1_pixel_indices[flip_bit][subblock_index][i];

				pPixels[i] = unpacked_block[idx];
			}
		}
	}
								
	bool etc1_optimizer::compute()
	{
		assert(m_pResult->m_pSelectors);

		if ((m_pParams->m_pForce_selectors) || (m_pParams->m_pEval_solution_override))
		{
			assert(m_pParams->m_quality >= cETCQualitySlow);
		}

		const uint32_t n = m_pParams->m_num_src_pixels;

		if (m_pParams->m_cluster_fit)
		{
			if (m_pParams->m_quality == cETCQualityFast)
				compute_internal_cluster_fit(4);
			else if (m_pParams->m_quality == cETCQualityMedium)
				compute_internal_cluster_fit(32);
			else if (m_pParams->m_quality == cETCQualitySlow)
				compute_internal_cluster_fit(64);
			else
				compute_internal_cluster_fit(BASISU_ETC1_CLUSTER_FIT_ORDER_TABLE_SIZE);
		}
		else
			compute_internal_neighborhood(m_br, m_bg, m_bb);

		if (!m_best_solution.m_valid)
		{
			m_pResult->m_error = UINT32_MAX;
			return false;
		}

		const uint8_t* pSelectors = &m_best_solution.m_selectors[0];

#ifdef BASISU_BUILD_DEBUG
		if (m_pParams->m_pEval_solution_override == nullptr)
		{
			color_rgba block_colors[4];
			m_best_solution.m_coords.get_block_colors(block_colors);

			const color_rgba* pSrc_pixels = m_pParams->m_pSrc_pixels;
			uint64_t actual_error = 0;
			for (uint32_t i = 0; i < n; i++)
			{
				if ((m_pParams->m_perceptual) && (m_pParams->m_quality >= cETCQualitySlow))
					actual_error += color_distance(true, pSrc_pixels[i], block_colors[pSelectors[i]], false);
				else
					actual_error += color_distance(pSrc_pixels[i], block_colors[pSelectors[i]], false);
			}
			assert(actual_error == m_best_solution.m_error);
		}
#endif      

		m_pResult->m_error = m_best_solution.m_error;

		m_pResult->m_block_color_unscaled = m_best_solution.m_coords.m_unscaled_color;
		m_pResult->m_block_color4 = m_best_solution.m_coords.m_color4;

		m_pResult->m_block_inten_table = m_best_solution.m_coords.m_inten_table;
		memcpy(m_pResult->m_pSelectors, pSelectors, n);
		m_pResult->m_n = n;

		return true;
	}

	void etc1_optimizer::refine_solution(uint32_t max_refinement_trials)
	{
		// Now we have the input block, the avg. color of the input pixels, a set of trial selector indices, and the block color+intensity index.
		// Now, for each component, attempt to refine the current solution by solving a simple linear equation. For example, for 4 colors:
		// The goal is:
		// pixel0 - (block_color+inten_table[selector0]) + pixel1 - (block_color+inten_table[selector1]) + pixel2 - (block_color+inten_table[selector2]) + pixel3 - (block_color+inten_table[selector3]) = 0
		// Rearranging this:
		// (pixel0 + pixel1 + pixel2 + pixel3) - (block_color+inten_table[selector0]) - (block_color+inten_table[selector1]) - (block_color+inten_table[selector2]) - (block_color+inten_table[selector3]) = 0
		// (pixel0 + pixel1 + pixel2 + pixel3) - block_color - inten_table[selector0] - block_color-inten_table[selector1] - block_color-inten_table[selector2] - block_color-inten_table[selector3] = 0
		// (pixel0 + pixel1 + pixel2 + pixel3) - 4*block_color - inten_table[selector0] - inten_table[selector1] - inten_table[selector2] - inten_table[selector3] = 0
		// (pixel0 + pixel1 + pixel2 + pixel3) - 4*block_color - (inten_table[selector0] + inten_table[selector1] + inten_table[selector2] + inten_table[selector3]) = 0
		// (pixel0 + pixel1 + pixel2 + pixel3)/4 - block_color - (inten_table[selector0] + inten_table[selector1] + inten_table[selector2] + inten_table[selector3])/4 = 0
		// block_color = (pixel0 + pixel1 + pixel2 + pixel3)/4 - (inten_table[selector0] + inten_table[selector1] + inten_table[selector2] + inten_table[selector3])/4
		// So what this means:
		// optimal_block_color = avg_input - avg_inten_delta
		// So the optimal block color can be computed by taking the average block color and subtracting the current average of the intensity delta.
		// Unfortunately, optimal_block_color must then be quantized to 555 or 444 so it's not always possible to improve matters using this formula.
		// Also, the above formula is for unclamped intensity deltas. The actual implementation takes into account clamping.

		const uint32_t n = m_pParams->m_num_src_pixels;

		for (uint32_t refinement_trial = 0; refinement_trial < max_refinement_trials; refinement_trial++)
		{
			const uint8_t* pSelectors = &m_best_solution.m_selectors[0];
			const int* pInten_table = g_etc1_inten_tables[m_best_solution.m_coords.m_inten_table];

			int delta_sum_r = 0, delta_sum_g = 0, delta_sum_b = 0;
			const color_rgba base_color(m_best_solution.m_coords.get_scaled_color());
			for (uint32_t r = 0; r < n; r++)
			{
				const uint32_t s = *pSelectors++;
				const int yd_temp = pInten_table[s];
				// Compute actual delta being applied to each pixel, taking into account clamping.
				delta_sum_r += clamp<int>(base_color.r + yd_temp, 0, 255) - base_color.r;
				delta_sum_g += clamp<int>(base_color.g + yd_temp, 0, 255) - base_color.g;
				delta_sum_b += clamp<int>(base_color.b + yd_temp, 0, 255) - base_color.b;
			}

			if ((!delta_sum_r) && (!delta_sum_g) && (!delta_sum_b))
				break;

			const float avg_delta_r_f = static_cast<float>(delta_sum_r) / n;
			const float avg_delta_g_f = static_cast<float>(delta_sum_g) / n;
			const float avg_delta_b_f = static_cast<float>(delta_sum_b) / n;
			const int br1 = clamp<int>(static_cast<uint32_t>((m_avg_color[0] - avg_delta_r_f) * m_limit / 255.0f + .5f), 0, m_limit);
			const int bg1 = clamp<int>(static_cast<uint32_t>((m_avg_color[1] - avg_delta_g_f) * m_limit / 255.0f + .5f), 0, m_limit);
			const int bb1 = clamp<int>(static_cast<uint32_t>((m_avg_color[2] - avg_delta_b_f) * m_limit / 255.0f + .5f), 0, m_limit);

#if BASISU_DEBUG_ETC_ENCODER_DEEPER
			printf("Refinement trial %u, avg_delta %f %f %f\n", refinement_trial, avg_delta_r_f, avg_delta_g_f, avg_delta_b_f);
#endif

			if (!evaluate_solution(etc1_solution_coordinates(br1, bg1, bb1, 0, m_pParams->m_use_color4), m_trial_solution, &m_best_solution))
				break;

		}  // refinement_trial
	}

	void etc1_optimizer::compute_internal_neighborhood(int scan_r, int scan_g, int scan_b)
	{
		if (m_best_solution.m_error == 0)
			return;

		const uint32_t n = m_pParams->m_num_src_pixels;
		const int scan_delta_size = m_pParams->m_scan_delta_size;

		// Scan through a subset of the 3D lattice centered around the avg block color trying each 3D (555 or 444) lattice point as a potential block color.
		// Each time a better solution is found try to refine the current solution's block color based of the current selectors and intensity table index.
		for (int zdi = 0; zdi < scan_delta_size; zdi++)
		{
			const int zd = m_pParams->m_pScan_deltas[zdi];
			const int mbb = scan_b + zd;
			if (mbb < 0) continue; else if (mbb > m_limit) break;

			for (int ydi = 0; ydi < scan_delta_size; ydi++)
			{
				const int yd = m_pParams->m_pScan_deltas[ydi];
				const int mbg = scan_g + yd;
				if (mbg < 0) continue; else if (mbg > m_limit) break;

				for (int xdi = 0; xdi < scan_delta_size; xdi++)
				{
					const int xd = m_pParams->m_pScan_deltas[xdi];
					const int mbr = scan_r + xd;
					if (mbr < 0) continue; else if (mbr > m_limit) break;

					etc1_solution_coordinates coords(mbr, mbg, mbb, 0, m_pParams->m_use_color4);

					if (!evaluate_solution(coords, m_trial_solution, &m_best_solution))
						continue;

					if (m_pParams->m_refinement)
					{
						refine_solution((m_pParams->m_quality == cETCQualityFast) ? 2 : (((xd | yd | zd) == 0) ? 4 : 2));
					}

				} // xdi
			} // ydi
		} // zdi
	}

	void etc1_optimizer::compute_internal_cluster_fit(uint32_t total_perms_to_try)
	{
		if ((!m_best_solution.m_valid) || ((m_br != m_best_solution.m_coords.m_unscaled_color.r) || (m_bg != m_best_solution.m_coords.m_unscaled_color.g) || (m_bb != m_best_solution.m_coords.m_unscaled_color.b)))
		{
			evaluate_solution(etc1_solution_coordinates(m_br, m_bg, m_bb, 0, m_pParams->m_use_color4), m_trial_solution, &m_best_solution);
		}

		if ((m_best_solution.m_error == 0) || (!m_best_solution.m_valid))
			return;

		for (uint32_t i = 0; i < total_perms_to_try; i++)
		{
			int delta_sum_r = 0, delta_sum_g = 0, delta_sum_b = 0;

			const int *pInten_table = g_etc1_inten_tables[m_best_solution.m_coords.m_inten_table];
			const color_rgba base_color(m_best_solution.m_coords.get_scaled_color());

			const uint8_t *pNum_selectors = g_cluster_fit_order_tab[i].m_v;

			for (uint32_t q = 0; q < 4; q++)
			{
				const int yd_temp = pInten_table[q];

				delta_sum_r += pNum_selectors[q] * (clamp<int>(base_color.r + yd_temp, 0, 255) - base_color.r);
				delta_sum_g += pNum_selectors[q] * (clamp<int>(base_color.g + yd_temp, 0, 255) - base_color.g);
				delta_sum_b += pNum_selectors[q] * (clamp<int>(base_color.b + yd_temp, 0, 255) - base_color.b);
			}

			if ((!delta_sum_r) && (!delta_sum_g) && (!delta_sum_b))
				continue;

			const float avg_delta_r_f = static_cast<float>(delta_sum_r) / 8;
			const float avg_delta_g_f = static_cast<float>(delta_sum_g) / 8;
			const float avg_delta_b_f = static_cast<float>(delta_sum_b) / 8;

			const int br1 = clamp<int>(static_cast<uint32_t>((m_avg_color[0] - avg_delta_r_f) * m_limit / 255.0f + .5f), 0, m_limit);
			const int bg1 = clamp<int>(static_cast<uint32_t>((m_avg_color[1] - avg_delta_g_f) * m_limit / 255.0f + .5f), 0, m_limit);
			const int bb1 = clamp<int>(static_cast<uint32_t>((m_avg_color[2] - avg_delta_b_f) * m_limit / 255.0f + .5f), 0, m_limit);

#if BASISU_DEBUG_ETC_ENCODER_DEEPER
			printf("Second refinement trial %u, avg_delta %f %f %f\n", i, avg_delta_r_f, avg_delta_g_f, avg_delta_b_f);
#endif

			evaluate_solution(etc1_solution_coordinates(br1, bg1, bb1, 0, m_pParams->m_use_color4), m_trial_solution, &m_best_solution);

			if (m_best_solution.m_error == 0)
				break;
		}
	}

	void etc1_optimizer::init(const params& params, results& result)
	{
		m_pParams = &params;
		m_pResult = &result;

		const uint32_t n = m_pParams->m_num_src_pixels;

		m_selectors.resize(n);
		m_best_selectors.resize(n);
		m_temp_selectors.resize(n);
		m_trial_solution.m_selectors.resize(n);
		m_best_solution.m_selectors.resize(n);

		m_limit = m_pParams->m_use_color4 ? 15 : 31;

		vec3F avg_color(0.0f);

		m_luma.resize(n);
		m_sorted_luma_indices.resize(n);
		m_sorted_luma.resize(n);
		
		for (uint32_t i = 0; i < n; i++)
		{
			const color_rgba& c = m_pParams->m_pSrc_pixels[i];
			const vec3F fc(c.r, c.g, c.b);

			avg_color += fc;

			m_luma[i] = static_cast<uint16_t>(c.r + c.g + c.b);
			m_sorted_luma_indices[i] = i;
		}
		avg_color /= static_cast<float>(n);
		m_avg_color = avg_color;

		m_br = clamp<int>(static_cast<uint32_t>(m_avg_color[0] * m_limit / 255.0f + .5f), 0, m_limit);
		m_bg = clamp<int>(static_cast<uint32_t>(m_avg_color[1] * m_limit / 255.0f + .5f), 0, m_limit);
		m_bb = clamp<int>(static_cast<uint32_t>(m_avg_color[2] * m_limit / 255.0f + .5f), 0, m_limit);

#if BASISU_DEBUG_ETC_ENCODER_DEEPER
		printf("Avg block color: %u %u %u\n", m_br, m_bg, m_bb);
#endif

		if (m_pParams->m_quality <= cETCQualityMedium)
		{
			indirect_sort(n, &m_sorted_luma_indices[0], &m_luma[0]);

			m_pSorted_luma = &m_sorted_luma[0];
			m_pSorted_luma_indices = &m_sorted_luma_indices[0];
			
			for (uint32_t i = 0; i < n; i++)
				m_pSorted_luma[i] = m_luma[m_pSorted_luma_indices[i]];
		}

		m_best_solution.m_coords.clear();
		m_best_solution.m_valid = false;
		m_best_solution.m_error = UINT64_MAX;

		m_solutions_tried.clear();
	}

	bool etc1_optimizer::evaluate_solution_slow(const etc1_solution_coordinates& coords, potential_solution& trial_solution, potential_solution* pBest_solution)
	{
		uint32_t k = coords.m_unscaled_color.r | (coords.m_unscaled_color.g << 8) | (coords.m_unscaled_color.b << 16);
		if (!m_solutions_tried.insert(k).second)
			return false;

#if BASISU_DEBUG_ETC_ENCODER_DEEPER
		printf("Eval solution: %u %u %u\n", coords.m_unscaled_color.r, coords.m_unscaled_color.g, coords.m_unscaled_color.b);
#endif

		trial_solution.m_valid = false;

		if (m_pParams->m_constrain_against_base_color5)
		{
			const int dr = (int)coords.m_unscaled_color.r - (int)m_pParams->m_base_color5.r;
			const int dg = (int)coords.m_unscaled_color.g - (int)m_pParams->m_base_color5.g;
			const int db = (int)coords.m_unscaled_color.b - (int)m_pParams->m_base_color5.b;

			if ((minimum(dr, dg, db) < cETC1ColorDeltaMin) || (maximum(dr, dg, db) > cETC1ColorDeltaMax))
			{
#if BASISU_DEBUG_ETC_ENCODER_DEEPER
				printf("Eval failed due to constraint from %u %u %u\n", m_pParams->m_base_color5.r, m_pParams->m_base_color5.g, m_pParams->m_base_color5.b);
#endif
				return false;
			}
		}

		const color_rgba base_color(coords.get_scaled_color());

		const uint32_t n = m_pParams->m_num_src_pixels;
		assert(trial_solution.m_selectors.size() == n);

		trial_solution.m_error = UINT64_MAX;

		const uint8_t *pSelectors_to_use = m_pParams->m_pForce_selectors;

		for (uint32_t inten_table = 0; inten_table < cETC1IntenModifierValues; inten_table++)
		{
			const int* pInten_table = g_etc1_inten_tables[inten_table];

			color_rgba block_colors[4];
			for (uint32_t s = 0; s < 4; s++)
			{
				const int yd = pInten_table[s];
				block_colors[s].set(base_color.r + yd, base_color.g + yd, base_color.b + yd, 255);
			}

			uint64_t total_error = 0;

			const color_rgba* pSrc_pixels = m_pParams->m_pSrc_pixels;
			for (uint32_t c = 0; c < n; c++)
			{
				const color_rgba& src_pixel = *pSrc_pixels++;

				uint32_t best_selector_index = 0;
				uint32_t best_error = 0;

				if (pSelectors_to_use)
				{
					best_selector_index = pSelectors_to_use[c];
					best_error = color_distance(m_pParams->m_perceptual, src_pixel, block_colors[best_selector_index], false);
				}
				else
				{
					best_error = color_distance(m_pParams->m_perceptual, src_pixel, block_colors[0], false);

					uint32_t trial_error = color_distance(m_pParams->m_perceptual, src_pixel, block_colors[1], false);
					if (trial_error < best_error)
					{
						best_error = trial_error;
						best_selector_index = 1;
					}

					trial_error = color_distance(m_pParams->m_perceptual, src_pixel, block_colors[2], false);
					if (trial_error < best_error)
					{
						best_error = trial_error;
						best_selector_index = 2;
					}

					trial_error = color_distance(m_pParams->m_perceptual, src_pixel, block_colors[3], false);
					if (trial_error < best_error)
					{
						best_error = trial_error;
						best_selector_index = 3;
					}
				}

				m_temp_selectors[c] = static_cast<uint8_t>(best_selector_index);

				total_error += best_error;
				if ((m_pParams->m_pEval_solution_override == nullptr) && (total_error >= trial_solution.m_error))
					break;
			}

			if (m_pParams->m_pEval_solution_override)
			{
				if (!(*m_pParams->m_pEval_solution_override)(total_error, *m_pParams, block_colors, &m_temp_selectors[0], coords))
					return false;
			}

			if (total_error < trial_solution.m_error)
			{
				trial_solution.m_error = total_error;
				trial_solution.m_coords.m_inten_table = inten_table;
				trial_solution.m_selectors.swap(m_temp_selectors);
				trial_solution.m_valid = true;
			}
		}
		trial_solution.m_coords.m_unscaled_color = coords.m_unscaled_color;
		trial_solution.m_coords.m_color4 = m_pParams->m_use_color4;

#if BASISU_DEBUG_ETC_ENCODER_DEEPER
		printf("Eval done: %u error: %I64u best error so far: %I64u\n", (trial_solution.m_error < pBest_solution->m_error), trial_solution.m_error, pBest_solution->m_error);
#endif

		bool success = false;
		if (pBest_solution)
		{
			if (trial_solution.m_error < pBest_solution->m_error)
			{
				*pBest_solution = trial_solution;
				success = true;
			}
		}

		return success;
	}

	bool etc1_optimizer::evaluate_solution_fast(const etc1_solution_coordinates& coords, potential_solution& trial_solution, potential_solution* pBest_solution)
	{
		uint32_t k = coords.m_unscaled_color.r | (coords.m_unscaled_color.g << 8) | (coords.m_unscaled_color.b << 16);
		if (!m_solutions_tried.insert(k).second)
			return false;

#if BASISU_DEBUG_ETC_ENCODER_DEEPER
		printf("Eval solution fast: %u %u %u\n", coords.m_unscaled_color.r, coords.m_unscaled_color.g, coords.m_unscaled_color.b);
#endif

		if (m_pParams->m_constrain_against_base_color5)
		{
			const int dr = (int)coords.m_unscaled_color.r - (int)m_pParams->m_base_color5.r;
			const int dg = (int)coords.m_unscaled_color.g - (int)m_pParams->m_base_color5.g;
			const int db = (int)coords.m_unscaled_color.b - (int)m_pParams->m_base_color5.b;

			if ((minimum(dr, dg, db) < cETC1ColorDeltaMin) || (maximum(dr, dg, db) > cETC1ColorDeltaMax))
			{
				trial_solution.m_valid = false;

#if BASISU_DEBUG_ETC_ENCODER_DEEPER
				printf("Eval failed due to constraint from %u %u %u\n", m_pParams->m_base_color5.r, m_pParams->m_base_color5.g, m_pParams->m_base_color5.b);
#endif
				return false;
			}
		}

		const color_rgba base_color(coords.get_scaled_color());

		const uint32_t n = m_pParams->m_num_src_pixels;
		assert(trial_solution.m_selectors.size() == n);

		trial_solution.m_error = UINT64_MAX;

		for (int inten_table = cETC1IntenModifierValues - 1; inten_table >= 0; --inten_table)
		{
			const int* pInten_table = g_etc1_inten_tables[inten_table];

			uint32_t block_inten[4];
			color_rgba block_colors[4];
			for (uint32_t s = 0; s < 4; s++)
			{
				const int yd = pInten_table[s];
				color_rgba block_color(base_color.r + yd, base_color.g + yd, base_color.b + yd, 255);
				block_colors[s] = block_color;
				block_inten[s] = block_color.r + block_color.g + block_color.b;
			}

			// evaluate_solution_fast() enforces/assumesd a total ordering of the input colors along the intensity (1,1,1) axis to more quickly classify the inputs to selectors.
			// The inputs colors have been presorted along the projection onto this axis, and ETC1 block colors are always ordered along the intensity axis, so this classification is fast.
			// 0   1   2   3
			//   01  12  23
			const uint32_t block_inten_midpoints[3] = { block_inten[0] + block_inten[1], block_inten[1] + block_inten[2], block_inten[2] + block_inten[3] };

			uint64_t total_error = 0;
			const color_rgba* pSrc_pixels = m_pParams->m_pSrc_pixels;
			if ((m_pSorted_luma[n - 1] * 2) < block_inten_midpoints[0])
			{
				if (block_inten[0] > m_pSorted_luma[n - 1])
				{
					const uint32_t min_error = iabs((int)block_inten[0] - (int)m_pSorted_luma[n - 1]);
					if (min_error >= trial_solution.m_error)
						continue;
				}

				memset(&m_temp_selectors[0], 0, n);

				for (uint32_t c = 0; c < n; c++)
					total_error += color_distance(block_colors[0], pSrc_pixels[c], false);
			}
			else if ((m_pSorted_luma[0] * 2) >= block_inten_midpoints[2])
			{
				if (m_pSorted_luma[0] > block_inten[3])
				{
					const uint32_t min_error = iabs((int)m_pSorted_luma[0] - (int)block_inten[3]);
					if (min_error >= trial_solution.m_error)
						continue;
				}

				memset(&m_temp_selectors[0], 3, n);

				for (uint32_t c = 0; c < n; c++)
					total_error += color_distance(block_colors[3], pSrc_pixels[c], false);
			}
			else
			{
				uint32_t cur_selector = 0, c;
				for (c = 0; c < n; c++)
				{
					const uint32_t y = m_pSorted_luma[c];
					while ((y * 2) >= block_inten_midpoints[cur_selector])
						if (++cur_selector > 2)
							goto done;
					const uint32_t sorted_pixel_index = m_pSorted_luma_indices[c];
					m_temp_selectors[sorted_pixel_index] = static_cast<uint8_t>(cur_selector);
					total_error += color_distance(block_colors[cur_selector], pSrc_pixels[sorted_pixel_index], false);
				}
			done:
				while (c < n)
				{
					const uint32_t sorted_pixel_index = m_pSorted_luma_indices[c];
					m_temp_selectors[sorted_pixel_index] = 3;
					total_error += color_distance(block_colors[3], pSrc_pixels[sorted_pixel_index], false);
					++c;
				}
			}

			if (total_error < trial_solution.m_error)
			{
				trial_solution.m_error = total_error;
				trial_solution.m_coords.m_inten_table = inten_table;
				trial_solution.m_selectors.swap(m_temp_selectors);
				trial_solution.m_valid = true;
				if (!total_error)
					break;
			}
		}
		trial_solution.m_coords.m_unscaled_color = coords.m_unscaled_color;
		trial_solution.m_coords.m_color4 = m_pParams->m_use_color4;

#if BASISU_DEBUG_ETC_ENCODER_DEEPER
		printf("Eval done: %u error: %I64u best error so far: %I64u\n", (trial_solution.m_error < pBest_solution->m_error), trial_solution.m_error, pBest_solution->m_error);
#endif

		bool success = false;
		if (pBest_solution)
		{
			if (trial_solution.m_error < pBest_solution->m_error)
			{
				*pBest_solution = trial_solution;
				success = true;
			}
		}

		return success;
	}

} // namespace basisu
