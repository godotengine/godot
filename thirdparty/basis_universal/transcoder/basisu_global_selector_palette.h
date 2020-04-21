// basisu_global_selector_palette.h
// Copyright (C) 2019-2020 Binomial LLC. All Rights Reserved.
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
#pragma once
#include "basisu_transcoder_internal.h"
#include <algorithm>

namespace basist
{
	class etc1_global_palette_entry_modifier
	{
	public:
		enum { cTotalBits = 15, cTotalValues = 1 << cTotalBits };

		etc1_global_palette_entry_modifier(uint32_t index = 0)
		{
#ifdef _DEBUG
			static bool s_tested;
			if (!s_tested)
			{
				s_tested = true;
				for (uint32_t i = 0; i < cTotalValues; i++)
				{
					etc1_global_palette_entry_modifier m(i);
					etc1_global_palette_entry_modifier n = m;

					assert(n.get_index() == i);
				}
			}
#endif

			set_index(index);
		}

		void set_index(uint32_t index)
		{
			assert(index < cTotalValues);
			m_rot = index & 3;
			m_flip = (index >> 2) & 1;
			m_inv = (index >> 3) & 1;
			m_contrast = (index >> 4) & 3;
			m_shift = (index >> 6) & 1;
			m_median = (index >> 7) & 1;
			m_div = (index >> 8) & 1;
			m_rand = (index >> 9) & 1;
			m_dilate = (index >> 10) & 1;
			m_shift_x = (index >> 11) & 1;
			m_shift_y = (index >> 12) & 1;
			m_erode = (index >> 13) & 1;
			m_high_pass = (index >> 14) & 1;
		}

		uint32_t get_index() const
		{
			return m_rot | (m_flip << 2) | (m_inv << 3) | (m_contrast << 4) | (m_shift << 6) | (m_median << 7) | (m_div << 8) | (m_rand << 9) | (m_dilate << 10) | (m_shift_x << 11) | (m_shift_y << 12) | (m_erode << 13) | (m_high_pass << 14);
		}

		void clear()
		{
			basisu::clear_obj(*this);
		}

		uint8_t m_contrast;
		bool m_rand;
		bool m_median;
		bool m_div;
		bool m_shift;
		bool m_inv;
		bool m_flip;
		bool m_dilate;
		bool m_shift_x;
		bool m_shift_y;
		bool m_erode;
		bool m_high_pass;
		uint8_t m_rot;
	};

	enum modifier_types
	{
		cModifierContrast,
		cModifierRand,
		cModifierMedian,
		cModifierDiv,
		cModifierShift,
		cModifierInv,
		cModifierFlippedAndRotated,
		cModifierDilate,
		cModifierShiftX,
		cModifierShiftY,
		cModifierErode,
		cModifierHighPass,
		cTotalModifiers
	};

#define ETC1_GLOBAL_SELECTOR_CODEBOOK_MAX_MOD_BITS (etc1_global_palette_entry_modifier::cTotalBits)

	struct etc1_selector_palette_entry
	{
		etc1_selector_palette_entry()
		{
			clear();
		}

		void clear()
		{
			basisu::clear_obj(*this);
		}

		uint8_t operator[] (uint32_t i) const { assert(i < 16); return m_selectors[i]; }
		uint8_t&operator[] (uint32_t i) { assert(i < 16); return m_selectors[i]; }

		void set_uint32(uint32_t v)
		{
			for (uint32_t byte_index = 0; byte_index < 4; byte_index++)
			{
				uint32_t b = (v >> (byte_index * 8)) & 0xFF;

				m_selectors[byte_index * 4 + 0] = b & 3;
				m_selectors[byte_index * 4 + 1] = (b >> 2) & 3;
				m_selectors[byte_index * 4 + 2] = (b >> 4) & 3;
				m_selectors[byte_index * 4 + 3] = (b >> 6) & 3;
			}
		}

		uint32_t get_uint32() const
		{
			return get_byte(0) | (get_byte(1) << 8) | (get_byte(2) << 16) | (get_byte(3) << 24);
		}

		uint32_t get_byte(uint32_t byte_index) const
		{
			assert(byte_index < 4);

			return m_selectors[byte_index * 4 + 0] |
				(m_selectors[byte_index * 4 + 1] << 2) |
				(m_selectors[byte_index * 4 + 2] << 4) |
				(m_selectors[byte_index * 4 + 3] << 6);
		}

		uint8_t operator()(uint32_t x, uint32_t y) const { assert((x < 4) && (y < 4)); return m_selectors[x + y * 4]; }
		uint8_t&operator()(uint32_t x, uint32_t y) { assert((x < 4) && (y < 4)); return m_selectors[x + y * 4]; }

		uint32_t calc_distance(const etc1_selector_palette_entry &other) const
		{
			uint32_t dist = 0;
			for (uint32_t i = 0; i < 8; i++)
			{
				int delta = static_cast<int>(m_selectors[i]) - static_cast<int>(other.m_selectors[i]);
				dist += delta * delta;
			}
			return dist;
		}

#if 0
		uint32_t calc_hamming_dist(const etc1_selector_palette_entry &other) const
		{
			uint32_t dist = 0;
			for (uint32_t i = 0; i < 4; i++)
				dist += g_hamming_dist[get_byte(i) ^ other.get_byte(i)];
			return dist;
		}
#endif

		etc1_selector_palette_entry get_inverted() const
		{
			etc1_selector_palette_entry result;

			for (uint32_t i = 0; i < 16; i++)
				result.m_selectors[i] = 3 - m_selectors[i];

			return result;
		}

		etc1_selector_palette_entry get_divided() const
		{
			etc1_selector_palette_entry result;

			const uint8_t div_selector[4] = { 2, 0, 3, 1 };

			for (uint32_t i = 0; i < 16; i++)
				result.m_selectors[i] = div_selector[m_selectors[i]];

			return result;
		}

		etc1_selector_palette_entry get_shifted(int delta) const
		{
			etc1_selector_palette_entry result;

			for (uint32_t i = 0; i < 16; i++)
				result.m_selectors[i] = static_cast<uint8_t>(basisu::clamp<int>(m_selectors[i] + delta, 0, 3));

			return result;
		}

		etc1_selector_palette_entry get_randomized() const
		{
			uint32_t seed = get_uint32();

			etc1_selector_palette_entry result;

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					int s = (*this)(x, y);

					// between 0 and 10
					uint32_t i = basisd_urand(seed, 6) + basisd_urand(seed, 6);
					if (i == 0)
						s -= 2;
					else if (i == 10)
						s += 2;
					else if (i < 3)
						s -= 1;
					else if (i > 7)
						s += 1;

					result(x, y) = static_cast<uint8_t>(basisu::clamp<int>(s, 0, 3));
				}
			}

			return result;
		}

		etc1_selector_palette_entry get_contrast(int table_index) const
		{
			assert(table_index < 4);

			etc1_selector_palette_entry result;

			static const uint8_t s_contrast_tables[4][4] =
			{
				{ 0, 1, 2, 3 }, // not used
				{ 0, 0, 3, 3 },
				{ 1, 1, 2, 2 },
				{ 1, 1, 3, 3 }
			};

			for (uint32_t i = 0; i < 16; i++)
			{
				result[i] = s_contrast_tables[table_index][(*this)[i]];
			}

			return result;
		}

		etc1_selector_palette_entry get_dilated() const
		{
			etc1_selector_palette_entry result;

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t max_selector = 0;

					for (int yd = -1; yd <= 1; yd++)
					{
						int fy = y + yd;
						if ((fy < 0) || (fy > 3))
							continue;

						for (int xd = -1; xd <= 1; xd++)
						{
							int fx = x + xd;
							if ((fx < 0) || (fx > 3))
								continue;

							max_selector = basisu::maximum<uint32_t>(max_selector, (*this)(fx, fy));
						}
					}

					result(x, y) = static_cast<uint8_t>(max_selector);
				}
			}

			return result;
		}

		etc1_selector_palette_entry get_eroded() const
		{
			etc1_selector_palette_entry result;

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t min_selector = 99;

					for (int yd = -1; yd <= 1; yd++)
					{
						int fy = y + yd;
						if ((fy < 0) || (fy > 3))
							continue;

						for (int xd = -1; xd <= 1; xd++)
						{
							int fx = x + xd;
							if ((fx < 0) || (fx > 3))
								continue;

							min_selector = basisu::minimum<uint32_t>(min_selector, (*this)(fx, fy));
						}
					}

					result(x, y) = static_cast<uint8_t>(min_selector);
				}
			}

			return result;
		}

		etc1_selector_palette_entry get_shift_x() const
		{
			etc1_selector_palette_entry result;

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					int sx = x - 1;
					if (sx < 0)
						sx = 0;

					result(x, y) = (*this)(sx, y);
				}
			}

			return result;
		}

		etc1_selector_palette_entry get_shift_y() const
		{
			etc1_selector_palette_entry result;

			for (uint32_t y = 0; y < 4; y++)
			{
				int sy = y - 1;
				if (sy < 0)
					sy = 3;

				for (uint32_t x = 0; x < 4; x++)
					result(x, y) = (*this)(x, sy);
			}

			return result;
		}

		etc1_selector_palette_entry get_median() const
		{
			etc1_selector_palette_entry result;

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					// ABC
					// D F
					// GHI

					uint8_t selectors[8];
					uint32_t n = 0;

					for (int yd = -1; yd <= 1; yd++)
					{
						int fy = y + yd;
						if ((fy < 0) || (fy > 3))
							continue;

						for (int xd = -1; xd <= 1; xd++)
						{
							if ((xd | yd) == 0)
								continue;

							int fx = x + xd;
							if ((fx < 0) || (fx > 3))
								continue;

							selectors[n++] = (*this)(fx, fy);
						}
					}

					std::sort(selectors, selectors + n);

					result(x, y) = selectors[n / 2];
				}
			}

			return result;
		}

		etc1_selector_palette_entry get_high_pass() const
		{
			etc1_selector_palette_entry result;

			static const int kernel[3][3] =
			{
				{ 0,  -1,  0 },
				{ -1,  8, -1 },
				{ 0,  -1,  0 }
			};

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					// ABC
					// D F
					// GHI

					int sum = 0;

					for (int yd = -1; yd <= 1; yd++)
					{
						int fy = y + yd;
						fy = basisu::clamp<int>(fy, 0, 3);

						for (int xd = -1; xd <= 1; xd++)
						{
							int fx = x + xd;
							fx = basisu::clamp<int>(fx, 0, 3);

							int k = (*this)(fx, fy);
							sum += k * kernel[yd + 1][xd + 1];
						}
					}

					sum = sum / 4;

					result(x, y) = static_cast<uint8_t>(basisu::clamp<int>(sum, 0, 3));
				}
			}

			return result;
		}

		etc1_selector_palette_entry get_flipped_and_rotated(bool flip, uint32_t rotation_index) const
		{
			etc1_selector_palette_entry temp;

			if (flip)
			{
				for (uint32_t y = 0; y < 4; y++)
					for (uint32_t x = 0; x < 4; x++)
						temp(x, y) = (*this)(x, 3 - y);
			}
			else
			{
				temp = *this;
			}

			etc1_selector_palette_entry result;

			switch (rotation_index)
			{
			case 0:
				result = temp;
				break;
			case 1:
				for (uint32_t y = 0; y < 4; y++)
					for (uint32_t x = 0; x < 4; x++)
						result(x, y) = temp(y, 3 - x);
				break;
			case 2:
				for (uint32_t y = 0; y < 4; y++)
					for (uint32_t x = 0; x < 4; x++)
						result(x, y) = temp(3 - x, 3 - y);
				break;
			case 3:
				for (uint32_t y = 0; y < 4; y++)
					for (uint32_t x = 0; x < 4; x++)
						result(x, y) = temp(3 - y, x);
				break;
			default:
				assert(0);
				break;
			}

			return result;
		}

		etc1_selector_palette_entry get_modified(const etc1_global_palette_entry_modifier &modifier) const
		{
			etc1_selector_palette_entry r(*this);

			if (modifier.m_shift_x)
				r = r.get_shift_x();

			if (modifier.m_shift_y)
				r = r.get_shift_y();

			r = r.get_flipped_and_rotated(modifier.m_flip != 0, modifier.m_rot);

			if (modifier.m_dilate)
				r = r.get_dilated();

			if (modifier.m_erode)
				r = r.get_eroded();

			if (modifier.m_high_pass)
				r = r.get_high_pass();

			if (modifier.m_rand)
				r = r.get_randomized();

			if (modifier.m_div)
				r = r.get_divided();

			if (modifier.m_shift)
				r = r.get_shifted(1);

			if (modifier.m_contrast)
				r = r.get_contrast(modifier.m_contrast);

			if (modifier.m_inv)
				r = r.get_inverted();

			if (modifier.m_median)
				r = r.get_median();

			return r;
		}

		etc1_selector_palette_entry apply_modifier(modifier_types mod_type, const etc1_global_palette_entry_modifier &modifier) const
		{
			switch (mod_type)
			{
			case cModifierContrast:
				return get_contrast(modifier.m_contrast);
			case cModifierRand:
				return get_randomized();
			case cModifierMedian:
				return get_median();
			case cModifierDiv:
				return get_divided();
			case cModifierShift:
				return get_shifted(1);
			case cModifierInv:
				return get_inverted();
			case cModifierFlippedAndRotated:
				return get_flipped_and_rotated(modifier.m_flip != 0, modifier.m_rot);
			case cModifierDilate:
				return get_dilated();
			case cModifierShiftX:
				return get_shift_x();
			case cModifierShiftY:
				return get_shift_y();
			case cModifierErode:
				return get_eroded();
			case cModifierHighPass:
				return get_high_pass();
			default:
				assert(0);
				break;
			}

			return *this;
		}

		etc1_selector_palette_entry get_modified(const etc1_global_palette_entry_modifier &modifier, uint32_t num_order, const modifier_types *pOrder) const
		{
			etc1_selector_palette_entry r(*this);

			for (uint32_t i = 0; i < num_order; i++)
			{
				r = r.apply_modifier(pOrder[i], modifier);
			}

			return r;
		}

		bool operator< (const etc1_selector_palette_entry &other) const
		{
			for (uint32_t i = 0; i < 16; i++)
			{
				if (m_selectors[i] < other.m_selectors[i])
					return true;
				else if (m_selectors[i] != other.m_selectors[i])
					return false;
			}

			return false;
		}

		bool operator== (const etc1_selector_palette_entry &other) const
		{
			for (uint32_t i = 0; i < 16; i++)
			{
				if (m_selectors[i] != other.m_selectors[i])
					return false;
			}

			return true;
		}

	private:
		uint8_t m_selectors[16];
	};

	typedef std::vector<etc1_selector_palette_entry> etc1_selector_palette_entry_vec;

	extern const uint32_t g_global_selector_cb[];
	extern const uint32_t g_global_selector_cb_size;

#define ETC1_GLOBAL_SELECTOR_CODEBOOK_MAX_PAL_BITS (12)

	struct etc1_global_selector_codebook_entry_id
	{
		uint32_t m_palette_index;
		etc1_global_palette_entry_modifier m_modifier;

		etc1_global_selector_codebook_entry_id(uint32_t palette_index, const etc1_global_palette_entry_modifier &modifier) : m_palette_index(palette_index), m_modifier(modifier) { }

		etc1_global_selector_codebook_entry_id() { }

		void set(uint32_t palette_index, const etc1_global_palette_entry_modifier &modifier) { m_palette_index = palette_index; m_modifier = modifier; }
	};

	typedef std::vector<etc1_global_selector_codebook_entry_id> etc1_global_selector_codebook_entry_id_vec;

	class etc1_global_selector_codebook
	{
	public:
		etc1_global_selector_codebook() { }
		etc1_global_selector_codebook(uint32_t N, const uint32_t *pEntries) { init(N, pEntries); }

		void init(uint32_t N, const uint32_t* pEntries);

		void print_code(FILE *pFile);

		void clear()
		{
			m_palette.clear();
		}

		uint32_t size() const { return (uint32_t)m_palette.size(); }

		const etc1_selector_palette_entry_vec &get_palette() const
		{
			return m_palette;
		}

		etc1_selector_palette_entry get_entry(uint32_t palette_index) const
		{
			return m_palette[palette_index];
		}

		etc1_selector_palette_entry get_entry(uint32_t palette_index, const etc1_global_palette_entry_modifier &modifier) const
		{
			return m_palette[palette_index].get_modified(modifier);
		}

		etc1_selector_palette_entry get_entry(const etc1_global_selector_codebook_entry_id &id) const
		{
			return m_palette[id.m_palette_index].get_modified(id.m_modifier);
		}

		etc1_selector_palette_entry_vec m_palette;
	};

} // namespace basist
