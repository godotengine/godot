// basisu_transcoder.cpp
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

#include "basisu_transcoder.h"
#include <limits.h>
#include <vector>

// The supported .basis file header version. Keep in sync with BASIS_FILE_VERSION.
#define BASISD_SUPPORTED_BASIS_VERSION (0x13)

// Set to 1 for fuzz testing. This will disable all CRC16 checks on headers and compressed data.
#ifndef BASISU_NO_HEADER_OR_DATA_CRC16_CHECKS
#define BASISU_NO_HEADER_OR_DATA_CRC16_CHECKS 0
#endif

#ifndef BASISD_SUPPORT_DXT1
#define BASISD_SUPPORT_DXT1 1
#endif

#ifndef BASISD_SUPPORT_DXT5A
#define BASISD_SUPPORT_DXT5A 1
#endif

// Disable all BC7 transcoders if necessary (useful when cross compiling to Javascript)
#if defined(BASISD_SUPPORT_BC7) && !BASISD_SUPPORT_BC7
	#ifndef BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY
	#define BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY 0
	#endif
	#ifndef BASISD_SUPPORT_BC7_MODE5
	#define BASISD_SUPPORT_BC7_MODE5 0
	#endif
#endif // !BASISD_SUPPORT_BC7

// BC7 mode 6 opaque only is the highest quality (compared to ETC1), but the tables are massive.
// For web/mobile use you probably should disable this.
#ifndef BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY
#define BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY 1
#endif

// BC7 mode 5 supports both opaque and opaque+alpha textures, and uses substantially less memory than BC7 mode 6 and even BC1.
#ifndef BASISD_SUPPORT_BC7_MODE5
#define BASISD_SUPPORT_BC7_MODE5 1
#endif

#ifndef BASISD_SUPPORT_PVRTC1
#define BASISD_SUPPORT_PVRTC1 1
#endif

#ifndef BASISD_SUPPORT_ETC2_EAC_A8
#define BASISD_SUPPORT_ETC2_EAC_A8 1
#endif

#ifndef BASISD_SUPPORT_ASTC
#define BASISD_SUPPORT_ASTC 1
#endif

// Note that if BASISD_SUPPORT_ATC is enabled, BASISD_SUPPORT_DXT5A should also be enabled for alpha support.
#ifndef BASISD_SUPPORT_ATC
#define BASISD_SUPPORT_ATC 1
#endif

// Support for ETC2 EAC R11 and ETC2 EAC RG11
#ifndef BASISD_SUPPORT_ETC2_EAC_RG11
#define BASISD_SUPPORT_ETC2_EAC_RG11 1
#endif

// If BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY is 1, opaque blocks will be transcoded to ASTC at slightly higher quality (higher than BC1), but the transcoder tables will be 2x as large.
// This impacts grayscale and grayscale+alpha textures the most.
#ifndef BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY
	#ifdef __EMSCRIPTEN__
		// Let's assume size matters more than quality when compiling with emscripten.
		#define BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY 0
	#else
		// Compiling native, so an extra 64K lookup table is probably acceptable.
		#define BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY 1
	#endif
#endif

#ifndef BASISD_SUPPORT_FXT1
#define BASISD_SUPPORT_FXT1 1
#endif

#ifndef BASISD_SUPPORT_PVRTC2
#define BASISD_SUPPORT_PVRTC2 1
#endif

#if BASISD_SUPPORT_PVRTC2
#if !BASISD_SUPPORT_ATC
#error BASISD_SUPPORT_ATC must be 1 if BASISD_SUPPORT_PVRTC2 is 1
#endif
#endif

#if BASISD_SUPPORT_ATC
#if !BASISD_SUPPORT_DXT5A
#error BASISD_SUPPORT_DXT5A must be 1 if BASISD_SUPPORT_ATC is 1
#endif
#endif

#define BASISD_WRITE_NEW_BC7_TABLES					0
#define BASISD_WRITE_NEW_BC7_MODE5_TABLES			0
#define BASISD_WRITE_NEW_DXT1_TABLES				0
#define BASISD_WRITE_NEW_ETC2_EAC_A8_TABLES		0
#define BASISD_WRITE_NEW_ASTC_TABLES				0
#define BASISD_WRITE_NEW_ATC_TABLES					0
#define BASISD_WRITE_NEW_ETC2_EAC_R11_TABLES		0

#ifndef BASISD_ENABLE_DEBUG_FLAGS
#define BASISD_ENABLE_DEBUG_FLAGS	0
#endif

namespace basisu
{
	bool g_debug_printf;

	void enable_debug_printf(bool enabled)
	{
		g_debug_printf = enabled;
	}

	void debug_printf(const char* pFmt, ...)
	{
#if BASISU_DEVEL_MESSAGES	
		g_debug_printf = true;
#endif
		if (g_debug_printf)
		{
			va_list args;
			va_start(args, pFmt);
			vprintf(pFmt, args);
			va_end(args);
		}
	}
} // namespace basisu

namespace basist
{
#if BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY
#include "basisu_transcoder_tables_bc7_m6.inc"
#endif

#if BASISD_ENABLE_DEBUG_FLAGS
	static uint32_t g_debug_flags = 0;
#endif

	uint32_t get_debug_flags()
	{
#if BASISD_ENABLE_DEBUG_FLAGS
		return g_debug_flags;
#else
		return 0;
#endif
	}

	void set_debug_flags(uint32_t f)
	{
		(void)f;
#if BASISD_ENABLE_DEBUG_FLAGS
		g_debug_flags = f;
#endif
	}
	uint16_t crc16(const void* r, size_t size, uint16_t crc)
	{
		crc = ~crc;

		const uint8_t* p = reinterpret_cast<const uint8_t*>(r);
		for (; size; --size)
		{
			const uint16_t q = *p++ ^ (crc >> 8);
			uint16_t k = (q >> 4) ^ q;
			crc = (((crc << 8) ^ k) ^ (k << 5)) ^ (k << 12);
		}

		return static_cast<uint16_t>(~crc);
	}

	const uint32_t g_global_selector_cb[] =
#include "basisu_global_selector_cb.h"
		;

	const uint32_t g_global_selector_cb_size = sizeof(g_global_selector_cb) / sizeof(g_global_selector_cb[0]);

	void etc1_global_selector_codebook::init(uint32_t N, const uint32_t* pEntries)
	{
		m_palette.resize(N);
		for (uint32_t i = 0; i < N; i++)
			m_palette[i].set_uint32(pEntries[i]);
	}

	void etc1_global_selector_codebook::print_code(FILE* pFile)
	{
		fprintf(pFile, "{\n");
		for (uint32_t i = 0; i < m_palette.size(); i++)
		{
			fprintf(pFile, "0x%X,", m_palette[i].get_uint32());
			if ((i & 15) == 15)
				fprintf(pFile, "\n");
		}
		fprintf(pFile, "\n}\n");
	}

	enum etc_constants
	{
		cETC1BytesPerBlock = 8U,

		cETC1SelectorBits = 2U,
		cETC1SelectorValues = 1U << cETC1SelectorBits,
		cETC1SelectorMask = cETC1SelectorValues - 1U,

		cETC1BlockShift = 2U,
		cETC1BlockSize = 1U << cETC1BlockShift,

		cETC1LSBSelectorIndicesBitOffset = 0,
		cETC1MSBSelectorIndicesBitOffset = 16,

		cETC1FlipBitOffset = 32,
		cETC1DiffBitOffset = 33,

		cETC1IntenModifierNumBits = 3,
		cETC1IntenModifierValues = 1 << cETC1IntenModifierNumBits,
		cETC1RightIntenModifierTableBitOffset = 34,
		cETC1LeftIntenModifierTableBitOffset = 37,

		// Base+Delta encoding (5 bit bases, 3 bit delta)
		cETC1BaseColorCompNumBits = 5,
		cETC1BaseColorCompMax = 1 << cETC1BaseColorCompNumBits,

		cETC1DeltaColorCompNumBits = 3,
		cETC1DeltaColorComp = 1 << cETC1DeltaColorCompNumBits,
		cETC1DeltaColorCompMax = 1 << cETC1DeltaColorCompNumBits,

		cETC1BaseColor5RBitOffset = 59,
		cETC1BaseColor5GBitOffset = 51,
		cETC1BaseColor5BBitOffset = 43,

		cETC1DeltaColor3RBitOffset = 56,
		cETC1DeltaColor3GBitOffset = 48,
		cETC1DeltaColor3BBitOffset = 40,

		// Absolute (non-delta) encoding (two 4-bit per component bases)
		cETC1AbsColorCompNumBits = 4,
		cETC1AbsColorCompMax = 1 << cETC1AbsColorCompNumBits,

		cETC1AbsColor4R1BitOffset = 60,
		cETC1AbsColor4G1BitOffset = 52,
		cETC1AbsColor4B1BitOffset = 44,

		cETC1AbsColor4R2BitOffset = 56,
		cETC1AbsColor4G2BitOffset = 48,
		cETC1AbsColor4B2BitOffset = 40,

		cETC1ColorDeltaMin = -4,
		cETC1ColorDeltaMax = 3,

		// Delta3:
		// 0   1   2   3   4   5   6   7
		// 000 001 010 011 100 101 110 111
		// 0   1   2   3   -4  -3  -2  -1
	};

#define DECLARE_ETC1_INTEN_TABLE(name, N) \
	static const int name[cETC1IntenModifierValues][cETC1SelectorValues] = \
	{ \
		{ N * -8,  N * -2,   N * 2,   N * 8 },{ N * -17,  N * -5,  N * 5,  N * 17 },{ N * -29,  N * -9,   N * 9,  N * 29 },{ N * -42, N * -13, N * 13,  N * 42 }, \
		{ N * -60, N * -18, N * 18,  N * 60 },{ N * -80, N * -24, N * 24,  N * 80 },{ N * -106, N * -33, N * 33, N * 106 },{ N * -183, N * -47, N * 47, N * 183 } \
	};

	DECLARE_ETC1_INTEN_TABLE(g_etc1_inten_tables, 1);
	DECLARE_ETC1_INTEN_TABLE(g_etc1_inten_tables16, 16);
	DECLARE_ETC1_INTEN_TABLE(g_etc1_inten_tables48, 3 * 16);
	
	static const uint8_t g_etc_5_to_8[32] = { 0, 8, 16, 24, 33, 41, 49, 57, 66, 74, 82, 90, 99, 107, 115, 123, 132, 140, 148, 156, 165, 173, 181, 189, 198, 206, 214, 222, 231, 239, 247, 255 };

	struct decoder_etc_block
	{
		// big endian uint64:
		// bit ofs:  56  48  40  32  24  16   8   0
		// byte ofs: b0, b1, b2, b3, b4, b5, b6, b7 
		union
		{
			uint64_t m_uint64;

			uint32_t m_uint32[2];

			uint8_t m_bytes[8];

			struct
			{
				signed m_dred2 : 3;
				uint32_t m_red1 : 5;

				signed m_dgreen2 : 3;
				uint32_t m_green1 : 5;

				signed m_dblue2 : 3;
				uint32_t m_blue1 : 5;

				uint32_t m_flip : 1;
				uint32_t m_diff : 1;
				uint32_t m_cw2 : 3;
				uint32_t m_cw1 : 3;

				uint32_t m_selectors;
			} m_differential;
		};

		inline void clear()
		{
			assert(sizeof(*this) == 8);
			basisu::clear_obj(*this);
		}

		inline void set_byte_bits(uint32_t ofs, uint32_t num, uint32_t bits)
		{
			assert((ofs + num) <= 64U);
			assert(num && (num < 32U));
			assert((ofs >> 3) == ((ofs + num - 1) >> 3));
			assert(bits < (1U << num));
			const uint32_t byte_ofs = 7 - (ofs >> 3);
			const uint32_t byte_bit_ofs = ofs & 7;
			const uint32_t mask = (1 << num) - 1;
			m_bytes[byte_ofs] &= ~(mask << byte_bit_ofs);
			m_bytes[byte_ofs] |= (bits << byte_bit_ofs);
		}

		inline void set_flip_bit(bool flip)
		{
			m_bytes[3] &= ~1;
			m_bytes[3] |= static_cast<uint8_t>(flip);
		}

		inline void set_diff_bit(bool diff)
		{
			m_bytes[3] &= ~2;
			m_bytes[3] |= (static_cast<uint32_t>(diff) << 1);
		}

		// Sets intensity modifier table (0-7) used by subblock subblock_id (0 or 1)
		inline void set_inten_table(uint32_t subblock_id, uint32_t t)
		{
			assert(subblock_id < 2);
			assert(t < 8);
			const uint32_t ofs = subblock_id ? 2 : 5;
			m_bytes[3] &= ~(7 << ofs);
			m_bytes[3] |= (t << ofs);
		}

		// Selector "val" ranges from 0-3 and is a direct index into g_etc1_inten_tables.
		inline void set_selector(uint32_t x, uint32_t y, uint32_t val)
		{
			assert((x | y | val) < 4);
			const uint32_t bit_index = x * 4 + y;

			uint8_t* p = &m_bytes[7 - (bit_index >> 3)];

			const uint32_t byte_bit_ofs = bit_index & 7;
			const uint32_t mask = 1 << byte_bit_ofs;

			static const uint8_t s_selector_index_to_etc1[4] = { 3, 2, 0, 1 };
			const uint32_t etc1_val = s_selector_index_to_etc1[val];

			const uint32_t lsb = etc1_val & 1;
			const uint32_t msb = etc1_val >> 1;

			p[0] &= ~mask;
			p[0] |= (lsb << byte_bit_ofs);

			p[-2] &= ~mask;
			p[-2] |= (msb << byte_bit_ofs);
		}

		// Returned encoded selector value ranges from 0-3 (this is NOT a direct index into g_etc1_inten_tables, see get_selector())
		inline uint32_t get_raw_selector(uint32_t x, uint32_t y) const
		{
			assert((x | y) < 4);

			const uint32_t bit_index = x * 4 + y;
			const uint32_t byte_bit_ofs = bit_index & 7;
			const uint8_t* p = &m_bytes[7 - (bit_index >> 3)];
			const uint32_t lsb = (p[0] >> byte_bit_ofs) & 1;
			const uint32_t msb = (p[-2] >> byte_bit_ofs) & 1;
			const uint32_t val = lsb | (msb << 1);

			return val;
		}

		// Returned selector value ranges from 0-3 and is a direct index into g_etc1_inten_tables.
		inline uint32_t get_selector(uint32_t x, uint32_t y) const
		{
			static const uint8_t s_etc1_to_selector_index[cETC1SelectorValues] = { 2, 3, 1, 0 };
			return s_etc1_to_selector_index[get_raw_selector(x, y)];
		}

		inline void set_raw_selector_bits(uint32_t bits)
		{
			m_bytes[4] = static_cast<uint8_t>(bits);
			m_bytes[5] = static_cast<uint8_t>(bits >> 8);
			m_bytes[6] = static_cast<uint8_t>(bits >> 16);
			m_bytes[7] = static_cast<uint8_t>(bits >> 24);
		}

		inline bool are_all_selectors_the_same() const
		{
			uint32_t v = *reinterpret_cast<const uint32_t*>(&m_bytes[4]);

			if ((v == 0xFFFFFFFF) || (v == 0xFFFF) || (!v) || (v == 0xFFFF0000))
				return true;

			return false;
		}

		inline void set_raw_selector_bits(uint8_t byte0, uint8_t byte1, uint8_t byte2, uint8_t byte3)
		{
			m_bytes[4] = byte0;
			m_bytes[5] = byte1;
			m_bytes[6] = byte2;
			m_bytes[7] = byte3;
		}

		inline uint32_t get_raw_selector_bits() const
		{
			return m_bytes[4] | (m_bytes[5] << 8) | (m_bytes[6] << 16) | (m_bytes[7] << 24);
		}

		inline void set_base4_color(uint32_t idx, uint16_t c)
		{
			if (idx)
			{
				set_byte_bits(cETC1AbsColor4R2BitOffset, 4, (c >> 8) & 15);
				set_byte_bits(cETC1AbsColor4G2BitOffset, 4, (c >> 4) & 15);
				set_byte_bits(cETC1AbsColor4B2BitOffset, 4, c & 15);
			}
			else
			{
				set_byte_bits(cETC1AbsColor4R1BitOffset, 4, (c >> 8) & 15);
				set_byte_bits(cETC1AbsColor4G1BitOffset, 4, (c >> 4) & 15);
				set_byte_bits(cETC1AbsColor4B1BitOffset, 4, c & 15);
			}
		}

		inline void set_base5_color(uint16_t c)
		{
			set_byte_bits(cETC1BaseColor5RBitOffset, 5, (c >> 10) & 31);
			set_byte_bits(cETC1BaseColor5GBitOffset, 5, (c >> 5) & 31);
			set_byte_bits(cETC1BaseColor5BBitOffset, 5, c & 31);
		}

		void set_delta3_color(uint16_t c)
		{
			set_byte_bits(cETC1DeltaColor3RBitOffset, 3, (c >> 6) & 7);
			set_byte_bits(cETC1DeltaColor3GBitOffset, 3, (c >> 3) & 7);
			set_byte_bits(cETC1DeltaColor3BBitOffset, 3, c & 7);
		}

		void set_block_color4(const color32& c0_unscaled, const color32& c1_unscaled)
		{
			set_diff_bit(false);

			set_base4_color(0, pack_color4(c0_unscaled, false));
			set_base4_color(1, pack_color4(c1_unscaled, false));
		}

		void set_block_color5(const color32& c0_unscaled, const color32& c1_unscaled)
		{
			set_diff_bit(true);

			set_base5_color(pack_color5(c0_unscaled, false));

			int dr = c1_unscaled.r - c0_unscaled.r;
			int dg = c1_unscaled.g - c0_unscaled.g;
			int db = c1_unscaled.b - c0_unscaled.b;

			set_delta3_color(pack_delta3(dr, dg, db));
		}

		bool set_block_color5_check(const color32& c0_unscaled, const color32& c1_unscaled)
		{
			set_diff_bit(true);

			set_base5_color(pack_color5(c0_unscaled, false));

			int dr = c1_unscaled.r - c0_unscaled.r;
			int dg = c1_unscaled.g - c0_unscaled.g;
			int db = c1_unscaled.b - c0_unscaled.b;

			if (((dr < cETC1ColorDeltaMin) || (dr > cETC1ColorDeltaMax)) ||
				((dg < cETC1ColorDeltaMin) || (dg > cETC1ColorDeltaMax)) ||
				((db < cETC1ColorDeltaMin) || (db > cETC1ColorDeltaMax)))
				return false;

			set_delta3_color(pack_delta3(dr, dg, db));

			return true;
		}

		inline uint32_t get_byte_bits(uint32_t ofs, uint32_t num) const
		{
			assert((ofs + num) <= 64U);
			assert(num && (num <= 8U));
			assert((ofs >> 3) == ((ofs + num - 1) >> 3));
			const uint32_t byte_ofs = 7 - (ofs >> 3);
			const uint32_t byte_bit_ofs = ofs & 7;
			return (m_bytes[byte_ofs] >> byte_bit_ofs) & ((1 << num) - 1);
		}

		inline uint16_t get_base5_color() const
		{
			const uint32_t r = get_byte_bits(cETC1BaseColor5RBitOffset, 5);
			const uint32_t g = get_byte_bits(cETC1BaseColor5GBitOffset, 5);
			const uint32_t b = get_byte_bits(cETC1BaseColor5BBitOffset, 5);
			return static_cast<uint16_t>(b | (g << 5U) | (r << 10U));
		}

		inline color32 get_base5_color_unscaled() const
		{
			return color32(m_differential.m_red1, m_differential.m_green1, m_differential.m_blue1, 255);
		}

		inline uint32_t get_inten_table(uint32_t subblock_id) const
		{
			assert(subblock_id < 2);
			const uint32_t ofs = subblock_id ? 2 : 5;
			return (m_bytes[3] >> ofs) & 7;
		}

		static uint16_t pack_color4(const color32& color, bool scaled, uint32_t bias = 127U)
		{
			return pack_color4(color.r, color.g, color.b, scaled, bias);
		}

		static uint16_t pack_color4(uint32_t r, uint32_t g, uint32_t b, bool scaled, uint32_t bias = 127U)
		{
			if (scaled)
			{
				r = (r * 15U + bias) / 255U;
				g = (g * 15U + bias) / 255U;
				b = (b * 15U + bias) / 255U;
			}

			r = basisu::minimum(r, 15U);
			g = basisu::minimum(g, 15U);
			b = basisu::minimum(b, 15U);

			return static_cast<uint16_t>(b | (g << 4U) | (r << 8U));
		}

		static uint16_t pack_color5(const color32& color, bool scaled, uint32_t bias = 127U)
		{
			return pack_color5(color.r, color.g, color.b, scaled, bias);
		}

		static uint16_t pack_color5(uint32_t r, uint32_t g, uint32_t b, bool scaled, uint32_t bias = 127U)
		{
			if (scaled)
			{
				r = (r * 31U + bias) / 255U;
				g = (g * 31U + bias) / 255U;
				b = (b * 31U + bias) / 255U;
			}

			r = basisu::minimum(r, 31U);
			g = basisu::minimum(g, 31U);
			b = basisu::minimum(b, 31U);

			return static_cast<uint16_t>(b | (g << 5U) | (r << 10U));
		}

		uint16_t pack_delta3(const color32& color)
		{
			return pack_delta3(color.r, color.g, color.b);
		}

		uint16_t pack_delta3(int r, int g, int b)
		{
			assert((r >= cETC1ColorDeltaMin) && (r <= cETC1ColorDeltaMax));
			assert((g >= cETC1ColorDeltaMin) && (g <= cETC1ColorDeltaMax));
			assert((b >= cETC1ColorDeltaMin) && (b <= cETC1ColorDeltaMax));
			if (r < 0) r += 8;
			if (g < 0) g += 8;
			if (b < 0) b += 8;
			return static_cast<uint16_t>(b | (g << 3) | (r << 6));
		}

		static color32 unpack_color5(uint16_t packed_color5, bool scaled, uint32_t alpha = 255)
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

			return color32(r, g, b, alpha);
		}

		static void unpack_color5(uint32_t& r, uint32_t& g, uint32_t& b, uint16_t packed_color5, bool scaled)
		{
			color32 c(unpack_color5(packed_color5, scaled, 0));
			r = c.r;
			g = c.g;
			b = c.b;
		}

		static void get_diff_subblock_colors(color32* pDst, uint16_t packed_color5, uint32_t table_idx)
		{
			assert(table_idx < cETC1IntenModifierValues);
			const int* pInten_modifer_table = &g_etc1_inten_tables[table_idx][0];

			uint32_t r, g, b;
			unpack_color5(r, g, b, packed_color5, true);

			const int ir = static_cast<int>(r), ig = static_cast<int>(g), ib = static_cast<int>(b);

			const int y0 = pInten_modifer_table[0];
			pDst[0].set(clamp255(ir + y0), clamp255(ig + y0), clamp255(ib + y0), 255);

			const int y1 = pInten_modifer_table[1];
			pDst[1].set(clamp255(ir + y1), clamp255(ig + y1), clamp255(ib + y1), 255);

			const int y2 = pInten_modifer_table[2];
			pDst[2].set(clamp255(ir + y2), clamp255(ig + y2), clamp255(ib + y2), 255);

			const int y3 = pInten_modifer_table[3];
			pDst[3].set(clamp255(ir + y3), clamp255(ig + y3), clamp255(ib + y3), 255);
		}

		static int clamp255(int x)
		{
			if (x & 0xFFFFFF00)
			{
				if (x < 0)
					x = 0;
				else if (x > 255)
					x = 255;
			}

			return x;
		}

		static void get_block_colors5(color32* pBlock_colors, const color32& base_color5, uint32_t inten_table)
		{
			color32 b(base_color5);

			b.r = (b.r << 3) | (b.r >> 2);
			b.g = (b.g << 3) | (b.g >> 2);
			b.b = (b.b << 3) | (b.b >> 2);

			const int* pInten_table = g_etc1_inten_tables[inten_table];

			pBlock_colors[0].set(clamp255(b.r + pInten_table[0]), clamp255(b.g + pInten_table[0]), clamp255(b.b + pInten_table[0]), 255);
			pBlock_colors[1].set(clamp255(b.r + pInten_table[1]), clamp255(b.g + pInten_table[1]), clamp255(b.b + pInten_table[1]), 255);
			pBlock_colors[2].set(clamp255(b.r + pInten_table[2]), clamp255(b.g + pInten_table[2]), clamp255(b.b + pInten_table[2]), 255);
			pBlock_colors[3].set(clamp255(b.r + pInten_table[3]), clamp255(b.g + pInten_table[3]), clamp255(b.b + pInten_table[3]), 255);
		}

		static void get_block_color5(const color32& base_color5, uint32_t inten_table, uint32_t index, uint32_t& r, uint32_t &g, uint32_t &b)
		{
			assert(index < 4);

			uint32_t br = (base_color5.r << 3) | (base_color5.r >> 2);
			uint32_t bg = (base_color5.g << 3) | (base_color5.g >> 2);
			uint32_t bb = (base_color5.b << 3) | (base_color5.b >> 2);

			const int* pInten_table = g_etc1_inten_tables[inten_table];

			r = clamp255(br + pInten_table[index]);
			g = clamp255(bg + pInten_table[index]);
			b = clamp255(bb + pInten_table[index]);
		}

		static void get_block_color5_r(const color32& base_color5, uint32_t inten_table, uint32_t index, uint32_t &r)
		{
			assert(index < 4);
						
			uint32_t br = (base_color5.r << 3) | (base_color5.r >> 2);

			const int* pInten_table = g_etc1_inten_tables[inten_table];

			r = clamp255(br + pInten_table[index]);
		}

		static void get_block_colors5_g(int* pBlock_colors, const color32& base_color5, uint32_t inten_table)
		{
			const int g = (base_color5.g << 3) | (base_color5.g >> 2);

			const int* pInten_table = g_etc1_inten_tables[inten_table];

			pBlock_colors[0] = clamp255(g + pInten_table[0]);
			pBlock_colors[1] = clamp255(g + pInten_table[1]);
			pBlock_colors[2] = clamp255(g + pInten_table[2]);
			pBlock_colors[3] = clamp255(g + pInten_table[3]);
		}

		static void get_block_colors5_bounds(color32* pBlock_colors, const color32& base_color5, uint32_t inten_table, uint32_t l = 0, uint32_t h = 3)
		{
			color32 b(base_color5);

			b.r = (b.r << 3) | (b.r >> 2);
			b.g = (b.g << 3) | (b.g >> 2);
			b.b = (b.b << 3) | (b.b >> 2);

			const int* pInten_table = g_etc1_inten_tables[inten_table];

			pBlock_colors[0].set(clamp255(b.r + pInten_table[l]), clamp255(b.g + pInten_table[l]), clamp255(b.b + pInten_table[l]), 255);
			pBlock_colors[1].set(clamp255(b.r + pInten_table[h]), clamp255(b.g + pInten_table[h]), clamp255(b.b + pInten_table[h]), 255);
		}

		static void get_block_colors5_bounds_g(uint32_t* pBlock_colors, const color32& base_color5, uint32_t inten_table, uint32_t l = 0, uint32_t h = 3)
		{
			color32 b(base_color5);

			b.g = (b.g << 3) | (b.g >> 2);

			const int* pInten_table = g_etc1_inten_tables[inten_table];

			pBlock_colors[0] = clamp255(b.g + pInten_table[l]);
			pBlock_colors[1] = clamp255(b.g + pInten_table[h]);
		}
	};

	enum dxt_constants
	{
		cDXT1SelectorBits = 2U, cDXT1SelectorValues = 1U << cDXT1SelectorBits, cDXT1SelectorMask = cDXT1SelectorValues - 1U,
		cDXT5SelectorBits = 3U, cDXT5SelectorValues = 1U << cDXT5SelectorBits, cDXT5SelectorMask = cDXT5SelectorValues - 1U,
	};

	static const uint8_t g_etc1_x_selector_unpack[4][256] =
	{
		{
			0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
			0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
			0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
			0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
			0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
			0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
			0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
			0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
		},
		{
			0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
			2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
			0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
			2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
			0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
			2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
			0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
			2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
		},

		{
			0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
			0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
			2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
			2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
			0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
			0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
			2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
			2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
		},

		{
			0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
			2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
			2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
			2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
			2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
		}
	};

	struct dxt1_block
	{
		enum { cTotalEndpointBytes = 2, cTotalSelectorBytes = 4 };

		uint8_t m_low_color[cTotalEndpointBytes];
		uint8_t m_high_color[cTotalEndpointBytes];
		uint8_t m_selectors[cTotalSelectorBytes];

		inline void clear() { basisu::clear_obj(*this); }

		inline uint32_t get_high_color() const { return m_high_color[0] | (m_high_color[1] << 8U); }
		inline uint32_t get_low_color() const { return m_low_color[0] | (m_low_color[1] << 8U); }
		inline void set_low_color(uint16_t c) { m_low_color[0] = static_cast<uint8_t>(c & 0xFF); m_low_color[1] = static_cast<uint8_t>((c >> 8) & 0xFF); }
		inline void set_high_color(uint16_t c) { m_high_color[0] = static_cast<uint8_t>(c & 0xFF); m_high_color[1] = static_cast<uint8_t>((c >> 8) & 0xFF); }
		inline uint32_t get_selector(uint32_t x, uint32_t y) const { assert((x < 4U) && (y < 4U)); return (m_selectors[y] >> (x * cDXT1SelectorBits)) & cDXT1SelectorMask; }
		inline void set_selector(uint32_t x, uint32_t y, uint32_t val) { assert((x < 4U) && (y < 4U) && (val < 4U)); m_selectors[y] &= (~(cDXT1SelectorMask << (x * cDXT1SelectorBits))); m_selectors[y] |= (val << (x * cDXT1SelectorBits)); }

		static uint16_t pack_color(const color32& color, bool scaled, uint32_t bias = 127U)
		{
			uint32_t r = color.r, g = color.g, b = color.b;
			if (scaled)
			{
				r = (r * 31U + bias) / 255U;
				g = (g * 63U + bias) / 255U;
				b = (b * 31U + bias) / 255U;
			}
			return static_cast<uint16_t>(basisu::minimum(b, 31U) | (basisu::minimum(g, 63U) << 5U) | (basisu::minimum(r, 31U) << 11U));
		}

		static uint16_t pack_unscaled_color(uint32_t r, uint32_t g, uint32_t b) { return static_cast<uint16_t>(b | (g << 5U) | (r << 11U)); }
	};

	struct dxt_selector_range
	{
		uint32_t m_low;
		uint32_t m_high;
	};

#if BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY
	static dxt_selector_range g_etc1_to_bc7_selector_ranges[] =
	{
		{ 0, 0 },
		{ 1, 1 },
		{ 2, 2 },
		{ 3, 3 },

		{ 0, 3 },

		{ 1, 3 },
		{ 0, 2 },

		{ 1, 2 },

		{ 2, 3 },
		{ 0, 1 },
	};
	const uint32_t NUM_ETC1_TO_BC7_M6_SELECTOR_RANGES = sizeof(g_etc1_to_bc7_selector_ranges) / sizeof(g_etc1_to_bc7_selector_ranges[0]);

	static uint32_t g_etc1_to_bc7_m6_selector_range_index[4][4];
		
	static const uint8_t g_etc1_to_bc7_selector_mappings[][4] =
	{
#if 1
		{ 5 * 0, 5 * 0, 5 * 0, 5 * 0 },
		{ 5 * 0, 5 * 0, 5 * 0, 5 * 1 },
		{ 5 * 0, 5 * 0, 5 * 0, 5 * 2 },
		{ 5 * 0, 5 * 0, 5 * 0, 5 * 3 },
		{ 5 * 0, 5 * 0, 5 * 1, 5 * 1 },
		{ 5 * 0, 5 * 0, 5 * 1, 5 * 2 },
		{ 5 * 0, 5 * 0, 5 * 1, 5 * 3 },
		{ 5 * 0, 5 * 0, 5 * 2, 5 * 2 },
		{ 5 * 0, 5 * 0, 5 * 2, 5 * 3 },
		{ 5 * 0, 5 * 0, 5 * 3, 5 * 3 },
		{ 5 * 0, 5 * 1, 5 * 1, 5 * 1 },
		{ 5 * 0, 5 * 1, 5 * 1, 5 * 2 },
		{ 5 * 0, 5 * 1, 5 * 1, 5 * 3 },
		{ 5 * 0, 5 * 1, 5 * 2, 5 * 2 },
		{ 5 * 0, 5 * 1, 5 * 2, 5 * 3 },
		{ 5 * 0, 5 * 1, 5 * 3, 5 * 3 },
		{ 5 * 0, 5 * 2, 5 * 2, 5 * 2 },
		{ 5 * 0, 5 * 2, 5 * 2, 5 * 3 },
		{ 5 * 0, 5 * 2, 5 * 3, 5 * 3 },
		{ 5 * 0, 5 * 3, 5 * 3, 5 * 3 },
		{ 5 * 1, 5 * 1, 5 * 1, 5 * 1 },
		{ 5 * 1, 5 * 1, 5 * 1, 5 * 2 },
		{ 5 * 1, 5 * 1, 5 * 1, 5 * 3 },
		{ 5 * 1, 5 * 1, 5 * 2, 5 * 2 },
		{ 5 * 1, 5 * 1, 5 * 2, 5 * 3 },
		{ 5 * 1, 5 * 1, 5 * 3, 5 * 3 },
		{ 5 * 1, 5 * 2, 5 * 2, 5 * 2 },
		{ 5 * 1, 5 * 2, 5 * 2, 5 * 3 },
		{ 5 * 1, 5 * 2, 5 * 3, 5 * 3 },
		{ 5 * 1, 5 * 3, 5 * 3, 5 * 3 },
		{ 5 * 2, 5 * 2, 5 * 2, 5 * 2 },
		{ 5 * 2, 5 * 2, 5 * 2, 5 * 3 },
		{ 5 * 2, 5 * 2, 5 * 3, 5 * 3 },
		{ 5 * 2, 5 * 3, 5 * 3, 5 * 3 },
		{ 5 * 3, 5 * 3, 5 * 3, 5 * 3 },

		{ 0, 1, 2, 3 },
		{ 0, 0, 1, 1 },
		{ 0, 0, 0, 1 },
		{ 0, 2, 4, 6 },
		{ 0, 3, 6, 9 },
		{ 0, 4, 8, 12 },

		{ 0, 4, 9, 15 },
		{ 0, 6, 11, 15 },

		{ 1, 2, 3, 4 },
		{ 1, 3, 5, 7 },

		{ 1, 8, 8, 14 },
#else
		{ 5 * 0, 5 * 0, 5 * 1, 5 * 1 },
		{ 5 * 0, 5 * 0, 5 * 1, 5 * 2 },
		{ 5 * 0, 5 * 0, 5 * 1, 5 * 3 },
		{ 5 * 0, 5 * 0, 5 * 2, 5 * 3 },
		{ 5 * 0, 5 * 1, 5 * 1, 5 * 1 },
		{ 5 * 0, 5 * 1, 5 * 2, 5 * 2 },
		{ 5 * 0, 5 * 1, 5 * 2, 5 * 3 },
		{ 5 * 0, 5 * 2, 5 * 3, 5 * 3 },
		{ 5 * 1, 5 * 2, 5 * 2, 5 * 2 },
#endif
		{ 5 * 1, 5 * 2, 5 * 3, 5 * 3 },
		{ 8, 8, 8, 8 },
	};
	const uint32_t NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS = sizeof(g_etc1_to_bc7_selector_mappings) / sizeof(g_etc1_to_bc7_selector_mappings[0]);

	static uint8_t g_etc1_to_bc7_selector_mappings_inv[NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS][4];

	// encoding from LSB to MSB: low8, high8, error16, size is [32*8][NUM_ETC1_TO_BC7_M6_SELECTOR_RANGES][NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS]
	extern const uint32_t* g_etc1_to_bc7_m6_table[];

	const uint16_t s_bptc_table_aWeight4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

#if BASISD_WRITE_NEW_BC7_TABLES
	static void create_etc1_to_bc7_m6_conversion_table()
	{
		FILE* pFile = NULL;

		pFile = fopen("basisu_decoder_tables_bc7_m6.inc", "w");

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				fprintf(pFile, "static const uint32_t g_etc1_to_bc7_m6_table%u[] = {\n", g + inten * 32);
				uint32_t n = 0;

				for (uint32_t sr = 0; sr < NUM_ETC1_TO_BC7_M6_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1_to_bc7_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1_to_bc7_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 127; hi++)
						{
							for (uint32_t lo = 0; lo <= 127; lo++)
							{
								uint32_t bc7_block_colors[16];

								bc7_block_colors[0] = lo << 1;
								bc7_block_colors[15] = (hi << 1) | 1;

								for (uint32_t i = 1; i < 15; i++)
									bc7_block_colors[i] = (bc7_block_colors[0] * (64 - s_bptc_table_aWeight4[i]) + bc7_block_colors[15] * s_bptc_table_aWeight4[i] + 32) >> 6;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = (int)block_colors[s].g - (int)bc7_block_colors[g_etc1_to_bc7_selector_mappings[m][s]];

									total_err += err * err;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							} // lo

						} // hi

						best_err = basisu::minimum<uint32_t>(best_err, 0xFFFF);

						const uint32_t index = (g + inten * 32) * (NUM_ETC1_TO_BC7_M6_SELECTOR_RANGES * NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS) + (sr * NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS) + m;

						uint32_t v = best_err | (best_lo << 18) | (best_hi << 25);

						fprintf(pFile, "0x%X,", v);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");

					} // m
				} // sr

				fprintf(pFile, "};\n");

			} // g
		} // inten

		fprintf(pFile, "const uint32_t *g_etc1_to_bc7_m6_table[] = {\n");

		for (uint32_t i = 0; i < 32 * 8; i++)
		{
			fprintf(pFile, "g_etc1_to_bc7_m6_table%u, ", i);
			if ((i & 15) == 15)
				fprintf(pFile, "\n");
		}

		fprintf(pFile, "};\n");
		fclose(pFile);
	}
#endif
#endif

	struct etc1_to_dxt1_56_solution
	{
		uint8_t m_lo;
		uint8_t m_hi;
		uint16_t m_err;
	};

#if BASISD_SUPPORT_DXT1
	static dxt_selector_range g_etc1_to_dxt1_selector_ranges[] =
	{
		{ 0, 3 },

		{ 1, 3 },
		{ 0, 2 },

		{ 1, 2 },

		{ 2, 3 },
		{ 0, 1 },
	};

	const uint32_t NUM_ETC1_TO_DXT1_SELECTOR_RANGES = sizeof(g_etc1_to_dxt1_selector_ranges) / sizeof(g_etc1_to_dxt1_selector_ranges[0]);

	static uint32_t g_etc1_to_dxt1_selector_range_index[4][4];

	const uint32_t NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS = 10;
	static const uint8_t g_etc1_to_dxt1_selector_mappings[NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS][4] =
	{
		{ 0, 0, 1, 1 },
		{ 0, 0, 1, 2 },
		{ 0, 0, 1, 3 },
		{ 0, 0, 2, 3 },
		{ 0, 1, 1, 1 },
		{ 0, 1, 2, 2 },
		{ 0, 1, 2, 3 },
		{ 0, 2, 3, 3 },
		{ 1, 2, 2, 2 },
		{ 1, 2, 3, 3 },
	};
	
	static uint8_t g_etc1_to_dxt1_selector_mappings_raw_dxt1_256[NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS][256];
	static uint8_t g_etc1_to_dxt1_selector_mappings_raw_dxt1_inv_256[NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS][256];

	static const etc1_to_dxt1_56_solution g_etc1_to_dxt_6[32 * 8 * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS * NUM_ETC1_TO_DXT1_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_dxt1_6.inc"
	};

	static const etc1_to_dxt1_56_solution g_etc1_to_dxt_5[32 * 8 * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS * NUM_ETC1_TO_DXT1_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_dxt1_5.inc"
	};

	// First saw the idea for optimal BC1 single-color block encoding using lookup tables in ryg_dxt.
	struct bc1_match_entry
	{
		uint8_t m_hi;
		uint8_t m_lo;
	};
	static bc1_match_entry g_bc1_match5_equals_1[256], g_bc1_match6_equals_1[256]; // selector 1, allow equals hi/lo
	static bc1_match_entry g_bc1_match5_equals_0[256], g_bc1_match6_equals_0[256]; // selector 0, allow equals hi/lo

	static void prepare_bc1_single_color_table(bc1_match_entry* pTable, const uint8_t* pExpand, int size0, int size1, int sel)
	{
		for (int i = 0; i < 256; i++)
		{
			int lowest_e = 256;
			for (int lo = 0; lo < size0; lo++)
			{
				for (int hi = 0; hi < size1; hi++)
				{
					const int lo_e = pExpand[lo], hi_e = pExpand[hi];
					int e;

					if (sel == 1)
					{
						// Selector 1
						e = abs(((hi_e * 2 + lo_e) / 3) - i) + ((abs(hi_e - lo_e) >> 5));
					}
					else
					{
						assert(sel == 0);

						// Selector 0
						e = abs(hi_e - i);
					}

					if (e < lowest_e)
					{
						pTable[i].m_hi = static_cast<uint8_t>(hi);
						pTable[i].m_lo = static_cast<uint8_t>(lo);

						lowest_e = e;
					}

				} // hi
			} // lo
		}
	}
#endif // BASISD_SUPPORT_DXT1

#if BASISD_WRITE_NEW_DXT1_TABLES
	static void create_etc1_to_dxt1_5_conversion_table()
	{
		FILE* pFile = nullptr;
		fopen_s(&pFile, "basisu_transcoder_tables_dxt1_5.inc", "w");

		uint32_t n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1_TO_DXT1_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1_to_dxt1_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1_to_dxt1_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 31; hi++)
						{
							for (uint32_t lo = 0; lo <= 31; lo++)
							{
								//if (lo == hi) continue;

								uint32_t colors[4];

								colors[0] = (lo << 3) | (lo >> 2);
								colors[3] = (hi << 3) | (hi >> 2);

								colors[1] = (colors[0] * 2 + colors[3]) / 3;
								colors[2] = (colors[3] * 2 + colors[0]) / 3;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1_to_dxt1_selector_mappings[m][s]];

									total_err += err * err;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						assert(best_err <= 0xFFFF);

						//table[g + inten * 32].m_solutions[sr][m].m_lo = static_cast<uint8_t>(best_lo);
						//table[g + inten * 32].m_solutions[sr][m].m_hi = static_cast<uint8_t>(best_hi);
						//table[g + inten * 32].m_solutions[sr][m].m_err = static_cast<uint16_t>(best_err);

						//assert(best_lo != best_hi);
						fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, (uint32_t)best_err);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m
				} // sr
			} // g
		} // inten

		fclose(pFile);
	}

	static void create_etc1_to_dxt1_6_conversion_table()
	{
		FILE* pFile = nullptr;
		fopen_s(&pFile, "basisu_transcoder_tables_dxt1_6.inc", "w");

		uint32_t n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1_TO_DXT1_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1_to_dxt1_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1_to_dxt1_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 63; hi++)
						{
							for (uint32_t lo = 0; lo <= 63; lo++)
							{
								//if (lo == hi) continue;

								uint32_t colors[4];

								colors[0] = (lo << 2) | (lo >> 4);
								colors[3] = (hi << 2) | (hi >> 4);

								colors[1] = (colors[0] * 2 + colors[3]) / 3;
								colors[2] = (colors[3] * 2 + colors[0]) / 3;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1_to_dxt1_selector_mappings[m][s]];

									total_err += err * err;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						assert(best_err <= 0xFFFF);

						//table[g + inten * 32].m_solutions[sr][m].m_lo = static_cast<uint8_t>(best_lo);
						//table[g + inten * 32].m_solutions[sr][m].m_hi = static_cast<uint8_t>(best_hi);
						//table[g + inten * 32].m_solutions[sr][m].m_err = static_cast<uint16_t>(best_err);

						//assert(best_lo != best_hi);
						fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, (uint32_t)best_err);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");

					} // m
				} // sr
			} // g
		} // inten

		fclose(pFile);
	}
#endif

#if BASISD_SUPPORT_ETC2_EAC_A8 || BASISD_SUPPORT_ETC2_EAC_RG11
	static const int8_t g_eac_modifier_table[16][8] =
	{
		{ -3, -6, -9, -15, 2, 5, 8, 14 },
		{ -3, -7, -10, -13, 2, 6, 9, 12 },
		{ -2, -5, -8, -13, 1, 4, 7, 12 },
		{ -2, -4, -6, -13, 1, 3, 5, 12 },
		{ -3, -6, -8, -12, 2, 5, 7, 11 },
		{ -3, -7, -9, -11, 2, 6, 8, 10 },
		{ -4, -7, -8, -11, 3, 6, 7, 10 },
		{ -3, -5, -8, -11, 2, 4, 7, 10 },

		{ -2, -6, -8, -10, 1, 5, 7, 9 },
		{ -2, -5, -8, -10, 1, 4, 7, 9 },
		{ -2, -4, -8, -10, 1, 3, 7, 9 },
		{ -2, -5, -7, -10, 1, 4, 6, 9 },
		{ -3, -4, -7, -10, 2, 3, 6, 9 },
		{ -1, -2, -3, -10, 0, 1, 2, 9 }, // entry 13
		{ -4, -6, -8, -9, 3, 5, 7, 8 },
		{ -3, -5, -7, -9, 2, 4, 6, 8 }
	};

	// Used by ETC2 EAC A8 and ETC2 EAC R11/RG11.
	struct eac_block
	{
		uint16_t m_base : 8;

		uint16_t m_table : 4;
		uint16_t m_multiplier : 4;

		uint8_t m_selectors[6];

		uint32_t get_selector(uint32_t x, uint32_t y) const
		{
			assert((x < 4) && (y < 4));

			const uint32_t ofs = 45 - (y + x * 4) * 3;

			const uint64_t pixels = get_selector_bits();

			return (pixels >> ofs) & 7;
		}

		void set_selector(uint32_t x, uint32_t y, uint32_t s)
		{
			assert((x < 4) && (y < 4) && (s < 8));

			const uint32_t ofs = 45 - (y + x * 4) * 3;

			uint64_t pixels = get_selector_bits();

			pixels &= ~(7ULL << ofs);
			pixels |= (static_cast<uint64_t>(s) << ofs);

			set_selector_bits(pixels);
		}

		uint64_t get_selector_bits() const
		{
			uint64_t pixels = ((uint64_t)m_selectors[0] << 40) | ((uint64_t)m_selectors[1] << 32) |
				((uint64_t)m_selectors[2] << 24) |
				((uint64_t)m_selectors[3] << 16) | ((uint64_t)m_selectors[4] << 8) | m_selectors[5];
			return pixels;
		}

		void set_selector_bits(uint64_t pixels)
		{
			m_selectors[0] = (uint8_t)(pixels >> 40);
			m_selectors[1] = (uint8_t)(pixels >> 32);
			m_selectors[2] = (uint8_t)(pixels >> 24);
			m_selectors[3] = (uint8_t)(pixels >> 16);
			m_selectors[4] = (uint8_t)(pixels >> 8);
			m_selectors[5] = (uint8_t)(pixels);
		}
	};

	static const dxt_selector_range s_etc2_eac_selector_ranges[] =
	{
		{ 0, 3 },

		{ 1, 3 },
		{ 0, 2 },

		{ 1, 2 },
	};

	const uint32_t NUM_ETC2_EAC_SELECTOR_RANGES = sizeof(s_etc2_eac_selector_ranges) / sizeof(s_etc2_eac_selector_ranges[0]);

	struct etc1_g_to_eac_conversion
	{
		uint8_t m_base;
		uint8_t m_table_mul; // mul*16+table
		uint16_t m_trans; // translates ETC1 selectors to ETC2_EAC_A8
	};
#endif // BASISD_SUPPORT_ETC2_EAC_A8 || BASISD_SUPPORT_ETC2_EAC_RG11

#if BASISD_SUPPORT_ETC2_EAC_A8

#if BASISD_WRITE_NEW_ETC2_EAC_A8_TABLES
	struct pack_eac_a8_results
	{
		uint32_t m_base;
		uint32_t m_table;
		uint32_t m_multiplier;
		std::vector<uint8_t> m_selectors;
		std::vector<uint8_t> m_selectors_temp;
	};

	static uint64_t pack_eac_a8_exhaustive(pack_eac_a8_results& results, const uint8_t* pPixels, uint32_t num_pixels)
	{
		results.m_selectors.resize(num_pixels);
		results.m_selectors_temp.resize(num_pixels);

		uint64_t best_err = UINT64_MAX;

		for (uint32_t base_color = 0; base_color < 256; base_color++)
		{
			for (uint32_t multiplier = 1; multiplier < 16; multiplier++)
			{
				for (uint32_t table = 0; table < 16; table++)
				{
					uint64_t total_err = 0;

					for (uint32_t i = 0; i < num_pixels; i++)
					{
						const int a = pPixels[i];

						uint32_t best_s_err = UINT32_MAX;
						uint32_t best_s = 0;
						for (uint32_t s = 0; s < 8; s++)
						{
							int v = (int)multiplier * g_eac_modifier_table[table][s] + (int)base_color;
							if (v < 0)
								v = 0;
							else if (v > 255)
								v = 255;

							uint32_t err = abs(a - v);
							if (err < best_s_err)
							{
								best_s_err = err;
								best_s = s;
							}
						}

						results.m_selectors_temp[i] = static_cast<uint8_t>(best_s);

						total_err += best_s_err * best_s_err;
						if (total_err >= best_err)
							break;
					}

					if (total_err < best_err)
					{
						best_err = total_err;
						results.m_base = base_color;
						results.m_multiplier = multiplier;
						results.m_table = table;
						results.m_selectors.swap(results.m_selectors_temp);
					}

				} // table

			} // multiplier

		} // base_color

		return best_err;
	}
#endif // BASISD_WRITE_NEW_ETC2_EAC_A8_TABLES
		
	static
#if !BASISD_WRITE_NEW_ETC2_EAC_A8_TABLES		
		const
#endif
		etc1_g_to_eac_conversion s_etc1_g_to_etc2_a8[32 * 8][NUM_ETC2_EAC_SELECTOR_RANGES] =
	{
		{ { 0,1,3328 },{ 0,1,3328 },{ 0,1,256 },{ 0,1,256 } },
		{ { 0,226,3936 },{ 0,226,3936 },{ 0,81,488 },{ 0,81,488 } },
		{ { 6,178,4012 },{ 6,178,4008 },{ 0,146,501 },{ 0,130,496 } },
		{ { 14,178,4012 },{ 14,178,4008 },{ 8,146,501 },{ 6,82,496 } },
		{ { 23,178,4012 },{ 23,178,4008 },{ 17,146,501 },{ 3,228,496 } },
		{ { 31,178,4012 },{ 31,178,4008 },{ 25,146,501 },{ 11,228,496 } },
		{ { 39,178,4012 },{ 39,178,4008 },{ 33,146,501 },{ 19,228,496 } },
		{ { 47,178,4012 },{ 47,178,4008 },{ 41,146,501 },{ 27,228,496 } },
		{ { 56,178,4012 },{ 56,178,4008 },{ 50,146,501 },{ 36,228,496 } },
		{ { 64,178,4012 },{ 64,178,4008 },{ 58,146,501 },{ 44,228,496 } },
		{ { 72,178,4012 },{ 72,178,4008 },{ 66,146,501 },{ 52,228,496 } },
		{ { 80,178,4012 },{ 80,178,4008 },{ 74,146,501 },{ 60,228,496 } },
		{ { 89,178,4012 },{ 89,178,4008 },{ 83,146,501 },{ 69,228,496 } },
		{ { 97,178,4012 },{ 97,178,4008 },{ 91,146,501 },{ 77,228,496 } },
		{ { 105,178,4012 },{ 105,178,4008 },{ 99,146,501 },{ 85,228,496 } },
		{ { 113,178,4012 },{ 113,178,4008 },{ 107,146,501 },{ 93,228,496 } },
		{ { 122,178,4012 },{ 122,178,4008 },{ 116,146,501 },{ 102,228,496 } },
		{ { 130,178,4012 },{ 130,178,4008 },{ 124,146,501 },{ 110,228,496 } },
		{ { 138,178,4012 },{ 138,178,4008 },{ 132,146,501 },{ 118,228,496 } },
		{ { 146,178,4012 },{ 146,178,4008 },{ 140,146,501 },{ 126,228,496 } },
		{ { 155,178,4012 },{ 155,178,4008 },{ 149,146,501 },{ 135,228,496 } },
		{ { 163,178,4012 },{ 163,178,4008 },{ 157,146,501 },{ 143,228,496 } },
		{ { 171,178,4012 },{ 171,178,4008 },{ 165,146,501 },{ 151,228,496 } },
		{ { 179,178,4012 },{ 179,178,4008 },{ 173,146,501 },{ 159,228,496 } },
		{ { 188,178,4012 },{ 188,178,4008 },{ 182,146,501 },{ 168,228,496 } },
		{ { 196,178,4012 },{ 196,178,4008 },{ 190,146,501 },{ 176,228,496 } },
		{ { 204,178,4012 },{ 204,178,4008 },{ 198,146,501 },{ 184,228,496 } },
		{ { 212,178,4012 },{ 212,178,4008 },{ 206,146,501 },{ 192,228,496 } },
		{ { 221,178,4012 },{ 221,178,4008 },{ 215,146,501 },{ 201,228,496 } },
		{ { 229,178,4012 },{ 229,178,4008 },{ 223,146,501 },{ 209,228,496 } },
		{ { 235,66,4012 },{ 221,100,4008 },{ 231,146,501 },{ 217,228,496 } },
		{ { 211,102,4085 },{ 118,31,4080 },{ 211,102,501 },{ 118,31,496 } },
		{ { 1,2,3328 },{ 1,2,3328 },{ 0,1,320 },{ 0,1,320 } },
		{ { 7,162,3905 },{ 7,162,3904 },{ 1,17,480 },{ 1,17,480 } },
		{ { 15,162,3906 },{ 15,162,3904 },{ 1,117,352 },{ 1,117,352 } },
		{ { 23,162,3906 },{ 23,162,3904 },{ 5,34,500 },{ 4,53,424 } },
		{ { 32,162,3906 },{ 32,162,3904 },{ 14,34,500 },{ 3,69,424 } },
		{ { 40,162,3906 },{ 40,162,3904 },{ 22,34,500 },{ 1,133,496 } },
		{ { 48,162,3906 },{ 48,162,3904 },{ 30,34,500 },{ 4,85,496 } },
		{ { 56,162,3906 },{ 56,162,3904 },{ 38,34,500 },{ 12,85,496 } },
		{ { 65,162,3906 },{ 65,162,3904 },{ 47,34,500 },{ 1,106,424 } },
		{ { 73,162,3906 },{ 73,162,3904 },{ 55,34,500 },{ 9,106,424 } },
		{ { 81,162,3906 },{ 81,162,3904 },{ 63,34,500 },{ 7,234,496 } },
		{ { 89,162,3906 },{ 89,162,3904 },{ 71,34,500 },{ 15,234,496 } },
		{ { 98,162,3906 },{ 98,162,3904 },{ 80,34,500 },{ 24,234,496 } },
		{ { 106,162,3906 },{ 106,162,3904 },{ 88,34,500 },{ 32,234,496 } },
		{ { 114,162,3906 },{ 114,162,3904 },{ 96,34,500 },{ 40,234,496 } },
		{ { 122,162,3906 },{ 122,162,3904 },{ 104,34,500 },{ 48,234,496 } },
		{ { 131,162,3906 },{ 131,162,3904 },{ 113,34,500 },{ 57,234,496 } },
		{ { 139,162,3906 },{ 139,162,3904 },{ 121,34,500 },{ 65,234,496 } },
		{ { 147,162,3906 },{ 147,162,3904 },{ 129,34,500 },{ 73,234,496 } },
		{ { 155,162,3906 },{ 155,162,3904 },{ 137,34,500 },{ 81,234,496 } },
		{ { 164,162,3906 },{ 164,162,3904 },{ 146,34,500 },{ 90,234,496 } },
		{ { 172,162,3906 },{ 172,162,3904 },{ 154,34,500 },{ 98,234,496 } },
		{ { 180,162,3906 },{ 180,162,3904 },{ 162,34,500 },{ 106,234,496 } },
		{ { 188,162,3906 },{ 188,162,3904 },{ 170,34,500 },{ 114,234,496 } },
		{ { 197,162,3906 },{ 197,162,3904 },{ 179,34,500 },{ 123,234,496 } },
		{ { 205,162,3906 },{ 205,162,3904 },{ 187,34,500 },{ 131,234,496 } },
		{ { 213,162,3906 },{ 213,162,3904 },{ 195,34,500 },{ 139,234,496 } },
		{ { 221,162,3906 },{ 221,162,3904 },{ 203,34,500 },{ 147,234,496 } },
		{ { 230,162,3906 },{ 230,162,3904 },{ 212,34,500 },{ 156,234,496 } },
		{ { 238,162,3906 },{ 174,106,4008 },{ 220,34,500 },{ 164,234,496 } },
		{ { 240,178,4001 },{ 182,106,4008 },{ 228,34,500 },{ 172,234,496 } },
		{ { 166,108,4085 },{ 115,31,4080 },{ 166,108,501 },{ 115,31,496 } },
		{ { 1,68,3328 },{ 1,68,3328 },{ 0,17,384 },{ 0,17,384 } },
		{ { 1,148,3904 },{ 1,148,3904 },{ 1,2,384 },{ 1,2,384 } },
		{ { 21,18,3851 },{ 21,18,3848 },{ 1,50,488 },{ 1,50,488 } },
		{ { 27,195,3851 },{ 29,18,3848 },{ 0,67,488 },{ 0,67,488 } },
		{ { 34,195,3907 },{ 38,18,3848 },{ 20,66,482 },{ 0,3,496 } },
		{ { 42,195,3907 },{ 46,18,3848 },{ 28,66,482 },{ 2,6,424 } },
		{ { 50,195,3907 },{ 54,18,3848 },{ 36,66,482 },{ 4,22,424 } },
		{ { 58,195,3907 },{ 62,18,3848 },{ 44,66,482 },{ 3,73,424 } },
		{ { 67,195,3907 },{ 71,18,3848 },{ 53,66,482 },{ 3,22,496 } },
		{ { 75,195,3907 },{ 79,18,3848 },{ 61,66,482 },{ 2,137,496 } },
		{ { 83,195,3907 },{ 87,18,3848 },{ 69,66,482 },{ 1,89,496 } },
		{ { 91,195,3907 },{ 95,18,3848 },{ 77,66,482 },{ 9,89,496 } },
		{ { 100,195,3907 },{ 104,18,3848 },{ 86,66,482 },{ 18,89,496 } },
		{ { 108,195,3907 },{ 112,18,3848 },{ 94,66,482 },{ 26,89,496 } },
		{ { 116,195,3907 },{ 120,18,3848 },{ 102,66,482 },{ 34,89,496 } },
		{ { 124,195,3907 },{ 128,18,3848 },{ 110,66,482 },{ 42,89,496 } },
		{ { 133,195,3907 },{ 137,18,3848 },{ 119,66,482 },{ 51,89,496 } },
		{ { 141,195,3907 },{ 145,18,3848 },{ 127,66,482 },{ 59,89,496 } },
		{ { 149,195,3907 },{ 153,18,3848 },{ 135,66,482 },{ 67,89,496 } },
		{ { 157,195,3907 },{ 161,18,3848 },{ 143,66,482 },{ 75,89,496 } },
		{ { 166,195,3907 },{ 170,18,3848 },{ 152,66,482 },{ 84,89,496 } },
		{ { 174,195,3907 },{ 178,18,3848 },{ 160,66,482 },{ 92,89,496 } },
		{ { 182,195,3907 },{ 186,18,3848 },{ 168,66,482 },{ 100,89,496 } },
		{ { 190,195,3907 },{ 194,18,3848 },{ 176,66,482 },{ 108,89,496 } },
		{ { 199,195,3907 },{ 203,18,3848 },{ 185,66,482 },{ 117,89,496 } },
		{ { 207,195,3907 },{ 211,18,3848 },{ 193,66,482 },{ 125,89,496 } },
		{ { 215,195,3907 },{ 219,18,3848 },{ 201,66,482 },{ 133,89,496 } },
		{ { 223,195,3907 },{ 227,18,3848 },{ 209,66,482 },{ 141,89,496 } },
		{ { 231,195,3907 },{ 168,89,4008 },{ 218,66,482 },{ 150,89,496 } },
		{ { 236,18,3907 },{ 176,89,4008 },{ 226,66,482 },{ 158,89,496 } },
		{ { 158,90,4085 },{ 103,31,4080 },{ 158,90,501 },{ 103,31,496 } },
		{ { 166,90,4085 },{ 111,31,4080 },{ 166,90,501 },{ 111,31,496 } },
		{ { 0,70,3328 },{ 0,70,3328 },{ 0,45,256 },{ 0,45,256 } },
		{ { 0,117,3904 },{ 0,117,3904 },{ 0,35,384 },{ 0,35,384 } },
		{ { 13,165,3905 },{ 13,165,3904 },{ 3,221,416 },{ 3,221,416 } },
		{ { 21,165,3906 },{ 21,165,3904 },{ 11,221,416 },{ 11,221,416 } },
		{ { 30,165,3906 },{ 30,165,3904 },{ 7,61,352 },{ 7,61,352 } },
		{ { 38,165,3906 },{ 38,165,3904 },{ 2,125,352 },{ 2,125,352 } },
		{ { 46,165,3906 },{ 46,165,3904 },{ 2,37,500 },{ 10,125,352 } },
		{ { 54,165,3906 },{ 54,165,3904 },{ 10,37,500 },{ 5,61,424 } },
		{ { 63,165,3906 },{ 63,165,3904 },{ 19,37,500 },{ 1,189,424 } },
		{ { 4,254,4012 },{ 71,165,3904 },{ 27,37,500 },{ 9,189,424 } },
		{ { 12,254,4012 },{ 79,165,3904 },{ 35,37,500 },{ 4,77,424 } },
		{ { 20,254,4012 },{ 87,165,3904 },{ 43,37,500 },{ 12,77,424 } },
		{ { 29,254,4012 },{ 96,165,3904 },{ 52,37,500 },{ 8,93,424 } },
		{ { 37,254,4012 },{ 104,165,3904 },{ 60,37,500 },{ 3,141,496 } },
		{ { 45,254,4012 },{ 112,165,3904 },{ 68,37,500 },{ 11,141,496 } },
		{ { 53,254,4012 },{ 120,165,3904 },{ 76,37,500 },{ 6,93,496 } },
		{ { 62,254,4012 },{ 129,165,3904 },{ 85,37,500 },{ 15,93,496 } },
		{ { 70,254,4012 },{ 137,165,3904 },{ 93,37,500 },{ 23,93,496 } },
		{ { 78,254,4012 },{ 145,165,3904 },{ 101,37,500 },{ 31,93,496 } },
		{ { 86,254,4012 },{ 153,165,3904 },{ 109,37,500 },{ 39,93,496 } },
		{ { 95,254,4012 },{ 162,165,3904 },{ 118,37,500 },{ 48,93,496 } },
		{ { 103,254,4012 },{ 170,165,3904 },{ 126,37,500 },{ 56,93,496 } },
		{ { 111,254,4012 },{ 178,165,3904 },{ 134,37,500 },{ 64,93,496 } },
		{ { 119,254,4012 },{ 186,165,3904 },{ 142,37,500 },{ 72,93,496 } },
		{ { 128,254,4012 },{ 195,165,3904 },{ 151,37,500 },{ 81,93,496 } },
		{ { 136,254,4012 },{ 203,165,3904 },{ 159,37,500 },{ 89,93,496 } },
		{ { 212,165,3906 },{ 136,77,4008 },{ 167,37,500 },{ 97,93,496 } },
		{ { 220,165,3394 },{ 131,93,4008 },{ 175,37,500 },{ 105,93,496 } },
		{ { 214,181,4001 },{ 140,93,4008 },{ 184,37,500 },{ 114,93,496 } },
		{ { 222,181,4001 },{ 148,93,4008 },{ 192,37,500 },{ 122,93,496 } },
		{ { 114,95,4085 },{ 99,31,4080 },{ 114,95,501 },{ 99,31,496 } },
		{ { 122,95,4085 },{ 107,31,4080 },{ 122,95,501 },{ 107,31,496 } },
		{ { 0,102,3840 },{ 0,102,3840 },{ 0,18,384 },{ 0,18,384 } },
		{ { 5,167,3904 },{ 5,167,3904 },{ 0,13,256 },{ 0,13,256 } },
		{ { 4,54,3968 },{ 4,54,3968 },{ 1,67,448 },{ 1,67,448 } },
		{ { 30,198,3850 },{ 30,198,3848 },{ 0,3,480 },{ 0,3,480 } },
		{ { 39,198,3850 },{ 39,198,3848 },{ 3,52,488 },{ 3,52,488 } },
		{ { 47,198,3851 },{ 47,198,3848 },{ 3,4,488 },{ 3,4,488 } },
		{ { 55,198,3851 },{ 55,198,3848 },{ 1,70,488 },{ 1,70,488 } },
		{ { 54,167,3906 },{ 63,198,3848 },{ 3,22,488 },{ 3,22,488 } },
		{ { 62,167,3906 },{ 72,198,3848 },{ 24,118,488 },{ 0,6,496 } },
		{ { 70,167,3906 },{ 80,198,3848 },{ 32,118,488 },{ 2,89,488 } },
		{ { 78,167,3906 },{ 88,198,3848 },{ 40,118,488 },{ 1,73,496 } },
		{ { 86,167,3906 },{ 96,198,3848 },{ 48,118,488 },{ 0,28,424 } },
		{ { 95,167,3906 },{ 105,198,3848 },{ 57,118,488 },{ 9,28,424 } },
		{ { 103,167,3906 },{ 113,198,3848 },{ 65,118,488 },{ 5,108,496 } },
		{ { 111,167,3906 },{ 121,198,3848 },{ 73,118,488 },{ 13,108,496 } },
		{ { 119,167,3906 },{ 129,198,3848 },{ 81,118,488 },{ 21,108,496 } },
		{ { 128,167,3906 },{ 138,198,3848 },{ 90,118,488 },{ 6,28,496 } },
		{ { 136,167,3906 },{ 146,198,3848 },{ 98,118,488 },{ 14,28,496 } },
		{ { 144,167,3906 },{ 154,198,3848 },{ 106,118,488 },{ 22,28,496 } },
		{ { 152,167,3906 },{ 162,198,3848 },{ 114,118,488 },{ 30,28,496 } },
		{ { 161,167,3906 },{ 171,198,3848 },{ 123,118,488 },{ 39,28,496 } },
		{ { 169,167,3906 },{ 179,198,3848 },{ 131,118,488 },{ 47,28,496 } },
		{ { 177,167,3906 },{ 187,198,3848 },{ 139,118,488 },{ 55,28,496 } },
		{ { 185,167,3906 },{ 195,198,3848 },{ 147,118,488 },{ 63,28,496 } },
		{ { 194,167,3906 },{ 120,12,4008 },{ 156,118,488 },{ 72,28,496 } },
		{ { 206,198,3907 },{ 116,28,4008 },{ 164,118,488 },{ 80,28,496 } },
		{ { 214,198,3907 },{ 124,28,4008 },{ 172,118,488 },{ 88,28,496 } },
		{ { 222,198,3395 },{ 132,28,4008 },{ 180,118,488 },{ 96,28,496 } },
		{ { 207,134,4001 },{ 141,28,4008 },{ 189,118,488 },{ 105,28,496 } },
		{ { 95,30,4085 },{ 86,31,4080 },{ 95,30,501 },{ 86,31,496 } },
		{ { 103,30,4085 },{ 94,31,4080 },{ 103,30,501 },{ 94,31,496 } },
		{ { 111,30,4085 },{ 102,31,4080 },{ 111,30,501 },{ 102,31,496 } },
		{ { 0,104,3840 },{ 0,104,3840 },{ 0,18,448 },{ 0,18,448 } },
		{ { 4,39,3904 },{ 4,39,3904 },{ 0,4,384 },{ 0,4,384 } },
		{ { 0,56,3968 },{ 0,56,3968 },{ 0,84,448 },{ 0,84,448 } },
		{ { 6,110,3328 },{ 6,110,3328 },{ 0,20,448 },{ 0,20,448 } },
		{ { 41,200,3850 },{ 41,200,3848 },{ 1,4,480 },{ 1,4,480 } },
		{ { 49,200,3850 },{ 49,200,3848 },{ 1,8,416 },{ 1,8,416 } },
		{ { 57,200,3851 },{ 57,200,3848 },{ 1,38,488 },{ 1,38,488 } },
		{ { 65,200,3851 },{ 65,200,3848 },{ 1,120,488 },{ 1,120,488 } },
		{ { 74,200,3851 },{ 74,200,3848 },{ 2,72,488 },{ 2,72,488 } },
		{ { 69,6,3907 },{ 82,200,3848 },{ 2,24,488 },{ 2,24,488 } },
		{ { 77,6,3907 },{ 90,200,3848 },{ 26,120,488 },{ 10,24,488 } },
		{ { 97,63,3330 },{ 98,200,3848 },{ 34,120,488 },{ 2,8,496 } },
		{ { 106,63,3330 },{ 107,200,3848 },{ 43,120,488 },{ 3,92,488 } },
		{ { 114,63,3330 },{ 115,200,3848 },{ 51,120,488 },{ 11,92,488 } },
		{ { 122,63,3330 },{ 123,200,3848 },{ 59,120,488 },{ 7,76,496 } },
		{ { 130,63,3330 },{ 131,200,3848 },{ 67,120,488 },{ 15,76,496 } },
		{ { 139,63,3330 },{ 140,200,3848 },{ 76,120,488 },{ 24,76,496 } },
		{ { 147,63,3330 },{ 148,200,3848 },{ 84,120,488 },{ 32,76,496 } },
		{ { 155,63,3330 },{ 156,200,3848 },{ 92,120,488 },{ 40,76,496 } },
		{ { 163,63,3330 },{ 164,200,3848 },{ 100,120,488 },{ 48,76,496 } },
		{ { 172,63,3330 },{ 173,200,3848 },{ 109,120,488 },{ 57,76,496 } },
		{ { 184,6,3851 },{ 181,200,3848 },{ 117,120,488 },{ 65,76,496 } },
		{ { 192,6,3851 },{ 133,28,3936 },{ 125,120,488 },{ 73,76,496 } },
		{ { 189,200,3907 },{ 141,28,3936 },{ 133,120,488 },{ 81,76,496 } },
		{ { 198,200,3907 },{ 138,108,4000 },{ 142,120,488 },{ 90,76,496 } },
		{ { 206,200,3907 },{ 146,108,4000 },{ 150,120,488 },{ 98,76,496 } },
		{ { 214,200,3395 },{ 154,108,4000 },{ 158,120,488 },{ 106,76,496 } },
		{ { 190,136,4001 },{ 162,108,4000 },{ 166,120,488 },{ 114,76,496 } },
		{ { 123,30,4076 },{ 87,15,4080 },{ 123,30,492 },{ 87,15,496 } },
		{ { 117,110,4084 },{ 80,31,4080 },{ 117,110,500 },{ 80,31,496 } },
		{ { 125,110,4084 },{ 88,31,4080 },{ 125,110,500 },{ 88,31,496 } },
		{ { 133,110,4084 },{ 96,31,4080 },{ 133,110,500 },{ 96,31,496 } },
		{ { 9,56,3904 },{ 9,56,3904 },{ 0,67,448 },{ 0,67,448 } },
		{ { 1,8,3904 },{ 1,8,3904 },{ 1,84,448 },{ 1,84,448 } },
		{ { 1,124,3904 },{ 1,124,3904 },{ 0,39,384 },{ 0,39,384 } },
		{ { 9,124,3904 },{ 9,124,3904 },{ 1,4,448 },{ 1,4,448 } },
		{ { 6,76,3904 },{ 6,76,3904 },{ 0,70,448 },{ 0,70,448 } },
		{ { 62,6,3859 },{ 62,6,3856 },{ 2,38,480 },{ 2,38,480 } },
		{ { 70,6,3859 },{ 70,6,3856 },{ 5,43,416 },{ 5,43,416 } },
		{ { 78,6,3859 },{ 78,6,3856 },{ 2,11,416 },{ 2,11,416 } },
		{ { 87,6,3859 },{ 87,6,3856 },{ 0,171,488 },{ 0,171,488 } },
		{ { 67,8,3906 },{ 95,6,3856 },{ 8,171,488 },{ 8,171,488 } },
		{ { 75,8,3907 },{ 103,6,3856 },{ 5,123,488 },{ 5,123,488 } },
		{ { 83,8,3907 },{ 111,6,3856 },{ 2,75,488 },{ 2,75,488 } },
		{ { 92,8,3907 },{ 120,6,3856 },{ 0,27,488 },{ 0,27,488 } },
		{ { 100,8,3907 },{ 128,6,3856 },{ 8,27,488 },{ 8,27,488 } },
		{ { 120,106,3843 },{ 136,6,3856 },{ 100,6,387 },{ 16,27,488 } },
		{ { 128,106,3843 },{ 144,6,3856 },{ 108,6,387 },{ 2,11,496 } },
		{ { 137,106,3843 },{ 153,6,3856 },{ 117,6,387 },{ 11,11,496 } },
		{ { 145,106,3843 },{ 161,6,3856 },{ 125,6,387 },{ 19,11,496 } },
		{ { 163,8,3851 },{ 137,43,3904 },{ 133,6,387 },{ 27,11,496 } },
		{ { 171,8,3851 },{ 101,11,4000 },{ 141,6,387 },{ 35,11,496 } },
		{ { 180,8,3851 },{ 110,11,4000 },{ 150,6,387 },{ 44,11,496 } },
		{ { 188,8,3851 },{ 118,11,4000 },{ 158,6,387 },{ 52,11,496 } },
		{ { 172,72,3907 },{ 126,11,4000 },{ 166,6,387 },{ 60,11,496 } },
		{ { 174,6,3971 },{ 134,11,4000 },{ 174,6,387 },{ 68,11,496 } },
		{ { 183,6,3971 },{ 143,11,4000 },{ 183,6,387 },{ 77,11,496 } },
		{ { 191,6,3971 },{ 151,11,4000 },{ 191,6,387 },{ 85,11,496 } },
		{ { 199,6,3971 },{ 159,11,4000 },{ 199,6,387 },{ 93,11,496 } },
		{ { 92,12,4084 },{ 69,15,4080 },{ 92,12,500 },{ 69,15,496 } },
		{ { 101,12,4084 },{ 78,15,4080 },{ 101,12,500 },{ 78,15,496 } },
		{ { 109,12,4084 },{ 86,15,4080 },{ 109,12,500 },{ 86,15,496 } },
		{ { 117,12,4084 },{ 79,31,4080 },{ 117,12,500 },{ 79,31,496 } },
		{ { 125,12,4084 },{ 87,31,4080 },{ 125,12,500 },{ 87,31,496 } },
		{ { 71,8,3602 },{ 71,8,3600 },{ 2,21,384 },{ 2,21,384 } },
		{ { 79,8,3611 },{ 79,8,3608 },{ 0,69,448 },{ 0,69,448 } },
		{ { 87,8,3611 },{ 87,8,3608 },{ 0,23,384 },{ 0,23,384 } },
		{ { 95,8,3611 },{ 95,8,3608 },{ 1,5,448 },{ 1,5,448 } },
		{ { 104,8,3611 },{ 104,8,3608 },{ 0,88,448 },{ 0,88,448 } },
		{ { 112,8,3611 },{ 112,8,3608 },{ 0,72,448 },{ 0,72,448 } },
		{ { 120,8,3611 },{ 121,8,3608 },{ 36,21,458 },{ 36,21,456 } },
		{ { 133,47,3091 },{ 129,8,3608 },{ 44,21,458 },{ 44,21,456 } },
		{ { 142,47,3091 },{ 138,8,3608 },{ 53,21,459 },{ 53,21,456 } },
		{ { 98,12,3850 },{ 98,12,3848 },{ 61,21,459 },{ 61,21,456 } },
		{ { 106,12,3850 },{ 106,12,3848 },{ 10,92,480 },{ 69,21,456 } },
		{ { 114,12,3851 },{ 114,12,3848 },{ 18,92,480 },{ 77,21,456 } },
		{ { 87,12,3906 },{ 87,12,3904 },{ 3,44,488 },{ 86,21,456 } },
		{ { 95,12,3906 },{ 95,12,3904 },{ 11,44,488 },{ 94,21,456 } },
		{ { 103,12,3906 },{ 103,12,3904 },{ 19,44,488 },{ 102,21,456 } },
		{ { 111,12,3907 },{ 111,12,3904 },{ 27,44,489 },{ 110,21,456 } },
		{ { 120,12,3907 },{ 120,12,3904 },{ 36,44,489 },{ 119,21,456 } },
		{ { 128,12,3907 },{ 128,12,3904 },{ 44,44,489 },{ 127,21,456 } },
		{ { 136,12,3907 },{ 136,12,3904 },{ 52,44,489 },{ 135,21,456 } },
		{ { 144,12,3907 },{ 144,12,3904 },{ 60,44,489 },{ 143,21,456 } },
		{ { 153,12,3907 },{ 153,12,3904 },{ 69,44,490 },{ 152,21,456 } },
		{ { 161,12,3395 },{ 149,188,3968 },{ 77,44,490 },{ 160,21,456 } },
		{ { 169,12,3395 },{ 198,21,3928 },{ 85,44,490 },{ 168,21,456 } },
		{ { 113,95,4001 },{ 201,69,3992 },{ 125,8,483 },{ 176,21,456 } },
		{ { 122,95,4001 },{ 200,21,3984 },{ 134,8,483 },{ 185,21,456 } },
		{ { 142,8,4067 },{ 208,21,3984 },{ 142,8,483 },{ 193,21,456 } },
		{ { 151,8,4067 },{ 47,15,4080 },{ 151,8,483 },{ 47,15,496 } },
		{ { 159,8,4067 },{ 55,15,4080 },{ 159,8,483 },{ 55,15,496 } },
		{ { 168,8,4067 },{ 64,15,4080 },{ 168,8,483 },{ 64,15,496 } },
		{ { 160,40,4075 },{ 72,15,4080 },{ 160,40,491 },{ 72,15,496 } },
		{ { 168,40,4075 },{ 80,15,4080 },{ 168,40,491 },{ 80,15,496 } },
		{ { 144,8,4082 },{ 88,15,4080 },{ 144,8,498 },{ 88,15,496 } }
	};
#endif // BASISD_SUPPORT_ETC2_EAC_A8

#if BASISD_WRITE_NEW_ETC2_EAC_A8_TABLES
	static void create_etc2_eac_a8_conversion_table()
	{
		FILE* pFile = fopen("basisu_decoder_tables_etc2_eac_a8.inc", "w");

		for (uint32_t inten = 0; inten < 8; inten++)
		{
			for (uint32_t base = 0; base < 32; base++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(base, base, base, 255), false), inten);

				fprintf(pFile, "{");

				for (uint32_t sel_range = 0; sel_range < NUM_ETC2_EAC_SELECTOR_RANGES; sel_range++)
				{
					const uint32_t low_selector = s_etc2_eac_selector_ranges[sel_range].m_low;
					const uint32_t high_selector = s_etc2_eac_selector_ranges[sel_range].m_high;

					// We have a ETC1 base color and intensity, and a used selector range from low_selector-high_selector.
					// Now find the best ETC2 EAC A8 base/table/multiplier that fits these colors.

					uint8_t pixels[4];
					uint32_t num_pixels = 0;
					for (uint32_t s = low_selector; s <= high_selector; s++)
						pixels[num_pixels++] = block_colors[s].g;

					pack_eac_a8_results pack_results;
					pack_eac_a8_exhaustive(pack_results, pixels, num_pixels);

					etc1_g_to_eac_conversion& c = s_etc1_g_to_etc2_a8[base + inten * 32][sel_range];

					c.m_base = pack_results.m_base;
					c.m_table_mul = pack_results.m_table * 16 + pack_results.m_multiplier;
					c.m_trans = 0;

					for (uint32_t s = 0; s < 4; s++)
					{
						if ((s < low_selector) || (s > high_selector))
							continue;

						uint32_t etc2_selector = pack_results.m_selectors[s - low_selector];

						c.m_trans |= (etc2_selector << (s * 3));
					}

					fprintf(pFile, "{%u,%u,%u}", c.m_base, c.m_table_mul, c.m_trans);
					if (sel_range < (NUM_ETC2_EAC_SELECTOR_RANGES - 1))
						fprintf(pFile, ",");
				}

				fprintf(pFile, "},\n");
			}
		}

		fclose(pFile);
	}
#endif

#if BASISD_WRITE_NEW_ETC2_EAC_R11_TABLES
	struct pack_eac_r11_results
	{
		uint32_t m_base;
		uint32_t m_table;
		uint32_t m_multiplier;
		std::vector<uint8_t> m_selectors;
		std::vector<uint8_t> m_selectors_temp;
	};

	static uint64_t pack_eac_r11_exhaustive(pack_eac_r11_results& results, const uint8_t* pPixels, uint32_t num_pixels)
	{
		results.m_selectors.resize(num_pixels);
		results.m_selectors_temp.resize(num_pixels);

		uint64_t best_err = UINT64_MAX;

		for (uint32_t base_color = 0; base_color < 256; base_color++)
		{
			for (uint32_t multiplier = 0; multiplier < 16; multiplier++)
			{
				for (uint32_t table = 0; table < 16; table++)
				{
					uint64_t total_err = 0;

					for (uint32_t i = 0; i < num_pixels; i++)
					{
						// Convert 8-bit input to 11-bits
						const int a = (pPixels[i] * 2047 + 128) / 255;

						uint32_t best_s_err = UINT32_MAX;
						uint32_t best_s = 0;
						for (uint32_t s = 0; s < 8; s++)
						{
							int v = (int)(multiplier ? (multiplier * 8) : 1) * g_eac_modifier_table[table][s] + (int)base_color * 8 + 4;
							if (v < 0)
								v = 0;
							else if (v > 2047)
								v = 2047;

							uint32_t err = abs(a - v);
							if (err < best_s_err)
							{
								best_s_err = err;
								best_s = s;
							}
						}

						results.m_selectors_temp[i] = static_cast<uint8_t>(best_s);

						total_err += best_s_err * best_s_err;
						if (total_err >= best_err)
							break;
					}

					if (total_err < best_err)
					{
						best_err = total_err;
						results.m_base = base_color;
						results.m_multiplier = multiplier;
						results.m_table = table;
						results.m_selectors.swap(results.m_selectors_temp);
					}

				} // table

			} // multiplier

		} // base_color

		return best_err;
	}

	static void create_etc2_eac_r11_conversion_table()
	{
		FILE* pFile = nullptr;
		fopen_s(&pFile, "basisu_decoder_tables_etc2_eac_r11.inc", "w");

		for (uint32_t inten = 0; inten < 8; inten++)
		{
			for (uint32_t base = 0; base < 32; base++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(base, base, base, 255), false), inten);

				fprintf(pFile, "{");

				for (uint32_t sel_range = 0; sel_range < NUM_ETC2_EAC_SELECTOR_RANGES; sel_range++)
				{
					const uint32_t low_selector = s_etc2_eac_selector_ranges[sel_range].m_low;
					const uint32_t high_selector = s_etc2_eac_selector_ranges[sel_range].m_high;

					// We have a ETC1 base color and intensity, and a used selector range from low_selector-high_selector.
					// Now find the best ETC2 EAC R11 base/table/multiplier that fits these colors.

					uint8_t pixels[4];
					uint32_t num_pixels = 0;
					for (uint32_t s = low_selector; s <= high_selector; s++)
						pixels[num_pixels++] = block_colors[s].g;

					pack_eac_r11_results pack_results;
					pack_eac_r11_exhaustive(pack_results, pixels, num_pixels);

					etc1_g_to_eac_conversion c;

					c.m_base = (uint8_t)pack_results.m_base;
					c.m_table_mul = (uint8_t)(pack_results.m_table * 16 + pack_results.m_multiplier);
					c.m_trans = 0;

					for (uint32_t s = 0; s < 4; s++)
					{
						if ((s < low_selector) || (s > high_selector))
							continue;

						uint32_t etc2_selector = pack_results.m_selectors[s - low_selector];

						c.m_trans |= (etc2_selector << (s * 3));
					}

					fprintf(pFile, "{%u,%u,%u}", c.m_base, c.m_table_mul, c.m_trans);
					if (sel_range < (NUM_ETC2_EAC_SELECTOR_RANGES - 1))
						fprintf(pFile, ",");
				}

				fprintf(pFile, "},\n");
			}
		}

		fclose(pFile);
	}
#endif // BASISD_WRITE_NEW_ETC2_EAC_R11_TABLES

#if BASISD_WRITE_NEW_ASTC_TABLES
	static void create_etc1_to_astc_conversion_table_0_47();
	static void create_etc1_to_astc_conversion_table_0_255();
#endif

#if BASISD_SUPPORT_ASTC
	static void transcoder_init_astc();
#endif

#if BASISD_WRITE_NEW_BC7_MODE5_TABLES
	static void create_etc1_to_bc7_m5_color_conversion_table();
	static void create_etc1_to_bc7_m5_alpha_conversion_table();
#endif

#if BASISD_SUPPORT_BC7_MODE5
	static void transcoder_init_bc7_mode5();
#endif

#if BASISD_WRITE_NEW_ATC_TABLES
	static void create_etc1s_to_atc_conversion_tables();
#endif

#if BASISD_SUPPORT_ATC
	static void transcoder_init_atc();
#endif

#if BASISD_SUPPORT_PVRTC2
	static void transcoder_init_pvrtc2();
#endif
		
	// Library global initialization. Requires ~9 milliseconds when compiled and executed natively on a Core i7 2.2 GHz.
	// If this is too slow, these computed tables can easilky be moved to be compiled in.
	void basisu_transcoder_init()
	{
		static bool s_initialized;
		if (s_initialized)
			return;

#if BASISD_SUPPORT_ASTC
		transcoder_init_astc();
#endif
				
#if BASISD_WRITE_NEW_ASTC_TABLES
		create_etc1_to_astc_conversion_table_0_47();
		create_etc1_to_astc_conversion_table_0_255();
		exit(0);
#endif

#if BASISD_WRITE_NEW_BC7_TABLES
		create_etc1_to_bc7_m6_conversion_table();
		exit(0);
#endif

#if BASISD_WRITE_NEW_BC7_MODE5_TABLES
		create_etc1_to_bc7_m5_color_conversion_table();
		create_etc1_to_bc7_m5_alpha_conversion_table();
		exit(0);
#endif

#if BASISD_WRITE_NEW_DXT1_TABLES
		create_etc1_to_dxt1_5_conversion_table();
		create_etc1_to_dxt1_6_conversion_table();
		exit(0);
#endif

#if BASISD_WRITE_NEW_ETC2_EAC_A8_TABLES
		create_etc2_eac_a8_conversion_table();
		exit(0);
#endif

#if BASISD_WRITE_NEW_ATC_TABLES
		create_etc1s_to_atc_conversion_tables();
		exit(0);
#endif

#if BASISD_WRITE_NEW_ETC2_EAC_R11_TABLES
		create_etc2_eac_r11_conversion_table();
		exit(0);
#endif

#if BASISD_SUPPORT_DXT1
		uint8_t bc1_expand5[32];
		for (int i = 0; i < 32; i++)
			bc1_expand5[i] = static_cast<uint8_t>((i << 3) | (i >> 2));
		prepare_bc1_single_color_table(g_bc1_match5_equals_1, bc1_expand5, 32, 32, 1);
		prepare_bc1_single_color_table(g_bc1_match5_equals_0, bc1_expand5, 1, 32, 0);

		uint8_t bc1_expand6[64];
		for (int i = 0; i < 64; i++)
			bc1_expand6[i] = static_cast<uint8_t>((i << 2) | (i >> 4));
		prepare_bc1_single_color_table(g_bc1_match6_equals_1, bc1_expand6, 64, 64, 1);
		prepare_bc1_single_color_table(g_bc1_match6_equals_0, bc1_expand6, 1, 64, 0);

		for (uint32_t i = 0; i < NUM_ETC1_TO_DXT1_SELECTOR_RANGES; i++)
		{
			uint32_t l = g_etc1_to_dxt1_selector_ranges[i].m_low;
			uint32_t h = g_etc1_to_dxt1_selector_ranges[i].m_high;
			g_etc1_to_dxt1_selector_range_index[l][h] = i;
		}

		for (uint32_t sm = 0; sm < NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS; sm++)
		{
			uint8_t etc1_to_dxt1_selector_mappings_raw_dxt1[4];
			uint8_t etc1_to_dxt1_selector_mappings_raw_dxt1_inv[4];

			for (uint32_t j = 0; j < 4; j++)
			{
				static const uint8_t s_linear_dxt1_to_dxt1[4] = { 0, 2, 3, 1 };
				static const uint8_t s_dxt1_inverted_xlat[4] = { 1, 0, 3, 2 };

				etc1_to_dxt1_selector_mappings_raw_dxt1[j] = (uint8_t)s_linear_dxt1_to_dxt1[g_etc1_to_dxt1_selector_mappings[sm][j]];
				etc1_to_dxt1_selector_mappings_raw_dxt1_inv[j] = (uint8_t)s_dxt1_inverted_xlat[etc1_to_dxt1_selector_mappings_raw_dxt1[j]];
			}

			for (uint32_t i = 0; i < 256; i++)
			{
				uint32_t k = 0, k_inv = 0;
				for (uint32_t s = 0; s < 4; s++)
				{
					k |= (etc1_to_dxt1_selector_mappings_raw_dxt1[(i >> (s * 2)) & 3] << (s * 2));
					k_inv |= (etc1_to_dxt1_selector_mappings_raw_dxt1_inv[(i >> (s * 2)) & 3] << (s * 2));
				}
				g_etc1_to_dxt1_selector_mappings_raw_dxt1_256[sm][i] = (uint8_t)k;
				g_etc1_to_dxt1_selector_mappings_raw_dxt1_inv_256[sm][i] = (uint8_t)k_inv;
			}
		}
#endif

#if BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY
		for (uint32_t i = 0; i < NUM_ETC1_TO_BC7_M6_SELECTOR_RANGES; i++)
		{
			uint32_t l = g_etc1_to_bc7_selector_ranges[i].m_low;
			uint32_t h = g_etc1_to_bc7_selector_ranges[i].m_high;
			g_etc1_to_bc7_m6_selector_range_index[l][h] = i;
		}

		for (uint32_t sm = 0; sm < NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS; sm++)
			for (uint32_t j = 0; j < 4; j++)
				g_etc1_to_bc7_selector_mappings_inv[sm][j] = 15 - g_etc1_to_bc7_selector_mappings[sm][j];
#endif

#if BASISD_SUPPORT_BC7_MODE5
		transcoder_init_bc7_mode5();
#endif

#if BASISD_SUPPORT_ATC
		transcoder_init_atc();
#endif

#if BASISD_SUPPORT_PVRTC2
		transcoder_init_pvrtc2();
#endif

		s_initialized = true;
	}

#if BASISD_SUPPORT_DXT1
	static void convert_etc1s_to_dxt1(dxt1_block* pDst_block, const endpoint *pEndpoints, const selector* pSelector, bool use_threecolor_blocks)
	{
#if !BASISD_WRITE_NEW_DXT1_TABLES
		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		const color32& base_color = pEndpoints->m_color5;
		const uint32_t inten_table = pEndpoints->m_inten5;

		if (low_selector == high_selector)
		{
			uint32_t r, g, b;
			decoder_etc_block::get_block_color5(base_color, inten_table, low_selector, r, g, b);

			uint32_t mask = 0xAA;
			uint32_t max16 = (g_bc1_match5_equals_1[r].m_hi << 11) | (g_bc1_match6_equals_1[g].m_hi << 5) | g_bc1_match5_equals_1[b].m_hi;
			uint32_t min16 = (g_bc1_match5_equals_1[r].m_lo << 11) | (g_bc1_match6_equals_1[g].m_lo << 5) | g_bc1_match5_equals_1[b].m_lo;

			if ((!use_threecolor_blocks) && (min16 == max16))
			{
				// This is an annoying edge case that impacts BC3.
				// This is to guarantee that BC3 blocks never use punchthrough alpha (3 color) mode, which isn't supported on some (all?) GPU's.
				mask = 0;

				// Make l > h
				if (min16 > 0)
					min16--;
				else
				{
					// l = h = 0
					assert(min16 == max16 && max16 == 0);

					max16 = 1;
					min16 = 0;
					mask = 0x55;
				}

				assert(max16 > min16);
			}

			if (max16 < min16)
			{
				std::swap(max16, min16);
				mask ^= 0x55;
			}

			pDst_block->set_low_color(static_cast<uint16_t>(max16));
			pDst_block->set_high_color(static_cast<uint16_t>(min16));
			pDst_block->m_selectors[0] = static_cast<uint8_t>(mask);
			pDst_block->m_selectors[1] = static_cast<uint8_t>(mask);
			pDst_block->m_selectors[2] = static_cast<uint8_t>(mask);
			pDst_block->m_selectors[3] = static_cast<uint8_t>(mask);

			return;
		}
		else if ((inten_table >= 7) && (pSelector->m_num_unique_selectors == 2) && (pSelector->m_lo_selector == 0) && (pSelector->m_hi_selector == 3))
		{
			color32 block_colors[4];

			decoder_etc_block::get_block_colors5(block_colors, base_color, inten_table);

			const uint32_t r0 = block_colors[0].r;
			const uint32_t g0 = block_colors[0].g;
			const uint32_t b0 = block_colors[0].b;

			const uint32_t r1 = block_colors[3].r;
			const uint32_t g1 = block_colors[3].g;
			const uint32_t b1 = block_colors[3].b;

			uint32_t max16 = (g_bc1_match5_equals_0[r0].m_hi << 11) | (g_bc1_match6_equals_0[g0].m_hi << 5) | g_bc1_match5_equals_0[b0].m_hi;
			uint32_t min16 = (g_bc1_match5_equals_0[r1].m_hi << 11) | (g_bc1_match6_equals_0[g1].m_hi << 5) | g_bc1_match5_equals_0[b1].m_hi;

			uint32_t l = 0, h = 1;

			if (min16 == max16)
			{
				// Make l > h
				if (min16 > 0)
				{
					min16--;

					l = 0;
					h = 0;
				}
				else
				{
					// l = h = 0
					assert(min16 == max16 && max16 == 0);

					max16 = 1;
					min16 = 0;

					l = 1;
					h = 1;
				}

				assert(max16 > min16);
			}

			if (max16 < min16)
			{
				std::swap(max16, min16);
				l = 1;
				h = 0;
			}

			pDst_block->set_low_color((uint16_t)max16);
			pDst_block->set_high_color((uint16_t)min16);

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t s = pSelector->get_selector(x, y);
					pDst_block->set_selector(x, y, (s == 3) ? h : l);
				}
			}

			return;
		}

		const uint32_t selector_range_table = g_etc1_to_dxt1_selector_range_index[low_selector][high_selector];

		//[32][8][RANGES][MAPPING]
		const etc1_to_dxt1_56_solution* pTable_r = &g_etc1_to_dxt_5[(inten_table * 32 + base_color.r) * (NUM_ETC1_TO_DXT1_SELECTOR_RANGES * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS];
		const etc1_to_dxt1_56_solution* pTable_g = &g_etc1_to_dxt_6[(inten_table * 32 + base_color.g) * (NUM_ETC1_TO_DXT1_SELECTOR_RANGES * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS];
		const etc1_to_dxt1_56_solution* pTable_b = &g_etc1_to_dxt_5[(inten_table * 32 + base_color.b) * (NUM_ETC1_TO_DXT1_SELECTOR_RANGES * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS];

		uint32_t best_err = UINT_MAX;
		uint32_t best_mapping = 0;

		assert(NUM_ETC1_TO_DXT1_SELECTOR_MAPPINGS == 10);
#define DO_ITER(m) { uint32_t total_err = pTable_r[m].m_err + pTable_g[m].m_err + pTable_b[m].m_err; if (total_err < best_err) { best_err = total_err; best_mapping = m; } }
		DO_ITER(0); DO_ITER(1); DO_ITER(2); DO_ITER(3); DO_ITER(4);
		DO_ITER(5); DO_ITER(6); DO_ITER(7); DO_ITER(8); DO_ITER(9);
#undef DO_ITER

		uint32_t l = dxt1_block::pack_unscaled_color(pTable_r[best_mapping].m_lo, pTable_g[best_mapping].m_lo, pTable_b[best_mapping].m_lo);
		uint32_t h = dxt1_block::pack_unscaled_color(pTable_r[best_mapping].m_hi, pTable_g[best_mapping].m_hi, pTable_b[best_mapping].m_hi);

		const uint8_t* pSelectors_xlat_256 = &g_etc1_to_dxt1_selector_mappings_raw_dxt1_256[best_mapping][0];

		if (l < h)
		{
			std::swap(l, h);
			pSelectors_xlat_256 = &g_etc1_to_dxt1_selector_mappings_raw_dxt1_inv_256[best_mapping][0];
		}
				
		pDst_block->set_low_color(static_cast<uint16_t>(l));
		pDst_block->set_high_color(static_cast<uint16_t>(h));

		if (l == h)
		{
			uint8_t mask = 0;

			if (!use_threecolor_blocks)
			{
				// This is an annoying edge case that impacts BC3.

				// Make l > h
				if (h > 0)
					h--;
				else
				{
					// l = h = 0
					assert(l == h && h == 0);

					h = 0;
					l = 1;
					mask = 0x55;
				}

				assert(l > h);
				pDst_block->set_low_color(static_cast<uint16_t>(l));
				pDst_block->set_high_color(static_cast<uint16_t>(h));
			}

			pDst_block->m_selectors[0] = mask;
			pDst_block->m_selectors[1] = mask;
			pDst_block->m_selectors[2] = mask;
			pDst_block->m_selectors[3] = mask;

			return;
		}

		pDst_block->m_selectors[0] = pSelectors_xlat_256[pSelector->m_selectors[0]];
		pDst_block->m_selectors[1] = pSelectors_xlat_256[pSelector->m_selectors[1]];
		pDst_block->m_selectors[2] = pSelectors_xlat_256[pSelector->m_selectors[2]];
		pDst_block->m_selectors[3] = pSelectors_xlat_256[pSelector->m_selectors[3]];
#endif
	}

#if BASISD_ENABLE_DEBUG_FLAGS
	static void convert_etc1s_to_dxt1_vis(dxt1_block* pDst_block, const endpoint* pEndpoints, const selector* pSelector, bool use_threecolor_blocks)
	{
		convert_etc1s_to_dxt1(pDst_block, pEndpoints, pSelector, use_threecolor_blocks);

		if (g_debug_flags & cDebugFlagVisBC1Sels)
		{
			uint32_t l = dxt1_block::pack_unscaled_color(31, 63, 31);
			uint32_t h = dxt1_block::pack_unscaled_color(0, 0, 0);
			pDst_block->set_low_color(static_cast<uint16_t>(l));
			pDst_block->set_high_color(static_cast<uint16_t>(h));
		}
		else if (g_debug_flags & cDebugFlagVisBC1Endpoints)
		{
			for (uint32_t y = 0; y < 4; y++)
				for (uint32_t x = 0; x < 4; x++)
					pDst_block->set_selector(x, y, (y < 2) ? 0 : 1);
		}
	}
#endif
#endif

#if BASISD_SUPPORT_FXT1
	struct fxt1_block
	{
		union
		{
			struct
			{
				uint64_t m_t00 : 2;
				uint64_t m_t01 : 2;
				uint64_t m_t02 : 2;
				uint64_t m_t03 : 2;
				uint64_t m_t04 : 2;
				uint64_t m_t05 : 2;
				uint64_t m_t06 : 2;
				uint64_t m_t07 : 2;
				uint64_t m_t08 : 2;
				uint64_t m_t09 : 2;
				uint64_t m_t10 : 2;
				uint64_t m_t11 : 2;
				uint64_t m_t12 : 2;
				uint64_t m_t13 : 2;
				uint64_t m_t14 : 2;
				uint64_t m_t15 : 2;
				uint64_t m_t16 : 2;
				uint64_t m_t17 : 2;
				uint64_t m_t18 : 2;
				uint64_t m_t19 : 2;
				uint64_t m_t20 : 2;
				uint64_t m_t21 : 2;
				uint64_t m_t22 : 2;
				uint64_t m_t23 : 2;
				uint64_t m_t24 : 2;
				uint64_t m_t25 : 2;
				uint64_t m_t26 : 2;
				uint64_t m_t27 : 2;
				uint64_t m_t28 : 2;
				uint64_t m_t29 : 2;
				uint64_t m_t30 : 2;
				uint64_t m_t31 : 2;
			} m_lo;
			uint64_t m_lo_bits;
			uint8_t m_sels[8];
		};
		union
		{
			struct
			{
#ifdef BASISU_USE_ORIGINAL_3DFX_FXT1_ENCODING
				uint64_t m_b1 : 5;
				uint64_t m_g1 : 5;
				uint64_t m_r1 : 5;
				uint64_t m_b0 : 5;
				uint64_t m_g0 : 5;
				uint64_t m_r0 : 5;
				uint64_t m_b3 : 5;
				uint64_t m_g3 : 5;
				uint64_t m_r3 : 5;
				uint64_t m_b2 : 5;
				uint64_t m_g2 : 5;
				uint64_t m_r2 : 5;
#else
				uint64_t m_b0 : 5;
				uint64_t m_g0 : 5;
				uint64_t m_r0 : 5;
				uint64_t m_b1 : 5;
				uint64_t m_g1 : 5;
				uint64_t m_r1 : 5;
				uint64_t m_b2 : 5;
				uint64_t m_g2 : 5;
				uint64_t m_r2 : 5;
				uint64_t m_b3 : 5;
				uint64_t m_g3 : 5;
				uint64_t m_r3 : 5;
#endif
				uint64_t m_alpha : 1;
				uint64_t m_glsb : 2;
				uint64_t m_mode : 1;
			} m_hi;
			uint64_t m_hi_bits;
		};
	};

	static uint8_t conv_dxt1_to_fxt1_sels(uint32_t sels)
	{
		static uint8_t s_conv_table[16] = { 0, 3, 1, 2, 12, 15, 13, 14, 4, 7, 5, 6, 8, 11, 9, 10 };
		return s_conv_table[sels & 15] | (s_conv_table[sels >> 4] << 4);
	}

	static void convert_etc1s_to_fxt1(void *pDst, const endpoint *pEndpoints, const selector *pSelectors, uint32_t fxt1_subblock)
	{
		fxt1_block* pBlock = static_cast<fxt1_block*>(pDst);

		// CC_MIXED is basically DXT1 with different encoding tricks.
		// So transcode ETC1S to DXT1, then transcode that to FXT1 which is easy and nearly lossless. 
		// (It's not completely lossless because FXT1 rounds in its color lerps while DXT1 doesn't, but it should be good enough.)
		dxt1_block blk;
		convert_etc1s_to_dxt1(&blk, pEndpoints, pSelectors, false);

		const uint32_t l = blk.get_low_color();
		const uint32_t h = blk.get_high_color();

		color32 color0((l >> 11) & 31, (l >> 5) & 63, l & 31, 255);
		color32 color1((h >> 11) & 31, (h >> 5) & 63, h & 31, 255);

		uint32_t g0 = color0.g & 1;
		uint32_t g1 = color1.g & 1;
		
		color0.g >>= 1;
		color1.g >>= 1;

		blk.m_selectors[0] = conv_dxt1_to_fxt1_sels(blk.m_selectors[0]);
		blk.m_selectors[1] = conv_dxt1_to_fxt1_sels(blk.m_selectors[1]);
		blk.m_selectors[2] = conv_dxt1_to_fxt1_sels(blk.m_selectors[2]);
		blk.m_selectors[3] = conv_dxt1_to_fxt1_sels(blk.m_selectors[3]);
		
		if ((blk.get_selector(0, 0) >> 1) != (g0 ^ g1))
		{
			std::swap(color0, color1);
			std::swap(g0, g1);

			blk.m_selectors[0] ^= 0xFF;
			blk.m_selectors[1] ^= 0xFF;
			blk.m_selectors[2] ^= 0xFF;
			blk.m_selectors[3] ^= 0xFF;
		}

		if (fxt1_subblock == 0)
		{
			pBlock->m_hi.m_mode = 1; 
			pBlock->m_hi.m_alpha = 0;
			pBlock->m_hi.m_glsb = g1 | (g1 << 1);
			pBlock->m_hi.m_r0 = color0.r;
			pBlock->m_hi.m_g0 = color0.g;
			pBlock->m_hi.m_b0 = color0.b;
			pBlock->m_hi.m_r1 = color1.r;
			pBlock->m_hi.m_g1 = color1.g;
			pBlock->m_hi.m_b1 = color1.b;
			pBlock->m_hi.m_r2 = color0.r;
			pBlock->m_hi.m_g2 = color0.g;
			pBlock->m_hi.m_b2 = color0.b;
			pBlock->m_hi.m_r3 = color1.r;
			pBlock->m_hi.m_g3 = color1.g;
			pBlock->m_hi.m_b3 = color1.b;
			pBlock->m_sels[0] = blk.m_selectors[0];
			pBlock->m_sels[1] = blk.m_selectors[1];
			pBlock->m_sels[2] = blk.m_selectors[2];
			pBlock->m_sels[3] = blk.m_selectors[3];

			static const uint8_t s_border_dup[4] = { 0, 85, 170, 255 };
			pBlock->m_sels[4] = s_border_dup[blk.m_selectors[0] >> 6];
			pBlock->m_sels[5] = s_border_dup[blk.m_selectors[1] >> 6];
			pBlock->m_sels[6] = s_border_dup[blk.m_selectors[2] >> 6];
			pBlock->m_sels[7] = s_border_dup[blk.m_selectors[3] >> 6];
		}
		else
		{
			pBlock->m_hi.m_glsb = (pBlock->m_hi.m_glsb & 1) | (g1 << 1);
			pBlock->m_hi.m_r2 = color0.r;
			pBlock->m_hi.m_g2 = color0.g;
			pBlock->m_hi.m_b2 = color0.b;
			pBlock->m_hi.m_r3 = color1.r;
			pBlock->m_hi.m_g3 = color1.g;
			pBlock->m_hi.m_b3 = color1.b;
			pBlock->m_sels[4] = blk.m_selectors[0];
			pBlock->m_sels[5] = blk.m_selectors[1];
			pBlock->m_sels[6] = blk.m_selectors[2];
			pBlock->m_sels[7] = blk.m_selectors[3];
		}
	}
#endif // BASISD_SUPPORT_FXT1
#if BASISD_SUPPORT_DXT5A
	static dxt_selector_range s_dxt5a_selector_ranges[] =
	{
		{ 0, 3 },

		{ 1, 3 },
		{ 0, 2 },

		{ 1, 2 },
	};

	const uint32_t NUM_DXT5A_SELECTOR_RANGES = sizeof(s_dxt5a_selector_ranges) / sizeof(s_dxt5a_selector_ranges[0]);

	struct etc1_g_to_dxt5a_conversion
	{
		uint8_t m_lo, m_hi;
		uint16_t m_trans;
	};

	static etc1_g_to_dxt5a_conversion g_etc1_g_to_dxt5a[32 * 8][NUM_DXT5A_SELECTOR_RANGES] =
	{
		{ { 8, 0, 393 },{ 8, 0, 392 },{ 2, 0, 9 },{ 2, 0, 8 }, }, { { 6, 16, 710 },{ 16, 6, 328 },{ 0, 10, 96 },{ 10, 6, 8 }, },
		{ { 28, 5, 1327 },{ 24, 14, 328 },{ 8, 18, 96 },{ 18, 14, 8 }, }, { { 36, 13, 1327 },{ 32, 22, 328 },{ 16, 26, 96 },{ 26, 22, 8 }, },
		{ { 45, 22, 1327 },{ 41, 31, 328 },{ 25, 35, 96 },{ 35, 31, 8 }, }, { { 53, 30, 1327 },{ 49, 39, 328 },{ 33, 43, 96 },{ 43, 39, 8 }, },
		{ { 61, 38, 1327 },{ 57, 47, 328 },{ 41, 51, 96 },{ 51, 47, 8 }, }, { { 69, 46, 1327 },{ 65, 55, 328 },{ 49, 59, 96 },{ 59, 55, 8 }, },
		{ { 78, 55, 1327 },{ 74, 64, 328 },{ 58, 68, 96 },{ 68, 64, 8 }, }, { { 86, 63, 1327 },{ 82, 72, 328 },{ 66, 76, 96 },{ 76, 72, 8 }, },
		{ { 94, 71, 1327 },{ 90, 80, 328 },{ 74, 84, 96 },{ 84, 80, 8 }, }, { { 102, 79, 1327 },{ 98, 88, 328 },{ 82, 92, 96 },{ 92, 88, 8 }, },
		{ { 111, 88, 1327 },{ 107, 97, 328 },{ 91, 101, 96 },{ 101, 97, 8 }, }, { { 119, 96, 1327 },{ 115, 105, 328 },{ 99, 109, 96 },{ 109, 105, 8 }, },
		{ { 127, 104, 1327 },{ 123, 113, 328 },{ 107, 117, 96 },{ 117, 113, 8 }, }, { { 135, 112, 1327 },{ 131, 121, 328 },{ 115, 125, 96 },{ 125, 121, 8 }, },
		{ { 144, 121, 1327 },{ 140, 130, 328 },{ 124, 134, 96 },{ 134, 130, 8 }, }, { { 152, 129, 1327 },{ 148, 138, 328 },{ 132, 142, 96 },{ 142, 138, 8 }, },
		{ { 160, 137, 1327 },{ 156, 146, 328 },{ 140, 150, 96 },{ 150, 146, 8 }, }, { { 168, 145, 1327 },{ 164, 154, 328 },{ 148, 158, 96 },{ 158, 154, 8 }, },
		{ { 177, 154, 1327 },{ 173, 163, 328 },{ 157, 167, 96 },{ 167, 163, 8 }, }, { { 185, 162, 1327 },{ 181, 171, 328 },{ 165, 175, 96 },{ 175, 171, 8 }, },
		{ { 193, 170, 1327 },{ 189, 179, 328 },{ 173, 183, 96 },{ 183, 179, 8 }, }, { { 201, 178, 1327 },{ 197, 187, 328 },{ 181, 191, 96 },{ 191, 187, 8 }, },
		{ { 210, 187, 1327 },{ 206, 196, 328 },{ 190, 200, 96 },{ 200, 196, 8 }, }, { { 218, 195, 1327 },{ 214, 204, 328 },{ 198, 208, 96 },{ 208, 204, 8 }, },
		{ { 226, 203, 1327 },{ 222, 212, 328 },{ 206, 216, 96 },{ 216, 212, 8 }, }, { { 234, 211, 1327 },{ 230, 220, 328 },{ 214, 224, 96 },{ 224, 220, 8 }, },
		{ { 243, 220, 1327 },{ 239, 229, 328 },{ 223, 233, 96 },{ 233, 229, 8 }, }, { { 251, 228, 1327 },{ 247, 237, 328 },{ 231, 241, 96 },{ 241, 237, 8 }, },
		{ { 239, 249, 3680 },{ 245, 249, 3648 },{ 239, 249, 96 },{ 249, 245, 8 }, }, { { 247, 253, 4040 },{ 255, 253, 8 },{ 247, 253, 456 },{ 255, 253, 8 }, },
		{ { 5, 17, 566 },{ 5, 17, 560 },{ 5, 0, 9 },{ 5, 0, 8 }, }, { { 25, 0, 313 },{ 25, 3, 328 },{ 13, 0, 49 },{ 13, 3, 8 }, },
		{ { 39, 0, 1329 },{ 33, 11, 328 },{ 11, 21, 70 },{ 21, 11, 8 }, }, { { 47, 7, 1329 },{ 41, 19, 328 },{ 29, 7, 33 },{ 29, 19, 8 }, },
		{ { 50, 11, 239 },{ 50, 28, 328 },{ 38, 16, 33 },{ 38, 28, 8 }, }, { { 92, 13, 2423 },{ 58, 36, 328 },{ 46, 24, 33 },{ 46, 36, 8 }, },
		{ { 100, 21, 2423 },{ 66, 44, 328 },{ 54, 32, 33 },{ 54, 44, 8 }, }, { { 86, 7, 1253 },{ 74, 52, 328 },{ 62, 40, 33 },{ 62, 52, 8 }, },
		{ { 95, 16, 1253 },{ 83, 61, 328 },{ 71, 49, 33 },{ 71, 61, 8 }, }, { { 103, 24, 1253 },{ 91, 69, 328 },{ 79, 57, 33 },{ 79, 69, 8 }, },
		{ { 111, 32, 1253 },{ 99, 77, 328 },{ 87, 65, 33 },{ 87, 77, 8 }, }, { { 119, 40, 1253 },{ 107, 85, 328 },{ 95, 73, 33 },{ 95, 85, 8 }, },
		{ { 128, 49, 1253 },{ 116, 94, 328 },{ 104, 82, 33 },{ 104, 94, 8 }, }, { { 136, 57, 1253 },{ 124, 102, 328 },{ 112, 90, 33 },{ 112, 102, 8 }, },
		{ { 144, 65, 1253 },{ 132, 110, 328 },{ 120, 98, 33 },{ 120, 110, 8 }, }, { { 152, 73, 1253 },{ 140, 118, 328 },{ 128, 106, 33 },{ 128, 118, 8 }, },
		{ { 161, 82, 1253 },{ 149, 127, 328 },{ 137, 115, 33 },{ 137, 127, 8 }, }, { { 169, 90, 1253 },{ 157, 135, 328 },{ 145, 123, 33 },{ 145, 135, 8 }, },
		{ { 177, 98, 1253 },{ 165, 143, 328 },{ 153, 131, 33 },{ 153, 143, 8 }, }, { { 185, 106, 1253 },{ 173, 151, 328 },{ 161, 139, 33 },{ 161, 151, 8 }, },
		{ { 194, 115, 1253 },{ 182, 160, 328 },{ 170, 148, 33 },{ 170, 160, 8 }, }, { { 202, 123, 1253 },{ 190, 168, 328 },{ 178, 156, 33 },{ 178, 168, 8 }, },
		{ { 210, 131, 1253 },{ 198, 176, 328 },{ 186, 164, 33 },{ 186, 176, 8 }, }, { { 218, 139, 1253 },{ 206, 184, 328 },{ 194, 172, 33 },{ 194, 184, 8 }, },
		{ { 227, 148, 1253 },{ 215, 193, 328 },{ 203, 181, 33 },{ 203, 193, 8 }, }, { { 235, 156, 1253 },{ 223, 201, 328 },{ 211, 189, 33 },{ 211, 201, 8 }, },
		{ { 243, 164, 1253 },{ 231, 209, 328 },{ 219, 197, 33 },{ 219, 209, 8 }, }, { { 183, 239, 867 },{ 239, 217, 328 },{ 227, 205, 33 },{ 227, 217, 8 }, },
		{ { 254, 214, 1329 },{ 248, 226, 328 },{ 236, 214, 33 },{ 236, 226, 8 }, }, { { 222, 244, 3680 },{ 234, 244, 3648 },{ 244, 222, 33 },{ 244, 234, 8 }, },
		{ { 230, 252, 3680 },{ 242, 252, 3648 },{ 252, 230, 33 },{ 252, 242, 8 }, }, { { 238, 250, 4040 },{ 255, 250, 8 },{ 238, 250, 456 },{ 255, 250, 8 }, },
		{ { 9, 29, 566 },{ 9, 29, 560 },{ 9, 0, 9 },{ 9, 0, 8 }, }, { { 17, 37, 566 },{ 17, 37, 560 },{ 17, 0, 9 },{ 17, 0, 8 }, },
		{ { 45, 0, 313 },{ 45, 0, 312 },{ 25, 0, 49 },{ 25, 7, 8 }, }, { { 14, 63, 2758 },{ 5, 53, 784 },{ 15, 33, 70 },{ 33, 15, 8 }, },
		{ { 71, 6, 1329 },{ 72, 4, 1328 },{ 42, 4, 33 },{ 42, 24, 8 }, }, { { 70, 3, 239 },{ 70, 2, 232 },{ 50, 12, 33 },{ 50, 32, 8 }, },
		{ { 0, 98, 2842 },{ 78, 10, 232 },{ 58, 20, 33 },{ 58, 40, 8 }, }, { { 97, 27, 1329 },{ 86, 18, 232 },{ 66, 28, 33 },{ 66, 48, 8 }, },
		{ { 0, 94, 867 },{ 95, 27, 232 },{ 75, 37, 33 },{ 75, 57, 8 }, }, { { 8, 102, 867 },{ 103, 35, 232 },{ 83, 45, 33 },{ 83, 65, 8 }, },
		{ { 12, 112, 867 },{ 111, 43, 232 },{ 91, 53, 33 },{ 91, 73, 8 }, }, { { 139, 2, 1253 },{ 119, 51, 232 },{ 99, 61, 33 },{ 99, 81, 8 }, },
		{ { 148, 13, 1253 },{ 128, 60, 232 },{ 108, 70, 33 },{ 108, 90, 8 }, }, { { 156, 21, 1253 },{ 136, 68, 232 },{ 116, 78, 33 },{ 116, 98, 8 }, },
		{ { 164, 29, 1253 },{ 144, 76, 232 },{ 124, 86, 33 },{ 124, 106, 8 }, }, { { 172, 37, 1253 },{ 152, 84, 232 },{ 132, 94, 33 },{ 132, 114, 8 }, },
		{ { 181, 46, 1253 },{ 161, 93, 232 },{ 141, 103, 33 },{ 141, 123, 8 }, }, { { 189, 54, 1253 },{ 169, 101, 232 },{ 149, 111, 33 },{ 149, 131, 8 }, },
		{ { 197, 62, 1253 },{ 177, 109, 232 },{ 157, 119, 33 },{ 157, 139, 8 }, }, { { 205, 70, 1253 },{ 185, 117, 232 },{ 165, 127, 33 },{ 165, 147, 8 }, },
		{ { 214, 79, 1253 },{ 194, 126, 232 },{ 174, 136, 33 },{ 174, 156, 8 }, }, { { 222, 87, 1253 },{ 202, 134, 232 },{ 182, 144, 33 },{ 182, 164, 8 }, },
		{ { 230, 95, 1253 },{ 210, 142, 232 },{ 190, 152, 33 },{ 190, 172, 8 }, }, { { 238, 103, 1253 },{ 218, 150, 232 },{ 198, 160, 33 },{ 198, 180, 8 }, },
		{ { 247, 112, 1253 },{ 227, 159, 232 },{ 207, 169, 33 },{ 207, 189, 8 }, }, { { 255, 120, 1253 },{ 235, 167, 232 },{ 215, 177, 33 },{ 215, 197, 8 }, },
		{ { 146, 243, 867 },{ 243, 175, 232 },{ 223, 185, 33 },{ 223, 205, 8 }, }, { { 184, 231, 3682 },{ 203, 251, 784 },{ 231, 193, 33 },{ 231, 213, 8 }, },
		{ { 193, 240, 3682 },{ 222, 240, 3648 },{ 240, 202, 33 },{ 240, 222, 8 }, }, { { 255, 210, 169 },{ 230, 248, 3648 },{ 248, 210, 33 },{ 248, 230, 8 }, },
		{ { 218, 238, 4040 },{ 255, 238, 8 },{ 218, 238, 456 },{ 255, 238, 8 }, }, { { 226, 246, 4040 },{ 255, 246, 8 },{ 226, 246, 456 },{ 255, 246, 8 }, },
		{ { 13, 42, 566 },{ 13, 42, 560 },{ 13, 0, 9 },{ 13, 0, 8 }, }, { { 50, 0, 329 },{ 50, 0, 328 },{ 21, 0, 9 },{ 21, 0, 8 }, },
		{ { 29, 58, 566 },{ 67, 2, 1352 },{ 3, 29, 70 },{ 29, 3, 8 }, }, { { 10, 79, 2758 },{ 76, 11, 1352 },{ 11, 37, 70 },{ 37, 11, 8 }, },
		{ { 7, 75, 790 },{ 7, 75, 784 },{ 20, 46, 70 },{ 46, 20, 8 }, }, { { 15, 83, 790 },{ 97, 1, 1328 },{ 28, 54, 70 },{ 54, 28, 8 }, },
		{ { 101, 7, 1329 },{ 105, 9, 1328 },{ 62, 0, 39 },{ 62, 36, 8 }, }, { { 99, 1, 239 },{ 99, 3, 232 },{ 1, 71, 98 },{ 70, 44, 8 }, },
		{ { 107, 11, 239 },{ 108, 12, 232 },{ 10, 80, 98 },{ 79, 53, 8 }, }, { { 115, 19, 239 },{ 116, 20, 232 },{ 18, 88, 98 },{ 87, 61, 8 }, },
		{ { 123, 27, 239 },{ 124, 28, 232 },{ 26, 96, 98 },{ 95, 69, 8 }, }, { { 131, 35, 239 },{ 132, 36, 232 },{ 34, 104, 98 },{ 103, 77, 8 }, },
		{ { 140, 44, 239 },{ 141, 45, 232 },{ 43, 113, 98 },{ 112, 86, 8 }, }, { { 148, 52, 239 },{ 149, 53, 232 },{ 51, 121, 98 },{ 120, 94, 8 }, },
		{ { 156, 60, 239 },{ 157, 61, 232 },{ 59, 129, 98 },{ 128, 102, 8 }, }, { { 164, 68, 239 },{ 165, 69, 232 },{ 67, 137, 98 },{ 136, 110, 8 }, },
		{ { 173, 77, 239 },{ 174, 78, 232 },{ 76, 146, 98 },{ 145, 119, 8 }, }, { { 181, 85, 239 },{ 182, 86, 232 },{ 84, 154, 98 },{ 153, 127, 8 }, },
		{ { 189, 93, 239 },{ 190, 94, 232 },{ 92, 162, 98 },{ 161, 135, 8 }, }, { { 197, 101, 239 },{ 198, 102, 232 },{ 100, 170, 98 },{ 169, 143, 8 }, },
		{ { 206, 110, 239 },{ 207, 111, 232 },{ 109, 179, 98 },{ 178, 152, 8 }, }, { { 214, 118, 239 },{ 215, 119, 232 },{ 117, 187, 98 },{ 186, 160, 8 }, },
		{ { 222, 126, 239 },{ 223, 127, 232 },{ 125, 195, 98 },{ 194, 168, 8 }, }, { { 230, 134, 239 },{ 231, 135, 232 },{ 133, 203, 98 },{ 202, 176, 8 }, },
		{ { 239, 143, 239 },{ 240, 144, 232 },{ 142, 212, 98 },{ 211, 185, 8 }, }, { { 247, 151, 239 },{ 180, 248, 784 },{ 150, 220, 98 },{ 219, 193, 8 }, },
		{ { 159, 228, 3682 },{ 201, 227, 3648 },{ 158, 228, 98 },{ 227, 201, 8 }, }, { { 181, 249, 3928 },{ 209, 235, 3648 },{ 166, 236, 98 },{ 235, 209, 8 }, },
		{ { 255, 189, 169 },{ 218, 244, 3648 },{ 175, 245, 98 },{ 244, 218, 8 }, }, { { 197, 226, 4040 },{ 226, 252, 3648 },{ 183, 253, 98 },{ 252, 226, 8 }, },
		{ { 205, 234, 4040 },{ 255, 234, 8 },{ 205, 234, 456 },{ 255, 234, 8 }, }, { { 213, 242, 4040 },{ 255, 242, 8 },{ 213, 242, 456 },{ 255, 242, 8 }, },
		{ { 18, 60, 566 },{ 18, 60, 560 },{ 18, 0, 9 },{ 18, 0, 8 }, }, { { 26, 68, 566 },{ 26, 68, 560 },{ 26, 0, 9 },{ 26, 0, 8 }, },
		{ { 34, 76, 566 },{ 34, 76, 560 },{ 34, 0, 9 },{ 34, 0, 8 }, }, { { 5, 104, 2758 },{ 98, 5, 1352 },{ 42, 0, 57 },{ 42, 6, 8 }, },
		{ { 92, 0, 313 },{ 93, 1, 312 },{ 15, 51, 70 },{ 51, 15, 8 }, }, { { 3, 101, 790 },{ 3, 101, 784 },{ 0, 59, 88 },{ 59, 23, 8 }, },
		{ { 14, 107, 790 },{ 11, 109, 784 },{ 31, 67, 70 },{ 67, 31, 8 }, }, { { 19, 117, 790 },{ 19, 117, 784 },{ 39, 75, 70 },{ 75, 39, 8 }, },
		{ { 28, 126, 790 },{ 28, 126, 784 },{ 83, 5, 33 },{ 84, 48, 8 }, }, { { 132, 0, 239 },{ 36, 134, 784 },{ 91, 13, 33 },{ 92, 56, 8 }, },
		{ { 142, 4, 239 },{ 44, 142, 784 },{ 99, 21, 33 },{ 100, 64, 8 }, }, { { 150, 12, 239 },{ 52, 150, 784 },{ 107, 29, 33 },{ 108, 72, 8 }, },
		{ { 159, 21, 239 },{ 61, 159, 784 },{ 116, 38, 33 },{ 117, 81, 8 }, }, { { 167, 29, 239 },{ 69, 167, 784 },{ 124, 46, 33 },{ 125, 89, 8 }, },
		{ { 175, 37, 239 },{ 77, 175, 784 },{ 132, 54, 33 },{ 133, 97, 8 }, }, { { 183, 45, 239 },{ 85, 183, 784 },{ 140, 62, 33 },{ 141, 105, 8 }, },
		{ { 192, 54, 239 },{ 94, 192, 784 },{ 149, 71, 33 },{ 150, 114, 8 }, }, { { 200, 62, 239 },{ 102, 200, 784 },{ 157, 79, 33 },{ 158, 122, 8 }, },
		{ { 208, 70, 239 },{ 110, 208, 784 },{ 165, 87, 33 },{ 166, 130, 8 }, }, { { 216, 78, 239 },{ 118, 216, 784 },{ 173, 95, 33 },{ 174, 138, 8 }, },
		{ { 225, 87, 239 },{ 127, 225, 784 },{ 182, 104, 33 },{ 183, 147, 8 }, }, { { 233, 95, 239 },{ 135, 233, 784 },{ 190, 112, 33 },{ 191, 155, 8 }, },
		{ { 241, 103, 239 },{ 143, 241, 784 },{ 198, 120, 33 },{ 199, 163, 8 }, }, { { 111, 208, 3682 },{ 151, 249, 784 },{ 206, 128, 33 },{ 207, 171, 8 }, },
		{ { 120, 217, 3682 },{ 180, 216, 3648 },{ 215, 137, 33 },{ 216, 180, 8 }, }, { { 128, 225, 3682 },{ 188, 224, 3648 },{ 223, 145, 33 },{ 224, 188, 8 }, },
		{ { 155, 253, 3928 },{ 196, 232, 3648 },{ 231, 153, 33 },{ 232, 196, 8 }, }, { { 144, 241, 3682 },{ 204, 240, 3648 },{ 239, 161, 33 },{ 240, 204, 8 }, },
		{ { 153, 250, 3682 },{ 213, 249, 3648 },{ 248, 170, 33 },{ 249, 213, 8 }, }, { { 179, 221, 4040 },{ 255, 221, 8 },{ 179, 221, 456 },{ 255, 221, 8 }, },
		{ { 187, 229, 4040 },{ 255, 229, 8 },{ 187, 229, 456 },{ 255, 229, 8 }, }, { { 195, 237, 4040 },{ 255, 237, 8 },{ 195, 237, 456 },{ 255, 237, 8 }, },
		{ { 24, 80, 566 },{ 24, 80, 560 },{ 24, 0, 9 },{ 24, 0, 8 }, }, { { 32, 88, 566 },{ 32, 88, 560 },{ 32, 0, 9 },{ 32, 0, 8 }, },
		{ { 40, 96, 566 },{ 40, 96, 560 },{ 40, 0, 9 },{ 40, 0, 8 }, }, { { 48, 104, 566 },{ 48, 104, 560 },{ 48, 0, 9 },{ 48, 0, 8 }, },
		{ { 9, 138, 2758 },{ 130, 7, 1352 },{ 9, 57, 70 },{ 57, 9, 8 }, }, { { 119, 0, 313 },{ 120, 0, 312 },{ 17, 65, 70 },{ 65, 17, 8 }, },
		{ { 0, 128, 784 },{ 128, 6, 312 },{ 25, 73, 70 },{ 73, 25, 8 }, }, { { 6, 137, 790 },{ 5, 136, 784 },{ 33, 81, 70 },{ 81, 33, 8 }, },
		{ { 42, 171, 2758 },{ 14, 145, 784 },{ 42, 90, 70 },{ 90, 42, 8 }, }, { { 50, 179, 2758 },{ 22, 153, 784 },{ 50, 98, 70 },{ 98, 50, 8 }, },
		{ { 58, 187, 2758 },{ 30, 161, 784 },{ 58, 106, 70 },{ 106, 58, 8 }, }, { { 191, 18, 1329 },{ 38, 169, 784 },{ 112, 9, 33 },{ 114, 66, 8 }, },
		{ { 176, 0, 239 },{ 47, 178, 784 },{ 121, 18, 33 },{ 123, 75, 8 }, }, { { 187, 1, 239 },{ 55, 186, 784 },{ 129, 26, 33 },{ 131, 83, 8 }, },
		{ { 195, 10, 239 },{ 63, 194, 784 },{ 137, 34, 33 },{ 139, 91, 8 }, }, { { 203, 18, 239 },{ 71, 202, 784 },{ 145, 42, 33 },{ 147, 99, 8 }, },
		{ { 212, 27, 239 },{ 80, 211, 784 },{ 154, 51, 33 },{ 156, 108, 8 }, }, { { 220, 35, 239 },{ 88, 219, 784 },{ 162, 59, 33 },{ 164, 116, 8 }, },
		{ { 228, 43, 239 },{ 96, 227, 784 },{ 170, 67, 33 },{ 172, 124, 8 }, }, { { 236, 51, 239 },{ 104, 235, 784 },{ 178, 75, 33 },{ 180, 132, 8 }, },
		{ { 245, 60, 239 },{ 113, 244, 784 },{ 187, 84, 33 },{ 189, 141, 8 }, }, { { 91, 194, 3680 },{ 149, 197, 3648 },{ 195, 92, 33 },{ 197, 149, 8 }, },
		{ { 99, 202, 3680 },{ 157, 205, 3648 },{ 203, 100, 33 },{ 205, 157, 8 }, }, { { 107, 210, 3680 },{ 165, 213, 3648 },{ 211, 108, 33 },{ 213, 165, 8 }, },
		{ { 119, 249, 3928 },{ 174, 222, 3648 },{ 220, 117, 33 },{ 222, 174, 8 }, }, { { 127, 255, 856 },{ 182, 230, 3648 },{ 228, 125, 33 },{ 230, 182, 8 }, },
		{ { 255, 135, 169 },{ 190, 238, 3648 },{ 236, 133, 33 },{ 238, 190, 8 }, }, { { 140, 243, 3680 },{ 198, 246, 3648 },{ 244, 141, 33 },{ 246, 198, 8 }, },
		{ { 151, 207, 4040 },{ 255, 207, 8 },{ 151, 207, 456 },{ 255, 207, 8 }, }, { { 159, 215, 4040 },{ 255, 215, 8 },{ 159, 215, 456 },{ 255, 215, 8 }, },
		{ { 167, 223, 4040 },{ 255, 223, 8 },{ 167, 223, 456 },{ 255, 223, 8 }, }, { { 175, 231, 4040 },{ 255, 231, 8 },{ 175, 231, 456 },{ 255, 231, 8 }, },
		{ { 33, 106, 566 },{ 33, 106, 560 },{ 33, 0, 9 },{ 33, 0, 8 }, }, { { 41, 114, 566 },{ 41, 114, 560 },{ 41, 0, 9 },{ 41, 0, 8 }, },
		{ { 49, 122, 566 },{ 49, 122, 560 },{ 49, 0, 9 },{ 49, 0, 8 }, }, { { 57, 130, 566 },{ 57, 130, 560 },{ 57, 0, 9 },{ 57, 0, 8 }, },
		{ { 66, 139, 566 },{ 66, 139, 560 },{ 66, 0, 9 },{ 66, 0, 8 }, }, { { 74, 147, 566 },{ 170, 7, 1352 },{ 8, 74, 70 },{ 74, 8, 8 }, },
		{ { 152, 0, 313 },{ 178, 15, 1352 },{ 0, 82, 80 },{ 82, 16, 8 }, }, { { 162, 0, 313 },{ 186, 23, 1352 },{ 24, 90, 70 },{ 90, 24, 8 }, },
		{ { 0, 171, 784 },{ 195, 32, 1352 },{ 33, 99, 70 },{ 99, 33, 8 }, }, { { 6, 179, 790 },{ 203, 40, 1352 },{ 41, 107, 70 },{ 107, 41, 8 }, },
		{ { 15, 187, 790 },{ 211, 48, 1352 },{ 115, 0, 41 },{ 115, 49, 8 }, }, { { 61, 199, 710 },{ 219, 56, 1352 },{ 57, 123, 70 },{ 123, 57, 8 }, },
		{ { 70, 208, 710 },{ 228, 65, 1352 },{ 66, 132, 70 },{ 132, 66, 8 }, }, { { 78, 216, 710 },{ 236, 73, 1352 },{ 74, 140, 70 },{ 140, 74, 8 }, },
		{ { 86, 224, 710 },{ 244, 81, 1352 },{ 145, 7, 33 },{ 148, 82, 8 }, }, { { 222, 8, 233 },{ 252, 89, 1352 },{ 153, 15, 33 },{ 156, 90, 8 }, },
		{ { 235, 0, 239 },{ 241, 101, 328 },{ 166, 6, 39 },{ 165, 99, 8 }, }, { { 32, 170, 3680 },{ 249, 109, 328 },{ 0, 175, 98 },{ 173, 107, 8 }, },
		{ { 40, 178, 3680 },{ 115, 181, 3648 },{ 8, 183, 98 },{ 181, 115, 8 }, }, { { 48, 186, 3680 },{ 123, 189, 3648 },{ 16, 191, 98 },{ 189, 123, 8 }, },
		{ { 57, 195, 3680 },{ 132, 198, 3648 },{ 25, 200, 98 },{ 198, 132, 8 }, }, { { 67, 243, 3928 },{ 140, 206, 3648 },{ 33, 208, 98 },{ 206, 140, 8 }, },
		{ { 76, 251, 3928 },{ 148, 214, 3648 },{ 41, 216, 98 },{ 214, 148, 8 }, }, { { 86, 255, 856 },{ 156, 222, 3648 },{ 49, 224, 98 },{ 222, 156, 8 }, },
		{ { 255, 93, 169 },{ 165, 231, 3648 },{ 58, 233, 98 },{ 231, 165, 8 }, }, { { 98, 236, 3680 },{ 173, 239, 3648 },{ 66, 241, 98 },{ 239, 173, 8 }, },
		{ { 108, 181, 4040 },{ 181, 247, 3648 },{ 74, 249, 98 },{ 247, 181, 8 }, }, { { 116, 189, 4040 },{ 255, 189, 8 },{ 116, 189, 456 },{ 255, 189, 8 }, },
		{ { 125, 198, 4040 },{ 255, 198, 8 },{ 125, 198, 456 },{ 255, 198, 8 }, }, { { 133, 206, 4040 },{ 255, 206, 8 },{ 133, 206, 456 },{ 255, 206, 8 }, },
		{ { 141, 214, 4040 },{ 255, 214, 8 },{ 141, 214, 456 },{ 255, 214, 8 }, }, { { 149, 222, 4040 },{ 255, 222, 8 },{ 149, 222, 456 },{ 255, 222, 8 }, },
		{ { 47, 183, 566 },{ 47, 183, 560 },{ 47, 0, 9 },{ 47, 0, 8 }, }, { { 55, 191, 566 },{ 55, 191, 560 },{ 55, 0, 9 },{ 55, 0, 8 }, },
		{ { 63, 199, 566 },{ 63, 199, 560 },{ 63, 0, 9 },{ 63, 0, 8 }, }, { { 71, 207, 566 },{ 71, 207, 560 },{ 71, 0, 9 },{ 71, 0, 8 }, },
		{ { 80, 216, 566 },{ 80, 216, 560 },{ 80, 0, 9 },{ 80, 0, 8 }, }, { { 88, 224, 566 },{ 88, 224, 560 },{ 88, 0, 9 },{ 88, 0, 8 }, },
		{ { 3, 233, 710 },{ 3, 233, 704 },{ 2, 96, 70 },{ 96, 2, 8 }, }, { { 11, 241, 710 },{ 11, 241, 704 },{ 10, 104, 70 },{ 104, 10, 8 }, },
		{ { 20, 250, 710 },{ 20, 250, 704 },{ 19, 113, 70 },{ 113, 19, 8 }, }, { { 27, 121, 3654 },{ 27, 121, 3648 },{ 27, 121, 70 },{ 121, 27, 8 }, },
		{ { 35, 129, 3654 },{ 35, 129, 3648 },{ 35, 129, 70 },{ 129, 35, 8 }, }, { { 43, 137, 3654 },{ 43, 137, 3648 },{ 43, 137, 70 },{ 137, 43, 8 }, },
		{ { 52, 146, 3654 },{ 52, 146, 3648 },{ 52, 146, 70 },{ 146, 52, 8 }, }, { { 60, 154, 3654 },{ 60, 154, 3648 },{ 60, 154, 70 },{ 154, 60, 8 }, },
		{ { 68, 162, 3654 },{ 68, 162, 3648 },{ 68, 162, 70 },{ 162, 68, 8 }, }, { { 76, 170, 3654 },{ 76, 170, 3648 },{ 76, 170, 70 },{ 170, 76, 8 }, },
		{ { 85, 179, 3654 },{ 85, 179, 3648 },{ 85, 179, 70 },{ 179, 85, 8 }, }, { { 93, 187, 3654 },{ 93, 187, 3648 },{ 93, 187, 70 },{ 187, 93, 8 }, },
		{ { 101, 195, 3654 },{ 101, 195, 3648 },{ 101, 195, 70 },{ 195, 101, 8 }, }, { { 109, 203, 3654 },{ 109, 203, 3648 },{ 109, 203, 70 },{ 203, 109, 8 }, },
		{ { 118, 212, 3654 },{ 118, 212, 3648 },{ 118, 212, 70 },{ 212, 118, 8 }, }, { { 126, 220, 3654 },{ 126, 220, 3648 },{ 126, 220, 70 },{ 220, 126, 8 }, },
		{ { 134, 228, 3654 },{ 134, 228, 3648 },{ 134, 228, 70 },{ 228, 134, 8 }, }, { { 5, 236, 3680 },{ 142, 236, 3648 },{ 5, 236, 96 },{ 236, 142, 8 }, },
		{ { 14, 245, 3680 },{ 151, 245, 3648 },{ 14, 245, 96 },{ 245, 151, 8 }, }, { { 23, 159, 4040 },{ 159, 253, 3648 },{ 23, 159, 456 },{ 253, 159, 8 }, },
		{ { 31, 167, 4040 },{ 255, 167, 8 },{ 31, 167, 456 },{ 255, 167, 8 }, }, { { 39, 175, 4040 },{ 255, 175, 8 },{ 39, 175, 456 },{ 255, 175, 8 }, },
		{ { 48, 184, 4040 },{ 255, 184, 8 },{ 48, 184, 456 },{ 255, 184, 8 }, }, { { 56, 192, 4040 },{ 255, 192, 8 },{ 56, 192, 456 },{ 255, 192, 8 }, },
		{ { 64, 200, 4040 },{ 255, 200, 8 },{ 64, 200, 456 },{ 255, 200, 8 }, },{ { 72, 208, 4040 },{ 255, 208, 8 },{ 72, 208, 456 },{ 255, 208, 8 }, },

	};

	struct dxt5a_block
	{
		uint8_t m_endpoints[2];

		enum { cTotalSelectorBytes = 6 };
		uint8_t m_selectors[cTotalSelectorBytes];

		inline void clear()
		{
			basisu::clear_obj(*this);
		}

		inline uint32_t get_low_alpha() const
		{
			return m_endpoints[0];
		}

		inline uint32_t get_high_alpha() const
		{
			return m_endpoints[1];
		}

		inline void set_low_alpha(uint32_t i)
		{
			assert(i <= UINT8_MAX);
			m_endpoints[0] = static_cast<uint8_t>(i);
		}

		inline void set_high_alpha(uint32_t i)
		{
			assert(i <= UINT8_MAX);
			m_endpoints[1] = static_cast<uint8_t>(i);
		}

		inline bool is_alpha6_block() const { return get_low_alpha() <= get_high_alpha(); }

		uint32_t get_endpoints_as_word() const { return m_endpoints[0] | (m_endpoints[1] << 8); }
		uint32_t get_selectors_as_word(uint32_t index) { assert(index < 3); return m_selectors[index * 2] | (m_selectors[index * 2 + 1] << 8); }

		inline uint32_t get_selector(uint32_t x, uint32_t y) const
		{
			assert((x < 4U) && (y < 4U));

			uint32_t selector_index = (y * 4) + x;
			uint32_t bit_index = selector_index * cDXT5SelectorBits;

			uint32_t byte_index = bit_index >> 3;
			uint32_t bit_ofs = bit_index & 7;

			uint32_t v = m_selectors[byte_index];
			if (byte_index < (cTotalSelectorBytes - 1))
				v |= (m_selectors[byte_index + 1] << 8);

			return (v >> bit_ofs) & 7;
		}

		inline void set_selector(uint32_t x, uint32_t y, uint32_t val)
		{
			assert((x < 4U) && (y < 4U) && (val < 8U));

			uint32_t selector_index = (y * 4) + x;
			uint32_t bit_index = selector_index * cDXT5SelectorBits;

			uint32_t byte_index = bit_index >> 3;
			uint32_t bit_ofs = bit_index & 7;

			uint32_t v = m_selectors[byte_index];
			if (byte_index < (cTotalSelectorBytes - 1))
				v |= (m_selectors[byte_index + 1] << 8);

			v &= (~(7 << bit_ofs));
			v |= (val << bit_ofs);

			m_selectors[byte_index] = static_cast<uint8_t>(v);
			if (byte_index < (cTotalSelectorBytes - 1))
				m_selectors[byte_index + 1] = static_cast<uint8_t>(v >> 8);
		}

		enum { cMaxSelectorValues = 8 };

		static uint32_t get_block_values6(color32* pDst, uint32_t l, uint32_t h)
		{
			pDst[0].a = static_cast<uint8_t>(l);
			pDst[1].a = static_cast<uint8_t>(h);
			pDst[2].a = static_cast<uint8_t>((l * 4 + h) / 5);
			pDst[3].a = static_cast<uint8_t>((l * 3 + h * 2) / 5);
			pDst[4].a = static_cast<uint8_t>((l * 2 + h * 3) / 5);
			pDst[5].a = static_cast<uint8_t>((l + h * 4) / 5);
			pDst[6].a = 0;
			pDst[7].a = 255;
			return 6;
		}

		static uint32_t get_block_values8(color32* pDst, uint32_t l, uint32_t h)
		{
			pDst[0].a = static_cast<uint8_t>(l);
			pDst[1].a = static_cast<uint8_t>(h);
			pDst[2].a = static_cast<uint8_t>((l * 6 + h) / 7);
			pDst[3].a = static_cast<uint8_t>((l * 5 + h * 2) / 7);
			pDst[4].a = static_cast<uint8_t>((l * 4 + h * 3) / 7);
			pDst[5].a = static_cast<uint8_t>((l * 3 + h * 4) / 7);
			pDst[6].a = static_cast<uint8_t>((l * 2 + h * 5) / 7);
			pDst[7].a = static_cast<uint8_t>((l + h * 6) / 7);
			return 8;
		}

		static uint32_t get_block_values(color32* pDst, uint32_t l, uint32_t h)
		{
			if (l > h)
				return get_block_values8(pDst, l, h);
			else
				return get_block_values6(pDst, l, h);
		}
	};

	static void convert_etc1s_to_dxt5a(dxt5a_block* pDst_block, const endpoint* pEndpoints, const selector* pSelector)
	{
		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		const color32& base_color = pEndpoints->m_color5;
		const uint32_t inten_table = pEndpoints->m_inten5;

		if (low_selector == high_selector)
		{
			uint32_t r;
			decoder_etc_block::get_block_color5_r(base_color, inten_table, low_selector, r);
						
			pDst_block->set_low_alpha(r);
			pDst_block->set_high_alpha(r);
			pDst_block->m_selectors[0] = 0;
			pDst_block->m_selectors[1] = 0;
			pDst_block->m_selectors[2] = 0;
			pDst_block->m_selectors[3] = 0;
			pDst_block->m_selectors[4] = 0;
			pDst_block->m_selectors[5] = 0;
			return;
		}
		else if (pSelector->m_num_unique_selectors == 2)
		{
			color32 block_colors[4];

			decoder_etc_block::get_block_colors5(block_colors, base_color, inten_table);

			const uint32_t r0 = block_colors[low_selector].r;
			const uint32_t r1 = block_colors[high_selector].r;

			pDst_block->set_low_alpha(r0);
			pDst_block->set_high_alpha(r1);

			// TODO: Optimize this
			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t s = pSelector->get_selector(x, y);
					pDst_block->set_selector(x, y, (s == high_selector) ? 1 : 0);
				}
			}

			return;
		}

		uint32_t selector_range_table = 0;
		for (selector_range_table = 0; selector_range_table < NUM_DXT5A_SELECTOR_RANGES; selector_range_table++)
			if ((low_selector == s_dxt5a_selector_ranges[selector_range_table].m_low) && (high_selector == s_dxt5a_selector_ranges[selector_range_table].m_high))
				break;
		if (selector_range_table >= NUM_DXT5A_SELECTOR_RANGES)
			selector_range_table = 0;

		const etc1_g_to_dxt5a_conversion* pTable_entry = &g_etc1_g_to_dxt5a[base_color.r + inten_table * 32][selector_range_table];

		pDst_block->set_low_alpha(pTable_entry->m_lo);
		pDst_block->set_high_alpha(pTable_entry->m_hi);

		// TODO: Optimize this (like ETC1->BC1)
		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
			{
				uint32_t s = pSelector->get_selector(x, y);

				uint32_t ds = (pTable_entry->m_trans >> (s * 3)) & 7;

				pDst_block->set_selector(x, y, ds);
			}
		}
	}
#endif //BASISD_SUPPORT_DXT5A

	// PVRTC

#if BASISD_SUPPORT_PVRTC1
	static const  uint16_t g_pvrtc_swizzle_table[256] =
	{
		0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015, 0x0040, 0x0041, 0x0044, 0x0045, 0x0050, 0x0051, 0x0054, 0x0055, 0x0100, 0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115, 0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155,
		0x0400, 0x0401, 0x0404, 0x0405, 0x0410, 0x0411, 0x0414, 0x0415, 0x0440, 0x0441, 0x0444, 0x0445, 0x0450, 0x0451, 0x0454, 0x0455, 0x0500, 0x0501, 0x0504, 0x0505, 0x0510, 0x0511, 0x0514, 0x0515, 0x0540, 0x0541, 0x0544, 0x0545, 0x0550, 0x0551, 0x0554, 0x0555,
		0x1000, 0x1001, 0x1004, 0x1005, 0x1010, 0x1011, 0x1014, 0x1015, 0x1040, 0x1041, 0x1044, 0x1045, 0x1050, 0x1051, 0x1054, 0x1055, 0x1100, 0x1101, 0x1104, 0x1105, 0x1110, 0x1111, 0x1114, 0x1115, 0x1140, 0x1141, 0x1144, 0x1145, 0x1150, 0x1151, 0x1154, 0x1155,
		0x1400, 0x1401, 0x1404, 0x1405, 0x1410, 0x1411, 0x1414, 0x1415, 0x1440, 0x1441, 0x1444, 0x1445, 0x1450, 0x1451, 0x1454, 0x1455, 0x1500, 0x1501, 0x1504, 0x1505, 0x1510, 0x1511, 0x1514, 0x1515, 0x1540, 0x1541, 0x1544, 0x1545, 0x1550, 0x1551, 0x1554, 0x1555,
		0x4000, 0x4001, 0x4004, 0x4005, 0x4010, 0x4011, 0x4014, 0x4015, 0x4040, 0x4041, 0x4044, 0x4045, 0x4050, 0x4051, 0x4054, 0x4055, 0x4100, 0x4101, 0x4104, 0x4105, 0x4110, 0x4111, 0x4114, 0x4115, 0x4140, 0x4141, 0x4144, 0x4145, 0x4150, 0x4151, 0x4154, 0x4155,
		0x4400, 0x4401, 0x4404, 0x4405, 0x4410, 0x4411, 0x4414, 0x4415, 0x4440, 0x4441, 0x4444, 0x4445, 0x4450, 0x4451, 0x4454, 0x4455, 0x4500, 0x4501, 0x4504, 0x4505, 0x4510, 0x4511, 0x4514, 0x4515, 0x4540, 0x4541, 0x4544, 0x4545, 0x4550, 0x4551, 0x4554, 0x4555,
		0x5000, 0x5001, 0x5004, 0x5005, 0x5010, 0x5011, 0x5014, 0x5015, 0x5040, 0x5041, 0x5044, 0x5045, 0x5050, 0x5051, 0x5054, 0x5055, 0x5100, 0x5101, 0x5104, 0x5105, 0x5110, 0x5111, 0x5114, 0x5115, 0x5140, 0x5141, 0x5144, 0x5145, 0x5150, 0x5151, 0x5154, 0x5155,
		0x5400, 0x5401, 0x5404, 0x5405, 0x5410, 0x5411, 0x5414, 0x5415, 0x5440, 0x5441, 0x5444, 0x5445, 0x5450, 0x5451, 0x5454, 0x5455, 0x5500, 0x5501, 0x5504, 0x5505, 0x5510, 0x5511, 0x5514, 0x5515, 0x5540, 0x5541, 0x5544, 0x5545, 0x5550, 0x5551, 0x5554, 0x5555
	};

	// Note we can't use simple calculations to convert PVRTC1 encoded endpoint components to/from 8-bits, due to hardware approximations.
	static const uint8_t g_pvrtc_5[32] = { 0,8,16,24,33,41,49,57,66,74,82,90,99,107,115,123,132,140,148,156,165,173,181,189,198,206,214,222,231,239,247,255 };
	static const uint8_t g_pvrtc_4[16] = { 0,16,33,49,66,82,99,115,140,156,173,189,206,222,239,255 };
	static const uint8_t g_pvrtc_3[8] = { 0,33,74,107,148,181,222,255 };
	static const uint8_t g_pvrtc_alpha[9] = { 0,34,68,102,136,170,204,238,255 };
		
	static const uint8_t g_pvrtc_5_floor[256] =
	{
		0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,
		3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
		7,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11,
		11,11,11,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,15,15,15,15,15,
		15,15,15,15,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,19,19,19,19,
		19,19,19,19,19,20,20,20,20,20,20,20,20,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22,22,23,23,23,
		23,23,23,23,23,23,24,24,24,24,24,24,24,24,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,27,27,
		27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,30,30,30,30,30,30,30,30,31
	};

	static const uint8_t g_pvrtc_5_ceil[256] =
	{
		0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,
		4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,
		8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,12,12,12,12,12,
		12,12,12,12,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,16,16,16,16,
		16,16,16,16,16,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,19,19,19,19,19,19,19,19,20,20,20,
		20,20,20,20,20,20,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22,22,23,23,23,23,23,23,23,23,24,24,
		24,24,24,24,24,24,24,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,28,
		28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,30,30,30,30,30,30,30,30,31,31,31,31,31,31,31,31
	};
		
	static const uint8_t g_pvrtc_4_floor[256] =
	{
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
		3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
		5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,
		7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,
		9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,
		11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,
		13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,15
	};

	static const uint8_t g_pvrtc_4_ceil[256] =
	{
		0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
		2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
		4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,
		6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,
		8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,
		10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,
		12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,
		14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
	};
		
	static const uint8_t g_pvrtc_3_floor[256] =
	{
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
		2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
		3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,
		4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,
		5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,
		6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7
	};

	static const uint8_t g_pvrtc_3_ceil[256] =
	{
		0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
		2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
		3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
		4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,
		5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,
		6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
	};
		
	static const uint8_t g_pvrtc_alpha_floor[256] =
	{
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
		2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
		3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
		4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
		5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
		6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8
	};

	static const uint8_t g_pvrtc_alpha_ceil[256] =
	{
		0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
		2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
		3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
		4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
		5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
		6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
	};

	struct pvrtc4_block
	{
		uint32_t m_modulation;
		uint32_t m_endpoints;

		pvrtc4_block() : m_modulation(0), m_endpoints(0) { }

		inline bool operator== (const pvrtc4_block& rhs) const
		{
			return (m_modulation == rhs.m_modulation) && (m_endpoints == rhs.m_endpoints);
		}

		inline void clear()
		{
			m_modulation = 0;
			m_endpoints = 0;
		}

		inline bool get_block_uses_transparent_modulation() const
		{
			return (m_endpoints & 1) != 0;
		}

		inline void set_block_uses_transparent_modulation(bool m)
		{
			m_endpoints = (m_endpoints & ~1U) | static_cast<uint32_t>(m);
		}

		inline bool is_endpoint_opaque(uint32_t endpoint_index) const
		{
			static const uint32_t s_bitmasks[2] = { 0x8000U, 0x80000000U };
			return (m_endpoints & s_bitmasks[basisu::open_range_check(endpoint_index, 2U)]) != 0;
		}

		inline void set_endpoint_opaque(uint32_t endpoint_index, bool opaque)
		{
			assert(endpoint_index < 2);
			static const uint32_t s_bitmasks[2] = { 0x8000U, 0x80000000U };
			if (opaque)
				m_endpoints |= s_bitmasks[endpoint_index];
			else
				m_endpoints &= ~s_bitmasks[endpoint_index];
		}

		inline color32 get_endpoint_5554(uint32_t endpoint_index) const
		{
			assert(endpoint_index < 2);
			static const uint32_t s_endpoint_mask[2] = { 0xFFFE, 0xFFFF };
			uint32_t packed = (m_endpoints >> (basisu::open_range_check(endpoint_index, 2U) ? 16 : 0)) & s_endpoint_mask[endpoint_index];

			uint32_t r, g, b, a;
			if (packed & 0x8000)
			{
				// opaque 554 or 555
				r = (packed >> 10) & 31;
				g = (packed >> 5) & 31;
				b = packed & 31;

				if (!endpoint_index)
					b |= (b >> 4);

				a = 0xF;
			}
			else
			{
				// translucent 4433 or 4443
				r = (packed >> 7) & 0x1E;
				g = (packed >> 3) & 0x1E;
				b = (packed & 0xF) << 1;

				r |= (r >> 4);
				g |= (g >> 4);

				if (!endpoint_index)
					b |= (b >> 3);
				else
					b |= (b >> 4);

				a = (packed >> 11) & 0xE;
			}

			assert((r < 32) && (g < 32) && (b < 32) && (a < 16));
					
			return color32(r, g, b, a);
		}
				
		inline color32 get_endpoint_8888(uint32_t endpoint_index) const
		{
			assert(endpoint_index < 2);
			static const uint32_t s_endpoint_mask[2] = { 0xFFFE, 0xFFFF };
			uint32_t packed = (m_endpoints >> (basisu::open_range_check(endpoint_index, 2U) ? 16 : 0)) & s_endpoint_mask[endpoint_index];

			uint32_t r, g, b, a;
			if (packed & 0x8000)
			{
				// opaque 554 or 555
				// 1RRRRRGGGGGBBBBM
				// 1RRRRRGGGGGBBBBB
				r = (packed >> 10) & 31;
				g = (packed >> 5) & 31;
				b = packed & 31;

				r = g_pvrtc_5[r];
				g = g_pvrtc_5[g];

				if (!endpoint_index)
					b = g_pvrtc_4[b >> 1];
				else
					b = g_pvrtc_5[b];

				a = 255;
			}
			else
			{
				// translucent 4433 or 4443
				// 0AAA RRRR GGGG BBBM
				// 0AAA RRRR GGGG BBBB
				r = (packed >> 8) & 0xF;
				g = (packed >> 4) & 0xF;
				b = packed & 0xF;
				a = (packed >> 12) & 7;

				r = g_pvrtc_4[r];
				g = g_pvrtc_4[g];

				if (!endpoint_index)
					b = g_pvrtc_3[b >> 1];
				else
					b = g_pvrtc_4[b];

				a = g_pvrtc_alpha[a];
			}
			
			return color32(r, g, b, a);
		}

		inline uint32_t get_endpoint_l8(uint32_t endpoint_index) const
		{
			color32 c(get_endpoint_8888(endpoint_index));
			return c.r + c.g + c.b + c.a;
		}
				
		inline uint32_t get_opaque_endpoint_l0() const
		{
			uint32_t packed = m_endpoints & 0xFFFE;

			uint32_t r, g, b;
			assert(packed & 0x8000);

			// opaque 554 or 555
			r = (packed >> 10) & 31;
			g = (packed >> 5) & 31;
			b = packed & 31;
			b |= (b >> 4);

			return r + g + b;
		}

		inline uint32_t get_opaque_endpoint_l1() const
		{
			uint32_t packed = m_endpoints >> 16;

			uint32_t r, g, b;
			assert(packed & 0x8000);

			// opaque 554 or 555
			r = (packed >> 10) & 31;
			g = (packed >> 5) & 31;
			b = packed & 31;

			return r + g + b;
		}

		static uint32_t get_component_precision_in_bits(uint32_t c, uint32_t endpoint_index, bool opaque_endpoint)
		{
			static const uint32_t s_comp_prec[4][4] =
			{
				// R0 G0 B0 A0      R1 G1 B1 A1
				{ 4, 4, 3, 3 },{ 4, 4, 4, 3 }, // transparent endpoint

				{ 5, 5, 4, 0 },{ 5, 5, 5, 0 }  // opaque endpoint
			};
			return s_comp_prec[basisu::open_range_check(endpoint_index, 2U) + (opaque_endpoint * 2)][basisu::open_range_check(c, 4U)];
		}

		static color32 get_color_precision_in_bits(uint32_t endpoint_index, bool opaque_endpoint)
		{
			static const color32 s_color_prec[4] =
			{
				color32(4, 4, 3, 3), color32(4, 4, 4, 3), // transparent endpoint
				color32(5, 5, 4, 0), color32(5, 5, 5, 0)  // opaque endpoint
			};
			return s_color_prec[basisu::open_range_check(endpoint_index, 2U) + (opaque_endpoint * 2)];
		}

		inline void set_opaque_endpoint_floor(uint32_t endpoint_index, const color32& c)
		{
			assert(endpoint_index < 2);
			const uint32_t m = m_endpoints & 1;

			uint32_t r = g_pvrtc_5_floor[c[0]], g = g_pvrtc_5_floor[c[1]], b = c[2];

			if (!endpoint_index)
				b = g_pvrtc_4_floor[b] << 1;
			else
				b = g_pvrtc_5_floor[b];

			// rgba=555 here
			assert((r < 32) && (g < 32) && (b < 32));

			// 1RRRRRGGGGGBBBBM
			// 1RRRRRGGGGGBBBBB

			// opaque 554 or 555
			uint32_t packed = 0x8000 | (r << 10) | (g << 5) | b;
			if (!endpoint_index)
				packed = (packed & ~1) | m;

			assert(packed <= 0xFFFF);

			if (endpoint_index)
				m_endpoints = (m_endpoints & 0xFFFFU) | (packed << 16);
			else
				m_endpoints = (m_endpoints & 0xFFFF0000U) | packed;
		}

		inline void set_opaque_endpoint_ceil(uint32_t endpoint_index, const color32& c)
		{
			assert(endpoint_index < 2);
			const uint32_t m = m_endpoints & 1;

			uint32_t r = g_pvrtc_5_ceil[c[0]], g = g_pvrtc_5_ceil[c[1]], b = c[2];

			if (!endpoint_index)
				b = g_pvrtc_4_ceil[b] << 1;
			else
				b = g_pvrtc_5_ceil[b];

			// rgba=555 here
			assert((r < 32) && (g < 32) && (b < 32));

			// 1RRRRRGGGGGBBBBM
			// 1RRRRRGGGGGBBBBB

			// opaque 554 or 555
			uint32_t packed = 0x8000 | (r << 10) | (g << 5) | b;
			if (!endpoint_index)
				packed |= m;

			assert(packed <= 0xFFFF);

			if (endpoint_index)
				m_endpoints = (m_endpoints & 0xFFFFU) | (packed << 16);
			else
				m_endpoints = (m_endpoints & 0xFFFF0000U) | packed;
		}
				
		// opaque endpoints:	554 or 555
		// transparent endpoints: 3443 or 3444
		inline void set_endpoint_raw(uint32_t endpoint_index, const color32& c, bool opaque_endpoint)
		{
			assert(endpoint_index < 2);
			const uint32_t m = m_endpoints & 1;
			uint32_t r = c[0], g = c[1], b = c[2], a = c[3];

			uint32_t packed;

			if (opaque_endpoint)
			{
				if (!endpoint_index)
				{
					// 554
					// 1RRRRRGGGGGBBBBM
					assert((r < 32) && (g < 32) && (b < 16));
					packed = 0x8000 | (r << 10) | (g << 5) | (b << 1) | m;
				}
				else
				{
					// 555
					// 1RRRRRGGGGGBBBBB
					assert((r < 32) && (g < 32) && (b < 32));
					packed = 0x8000 | (r << 10) | (g << 5) | b;
				}
			}
			else
			{
				if (!endpoint_index)
				{
					// 3443
					// 0AAA RRRR GGGG BBBM
					assert((r < 16) && (g < 16) && (b < 8) && (a < 8));
					packed = (a << 12) | (r << 8) | (g << 4) | (b << 1) | m;
				}
				else
				{
					// 3444
					// 0AAA RRRR GGGG BBBB
					assert((r < 16) && (g < 16) && (b < 16) && (a < 8));
					packed = (a << 12) | (r << 8) | (g << 4) | b;
				}
			}

			assert(packed <= 0xFFFF);

			if (endpoint_index)
				m_endpoints = (m_endpoints & 0xFFFFU) | (packed << 16);
			else
				m_endpoints = (m_endpoints & 0xFFFF0000U) | packed;
		}
				
		inline void set_endpoint_floor(uint32_t endpoint_index, const color32& c)
		{
			assert(endpoint_index < 2);

			int a = g_pvrtc_alpha_floor[c.a];
			if (a == 8)
			{
				// 554 or 555
				uint32_t r = g_pvrtc_5_floor[c[0]], g = g_pvrtc_5_floor[c[1]], b = c[2];

				if (!endpoint_index)
					b = g_pvrtc_4_floor[b];
				else
					b = g_pvrtc_5_floor[b];

				set_endpoint_raw(endpoint_index, color32(r, g, b, a), true);
			}
			else
			{
				// 4433 or 4443
				uint32_t r = g_pvrtc_4_floor[c[0]], g = g_pvrtc_4_floor[c[1]], b = c[2];

				if (!endpoint_index)
					b = g_pvrtc_3_floor[b];
				else
					b = g_pvrtc_4_floor[b];

				set_endpoint_raw(endpoint_index, color32(r, g, b, a), false);
			}
		}

		inline void set_endpoint_ceil(uint32_t endpoint_index, const color32& c)
		{
			assert(endpoint_index < 2);

			int a = g_pvrtc_alpha_ceil[c.a];
			if (a == 8)
			{
				// 554 or 555
				uint32_t r = g_pvrtc_5_ceil[c[0]], g = g_pvrtc_5_ceil[c[1]], b = c[2];

				if (!endpoint_index)
					b = g_pvrtc_4_ceil[b];
				else
					b = g_pvrtc_5_ceil[b];

				set_endpoint_raw(endpoint_index, color32(r, g, b, a), true);
			}
			else
			{
				// 4433 or 4443
				uint32_t r = g_pvrtc_4_ceil[c[0]], g = g_pvrtc_4_ceil[c[1]], b = c[2];

				if (!endpoint_index)
					b = g_pvrtc_3_ceil[b];
				else
					b = g_pvrtc_4_ceil[b];

				set_endpoint_raw(endpoint_index, color32(r, g, b, a), false);
			}
		}

		inline uint32_t get_modulation(uint32_t x, uint32_t y) const
		{
			assert((x < 4) && (y < 4));
			return (m_modulation >> ((y * 4 + x) * 2)) & 3;
		}

		// Scaled by 8
		inline const uint32_t* get_scaled_modulation_values(bool block_uses_transparent_modulation) const
		{
			static const uint32_t s_block_scales[2][4] = { { 0, 3, 5, 8 },{ 0, 4, 4, 8 } };
			return s_block_scales[block_uses_transparent_modulation];
		}

		// Scaled by 8
		inline uint32_t get_scaled_modulation(uint32_t x, uint32_t y) const
		{
			return get_scaled_modulation_values(get_block_uses_transparent_modulation())[get_modulation(x, y)];
		}

		inline void set_modulation(uint32_t x, uint32_t y, uint32_t s)
		{
			assert((x < 4) && (y < 4) && (s < 4));
			uint32_t n = (y * 4 + x) * 2;
			m_modulation = (m_modulation & (~(3 << n))) | (s << n);
			assert(get_modulation(x, y) == s);
		}

		// Assumes modulation was initialized to 0
		inline void set_modulation_fast(uint32_t x, uint32_t y, uint32_t s)
		{
			assert((x < 4) && (y < 4) && (s < 4));
			uint32_t n = (y * 4 + x) * 2;
			m_modulation |= (s << n);
			assert(get_modulation(x, y) == s);
		}
	};

	static const uint8_t g_pvrtc_bilinear_weights[16][4] =
	{
		{ 4, 4, 4, 4 }, { 2, 6, 2, 6 }, { 8, 0, 8, 0 }, { 6, 2, 6, 2 },
		{ 2, 2, 6, 6 }, { 1, 3, 3, 9 }, { 4, 0, 12, 0 }, { 3, 1, 9, 3 },
		{ 8, 8, 0, 0 }, { 4, 12, 0, 0 }, { 16, 0, 0, 0 }, { 12, 4, 0, 0 },
		{ 6, 6, 2, 2 }, { 3, 9, 1, 3 }, { 12, 0, 4, 0 }, { 9, 3, 3, 1 },
	};

	struct pvrtc1_temp_block
	{
		decoder_etc_block m_etc1_block;
		uint32_t m_pvrtc_endpoints;
	};

	static inline uint32_t get_opaque_endpoint_l0(uint32_t endpoints)
	{
		uint32_t packed = endpoints;

		uint32_t r, g, b;
		assert(packed & 0x8000);

		r = (packed >> 10) & 31;
		g = (packed >> 5) & 31;
		b = packed & 30;
		b |= (b >> 4);

		return r + g + b;
	}

	static inline uint32_t get_opaque_endpoint_l1(uint32_t endpoints)
	{
		uint32_t packed = endpoints >> 16;

		uint32_t r, g, b;
		assert(packed & 0x8000);

		r = (packed >> 10) & 31;
		g = (packed >> 5) & 31;
		b = packed & 31;

		return r + g + b;
	}

	static color32 get_endpoint_8888(uint32_t endpoints, uint32_t endpoint_index)
	{
		assert(endpoint_index < 2);
		static const uint32_t s_endpoint_mask[2] = { 0xFFFE, 0xFFFF };
		uint32_t packed = (endpoints >> (basisu::open_range_check(endpoint_index, 2U) ? 16 : 0)) & s_endpoint_mask[endpoint_index];

		uint32_t r, g, b, a;
		if (packed & 0x8000)
		{
			// opaque 554 or 555
			// 1RRRRRGGGGGBBBBM
			// 1RRRRRGGGGGBBBBB
			r = (packed >> 10) & 31;
			g = (packed >> 5) & 31;
			b = packed & 31;

			r = g_pvrtc_5[r];
			g = g_pvrtc_5[g];

			if (!endpoint_index)
				b = g_pvrtc_4[b >> 1];
			else
				b = g_pvrtc_5[b];

			a = 255;
		}
		else
		{
			// translucent 4433 or 4443
			// 0AAA RRRR GGGG BBBM
			// 0AAA RRRR GGGG BBBB
			r = (packed >> 8) & 0xF;
			g = (packed >> 4) & 0xF;
			b = packed & 0xF;
			a = (packed >> 12) & 7;

			r = g_pvrtc_4[r];
			g = g_pvrtc_4[g];

			if (!endpoint_index)
				b = g_pvrtc_3[b >> 1];
			else
				b = g_pvrtc_4[b];

			a = g_pvrtc_alpha[a];
		}

		return color32(r, g, b, a);
	}

	static uint32_t get_endpoint_l8(uint32_t endpoints, uint32_t endpoint_index)
	{
		color32 c(get_endpoint_8888(endpoints, endpoint_index));
		return c.r + c.g + c.b + c.a;
	}

	// TODO: Support decoding a non-pow2 ETC1S texture into the next larger pow2 PVRTC texture.
	static void fixup_pvrtc1_4_modulation_rgb(const decoder_etc_block* pETC_Blocks, const uint32_t* pPVRTC_endpoints, void* pDst_blocks, uint32_t num_blocks_x, uint32_t num_blocks_y)
	{
		const uint32_t x_mask = num_blocks_x - 1;
		const uint32_t y_mask = num_blocks_y - 1;
		const uint32_t x_bits = basisu::total_bits(x_mask);
		const uint32_t y_bits = basisu::total_bits(y_mask);
		const uint32_t min_bits = basisu::minimum(x_bits, y_bits);
		const uint32_t max_bits = basisu::maximum(x_bits, y_bits);
		const uint32_t swizzle_mask = (1 << (min_bits * 2)) - 1;

		uint32_t block_index = 0;

		// really 3x3
		int e0[4][4], e1[4][4];

		for (int y = 0; y < static_cast<int>(num_blocks_y); y++)
		{
			const uint32_t* pE_rows[3];

			for (int ey = 0; ey < 3; ey++)
			{
				int by = y + ey - 1; 

				const uint32_t* pE = &pPVRTC_endpoints[(by & y_mask) * num_blocks_x];

				pE_rows[ey] = pE;

				for (int ex = 0; ex < 3; ex++)
				{
					int bx = 0 + ex - 1; 

					const uint32_t e = pE[bx & x_mask];

					e0[ex][ey] = (get_opaque_endpoint_l0(e) * 255) / 31;
					e1[ex][ey] = (get_opaque_endpoint_l1(e) * 255) / 31;
				}
			}

			const uint32_t y_swizzle = (g_pvrtc_swizzle_table[y >> 8] << 16) | g_pvrtc_swizzle_table[y & 0xFF];

			for (int x = 0; x < static_cast<int>(num_blocks_x); x++, block_index++)
			{
				const decoder_etc_block& src_block = pETC_Blocks[block_index];

				const uint32_t x_swizzle = (g_pvrtc_swizzle_table[x >> 8] << 17) | (g_pvrtc_swizzle_table[x & 0xFF] << 1);

				uint32_t swizzled = x_swizzle | y_swizzle;
				if (num_blocks_x != num_blocks_y)
				{
					swizzled &= swizzle_mask;

					if (num_blocks_x > num_blocks_y)
						swizzled |= ((x >> min_bits) << (min_bits * 2));
					else
						swizzled |= ((y >> min_bits) << (min_bits * 2));
				}

				pvrtc4_block* pDst_block = static_cast<pvrtc4_block*>(pDst_blocks) + swizzled;
				pDst_block->m_endpoints = pPVRTC_endpoints[block_index];

				uint32_t base_r = g_etc_5_to_8[src_block.m_differential.m_red1];
				uint32_t base_g = g_etc_5_to_8[src_block.m_differential.m_green1];
				uint32_t base_b = g_etc_5_to_8[src_block.m_differential.m_blue1];

				const int* pInten_table48 = g_etc1_inten_tables48[src_block.m_differential.m_cw1];
				int by = (base_r + base_g + base_b) * 16;
				int block_colors_y_x16[4];
				block_colors_y_x16[0] = by + pInten_table48[2];
				block_colors_y_x16[1] = by + pInten_table48[3];
				block_colors_y_x16[2] = by + pInten_table48[1];
				block_colors_y_x16[3] = by + pInten_table48[0];

				{
					const uint32_t ex = 2;
					int bx = x + ex - 1;
					bx &= x_mask;

#define DO_ROW(ey) \
					{ \
						const uint32_t e = pE_rows[ey][bx]; \
						e0[ex][ey] = (get_opaque_endpoint_l0(e) * 255) / 31; \
						e1[ex][ey] = (get_opaque_endpoint_l1(e) * 255) / 31; \
					}

					DO_ROW(0);
					DO_ROW(1);
					DO_ROW(2);
#undef DO_ROW
				}

				uint32_t mod = 0;

				uint32_t lookup_x[4];

#define DO_LOOKUP(lx) { \
					const uint32_t byte_ofs = 7 - (((lx) * 4) >> 3); \
					const uint32_t lsb_bits = src_block.m_bytes[byte_ofs] >> (((lx) & 1) * 4); \
					const uint32_t msb_bits = src_block.m_bytes[byte_ofs - 2] >> (((lx) & 1) * 4); \
					lookup_x[lx] = (lsb_bits & 0xF) | ((msb_bits & 0xF) << 4); }

				DO_LOOKUP(0);
				DO_LOOKUP(1);
				DO_LOOKUP(2);
				DO_LOOKUP(3);
#undef DO_LOOKUP

#define DO_PIX(lx, ly, w0, w1, w2, w3) \
				{ \
					int ca_l = a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3; \
					int cb_l = b0 * w0 + b1 * w1 + b2 * w2 + b3 * w3; \
					int cl = block_colors_y_x16[g_etc1_x_selector_unpack[ly][lookup_x[lx]]]; \
					int dl = cb_l - ca_l; \
					int vl = cl - ca_l; \
					int p = vl * 16; \
					if (ca_l > cb_l) { p = -p; dl = -dl; } \
					uint32_t m = 0; \
					if (p > 3 * dl) m = (uint32_t)(1 << ((ly) * 8 + (lx) * 2)); \
					if (p > 8 * dl) m = (uint32_t)(2 << ((ly) * 8 + (lx) * 2)); \
					if (p > 13 * dl) m = (uint32_t)(3 << ((ly) * 8 + (lx) * 2)); \
					mod |= m; \
				}

				{
					const uint32_t ex = 0, ey = 0;
					const int a0 = e0[ex][ey], a1 = e0[ex + 1][ey], a2 = e0[ex][ey + 1], a3 = e0[ex + 1][ey + 1];
					const int b0 = e1[ex][ey], b1 = e1[ex + 1][ey], b2 = e1[ex][ey + 1], b3 = e1[ex + 1][ey + 1];
					DO_PIX(0, 0, 4, 4, 4, 4);
					DO_PIX(1, 0, 2, 6, 2, 6);
					DO_PIX(0, 1, 2, 2, 6, 6);
					DO_PIX(1, 1, 1, 3, 3, 9);
				}

				{
					const uint32_t ex = 1, ey = 0;
					const int a0 = e0[ex][ey], a1 = e0[ex + 1][ey], a2 = e0[ex][ey + 1], a3 = e0[ex + 1][ey + 1];
					const int b0 = e1[ex][ey], b1 = e1[ex + 1][ey], b2 = e1[ex][ey + 1], b3 = e1[ex + 1][ey + 1];
					DO_PIX(2, 0, 8, 0, 8, 0);
					DO_PIX(3, 0, 6, 2, 6, 2);
					DO_PIX(2, 1, 4, 0, 12, 0);
					DO_PIX(3, 1, 3, 1, 9, 3);
				}

				{
					const uint32_t ex = 0, ey = 1;
					const int a0 = e0[ex][ey], a1 = e0[ex + 1][ey], a2 = e0[ex][ey + 1], a3 = e0[ex + 1][ey + 1];
					const int b0 = e1[ex][ey], b1 = e1[ex + 1][ey], b2 = e1[ex][ey + 1], b3 = e1[ex + 1][ey + 1];
					DO_PIX(0, 2, 8, 8, 0, 0);
					DO_PIX(1, 2, 4, 12, 0, 0);
					DO_PIX(0, 3, 6, 6, 2, 2);
					DO_PIX(1, 3, 3, 9, 1, 3);
				}

				{
					const uint32_t ex = 1, ey = 1;
					const int a0 = e0[ex][ey], a1 = e0[ex + 1][ey], a2 = e0[ex][ey + 1], a3 = e0[ex + 1][ey + 1];
					const int b0 = e1[ex][ey], b1 = e1[ex + 1][ey], b2 = e1[ex][ey + 1], b3 = e1[ex + 1][ey + 1];
					DO_PIX(2, 2, 16, 0, 0, 0);
					DO_PIX(3, 2, 12, 4, 0, 0);
					DO_PIX(2, 3, 12, 0, 4, 0);
					DO_PIX(3, 3, 9, 3, 3, 1);
				}
#undef DO_PIX

				pDst_block->m_modulation = mod;

				e0[0][0] = e0[1][0]; e0[1][0] = e0[2][0];
				e0[0][1] = e0[1][1]; e0[1][1] = e0[2][1];
				e0[0][2] = e0[1][2]; e0[1][2] = e0[2][2];

				e1[0][0] = e1[1][0]; e1[1][0] = e1[2][0];
				e1[0][1] = e1[1][1]; e1[1][1] = e1[2][1];
				e1[0][2] = e1[1][2]; e1[1][2] = e1[2][2];

			} // x
		} // y
	}

	static void fixup_pvrtc1_4_modulation_rgba(
		const decoder_etc_block* pETC_Blocks, 
		const uint32_t* pPVRTC_endpoints, 
		void* pDst_blocks, uint32_t num_blocks_x, uint32_t num_blocks_y, void *pAlpha_blocks,
		const endpoint* pEndpoints, const selector* pSelectors)
	{
		const uint32_t x_mask = num_blocks_x - 1;
		const uint32_t y_mask = num_blocks_y - 1;
		const uint32_t x_bits = basisu::total_bits(x_mask);
		const uint32_t y_bits = basisu::total_bits(y_mask);
		const uint32_t min_bits = basisu::minimum(x_bits, y_bits);
		const uint32_t max_bits = basisu::maximum(x_bits, y_bits);
		const uint32_t swizzle_mask = (1 << (min_bits * 2)) - 1;

		uint32_t block_index = 0;

		// really 3x3
		int e0[4][4], e1[4][4];

		for (int y = 0; y < static_cast<int>(num_blocks_y); y++)
		{
			const uint32_t* pE_rows[3];

			for (int ey = 0; ey < 3; ey++)
			{
				int by = y + ey - 1; 

				const uint32_t* pE = &pPVRTC_endpoints[(by & y_mask) * num_blocks_x];

				pE_rows[ey] = pE;

				for (int ex = 0; ex < 3; ex++)
				{
					int bx = 0 + ex - 1; 

					const uint32_t e = pE[bx & x_mask];

					e0[ex][ey] = get_endpoint_l8(e, 0);
					e1[ex][ey] = get_endpoint_l8(e, 1);
				}
			}

			const uint32_t y_swizzle = (g_pvrtc_swizzle_table[y >> 8] << 16) | g_pvrtc_swizzle_table[y & 0xFF];

			for (int x = 0; x < static_cast<int>(num_blocks_x); x++, block_index++)
			{
				const decoder_etc_block& src_block = pETC_Blocks[block_index];
				
				const uint16_t* pSrc_alpha_block = reinterpret_cast<const uint16_t*>(static_cast<const uint32_t*>(pAlpha_blocks) + x + (y * num_blocks_x));
				const endpoint* pAlpha_endpoints = &pEndpoints[pSrc_alpha_block[0]];
				const selector* pAlpha_selectors = &pSelectors[pSrc_alpha_block[1]];
				
				const uint32_t x_swizzle = (g_pvrtc_swizzle_table[x >> 8] << 17) | (g_pvrtc_swizzle_table[x & 0xFF] << 1);
				
				uint32_t swizzled = x_swizzle | y_swizzle;
				if (num_blocks_x != num_blocks_y)
				{
					swizzled &= swizzle_mask;

					if (num_blocks_x > num_blocks_y)
						swizzled |= ((x >> min_bits) << (min_bits * 2));
					else
						swizzled |= ((y >> min_bits) << (min_bits * 2));
				}

				pvrtc4_block* pDst_block = static_cast<pvrtc4_block*>(pDst_blocks) + swizzled;
				pDst_block->m_endpoints = pPVRTC_endpoints[block_index];

				uint32_t base_r = g_etc_5_to_8[src_block.m_differential.m_red1];
				uint32_t base_g = g_etc_5_to_8[src_block.m_differential.m_green1];
				uint32_t base_b = g_etc_5_to_8[src_block.m_differential.m_blue1];

				const int* pInten_table48 = g_etc1_inten_tables48[src_block.m_differential.m_cw1];
				int by = (base_r + base_g + base_b) * 16;
				int block_colors_y_x16[4];
				block_colors_y_x16[0] = basisu::clamp<int>(by + pInten_table48[0], 0, 48 * 255);
				block_colors_y_x16[1] = basisu::clamp<int>(by + pInten_table48[1], 0, 48 * 255);
				block_colors_y_x16[2] = basisu::clamp<int>(by + pInten_table48[2], 0, 48 * 255);
				block_colors_y_x16[3] = basisu::clamp<int>(by + pInten_table48[3], 0, 48 * 255);

				uint32_t alpha_base_g = g_etc_5_to_8[pAlpha_endpoints->m_color5.g] * 16;
				const int* pInten_table16 = g_etc1_inten_tables16[pAlpha_endpoints->m_inten5];
				int alpha_block_colors_x16[4];
				alpha_block_colors_x16[0] = basisu::clamp<int>(alpha_base_g + pInten_table16[0], 0, 16 * 255);
				alpha_block_colors_x16[1] = basisu::clamp<int>(alpha_base_g + pInten_table16[1], 0, 16 * 255);
				alpha_block_colors_x16[2] = basisu::clamp<int>(alpha_base_g + pInten_table16[2], 0, 16 * 255);
				alpha_block_colors_x16[3] = basisu::clamp<int>(alpha_base_g + pInten_table16[3], 0, 16 * 255);

				// clamp((base_r + base_g + base_b) * 16 + color_inten[s] * 48) + clamp(alpha_base_g * 16 + alpha_inten[as] * 16)

				{
					const uint32_t ex = 2;
					int bx = x + ex - 1;
					bx &= x_mask;

#define DO_ROW(ey) \
					{ \
						const uint32_t e = pE_rows[ey][bx]; \
						e0[ex][ey] = get_endpoint_l8(e, 0); \
						e1[ex][ey] = get_endpoint_l8(e, 1); \
					}

					DO_ROW(0);
					DO_ROW(1);
					DO_ROW(2);
#undef DO_ROW
				}

				uint32_t mod = 0;

#define DO_PIX(lx, ly, w0, w1, w2, w3) \
				{ \
					int ca_l = a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3; \
					int cb_l = b0 * w0 + b1 * w1 + b2 * w2 + b3 * w3; \
					int cl = block_colors_y_x16[(src_block.m_bytes[4 + ly] >> (lx * 2)) & 3] + alpha_block_colors_x16[(pAlpha_selectors->m_selectors[ly] >> (lx * 2)) & 3]; \
					int dl = cb_l - ca_l; \
					int vl = cl - ca_l; \
					int p = vl * 16; \
					if (ca_l > cb_l) { p = -p; dl = -dl; } \
					uint32_t m = 0; \
					if (p > 3 * dl) m = (uint32_t)(1 << ((ly) * 8 + (lx) * 2)); \
					if (p > 8 * dl) m = (uint32_t)(2 << ((ly) * 8 + (lx) * 2)); \
					if (p > 13 * dl) m = (uint32_t)(3 << ((ly) * 8 + (lx) * 2)); \
					mod |= m; \
				}

				{
					const uint32_t ex = 0, ey = 0;
					const int a0 = e0[ex][ey], a1 = e0[ex + 1][ey], a2 = e0[ex][ey + 1], a3 = e0[ex + 1][ey + 1];
					const int b0 = e1[ex][ey], b1 = e1[ex + 1][ey], b2 = e1[ex][ey + 1], b3 = e1[ex + 1][ey + 1];
					DO_PIX(0, 0, 4, 4, 4, 4);
					DO_PIX(1, 0, 2, 6, 2, 6);
					DO_PIX(0, 1, 2, 2, 6, 6);
					DO_PIX(1, 1, 1, 3, 3, 9);
				}

				{
					const uint32_t ex = 1, ey = 0;
					const int a0 = e0[ex][ey], a1 = e0[ex + 1][ey], a2 = e0[ex][ey + 1], a3 = e0[ex + 1][ey + 1];
					const int b0 = e1[ex][ey], b1 = e1[ex + 1][ey], b2 = e1[ex][ey + 1], b3 = e1[ex + 1][ey + 1];
					DO_PIX(2, 0, 8, 0, 8, 0);
					DO_PIX(3, 0, 6, 2, 6, 2);
					DO_PIX(2, 1, 4, 0, 12, 0);
					DO_PIX(3, 1, 3, 1, 9, 3);
				}

				{
					const uint32_t ex = 0, ey = 1;
					const int a0 = e0[ex][ey], a1 = e0[ex + 1][ey], a2 = e0[ex][ey + 1], a3 = e0[ex + 1][ey + 1];
					const int b0 = e1[ex][ey], b1 = e1[ex + 1][ey], b2 = e1[ex][ey + 1], b3 = e1[ex + 1][ey + 1];
					DO_PIX(0, 2, 8, 8, 0, 0);
					DO_PIX(1, 2, 4, 12, 0, 0);
					DO_PIX(0, 3, 6, 6, 2, 2);
					DO_PIX(1, 3, 3, 9, 1, 3);
				}

				{
					const uint32_t ex = 1, ey = 1;
					const int a0 = e0[ex][ey], a1 = e0[ex + 1][ey], a2 = e0[ex][ey + 1], a3 = e0[ex + 1][ey + 1];
					const int b0 = e1[ex][ey], b1 = e1[ex + 1][ey], b2 = e1[ex][ey + 1], b3 = e1[ex + 1][ey + 1];
					DO_PIX(2, 2, 16, 0, 0, 0);
					DO_PIX(3, 2, 12, 4, 0, 0);
					DO_PIX(2, 3, 12, 0, 4, 0);
					DO_PIX(3, 3, 9, 3, 3, 1);
				}
#undef DO_PIX

				pDst_block->m_modulation = mod;

				e0[0][0] = e0[1][0]; e0[1][0] = e0[2][0];
				e0[0][1] = e0[1][1]; e0[1][1] = e0[2][1];
				e0[0][2] = e0[1][2]; e0[1][2] = e0[2][2];

				e1[0][0] = e1[1][0]; e1[1][0] = e1[2][0];
				e1[0][1] = e1[1][1]; e1[1][1] = e1[2][1];
				e1[0][2] = e1[1][2]; e1[1][2] = e1[2][2];

			} // x
		} // y
	}
#endif // BASISD_SUPPORT_PVRTC1

#if BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY
	struct bc7_mode_6
	{
		struct
		{
			uint64_t m_mode : 7;
			uint64_t m_r0 : 7;
			uint64_t m_r1 : 7;
			uint64_t m_g0 : 7;
			uint64_t m_g1 : 7;
			uint64_t m_b0 : 7;
			uint64_t m_b1 : 7;
			uint64_t m_a0 : 7;
			uint64_t m_a1 : 7;
			uint64_t m_p0 : 1;
		} m_lo;

		union
		{
			struct
			{
				uint64_t m_p1 : 1;
				uint64_t m_s00 : 3;
				uint64_t m_s10 : 4;
				uint64_t m_s20 : 4;
				uint64_t m_s30 : 4;

				uint64_t m_s01 : 4;
				uint64_t m_s11 : 4;
				uint64_t m_s21 : 4;
				uint64_t m_s31 : 4;

				uint64_t m_s02 : 4;
				uint64_t m_s12 : 4;
				uint64_t m_s22 : 4;
				uint64_t m_s32 : 4;

				uint64_t m_s03 : 4;
				uint64_t m_s13 : 4;
				uint64_t m_s23 : 4;
				uint64_t m_s33 : 4;

			} m_hi;

			uint64_t m_hi_bits;
		};
	};

	static void convert_etc1s_to_bc7_m6(bc7_mode_6* pDst_block, const endpoint *pEndpoint, const selector* pSelector)
	{
#if !BASISD_WRITE_NEW_BC7_TABLES
		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;
				
		const uint32_t base_color_r = pEndpoint->m_color5.r;
		const uint32_t base_color_g = pEndpoint->m_color5.g;
		const uint32_t base_color_b = pEndpoint->m_color5.b;
		const uint32_t inten_table = pEndpoint->m_inten5;

		if (pSelector->m_num_unique_selectors <= 2)
		{
			// Only two unique selectors so just switch to block truncation coding (BTC) to avoid quality issues on extreme blocks.
			pDst_block->m_lo.m_mode = 64;

			pDst_block->m_lo.m_a0 = 127;
			pDst_block->m_lo.m_a1 = 127;

			color32 block_colors[4];

			decoder_etc_block::get_block_colors5(block_colors, color32(base_color_r, base_color_g, base_color_b, 255), inten_table);

			const uint32_t r0 = block_colors[low_selector].r;
			const uint32_t g0 = block_colors[low_selector].g;
			const uint32_t b0 = block_colors[low_selector].b;
			const uint32_t low_bits0 = (r0 & 1) + (g0 & 1) + (b0 & 1);
			uint32_t p0 = low_bits0 >= 2;

			const uint32_t r1 = block_colors[high_selector].r;
			const uint32_t g1 = block_colors[high_selector].g;
			const uint32_t b1 = block_colors[high_selector].b;
			const uint32_t low_bits1 = (r1 & 1) + (g1 & 1) + (b1 & 1);
			uint32_t p1 = low_bits1 >= 2;

			pDst_block->m_lo.m_r0 = r0 >> 1;
			pDst_block->m_lo.m_g0 = g0 >> 1;
			pDst_block->m_lo.m_b0 = b0 >> 1;
			pDst_block->m_lo.m_p0 = p0;

			pDst_block->m_lo.m_r1 = r1 >> 1;
			pDst_block->m_lo.m_g1 = g1 >> 1;
			pDst_block->m_lo.m_b1 = b1 >> 1;

			uint32_t output_low_selector = 0;
			uint32_t output_bit_offset = 1;
			uint64_t output_hi_bits = p1;

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t s = pSelector->get_selector(x, y);
					uint32_t os = (s == low_selector) ? output_low_selector : (15 ^ output_low_selector);

					uint32_t num_bits = 4;

					if ((x | y) == 0)
					{
						if (os & 8)
						{
							pDst_block->m_lo.m_r0 = r1 >> 1;
							pDst_block->m_lo.m_g0 = g1 >> 1;
							pDst_block->m_lo.m_b0 = b1 >> 1;
							pDst_block->m_lo.m_p0 = p1;

							pDst_block->m_lo.m_r1 = r0 >> 1;
							pDst_block->m_lo.m_g1 = g0 >> 1;
							pDst_block->m_lo.m_b1 = b0 >> 1;

							output_hi_bits &= ~1ULL;
							output_hi_bits |= p0;
							std::swap(p0, p1);

							output_low_selector = 15;
							os = 0;
						}

						num_bits = 3;
					}

					output_hi_bits |= (static_cast<uint64_t>(os) << output_bit_offset);
					output_bit_offset += num_bits;
				}
			}

			pDst_block->m_hi_bits = output_hi_bits;

			assert(pDst_block->m_hi.m_p1 == p1);

			return;
		}

		uint32_t selector_range_table = g_etc1_to_bc7_m6_selector_range_index[low_selector][high_selector];

		const uint32_t* pTable_r = g_etc1_to_bc7_m6_table[base_color_r + inten_table * 32] + (selector_range_table * NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS);
		const uint32_t* pTable_g = g_etc1_to_bc7_m6_table[base_color_g + inten_table * 32] + (selector_range_table * NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS);
		const uint32_t* pTable_b = g_etc1_to_bc7_m6_table[base_color_b + inten_table * 32] + (selector_range_table * NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS);

#if 1
		assert(NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS == 48);

		uint32_t best_err0 = UINT_MAX, best_err1 = UINT_MAX;

#define DO_ITER2(idx) \
		{  \
			uint32_t v0 = ((pTable_r[(idx)+0] + pTable_g[(idx)+0] + pTable_b[(idx)+0]) << 14) | ((idx) + 0); if (v0 < best_err0) best_err0 = v0; \
			uint32_t v1 = ((pTable_r[(idx)+1] + pTable_g[(idx)+1] + pTable_b[(idx)+1]) << 14) | ((idx) + 1); if (v1 < best_err1) best_err1 = v1; \
		}
#define DO_ITER4(idx) DO_ITER2(idx); DO_ITER2((idx) + 2);
#define DO_ITER8(idx) DO_ITER4(idx); DO_ITER4((idx) + 4);
#define DO_ITER16(idx) DO_ITER8(idx); DO_ITER8((idx) + 8);

		DO_ITER16(0);
		DO_ITER16(16);
		DO_ITER16(32);
#undef DO_ITER2
#undef DO_ITER4
#undef DO_ITER8
#undef DO_ITER16

		uint32_t best_err = basisu::minimum(best_err0, best_err1);
		uint32_t best_mapping = best_err & 0xFF;
		//best_err >>= 14;
#else
		uint32_t best_err = UINT_MAX;
		uint32_t best_mapping = 0;
		assert((NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS % 2) == 0);
		for (uint32_t m = 0; m < NUM_ETC1_TO_BC7_M6_SELECTOR_MAPPINGS; m += 2)
		{
#define DO_ITER(idx)	{ uint32_t total_err = (pTable_r[idx] + pTable_g[idx] + pTable_b[idx]) & 0x3FFFF; if (total_err < best_err) { best_err = total_err; best_mapping = idx; } }
			DO_ITER(m);
			DO_ITER(m + 1);
#undef DO_ITER
		}
#endif		

		pDst_block->m_lo.m_mode = 64;

		pDst_block->m_lo.m_a0 = 127;
		pDst_block->m_lo.m_a1 = 127;

		uint64_t v = 0;
		const uint8_t* pSelectors_xlat;

		if (g_etc1_to_bc7_selector_mappings[best_mapping][pSelector->get_selector(0, 0)] & 8)
		{
			pDst_block->m_lo.m_r1 = (pTable_r[best_mapping] >> 18) & 0x7F;
			pDst_block->m_lo.m_g1 = (pTable_g[best_mapping] >> 18) & 0x7F;
			pDst_block->m_lo.m_b1 = (pTable_b[best_mapping] >> 18) & 0x7F;

			pDst_block->m_lo.m_r0 = (pTable_r[best_mapping] >> 25) & 0x7F;
			pDst_block->m_lo.m_g0 = (pTable_g[best_mapping] >> 25) & 0x7F;
			pDst_block->m_lo.m_b0 = (pTable_b[best_mapping] >> 25) & 0x7F;

			pDst_block->m_lo.m_p0 = 1;
			pDst_block->m_hi.m_p1 = 0;

			v = 0;
			pSelectors_xlat = &g_etc1_to_bc7_selector_mappings_inv[best_mapping][0];
		}
		else
		{
			pDst_block->m_lo.m_r0 = (pTable_r[best_mapping] >> 18) & 0x7F;
			pDst_block->m_lo.m_g0 = (pTable_g[best_mapping] >> 18) & 0x7F;
			pDst_block->m_lo.m_b0 = (pTable_b[best_mapping] >> 18) & 0x7F;

			pDst_block->m_lo.m_r1 = (pTable_r[best_mapping] >> 25) & 0x7F;
			pDst_block->m_lo.m_g1 = (pTable_g[best_mapping] >> 25) & 0x7F;
			pDst_block->m_lo.m_b1 = (pTable_b[best_mapping] >> 25) & 0x7F;

			pDst_block->m_lo.m_p0 = 0;
			pDst_block->m_hi.m_p1 = 1;

			v = 1;
			pSelectors_xlat = &g_etc1_to_bc7_selector_mappings[best_mapping][0];
		}

		uint64_t v1 = 0, v2 = 0, v3 = 0;

#define DO_X(x, s0, s1, s2, s3) { \
		v |= ((uint64_t)pSelectors_xlat[(pSelector->m_selectors[0] >> ((x) * 2)) & 3] << (s0)); \
		v1 |= ((uint64_t)pSelectors_xlat[(pSelector->m_selectors[1] >> ((x) * 2)) & 3] << (s1)); \
		v2 |= ((uint64_t)pSelectors_xlat[(pSelector->m_selectors[2] >> ((x) * 2)) & 3] << (s2)); \
		v3 |= ((uint64_t)pSelectors_xlat[(pSelector->m_selectors[3] >> ((x) * 2)) & 3] << (s3)); }

		// 1  4  8  12
		// 16 20 24 28
		// 32 36 40 44
		// 48 52 56 60

		DO_X(0, 1, 16, 32, 48);
		DO_X(1, 4, 20, 36, 52);
		DO_X(2, 8, 24, 40, 56);
		DO_X(3, 12, 28, 44, 60);
#undef DO_X

		pDst_block->m_hi_bits = v | v1 | v2 | v3;
#endif

	}
#endif // BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY

#if BASISD_SUPPORT_BC7_MODE5
	static dxt_selector_range g_etc1_to_bc7_m5_selector_ranges[] =
	{
		{ 0, 3 },
		{ 1, 3 },
		{ 0, 2 },
		{ 1, 2 },
		{ 2, 3 },
		{ 0, 1 },
	};

	const uint32_t NUM_ETC1_TO_BC7_M5_SELECTOR_RANGES = sizeof(g_etc1_to_bc7_m5_selector_ranges) / sizeof(g_etc1_to_bc7_m5_selector_ranges[0]);

	static uint32_t g_etc1_to_bc7_m5_selector_range_index[4][4];
	
	const uint32_t NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS = 10;
	static const uint8_t g_etc1_to_bc7_m5_selector_mappings[NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS][4] =
	{
		{ 0, 0, 1, 1 },
		{ 0, 0, 1, 2 },
		{ 0, 0, 1, 3 },
		{ 0, 0, 2, 3 },
		{ 0, 1, 1, 1 },
		{ 0, 1, 2, 2 },
		{ 0, 1, 2, 3 },
		{ 0, 2, 3, 3 },
		{ 1, 2, 2, 2 },
		{ 1, 2, 3, 3 },
	};

	struct etc1_to_bc7_m5_solution
	{
		uint8_t m_lo;
		uint8_t m_hi;
		uint16_t m_err;
	};
		
	static const etc1_to_bc7_m5_solution g_etc1_to_bc7_m5_color[32 * 8 * NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS * NUM_ETC1_TO_BC7_M5_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_bc7_m5_color.inc"
	};
	
	static dxt_selector_range g_etc1_to_bc7_m5a_selector_ranges[] =
	{
		{ 0, 3 },
		{ 1, 3 },
		{ 0, 2 },
		{ 1, 2 },
		{ 2, 3 },
		{ 0, 1 }
	};

	const uint32_t NUM_ETC1_TO_BC7_M5A_SELECTOR_RANGES = sizeof(g_etc1_to_bc7_m5a_selector_ranges) / sizeof(g_etc1_to_bc7_m5a_selector_ranges[0]);

	static uint32_t g_etc1_to_bc7_m5a_selector_range_index[4][4];

	struct etc1_g_to_bc7_m5a_conversion
	{
		uint8_t m_lo, m_hi;
		uint8_t m_trans;
	};

	static etc1_g_to_bc7_m5a_conversion g_etc1_g_to_bc7_m5a[8 * 32 * NUM_ETC1_TO_BC7_M5A_SELECTOR_RANGES] =
	{
		#include "basisu_transcoder_tables_bc7_m5_alpha.inc"
	};
	
	static inline uint32_t set_block_bits(uint8_t* pBytes, uint32_t val, uint32_t num_bits, uint32_t cur_ofs)
	{
		assert(num_bits < 32);
		assert(val < (1ULL << num_bits));

		uint32_t mask = (1 << num_bits) - 1;

		while (num_bits)
		{
			const uint32_t n = basisu::minimum<uint32_t>(8 - (cur_ofs & 7), num_bits);

			pBytes[cur_ofs >> 3] &= ~static_cast<uint8_t>(mask << (cur_ofs & 7));
			pBytes[cur_ofs >> 3] |= static_cast<uint8_t>(val << (cur_ofs & 7));

			val >>= n;
			mask >>= n;

			num_bits -= n;
			cur_ofs += n;
		}

		return cur_ofs;
	}

	struct bc7_mode_5
	{
		union
		{
			struct
			{
				uint64_t m_mode : 6;
				uint64_t m_rot : 2;

				uint64_t m_r0 : 7;
				uint64_t m_r1 : 7;
				uint64_t m_g0 : 7;
				uint64_t m_g1 : 7;
				uint64_t m_b0 : 7;
				uint64_t m_b1 : 7;
				uint64_t m_a0 : 8;
				uint64_t m_a1_0 : 6;

			} m_lo;

			uint64_t m_lo_bits;
		};

		union
		{
			struct
			{
				uint64_t m_a1_1 : 2;

				// bit 2
				uint64_t m_c00 : 1;
				uint64_t m_c10 : 2;
				uint64_t m_c20 : 2;
				uint64_t m_c30 : 2;

				uint64_t m_c01 : 2;
				uint64_t m_c11 : 2;
				uint64_t m_c21 : 2;
				uint64_t m_c31 : 2;

				uint64_t m_c02 : 2;
				uint64_t m_c12 : 2;
				uint64_t m_c22 : 2;
				uint64_t m_c32 : 2;

				uint64_t m_c03 : 2;
				uint64_t m_c13 : 2;
				uint64_t m_c23 : 2;
				uint64_t m_c33 : 2;

				// bit 33
				uint64_t m_a00 : 1;
				uint64_t m_a10 : 2;
				uint64_t m_a20 : 2;
				uint64_t m_a30 : 2;

				uint64_t m_a01 : 2;
				uint64_t m_a11 : 2;
				uint64_t m_a21 : 2;
				uint64_t m_a31 : 2;

				uint64_t m_a02 : 2;
				uint64_t m_a12 : 2;
				uint64_t m_a22 : 2;
				uint64_t m_a32 : 2;

				uint64_t m_a03 : 2;
				uint64_t m_a13 : 2;
				uint64_t m_a23 : 2;
				uint64_t m_a33 : 2;

			} m_hi;

			uint64_t m_hi_bits;
		};
	};

#if BASISD_WRITE_NEW_BC7_MODE5_TABLES
	static void create_etc1_to_bc7_m5_color_conversion_table()
	{
		FILE* pFile = nullptr;
		fopen_s(&pFile, "basisu_transcoder_tables_bc7_m5_color.inc", "w");

		uint32_t n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1_TO_BC7_M5_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1_to_bc7_m5_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1_to_bc7_m5_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 127; hi++)
						{
							for (uint32_t lo = 0; lo <= 127; lo++)
							{
								uint32_t colors[4];

								colors[0] = (lo << 1) | (lo >> 6);
								colors[3] = (hi << 1) | (hi >> 6);

								colors[1] = (colors[0] * (64 - 21) + colors[3] * 21 + 32) / 64;
								colors[2] = (colors[0] * (64 - 43) + colors[3] * 43 + 32) / 64;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1_to_bc7_m5_selector_mappings[m][s]];

									int err_scale = 1;
									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										err_scale = 5;

									total_err += (err * err) * err_scale;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						best_err = basisu::minimum<uint32_t>(best_err, 0xFFFF);

						fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, (uint32_t)best_err);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m
				} // sr
			} // g
		} // inten

		fclose(pFile);
	}

	static void create_etc1_to_bc7_m5_alpha_conversion_table()
	{
		FILE* pFile = nullptr;
		fopen_s(&pFile, "basisu_transcoder_tables_bc7_m5_alpha.inc", "w");

		uint32_t n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1_TO_BC7_M5A_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1_to_bc7_m5a_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1_to_bc7_m5a_selector_ranges[sr].m_high;

					uint32_t best_lo = 0;
					uint32_t best_hi = 0;
					uint64_t best_err = UINT64_MAX;
					uint32_t best_output_selectors = 0;

					for (uint32_t hi = 0; hi <= 255; hi++)
					{
						for (uint32_t lo = 0; lo <= 255; lo++)
						{
							uint32_t colors[4];

							colors[0] = lo;
							colors[3] = hi;

							colors[1] = (colors[0] * (64 - 21) + colors[3] * 21 + 32) / 64;
							colors[2] = (colors[0] * (64 - 43) + colors[3] * 43 + 32) / 64;

							uint64_t total_err = 0;
							uint32_t output_selectors = 0;

							for (uint32_t s = low_selector; s <= high_selector; s++)
							{
								int best_mapping_err = INT_MAX;
								int best_k = 0;
								for (int k = 0; k < 4; k++)
								{
									int mapping_err = block_colors[s].g - colors[k];
									mapping_err *= mapping_err;

									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										mapping_err *= 5;

									if (mapping_err < best_mapping_err)
									{
										best_mapping_err = mapping_err;
										best_k = k;
									}
								} // k
								
								total_err += best_mapping_err;
								output_selectors |= (best_k << (s * 2));
							} // s

							if (total_err < best_err)
							{
								best_err = total_err;
								best_lo = lo;
								best_hi = hi;
								best_output_selectors = output_selectors;
							}

						} // lo
					} // hi
										
					fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, best_output_selectors);
					n++;
					if ((n & 31) == 31)
						fprintf(pFile, "\n");

				} // sr
			} // g
		} // inten

		fclose(pFile);
	}
#endif // BASISD_WRITE_NEW_BC7_MODE5_TABLES

	struct bc7_m5_match_entry
	{
		uint8_t m_hi;
		uint8_t m_lo;
	};

	static bc7_m5_match_entry g_bc7_m5_equals_1[256] =
	{
		{0,0},{1,0},{3,0},{4,0},{6,0},{7,0},{9,0},{10,0},{12,0},{13,0},{15,0},{16,0},{18,0},{20,0},{21,0},{23,0},
		{24,0},{26,0},{27,0},{29,0},{30,0},{32,0},{33,0},{35,0},{36,0},{38,0},{39,0},{41,0},{42,0},{44,0},{45,0},{47,0},
		{48,0},{50,0},{52,0},{53,0},{55,0},{56,0},{58,0},{59,0},{61,0},{62,0},{64,0},{65,0},{66,0},{68,0},{69,0},{71,0},
		{72,0},{74,0},{75,0},{77,0},{78,0},{80,0},{82,0},{83,0},{85,0},{86,0},{88,0},{89,0},{91,0},{92,0},{94,0},{95,0},
		{97,0},{98,0},{100,0},{101,0},{103,0},{104,0},{106,0},{107,0},{109,0},{110,0},{112,0},{114,0},{115,0},{117,0},{118,0},{120,0},
		{121,0},{123,0},{124,0},{126,0},{127,0},{127,1},{126,2},{126,3},{127,3},{127,4},{126,5},{126,6},{127,6},{127,7},{126,8},{126,9},
		{127,9},{127,10},{126,11},{126,12},{127,12},{127,13},{126,14},{125,15},{127,15},{126,16},{126,17},{127,17},{127,18},{126,19},{126,20},{127,20},
		{127,21},{126,22},{126,23},{127,23},{127,24},{126,25},{126,26},{127,26},{127,27},{126,28},{126,29},{127,29},{127,30},{126,31},{126,32},{127,32},
		{127,33},{126,34},{126,35},{127,35},{127,36},{126,37},{126,38},{127,38},{127,39},{126,40},{126,41},{127,41},{127,42},{126,43},{126,44},{127,44},
		{127,45},{126,46},{125,47},{127,47},{126,48},{126,49},{127,49},{127,50},{126,51},{126,52},{127,52},{127,53},{126,54},{126,55},{127,55},{127,56},
		{126,57},{126,58},{127,58},{127,59},{126,60},{126,61},{127,61},{127,62},{126,63},{125,64},{126,64},{126,65},{127,65},{127,66},{126,67},{126,68},
		{127,68},{127,69},{126,70},{126,71},{127,71},{127,72},{126,73},{126,74},{127,74},{127,75},{126,76},{125,77},{127,77},{126,78},{126,79},{127,79},
		{127,80},{126,81},{126,82},{127,82},{127,83},{126,84},{126,85},{127,85},{127,86},{126,87},{126,88},{127,88},{127,89},{126,90},{126,91},{127,91},
		{127,92},{126,93},{126,94},{127,94},{127,95},{126,96},{126,97},{127,97},{127,98},{126,99},{126,100},{127,100},{127,101},{126,102},{126,103},{127,103},
		{127,104},{126,105},{126,106},{127,106},{127,107},{126,108},{125,109},{127,109},{126,110},{126,111},{127,111},{127,112},{126,113},{126,114},{127,114},{127,115},
		{126,116},{126,117},{127,117},{127,118},{126,119},{126,120},{127,120},{127,121},{126,122},{126,123},{127,123},{127,124},{126,125},{126,126},{127,126},{127,127}
	};
	
	static void transcoder_init_bc7_mode5()
	{
#if 0
		// This is a little too much work to do at init time, so precompute it.
		for (int i = 0; i < 256; i++)
		{
			int lowest_e = 256;
			for (int lo = 0; lo < 128; lo++)
			{
				for (int hi = 0; hi < 128; hi++)
				{
					const int lo_e = (lo << 1) | (lo >> 6);
					const int hi_e = (hi << 1) | (hi >> 6);

					// Selector 1
					int v = (lo_e * (64 - 21) + hi_e * 21 + 32) >> 6;
					int e = abs(v - i);

					if (e < lowest_e)
					{
						g_bc7_m5_equals_1[i].m_hi = static_cast<uint8_t>(hi);
						g_bc7_m5_equals_1[i].m_lo = static_cast<uint8_t>(lo);

						lowest_e = e;
					}

				} // hi
								
			} // lo
			
			printf("{%u,%u},", g_bc7_m5_equals_1[i].m_hi, g_bc7_m5_equals_1[i].m_lo);
			if ((i & 15) == 15) printf("\n");
		}
#endif

		for (uint32_t i = 0; i < NUM_ETC1_TO_BC7_M5_SELECTOR_RANGES; i++)
		{
			uint32_t l = g_etc1_to_bc7_m5_selector_ranges[i].m_low;
			uint32_t h = g_etc1_to_bc7_m5_selector_ranges[i].m_high;
			g_etc1_to_bc7_m5_selector_range_index[l][h] = i;
		}

		for (uint32_t i = 0; i < NUM_ETC1_TO_BC7_M5A_SELECTOR_RANGES; i++)
		{
			uint32_t l = g_etc1_to_bc7_m5a_selector_ranges[i].m_low;
			uint32_t h = g_etc1_to_bc7_m5a_selector_ranges[i].m_high;
			g_etc1_to_bc7_m5a_selector_range_index[l][h] = i;
		}
	}

	static void convert_etc1s_to_bc7_m5_color(void* pDst, const endpoint* pEndpoints, const selector* pSelector)
	{
		bc7_mode_5* pDst_block = static_cast<bc7_mode_5*>(pDst);
				
		static_cast<uint64_t*>(pDst)[0] = 0;
		static_cast<uint64_t*>(pDst)[1] = 0;

		pDst_block->m_lo.m_mode = 1 << 5;
		pDst_block->m_lo.m_a0 = 255;
		pDst_block->m_lo.m_a1_0 = 63;
		pDst_block->m_hi.m_a1_1 = 3;

		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		const uint32_t base_color_r = pEndpoints->m_color5.r;
		const uint32_t base_color_g = pEndpoints->m_color5.g;
		const uint32_t base_color_b = pEndpoints->m_color5.b;
		const uint32_t inten_table = pEndpoints->m_inten5;

		if (pSelector->m_num_unique_selectors == 1)
		{
			// Solid color block - use precomputed tables and set selectors to 1.
			uint32_t r, g, b;
			decoder_etc_block::get_block_color5(pEndpoints->m_color5, inten_table, low_selector, r, g, b);

			pDst_block->m_lo.m_r0 = g_bc7_m5_equals_1[r].m_lo;
			pDst_block->m_lo.m_g0 = g_bc7_m5_equals_1[g].m_lo;
			pDst_block->m_lo.m_b0 = g_bc7_m5_equals_1[b].m_lo;

			pDst_block->m_lo.m_r1 = g_bc7_m5_equals_1[r].m_hi;
			pDst_block->m_lo.m_g1 = g_bc7_m5_equals_1[g].m_hi;
			pDst_block->m_lo.m_b1 = g_bc7_m5_equals_1[b].m_hi;

			set_block_bits((uint8_t*)pDst, 0x2aaaaaab, 31, 66);
			return;
		}
		else if (pSelector->m_num_unique_selectors == 2)
		{
			// Only one or two unique selectors, so just switch to block truncation coding (BTC) to avoid quality issues on extreme blocks.
			color32 block_colors[4];

			decoder_etc_block::get_block_colors5(block_colors, color32(base_color_r, base_color_g, base_color_b, 255), inten_table);

			const uint32_t r0 = block_colors[low_selector].r;
			const uint32_t g0 = block_colors[low_selector].g;
			const uint32_t b0 = block_colors[low_selector].b;

			const uint32_t r1 = block_colors[high_selector].r;
			const uint32_t g1 = block_colors[high_selector].g;
			const uint32_t b1 = block_colors[high_selector].b;

			pDst_block->m_lo.m_r0 = r0 >> 1;
			pDst_block->m_lo.m_g0 = g0 >> 1;
			pDst_block->m_lo.m_b0 = b0 >> 1;

			pDst_block->m_lo.m_r1 = r1 >> 1;
			pDst_block->m_lo.m_g1 = g1 >> 1;
			pDst_block->m_lo.m_b1 = b1 >> 1;

			uint32_t output_low_selector = 0, output_bit_offset = 0, output_bits = 0;

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t s = pSelector->get_selector(x, y);
					uint32_t os = (s == low_selector) ? output_low_selector : (3 ^ output_low_selector);

					uint32_t num_bits = 2;

					if ((x | y) == 0)
					{
						if (os & 2)
						{
							pDst_block->m_lo.m_r0 = r1 >> 1;
							pDst_block->m_lo.m_g0 = g1 >> 1;
							pDst_block->m_lo.m_b0 = b1 >> 1;

							pDst_block->m_lo.m_r1 = r0 >> 1;
							pDst_block->m_lo.m_g1 = g0 >> 1;
							pDst_block->m_lo.m_b1 = b0 >> 1;

							output_low_selector = 3;
							os = 0;
						}

						num_bits = 1;
					}

					output_bits |= (os << output_bit_offset);
					output_bit_offset += num_bits;
				}
			}

			set_block_bits((uint8_t*)pDst, output_bits, 31, 66);
			return;
		}

		const uint32_t selector_range_table = g_etc1_to_bc7_m5_selector_range_index[low_selector][high_selector];

		//[32][8][RANGES][MAPPING]
		const etc1_to_bc7_m5_solution* pTable_r = &g_etc1_to_bc7_m5_color[(inten_table * 32 + base_color_r) * (NUM_ETC1_TO_BC7_M5_SELECTOR_RANGES * NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS];
		const etc1_to_bc7_m5_solution* pTable_g = &g_etc1_to_bc7_m5_color[(inten_table * 32 + base_color_g) * (NUM_ETC1_TO_BC7_M5_SELECTOR_RANGES * NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS];
		const etc1_to_bc7_m5_solution* pTable_b = &g_etc1_to_bc7_m5_color[(inten_table * 32 + base_color_b) * (NUM_ETC1_TO_BC7_M5_SELECTOR_RANGES * NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS];

		uint32_t best_err = UINT_MAX;
		uint32_t best_mapping = 0;

		assert(NUM_ETC1_TO_BC7_M5_SELECTOR_MAPPINGS == 10);
#define DO_ITER(m) { uint32_t total_err = pTable_r[m].m_err + pTable_g[m].m_err + pTable_b[m].m_err; if (total_err < best_err) { best_err = total_err; best_mapping = m; } }
		DO_ITER(0); DO_ITER(1); DO_ITER(2); DO_ITER(3); DO_ITER(4);
		DO_ITER(5); DO_ITER(6); DO_ITER(7); DO_ITER(8); DO_ITER(9);
#undef DO_ITER

		const uint8_t* pSelectors_xlat = &g_etc1_to_bc7_m5_selector_mappings[best_mapping][0];

		uint32_t s_inv = 0;
		if (pSelectors_xlat[pSelector->get_selector(0, 0)] & 2)
		{
			pDst_block->m_lo.m_r0 = pTable_r[best_mapping].m_hi;
			pDst_block->m_lo.m_g0 = pTable_g[best_mapping].m_hi;
			pDst_block->m_lo.m_b0 = pTable_b[best_mapping].m_hi;

			pDst_block->m_lo.m_r1 = pTable_r[best_mapping].m_lo;
			pDst_block->m_lo.m_g1 = pTable_g[best_mapping].m_lo;
			pDst_block->m_lo.m_b1 = pTable_b[best_mapping].m_lo;
			
			s_inv = 3;
		}
		else
		{
			pDst_block->m_lo.m_r0 = pTable_r[best_mapping].m_lo;
			pDst_block->m_lo.m_g0 = pTable_g[best_mapping].m_lo;
			pDst_block->m_lo.m_b0 = pTable_b[best_mapping].m_lo;

			pDst_block->m_lo.m_r1 = pTable_r[best_mapping].m_hi;
			pDst_block->m_lo.m_g1 = pTable_g[best_mapping].m_hi;
			pDst_block->m_lo.m_b1 = pTable_b[best_mapping].m_hi;
		}

		uint32_t output_bits = 0, output_bit_ofs = 0;

		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
			{
				const uint32_t s = pSelector->get_selector(x, y);
				
				const uint32_t os = pSelectors_xlat[s] ^ s_inv;

				output_bits |= (os << output_bit_ofs);

				output_bit_ofs += (((x | y) == 0) ? 1 : 2);
			}
		}

		set_block_bits((uint8_t*)pDst, output_bits, 31, 66);
	}

	static void convert_etc1s_to_bc7_m5_alpha(void* pDst, const endpoint* pEndpoints, const selector* pSelector)
	{
		bc7_mode_5* pDst_block = static_cast<bc7_mode_5*>(pDst);

		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		const uint32_t base_color_r = pEndpoints->m_color5.r;
		const uint32_t inten_table = pEndpoints->m_inten5;

		if (pSelector->m_num_unique_selectors == 1)
		{
			uint32_t r;
			decoder_etc_block::get_block_color5_r(pEndpoints->m_color5, inten_table, low_selector, r);

			pDst_block->m_lo.m_a0 = r;
			pDst_block->m_lo.m_a1_0 = r & 63;
			pDst_block->m_hi.m_a1_1 = r >> 6;
					  
			return;
		}
		else if (pSelector->m_num_unique_selectors == 2)
		{
			// Only one or two unique selectors, so just switch to block truncation coding (BTC) to avoid quality issues on extreme blocks.
			int block_colors[4];

			decoder_etc_block::get_block_colors5_g(block_colors, pEndpoints->m_color5, inten_table);

			pDst_block->m_lo.m_a0 = block_colors[low_selector];
			pDst_block->m_lo.m_a1_0 = block_colors[high_selector] & 63;
			pDst_block->m_hi.m_a1_1 = block_colors[high_selector] >> 6;

			uint32_t output_low_selector = 0, output_bit_offset = 0, output_bits = 0;

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					const uint32_t s = pSelector->get_selector(x, y);
					uint32_t os = (s == low_selector) ? output_low_selector : (3 ^ output_low_selector);

					uint32_t num_bits = 2;

					if ((x | y) == 0)
					{
						if (os & 2)
						{
							pDst_block->m_lo.m_a0 = block_colors[high_selector];
							pDst_block->m_lo.m_a1_0 = block_colors[low_selector] & 63;
							pDst_block->m_hi.m_a1_1 = block_colors[low_selector] >> 6;

							output_low_selector = 3;
							os = 0;
						}

						num_bits = 1;
					}

					output_bits |= (os << output_bit_offset);
					output_bit_offset += num_bits;
				}
			}

			set_block_bits((uint8_t*)pDst, output_bits, 31, 97);
			return;
		}

		const uint32_t selector_range_table = g_etc1_to_bc7_m5a_selector_range_index[low_selector][high_selector];
						
		const etc1_g_to_bc7_m5a_conversion* pTable = &g_etc1_g_to_bc7_m5a[inten_table * (32 * NUM_ETC1_TO_BC7_M5A_SELECTOR_RANGES) + base_color_r * NUM_ETC1_TO_BC7_M5A_SELECTOR_RANGES + selector_range_table];

		pDst_block->m_lo.m_a0 = pTable->m_lo;
		pDst_block->m_lo.m_a1_0 = pTable->m_hi & 63;
		pDst_block->m_hi.m_a1_1 = pTable->m_hi >> 6;

		uint32_t output_bit_offset = 0, output_bits = 0, selector_trans = pTable->m_trans;

		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
			{
				const uint32_t s = pSelector->get_selector(x, y);
				uint32_t os = (selector_trans >> (s * 2)) & 3;

				uint32_t num_bits = 2;

				if ((x | y) == 0)
				{
					if (os & 2)
					{
						pDst_block->m_lo.m_a0 = pTable->m_hi;
						pDst_block->m_lo.m_a1_0 = pTable->m_lo & 63;
						pDst_block->m_hi.m_a1_1 = pTable->m_lo >> 6;

						selector_trans ^= 0xFF;
						os ^= 3;
					}

					num_bits = 1;
				}

				output_bits |= (os << output_bit_offset);
				output_bit_offset += num_bits;
			}
		}

		set_block_bits((uint8_t*)pDst, output_bits, 31, 97);
	}
#endif // BASISD_SUPPORT_BC7_MODE5
	
#if BASISD_SUPPORT_ETC2_EAC_A8
	static void convert_etc1s_to_etc2_eac_a8(eac_block* pDst_block, const endpoint* pEndpoints, const selector* pSelector)
	{
		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		const color32& base_color = pEndpoints->m_color5;
		const uint32_t inten_table = pEndpoints->m_inten5;

		if (low_selector == high_selector)
		{
			uint32_t r;
			decoder_etc_block::get_block_color5_r(base_color, inten_table, low_selector, r);

			// Constant alpha block
			// Select table 13, use selector 4 (0), set multiplier to 1 and base color g
			pDst_block->m_base = r;
			pDst_block->m_table = 13;
			pDst_block->m_multiplier = 1;

			// selectors are all 4's
			static const uint8_t s_etc2_eac_a8_sel4[6] = { 0x92, 0x49, 0x24, 0x92, 0x49, 0x24 };
			memcpy(pDst_block->m_selectors, s_etc2_eac_a8_sel4, sizeof(s_etc2_eac_a8_sel4));

			return;
		}

		uint32_t selector_range_table = 0;
		for (selector_range_table = 0; selector_range_table < NUM_ETC2_EAC_SELECTOR_RANGES; selector_range_table++)
			if ((low_selector == s_etc2_eac_selector_ranges[selector_range_table].m_low) && (high_selector == s_etc2_eac_selector_ranges[selector_range_table].m_high))
				break;
		if (selector_range_table >= NUM_ETC2_EAC_SELECTOR_RANGES)
			selector_range_table = 0;

		const etc1_g_to_eac_conversion* pTable_entry = &s_etc1_g_to_etc2_a8[base_color.r + inten_table * 32][selector_range_table];

		pDst_block->m_base = pTable_entry->m_base;
		pDst_block->m_table = pTable_entry->m_table_mul >> 4;
		pDst_block->m_multiplier = pTable_entry->m_table_mul & 15;

		uint64_t selector_bits = 0;

		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
			{
				uint32_t s = pSelector->get_selector(x, y);

				uint32_t ds = (pTable_entry->m_trans >> (s * 3)) & 7;

				const uint32_t dst_ofs = 45 - (y + x * 4) * 3;
				selector_bits |= (static_cast<uint64_t>(ds) << dst_ofs);
			}
		}

		pDst_block->set_selector_bits(selector_bits);
	}
#endif // BASISD_SUPPORT_ETC2_EAC_A8

#if BASISD_SUPPORT_ETC2_EAC_RG11
	static const etc1_g_to_eac_conversion s_etc1_g_to_etc2_r11[32 * 8][NUM_ETC2_EAC_SELECTOR_RANGES] =
	{
		{{0,1,3328},{0,1,3328},{0,16,457},{0,16,456}},
		{{0,226,3936},{0,226,3936},{0,17,424},{8,0,472}},
		{{6,178,4012},{6,178,4008},{0,146,501},{16,0,472}},
		{{14,178,4012},{14,178,4008},{8,146,501},{24,0,472}},
		{{23,178,4012},{23,178,4008},{17,146,501},{33,0,472}},
		{{31,178,4012},{31,178,4008},{25,146,501},{41,0,472}},
		{{39,178,4012},{39,178,4008},{33,146,501},{49,0,472}},
		{{47,178,4012},{47,178,4008},{41,146,501},{27,228,496}},
		{{56,178,4012},{56,178,4008},{50,146,501},{36,228,496}},
		{{64,178,4012},{64,178,4008},{58,146,501},{44,228,496}},
		{{72,178,4012},{72,178,4008},{66,146,501},{52,228,496}},
		{{80,178,4012},{80,178,4008},{74,146,501},{60,228,496}},
		{{89,178,4012},{89,178,4008},{83,146,501},{69,228,496}},
		{{97,178,4012},{97,178,4008},{91,146,501},{77,228,496}},
		{{105,178,4012},{105,178,4008},{99,146,501},{85,228,496}},
		{{113,178,4012},{113,178,4008},{107,146,501},{93,228,496}},
		{{122,178,4012},{122,178,4008},{116,146,501},{102,228,496}},
		{{130,178,4012},{130,178,4008},{124,146,501},{110,228,496}},
		{{138,178,4012},{138,178,4008},{132,146,501},{118,228,496}},
		{{146,178,4012},{146,178,4008},{140,146,501},{126,228,496}},
		{{155,178,4012},{155,178,4008},{149,146,501},{135,228,496}},
		{{163,178,4012},{163,178,4008},{157,146,501},{143,228,496}},
		{{171,178,4012},{171,178,4008},{165,146,501},{151,228,496}},
		{{179,178,4012},{179,178,4008},{173,146,501},{159,228,496}},
		{{188,178,4012},{188,178,4008},{182,146,501},{168,228,496}},
		{{196,178,4012},{196,178,4008},{190,146,501},{176,228,496}},
		{{204,178,4012},{204,178,4008},{198,146,501},{184,228,496}},
		{{212,178,4012},{212,178,4008},{206,146,501},{192,228,496}},
		{{221,178,4012},{221,178,4008},{215,146,501},{201,228,496}},
		{{229,178,4012},{229,178,4008},{223,146,501},{209,228,496}},
		{{235,66,4012},{221,100,4008},{231,146,501},{217,228,496}},
		{{211,102,4085},{254,32,4040},{211,102,501},{254,32,456}},
		{{0,2,3328},{0,2,3328},{0,1,320},{0,1,320}},
		{{7,162,3905},{7,162,3904},{0,17,480},{0,17,480}},
		{{15,162,3906},{15,162,3904},{1,117,352},{1,117,352}},
		{{23,162,3906},{23,162,3904},{5,34,500},{4,53,424}},
		{{32,162,3906},{32,162,3904},{14,34,500},{3,69,424}},
		{{40,162,3906},{40,162,3904},{22,34,500},{1,133,496}},
		{{48,162,3906},{48,162,3904},{30,34,500},{4,85,496}},
		{{56,162,3906},{56,162,3904},{38,34,500},{12,85,496}},
		{{65,162,3906},{65,162,3904},{47,34,500},{1,106,424}},
		{{73,162,3906},{73,162,3904},{55,34,500},{9,106,424}},
		{{81,162,3906},{81,162,3904},{63,34,500},{7,234,496}},
		{{89,162,3906},{89,162,3904},{71,34,500},{15,234,496}},
		{{98,162,3906},{98,162,3904},{80,34,500},{24,234,496}},
		{{106,162,3906},{106,162,3904},{88,34,500},{32,234,496}},
		{{114,162,3906},{114,162,3904},{96,34,500},{40,234,496}},
		{{122,162,3906},{122,162,3904},{104,34,500},{48,234,496}},
		{{131,162,3906},{131,162,3904},{113,34,500},{57,234,496}},
		{{139,162,3906},{139,162,3904},{121,34,500},{65,234,496}},
		{{147,162,3906},{147,162,3904},{129,34,500},{73,234,496}},
		{{155,162,3906},{155,162,3904},{137,34,500},{81,234,496}},
		{{164,162,3906},{164,162,3904},{146,34,500},{90,234,496}},
		{{172,162,3906},{172,162,3904},{154,34,500},{98,234,496}},
		{{180,162,3906},{180,162,3904},{162,34,500},{106,234,496}},
		{{188,162,3906},{188,162,3904},{170,34,500},{114,234,496}},
		{{197,162,3906},{197,162,3904},{179,34,500},{123,234,496}},
		{{205,162,3906},{205,162,3904},{187,34,500},{131,234,496}},
		{{213,162,3906},{213,162,3904},{195,34,500},{139,234,496}},
		{{221,162,3906},{221,162,3904},{203,34,500},{147,234,496}},
		{{230,162,3906},{230,162,3904},{212,34,500},{156,234,496}},
		{{238,162,3906},{174,106,4008},{220,34,500},{164,234,496}},
		{{240,178,4001},{182,106,4008},{228,34,500},{172,234,496}},
		{{166,108,4085},{115,31,4080},{166,108,501},{115,31,496}},
		{{1,68,3328},{1,68,3328},{0,1,384},{0,1,384}},
		{{1,51,3968},{1,51,3968},{0,2,384},{0,2,384}},
		{{21,18,3851},{21,18,3848},{1,50,488},{1,50,488}},
		{{26,195,3851},{29,18,3848},{0,67,488},{0,67,488}},
		{{35,195,3851},{38,18,3848},{12,115,488},{0,3,496}},
		{{43,195,3851},{46,18,3848},{20,115,488},{2,6,424}},
		{{51,195,3851},{54,18,3848},{36,66,482},{4,22,424}},
		{{59,195,3851},{62,18,3848},{44,66,482},{3,73,424}},
		{{68,195,3851},{71,18,3848},{53,66,482},{3,22,496}},
		{{76,195,3851},{79,18,3848},{61,66,482},{2,137,496}},
		{{84,195,3851},{87,18,3848},{69,66,482},{1,89,496}},
		{{92,195,3851},{95,18,3848},{77,66,482},{9,89,496}},
		{{101,195,3851},{104,18,3848},{86,66,482},{18,89,496}},
		{{109,195,3851},{112,18,3848},{94,66,482},{26,89,496}},
		{{117,195,3851},{120,18,3848},{102,66,482},{34,89,496}},
		{{125,195,3851},{128,18,3848},{110,66,482},{42,89,496}},
		{{134,195,3851},{137,18,3848},{119,66,482},{51,89,496}},
		{{141,195,3907},{145,18,3848},{127,66,482},{59,89,496}},
		{{149,195,3907},{153,18,3848},{135,66,482},{67,89,496}},
		{{157,195,3907},{161,18,3848},{143,66,482},{75,89,496}},
		{{166,195,3907},{170,18,3848},{152,66,482},{84,89,496}},
		{{174,195,3907},{178,18,3848},{160,66,482},{92,89,496}},
		{{182,195,3907},{186,18,3848},{168,66,482},{100,89,496}},
		{{190,195,3907},{194,18,3848},{176,66,482},{108,89,496}},
		{{199,195,3907},{203,18,3848},{185,66,482},{117,89,496}},
		{{207,195,3907},{211,18,3848},{193,66,482},{125,89,496}},
		{{215,195,3907},{219,18,3848},{201,66,482},{133,89,496}},
		{{223,195,3907},{227,18,3848},{209,66,482},{141,89,496}},
		{{232,195,3907},{168,89,4008},{218,66,482},{150,89,496}},
		{{236,18,3907},{176,89,4008},{226,66,482},{158,89,496}},
		{{158,90,4085},{103,31,4080},{158,90,501},{103,31,496}},
		{{166,90,4085},{111,31,4080},{166,90,501},{111,31,496}},
		{{0,70,3328},{0,70,3328},{0,17,448},{0,17,448}},
		{{0,117,3904},{0,117,3904},{0,35,384},{0,35,384}},
		{{13,165,3905},{13,165,3904},{2,211,480},{2,211,480}},
		{{21,165,3906},{21,165,3904},{1,51,488},{1,51,488}},
		{{30,165,3906},{30,165,3904},{7,61,352},{7,61,352}},
		{{38,165,3906},{38,165,3904},{2,125,352},{2,125,352}},
		{{46,165,3906},{46,165,3904},{1,37,500},{10,125,352}},
		{{54,165,3906},{54,165,3904},{9,37,500},{5,61,424}},
		{{63,165,3906},{63,165,3904},{18,37,500},{1,189,424}},
		{{71,165,3906},{71,165,3904},{26,37,500},{9,189,424}},
		{{79,165,3906},{79,165,3904},{34,37,500},{4,77,424}},
		{{87,165,3906},{87,165,3904},{42,37,500},{12,77,424}},
		{{96,165,3906},{96,165,3904},{51,37,500},{8,93,424}},
		{{104,165,3906},{104,165,3904},{59,37,500},{3,141,496}},
		{{112,165,3906},{112,165,3904},{68,37,500},{11,141,496}},
		{{120,165,3906},{120,165,3904},{76,37,500},{6,93,496}},
		{{129,165,3906},{129,165,3904},{85,37,500},{15,93,496}},
		{{70,254,4012},{137,165,3904},{93,37,500},{23,93,496}},
		{{145,165,3906},{145,165,3904},{101,37,500},{31,93,496}},
		{{86,254,4012},{153,165,3904},{109,37,500},{39,93,496}},
		{{163,165,3906},{162,165,3904},{118,37,500},{48,93,496}},
		{{171,165,3906},{170,165,3904},{126,37,500},{56,93,496}},
		{{179,165,3906},{178,165,3904},{134,37,500},{64,93,496}},
		{{187,165,3906},{187,165,3904},{142,37,500},{72,93,496}},
		{{196,165,3906},{196,165,3904},{151,37,500},{81,93,496}},
		{{204,165,3906},{204,165,3904},{159,37,500},{89,93,496}},
		{{212,165,3906},{136,77,4008},{167,37,500},{97,93,496}},
		{{220,165,3906},{131,93,4008},{175,37,500},{105,93,496}},
		{{214,181,4001},{140,93,4008},{184,37,500},{114,93,496}},
		{{222,181,4001},{148,93,4008},{192,37,500},{122,93,496}},
		{{115,95,4085},{99,31,4080},{115,95,501},{99,31,496}},
		{{123,95,4085},{107,31,4080},{123,95,501},{107,31,496}},
		{{0,102,3840},{0,102,3840},{0,18,384},{0,18,384}},
		{{5,167,3904},{5,167,3904},{0,13,256},{0,13,256}},
		{{4,54,3968},{4,54,3968},{1,67,448},{1,67,448}},
		{{30,198,3850},{30,198,3848},{0,3,480},{0,3,480}},
		{{39,198,3850},{39,198,3848},{3,52,488},{3,52,488}},
		{{47,198,3851},{47,198,3848},{3,4,488},{3,4,488}},
		{{55,198,3851},{55,198,3848},{1,70,488},{1,70,488}},
		{{53,167,3906},{63,198,3848},{3,22,488},{3,22,488}},
		{{62,167,3906},{72,198,3848},{24,118,488},{0,6,496}},
		{{70,167,3906},{80,198,3848},{32,118,488},{2,89,488}},
		{{78,167,3906},{88,198,3848},{40,118,488},{1,73,496}},
		{{86,167,3906},{96,198,3848},{48,118,488},{0,28,424}},
		{{95,167,3906},{105,198,3848},{57,118,488},{9,28,424}},
		{{103,167,3906},{113,198,3848},{65,118,488},{5,108,496}},
		{{111,167,3906},{121,198,3848},{73,118,488},{13,108,496}},
		{{119,167,3906},{129,198,3848},{81,118,488},{21,108,496}},
		{{128,167,3906},{138,198,3848},{90,118,488},{6,28,496}},
		{{136,167,3906},{146,198,3848},{98,118,488},{14,28,496}},
		{{145,167,3906},{154,198,3848},{106,118,488},{22,28,496}},
		{{153,167,3906},{162,198,3848},{114,118,488},{30,28,496}},
		{{162,167,3906},{171,198,3848},{123,118,488},{39,28,496}},
		{{170,167,3906},{179,198,3848},{131,118,488},{47,28,496}},
		{{178,167,3906},{187,198,3848},{139,118,488},{55,28,496}},
		{{186,167,3906},{195,198,3848},{147,118,488},{63,28,496}},
		{{194,167,3906},{120,12,4008},{156,118,488},{72,28,496}},
		{{206,198,3907},{116,28,4008},{164,118,488},{80,28,496}},
		{{214,198,3907},{124,28,4008},{172,118,488},{88,28,496}},
		{{222,198,3395},{132,28,4008},{180,118,488},{96,28,496}},
		{{207,134,4001},{141,28,4008},{189,118,488},{105,28,496}},
		{{95,30,4085},{86,31,4080},{95,30,501},{86,31,496}},
		{{103,30,4085},{94,31,4080},{103,30,501},{94,31,496}},
		{{111,30,4085},{102,31,4080},{111,30,501},{102,31,496}},
		{{0,104,3840},{0,104,3840},{0,18,448},{0,18,448}},
		{{4,39,3904},{4,39,3904},{0,4,384},{0,4,384}},
		{{0,56,3968},{0,56,3968},{0,84,448},{0,84,448}},
		{{6,110,3328},{6,110,3328},{0,20,448},{0,20,448}},
		{{41,200,3850},{41,200,3848},{1,4,480},{1,4,480}},
		{{49,200,3850},{49,200,3848},{1,8,416},{1,8,416}},
		{{57,200,3851},{57,200,3848},{1,38,488},{1,38,488}},
		{{65,200,3851},{65,200,3848},{1,120,488},{1,120,488}},
		{{74,200,3851},{74,200,3848},{2,72,488},{2,72,488}},
		{{68,6,3907},{82,200,3848},{2,24,488},{2,24,488}},
		{{77,6,3907},{90,200,3848},{26,120,488},{10,24,488}},
		{{97,63,3330},{98,200,3848},{34,120,488},{2,8,496}},
		{{106,63,3330},{107,200,3848},{43,120,488},{3,92,488}},
		{{114,63,3330},{115,200,3848},{51,120,488},{11,92,488}},
		{{122,63,3330},{123,200,3848},{59,120,488},{7,76,496}},
		{{130,63,3330},{131,200,3848},{67,120,488},{15,76,496}},
		{{139,63,3330},{140,200,3848},{76,120,488},{24,76,496}},
		{{147,63,3330},{148,200,3848},{84,120,488},{32,76,496}},
		{{155,63,3330},{156,200,3848},{92,120,488},{40,76,496}},
		{{164,63,3330},{164,200,3848},{100,120,488},{48,76,496}},
		{{173,63,3330},{173,200,3848},{109,120,488},{57,76,496}},
		{{184,6,3851},{181,200,3848},{117,120,488},{65,76,496}},
		{{192,6,3851},{133,28,3936},{125,120,488},{73,76,496}},
		{{189,200,3907},{141,28,3936},{133,120,488},{81,76,496}},
		{{198,200,3907},{138,108,4000},{142,120,488},{90,76,496}},
		{{206,200,3907},{146,108,4000},{150,120,488},{98,76,496}},
		{{214,200,3395},{154,108,4000},{158,120,488},{106,76,496}},
		{{190,136,4001},{162,108,4000},{166,120,488},{114,76,496}},
		{{123,30,4076},{87,15,4080},{123,30,492},{87,15,496}},
		{{117,110,4084},{80,31,4080},{117,110,500},{80,31,496}},
		{{125,110,4084},{88,31,4080},{125,110,500},{88,31,496}},
		{{133,110,4084},{96,31,4080},{133,110,500},{96,31,496}},
		{{9,56,3904},{9,56,3904},{0,67,448},{0,67,448}},
		{{1,8,3904},{1,8,3904},{1,84,448},{1,84,448}},
		{{1,124,3904},{1,124,3904},{0,39,384},{0,39,384}},
		{{9,124,3904},{9,124,3904},{1,4,448},{1,4,448}},
		{{6,76,3904},{6,76,3904},{0,70,448},{0,70,448}},
		{{62,6,3859},{62,6,3856},{2,38,480},{2,38,480}},
		{{70,6,3859},{70,6,3856},{5,43,416},{5,43,416}},
		{{78,6,3859},{78,6,3856},{2,11,416},{2,11,416}},
		{{87,6,3859},{87,6,3856},{0,171,488},{0,171,488}},
		{{67,8,3906},{95,6,3856},{8,171,488},{8,171,488}},
		{{75,8,3907},{103,6,3856},{5,123,488},{5,123,488}},
		{{83,8,3907},{111,6,3856},{2,75,488},{2,75,488}},
		{{92,8,3907},{120,6,3856},{0,27,488},{0,27,488}},
		{{100,8,3907},{128,6,3856},{8,27,488},{8,27,488}},
		{{120,106,3843},{136,6,3856},{99,6,387},{16,27,488}},
		{{128,106,3843},{144,6,3856},{107,6,387},{2,11,496}},
		{{137,106,3843},{153,6,3856},{117,6,387},{11,11,496}},
		{{145,106,3843},{161,6,3856},{125,6,387},{19,11,496}},
		{{163,8,3851},{137,43,3904},{133,6,387},{27,11,496}},
		{{171,8,3851},{145,43,3904},{141,6,387},{35,11,496}},
		{{180,8,3851},{110,11,4000},{150,6,387},{44,11,496}},
		{{188,8,3851},{118,11,4000},{158,6,387},{52,11,496}},
		{{172,72,3907},{126,11,4000},{166,6,387},{60,11,496}},
		{{174,6,3971},{134,11,4000},{174,6,387},{68,11,496}},
		{{183,6,3971},{143,11,4000},{183,6,387},{77,11,496}},
		{{191,6,3971},{151,11,4000},{191,6,387},{85,11,496}},
		{{199,6,3971},{159,11,4000},{199,6,387},{93,11,496}},
		{{92,12,4084},{69,15,4080},{92,12,500},{69,15,496}},
		{{101,12,4084},{78,15,4080},{101,12,500},{78,15,496}},
		{{110,12,4084},{86,15,4080},{110,12,500},{86,15,496}},
		{{118,12,4084},{79,31,4080},{118,12,500},{79,31,496}},
		{{126,12,4084},{87,31,4080},{126,12,500},{87,31,496}},
		{{71,8,3602},{71,8,3600},{2,21,384},{2,21,384}},
		{{79,8,3611},{79,8,3608},{0,69,448},{0,69,448}},
		{{87,8,3611},{87,8,3608},{0,23,384},{0,23,384}},
		{{95,8,3611},{95,8,3608},{1,5,448},{1,5,448}},
		{{104,8,3611},{104,8,3608},{0,88,448},{0,88,448}},
		{{112,8,3611},{112,8,3608},{0,72,448},{0,72,448}},
		{{120,8,3611},{121,8,3608},{36,21,458},{36,21,456}},
		{{133,47,3091},{129,8,3608},{44,21,458},{44,21,456}},
		{{142,47,3091},{138,8,3608},{53,21,459},{53,21,456}},
		{{98,12,3850},{98,12,3848},{61,21,459},{61,21,456}},
		{{106,12,3850},{106,12,3848},{10,92,480},{69,21,456}},
		{{114,12,3851},{114,12,3848},{18,92,480},{77,21,456}},
		{{123,12,3851},{123,12,3848},{3,44,488},{86,21,456}},
		{{95,12,3906},{95,12,3904},{11,44,488},{94,21,456}},
		{{103,12,3906},{103,12,3904},{19,44,488},{102,21,456}},
		{{111,12,3907},{111,12,3904},{27,44,489},{110,21,456}},
		{{120,12,3907},{120,12,3904},{36,44,489},{119,21,456}},
		{{128,12,3907},{128,12,3904},{44,44,489},{127,21,456}},
		{{136,12,3907},{136,12,3904},{52,44,489},{135,21,456}},
		{{144,12,3907},{144,12,3904},{60,44,490},{144,21,456}},
		{{153,12,3907},{153,12,3904},{69,44,490},{153,21,456}},
		{{161,12,3395},{149,188,3968},{77,44,490},{161,21,456}},
		{{169,12,3395},{199,21,3928},{85,44,490},{169,21,456}},
		{{113,95,4001},{202,69,3992},{125,8,483},{177,21,456}},
		{{122,95,4001},{201,21,3984},{134,8,483},{186,21,456}},
		{{143,8,4067},{209,21,3984},{142,8,483},{194,21,456}},
		{{151,8,4067},{47,15,4080},{151,8,483},{47,15,496}},
		{{159,8,4067},{55,15,4080},{159,8,483},{55,15,496}},
		{{168,8,4067},{64,15,4080},{168,8,483},{64,15,496}},
		{{160,40,4075},{72,15,4080},{160,40,491},{72,15,496}},
		{{168,40,4075},{80,15,4080},{168,40,491},{80,15,496}},
		{{144,8,4082},{88,15,4080},{144,8,498},{88,15,496}},
	};

	static void convert_etc1s_to_etc2_eac_r11(eac_block* pDst_block, const endpoint* pEndpoints, const selector* pSelector)
	{
		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		const color32& base_color = pEndpoints->m_color5;
		const uint32_t inten_table = pEndpoints->m_inten5;

		if (low_selector == high_selector)
		{
			uint32_t r;
			decoder_etc_block::get_block_color5_r(base_color, inten_table, low_selector, r);

			// Constant alpha block
			// Select table 13, use selector 4 (0), set multiplier to 1 and base color r
			pDst_block->m_base = r;
			pDst_block->m_table = 13;
			pDst_block->m_multiplier = 1;

			// selectors are all 4's
			static const uint8_t s_etc2_eac_r11_sel4[6] = { 0x92, 0x49, 0x24, 0x92, 0x49, 0x24 };
			memcpy(pDst_block->m_selectors, s_etc2_eac_r11_sel4, sizeof(s_etc2_eac_r11_sel4));

			return;
		}

		uint32_t selector_range_table = 0;
		for (selector_range_table = 0; selector_range_table < NUM_ETC2_EAC_SELECTOR_RANGES; selector_range_table++)
			if ((low_selector == s_etc2_eac_selector_ranges[selector_range_table].m_low) && (high_selector == s_etc2_eac_selector_ranges[selector_range_table].m_high))
				break;
		if (selector_range_table >= NUM_ETC2_EAC_SELECTOR_RANGES)
			selector_range_table = 0;

		const etc1_g_to_eac_conversion* pTable_entry = &s_etc1_g_to_etc2_r11[base_color.r + inten_table * 32][selector_range_table];

		pDst_block->m_base = pTable_entry->m_base;
		pDst_block->m_table = pTable_entry->m_table_mul >> 4;
		pDst_block->m_multiplier = pTable_entry->m_table_mul & 15;

		uint64_t selector_bits = 0;

		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
			{
				uint32_t s = pSelector->get_selector(x, y);

				uint32_t ds = (pTable_entry->m_trans >> (s * 3)) & 7;

				const uint32_t dst_ofs = 45 - (y + x * 4) * 3;
				selector_bits |= (static_cast<uint64_t>(ds) << dst_ofs);
			}
		}

		pDst_block->set_selector_bits(selector_bits);
	}
#endif // BASISD_SUPPORT_ETC2_EAC_RG11

// ASTC
	struct etc1_to_astc_solution
	{
		uint8_t m_lo;
		uint8_t m_hi;
		uint16_t m_err;
	};

#if BASISD_SUPPORT_ASTC
	static dxt_selector_range g_etc1_to_astc_selector_ranges[] =
	{
		{ 0, 3 },

		{ 1, 3 },
		{ 0, 2 },

		{ 1, 2 },

		{ 2, 3 },
		{ 0, 1 },
	};

	const uint32_t NUM_ETC1_TO_ASTC_SELECTOR_RANGES = sizeof(g_etc1_to_astc_selector_ranges) / sizeof(g_etc1_to_astc_selector_ranges[0]);

	static uint32_t g_etc1_to_astc_selector_range_index[4][4];

	const uint32_t NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS = 10;
	static const uint8_t g_etc1_to_astc_selector_mappings[NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS][4] =
	{
		{ 0, 0, 1, 1 },
		{ 0, 0, 1, 2 },
		{ 0, 0, 1, 3 },
		{ 0, 0, 2, 3 },
		{ 0, 1, 1, 1 },
		{ 0, 1, 2, 2 },
		{ 0, 1, 2, 3 },
		{ 0, 2, 3, 3 },
		{ 1, 2, 2, 2 },
		{ 1, 2, 3, 3 },
	};

	static const etc1_to_astc_solution g_etc1_to_astc[32 * 8 * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS * NUM_ETC1_TO_ASTC_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_astc.inc"
	};

	// The best selector mapping to use given a base base+inten table and used selector range for converting grayscale data.
	static uint8_t g_etc1_to_astc_best_grayscale_mapping[32][8][NUM_ETC1_TO_ASTC_SELECTOR_RANGES];
	
#if BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY
	static const etc1_to_astc_solution g_etc1_to_astc_0_255[32 * 8 * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS * NUM_ETC1_TO_ASTC_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_astc_0_255.inc"
	};
	static uint8_t g_etc1_to_astc_best_grayscale_mapping_0_255[32][8][NUM_ETC1_TO_ASTC_SELECTOR_RANGES];
#endif

	static uint32_t g_ise_to_unquant[48];

#if BASISD_WRITE_NEW_ASTC_TABLES
	static void create_etc1_to_astc_conversion_table_0_47()
	{
		FILE* pFile = nullptr;
		fopen_s(&pFile, "basisu_transcoder_tables_astc.inc", "w");

		uint32_t n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1_TO_ASTC_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1_to_astc_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1_to_astc_selector_ranges[sr].m_high;

					uint32_t mapping_best_low[NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
					uint32_t mapping_best_high[NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
					uint64_t mapping_best_err[NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
					uint64_t highest_best_err = 0;

					for (uint32_t m = 0; m < NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 47; hi++)
						{
							for (uint32_t lo = 0; lo <= 47; lo++)
							{
								uint32_t colors[4];

								for (uint32_t s = 0; s < 4; s++)
								{
									uint32_t s_scaled = s | (s << 2) | (s << 4);
									if (s_scaled > 32)
										s_scaled++;

									uint32_t c0 = g_ise_to_unquant[lo] | (g_ise_to_unquant[lo] << 8);
									uint32_t c1 = g_ise_to_unquant[hi] | (g_ise_to_unquant[hi] << 8);
									colors[s] = ((c0 * (64 - s_scaled) + c1 * s_scaled + 32) / 64) >> 8;
								}

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1_to_astc_selector_mappings[m][s]];

									int err_scale = 1;
									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										err_scale = 8;

									total_err += (err * err) * err_scale;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						mapping_best_low[m] = best_lo;
						mapping_best_high[m] = best_hi;
						mapping_best_err[m] = best_err;
						highest_best_err = basisu::maximum(highest_best_err, best_err);
												
					} // m

					for (uint32_t m = 0; m < NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS; m++)
					{
						uint64_t err = mapping_best_err[m];

						err = basisu::minimum<uint64_t>(err, 0xFFFF);

						fprintf(pFile, "{%u,%u,%u},", mapping_best_low[m], mapping_best_high[m], (uint32_t)err);

						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m

				} // sr
			} // g
		} // inten

		fclose(pFile);
	}

	static void create_etc1_to_astc_conversion_table_0_255()
	{
		FILE* pFile = nullptr;
		fopen_s(&pFile, "basisu_transcoder_tables_astc_0_255.inc", "w");

		uint32_t n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1_TO_ASTC_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1_to_astc_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1_to_astc_selector_ranges[sr].m_high;

					uint32_t mapping_best_low[NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
					uint32_t mapping_best_high[NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
					uint64_t mapping_best_err[NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
					uint64_t highest_best_err = 0;

					for (uint32_t m = 0; m < NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 255; hi++)
						{
							for (uint32_t lo = 0; lo <= 255; lo++)
							{
								uint32_t colors[4];

								for (uint32_t s = 0; s < 4; s++)
								{
									uint32_t s_scaled = s | (s << 2) | (s << 4);
									if (s_scaled > 32)
										s_scaled++;

									uint32_t c0 = lo | (lo << 8);
									uint32_t c1 = hi | (hi << 8);
									colors[s] = ((c0 * (64 - s_scaled) + c1 * s_scaled + 32) / 64) >> 8;
								}

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1_to_astc_selector_mappings[m][s]];

									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									int err_scale = 1;
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										err_scale = 8;

									total_err += (err * err) * err_scale;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						mapping_best_low[m] = best_lo;
						mapping_best_high[m] = best_hi;
						mapping_best_err[m] = best_err;
						highest_best_err = basisu::maximum(highest_best_err, best_err);
					} // m

					for (uint32_t m = 0; m < NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS; m++)
					{
						uint64_t err = mapping_best_err[m];

						err = basisu::minimum<uint64_t>(err, 0xFFFF);
						
						fprintf(pFile, "{%u,%u,%u},", mapping_best_low[m], mapping_best_high[m], (uint32_t)err);
						
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m

				} // sr
			} // g
		} // inten

		fclose(pFile);
	}
#endif

#endif

#if BASISD_SUPPORT_ASTC
	struct astc_block_params
	{
		// 2 groups of 5, but only a max of 8 are used (RRGGBBAA00)
		uint8_t m_endpoints[10]; 
		uint8_t m_weights[32];
	};
		
	// Table encodes 5 trits to 8 output bits. 3^5 entries.
	// Inverse of the trit bit manipulation process in https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#astc-integer-sequence-encoding
	static const uint8_t g_astc_trit_encode[243] = { 0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26, 3, 7, 11, 19, 23, 27, 12, 13, 14, 32, 33, 34, 36, 37, 38, 40, 41, 42, 48, 49, 50, 52, 53, 54, 56, 57, 58, 35, 39, 
		43, 51, 55, 59, 44, 45, 46, 64, 65, 66, 68, 69, 70, 72, 73, 74, 80, 81, 82, 84, 85, 86, 88, 89, 90, 67, 71, 75, 83, 87, 91, 76, 77, 78, 128, 129, 130, 132, 133, 134, 136, 137, 138, 144, 145, 146, 148, 149, 150, 152, 153, 154, 
		131, 135, 139, 147, 151, 155, 140, 141, 142, 160, 161, 162, 164, 165, 166, 168, 169, 170, 176, 177, 178, 180, 181, 182, 184, 185, 186, 163, 167, 171, 179, 183, 187, 172, 173, 174, 192, 193, 194, 196, 197, 198, 200, 201, 202, 
		208, 209, 210, 212, 213, 214, 216, 217, 218, 195, 199, 203, 211, 215, 219, 204, 205, 206, 96, 97, 98, 100, 101, 102, 104, 105, 106, 112, 113, 114, 116, 117, 118, 120, 121, 122, 99, 103, 107, 115, 119, 123, 108, 109, 110, 224, 
		225, 226, 228, 229, 230, 232, 233, 234, 240, 241, 242, 244, 245, 246, 248, 249, 250, 227, 231, 235, 243, 247, 251, 236, 237, 238, 28, 29, 30, 60, 61, 62, 92, 93, 94, 156, 157, 158, 188, 189, 190, 220, 221, 222, 31, 63, 95, 159, 
		191, 223, 124, 125, 126 };

	// Writes bits to output in an endian safe way
	static inline void astc_set_bits(uint32_t *pOutput, int &bit_pos, uint32_t value, int total_bits)
	{
		uint8_t* pBytes = reinterpret_cast<uint8_t*>(pOutput);
		
		while (total_bits)
		{
			const uint32_t bits_to_write = std::min(total_bits, 8 - (bit_pos & 7));

			pBytes[bit_pos >> 3] |= static_cast<uint8_t>(value << (bit_pos & 7));

			bit_pos += bits_to_write;
			total_bits -= bits_to_write;
			value >>= bits_to_write;
		}
	}

	// Extracts bits [low,high]
	static inline uint32_t astc_extract_bits(uint32_t bits, int low, int high)
	{
		return (bits >> low) & ((1 << (high - low + 1)) - 1);
	}

	// Encodes 5 values to output, usable for any range that uses trits and bits
	static void astc_encode_trits(uint32_t *pOutput, const uint8_t *pValues, int& bit_pos, int n)
	{
		// First extract the trits and the bits from the 5 input values
		int trits = 0, bits[5];
		const uint32_t bit_mask = (1 << n) - 1;
		for (int i = 0; i < 5; i++)
		{
			static const int s_muls[5] = { 1, 3, 9, 27, 81 };
			
			const int t = pValues[i] >> n;
			
			trits += t * s_muls[i];
			bits[i] = pValues[i] & bit_mask;
		}

		// Encode the trits, by inverting the bit manipulations done by the decoder, converting 5 trits into 8-bits.
		// See https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#astc-integer-sequence-encoding

		assert(trits < 243);
		const int T = g_astc_trit_encode[trits];
		
		// Now interleave the 8 encoded trit bits with the bits to form the encoded output. See table 94.
		astc_set_bits(pOutput, bit_pos, bits[0] | (astc_extract_bits(T, 0, 1) << n) | (bits[1] << (2 + n)), n * 2 + 2);

		astc_set_bits(pOutput, bit_pos, astc_extract_bits(T, 2, 3) | (bits[2] << 2) | (astc_extract_bits(T, 4, 4) << (2 + n)) | (bits[3] << (3 + n)) | (astc_extract_bits(T, 5, 6) << (3 + n * 2)) | 
			(bits[4] << (5 + n * 2)) | (astc_extract_bits(T, 7, 7) << (5 + n * 3)), n * 3 + 6);
	}

	// Packs a single format ASTC block using Color Endpoint Mode 12 (LDR RGBA direct), endpoint BISE range 13, 2-bit weights (range 2). 
	// We're always going to output blocks containing alpha, even if the input doesn't have alpha, for simplicity.
	// Each block always has 4x4 weights, uses range 13 BISE encoding on the endpoints (0-47), and each weight ranges from 0-3. This encoding should be roughly equal in quality vs. BC1 for color.
	// 8 total endpoints, stored as RGBA LH LH LH LH order, each ranging from 0-47. 
	// Note the input [0,47] endpoint values are not linear - they are encoded as outlined in the ASTC spec:
	// https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#astc-endpoint-unquantization
	// 32 total weights, stored as 16 CA CA, each ranging from 0-3.
	static void astc_pack_block_cem_12_weight_range2(uint32_t *pOutput, const astc_block_params* pBlock)
	{
		uint8_t* pBytes = reinterpret_cast<uint8_t*>(pOutput);

		// Write constant block mode, color component selector, number of partitions, color endpoint mode
		// https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#_block_mode
		pBytes[0] = 0x42; pBytes[1] = 0x84; pBytes[2] = 0x01; pBytes[3] = 0x00;
		pBytes[4] = 0x00; pBytes[5] = 0x00; pBytes[6] = 0x00; pBytes[7] = 0xc0;

		pOutput[2] = 0;
		pOutput[3] = 0;

		// Pack 8 endpoints (each ranging between [0,47]) using BISE starting at bit 17
		int bit_pos = 17;
		astc_encode_trits(pOutput, pBlock->m_endpoints, bit_pos, 4);
		astc_encode_trits(pOutput, pBlock->m_endpoints + 5, bit_pos, 4);

		// Pack 32 2-bit weights, which are stored from the top down into the block in opposite bit order.
				
		for (uint32_t i = 0; i < 32; i++)
		{
			static const uint8_t s_reverse_bits[4] = { 0, 2, 1, 3 };
			const uint32_t ofs = 126 - (i * 2);
			pBytes[ofs >> 3] |= (s_reverse_bits[pBlock->m_weights[i]] << (ofs & 7));
		}
	}

	// CEM mode 12 (LDR RGBA Direct), 8-bit endpoints, 1-bit weights 
	// This ASTC mode is basically block truncation coding (BTC) using 1-bit weights and 8-bit/component endpoints - very convenient.
	static void astc_pack_block_cem_12_weight_range0(uint32_t* pOutput, const astc_block_params* pBlock)
	{
		uint8_t* pBytes = reinterpret_cast<uint8_t*>(pOutput);

		// Write constant block mode, color component selector, number of partitions, color endpoint mode
		// https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#_block_mode
		pBytes[0] = 0x41; pBytes[1] = 0x84; pBytes[2] = 0x01; pBytes[3] = 0x00;
		pOutput[1] = 0;
		pBytes[8] = 0x00; pBytes[9] = 0x00; pBytes[10] = 0x00; pBytes[11] = 0xc0;
		pOutput[3] = 0;

		// Pack 8 endpoints (each ranging between [0,255]) as 8-bits starting at bit 17
		int bit_pos = 17;
		for (uint32_t i = 0; i < 8; i++)
			astc_set_bits(pOutput, bit_pos, pBlock->m_endpoints[i], 8);

		// Pack 32 1-bit weights, which are stored from the top down into the block in opposite bit order.
		for (uint32_t i = 0; i < 32; i++)
		{
			const uint32_t ofs = 127 - i;
			pBytes[ofs >> 3] |= (pBlock->m_weights[i] << (ofs & 7));
		}
	}

#if BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY
	// Optional 8-bit endpoint packing functions.

	// CEM mode 4 (LDR Luminance+Alpha Direct), 8-bit endpoints, 2 bit weights
	static void astc_pack_block_cem_4_weight_range2(uint32_t* pOutput, const astc_block_params* pBlock)
	{
		uint8_t* pBytes = reinterpret_cast<uint8_t*>(pOutput);

		// Write constant block mode, color component selector, number of partitions, color endpoint mode
		// https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#_block_mode
		pBytes[0] = 0x42; pBytes[1] = 0x84; pBytes[2] = 0x00; pBytes[3] = 0x00;
		pBytes[4] = 0x00; pBytes[5] = 0x00; pBytes[6] = 0x00; pBytes[7] = 0xc0;
		
		pOutput[2] = 0;
		pOutput[3] = 0;

		// Pack 4 endpoints (each ranging between [0,255]) as 8-bits starting at bit 17
		int bit_pos = 17;
		for (uint32_t i = 0; i < 4; i++)
			astc_set_bits(pOutput, bit_pos, pBlock->m_endpoints[i], 8);

		// Pack 32 2-bit weights, which are stored from the top down into the block in opposite bit order.
		for (uint32_t i = 0; i < 32; i++)
		{
			static const uint8_t s_reverse_bits[4] = { 0, 2, 1, 3 };
			const uint32_t ofs = 126 - (i * 2);
			pBytes[ofs >> 3] |= (s_reverse_bits[pBlock->m_weights[i]] << (ofs & 7));
		}
	}

	// CEM mode 8 (LDR RGB Direct), 8-bit endpoints, 2 bit weights
	static void astc_pack_block_cem_8_weight_range2(uint32_t* pOutput, const astc_block_params* pBlock)
	{
		uint8_t* pBytes = reinterpret_cast<uint8_t*>(pOutput);

		// Write constant block mode, color component selector, number of partitions, color endpoint mode
		// https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#_block_mode
		pBytes[0] = 0x42; pBytes[1] = 0x00; pBytes[2] = 0x01; pBytes[3] = 0x00;
		
		pOutput[1] = 0;
		pOutput[2] = 0;
		pOutput[3] = 0;

		// Pack 6 endpoints (each ranging between [0,255]) as 8-bits starting at bit 17
		int bit_pos = 17;
		for (uint32_t i = 0; i < 6; i++)
			astc_set_bits(pOutput, bit_pos, pBlock->m_endpoints[i], 8);

		// Pack 16 2-bit weights, which are stored from the top down into the block in opposite bit order.
		for (uint32_t i = 0; i < 16; i++)
		{
			static const uint8_t s_reverse_bits[4] = { 0, 2, 1, 3 };
			const uint32_t ofs = 126 - (i * 2);
			pBytes[ofs >> 3] |= (s_reverse_bits[pBlock->m_weights[i]] << (ofs & 7));
		}
	}
#endif

	// Optimal quantized [0,47] entry to use given [0,255] input
	static uint8_t g_astc_single_color_encoding_0[256];

	// Optimal quantized [0,47] low/high values given [0,255] input assuming a selector of 1
	static struct
	{
		uint8_t m_lo, m_hi;
	} g_astc_single_color_encoding_1[256];
		
	static void transcoder_init_astc()
	{
		for (uint32_t base_color = 0; base_color < 32; base_color++)
		{
			for (uint32_t inten_table = 0; inten_table < 8; inten_table++)
			{
				for (uint32_t range_index = 0; range_index < NUM_ETC1_TO_ASTC_SELECTOR_RANGES; range_index++)
				{
					const etc1_to_astc_solution* pTable_g = &g_etc1_to_astc[(inten_table * 32 + base_color) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + range_index * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];

					uint32_t best_mapping = 0;
					uint32_t best_err = UINT32_MAX;
					for (uint32_t mapping_index = 0; mapping_index < NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS; mapping_index++)
					{
						if (pTable_g[mapping_index].m_err < best_err)
						{
							best_err = pTable_g[mapping_index].m_err;
							best_mapping = mapping_index;
						}
					}

					g_etc1_to_astc_best_grayscale_mapping[base_color][inten_table][range_index] = static_cast<uint8_t>(best_mapping);
				}
			}
		}

#if BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY
		for (uint32_t base_color = 0; base_color < 32; base_color++)
		{
			for (uint32_t inten_table = 0; inten_table < 8; inten_table++)
			{
				for (uint32_t range_index = 0; range_index < NUM_ETC1_TO_ASTC_SELECTOR_RANGES; range_index++)
				{
					const etc1_to_astc_solution* pTable_g = &g_etc1_to_astc_0_255[(inten_table * 32 + base_color) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + range_index * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];

					uint32_t best_mapping = 0;
					uint32_t best_err = UINT32_MAX;
					for (uint32_t mapping_index = 0; mapping_index < NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS; mapping_index++)
					{
						if (pTable_g[mapping_index].m_err < best_err)
						{
							best_err = pTable_g[mapping_index].m_err;
							best_mapping = mapping_index;
						}
					}

					g_etc1_to_astc_best_grayscale_mapping_0_255[base_color][inten_table][range_index] = static_cast<uint8_t>(best_mapping);
				}
			}
		}
#endif

		for (uint32_t i = 0; i < NUM_ETC1_TO_ASTC_SELECTOR_RANGES; i++)
		{
			uint32_t l = g_etc1_to_astc_selector_ranges[i].m_low;
			uint32_t h = g_etc1_to_astc_selector_ranges[i].m_high;
			g_etc1_to_astc_selector_range_index[l][h] = i;
		}

		// Endpoint dequantization, see:
		// https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#astc-endpoint-unquantization
		for (uint32_t trit = 0; trit < 3; trit++)
		{
			for (uint32_t bit = 0; bit < 16; bit++)
			{
				const uint32_t A = (bit & 1) ? 511 : 0;
				const uint32_t B = (bit >> 1) | ((bit >> 1) << 6);
				const uint32_t C = 22;
				const uint32_t D = trit;

				uint32_t unq = D * C + B;
				unq = unq ^ A;
				unq = (A & 0x80) | (unq >> 2);

				g_ise_to_unquant[bit | (trit << 4)] = unq;
			}
		}
				
		// Compute table used for optimal single color encoding.
		for (int i = 0; i < 256; i++)
		{
			int lowest_e = INT_MAX;

			for (int lo = 0; lo < 48; lo++)
			{
				for (int hi = 0; hi < 48; hi++)
				{
					const int lo_v = g_ise_to_unquant[lo];
					const int hi_v = g_ise_to_unquant[hi];

					int l = lo_v | (lo_v << 8);
					int h = hi_v | (hi_v << 8);
										
					int v = ((l * (64 - 21) + (h * 21) + 32) / 64) >> 8;
					
					int e = abs(v - i);

					if (e < lowest_e)
					{
						g_astc_single_color_encoding_1[i].m_hi = static_cast<uint8_t>(hi);
						g_astc_single_color_encoding_1[i].m_lo = static_cast<uint8_t>(lo);

						lowest_e = e;
					}

				} // hi
			} // lo
		}

		for (int i = 0; i < 256; i++)
		{
			int lowest_e = INT_MAX;

			for (int lo = 0; lo < 48; lo++)
			{
				const int lo_v = g_ise_to_unquant[lo];
				
				int e = abs(lo_v - i);

				if (e < lowest_e)
				{
					g_astc_single_color_encoding_0[i] = static_cast<uint8_t>(lo);

					lowest_e = e;
				}
			} // lo
		}
	}

	// Converts opaque or color+alpha ETC1S block to ASTC 4x4.
	// This function tries to use the best ASTC mode given the block's actual contents.
	static void convert_etc1s_to_astc_4x4(void* pDst_block, const endpoint* pEndpoints, const selector* pSelector, 
		bool transcode_alpha, const endpoint *pEndpoint_codebook, const selector *pSelector_codebook)
	{
		astc_block_params blk;

		blk.m_endpoints[8] = 0;
		blk.m_endpoints[9] = 0;

		int constant_alpha_val = 255;
		int num_unique_alpha_selectors = 1;

		if (transcode_alpha)
		{
			const selector& alpha_selectors = pSelector_codebook[((uint16_t*)pDst_block)[1]];

			num_unique_alpha_selectors = alpha_selectors.m_num_unique_selectors;

			if (num_unique_alpha_selectors == 1)
			{
				const endpoint& alpha_endpoint = pEndpoint_codebook[((uint16_t*)pDst_block)[0]];

				const color32& alpha_base_color = alpha_endpoint.m_color5;
				const uint32_t alpha_inten_table = alpha_endpoint.m_inten5;

				int alpha_block_colors[4];
				decoder_etc_block::get_block_colors5_g(alpha_block_colors, alpha_base_color, alpha_inten_table);

				constant_alpha_val = alpha_block_colors[alpha_selectors.m_lo_selector];
			}
		}

		const color32& base_color = pEndpoints->m_color5;
		const uint32_t inten_table = pEndpoints->m_inten5;

		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		// Handle solid color or BTC blocks, which can always be encoded from ETC1S to ASTC losslessly.
		if ((pSelector->m_num_unique_selectors == 1) && (num_unique_alpha_selectors == 1))
		{
			// Both color and alpha are constant, write a solid color block and exit.
			// See https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#astc-void-extent-blocks
			uint32_t r, g, b;
			decoder_etc_block::get_block_color5(base_color, inten_table, low_selector, r, g, b);
						
			uint32_t* pOutput = static_cast<uint32_t*>(pDst_block);
			uint8_t* pBytes = reinterpret_cast<uint8_t*>(pDst_block);

			pBytes[0] = 0xfc; pBytes[1] = 0xfd; pBytes[2] = 0xff; pBytes[3] = 0xff;

			pOutput[1] = 0xffffffff;
			pOutput[2] = 0;
			pOutput[3] = 0;

			int bit_pos = 64;
			astc_set_bits(pOutput, bit_pos, r | (r << 8), 16);
			astc_set_bits(pOutput, bit_pos, g | (g << 8), 16);
			astc_set_bits(pOutput, bit_pos, b | (b << 8), 16);
			astc_set_bits(pOutput, bit_pos, constant_alpha_val | (constant_alpha_val << 8), 16);

			return;
		}
		else if ((pSelector->m_num_unique_selectors <= 2) && (num_unique_alpha_selectors <= 2))
		{
			// Both color and alpha use <= 2 unique selectors each. 
			// Use block truncation coding, which is lossless with ASTC (8-bit endpoints, 1-bit weights).
			color32 block_colors[4];
			decoder_etc_block::get_block_colors5(block_colors, base_color, inten_table);

			blk.m_endpoints[0] = block_colors[low_selector].r;
			blk.m_endpoints[2] = block_colors[low_selector].g;
			blk.m_endpoints[4] = block_colors[low_selector].b;

			blk.m_endpoints[1] = block_colors[high_selector].r;
			blk.m_endpoints[3] = block_colors[high_selector].g;
			blk.m_endpoints[5] = block_colors[high_selector].b;

			int s0 = blk.m_endpoints[0] + blk.m_endpoints[2] + blk.m_endpoints[4];
			int s1 = blk.m_endpoints[1] + blk.m_endpoints[3] + blk.m_endpoints[5];
			bool invert = false;
			if (s1 < s0)
			{
				std::swap(blk.m_endpoints[0], blk.m_endpoints[1]);
				std::swap(blk.m_endpoints[2], blk.m_endpoints[3]);
				std::swap(blk.m_endpoints[4], blk.m_endpoints[5]);
				invert = true;
			}

			if (transcode_alpha)
			{
				const endpoint& alpha_endpoint = pEndpoint_codebook[((uint16_t*)pDst_block)[0]];
				const selector& alpha_selectors = pSelector_codebook[((uint16_t*)pDst_block)[1]];

				const color32& alpha_base_color = alpha_endpoint.m_color5;
				const uint32_t alpha_inten_table = alpha_endpoint.m_inten5;

				const uint32_t alpha_low_selector = alpha_selectors.m_lo_selector;
				const uint32_t alpha_high_selector = alpha_selectors.m_hi_selector;

				int alpha_block_colors[4];
				decoder_etc_block::get_block_colors5_g(alpha_block_colors, alpha_base_color, alpha_inten_table);

				blk.m_endpoints[6] = static_cast<uint8_t>(alpha_block_colors[alpha_low_selector]);
				blk.m_endpoints[7] = static_cast<uint8_t>(alpha_block_colors[alpha_high_selector]);

				for (uint32_t y = 0; y < 4; y++)
				{
					for (uint32_t x = 0; x < 4; x++)
					{
						uint32_t s = alpha_selectors.get_selector(x, y);
						s = (s == alpha_high_selector) ? 1 : 0;
												
						blk.m_weights[(x + y * 4) * 2 + 1] = static_cast<uint8_t>(s);
					} // x
				} // y
			}
			else
			{
				blk.m_endpoints[6] = 255;
				blk.m_endpoints[7] = 255;

				for (uint32_t i = 0; i < 16; i++)
					blk.m_weights[i * 2 + 1] = 0;
			}

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t s = pSelector->get_selector(x, y);

					s = (s == high_selector) ? 1 : 0;

					if (invert)
						s = 1 - s;

					blk.m_weights[(x + y * 4) * 2] = static_cast<uint8_t>(s);
				} // x
			} // y

			astc_pack_block_cem_12_weight_range0(reinterpret_cast<uint32_t*>(pDst_block), &blk);

			return;
		}
				
		// Either alpha and/or color use > 2 unique selectors each, so we must do something more complex.
				
#if BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY
		// The optional higher quality modes use 8-bits endpoints vs. [0,47] endpoints.
		
		// If the block's base color is grayscale, all pixels are grayscale, so encode the block as Luminance+Alpha.
		if ((base_color.r == base_color.g) && (base_color.r == base_color.b))
		{
			if (transcode_alpha)
			{
				const endpoint& alpha_endpoint = pEndpoint_codebook[((uint16_t*)pDst_block)[0]];
				const selector& alpha_selectors = pSelector_codebook[((uint16_t*)pDst_block)[1]];

				const color32& alpha_base_color = alpha_endpoint.m_color5;
				const uint32_t alpha_inten_table = alpha_endpoint.m_inten5;

				const uint32_t alpha_low_selector = alpha_selectors.m_lo_selector;
				const uint32_t alpha_high_selector = alpha_selectors.m_hi_selector;

				if (num_unique_alpha_selectors <= 2)
				{
					// Simple alpha block with only 1 or 2 unique values, so use BTC. This is lossless.
					int alpha_block_colors[4];
					decoder_etc_block::get_block_colors5_g(alpha_block_colors, alpha_base_color, alpha_inten_table);

					blk.m_endpoints[2] = static_cast<uint8_t>(alpha_block_colors[alpha_low_selector]);
					blk.m_endpoints[3] = static_cast<uint8_t>(alpha_block_colors[alpha_high_selector]);

					for (uint32_t i = 0; i < 16; i++)
					{
						uint32_t s = alpha_selectors.get_selector(i & 3, i >> 2);
						blk.m_weights[i * 2 + 1] = (s == alpha_high_selector) ? 3 : 0;
					}
				}
				else
				{
					// Convert ETC1S alpha
					const uint32_t alpha_selector_range_table = g_etc1_to_astc_selector_range_index[alpha_low_selector][alpha_high_selector];
										
					//[32][8][RANGES][MAPPING]
					const etc1_to_astc_solution* pTable_g = &g_etc1_to_astc_0_255[(alpha_inten_table * 32 + alpha_base_color.g) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + alpha_selector_range_table * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];

					const uint32_t best_mapping = g_etc1_to_astc_best_grayscale_mapping_0_255[alpha_base_color.g][alpha_inten_table][alpha_selector_range_table];

					blk.m_endpoints[2] = pTable_g[best_mapping].m_lo;
					blk.m_endpoints[3] = pTable_g[best_mapping].m_hi;
										
					const uint8_t* pSelectors_xlat = &g_etc1_to_astc_selector_mappings[best_mapping][0];

					for (uint32_t y = 0; y < 4; y++)
					{
						for (uint32_t x = 0; x < 4; x++)
						{
							uint32_t s = alpha_selectors.get_selector(x, y);
							uint32_t as = pSelectors_xlat[s];

							blk.m_weights[(x + y * 4) * 2 + 1] = static_cast<uint8_t>(as);
						} // x
					} // y
				}
			}
			else
			{
				// No alpha slice - set output alpha to all 255's
				blk.m_endpoints[2] = 255;
				blk.m_endpoints[3] = 255;

				for (uint32_t i = 0; i < 16; i++)
					blk.m_weights[i * 2 + 1] = 0;
			}

			if (pSelector->m_num_unique_selectors <= 2)
			{
				// Simple color block with only 1 or 2 unique values, so use BTC. This is lossless.
				int block_colors[4];
				decoder_etc_block::get_block_colors5_g(block_colors, base_color, inten_table);

				blk.m_endpoints[0] = static_cast<uint8_t>(block_colors[low_selector]);
				blk.m_endpoints[1] = static_cast<uint8_t>(block_colors[high_selector]);

				for (uint32_t i = 0; i < 16; i++)
				{
					uint32_t s = pSelector->get_selector(i & 3, i >> 2);
					blk.m_weights[i * 2] = (s == high_selector) ? 3 : 0;
				}
			}
			else
			{
				// Convert ETC1S alpha
				const uint32_t selector_range_table = g_etc1_to_astc_selector_range_index[low_selector][high_selector];
								
				//[32][8][RANGES][MAPPING]
				const etc1_to_astc_solution* pTable_g = &g_etc1_to_astc_0_255[(inten_table * 32 + base_color.g) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
								
				const uint32_t best_mapping = g_etc1_to_astc_best_grayscale_mapping_0_255[base_color.g][inten_table][selector_range_table];

				blk.m_endpoints[0] = pTable_g[best_mapping].m_lo;
				blk.m_endpoints[1] = pTable_g[best_mapping].m_hi;

				const uint8_t* pSelectors_xlat = &g_etc1_to_astc_selector_mappings[best_mapping][0];

				for (uint32_t y = 0; y < 4; y++)
				{
					for (uint32_t x = 0; x < 4; x++)
					{
						uint32_t s = pSelector->get_selector(x, y);
						uint32_t as = pSelectors_xlat[s];

						blk.m_weights[(x + y * 4) * 2] = static_cast<uint8_t>(as);
					} // x
				} // y
			}

			astc_pack_block_cem_4_weight_range2(reinterpret_cast<uint32_t*>(pDst_block), &blk);
			return;
		}

		// The block isn't grayscale and it uses > 2 unique selectors for opaque and/or alpha.
		// Check for fully opaque blocks, if so use 8-bit endpoints for slightly higher opaque quality (higher than BC1, but lower than BC7 mode 6 opaque).
		if ((num_unique_alpha_selectors == 1) && (constant_alpha_val == 255))
		{
			// Convert ETC1S color
			const uint32_t selector_range_table = g_etc1_to_astc_selector_range_index[low_selector][high_selector];

			//[32][8][RANGES][MAPPING]
			const etc1_to_astc_solution* pTable_r = &g_etc1_to_astc_0_255[(inten_table * 32 + base_color.r) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
			const etc1_to_astc_solution* pTable_g = &g_etc1_to_astc_0_255[(inten_table * 32 + base_color.g) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
			const etc1_to_astc_solution* pTable_b = &g_etc1_to_astc_0_255[(inten_table * 32 + base_color.b) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];

			uint32_t best_err = UINT_MAX;
			uint32_t best_mapping = 0;

			assert(NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS == 10);
#define DO_ITER(m) { uint32_t total_err = pTable_r[m].m_err + pTable_g[m].m_err + pTable_b[m].m_err; if (total_err < best_err) { best_err = total_err; best_mapping = m; } }
			DO_ITER(0); DO_ITER(1); DO_ITER(2); DO_ITER(3); DO_ITER(4);
			DO_ITER(5); DO_ITER(6); DO_ITER(7); DO_ITER(8); DO_ITER(9);
#undef DO_ITER

			blk.m_endpoints[0] = pTable_r[best_mapping].m_lo;
			blk.m_endpoints[1] = pTable_r[best_mapping].m_hi;

			blk.m_endpoints[2] = pTable_g[best_mapping].m_lo;
			blk.m_endpoints[3] = pTable_g[best_mapping].m_hi;

			blk.m_endpoints[4] = pTable_b[best_mapping].m_lo;
			blk.m_endpoints[5] = pTable_b[best_mapping].m_hi;

			int s0 = blk.m_endpoints[0] + blk.m_endpoints[2] + blk.m_endpoints[4];
			int s1 = blk.m_endpoints[1] + blk.m_endpoints[3] + blk.m_endpoints[5];
			bool invert = false;

			if (s1 < s0)
			{
				std::swap(blk.m_endpoints[0], blk.m_endpoints[1]);
				std::swap(blk.m_endpoints[2], blk.m_endpoints[3]);
				std::swap(blk.m_endpoints[4], blk.m_endpoints[5]);
				invert = true;
			}

			const uint8_t* pSelectors_xlat = &g_etc1_to_astc_selector_mappings[best_mapping][0];

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t s = pSelector->get_selector(x, y);
					uint32_t as = pSelectors_xlat[s];
					if (invert)
						as = 3 - as;

					blk.m_weights[x + y * 4] = static_cast<uint8_t>(as);
				} // x
			} // y

			// Now pack to ASTC
			astc_pack_block_cem_8_weight_range2(reinterpret_cast<uint32_t*>(pDst_block), &blk);
			return;
		}
#endif //#if BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY

		// Nothing else worked, so fall back to CEM Mode 12 (LDR RGBA Direct), [0,47] endpoints, weight range 2 (2-bit weights), dual planes.
		// This mode can handle everything, but at slightly less quality than BC1.
		if (transcode_alpha)
		{
			const endpoint& alpha_endpoint = pEndpoint_codebook[((uint16_t*)pDst_block)[0]];
			const selector& alpha_selectors = pSelector_codebook[((uint16_t*)pDst_block)[1]];

			const color32& alpha_base_color = alpha_endpoint.m_color5;
			const uint32_t alpha_inten_table = alpha_endpoint.m_inten5;

			const uint32_t alpha_low_selector = alpha_selectors.m_lo_selector;
			const uint32_t alpha_high_selector = alpha_selectors.m_hi_selector;

			if (alpha_low_selector == alpha_high_selector)
			{
				// Solid alpha block - use precomputed tables.
				int alpha_block_colors[4];
				decoder_etc_block::get_block_colors5_g(alpha_block_colors, alpha_base_color, alpha_inten_table);

				const uint32_t g = alpha_block_colors[alpha_low_selector];

				blk.m_endpoints[6] = g_astc_single_color_encoding_1[g].m_lo;
				blk.m_endpoints[7] = g_astc_single_color_encoding_1[g].m_hi;

				for (uint32_t i = 0; i < 16; i++)
					blk.m_weights[i * 2 + 1] = 1;
			}
			else if ((alpha_inten_table >= 7) && (alpha_selectors.m_num_unique_selectors == 2) && (alpha_low_selector == 0) && (alpha_high_selector == 3))
			{
				// Handle outlier case where only the two outer colors are used with inten table 7.
				color32 alpha_block_colors[4];

				decoder_etc_block::get_block_colors5(alpha_block_colors, alpha_base_color, alpha_inten_table);

				const uint32_t g0 = alpha_block_colors[0].g;
				const uint32_t g1 = alpha_block_colors[3].g;

				blk.m_endpoints[6] = g_astc_single_color_encoding_0[g0];
				blk.m_endpoints[7] = g_astc_single_color_encoding_0[g1];

				for (uint32_t y = 0; y < 4; y++)
				{
					for (uint32_t x = 0; x < 4; x++)
					{
						uint32_t s = alpha_selectors.get_selector(x, y);
						uint32_t as = (s == alpha_high_selector) ? 3 : 0;

						blk.m_weights[(x + y * 4) * 2 + 1] = static_cast<uint8_t>(as);
					} // x
				} // y
			}
			else
			{
				// Convert ETC1S alpha
				const uint32_t alpha_selector_range_table = g_etc1_to_astc_selector_range_index[alpha_low_selector][alpha_high_selector];
								
				//[32][8][RANGES][MAPPING]
				const etc1_to_astc_solution* pTable_g = &g_etc1_to_astc[(alpha_inten_table * 32 + alpha_base_color.g) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + alpha_selector_range_table * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];

				const uint32_t best_mapping = g_etc1_to_astc_best_grayscale_mapping[alpha_base_color.g][alpha_inten_table][alpha_selector_range_table];

				blk.m_endpoints[6] = pTable_g[best_mapping].m_lo;
				blk.m_endpoints[7] = pTable_g[best_mapping].m_hi;

				const uint8_t* pSelectors_xlat = &g_etc1_to_astc_selector_mappings[best_mapping][0];

				for (uint32_t y = 0; y < 4; y++)
				{
					for (uint32_t x = 0; x < 4; x++)
					{
						uint32_t s = alpha_selectors.get_selector(x, y);
						uint32_t as = pSelectors_xlat[s];

						blk.m_weights[(x + y * 4) * 2 + 1] = static_cast<uint8_t>(as);
					} // x
				} // y
			}
		}
		else
		{
			// No alpha slice - set output alpha to all 255's
			// 1 is 255 when dequantized
			blk.m_endpoints[6] = 1;
			blk.m_endpoints[7] = 1;

			for (uint32_t i = 0; i < 16; i++)
				blk.m_weights[i * 2 + 1] = 0;
		}

		if (low_selector == high_selector)
		{
			// Solid color block - use precomputed tables of optimal endpoints assuming selector weights are all 1.
			color32 block_colors[4];

			decoder_etc_block::get_block_colors5(block_colors, base_color, inten_table);

			const uint32_t r = block_colors[low_selector].r;
			const uint32_t g = block_colors[low_selector].g;
			const uint32_t b = block_colors[low_selector].b;
						
			blk.m_endpoints[0] = g_astc_single_color_encoding_1[r].m_lo;
			blk.m_endpoints[1] = g_astc_single_color_encoding_1[r].m_hi;

			blk.m_endpoints[2] = g_astc_single_color_encoding_1[g].m_lo;
			blk.m_endpoints[3] = g_astc_single_color_encoding_1[g].m_hi;

			blk.m_endpoints[4] = g_astc_single_color_encoding_1[b].m_lo;
			blk.m_endpoints[5] = g_astc_single_color_encoding_1[b].m_hi;

			int s0 = g_ise_to_unquant[blk.m_endpoints[0]] + g_ise_to_unquant[blk.m_endpoints[2]] + g_ise_to_unquant[blk.m_endpoints[4]];
			int s1 = g_ise_to_unquant[blk.m_endpoints[1]] + g_ise_to_unquant[blk.m_endpoints[3]] + g_ise_to_unquant[blk.m_endpoints[5]];
			bool invert = false;

			if (s1 < s0)
			{
				std::swap(blk.m_endpoints[0], blk.m_endpoints[1]);
				std::swap(blk.m_endpoints[2], blk.m_endpoints[3]);
				std::swap(blk.m_endpoints[4], blk.m_endpoints[5]);
				invert = true;
			}

			for (uint32_t i = 0; i < 16; i++)
				blk.m_weights[i * 2] = invert ? 2 : 1;
		}
		else if ((inten_table >= 7) && (pSelector->m_num_unique_selectors == 2) && (pSelector->m_lo_selector == 0) && (pSelector->m_hi_selector == 3))
		{
			// Handle outlier case where only the two outer colors are used with inten table 7.
			color32 block_colors[4];

			decoder_etc_block::get_block_colors5(block_colors, base_color, inten_table);

			const uint32_t r0 = block_colors[0].r;
			const uint32_t g0 = block_colors[0].g;
			const uint32_t b0 = block_colors[0].b;

			const uint32_t r1 = block_colors[3].r;
			const uint32_t g1 = block_colors[3].g;
			const uint32_t b1 = block_colors[3].b;

			blk.m_endpoints[0] = g_astc_single_color_encoding_0[r0];
			blk.m_endpoints[1] = g_astc_single_color_encoding_0[r1];

			blk.m_endpoints[2] = g_astc_single_color_encoding_0[g0];
			blk.m_endpoints[3] = g_astc_single_color_encoding_0[g1];

			blk.m_endpoints[4] = g_astc_single_color_encoding_0[b0];
			blk.m_endpoints[5] = g_astc_single_color_encoding_0[b1];

			int s0 = g_ise_to_unquant[blk.m_endpoints[0]] + g_ise_to_unquant[blk.m_endpoints[2]] + g_ise_to_unquant[blk.m_endpoints[4]];
			int s1 = g_ise_to_unquant[blk.m_endpoints[1]] + g_ise_to_unquant[blk.m_endpoints[3]] + g_ise_to_unquant[blk.m_endpoints[5]];
			bool invert = false;

			if (s1 < s0)
			{
				std::swap(blk.m_endpoints[0], blk.m_endpoints[1]);
				std::swap(blk.m_endpoints[2], blk.m_endpoints[3]);
				std::swap(blk.m_endpoints[4], blk.m_endpoints[5]);
				invert = true;
			}

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t s = pSelector->get_selector(x, y);
					uint32_t as = (s == low_selector) ? 0 : 3;

					if (invert)
						as = 3 - as;

					blk.m_weights[(x + y * 4) * 2] = static_cast<uint8_t>(as);
				} // x
			} // y
		}
		else
		{
			// Convert ETC1S color
			const uint32_t selector_range_table = g_etc1_to_astc_selector_range_index[low_selector][high_selector];

			//[32][8][RANGES][MAPPING]
			const etc1_to_astc_solution* pTable_r = &g_etc1_to_astc[(inten_table * 32 + base_color.r) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
			const etc1_to_astc_solution* pTable_g = &g_etc1_to_astc[(inten_table * 32 + base_color.g) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];
			const etc1_to_astc_solution* pTable_b = &g_etc1_to_astc[(inten_table * 32 + base_color.b) * (NUM_ETC1_TO_ASTC_SELECTOR_RANGES * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS];

			uint32_t best_err = UINT_MAX;
			uint32_t best_mapping = 0;

			assert(NUM_ETC1_TO_ASTC_SELECTOR_MAPPINGS == 10);
#define DO_ITER(m) { uint32_t total_err = pTable_r[m].m_err + pTable_g[m].m_err + pTable_b[m].m_err; if (total_err < best_err) { best_err = total_err; best_mapping = m; } }
			DO_ITER(0); DO_ITER(1); DO_ITER(2); DO_ITER(3); DO_ITER(4);
			DO_ITER(5); DO_ITER(6); DO_ITER(7); DO_ITER(8); DO_ITER(9);
#undef DO_ITER

			blk.m_endpoints[0] = pTable_r[best_mapping].m_lo;
			blk.m_endpoints[1] = pTable_r[best_mapping].m_hi;

			blk.m_endpoints[2] = pTable_g[best_mapping].m_lo;
			blk.m_endpoints[3] = pTable_g[best_mapping].m_hi;

			blk.m_endpoints[4] = pTable_b[best_mapping].m_lo;
			blk.m_endpoints[5] = pTable_b[best_mapping].m_hi;
						
			int s0 = g_ise_to_unquant[blk.m_endpoints[0]] + g_ise_to_unquant[blk.m_endpoints[2]] + g_ise_to_unquant[blk.m_endpoints[4]];
			int s1 = g_ise_to_unquant[blk.m_endpoints[1]] + g_ise_to_unquant[blk.m_endpoints[3]] + g_ise_to_unquant[blk.m_endpoints[5]];
			bool invert = false;

			if (s1 < s0)
			{
				std::swap(blk.m_endpoints[0], blk.m_endpoints[1]);
				std::swap(blk.m_endpoints[2], blk.m_endpoints[3]);
				std::swap(blk.m_endpoints[4], blk.m_endpoints[5]);
				invert = true;
			}

			const uint8_t* pSelectors_xlat = &g_etc1_to_astc_selector_mappings[best_mapping][0];

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					uint32_t s = pSelector->get_selector(x, y);
					uint32_t as = pSelectors_xlat[s];
					if (invert)
						as = 3 - as;

					blk.m_weights[(x + y * 4) * 2] = static_cast<uint8_t>(as);
				} // x
			} // y
		}

		// Now pack to ASTC
		astc_pack_block_cem_12_weight_range2(reinterpret_cast<uint32_t *>(pDst_block), &blk);
	}
#endif

#if BASISD_SUPPORT_ATC
	// ATC and PVRTC2 both use these tables.
	struct etc1s_to_atc_solution
	{
		uint8_t m_lo;
		uint8_t m_hi;
		uint16_t m_err;
	};

	static dxt_selector_range g_etc1s_to_atc_selector_ranges[] =
	{
		{ 0, 3 },
		{ 1, 3 },
		{ 0, 2 },
		{ 1, 2 },
		{ 2, 3 },
		{ 0, 1 },
	};

	const uint32_t NUM_ETC1S_TO_ATC_SELECTOR_RANGES = sizeof(g_etc1s_to_atc_selector_ranges) / sizeof(g_etc1s_to_atc_selector_ranges[0]);

	static uint32_t g_etc1s_to_atc_selector_range_index[4][4];

	const uint32_t NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS = 10;
	static const uint8_t g_etc1s_to_atc_selector_mappings[NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS][4] =
	{
		{ 0, 0, 1, 1 },
		{ 0, 0, 1, 2 },
		{ 0, 0, 1, 3 },
		{ 0, 0, 2, 3 },
		{ 0, 1, 1, 1 },
		{ 0, 1, 2, 2 },
		{ 0, 1, 2, 3 }, //6 - identity
		{ 0, 2, 3, 3 },
		{ 1, 2, 2, 2 },
		{ 1, 2, 3, 3 },
	};
	const uint32_t ATC_IDENTITY_SELECTOR_MAPPING_INDEX = 6;

#if BASISD_SUPPORT_PVRTC2
	static const etc1s_to_atc_solution g_etc1s_to_pvrtc2_45[32 * 8 * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS * NUM_ETC1S_TO_ATC_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_pvrtc2_45.inc"
	};
		
	static const etc1s_to_atc_solution g_etc1s_to_pvrtc2_alpha_33[32 * 8 * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS * NUM_ETC1S_TO_ATC_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_pvrtc2_alpha_33.inc"
	};
#endif

	static const etc1s_to_atc_solution g_etc1s_to_atc_55[32 * 8 * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS * NUM_ETC1S_TO_ATC_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_atc_55.inc"
	};

	static const etc1s_to_atc_solution g_etc1s_to_atc_56[32 * 8 * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS * NUM_ETC1S_TO_ATC_SELECTOR_RANGES] = {
#include "basisu_transcoder_tables_atc_56.inc"
	};

	struct atc_match_entry
	{
		uint8_t m_lo;
		uint8_t m_hi;
	};
	static atc_match_entry g_pvrtc2_match45_equals_1[256], g_atc_match55_equals_1[256], g_atc_match56_equals_1[256]; // selector 1
	static atc_match_entry g_pvrtc2_match4[256], g_atc_match5[256], g_atc_match6[256];

	static void prepare_atc_single_color_table(atc_match_entry* pTable, int size0, int size1, int sel)
	{
		for (int i = 0; i < 256; i++)
		{
			int lowest_e = 256;
			for (int lo = 0; lo < size0; lo++)
			{
				int lo_e = lo;
				if (size0 == 16)
				{
					lo_e = (lo_e << 1) | (lo_e >> 3);
					lo_e = (lo_e << 3) | (lo_e >> 2);
				}
				else if (size0 == 32)
					lo_e = (lo_e << 3) | (lo_e >> 2);
				else
					lo_e = (lo_e << 2) | (lo_e >> 4);

				for (int hi = 0; hi < size1; hi++)
				{
					int hi_e = hi;
					if (size1 == 16)
					{
						// This is only for PVRTC2 - expand to 5 then 8
						hi_e = (hi_e << 1) | (hi_e >> 3);
						hi_e = (hi_e << 3) | (hi_e >> 2);
					}
					else if (size1 == 32)
						hi_e = (hi_e << 3) | (hi_e >> 2);
					else
						hi_e = (hi_e << 2) | (hi_e >> 4);

					int e;

					if (sel == 1)
					{
						// Selector 1
						e = abs(((lo_e * 5 + hi_e * 3) / 8) - i);
					}
					else
					{
						assert(sel == 3);

						// Selector 3
						e = abs(hi_e - i);
					}

					if (e < lowest_e)
					{
						pTable[i].m_lo = static_cast<uint8_t>(lo);
						pTable[i].m_hi = static_cast<uint8_t>(hi);

						lowest_e = e;
					}

				} // hi
			} // lo
		} // i
	}

	static void transcoder_init_atc()
	{
		prepare_atc_single_color_table(g_pvrtc2_match45_equals_1, 16, 32, 1);
		prepare_atc_single_color_table(g_atc_match55_equals_1, 32, 32, 1); 
		prepare_atc_single_color_table(g_atc_match56_equals_1, 32, 64, 1); 

		prepare_atc_single_color_table(g_pvrtc2_match4, 1, 16, 3);
		prepare_atc_single_color_table(g_atc_match5, 1, 32, 3);
		prepare_atc_single_color_table(g_atc_match6, 1, 64, 3);

		for (uint32_t i = 0; i < NUM_ETC1S_TO_ATC_SELECTOR_RANGES; i++)
		{
			uint32_t l = g_etc1s_to_atc_selector_ranges[i].m_low;
			uint32_t h = g_etc1s_to_atc_selector_ranges[i].m_high;
			g_etc1s_to_atc_selector_range_index[l][h] = i;
		}
	}

	struct atc_block
	{
		uint8_t m_lo[2];
		uint8_t m_hi[2];
		uint8_t m_sels[4];

		void set_low_color(uint32_t r, uint32_t g, uint32_t b)
		{
			assert((r < 32) && (g < 32) && (b < 32));
			uint32_t x = (r << 10) | (g << 5) | b;
			m_lo[0] = x & 0xFF;
			m_lo[1] = (x >> 8) & 0xFF;
		}

		void set_high_color(uint32_t r, uint32_t g, uint32_t b)
		{
			assert((r < 32) && (g < 64) && (b < 32));
			uint32_t x = (r << 11) | (g << 5) | b;
			m_hi[0] = x & 0xFF;
			m_hi[1] = (x >> 8) & 0xFF;
		}
	};

	static void convert_etc1s_to_atc(void* pDst, const endpoint* pEndpoints, const selector* pSelector)
	{
		atc_block* pBlock = static_cast<atc_block*>(pDst);

		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		const color32& base_color = pEndpoints->m_color5;
		const uint32_t inten_table = pEndpoints->m_inten5;

		if (low_selector == high_selector)
		{
			uint32_t r, g, b;
			decoder_etc_block::get_block_color5(base_color, inten_table, low_selector, r, g, b);

			pBlock->set_low_color(g_atc_match55_equals_1[r].m_lo, g_atc_match56_equals_1[g].m_lo, g_atc_match55_equals_1[b].m_lo);
			pBlock->set_high_color(g_atc_match55_equals_1[r].m_hi, g_atc_match56_equals_1[g].m_hi, g_atc_match55_equals_1[b].m_hi);
						
			pBlock->m_sels[0] = 0x55;
			pBlock->m_sels[1] = 0x55;
			pBlock->m_sels[2] = 0x55;
			pBlock->m_sels[3] = 0x55;

			return;
		}
		else if ((inten_table >= 7) && (pSelector->m_num_unique_selectors == 2) && (pSelector->m_lo_selector == 0) && (pSelector->m_hi_selector == 3))
		{
			color32 block_colors[4];
			decoder_etc_block::get_block_colors5(block_colors, base_color, inten_table);

			const uint32_t r0 = block_colors[0].r;
			const uint32_t g0 = block_colors[0].g;
			const uint32_t b0 = block_colors[0].b;

			const uint32_t r1 = block_colors[3].r;
			const uint32_t g1 = block_colors[3].g;
			const uint32_t b1 = block_colors[3].b;

			pBlock->set_low_color(g_atc_match5[r0].m_hi, g_atc_match5[g0].m_hi, g_atc_match5[b0].m_hi);
			pBlock->set_high_color(g_atc_match5[r1].m_hi, g_atc_match6[g1].m_hi, g_atc_match5[b1].m_hi);

			pBlock->m_sels[0] = pSelector->m_selectors[0];
			pBlock->m_sels[1] = pSelector->m_selectors[1];
			pBlock->m_sels[2] = pSelector->m_selectors[2];
			pBlock->m_sels[3] = pSelector->m_selectors[3];

			return;
		}

		const uint32_t selector_range_table = g_etc1s_to_atc_selector_range_index[low_selector][high_selector];

		//[32][8][RANGES][MAPPING]
		const etc1s_to_atc_solution* pTable_r = &g_etc1s_to_atc_55[(inten_table * 32 + base_color.r) * (NUM_ETC1S_TO_ATC_SELECTOR_RANGES * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS];
		const etc1s_to_atc_solution* pTable_g = &g_etc1s_to_atc_56[(inten_table * 32 + base_color.g) * (NUM_ETC1S_TO_ATC_SELECTOR_RANGES * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS];
		const etc1s_to_atc_solution* pTable_b = &g_etc1s_to_atc_55[(inten_table * 32 + base_color.b) * (NUM_ETC1S_TO_ATC_SELECTOR_RANGES * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS];

		uint32_t best_err = UINT_MAX;
		uint32_t best_mapping = 0;

		assert(NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS == 10);
#define DO_ITER(m) { uint32_t total_err = pTable_r[m].m_err + pTable_g[m].m_err + pTable_b[m].m_err; if (total_err < best_err) { best_err = total_err; best_mapping = m; } }
		DO_ITER(0); DO_ITER(1); DO_ITER(2); DO_ITER(3); DO_ITER(4);
		DO_ITER(5); DO_ITER(6); DO_ITER(7); DO_ITER(8); DO_ITER(9);
#undef DO_ITER

		pBlock->set_low_color(pTable_r[best_mapping].m_lo, pTable_g[best_mapping].m_lo, pTable_b[best_mapping].m_lo);
		pBlock->set_high_color(pTable_r[best_mapping].m_hi, pTable_g[best_mapping].m_hi, pTable_b[best_mapping].m_hi);

		if (ATC_IDENTITY_SELECTOR_MAPPING_INDEX == best_mapping)
		{
			pBlock->m_sels[0] = pSelector->m_selectors[0];
			pBlock->m_sels[1] = pSelector->m_selectors[1];
			pBlock->m_sels[2] = pSelector->m_selectors[2];
			pBlock->m_sels[3] = pSelector->m_selectors[3];
		}
		else
		{
			const uint8_t* pSelectors_xlat = &g_etc1s_to_atc_selector_mappings[best_mapping][0];

			const uint32_t sel_bits0 = pSelector->m_selectors[0];
			const uint32_t sel_bits1 = pSelector->m_selectors[1];
			const uint32_t sel_bits2 = pSelector->m_selectors[2];
			const uint32_t sel_bits3 = pSelector->m_selectors[3];

			uint32_t atc_sels0 = 0, atc_sels1 = 0, atc_sels2 = 0, atc_sels3 = 0;

#define DO_X(x) { \
			const uint32_t x_shift = (x) * 2; \
			atc_sels0 |= (pSelectors_xlat[(sel_bits0 >> x_shift) & 3] << x_shift); \
			atc_sels1 |= (pSelectors_xlat[(sel_bits1 >> x_shift) & 3] << x_shift); \
			atc_sels2 |= (pSelectors_xlat[(sel_bits2 >> x_shift) & 3] << x_shift); \
			atc_sels3 |= (pSelectors_xlat[(sel_bits3 >> x_shift) & 3] << x_shift); }

			DO_X(0);
			DO_X(1);
			DO_X(2);
			DO_X(3);
#undef DO_X

			pBlock->m_sels[0] = (uint8_t)atc_sels0;
			pBlock->m_sels[1] = (uint8_t)atc_sels1;
			pBlock->m_sels[2] = (uint8_t)atc_sels2;
			pBlock->m_sels[3] = (uint8_t)atc_sels3;
		}
	}

#if BASISD_WRITE_NEW_ATC_TABLES
	static void create_etc1s_to_atc_conversion_tables()
	{
		// ATC 55
		FILE* pFile = nullptr;
		fopen_s(&pFile, "basisu_transcoder_tables_atc_55.inc", "w");

		uint32_t n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1S_TO_ATC_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1s_to_atc_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1s_to_atc_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 31; hi++)
						{
							for (uint32_t lo = 0; lo <= 31; lo++)
							{
								uint32_t colors[4];

								colors[0] = (lo << 3) | (lo >> 2);
								colors[3] = (hi << 3) | (hi >> 2);

								colors[1] = (colors[0] * 5 + colors[3] * 3) / 8;
								colors[2] = (colors[3] * 5 + colors[0] * 3) / 8;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1s_to_atc_selector_mappings[m][s]];

									int err_scale = 1;
									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										err_scale = 5;

									total_err += (err * err) * err_scale;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						//assert(best_err <= 0xFFFF);
						best_err = basisu::minimum<uint32_t>(best_err, 0xFFFF);

						fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, (uint32_t)best_err);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m
				} // sr
			} // g
		} // inten

		fclose(pFile);
		pFile = nullptr;

		// ATC 56
		fopen_s(&pFile, "basisu_transcoder_tables_atc_56.inc", "w");

		n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1S_TO_ATC_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1s_to_atc_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1s_to_atc_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 63; hi++)
						{
							for (uint32_t lo = 0; lo <= 31; lo++)
							{
								uint32_t colors[4];

								colors[0] = (lo << 3) | (lo >> 2);
								colors[3] = (hi << 2) | (hi >> 4);

								colors[1] = (colors[0] * 5 + colors[3] * 3) / 8;
								colors[2] = (colors[3] * 5 + colors[0] * 3) / 8;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1s_to_atc_selector_mappings[m][s]];

									int err_scale = 1;
									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										err_scale = 5;

									total_err += (err * err) * err_scale;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						//assert(best_err <= 0xFFFF);
						best_err = basisu::minimum<uint32_t>(best_err, 0xFFFF);

						fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, (uint32_t)best_err);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m
				} // sr
			} // g
		} // inten

		fclose(pFile);
		
		// PVRTC2 45
		fopen_s(&pFile, "basisu_transcoder_tables_pvrtc2_45.inc", "w");

		n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1S_TO_ATC_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1s_to_atc_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1s_to_atc_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 31; hi++)
						{
							for (uint32_t lo = 0; lo <= 15; lo++)
							{
								uint32_t colors[4];

								colors[0] = (lo << 1) | (lo >> 3);
								colors[0] = (colors[0] << 3) | (colors[0] >> 2);

								colors[3] = (hi << 3) | (hi >> 2);

								colors[1] = (colors[0] * 5 + colors[3] * 3) / 8;
								colors[2] = (colors[3] * 5 + colors[0] * 3) / 8;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1s_to_atc_selector_mappings[m][s]];

									int err_scale = 1;
									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										err_scale = 5;

									total_err += (err * err) * err_scale;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						//assert(best_err <= 0xFFFF);
						best_err = basisu::minimum<uint32_t>(best_err, 0xFFFF);

						fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, (uint32_t)best_err);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m
				} // sr
			} // g
		} // inten

		fclose(pFile);

#if 0
		// PVRTC2 34
		fopen_s(&pFile, "basisu_transcoder_tables_pvrtc2_34.inc", "w");

		n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1S_TO_ATC_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1s_to_atc_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1s_to_atc_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 15; hi++)
						{
							for (uint32_t lo = 0; lo <= 7; lo++)
							{
								uint32_t colors[4];

								colors[0] = (lo << 2) | (lo >> 1);
								colors[0] = (colors[0] << 3) | (colors[0] >> 2);

								colors[3] = (hi << 1) | (hi >> 3);
								colors[3] = (colors[3] << 3) | (colors[3] >> 2);

								colors[1] = (colors[0] * 5 + colors[3] * 3) / 8;
								colors[2] = (colors[3] * 5 + colors[0] * 3) / 8;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1s_to_atc_selector_mappings[m][s]];

									int err_scale = 1;
									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										err_scale = 5;

									total_err += (err * err) * err_scale;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						//assert(best_err <= 0xFFFF);
						best_err = basisu::minimum<uint32_t>(best_err, 0xFFFF);

						fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, (uint32_t)best_err);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m
				} // sr
			} // g
		} // inten

		fclose(pFile);
#endif
#if 0
		// PVRTC2 44
		fopen_s(&pFile, "basisu_transcoder_tables_pvrtc2_44.inc", "w");

		n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1S_TO_ATC_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1s_to_atc_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1s_to_atc_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 15; hi++)
						{
							for (uint32_t lo = 0; lo <= 15; lo++)
							{
								uint32_t colors[4];

								colors[0] = (lo << 1) | (lo >> 3);
								colors[0] = (colors[0] << 3) | (colors[0] >> 2);

								colors[3] = (hi << 1) | (hi >> 3);
								colors[3] = (colors[3] << 3) | (colors[3] >> 2);

								colors[1] = (colors[0] * 5 + colors[3] * 3) / 8;
								colors[2] = (colors[3] * 5 + colors[0] * 3) / 8;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1s_to_atc_selector_mappings[m][s]];

									int err_scale = 1;
									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										err_scale = 5;

									total_err += (err * err) * err_scale;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						//assert(best_err <= 0xFFFF);
						best_err = basisu::minimum<uint32_t>(best_err, 0xFFFF);

						fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, (uint32_t)best_err);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m
				} // sr
			} // g
		} // inten

		fclose(pFile);
#endif

		// PVRTC2 alpha 33
		fopen_s(&pFile, "basisu_transcoder_tables_pvrtc2_alpha_33.inc", "w");

		n = 0;

		for (int inten = 0; inten < 8; inten++)
		{
			for (uint32_t g = 0; g < 32; g++)
			{
				color32 block_colors[4];
				decoder_etc_block::get_diff_subblock_colors(block_colors, decoder_etc_block::pack_color5(color32(g, g, g, 255), false), inten);

				for (uint32_t sr = 0; sr < NUM_ETC1S_TO_ATC_SELECTOR_RANGES; sr++)
				{
					const uint32_t low_selector = g_etc1s_to_atc_selector_ranges[sr].m_low;
					const uint32_t high_selector = g_etc1s_to_atc_selector_ranges[sr].m_high;

					for (uint32_t m = 0; m < NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS; m++)
					{
						uint32_t best_lo = 0;
						uint32_t best_hi = 0;
						uint64_t best_err = UINT64_MAX;

						for (uint32_t hi = 0; hi <= 7; hi++)
						{
							for (uint32_t lo = 0; lo <= 7; lo++)
							{
								uint32_t colors[4];

								colors[0] = (lo << 1);
								colors[0] = (colors[0] << 4) | colors[0];

								colors[3] = (hi << 1) | 1;
								colors[3] = (colors[3] << 4) | colors[3];

								colors[1] = (colors[0] * 5 + colors[3] * 3) / 8;
								colors[2] = (colors[3] * 5 + colors[0] * 3) / 8;

								uint64_t total_err = 0;

								for (uint32_t s = low_selector; s <= high_selector; s++)
								{
									int err = block_colors[s].g - colors[g_etc1s_to_atc_selector_mappings[m][s]];

									int err_scale = 1;
									// Special case when the intensity table is 7, low_selector is 0, and high_selector is 3. In this extreme case, it's likely the encoder is trying to strongly favor 
									// the low/high selectors which are clamping to either 0 or 255.
									if (((inten == 7) && (low_selector == 0) && (high_selector == 3)) && ((s == 0) || (s == 3)))
										err_scale = 5;

									total_err += (err * err) * err_scale;
								}

								if (total_err < best_err)
								{
									best_err = total_err;
									best_lo = lo;
									best_hi = hi;
								}
							}
						}

						//assert(best_err <= 0xFFFF);
						best_err = basisu::minimum<uint32_t>(best_err, 0xFFFF);

						fprintf(pFile, "{%u,%u,%u},", best_lo, best_hi, (uint32_t)best_err);
						n++;
						if ((n & 31) == 31)
							fprintf(pFile, "\n");
					} // m
				} // sr
			} // g
		} // inten

		fclose(pFile);
	}
#endif // BASISD_WRITE_NEW_ATC_TABLES

#endif // BASISD_SUPPORT_ATC

#if BASISD_SUPPORT_PVRTC2
	struct pvrtc2_block
	{
		uint8_t m_modulation[4];

		union
		{
			union
			{
				// Opaque mode: RGB colora=554 and colorb=555
				struct
				{
					uint32_t m_mod_flag : 1;
					uint32_t m_blue_a : 4;
					uint32_t m_green_a : 5;
					uint32_t m_red_a : 5;
					uint32_t m_hard_flag : 1;
					uint32_t m_blue_b : 5;
					uint32_t m_green_b : 5;
					uint32_t m_red_b : 5;
					uint32_t m_opaque_flag : 1;

				} m_opaque_color_data;

				// Transparent mode: RGBA colora=4433 and colorb=4443
				struct
				{
					uint32_t m_mod_flag : 1;
					uint32_t m_blue_a : 3;
					uint32_t m_green_a : 4;
					uint32_t m_red_a : 4;
					uint32_t m_alpha_a : 3;
					uint32_t m_hard_flag : 1;
					uint32_t m_blue_b : 4;
					uint32_t m_green_b : 4;
					uint32_t m_red_b : 4;
					uint32_t m_alpha_b : 3;
					uint32_t m_opaque_flag : 1;

				} m_trans_color_data;
			};

			uint32_t m_color_data_bits;
		};

		// 554
		void set_low_color(uint32_t r, uint32_t g, uint32_t b)
		{
			assert((r < 32) && (g < 32) && (b < 16));
			m_opaque_color_data.m_red_a = r;
			m_opaque_color_data.m_green_a = g;
			m_opaque_color_data.m_blue_a = b;
		}

		// 555
		void set_high_color(uint32_t r, uint32_t g, uint32_t b)
		{
			assert((r < 32) && (g < 32) && (b < 32));
			m_opaque_color_data.m_red_b = r;
			m_opaque_color_data.m_green_b = g;
			m_opaque_color_data.m_blue_b = b;
		}

		// 4433
		void set_trans_low_color(uint32_t r, uint32_t g, uint32_t b, uint32_t a)
		{
			assert((r < 16) && (g < 16) && (b < 8) && (a < 8));
			m_trans_color_data.m_red_a = r;
			m_trans_color_data.m_green_a = g;
			m_trans_color_data.m_blue_a = b;
			m_trans_color_data.m_alpha_a = a;
		}

		// 4443
		void set_trans_high_color(uint32_t r, uint32_t g, uint32_t b, uint32_t a)
		{
			assert((r < 16) && (g < 16) && (b < 16) && (a < 8));
			m_trans_color_data.m_red_b = r;
			m_trans_color_data.m_green_b = g;
			m_trans_color_data.m_blue_b = b;
			m_trans_color_data.m_alpha_b = a;
		}
	};

	static struct
	{
		uint8_t m_l, m_h;
	} g_pvrtc2_trans_match34[256];

	static struct
	{
		uint8_t m_l, m_h;
	} g_pvrtc2_trans_match44[256];
		
	static struct
	{
		uint8_t m_l, m_h;
	} g_pvrtc2_alpha_match33[256];
	
	static struct
	{
		uint8_t m_l, m_h;
	} g_pvrtc2_alpha_match33_0[256];

	static struct
	{
		uint8_t m_l, m_h;
	} g_pvrtc2_alpha_match33_3[256];
		
	// PVRTC2 can be forced to look like a slightly weaker variant of ATC/BC1, so that's what we do here for simplicity.
	static void convert_etc1s_to_pvrtc2_rgb(void* pDst, const endpoint* pEndpoints, const selector* pSelector)
	{
		pvrtc2_block* pBlock = static_cast<pvrtc2_block*>(pDst);

		pBlock->m_opaque_color_data.m_hard_flag = 1;
		pBlock->m_opaque_color_data.m_mod_flag = 0;
		pBlock->m_opaque_color_data.m_opaque_flag = 1;

		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		const color32& base_color = pEndpoints->m_color5;
		const uint32_t inten_table = pEndpoints->m_inten5;

		if (low_selector == high_selector)
		{
			uint32_t r, g, b;
			decoder_etc_block::get_block_color5(base_color, inten_table, low_selector, r, g, b);

			pBlock->set_low_color(g_atc_match55_equals_1[r].m_lo, g_atc_match55_equals_1[g].m_lo, g_pvrtc2_match45_equals_1[b].m_lo);
			pBlock->set_high_color(g_atc_match55_equals_1[r].m_hi, g_atc_match55_equals_1[g].m_hi, g_pvrtc2_match45_equals_1[b].m_hi);

			pBlock->m_modulation[0] = 0x55;
			pBlock->m_modulation[1] = 0x55;
			pBlock->m_modulation[2] = 0x55;
			pBlock->m_modulation[3] = 0x55;

			return;
		}
		else if ((inten_table >= 7) && (pSelector->m_num_unique_selectors == 2) && (pSelector->m_lo_selector == 0) && (pSelector->m_hi_selector == 3))
		{
			color32 block_colors[4];
			decoder_etc_block::get_block_colors5(block_colors, base_color, inten_table);

			const uint32_t r0 = block_colors[0].r;
			const uint32_t g0 = block_colors[0].g;
			const uint32_t b0 = block_colors[0].b;

			const uint32_t r1 = block_colors[3].r;
			const uint32_t g1 = block_colors[3].g;
			const uint32_t b1 = block_colors[3].b;

			pBlock->set_low_color(g_atc_match5[r0].m_hi, g_atc_match5[g0].m_hi, g_pvrtc2_match4[b0].m_hi);
			pBlock->set_high_color(g_atc_match5[r1].m_hi, g_atc_match5[g1].m_hi, g_atc_match5[b1].m_hi);

			pBlock->m_modulation[0] = pSelector->m_selectors[0];
			pBlock->m_modulation[1] = pSelector->m_selectors[1];
			pBlock->m_modulation[2] = pSelector->m_selectors[2];
			pBlock->m_modulation[3] = pSelector->m_selectors[3];

			return;
		}

		const uint32_t selector_range_table = g_etc1s_to_atc_selector_range_index[low_selector][high_selector];

		//[32][8][RANGES][MAPPING]
		const etc1s_to_atc_solution* pTable_r = &g_etc1s_to_atc_55[(inten_table * 32 + base_color.r) * (NUM_ETC1S_TO_ATC_SELECTOR_RANGES * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS];
		const etc1s_to_atc_solution* pTable_g = &g_etc1s_to_atc_55[(inten_table * 32 + base_color.g) * (NUM_ETC1S_TO_ATC_SELECTOR_RANGES * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS];
		const etc1s_to_atc_solution* pTable_b = &g_etc1s_to_pvrtc2_45[(inten_table * 32 + base_color.b) * (NUM_ETC1S_TO_ATC_SELECTOR_RANGES * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS) + selector_range_table * NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS];

		uint32_t best_err = UINT_MAX;
		uint32_t best_mapping = 0;

		assert(NUM_ETC1S_TO_ATC_SELECTOR_MAPPINGS == 10);
#define DO_ITER(m) { uint32_t total_err = pTable_r[m].m_err + pTable_g[m].m_err + pTable_b[m].m_err; if (total_err < best_err) { best_err = total_err; best_mapping = m; } }
		DO_ITER(0); DO_ITER(1); DO_ITER(2); DO_ITER(3); DO_ITER(4);
		DO_ITER(5); DO_ITER(6); DO_ITER(7); DO_ITER(8); DO_ITER(9);
#undef DO_ITER

		pBlock->set_low_color(pTable_r[best_mapping].m_lo, pTable_g[best_mapping].m_lo, pTable_b[best_mapping].m_lo);
		pBlock->set_high_color(pTable_r[best_mapping].m_hi, pTable_g[best_mapping].m_hi, pTable_b[best_mapping].m_hi);

		if (ATC_IDENTITY_SELECTOR_MAPPING_INDEX == best_mapping)
		{
			pBlock->m_modulation[0] = pSelector->m_selectors[0];
			pBlock->m_modulation[1] = pSelector->m_selectors[1];
			pBlock->m_modulation[2] = pSelector->m_selectors[2];
			pBlock->m_modulation[3] = pSelector->m_selectors[3];
		}
		else
		{
			// TODO: We could make this faster using several precomputed 256 entry tables, like ETC1S->BC1 does.
			const uint8_t* pSelectors_xlat = &g_etc1s_to_atc_selector_mappings[best_mapping][0];

			const uint32_t sel_bits0 = pSelector->m_selectors[0];
			const uint32_t sel_bits1 = pSelector->m_selectors[1];
			const uint32_t sel_bits2 = pSelector->m_selectors[2];
			const uint32_t sel_bits3 = pSelector->m_selectors[3];

			uint32_t sels0 = 0, sels1 = 0, sels2 = 0, sels3 = 0;

#define DO_X(x) { \
			const uint32_t x_shift = (x) * 2; \
			sels0 |= (pSelectors_xlat[(sel_bits0 >> x_shift) & 3] << x_shift); \
			sels1 |= (pSelectors_xlat[(sel_bits1 >> x_shift) & 3] << x_shift); \
			sels2 |= (pSelectors_xlat[(sel_bits2 >> x_shift) & 3] << x_shift); \
			sels3 |= (pSelectors_xlat[(sel_bits3 >> x_shift) & 3] << x_shift); }

			DO_X(0);
			DO_X(1);
			DO_X(2);
			DO_X(3);
#undef DO_X

			pBlock->m_modulation[0] = (uint8_t)sels0;
			pBlock->m_modulation[1] = (uint8_t)sels1;
			pBlock->m_modulation[2] = (uint8_t)sels2;
			pBlock->m_modulation[3] = (uint8_t)sels3;
		}
	}

	typedef struct { float c[4]; } vec4F;

	static inline int32_t clampi(int32_t value, int32_t low, int32_t high) { if (value < low) value = low; else if (value > high) value = high;	return value; }
	static inline float clampf(float value, float low, float high) { if (value < low) value = low; else if (value > high) value = high;	return value; }
	static inline float saturate(float value) { return clampf(value, 0, 1.0f); }
	static inline vec4F* vec4F_set_scalar(vec4F* pV, float x) { pV->c[0] = x; pV->c[1] = x; pV->c[2] = x;	pV->c[3] = x;	return pV; }
	static inline vec4F* vec4F_set(vec4F* pV, float x, float y, float z, float w) { pV->c[0] = x;	pV->c[1] = y;	pV->c[2] = z;	pV->c[3] = w;	return pV; }
	static inline vec4F* vec4F_saturate_in_place(vec4F* pV) { pV->c[0] = saturate(pV->c[0]); pV->c[1] = saturate(pV->c[1]); pV->c[2] = saturate(pV->c[2]); pV->c[3] = saturate(pV->c[3]); return pV; }
	static inline vec4F vec4F_saturate(const vec4F* pV) { vec4F res; res.c[0] = saturate(pV->c[0]); res.c[1] = saturate(pV->c[1]); res.c[2] = saturate(pV->c[2]); res.c[3] = saturate(pV->c[3]); return res; }
	static inline vec4F vec4F_from_color(const color32* pC) { vec4F res; vec4F_set(&res, pC->c[0], pC->c[1], pC->c[2], pC->c[3]); return res; }
	static inline vec4F vec4F_add(const vec4F* pLHS, const vec4F* pRHS) { vec4F res; vec4F_set(&res, pLHS->c[0] + pRHS->c[0], pLHS->c[1] + pRHS->c[1], pLHS->c[2] + pRHS->c[2], pLHS->c[3] + pRHS->c[3]); return res; }
	static inline vec4F vec4F_sub(const vec4F* pLHS, const vec4F* pRHS) { vec4F res; vec4F_set(&res, pLHS->c[0] - pRHS->c[0], pLHS->c[1] - pRHS->c[1], pLHS->c[2] - pRHS->c[2], pLHS->c[3] - pRHS->c[3]); return res; }
	static inline float vec4F_dot(const vec4F* pLHS, const vec4F* pRHS) { return pLHS->c[0] * pRHS->c[0] + pLHS->c[1] * pRHS->c[1] + pLHS->c[2] * pRHS->c[2] + pLHS->c[3] * pRHS->c[3]; }
	static inline vec4F vec4F_mul(const vec4F* pLHS, float s) { vec4F res; vec4F_set(&res, pLHS->c[0] * s, pLHS->c[1] * s, pLHS->c[2] * s, pLHS->c[3] * s); return res; }
	static inline vec4F* vec4F_normalize_in_place(vec4F* pV) { float s = pV->c[0] * pV->c[0] + pV->c[1] * pV->c[1] + pV->c[2] * pV->c[2] + pV->c[3] * pV->c[3]; if (s != 0.0f) { s = 1.0f / sqrtf(s); pV->c[0] *= s; pV->c[1] *= s; pV->c[2] *= s; pV->c[3] *= s; } return pV; }

	static color32 convert_rgba_5554_to_8888(const color32& col)
	{
		return color32((col[0] << 3) | (col[0] >> 2), (col[1] << 3) | (col[1] >> 2), (col[2] << 3) | (col[2] >> 2), (col[3] << 4) | col[3]);
	}

	static inline int sq(int x) { return x * x; }
				
	// PVRTC2 is a slightly borked format for alpha: In Non-Interpolated mode, the way AlphaB8 is exanded from 4 to 8 bits means it can never be 0. 
	// This is actually very bad, because on 100% transparent blocks which have non-trivial color pixels, part of the color channel will leak into alpha! 
	// And there's nothing straightforward we can do because using the other modes is too expensive/complex. I can see why Apple didn't adopt it.
	static void convert_etc1s_to_pvrtc2_rgba(void* pDst, const endpoint* pEndpoints, const selector* pSelector, const endpoint* pEndpoint_codebook, const selector* pSelector_codebook)
	{
		pvrtc2_block* pBlock = static_cast<pvrtc2_block*>(pDst);

		const endpoint& alpha_endpoint = pEndpoint_codebook[((uint16_t*)pBlock)[0]];
		const selector& alpha_selectors = pSelector_codebook[((uint16_t*)pBlock)[1]];

		pBlock->m_opaque_color_data.m_hard_flag = 1;
		pBlock->m_opaque_color_data.m_mod_flag = 0;
		pBlock->m_opaque_color_data.m_opaque_flag = 0;

		const int num_unique_alpha_selectors = alpha_selectors.m_num_unique_selectors;

		const color32& alpha_base_color = alpha_endpoint.m_color5;
		const uint32_t alpha_inten_table = alpha_endpoint.m_inten5;

		int constant_alpha_val = -1;

		int alpha_block_colors[4];
		decoder_etc_block::get_block_colors5_g(alpha_block_colors, alpha_base_color, alpha_inten_table);

		if (num_unique_alpha_selectors == 1)
		{
			constant_alpha_val = alpha_block_colors[alpha_selectors.m_lo_selector];
		}
		else
		{
			constant_alpha_val = alpha_block_colors[alpha_selectors.m_lo_selector];

			for (uint32_t i = alpha_selectors.m_lo_selector + 1; i <= alpha_selectors.m_hi_selector; i++)
			{
				if (constant_alpha_val != alpha_block_colors[i])
				{
					constant_alpha_val = -1;
					break;
				}
			}
		}

		if (constant_alpha_val >= 250)
		{
			// It's opaque enough, so don't bother trying to encode it as an alpha block.
			convert_etc1s_to_pvrtc2_rgb(pDst, pEndpoints, pSelector);
			return;
		}

		const color32& base_color = pEndpoints->m_color5;
		const uint32_t inten_table = pEndpoints->m_inten5;

		const uint32_t low_selector = pSelector->m_lo_selector;
		const uint32_t high_selector = pSelector->m_hi_selector;

		const int num_unique_color_selectors = pSelector->m_num_unique_selectors;
				
		// We need to reencode the block at the pixel level, unfortunately, from two ETC1S planes.
		// Do 4D incremental PCA, project all pixels to this hyperline, then quantize to packed endpoints and compute the modulation values.
		const int br = (base_color.r << 3) | (base_color.r >> 2);
		const int bg = (base_color.g << 3) | (base_color.g >> 2);
		const int bb = (base_color.b << 3) | (base_color.b >> 2);
		
		color32 block_cols[4];
		for (uint32_t i = 0; i < 4; i++)
		{
			const int ci = g_etc1_inten_tables[inten_table][i];
			block_cols[i].set_clamped(br + ci, bg + ci, bb + ci, alpha_block_colors[i]);
		}

		bool solid_color_block = true;
		if (num_unique_color_selectors > 1)
		{
			for (uint32_t i = low_selector + 1; i <= high_selector; i++)
			{
				if ((block_cols[low_selector].r != block_cols[i].r) || (block_cols[low_selector].g != block_cols[i].g) || (block_cols[low_selector].b != block_cols[i].b))
				{
					solid_color_block = false;
					break;
				}
			}
		}

		if ((solid_color_block) && (constant_alpha_val >= 0))
		{
			// Constant color/alpha block.
			// This is more complex than it may seem because of the way color and alpha are packed in PVRTC2. We need to evaluate mod0, mod1 and mod3 encodings to find the best one.
			uint32_t r, g, b;
			decoder_etc_block::get_block_color5(base_color, inten_table, low_selector, r, g, b);

			// Mod 0
			uint32_t lr0 = (r * 15 + 128) / 255, lg0 = (g * 15 + 128) / 255, lb0 = (b * 7 + 128) / 255; 
			uint32_t la0 = g_pvrtc2_alpha_match33_0[constant_alpha_val].m_l;

			uint32_t cr0 = (lr0 << 1) | (lr0 >> 3);
			uint32_t cg0 = (lg0 << 1) | (lg0 >> 3);
			uint32_t cb0 = (lb0 << 2) | (lb0 >> 1);
			uint32_t ca0 = (la0 << 1);
			
			cr0 = (cr0 << 3) | (cr0 >> 2);
			cg0 = (cg0 << 3) | (cg0 >> 2);
			cb0 = (cb0 << 3) | (cb0 >> 2);
			ca0 = (ca0 << 4) | ca0;

			uint32_t err0 = sq(cr0 - r) + sq(cg0 - g) + sq(cb0 - b) + sq(ca0 - constant_alpha_val) * 2;

			// If the alpha is < 3 or so we're kinda screwed. It's better to have some RGB error than it is to turn a 100% transparent area slightly opaque.
			if ((err0 == 0) || (constant_alpha_val < 3))
			{
				pBlock->set_trans_low_color(lr0, lg0, lb0, la0);
				pBlock->set_trans_high_color(0, 0, 0, 0);

				pBlock->m_modulation[0] = 0;
				pBlock->m_modulation[1] = 0;
				pBlock->m_modulation[2] = 0;
				pBlock->m_modulation[3] = 0;
				return;
			}

			// Mod 3
			uint32_t lr3 = (r * 15 + 128) / 255, lg3 = (g * 15 + 128) / 255, lb3 = (b * 15 + 128) / 255;
			uint32_t la3 = g_pvrtc2_alpha_match33_3[constant_alpha_val].m_l;

			uint32_t cr3 = (lr3 << 1) | (lr3 >> 3);
			uint32_t cg3 = (lg3 << 1) | (lg3 >> 3);
			uint32_t cb3 = (lb3 << 1) | (lb3 >> 3);
			uint32_t ca3 = (la3 << 1) | 1;
			
			cr3 = (cr3 << 3) | (cr3 >> 2);
			cg3 = (cg3 << 3) | (cg3 >> 2);
			cb3 = (cb3 << 3) | (cb3 >> 2);
			ca3 = (ca3 << 4) | ca3;

			uint32_t err3 = sq(cr3 - r) + sq(cg3 - g) + sq(cb3 - b) + sq(ca3 - constant_alpha_val) * 2;
			
			// Mod 1
			uint32_t lr1 = g_pvrtc2_trans_match44[r].m_l, lg1 = g_pvrtc2_trans_match44[g].m_l, lb1 = g_pvrtc2_trans_match34[b].m_l;
			uint32_t hr1 = g_pvrtc2_trans_match44[r].m_h, hg1 = g_pvrtc2_trans_match44[g].m_h, hb1 = g_pvrtc2_trans_match34[b].m_h;
			uint32_t la1 = g_pvrtc2_alpha_match33[constant_alpha_val].m_l, ha1 = g_pvrtc2_alpha_match33[constant_alpha_val].m_h;

			uint32_t clr1 = (lr1 << 1) | (lr1 >> 3);
			uint32_t clg1 = (lg1 << 1) | (lg1 >> 3);
			uint32_t clb1 = (lb1 << 2) | (lb1 >> 1);
			uint32_t cla1 = (la1 << 1);

			clr1 = (clr1 << 3) | (clr1 >> 2);
			clg1 = (clg1 << 3) | (clg1 >> 2);
			clb1 = (clb1 << 3) | (clb1 >> 2);
			cla1 = (cla1 << 4) | cla1;

			uint32_t chr1 = (hr1 << 1) | (hr1 >> 3);
			uint32_t chg1 = (hg1 << 1) | (hg1 >> 3);
			uint32_t chb1 = (hb1 << 1) | (hb1 >> 3);
			uint32_t cha1 = (ha1 << 1) | 1;

			chr1 = (chr1 << 3) | (chr1 >> 2);
			chg1 = (chg1 << 3) | (chg1 >> 2);
			chb1 = (chb1 << 3) | (chb1 >> 2);
			cha1 = (cha1 << 4) | cha1;

			uint32_t r1 = (clr1 * 5 + chr1 * 3) / 8;
			uint32_t g1 = (clg1 * 5 + chg1 * 3) / 8;
			uint32_t b1 = (clb1 * 5 + chb1 * 3) / 8;
			uint32_t a1 = (cla1 * 5 + cha1 * 3) / 8;

			uint32_t err1 = sq(r1 - r) + sq(g1 - g) + sq(b1 - b) + sq(a1 - constant_alpha_val) * 2;

			if ((err1 < err0) && (err1 < err3))
			{
				pBlock->set_trans_low_color(lr1, lg1, lb1, la1);
				pBlock->set_trans_high_color(hr1, hg1, hb1, ha1);

				pBlock->m_modulation[0] = 0x55;
				pBlock->m_modulation[1] = 0x55;
				pBlock->m_modulation[2] = 0x55;
				pBlock->m_modulation[3] = 0x55;
			}
			else if (err0 < err3)
			{
				pBlock->set_trans_low_color(lr0, lg0, lb0, la0);
				pBlock->set_trans_high_color(0, 0, 0, 0);

				pBlock->m_modulation[0] = 0;
				pBlock->m_modulation[1] = 0;
				pBlock->m_modulation[2] = 0;
				pBlock->m_modulation[3] = 0;
			}
			else
			{
				pBlock->set_trans_low_color(0, 0, 0, 0);
				pBlock->set_trans_high_color(lr3, lg3, lb3, la3);

				pBlock->m_modulation[0] = 0xFF;
				pBlock->m_modulation[1] = 0xFF;
				pBlock->m_modulation[2] = 0xFF;
				pBlock->m_modulation[3] = 0xFF;
			}

			return;
		}

		// It's a complex block with non-solid color and/or alpha pixels.
		vec4F minColor, maxColor;

		if (solid_color_block)
		{
			// It's a solid color block.
			uint32_t low_a = block_cols[alpha_selectors.m_lo_selector].a;
			uint32_t high_a = block_cols[alpha_selectors.m_hi_selector].a;
			
			const float S = 1.0f / 255.0f;
			vec4F_set(&minColor, block_cols[low_selector].r * S, block_cols[low_selector].g * S, block_cols[low_selector].b * S, low_a * S);
			vec4F_set(&maxColor, block_cols[low_selector].r * S, block_cols[low_selector].g * S, block_cols[low_selector].b * S, high_a * S);
		}
		else if (constant_alpha_val >= 0)
		{
			// It's a solid alpha block.
			const float S = 1.0f / 255.0f;
			vec4F_set(&minColor, block_cols[low_selector].r * S, block_cols[low_selector].g * S, block_cols[low_selector].b * S, constant_alpha_val * S);
			vec4F_set(&maxColor, block_cols[high_selector].r * S, block_cols[high_selector].g * S, block_cols[high_selector].b * S, constant_alpha_val * S);
	   }
		// See if any of the block colors got clamped - if so the principle axis got distorted (it's no longer just the ETC1S luma axis). 
		// To keep quality up we need to use full 4D PCA in this case.
		else	if ((block_cols[low_selector].c[0] == 0) || (block_cols[high_selector].c[0] == 255) ||
				(block_cols[low_selector].c[1] == 0) || (block_cols[high_selector].c[1] == 255) ||
				(block_cols[low_selector].c[2] == 0) || (block_cols[high_selector].c[2] == 255) ||
				(block_cols[alpha_selectors.m_lo_selector].c[3] == 0) || (block_cols[alpha_selectors.m_hi_selector].c[3] == 255))
		{
			// Find principle component of RGBA colors treated as 4D vectors.
			color32 pixels[16];

			uint32_t sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
			for (uint32_t i = 0; i < 16; i++)
			{
				color32 rgb(block_cols[pSelector->get_selector(i & 3, i >> 2)]);
				uint32_t a = block_cols[alpha_selectors.get_selector(i & 3, i >> 2)].a;

				pixels[i].set(rgb.r, rgb.g, rgb.b, a);

				sum_r += rgb.r;
				sum_g += rgb.g;
				sum_b += rgb.b;
				sum_a += a;
			}

			vec4F meanColor;
			vec4F_set(&meanColor, (float)sum_r, (float)sum_g, (float)sum_b, (float)sum_a);
			vec4F meanColorScaled = vec4F_mul(&meanColor, 1.0f / 16.0f);

			meanColor = vec4F_mul(&meanColor, 1.0f / (float)(16.0f * 255.0f));
			vec4F_saturate_in_place(&meanColor);

			vec4F axis;
			vec4F_set_scalar(&axis, 0.0f);
			// Why this incremental method? Because it's stable and predictable. Covar+power method can require a lot of iterations to converge in 4D.
			for (uint32_t i = 0; i < 16; i++)
			{
				vec4F color = vec4F_from_color(&pixels[i]);
				color = vec4F_sub(&color, &meanColorScaled);
				vec4F a = vec4F_mul(&color, color.c[0]);
				vec4F b = vec4F_mul(&color, color.c[1]);
				vec4F c = vec4F_mul(&color, color.c[2]);
				vec4F d = vec4F_mul(&color, color.c[3]);
				vec4F n = i ? axis : color;
				vec4F_normalize_in_place(&n);
				axis.c[0] += vec4F_dot(&a, &n);
				axis.c[1] += vec4F_dot(&b, &n);
				axis.c[2] += vec4F_dot(&c, &n);
				axis.c[3] += vec4F_dot(&d, &n);
			}

			vec4F_normalize_in_place(&axis);

			if (vec4F_dot(&axis, &axis) < .5f)
				vec4F_set_scalar(&axis, .5f);

			float l = 1e+9f, h = -1e+9f;

			for (uint32_t i = 0; i < 16; i++)
			{
				vec4F color = vec4F_from_color(&pixels[i]);

				vec4F q = vec4F_sub(&color, &meanColorScaled);
				float d = vec4F_dot(&q, &axis);

				l = basisu::minimum(l, d);
				h = basisu::maximum(h, d);
			}

			l *= (1.0f / 255.0f);
			h *= (1.0f / 255.0f);

			vec4F b0 = vec4F_mul(&axis, l);
			vec4F b1 = vec4F_mul(&axis, h);
			vec4F c0 = vec4F_add(&meanColor, &b0);
			vec4F c1 = vec4F_add(&meanColor, &b1);
			minColor = vec4F_saturate(&c0);
			maxColor = vec4F_saturate(&c1);
			if (minColor.c[3] > maxColor.c[3])
				std::swap(minColor, maxColor);
		}
		else
		{
			// We know the RGB axis is luma, because it's an ETC1S block and none of the block colors got clamped. So we only need to use 2D PCA.
			// We project each LA vector onto two 2D lines with axes (1,1) and (1,-1) and find the largest projection to determine if axis A is flipped relative to L.
			uint32_t block_cols_l[4], block_cols_a[4];
			for (uint32_t i = 0; i < 4; i++)
			{
				block_cols_l[i] = block_cols[i].r + block_cols[i].g + block_cols[i].b;
				block_cols_a[i] = block_cols[i].a * 3;
			}

			int p0_min = INT_MAX, p0_max = INT_MIN;
			int p1_min = INT_MAX, p1_max = INT_MIN;
			for (uint32_t y = 0; y < 4; y++)
			{
				const uint32_t cs = pSelector->m_selectors[y];
				const uint32_t as = alpha_selectors.m_selectors[y];

				{
					const int l = block_cols_l[cs & 3];
					const int a = block_cols_a[as & 3];
					const int p0 = l + a; p0_min = basisu::minimum(p0_min, p0); p0_max = basisu::maximum(p0_max, p0);
					const int p1 = l - a; p1_min = basisu::minimum(p1_min, p1); p1_max = basisu::maximum(p1_max, p1);
				}
				{
					const int l = block_cols_l[(cs >> 2) & 3];
					const int a = block_cols_a[(as >> 2) & 3];
					const int p0 = l + a; p0_min = basisu::minimum(p0_min, p0); p0_max = basisu::maximum(p0_max, p0);
					const int p1 = l - a; p1_min = basisu::minimum(p1_min, p1); p1_max = basisu::maximum(p1_max, p1);
				}
				{
					const int l = block_cols_l[(cs >> 4) & 3];
					const int a = block_cols_a[(as >> 4) & 3];
					const int p0 = l + a; p0_min = basisu::minimum(p0_min, p0); p0_max = basisu::maximum(p0_max, p0);
					const int p1 = l - a; p1_min = basisu::minimum(p1_min, p1); p1_max = basisu::maximum(p1_max, p1);
				}
				{
					const int l = block_cols_l[cs >> 6];
					const int a = block_cols_a[as >> 6];
					const int p0 = l + a; p0_min = basisu::minimum(p0_min, p0); p0_max = basisu::maximum(p0_max, p0);
					const int p1 = l - a; p1_min = basisu::minimum(p1_min, p1); p1_max = basisu::maximum(p1_max, p1);
				}
			}

			int dist0 = p0_max - p0_min;
			int dist1 = p1_max - p1_min;

			const float S = 1.0f / 255.0f;

			vec4F_set(&minColor, block_cols[low_selector].r * S, block_cols[low_selector].g * S, block_cols[low_selector].b * S, block_cols[alpha_selectors.m_lo_selector].a * S);
			vec4F_set(&maxColor, block_cols[high_selector].r * S, block_cols[high_selector].g * S, block_cols[high_selector].b * S, block_cols[alpha_selectors.m_hi_selector].a * S);

			// See if the A component of the principle axis is flipped relative to L. If so, we need to flip either RGB or A bounds.
			if (dist1 > dist0)
			{
				std::swap(minColor.c[0], maxColor.c[0]);
				std::swap(minColor.c[1], maxColor.c[1]);
				std::swap(minColor.c[2], maxColor.c[2]);
			}
		}

		// 4433 4443
		color32 trialMinColor, trialMaxColor;
				
		trialMinColor.set_clamped((int)(minColor.c[0] * 15.0f + .5f), (int)(minColor.c[1] * 15.0f + .5f), (int)(minColor.c[2] * 7.0f + .5f), (int)(minColor.c[3] * 7.0f + .5f));
		trialMaxColor.set_clamped((int)(maxColor.c[0] * 15.0f + .5f), (int)(maxColor.c[1] * 15.0f + .5f), (int)(maxColor.c[2] * 15.0f + .5f), (int)(maxColor.c[3] * 7.0f + .5f));
				
		pBlock->set_trans_low_color(trialMinColor.r, trialMinColor.g, trialMinColor.b, trialMinColor.a);
		pBlock->set_trans_high_color(trialMaxColor.r, trialMaxColor.g, trialMaxColor.b, trialMaxColor.a);

		color32 color_a((trialMinColor.r << 1) | (trialMinColor.r >> 3), (trialMinColor.g << 1) | (trialMinColor.g >> 3), (trialMinColor.b << 2) | (trialMinColor.b >> 1), trialMinColor.a << 1);
		color32 color_b((trialMaxColor.r << 1) | (trialMaxColor.r >> 3), (trialMaxColor.g << 1) | (trialMaxColor.g >> 3), (trialMaxColor.b << 1) | (trialMaxColor.b >> 3), (trialMaxColor.a << 1) | 1);

		color32 color0(convert_rgba_5554_to_8888(color_a));
		color32 color3(convert_rgba_5554_to_8888(color_b));

		const int lr = color0.r;
		const int lg = color0.g;
		const int lb = color0.b;
		const int la = color0.a;

		const int axis_r = color3.r - lr;
		const int axis_g = color3.g - lg;
		const int axis_b = color3.b - lb;
		const int axis_a = color3.a - la;
		const int len_a = (axis_r * axis_r) + (axis_g * axis_g) + (axis_b * axis_b) + (axis_a * axis_a);

		const int thresh01 = (len_a * 3) / 16;
		const int thresh12 = len_a >> 1;
		const int thresh23 = (len_a * 13) / 16;

		if ((axis_r | axis_g | axis_b) == 0)
		{
			int ca_sel[4];

			for (uint32_t i = 0; i < 4; i++)
			{
				int ca = (block_cols[i].a - la) * axis_a;
				ca_sel[i] = (ca >= thresh23) + (ca >= thresh12) + (ca >= thresh01);
			}

			for (uint32_t y = 0; y < 4; y++)
			{
				const uint32_t a_sels = alpha_selectors.m_selectors[y];

				uint32_t sel = ca_sel[a_sels & 3] | (ca_sel[(a_sels >> 2) & 3] << 2) | (ca_sel[(a_sels >> 4) & 3] << 4) | (ca_sel[a_sels >> 6] << 6);

				pBlock->m_modulation[y] = (uint8_t)sel;
			}
		}
		else
		{
			int cy[4], ca[4];

			for (uint32_t i = 0; i < 4; i++)
			{
				cy[i] = (block_cols[i].r - lr) * axis_r + (block_cols[i].g - lg) * axis_g + (block_cols[i].b - lb) * axis_b;
				ca[i] = (block_cols[i].a - la) * axis_a;
			}

			for (uint32_t y = 0; y < 4; y++)
			{
				const uint32_t c_sels = pSelector->m_selectors[y];
				const uint32_t a_sels = alpha_selectors.m_selectors[y];

				const int d0 = cy[c_sels & 3] + ca[a_sels & 3];
				const int d1 = cy[(c_sels >> 2) & 3] + ca[(a_sels >> 2) & 3];
				const int d2 = cy[(c_sels >> 4) & 3] + ca[(a_sels >> 4) & 3];
				const int d3 = cy[c_sels >> 6] + ca[a_sels >> 6];

				uint32_t sel = ((d0 >= thresh23) + (d0 >= thresh12) + (d0 >= thresh01)) |
					(((d1 >= thresh23) + (d1 >= thresh12) + (d1 >= thresh01)) << 2) |
					(((d2 >= thresh23) + (d2 >= thresh12) + (d2 >= thresh01)) << 4) |
					(((d3 >= thresh23) + (d3 >= thresh12) + (d3 >= thresh01)) << 6);

				pBlock->m_modulation[y] = (uint8_t)sel;
			}
		}
	}
		
	static void transcoder_init_pvrtc2()
	{
		for (uint32_t v = 0; v < 256; v++)
		{
			int best_l = 0, best_h = 0, lowest_err = INT_MAX;

			for (uint32_t l = 0; l < 8; l++)
			{
				uint32_t le = (l << 1);
				le = (le << 4) | le;

				for (uint32_t h = 0; h < 8; h++)
				{
					uint32_t he = (h << 1) | 1;
					he = (he << 4) | he;

					uint32_t m = (le * 5 + he * 3) / 8;

					int err = labs((int)v - (int)m);
					if (err < lowest_err)
					{
						lowest_err = err;
						best_l = l;
						best_h = h;
					}
				}
			}

			g_pvrtc2_alpha_match33[v].m_l = (uint8_t)best_l;
			g_pvrtc2_alpha_match33[v].m_h = (uint8_t)best_h;
		}

		for (uint32_t v = 0; v < 256; v++)
		{
			int best_l = 0, best_h = 0, lowest_err = INT_MAX;

			for (uint32_t l = 0; l < 8; l++)
			{
				uint32_t le = (l << 1);
				le = (le << 4) | le;

				int err = labs((int)v - (int)le);
				if (err < lowest_err)
				{
					lowest_err = err;
					best_l = l;
					best_h = l;
				}
			}

			g_pvrtc2_alpha_match33_0[v].m_l = (uint8_t)best_l;
			g_pvrtc2_alpha_match33_0[v].m_h = (uint8_t)best_h;
		}

		for (uint32_t v = 0; v < 256; v++)
		{
			int best_l = 0, best_h = 0, lowest_err = INT_MAX;

			for (uint32_t h = 0; h < 8; h++)
			{
				uint32_t he = (h << 1) | 1;
				he = (he << 4) | he;

				int err = labs((int)v - (int)he);
				if (err < lowest_err)
				{
					lowest_err = err;
					best_l = h;
					best_h = h;
				}
			}

			g_pvrtc2_alpha_match33_3[v].m_l = (uint8_t)best_l;
			g_pvrtc2_alpha_match33_3[v].m_h = (uint8_t)best_h;
		}

		for (uint32_t v = 0; v < 256; v++)
		{
			int best_l = 0, best_h = 0, lowest_err = INT_MAX;

			for (uint32_t l = 0; l < 8; l++)
			{
				uint32_t le = (l << 2) | (l >> 1);
				le = (le << 3) | (le >> 2);

				for (uint32_t h = 0; h < 16; h++)
				{
					uint32_t he = (h << 1) | (h >> 3);
					he = (he << 3) | (he >> 2);

					uint32_t m = (le * 5 + he * 3) / 8;

					int err = labs((int)v - (int)m);
					if (err < lowest_err)
					{
						lowest_err = err;
						best_l = l;
						best_h = h;
					}
				}
			}

			g_pvrtc2_trans_match34[v].m_l = (uint8_t)best_l;
			g_pvrtc2_trans_match34[v].m_h = (uint8_t)best_h;
		}
				
		for (uint32_t v = 0; v < 256; v++)
		{
			int best_l = 0, best_h = 0, lowest_err = INT_MAX;

			for (uint32_t l = 0; l < 16; l++)
			{
				uint32_t le = (l << 1) | (l >> 3);
				le = (le << 3) | (le >> 2);

				for (uint32_t h = 0; h < 16; h++)
				{
					uint32_t he = (h << 1) | (h >> 3);
					he = (he << 3) | (he >> 2);

					uint32_t m = (le * 5 + he * 3) / 8;

					int err = labs((int)v - (int)m);
					if (err < lowest_err)
					{
						lowest_err = err;
						best_l = l;
						best_h = h;
					}
				}
			}

			g_pvrtc2_trans_match44[v].m_l = (uint8_t)best_l;
			g_pvrtc2_trans_match44[v].m_h = (uint8_t)best_h;
		}
	}
#endif // BASISD_SUPPORT_PVRTC2

	basisu_lowlevel_transcoder::basisu_lowlevel_transcoder(const etc1_global_selector_codebook* pGlobal_sel_codebook) :
		m_pGlobal_sel_codebook(pGlobal_sel_codebook),
		m_selector_history_buf_size(0)
	{
	}

	bool basisu_lowlevel_transcoder::decode_palettes(
		uint32_t num_endpoints, const uint8_t* pEndpoints_data, uint32_t endpoints_data_size,
		uint32_t num_selectors, const uint8_t* pSelectors_data, uint32_t selectors_data_size)
	{
		bitwise_decoder sym_codec;

		huffman_decoding_table color5_delta_model0, color5_delta_model1, color5_delta_model2, inten_delta_model;

		if (!sym_codec.init(pEndpoints_data, endpoints_data_size))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 0\n");
			return false;
		}

		if (!sym_codec.read_huffman_table(color5_delta_model0))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 1\n");
			return false;
		}

		if (!sym_codec.read_huffman_table(color5_delta_model1))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 1a\n");
			return false;
		}

		if (!sym_codec.read_huffman_table(color5_delta_model2))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 2a\n");
			return false;
		}

		if (!sym_codec.read_huffman_table(inten_delta_model))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 2b\n");
			return false;
		}

		if (!color5_delta_model0.is_valid() || !color5_delta_model1.is_valid() || !color5_delta_model2.is_valid() || !inten_delta_model.is_valid())
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 2b\n");
			return false;
		}

		const bool endpoints_are_grayscale = sym_codec.get_bits(1) != 0;

		m_endpoints.resize(num_endpoints);

		color32 prev_color5(16, 16, 16, 0);
		uint32_t prev_inten = 0;

		for (uint32_t i = 0; i < num_endpoints; i++)
		{
			uint32_t inten_delta = sym_codec.decode_huffman(inten_delta_model);
			m_endpoints[i].m_inten5 = static_cast<uint8_t>((inten_delta + prev_inten) & 7);
			prev_inten = m_endpoints[i].m_inten5;

			for (uint32_t c = 0; c < (endpoints_are_grayscale ? 1U : 3U); c++)
			{
				int delta;
				if (prev_color5[c] <= basist::COLOR5_PAL0_PREV_HI)
					delta = sym_codec.decode_huffman(color5_delta_model0);
				else if (prev_color5[c] <= basist::COLOR5_PAL1_PREV_HI)
					delta = sym_codec.decode_huffman(color5_delta_model1);
				else
					delta = sym_codec.decode_huffman(color5_delta_model2);

				int v = (prev_color5[c] + delta) & 31;

				m_endpoints[i].m_color5[c] = static_cast<uint8_t>(v);

				prev_color5[c] = static_cast<uint8_t>(v);
			}

			if (endpoints_are_grayscale)
			{
				m_endpoints[i].m_color5[1] = m_endpoints[i].m_color5[0];
				m_endpoints[i].m_color5[2] = m_endpoints[i].m_color5[0];
			}
		}

		sym_codec.stop();

		m_selectors.resize(num_selectors);
		
		if (!sym_codec.init(pSelectors_data, selectors_data_size))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 5\n");
			return false;
		}

		basist::huffman_decoding_table delta_selector_pal_model;

		const bool used_global_selector_cb = (sym_codec.get_bits(1) == 1);

		if (used_global_selector_cb)
		{
			// global selector palette
			uint32_t pal_bits = sym_codec.get_bits(4);
			uint32_t mod_bits = sym_codec.get_bits(4);

			basist::huffman_decoding_table mod_model;
			if (mod_bits)
			{
				if (!sym_codec.read_huffman_table(mod_model))
				{
					BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 6\n");
					return false;
				}
				if (!mod_model.is_valid())
				{
					BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 6a\n");
					return false;
				}
			}

			for (uint32_t i = 0; i < num_selectors; i++)
			{
				uint32_t pal_index = 0;
				if (pal_bits)
					pal_index = sym_codec.get_bits(pal_bits);

				uint32_t mod_index = 0;
				if (mod_bits)
					mod_index = sym_codec.decode_huffman(mod_model);

				if (pal_index >= m_pGlobal_sel_codebook->size())
				{
					BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 7z\n");
					return false;
				}

				const etc1_selector_palette_entry e(m_pGlobal_sel_codebook->get_entry(pal_index, etc1_global_palette_entry_modifier(mod_index)));

				// TODO: Optimize this
				for (uint32_t y = 0; y < 4; y++)
					for (uint32_t x = 0; x < 4; x++)
						m_selectors[i].set_selector(x, y, e[x + y * 4]);

				m_selectors[i].init_flags();
			}
		}
		else
		{
			const bool used_hybrid_selector_cb = (sym_codec.get_bits(1) == 1);

			if (used_hybrid_selector_cb)
			{
				const uint32_t pal_bits = sym_codec.get_bits(4);
				const uint32_t mod_bits = sym_codec.get_bits(4);

				basist::huffman_decoding_table uses_global_cb_bitflags_model;
				if (!sym_codec.read_huffman_table(uses_global_cb_bitflags_model))
				{
					BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 7\n");
					return false;
				}
				if (!uses_global_cb_bitflags_model.is_valid())
				{
					BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 7a\n");
					return false;
				}

				basist::huffman_decoding_table global_mod_indices_model;
				if (mod_bits)
				{
					if (!sym_codec.read_huffman_table(global_mod_indices_model))
					{
						BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 8\n");
						return false;
					}
					if (!global_mod_indices_model.is_valid())
					{
						BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 8a\n");
						return false;
					}
				}

				uint32_t cur_uses_global_cb_bitflags = 0;
				uint32_t uses_global_cb_bitflags_remaining = 0;

				for (uint32_t q = 0; q < num_selectors; q++)
				{
					if (!uses_global_cb_bitflags_remaining)
					{
						cur_uses_global_cb_bitflags = sym_codec.decode_huffman(uses_global_cb_bitflags_model);

						uses_global_cb_bitflags_remaining = 8;
					}
					uses_global_cb_bitflags_remaining--;

					const bool used_global_cb_flag = (cur_uses_global_cb_bitflags & 1) != 0;
					cur_uses_global_cb_bitflags >>= 1;

					if (used_global_cb_flag)
					{
						const uint32_t pal_index = pal_bits ? sym_codec.get_bits(pal_bits) : 0;
						const uint32_t mod_index = mod_bits ? sym_codec.decode_huffman(global_mod_indices_model) : 0;

						if (pal_index >= m_pGlobal_sel_codebook->size())
						{
							BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 8b\n");
							return false;
						}

						const etc1_selector_palette_entry e(m_pGlobal_sel_codebook->get_entry(pal_index, etc1_global_palette_entry_modifier(mod_index)));

						for (uint32_t y = 0; y < 4; y++)
							for (uint32_t x = 0; x < 4; x++)
								m_selectors[q].set_selector(x, y, e[x + y * 4]);
					}
					else
					{
						for (uint32_t j = 0; j < 4; j++)
						{
							uint32_t cur_byte = sym_codec.get_bits(8);

							for (uint32_t k = 0; k < 4; k++)
								m_selectors[q].set_selector(k, j, (cur_byte >> (k * 2)) & 3);
						}
					}

					m_selectors[q].init_flags();
				}
			}
			else
			{
				const bool used_raw_encoding = (sym_codec.get_bits(1) == 1);

				if (used_raw_encoding)
				{
					for (uint32_t i = 0; i < num_selectors; i++)
					{
						for (uint32_t j = 0; j < 4; j++)
						{
							uint32_t cur_byte = sym_codec.get_bits(8);

							for (uint32_t k = 0; k < 4; k++)
								m_selectors[i].set_selector(k, j, (cur_byte >> (k * 2)) & 3);
						}

						m_selectors[i].init_flags();
					}
				}
				else
				{
					if (!sym_codec.read_huffman_table(delta_selector_pal_model))
					{
						BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 10\n");
						return false;
					}

					if ((num_selectors > 1) && (!delta_selector_pal_model.is_valid()))
					{
						BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_palettes: fail 10a\n");
						return false;
					}

					uint8_t prev_bytes[4] = { 0, 0, 0, 0 };

					for (uint32_t i = 0; i < num_selectors; i++)
					{
						if (!i)
						{
							for (uint32_t j = 0; j < 4; j++)
							{
								uint32_t cur_byte = sym_codec.get_bits(8);
								prev_bytes[j] = static_cast<uint8_t>(cur_byte);

								for (uint32_t k = 0; k < 4; k++)
									m_selectors[i].set_selector(k, j, (cur_byte >> (k * 2)) & 3);
							}
							m_selectors[i].init_flags();
							continue;
						}

						for (uint32_t j = 0; j < 4; j++)
						{
							int delta_byte = sym_codec.decode_huffman(delta_selector_pal_model);

							uint32_t cur_byte = delta_byte ^ prev_bytes[j];
							prev_bytes[j] = static_cast<uint8_t>(cur_byte);

							for (uint32_t k = 0; k < 4; k++)
								m_selectors[i].set_selector(k, j, (cur_byte >> (k * 2)) & 3);
						}
						m_selectors[i].init_flags();
					}
				}
			}
		}

		sym_codec.stop();

		return true;
	}

	bool basisu_lowlevel_transcoder::decode_tables(const uint8_t* pTable_data, uint32_t table_data_size)
	{
		basist::bitwise_decoder sym_codec;
		if (!sym_codec.init(pTable_data, table_data_size))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_tables: fail 0\n");
			return false;
		}

		if (!sym_codec.read_huffman_table(m_endpoint_pred_model))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_tables: fail 1\n");
			return false;
		}

		if (m_endpoint_pred_model.get_code_sizes().size() == 0)
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_tables: fail 1a\n");
			return false;
		}

		if (!sym_codec.read_huffman_table(m_delta_endpoint_model))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_tables: fail 2\n");
			return false;
		}

		if (m_delta_endpoint_model.get_code_sizes().size() == 0)
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_tables: fail 2a\n");
			return false;
		}

		if (!sym_codec.read_huffman_table(m_selector_model))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_tables: fail 3\n");
			return false;
		}

		if (m_selector_model.get_code_sizes().size() == 0)
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_tables: fail 3a\n");
			return false;
		}

		if (!sym_codec.read_huffman_table(m_selector_history_buf_rle_model))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_tables: fail 4\n");
			return false;
		}

		if (m_selector_history_buf_rle_model.get_code_sizes().size() == 0)
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::decode_tables: fail 4a\n");
			return false;
		}

		m_selector_history_buf_size = sym_codec.get_bits(13);

		sym_codec.stop();

		return true;
	}

	bool basisu_lowlevel_transcoder::transcode_slice(void* pDst_blocks, uint32_t num_blocks_x, uint32_t num_blocks_y, const uint8_t* pImage_data, uint32_t image_data_size, block_format fmt,
		uint32_t output_block_or_pixel_stride_in_bytes, bool bc1_allow_threecolor_blocks, const basis_file_header& header, const basis_slice_desc& slice_desc, uint32_t output_row_pitch_in_blocks_or_pixels,
		basisu_transcoder_state* pState, bool transcode_alpha, void *pAlpha_blocks, uint32_t output_rows_in_pixels)
	{
		(void)transcode_alpha;
		(void)pAlpha_blocks;

		if (!pState)
			pState = &m_def_state;

		const bool is_video = (header.m_tex_type == cBASISTexTypeVideoFrames);
		const uint32_t total_blocks = num_blocks_x * num_blocks_y;

		if (!output_row_pitch_in_blocks_or_pixels)
		{
			if (basis_block_format_is_uncompressed(fmt))
				output_row_pitch_in_blocks_or_pixels = slice_desc.m_orig_width;
			else
			{
				if (fmt == block_format::cFXT1_RGB)
					output_row_pitch_in_blocks_or_pixels = (slice_desc.m_orig_width + 7) / 8;
				else
					output_row_pitch_in_blocks_or_pixels = num_blocks_x;
			}
		}

		if (basis_block_format_is_uncompressed(fmt))
		{
			if (!output_rows_in_pixels)
				output_rows_in_pixels = slice_desc.m_orig_height;
		}
		
		std::vector<uint32_t>* pPrev_frame_indices = nullptr;
		if (is_video)
		{
			// TODO: Add check to make sure the caller hasn't tried skipping past p-frames
			const bool alpha_flag = (slice_desc.m_flags & cSliceDescFlagsIsAlphaData) != 0;
			const uint32_t level_index = slice_desc.m_level_index;

			if (level_index >= basisu_transcoder_state::cMaxPrevFrameLevels)
			{
				BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: unsupported level_index\n");
				return false;
			}

			pPrev_frame_indices = &pState->m_prev_frame_indices[alpha_flag][level_index];
			if (pPrev_frame_indices->size() < total_blocks)
				pPrev_frame_indices->resize(total_blocks);
		}

		basist::bitwise_decoder sym_codec;

		if (!sym_codec.init(pImage_data, image_data_size))
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: sym_codec.init failed\n");
			return false;
		}

		approx_move_to_front selector_history_buf(m_selector_history_buf_size);

		const uint32_t SELECTOR_HISTORY_BUF_FIRST_SYMBOL_INDEX = (uint32_t)m_selectors.size();
		const uint32_t SELECTOR_HISTORY_BUF_RLE_SYMBOL_INDEX = m_selector_history_buf_size + SELECTOR_HISTORY_BUF_FIRST_SYMBOL_INDEX;
		uint32_t cur_selector_rle_count = 0;

		decoder_etc_block block;
		memset(&block, 0, sizeof(block));

		block.set_flip_bit(true);
		block.set_diff_bit(true);

		void* pPVRTC_work_mem = nullptr;
		uint32_t* pPVRTC_endpoints = nullptr;
		if ((fmt == block_format::cPVRTC1_4_RGB) || (fmt == block_format::cPVRTC1_4_RGBA))
		{
			pPVRTC_work_mem = malloc(num_blocks_x * num_blocks_y * (sizeof(decoder_etc_block) + sizeof(uint32_t)));
			if (!pPVRTC_work_mem)
			{
				BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: malloc failed\n");
				return false;
			}
			pPVRTC_endpoints = (uint32_t*) & ((decoder_etc_block*)pPVRTC_work_mem)[num_blocks_x * num_blocks_y];
		}

		if (pState->m_block_endpoint_preds[0].size() < num_blocks_x)
		{
			pState->m_block_endpoint_preds[0].resize(num_blocks_x);
			pState->m_block_endpoint_preds[1].resize(num_blocks_x);
		}

		uint32_t cur_pred_bits = 0;
		int prev_endpoint_pred_sym = 0;
		int endpoint_pred_repeat_count = 0;
		uint32_t prev_endpoint_index = 0;

		for (uint32_t block_y = 0; block_y < num_blocks_y; block_y++)
		{
			const uint32_t cur_block_endpoint_pred_array = block_y & 1;

			for (uint32_t block_x = 0; block_x < num_blocks_x; block_x++)
			{
				// Decode endpoint index predictor symbols
				if ((block_x & 1) == 0)
				{
					if ((block_y & 1) == 0)
					{
						if (endpoint_pred_repeat_count)
						{
							endpoint_pred_repeat_count--;
							cur_pred_bits = prev_endpoint_pred_sym;
						}
						else
						{
							cur_pred_bits = sym_codec.decode_huffman(m_endpoint_pred_model);
							if (cur_pred_bits == ENDPOINT_PRED_REPEAT_LAST_SYMBOL)
							{
								endpoint_pred_repeat_count = sym_codec.decode_vlc(ENDPOINT_PRED_COUNT_VLC_BITS) + ENDPOINT_PRED_MIN_REPEAT_COUNT - 1;

								cur_pred_bits = prev_endpoint_pred_sym;
							}
							else
							{
								prev_endpoint_pred_sym = cur_pred_bits;
							}
						}

						pState->m_block_endpoint_preds[cur_block_endpoint_pred_array ^ 1][block_x].m_pred_bits = (uint8_t)(cur_pred_bits >> 4);
					}
					else
					{
						cur_pred_bits = pState->m_block_endpoint_preds[cur_block_endpoint_pred_array][block_x].m_pred_bits;
					}
				}

				// Decode endpoint index
				uint32_t endpoint_index, selector_index = 0;

				const uint32_t pred = cur_pred_bits & 3;
				cur_pred_bits >>= 2;

				if (pred == 0)
				{
					// Left
					if (!block_x)
					{
						BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: invalid datastream (0)\n");
						if (pPVRTC_work_mem)
							free(pPVRTC_work_mem);
						return false;
					}

					endpoint_index = prev_endpoint_index;
				}
				else if (pred == 1)
				{
					// Upper
					if (!block_y)
					{
						BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: invalid datastream (1)\n");
						if (pPVRTC_work_mem)
							free(pPVRTC_work_mem);
						return false;
					}

					endpoint_index = pState->m_block_endpoint_preds[cur_block_endpoint_pred_array ^ 1][block_x].m_endpoint_index;
				}
				else if (pred == 2)
				{
					if (is_video)
					{
						assert(pred == CR_ENDPOINT_PRED_INDEX);
						endpoint_index = (*pPrev_frame_indices)[block_x + block_y * num_blocks_x];
						selector_index = endpoint_index >> 16;
						endpoint_index &= 0xFFFFU;
					}
					else
					{
						// Upper left
						if ((!block_x) || (!block_y))
						{
							BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: invalid datastream (2)\n");
							if (pPVRTC_work_mem)
								free(pPVRTC_work_mem);
							return false;
						}

						endpoint_index = pState->m_block_endpoint_preds[cur_block_endpoint_pred_array ^ 1][block_x - 1].m_endpoint_index;
					}
				}
				else
				{
					// Decode and apply delta
					const uint32_t delta_sym = sym_codec.decode_huffman(m_delta_endpoint_model);

					endpoint_index = delta_sym + prev_endpoint_index;
					if (endpoint_index >= m_endpoints.size())
						endpoint_index -= (int)m_endpoints.size();
				}

				pState->m_block_endpoint_preds[cur_block_endpoint_pred_array][block_x].m_endpoint_index = (uint16_t)endpoint_index;

				prev_endpoint_index = endpoint_index;

				// Decode selector index
				if ((!is_video) || (pred != CR_ENDPOINT_PRED_INDEX))
				{
					int selector_sym;
					if (cur_selector_rle_count > 0)
					{
						cur_selector_rle_count--;

						selector_sym = (int)m_selectors.size();
					}
					else
					{
						selector_sym = sym_codec.decode_huffman(m_selector_model);

						if (selector_sym == static_cast<int>(SELECTOR_HISTORY_BUF_RLE_SYMBOL_INDEX))
						{
							int run_sym = sym_codec.decode_huffman(m_selector_history_buf_rle_model);

							if (run_sym == (SELECTOR_HISTORY_BUF_RLE_COUNT_TOTAL - 1))
								cur_selector_rle_count = sym_codec.decode_vlc(7) + SELECTOR_HISTORY_BUF_RLE_COUNT_THRESH;
							else
								cur_selector_rle_count = run_sym + SELECTOR_HISTORY_BUF_RLE_COUNT_THRESH;

							if (cur_selector_rle_count > total_blocks)
							{
								// The file is corrupted or we've got a bug.
								BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: invalid datastream (3)\n");
								if (pPVRTC_work_mem)
									free(pPVRTC_work_mem);
								return false;
							}

							selector_sym = (int)m_selectors.size();

							cur_selector_rle_count--;
						}
					}

					if (selector_sym >= (int)m_selectors.size())
					{
						assert(m_selector_history_buf_size > 0);

						int history_buf_index = selector_sym - (int)m_selectors.size();

						if (history_buf_index >= (int)selector_history_buf.size())
						{
							// The file is corrupted or we've got a bug.
							BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: invalid datastream (4)\n");
							if (pPVRTC_work_mem)
								free(pPVRTC_work_mem);
							return false;
						}

						selector_index = selector_history_buf[history_buf_index];

						if (history_buf_index != 0)
							selector_history_buf.use(history_buf_index);
					}
					else
					{
						selector_index = selector_sym;

						if (m_selector_history_buf_size)
							selector_history_buf.add(selector_index);
					}
				}

				if ((endpoint_index >= m_endpoints.size()) || (selector_index >= m_selectors.size()))
				{
					// The file is corrupted or we've got a bug.
					BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: invalid datastream (5)\n");
					if (pPVRTC_work_mem)
						free(pPVRTC_work_mem);
					return false;
				}

				if (is_video)
					(*pPrev_frame_indices)[block_x + block_y * num_blocks_x] = endpoint_index | (selector_index << 16);

#if BASISD_ENABLE_DEBUG_FLAGS
				if ((g_debug_flags & cDebugFlagVisCRs) && ((fmt == block_format::cETC1) || (fmt == block_format::cBC1)))
				{
					if ((is_video) && (pred == 2))
					{
						decoder_etc_block* pDst_block = reinterpret_cast<decoder_etc_block*>(static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes);
						memset(pDst_block, 0xFF, 8);
						continue;
					}
				}
#endif

				const endpoint* pEndpoints = &m_endpoints[endpoint_index];
				const selector* pSelector = &m_selectors[selector_index];

				switch (fmt)
				{
				case block_format::cETC1:
				{
					decoder_etc_block* pDst_block = reinterpret_cast<decoder_etc_block*>(static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes);
					
					block.set_base5_color(decoder_etc_block::pack_color5(pEndpoints->m_color5, false));
					block.set_inten_table(0, pEndpoints->m_inten5);
					block.set_inten_table(1, pEndpoints->m_inten5);

					pDst_block->m_uint32[0] = block.m_uint32[0];
					pDst_block->set_raw_selector_bits(pSelector->m_bytes[0], pSelector->m_bytes[1], pSelector->m_bytes[2], pSelector->m_bytes[3]);

					break;
				}
				case block_format::cBC1:
				{
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;

#if BASISD_SUPPORT_DXT1
#if BASISD_ENABLE_DEBUG_FLAGS
					if (g_debug_flags & (cDebugFlagVisBC1Sels | cDebugFlagVisBC1Endpoints))
						convert_etc1s_to_dxt1_vis(static_cast<dxt1_block*>(pDst_block), pEndpoints, pSelector, bc1_allow_threecolor_blocks);
					else
#endif
						convert_etc1s_to_dxt1(static_cast<dxt1_block*>(pDst_block), pEndpoints, pSelector, bc1_allow_threecolor_blocks);
#else
					assert(0);
#endif
					break;
				}
				case block_format::cBC4:
				{
#if BASISD_SUPPORT_DXT5A
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					convert_etc1s_to_dxt5a(static_cast<dxt5a_block*>(pDst_block), pEndpoints, pSelector);
#else
					assert(0);
#endif
					break;
				}
				case block_format::cPVRTC1_4_RGB:
				{
#if BASISD_SUPPORT_PVRTC1
					block.set_base5_color(decoder_etc_block::pack_color5(pEndpoints->m_color5, false));
					block.set_inten_table(0, pEndpoints->m_inten5);
					block.set_inten_table(1, pEndpoints->m_inten5);
					block.set_raw_selector_bits(pSelector->m_bytes[0], pSelector->m_bytes[1], pSelector->m_bytes[2], pSelector->m_bytes[3]);

					((decoder_etc_block*)pPVRTC_work_mem)[block_x + block_y * num_blocks_x] = block;

					const color32& base_color = pEndpoints->m_color5;
					const uint32_t inten_table = pEndpoints->m_inten5;

					const uint32_t low_selector = pSelector->m_lo_selector;
					const uint32_t high_selector = pSelector->m_hi_selector;

					// Get block's RGB bounding box 
					color32 block_colors[2];
					decoder_etc_block::get_block_colors5_bounds(block_colors, base_color, inten_table, low_selector, high_selector);

					assert(block_colors[0][0] <= block_colors[1][0]);
					assert(block_colors[0][1] <= block_colors[1][1]);
					assert(block_colors[0][2] <= block_colors[1][2]);

					// Set PVRTC1 endpoints to floor/ceil of bounding box's coordinates.
					pvrtc4_block temp;
					temp.set_opaque_endpoint_floor(0, block_colors[0]);
					temp.set_opaque_endpoint_ceil(1, block_colors[1]);

					pPVRTC_endpoints[block_x + block_y * num_blocks_x] = temp.m_endpoints;
#else
					assert(0);
#endif	

					break;
				}
				case block_format::cPVRTC1_4_RGBA:
				{
#if BASISD_SUPPORT_PVRTC1
					assert(pAlpha_blocks);
					
					block.set_base5_color(decoder_etc_block::pack_color5(pEndpoints->m_color5, false));
					block.set_inten_table(0, pEndpoints->m_inten5);
					block.set_inten_table(1, pEndpoints->m_inten5);
					block.set_raw_selector_bits(pSelector->m_selectors[0], pSelector->m_selectors[1], pSelector->m_selectors[2], pSelector->m_selectors[3]);

					((decoder_etc_block*)pPVRTC_work_mem)[block_x + block_y * num_blocks_x] = block;

					// Get block's RGBA bounding box 
					const color32& base_color = pEndpoints->m_color5;
					const uint32_t inten_table = pEndpoints->m_inten5;
					const uint32_t low_selector = pSelector->m_lo_selector;
					const uint32_t high_selector = pSelector->m_hi_selector;
					color32 block_colors[2];
					decoder_etc_block::get_block_colors5_bounds(block_colors, base_color, inten_table, low_selector, high_selector);

					assert(block_colors[0][0] <= block_colors[1][0]);
					assert(block_colors[0][1] <= block_colors[1][1]);
					assert(block_colors[0][2] <= block_colors[1][2]);

					const uint16_t* pAlpha_block = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(pAlpha_blocks) + (block_x + block_y * num_blocks_x) * sizeof(uint32_t));

					const endpoint* pAlpha_endpoints = &m_endpoints[pAlpha_block[0]];
					const selector* pAlpha_selector = &m_selectors[pAlpha_block[1]];

					const color32& alpha_base_color = pAlpha_endpoints->m_color5;
					const uint32_t alpha_inten_table = pAlpha_endpoints->m_inten5;
					const uint32_t alpha_low_selector = pAlpha_selector->m_lo_selector;
					const uint32_t alpha_high_selector = pAlpha_selector->m_hi_selector;
					uint32_t alpha_block_colors[2];
					decoder_etc_block::get_block_colors5_bounds_g(alpha_block_colors, alpha_base_color, alpha_inten_table, alpha_low_selector, alpha_high_selector);
					assert(alpha_block_colors[0] <= alpha_block_colors[1]);
					block_colors[0].a = (uint8_t)alpha_block_colors[0];
					block_colors[1].a = (uint8_t)alpha_block_colors[1];

					// Set PVRTC1 endpoints to floor/ceil of bounding box's coordinates.
					pvrtc4_block temp;
					temp.set_endpoint_floor(0, block_colors[0]);
					temp.set_endpoint_ceil(1, block_colors[1]);

					pPVRTC_endpoints[block_x + block_y * num_blocks_x] = temp.m_endpoints;
#else
					assert(0);
#endif	

					break;
				}
				case block_format::cBC7_M6_OPAQUE_ONLY:
				{
#if BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					convert_etc1s_to_bc7_m6(static_cast<bc7_mode_6*>(pDst_block), pEndpoints, pSelector);
#else	
					assert(0);
#endif
					break;
				}
				case block_format::cBC7_M5_COLOR:
				{
#if BASISD_SUPPORT_BC7_MODE5
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					convert_etc1s_to_bc7_m5_color(pDst_block, pEndpoints, pSelector);
#else
					assert(0);
#endif
					break;
				}
				case block_format::cBC7_M5_ALPHA:
				{
#if BASISD_SUPPORT_BC7_MODE5
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					convert_etc1s_to_bc7_m5_alpha(pDst_block, pEndpoints, pSelector);
#else
					assert(0);
#endif
					break;
				}
				case block_format::cETC2_EAC_A8:
				{
#if BASISD_SUPPORT_ETC2_EAC_A8
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					convert_etc1s_to_etc2_eac_a8(static_cast<eac_block*>(pDst_block), pEndpoints, pSelector);
#else
					assert(0);
#endif
					break;
				}
				case block_format::cASTC_4x4:
				{
#if BASISD_SUPPORT_ASTC
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					convert_etc1s_to_astc_4x4(pDst_block, pEndpoints, pSelector, transcode_alpha, &m_endpoints[0], &m_selectors[0]);
#else
					assert(0);
#endif
					break;
				}
				case block_format::cATC_RGB:
				{
#if BASISD_SUPPORT_ATC
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					convert_etc1s_to_atc(pDst_block, pEndpoints, pSelector);
#else
					assert(0);
#endif
					break;
				}
				case block_format::cFXT1_RGB:
				{
#if BASISD_SUPPORT_FXT1
					const uint32_t fxt1_block_x = block_x >> 1;
					const uint32_t fxt1_block_y = block_y;
					const uint32_t fxt1_subblock = block_x & 1;

					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (fxt1_block_x + fxt1_block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;

					convert_etc1s_to_fxt1(pDst_block, pEndpoints, pSelector, fxt1_subblock);
#else
					assert(0);
#endif
					break;
				}
				case block_format::cPVRTC2_4_RGB:
				{
#if BASISD_SUPPORT_PVRTC2
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					convert_etc1s_to_pvrtc2_rgb(pDst_block, pEndpoints, pSelector);
#endif
					break;
				}
				case block_format::cPVRTC2_4_RGBA:
				{
#if BASISD_SUPPORT_PVRTC2
					assert(transcode_alpha);

					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					
					convert_etc1s_to_pvrtc2_rgba(pDst_block, pEndpoints, pSelector, &m_endpoints[0], &m_selectors[0]);
#endif
					break;
				}
				case block_format::cIndices:
				{
					uint16_t* pDst_block = reinterpret_cast<uint16_t *>(static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes);
					pDst_block[0] = static_cast<uint16_t>(endpoint_index);
					pDst_block[1] = static_cast<uint16_t>(selector_index);
					break;
				}
				case block_format::cA32:
				{
					assert(sizeof(uint32_t) == output_block_or_pixel_stride_in_bytes);
					uint8_t* pDst_pixels = static_cast<uint8_t*>(pDst_blocks) + (block_x * 4 + block_y * 4 * output_row_pitch_in_blocks_or_pixels) * sizeof(uint32_t);
										
					const uint32_t max_x = basisu::minimum<int>(4, output_row_pitch_in_blocks_or_pixels - block_x * 4);
					const uint32_t max_y = basisu::minimum<int>(4, output_rows_in_pixels - block_y * 4);
					
					int colors[4];
					decoder_etc_block::get_block_colors5_g(colors, pEndpoints->m_color5, pEndpoints->m_inten5);

					if (max_x == 4)
					{
						for (uint32_t y = 0; y < max_y; y++)
						{
							const uint32_t s = pSelector->m_selectors[y];

							pDst_pixels[3] = static_cast<uint8_t>(colors[s & 3]);
							pDst_pixels[3+4] = static_cast<uint8_t>(colors[(s >> 2) & 3]);
							pDst_pixels[3+8] = static_cast<uint8_t>(colors[(s >> 4) & 3]);
							pDst_pixels[3+12] = static_cast<uint8_t>(colors[(s >> 6) & 3]);
							
							pDst_pixels += output_row_pitch_in_blocks_or_pixels * sizeof(uint32_t);
						}
					}
					else
					{
						for (uint32_t y = 0; y < max_y; y++)
						{
							const uint32_t s = pSelector->m_selectors[y];

							for (uint32_t x = 0; x < max_x; x++)
								pDst_pixels[3 + 4 * x] = static_cast<uint8_t>(colors[(s >> (x * 2)) & 3]);

							pDst_pixels += output_row_pitch_in_blocks_or_pixels * sizeof(uint32_t);
						}
					}

					break;
				}
				case block_format::cRGB32:
				{
					assert(sizeof(uint32_t) == output_block_or_pixel_stride_in_bytes);
					uint8_t* pDst_pixels = static_cast<uint8_t*>(pDst_blocks) + (block_x * 4 + block_y * 4 * output_row_pitch_in_blocks_or_pixels) * sizeof(uint32_t);

					const uint32_t max_x = basisu::minimum<int>(4, output_row_pitch_in_blocks_or_pixels - block_x * 4);
					const uint32_t max_y = basisu::minimum<int>(4, output_rows_in_pixels - block_y * 4);

					color32 colors[4];
					decoder_etc_block::get_block_colors5(colors, pEndpoints->m_color5, pEndpoints->m_inten5);
					
					for (uint32_t y = 0; y < max_y; y++)
					{
						const uint32_t s = pSelector->m_selectors[y];

						for (uint32_t x = 0; x < max_x; x++)
						{
							const color32& c = colors[(s >> (x * 2)) & 3];

							pDst_pixels[0 + 4 * x] = c.r;
							pDst_pixels[1 + 4 * x] = c.g;
							pDst_pixels[2 + 4 * x] = c.b;
						}

						pDst_pixels += output_row_pitch_in_blocks_or_pixels * sizeof(uint32_t);
					}

					break;
				}
				case block_format::cRGBA32:
				{
					assert(sizeof(uint32_t) == output_block_or_pixel_stride_in_bytes);
					uint8_t* pDst_pixels = static_cast<uint8_t*>(pDst_blocks) + (block_x * 4 + block_y * 4 * output_row_pitch_in_blocks_or_pixels) * sizeof(uint32_t);

					const uint32_t max_x = basisu::minimum<int>(4, output_row_pitch_in_blocks_or_pixels - block_x * 4);
					const uint32_t max_y = basisu::minimum<int>(4, output_rows_in_pixels - block_y * 4);

					color32 colors[4];
					decoder_etc_block::get_block_colors5(colors, pEndpoints->m_color5, pEndpoints->m_inten5);

					for (uint32_t y = 0; y < max_y; y++)
					{
						const uint32_t s = pSelector->m_selectors[y];

						for (uint32_t x = 0; x < max_x; x++)
						{
							const color32& c = colors[(s >> (x * 2)) & 3];

							pDst_pixels[0 + 4 * x] = c.r;
							pDst_pixels[1 + 4 * x] = c.g;
							pDst_pixels[2 + 4 * x] = c.b;
							pDst_pixels[3 + 4 * x] = 255;
						}

						pDst_pixels += output_row_pitch_in_blocks_or_pixels * sizeof(uint32_t);
					}

					break;
				}
				case block_format::cRGB565:
				case block_format::cBGR565:
				{
					assert(sizeof(uint16_t) == output_block_or_pixel_stride_in_bytes);
					uint8_t* pDst_pixels = static_cast<uint8_t*>(pDst_blocks) + (block_x * 4 + block_y * 4 * output_row_pitch_in_blocks_or_pixels) * sizeof(uint16_t);

					const uint32_t max_x = basisu::minimum<int>(4, output_row_pitch_in_blocks_or_pixels - block_x * 4);
					const uint32_t max_y = basisu::minimum<int>(4, output_rows_in_pixels - block_y * 4);

					color32 colors[4];
					decoder_etc_block::get_block_colors5(colors, pEndpoints->m_color5, pEndpoints->m_inten5);

					uint16_t packed_colors[4];
					if (fmt == block_format::cRGB565)
					{
						for (uint32_t i = 0; i < 4; i++)
							packed_colors[i] = static_cast<uint16_t>(((colors[i].r >> 3) << 11) | ((colors[i].g >> 2) << 5) | (colors[i].b >> 3));
					}
					else
					{
						for (uint32_t i = 0; i < 4; i++)
							packed_colors[i] = static_cast<uint16_t>(((colors[i].b >> 3) << 11) | ((colors[i].g >> 2) << 5) | (colors[i].r >> 3));
					}

					for (uint32_t y = 0; y < max_y; y++)
					{
						const uint32_t s = pSelector->m_selectors[y];

						for (uint32_t x = 0; x < max_x; x++)
							reinterpret_cast<uint16_t *>(pDst_pixels)[x] = packed_colors[(s >> (x * 2)) & 3];

						pDst_pixels += output_row_pitch_in_blocks_or_pixels * sizeof(uint16_t);
					}

					break;
				}
				case block_format::cRGBA4444_COLOR:
				{
					assert(sizeof(uint16_t) == output_block_or_pixel_stride_in_bytes);
					uint8_t* pDst_pixels = static_cast<uint8_t*>(pDst_blocks) + (block_x * 4 + block_y * 4 * output_row_pitch_in_blocks_or_pixels) * sizeof(uint16_t);

					const uint32_t max_x = basisu::minimum<int>(4, output_row_pitch_in_blocks_or_pixels - block_x * 4);
					const uint32_t max_y = basisu::minimum<int>(4, output_rows_in_pixels - block_y * 4);

					color32 colors[4];
					decoder_etc_block::get_block_colors5(colors, pEndpoints->m_color5, pEndpoints->m_inten5);

					uint16_t packed_colors[4];
					for (uint32_t i = 0; i < 4; i++)
						packed_colors[i] = static_cast<uint16_t>(((colors[i].r >> 4) << 12) | ((colors[i].g >> 4) << 8) | ((colors[i].b >> 4) << 4));

					for (uint32_t y = 0; y < max_y; y++)
					{
						const uint32_t s = pSelector->m_selectors[y];

						for (uint32_t x = 0; x < max_x; x++)
						{
							uint16_t cur = reinterpret_cast<uint16_t*>(pDst_pixels)[x];
							cur = (cur & 0xF) | packed_colors[(s >> (x * 2)) & 3];
							reinterpret_cast<uint16_t*>(pDst_pixels)[x] = cur;
						}

						pDst_pixels += output_row_pitch_in_blocks_or_pixels * sizeof(uint16_t);
					}

					break;
				}
				case block_format::cRGBA4444_COLOR_OPAQUE:
				{
					assert(sizeof(uint16_t) == output_block_or_pixel_stride_in_bytes);
					uint8_t* pDst_pixels = static_cast<uint8_t*>(pDst_blocks) + (block_x * 4 + block_y * 4 * output_row_pitch_in_blocks_or_pixels) * sizeof(uint16_t);

					const uint32_t max_x = basisu::minimum<int>(4, output_row_pitch_in_blocks_or_pixels - block_x * 4);
					const uint32_t max_y = basisu::minimum<int>(4, output_rows_in_pixels - block_y * 4);

					color32 colors[4];
					decoder_etc_block::get_block_colors5(colors, pEndpoints->m_color5, pEndpoints->m_inten5);

					uint16_t packed_colors[4];
					for (uint32_t i = 0; i < 4; i++)
						packed_colors[i] = static_cast<uint16_t>(((colors[i].r >> 4) << 12) | ((colors[i].g >> 4) << 8) | ((colors[i].b >> 4) << 4) |  0xF);

					for (uint32_t y = 0; y < max_y; y++)
					{
						const uint32_t s = pSelector->m_selectors[y];

						for (uint32_t x = 0; x < max_x; x++)
							reinterpret_cast<uint16_t*>(pDst_pixels)[x] = packed_colors[(s >> (x * 2)) & 3];

						pDst_pixels += output_row_pitch_in_blocks_or_pixels * sizeof(uint16_t);
					}

					break;
				}
				case block_format::cRGBA4444_ALPHA:
				{
					assert(sizeof(uint16_t) == output_block_or_pixel_stride_in_bytes);
					uint8_t* pDst_pixels = static_cast<uint8_t*>(pDst_blocks) + (block_x * 4 + block_y * 4 * output_row_pitch_in_blocks_or_pixels) * sizeof(uint16_t);

					const uint32_t max_x = basisu::minimum<int>(4, output_row_pitch_in_blocks_or_pixels - block_x * 4);
					const uint32_t max_y = basisu::minimum<int>(4, output_rows_in_pixels - block_y * 4);

					color32 colors[4];
					decoder_etc_block::get_block_colors5(colors, pEndpoints->m_color5, pEndpoints->m_inten5);

					uint16_t packed_colors[4];
					for (uint32_t i = 0; i < 4; i++)
						packed_colors[i] = colors[i].g >> 4;

					for (uint32_t y = 0; y < max_y; y++)
					{
						const uint32_t s = pSelector->m_selectors[y];

						for (uint32_t x = 0; x < max_x; x++)
							reinterpret_cast<uint16_t*>(pDst_pixels)[x] = packed_colors[(s >> (x * 2)) & 3];

						pDst_pixels += output_row_pitch_in_blocks_or_pixels * sizeof(uint16_t);
					}

					break;
				}
				case block_format::cETC2_EAC_R11:
				{
#if BASISD_SUPPORT_ETC2_EAC_RG11
					void* pDst_block = static_cast<uint8_t*>(pDst_blocks) + (block_x + block_y * output_row_pitch_in_blocks_or_pixels) * output_block_or_pixel_stride_in_bytes;
					convert_etc1s_to_etc2_eac_r11(static_cast<eac_block*>(pDst_block), pEndpoints, pSelector);
#else
					assert(0);
#endif
					break;
				}
				default:
				{
					assert(0);
					break;
				}
				}

			} // block_x

		} // block-y

		if (endpoint_pred_repeat_count != 0)
		{
			BASISU_DEVEL_ERROR("basisu_lowlevel_transcoder::transcode_slice: endpoint_pred_repeat_count != 0. The file is corrupted or this is a bug\n");
			return false;
		}

		//assert(endpoint_pred_repeat_count == 0);

#if BASISD_SUPPORT_PVRTC1
		// PVRTC post process - create per-pixel modulation values.
		if (fmt == block_format::cPVRTC1_4_RGB)
			fixup_pvrtc1_4_modulation_rgb((decoder_etc_block*)pPVRTC_work_mem, pPVRTC_endpoints, pDst_blocks, num_blocks_x, num_blocks_y);
		else if (fmt == block_format::cPVRTC1_4_RGBA)
			fixup_pvrtc1_4_modulation_rgba((decoder_etc_block*)pPVRTC_work_mem, pPVRTC_endpoints, pDst_blocks, num_blocks_x, num_blocks_y, pAlpha_blocks, &m_endpoints[0], &m_selectors[0]);
#endif // BASISD_SUPPORT_PVRTC1

		if (pPVRTC_work_mem)
			free(pPVRTC_work_mem);

		return true;
	}

	basisu_transcoder::basisu_transcoder(const etc1_global_selector_codebook* pGlobal_sel_codebook) :
		m_lowlevel_decoder(pGlobal_sel_codebook)
	{
	}

	bool basisu_transcoder::validate_file_checksums(const void* pData, uint32_t data_size, bool full_validation) const
	{
		if (!validate_header(pData, data_size))
			return false;

		const basis_file_header* pHeader = reinterpret_cast<const basis_file_header*>(pData);

#if !BASISU_NO_HEADER_OR_DATA_CRC16_CHECKS
		if (crc16(&pHeader->m_data_size, sizeof(basis_file_header) - BASISU_OFFSETOF(basis_file_header, m_data_size), 0) != pHeader->m_header_crc16)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: header CRC check failed\n");
			return false;
		}

		if (full_validation)
		{
			if (crc16(reinterpret_cast<const uint8_t*>(pData) + sizeof(basis_file_header), pHeader->m_data_size, 0) != pHeader->m_data_crc16)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: data CRC check failed\n");
				return false;
			}
		}
#endif		

		return true;
	}

	bool basisu_transcoder::validate_header_quick(const void* pData, uint32_t data_size) const
	{
		if (data_size <= sizeof(basis_file_header))
			return false;

		const basis_file_header* pHeader = reinterpret_cast<const basis_file_header*>(pData);

		if ((pHeader->m_sig != basis_file_header::cBASISSigValue) || (pHeader->m_ver != BASISD_SUPPORTED_BASIS_VERSION) || (pHeader->m_header_size != sizeof(basis_file_header)))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: header has an invalid signature, or file version is unsupported\n");
			return false;
		}

		uint32_t expected_file_size = sizeof(basis_file_header) + pHeader->m_data_size;
		if (data_size < expected_file_size)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: source buffer is too small\n");
			return false;
		}

		if ((!pHeader->m_total_slices) || (!pHeader->m_total_images))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::validate_header_quick: header is invalid\n");
			return false;
		}

		if ((pHeader->m_slice_desc_file_ofs >= data_size) ||
			((data_size - pHeader->m_slice_desc_file_ofs) < (sizeof(basis_slice_desc) * pHeader->m_total_slices))
			)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::validate_header_quick: passed in buffer is too small or data is corrupted\n");
			return false;
		}

		return true;
	}

	bool basisu_transcoder::validate_header(const void* pData, uint32_t data_size) const
	{
		if (data_size <= sizeof(basis_file_header))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: input source buffer is too small\n");
			return false;
		}

		const basis_file_header* pHeader = reinterpret_cast<const basis_file_header*>(pData);

		if ((pHeader->m_sig != basis_file_header::cBASISSigValue) || (pHeader->m_ver != BASISD_SUPPORTED_BASIS_VERSION) || (pHeader->m_header_size != sizeof(basis_file_header)))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: header has an invalid signature, or file version is unsupported\n");
			return false;
		}

		uint32_t expected_file_size = sizeof(basis_file_header) + pHeader->m_data_size;
		if (data_size < expected_file_size)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: input source buffer is too small, or header is corrupted\n");
			return false;
		}

		if ((!pHeader->m_total_images) || (!pHeader->m_total_slices))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: invalid basis file (total images or slices are 0)\n");
			return false;
		}

		if (pHeader->m_total_images > pHeader->m_total_slices)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: invalid basis file (too many images)\n");
			return false;
		}

		if (pHeader->m_flags & cBASISHeaderFlagHasAlphaSlices)
		{
			if (pHeader->m_total_slices & 1)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: invalid alpha basis file\n");
				return false;
			}
		}

		if ((pHeader->m_flags & cBASISHeaderFlagETC1S) == 0)
		{
			// We only support ETC1S in basis universal
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: invalid basis file (ETC1S flag check)\n");
			return false;
		}

		if ((pHeader->m_slice_desc_file_ofs >= data_size) ||
			((data_size - pHeader->m_slice_desc_file_ofs) < (sizeof(basis_slice_desc) * pHeader->m_total_slices))
			)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::validate_header_quick: passed in buffer is too small or data is corrupted\n");
			return false;
		}

		return true;
	}

	basis_texture_type basisu_transcoder::get_texture_type(const void* pData, uint32_t data_size) const
	{
		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_texture_type: header validation failed\n");
			return cBASISTexType2DArray;
		}

		const basis_file_header* pHeader = static_cast<const basis_file_header*>(pData);

		basis_texture_type btt = static_cast<basis_texture_type>(static_cast<uint8_t>(pHeader->m_tex_type));

		if (btt >= cBASISTexTypeTotal)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::validate_header_quick: header's texture type field is invalid\n");
			return cBASISTexType2DArray;
		}

		return btt;
	}

	bool basisu_transcoder::get_userdata(const void* pData, uint32_t data_size, uint32_t& userdata0, uint32_t& userdata1) const
	{
		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_userdata: header validation failed\n");
			return false;
		}

		const basis_file_header* pHeader = static_cast<const basis_file_header*>(pData);

		userdata0 = pHeader->m_userdata0;
		userdata1 = pHeader->m_userdata1;
		return true;
	}

	uint32_t basisu_transcoder::get_total_images(const void* pData, uint32_t data_size) const
	{
		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_images: header validation failed\n");
			return 0;
		}

		const basis_file_header* pHeader = static_cast<const basis_file_header*>(pData);

		return pHeader->m_total_images;
	}

	bool basisu_transcoder::get_image_info(const void* pData, uint32_t data_size, basisu_image_info& image_info, uint32_t image_index) const
	{
		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_info: header validation failed\n");
			return false;
		}

		int slice_index = find_first_slice_index(pData, data_size, image_index, 0);
		if (slice_index < 0)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_info: invalid slice index\n");
			return false;
		}

		const basis_file_header* pHeader = static_cast<const basis_file_header*>(pData);

		if (image_index >= pHeader->m_total_images)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_info: invalid image_index\n");
			return false;
		}

		const basis_slice_desc* pSlice_descs = reinterpret_cast<const basis_slice_desc*>(static_cast<const uint8_t*>(pData) + pHeader->m_slice_desc_file_ofs);

		uint32_t total_levels = 1;
		for (uint32_t i = slice_index + 1; i < pHeader->m_total_slices; i++)
			if (pSlice_descs[i].m_image_index == image_index)
				total_levels = basisu::maximum<uint32_t>(total_levels, pSlice_descs[i].m_level_index + 1);
			else
				break;

		if (total_levels > 16)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_info: invalid image_index\n");
			return false;
		}

		const basis_slice_desc& slice_desc = pSlice_descs[slice_index];

		image_info.m_image_index = image_index;
		image_info.m_total_levels = total_levels;
		image_info.m_alpha_flag = (pHeader->m_flags & cBASISHeaderFlagHasAlphaSlices) != 0;
		image_info.m_iframe_flag = (slice_desc.m_flags & cSliceDescFlagsFrameIsIFrame) != 0;
		image_info.m_width = slice_desc.m_num_blocks_x * 4;
		image_info.m_height = slice_desc.m_num_blocks_y * 4;
		image_info.m_orig_width = slice_desc.m_orig_width;
		image_info.m_orig_height = slice_desc.m_orig_height;
		image_info.m_num_blocks_x = slice_desc.m_num_blocks_x;
		image_info.m_num_blocks_y = slice_desc.m_num_blocks_y;
		image_info.m_total_blocks = image_info.m_num_blocks_x * image_info.m_num_blocks_y;
		image_info.m_first_slice_index = slice_index;

		return true;
	}

	uint32_t basisu_transcoder::get_total_image_levels(const void* pData, uint32_t data_size, uint32_t image_index) const
	{
		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_image_levels: header validation failed\n");
			return false;
		}

		int slice_index = find_first_slice_index(pData, data_size, image_index, 0);
		if (slice_index < 0)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_image_levels: failed finding slice\n");
			return false;
		}

		const basis_file_header* pHeader = static_cast<const basis_file_header*>(pData);

		if (image_index >= pHeader->m_total_images)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_image_levels: invalid image_index\n");
			return false;
		}

		const basis_slice_desc* pSlice_descs = reinterpret_cast<const basis_slice_desc*>(static_cast<const uint8_t*>(pData) + pHeader->m_slice_desc_file_ofs);

		uint32_t total_levels = 1;
		for (uint32_t i = slice_index + 1; i < pHeader->m_total_slices; i++)
			if (pSlice_descs[i].m_image_index == image_index)
				total_levels = basisu::maximum<uint32_t>(total_levels, pSlice_descs[i].m_level_index + 1);
			else
				break;

		const uint32_t cMaxSupportedLevels = 16;
		if (total_levels > cMaxSupportedLevels)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_total_image_levels: invalid image levels!\n");
			return false;
		}

		return total_levels;
	}

	bool basisu_transcoder::get_image_level_desc(const void* pData, uint32_t data_size, uint32_t image_index, uint32_t level_index, uint32_t& orig_width, uint32_t& orig_height, uint32_t& total_blocks) const
	{
		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_level_desc: header validation failed\n");
			return false;
		}

		int slice_index = find_first_slice_index(pData, data_size, image_index, level_index);
		if (slice_index < 0)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_level_desc: failed finding slice\n");
			return false;
		}

		const basis_file_header* pHeader = static_cast<const basis_file_header*>(pData);

		if (image_index >= pHeader->m_total_images)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_level_desc: invalid image_index\n");
			return false;
		}

		const basis_slice_desc* pSlice_descs = reinterpret_cast<const basis_slice_desc*>(static_cast<const uint8_t*>(pData) + pHeader->m_slice_desc_file_ofs);

		const basis_slice_desc& slice_desc = pSlice_descs[slice_index];

		orig_width = slice_desc.m_orig_width;
		orig_height = slice_desc.m_orig_height;
		total_blocks = slice_desc.m_num_blocks_x * slice_desc.m_num_blocks_y;

		return true;
	}

	bool basisu_transcoder::get_image_level_info(const void* pData, uint32_t data_size, basisu_image_level_info& image_info, uint32_t image_index, uint32_t level_index) const
	{
		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_level_info: validate_file_checksums failed\n");
			return false;
		}

		int slice_index = find_first_slice_index(pData, data_size, image_index, level_index);
		if (slice_index < 0)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_level_info: failed finding slice\n");
			return false;
		}

		const basis_file_header* pHeader = static_cast<const basis_file_header*>(pData);

		if (image_index >= pHeader->m_total_images)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_image_level_info: invalid image_index\n");
			return false;
		}

		const basis_slice_desc* pSlice_descs = reinterpret_cast<const basis_slice_desc*>(static_cast<const uint8_t*>(pData) + pHeader->m_slice_desc_file_ofs);

		const basis_slice_desc& slice_desc = pSlice_descs[slice_index];

		image_info.m_image_index = image_index;
		image_info.m_level_index = level_index;
		image_info.m_alpha_flag = (pHeader->m_flags & cBASISHeaderFlagHasAlphaSlices) != 0;
		image_info.m_iframe_flag = (slice_desc.m_flags & cSliceDescFlagsFrameIsIFrame) != 0;
		image_info.m_width = slice_desc.m_num_blocks_x * 4;
		image_info.m_height = slice_desc.m_num_blocks_y * 4;
		image_info.m_orig_width = slice_desc.m_orig_width;
		image_info.m_orig_height = slice_desc.m_orig_height;
		image_info.m_num_blocks_x = slice_desc.m_num_blocks_x;
		image_info.m_num_blocks_y = slice_desc.m_num_blocks_y;
		image_info.m_total_blocks = image_info.m_num_blocks_x * image_info.m_num_blocks_y;
		image_info.m_first_slice_index = slice_index;

		return true;
	}

	bool basisu_transcoder::get_file_info(const void* pData, uint32_t data_size, basisu_file_info& file_info) const
	{
		if (!validate_file_checksums(pData, data_size, false))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_file_info: validate_file_checksums failed\n");
			return false;
		}

		const basis_file_header* pHeader = static_cast<const basis_file_header*>(pData);
		const basis_slice_desc* pSlice_descs = reinterpret_cast<const basis_slice_desc*>(static_cast<const uint8_t*>(pData) + pHeader->m_slice_desc_file_ofs);

		file_info.m_version = pHeader->m_ver;

		file_info.m_total_header_size = sizeof(basis_file_header) + pHeader->m_total_slices * sizeof(basis_slice_desc);

		file_info.m_total_selectors = pHeader->m_total_selectors;
		file_info.m_selector_codebook_size = pHeader->m_selector_cb_file_size;

		file_info.m_total_endpoints = pHeader->m_total_endpoints;
		file_info.m_endpoint_codebook_size = pHeader->m_endpoint_cb_file_size;

		file_info.m_tables_size = pHeader->m_tables_file_size;

		file_info.m_etc1s = (pHeader->m_flags & cBASISHeaderFlagETC1S) != 0;
		file_info.m_y_flipped = (pHeader->m_flags & cBASISHeaderFlagYFlipped) != 0;
		file_info.m_has_alpha_slices = (pHeader->m_flags & cBASISHeaderFlagHasAlphaSlices) != 0;

		const uint32_t total_slices = pHeader->m_total_slices;

		file_info.m_slice_info.resize(total_slices);

		file_info.m_slices_size = 0;

		file_info.m_tex_type = static_cast<basis_texture_type>(static_cast<uint8_t>(pHeader->m_tex_type));

		if (file_info.m_tex_type > cBASISTexTypeTotal)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::get_file_info: invalid texture type, file is corrupted\n");
			return false;
		}

		file_info.m_us_per_frame = pHeader->m_us_per_frame;
		file_info.m_userdata0 = pHeader->m_userdata0;
		file_info.m_userdata1 = pHeader->m_userdata1;

		file_info.m_image_mipmap_levels.resize(0);
		file_info.m_image_mipmap_levels.resize(pHeader->m_total_images);

		file_info.m_total_images = pHeader->m_total_images;

		for (uint32_t i = 0; i < total_slices; i++)
		{
			file_info.m_slices_size += pSlice_descs[i].m_file_size;

			basisu_slice_info& slice_info = file_info.m_slice_info[i];

			slice_info.m_orig_width = pSlice_descs[i].m_orig_width;
			slice_info.m_orig_height = pSlice_descs[i].m_orig_height;
			slice_info.m_width = pSlice_descs[i].m_num_blocks_x * 4;
			slice_info.m_height = pSlice_descs[i].m_num_blocks_y * 4;
			slice_info.m_num_blocks_x = pSlice_descs[i].m_num_blocks_x;
			slice_info.m_num_blocks_y = pSlice_descs[i].m_num_blocks_y;
			slice_info.m_total_blocks = slice_info.m_num_blocks_x * slice_info.m_num_blocks_y;
			slice_info.m_compressed_size = pSlice_descs[i].m_file_size;
			slice_info.m_slice_index = i;
			slice_info.m_image_index = pSlice_descs[i].m_image_index;
			slice_info.m_level_index = pSlice_descs[i].m_level_index;
			slice_info.m_unpacked_slice_crc16 = pSlice_descs[i].m_slice_data_crc16;
			slice_info.m_alpha_flag = (pSlice_descs[i].m_flags & cSliceDescFlagsIsAlphaData) != 0;
			slice_info.m_iframe_flag = (pSlice_descs[i].m_flags & cSliceDescFlagsFrameIsIFrame) != 0;

			if (pSlice_descs[i].m_image_index >= pHeader->m_total_images)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::get_file_info: slice desc's image index is invalid\n");
				return false;
			}

			file_info.m_image_mipmap_levels[pSlice_descs[i].m_image_index] = basisu::maximum<uint32_t>(file_info.m_image_mipmap_levels[pSlice_descs[i].m_image_index], pSlice_descs[i].m_level_index + 1);

			if (file_info.m_image_mipmap_levels[pSlice_descs[i].m_image_index] > 16)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::get_file_info: slice mipmap level is invalid\n");
				return false;
			}
		}

		return true;
	}

	bool basisu_transcoder::start_transcoding(const void* pData, uint32_t data_size) const
	{
		if (m_lowlevel_decoder.m_endpoints.size())
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::start_transcoding: already called start_transcoding\n");
			return true;
		}

		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::start_transcoding: header validation failed\n");
			return false;
		}

		const basis_file_header* pHeader = reinterpret_cast<const basis_file_header*>(pData);

		const uint8_t* pDataU8 = static_cast<const uint8_t*>(pData);

		if (!pHeader->m_endpoint_cb_file_size || !pHeader->m_selector_cb_file_size || !pHeader->m_tables_file_size)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::start_transcoding: file is corrupted (0)\n");
		}

		if ((pHeader->m_endpoint_cb_file_ofs > data_size) || (pHeader->m_selector_cb_file_ofs > data_size) || (pHeader->m_tables_file_ofs > data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::start_transcoding: file is corrupted or passed in buffer too small (1)\n");
			return false;
		}

		if (pHeader->m_endpoint_cb_file_size > (data_size - pHeader->m_endpoint_cb_file_ofs))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::start_transcoding: file is corrupted or passed in buffer too small (2)\n");
			return false;
		}

		if (pHeader->m_selector_cb_file_size > (data_size - pHeader->m_selector_cb_file_ofs))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::start_transcoding: file is corrupted or passed in buffer too small (3)\n");
			return false;
		}

		if (pHeader->m_tables_file_size > (data_size - pHeader->m_tables_file_ofs))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::start_transcoding: file is corrupted or passed in buffer too small (3)\n");
			return false;
		}

		if (!m_lowlevel_decoder.decode_palettes(
			pHeader->m_total_endpoints, pDataU8 + pHeader->m_endpoint_cb_file_ofs, pHeader->m_endpoint_cb_file_size,
			pHeader->m_total_selectors, pDataU8 + pHeader->m_selector_cb_file_ofs, pHeader->m_selector_cb_file_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::start_transcoding: decode_palettes failed\n");
			return false;
		}

		if (!m_lowlevel_decoder.decode_tables(pDataU8 + pHeader->m_tables_file_ofs, pHeader->m_tables_file_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::start_transcoding: decode_tables failed\n");
			return false;
		}

		return true;
	}

	bool basisu_transcoder::transcode_slice(const void* pData, uint32_t data_size, uint32_t slice_index, void* pOutput_blocks, uint32_t output_blocks_buf_size_in_blocks_or_pixels, block_format fmt,
		uint32_t output_block_or_pixel_stride_in_bytes, uint32_t decode_flags, uint32_t output_row_pitch_in_blocks_or_pixels, basisu_transcoder_state* pState, void *pAlpha_blocks, uint32_t output_rows_in_pixels) const
	{
		if (!m_lowlevel_decoder.m_endpoints.size())
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: must call start_transcoding first\n");
			return false;
		}

		if (decode_flags & cDecodeFlagsPVRTCDecodeToNextPow2)
		{
			// TODO: Not yet supported
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: cDecodeFlagsPVRTCDecodeToNextPow2 currently unsupported\n");
			return false;
		}

		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: header validation failed\n");
			return false;
		}

		const basis_file_header* pHeader = reinterpret_cast<const basis_file_header*>(pData);

		const uint8_t* pDataU8 = static_cast<const uint8_t*>(pData);

		if (slice_index >= pHeader->m_total_slices)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: slice_index >= pHeader->m_total_slices\n");
			return false;
		}

		const basis_slice_desc& slice_desc = reinterpret_cast<const basis_slice_desc*>(pDataU8 + pHeader->m_slice_desc_file_ofs)[slice_index];

		uint32_t total_4x4_blocks = slice_desc.m_num_blocks_x * slice_desc.m_num_blocks_y;
		
		if (basis_block_format_is_uncompressed(fmt))
		{
			// Assume the output buffer is orig_width by orig_height
			if (!output_row_pitch_in_blocks_or_pixels)
				output_row_pitch_in_blocks_or_pixels = slice_desc.m_orig_width;

			if (!output_rows_in_pixels)
				output_rows_in_pixels = slice_desc.m_orig_height;

			// Now make sure the output buffer is large enough, or we'll overwrite memory.
			if (output_blocks_buf_size_in_blocks_or_pixels < (output_rows_in_pixels * output_row_pitch_in_blocks_or_pixels))
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: output_blocks_buf_size_in_blocks_or_pixels < (output_rows_in_pixels * output_row_pitch_in_blocks_or_pixels)\n");
				return false;
			}
		}
		else if (fmt == block_format::cFXT1_RGB)
		{
			const uint32_t num_blocks_fxt1_x = (slice_desc.m_orig_width + 7) / 8;
			const uint32_t num_blocks_fxt1_y = (slice_desc.m_orig_height + 3) / 4;
			const uint32_t total_blocks_fxt1 = num_blocks_fxt1_x * num_blocks_fxt1_y;

			if (output_blocks_buf_size_in_blocks_or_pixels < total_blocks_fxt1)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: output_blocks_buf_size_in_blocks_or_pixels < total_blocks_fxt1\n");
				return false;
			}
		}
		else
		{
			if (output_blocks_buf_size_in_blocks_or_pixels < total_4x4_blocks)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: output_blocks_buf_size_in_blocks_or_pixels < total_blocks\n");
				return false;
			}
		}

		if (fmt != block_format::cETC1)
		{
			if ((fmt == block_format::cPVRTC1_4_RGB) || (fmt == block_format::cPVRTC1_4_RGBA))
			{
				if ((!basisu::is_pow2(slice_desc.m_num_blocks_x * 4)) || (!basisu::is_pow2(slice_desc.m_num_blocks_y * 4)))
				{
					// PVRTC1 only supports power of 2 dimensions
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: PVRTC1 only supports power of 2 dimensions\n");
					return false;
				}
			}
		}

		if (slice_desc.m_file_ofs > data_size)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: invalid slice_desc.m_file_ofs, or passed in buffer too small\n");
			return false;
		}

		const uint32_t data_size_left = data_size - slice_desc.m_file_ofs;
		if (data_size_left < slice_desc.m_file_size)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_slice: invalid slice_desc.m_file_size, or passed in buffer too small\n");
			return false;
		}

		return m_lowlevel_decoder.transcode_slice(pOutput_blocks, slice_desc.m_num_blocks_x, slice_desc.m_num_blocks_y,
			pDataU8 + slice_desc.m_file_ofs, slice_desc.m_file_size,
			fmt, output_block_or_pixel_stride_in_bytes, (decode_flags & cDecodeFlagsBC1ForbidThreeColorBlocks) == 0, *pHeader, slice_desc, output_row_pitch_in_blocks_or_pixels, pState,
			(decode_flags & cDecodeFlagsOutputHasAlphaIndices) != 0, pAlpha_blocks, output_rows_in_pixels);
	}

	int basisu_transcoder::find_first_slice_index(const void* pData, uint32_t data_size, uint32_t image_index, uint32_t level_index) const
	{
		(void)data_size;

		const basis_file_header* pHeader = reinterpret_cast<const basis_file_header*>(pData);
		const uint8_t* pDataU8 = static_cast<const uint8_t*>(pData);

		// For very large basis files this search could be painful
		// TODO: Binary search this
		for (uint32_t slice_iter = 0; slice_iter < pHeader->m_total_slices; slice_iter++)
		{
			const basis_slice_desc& slice_desc = reinterpret_cast<const basis_slice_desc*>(pDataU8 + pHeader->m_slice_desc_file_ofs)[slice_iter];
			if ((slice_desc.m_image_index == image_index) && (slice_desc.m_level_index == level_index))
				return slice_iter;
		}

		BASISU_DEVEL_ERROR("basisu_transcoder::find_first_slice_index: didn't find slice\n");

		return -1;
	}

	int basisu_transcoder::find_slice(const void* pData, uint32_t data_size, uint32_t image_index, uint32_t level_index, bool alpha_data) const
	{
		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::find_slice: header validation failed\n");
			return false;
		}

		const basis_file_header* pHeader = reinterpret_cast<const basis_file_header*>(pData);
		const uint8_t* pDataU8 = static_cast<const uint8_t*>(pData);
		const basis_slice_desc* pSlice_descs = reinterpret_cast<const basis_slice_desc*>(pDataU8 + pHeader->m_slice_desc_file_ofs);

		// For very large basis files this search could be painful
		// TODO: Binary search this
		for (uint32_t slice_iter = 0; slice_iter < pHeader->m_total_slices; slice_iter++)
		{
			const basis_slice_desc& slice_desc = pSlice_descs[slice_iter];
			if ((slice_desc.m_image_index == image_index) && (slice_desc.m_level_index == level_index))
			{
				const bool slice_alpha = (slice_desc.m_flags & cSliceDescFlagsIsAlphaData) != 0;
				if (slice_alpha == alpha_data)
					return slice_iter;
			}
		}

		BASISU_DEVEL_ERROR("basisu_transcoder::find_slice: didn't find slice\n");

		return -1;
	}

	static void write_opaque_alpha_blocks(
		uint32_t num_blocks_x, uint32_t num_blocks_y,
		void* pOutput_blocks, uint32_t output_blocks_buf_size_in_blocks_or_pixels, block_format fmt,
		uint32_t block_stride_in_bytes, uint32_t output_row_pitch_in_blocks_or_pixels)
	{
		BASISU_NOTE_UNUSED(output_blocks_buf_size_in_blocks_or_pixels);

		if (!output_row_pitch_in_blocks_or_pixels)
			output_row_pitch_in_blocks_or_pixels = num_blocks_x;
				
		if ((fmt == block_format::cETC2_EAC_A8) || (fmt == block_format::cETC2_EAC_R11))
		{
#if BASISD_SUPPORT_ETC2_EAC_A8
			eac_block blk;
			blk.m_base = 255;
			blk.m_multiplier = 1;
			blk.m_table = 13;

			// Selectors are all 4's
			static const uint8_t s_etc2_eac_a8_sel4[6] = { 0x92, 0x49, 0x24, 0x92, 0x49, 0x24 };
			memcpy(&blk.m_selectors, s_etc2_eac_a8_sel4, sizeof(s_etc2_eac_a8_sel4));

			for (uint32_t y = 0; y < num_blocks_y; y++)
			{
				uint32_t dst_ofs = y * output_row_pitch_in_blocks_or_pixels * block_stride_in_bytes;
				for (uint32_t x = 0; x < num_blocks_x; x++)
				{
					memcpy((uint8_t*)pOutput_blocks + dst_ofs, &blk, sizeof(blk));
					dst_ofs += block_stride_in_bytes;
				}
			}
#endif
		}
		else if (fmt == block_format::cBC4)
		{
#if BASISD_SUPPORT_DXT5A
			dxt5a_block blk;
			blk.m_endpoints[0] = 255;
			blk.m_endpoints[1] = 255;
			memset(blk.m_selectors, 0, sizeof(blk.m_selectors));

			for (uint32_t y = 0; y < num_blocks_y; y++)
			{
				uint32_t dst_ofs = y * output_row_pitch_in_blocks_or_pixels * block_stride_in_bytes;
				for (uint32_t x = 0; x < num_blocks_x; x++)
				{
					memcpy((uint8_t*)pOutput_blocks + dst_ofs, &blk, sizeof(blk));
					dst_ofs += block_stride_in_bytes;
				}
			}
#endif
		}
	}

	bool basisu_transcoder::transcode_image_level(
		const void* pData, uint32_t data_size,
		uint32_t image_index, uint32_t level_index,
		void* pOutput_blocks, uint32_t output_blocks_buf_size_in_blocks_or_pixels,
		transcoder_texture_format fmt,
		uint32_t decode_flags, uint32_t output_row_pitch_in_blocks_or_pixels, basisu_transcoder_state *pState, uint32_t output_rows_in_pixels) const
	{
		const uint32_t bytes_per_block = basis_get_bytes_per_block(fmt);

		if (!m_lowlevel_decoder.m_endpoints.size())
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: must call start_transcoding() first\n");
			return false;
		}

		const bool transcode_alpha_data_to_opaque_formats = (decode_flags & cDecodeFlagsTranscodeAlphaDataToOpaqueFormats) != 0;

		if (decode_flags & cDecodeFlagsPVRTCDecodeToNextPow2)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: cDecodeFlagsPVRTCDecodeToNextPow2 currently unsupported\n");
			// TODO: Not yet supported
			return false;
		}

		if (!validate_header_quick(pData, data_size))
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: header validation failed\n");
			return false;
		}

		const basis_file_header* pHeader = reinterpret_cast<const basis_file_header*>(pData);

		const uint8_t* pDataU8 = static_cast<const uint8_t*>(pData);

		const basis_slice_desc* pSlice_descs = reinterpret_cast<const basis_slice_desc*>(pDataU8 + pHeader->m_slice_desc_file_ofs);

		const bool basis_file_has_alpha_slices = (pHeader->m_flags & cBASISHeaderFlagHasAlphaSlices) != 0;

		int slice_index = find_first_slice_index(pData, data_size, image_index, level_index);
		if (slice_index < 0)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: failed finding slice index\n");
			// Unable to find the requested image/level 
			return false;
		}

		if ((fmt == transcoder_texture_format::cTFPVRTC1_4_RGBA) && (!basis_file_has_alpha_slices))
		{
			// Switch to PVRTC1 RGB if the input doesn't have alpha.
			fmt = transcoder_texture_format::cTFPVRTC1_4_RGB;
		}
				
		if (pSlice_descs[slice_index].m_flags & cSliceDescFlagsIsAlphaData)
		{
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: alpha basis file has out of order alpha slice\n");

			// The first slice shouldn't have alpha data in a properly formed basis file
			return false;
		}

		if (basis_file_has_alpha_slices)
		{
			// The alpha data should immediately follow the color data, and have the same resolution.
			if ((slice_index + 1U) >= pHeader->m_total_slices)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: alpha basis file has missing alpha slice\n");
				// basis file is missing the alpha slice
				return false;
			}

			// Basic sanity checks
			if ((pSlice_descs[slice_index + 1].m_flags & cSliceDescFlagsIsAlphaData) == 0)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: alpha basis file has missing alpha slice (flag check)\n");
				// This slice should have alpha data
				return false;
			}

			if ((pSlice_descs[slice_index].m_num_blocks_x != pSlice_descs[slice_index + 1].m_num_blocks_x) || (pSlice_descs[slice_index].m_num_blocks_y != pSlice_descs[slice_index + 1].m_num_blocks_y))
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: alpha basis file slice dimensions bad\n");
				// Alpha slice should have been the same res as the color slice
				return false;
			}
		}
								
		bool status = false;

		const uint32_t total_slice_blocks = pSlice_descs[slice_index].m_num_blocks_x * pSlice_descs[slice_index].m_num_blocks_y;

		if (((fmt == transcoder_texture_format::cTFPVRTC1_4_RGB) || (fmt == transcoder_texture_format::cTFPVRTC1_4_RGBA)) && (output_blocks_buf_size_in_blocks_or_pixels > total_slice_blocks))
		{
			// The transcoder doesn't write beyond total_slice_blocks, so we need to clear the rest ourselves.
			// For GL usage, PVRTC1 4bpp image size is (max(width, 8)* max(height, 8) * 4 + 7) / 8. 
			// However, for KTX and internally in Basis this formula isn't used, it's just ((width+3)/4) * ((height+3)/4) * bytes_per_block. This is all the transcoder actually writes to memory.
			memset(static_cast<uint8_t*>(pOutput_blocks) + total_slice_blocks * bytes_per_block, 0, (output_blocks_buf_size_in_blocks_or_pixels - total_slice_blocks) * bytes_per_block);
		}
				
		switch (fmt)
		{
		case transcoder_texture_format::cTFETC1_RGB:
		{
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cETC1, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to ETC1 failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFBC1_RGB:
		{
#if !BASISD_SUPPORT_DXT1
			return false;
#endif
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC1, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to BC1 failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFBC4_R:
		{
#if !BASISD_SUPPORT_DXT5A
			return false;
#endif
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC4, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to BC4 failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFPVRTC1_4_RGB:
		{
#if !BASISD_SUPPORT_PVRTC1
			return false;
#endif
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			// output_row_pitch_in_blocks_or_pixels is actually ignored because we're transcoding to PVRTC1. (Print a dev warning if it's != 0?)
			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cPVRTC1_4_RGB, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to PVRTC1 4 RGB failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFPVRTC1_4_RGBA:
		{
#if !BASISD_SUPPORT_PVRTC1
			return false;
#endif
			assert(basis_file_has_alpha_slices);

			// Temp buffer to hold alpha block endpoint/selector indices
			std::vector<uint32_t> temp_block_indices(total_slice_blocks);

			// First transcode alpha data to temp buffer
			status = transcode_slice(pData, data_size, slice_index + 1, &temp_block_indices[0], total_slice_blocks, block_format::cIndices, sizeof(uint32_t), decode_flags, pSlice_descs[slice_index].m_num_blocks_x, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to PVRTC1 4 RGBA failed (0)\n");
			}
			else
			{
				// output_row_pitch_in_blocks_or_pixels is actually ignored because we're transcoding to PVRTC1. (Print a dev warning if it's != 0?)
				status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cPVRTC1_4_RGBA, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState, &temp_block_indices[0]);
				if (!status)
				{
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to PVRTC1 4 RGBA failed (1)\n");
				}
			}

			break;
		}
		case transcoder_texture_format::cTFBC7_M6_RGB:
		{
#if !BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY
			return false;
#endif
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC7_M6_OPAQUE_ONLY, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to BC7 m6 opaque only failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFBC7_M5_RGBA:
		{
#if !BASISD_SUPPORT_BC7_MODE5
			return false;
#else
			assert(bytes_per_block == 16);

			// First transcode the color slice. The cBC7_M5_COLOR transcoder will output opaque mode 5 blocks.
			status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC7_M5_COLOR, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);

			if ((status) && (basis_file_has_alpha_slices))
			{
				// Now transcode the alpha slice. The cBC7_M5_ALPHA transcoder will now change the opaque mode 5 blocks to blocks with alpha.
				status = transcode_slice(pData, data_size, slice_index + 1, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC7_M5_ALPHA, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			}

			break;
#endif
		}
		case transcoder_texture_format::cTFETC2_RGBA:
		{
#if !BASISD_SUPPORT_ETC2_EAC_A8
			return false;
#endif
			assert(bytes_per_block == 16);

			if (basis_file_has_alpha_slices)
			{
				// First decode the alpha data 
				status = transcode_slice(pData, data_size, slice_index + 1, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cETC2_EAC_A8, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			}
			else
			{
				write_opaque_alpha_blocks(pSlice_descs[slice_index].m_num_blocks_x, pSlice_descs[slice_index].m_num_blocks_y, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cETC2_EAC_A8, 16, output_row_pitch_in_blocks_or_pixels);
				status = true;
			}

			if (status)
			{
				// Now decode the color data
				status = transcode_slice(pData, data_size, slice_index, (uint8_t*)pOutput_blocks + 8, output_blocks_buf_size_in_blocks_or_pixels, block_format::cETC1, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
				if (!status)
				{
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to ETC2 RGB failed\n");
				}
			}
			else
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to ETC2 A failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFBC3_RGBA:
		{
#if !BASISD_SUPPORT_DXT1
			return false;
#endif
#if !BASISD_SUPPORT_DXT5A
			return false;
#endif
			assert(bytes_per_block == 16);

			// First decode the alpha data 
			if (basis_file_has_alpha_slices)
			{
				status = transcode_slice(pData, data_size, slice_index + 1, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC4, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			}
			else
			{
				write_opaque_alpha_blocks(pSlice_descs[slice_index].m_num_blocks_x, pSlice_descs[slice_index].m_num_blocks_y, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC4, 16, output_row_pitch_in_blocks_or_pixels);
				status = true;
			}

			if (status)
			{
				// Now decode the color data. Forbid 3 color blocks, which aren't allowed in BC3.
				status = transcode_slice(pData, data_size, slice_index, (uint8_t*)pOutput_blocks + 8, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC1, 16, decode_flags | cDecodeFlagsBC1ForbidThreeColorBlocks, output_row_pitch_in_blocks_or_pixels, pState);
				if (!status)
				{
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to BC3 RGB failed\n");
				}
			}
			else
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to BC3 A failed\n");
			}

			break;
		}
		case transcoder_texture_format::cTFBC5_RG:
		{
#if !BASISD_SUPPORT_DXT5A
			return false;
#endif
			assert(bytes_per_block == 16);

			// Decode the R data (actually the green channel of the color data slice in the basis file)
			status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC4, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (status)
			{
				if (basis_file_has_alpha_slices)
				{
					// Decode the G data (actually the green channel of the alpha data slice in the basis file)
					status = transcode_slice(pData, data_size, slice_index + 1, (uint8_t*)pOutput_blocks + 8, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC4, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
					if (!status)
					{
						BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to BC5 1 failed\n");
					}
				}
				else
				{
					write_opaque_alpha_blocks(pSlice_descs[slice_index].m_num_blocks_x, pSlice_descs[slice_index].m_num_blocks_y, (uint8_t*)pOutput_blocks + 8, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC4, 16, output_row_pitch_in_blocks_or_pixels);
					status = true;
				}
			}
			else
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to BC5 channel 0 failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFASTC_4x4_RGBA:
		{
#if !BASISD_SUPPORT_ASTC
			return false;
#endif
			assert(bytes_per_block == 16);

			if (basis_file_has_alpha_slices)
			{
				// First decode the alpha data to the output (we're using the output texture as a temp buffer here).
				status = transcode_slice(pData, data_size, slice_index + 1, (uint8_t*)pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cIndices, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
				if (!status)
				{
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to failed\n");
				}
				else
				{
					// Now decode the color data and transcode to ASTC. The transcoder function will read the alpha selector data from the output texture as it converts and
					// transcode both the alpha and color data at the same time to ASTC.
					status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cASTC_4x4, 16, decode_flags | cDecodeFlagsOutputHasAlphaIndices, output_row_pitch_in_blocks_or_pixels, pState);
				}
			}
			else
				status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cASTC_4x4, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);

			break;
		}
		case transcoder_texture_format::cTFATC_RGB:
		{
#if !BASISD_SUPPORT_ATC
			return false;
#endif
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cATC_RGB, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to ATC_RGB failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFATC_RGBA:
		{
#if !BASISD_SUPPORT_ATC
			return false;
#endif
#if !BASISD_SUPPORT_DXT5A
			return false;
#endif
			assert(bytes_per_block == 16);

			// First decode the alpha data 
			if (basis_file_has_alpha_slices)
			{
				status = transcode_slice(pData, data_size, slice_index + 1, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC4, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			}
			else
			{
				write_opaque_alpha_blocks(pSlice_descs[slice_index].m_num_blocks_x, pSlice_descs[slice_index].m_num_blocks_y, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cBC4, 16, output_row_pitch_in_blocks_or_pixels);
				status = true;
			}

			if (status)
			{
				status = transcode_slice(pData, data_size, slice_index, (uint8_t*)pOutput_blocks + 8, output_blocks_buf_size_in_blocks_or_pixels, block_format::cATC_RGB, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
				if (!status)
				{
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to ATC RGB failed\n");
				}
			}
			else
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to ATC A failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFPVRTC2_4_RGB:
		{
#if !BASISD_SUPPORT_PVRTC2
			return false;
#endif
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cPVRTC2_4_RGB, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to cPVRTC2_4_RGB failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFPVRTC2_4_RGBA:
		{
#if !BASISD_SUPPORT_PVRTC2
			return false;
#endif
			if (basis_file_has_alpha_slices)
			{
				// First decode the alpha data to the output (we're using the output texture as a temp buffer here).
				status = transcode_slice(pData, data_size, slice_index + 1, (uint8_t*)pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cIndices, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
				if (!status)
				{
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to failed\n");
				}
				else
				{
					// Now decode the color data and transcode to PVRTC2 RGBA. 
					status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cPVRTC2_4_RGBA, bytes_per_block, decode_flags | cDecodeFlagsOutputHasAlphaIndices, output_row_pitch_in_blocks_or_pixels, pState);
				}
			}
			else
				status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cPVRTC2_4_RGB, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);

			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to cPVRTC2_4_RGBA failed\n");
			}

			break;
		}
		case transcoder_texture_format::cTFRGBA32:
		{
			// Raw 32bpp pixels, decoded in the usual raster order (NOT block order) into an image in memory.

			// First decode the alpha data 
			if (basis_file_has_alpha_slices)
				status = transcode_slice(pData, data_size, slice_index + 1, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cA32, sizeof(uint32_t), decode_flags, output_row_pitch_in_blocks_or_pixels, pState, nullptr, output_rows_in_pixels);
			else
				status = true;

			if (status)
			{
				status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, basis_file_has_alpha_slices ? block_format::cRGB32 : block_format::cRGBA32, sizeof(uint32_t), decode_flags, output_row_pitch_in_blocks_or_pixels, pState, nullptr, output_rows_in_pixels);
				if (!status)
				{
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to RGBA32 RGB failed\n");
				}
			}
			else
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to RGBA32 A failed\n");
			}

			break;
		}
		case transcoder_texture_format::cTFRGB565:
		case transcoder_texture_format::cTFBGR565:
		{
			// Raw 16bpp pixels, decoded in the usual raster order (NOT block order) into an image in memory.
			
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, (fmt == transcoder_texture_format::cTFRGB565) ? block_format::cRGB565 : block_format::cBGR565, sizeof(uint16_t), decode_flags, output_row_pitch_in_blocks_or_pixels, pState, nullptr, output_rows_in_pixels);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to RGB565 RGB failed\n");
			}

			break;
		}
		case transcoder_texture_format::cTFRGBA4444:
		{
			// Raw 16bpp pixels, decoded in the usual raster order (NOT block order) into an image in memory.

			// First decode the alpha data 
			if (basis_file_has_alpha_slices)
				status = transcode_slice(pData, data_size, slice_index + 1, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cRGBA4444_ALPHA, sizeof(uint16_t), decode_flags, output_row_pitch_in_blocks_or_pixels, pState, nullptr, output_rows_in_pixels);
			else
				status = true;

			if (status)
			{
				status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, basis_file_has_alpha_slices ? block_format::cRGBA4444_COLOR : block_format::cRGBA4444_COLOR_OPAQUE, sizeof(uint16_t), decode_flags, output_row_pitch_in_blocks_or_pixels, pState, nullptr, output_rows_in_pixels);
				if (!status)
				{
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to RGBA4444 RGB failed\n");
				}
			}
			else
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to RGBA4444 A failed\n");
			}

			break;
		}
		case transcoder_texture_format::cTFFXT1_RGB:
		{
#if !BASISD_SUPPORT_FXT1
			return false;
#endif
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cFXT1_RGB, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to FXT1_RGB failed\n");
			}
			break;
		}
		case transcoder_texture_format::cTFETC2_EAC_R11:
		{
#if !BASISD_SUPPORT_ETC2_EAC_RG11
			return false;
#endif
			uint32_t slice_index_to_decode = slice_index;
			// If the caller wants us to transcode the mip level's alpha data, then use the next slice.
			if ((basis_file_has_alpha_slices) && (transcode_alpha_data_to_opaque_formats))
				slice_index_to_decode++;

			status = transcode_slice(pData, data_size, slice_index_to_decode, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cETC2_EAC_R11, bytes_per_block, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			if (!status)
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to ETC2_EAC_R11 failed\n");
			}

			break;
		}
		case transcoder_texture_format::cTFETC2_EAC_RG11:
		{
#if !BASISD_SUPPORT_ETC2_EAC_RG11
			return false;
#endif
			assert(bytes_per_block == 16);

			if (basis_file_has_alpha_slices)
			{
				// First decode the alpha data to G
				status = transcode_slice(pData, data_size, slice_index + 1, (uint8_t *)pOutput_blocks + 8, output_blocks_buf_size_in_blocks_or_pixels, block_format::cETC2_EAC_R11, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
			}
			else
			{
				write_opaque_alpha_blocks(pSlice_descs[slice_index].m_num_blocks_x, pSlice_descs[slice_index].m_num_blocks_y, (uint8_t *)pOutput_blocks + 8, output_blocks_buf_size_in_blocks_or_pixels, block_format::cETC2_EAC_R11, 16, output_row_pitch_in_blocks_or_pixels);
				status = true;
			}

			if (status)
			{
				// Now decode the color data to R
				status = transcode_slice(pData, data_size, slice_index, pOutput_blocks, output_blocks_buf_size_in_blocks_or_pixels, block_format::cETC2_EAC_R11, 16, decode_flags, output_row_pitch_in_blocks_or_pixels, pState);
				if (!status)
				{
					BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to ETC2_EAC_R11 R failed\n");
				}
			}
			else
			{
				BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: transcode_slice() to ETC2_EAC_R11 G failed\n");
			}

			break;
		}
		default:
		{
			assert(0);
			BASISU_DEVEL_ERROR("basisu_transcoder::transcode_image_level: Invalid fmt\n");
			break;
		}
		}

		return status;
	}

	uint32_t basis_get_bytes_per_block(transcoder_texture_format fmt)
	{
		switch (fmt)
		{
		case transcoder_texture_format::cTFETC1_RGB:
		case transcoder_texture_format::cTFBC1_RGB:
		case transcoder_texture_format::cTFBC4_R:
		case transcoder_texture_format::cTFPVRTC1_4_RGB:
		case transcoder_texture_format::cTFPVRTC1_4_RGBA:
		case transcoder_texture_format::cTFATC_RGB:
		case transcoder_texture_format::cTFPVRTC2_4_RGB:
		case transcoder_texture_format::cTFPVRTC2_4_RGBA:
		case transcoder_texture_format::cTFETC2_EAC_R11:
			return 8;
		case transcoder_texture_format::cTFBC7_M6_RGB:
		case transcoder_texture_format::cTFBC7_M5_RGBA:
		case transcoder_texture_format::cTFETC2_RGBA:
		case transcoder_texture_format::cTFBC3_RGBA:
		case transcoder_texture_format::cTFBC5_RG:
		case transcoder_texture_format::cTFASTC_4x4_RGBA:
		case transcoder_texture_format::cTFATC_RGBA:
		case transcoder_texture_format::cTFFXT1_RGB:
		case transcoder_texture_format::cTFETC2_EAC_RG11:
			return 16;
		case transcoder_texture_format::cTFRGBA32:
			return sizeof(uint32_t) * 16;
		case transcoder_texture_format::cTFRGB565:
		case transcoder_texture_format::cTFBGR565:
		case transcoder_texture_format::cTFRGBA4444:
			return sizeof(uint16_t) * 16;
		default:
			assert(0);
			BASISU_DEVEL_ERROR("basis_get_basisu_texture_format: Invalid fmt\n");
			break;
		}
		return 0;
	}

	const char* basis_get_format_name(transcoder_texture_format fmt)
	{
		switch (fmt)
		{
		case transcoder_texture_format::cTFETC1_RGB: return "ETC1_RGB";
		case transcoder_texture_format::cTFBC1_RGB: return "BC1_RGB";
		case transcoder_texture_format::cTFBC4_R: return "BC4_R";
		case transcoder_texture_format::cTFPVRTC1_4_RGB: return "PVRTC1_4_RGB";
		case transcoder_texture_format::cTFPVRTC1_4_RGBA: return "PVRTC1_4_RGBA";
		case transcoder_texture_format::cTFBC7_M6_RGB: return "BC7_M6_RGB";
		case transcoder_texture_format::cTFBC7_M5_RGBA: return "BC7_M5_RGBA";
		case transcoder_texture_format::cTFETC2_RGBA: return "ETC2_RGBA";
		case transcoder_texture_format::cTFBC3_RGBA: return "BC3_RGBA";
		case transcoder_texture_format::cTFBC5_RG: return "BC5_RG";
		case transcoder_texture_format::cTFASTC_4x4_RGBA: return "ASTC_RGBA";
		case transcoder_texture_format::cTFATC_RGB: return "ATC_RGB";
		case transcoder_texture_format::cTFATC_RGBA: return "ATC_RGBA";
		case transcoder_texture_format::cTFRGBA32: return "RGBA32";
		case transcoder_texture_format::cTFRGB565: return "RGB565";
		case transcoder_texture_format::cTFBGR565: return "BGR565";
		case transcoder_texture_format::cTFRGBA4444: return "RGBA4444";
		case transcoder_texture_format::cTFFXT1_RGB: return "FXT1_RGB";
		case transcoder_texture_format::cTFPVRTC2_4_RGB: return "PVRTC2_4_RGB";
		case transcoder_texture_format::cTFPVRTC2_4_RGBA: return "PVRTC2_4_RGBA";
		case transcoder_texture_format::cTFETC2_EAC_R11: return "ETC2_EAC_R11";
		case transcoder_texture_format::cTFETC2_EAC_RG11: return "ETC2_EAC_RG11";
		default:
			assert(0);
			BASISU_DEVEL_ERROR("basis_get_basisu_texture_format: Invalid fmt\n");
			break;
		}
		return "";
	}

	const char* basis_get_texture_type_name(basis_texture_type tex_type)
	{
		switch (tex_type)
		{
		case cBASISTexType2D: return "2D";
		case cBASISTexType2DArray: return "2D array";
		case cBASISTexTypeCubemapArray: return "cubemap array";
		case cBASISTexTypeVideoFrames: return "video";
		case cBASISTexTypeVolume: return "3D";
		default:
			assert(0);
			BASISU_DEVEL_ERROR("basis_get_texture_type_name: Invalid tex_type\n");
			break;
		}
		return "";
	}

	bool basis_transcoder_format_has_alpha(transcoder_texture_format fmt)
	{
		switch (fmt)
		{
		case transcoder_texture_format::cTFETC2_RGBA:
		case transcoder_texture_format::cTFBC3_RGBA:
		case transcoder_texture_format::cTFASTC_4x4_RGBA:
		case transcoder_texture_format::cTFBC7_M5_RGBA:
		case transcoder_texture_format::cTFPVRTC1_4_RGBA:
		case transcoder_texture_format::cTFPVRTC2_4_RGBA:
		case transcoder_texture_format::cTFATC_RGBA:
		case transcoder_texture_format::cTFRGBA32:
		case transcoder_texture_format::cTFRGBA4444:
			return true;
		default:
			break;
		}
		return false;
	}

	basisu::texture_format basis_get_basisu_texture_format(transcoder_texture_format fmt)
	{
		switch (fmt)
		{
		case transcoder_texture_format::cTFETC1_RGB: return basisu::texture_format::cETC1;
		case transcoder_texture_format::cTFBC1_RGB: return basisu::texture_format::cBC1;
		case transcoder_texture_format::cTFBC4_R: return basisu::texture_format::cBC4;
		case transcoder_texture_format::cTFPVRTC1_4_RGB: return basisu::texture_format::cPVRTC1_4_RGB;
		case transcoder_texture_format::cTFPVRTC1_4_RGBA: return basisu::texture_format::cPVRTC1_4_RGBA;
		case transcoder_texture_format::cTFBC7_M6_RGB: return basisu::texture_format::cBC7;
		case transcoder_texture_format::cTFBC7_M5_RGBA: return basisu::texture_format::cBC7;
		case transcoder_texture_format::cTFETC2_RGBA: return basisu::texture_format::cETC2_RGBA;
		case transcoder_texture_format::cTFBC3_RGBA: return basisu::texture_format::cBC3;
		case transcoder_texture_format::cTFBC5_RG: return basisu::texture_format::cBC5;
		case transcoder_texture_format::cTFASTC_4x4_RGBA: return basisu::texture_format::cASTC4x4;
		case transcoder_texture_format::cTFATC_RGB: return basisu::texture_format::cATC_RGB;
		case transcoder_texture_format::cTFATC_RGBA: return basisu::texture_format::cATC_RGBA_INTERPOLATED_ALPHA;
		case transcoder_texture_format::cTFRGBA32: return basisu::texture_format::cRGBA32;
		case transcoder_texture_format::cTFRGB565: return basisu::texture_format::cRGB565;
		case transcoder_texture_format::cTFBGR565: return basisu::texture_format::cBGR565;
		case transcoder_texture_format::cTFRGBA4444: return basisu::texture_format::cRGBA4444;
		case transcoder_texture_format::cTFFXT1_RGB: return basisu::texture_format::cFXT1_RGB;
		case transcoder_texture_format::cTFPVRTC2_4_RGB: return basisu::texture_format::cPVRTC2_4_RGBA;
		case transcoder_texture_format::cTFPVRTC2_4_RGBA: return basisu::texture_format::cPVRTC2_4_RGBA;
		case transcoder_texture_format::cTFETC2_EAC_R11: return basisu::texture_format::cETC2_R11_EAC;
		case transcoder_texture_format::cTFETC2_EAC_RG11: return basisu::texture_format::cETC2_RG11_EAC;
		default:
			assert(0);
			BASISU_DEVEL_ERROR("basis_get_basisu_texture_format: Invalid fmt\n");
			break;
		}
		return basisu::texture_format::cInvalidTextureFormat;
	}

	bool basis_transcoder_format_is_uncompressed(transcoder_texture_format tex_type)
	{
		switch (tex_type)
		{
		case transcoder_texture_format::cTFRGBA32:
		case transcoder_texture_format::cTFRGB565:
		case transcoder_texture_format::cTFBGR565:
		case transcoder_texture_format::cTFRGBA4444:
			return true;
		default:
			break;
		}
		return false;
	}

	bool basis_block_format_is_uncompressed(block_format tex_type)
	{
		switch (tex_type)
		{
		case block_format::cRGB32:
		case block_format::cRGBA32:
		case block_format::cA32:
		case block_format::cRGB565:
		case block_format::cBGR565:
		case block_format::cRGBA4444_COLOR:
		case block_format::cRGBA4444_ALPHA:
		case block_format::cRGBA4444_COLOR_OPAQUE:
			return true;
		default:
			break;
		}
		return false;
	}
	
	uint32_t basis_get_uncompressed_bytes_per_pixel(transcoder_texture_format fmt)
	{
		switch (fmt)
		{
		case transcoder_texture_format::cTFRGBA32:
			return sizeof(uint32_t); 
		case transcoder_texture_format::cTFRGB565:
		case transcoder_texture_format::cTFBGR565:
		case transcoder_texture_format::cTFRGBA4444:
			return sizeof(uint16_t);
		default:
			break;
		}
		return 0;
	}
	
	uint32_t basis_get_block_width(transcoder_texture_format tex_type)
	{
		switch (tex_type)
		{
			case transcoder_texture_format::cTFFXT1_RGB:
				return 8;
			default:
				break;
		}
		return 4;
	}

	uint32_t basis_get_block_height(transcoder_texture_format tex_type)
	{
		(void)tex_type;
		return 4;
	}
	
	bool basis_is_format_supported(transcoder_texture_format tex_type)
	{
		switch (tex_type)
		{
			// ETC1 and uncompressed are always supported.
		case transcoder_texture_format::cTFETC1_RGB: 
		case transcoder_texture_format::cTFRGBA32: 
		case transcoder_texture_format::cTFRGB565: 
		case transcoder_texture_format::cTFBGR565: 
		case transcoder_texture_format::cTFRGBA4444: 
			return true;
#if BASISD_SUPPORT_DXT1
		case transcoder_texture_format::cTFBC1_RGB:
			return true;
#endif
#if BASISD_SUPPORT_DXT5A
		case transcoder_texture_format::cTFBC4_R:
		case transcoder_texture_format::cTFBC5_RG:
			return true;
#endif
#if BASISD_SUPPORT_DXT1 && BASISD_SUPPORT_DXT5A
		case transcoder_texture_format::cTFBC3_RGBA:
			return true;
#endif
#if BASISD_SUPPORT_PVRTC1
		case transcoder_texture_format::cTFPVRTC1_4_RGB: 
		case transcoder_texture_format::cTFPVRTC1_4_RGBA: 
			return true;
#endif
#if BASISD_SUPPORT_BC7_MODE6_OPAQUE_ONLY
		case transcoder_texture_format::cTFBC7_M6_RGB: 
			return true;
#endif
#if BASISD_SUPPORT_BC7_MODE5
		case transcoder_texture_format::cTFBC7_M5_RGBA:
			return true;
#endif
#if BASISD_SUPPORT_ETC2_EAC_A8
		case transcoder_texture_format::cTFETC2_RGBA: 
			return true;
#endif
#if BASISD_SUPPORT_ASTC		
		case transcoder_texture_format::cTFASTC_4x4_RGBA: 
			return true;
#endif
#if BASISD_SUPPORT_ATC
		case transcoder_texture_format::cTFATC_RGB: 
		case transcoder_texture_format::cTFATC_RGBA:
			return true;
#endif
#if BASISD_SUPPORT_FXT1
		case transcoder_texture_format::cTFFXT1_RGB:
			return true;
#endif
#if BASISD_SUPPORT_PVRTC2
		case transcoder_texture_format::cTFPVRTC2_4_RGB:
		case transcoder_texture_format::cTFPVRTC2_4_RGBA:
			return true;
#endif
#if BASISD_SUPPORT_ETC2_EAC_RG11
		case transcoder_texture_format::cTFETC2_EAC_R11:
		case transcoder_texture_format::cTFETC2_EAC_RG11:
			return true;
#endif
		default:
			break;
		}

		return false;
	}

} // namespace basist

