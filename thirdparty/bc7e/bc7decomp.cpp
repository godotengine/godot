// File: bc7decomp.c - Richard Geldreich, Jr. 3/31/2020 - MIT license or public domain (see end of file)
#include "bc7decomp.h"
#include <string.h>

#if (defined(_M_AMD64) || defined(_M_X64) || defined(__SSE2__))
#  define BC7DECOMP_USE_SSE2
#endif

#ifdef BC7DECOMP_USE_SSE2
#include <immintrin.h>
#include <emmintrin.h>
#endif

namespace bc7decomp
{

#ifdef BC7DECOMP_USE_SSE2
	const __m128i g_bc7_weights4_sse2[8] =
	{
		_mm_set_epi16(4, 4, 4, 4, 0, 0, 0, 0),
		_mm_set_epi16(13, 13, 13, 13, 9, 9, 9, 9),
		_mm_set_epi16(21, 21, 21, 21, 17, 17, 17, 17),
		_mm_set_epi16(30, 30, 30, 30, 26, 26, 26, 26),
		_mm_set_epi16(38, 38, 38, 38, 34, 34, 34, 34),
		_mm_set_epi16(47, 47, 47, 47, 43, 43, 43, 43),
		_mm_set_epi16(55, 55, 55, 55, 51, 51, 51, 51),
		_mm_set_epi16(64, 64, 64, 64, 60, 60, 60, 60),
	};
#endif

const uint32_t g_bc7_weights2[4] = { 0, 21, 43, 64 };
const uint32_t g_bc7_weights3[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
const uint32_t g_bc7_weights4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

const uint8_t g_bc7_partition2[64 * 16] =
{
	0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,		0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,		0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,		0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,		0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,		0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,
	0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,		0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,		0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
	0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,		0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,		0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,		0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,		0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,		0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,		0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,		0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
	0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,		0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,		0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,		0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,		0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,		0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,		0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,		0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,
	0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,		0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,		0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,		0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,		0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,		0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,		0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,		0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,
	0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,		0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,		0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,		0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,		0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,		0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,		0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,		0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,
	0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,		0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,		0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,		0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,		0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1,		0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,		0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,
	0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1,		0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,		0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,		0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,		0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,		0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,		0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,		0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1
};

const uint8_t g_bc7_partition3[64 * 16] =
{
	0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2,		0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1,		0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1,		0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1,		0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2,		0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2,		0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1,		0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1,
	0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,		0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,		0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,		0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,		0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2,		0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2,		0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2,		0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0,
	0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2,		0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0,		0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2,		0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1,		0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2,		0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1,		0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2,		0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0,
	0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0,		0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2,		0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0,		0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1,		0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2,		0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2,		0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1,		0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1,
	0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2,		0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1,		0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2,		0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0,		0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0,		0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,		0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0,		0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1,
	0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1,		0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2,		0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1,		0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2,		0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1,		0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1,		0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1,		0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1,
	0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2,		0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1,		0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2,		0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2,		0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2,		0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2,		0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2,		0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2,
	0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2,		0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2,		0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2,		0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2,		0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1,		0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2,		0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2,		0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0,
};

const uint8_t g_bc7_table_anchor_index_second_subset[64] = { 15,15,15,15,15,15,15,15,		15,15,15,15,15,15,15,15,		15, 2, 8, 2, 2, 8, 8,15,		2, 8, 2, 2, 8, 8, 2, 2,		15,15, 6, 8, 2, 8,15,15,		2, 8, 2, 2, 2,15,15, 6,		6, 2, 6, 8,15,15, 2, 2,		15,15,15,15,15, 2, 2,15 };

const uint8_t g_bc7_table_anchor_index_third_subset_1[64] =
{
	3, 3,15,15, 8, 3,15,15,		8, 8, 6, 6, 6, 5, 3, 3,		3, 3, 8,15, 3, 3, 6,10,		5, 8, 8, 6, 8, 5,15,15,		8,15, 3, 5, 6,10, 8,15,		15, 3,15, 5,15,15,15,15,		3,15, 5, 5, 5, 8, 5,10,		5,10, 8,13,15,12, 3, 3
};

const uint8_t g_bc7_table_anchor_index_third_subset_2[64] =
{
	15, 8, 8, 3,15,15, 3, 8,		15,15,15,15,15,15,15, 8,		15, 8,15, 3,15, 8,15, 8,		3,15, 6,10,15,15,10, 8,		15, 3,15,10,10, 8, 9,10,		6,15, 8,15, 3, 6, 6, 8,		15, 3,15,15,15,15,15,15,		15,15,15,15, 3,15,15, 8
};

const uint8_t g_bc7_first_byte_to_mode[256] =
{
	8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
};

inline void insert_weight_zero(uint64_t& index_bits, uint32_t bits_per_index, uint32_t offset)
{
	uint64_t LOW_BIT_MASK = (static_cast<uint64_t>(1) << ((bits_per_index * (offset + 1)) - 1)) - 1;
	uint64_t HIGH_BIT_MASK = ~LOW_BIT_MASK;

	index_bits = ((index_bits & HIGH_BIT_MASK) << 1) | (index_bits & LOW_BIT_MASK);
}

// BC7 mode 0-7 decompression.
// Instead of one monster routine to unpack all the BC7 modes, we're lumping the 3 subset, 2 subset, 1 subset, and dual plane modes together into simple shared routines.

static inline uint32_t bc7_dequant(uint32_t val, uint32_t pbit, uint32_t val_bits) { assert(val < (1U << val_bits)); assert(pbit < 2); assert(val_bits >= 4 && val_bits <= 8); const uint32_t total_bits = val_bits + 1; val = (val << 1) | pbit; val <<= (8 - total_bits); val |= (val >> total_bits); assert(val <= 255); return val; }
static inline uint32_t bc7_dequant(uint32_t val, uint32_t val_bits) { assert(val < (1U << val_bits)); assert(val_bits >= 4 && val_bits <= 8); val <<= (8 - val_bits); val |= (val >> val_bits); assert(val <= 255); return val; }

static inline uint32_t bc7_interp2(uint32_t l, uint32_t h, uint32_t w) { assert(w < 4); return (l * (64 - g_bc7_weights2[w]) + h * g_bc7_weights2[w] + 32) >> 6; }
static inline uint32_t bc7_interp3(uint32_t l, uint32_t h, uint32_t w) { assert(w < 8); return (l * (64 - g_bc7_weights3[w]) + h * g_bc7_weights3[w] + 32) >> 6; }
static inline uint32_t bc7_interp4(uint32_t l, uint32_t h, uint32_t w) { assert(w < 16); return (l * (64 - g_bc7_weights4[w]) + h * g_bc7_weights4[w] + 32) >> 6; }
static inline uint32_t bc7_interp(uint32_t l, uint32_t h, uint32_t w, uint32_t bits)
{
	assert(l <= 255 && h <= 255);
	switch (bits)
	{
	case 2: return bc7_interp2(l, h, w);
	case 3: return bc7_interp3(l, h, w);
	case 4: return bc7_interp4(l, h, w);
	default: 
		break;
	}
	return 0;
}


#ifdef BC7DECOMP_USE_SSE2
static inline __m128i bc7_interp_sse2(__m128i l, __m128i h, __m128i w, __m128i iw)
{
	return _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(l, iw), _mm_mullo_epi16(h, w)), _mm_set1_epi16(32)), 6);
}

static inline void bc7_interp2_sse2(const color_rgba* endpoint_pair, color_rgba* out_colors)
{
	__m128i endpoints = _mm_loadu_si64(endpoint_pair);
	__m128i endpoints_16 = _mm_unpacklo_epi8(endpoints, _mm_setzero_si128());

	__m128i endpoints_16_swapped = _mm_shuffle_epi32(endpoints_16, _MM_SHUFFLE(1, 0, 3, 2));

	// Interpolated colors will be color 1 and 2
	__m128i interpolated_colors = bc7_interp_sse2(endpoints_16, endpoints_16_swapped, _mm_set1_epi16(21), _mm_set1_epi16(43));

	// all_colors will be 1, 2, 0, 3
	__m128i all_colors = _mm_packus_epi16(interpolated_colors, endpoints_16);

	all_colors = _mm_shuffle_epi32(all_colors, _MM_SHUFFLE(3, 1, 0, 2));

	_mm_storeu_si128(reinterpret_cast<__m128i*>(out_colors), all_colors);
}

static inline void bc7_interp3_sse2(const color_rgba* endpoint_pair, color_rgba* out_colors)
{
	__m128i endpoints = _mm_loadu_si64(endpoint_pair);
	__m128i endpoints_16bit = _mm_unpacklo_epi8(endpoints, _mm_setzero_si128());
	__m128i endpoints_16bit_swapped = _mm_shuffle_epi32(endpoints_16bit, _MM_SHUFFLE(1, 0, 3, 2));

	__m128i interpolated_16 = bc7_interp_sse2(endpoints_16bit, endpoints_16bit_swapped, _mm_set1_epi16(9), _mm_set1_epi16(55));
	__m128i interpolated_23 = bc7_interp_sse2(endpoints_16bit, endpoints_16bit_swapped, _mm_set_epi16(37, 37, 37, 37, 18, 18, 18, 18), _mm_set_epi16(27, 27, 27, 27, 46, 46, 46, 46));
	__m128i interpolated_45 = bc7_interp_sse2(endpoints_16bit, endpoints_16bit_swapped, _mm_set_epi16(18, 18, 18, 18, 37, 37, 37, 37), _mm_set_epi16(46, 46, 46, 46, 27, 27, 27, 27));

	__m128i interpolated_01 = _mm_unpacklo_epi64(endpoints_16bit, interpolated_16);
	__m128i interpolated_67 = _mm_unpackhi_epi64(interpolated_16, endpoints_16bit);

	__m128i all_colors_0 = _mm_packus_epi16(interpolated_01, interpolated_23);
	__m128i all_colors_1 = _mm_packus_epi16(interpolated_45, interpolated_67);

	_mm_storeu_si128(reinterpret_cast<__m128i*>(out_colors), all_colors_0);
	_mm_storeu_si128(reinterpret_cast<__m128i*>(out_colors + 4), all_colors_1);
}
#endif

bool unpack_bc7_mode0_2(uint32_t mode, const uint64_t* data_chunks, color_rgba* pPixels)
{
	//const uint32_t SUBSETS = 3;
	const uint32_t ENDPOINTS = 6;
	const uint32_t COMPS = 3;
	const uint32_t WEIGHT_BITS = (mode == 0) ? 3 : 2;
	const uint32_t WEIGHT_MASK = (1 << WEIGHT_BITS) - 1;
	const uint32_t ENDPOINT_BITS = (mode == 0) ? 4 : 5;
	const uint32_t ENDPOINT_MASK = (1 << ENDPOINT_BITS) - 1;
	const uint32_t PBITS = (mode == 0) ? 6 : 0;
	const uint32_t WEIGHT_VALS = 1 << WEIGHT_BITS;
	const uint32_t PART_BITS = (mode == 0) ? 4 : 6;
	const uint32_t PART_MASK = (1 << PART_BITS) - 1;

	const uint64_t low_chunk = data_chunks[0];
	const uint64_t high_chunk = data_chunks[1];

	const uint32_t part = (low_chunk >> (mode + 1)) & PART_MASK;

	uint64_t channel_read_chunks[3] = { 0, 0, 0 };

	if (mode == 0)
	{
		channel_read_chunks[0] = low_chunk >> 5;
		channel_read_chunks[1] = low_chunk >> 29;
		channel_read_chunks[2] = ((low_chunk >> 53) | (high_chunk << 11));
	}
	else
	{
		channel_read_chunks[0] = low_chunk >> 9;
		channel_read_chunks[1] = ((low_chunk >> 39) | (high_chunk << 25));
		channel_read_chunks[2] = high_chunk >> 5;
	}

	color_rgba endpoints[ENDPOINTS];
	for (uint32_t c = 0; c < COMPS; c++)
	{
		uint64_t channel_read_chunk = channel_read_chunks[c];
		for (uint32_t e = 0; e < ENDPOINTS; e++)
		{
			endpoints[e][c] = static_cast<uint8_t>(channel_read_chunk & ENDPOINT_MASK);
			channel_read_chunk >>= ENDPOINT_BITS;
		}
	}

	uint32_t pbits[6];
	if (mode == 0)
	{
		uint8_t p_bits_chunk = static_cast<uint8_t>((high_chunk >> 13) & 0xff);

		for (uint32_t p = 0; p < PBITS; p++)
			pbits[p] = (p_bits_chunk >> p) & 1;
	}

	uint64_t weights_read_chunk = high_chunk >> (67 - 16 * WEIGHT_BITS);
	insert_weight_zero(weights_read_chunk, WEIGHT_BITS, 0);
	insert_weight_zero(weights_read_chunk, WEIGHT_BITS, std::min(g_bc7_table_anchor_index_third_subset_1[part], g_bc7_table_anchor_index_third_subset_2[part]));
	insert_weight_zero(weights_read_chunk, WEIGHT_BITS, std::max(g_bc7_table_anchor_index_third_subset_1[part], g_bc7_table_anchor_index_third_subset_2[part]));

	uint32_t weights[16];
	for (uint32_t i = 0; i < 16; i++)
	{
		weights[i] = static_cast<uint32_t>(weights_read_chunk & WEIGHT_MASK);
		weights_read_chunk >>= WEIGHT_BITS;
	}

	for (uint32_t e = 0; e < ENDPOINTS; e++)
		for (uint32_t c = 0; c < 4; c++)
			endpoints[e][c] = static_cast<uint8_t>((c == 3) ? 255 : (PBITS ? bc7_dequant(endpoints[e][c], pbits[e], ENDPOINT_BITS) : bc7_dequant(endpoints[e][c], ENDPOINT_BITS)));

	color_rgba block_colors[3][8];

#ifdef BC7DECOMP_USE_SSE2
	for (uint32_t s = 0; s < 3; s++)
	{
		if (WEIGHT_BITS == 2)
			bc7_interp2_sse2(endpoints + s * 2, block_colors[s]);
		else
			bc7_interp3_sse2(endpoints + s * 2, block_colors[s]);
	}
#else
	for (uint32_t s = 0; s < 3; s++)
		for (uint32_t i = 0; i < WEIGHT_VALS; i++)
		{
			for (uint32_t c = 0; c < 3; c++)
				block_colors[s][i][c] = static_cast<uint8_t>(bc7_interp(endpoints[s * 2 + 0][c], endpoints[s * 2 + 1][c], i, WEIGHT_BITS));
			block_colors[s][i][3] = 255;
		}
#endif

	for (uint32_t i = 0; i < 16; i++)
		pPixels[i] = block_colors[g_bc7_partition3[part * 16 + i]][weights[i]];

	return true;
}

bool unpack_bc7_mode1_3_7(uint32_t mode, const uint64_t* data_chunks, color_rgba* pPixels)
{
	//const uint32_t SUBSETS = 2;
	const uint32_t ENDPOINTS = 4;
	const uint32_t COMPS = (mode == 7) ? 4 : 3;
	const uint32_t WEIGHT_BITS = (mode == 1) ? 3 : 2;
	const uint32_t WEIGHT_MASK = (1 << WEIGHT_BITS) - 1;
	const uint32_t ENDPOINT_BITS = (mode == 7) ? 5 : ((mode == 1) ? 6 : 7);
	const uint32_t ENDPOINT_MASK = (1 << ENDPOINT_BITS) - 1;
	const uint32_t PBITS = (mode == 1) ? 2 : 4;
	const uint32_t SHARED_PBITS = (mode == 1) ? true : false;
	const uint32_t WEIGHT_VALS = 1 << WEIGHT_BITS;

	const uint64_t low_chunk = data_chunks[0];
	const uint64_t high_chunk = data_chunks[1];

	const uint32_t part = ((low_chunk >> (mode + 1)) & 0x3f);

	color_rgba endpoints[ENDPOINTS];

	uint64_t channel_read_chunks[4] = { 0, 0, 0, 0 };
	uint64_t p_read_chunk = 0;
	channel_read_chunks[0] = (low_chunk >> (mode + 7));
	uint64_t weight_read_chunk;

	switch (mode)
	{
	case 1:
		channel_read_chunks[1] = (low_chunk >> 32);
		channel_read_chunks[2] = ((low_chunk >> 56) | (high_chunk << 8));
		p_read_chunk = high_chunk >> 16;
		weight_read_chunk = high_chunk >> 18;
		break;
	case 3:
		channel_read_chunks[1] = ((low_chunk >> 38) | (high_chunk << 26));
		channel_read_chunks[2] = high_chunk >> 2;
		p_read_chunk = high_chunk >> 30;
		weight_read_chunk = high_chunk >> 34;
		break;
	case 7:
		channel_read_chunks[1] = low_chunk >> 34;
		channel_read_chunks[2] = ((low_chunk >> 54) | (high_chunk << 10));
		channel_read_chunks[3] = high_chunk >> 10;
		p_read_chunk = (high_chunk >> 30);
		weight_read_chunk = (high_chunk >> 34);
		break;
	default:
		return false;
	};

	for (uint32_t c = 0; c < COMPS; c++)
	{
		uint64_t channel_read_chunk = channel_read_chunks[c];
		for (uint32_t e = 0; e < ENDPOINTS; e++)
		{
			endpoints[e][c] = static_cast<uint8_t>(channel_read_chunk & ENDPOINT_MASK);
			channel_read_chunk >>= ENDPOINT_BITS;
		}
	}
		
	uint32_t pbits[4];
	for (uint32_t p = 0; p < PBITS; p++)
		pbits[p] = (p_read_chunk >> p) & 1;

	insert_weight_zero(weight_read_chunk, WEIGHT_BITS, 0);
	insert_weight_zero(weight_read_chunk, WEIGHT_BITS, g_bc7_table_anchor_index_second_subset[part]);

	uint32_t weights[16];
	for (uint32_t i = 0; i < 16; i++)
	{
		weights[i] = static_cast<uint32_t>(weight_read_chunk & WEIGHT_MASK);
		weight_read_chunk >>= WEIGHT_BITS;
	}

	for (uint32_t e = 0; e < ENDPOINTS; e++)
		for (uint32_t c = 0; c < 4; c++)
			endpoints[e][c] = static_cast<uint8_t>((mode != 7U && c == 3U) ? 255 : bc7_dequant(endpoints[e][c], pbits[SHARED_PBITS ? (e >> 1) : e], ENDPOINT_BITS));
		
	color_rgba block_colors[2][8];
#ifdef BC7DECOMP_USE_SSE2
	for (uint32_t s = 0; s < 2; s++)
	{
		if (WEIGHT_BITS == 2)
			bc7_interp2_sse2(endpoints + s * 2, block_colors[s]);
		else
			bc7_interp3_sse2(endpoints + s * 2, block_colors[s]);
	}
#else
	for (uint32_t s = 0; s < 2; s++)
		for (uint32_t i = 0; i < WEIGHT_VALS; i++)
		{
			for (uint32_t c = 0; c < COMPS; c++)
				block_colors[s][i][c] = static_cast<uint8_t>(bc7_interp(endpoints[s * 2 + 0][c], endpoints[s * 2 + 1][c], i, WEIGHT_BITS));
			block_colors[s][i][3] = (COMPS == 3) ? 255 : block_colors[s][i][3];
		}
#endif

	for (uint32_t i = 0; i < 16; i++)
		pPixels[i] = block_colors[g_bc7_partition2[part * 16 + i]][weights[i]];

	return true;
}

bool unpack_bc7_mode4_5(uint32_t mode, const uint64_t* data_chunks, color_rgba* pPixels)
{
	const uint32_t ENDPOINTS = 2;
	//const uint32_t COMPS = 4;
	const uint32_t WEIGHT_BITS = 2;
	const uint32_t WEIGHT_MASK = (1 << WEIGHT_BITS) - 1;
	const uint32_t A_WEIGHT_BITS = (mode == 4) ? 3 : 2;
	const uint32_t A_WEIGHT_MASK = (1 << A_WEIGHT_BITS) - 1;
	const uint32_t ENDPOINT_BITS = (mode == 4) ? 5 : 7;
	const uint32_t ENDPOINT_MASK = (1 << ENDPOINT_BITS) - 1;
	const uint32_t A_ENDPOINT_BITS = (mode == 4) ? 6 : 8;
	const uint32_t A_ENDPOINT_MASK = (1 << A_ENDPOINT_BITS) - 1;
	//const uint32_t WEIGHT_VALS = 1 << WEIGHT_BITS;
	//const uint32_t A_WEIGHT_VALS = 1 << A_WEIGHT_BITS;

	const uint64_t low_chunk = data_chunks[0];
	const uint64_t high_chunk = data_chunks[1];

	const uint32_t comp_rot = (low_chunk >> (mode + 1)) & 0x3;
	const uint32_t index_mode = (mode == 4) ? static_cast<uint32_t>((low_chunk >> 7) & 1) : 0;

	uint64_t color_read_bits = low_chunk >> 8;

	color_rgba endpoints[ENDPOINTS];
	for (uint32_t c = 0; c < 3; c++)
	{
		for (uint32_t e = 0; e < ENDPOINTS; e++)
		{
			endpoints[e][c] = static_cast<uint8_t>(color_read_bits & ENDPOINT_MASK);
			color_read_bits >>= ENDPOINT_BITS;
		}
	}

	endpoints[0][3] = static_cast<uint8_t>(color_read_bits & ENDPOINT_MASK);

	uint64_t rgb_weights_chunk;
	uint64_t a_weights_chunk;
	if (mode == 4)
	{
		endpoints[0][3] = static_cast<uint8_t>(color_read_bits & A_ENDPOINT_MASK);
		endpoints[1][3] = static_cast<uint8_t>((color_read_bits >> A_ENDPOINT_BITS) & A_ENDPOINT_MASK);
		rgb_weights_chunk = ((low_chunk >> 50) | (high_chunk << 14));
		a_weights_chunk = high_chunk >> 17;
	}
	else if (mode == 5)
	{
		endpoints[0][3] = static_cast<uint8_t>(color_read_bits & A_ENDPOINT_MASK);
		endpoints[1][3] = static_cast<uint8_t>(((low_chunk >> 58) | (high_chunk << 6)) & A_ENDPOINT_MASK);
		rgb_weights_chunk = high_chunk >> 2;
		a_weights_chunk = high_chunk >> 33;
	}
	else
		return false;

	insert_weight_zero(rgb_weights_chunk, WEIGHT_BITS, 0);
	insert_weight_zero(a_weights_chunk, A_WEIGHT_BITS, 0);

	const uint32_t weight_bits[2] = { index_mode ? A_WEIGHT_BITS : WEIGHT_BITS,  index_mode ? WEIGHT_BITS : A_WEIGHT_BITS };
	const uint32_t weight_mask[2] = { index_mode ? A_WEIGHT_MASK : WEIGHT_MASK,  index_mode ? WEIGHT_MASK : A_WEIGHT_MASK };

	uint32_t weights[16], a_weights[16];

	if (index_mode)
		std::swap(rgb_weights_chunk, a_weights_chunk);

	for (uint32_t i = 0; i < 16; i++)
	{
		weights[i] = (rgb_weights_chunk & weight_mask[0]);
		rgb_weights_chunk >>= weight_bits[0];
	}

	for (uint32_t i = 0; i < 16; i++)
	{
		a_weights[i] = (a_weights_chunk & weight_mask[1]);
		a_weights_chunk >>= weight_bits[1];
	}

	for (uint32_t e = 0; e < ENDPOINTS; e++)
		for (uint32_t c = 0; c < 4; c++)
			endpoints[e][c] = static_cast<uint8_t>(bc7_dequant(endpoints[e][c], (c == 3) ? A_ENDPOINT_BITS : ENDPOINT_BITS));

	color_rgba block_colors[8];
#ifdef BC7DECOMP_USE_SSE2
	if (weight_bits[0] == 3)
		bc7_interp3_sse2(endpoints, block_colors);
	else
		bc7_interp2_sse2(endpoints, block_colors);
#else
	for (uint32_t i = 0; i < (1U << weight_bits[0]); i++)
		for (uint32_t c = 0; c < 3; c++)
			block_colors[i][c] = static_cast<uint8_t>(bc7_interp(endpoints[0][c], endpoints[1][c], i, weight_bits[0]));
#endif

	for (uint32_t i = 0; i < (1U << weight_bits[1]); i++)
		block_colors[i][3] = static_cast<uint8_t>(bc7_interp(endpoints[0][3], endpoints[1][3], i, weight_bits[1]));

	for (uint32_t i = 0; i < 16; i++)
	{
		pPixels[i] = block_colors[weights[i]];
		pPixels[i].a = block_colors[a_weights[i]].a;
		if (comp_rot >= 1)
			std::swap(pPixels[i].a, pPixels[i].m_comps[comp_rot - 1]);
	}

	return true;
}

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

bool unpack_bc7_mode6(const void *pBlock_bits, color_rgba *pPixels)
{
	static_assert(sizeof(bc7_mode_6) == 16, "sizeof(bc7_mode_6) == 16");

	const bc7_mode_6 &block = *static_cast<const bc7_mode_6 *>(pBlock_bits);

	if (block.m_lo.m_mode != (1 << 6))
		return false;

	const uint32_t r0 = static_cast<uint32_t>((block.m_lo.m_r0 << 1) | block.m_lo.m_p0);
	const uint32_t g0 = static_cast<uint32_t>((block.m_lo.m_g0 << 1) | block.m_lo.m_p0);
	const uint32_t b0 = static_cast<uint32_t>((block.m_lo.m_b0 << 1) | block.m_lo.m_p0);
	const uint32_t a0 = static_cast<uint32_t>((block.m_lo.m_a0 << 1) | block.m_lo.m_p0);
	const uint32_t r1 = static_cast<uint32_t>((block.m_lo.m_r1 << 1) | block.m_hi.m_p1);
	const uint32_t g1 = static_cast<uint32_t>((block.m_lo.m_g1 << 1) | block.m_hi.m_p1);
	const uint32_t b1 = static_cast<uint32_t>((block.m_lo.m_b1 << 1) | block.m_hi.m_p1);
	const uint32_t a1 = static_cast<uint32_t>((block.m_lo.m_a1 << 1) | block.m_hi.m_p1);

	color_rgba vals[16];
#ifdef BC7DECOMP_USE_SSE2
	__m128i vep0 = _mm_set_epi16((short)a0, (short)b0, (short)g0, (short)r0, (short)a0, (short)b0, (short)g0, (short)r0);
	__m128i vep1 = _mm_set_epi16((short)a1, (short)b1, (short)g1, (short)r1, (short)a1, (short)b1, (short)g1, (short)r1);

	for (uint32_t i = 0; i < 16; i += 4)
	{
		const __m128i w0 = g_bc7_weights4_sse2[i / 4 * 2 + 0];
		const __m128i w1 = g_bc7_weights4_sse2[i / 4 * 2 + 1];

		const __m128i iw0 = _mm_sub_epi16(_mm_set1_epi16(64), w0);
		const __m128i iw1 = _mm_sub_epi16(_mm_set1_epi16(64), w1);

		__m128i first_half = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(vep0, iw0), _mm_mullo_epi16(vep1, w0)), _mm_set1_epi16(32)), 6);
		__m128i second_half = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(vep0, iw1), _mm_mullo_epi16(vep1, w1)), _mm_set1_epi16(32)), 6);
		__m128i combined = _mm_packus_epi16(first_half, second_half);

		_mm_storeu_si128(reinterpret_cast<__m128i*>(vals + i), combined);
	}
#else
	for (uint32_t i = 0; i < 16; i++)
	{
		const uint32_t w = g_bc7_weights4[i];
		const uint32_t iw = 64 - w;
		vals[i].set_noclamp_rgba(
			(r0 * iw + r1 * w + 32) >> 6,
			(g0 * iw + g1 * w + 32) >> 6,
			(b0 * iw + b1 * w + 32) >> 6,
			(a0 * iw + a1 * w + 32) >> 6);
	}
#endif

	pPixels[0] = vals[block.m_hi.m_s00];
	pPixels[1] = vals[block.m_hi.m_s10];
	pPixels[2] = vals[block.m_hi.m_s20];
	pPixels[3] = vals[block.m_hi.m_s30];

	pPixels[4] = vals[block.m_hi.m_s01];
	pPixels[5] = vals[block.m_hi.m_s11];
	pPixels[6] = vals[block.m_hi.m_s21];
	pPixels[7] = vals[block.m_hi.m_s31];

	pPixels[8] = vals[block.m_hi.m_s02];
	pPixels[9] = vals[block.m_hi.m_s12];
	pPixels[10] = vals[block.m_hi.m_s22];
	pPixels[11] = vals[block.m_hi.m_s32];

	pPixels[12] = vals[block.m_hi.m_s03];
	pPixels[13] = vals[block.m_hi.m_s13];
	pPixels[14] = vals[block.m_hi.m_s23];
	pPixels[15] = vals[block.m_hi.m_s33];

	return true;
}

bool unpack_bc7(const void *pBlock, color_rgba *pPixels)
{
	const uint8_t *block_bytes = static_cast<const uint8_t*>(pBlock);
	uint8_t mode = g_bc7_first_byte_to_mode[block_bytes[0]];

	uint64_t data_chunks[2];

	uint64_t endian_check = 1;
	if (*reinterpret_cast<const uint8_t*>(&endian_check) == 1)
		memcpy(data_chunks, pBlock, 16);
	else
	{
		data_chunks[0] = data_chunks[1] = 0;
		for (int chunk_index = 0; chunk_index < 2; chunk_index++)
		{
			for (int byte_index = 0; byte_index < 8; byte_index++)
				data_chunks[chunk_index] |= static_cast<uint64_t>(block_bytes[chunk_index * 8 + byte_index]) << (byte_index * 8);
		}
	}

	switch (mode)
	{
	case 0:
	case 2:
		return unpack_bc7_mode0_2(mode, data_chunks, pPixels);
	case 1:
	case 3:
	case 7:
		return unpack_bc7_mode1_3_7(mode, data_chunks, pPixels);
	case 4:
	case 5:
		return unpack_bc7_mode4_5(mode, data_chunks, pPixels);
	case 6:
		return unpack_bc7_mode6(data_chunks, pPixels);
	default:
		memset(pPixels, 0, sizeof(color_rgba) * 16);
		break;
	}

	return false;
}

} // namespace bc7decomp

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright(c) 2020 Richard Geldreich, Jr.
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files(the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions :
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain(www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non - commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain.We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors.We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/

