// File: bc7decomp.c - Richard Geldreich, Jr. 3/31/2020 - MIT license or public domain (see end of file)
#include "bc7decomp.h"

using namespace bc7decomp;

namespace bc7decomp_ref
{

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

inline uint32_t read_bits32(const uint8_t* pBuf, uint32_t& bit_offset, uint32_t codesize)
{
	assert(codesize <= 32);
	uint32_t bits = 0;
	uint32_t total_bits = 0;

	while (total_bits < codesize)
	{
		uint32_t byte_bit_offset = bit_offset & 7;
		uint32_t bits_to_read = std::min<int>(codesize - total_bits, 8 - byte_bit_offset);

		uint32_t byte_bits = pBuf[bit_offset >> 3] >> byte_bit_offset;
		byte_bits &= ((1 << bits_to_read) - 1);

		bits |= (byte_bits << total_bits);

		total_bits += bits_to_read;
		bit_offset += bits_to_read;
	}

	return bits;
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
		
bool unpack_bc7_mode0_2(uint32_t mode, const void* pBlock_bits, color_rgba* pPixels)
{
	//const uint32_t SUBSETS = 3;
	const uint32_t ENDPOINTS = 6;
	const uint32_t COMPS = 3;
	const uint32_t WEIGHT_BITS = (mode == 0) ? 3 : 2;
	const uint32_t ENDPOINT_BITS = (mode == 0) ? 4 : 5;
	const uint32_t PBITS = (mode == 0) ? 6 : 0;
	const uint32_t WEIGHT_VALS = 1 << WEIGHT_BITS;
		
	uint32_t bit_offset = 0;
	const uint8_t* pBuf = static_cast<const uint8_t*>(pBlock_bits);

	if (read_bits32(pBuf, bit_offset, mode + 1) != (1U << mode)) return false;

	const uint32_t part = read_bits32(pBuf, bit_offset, (mode == 0) ? 4 : 6);

	color_rgba endpoints[ENDPOINTS];
	for (uint32_t c = 0; c < COMPS; c++)
		for (uint32_t e = 0; e < ENDPOINTS; e++)
			endpoints[e][c] = (uint8_t)read_bits32(pBuf, bit_offset, ENDPOINT_BITS);

	uint32_t pbits[6];
	for (uint32_t p = 0; p < PBITS; p++)
		pbits[p] = read_bits32(pBuf, bit_offset, 1);

	uint32_t weights[16];
	for (uint32_t i = 0; i < 16; i++)
		weights[i] = read_bits32(pBuf, bit_offset, ((!i) || (i == g_bc7_table_anchor_index_third_subset_1[part]) || (i == g_bc7_table_anchor_index_third_subset_2[part])) ? (WEIGHT_BITS - 1) : WEIGHT_BITS);

	assert(bit_offset == 128);

	for (uint32_t e = 0; e < ENDPOINTS; e++)
		for (uint32_t c = 0; c < 4; c++)
			endpoints[e][c] = (uint8_t)((c == 3) ? 255 : (PBITS ? bc7_dequant(endpoints[e][c], pbits[e], ENDPOINT_BITS) : bc7_dequant(endpoints[e][c], ENDPOINT_BITS)));

	color_rgba block_colors[3][8];
	for (uint32_t s = 0; s < 3; s++)
		for (uint32_t i = 0; i < WEIGHT_VALS; i++)
		{
			for (uint32_t c = 0; c < 3; c++)
				block_colors[s][i][c] = (uint8_t)bc7_interp(endpoints[s * 2 + 0][c], endpoints[s * 2 + 1][c], i, WEIGHT_BITS);
			block_colors[s][i][3] = 255;
		}

	for (uint32_t i = 0; i < 16; i++)
		pPixels[i] = block_colors[g_bc7_partition3[part * 16 + i]][weights[i]];

	return true;
}

bool unpack_bc7_mode1_3_7(uint32_t mode, const void* pBlock_bits, color_rgba* pPixels)
{
	//const uint32_t SUBSETS = 2;
	const uint32_t ENDPOINTS = 4;
	const uint32_t COMPS = (mode == 7) ? 4 : 3;
	const uint32_t WEIGHT_BITS = (mode == 1) ? 3 : 2;
	const uint32_t ENDPOINT_BITS = (mode == 7) ? 5 : ((mode == 1) ? 6 : 7);
	const uint32_t PBITS = (mode == 1) ? 2 : 4;
	const uint32_t SHARED_PBITS = (mode == 1) ? true : false;
	const uint32_t WEIGHT_VALS = 1 << WEIGHT_BITS;
		
	uint32_t bit_offset = 0;
	const uint8_t* pBuf = static_cast<const uint8_t*>(pBlock_bits);

	if (read_bits32(pBuf, bit_offset, mode + 1) != (1U << mode)) return false;

	const uint32_t part = read_bits32(pBuf, bit_offset, 6);

	color_rgba endpoints[ENDPOINTS];
	for (uint32_t c = 0; c < COMPS; c++)
		for (uint32_t e = 0; e < ENDPOINTS; e++)
			endpoints[e][c] = (uint8_t)read_bits32(pBuf, bit_offset, ENDPOINT_BITS);
		
	uint32_t pbits[4];
	for (uint32_t p = 0; p < PBITS; p++)
		pbits[p] = read_bits32(pBuf, bit_offset, 1);
						
	uint32_t weights[16];
	for (uint32_t i = 0; i < 16; i++)
		weights[i] = read_bits32(pBuf, bit_offset, ((!i) || (i == g_bc7_table_anchor_index_second_subset[part])) ? (WEIGHT_BITS - 1) : WEIGHT_BITS);
		
	assert(bit_offset == 128);

	for (uint32_t e = 0; e < ENDPOINTS; e++)
		for (uint32_t c = 0; c < 4; c++)
			endpoints[e][c] = (uint8_t)((c == ((mode == 7U) ? 4U : 3U)) ? 255 : bc7_dequant(endpoints[e][c], pbits[SHARED_PBITS ? (e >> 1) : e], ENDPOINT_BITS));
		
	color_rgba block_colors[2][8];
	for (uint32_t s = 0; s < 2; s++)
		for (uint32_t i = 0; i < WEIGHT_VALS; i++)
		{
			for (uint32_t c = 0; c < COMPS; c++)
				block_colors[s][i][c] = (uint8_t)bc7_interp(endpoints[s * 2 + 0][c], endpoints[s * 2 + 1][c], i, WEIGHT_BITS);
			block_colors[s][i][3] = (COMPS == 3) ? 255 : block_colors[s][i][3];
		}

	for (uint32_t i = 0; i < 16; i++)
		pPixels[i] = block_colors[g_bc7_partition2[part * 16 + i]][weights[i]];

	return true;
}

bool unpack_bc7_mode4_5(uint32_t mode, const void* pBlock_bits, color_rgba* pPixels)
{
	const uint32_t ENDPOINTS = 2;
	const uint32_t COMPS = 4;
	const uint32_t WEIGHT_BITS = 2;
	const uint32_t A_WEIGHT_BITS = (mode == 4) ? 3 : 2;
	const uint32_t ENDPOINT_BITS = (mode == 4) ? 5 : 7;
	const uint32_t A_ENDPOINT_BITS = (mode == 4) ? 6 : 8;
	//const uint32_t WEIGHT_VALS = 1 << WEIGHT_BITS;
	//const uint32_t A_WEIGHT_VALS = 1 << A_WEIGHT_BITS;

	uint32_t bit_offset = 0;
	const uint8_t* pBuf = static_cast<const uint8_t*>(pBlock_bits);

	if (read_bits32(pBuf, bit_offset, mode + 1) != (1U << mode)) return false;

	const uint32_t comp_rot = read_bits32(pBuf, bit_offset, 2);
	const uint32_t index_mode = (mode == 4) ? read_bits32(pBuf, bit_offset, 1) : 0;

	color_rgba endpoints[ENDPOINTS];
	for (uint32_t c = 0; c < COMPS; c++)
		for (uint32_t e = 0; e < ENDPOINTS; e++)
			endpoints[e][c] = (uint8_t)read_bits32(pBuf, bit_offset, (c == 3) ? A_ENDPOINT_BITS : ENDPOINT_BITS);
		
	const uint32_t weight_bits[2] = { index_mode ? A_WEIGHT_BITS : WEIGHT_BITS,  index_mode ? WEIGHT_BITS : A_WEIGHT_BITS };
		
	uint32_t weights[16], a_weights[16];
		
	for (uint32_t i = 0; i < 16; i++)
		(index_mode ? a_weights : weights)[i] = read_bits32(pBuf, bit_offset, weight_bits[index_mode] - ((!i) ? 1 : 0));

	for (uint32_t i = 0; i < 16; i++)
		(index_mode ? weights : a_weights)[i] = read_bits32(pBuf, bit_offset, weight_bits[1 - index_mode] - ((!i) ? 1 : 0));

	assert(bit_offset == 128);

	for (uint32_t e = 0; e < ENDPOINTS; e++)
		for (uint32_t c = 0; c < 4; c++)
			endpoints[e][c] = (uint8_t)bc7_dequant(endpoints[e][c], (c == 3) ? A_ENDPOINT_BITS : ENDPOINT_BITS);

	color_rgba block_colors[8];
	for (uint32_t i = 0; i < (1U << weight_bits[0]); i++)
		for (uint32_t c = 0; c < 3; c++)
			block_colors[i][c] = (uint8_t)bc7_interp(endpoints[0][c], endpoints[1][c], i, weight_bits[0]);

	for (uint32_t i = 0; i < (1U << weight_bits[1]); i++)
		block_colors[i][3] = (uint8_t)bc7_interp(endpoints[0][3], endpoints[1][3], i, weight_bits[1]);

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

	const uint32_t r0 = (uint32_t)((block.m_lo.m_r0 << 1) | block.m_lo.m_p0);
	const uint32_t g0 = (uint32_t)((block.m_lo.m_g0 << 1) | block.m_lo.m_p0);
	const uint32_t b0 = (uint32_t)((block.m_lo.m_b0 << 1) | block.m_lo.m_p0);
	const uint32_t a0 = (uint32_t)((block.m_lo.m_a0 << 1) | block.m_lo.m_p0);
	const uint32_t r1 = (uint32_t)((block.m_lo.m_r1 << 1) | block.m_hi.m_p1);
	const uint32_t g1 = (uint32_t)((block.m_lo.m_g1 << 1) | block.m_hi.m_p1);
	const uint32_t b1 = (uint32_t)((block.m_lo.m_b1 << 1) | block.m_hi.m_p1);
	const uint32_t a1 = (uint32_t)((block.m_lo.m_a1 << 1) | block.m_hi.m_p1);

	color_rgba vals[16];
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

bool unpack_bc7(const void *pBlock, bc7decomp::color_rgba *pPixels)
{
	const uint32_t first_byte = static_cast<const uint8_t*>(pBlock)[0];

	for (uint32_t mode = 0; mode <= 7; mode++)
	{
		if (first_byte & (1U << mode))
		{
			switch (mode)
			{
			case 0:
			case 2:
				return unpack_bc7_mode0_2(mode, pBlock, pPixels);
			case 1:
			case 3:
			case 7:
				return unpack_bc7_mode1_3_7(mode, pBlock, pPixels);
			case 4:
			case 5:
				return unpack_bc7_mode4_5(mode, pBlock, pPixels);
			case 6:
				return unpack_bc7_mode6(pBlock, pPixels);
			default:
				break;
			}
		}
	}

	return false;
}

} // namespace bc7decomp_ref

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

