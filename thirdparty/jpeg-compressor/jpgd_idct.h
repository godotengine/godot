// Copyright 2009 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//
// From:
// https://software.intel.com/sites/default/files/m/d/4/1/d/8/UsingIntelAVXToImplementIDCT-r1_5.pdf
// https://software.intel.com/file/29048
//
// Requires SSE
//
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <immintrin.h>

#ifdef _MSC_VER
	#define JPGD_SIMD_ALIGN(type, name) __declspec(align(16)) type name
#else
	#define JPGD_SIMD_ALIGN(type, name) type name __attribute__((aligned(16)))
#endif

#define BITS_INV_ACC 4
#define SHIFT_INV_ROW 16 - BITS_INV_ACC
#define SHIFT_INV_COL 1 + BITS_INV_ACC
const short IRND_INV_ROW = 1024 * (6 - BITS_INV_ACC);	//1 << (SHIFT_INV_ROW-1)
const short IRND_INV_COL = 16 * (BITS_INV_ACC - 3);		// 1 << (SHIFT_INV_COL-1)
const short IRND_INV_CORR = IRND_INV_COL - 1;			// correction -1.0 and round

JPGD_SIMD_ALIGN(short, shortM128_one_corr[8]) = {1, 1, 1, 1, 1, 1, 1, 1};
JPGD_SIMD_ALIGN(short, shortM128_round_inv_row[8]) = {IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0};
JPGD_SIMD_ALIGN(short, shortM128_round_inv_col[8]) = {IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL};
JPGD_SIMD_ALIGN(short, shortM128_round_inv_corr[8])= {IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR};
JPGD_SIMD_ALIGN(short, shortM128_tg_1_16[8]) = {13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036}; // tg * (2<<16) + 0.5
JPGD_SIMD_ALIGN(short, shortM128_tg_2_16[8]) = {27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146}; // tg * (2<<16) + 0.5
JPGD_SIMD_ALIGN(short, shortM128_tg_3_16[8]) = {-21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746}; // tg * (2<<16) + 0.5
JPGD_SIMD_ALIGN(short, shortM128_cos_4_16[8]) = {-19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195};// cos * (2<<16) + 0.5

//-----------------------------------------------------------------------------
// Table for rows 0,4 - constants are multiplied on cos_4_16
// w15 w14 w11 w10 w07 w06 w03 w02
// w29 w28 w25 w24 w21 w20 w17 w16
// w31 w30 w27 w26 w23 w22 w19 w18
//movq -> w05 w04 w01 w00
JPGD_SIMD_ALIGN(short, shortM128_tab_i_04[]) = {
	16384, 21407, 16384, 8867,
	16384, -8867, 16384, -21407, // w13 w12 w09 w08
	16384, 8867, -16384, -21407, // w07 w06 w03 w02
	-16384, 21407, 16384, -8867, // w15 w14 w11 w10
	22725, 19266, 19266, -4520, // w21 w20 w17 w16
	12873, -22725, 4520, -12873, // w29 w28 w25 w24
	12873, 4520, -22725, -12873, // w23 w22 w19 w18
	4520, 19266, 19266, -22725}; // w31 w30 w27 w26

	// Table for rows 1,7 - constants are multiplied on cos_1_16
//movq -> w05 w04 w01 w00
JPGD_SIMD_ALIGN(short, shortM128_tab_i_17[]) = {
	22725, 29692, 22725, 12299,
	22725, -12299, 22725, -29692, // w13 w12 w09 w08
	22725, 12299, -22725, -29692, // w07 w06 w03 w02
	-22725, 29692, 22725, -12299, // w15 w14 w11 w10
	31521, 26722, 26722, -6270, // w21 w20 w17 w16
	17855, -31521, 6270, -17855, // w29 w28 w25 w24
	17855, 6270, -31521, -17855, // w23 w22 w19 w18
	6270, 26722, 26722, -31521}; // w31 w30 w27 w26

// Table for rows 2,6 - constants are multiplied on cos_2_16
//movq -> w05 w04 w01 w00
JPGD_SIMD_ALIGN(short, shortM128_tab_i_26[]) = {
	21407, 27969, 21407, 11585,
	21407, -11585, 21407, -27969, // w13 w12 w09 w08
	21407, 11585, -21407, -27969, // w07 w06 w03 w02
	-21407, 27969, 21407, -11585, // w15 w14 w11 w10
	29692, 25172, 25172, -5906,	// w21 w20 w17 w16
	16819, -29692, 5906, -16819, // w29 w28 w25 w24
	16819, 5906, -29692, -16819, // w23 w22 w19 w18
	5906, 25172, 25172, -29692}; // w31 w30 w27 w26
// Table for rows 3,5 - constants are multiplied on cos_3_16
//movq -> w05 w04 w01 w00
JPGD_SIMD_ALIGN(short, shortM128_tab_i_35[]) = {
	19266, 25172, 19266, 10426,
	19266, -10426, 19266, -25172, // w13 w12 w09 w08
	19266, 10426, -19266, -25172, // w07 w06 w03 w02
	-19266, 25172, 19266, -10426, // w15 w14 w11 w10
	26722, 22654, 22654, -5315, // w21 w20 w17 w16
	15137, -26722, 5315, -15137, // w29 w28 w25 w24
	15137, 5315, -26722, -15137, // w23 w22 w19 w18
	5315, 22654, 22654, -26722}; // w31 w30 w27 w26

JPGD_SIMD_ALIGN(short, shortM128_128[8]) = { 128, 128, 128, 128, 128, 128, 128, 128 };

void idctSSEShortU8(const short *pInput, uint8_t * pOutputUB)
{
	__m128i r_xmm0, r_xmm4;
	__m128i r_xmm1, r_xmm2, r_xmm3, r_xmm5, r_xmm6, r_xmm7;
	__m128i row0, row1, row2, row3, row4, row5, row6, row7;
	short * pTab_i_04 = shortM128_tab_i_04;
	short * pTab_i_26 = shortM128_tab_i_26;

	//Get pointers for this input and output
	pTab_i_04 = shortM128_tab_i_04;
	pTab_i_26 = shortM128_tab_i_26;

	//Row 1 and Row 3
	r_xmm0 = _mm_load_si128((__m128i *) pInput);
	r_xmm4 = _mm_load_si128((__m128i *) (&pInput[2*8]));

	// *** Work on the data in xmm0
	//low shuffle mask = 0xd8 = 11 01 10 00
	//get short 2 and short 0 into ls 32-bits
	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);

	// copy short 2 and short 0 to all locations
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
		
	// add to those copies
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i *) pTab_i_04));

	// shuffle mask = 0x55 = 01 01 01 01
	// copy short 3 and short 1 to all locations
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
		
	// high shuffle mask = 0xd8 = 11 01 10 00
	// get short 6 and short 4 into bit positions 64-95
	// get short 7 and short 5 into bit positions 96-127
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
		
	// add to short 3 and short 1
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i *) &pTab_i_04[16]));
		
	// shuffle mask = 0xaa = 10 10 10 10
	// copy short 6 and short 4 to all locations
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
		
	// shuffle mask = 0xaa = 11 11 11 11
	// copy short 7 and short 5 to all locations
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
		
	// add to short 6 and short 4
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i *) &pTab_i_04[8])); 
		
	// *** Work on the data in xmm4
	// high shuffle mask = 0xd8 11 01 10 00
	// get short 6 and short 4 into bit positions 64-95
	// get short 7 and short 5 into bit positions 96-127
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
		
	// (xmm0 short 2 and short 0 plus pSi) + some constants
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i *) shortM128_round_inv_row));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i *) &pTab_i_04[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i *) &shortM128_tab_i_26[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i *) &shortM128_tab_i_26[8])); 
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i *) &shortM128_tab_i_26[16])); 
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, 12);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i *) shortM128_round_inv_row));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i *) &shortM128_tab_i_26[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, 12);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	row0 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, 12);
	r_xmm4 = _mm_srai_epi32(r_xmm4, 12);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	row2 = _mm_packs_epi32(r_xmm4, r_xmm6);

	//Row 5 and row 7
	r_xmm0 = _mm_load_si128((__m128i *) (&pInput[4*8]));
	r_xmm4 = _mm_load_si128((__m128i *) (&pInput[6*8]));

	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i *) pTab_i_04));
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i *) &pTab_i_04[16]));
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i *) &pTab_i_04[8])); 
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i *) shortM128_round_inv_row));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i *) &pTab_i_04[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i *) &shortM128_tab_i_26[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i *) &shortM128_tab_i_26[8])); 
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i *) &shortM128_tab_i_26[16])); 
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, 12);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i *) shortM128_round_inv_row));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i *) &shortM128_tab_i_26[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, 12);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	row4 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, 12);
	r_xmm4 = _mm_srai_epi32(r_xmm4, 12);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	row6 = _mm_packs_epi32(r_xmm4, r_xmm6);

	//Row 4 and row 2
	pTab_i_04 = shortM128_tab_i_35;
	pTab_i_26 = shortM128_tab_i_17;
	r_xmm0 = _mm_load_si128((__m128i *) (&pInput[3*8]));
	r_xmm4 = _mm_load_si128((__m128i *) (&pInput[1*8]));

	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i *) pTab_i_04));
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i *) &pTab_i_04[16]));
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i *) &pTab_i_04[8])); 
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i *) shortM128_round_inv_row));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i *) &pTab_i_04[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i *) &pTab_i_26[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i *) &pTab_i_26[8])); 
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i *) &pTab_i_26[16])); 
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, 12);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i *) shortM128_round_inv_row));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i *) &pTab_i_26[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, 12);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	row3 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, 12);
	r_xmm4 = _mm_srai_epi32(r_xmm4, 12);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	row1 = _mm_packs_epi32(r_xmm4, r_xmm6);

	//Row 6 and row 8
	r_xmm0 = _mm_load_si128((__m128i *) (&pInput[5*8]));
	r_xmm4 = _mm_load_si128((__m128i *) (&pInput[7*8]));

	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i *) pTab_i_04));
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i *) &pTab_i_04[16]));
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i *) &pTab_i_04[8])); 
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i *) shortM128_round_inv_row));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i *) &pTab_i_04[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i *) &pTab_i_26[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i *) &pTab_i_26[8])); 
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i *) &pTab_i_26[16])); 
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, 12);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i *) shortM128_round_inv_row));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i *) &pTab_i_26[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, 12);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	row5 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, 12);
	r_xmm4 = _mm_srai_epi32(r_xmm4, 12);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	row7 = _mm_packs_epi32(r_xmm4, r_xmm6);

	r_xmm1 = _mm_load_si128((__m128i *) shortM128_tg_3_16);
	r_xmm2 = row5;
	r_xmm3 = row3;
	r_xmm0 = _mm_mulhi_epi16(row5, r_xmm1);

	r_xmm1 = _mm_mulhi_epi16(r_xmm1, r_xmm3);
	r_xmm5 = _mm_load_si128((__m128i *) shortM128_tg_1_16);
	r_xmm6 = row7;
	r_xmm4 = _mm_mulhi_epi16(row7, r_xmm5);

	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm2);
	r_xmm5 = _mm_mulhi_epi16(r_xmm5, row1);
	r_xmm1 = _mm_adds_epi16(r_xmm1, r_xmm3);
	r_xmm7 = row6;

	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm3);
	r_xmm3 = _mm_load_si128((__m128i *) shortM128_tg_2_16);
	r_xmm2 = _mm_subs_epi16(r_xmm2, r_xmm1);
	r_xmm7 = _mm_mulhi_epi16(r_xmm7, r_xmm3);
	r_xmm1 = r_xmm0;
	r_xmm3 = _mm_mulhi_epi16(r_xmm3, row2);
	r_xmm5 = _mm_subs_epi16(r_xmm5, r_xmm6);
	r_xmm4 = _mm_adds_epi16(r_xmm4, row1);
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm4);
	r_xmm0 = _mm_adds_epi16(r_xmm0, *((__m128i *) shortM128_one_corr));
	r_xmm4 = _mm_subs_epi16(r_xmm4, r_xmm1);
	r_xmm6 = r_xmm5;
	r_xmm5 = _mm_subs_epi16(r_xmm5, r_xmm2);
	r_xmm5 = _mm_adds_epi16(r_xmm5, *((__m128i *) shortM128_one_corr));
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm2);

	//Intermediate results, needed later
	__m128i temp3, temp7;
	temp7 = r_xmm0;

	r_xmm1 = r_xmm4;
	r_xmm0 = _mm_load_si128((__m128i *) shortM128_cos_4_16);
	r_xmm4 = _mm_adds_epi16(r_xmm4, r_xmm5);
	r_xmm2 = _mm_load_si128((__m128i *) shortM128_cos_4_16);
	r_xmm2 = _mm_mulhi_epi16(r_xmm2, r_xmm4);

	//Intermediate results, needed later
	temp3 = r_xmm6;

	r_xmm1 = _mm_subs_epi16(r_xmm1, r_xmm5);
	r_xmm7 = _mm_adds_epi16(r_xmm7, row2);
	r_xmm3 = _mm_subs_epi16(r_xmm3, row6);
	r_xmm6 = row0;
	r_xmm0 = _mm_mulhi_epi16(r_xmm0, r_xmm1);
	r_xmm5 = row4;
	r_xmm5 = _mm_adds_epi16(r_xmm5, r_xmm6);
	r_xmm6 = _mm_subs_epi16(r_xmm6, row4);
	r_xmm4 = _mm_adds_epi16(r_xmm4, r_xmm2);

	r_xmm4 = _mm_or_si128(r_xmm4, *((__m128i *) shortM128_one_corr));
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm1);
	r_xmm0 = _mm_or_si128(r_xmm0, *((__m128i *) shortM128_one_corr));

	r_xmm2 = r_xmm5;
	r_xmm5 = _mm_adds_epi16(r_xmm5, r_xmm7);
	r_xmm1 = r_xmm6;
	r_xmm5 = _mm_adds_epi16(r_xmm5, *((__m128i *) shortM128_round_inv_col));
	r_xmm2 = _mm_subs_epi16(r_xmm2, r_xmm7);
	r_xmm7 = temp7;
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm3);
	r_xmm6 = _mm_adds_epi16(r_xmm6, *((__m128i *) shortM128_round_inv_col));
	r_xmm7 = _mm_adds_epi16(r_xmm7, r_xmm5);
	r_xmm7 = _mm_srai_epi16(r_xmm7, SHIFT_INV_COL);
	r_xmm1 = _mm_subs_epi16(r_xmm1, r_xmm3);
	r_xmm1 = _mm_adds_epi16(r_xmm1, *((__m128i *) shortM128_round_inv_corr));
	r_xmm3 = r_xmm6;
	r_xmm2 = _mm_adds_epi16(r_xmm2, *((__m128i *) shortM128_round_inv_corr));
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm4);

	//Store results for row 0
	//_mm_store_si128((__m128i *) pOutput, r_xmm7);
	__m128i r0 = r_xmm7;

	r_xmm6 = _mm_srai_epi16(r_xmm6, SHIFT_INV_COL);
	r_xmm7 = r_xmm1;
	r_xmm1 = _mm_adds_epi16(r_xmm1, r_xmm0);

	//Store results for row 1
	//_mm_store_si128((__m128i *) (&pOutput[1*8]), r_xmm6); 
	__m128i r1 = r_xmm6;

	r_xmm1 = _mm_srai_epi16(r_xmm1, SHIFT_INV_COL);
	r_xmm6 = temp3;
	r_xmm7 = _mm_subs_epi16(r_xmm7, r_xmm0);
	r_xmm7 = _mm_srai_epi16(r_xmm7, SHIFT_INV_COL);

	//Store results for row 2
	//_mm_store_si128((__m128i *) (&pOutput[2*8]), r_xmm1); 
	__m128i r2 = r_xmm1;

	r_xmm5 = _mm_subs_epi16(r_xmm5, temp7); 
	r_xmm5 = _mm_srai_epi16(r_xmm5, SHIFT_INV_COL);

	//Store results for row 7
	//_mm_store_si128((__m128i *) (&pOutput[7*8]), r_xmm5); 
	__m128i r7 = r_xmm5;

	r_xmm3 = _mm_subs_epi16(r_xmm3, r_xmm4);
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm2);
	r_xmm2 = _mm_subs_epi16(r_xmm2, temp3); 
	r_xmm6 = _mm_srai_epi16(r_xmm6, SHIFT_INV_COL);
	r_xmm2 = _mm_srai_epi16(r_xmm2, SHIFT_INV_COL);

	//Store results for row 3
	//_mm_store_si128((__m128i *) (&pOutput[3*8]), r_xmm6); 
	__m128i r3 = r_xmm6;

	r_xmm3 = _mm_srai_epi16(r_xmm3, SHIFT_INV_COL);

	//Store results for rows 4, 5, and 6
	//_mm_store_si128((__m128i *) (&pOutput[4*8]), r_xmm2);
	//_mm_store_si128((__m128i *) (&pOutput[5*8]), r_xmm7);
	//_mm_store_si128((__m128i *) (&pOutput[6*8]), r_xmm3);

	__m128i r4 = r_xmm2;
	__m128i r5 = r_xmm7;
	__m128i r6 = r_xmm3;

	r0 = _mm_add_epi16(*(const __m128i *)shortM128_128, r0);
	r1 = _mm_add_epi16(*(const __m128i *)shortM128_128, r1);
	r2 = _mm_add_epi16(*(const __m128i *)shortM128_128, r2);
	r3 = _mm_add_epi16(*(const __m128i *)shortM128_128, r3);
	r4 = _mm_add_epi16(*(const __m128i *)shortM128_128, r4);
	r5 = _mm_add_epi16(*(const __m128i *)shortM128_128, r5);
	r6 = _mm_add_epi16(*(const __m128i *)shortM128_128, r6);
	r7 = _mm_add_epi16(*(const __m128i *)shortM128_128, r7);

	((__m128i *)pOutputUB)[0] = _mm_packus_epi16(r0, r1);
	((__m128i *)pOutputUB)[1] = _mm_packus_epi16(r2, r3);
	((__m128i *)pOutputUB)[2] = _mm_packus_epi16(r4, r5);
	((__m128i *)pOutputUB)[3] = _mm_packus_epi16(r6, r7);
}
