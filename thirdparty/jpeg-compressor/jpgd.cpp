// jpgd.cpp - C++ class for JPEG decompression. Written by Richard Geldreich <richgel99@gmail.com> between 1994-2020.
// Supports progressive and baseline sequential JPEG image files, and the most common chroma subsampling factors: Y, H1V1, H2V1, H1V2, and H2V2.
// Supports box and linear chroma upsampling.
//
// Released under two licenses. You are free to choose which license you want:
// License 1: 
// Public Domain
//
// License 2:
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
//
// Alex Evans: Linear memory allocator (taken from jpge.h).
// v1.04, May. 19, 2012: Code tweaks to fix VS2008 static code analysis warnings
// v2.00, March 20, 2020: Fuzzed with zzuf and afl. Fixed several issues, converted most assert()'s to run-time checks. Added chroma upsampling. Removed freq. domain upsampling. gcc/clang warnings.
//
// Important:
// #define JPGD_USE_SSE2 to 0 to completely disable SSE2 usage.
//
#include "jpgd.h"
#include <string.h>
#include <algorithm>
#include <assert.h>

#ifdef _MSC_VER
#pragma warning (disable : 4611) // warning C4611: interaction between '_setjmp' and C++ object destruction is non-portable
#endif

#ifndef JPGD_USE_SSE2

	#if defined(__GNUC__)
		#if defined(__SSE2__)
			#define JPGD_USE_SSE2 (1)
		#endif
	#elif defined(_MSC_VER)
		#if defined(_M_X64)
			#define JPGD_USE_SSE2 (1)
		#endif
	#endif

#endif

#define JPGD_TRUE (1)
#define JPGD_FALSE (0)

#define JPGD_MAX(a,b) (((a)>(b)) ? (a) : (b))
#define JPGD_MIN(a,b) (((a)<(b)) ? (a) : (b))

namespace jpgd {

	static inline void* jpgd_malloc(size_t nSize) { return malloc(nSize); }
	static inline void jpgd_free(void* p) { free(p); }

	// DCT coefficients are stored in this sequence.
	static int g_ZAG[64] = { 0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63 };

	enum JPEG_MARKER
	{
		M_SOF0 = 0xC0, M_SOF1 = 0xC1, M_SOF2 = 0xC2, M_SOF3 = 0xC3, M_SOF5 = 0xC5, M_SOF6 = 0xC6, M_SOF7 = 0xC7, M_JPG = 0xC8,
		M_SOF9 = 0xC9, M_SOF10 = 0xCA, M_SOF11 = 0xCB, M_SOF13 = 0xCD, M_SOF14 = 0xCE, M_SOF15 = 0xCF, M_DHT = 0xC4, M_DAC = 0xCC,
		M_RST0 = 0xD0, M_RST1 = 0xD1, M_RST2 = 0xD2, M_RST3 = 0xD3, M_RST4 = 0xD4, M_RST5 = 0xD5, M_RST6 = 0xD6, M_RST7 = 0xD7,
		M_SOI = 0xD8, M_EOI = 0xD9, M_SOS = 0xDA, M_DQT = 0xDB, M_DNL = 0xDC, M_DRI = 0xDD, M_DHP = 0xDE, M_EXP = 0xDF,
		M_APP0 = 0xE0, M_APP15 = 0xEF, M_JPG0 = 0xF0, M_JPG13 = 0xFD, M_COM = 0xFE, M_TEM = 0x01, M_ERROR = 0x100, RST0 = 0xD0
	};

	enum JPEG_SUBSAMPLING { JPGD_GRAYSCALE = 0, JPGD_YH1V1, JPGD_YH2V1, JPGD_YH1V2, JPGD_YH2V2 };

#if JPGD_USE_SSE2
#include "jpgd_idct.h"
#endif

#define CONST_BITS  13
#define PASS1_BITS  2
#define SCALEDONE ((int32)1)

#define FIX_0_298631336  ((int32)2446)        /* FIX(0.298631336) */
#define FIX_0_390180644  ((int32)3196)        /* FIX(0.390180644) */
#define FIX_0_541196100  ((int32)4433)        /* FIX(0.541196100) */
#define FIX_0_765366865  ((int32)6270)        /* FIX(0.765366865) */
#define FIX_0_899976223  ((int32)7373)        /* FIX(0.899976223) */
#define FIX_1_175875602  ((int32)9633)        /* FIX(1.175875602) */
#define FIX_1_501321110  ((int32)12299)       /* FIX(1.501321110) */
#define FIX_1_847759065  ((int32)15137)       /* FIX(1.847759065) */
#define FIX_1_961570560  ((int32)16069)       /* FIX(1.961570560) */
#define FIX_2_053119869  ((int32)16819)       /* FIX(2.053119869) */
#define FIX_2_562915447  ((int32)20995)       /* FIX(2.562915447) */
#define FIX_3_072711026  ((int32)25172)       /* FIX(3.072711026) */

#define DESCALE(x,n)  (((x) + (SCALEDONE << ((n)-1))) >> (n))
#define DESCALE_ZEROSHIFT(x,n)  (((x) + (128 << (n)) + (SCALEDONE << ((n)-1))) >> (n))

#define MULTIPLY(var, cnst)  ((var) * (cnst))

#define CLAMP(i) ((static_cast<uint>(i) > 255) ? (((~i) >> 31) & 0xFF) : (i))

	static inline int left_shifti(int val, uint32_t bits)
	{
		return static_cast<int>(static_cast<uint32_t>(val) << bits);
	}

	// Compiler creates a fast path 1D IDCT for X non-zero columns
	template <int NONZERO_COLS>
	struct Row
	{
		static void idct(int* pTemp, const jpgd_block_coeff_t* pSrc)
		{
			// ACCESS_COL() will be optimized at compile time to either an array access, or 0. Good compilers will then optimize out muls against 0.
#define ACCESS_COL(x) (((x) < NONZERO_COLS) ? (int)pSrc[x] : 0)

			const int z2 = ACCESS_COL(2), z3 = ACCESS_COL(6);

			const int z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
			const int tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065);
			const int tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865);

			const int tmp0 = left_shifti(ACCESS_COL(0) + ACCESS_COL(4), CONST_BITS);
			const int tmp1 = left_shifti(ACCESS_COL(0) - ACCESS_COL(4), CONST_BITS);

			const int tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;

			const int atmp0 = ACCESS_COL(7), atmp1 = ACCESS_COL(5), atmp2 = ACCESS_COL(3), atmp3 = ACCESS_COL(1);

			const int bz1 = atmp0 + atmp3, bz2 = atmp1 + atmp2, bz3 = atmp0 + atmp2, bz4 = atmp1 + atmp3;
			const int bz5 = MULTIPLY(bz3 + bz4, FIX_1_175875602);

			const int az1 = MULTIPLY(bz1, -FIX_0_899976223);
			const int az2 = MULTIPLY(bz2, -FIX_2_562915447);
			const int az3 = MULTIPLY(bz3, -FIX_1_961570560) + bz5;
			const int az4 = MULTIPLY(bz4, -FIX_0_390180644) + bz5;

			const int btmp0 = MULTIPLY(atmp0, FIX_0_298631336) + az1 + az3;
			const int btmp1 = MULTIPLY(atmp1, FIX_2_053119869) + az2 + az4;
			const int btmp2 = MULTIPLY(atmp2, FIX_3_072711026) + az2 + az3;
			const int btmp3 = MULTIPLY(atmp3, FIX_1_501321110) + az1 + az4;

			pTemp[0] = DESCALE(tmp10 + btmp3, CONST_BITS - PASS1_BITS);
			pTemp[7] = DESCALE(tmp10 - btmp3, CONST_BITS - PASS1_BITS);
			pTemp[1] = DESCALE(tmp11 + btmp2, CONST_BITS - PASS1_BITS);
			pTemp[6] = DESCALE(tmp11 - btmp2, CONST_BITS - PASS1_BITS);
			pTemp[2] = DESCALE(tmp12 + btmp1, CONST_BITS - PASS1_BITS);
			pTemp[5] = DESCALE(tmp12 - btmp1, CONST_BITS - PASS1_BITS);
			pTemp[3] = DESCALE(tmp13 + btmp0, CONST_BITS - PASS1_BITS);
			pTemp[4] = DESCALE(tmp13 - btmp0, CONST_BITS - PASS1_BITS);
		}
	};

	template <>
	struct Row<0>
	{
		static void idct(int* pTemp, const jpgd_block_coeff_t* pSrc)
		{
			(void)pTemp; 
			(void)pSrc;
		}
	};

	template <>
	struct Row<1>
	{
		static void idct(int* pTemp, const jpgd_block_coeff_t* pSrc)
		{
			const int dcval = left_shifti(pSrc[0], PASS1_BITS);

			pTemp[0] = dcval;
			pTemp[1] = dcval;
			pTemp[2] = dcval;
			pTemp[3] = dcval;
			pTemp[4] = dcval;
			pTemp[5] = dcval;
			pTemp[6] = dcval;
			pTemp[7] = dcval;
		}
	};

	// Compiler creates a fast path 1D IDCT for X non-zero rows
	template <int NONZERO_ROWS>
	struct Col
	{
		static void idct(uint8* pDst_ptr, const int* pTemp)
		{
			// ACCESS_ROW() will be optimized at compile time to either an array access, or 0.
#define ACCESS_ROW(x) (((x) < NONZERO_ROWS) ? pTemp[x * 8] : 0)

			const int z2 = ACCESS_ROW(2);
			const int z3 = ACCESS_ROW(6);

			const int z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
			const int tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065);
			const int tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865);

			const int tmp0 = left_shifti(ACCESS_ROW(0) + ACCESS_ROW(4), CONST_BITS);
			const int tmp1 = left_shifti(ACCESS_ROW(0) - ACCESS_ROW(4), CONST_BITS);

			const int tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;

			const int atmp0 = ACCESS_ROW(7), atmp1 = ACCESS_ROW(5), atmp2 = ACCESS_ROW(3), atmp3 = ACCESS_ROW(1);

			const int bz1 = atmp0 + atmp3, bz2 = atmp1 + atmp2, bz3 = atmp0 + atmp2, bz4 = atmp1 + atmp3;
			const int bz5 = MULTIPLY(bz3 + bz4, FIX_1_175875602);

			const int az1 = MULTIPLY(bz1, -FIX_0_899976223);
			const int az2 = MULTIPLY(bz2, -FIX_2_562915447);
			const int az3 = MULTIPLY(bz3, -FIX_1_961570560) + bz5;
			const int az4 = MULTIPLY(bz4, -FIX_0_390180644) + bz5;

			const int btmp0 = MULTIPLY(atmp0, FIX_0_298631336) + az1 + az3;
			const int btmp1 = MULTIPLY(atmp1, FIX_2_053119869) + az2 + az4;
			const int btmp2 = MULTIPLY(atmp2, FIX_3_072711026) + az2 + az3;
			const int btmp3 = MULTIPLY(atmp3, FIX_1_501321110) + az1 + az4;

			int i = DESCALE_ZEROSHIFT(tmp10 + btmp3, CONST_BITS + PASS1_BITS + 3);
			pDst_ptr[8 * 0] = (uint8)CLAMP(i);

			i = DESCALE_ZEROSHIFT(tmp10 - btmp3, CONST_BITS + PASS1_BITS + 3);
			pDst_ptr[8 * 7] = (uint8)CLAMP(i);

			i = DESCALE_ZEROSHIFT(tmp11 + btmp2, CONST_BITS + PASS1_BITS + 3);
			pDst_ptr[8 * 1] = (uint8)CLAMP(i);

			i = DESCALE_ZEROSHIFT(tmp11 - btmp2, CONST_BITS + PASS1_BITS + 3);
			pDst_ptr[8 * 6] = (uint8)CLAMP(i);

			i = DESCALE_ZEROSHIFT(tmp12 + btmp1, CONST_BITS + PASS1_BITS + 3);
			pDst_ptr[8 * 2] = (uint8)CLAMP(i);

			i = DESCALE_ZEROSHIFT(tmp12 - btmp1, CONST_BITS + PASS1_BITS + 3);
			pDst_ptr[8 * 5] = (uint8)CLAMP(i);

			i = DESCALE_ZEROSHIFT(tmp13 + btmp0, CONST_BITS + PASS1_BITS + 3);
			pDst_ptr[8 * 3] = (uint8)CLAMP(i);

			i = DESCALE_ZEROSHIFT(tmp13 - btmp0, CONST_BITS + PASS1_BITS + 3);
			pDst_ptr[8 * 4] = (uint8)CLAMP(i);
		}
	};

	template <>
	struct Col<1>
	{
		static void idct(uint8* pDst_ptr, const int* pTemp)
		{
			int dcval = DESCALE_ZEROSHIFT(pTemp[0], PASS1_BITS + 3);
			const uint8 dcval_clamped = (uint8)CLAMP(dcval);
			pDst_ptr[0 * 8] = dcval_clamped;
			pDst_ptr[1 * 8] = dcval_clamped;
			pDst_ptr[2 * 8] = dcval_clamped;
			pDst_ptr[3 * 8] = dcval_clamped;
			pDst_ptr[4 * 8] = dcval_clamped;
			pDst_ptr[5 * 8] = dcval_clamped;
			pDst_ptr[6 * 8] = dcval_clamped;
			pDst_ptr[7 * 8] = dcval_clamped;
		}
	};

	static const uint8 s_idct_row_table[] =
	{
	  1,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0, 2,1,0,0,0,0,0,0, 2,1,1,0,0,0,0,0, 2,2,1,0,0,0,0,0, 3,2,1,0,0,0,0,0, 4,2,1,0,0,0,0,0, 4,3,1,0,0,0,0,0,
	  4,3,2,0,0,0,0,0, 4,3,2,1,0,0,0,0, 4,3,2,1,1,0,0,0, 4,3,2,2,1,0,0,0, 4,3,3,2,1,0,0,0, 4,4,3,2,1,0,0,0, 5,4,3,2,1,0,0,0, 6,4,3,2,1,0,0,0,
	  6,5,3,2,1,0,0,0, 6,5,4,2,1,0,0,0, 6,5,4,3,1,0,0,0, 6,5,4,3,2,0,0,0, 6,5,4,3,2,1,0,0, 6,5,4,3,2,1,1,0, 6,5,4,3,2,2,1,0, 6,5,4,3,3,2,1,0,
	  6,5,4,4,3,2,1,0, 6,5,5,4,3,2,1,0, 6,6,5,4,3,2,1,0, 7,6,5,4,3,2,1,0, 8,6,5,4,3,2,1,0, 8,7,5,4,3,2,1,0, 8,7,6,4,3,2,1,0, 8,7,6,5,3,2,1,0,
	  8,7,6,5,4,2,1,0, 8,7,6,5,4,3,1,0, 8,7,6,5,4,3,2,0, 8,7,6,5,4,3,2,1, 8,7,6,5,4,3,2,2, 8,7,6,5,4,3,3,2, 8,7,6,5,4,4,3,2, 8,7,6,5,5,4,3,2,
	  8,7,6,6,5,4,3,2, 8,7,7,6,5,4,3,2, 8,8,7,6,5,4,3,2, 8,8,8,6,5,4,3,2, 8,8,8,7,5,4,3,2, 8,8,8,7,6,4,3,2, 8,8,8,7,6,5,3,2, 8,8,8,7,6,5,4,2,
	  8,8,8,7,6,5,4,3, 8,8,8,7,6,5,4,4, 8,8,8,7,6,5,5,4, 8,8,8,7,6,6,5,4, 8,8,8,7,7,6,5,4, 8,8,8,8,7,6,5,4, 8,8,8,8,8,6,5,4, 8,8,8,8,8,7,5,4,
	  8,8,8,8,8,7,6,4, 8,8,8,8,8,7,6,5, 8,8,8,8,8,7,6,6, 8,8,8,8,8,7,7,6, 8,8,8,8,8,8,7,6, 8,8,8,8,8,8,8,6, 8,8,8,8,8,8,8,7, 8,8,8,8,8,8,8,8,
	};

	static const uint8 s_idct_col_table[] = 
	{ 
		1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
		7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8 
	};

	// Scalar "fast pathing" IDCT.
	static void idct(const jpgd_block_coeff_t* pSrc_ptr, uint8* pDst_ptr, int block_max_zag, bool use_simd)
	{
		(void)use_simd;

		assert(block_max_zag >= 1);
		assert(block_max_zag <= 64);
				
		if (block_max_zag <= 1)
		{
			int k = ((pSrc_ptr[0] + 4) >> 3) + 128;
			k = CLAMP(k);
			k = k | (k << 8);
			k = k | (k << 16);

			for (int i = 8; i > 0; i--)
			{
				*(int*)&pDst_ptr[0] = k;
				*(int*)&pDst_ptr[4] = k;
				pDst_ptr += 8;
			}
			return;
		}

#if JPGD_USE_SSE2
		if (use_simd)
		{
			assert((((uintptr_t)pSrc_ptr) & 15) == 0);
			assert((((uintptr_t)pDst_ptr) & 15) == 0);
			idctSSEShortU8(pSrc_ptr, pDst_ptr);
			return;
		}
#endif

		int temp[64];

		const jpgd_block_coeff_t* pSrc = pSrc_ptr;
		int* pTemp = temp;

		const uint8* pRow_tab = &s_idct_row_table[(block_max_zag - 1) * 8];
		int i;
		for (i = 8; i > 0; i--, pRow_tab++)
		{
			switch (*pRow_tab)
			{
			case 0: Row<0>::idct(pTemp, pSrc); break;
			case 1: Row<1>::idct(pTemp, pSrc); break;
			case 2: Row<2>::idct(pTemp, pSrc); break;
			case 3: Row<3>::idct(pTemp, pSrc); break;
			case 4: Row<4>::idct(pTemp, pSrc); break;
			case 5: Row<5>::idct(pTemp, pSrc); break;
			case 6: Row<6>::idct(pTemp, pSrc); break;
			case 7: Row<7>::idct(pTemp, pSrc); break;
			case 8: Row<8>::idct(pTemp, pSrc); break;
			}

			pSrc += 8;
			pTemp += 8;
		}

		pTemp = temp;

		const int nonzero_rows = s_idct_col_table[block_max_zag - 1];
		for (i = 8; i > 0; i--)
		{
			switch (nonzero_rows)
			{
			case 1: Col<1>::idct(pDst_ptr, pTemp); break;
			case 2: Col<2>::idct(pDst_ptr, pTemp); break;
			case 3: Col<3>::idct(pDst_ptr, pTemp); break;
			case 4: Col<4>::idct(pDst_ptr, pTemp); break;
			case 5: Col<5>::idct(pDst_ptr, pTemp); break;
			case 6: Col<6>::idct(pDst_ptr, pTemp); break;
			case 7: Col<7>::idct(pDst_ptr, pTemp); break;
			case 8: Col<8>::idct(pDst_ptr, pTemp); break;
			}

			pTemp++;
			pDst_ptr++;
		}
	}

	// Retrieve one character from the input stream.
	inline uint jpeg_decoder::get_char()
	{
		// Any bytes remaining in buffer?
		if (!m_in_buf_left)
		{
			// Try to get more bytes.
			prep_in_buffer();
			// Still nothing to get?
			if (!m_in_buf_left)
			{
				// Pad the end of the stream with 0xFF 0xD9 (EOI marker)
				int t = m_tem_flag;
				m_tem_flag ^= 1;
				if (t)
					return 0xD9;
				else
					return 0xFF;
			}
		}

		uint c = *m_pIn_buf_ofs++;
		m_in_buf_left--;

		return c;
	}

	// Same as previous method, except can indicate if the character is a pad character or not.
	inline uint jpeg_decoder::get_char(bool* pPadding_flag)
	{
		if (!m_in_buf_left)
		{
			prep_in_buffer();
			if (!m_in_buf_left)
			{
				*pPadding_flag = true;
				int t = m_tem_flag;
				m_tem_flag ^= 1;
				if (t)
					return 0xD9;
				else
					return 0xFF;
			}
		}

		*pPadding_flag = false;

		uint c = *m_pIn_buf_ofs++;
		m_in_buf_left--;

		return c;
	}

	// Inserts a previously retrieved character back into the input buffer.
	inline void jpeg_decoder::stuff_char(uint8 q)
	{
		// This could write before the input buffer, but we've placed another array there.
		*(--m_pIn_buf_ofs) = q;
		m_in_buf_left++;
	}

	// Retrieves one character from the input stream, but does not read past markers. Will continue to return 0xFF when a marker is encountered.
	inline uint8 jpeg_decoder::get_octet()
	{
		bool padding_flag;
		int c = get_char(&padding_flag);

		if (c == 0xFF)
		{
			if (padding_flag)
				return 0xFF;

			c = get_char(&padding_flag);
			if (padding_flag)
			{
				stuff_char(0xFF);
				return 0xFF;
			}

			if (c == 0x00)
				return 0xFF;
			else
			{
				stuff_char(static_cast<uint8>(c));
				stuff_char(0xFF);
				return 0xFF;
			}
		}

		return static_cast<uint8>(c);
	}

	// Retrieves a variable number of bits from the input stream. Does not recognize markers.
	inline uint jpeg_decoder::get_bits(int num_bits)
	{
		if (!num_bits)
			return 0;

		uint i = m_bit_buf >> (32 - num_bits);

		if ((m_bits_left -= num_bits) <= 0)
		{
			m_bit_buf <<= (num_bits += m_bits_left);

			uint c1 = get_char();
			uint c2 = get_char();
			m_bit_buf = (m_bit_buf & 0xFFFF0000) | (c1 << 8) | c2;

			m_bit_buf <<= -m_bits_left;

			m_bits_left += 16;

			assert(m_bits_left >= 0);
		}
		else
			m_bit_buf <<= num_bits;

		return i;
	}

	// Retrieves a variable number of bits from the input stream. Markers will not be read into the input bit buffer. Instead, an infinite number of all 1's will be returned when a marker is encountered.
	inline uint jpeg_decoder::get_bits_no_markers(int num_bits)
	{
		if (!num_bits)
			return 0;

		assert(num_bits <= 16);

		uint i = m_bit_buf >> (32 - num_bits);

		if ((m_bits_left -= num_bits) <= 0)
		{
			m_bit_buf <<= (num_bits += m_bits_left);

			if ((m_in_buf_left < 2) || (m_pIn_buf_ofs[0] == 0xFF) || (m_pIn_buf_ofs[1] == 0xFF))
			{
				uint c1 = get_octet();
				uint c2 = get_octet();
				m_bit_buf |= (c1 << 8) | c2;
			}
			else
			{
				m_bit_buf |= ((uint)m_pIn_buf_ofs[0] << 8) | m_pIn_buf_ofs[1];
				m_in_buf_left -= 2;
				m_pIn_buf_ofs += 2;
			}

			m_bit_buf <<= -m_bits_left;

			m_bits_left += 16;

			assert(m_bits_left >= 0);
		}
		else
			m_bit_buf <<= num_bits;

		return i;
	}

	// Decodes a Huffman encoded symbol.
	inline int jpeg_decoder::huff_decode(huff_tables* pH)
	{
		if (!pH)
			stop_decoding(JPGD_DECODE_ERROR);

		int symbol;
		// Check first 8-bits: do we have a complete symbol?
		if ((symbol = pH->look_up[m_bit_buf >> 24]) < 0)
		{
			// Decode more bits, use a tree traversal to find symbol.
			int ofs = 23;
			do
			{
				unsigned int idx = -(int)(symbol + ((m_bit_buf >> ofs) & 1));

				// This should never happen, but to be safe I'm turning these asserts into a run-time check.
				if ((idx >= JPGD_HUFF_TREE_MAX_LENGTH) || (ofs < 0))
					stop_decoding(JPGD_DECODE_ERROR);

				symbol = pH->tree[idx];
				ofs--;
			} while (symbol < 0);

			get_bits_no_markers(8 + (23 - ofs));
		}
		else
		{
			assert(symbol < JPGD_HUFF_CODE_SIZE_MAX_LENGTH);
			get_bits_no_markers(pH->code_size[symbol]);
		}

		return symbol;
	}

	// Decodes a Huffman encoded symbol.
	inline int jpeg_decoder::huff_decode(huff_tables* pH, int& extra_bits)
	{
		int symbol;

		if (!pH)
			stop_decoding(JPGD_DECODE_ERROR);

		// Check first 8-bits: do we have a complete symbol?
		if ((symbol = pH->look_up2[m_bit_buf >> 24]) < 0)
		{
			// Use a tree traversal to find symbol.
			int ofs = 23;
			do
			{
				unsigned int idx = -(int)(symbol + ((m_bit_buf >> ofs) & 1));

				// This should never happen, but to be safe I'm turning these asserts into a run-time check.
				if ((idx >= JPGD_HUFF_TREE_MAX_LENGTH) || (ofs < 0))
					stop_decoding(JPGD_DECODE_ERROR);

				symbol = pH->tree[idx];
				ofs--;
			} while (symbol < 0);

			get_bits_no_markers(8 + (23 - ofs));

			extra_bits = get_bits_no_markers(symbol & 0xF);
		}
		else
		{
			if (symbol & 0x8000)
			{
				//get_bits_no_markers((symbol >> 8) & 31);
				assert(((symbol >> 8) & 31) <= 15);
				get_bits_no_markers((symbol >> 8) & 15);
				extra_bits = symbol >> 16;
			}
			else
			{
				int code_size = (symbol >> 8) & 31;
				int num_extra_bits = symbol & 0xF;
				int bits = code_size + num_extra_bits;

				if (bits <= 16)
					extra_bits = get_bits_no_markers(bits) & ((1 << num_extra_bits) - 1);
				else
				{
					get_bits_no_markers(code_size);
					extra_bits = get_bits_no_markers(num_extra_bits);
				}
			}

			symbol &= 0xFF;
		}

		return symbol;
	}

	// Tables and macro used to fully decode the DPCM differences.
	static const int s_extend_test[16] = { 0, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000 };
	static const int s_extend_offset[16] = { 0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047, -4095, -8191, -16383, -32767 };
	//static const int s_extend_mask[] = { 0, (1 << 0), (1 << 1), (1 << 2), (1 << 3), (1 << 4), (1 << 5), (1 << 6), (1 << 7), (1 << 8), (1 << 9), (1 << 10), (1 << 11), (1 << 12), (1 << 13), (1 << 14), (1 << 15), (1 << 16) };

#define JPGD_HUFF_EXTEND(x, s) (((x) < s_extend_test[s & 15]) ? ((x) + s_extend_offset[s & 15]) : (x))

	// Unconditionally frees all allocated m_blocks.
	void jpeg_decoder::free_all_blocks()
	{
		m_pStream = nullptr;
		for (mem_block* b = m_pMem_blocks; b; )
		{
			mem_block* n = b->m_pNext;
			jpgd_free(b);
			b = n;
		}
		m_pMem_blocks = nullptr;
	}

	// This method handles all errors. It will never return.
	// It could easily be changed to use C++ exceptions.
	JPGD_NORETURN void jpeg_decoder::stop_decoding(jpgd_status status)
	{
		m_error_code = status;
		free_all_blocks();
		longjmp(m_jmp_state, status);
	}
		
	void* jpeg_decoder::alloc(size_t nSize, bool zero)
	{
		nSize = (JPGD_MAX(nSize, 1) + 3) & ~3;
		char* rv = nullptr;
		for (mem_block* b = m_pMem_blocks; b; b = b->m_pNext)
		{
			if ((b->m_used_count + nSize) <= b->m_size)
			{
				rv = b->m_data + b->m_used_count;
				b->m_used_count += nSize;
				break;
			}
		}
		if (!rv)
		{
			int capacity = JPGD_MAX(32768 - 256, (nSize + 2047) & ~2047);
			mem_block* b = (mem_block*)jpgd_malloc(sizeof(mem_block) + capacity);
			if (!b)
			{
				stop_decoding(JPGD_NOTENOUGHMEM);
			}

			b->m_pNext = m_pMem_blocks;
			m_pMem_blocks = b;
			b->m_used_count = nSize;
			b->m_size = capacity;
			rv = b->m_data;
		}
		if (zero) memset(rv, 0, nSize);
		return rv;
	}

	void* jpeg_decoder::alloc_aligned(size_t nSize, uint32_t align, bool zero)
	{
		assert((align >= 1U) && ((align & (align - 1U)) == 0U));
		void *p = alloc(nSize + align - 1U, zero);
		p = (void *)( ((uintptr_t)p + (align - 1U)) & ~((uintptr_t)(align - 1U)) );
		return p;
	}

	void jpeg_decoder::word_clear(void* p, uint16 c, uint n)
	{
		uint8* pD = (uint8*)p;
		const uint8 l = c & 0xFF, h = (c >> 8) & 0xFF;
		while (n)
		{
			pD[0] = l;
			pD[1] = h;
			pD += 2;
			n--;
		}
	}

	// Refill the input buffer.
	// This method will sit in a loop until (A) the buffer is full or (B)
	// the stream's read() method reports and end of file condition.
	void jpeg_decoder::prep_in_buffer()
	{
		m_in_buf_left = 0;
		m_pIn_buf_ofs = m_in_buf;

		if (m_eof_flag)
			return;

		do
		{
			int bytes_read = m_pStream->read(m_in_buf + m_in_buf_left, JPGD_IN_BUF_SIZE - m_in_buf_left, &m_eof_flag);
			if (bytes_read == -1)
				stop_decoding(JPGD_STREAM_READ);

			m_in_buf_left += bytes_read;
		} while ((m_in_buf_left < JPGD_IN_BUF_SIZE) && (!m_eof_flag));

		m_total_bytes_read += m_in_buf_left;

		// Pad the end of the block with M_EOI (prevents the decompressor from going off the rails if the stream is invalid).
		// (This dates way back to when this decompressor was written in C/asm, and the all-asm Huffman decoder did some fancy things to increase perf.)
		word_clear(m_pIn_buf_ofs + m_in_buf_left, 0xD9FF, 64);
	}

	// Read a Huffman code table.
	void jpeg_decoder::read_dht_marker()
	{
		int i, index, count;
		uint8 huff_num[17];
		uint8 huff_val[256];

		uint num_left = get_bits(16);

		if (num_left < 2)
			stop_decoding(JPGD_BAD_DHT_MARKER);

		num_left -= 2;

		while (num_left)
		{
			index = get_bits(8);

			huff_num[0] = 0;

			count = 0;

			for (i = 1; i <= 16; i++)
			{
				huff_num[i] = static_cast<uint8>(get_bits(8));
				count += huff_num[i];
			}

			if (count > 255)
				stop_decoding(JPGD_BAD_DHT_COUNTS);

			bool symbol_present[256];
			memset(symbol_present, 0, sizeof(symbol_present));

			for (i = 0; i < count; i++)
			{
				const int s = get_bits(8);

				// Check for obviously bogus tables.
				if (symbol_present[s])
					stop_decoding(JPGD_BAD_DHT_COUNTS);

				huff_val[i] = static_cast<uint8_t>(s);
				symbol_present[s] = true;
			}

			i = 1 + 16 + count;

			if (num_left < (uint)i)
				stop_decoding(JPGD_BAD_DHT_MARKER);

			num_left -= i;

			if ((index & 0x10) > 0x10)
				stop_decoding(JPGD_BAD_DHT_INDEX);

			index = (index & 0x0F) + ((index & 0x10) >> 4) * (JPGD_MAX_HUFF_TABLES >> 1);

			if (index >= JPGD_MAX_HUFF_TABLES)
				stop_decoding(JPGD_BAD_DHT_INDEX);

			if (!m_huff_num[index])
				m_huff_num[index] = (uint8*)alloc(17);

			if (!m_huff_val[index])
				m_huff_val[index] = (uint8*)alloc(256);

			m_huff_ac[index] = (index & 0x10) != 0;
			memcpy(m_huff_num[index], huff_num, 17);
			memcpy(m_huff_val[index], huff_val, 256);
		}
	}

	// Read a quantization table.
	void jpeg_decoder::read_dqt_marker()
	{
		int n, i, prec;
		uint num_left;
		uint temp;

		num_left = get_bits(16);

		if (num_left < 2)
			stop_decoding(JPGD_BAD_DQT_MARKER);

		num_left -= 2;

		while (num_left)
		{
			n = get_bits(8);
			prec = n >> 4;
			n &= 0x0F;

			if (n >= JPGD_MAX_QUANT_TABLES)
				stop_decoding(JPGD_BAD_DQT_TABLE);

			if (!m_quant[n])
				m_quant[n] = (jpgd_quant_t*)alloc(64 * sizeof(jpgd_quant_t));

			// read quantization entries, in zag order
			for (i = 0; i < 64; i++)
			{
				temp = get_bits(8);

				if (prec)
					temp = (temp << 8) + get_bits(8);

				m_quant[n][i] = static_cast<jpgd_quant_t>(temp);
			}

			i = 64 + 1;

			if (prec)
				i += 64;

			if (num_left < (uint)i)
				stop_decoding(JPGD_BAD_DQT_LENGTH);

			num_left -= i;
		}
	}

	// Read the start of frame (SOF) marker.
	void jpeg_decoder::read_sof_marker()
	{
		int i;
		uint num_left;

		num_left = get_bits(16);

		/* precision: sorry, only 8-bit precision is supported */
		if (get_bits(8) != 8)
			stop_decoding(JPGD_BAD_PRECISION);

		m_image_y_size = get_bits(16);

		if ((m_image_y_size < 1) || (m_image_y_size > JPGD_MAX_HEIGHT))
			stop_decoding(JPGD_BAD_HEIGHT);

		m_image_x_size = get_bits(16);

		if ((m_image_x_size < 1) || (m_image_x_size > JPGD_MAX_WIDTH))
			stop_decoding(JPGD_BAD_WIDTH);

		m_comps_in_frame = get_bits(8);

		if (m_comps_in_frame > JPGD_MAX_COMPONENTS)
			stop_decoding(JPGD_TOO_MANY_COMPONENTS);

		if (num_left != (uint)(m_comps_in_frame * 3 + 8))
			stop_decoding(JPGD_BAD_SOF_LENGTH);

		for (i = 0; i < m_comps_in_frame; i++)
		{
			m_comp_ident[i] = get_bits(8);
			m_comp_h_samp[i] = get_bits(4);
			m_comp_v_samp[i] = get_bits(4);

			if (!m_comp_h_samp[i] || !m_comp_v_samp[i] || (m_comp_h_samp[i] > 2) || (m_comp_v_samp[i] > 2))
				stop_decoding(JPGD_UNSUPPORTED_SAMP_FACTORS);

			m_comp_quant[i] = get_bits(8);
			if (m_comp_quant[i] >= JPGD_MAX_QUANT_TABLES)
				stop_decoding(JPGD_DECODE_ERROR);
		}
	}

	// Used to skip unrecognized markers.
	void jpeg_decoder::skip_variable_marker()
	{
		uint num_left;

		num_left = get_bits(16);

		if (num_left < 2)
			stop_decoding(JPGD_BAD_VARIABLE_MARKER);

		num_left -= 2;

		while (num_left)
		{
			get_bits(8);
			num_left--;
		}
	}

	// Read a define restart interval (DRI) marker.
	void jpeg_decoder::read_dri_marker()
	{
		if (get_bits(16) != 4)
			stop_decoding(JPGD_BAD_DRI_LENGTH);

		m_restart_interval = get_bits(16);
	}

	// Read a start of scan (SOS) marker.
	void jpeg_decoder::read_sos_marker()
	{
		uint num_left;
		int i, ci, n, c, cc;

		num_left = get_bits(16);

		n = get_bits(8);

		m_comps_in_scan = n;

		num_left -= 3;

		if ((num_left != (uint)(n * 2 + 3)) || (n < 1) || (n > JPGD_MAX_COMPS_IN_SCAN))
			stop_decoding(JPGD_BAD_SOS_LENGTH);

		for (i = 0; i < n; i++)
		{
			cc = get_bits(8);
			c = get_bits(8);
			num_left -= 2;

			for (ci = 0; ci < m_comps_in_frame; ci++)
				if (cc == m_comp_ident[ci])
					break;

			if (ci >= m_comps_in_frame)
				stop_decoding(JPGD_BAD_SOS_COMP_ID);

			if (ci >= JPGD_MAX_COMPONENTS)
				stop_decoding(JPGD_DECODE_ERROR);

			m_comp_list[i] = ci;

			m_comp_dc_tab[ci] = (c >> 4) & 15;
			m_comp_ac_tab[ci] = (c & 15) + (JPGD_MAX_HUFF_TABLES >> 1);

			if (m_comp_dc_tab[ci] >= JPGD_MAX_HUFF_TABLES)
				stop_decoding(JPGD_DECODE_ERROR);

			if (m_comp_ac_tab[ci] >= JPGD_MAX_HUFF_TABLES)
				stop_decoding(JPGD_DECODE_ERROR);
		}

		m_spectral_start = get_bits(8);
		m_spectral_end = get_bits(8);
		m_successive_high = get_bits(4);
		m_successive_low = get_bits(4);

		if (!m_progressive_flag)
		{
			m_spectral_start = 0;
			m_spectral_end = 63;
		}

		num_left -= 3;

		/* read past whatever is num_left */
		while (num_left)
		{
			get_bits(8);
			num_left--;
		}
	}

	// Finds the next marker.
	int jpeg_decoder::next_marker()
	{
		uint c, bytes;

		bytes = 0;

		do
		{
			do
			{
				bytes++;
				c = get_bits(8);
			} while (c != 0xFF);

			do
			{
				c = get_bits(8);
			} while (c == 0xFF);

		} while (c == 0);

		// If bytes > 0 here, there where extra bytes before the marker (not good).

		return c;
	}

	// Process markers. Returns when an SOFx, SOI, EOI, or SOS marker is
	// encountered.
	int jpeg_decoder::process_markers()
	{
		int c;

		for (; ; )
		{
			c = next_marker();

			switch (c)
			{
			case M_SOF0:
			case M_SOF1:
			case M_SOF2:
			case M_SOF3:
			case M_SOF5:
			case M_SOF6:
			case M_SOF7:
				//      case M_JPG:
			case M_SOF9:
			case M_SOF10:
			case M_SOF11:
			case M_SOF13:
			case M_SOF14:
			case M_SOF15:
			case M_SOI:
			case M_EOI:
			case M_SOS:
			{
				return c;
			}
			case M_DHT:
			{
				read_dht_marker();
				break;
			}
			// No arithmitic support - dumb patents!
			case M_DAC:
			{
				stop_decoding(JPGD_NO_ARITHMITIC_SUPPORT);
				break;
			}
			case M_DQT:
			{
				read_dqt_marker();
				break;
			}
			case M_DRI:
			{
				read_dri_marker();
				break;
			}
			//case M_APP0:  /* no need to read the JFIF marker */
			case M_JPG:
			case M_RST0:    /* no parameters */
			case M_RST1:
			case M_RST2:
			case M_RST3:
			case M_RST4:
			case M_RST5:
			case M_RST6:
			case M_RST7:
			case M_TEM:
			{
				stop_decoding(JPGD_UNEXPECTED_MARKER);
				break;
			}
			default:    /* must be DNL, DHP, EXP, APPn, JPGn, COM, or RESn or APP0 */
			{
				skip_variable_marker();
				break;
			}
			}
		}
	}

	// Finds the start of image (SOI) marker.
	void jpeg_decoder::locate_soi_marker()
	{
		uint lastchar, thischar;
		uint bytesleft;

		lastchar = get_bits(8);

		thischar = get_bits(8);

		/* ok if it's a normal JPEG file without a special header */

		if ((lastchar == 0xFF) && (thischar == M_SOI))
			return;

		bytesleft = 4096;

		for (; ; )
		{
			if (--bytesleft == 0)
				stop_decoding(JPGD_NOT_JPEG);

			lastchar = thischar;

			thischar = get_bits(8);

			if (lastchar == 0xFF)
			{
				if (thischar == M_SOI)
					break;
				else if (thischar == M_EOI) // get_bits will keep returning M_EOI if we read past the end
					stop_decoding(JPGD_NOT_JPEG);
			}
		}

		// Check the next character after marker: if it's not 0xFF, it can't be the start of the next marker, so the file is bad.
		thischar = (m_bit_buf >> 24) & 0xFF;

		if (thischar != 0xFF)
			stop_decoding(JPGD_NOT_JPEG);
	}

	// Find a start of frame (SOF) marker.
	void jpeg_decoder::locate_sof_marker()
	{
		locate_soi_marker();

		int c = process_markers();

		switch (c)
		{
		case M_SOF2:
		{
			m_progressive_flag = JPGD_TRUE;
			read_sof_marker();
			break;
		}
		case M_SOF0:  /* baseline DCT */
		case M_SOF1:  /* extended sequential DCT */
		{
			read_sof_marker();
			break;
		}
		case M_SOF9:  /* Arithmitic coding */
		{
			stop_decoding(JPGD_NO_ARITHMITIC_SUPPORT);
			break;
		}
		default:
		{
			stop_decoding(JPGD_UNSUPPORTED_MARKER);
			break;
		}
		}
	}

	// Find a start of scan (SOS) marker.
	int jpeg_decoder::locate_sos_marker()
	{
		int c;

		c = process_markers();

		if (c == M_EOI)
			return JPGD_FALSE;
		else if (c != M_SOS)
			stop_decoding(JPGD_UNEXPECTED_MARKER);

		read_sos_marker();

		return JPGD_TRUE;
	}

	// Reset everything to default/uninitialized state.
	void jpeg_decoder::init(jpeg_decoder_stream* pStream, uint32_t flags)
	{
		m_flags = flags;
		m_pMem_blocks = nullptr;
		m_error_code = JPGD_SUCCESS;
		m_ready_flag = false;
		m_image_x_size = m_image_y_size = 0;
		m_pStream = pStream;
		m_progressive_flag = JPGD_FALSE;
				
		memset(m_huff_ac, 0, sizeof(m_huff_ac));
		memset(m_huff_num, 0, sizeof(m_huff_num));
		memset(m_huff_val, 0, sizeof(m_huff_val));
		memset(m_quant, 0, sizeof(m_quant));

		m_scan_type = 0;
		m_comps_in_frame = 0;

		memset(m_comp_h_samp, 0, sizeof(m_comp_h_samp));
		memset(m_comp_v_samp, 0, sizeof(m_comp_v_samp));
		memset(m_comp_quant, 0, sizeof(m_comp_quant));
		memset(m_comp_ident, 0, sizeof(m_comp_ident));
		memset(m_comp_h_blocks, 0, sizeof(m_comp_h_blocks));
		memset(m_comp_v_blocks, 0, sizeof(m_comp_v_blocks));

		m_comps_in_scan = 0;
		memset(m_comp_list, 0, sizeof(m_comp_list));
		memset(m_comp_dc_tab, 0, sizeof(m_comp_dc_tab));
		memset(m_comp_ac_tab, 0, sizeof(m_comp_ac_tab));

		m_spectral_start = 0;
		m_spectral_end = 0;
		m_successive_low = 0;
		m_successive_high = 0;
		m_max_mcu_x_size = 0;
		m_max_mcu_y_size = 0;
		m_blocks_per_mcu = 0;
		m_max_blocks_per_row = 0;
		m_mcus_per_row = 0;
		m_mcus_per_col = 0;

		memset(m_mcu_org, 0, sizeof(m_mcu_org));

		m_total_lines_left = 0;
		m_mcu_lines_left = 0;
		m_num_buffered_scanlines = 0;
		m_real_dest_bytes_per_scan_line = 0;
		m_dest_bytes_per_scan_line = 0;
		m_dest_bytes_per_pixel = 0;

		memset(m_pHuff_tabs, 0, sizeof(m_pHuff_tabs));

		memset(m_dc_coeffs, 0, sizeof(m_dc_coeffs));
		memset(m_ac_coeffs, 0, sizeof(m_ac_coeffs));
		memset(m_block_y_mcu, 0, sizeof(m_block_y_mcu));

		m_eob_run = 0;

		m_pIn_buf_ofs = m_in_buf;
		m_in_buf_left = 0;
		m_eof_flag = false;
		m_tem_flag = 0;

		memset(m_in_buf_pad_start, 0, sizeof(m_in_buf_pad_start));
		memset(m_in_buf, 0, sizeof(m_in_buf));
		memset(m_in_buf_pad_end, 0, sizeof(m_in_buf_pad_end));

		m_restart_interval = 0;
		m_restarts_left = 0;
		m_next_restart_num = 0;

		m_max_mcus_per_row = 0;
		m_max_blocks_per_mcu = 0;
		m_max_mcus_per_col = 0;

		memset(m_last_dc_val, 0, sizeof(m_last_dc_val));
		m_pMCU_coefficients = nullptr;
		m_pSample_buf = nullptr;
		m_pSample_buf_prev = nullptr;
		m_sample_buf_prev_valid = false;

		m_total_bytes_read = 0;

		m_pScan_line_0 = nullptr;
		m_pScan_line_1 = nullptr;

		// Ready the input buffer.
		prep_in_buffer();

		// Prime the bit buffer.
		m_bits_left = 16;
		m_bit_buf = 0;

		get_bits(16);
		get_bits(16);

		for (int i = 0; i < JPGD_MAX_BLOCKS_PER_MCU; i++)
			m_mcu_block_max_zag[i] = 64;

		m_has_sse2 = false;

#if JPGD_USE_SSE2
#ifdef _MSC_VER
		int cpu_info[4];
		__cpuid(cpu_info, 1);
		const int cpu_info3 = cpu_info[3];
		m_has_sse2 = ((cpu_info3 >> 26U) & 1U) != 0U;
#else
		m_has_sse2 = true;
#endif
#endif
	}

#define SCALEBITS 16
#define ONE_HALF  ((int) 1 << (SCALEBITS-1))
#define FIX(x)    ((int) ((x) * (1L<<SCALEBITS) + 0.5f))

	// Create a few tables that allow us to quickly convert YCbCr to RGB.
	void jpeg_decoder::create_look_ups()
	{
		for (int i = 0; i <= 255; i++)
		{
			int k = i - 128;
			m_crr[i] = (FIX(1.40200f) * k + ONE_HALF) >> SCALEBITS;
			m_cbb[i] = (FIX(1.77200f) * k + ONE_HALF) >> SCALEBITS;
			m_crg[i] = (-FIX(0.71414f)) * k;
			m_cbg[i] = (-FIX(0.34414f)) * k + ONE_HALF;
		}
	}

	// This method throws back into the stream any bytes that where read
	// into the bit buffer during initial marker scanning.
	void jpeg_decoder::fix_in_buffer()
	{
		// In case any 0xFF's where pulled into the buffer during marker scanning.
		assert((m_bits_left & 7) == 0);

		if (m_bits_left == 16)
			stuff_char((uint8)(m_bit_buf & 0xFF));

		if (m_bits_left >= 8)
			stuff_char((uint8)((m_bit_buf >> 8) & 0xFF));

		stuff_char((uint8)((m_bit_buf >> 16) & 0xFF));
		stuff_char((uint8)((m_bit_buf >> 24) & 0xFF));

		m_bits_left = 16;
		get_bits_no_markers(16);
		get_bits_no_markers(16);
	}

	void jpeg_decoder::transform_mcu(int mcu_row)
	{
		jpgd_block_coeff_t* pSrc_ptr = m_pMCU_coefficients;
		if (mcu_row * m_blocks_per_mcu >= m_max_blocks_per_row)
			stop_decoding(JPGD_DECODE_ERROR);

		uint8* pDst_ptr = m_pSample_buf + mcu_row * m_blocks_per_mcu * 64;

		for (int mcu_block = 0; mcu_block < m_blocks_per_mcu; mcu_block++)
		{
			idct(pSrc_ptr, pDst_ptr, m_mcu_block_max_zag[mcu_block], ((m_flags & cFlagDisableSIMD) == 0) && m_has_sse2);
			pSrc_ptr += 64;
			pDst_ptr += 64;
		}
	}

	// Loads and dequantizes the next row of (already decoded) coefficients.
	// Progressive images only.
	void jpeg_decoder::load_next_row()
	{
		int i;
		jpgd_block_coeff_t* p;
		jpgd_quant_t* q;
		int mcu_row, mcu_block, row_block = 0;
		int component_num, component_id;
		int block_x_mcu[JPGD_MAX_COMPONENTS];

		memset(block_x_mcu, 0, JPGD_MAX_COMPONENTS * sizeof(int));

		for (mcu_row = 0; mcu_row < m_mcus_per_row; mcu_row++)
		{
			int block_x_mcu_ofs = 0, block_y_mcu_ofs = 0;

			for (mcu_block = 0; mcu_block < m_blocks_per_mcu; mcu_block++)
			{
				component_id = m_mcu_org[mcu_block];
				if (m_comp_quant[component_id] >= JPGD_MAX_QUANT_TABLES)
					stop_decoding(JPGD_DECODE_ERROR);

				q = m_quant[m_comp_quant[component_id]];

				p = m_pMCU_coefficients + 64 * mcu_block;

				jpgd_block_coeff_t* pAC = coeff_buf_getp(m_ac_coeffs[component_id], block_x_mcu[component_id] + block_x_mcu_ofs, m_block_y_mcu[component_id] + block_y_mcu_ofs);
				jpgd_block_coeff_t* pDC = coeff_buf_getp(m_dc_coeffs[component_id], block_x_mcu[component_id] + block_x_mcu_ofs, m_block_y_mcu[component_id] + block_y_mcu_ofs);
				p[0] = pDC[0];
				memcpy(&p[1], &pAC[1], 63 * sizeof(jpgd_block_coeff_t));

				for (i = 63; i > 0; i--)
					if (p[g_ZAG[i]])
						break;

				m_mcu_block_max_zag[mcu_block] = i + 1;

				for (; i >= 0; i--)
					if (p[g_ZAG[i]])
						p[g_ZAG[i]] = static_cast<jpgd_block_coeff_t>(p[g_ZAG[i]] * q[i]);

				row_block++;

				if (m_comps_in_scan == 1)
					block_x_mcu[component_id]++;
				else
				{
					if (++block_x_mcu_ofs == m_comp_h_samp[component_id])
					{
						block_x_mcu_ofs = 0;

						if (++block_y_mcu_ofs == m_comp_v_samp[component_id])
						{
							block_y_mcu_ofs = 0;

							block_x_mcu[component_id] += m_comp_h_samp[component_id];
						}
					}
				}
			}

			transform_mcu(mcu_row);
		}

		if (m_comps_in_scan == 1)
			m_block_y_mcu[m_comp_list[0]]++;
		else
		{
			for (component_num = 0; component_num < m_comps_in_scan; component_num++)
			{
				component_id = m_comp_list[component_num];

				m_block_y_mcu[component_id] += m_comp_v_samp[component_id];
			}
		}
	}

	// Restart interval processing.
	void jpeg_decoder::process_restart()
	{
		int i;
		int c = 0;

		// Align to a byte boundry
		// FIXME: Is this really necessary? get_bits_no_markers() never reads in markers!
		//get_bits_no_markers(m_bits_left & 7);

		// Let's scan a little bit to find the marker, but not _too_ far.
		// 1536 is a "fudge factor" that determines how much to scan.
		for (i = 1536; i > 0; i--)
			if (get_char() == 0xFF)
				break;

		if (i == 0)
			stop_decoding(JPGD_BAD_RESTART_MARKER);

		for (; i > 0; i--)
			if ((c = get_char()) != 0xFF)
				break;

		if (i == 0)
			stop_decoding(JPGD_BAD_RESTART_MARKER);

		// Is it the expected marker? If not, something bad happened.
		if (c != (m_next_restart_num + M_RST0))
			stop_decoding(JPGD_BAD_RESTART_MARKER);

		// Reset each component's DC prediction values.
		memset(&m_last_dc_val, 0, m_comps_in_frame * sizeof(uint));

		m_eob_run = 0;

		m_restarts_left = m_restart_interval;

		m_next_restart_num = (m_next_restart_num + 1) & 7;

		// Get the bit buffer going again...

		m_bits_left = 16;
		get_bits_no_markers(16);
		get_bits_no_markers(16);
	}

	static inline int dequantize_ac(int c, int q) { c *= q; return c; }

	// Decodes and dequantizes the next row of coefficients.
	void jpeg_decoder::decode_next_row()
	{
		int row_block = 0;

		for (int mcu_row = 0; mcu_row < m_mcus_per_row; mcu_row++)
		{
			if ((m_restart_interval) && (m_restarts_left == 0))
				process_restart();

			jpgd_block_coeff_t* p = m_pMCU_coefficients;
			for (int mcu_block = 0; mcu_block < m_blocks_per_mcu; mcu_block++, p += 64)
			{
				int component_id = m_mcu_org[mcu_block];
				if (m_comp_quant[component_id] >= JPGD_MAX_QUANT_TABLES)
					stop_decoding(JPGD_DECODE_ERROR);

				jpgd_quant_t* q = m_quant[m_comp_quant[component_id]];

				int r, s;
				s = huff_decode(m_pHuff_tabs[m_comp_dc_tab[component_id]], r);
				if (s >= 16)
					stop_decoding(JPGD_DECODE_ERROR);

				s = JPGD_HUFF_EXTEND(r, s);

				m_last_dc_val[component_id] = (s += m_last_dc_val[component_id]);

				p[0] = static_cast<jpgd_block_coeff_t>(s * q[0]);

				int prev_num_set = m_mcu_block_max_zag[mcu_block];

				huff_tables* pH = m_pHuff_tabs[m_comp_ac_tab[component_id]];

				int k;
				for (k = 1; k < 64; k++)
				{
					int extra_bits;
					s = huff_decode(pH, extra_bits);

					r = s >> 4;
					s &= 15;

					if (s)
					{
						if (r)
						{
							if ((k + r) > 63)
								stop_decoding(JPGD_DECODE_ERROR);

							if (k < prev_num_set)
							{
								int n = JPGD_MIN(r, prev_num_set - k);
								int kt = k;
								while (n--)
									p[g_ZAG[kt++]] = 0;
							}

							k += r;
						}

						s = JPGD_HUFF_EXTEND(extra_bits, s);

						if (k >= 64)
							stop_decoding(JPGD_DECODE_ERROR);

						p[g_ZAG[k]] = static_cast<jpgd_block_coeff_t>(dequantize_ac(s, q[k])); //s * q[k];
					}
					else
					{
						if (r == 15)
						{
							if ((k + 16) > 64)
								stop_decoding(JPGD_DECODE_ERROR);

							if (k < prev_num_set)
							{
								int n = JPGD_MIN(16, prev_num_set - k);
								int kt = k;
								while (n--)
								{
									if (kt > 63)
										stop_decoding(JPGD_DECODE_ERROR);
									p[g_ZAG[kt++]] = 0;
								}
							}

							k += 16 - 1; // - 1 because the loop counter is k

							if (p[g_ZAG[k & 63]] != 0)
								stop_decoding(JPGD_DECODE_ERROR);
						}
						else
							break;
					}
				}

				if (k < prev_num_set)
				{
					int kt = k;
					while (kt < prev_num_set)
						p[g_ZAG[kt++]] = 0;
				}

				m_mcu_block_max_zag[mcu_block] = k;

				row_block++;
			}

			transform_mcu(mcu_row);

			m_restarts_left--;
		}
	}

	// YCbCr H1V1 (1x1:1:1, 3 m_blocks per MCU) to RGB
	void jpeg_decoder::H1V1Convert()
	{
		int row = m_max_mcu_y_size - m_mcu_lines_left;
		uint8* d = m_pScan_line_0;
		uint8* s = m_pSample_buf + row * 8;

		for (int i = m_max_mcus_per_row; i > 0; i--)
		{
			for (int j = 0; j < 8; j++)
			{
				int y = s[j];
				int cb = s[64 + j];
				int cr = s[128 + j];

				d[0] = clamp(y + m_crr[cr]);
				d[1] = clamp(y + ((m_crg[cr] + m_cbg[cb]) >> 16));
				d[2] = clamp(y + m_cbb[cb]);
				d[3] = 255;

				d += 4;
			}

			s += 64 * 3;
		}
	}

	// YCbCr H2V1 (2x1:1:1, 4 m_blocks per MCU) to RGB
	void jpeg_decoder::H2V1Convert()
	{
		int row = m_max_mcu_y_size - m_mcu_lines_left;
		uint8* d0 = m_pScan_line_0;
		uint8* y = m_pSample_buf + row * 8;
		uint8* c = m_pSample_buf + 2 * 64 + row * 8;

		for (int i = m_max_mcus_per_row; i > 0; i--)
		{
			for (int l = 0; l < 2; l++)
			{
				for (int j = 0; j < 4; j++)
				{
					int cb = c[0];
					int cr = c[64];

					int rc = m_crr[cr];
					int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
					int bc = m_cbb[cb];

					int yy = y[j << 1];
					d0[0] = clamp(yy + rc);
					d0[1] = clamp(yy + gc);
					d0[2] = clamp(yy + bc);
					d0[3] = 255;

					yy = y[(j << 1) + 1];
					d0[4] = clamp(yy + rc);
					d0[5] = clamp(yy + gc);
					d0[6] = clamp(yy + bc);
					d0[7] = 255;

					d0 += 8;

					c++;
				}
				y += 64;
			}

			y += 64 * 4 - 64 * 2;
			c += 64 * 4 - 8;
		}
	}

	// YCbCr H2V1 (2x1:1:1, 4 m_blocks per MCU) to RGB
	void jpeg_decoder::H2V1ConvertFiltered()
	{
		const uint BLOCKS_PER_MCU = 4;
		int row = m_max_mcu_y_size - m_mcu_lines_left;
		uint8* d0 = m_pScan_line_0;

		const int half_image_x_size = (m_image_x_size >> 1) - 1;
		const int row_x8 = row * 8;

		for (int x = 0; x < m_image_x_size; x++)
		{
			int y = m_pSample_buf[check_sample_buf_ofs((x >> 4) * BLOCKS_PER_MCU * 64 + ((x & 8) ? 64 : 0) + (x & 7) + row_x8)];

			int c_x0 = (x - 1) >> 1;
			int c_x1 = JPGD_MIN(c_x0 + 1, half_image_x_size);
			c_x0 = JPGD_MAX(c_x0, 0);

			int a = (c_x0 >> 3) * BLOCKS_PER_MCU * 64 + (c_x0 & 7) + row_x8 + 128;
			int cb0 = m_pSample_buf[check_sample_buf_ofs(a)];
			int cr0 = m_pSample_buf[check_sample_buf_ofs(a + 64)];

			int b = (c_x1 >> 3) * BLOCKS_PER_MCU * 64 + (c_x1 & 7) + row_x8 + 128;
			int cb1 = m_pSample_buf[check_sample_buf_ofs(b)];
			int cr1 = m_pSample_buf[check_sample_buf_ofs(b + 64)];

			int w0 = (x & 1) ? 3 : 1;
			int w1 = (x & 1) ? 1 : 3;

			int cb = (cb0 * w0 + cb1 * w1 + 2) >> 2;
			int cr = (cr0 * w0 + cr1 * w1 + 2) >> 2;

			int rc = m_crr[cr];
			int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
			int bc = m_cbb[cb];

			d0[0] = clamp(y + rc);
			d0[1] = clamp(y + gc);
			d0[2] = clamp(y + bc);
			d0[3] = 255;

			d0 += 4;
		}
	}

	// YCbCr H2V1 (1x2:1:1, 4 m_blocks per MCU) to RGB
	void jpeg_decoder::H1V2Convert()
	{
		int row = m_max_mcu_y_size - m_mcu_lines_left;
		uint8* d0 = m_pScan_line_0;
		uint8* d1 = m_pScan_line_1;
		uint8* y;
		uint8* c;

		if (row < 8)
			y = m_pSample_buf + row * 8;
		else
			y = m_pSample_buf + 64 * 1 + (row & 7) * 8;

		c = m_pSample_buf + 64 * 2 + (row >> 1) * 8;

		for (int i = m_max_mcus_per_row; i > 0; i--)
		{
			for (int j = 0; j < 8; j++)
			{
				int cb = c[0 + j];
				int cr = c[64 + j];

				int rc = m_crr[cr];
				int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
				int bc = m_cbb[cb];

				int yy = y[j];
				d0[0] = clamp(yy + rc);
				d0[1] = clamp(yy + gc);
				d0[2] = clamp(yy + bc);
				d0[3] = 255;

				yy = y[8 + j];
				d1[0] = clamp(yy + rc);
				d1[1] = clamp(yy + gc);
				d1[2] = clamp(yy + bc);
				d1[3] = 255;

				d0 += 4;
				d1 += 4;
			}

			y += 64 * 4;
			c += 64 * 4;
		}
	}

	// YCbCr H2V1 (1x2:1:1, 4 m_blocks per MCU) to RGB
	void jpeg_decoder::H1V2ConvertFiltered()
	{
		const uint BLOCKS_PER_MCU = 4;
		int y = m_image_y_size - m_total_lines_left;
		int row = y & 15;

		const int half_image_y_size = (m_image_y_size >> 1) - 1;

		uint8* d0 = m_pScan_line_0;

		const int w0 = (row & 1) ? 3 : 1;
		const int w1 = (row & 1) ? 1 : 3;

		int c_y0 = (y - 1) >> 1;
		int c_y1 = JPGD_MIN(c_y0 + 1, half_image_y_size);

		const uint8_t* p_YSamples = m_pSample_buf;
		const uint8_t* p_C0Samples = m_pSample_buf;
		if ((c_y0 >= 0) && (((row & 15) == 0) || ((row & 15) == 15)) && (m_total_lines_left > 1))
		{
			assert(y > 0);
			assert(m_sample_buf_prev_valid);

			if ((row & 15) == 15)
				p_YSamples = m_pSample_buf_prev;

			p_C0Samples = m_pSample_buf_prev;
		}

		const int y_sample_base_ofs = ((row & 8) ? 64 : 0) + (row & 7) * 8;
		const int y0_base = (c_y0 & 7) * 8 + 128;
		const int y1_base = (c_y1 & 7) * 8 + 128;

		for (int x = 0; x < m_image_x_size; x++)
		{
			const int base_ofs = (x >> 3) * BLOCKS_PER_MCU * 64 + (x & 7);

			int y_sample = p_YSamples[check_sample_buf_ofs(base_ofs + y_sample_base_ofs)];

			int a = base_ofs + y0_base;
			int cb0_sample = p_C0Samples[check_sample_buf_ofs(a)];
			int cr0_sample = p_C0Samples[check_sample_buf_ofs(a + 64)];

			int b = base_ofs + y1_base;
			int cb1_sample = m_pSample_buf[check_sample_buf_ofs(b)];
			int cr1_sample = m_pSample_buf[check_sample_buf_ofs(b + 64)];

			int cb = (cb0_sample * w0 + cb1_sample * w1 + 2) >> 2;
			int cr = (cr0_sample * w0 + cr1_sample * w1 + 2) >> 2;

			int rc = m_crr[cr];
			int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
			int bc = m_cbb[cb];

			d0[0] = clamp(y_sample + rc);
			d0[1] = clamp(y_sample + gc);
			d0[2] = clamp(y_sample + bc);
			d0[3] = 255;

			d0 += 4;
		}
	}

	// YCbCr H2V2 (2x2:1:1, 6 m_blocks per MCU) to RGB
	void jpeg_decoder::H2V2Convert()
	{
		int row = m_max_mcu_y_size - m_mcu_lines_left;
		uint8* d0 = m_pScan_line_0;
		uint8* d1 = m_pScan_line_1;
		uint8* y;
		uint8* c;

		if (row < 8)
			y = m_pSample_buf + row * 8;
		else
			y = m_pSample_buf + 64 * 2 + (row & 7) * 8;

		c = m_pSample_buf + 64 * 4 + (row >> 1) * 8;

		for (int i = m_max_mcus_per_row; i > 0; i--)
		{
			for (int l = 0; l < 2; l++)
			{
				for (int j = 0; j < 8; j += 2)
				{
					int cb = c[0];
					int cr = c[64];

					int rc = m_crr[cr];
					int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
					int bc = m_cbb[cb];

					int yy = y[j];
					d0[0] = clamp(yy + rc);
					d0[1] = clamp(yy + gc);
					d0[2] = clamp(yy + bc);
					d0[3] = 255;

					yy = y[j + 1];
					d0[4] = clamp(yy + rc);
					d0[5] = clamp(yy + gc);
					d0[6] = clamp(yy + bc);
					d0[7] = 255;

					yy = y[j + 8];
					d1[0] = clamp(yy + rc);
					d1[1] = clamp(yy + gc);
					d1[2] = clamp(yy + bc);
					d1[3] = 255;

					yy = y[j + 8 + 1];
					d1[4] = clamp(yy + rc);
					d1[5] = clamp(yy + gc);
					d1[6] = clamp(yy + bc);
					d1[7] = 255;

					d0 += 8;
					d1 += 8;

					c++;
				}
				y += 64;
			}

			y += 64 * 6 - 64 * 2;
			c += 64 * 6 - 8;
		}
	}

	uint32_t jpeg_decoder::H2V2ConvertFiltered()
	{
		const uint BLOCKS_PER_MCU = 6;
		int y = m_image_y_size - m_total_lines_left;
		int row = y & 15;

		const int half_image_y_size = (m_image_y_size >> 1) - 1;

		uint8* d0 = m_pScan_line_0;

		int c_y0 = (y - 1) >> 1;
		int c_y1 = JPGD_MIN(c_y0 + 1, half_image_y_size);

		const uint8_t* p_YSamples = m_pSample_buf;
		const uint8_t* p_C0Samples = m_pSample_buf;
		if ((c_y0 >= 0) && (((row & 15) == 0) || ((row & 15) == 15)) && (m_total_lines_left > 1))
		{
			assert(y > 0);
			assert(m_sample_buf_prev_valid);

			if ((row & 15) == 15)
				p_YSamples = m_pSample_buf_prev;

			p_C0Samples = m_pSample_buf_prev;
		}

		const int y_sample_base_ofs = ((row & 8) ? 128 : 0) + (row & 7) * 8;
		const int y0_base = (c_y0 & 7) * 8 + 256;
		const int y1_base = (c_y1 & 7) * 8 + 256;

		const int half_image_x_size = (m_image_x_size >> 1) - 1;

		static const uint8_t s_muls[2][2][4] =
		{
			{ { 1, 3, 3, 9 }, { 3, 9, 1, 3 }, },
			{ { 3, 1, 9, 3 }, { 9, 3, 3, 1 } }
		};

		if (((row & 15) >= 1) && ((row & 15) <= 14))
		{
			assert((row & 1) == 1);
			assert(((y + 1 - 1) >> 1) == c_y0);

			assert(p_YSamples == m_pSample_buf);
			assert(p_C0Samples == m_pSample_buf);

			uint8* d1 = m_pScan_line_1;
			const int y_sample_base_ofs1 = (((row + 1) & 8) ? 128 : 0) + ((row + 1) & 7) * 8;

			for (int x = 0; x < m_image_x_size; x++)
			{
				int k = (x >> 4) * BLOCKS_PER_MCU * 64 + ((x & 8) ? 64 : 0) + (x & 7);
				int y_sample0 = p_YSamples[check_sample_buf_ofs(k + y_sample_base_ofs)];
				int y_sample1 = p_YSamples[check_sample_buf_ofs(k + y_sample_base_ofs1)];

				int c_x0 = (x - 1) >> 1;
				int c_x1 = JPGD_MIN(c_x0 + 1, half_image_x_size);
				c_x0 = JPGD_MAX(c_x0, 0);

				int a = (c_x0 >> 3) * BLOCKS_PER_MCU * 64 + (c_x0 & 7);
				int cb00_sample = p_C0Samples[check_sample_buf_ofs(a + y0_base)];
				int cr00_sample = p_C0Samples[check_sample_buf_ofs(a + y0_base + 64)];

				int cb01_sample = m_pSample_buf[check_sample_buf_ofs(a + y1_base)];
				int cr01_sample = m_pSample_buf[check_sample_buf_ofs(a + y1_base + 64)];

				int b = (c_x1 >> 3) * BLOCKS_PER_MCU * 64 + (c_x1 & 7);
				int cb10_sample = p_C0Samples[check_sample_buf_ofs(b + y0_base)];
				int cr10_sample = p_C0Samples[check_sample_buf_ofs(b + y0_base + 64)];

				int cb11_sample = m_pSample_buf[check_sample_buf_ofs(b + y1_base)];
				int cr11_sample = m_pSample_buf[check_sample_buf_ofs(b + y1_base + 64)];

				{
					const uint8_t* pMuls = &s_muls[row & 1][x & 1][0];
					int cb = (cb00_sample * pMuls[0] + cb01_sample * pMuls[1] + cb10_sample * pMuls[2] + cb11_sample * pMuls[3] + 8) >> 4;
					int cr = (cr00_sample * pMuls[0] + cr01_sample * pMuls[1] + cr10_sample * pMuls[2] + cr11_sample * pMuls[3] + 8) >> 4;

					int rc = m_crr[cr];
					int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
					int bc = m_cbb[cb];

					d0[0] = clamp(y_sample0 + rc);
					d0[1] = clamp(y_sample0 + gc);
					d0[2] = clamp(y_sample0 + bc);
					d0[3] = 255;

					d0 += 4;
				}

				{
					const uint8_t* pMuls = &s_muls[(row + 1) & 1][x & 1][0];
					int cb = (cb00_sample * pMuls[0] + cb01_sample * pMuls[1] + cb10_sample * pMuls[2] + cb11_sample * pMuls[3] + 8) >> 4;
					int cr = (cr00_sample * pMuls[0] + cr01_sample * pMuls[1] + cr10_sample * pMuls[2] + cr11_sample * pMuls[3] + 8) >> 4;

					int rc = m_crr[cr];
					int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
					int bc = m_cbb[cb];

					d1[0] = clamp(y_sample1 + rc);
					d1[1] = clamp(y_sample1 + gc);
					d1[2] = clamp(y_sample1 + bc);
					d1[3] = 255;

					d1 += 4;
				}

				if (((x & 1) == 1) && (x < m_image_x_size - 1))
				{
					const int nx = x + 1;
					assert(c_x0 == (nx - 1) >> 1);

					k = (nx >> 4) * BLOCKS_PER_MCU * 64 + ((nx & 8) ? 64 : 0) + (nx & 7);
					y_sample0 = p_YSamples[check_sample_buf_ofs(k + y_sample_base_ofs)];
					y_sample1 = p_YSamples[check_sample_buf_ofs(k + y_sample_base_ofs1)];

					{
						const uint8_t* pMuls = &s_muls[row & 1][nx & 1][0];
						int cb = (cb00_sample * pMuls[0] + cb01_sample * pMuls[1] + cb10_sample * pMuls[2] + cb11_sample * pMuls[3] + 8) >> 4;
						int cr = (cr00_sample * pMuls[0] + cr01_sample * pMuls[1] + cr10_sample * pMuls[2] + cr11_sample * pMuls[3] + 8) >> 4;

						int rc = m_crr[cr];
						int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
						int bc = m_cbb[cb];

						d0[0] = clamp(y_sample0 + rc);
						d0[1] = clamp(y_sample0 + gc);
						d0[2] = clamp(y_sample0 + bc);
						d0[3] = 255;

						d0 += 4;
					}

					{
						const uint8_t* pMuls = &s_muls[(row + 1) & 1][nx & 1][0];
						int cb = (cb00_sample * pMuls[0] + cb01_sample * pMuls[1] + cb10_sample * pMuls[2] + cb11_sample * pMuls[3] + 8) >> 4;
						int cr = (cr00_sample * pMuls[0] + cr01_sample * pMuls[1] + cr10_sample * pMuls[2] + cr11_sample * pMuls[3] + 8) >> 4;

						int rc = m_crr[cr];
						int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
						int bc = m_cbb[cb];

						d1[0] = clamp(y_sample1 + rc);
						d1[1] = clamp(y_sample1 + gc);
						d1[2] = clamp(y_sample1 + bc);
						d1[3] = 255;

						d1 += 4;
					}

					++x;
				}
			}

			return 2;
		}
		else
		{
			for (int x = 0; x < m_image_x_size; x++)
			{
				int y_sample = p_YSamples[check_sample_buf_ofs((x >> 4) * BLOCKS_PER_MCU * 64 + ((x & 8) ? 64 : 0) + (x & 7) + y_sample_base_ofs)];

				int c_x0 = (x - 1) >> 1;
				int c_x1 = JPGD_MIN(c_x0 + 1, half_image_x_size);
				c_x0 = JPGD_MAX(c_x0, 0);

				int a = (c_x0 >> 3) * BLOCKS_PER_MCU * 64 + (c_x0 & 7);
				int cb00_sample = p_C0Samples[check_sample_buf_ofs(a + y0_base)];
				int cr00_sample = p_C0Samples[check_sample_buf_ofs(a + y0_base + 64)];

				int cb01_sample = m_pSample_buf[check_sample_buf_ofs(a + y1_base)];
				int cr01_sample = m_pSample_buf[check_sample_buf_ofs(a + y1_base + 64)];

				int b = (c_x1 >> 3) * BLOCKS_PER_MCU * 64 + (c_x1 & 7);
				int cb10_sample = p_C0Samples[check_sample_buf_ofs(b + y0_base)];
				int cr10_sample = p_C0Samples[check_sample_buf_ofs(b + y0_base + 64)];

				int cb11_sample = m_pSample_buf[check_sample_buf_ofs(b + y1_base)];
				int cr11_sample = m_pSample_buf[check_sample_buf_ofs(b + y1_base + 64)];

				const uint8_t* pMuls = &s_muls[row & 1][x & 1][0];
				int cb = (cb00_sample * pMuls[0] + cb01_sample * pMuls[1] + cb10_sample * pMuls[2] + cb11_sample * pMuls[3] + 8) >> 4;
				int cr = (cr00_sample * pMuls[0] + cr01_sample * pMuls[1] + cr10_sample * pMuls[2] + cr11_sample * pMuls[3] + 8) >> 4;

				int rc = m_crr[cr];
				int gc = ((m_crg[cr] + m_cbg[cb]) >> 16);
				int bc = m_cbb[cb];

				d0[0] = clamp(y_sample + rc);
				d0[1] = clamp(y_sample + gc);
				d0[2] = clamp(y_sample + bc);
				d0[3] = 255;

				d0 += 4;
			}

			return 1;
		}
	}

	// Y (1 block per MCU) to 8-bit grayscale
	void jpeg_decoder::gray_convert()
	{
		int row = m_max_mcu_y_size - m_mcu_lines_left;
		uint8* d = m_pScan_line_0;
		uint8* s = m_pSample_buf + row * 8;

		for (int i = m_max_mcus_per_row; i > 0; i--)
		{
			*(uint*)d = *(uint*)s;
			*(uint*)(&d[4]) = *(uint*)(&s[4]);

			s += 64;
			d += 8;
		}
	}

	// Find end of image (EOI) marker, so we can return to the user the exact size of the input stream.
	void jpeg_decoder::find_eoi()
	{
		if (!m_progressive_flag)
		{
			// Attempt to read the EOI marker.
			//get_bits_no_markers(m_bits_left & 7);

			// Prime the bit buffer
			m_bits_left = 16;
			get_bits(16);
			get_bits(16);

			// The next marker _should_ be EOI
			process_markers();
		}

		m_total_bytes_read -= m_in_buf_left;
	}

	int jpeg_decoder::decode_next_mcu_row()
	{
		if (::setjmp(m_jmp_state))
			return JPGD_FAILED;

		const bool chroma_y_filtering = ((m_flags & cFlagBoxChromaFiltering) == 0) && ((m_scan_type == JPGD_YH2V2) || (m_scan_type == JPGD_YH1V2));
		if (chroma_y_filtering)
		{
			std::swap(m_pSample_buf, m_pSample_buf_prev);

			m_sample_buf_prev_valid = true;
		}

		if (m_progressive_flag)
			load_next_row();
		else
			decode_next_row();

		// Find the EOI marker if that was the last row.
		if (m_total_lines_left <= m_max_mcu_y_size)
			find_eoi();

		m_mcu_lines_left = m_max_mcu_y_size;
		return 0;
	}

	int jpeg_decoder::decode(const void** pScan_line, uint* pScan_line_len)
	{
		if ((m_error_code) || (!m_ready_flag))
			return JPGD_FAILED;

		if (m_total_lines_left == 0)
			return JPGD_DONE;

		const bool chroma_y_filtering = ((m_flags & cFlagBoxChromaFiltering) == 0) && ((m_scan_type == JPGD_YH2V2) || (m_scan_type == JPGD_YH1V2));

		bool get_another_mcu_row = false;
		bool got_mcu_early = false;
		if (chroma_y_filtering)
		{
			if (m_total_lines_left == m_image_y_size)
				get_another_mcu_row = true;
			else if ((m_mcu_lines_left == 1) && (m_total_lines_left > 1))
			{
				get_another_mcu_row = true;
				got_mcu_early = true;
			}
		}
		else
		{
			get_another_mcu_row = (m_mcu_lines_left == 0);
		}

		if (get_another_mcu_row)
		{
			int status = decode_next_mcu_row();
			if (status != 0)
				return status;
		}

		switch (m_scan_type)
		{
		case JPGD_YH2V2:
		{
			if ((m_flags & cFlagBoxChromaFiltering) == 0)
			{
				if (m_num_buffered_scanlines == 1)
				{
					*pScan_line = m_pScan_line_1;
				}
				else if (m_num_buffered_scanlines == 0)
				{
					m_num_buffered_scanlines = H2V2ConvertFiltered();
					*pScan_line = m_pScan_line_0;
				}

				m_num_buffered_scanlines--;
			}
			else
			{
				if ((m_mcu_lines_left & 1) == 0)
				{
					H2V2Convert();
					*pScan_line = m_pScan_line_0;
				}
				else
					*pScan_line = m_pScan_line_1;
			}

			break;
		}
		case JPGD_YH2V1:
		{
			if ((m_flags & cFlagBoxChromaFiltering) == 0)
				H2V1ConvertFiltered();
			else
				H2V1Convert();
			*pScan_line = m_pScan_line_0;
			break;
		}
		case JPGD_YH1V2:
		{
			if (chroma_y_filtering)
			{
				H1V2ConvertFiltered();
				*pScan_line = m_pScan_line_0;
			}
			else
			{
				if ((m_mcu_lines_left & 1) == 0)
				{
					H1V2Convert();
					*pScan_line = m_pScan_line_0;
				}
				else
					*pScan_line = m_pScan_line_1;
			}

			break;
		}
		case JPGD_YH1V1:
		{
			H1V1Convert();
			*pScan_line = m_pScan_line_0;
			break;
		}
		case JPGD_GRAYSCALE:
		{
			gray_convert();
			*pScan_line = m_pScan_line_0;

			break;
		}
		}

		*pScan_line_len = m_real_dest_bytes_per_scan_line;

		if (!got_mcu_early)
		{
			m_mcu_lines_left--;
		}

		m_total_lines_left--;

		return JPGD_SUCCESS;
	}

	// Creates the tables needed for efficient Huffman decoding.
	void jpeg_decoder::make_huff_table(int index, huff_tables* pH)
	{
		int p, i, l, si;
		uint8 huffsize[258];
		uint huffcode[258];
		uint code;
		uint subtree;
		int code_size;
		int lastp;
		int nextfreeentry;
		int currententry;

		pH->ac_table = m_huff_ac[index] != 0;

		p = 0;

		for (l = 1; l <= 16; l++)
		{
			for (i = 1; i <= m_huff_num[index][l]; i++)
			{
				if (p >= 257)
					stop_decoding(JPGD_DECODE_ERROR);
				huffsize[p++] = static_cast<uint8>(l);
			}
		}

		assert(p < 258);
		huffsize[p] = 0;

		lastp = p;

		code = 0;
		si = huffsize[0];
		p = 0;

		while (huffsize[p])
		{
			while (huffsize[p] == si)
			{
				if (p >= 257)
					stop_decoding(JPGD_DECODE_ERROR);
				huffcode[p++] = code;
				code++;
			}

			code <<= 1;
			si++;
		}

		memset(pH->look_up, 0, sizeof(pH->look_up));
		memset(pH->look_up2, 0, sizeof(pH->look_up2));
		memset(pH->tree, 0, sizeof(pH->tree));
		memset(pH->code_size, 0, sizeof(pH->code_size));

		nextfreeentry = -1;

		p = 0;

		while (p < lastp)
		{
			i = m_huff_val[index][p];

			code = huffcode[p];
			code_size = huffsize[p];

			assert(i < JPGD_HUFF_CODE_SIZE_MAX_LENGTH);
			pH->code_size[i] = static_cast<uint8>(code_size);

			if (code_size <= 8)
			{
				code <<= (8 - code_size);

				for (l = 1 << (8 - code_size); l > 0; l--)
				{
					if (code >= 256)
						stop_decoding(JPGD_DECODE_ERROR);

					pH->look_up[code] = i;

					bool has_extrabits = false;
					int extra_bits = 0;
					int num_extra_bits = i & 15;

					int bits_to_fetch = code_size;
					if (num_extra_bits)
					{
						int total_codesize = code_size + num_extra_bits;
						if (total_codesize <= 8)
						{
							has_extrabits = true;
							extra_bits = ((1 << num_extra_bits) - 1) & (code >> (8 - total_codesize));

							if (extra_bits > 0x7FFF)
								stop_decoding(JPGD_DECODE_ERROR);

							bits_to_fetch += num_extra_bits;
						}
					}

					if (!has_extrabits)
						pH->look_up2[code] = i | (bits_to_fetch << 8);
					else
						pH->look_up2[code] = i | 0x8000 | (extra_bits << 16) | (bits_to_fetch << 8);

					code++;
				}
			}
			else
			{
				subtree = (code >> (code_size - 8)) & 0xFF;

				currententry = pH->look_up[subtree];

				if (currententry == 0)
				{
					pH->look_up[subtree] = currententry = nextfreeentry;
					pH->look_up2[subtree] = currententry = nextfreeentry;

					nextfreeentry -= 2;
				}

				code <<= (16 - (code_size - 8));

				for (l = code_size; l > 9; l--)
				{
					if ((code & 0x8000) == 0)
						currententry--;

					unsigned int idx = -currententry - 1;

					if (idx >= JPGD_HUFF_TREE_MAX_LENGTH)
						stop_decoding(JPGD_DECODE_ERROR);

					if (pH->tree[idx] == 0)
					{
						pH->tree[idx] = nextfreeentry;

						currententry = nextfreeentry;

						nextfreeentry -= 2;
					}
					else
					{
						currententry = pH->tree[idx];
					}

					code <<= 1;
				}

				if ((code & 0x8000) == 0)
					currententry--;

				if ((-currententry - 1) >= JPGD_HUFF_TREE_MAX_LENGTH)
					stop_decoding(JPGD_DECODE_ERROR);

				pH->tree[-currententry - 1] = i;
			}

			p++;
		}
	}

	// Verifies the quantization tables needed for this scan are available.
	void jpeg_decoder::check_quant_tables()
	{
		for (int i = 0; i < m_comps_in_scan; i++)
			if (m_quant[m_comp_quant[m_comp_list[i]]] == nullptr)
				stop_decoding(JPGD_UNDEFINED_QUANT_TABLE);
	}

	// Verifies that all the Huffman tables needed for this scan are available.
	void jpeg_decoder::check_huff_tables()
	{
		for (int i = 0; i < m_comps_in_scan; i++)
		{
			if ((m_spectral_start == 0) && (m_huff_num[m_comp_dc_tab[m_comp_list[i]]] == nullptr))
				stop_decoding(JPGD_UNDEFINED_HUFF_TABLE);

			if ((m_spectral_end > 0) && (m_huff_num[m_comp_ac_tab[m_comp_list[i]]] == nullptr))
				stop_decoding(JPGD_UNDEFINED_HUFF_TABLE);
		}

		for (int i = 0; i < JPGD_MAX_HUFF_TABLES; i++)
			if (m_huff_num[i])
			{
				if (!m_pHuff_tabs[i])
					m_pHuff_tabs[i] = (huff_tables*)alloc(sizeof(huff_tables));

				make_huff_table(i, m_pHuff_tabs[i]);
			}
	}

	// Determines the component order inside each MCU.
	// Also calcs how many MCU's are on each row, etc.
	bool jpeg_decoder::calc_mcu_block_order()
	{
		int component_num, component_id;
		int max_h_samp = 0, max_v_samp = 0;

		for (component_id = 0; component_id < m_comps_in_frame; component_id++)
		{
			if (m_comp_h_samp[component_id] > max_h_samp)
				max_h_samp = m_comp_h_samp[component_id];

			if (m_comp_v_samp[component_id] > max_v_samp)
				max_v_samp = m_comp_v_samp[component_id];
		}

		for (component_id = 0; component_id < m_comps_in_frame; component_id++)
		{
			m_comp_h_blocks[component_id] = ((((m_image_x_size * m_comp_h_samp[component_id]) + (max_h_samp - 1)) / max_h_samp) + 7) / 8;
			m_comp_v_blocks[component_id] = ((((m_image_y_size * m_comp_v_samp[component_id]) + (max_v_samp - 1)) / max_v_samp) + 7) / 8;
		}

		if (m_comps_in_scan == 1)
		{
			m_mcus_per_row = m_comp_h_blocks[m_comp_list[0]];
			m_mcus_per_col = m_comp_v_blocks[m_comp_list[0]];
		}
		else
		{
			m_mcus_per_row = (((m_image_x_size + 7) / 8) + (max_h_samp - 1)) / max_h_samp;
			m_mcus_per_col = (((m_image_y_size + 7) / 8) + (max_v_samp - 1)) / max_v_samp;
		}

		if (m_comps_in_scan == 1)
		{
			m_mcu_org[0] = m_comp_list[0];

			m_blocks_per_mcu = 1;
		}
		else
		{
			m_blocks_per_mcu = 0;

			for (component_num = 0; component_num < m_comps_in_scan; component_num++)
			{
				int num_blocks;

				component_id = m_comp_list[component_num];

				num_blocks = m_comp_h_samp[component_id] * m_comp_v_samp[component_id];

				while (num_blocks--)
					m_mcu_org[m_blocks_per_mcu++] = component_id;
			}
		}

		if (m_blocks_per_mcu > m_max_blocks_per_mcu)
			return false;

		for (int mcu_block = 0; mcu_block < m_blocks_per_mcu; mcu_block++)
		{
			int comp_id = m_mcu_org[mcu_block];
			if (comp_id >= JPGD_MAX_QUANT_TABLES)
				return false;
		}

		return true;
	}

	// Starts a new scan.
	int jpeg_decoder::init_scan()
	{
		if (!locate_sos_marker())
			return JPGD_FALSE;

		if (!calc_mcu_block_order())
			return JPGD_FALSE;

		check_huff_tables();

		check_quant_tables();

		memset(m_last_dc_val, 0, m_comps_in_frame * sizeof(uint));

		m_eob_run = 0;

		if (m_restart_interval)
		{
			m_restarts_left = m_restart_interval;
			m_next_restart_num = 0;
		}

		fix_in_buffer();

		return JPGD_TRUE;
	}

	// Starts a frame. Determines if the number of components or sampling factors
	// are supported.
	void jpeg_decoder::init_frame()
	{
		int i;

		if (m_comps_in_frame == 1)
		{
			if ((m_comp_h_samp[0] != 1) || (m_comp_v_samp[0] != 1))
				stop_decoding(JPGD_UNSUPPORTED_SAMP_FACTORS);

			m_scan_type = JPGD_GRAYSCALE;
			m_max_blocks_per_mcu = 1;
			m_max_mcu_x_size = 8;
			m_max_mcu_y_size = 8;
		}
		else if (m_comps_in_frame == 3)
		{
			if (((m_comp_h_samp[1] != 1) || (m_comp_v_samp[1] != 1)) ||
				((m_comp_h_samp[2] != 1) || (m_comp_v_samp[2] != 1)))
				stop_decoding(JPGD_UNSUPPORTED_SAMP_FACTORS);

			if ((m_comp_h_samp[0] == 1) && (m_comp_v_samp[0] == 1))
			{
				m_scan_type = JPGD_YH1V1;

				m_max_blocks_per_mcu = 3;
				m_max_mcu_x_size = 8;
				m_max_mcu_y_size = 8;
			}
			else if ((m_comp_h_samp[0] == 2) && (m_comp_v_samp[0] == 1))
			{
				m_scan_type = JPGD_YH2V1;
				m_max_blocks_per_mcu = 4;
				m_max_mcu_x_size = 16;
				m_max_mcu_y_size = 8;
			}
			else if ((m_comp_h_samp[0] == 1) && (m_comp_v_samp[0] == 2))
			{
				m_scan_type = JPGD_YH1V2;
				m_max_blocks_per_mcu = 4;
				m_max_mcu_x_size = 8;
				m_max_mcu_y_size = 16;
			}
			else if ((m_comp_h_samp[0] == 2) && (m_comp_v_samp[0] == 2))
			{
				m_scan_type = JPGD_YH2V2;
				m_max_blocks_per_mcu = 6;
				m_max_mcu_x_size = 16;
				m_max_mcu_y_size = 16;
			}
			else
				stop_decoding(JPGD_UNSUPPORTED_SAMP_FACTORS);
		}
		else
			stop_decoding(JPGD_UNSUPPORTED_COLORSPACE);

		m_max_mcus_per_row = (m_image_x_size + (m_max_mcu_x_size - 1)) / m_max_mcu_x_size;
		m_max_mcus_per_col = (m_image_y_size + (m_max_mcu_y_size - 1)) / m_max_mcu_y_size;

		// These values are for the *destination* pixels: after conversion.
		if (m_scan_type == JPGD_GRAYSCALE)
			m_dest_bytes_per_pixel = 1;
		else
			m_dest_bytes_per_pixel = 4;

		m_dest_bytes_per_scan_line = ((m_image_x_size + 15) & 0xFFF0) * m_dest_bytes_per_pixel;

		m_real_dest_bytes_per_scan_line = (m_image_x_size * m_dest_bytes_per_pixel);

		// Initialize two scan line buffers.
		m_pScan_line_0 = (uint8*)alloc_aligned(m_dest_bytes_per_scan_line, true);
		if ((m_scan_type == JPGD_YH1V2) || (m_scan_type == JPGD_YH2V2))
			m_pScan_line_1 = (uint8*)alloc_aligned(m_dest_bytes_per_scan_line, true);

		m_max_blocks_per_row = m_max_mcus_per_row * m_max_blocks_per_mcu;

		// Should never happen
		if (m_max_blocks_per_row > JPGD_MAX_BLOCKS_PER_ROW)
			stop_decoding(JPGD_DECODE_ERROR);

		// Allocate the coefficient buffer, enough for one MCU
		m_pMCU_coefficients = (jpgd_block_coeff_t *)alloc_aligned(m_max_blocks_per_mcu * 64 * sizeof(jpgd_block_coeff_t));
				
		for (i = 0; i < m_max_blocks_per_mcu; i++)
			m_mcu_block_max_zag[i] = 64;

		m_pSample_buf = (uint8*)alloc_aligned(m_max_blocks_per_row * 64);
		m_pSample_buf_prev = (uint8*)alloc_aligned(m_max_blocks_per_row * 64);

		m_total_lines_left = m_image_y_size;

		m_mcu_lines_left = 0;

		create_look_ups();
	}

	// The coeff_buf series of methods originally stored the coefficients
	// into a "virtual" file which was located in EMS, XMS, or a disk file. A cache
	// was used to make this process more efficient. Now, we can store the entire
	// thing in RAM.
	jpeg_decoder::coeff_buf* jpeg_decoder::coeff_buf_open(int block_num_x, int block_num_y, int block_len_x, int block_len_y)
	{
		coeff_buf* cb = (coeff_buf*)alloc(sizeof(coeff_buf));

		cb->block_num_x = block_num_x;
		cb->block_num_y = block_num_y;
		cb->block_len_x = block_len_x;
		cb->block_len_y = block_len_y;
		cb->block_size = (block_len_x * block_len_y) * sizeof(jpgd_block_coeff_t);
		cb->pData = (uint8*)alloc(cb->block_size * block_num_x * block_num_y, true);
		return cb;
	}

	inline jpgd_block_coeff_t* jpeg_decoder::coeff_buf_getp(coeff_buf* cb, int block_x, int block_y)
	{
		if ((block_x >= cb->block_num_x) || (block_y >= cb->block_num_y))
			stop_decoding(JPGD_DECODE_ERROR);

		return (jpgd_block_coeff_t*)(cb->pData + block_x * cb->block_size + block_y * (cb->block_size * cb->block_num_x));
	}

	// The following methods decode the various types of m_blocks encountered
	// in progressively encoded images.
	void jpeg_decoder::decode_block_dc_first(jpeg_decoder* pD, int component_id, int block_x, int block_y)
	{
		int s, r;
		jpgd_block_coeff_t* p = pD->coeff_buf_getp(pD->m_dc_coeffs[component_id], block_x, block_y);

		if ((s = pD->huff_decode(pD->m_pHuff_tabs[pD->m_comp_dc_tab[component_id]])) != 0)
		{
			if (s >= 16)
				pD->stop_decoding(JPGD_DECODE_ERROR);

			r = pD->get_bits_no_markers(s);
			s = JPGD_HUFF_EXTEND(r, s);
		}

		pD->m_last_dc_val[component_id] = (s += pD->m_last_dc_val[component_id]);

		p[0] = static_cast<jpgd_block_coeff_t>(s << pD->m_successive_low);
	}

	void jpeg_decoder::decode_block_dc_refine(jpeg_decoder* pD, int component_id, int block_x, int block_y)
	{
		if (pD->get_bits_no_markers(1))
		{
			jpgd_block_coeff_t* p = pD->coeff_buf_getp(pD->m_dc_coeffs[component_id], block_x, block_y);

			p[0] |= (1 << pD->m_successive_low);
		}
	}

	void jpeg_decoder::decode_block_ac_first(jpeg_decoder* pD, int component_id, int block_x, int block_y)
	{
		int k, s, r;

		if (pD->m_eob_run)
		{
			pD->m_eob_run--;
			return;
		}

		jpgd_block_coeff_t* p = pD->coeff_buf_getp(pD->m_ac_coeffs[component_id], block_x, block_y);

		for (k = pD->m_spectral_start; k <= pD->m_spectral_end; k++)
		{
			unsigned int idx = pD->m_comp_ac_tab[component_id];
			if (idx >= JPGD_MAX_HUFF_TABLES)
				pD->stop_decoding(JPGD_DECODE_ERROR);

			s = pD->huff_decode(pD->m_pHuff_tabs[idx]);

			r = s >> 4;
			s &= 15;

			if (s)
			{
				if ((k += r) > 63)
					pD->stop_decoding(JPGD_DECODE_ERROR);

				r = pD->get_bits_no_markers(s);
				s = JPGD_HUFF_EXTEND(r, s);

				p[g_ZAG[k]] = static_cast<jpgd_block_coeff_t>(s << pD->m_successive_low);
			}
			else
			{
				if (r == 15)
				{
					if ((k += 15) > 63)
						pD->stop_decoding(JPGD_DECODE_ERROR);
				}
				else
				{
					pD->m_eob_run = 1 << r;

					if (r)
						pD->m_eob_run += pD->get_bits_no_markers(r);

					pD->m_eob_run--;

					break;
				}
			}
		}
	}

	void jpeg_decoder::decode_block_ac_refine(jpeg_decoder* pD, int component_id, int block_x, int block_y)
	{
		int s, k, r;

		int p1 = 1 << pD->m_successive_low;

		//int m1 = (-1) << pD->m_successive_low;
		int m1 = static_cast<int>((UINT32_MAX << pD->m_successive_low));

		jpgd_block_coeff_t* p = pD->coeff_buf_getp(pD->m_ac_coeffs[component_id], block_x, block_y);
		if (pD->m_spectral_end > 63)
			pD->stop_decoding(JPGD_DECODE_ERROR);

		k = pD->m_spectral_start;

		if (pD->m_eob_run == 0)
		{
			for (; k <= pD->m_spectral_end; k++)
			{
				unsigned int idx = pD->m_comp_ac_tab[component_id];
				if (idx >= JPGD_MAX_HUFF_TABLES)
					pD->stop_decoding(JPGD_DECODE_ERROR);

				s = pD->huff_decode(pD->m_pHuff_tabs[idx]);

				r = s >> 4;
				s &= 15;

				if (s)
				{
					if (s != 1)
						pD->stop_decoding(JPGD_DECODE_ERROR);

					if (pD->get_bits_no_markers(1))
						s = p1;
					else
						s = m1;
				}
				else
				{
					if (r != 15)
					{
						pD->m_eob_run = 1 << r;

						if (r)
							pD->m_eob_run += pD->get_bits_no_markers(r);

						break;
					}
				}

				do
				{
					jpgd_block_coeff_t* this_coef = p + g_ZAG[k & 63];

					if (*this_coef != 0)
					{
						if (pD->get_bits_no_markers(1))
						{
							if ((*this_coef & p1) == 0)
							{
								if (*this_coef >= 0)
									*this_coef = static_cast<jpgd_block_coeff_t>(*this_coef + p1);
								else
									*this_coef = static_cast<jpgd_block_coeff_t>(*this_coef + m1);
							}
						}
					}
					else
					{
						if (--r < 0)
							break;
					}

					k++;

				} while (k <= pD->m_spectral_end);

				if ((s) && (k < 64))
				{
					p[g_ZAG[k]] = static_cast<jpgd_block_coeff_t>(s);
				}
			}
		}

		if (pD->m_eob_run > 0)
		{
			for (; k <= pD->m_spectral_end; k++)
			{
				jpgd_block_coeff_t* this_coef = p + g_ZAG[k & 63]; // logical AND to shut up static code analysis

				if (*this_coef != 0)
				{
					if (pD->get_bits_no_markers(1))
					{
						if ((*this_coef & p1) == 0)
						{
							if (*this_coef >= 0)
								*this_coef = static_cast<jpgd_block_coeff_t>(*this_coef + p1);
							else
								*this_coef = static_cast<jpgd_block_coeff_t>(*this_coef + m1);
						}
					}
				}
			}

			pD->m_eob_run--;
		}
	}

	// Decode a scan in a progressively encoded image.
	void jpeg_decoder::decode_scan(pDecode_block_func decode_block_func)
	{
		int mcu_row, mcu_col, mcu_block;
		int block_x_mcu[JPGD_MAX_COMPONENTS], block_y_mcu[JPGD_MAX_COMPONENTS];

		memset(block_y_mcu, 0, sizeof(block_y_mcu));

		for (mcu_col = 0; mcu_col < m_mcus_per_col; mcu_col++)
		{
			int component_num, component_id;

			memset(block_x_mcu, 0, sizeof(block_x_mcu));

			for (mcu_row = 0; mcu_row < m_mcus_per_row; mcu_row++)
			{
				int block_x_mcu_ofs = 0, block_y_mcu_ofs = 0;

				if ((m_restart_interval) && (m_restarts_left == 0))
					process_restart();

				for (mcu_block = 0; mcu_block < m_blocks_per_mcu; mcu_block++)
				{
					component_id = m_mcu_org[mcu_block];

					decode_block_func(this, component_id, block_x_mcu[component_id] + block_x_mcu_ofs, block_y_mcu[component_id] + block_y_mcu_ofs);

					if (m_comps_in_scan == 1)
						block_x_mcu[component_id]++;
					else
					{
						if (++block_x_mcu_ofs == m_comp_h_samp[component_id])
						{
							block_x_mcu_ofs = 0;

							if (++block_y_mcu_ofs == m_comp_v_samp[component_id])
							{
								block_y_mcu_ofs = 0;
								block_x_mcu[component_id] += m_comp_h_samp[component_id];
							}
						}
					}
				}

				m_restarts_left--;
			}

			if (m_comps_in_scan == 1)
				block_y_mcu[m_comp_list[0]]++;
			else
			{
				for (component_num = 0; component_num < m_comps_in_scan; component_num++)
				{
					component_id = m_comp_list[component_num];
					block_y_mcu[component_id] += m_comp_v_samp[component_id];
				}
			}
		}
	}

	// Decode a progressively encoded image.
	void jpeg_decoder::init_progressive()
	{
		int i;

		if (m_comps_in_frame == 4)
			stop_decoding(JPGD_UNSUPPORTED_COLORSPACE);

		// Allocate the coefficient buffers.
		for (i = 0; i < m_comps_in_frame; i++)
		{
			m_dc_coeffs[i] = coeff_buf_open(m_max_mcus_per_row * m_comp_h_samp[i], m_max_mcus_per_col * m_comp_v_samp[i], 1, 1);
			m_ac_coeffs[i] = coeff_buf_open(m_max_mcus_per_row * m_comp_h_samp[i], m_max_mcus_per_col * m_comp_v_samp[i], 8, 8);
		}

		// See https://libjpeg-turbo.org/pmwiki/uploads/About/TwoIssueswiththeJPEGStandard.pdf
		uint32_t total_scans = 0;
		const uint32_t MAX_SCANS_TO_PROCESS = 1000;

		for (; ; )
		{
			int dc_only_scan, refinement_scan;
			pDecode_block_func decode_block_func;

			if (!init_scan())
				break;

			dc_only_scan = (m_spectral_start == 0);
			refinement_scan = (m_successive_high != 0);

			if ((m_spectral_start > m_spectral_end) || (m_spectral_end > 63))
				stop_decoding(JPGD_BAD_SOS_SPECTRAL);

			if (dc_only_scan)
			{
				if (m_spectral_end)
					stop_decoding(JPGD_BAD_SOS_SPECTRAL);
			}
			else if (m_comps_in_scan != 1)  /* AC scans can only contain one component */
				stop_decoding(JPGD_BAD_SOS_SPECTRAL);

			if ((refinement_scan) && (m_successive_low != m_successive_high - 1))
				stop_decoding(JPGD_BAD_SOS_SUCCESSIVE);

			if (dc_only_scan)
			{
				if (refinement_scan)
					decode_block_func = decode_block_dc_refine;
				else
					decode_block_func = decode_block_dc_first;
			}
			else
			{
				if (refinement_scan)
					decode_block_func = decode_block_ac_refine;
				else
					decode_block_func = decode_block_ac_first;
			}

			decode_scan(decode_block_func);

			m_bits_left = 16;
			get_bits(16);
			get_bits(16);

			total_scans++;
			if (total_scans > MAX_SCANS_TO_PROCESS)
				stop_decoding(JPGD_TOO_MANY_SCANS);
		}

		m_comps_in_scan = m_comps_in_frame;

		for (i = 0; i < m_comps_in_frame; i++)
			m_comp_list[i] = i;

		if (!calc_mcu_block_order())
			stop_decoding(JPGD_DECODE_ERROR);
	}

	void jpeg_decoder::init_sequential()
	{
		if (!init_scan())
			stop_decoding(JPGD_UNEXPECTED_MARKER);
	}

	void jpeg_decoder::decode_start()
	{
		init_frame();

		if (m_progressive_flag)
			init_progressive();
		else
			init_sequential();
	}

	void jpeg_decoder::decode_init(jpeg_decoder_stream* pStream, uint32_t flags)
	{
		init(pStream, flags);
		locate_sof_marker();
	}

	jpeg_decoder::jpeg_decoder(jpeg_decoder_stream* pStream, uint32_t flags)
	{
		if (::setjmp(m_jmp_state))
			return;
		decode_init(pStream, flags);
	}

	int jpeg_decoder::begin_decoding()
	{
		if (m_ready_flag)
			return JPGD_SUCCESS;

		if (m_error_code)
			return JPGD_FAILED;

		if (::setjmp(m_jmp_state))
			return JPGD_FAILED;

		decode_start();

		m_ready_flag = true;

		return JPGD_SUCCESS;
	}

	jpeg_decoder::~jpeg_decoder()
	{
		free_all_blocks();
	}

	jpeg_decoder_file_stream::jpeg_decoder_file_stream()
	{
		m_pFile = nullptr;
		m_eof_flag = false;
		m_error_flag = false;
	}

	void jpeg_decoder_file_stream::close()
	{
		if (m_pFile)
		{
			fclose(m_pFile);
			m_pFile = nullptr;
		}

		m_eof_flag = false;
		m_error_flag = false;
	}

	jpeg_decoder_file_stream::~jpeg_decoder_file_stream()
	{
		close();
	}

	bool jpeg_decoder_file_stream::open(const char* Pfilename)
	{
		close();

		m_eof_flag = false;
		m_error_flag = false;

#if defined(_MSC_VER)
		m_pFile = nullptr;
		fopen_s(&m_pFile, Pfilename, "rb");
#else
		m_pFile = fopen(Pfilename, "rb");
#endif
		return m_pFile != nullptr;
	}

	int jpeg_decoder_file_stream::read(uint8* pBuf, int max_bytes_to_read, bool* pEOF_flag)
	{
		if (!m_pFile)
			return -1;

		if (m_eof_flag)
		{
			*pEOF_flag = true;
			return 0;
		}

		if (m_error_flag)
			return -1;

		int bytes_read = static_cast<int>(fread(pBuf, 1, max_bytes_to_read, m_pFile));
		if (bytes_read < max_bytes_to_read)
		{
			if (ferror(m_pFile))
			{
				m_error_flag = true;
				return -1;
			}

			m_eof_flag = true;
			*pEOF_flag = true;
		}

		return bytes_read;
	}

	bool jpeg_decoder_mem_stream::open(const uint8* pSrc_data, uint size)
	{
		close();
		m_pSrc_data = pSrc_data;
		m_ofs = 0;
		m_size = size;
		return true;
	}

	int jpeg_decoder_mem_stream::read(uint8* pBuf, int max_bytes_to_read, bool* pEOF_flag)
	{
		*pEOF_flag = false;

		if (!m_pSrc_data)
			return -1;

		uint bytes_remaining = m_size - m_ofs;
		if ((uint)max_bytes_to_read > bytes_remaining)
		{
			max_bytes_to_read = bytes_remaining;
			*pEOF_flag = true;
		}

		memcpy(pBuf, m_pSrc_data + m_ofs, max_bytes_to_read);
		m_ofs += max_bytes_to_read;

		return max_bytes_to_read;
	}

	unsigned char* decompress_jpeg_image_from_stream(jpeg_decoder_stream* pStream, int* width, int* height, int* actual_comps, int req_comps, uint32_t flags)
	{
		if (!actual_comps)
			return nullptr;
		*actual_comps = 0;

		if ((!pStream) || (!width) || (!height) || (!req_comps))
			return nullptr;

		if ((req_comps != 1) && (req_comps != 3) && (req_comps != 4))
			return nullptr;

		jpeg_decoder decoder(pStream, flags);
		if (decoder.get_error_code() != JPGD_SUCCESS)
			return nullptr;

		const int image_width = decoder.get_width(), image_height = decoder.get_height();
		*width = image_width;
		*height = image_height;
		*actual_comps = decoder.get_num_components();

		if (decoder.begin_decoding() != JPGD_SUCCESS)
			return nullptr;

		const int dst_bpl = image_width * req_comps;

		uint8* pImage_data = (uint8*)jpgd_malloc(dst_bpl * image_height);
		if (!pImage_data)
			return nullptr;

		for (int y = 0; y < image_height; y++)
		{
			const uint8* pScan_line;
			uint scan_line_len;
			if (decoder.decode((const void**)&pScan_line, &scan_line_len) != JPGD_SUCCESS)
			{
				jpgd_free(pImage_data);
				return nullptr;
			}

			uint8* pDst = pImage_data + y * dst_bpl;

			if (((req_comps == 1) && (decoder.get_num_components() == 1)) || ((req_comps == 4) && (decoder.get_num_components() == 3)))
				memcpy(pDst, pScan_line, dst_bpl);
			else if (decoder.get_num_components() == 1)
			{
				if (req_comps == 3)
				{
					for (int x = 0; x < image_width; x++)
					{
						uint8 luma = pScan_line[x];
						pDst[0] = luma;
						pDst[1] = luma;
						pDst[2] = luma;
						pDst += 3;
					}
				}
				else
				{
					for (int x = 0; x < image_width; x++)
					{
						uint8 luma = pScan_line[x];
						pDst[0] = luma;
						pDst[1] = luma;
						pDst[2] = luma;
						pDst[3] = 255;
						pDst += 4;
					}
				}
			}
			else if (decoder.get_num_components() == 3)
			{
				if (req_comps == 1)
				{
					const int YR = 19595, YG = 38470, YB = 7471;
					for (int x = 0; x < image_width; x++)
					{
						int r = pScan_line[x * 4 + 0];
						int g = pScan_line[x * 4 + 1];
						int b = pScan_line[x * 4 + 2];
						*pDst++ = static_cast<uint8>((r * YR + g * YG + b * YB + 32768) >> 16);
					}
				}
				else
				{
					for (int x = 0; x < image_width; x++)
					{
						pDst[0] = pScan_line[x * 4 + 0];
						pDst[1] = pScan_line[x * 4 + 1];
						pDst[2] = pScan_line[x * 4 + 2];
						pDst += 3;
					}
				}
			}
		}

		return pImage_data;
	}

	unsigned char* decompress_jpeg_image_from_memory(const unsigned char* pSrc_data, int src_data_size, int* width, int* height, int* actual_comps, int req_comps, uint32_t flags)
	{
		jpgd::jpeg_decoder_mem_stream mem_stream(pSrc_data, src_data_size);
		return decompress_jpeg_image_from_stream(&mem_stream, width, height, actual_comps, req_comps, flags);
	}

	unsigned char* decompress_jpeg_image_from_file(const char* pSrc_filename, int* width, int* height, int* actual_comps, int req_comps, uint32_t flags)
	{
		jpgd::jpeg_decoder_file_stream file_stream;
		if (!file_stream.open(pSrc_filename))
			return nullptr;
		return decompress_jpeg_image_from_stream(&file_stream, width, height, actual_comps, req_comps, flags);
	}

} // namespace jpgd
