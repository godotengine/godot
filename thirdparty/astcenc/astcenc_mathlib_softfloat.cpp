// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2021 Arm Limited
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
 * @brief Soft-float library for IEEE-754.
 */
#if (ASTCENC_F16C == 0) && (ASTCENC_NEON == 0)

#include "astcenc_mathlib.h"

/*	sized soft-float types. These are mapped to the sized integer
    types of C99, instead of C's floating-point types; this is because
    the library needs to maintain exact, bit-level control on all
    operations on these data types. */
typedef uint16_t sf16;
typedef uint32_t sf32;

/******************************************
  helper functions and their lookup tables
 ******************************************/
/* count leading zeros functions. Only used when the input is nonzero. */

#if defined(__GNUC__) && (defined(__i386) || defined(__amd64))
#elif defined(__arm__) && defined(__ARMCC_VERSION)
#elif defined(__arm__) && defined(__GNUC__)
#else
	/* table used for the slow default versions. */
	static const uint8_t clz_table[256] =
	{
		8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
		3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
		2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
		2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};
#endif

/*
   32-bit count-leading-zeros function: use the Assembly instruction whenever possible. */
static uint32_t clz32(uint32_t inp)
{
	#if defined(__GNUC__) && (defined(__i386) || defined(__amd64))
		uint32_t bsr;
		__asm__("bsrl %1, %0": "=r"(bsr):"r"(inp | 1));
		return 31 - bsr;
	#else
		#if defined(__arm__) && defined(__ARMCC_VERSION)
			return __clz(inp);			/* armcc builtin */
		#else
			#if defined(__arm__) && defined(__GNUC__)
				uint32_t lz;
				__asm__("clz %0, %1": "=r"(lz):"r"(inp));
				return lz;
			#else
				/* slow default version */
				uint32_t summa = 24;
				if (inp >= UINT32_C(0x10000))
				{
					inp >>= 16;
					summa -= 16;
				}
				if (inp >= UINT32_C(0x100))
				{
					inp >>= 8;
					summa -= 8;
				}
				return summa + clz_table[inp];
			#endif
		#endif
	#endif
}

/* the five rounding modes that IEEE-754r defines */
typedef enum
{
	SF_UP = 0,				/* round towards positive infinity */
	SF_DOWN = 1,			/* round towards negative infinity */
	SF_TOZERO = 2,			/* round towards zero */
	SF_NEARESTEVEN = 3,		/* round toward nearest value; if mid-between, round to even value */
	SF_NEARESTAWAY = 4		/* round toward nearest value; if mid-between, round away from zero */
} roundmode;


static uint32_t rtne_shift32(uint32_t inp, uint32_t shamt)
{
	uint32_t vl1 = UINT32_C(1) << shamt;
	uint32_t inp2 = inp + (vl1 >> 1);	/* added 0.5 ULP */
	uint32_t msk = (inp | UINT32_C(1)) & vl1;	/* nonzero if odd. '| 1' forces it to 1 if the shamt is 0. */
	msk--;						/* negative if even, nonnegative if odd. */
	inp2 -= (msk >> 31);		/* subtract epsilon before shift if even. */
	inp2 >>= shamt;
	return inp2;
}

static uint32_t rtna_shift32(uint32_t inp, uint32_t shamt)
{
	uint32_t vl1 = (UINT32_C(1) << shamt) >> 1;
	inp += vl1;
	inp >>= shamt;
	return inp;
}

static uint32_t rtup_shift32(uint32_t inp, uint32_t shamt)
{
	uint32_t vl1 = UINT32_C(1) << shamt;
	inp += vl1;
	inp--;
	inp >>= shamt;
	return inp;
}

/* convert from FP16 to FP32. */
static sf32 sf16_to_sf32(sf16 inp)
{
	uint32_t inpx = inp;

	/*
		This table contains, for every FP16 sign/exponent value combination,
		the difference between the input FP16 value and the value obtained
		by shifting the correct FP32 result right by 13 bits.
		This table allows us to handle every case except denormals and NaN
		with just 1 table lookup, 2 shifts and 1 add.
	*/

	#define WITH_MSB(a) (UINT32_C(a) | (1u << 31))
	static const uint32_t tbl[64] =
	{
		WITH_MSB(0x00000), 0x1C000, 0x1C000, 0x1C000, 0x1C000, 0x1C000, 0x1C000,          0x1C000,
		         0x1C000,  0x1C000, 0x1C000, 0x1C000, 0x1C000, 0x1C000, 0x1C000,          0x1C000,
		         0x1C000,  0x1C000, 0x1C000, 0x1C000, 0x1C000, 0x1C000, 0x1C000,          0x1C000,
		         0x1C000,  0x1C000, 0x1C000, 0x1C000, 0x1C000, 0x1C000, 0x1C000, WITH_MSB(0x38000),
		WITH_MSB(0x38000), 0x54000, 0x54000, 0x54000, 0x54000, 0x54000, 0x54000,          0x54000,
		         0x54000,  0x54000, 0x54000, 0x54000, 0x54000, 0x54000, 0x54000,          0x54000,
		         0x54000,  0x54000, 0x54000, 0x54000, 0x54000, 0x54000, 0x54000,          0x54000,
		         0x54000,  0x54000, 0x54000, 0x54000, 0x54000, 0x54000, 0x54000, WITH_MSB(0x70000)
	};

	uint32_t res = tbl[inpx >> 10];
	res += inpx;

	/* Normal cases: MSB of 'res' not set. */
	if ((res & WITH_MSB(0)) == 0)
	{
		return res << 13;
	}

	/* Infinity and Zero: 10 LSB of 'res' not set. */
	if ((res & 0x3FF) == 0)
	{
		return res << 13;
	}

	/* NaN: the exponent field of 'inp' is non-zero. */
	if ((inpx & 0x7C00) != 0)
	{
		/* All NaNs are quietened. */
		return (res << 13) | 0x400000;
	}

	/* Denormal cases */
	uint32_t sign = (inpx & 0x8000) << 16;
	uint32_t mskval = inpx & 0x7FFF;
	uint32_t leadingzeroes = clz32(mskval);
	mskval <<= leadingzeroes;
	return (mskval >> 8) + ((0x85 - leadingzeroes) << 23) + sign;
}

/* Conversion routine that converts from FP32 to FP16. It supports denormals and all rounding modes. If a NaN is given as input, it is quietened. */
static sf16 sf32_to_sf16(sf32 inp, roundmode rmode)
{
	/* for each possible sign/exponent combination, store a case index. This gives a 512-byte table */
	static const uint8_t tab[512] {
		0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
		10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
		20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
		30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40,
		40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
		40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
		40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
		40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
		40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
		40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
		40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50,

		5, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
		15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
		15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
		15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
		15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
		15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
		15, 15, 15, 15, 15, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
		25, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
		35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 45,
		45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
		45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
		45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
		45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
		45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
		45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
		45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 55,
	};

	/* many of the cases below use a case-dependent magic constant. So we look up a magic constant before actually performing the switch. This table allows us to group cases, thereby minimizing code
	   size. */
	static const uint32_t tabx[60] {
		UINT32_C(0), UINT32_C(0), UINT32_C(0), UINT32_C(0), UINT32_C(0), UINT32_C(0x8000), UINT32_C(0x80000000), UINT32_C(0x8000), UINT32_C(0x8000), UINT32_C(0x8000),
		UINT32_C(1), UINT32_C(0), UINT32_C(0), UINT32_C(0), UINT32_C(0), UINT32_C(0x8000), UINT32_C(0x8001), UINT32_C(0x8000), UINT32_C(0x8000), UINT32_C(0x8000),
		UINT32_C(0), UINT32_C(0), UINT32_C(0), UINT32_C(0), UINT32_C(0), UINT32_C(0x8000), UINT32_C(0x8000), UINT32_C(0x8000), UINT32_C(0x8000), UINT32_C(0x8000),
		UINT32_C(0xC8001FFF), UINT32_C(0xC8000000), UINT32_C(0xC8000000), UINT32_C(0xC8000FFF), UINT32_C(0xC8001000),
		UINT32_C(0x58000000), UINT32_C(0x38001FFF), UINT32_C(0x58000000), UINT32_C(0x58000FFF), UINT32_C(0x58001000),
		UINT32_C(0x7C00), UINT32_C(0x7BFF), UINT32_C(0x7BFF), UINT32_C(0x7C00), UINT32_C(0x7C00),
		UINT32_C(0xFBFF), UINT32_C(0xFC00), UINT32_C(0xFBFF), UINT32_C(0xFC00), UINT32_C(0xFC00),
		UINT32_C(0x90000000), UINT32_C(0x90000000), UINT32_C(0x90000000), UINT32_C(0x90000000), UINT32_C(0x90000000),
		UINT32_C(0x20000000), UINT32_C(0x20000000), UINT32_C(0x20000000), UINT32_C(0x20000000), UINT32_C(0x20000000)
	};

	uint32_t p;
	uint32_t idx = rmode + tab[inp >> 23];
	uint32_t vlx = tabx[idx];
	switch (idx)
	{
		/*
			Positive number which may be Infinity or NaN.
			We need to check whether it is NaN; if it is, quieten it by setting the top bit of the mantissa.
			(If we don't do this quieting, then a NaN  that is distinguished only by having
			its low-order bits set, would be turned into an INF. */
	case 50:
	case 51:
	case 52:
	case 53:
	case 54:
	case 55:
	case 56:
	case 57:
	case 58:
	case 59:
		/*
			the input value is 0x7F800000 or 0xFF800000 if it is INF.
			By subtracting 1, we get 7F7FFFFF or FF7FFFFF, that is, bit 23 becomes zero.
			For NaNs, however, this operation will keep bit 23 with the value 1.
			We can then extract bit 23, and logical-OR bit 9 of the result with this
			bit in order to quieten the NaN (a Quiet NaN is a NaN where the top bit
			of the mantissa is set.)
		*/
		p = (inp - 1) & UINT32_C(0x800000);	/* zero if INF, nonzero if NaN. */
		return static_cast<sf16>(((inp + vlx) >> 13) | (p >> 14));
		/*
			positive, exponent = 0, round-mode == UP; need to check whether number actually is 0.
			If it is, then return 0, else return 1 (the smallest representable nonzero number)
		*/
	case 0:
		/*
			-inp will set the MSB if the input number is nonzero.
			Thus (-inp) >> 31 will turn into 0 if the input number is 0 and 1 otherwise.
		*/
		return static_cast<sf16>(static_cast<uint32_t>((-static_cast<int32_t>(inp))) >> 31);

		/*
			negative, exponent = , round-mode == DOWN, need to check whether number is
			actually 0. If it is, return 0x8000 ( float -0.0 )
			Else return the smallest negative number ( 0x8001 ) */
	case 6:
		/*
			in this case 'vlx' is 0x80000000. By subtracting the input value from it,
			we obtain a value that is 0 if the input value is in fact zero and has
			the MSB set if it isn't. We then right-shift the value by 31 places to
			get a value that is 0 if the input is -0.0 and 1 otherwise.
		*/
		return static_cast<sf16>(((vlx - inp) >> 31) + UINT32_C(0x8000));

		/*
			for all other cases involving underflow/overflow, we don't need to
			do actual tests; we just return 'vlx'.
		*/
	case 1:
	case 2:
	case 3:
	case 4:
	case 5:
	case 7:
	case 8:
	case 9:
	case 10:
	case 11:
	case 12:
	case 13:
	case 14:
	case 15:
	case 16:
	case 17:
	case 18:
	case 19:
	case 40:
	case 41:
	case 42:
	case 43:
	case 44:
	case 45:
	case 46:
	case 47:
	case 48:
	case 49:
		return static_cast<sf16>(vlx);

		/*
			for normal numbers, 'vlx' is the difference between the FP32 value of a number and the
			FP16 representation of the same number left-shifted by 13 places. In addition, a rounding constant is
			baked into 'vlx': for rounding-away-from zero, the constant is 2^13 - 1, causing roundoff away
			from zero. for round-to-nearest away, the constant is 2^12, causing roundoff away from zero.
			for round-to-nearest-even, the constant is 2^12 - 1. This causes correct round-to-nearest-even
			except for odd input numbers. For odd input numbers, we need to add 1 to the constant. */

		/* normal number, all rounding modes except round-to-nearest-even: */
	case 30:
	case 31:
	case 32:
	case 34:
	case 35:
	case 36:
	case 37:
	case 39:
		return static_cast<sf16>((inp + vlx) >> 13);

		/* normal number, round-to-nearest-even. */
	case 33:
	case 38:
		p = inp + vlx;
		p += (inp >> 13) & 1;
		return static_cast<sf16>(p >> 13);

		/*
			the various denormal cases. These are not expected to be common, so their performance is a bit
			less important. For each of these cases, we need to extract an exponent and a mantissa
			(including the implicit '1'!), and then right-shift the mantissa by a shift-amount that
			depends on the exponent. The shift must apply the correct rounding mode. 'vlx' is used to supply the
			sign of the resulting denormal number.
		*/
	case 21:
	case 22:
	case 25:
	case 27:
		/* denormal, round towards zero. */
		p = 126 - ((inp >> 23) & 0xFF);
		return static_cast<sf16>((((inp & UINT32_C(0x7FFFFF)) + UINT32_C(0x800000)) >> p) | vlx);
	case 20:
	case 26:
		/* denormal, round away from zero. */
		p = 126 - ((inp >> 23) & 0xFF);
		return static_cast<sf16>(rtup_shift32((inp & UINT32_C(0x7FFFFF)) + UINT32_C(0x800000), p) | vlx);
	case 24:
	case 29:
		/* denormal, round to nearest-away */
		p = 126 - ((inp >> 23) & 0xFF);
		return static_cast<sf16>(rtna_shift32((inp & UINT32_C(0x7FFFFF)) + UINT32_C(0x800000), p) | vlx);
	case 23:
	case 28:
		/* denormal, round to nearest-even. */
		p = 126 - ((inp >> 23) & 0xFF);
		return static_cast<sf16>(rtne_shift32((inp & UINT32_C(0x7FFFFF)) + UINT32_C(0x800000), p) | vlx);
	}

	return 0;
}

/* convert from soft-float to native-float */
float sf16_to_float(uint16_t p)
{
	if32 i;
	i.u = sf16_to_sf32(p);
	return i.f;
}

/* convert from native-float to soft-float */
uint16_t float_to_sf16(float p)
{
	if32 i;
	i.f = p;
	return sf32_to_sf16(i.u, SF_NEARESTEVEN);
}

#endif
