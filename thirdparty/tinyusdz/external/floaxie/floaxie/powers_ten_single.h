/*
 * Copyright 2015-2017 Alexey Chernov <4ernov@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * cached power literals and cached_power() function use code taken from
 * Florian Loitsch's original Grisu algorithms implementation
 * (http://florian.loitsch.com/publications/bench.tar.gz)
 * and "Printing Floating-Point Numbers Quickly and Accurately with
 * Integers" paper
 * (http://florian.loitsch.com/publications/dtoa-pldi2010.pdf)
 */

#ifndef FLOAXIE_POWERS_TEN_SINGLE_H
#define FLOAXIE_POWERS_TEN_SINGLE_H

#include <cstddef>

#include <external/floaxie/floaxie/powers_ten.h>

namespace floaxie
{
	/** \brief Specialization of **powers_ten** for single precision
	 * floating point type (`float`).
	 *
	 * Significand (mantissa) type is `std::uint32_t`, exponent type is
	 * `int` (typically 32 bit, at least 16 bit).
	 *
	 * Values are calculated for powers of 10 in the range of [-50, 50]
	 * exponent.
	 */
	template<> struct powers_ten<float>
	{
		/** \brief Pre-calculated binary 32-bit representation of mantissa of
		 * powers of 10 in the range of [-50, 50] for single precision
		 * floating point values.
		 */
		static constexpr std::uint32_t f[] =
		{
			0xef73d257,
			0x95a86376,
			0xbb127c54,
			0xe9d71b69,
			0x92267121,
			0xb6b00d6a,
			0xe45c10c4,
			0x8eb98a7b,
			0xb267ed19,
			0xdf01e860,
			0x8b61313c,
			0xae397d8b,
			0xd9c7dced,
			0x881cea14,
			0xaa242499,
			0xd4ad2dc0,
			0x84ec3c98,
			0xa6274bbe,
			0xcfb11ead,
			0x81ceb32c,
			0xa2425ff7,
			0xcad2f7f5,
			0xfd87b5f3,
			0x9e74d1b8,
			0xc6120625,
			0xf79687af,
			0x9abe14cd,
			0xc16d9a01,
			0xf1c90081,
			0x971da050,
			0xbce50865,
			0xec1e4a7e,
			0x9392ee8f,
			0xb877aa32,
			0xe69594bf,
			0x901d7cf7,
			0xb424dc35,
			0xe12e1342,
			0x8cbccc09,
			0xafebff0c,
			0xdbe6fecf,
			0x89705f41,
			0xabcc7712,
			0xd6bf94d6,
			0x8637bd06,
			0xa7c5ac47,
			0xd1b71759,
			0x83126e98,
			0xa3d70a3d,
			0xcccccccd,
			0x80000000,
			0xa0000000,
			0xc8000000,
			0xfa000000,
			0x9c400000,
			0xc3500000,
			0xf4240000,
			0x98968000,
			0xbebc2000,
			0xee6b2800,
			0x9502f900,
			0xba43b740,
			0xe8d4a510,
			0x9184e72a,
			0xb5e620f4,
			0xe35fa932,
			0x8e1bc9bf,
			0xb1a2bc2f,
			0xde0b6b3a,
			0x8ac72305,
			0xad78ebc6,
			0xd8d726b7,
			0x87867832,
			0xa968163f,
			0xd3c21bcf,
			0x84595161,
			0xa56fa5ba,
			0xcecb8f28,
			0x813f3979,
			0xa18f07d7,
			0xc9f2c9cd,
			0xfc6f7c40,
			0x9dc5ada8,
			0xc5371912,
			0xf684df57,
			0x9a130b96,
			0xc097ce7c,
			0xf0bdc21b,
			0x96769951,
			0xbc143fa5,
			0xeb194f8e,
			0x92efd1b9,
			0xb7abc627,
			0xe596b7b1,
			0x8f7e32ce,
			0xb35dbf82,
			0xe0352f63,
			0x8c213d9e,
			0xaf298d05,
			0xdaf3f046,
			0x88d8762c
		};

		/** \brief Pre-calculated values of binary exponent of powers of 10 in the
		 * range of [-50, 50].
		 */
		static constexpr int e[] =
		{
			-198,
			-194,
			-191,
			-188,
			-184,
			-181,
			-178,
			-174,
			-171,
			-168,
			-164,
			-161,
			-158,
			-154,
			-151,
			-148,
			-144,
			-141,
			-138,
			-134,
			-131,
			-128,
			-125,
			-121,
			-118,
			-115,
			-111,
			-108,
			-105,
			-101,
			-98,
			-95,
			-91,
			-88,
			-85,
			-81,
			-78,
			-75,
			-71,
			-68,
			-65,
			-61,
			-58,
			-55,
			-51,
			-48,
			-45,
			-41,
			-38,
			-35,
			-31,
			-28,
			-25,
			-22,
			-18,
			-15,
			-12,
			-8,
			-5,
			-2,
			2,
			5,
			8,
			12,
			15,
			18,
			22,
			25,
			28,
			32,
			35,
			38,
			42,
			45,
			48,
			52,
			55,
			58,
			62,
			65,
			68,
			71,
			75,
			78,
			81,
			85,
			88,
			91,
			95,
			98,
			101,
			105,
			108,
			111,
			115,
			118,
			121,
			125,
			128,
			131,
			135
		};

		/** \brief Offsef of the values for zero power in the arrays.
		 */
		static constexpr std::size_t pow_0_offset = 50;

		/** \brief Boundaries of possible powers of ten for the type.
		 */
		static constexpr std::pair<int, int> boundaries = { -50, 50 };
	};

	constexpr decltype(powers_ten<float>::f) powers_ten<float>::f;
	constexpr decltype(powers_ten<float>::e) powers_ten<float>::e;
	constexpr std::size_t powers_ten<float>::pow_0_offset;
	constexpr std::pair<int, int> powers_ten<float>::boundaries;
}

#endif // FLOAXIE_POWERS_TEN_SINGLE_H
