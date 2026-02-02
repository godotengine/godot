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

#ifndef FLOAXIE_CACHED_POWER_H
#define FLOAXIE_CACHED_POWER_H

#include <cstddef>
#include <cassert>

#include <external/floaxie/floaxie/powers_ten_single.h>
#include <external/floaxie/floaxie/powers_ten_double.h>

#include <external/floaxie/floaxie/diy_fp.h>

namespace floaxie
{
	/** \brief Returns pre-calculated `diy_fp` value of 10 in the specified
	 * power using pre-calculated and compiled version of binary mantissa
	 * and exponent.
	 *
	 * \tparam FloatType floating point type to call the values for.
	 */
	template<typename FloatType> inline diy_fp<FloatType> cached_power(int k) noexcept
	{
		assert(k >= -static_cast<int>(powers_ten<FloatType>::pow_0_offset));

		const std::size_t index = powers_ten<FloatType>::pow_0_offset + k;

		return diy_fp<FloatType>(powers_ten<FloatType>::f[index], powers_ten<FloatType>::e[index]);
	}
}

#endif // FLOAXIE_CACHED_POWER_H
