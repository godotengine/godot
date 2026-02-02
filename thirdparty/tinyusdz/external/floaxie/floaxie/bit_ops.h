/*
 * Copyright 2015-2019 Alexey Chernov <4ernov@gmail.com>
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
 * diy_fp class and helper functions use code and influenced by
 * Florian Loitsch's original Grisu algorithms implementation
 * (http://florian.loitsch.com/publications/bench.tar.gz)
 * and "Printing Floating-Point Numbers Quickly and Accurately with
 * Integers" paper
 * (http://florian.loitsch.com/publications/dtoa-pldi2010.pdf)
 */

#ifndef FLOAXIE_BIT_OPS_H
#define FLOAXIE_BIT_OPS_H

#include <limits>
#include <algorithm>
#include <cstddef>
#include <cassert>

#include <external/floaxie/floaxie/integer_of_size.h>

namespace floaxie
{
	/** \brief Calculates size of type in bits in compile time.
	 *
	 * \tparam NumericType type to calculate size in bits of.
	 *
	 * \return type size in bits.
	 */
	template<typename NumericType> constexpr std::size_t bit_size() noexcept
	{
		return sizeof(NumericType) * std::numeric_limits<unsigned char>::digits;
	}

	/** \brief Returns a value with bit of the specified power raised.
	 *
	 * Calculates a value, which equals to 2 in the specified power, i.e. with
	 * bit at \p `power` position is `1` and all the remaining bits are `0`.
	 *
	 * \tparam NumericType type to store the calculated value.
	 *
	 * \param power power (0-based index, counting right to left) of bit to
	 * raise.
	 *
	 * \return value of \p **NumericType** with \p **power**-th bit is `1` and
	 * all the remaining bits are `0`.
	 */
	template<typename NumericType> constexpr NumericType raised_bit(std::size_t power)
	{
		assert(power < bit_size<NumericType>());
		return NumericType(1) << power;
	}

	/** \brief Returns Most Significant Bit (MSB) value for the specified type.
	 *
	 * Calculates a value, which is equal to the value of Most
	 * Significant Bit of the integer type, which has the same length, as the
	 * specified one. The left most bit of the calculated value is equal to
	 * `1`, and the remaining bits are `0`.
	 *
	 * \tparam FloatType type to calculate MSB value for.
	 * \tparam NumericType integer type of the same size, as \p **FloatType**.
	 *
	 * \return value of Most Significant Bit (MSB).
	 */
	template<typename FloatType,
	typename NumericType = typename integer_of_size<sizeof(FloatType)>::type>
	constexpr NumericType msb_value() noexcept
	{
		return raised_bit<NumericType>(bit_size<NumericType>() - 1);
	}

	/** \brief Returns maximum unsigned integer value for the specified type.
	 *
	 * Calculates maximum value (using `std::numeric_limits`) of the integer
	 * type, which has the same length, as the specified one. Thus, all bits
	 * of the calculated value are equal to `1`.
	 *
	 * \tparam FloatType type to calculate MSB value for.
	 * \tparam NumericType integer type of the same size, as \p **FloatType**.
	 *
	 * \return maximum value of size the same as of the specified type.
	 */
	template<typename FloatType,
	typename NumericType = typename integer_of_size<sizeof(FloatType)>::type>
	constexpr NumericType max_integer_value() noexcept
	{
		return std::numeric_limits<NumericType>::max();
	}

	/** \brief Masks `n`-th bit of the specified value.
	 *
	 * Calculates a mask standing for the `n`-th bit, performs bitwise **AND**
	 * operation and returns the value of it.
	 *
	 * \param value the value, of which the specified bit is returned.
	 * \param power power (0-based right-to-left index) of bit to return.
	 *
	 * \return integer value, which has \p `power`-th bit of the \p **value**
	 * and the remaining bits equal to `0`.
	 *
	 */
	template<typename NumericType> constexpr bool nth_bit(NumericType value, std::size_t power) noexcept
	{
		return value & raised_bit<NumericType>(power);
	}

	/** \brief Returns Most Significant Bit (MSB) of the specified value.
	 *
	 * Masks the left most bit of the given value, performs bitwise **AND**
	 * operation with the mask and the value and returns the result.
	 *
	 * \tparam NumericType type of the value.
	 *
	 * \param value value to get the highest bit of.
	 *
	 * \return integer value, which left most bit of the \p **value** and the
	 * remaining bits equal to `0`.
	 */
	template<typename NumericType> constexpr bool highest_bit(NumericType value) noexcept
	{
		return nth_bit(value, bit_size<NumericType>() - 1);
	}

	/** \brief Returns mask of \p **n** bits from the right.
	 *
	 * \tparam NumericType type of the returned value.
	 * \param n number of bits from the right to mask.
	 *
	 * \return integer value with \p **n** right bits equal to `1` and the
	 * remaining bits equal to `0`.
	 */
	template<typename NumericType> constexpr NumericType mask(std::size_t n) noexcept
	{
		static_assert(!std::is_signed<NumericType>::value, "Unsigned integral type is expected.");
		return n < bit_size<NumericType>() ? raised_bit<NumericType>(n) - 1 : std::numeric_limits<NumericType>::max();
	}

	/** \brief Rectified linear function.
	 *
	 * Returns the argument (\p **value**), if it's positive and `0` otherwise.
	 *
	 * \param value the argument.
	 *
	 * \return \p **value**, if \p **value** > `0`, `0` otherwise.
	 *
	 */
	template<typename NumericType> constexpr typename std::make_unsigned<NumericType>::type positive_part(NumericType value) noexcept
	{
		return static_cast<typename std::make_unsigned<NumericType>::type>((std::max)(0, value));
	}

	/** \brief Return structure for `round_up` function. */
	struct round_result
	{
		/** \brief Round up result — flag indicating if the value should be
		 *rounded up (i.e. incremented).
		 */
		bool value;
		/** \brief Flag indicating if the rounding was accurate. */
		bool is_accurate;
	};

	/** \brief Detects if rounding up should be done.
	 *
	 * Applies IEEE algorithm of rounding up detection. The rounding criteria
	 * requires, that rounding bit (the bit to the right of target position,
	 * which rounding is being performed to) equals to `1`, and one of the
	 * following conditions is true: * - at least one bit to the right of the rounding bit equals to `1`
	 * - the bit in the target position equals to `1`
	 *
	 * \tparam NumericType type of \p **last_bits** parameter (auto-calculated).
	 *
	 * \param last_bits right suffix of the value, where rounding takes place.
	 * \param round_to_power the power (0-based right-to-left index) of the
	 * target position (which rounding is being performed to). According to the
	 * algorithm math it should be greater, than zero, otherwise behaviour is
	 * undefined.
	 *
	 * \returns `round_result` structure with the rounding decision.
	 */
	template<typename NumericType> inline round_result round_up(NumericType last_bits, std::size_t round_to_power) noexcept
	{
		round_result ret;

		assert(round_to_power > 0);

		const NumericType round_bit(raised_bit<NumericType>(round_to_power - 1));
		const NumericType check_mask(mask<NumericType>(round_to_power + 1) ^ round_bit);
		ret.is_accurate = (last_bits & mask<NumericType>(round_to_power)) != round_bit;
		ret.value = (last_bits & round_bit) && (last_bits & check_mask);

		return ret;
	}

	/** \brief `constexpr` version of `std::abs`, as the latter lacks `constepxr`.
	 *
	 * And is really not `constexpr` in e.g. Clang.
	 *
	 * \returns absolute value of the specified value.
	 */
	template<typename NumericType> constexpr NumericType constexpr_abs(NumericType value)
	{
		return NumericType(0) < value ? value : -value;
	}
}

#endif // FLOAXIE_BIT_OPS_H
