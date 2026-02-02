/*
 * Copyright 2015, 2016 Alexey Chernov <4ernov@gmail.com>
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
 */

#ifndef FLOAXIE_FRACTION_H
#define FLOAXIE_FRACTION_H

#include <cstddef>

#include <floaxie/static_pow.h>

namespace floaxie
{
	/** \brief Recursive structure template used to convert decimal common
	 * fraction to binary common fraction.
	 *
	 * The structure template hierarchy is used to calculate the binary common
	 * fraction value, which is approximately equal to given decimal common
	 * fraction.
	 *
	 * Example:
	 * ~~~
	 * 0.625 = 0.101
	 * ~~~
	 *
	 * \tparam T numeric type of input value and result.
	 * \tparam decimal_digits number of significant decimal digits in
	 * numerator (equals to the power of ten in denominator); for example, for
	 * 0.625 \p **decimal_digits** is 3.
	 * \tparam binary_digits number of significant binary digits in numerator
	 * of the result, which is defined by the necessary accuracy of
	 * result approximation; for example, for 0.101 \p **decimal_digits** is 3.
	 * \tparam current_binary_digit template cursor used to implement
	 * recursive descent.
	 * \tparam terminal automatically calculated flag of recursion termination.
	 */
	template
	<
		typename T,
		std::size_t decimal_digits,
		std::size_t binary_digits,
		std::size_t current_binary_digit,
		bool terminal = (binary_digits == current_binary_digit)
	>
	struct fraction_converter;

	/** \brief Intermediate step implementation of `fraction_converter`
	 * template.
	 */
	template
	<
		typename T,
		std::size_t decimal_digits,
		std::size_t binary_digits,
		std::size_t current_binary_digit
	>
	struct fraction_converter<T, decimal_digits, binary_digits, current_binary_digit, false>
	{
		/** \brief Calculates \p `current_binary_digit`-th digit of the result.
		 *
		 * \param decimal_numerator value of decimal numerator of common
		 * fraction to convert.
		 *
		 * \return properly shifted value of \p `current_binary_digit`-th
		 * digit of the result.
		 */
		static T convert(T decimal_numerator)
		{
			constexpr T numerator(static_pow<10, decimal_digits>());
			constexpr T denominator(static_pow<2, current_binary_digit>());
			constexpr T decimal_fraction(numerator / denominator);

			constexpr std::size_t shift_amount(binary_digits - current_binary_digit);

			const T decision(decimal_numerator >= decimal_fraction);
			const T residue(decimal_numerator - decision * decimal_fraction);

			return (decision << shift_amount) |
				fraction_converter<T, decimal_digits, binary_digits, current_binary_digit + 1>::convert(residue);
		}
	};

	/** \brief Terminal step implementation of `fraction_converter` template.
	 *
	 */
	template
	<
		typename T,
		std::size_t decimal_digits,
		std::size_t binary_digits,
		std::size_t current_binary_digit
	>
	struct fraction_converter<T, decimal_digits, binary_digits, current_binary_digit, true>
	{
		/** \brief Calculates least significant digit of the result.
		 *
		 * \param decimal_numerator value of decimal numerator of common
		 * fraction to convert.
		 *
		 * \return right most (least significant) digit of the result.
		 */
		static T convert(T decimal_numerator)
		{
			constexpr T numerator(static_pow<10, decimal_digits>());
			constexpr T denominator(static_pow<2, current_binary_digit>());
			constexpr T decimal_fraction(numerator / denominator);

			return decimal_numerator >= decimal_fraction;
		}
	};

	/** \brief Wrapper function to convert numerator of decimal common fraction
	 * to numerator of approximately equal binary common fraction with the
	 * specified accuracy.
	 *
	 * \tparam decimal_digits number of digits in decimal numerator to observe
	 * (input accuracy).
	 * \tparam binary_digits number of digits in binary numerator to generate
	 * (output accuracy).
	 *
	 * \return value of binary numerator with the specified accuracy as
	 * calculated by `fraction_converter`.
	 */
	template<std::size_t decimal_digits, std::size_t binary_digits, typename T> inline T convert_numerator(T decimal_numerator)
	{
		return fraction_converter<T, decimal_digits, binary_digits, 1>::convert(decimal_numerator);
	}
}

#endif // FLOAXIE_FRACTION_H
