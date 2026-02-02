/*
 * Copyright 2015-2022 Alexey Chernov <4ernov@gmail.com>
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
 * prettify_string() and fill_exponent() functions use code taken from
 * Florian Loitsch's original Grisu algorithms implementation
 * (http://florian.loitsch.com/publications/bench.tar.gz)
 * and "Printing Floating-Point Numbers Quickly and Accurately with
 * Integers" paper
 * (http://florian.loitsch.com/publications/dtoa-pldi2010.pdf)
 */

#ifndef FLOAXIE_PRETTIFY_H
#define FLOAXIE_PRETTIFY_H

#include <cstring>
#include <cassert>
#include <cstdlib>
#include <algorithm>

#include <external/floaxie/floaxie/static_pow.h>
#include <external/floaxie/floaxie/print.h>
#include <external/floaxie/floaxie/memwrap.h>

namespace floaxie
{
	/** \brief Format enumeration.
	 *
	 */
	enum format {
		/** \brief Decimal format. */
		decimal,
		/** \brief Decimal exponent (a.k.a. *scientific*) format. */
		scientific
	};

	/** \brief LUT of two-digit decimal values to speed up their printing. */
	constexpr const char digits_lut[200] = {
		'0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0', '7', '0', '8', '0', '9',
		'1', '0', '1', '1', '1', '2', '1', '3', '1', '4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9',
		'2', '0', '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9',
		'3', '0', '3', '1', '3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3', '7', '3', '8', '3', '9',
		'4', '0', '4', '1', '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4', '8', '4', '9',
		'5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9',
		'6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5', '6', '6', '6', '7', '6', '8', '6', '9',
		'7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6', '7', '7', '7', '8', '7', '9',
		'8', '0', '8', '1', '8', '2', '8', '3', '8', '4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9',
		'9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7', '9', '8', '9', '9'
	};

	/** \brief Detects the more appropriate format to print based on
	 * \p **threshold** value.
	 * \tparam threshold the maximum number of digits in the string
	 * representation, when decimal format can be used (otherwise decimal
	 * exponent or *scientific* format is used).
	 *
	 * \return value of the chosen format.
	 * \see `format`
	 */
	template<std::size_t threshold> inline format choose_format(const std::size_t field_width) noexcept
	{
		static_assert(threshold > static_pow<10, 1>(), "Only 10 ⩽ |threshold| ⩽ 100 is supported");

		return field_width > threshold ? format::scientific : format::decimal;
	}

	/** \brief Prints decimal exponent value.
	 *
	 * \tparam CharType character type (typically `char` or `wchar_t`) of the
	 * output buffer \p **buffer**.
	 *
	 * \param K decimal exponent value.
	 * \param buffer character buffer to print to.
	 *
	 * \return number of characters written to the buffer.
	 */
	template<typename CharType> inline std::size_t fill_exponent(unsigned int K, CharType* buffer) noexcept
	{
		const unsigned char hundreds = static_cast<unsigned char>(K / 100);
		K %= 100;
		buffer[0] = '0' + hundreds;
		buffer += (hundreds > 0);

		const char* d = digits_lut + K * 2;
		buffer[0] = d[0];
		buffer[1] = d[1];

		buffer[2] = '\0';

		return 2 + (hundreds > 0);
	}

	/** \brief Prints exponent (*scientific*) part of value representation in
	 * decimal exponent format.
	 *
	 * \tparam CharType character type (typically `char` or `wchar_t`) of the
	 * output buffer \p **buffer**.
	 *
	 * \param buffer character buffer with properly printed mantissa.
	 * \param len output parameter to return the length of printed
	 * representation.
	 * \param dot_pos number of character, where dot position should be placed.
	 *
	 * \return number of characters written to the buffer.
	 *
	 * \see `print_decimal()`
	 */
	template<typename CharType> inline std::size_t print_scientific(CharType* buffer, const unsigned int len, const int dot_pos) noexcept
	{
		const int K = dot_pos - 1;
		if (len > 1)
		{
			/* leave the first digit. then add a '.' and at the end 'e...' */
			wrap::memmove(buffer + 2, buffer + 1, len - 1);
			buffer[1] = '.';
			buffer += len;
		}

		/* add 'e...' */
		buffer[1] = 'e';
		buffer[2] = '-';
		buffer += K < 0;

		return len + /*dot*/(len > 1) + /*'e'*/1 + /*exp sign*/(K < 0) + fill_exponent(std::abs(K), buffer + 2);
	}

	/** \brief Formats decimal mantissa part of value representation.
	 *
	 * Tides up the printed digits in \p **buffer**, adding leading zeros,
	 * placing the decimal point into the proper place etc.
	 *
	 * \tparam CharType character type (typically `char` or `wchar_t`) of the
	 * output buffer \p **buffer**.
	 *
	 * \param buffer character buffer with printed digits.
	 * \param len length of current representation in \p **buffer**.
	 * \param k decimal exponent of the value.
	 *
	 * \return number of characters written to the buffer.
	 */
	template<typename CharType> inline std::size_t print_decimal(CharType* buffer, const unsigned int len, const int k) noexcept
	{
		const int dot_pos = static_cast<int>(len) + k;

		const unsigned int actual_dot_pos = dot_pos > 0 ? uint32_t(dot_pos) : 1;

		const unsigned int left_offset = dot_pos > 0 ? 0 : 2 - uint32_t(dot_pos);
		const unsigned int right_offset = positive_part(k);

		const unsigned int left_shift_src = positive_part(dot_pos);
		const unsigned int left_shift_dest = dot_pos > 0 ? left_shift_src + (k < 0) : left_offset;
		const unsigned int left_shift_len = positive_part(static_cast<int>(len) - static_cast<int>(left_shift_src));

		const unsigned int term_pos = len + right_offset + (left_shift_dest - left_shift_src);

		wrap::memmove(buffer + left_shift_dest, buffer + left_shift_src, left_shift_len);
		wrap::memset(buffer, CharType('0'), left_offset);
		wrap::memset(buffer + len, CharType('0'), right_offset);
		buffer[actual_dot_pos] = '.';
		buffer[term_pos] = '\0';

		return term_pos;
	}

	/** \brief Makes final format corrections to have the string representation
	 * be properly and pretty printed (with decimal point in place, exponent
	 * part, where appropriate etc.).
	 *
	 * \tparam decimal_scientific_threshold the maximum number of digits in the
	 * string representation, when decimal format can be used (otherwise
	 * decimal exponent or *scientific* format is used).
	 * \tparam CharType character type (typically `char` or `wchar_t`) of the
	 * output buffer \p **buffer**.
	 *
	 * \param buffer character buffer with printed digits.
	 * \param len length of current representation in \p **buffer**.
	 * \param k decimal exponent of the value.
	 *
	 * \return number of characters written to the buffer.
	 *
	 * \see `print_decimal()`
	 * \see `print_scientific()`
	 */
	template<std::size_t decimal_scientific_threshold, typename CharType>
	inline std::size_t prettify(CharType* buffer, const unsigned int len, const int k) noexcept
	{
		/* v = buffer * 10 ^ k
			dot_pos is such that 10 ^ (dot_pos - 1) <= v < 10 ^ dot_pos
			this way dot_pos gives the position of the comma.
		*/
		const int dot_pos = static_cast<int>(len) + k;

		// is always positive, since dot_pos is negative only when k is negative
		const std::size_t field_width = size_t((std::max)(dot_pos, -k));

		switch (choose_format<decimal_scientific_threshold>(field_width))
		{
		case format::decimal:
			return print_decimal(buffer, len, k);

		case format::scientific:
			return print_scientific(buffer, len, dot_pos);
		}

		// never reach here
		return 0;
	}
}

#endif // FLOAXIE_PRETTIFY_H
