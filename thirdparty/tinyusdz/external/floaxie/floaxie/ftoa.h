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
 */

#ifndef FLOAXIE_FTOA_H
#define FLOAXIE_FTOA_H

#include <string>
#include <type_traits>
#include <cmath>
#include <cstddef>
#include <cassert>

#include <external/floaxie/floaxie/grisu.h>
#include <external/floaxie/floaxie/prettify.h>

namespace floaxie
{
	/** \brief Returns maximum size of buffer can ever be required by `ftoa()`.
	 *
	 * Maximum size of buffer passed to `ftoa()` guaranteed not to lead to
	 * undefined behaviour.
	 *
	 * \tparam FloatType floating point type, which value is planned to be
	 * printed to the buffer.
	 *
	 * \return maximum size of buffer, which can ever be used in the very worst
	 * case.
	 */
	template<typename ValueType> constexpr std::size_t max_buffer_size() noexcept
	{
		typedef typename std::decay<ValueType>::type FloatType;

		// digits, '.' (or 'e' plus three-digit power with optional sign) and '\0'
		return max_digits<FloatType>() + 1 + 1 + 3 + 1;
	}

	/** \brief Prints floating point value to optimal string representation.
	 *
	 * The function prints the string representation of the specified floating
	 * point value using
	 * [**Grisu2**](http://florian.loitsch.com/publications/dtoa-pldi2010.pdf)
	 * algorithm and tries to get it as shorter, as possible. Usually it
	 * succeeds, but sometimes fails, and the output representation is not
	 * the shortest for this value. For the sake of speed improvement this is
	 * ignored, while there's **Grisu3** algorithm which rules this out
	 * informing the caller of the failure, so that it can call slower, but
	 * more accurate algorithm in this case.
	 *
	 * The format of the string representation is one of the following:
	 * 1. Decimal notation, which contains:
	 *  - minus sign ('-') in case of negative value
	 *  - sequence of one or more decimal digits optionally containing
	 *    decimal point character ('.')
	 * 2. Decimal exponent notation, which contains:
	 *  - minus ('-') sign in case of negative value
	 *  - sequence of one or more decimal digits optionally containing
	 *    decimal point character ('.')
	 *  - 'e' character followed by minus sign ('-') in case of negative
	 *    power of the value (i.e. the specified value is < 1) and
	 *    sequence of one, two of three decimal digits.
	 *
	 * \tparam FloatType type of floating point value, calculated using passed
	 * input parameter \p **v**.
	 * \tparam CharType character type (typically `char` or `wchar_t`) of the
	 * output buffer \p **buffer**.
	 *
	 * \param v floating point value to print.
	 * \param buffer character buffer of enough size (see `max_buffer_size()`)
	 * to print the representation to.
	 *
	 * \return number of characters actually written.
	 *
	 * \see `max_buffer_size()`
	 */
	template<typename FloatType, typename CharType> inline std::size_t ftoa(FloatType v, CharType* buffer) noexcept
	{
		if (std::isnan(v))
		{
			buffer[0] = 'n';
			buffer[1] = 'a';
			buffer[2] = 'n';
			buffer[3] = '\0';

			return 3;
		}
		else if (std::isinf(v))
		{
			if (v > 0)
			{
				buffer[0] = 'i';
				buffer[1] = 'n';
				buffer[2] = 'f';
				buffer[3] = '\0';

				return 3;
			}
			else
			{
				buffer[0] = '-';
				buffer[1] = 'i';
				buffer[2] = 'n';
				buffer[3] = 'f';
				buffer[4] = '\0';

				return 4;
			}
		}
		else if (v == 0)
		{
			buffer[0] = '0';
			buffer[1] = '\0';

			return 1;
		}
		else
		{
			*buffer = '-';
			buffer += v < 0;

			constexpr int alpha(grisu_parameters<FloatType>.alpha), gamma(grisu_parameters<FloatType>.gamma);
			constexpr unsigned int decimal_scientific_threshold(16);

			int len, K;

			grisu2<alpha, gamma>(v, buffer, &len, &K);
			return (v < 0) + prettify<decimal_scientific_threshold>(buffer, len, K);
		}
	}

	/** \brief Prints floating point value to optimal representation in
	 * `std::basic_string`.
	 *
	 * Wrapper function around `ftoa()`, which returns `std::basic_string`,
	 * rather than writing to the specified character buffer. This may be
	 * more useful, if working with `std::basic_string` strings is preferred.
	 * Please note, however, than common usage scenarios might be significantly
	 * slower, when many `std::basic_string`'s are created with this function
	 * and concatenated to each other, than when the outputs of `ftoa()` calls
	 * are written to one long buffer.
	 *
	 * \tparam FloatType type of floating point value, calculated using passed
	 * input parameter \p **v**.
	 * \tparam CharType character type (typically `char` or `wchar_t`) of the
	 * output buffer \p **buffer**.
	 *
	 * \param v floating point value to print.
	 * \param buffer character buffer of enough size (see `max_buffer_size()`)
	 * to print the representation to.
	 *
	 * \see `ftoa()`
	 */
	template<typename FloatType, typename CharType> inline std::basic_string<CharType> to_basic_string(FloatType v)
	{
		std::basic_string<CharType> result(max_buffer_size<FloatType>(), CharType());

		ftoa(v, &result.front());

		result.resize(std::char_traits<CharType>::length(result.data()));
		result.shrink_to_fit();

		return result;
	}

	/** \brief 'Specialization' of `to_basic_string()` template for `std::string`. */
	template<typename FloatType> inline std::string to_string(FloatType v)
	{
		return to_basic_string<FloatType, char>(v);
	}

	/** \brief 'Specialization' of `to_basic_string()` template for `std::wstring`. */
	template<typename FloatType> inline std::wstring to_wstring(FloatType v)
	{
		return to_basic_string<FloatType, wchar_t>(v);
	}

	/** \brief 'Specialization' of `to_basic_string()` template for `std::u16string`. */
	template<typename FloatType> inline std::u16string to_u16string(FloatType v)
	{
		return to_basic_string<FloatType, char16_t>(v);
	}

	/** \brief 'Specialization' of `to_basic_string()` template for `std::u32string`. */
	template<typename FloatType> inline std::u32string to_u32string(FloatType v)
	{
		return to_basic_string<FloatType, char32_t>(v);
	}
}

#endif // FLOAXIE_FTOA_H
