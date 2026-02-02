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
 */

#ifndef FLOAXIE_ATOF_H
#define FLOAXIE_ATOF_H

#include <string>

#include <floaxie/krosh.h>

#include <floaxie/default_fallback.h>

#include <floaxie/conversion_status.h>

/** \brief Floaxie functions templates.
 *
 * This namespace contains two main public floaxie functions (`atof()` and
 * `ftoa()`), as well as several helper functions (e.g. `max_buffer_size()`)
 * and internal type and function templates.
 */
namespace floaxie
{
	/** \brief Small decorator around returning value to help the client
	 * optionally receive minor error states along with it.
	 *
	 * \tparam FloatType target floating point type to store results.
	 */
	template<typename FloatType> struct value_and_status
	{
		/** \brief The returning result value itself. */
		FloatType value;
		/** \brief Conversion status indicating any problems occurred. */
		conversion_status status;

		/** \brief Constructs the object with empty value and successful status. */
		value_and_status() noexcept : value(), status(conversion_status::success) { }
		/** \brief Default conversion operator to `FloatType` to make use of the
		 * wrapper more transparent. */
		operator FloatType() const noexcept { return value; }
	};

	/** \brief Parses floating point string representation.
	 *
	 * Interprets string representation of floating point value using Krosh
	 * algorithm and, if successful, value of the specified type is returned.
	 *
	 * The accepted representation format is ordinary or exponential decimal
	 * floating point expression, containing:
	 *   - optional sign ('+' or '-')
	 *   - sequence of one or more decimal digits optionally containing decimal
	 *     point character ('.')
	 *   - optional 'e' of 'E' character followed by optional sign ('+' or '-')
	 *     and sequence of one or more decimal digits.
	 *
	 * Function doesn't expect any preceding spacing characters and treats the
	 * representation as incorrect, if there's any.
	 *
	 * \tparam FloatType target floating point type to store results.
	 * \tparam CharType character type (typically `char` or `wchar_t`) the input
	 * string \p **str** consists of.
	 * \tparam FallbackCallable fallback conversion function type, in case of
	 * Krosh is unsure if the result is correctly rounded (default is `strtof()`
	 * for `float`'s, `strtod()` for `double`'s, `strtold()` for `long double`'s).
	 *
	 * \param str buffer containing the string representation of the value.
	 * \param str_end out parameter, which will contain a pointer to first
	 * character after the parsed value in the specified buffer. If str_end is
	 * null, it is ignored.
	 * \param fallback_func pointer to fallback function. If omitted, by default
	 * is `strtof()` for `float`'s, `strtod()` for `double`'s, `strtold()` for
	 * `long double`'s. Null value will lead to undefined behaviour in case of
	 * algorithm is unsure and fall back to using it.
	 *
	 * \return structure containing the parsed value, if the
	 * input is correct (default constructed value otherwise) and status of the
	 * conversion made.
	 *
	 * \sa `value_and_status`
	 * \sa `conversion_status`
	 */
	template
	<
		typename FloatType,
		typename CharType,
		typename FallbackCallable = FloatType (const CharType*, CharType**)
	>
	inline value_and_status<FloatType> atof(const CharType* str, CharType** str_end, FallbackCallable fallback_func = default_fallback<FloatType, CharType>)
	{
		value_and_status<FloatType> result;

		const auto& cr(krosh<FloatType>(str));

		if (cr.str_end != str)
		{
			if (cr.is_accurate)
			{
				result.value = cr.value;
				result.status = cr.status;
			}
			else
			{
				result.value = fallback_func(str, str_end);
				result.status = check_errno(result.value);

				return result;
			}
		}

		if (str_end)
			*str_end = const_cast<CharType*>(cr.str_end);

		return result;
	}

	/** \brief Tiny overload for `atof()` function to allow passing `nullptr`
	 *  as `str_end` parameter.
	 */
	template
	<
		typename FloatType,
		typename CharType,
		typename FallbackCallable = FloatType (const CharType*, CharType**)
		>
	inline value_and_status<FloatType> atof(const CharType* str, std::nullptr_t str_end, FallbackCallable fallback_func = default_fallback<FloatType, CharType>)
	{
		return atof<FloatType, CharType, FallbackCallable>(str, static_cast<CharType**>(str_end), fallback_func);
	}

	/** \brief Parses floating point represented in `std::basic_string`.
	 *
	 * `atof()` adapter, which may be more useful for cases, where
	 * `std::basic_string` strings are widely used.
	 *
	 * \tparam FloatType target floating point type to store results.
	 * \tparam CharType character type (typically `char` or `wchar_t`) the input
	 * string \p **str** consists of.
	 * \tparam FallbackCallable fallback conversion function type, in case of
	 * Krosh is unsure if the result is correctly rounded (default is `strtof()`
	 * for `float`'s, `strtod()` for `double`'s, `strtold()` for `long double`'s).
	 *
	 * \param str string representation of the value.
	 * \param fallback_func pointer to fallback function. If omitted, by default
	 * is `strtof()` for `float`'s, `strtod()` for `double`'s, `strtold()` for
	 * `long double`'s. Null value will lead to undefined behaviour in case of
	 * algorithm is unsure and fall back to using it.
	 *
	 * \return structure containing the parsed value, if the
	 * input is correct (default constructed value otherwise) and status of the
	 * conversion made.
	 *
	 * \sa `value_and_status`
	 * \sa `conversion_status`
	 */
	template
	<
		typename FloatType,
		typename CharType,
		typename FallbackCallable = FloatType (const CharType*, CharType**)
	>
	inline value_and_status<FloatType> from_string(const std::basic_string<CharType>& str, FallbackCallable fallback_func = default_fallback<FloatType, CharType>)
	{
		return atof<FloatType>(str.c_str(), nullptr, fallback_func);
	}
}

#endif // FLOAXIE_ATOF_H
