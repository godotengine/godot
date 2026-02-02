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

#ifndef FLOAXIE_DEFAULT_FALLBACK_H
#define FLOAXIE_DEFAULT_FALLBACK_H

#include <cstdlib>
#include <cwchar>

#include <floaxie/conversion_status.h>

namespace floaxie
{
	/** \brief Function template to wrap C Standard Library floating point
	 * parse function based on floating-point type and character type (normal
	 * or wide).
	 *
	 * \tparam FloatType floating point type to parse.
	 * \tparam CharType character type of string to parse.
	 */
	template<typename FloatType, typename CharType> FloatType default_fallback(const CharType* str, CharType** str_end);

	/** \brief `float` and `char`. */
	template<> inline float default_fallback<float, char>(const char* str, char** str_end)
	{
		return std::strtof(str, str_end);
	}

	/** \brief `double` and `char`. */
	template<> inline double default_fallback<double, char>(const char* str, char** str_end)
	{
		return std::strtod(str, str_end);
	}

	/** \brief `long double` and `char`. */
	template<> inline long double default_fallback<long double, char>(const char* str, char** str_end)
	{
		return std::strtold(str, str_end);
	}

	/** \brief `float` and `wchar_t`. */
	template<> inline float default_fallback<float, wchar_t>(const wchar_t* str, wchar_t** str_end)
	{
		return std::wcstof(str, str_end);
	}

	/** \brief `double` and `wchar_t`. */
	template<> inline double default_fallback<double, wchar_t>(const wchar_t* str, wchar_t** str_end)
	{
		return std::wcstod(str, str_end);
	}

	/** \brief `long double` and `wchar_t`. */
	template<> inline long double default_fallback<long double, wchar_t>(const wchar_t* str, wchar_t** str_end)
	{
		return std::wcstold(str, str_end);
	}

	/** \brief Returns `conversion_status` based on `errno` value.
	 *
	 * Analyzes current value of `errno` together with the passed conversion
	 * result and returns `conversion_status` value for the case.
	 *
	 * \tparam FloatType floating-point type of the returned value passed.
	 *
	 * \p returned_value the value returned after the conversion.
	 *
	 * \return status of the last conversion.
	 *
	 * \sa `conversion_status`
	 */
	template<typename FloatType> conversion_status check_errno(FloatType returned_value)
	{
		if (errno != ERANGE)
			return conversion_status::success;

		return returned_value ? conversion_status::overflow : conversion_status::underflow;
	}
}

#endif // FLOAXIE_DEFAULT_FALLBACK_H
