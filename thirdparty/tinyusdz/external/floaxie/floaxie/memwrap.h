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

#ifndef FLOAXIE_MEMWRAP_H
#define FLOAXIE_MEMWRAP_H

#include <cstring>
#include <cwchar>

namespace floaxie
{
	namespace wrap
	{
		/** \brief wrapper to template `std::(w)memset` by character type.
		 *
		 * \tparam CharType character type used.
		*/
		template<typename CharType> inline CharType* memset(CharType* dest, CharType ch, std::size_t count);

		template<> inline char* memset(char* dest, char ch, std::size_t count)
		{
			return static_cast<char*>(std::memset(dest, ch, count));
		}

		template<> inline wchar_t* memset(wchar_t* dest, wchar_t ch, std::size_t count)
		{
			return std::wmemset(dest, ch, count);
		}

		/** \brief wrapper to template `std::(w)memmove` by character type.
		 *
		 * \tparam CharType character type used.
		 */
		template<typename CharType> inline CharType* memmove(CharType* dest, const CharType* src, std::size_t count);

		template<> inline char* memmove(char* dest, const char* src, std::size_t count)
		{
			return static_cast<char*>(std::memmove(dest, src, count));
		}

		template<> inline wchar_t* memmove(wchar_t* dest, const wchar_t* src, std::size_t count)
		{
			return std::wmemmove(dest, src, count);
		}
	}
}

#endif // FLOAXIE_MEMWRAP_H
