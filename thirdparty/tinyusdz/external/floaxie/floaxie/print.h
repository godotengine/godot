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
 *
 * Several utility functions as printing binary representation of floating
 * point values and other stuff.
 *
 */

#ifndef FLOAXIE_PRINT_H
#define FLOAXIE_PRINT_H

#include <bitset>
#include <string>
#include <cstdint>

#include <external/floaxie/floaxie/type_punning_cast.h>
#include <external/floaxie/floaxie/bit_ops.h>

namespace floaxie
{
	/** \brief Prints `double` value in binary format, splitting sign, exponent
	 * and mantissa parts with spaces.
	 *
	 * Useful for debugging purposes.
	 */
	std::string inline print_binary(double f)
	{
		auto s(std::bitset<64>(type_punning_cast<std::uint64_t>(f)).to_string());
		s.insert(1, 1, ' ');
		s.insert(13, 1, ' ');
		return s;
	}

	/** \brief Print arbitrary numeric value in binary format. */
	template<typename NumericType> inline std::string print_binary(NumericType v)
	{
		return std::bitset<bit_size<NumericType>()>(v).to_string();
	}

	/** \brief Print arbitrary numeric value as if it were `double`. */
	template<typename NumericType> inline std::string print_double_presentation(NumericType v)
	{
		auto s(std::bitset<64>(v).to_string());
		s.insert(1, 1, ' ');
		s.insert(54, 1, ' ');
		s.insert(56, 1, ' ');
		return s;
	}
}

#endif // FLOAXIE_PRINT_H
