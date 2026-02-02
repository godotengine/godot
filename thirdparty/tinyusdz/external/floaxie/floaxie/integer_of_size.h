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

#ifndef FLOAXIE_INTEGER_OF_SIZE_H
#define FLOAXIE_INTEGER_OF_SIZE_H

#include <cstdint>
#include <cstddef>

namespace floaxie
{
	/** \brief Identity type — hold the specified type in internal `typedef`.
	 *
	 * \tparam T type to hold.
	 */
	template<typename T> struct identity
	{
		/** \brief Held type. */
		typedef T type;
	};

	/** \brief Maps some of unsigned integer types to their sizes.
	 *
	 * Useful for choosing unsigned integer type of the same width as some
	 * target type (e.g. floating point) to increase possible accuracy.
	 *
	 * \tparam size size in bytes of the desired type.
	 */
	template<std::size_t size> struct integer_of_size : identity<std::uintmax_t()> {};

	/** \brief Specialization for 64-bit unsigned integer. */
	template<> struct integer_of_size<sizeof(std::uint64_t)> : identity<std::uint64_t> {};

	/** \brief Specialization for 32-bit unsigned integer. */
	template<> struct integer_of_size<sizeof(std::uint32_t)> : identity<std::uint32_t> {};
}

#endif // FLOAXIE_INTEGER_OF_SIZE_H
