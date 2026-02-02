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

#ifndef FLOAXIE_POWERS_TEN_H
#define FLOAXIE_POWERS_TEN_H

namespace floaxie
{
	/** \brief Structure template to store compile-time values of powers of ten.
	 *
	 * The template represents a structure family, which stores pre-calculated
	 * values of significands and exponents of powers of ten, represented in
	 * integer form of the specified precision.
	 *
	 * The exact values are written in specializations of the template for
	 * single precision or double precision floating point type.
	 *
	 * The template specializations are used in **Grisu** and **Krosh** algorithms
	 * to get the pre-calculated cached power of ten.
	 *
	 * \tparam FloatType floating point type, for which precision the values are
	 * calculated.
	 */
	template<typename FloatType> struct powers_ten;
}

#endif // FLOAXIE_POWERS_TEN_H
