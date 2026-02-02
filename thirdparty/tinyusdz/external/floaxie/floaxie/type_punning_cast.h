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
 * type_punning_cast is copy-paste of pseudo_cast as posted by plasmacel
 * at StackOverflow
 * (http://stackoverflow.com/questions/17789928/whats-a-proper-way-of-type-punning-a-float-to-an-int-and-vice-versa)
 *
 */

#ifndef FLOAXIE_TYPE_PUNNING_CAST
#define FLOAXIE_TYPE_PUNNING_CAST

#include <cstring>

namespace floaxie
{
	/** \brief Correct type-punning cast implementation to avoid any possible
	 * undefined behaviour.
	 *
	 * \see https://en.wikipedia.org/wiki/Type_punning
	 */
	template<typename T, typename U> inline T type_punning_cast(const U& x)
	{
		static_assert(sizeof(T) == sizeof(U),
				"type_punning_cast can't handle types with different size");

		T to;
		std::memcpy(&to, &x, sizeof(T));
		return to;
	}
}

#endif // FLOAXIE_TYPE_PUNNING_CAST
