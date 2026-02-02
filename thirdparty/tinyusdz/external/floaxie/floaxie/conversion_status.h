/*
 * Copyright 2015, 2016, 2017 Alexey Chernov <4ernov@gmail.com>
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

#ifndef FLOAXIE_CONVERSION_STATUS_H
#define FLOAXIE_CONVERSION_STATUS_H

namespace floaxie
{
	/** \brief Enumeration of possible conversion results, either successful or
	 * not.
	 */
	enum class conversion_status : unsigned char
	{

		success, /**< The conversion was successful. */
		underflow, /**< An underflow occurred during the conversion. */
		overflow /**< An overflow occurred during the conversion. */
	};
}

#endif // FLOAXIE_CONVERSION_STATUS_H
