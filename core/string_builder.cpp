/*************************************************************************/
/*  string_builder.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "string_builder.h"
#include "core/math/math_funcs.h"
#include <string.h>

StringBuilder &StringBuilder::append(const StringBuilder::Chunk &p_chunk)
{
	const double growthFactor = 1.7;
	const int minCapacity = 8;
	const int capacity = chunks.size();

	if (capacity <= 0) {
		Error err = chunks.resize(minCapacity);
		ERR_FAIL_COND_V(err, *this)
	}
	else if (capacity <= chunks_count) {
		int newCapacity = (int)Math::floor((double)capacity * growthFactor);
		ERR_FAIL_COND_V(newCapacity <= 0, *this)
		ERR_FAIL_COND_V(newCapacity <= capacity, *this)

		Error err = chunks.resize(newCapacity);
		ERR_FAIL_COND_V(err, *this)
	}

	chunks.set(chunks_count, p_chunk);

	chunks_count++;
	string_length += p_chunk.length;

	if (!resultCache.empty())
		resultCache.clear();

	return *this;
}

StringBuilder &StringBuilder::append(const String &p_string) {

	if (p_string.empty())
		return *this;

	append(Chunk(p_string));

	return *this;
}

StringBuilder &StringBuilder::append(const char *p_cstring) {

	Chunk c(p_cstring);

	if (c.length <= 0)
		return *this;

	append(c);

	return *this;
}

String StringBuilder::as_string() const {

	if (string_length == 0)
		return String();

	if (!resultCache.empty())
		return resultCache;

	CharType *buffer = memnew_arr(CharType, string_length);

	uint32_t current_position = 0;

	for (int i = 0; i < chunks_count; i++) {
		const Chunk& c = chunks.get(i);

		if (c.str_ptr == nullptr) {
			// Godot string
			memcpy(buffer + current_position, c.str_string.ptr(), c.length * sizeof(CharType));

			current_position += c.length;
		} else {
			// char* string
			for (uint32_t j = 0; j < c.length; j++) {
				buffer[current_position + j] = (const CharType)c.str_ptr[j];
			}

			current_position += c.length;
		}
	}

	// done
	CRASH_COND((uint32_t)current_position != string_length);

	resultCache = String(buffer, (int)string_length);

	memdelete_arr(buffer);

	return resultCache;
}
