/**************************************************************************/
/*  string_builder.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "string_builder.h"

#include <string.h>

StringBuilder &StringBuilder::append(const String &p_string) {
	if (p_string.is_empty()) {
		return *this;
	}

	uint32_t len = p_string.length();

	uint32_t total_length = string_length + len;
	if (capacity < total_length) {
		while (capacity < total_length) {
			capacity <<= 1;
		}
		buffer = (char32_t *)memrealloc(buffer, capacity * sizeof(char32_t));
	}

	memcpy(buffer + string_length, p_string.ptr(), len * sizeof(char32_t));

	string_length = total_length;
	strings++;

	return *this;
}

StringBuilder &StringBuilder::append(const char *p_cstring) {
	uint32_t len = strlen(p_cstring);

	uint32_t total_length = string_length + len;
	if (capacity < total_length) {
		while (capacity < total_length) {
			capacity <<= 1;
		}
		buffer = (char32_t *)memrealloc(buffer, capacity * sizeof(char32_t));
	}

	for (uint32_t i = 0; i < len; i++) {
		buffer[string_length + i] = p_cstring[i];
	}

	string_length = total_length;
	strings++;

	return *this;
}

String StringBuilder::as_string() const {
	return String(buffer, string_length);
}
