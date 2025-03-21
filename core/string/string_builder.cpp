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

	strings.push_back(p_string);

	string_length += p_string.length();

	return *this;
}

StringBuilder &StringBuilder::append(const char *p_cstring) {
	int32_t len = strlen(p_cstring);

	strings.push_back(Span<char>(p_cstring, len));

	string_length += len;

	return *this;
}

String StringBuilder::as_string() const {
	if (string_length == 0) {
		return "";
	}

	String string;
	string.resize(string_length + 1);
	char32_t *buffer = string.ptrw();

	int current_position = 0;

	for (const Either<String, Span<char>> &str : strings) {
		if (str.is<String>()) {
			// Godot string
			const String &s = str.get_unchecked<String>();

			memcpy(buffer + current_position, s.ptr(), s.length() * sizeof(char32_t));

			current_position += s.length();
		} else {
			const Span<char> &s = str.get_unchecked<Span<char>>();

			for (uint64_t i = 0; i < s.size(); i++) {
				buffer[current_position + i] = s.ptr()[i];
			}

			current_position += s.size();
		}
	}
	buffer[current_position] = 0;

	return string;
}
