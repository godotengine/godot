/*************************************************************************/
/*  string_buffer.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "string_buffer.h"

#include <string.h>

StringBuffer &StringBuffer::append(CharType p_char) {
	reserve(string_length + 2);
	current_buffer_ptr()[string_length++] = p_char;
	return *this;
}

StringBuffer &StringBuffer::append(const String &p_string) {
	return append(p_string.c_str());
}

StringBuffer &StringBuffer::append(const char *p_str) {
	int len = strlen(p_str);
	reserve(string_length + len + 1);

	CharType *buf = current_buffer_ptr();
	for (const char *c_ptr = p_str; c_ptr; ++c_ptr) {
		buf[string_length++] = *c_ptr;
	}
	return *this;
}

StringBuffer &StringBuffer::append(const CharType *p_str, int p_clip_to_len) {
	int len = 0;
	while ((p_clip_to_len < 0 || len < p_clip_to_len) && p_str[len]) {
		++len;
	}
	reserve(string_length + len + 1);
	memcpy(&(current_buffer_ptr()[string_length]), p_str, len * sizeof(CharType));
	string_length += len;

	return *this;
}

StringBuffer &StringBuffer::reserve(int p_size) {
	if (p_size < SHORT_BUFFER_SIZE || p_size < buffer.size())
		return *this;

	bool need_copy = string_length > 0 && buffer.empty();
	buffer.resize(next_power_of_2(p_size));
	if (need_copy) {
		memcpy(buffer.ptrw(), short_buffer, string_length * sizeof(CharType));
	}

	return *this;
}

int StringBuffer::length() const {
	return string_length;
}

String StringBuffer::as_string() {
	current_buffer_ptr()[string_length] = '\0';
	if (buffer.empty()) {
		return String(short_buffer);
	} else {
		buffer.resize(string_length + 1);
		return buffer;
	}
}

double StringBuffer::as_double() {
	current_buffer_ptr()[string_length] = '\0';
	return String::to_double(current_buffer_ptr());
}

int64_t StringBuffer::as_int() {
	current_buffer_ptr()[string_length] = '\0';
	return String::to_int(current_buffer_ptr());
}
