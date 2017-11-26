/*************************************************************************/
/*  string_buffer.h                                                      */
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
#ifndef STRING_BUFFER_H
#define STRING_BUFFER_H

#include "ustring.h"

class StringBuffer {
	static const int SHORT_BUFFER_SIZE = 64;

	CharType short_buffer[SHORT_BUFFER_SIZE];
	String buffer;
	int string_length = 0;

	_FORCE_INLINE_ CharType *current_buffer_ptr() {
		return static_cast<Vector<CharType> &>(buffer).empty() ? short_buffer : buffer.ptrw();
	}

public:
	StringBuffer &append(CharType p_char);
	StringBuffer &append(const String &p_string);
	StringBuffer &append(const char *p_str);
	StringBuffer &append(const CharType *p_str, int p_clip_to_len = -1);

	_FORCE_INLINE_ void operator+=(CharType p_char) {
		append(p_char);
	}

	_FORCE_INLINE_ void operator+=(const String &p_string) {
		append(p_string);
	}

	_FORCE_INLINE_ void operator+=(const char *p_str) {
		append(p_str);
	}

	_FORCE_INLINE_ void operator+=(const CharType *p_str) {
		append(p_str);
	}

	StringBuffer &reserve(int p_size);

	int length() const;

	String as_string();

	double as_double();
	int64_t as_int();

	_FORCE_INLINE_ operator String() {
		return as_string();
	}
};

#endif
