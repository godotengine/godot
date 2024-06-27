/**************************************************************************/
/*  char_utils.h                                                          */
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

#ifndef CHAR_UTILS_H
#define CHAR_UTILS_H

#include "core/typedefs.h"

#include "char_range.inc"

#define BSEARCH_CHAR_RANGE(m_array)                      \
	int low = 0;                                         \
	int high = sizeof(m_array) / sizeof(m_array[0]) - 1; \
	int middle;                                          \
                                                         \
	while (low <= high) {                                \
		middle = (low + high) / 2;                       \
                                                         \
		if (c < m_array[middle].start) {                 \
			high = middle - 1;                           \
		} else if (c > m_array[middle].end) {            \
			low = middle + 1;                            \
		} else {                                         \
			return true;                                 \
		}                                                \
	}                                                    \
                                                         \
	return false

static _FORCE_INLINE_ bool is_unicode_identifier_start(char32_t c) {
	BSEARCH_CHAR_RANGE(xid_start);
}

static _FORCE_INLINE_ bool is_unicode_identifier_continue(char32_t c) {
	BSEARCH_CHAR_RANGE(xid_continue);
}

static _FORCE_INLINE_ bool is_unicode_upper_case(char32_t c) {
	BSEARCH_CHAR_RANGE(uppercase_letter);
}

static _FORCE_INLINE_ bool is_unicode_lower_case(char32_t c) {
	BSEARCH_CHAR_RANGE(lowercase_letter);
}

static _FORCE_INLINE_ bool is_unicode_letter(char32_t c) {
	BSEARCH_CHAR_RANGE(unicode_letter);
}

#undef BSEARCH_CHAR_RANGE

static _FORCE_INLINE_ bool is_ascii_upper_case(char32_t c) {
	return (c >= 'A' && c <= 'Z');
}

static _FORCE_INLINE_ bool is_ascii_lower_case(char32_t c) {
	return (c >= 'a' && c <= 'z');
}

static _FORCE_INLINE_ bool is_digit(char32_t c) {
	return (c >= '0' && c <= '9');
}

static _FORCE_INLINE_ bool is_hex_digit(char32_t c) {
	return (is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
}

static _FORCE_INLINE_ bool is_binary_digit(char32_t c) {
	return (c == '0' || c == '1');
}

static _FORCE_INLINE_ bool is_ascii_alphabet_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static _FORCE_INLINE_ bool is_ascii_alphanumeric_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

static _FORCE_INLINE_ bool is_ascii_identifier_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static _FORCE_INLINE_ bool is_symbol(char32_t c) {
	return c != '_' && ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~') || c == '\t' || c == ' ');
}

static _FORCE_INLINE_ bool is_control(char32_t p_char) {
	return (p_char <= 0x001f) || (p_char >= 0x007f && p_char <= 0x009f);
}

static _FORCE_INLINE_ bool is_whitespace(char32_t p_char) {
	return (p_char == ' ') || (p_char == 0x00a0) || (p_char == 0x1680) || (p_char >= 0x2000 && p_char <= 0x200a) || (p_char == 0x202f) || (p_char == 0x205f) || (p_char == 0x3000) || (p_char == 0x2028) || (p_char == 0x2029) || (p_char >= 0x0009 && p_char <= 0x000d) || (p_char == 0x0085);
}

static _FORCE_INLINE_ bool is_linebreak(char32_t p_char) {
	return (p_char >= 0x000a && p_char <= 0x000d) || (p_char == 0x0085) || (p_char == 0x2028) || (p_char == 0x2029);
}

static _FORCE_INLINE_ bool is_punct(char32_t p_char) {
	return (p_char >= ' ' && p_char <= '/') || (p_char >= ':' && p_char <= '@') || (p_char >= '[' && p_char <= '^') || (p_char == '`') || (p_char >= '{' && p_char <= '~') || (p_char >= 0x2000 && p_char <= 0x206f) || (p_char >= 0x3000 && p_char <= 0x303f);
}

static _FORCE_INLINE_ bool is_underscore(char32_t p_char) {
	return (p_char == '_');
}

#endif // CHAR_UTILS_H
