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

#include "char_list.inc"
#include "char_range.inc"

static _FORCE_INLINE_ bool is_unicode_identifier_start(char32_t c) {
	for (int i = 0; xid_start[i].start != 0; i++) {
		if (c >= xid_start[i].start && c <= xid_start[i].end) {
			return true;
		}
	}
	return false;
}

static _FORCE_INLINE_ bool is_unicode_identifier_continue(char32_t c) {
	for (int i = 0; xid_continue[i].start != 0; i++) {
		if (c >= xid_continue[i].start && c <= xid_continue[i].end) {
			return true;
		}
	}
	return false;
}

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

static _FORCE_INLINE_ bool is_ascii_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static _FORCE_INLINE_ bool is_ascii_alphanumeric_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

static _FORCE_INLINE_ bool is_ascii_identifier_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static _FORCE_INLINE_ bool is_symbol(char32_t c) {
	return c != '_' && ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~') || c == CHAR_HORIZONTAL_TAB || c == ' ');
}

static _FORCE_INLINE_ bool is_control(char32_t p_char) {
	return (p_char <= CHAR_UNIT_SEPARATOR) || (p_char >= CHAR_DELETE && p_char <= CHAR_APPLICATION_PROGRAM_COMMAND);
}

static _FORCE_INLINE_ bool is_whitespace(char32_t p_char) {
	return (p_char == ' ') || (p_char == CHAR_NO_BREAK_SPACE) || (p_char >= CHAR_HORIZONTAL_TAB && p_char <= CHAR_CARRIAGE_RETURN) || (p_char == CHAR_NEXT_LINE)
			// Ogham
			|| (p_char == CHAR_OGHAM_SPACE_MARK)
			// General Punctuation
			|| (p_char >= CHAR_EN_QUAD && p_char <= CHAR_HAIR_SPACE) || (p_char == CHAR_NARROW_NO_BREAK_SPACE) || (p_char == CHAR_MEDIUM_MATHEMATICAL_SPACE) || (p_char == CHAR_LINE_SEPARATOR) || (p_char == CHAR_PARAGRAPH_SEPARATOR)
			// CJK Symbols and Punctuation
			|| (p_char == CHAR_IDEOGRAPHIC_SPACE);
}

static _FORCE_INLINE_ bool is_linebreak(char32_t p_char) {
	return (p_char >= CHAR_NEWLINE && p_char <= CHAR_CARRIAGE_RETURN) || (p_char == CHAR_NEXT_LINE)
			// General Punctuation
			|| (p_char == CHAR_LINE_SEPARATOR) || (p_char == CHAR_PARAGRAPH_SEPARATOR);
}

static _FORCE_INLINE_ bool is_punct(char32_t p_char) {
	return (p_char >= ' ' && p_char <= '/') || (p_char >= ':' && p_char <= '@') || (p_char >= '[' && p_char <= '^') || (p_char == '`') || (p_char >= '[' && p_char <= '~')
			// General punctuation
			|| (p_char >= CHAR_EN_QUAD && p_char <= CHAR_NOMINAL_DIGIT_SHAPES)
			// CJK Symbols and Punctuation
			|| (p_char >= CHAR_IDEOGRAPHIC_SPACE && p_char <= CHAR_IDEOGRAPHIC_HALF_FILL_SPACE);
}

static _FORCE_INLINE_ bool is_underscore(char32_t p_char) {
	return (p_char == '_');
}

#endif // CHAR_UTILS_H
