/*************************************************************************/
/*  string.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdnative/string.h"

#include "core/string/ustring.h"

static_assert(sizeof(godot_string) == sizeof(String), "String size mismatch");
static_assert(sizeof(godot_char_type) == sizeof(char32_t), "char32_t size mismatch");

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_string_new(godot_string *r_dest) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
}

void GDAPI godot_string_new_copy(godot_string *r_dest, const godot_string *p_src) {
	memnew_placement(r_dest, String(*(String *)p_src));
}

void GDAPI godot_string_new_with_latin1_chars(godot_string *r_dest, const char *p_contents) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	*dest = String(p_contents);
}

void GDAPI godot_string_new_with_utf8_chars(godot_string *r_dest, const char *p_contents) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	dest->parse_utf8(p_contents);
}

void GDAPI godot_string_new_with_utf16_chars(godot_string *r_dest, const char16_t *p_contents) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	dest->parse_utf16(p_contents);
}

void GDAPI godot_string_new_with_utf32_chars(godot_string *r_dest, const char32_t *p_contents) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	*dest = String((const char32_t *)p_contents);
}

void GDAPI godot_string_new_with_wide_chars(godot_string *r_dest, const wchar_t *p_contents) {
	String *dest = (String *)r_dest;
	if (sizeof(wchar_t) == 2) {
		// wchar_t is 16 bit, parse.
		memnew_placement(dest, String);
		dest->parse_utf16((const char16_t *)p_contents);
	} else {
		// wchar_t is 32 bit, copy.
		memnew_placement(dest, String);
		*dest = String((const char32_t *)p_contents);
	}
}

void GDAPI godot_string_new_with_latin1_chars_and_len(godot_string *r_dest, const char *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	*dest = String(p_contents, p_size);
}

void GDAPI godot_string_new_with_utf8_chars_and_len(godot_string *r_dest, const char *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	dest->parse_utf8(p_contents, p_size);
}

void GDAPI godot_string_new_with_utf16_chars_and_len(godot_string *r_dest, const char16_t *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	dest->parse_utf16(p_contents, p_size);
}

void GDAPI godot_string_new_with_utf32_chars_and_len(godot_string *r_dest, const char32_t *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	*dest = String((const char32_t *)p_contents, p_size);
}

void GDAPI godot_string_new_with_wide_chars_and_len(godot_string *r_dest, const wchar_t *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	if (sizeof(wchar_t) == 2) {
		// wchar_t is 16 bit, parse.
		memnew_placement(dest, String);
		dest->parse_utf16((const char16_t *)p_contents, p_size);
	} else {
		// wchar_t is 32 bit, copy.
		memnew_placement(dest, String);
		*dest = String((const char32_t *)p_contents, p_size);
	}
}

const char GDAPI *godot_string_to_latin1_chars(const godot_string *p_self) {
	String *self = (String *)p_self;
	return self->ascii(true).get_data();
}

const char GDAPI *godot_string_to_utf8_chars(const godot_string *p_self) {
	String *self = (String *)p_self;
	return self->utf8().get_data();
}

const char16_t GDAPI *godot_string_to_utf16_chars(const godot_string *p_self) {
	String *self = (String *)p_self;
	return self->utf16().get_data();
}

const char32_t GDAPI *godot_string_to_utf32_chars(const godot_string *p_self) {
	String *self = (String *)p_self;
	return self->get_data();
}

const wchar_t GDAPI *godot_string_to_wide_chars(const godot_string *p_self) {
	String *self = (String *)p_self;
	if (sizeof(wchar_t) == 2) {
		return (const wchar_t *)self->utf16().get_data();
	} else {
		return (const wchar_t *)self->get_data();
	}
}

char32_t GDAPI *godot_string_operator_index(godot_string *p_self, godot_int p_index) {
	String *self = (String *)p_self;
	return self->ptrw();
}

const char32_t GDAPI *godot_string_operator_index_const(const godot_string *p_self, godot_int p_index) {
	const String *self = (const String *)p_self;
	return self->ptr();
}

void GDAPI godot_string_destroy(godot_string *p_self) {
	String *self = (String *)p_self;
	self->~String();
}

#ifdef __cplusplus
}
#endif
