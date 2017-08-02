/*************************************************************************/
/*  string.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include <godot/string.h>

#include "string_db.h"
#include "ustring.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void _string_api_anchor() {
}

void GDAPI godot_string_new(godot_string *r_dest) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
}

void GDAPI godot_string_new_copy(godot_string *r_dest, const godot_string *p_src) {
	String *dest = (String *)r_dest;
	const String *src = (const String *)p_src;
	memnew_placement(dest, String(*src));
}

void GDAPI godot_string_new_data(godot_string *r_dest, const char *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String(String::utf8(p_contents, p_size)));
}

void GDAPI godot_string_new_unicode_data(godot_string *r_dest, const wchar_t *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String(p_contents, p_size));
}

void GDAPI godot_string_get_data(const godot_string *p_self, char *r_dest, int *p_size) {
	String *self = (String *)p_self;
	if (p_size != NULL) {
		*p_size = self->utf8().length();
	}
	if (r_dest != NULL) {
		memcpy(r_dest, self->utf8().get_data(), *p_size);
	}
}

wchar_t GDAPI *godot_string_operator_index(godot_string *p_self, const godot_int p_idx) {
	String *self = (String *)p_self;
	return &(self->operator[](p_idx));
}

const char GDAPI *godot_string_c_str(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	return self->utf8().get_data();
}

const wchar_t GDAPI *godot_string_unicode_str(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	return self->c_str();
}

godot_bool GDAPI godot_string_operator_equal(const godot_string *p_self, const godot_string *p_b) {
	const String *self = (const String *)p_self;
	const String *b = (const String *)p_b;
	return *self == *b;
}

godot_bool GDAPI godot_string_operator_less(const godot_string *p_self, const godot_string *p_b) {
	const String *self = (const String *)p_self;
	const String *b = (const String *)p_b;
	return *self < *b;
}

godot_string GDAPI godot_string_operator_plus(const godot_string *p_self, const godot_string *p_b) {
	godot_string ret;
	const String *self = (const String *)p_self;
	const String *b = (const String *)p_b;
	memnew_placement(&ret, String(*self + *b));
	return ret;
}

void GDAPI godot_string_destroy(godot_string *p_self) {
	String *self = (String *)p_self;
	self->~String();
}

#ifdef __cplusplus
}
#endif
