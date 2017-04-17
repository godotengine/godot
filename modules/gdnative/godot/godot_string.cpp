/*************************************************************************/
/*  godot_string.cpp                                                     */
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
#include "godot_string.h"

#include "string_db.h"
#include "ustring.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void _string_api_anchor() {
}

void GDAPI godot_string_new(godot_string *p_str) {
	String *p = (String *)p_str;
	memnew_placement(p, String);
	// *p = String(); // useless here
}

void GDAPI godot_string_new_data(godot_string *p_str, const char *p_contents, const int p_size) {
	String *p = (String *)p_str;
	memnew_placement(p, String);
	*p = String::utf8(p_contents, p_size);
}

void GDAPI godot_string_new_unicode_data(godot_string *p_str, const wchar_t *p_contents, const int p_size) {
	String *p = (String *)p_str;
	memnew_placement(p, String);
	*p = String(p_contents, p_size);
}

void GDAPI godot_string_get_data(const godot_string *p_str, char *p_dest, int *p_size) {
	String *p = (String *)p_str;
	if (p_size != NULL) {
		*p_size = p->utf8().length();
	}
	if (p_dest != NULL) {
		memcpy(p_dest, p->utf8().get_data(), *p_size);
	}
}

void GDAPI godot_string_copy_string(const godot_string *p_dest, const godot_string *p_src) {
	String *dest = (String *)p_dest;
	String *src = (String *)p_src;

	*dest = *src;
}

wchar_t GDAPI *godot_string_operator_index(godot_string *p_str, const godot_int p_idx) {
	String *s = (String *)p_str;
	return &(s->operator[](p_idx));
}

const char GDAPI *godot_string_c_str(const godot_string *p_str) {
	const String *s = (const String *)p_str;
	return s->utf8().get_data();
}

const wchar_t GDAPI *godot_string_unicode_str(const godot_string *p_str) {
	const String *s = (const String *)p_str;
	return s->c_str();
}

godot_bool GDAPI godot_string_operator_equal(const godot_string *p_a, const godot_string *p_b) {
	String *a = (String *)p_a;
	String *b = (String *)p_b;
	return *a == *b;
}

godot_bool GDAPI godot_string_operator_less(const godot_string *p_a, const godot_string *p_b) {
	String *a = (String *)p_a;
	String *b = (String *)p_b;
	return *a < *b;
}

void GDAPI godot_string_operator_plus(godot_string *p_dest, const godot_string *p_a, const godot_string *p_b) {
	String *dest = (String *)p_dest;
	const String *a = (String *)p_a;
	const String *b = (String *)p_b;

	String tmp = *a + *b;
	godot_string_new(p_dest);
	*dest = tmp;
}

void GDAPI godot_string_destroy(godot_string *p_str) {
	String *p = (String *)p_str;
	p->~String();
}

#ifdef __cplusplus
}
#endif
