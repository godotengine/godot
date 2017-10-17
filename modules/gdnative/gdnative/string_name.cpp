/*************************************************************************/
/*  string_name.cpp                                                           */
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
#include "gdnative/string_name.h"

#include "core/string_db.h"
#include "core/ustring.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void _string_name_api_anchor() {
}

void GDAPI godot_string_name_new(godot_string_name *r_dest, const godot_string *p_name) {
	StringName *dest = (StringName *)r_dest;
	const String *name = (const String *)p_name;
	memnew_placement(dest, StringName(*name));
}

void GDAPI godot_string_name_new_data(godot_string_name *r_dest, const char *p_name) {
	StringName *dest = (StringName *)r_dest;
	memnew_placement(dest, StringName(p_name));
}

godot_string GDAPI godot_string_name_get_name(const godot_string_name *p_self) {
	godot_string ret;
	const StringName *self = (const StringName *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

uint32_t GDAPI godot_string_name_get_hash(const godot_string_name *p_self) {
	const StringName *self = (const StringName *)p_self;
	return self->hash();
}

const void GDAPI *godot_string_name_get_data_unique_pointer(const godot_string_name *p_self) {
	const StringName *self = (const StringName *)p_self;
	return self->data_unique_pointer();
}

godot_bool GDAPI godot_string_name_operator_equal(const godot_string_name *p_self, const godot_string_name *p_other) {
	const StringName *self = (const StringName *)p_self;
	const StringName *other = (const StringName *)p_other;
	return self == other;
}

godot_bool GDAPI godot_string_name_operator_less(const godot_string_name *p_self, const godot_string_name *p_other) {
	const StringName *self = (const StringName *)p_self;
	const StringName *other = (const StringName *)p_other;
	return self < other;
}

void GDAPI godot_string_name_destroy(godot_string_name *p_self) {
	StringName *self = (StringName *)p_self;
	self->~StringName();
}

#ifdef __cplusplus
}
#endif
