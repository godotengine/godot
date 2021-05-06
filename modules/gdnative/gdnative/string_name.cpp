/*************************************************************************/
/*  string_name.cpp                                                      */
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

#include "gdnative/string_name.h"

#include "core/string/string_name.h"

static_assert(sizeof(godot_string_name) == sizeof(StringName), "StringName size mismatch");

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_string_name_new(godot_string_name *r_dest) {
	StringName *dest = (StringName *)r_dest;
	memnew_placement(dest, StringName);
}

void GDAPI godot_string_name_new_copy(godot_string_name *r_dest, const godot_string_name *p_src) {
	memnew_placement(r_dest, StringName(*(StringName *)p_src));
}

void GDAPI godot_string_name_new_with_latin1_chars(godot_string_name *r_dest, const char *p_contents) {
	StringName *dest = (StringName *)r_dest;
	memnew_placement(dest, StringName(p_contents));
}

void GDAPI godot_string_name_destroy(godot_string_name *p_self) {
	StringName *self = (StringName *)p_self;
	self->~StringName();
}

#ifdef __cplusplus
}
#endif
