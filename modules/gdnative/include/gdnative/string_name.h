/*************************************************************************/
/*  string_name.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GODOT_STRING_NAME_H
#define GODOT_STRING_NAME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <wchar.h>

#define GODOT_STRING_NAME_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_STRING_NAME_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_STRING_NAME_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_STRING_NAME_SIZE];
} godot_string_name;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/gdnative.h>

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_string_name_new(godot_string_name *r_dest, const godot_string *p_name);
void GDAPI godot_string_name_new_data(godot_string_name *r_dest, const char *p_name);

godot_string GDAPI godot_string_name_get_name(const godot_string_name *p_self);

uint32_t GDAPI godot_string_name_get_hash(const godot_string_name *p_self);
const void GDAPI *godot_string_name_get_data_unique_pointer(const godot_string_name *p_self);

godot_bool GDAPI godot_string_name_operator_equal(const godot_string_name *p_self, const godot_string_name *p_other);
godot_bool GDAPI godot_string_name_operator_less(const godot_string_name *p_self, const godot_string_name *p_other);

void GDAPI godot_string_name_destroy(godot_string_name *p_self);

#ifdef __cplusplus
}
#endif

#endif // GODOT_STRING_NAME_H
