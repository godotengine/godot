/*************************************************************************/
/*  godot_string.h                                                       */
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
#ifndef GODOT_STRING_H
#define GODOT_STRING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_STRING_TYPE_DEFINED
typedef struct godot_string {
	uint8_t _dont_touch_that[8];
} godot_string;
#endif

#include "../godot.h"

void GDAPI godot_string_new(godot_string *p_str);
void GDAPI godot_string_new_data(godot_string *p_str, const char *p_contents, const int p_size);

void GDAPI godot_string_get_data(const godot_string *p_str, char *p_dest, int *p_size);

void GDAPI godot_string_copy_string(const godot_string *p_dest, const godot_string *p_src);

wchar_t GDAPI *godot_string_operator_index(godot_string *p_str, const godot_int p_idx);
const char GDAPI *godot_string_c_str(const godot_string *p_str);

godot_bool GDAPI godot_string_operator_equal(const godot_string *p_a, const godot_string *p_b);
godot_bool GDAPI godot_string_operator_less(const godot_string *p_a, const godot_string *p_b);
void GDAPI godot_string_operator_plus(godot_string *p_dest, const godot_string *p_a, const godot_string *p_b);

// @Incomplete
// hmm, I guess exposing the whole API doesn't make much sense
// since the language used in the library has its own string funcs

void GDAPI godot_string_destroy(godot_string *p_str);

#ifdef __cplusplus
}
#endif

#endif // GODOT_STRING_H
