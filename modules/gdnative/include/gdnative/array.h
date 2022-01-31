/*************************************************************************/
/*  array.h                                                              */
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

#ifndef GODOT_ARRAY_H
#define GODOT_ARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define GODOT_ARRAY_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_ARRAY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_ARRAY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_ARRAY_SIZE];
} godot_array;
#endif

#include <gdnative/gdnative.h>
#include <gdnative/variant_struct.h>

void GDAPI godot_array_new(godot_array *p_self);
void GDAPI godot_array_new_copy(godot_array *r_dest, const godot_array *p_src);
void GDAPI godot_array_destroy(godot_array *p_self);
godot_variant GDAPI *godot_array_operator_index(godot_array *p_self, godot_int p_index);
const godot_variant GDAPI *godot_array_operator_index_const(const godot_array *p_self, godot_int p_index);

#ifdef __cplusplus
}
#endif

#endif // GODOT_ARRAY_H
