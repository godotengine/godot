/*************************************************************************/
/*  rid.h                                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef GODOT_RID_H
#define GODOT_RID_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define GODOT_RID_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_RID_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_RID_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_RID_SIZE];
} godot_rid;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/gdnative.h>

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_rid_new(godot_rid *r_dest);

godot_int GDAPI godot_rid_get_id(const godot_rid *p_self);

void GDAPI godot_rid_new_with_resource(godot_rid *r_dest, const godot_object *p_from);

godot_bool GDAPI godot_rid_operator_equal(const godot_rid *p_self, const godot_rid *p_b);

godot_bool GDAPI godot_rid_operator_less(const godot_rid *p_self, const godot_rid *p_b);

#ifdef __cplusplus
}
#endif

#endif // GODOT_RID_H
