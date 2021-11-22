/*************************************************************************/
/*  callable.h                                                           */
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

#ifndef GODOT_CALLABLE_H
#define GODOT_CALLABLE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Alignment hardcoded in `core/variant/callable.h`.
#define GODOT_CALLABLE_SIZE (16)

#ifndef GODOT_CORE_API_GODOT_CALLABLE_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_CALLABLE_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_CALLABLE_SIZE];
} godot_callable;
#endif

#include <gdnative/gdnative.h>

void GDAPI godot_callable_new(godot_callable *p_self);
void GDAPI godot_callable_new_copy(godot_callable *r_dest, const godot_callable *p_src);
void GDAPI godot_callable_destroy(godot_callable *p_self);

#ifdef __cplusplus
}
#endif

#endif
