/*************************************************************************/
/*  array.cpp                                                            */
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

#include "gdnative/array.h"

#include "core/os/memory.h"
#include "core/variant/array.h"

static_assert(sizeof(godot_array) == sizeof(Array), "Array size mismatch");

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_array_new(godot_array *p_self) {
	memnew_placement(p_self, Array);
}

void GDAPI godot_array_new_copy(godot_array *r_dest, const godot_array *p_src) {
	memnew_placement(r_dest, Array(*(Array *)p_src));
}

void GDAPI godot_array_destroy(godot_array *p_self) {
	((Array *)p_self)->~Array();
}

godot_variant GDAPI *godot_array_operator_index(godot_array *p_self, godot_int p_index) {
	Array *self = (Array *)p_self;
	return (godot_variant *)&self->operator[](p_index);
}

const godot_variant GDAPI *godot_array_operator_index_const(const godot_array *p_self, godot_int p_index) {
	const Array *self = (const Array *)p_self;
	return (const godot_variant *)&self->operator[](p_index);
}

#ifdef __cplusplus
}
#endif
