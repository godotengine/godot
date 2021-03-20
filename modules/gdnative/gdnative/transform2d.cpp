/*************************************************************************/
/*  transform2d.cpp                                                      */
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

#include "gdnative/transform2d.h"

#include "core/math/transform_2d.h"

static_assert(sizeof(godot_transform2d) == sizeof(Transform2D), "Transform2D size mismatch");

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_transform2d_new(godot_transform2d *p_self) {
	memnew_placement(p_self, Transform2D);
}

void GDAPI godot_transform2d_new_copy(godot_transform2d *r_dest, const godot_transform2d *p_src) {
	memnew_placement(r_dest, Transform2D(*(Transform2D *)p_src));
}

godot_vector2 GDAPI *godot_transform2d_operator_index(godot_transform2d *p_self, godot_int p_index) {
	Transform2D *self = (Transform2D *)p_self;
	return (godot_vector2 *)&self->operator[](p_index);
}

const godot_vector2 GDAPI *godot_transform2d_operator_index_const(const godot_transform2d *p_self, godot_int p_index) {
	const Transform2D *self = (const Transform2D *)p_self;
	return (const godot_vector2 *)&self->operator[](p_index);
}

#ifdef __cplusplus
}
#endif
