/*************************************************************************/
/*  vector3.cpp                                                          */
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

#include "gdnative/vector3.h"

#include "core/math/vector3.h"

static_assert(sizeof(godot_vector3) == sizeof(Vector3), "Vector3 size mismatch");
static_assert(sizeof(godot_vector3i) == sizeof(Vector3i), "Vector3i size mismatch");

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_vector3_new(godot_vector3 *p_self) {
	memnew_placement(p_self, Vector3);
}

void GDAPI godot_vector3_new_copy(godot_vector3 *r_dest, const godot_vector3 *p_src) {
	memnew_placement(r_dest, Vector3(*(Vector3 *)p_src));
}

void GDAPI godot_vector3i_new(godot_vector3i *p_self) {
	memnew_placement(p_self, Vector3i);
}

void GDAPI godot_vector3i_new_copy(godot_vector3i *r_dest, const godot_vector3i *p_src) {
	memnew_placement(r_dest, Vector3i(*(Vector3i *)p_src));
}

godot_real_t GDAPI *godot_vector3_operator_index(godot_vector3 *p_self, godot_int p_index) {
	Vector3 *self = (Vector3 *)p_self;
	return (godot_real_t *)&self->operator[](p_index);
}

const godot_real_t GDAPI *godot_vector3_operator_index_const(const godot_vector3 *p_self, godot_int p_index) {
	const Vector3 *self = (const Vector3 *)p_self;
	return (const godot_real_t *)&self->operator[](p_index);
}

int32_t GDAPI *godot_vector3i_operator_index(godot_vector3i *p_self, godot_int p_index) {
	Vector3i *self = (Vector3i *)p_self;
	return (int32_t *)&self->operator[](p_index);
}

const int32_t GDAPI *godot_vector3i_operator_index_const(const godot_vector3i *p_self, godot_int p_index) {
	const Vector3i *self = (const Vector3i *)p_self;
	return (const int32_t *)&self->operator[](p_index);
}

#ifdef __cplusplus
}
#endif
