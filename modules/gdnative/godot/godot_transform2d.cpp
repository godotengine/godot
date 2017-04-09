/*************************************************************************/
/*  godot_transform2d.cpp                                                */
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
#include "godot_transform2d.h"

#include "../godot.h"

#include "math/math_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

void _transform2d_api_anchor() {
}

void GDAPI godot_transform2d_new_identity(godot_transform2d *p_t) {
	Transform2D *t = (Transform2D *)p_t;
	*t = Transform2D();
}

void GDAPI godot_transform2d_new_elements(godot_transform2d *p_t, const godot_vector2 *p_a, const godot_vector2 *p_b, const godot_vector2 *p_c) {
	Transform2D *t = (Transform2D *)p_t;
	Vector2 *a = (Vector2 *)p_a;
	Vector2 *b = (Vector2 *)p_b;
	Vector2 *c = (Vector2 *)p_c;
	*t = Transform2D(a->x, a->y, b->x, b->y, c->x, c->y);
}

void GDAPI godot_transform2d_new(godot_transform2d *p_t, const godot_real p_rot, const godot_vector2 *p_pos) {
	Transform2D *t = (Transform2D *)p_t;
	Vector2 *pos = (Vector2 *)p_pos;
	*t = Transform2D(p_rot, *pos);
}

godot_vector2 const GDAPI *godot_transform2d_const_index(const godot_transform2d *p_t, const godot_int p_idx) {
	const Transform2D *t = (const Transform2D *)p_t;
	const Vector2 *e = &t->operator[](p_idx);
	return (godot_vector2 const *)e;
}

godot_vector2 GDAPI *godot_transform2d_index(godot_transform2d *p_t, const godot_int p_idx) {
	Transform2D *t = (Transform2D *)p_t;
	Vector2 *e = &t->operator[](p_idx);
	return (godot_vector2 *)e;
}

godot_vector2 GDAPI godot_transform2d_get_axis(const godot_transform2d *p_t, const godot_int p_axis) {
	return *godot_transform2d_const_index(p_t, p_axis);
}

void GDAPI godot_transform2d_set_axis(godot_transform2d *p_t, const godot_int p_axis, const godot_vector2 *p_vec) {
	godot_vector2 *origin_v = godot_transform2d_index(p_t, p_axis);
	*origin_v = *p_vec;
}

// @Incomplete
// See header file

#ifdef __cplusplus
}
#endif
