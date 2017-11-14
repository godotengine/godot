/*************************************************************************/
/*  rect2.cpp                                                            */
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
#include "gdnative/rect2.h"

#include "core/math/math_2d.h"
#include "core/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_rect2_new_with_position_and_size(godot_rect2 *r_dest, const godot_vector2 *p_pos, const godot_vector2 *p_size) {
	const Vector2 *position = (const Vector2 *)p_pos;
	const Vector2 *size = (const Vector2 *)p_size;
	Rect2 *dest = (Rect2 *)r_dest;
	*dest = Rect2(*position, *size);
}

void GDAPI godot_rect2_new(godot_rect2 *r_dest, const godot_real p_x, const godot_real p_y, const godot_real p_width, const godot_real p_height) {

	Rect2 *dest = (Rect2 *)r_dest;
	*dest = Rect2(p_x, p_y, p_width, p_height);
}

godot_string GDAPI godot_rect2_as_string(const godot_rect2 *p_self) {
	godot_string ret;
	const Rect2 *self = (const Rect2 *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

godot_real GDAPI godot_rect2_get_area(const godot_rect2 *p_self) {
	const Rect2 *self = (const Rect2 *)p_self;
	return self->get_area();
}

godot_bool GDAPI godot_rect2_intersects(const godot_rect2 *p_self, const godot_rect2 *p_b) {
	const Rect2 *self = (const Rect2 *)p_self;
	const Rect2 *b = (const Rect2 *)p_b;
	return self->intersects(*b);
}

godot_bool GDAPI godot_rect2_encloses(const godot_rect2 *p_self, const godot_rect2 *p_b) {
	const Rect2 *self = (const Rect2 *)p_self;
	const Rect2 *b = (const Rect2 *)p_b;
	return self->encloses(*b);
}

godot_bool GDAPI godot_rect2_has_no_area(const godot_rect2 *p_self) {
	const Rect2 *self = (const Rect2 *)p_self;
	return self->has_no_area();
}

godot_rect2 GDAPI godot_rect2_clip(const godot_rect2 *p_self, const godot_rect2 *p_b) {
	godot_rect2 dest;
	const Rect2 *self = (const Rect2 *)p_self;
	const Rect2 *b = (const Rect2 *)p_b;
	*((Rect2 *)&dest) = self->clip(*b);
	return dest;
}

godot_rect2 GDAPI godot_rect2_merge(const godot_rect2 *p_self, const godot_rect2 *p_b) {
	godot_rect2 dest;
	const Rect2 *self = (const Rect2 *)p_self;
	const Rect2 *b = (const Rect2 *)p_b;
	*((Rect2 *)&dest) = self->merge(*b);
	return dest;
}

godot_bool GDAPI godot_rect2_has_point(const godot_rect2 *p_self, const godot_vector2 *p_point) {
	const Rect2 *self = (const Rect2 *)p_self;
	const Vector2 *point = (const Vector2 *)p_point;
	return self->has_point(*point);
}

godot_rect2 GDAPI godot_rect2_grow(const godot_rect2 *p_self, const godot_real p_by) {
	godot_rect2 dest;
	const Rect2 *self = (const Rect2 *)p_self;

	*((Rect2 *)&dest) = self->grow(p_by);
	return dest;
}

godot_rect2 GDAPI godot_rect2_expand(const godot_rect2 *p_self, const godot_vector2 *p_to) {
	godot_rect2 dest;
	const Rect2 *self = (const Rect2 *)p_self;
	const Vector2 *to = (const Vector2 *)p_to;
	*((Rect2 *)&dest) = self->expand(*to);
	return dest;
}

godot_bool GDAPI godot_rect2_operator_equal(const godot_rect2 *p_self, const godot_rect2 *p_b) {
	const Rect2 *self = (const Rect2 *)p_self;
	const Rect2 *b = (const Rect2 *)p_b;
	return *self == *b;
}

godot_vector2 GDAPI godot_rect2_get_position(const godot_rect2 *p_self) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Rect2 *self = (const Rect2 *)p_self;
	*d = self->get_position();
	return dest;
}

godot_vector2 GDAPI godot_rect2_get_size(const godot_rect2 *p_self) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Rect2 *self = (const Rect2 *)p_self;
	*d = self->get_size();
	return dest;
}

void GDAPI godot_rect2_set_position(godot_rect2 *p_self, const godot_vector2 *p_pos) {
	Rect2 *self = (Rect2 *)p_self;
	const Vector2 *position = (const Vector2 *)p_pos;
	self->set_position(*position);
}

void GDAPI godot_rect2_set_size(godot_rect2 *p_self, const godot_vector2 *p_size) {
	Rect2 *self = (Rect2 *)p_self;
	const Vector2 *size = (const Vector2 *)p_size;
	self->set_size(*size);
}

#ifdef __cplusplus
}
#endif
