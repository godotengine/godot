/*************************************************************************/
/*  rect3.cpp                                                            */
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
#include "gdnative/rect3.h"

#include "core/math/rect3.h"
#include "core/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

void _rect3_api_anchor() {}

void GDAPI godot_rect3_new(godot_rect3 *r_dest, const godot_vector3 *p_pos, const godot_vector3 *p_size) {
	const Vector3 *pos = (const Vector3 *)p_pos;
	const Vector3 *size = (const Vector3 *)p_size;
	Rect3 *dest = (Rect3 *)r_dest;
	*dest = Rect3(*pos, *size);
}

godot_vector3 GDAPI godot_rect3_get_position(const godot_rect3 *p_self) {
	godot_vector3 raw_ret;
	const Rect3 *self = (const Rect3 *)p_self;
	Vector3 *ret = (Vector3 *)&raw_ret;
	*ret = self->position;
	return raw_ret;
}

void GDAPI godot_rect3_set_position(const godot_rect3 *p_self, const godot_vector3 *p_v) {
	Rect3 *self = (Rect3 *)p_self;
	const Vector3 *v = (const Vector3 *)p_v;
	self->position = *v;
}

godot_vector3 GDAPI godot_rect3_get_size(const godot_rect3 *p_self) {
	godot_vector3 raw_ret;
	const Rect3 *self = (const Rect3 *)p_self;
	Vector3 *ret = (Vector3 *)&raw_ret;
	*ret = self->size;
	return raw_ret;
}

void GDAPI godot_rect3_set_size(const godot_rect3 *p_self, const godot_vector3 *p_v) {
	Rect3 *self = (Rect3 *)p_self;
	const Vector3 *v = (const Vector3 *)p_v;
	self->size = *v;
}

godot_string GDAPI godot_rect3_as_string(const godot_rect3 *p_self) {
	godot_string ret;
	const Rect3 *self = (const Rect3 *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

godot_real GDAPI godot_rect3_get_area(const godot_rect3 *p_self) {
	const Rect3 *self = (const Rect3 *)p_self;
	return self->get_area();
}

godot_bool GDAPI godot_rect3_has_no_area(const godot_rect3 *p_self) {
	const Rect3 *self = (const Rect3 *)p_self;
	return self->has_no_area();
}

godot_bool GDAPI godot_rect3_has_no_surface(const godot_rect3 *p_self) {
	const Rect3 *self = (const Rect3 *)p_self;
	return self->has_no_surface();
}

godot_bool GDAPI godot_rect3_intersects(const godot_rect3 *p_self, const godot_rect3 *p_with) {
	const Rect3 *self = (const Rect3 *)p_self;
	const Rect3 *with = (const Rect3 *)p_with;
	return self->intersects(*with);
}

godot_bool GDAPI godot_rect3_encloses(const godot_rect3 *p_self, const godot_rect3 *p_with) {
	const Rect3 *self = (const Rect3 *)p_self;
	const Rect3 *with = (const Rect3 *)p_with;
	return self->encloses(*with);
}

godot_rect3 GDAPI godot_rect3_merge(const godot_rect3 *p_self, const godot_rect3 *p_with) {
	godot_rect3 dest;
	const Rect3 *self = (const Rect3 *)p_self;
	const Rect3 *with = (const Rect3 *)p_with;
	*((Rect3 *)&dest) = self->merge(*with);
	return dest;
}

godot_rect3 GDAPI godot_rect3_intersection(const godot_rect3 *p_self, const godot_rect3 *p_with) {
	godot_rect3 dest;
	const Rect3 *self = (const Rect3 *)p_self;
	const Rect3 *with = (const Rect3 *)p_with;
	*((Rect3 *)&dest) = self->intersection(*with);
	return dest;
}

godot_bool GDAPI godot_rect3_intersects_plane(const godot_rect3 *p_self, const godot_plane *p_plane) {
	const Rect3 *self = (const Rect3 *)p_self;
	const Plane *plane = (const Plane *)p_plane;
	return self->intersects_plane(*plane);
}

godot_bool GDAPI godot_rect3_intersects_segment(const godot_rect3 *p_self, const godot_vector3 *p_from, const godot_vector3 *p_to) {
	const Rect3 *self = (const Rect3 *)p_self;
	const Vector3 *from = (const Vector3 *)p_from;
	const Vector3 *to = (const Vector3 *)p_to;
	return self->intersects_segment(*from, *to);
}

godot_bool GDAPI godot_rect3_has_point(const godot_rect3 *p_self, const godot_vector3 *p_point) {
	const Rect3 *self = (const Rect3 *)p_self;
	const Vector3 *point = (const Vector3 *)p_point;
	return self->has_point(*point);
}

godot_vector3 GDAPI godot_rect3_get_support(const godot_rect3 *p_self, const godot_vector3 *p_dir) {
	godot_vector3 dest;
	const Rect3 *self = (const Rect3 *)p_self;
	const Vector3 *dir = (const Vector3 *)p_dir;
	*((Vector3 *)&dest) = self->get_support(*dir);
	return dest;
}

godot_vector3 GDAPI godot_rect3_get_longest_axis(const godot_rect3 *p_self) {
	godot_vector3 dest;
	const Rect3 *self = (const Rect3 *)p_self;
	*((Vector3 *)&dest) = self->get_longest_axis();
	return dest;
}

godot_int GDAPI godot_rect3_get_longest_axis_index(const godot_rect3 *p_self) {
	const Rect3 *self = (const Rect3 *)p_self;
	return self->get_longest_axis_index();
}

godot_real GDAPI godot_rect3_get_longest_axis_size(const godot_rect3 *p_self) {
	const Rect3 *self = (const Rect3 *)p_self;
	return self->get_longest_axis_size();
}

godot_vector3 GDAPI godot_rect3_get_shortest_axis(const godot_rect3 *p_self) {
	godot_vector3 dest;
	const Rect3 *self = (const Rect3 *)p_self;
	*((Vector3 *)&dest) = self->get_shortest_axis();
	return dest;
}

godot_int GDAPI godot_rect3_get_shortest_axis_index(const godot_rect3 *p_self) {
	const Rect3 *self = (const Rect3 *)p_self;
	return self->get_shortest_axis_index();
}

godot_real GDAPI godot_rect3_get_shortest_axis_size(const godot_rect3 *p_self) {
	const Rect3 *self = (const Rect3 *)p_self;
	return self->get_shortest_axis_size();
}

godot_rect3 GDAPI godot_rect3_expand(const godot_rect3 *p_self, const godot_vector3 *p_to_point) {
	godot_rect3 dest;
	const Rect3 *self = (const Rect3 *)p_self;
	const Vector3 *to_point = (const Vector3 *)p_to_point;
	*((Rect3 *)&dest) = self->expand(*to_point);
	return dest;
}

godot_rect3 GDAPI godot_rect3_grow(const godot_rect3 *p_self, const godot_real p_by) {
	godot_rect3 dest;
	const Rect3 *self = (const Rect3 *)p_self;

	*((Rect3 *)&dest) = self->grow(p_by);
	return dest;
}

godot_vector3 GDAPI godot_rect3_get_endpoint(const godot_rect3 *p_self, const godot_int p_idx) {
	godot_vector3 dest;
	const Rect3 *self = (const Rect3 *)p_self;

	*((Vector3 *)&dest) = self->get_endpoint(p_idx);
	return dest;
}

godot_bool GDAPI godot_rect3_operator_equal(const godot_rect3 *p_self, const godot_rect3 *p_b) {
	const Rect3 *self = (const Rect3 *)p_self;
	const Rect3 *b = (const Rect3 *)p_b;
	return *self == *b;
}

#ifdef __cplusplus
}
#endif
