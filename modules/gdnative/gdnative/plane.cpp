/**************************************************************************/
/*  plane.cpp                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "gdnative/plane.h"

#include "core/math/plane.h"
#include "core/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(sizeof(godot_plane) == sizeof(Plane), "Plane size mismatch");

void GDAPI godot_plane_new_with_reals(godot_plane *r_dest, const godot_real p_a, const godot_real p_b, const godot_real p_c, const godot_real p_d) {
	Plane *dest = (Plane *)r_dest;
	*dest = Plane(p_a, p_b, p_c, p_d);
}

void GDAPI godot_plane_new_with_vectors(godot_plane *r_dest, const godot_vector3 *p_v1, const godot_vector3 *p_v2, const godot_vector3 *p_v3) {
	const Vector3 *v1 = (const Vector3 *)p_v1;
	const Vector3 *v2 = (const Vector3 *)p_v2;
	const Vector3 *v3 = (const Vector3 *)p_v3;
	Plane *dest = (Plane *)r_dest;
	*dest = Plane(*v1, *v2, *v3);
}

void GDAPI godot_plane_new_with_normal(godot_plane *r_dest, const godot_vector3 *p_normal, const godot_real p_d) {
	const Vector3 *normal = (const Vector3 *)p_normal;
	Plane *dest = (Plane *)r_dest;
	*dest = Plane(*normal, p_d);
}

godot_string GDAPI godot_plane_as_string(const godot_plane *p_self) {
	godot_string ret;
	const Plane *self = (const Plane *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

godot_plane GDAPI godot_plane_normalized(const godot_plane *p_self) {
	godot_plane dest;
	const Plane *self = (const Plane *)p_self;
	*((Plane *)&dest) = self->normalized();
	return dest;
}

godot_vector3 GDAPI godot_plane_center(const godot_plane *p_self) {
	godot_vector3 dest;
	const Plane *self = (const Plane *)p_self;
	*((Vector3 *)&dest) = self->center();
	return dest;
}

godot_vector3 GDAPI godot_plane_get_any_point(const godot_plane *p_self) {
	godot_vector3 dest;
	const Plane *self = (const Plane *)p_self;
	*((Vector3 *)&dest) = self->get_any_point();
	return dest;
}

godot_bool GDAPI godot_plane_is_point_over(const godot_plane *p_self, const godot_vector3 *p_point) {
	const Plane *self = (const Plane *)p_self;
	const Vector3 *point = (const Vector3 *)p_point;
	return self->is_point_over(*point);
}

godot_real GDAPI godot_plane_distance_to(const godot_plane *p_self, const godot_vector3 *p_point) {
	const Plane *self = (const Plane *)p_self;
	const Vector3 *point = (const Vector3 *)p_point;
	return self->distance_to(*point);
}

godot_bool GDAPI godot_plane_has_point(const godot_plane *p_self, const godot_vector3 *p_point, const godot_real p_epsilon) {
	const Plane *self = (const Plane *)p_self;
	const Vector3 *point = (const Vector3 *)p_point;
	return self->has_point(*point, p_epsilon);
}

godot_vector3 GDAPI godot_plane_project(const godot_plane *p_self, const godot_vector3 *p_point) {
	godot_vector3 dest;
	const Plane *self = (const Plane *)p_self;
	const Vector3 *point = (const Vector3 *)p_point;
	*((Vector3 *)&dest) = self->project(*point);
	return dest;
}

godot_bool GDAPI godot_plane_intersect_3(const godot_plane *p_self, godot_vector3 *r_dest, const godot_plane *p_b, const godot_plane *p_c) {
	const Plane *self = (const Plane *)p_self;
	const Plane *b = (const Plane *)p_b;
	const Plane *c = (const Plane *)p_c;
	Vector3 *dest = (Vector3 *)r_dest;
	return self->intersect_3(*b, *c, dest);
}

godot_bool GDAPI godot_plane_intersects_ray(const godot_plane *p_self, godot_vector3 *r_dest, const godot_vector3 *p_from, const godot_vector3 *p_dir) {
	const Plane *self = (const Plane *)p_self;
	const Vector3 *from = (const Vector3 *)p_from;
	const Vector3 *dir = (const Vector3 *)p_dir;
	Vector3 *dest = (Vector3 *)r_dest;
	return self->intersects_ray(*from, *dir, dest);
}

godot_bool GDAPI godot_plane_intersects_segment(const godot_plane *p_self, godot_vector3 *r_dest, const godot_vector3 *p_begin, const godot_vector3 *p_end) {
	const Plane *self = (const Plane *)p_self;
	const Vector3 *begin = (const Vector3 *)p_begin;
	const Vector3 *end = (const Vector3 *)p_end;
	Vector3 *dest = (Vector3 *)r_dest;
	return self->intersects_segment(*begin, *end, dest);
}

godot_plane GDAPI godot_plane_operator_neg(const godot_plane *p_self) {
	godot_plane raw_dest;
	Plane *dest = (Plane *)&raw_dest;
	const Plane *self = (const Plane *)p_self;
	*dest = -(*self);
	return raw_dest;
}

godot_bool GDAPI godot_plane_operator_equal(const godot_plane *p_self, const godot_plane *p_b) {
	const Plane *self = (const Plane *)p_self;
	const Plane *b = (const Plane *)p_b;
	return *self == *b;
}

void GDAPI godot_plane_set_normal(godot_plane *p_self, const godot_vector3 *p_normal) {
	Plane *self = (Plane *)p_self;
	const Vector3 *normal = (const Vector3 *)p_normal;
	self->set_normal(*normal);
}

godot_vector3 GDAPI godot_plane_get_normal(const godot_plane *p_self) {
	const Plane *self = (const Plane *)p_self;
	const Vector3 normal = self->get_normal();
	godot_vector3 *v3 = (godot_vector3 *)&normal;
	return *v3;
}

godot_real GDAPI godot_plane_get_d(const godot_plane *p_self) {
	const Plane *self = (const Plane *)p_self;
	return self->d;
}

void GDAPI godot_plane_set_d(godot_plane *p_self, const godot_real p_d) {
	Plane *self = (Plane *)p_self;
	self->d = p_d;
}

#ifdef __cplusplus
}
#endif
