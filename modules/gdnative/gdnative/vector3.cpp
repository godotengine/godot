/*************************************************************************/
/*  vector3.cpp                                                          */
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
#include "gdnative/vector3.h"

#include "core/variant.h"
#include "core/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_vector3_new(godot_vector3 *r_dest, const godot_real p_x, const godot_real p_y, const godot_real p_z) {

	Vector3 *dest = (Vector3 *)r_dest;
	*dest = Vector3(p_x, p_y, p_z);
}

godot_string GDAPI godot_vector3_as_string(const godot_vector3 *p_self) {
	godot_string ret;
	const Vector3 *self = (const Vector3 *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

godot_int GDAPI godot_vector3_min_axis(const godot_vector3 *p_self) {
	const Vector3 *self = (const Vector3 *)p_self;
	return self->min_axis();
}

godot_int GDAPI godot_vector3_max_axis(const godot_vector3 *p_self) {
	const Vector3 *self = (const Vector3 *)p_self;
	return self->max_axis();
}

godot_real GDAPI godot_vector3_length(const godot_vector3 *p_self) {
	const Vector3 *self = (const Vector3 *)p_self;
	return self->length();
}

godot_real GDAPI godot_vector3_length_squared(const godot_vector3 *p_self) {
	const Vector3 *self = (const Vector3 *)p_self;
	return self->length_squared();
}

godot_bool GDAPI godot_vector3_is_normalized(const godot_vector3 *p_self) {
	const Vector3 *self = (const Vector3 *)p_self;
	return self->is_normalized();
}

godot_vector3 GDAPI godot_vector3_normalized(const godot_vector3 *p_self) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	*((Vector3 *)&dest) = self->normalized();
	return dest;
}

godot_vector3 GDAPI godot_vector3_inverse(const godot_vector3 *p_self) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	*((Vector3 *)&dest) = self->inverse();
	return dest;
}

godot_vector3 GDAPI godot_vector3_snapped(const godot_vector3 *p_self, const godot_vector3 *p_by) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *snap_axis = (const Vector3 *)p_by;

	*((Vector3 *)&dest) = self->snapped(*snap_axis);
	return dest;
}

godot_vector3 GDAPI godot_vector3_rotated(const godot_vector3 *p_self, const godot_vector3 *p_axis, const godot_real p_phi) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *axis = (const Vector3 *)p_axis;
	*((Vector3 *)&dest) = self->rotated(*axis, p_phi);
	return dest;
}

godot_vector3 GDAPI godot_vector3_linear_interpolate(const godot_vector3 *p_self, const godot_vector3 *p_b, const godot_real p_t) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	*((Vector3 *)&dest) = self->linear_interpolate(*b, p_t);
	return dest;
}

godot_vector3 GDAPI godot_vector3_cubic_interpolate(const godot_vector3 *p_self, const godot_vector3 *p_b, const godot_vector3 *p_pre_a, const godot_vector3 *p_post_b, const godot_real p_t) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	const Vector3 *pre_a = (const Vector3 *)p_pre_a;
	const Vector3 *post_b = (const Vector3 *)p_post_b;
	*((Vector3 *)&dest) = self->cubic_interpolate(*b, *pre_a, *post_b, p_t);
	return dest;
}

godot_real GDAPI godot_vector3_dot(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	return self->dot(*b);
}

godot_vector3 GDAPI godot_vector3_cross(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	*((Vector3 *)&dest) = self->cross(*b);
	return dest;
}

godot_basis GDAPI godot_vector3_outer(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	godot_basis dest;
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	*((Basis *)&dest) = self->outer(*b);
	return dest;
}

godot_basis GDAPI godot_vector3_to_diagonal_matrix(const godot_vector3 *p_self) {
	godot_basis dest;
	const Vector3 *self = (const Vector3 *)p_self;
	*((Basis *)&dest) = self->to_diagonal_matrix();
	return dest;
}

godot_vector3 GDAPI godot_vector3_abs(const godot_vector3 *p_self) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	*((Vector3 *)&dest) = self->abs();
	return dest;
}

godot_vector3 GDAPI godot_vector3_floor(const godot_vector3 *p_self) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	*((Vector3 *)&dest) = self->floor();
	return dest;
}

godot_vector3 GDAPI godot_vector3_ceil(const godot_vector3 *p_self) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	*((Vector3 *)&dest) = self->ceil();
	return dest;
}

godot_real GDAPI godot_vector3_distance_to(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	return self->distance_to(*b);
}

godot_real GDAPI godot_vector3_distance_squared_to(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	return self->distance_squared_to(*b);
}

godot_real GDAPI godot_vector3_angle_to(const godot_vector3 *p_self, const godot_vector3 *p_to) {
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *to = (const Vector3 *)p_to;
	return self->angle_to(*to);
}

godot_vector3 GDAPI godot_vector3_slide(const godot_vector3 *p_self, const godot_vector3 *p_n) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *n = (const Vector3 *)p_n;
	*((Vector3 *)&dest) = self->slide(*n);
	return dest;
}

godot_vector3 GDAPI godot_vector3_bounce(const godot_vector3 *p_self, const godot_vector3 *p_n) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *n = (const Vector3 *)p_n;
	*((Vector3 *)&dest) = self->bounce(*n);
	return dest;
}

godot_vector3 GDAPI godot_vector3_reflect(const godot_vector3 *p_self, const godot_vector3 *p_n) {
	godot_vector3 dest;
	const Vector3 *self = (const Vector3 *)p_self;
	const Vector3 *n = (const Vector3 *)p_n;
	*((Vector3 *)&dest) = self->reflect(*n);
	return dest;
}

godot_vector3 GDAPI godot_vector3_operator_add(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	godot_vector3 raw_dest;
	Vector3 *dest = (Vector3 *)&raw_dest;
	Vector3 *self = (Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	*dest = *self + *b;
	return raw_dest;
}

godot_vector3 GDAPI godot_vector3_operator_subtract(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	godot_vector3 raw_dest;
	Vector3 *dest = (Vector3 *)&raw_dest;
	Vector3 *self = (Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	*dest = *self - *b;
	return raw_dest;
}

godot_vector3 GDAPI godot_vector3_operator_multiply_vector(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	godot_vector3 raw_dest;
	Vector3 *dest = (Vector3 *)&raw_dest;
	Vector3 *self = (Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	*dest = *self * *b;
	return raw_dest;
}

godot_vector3 GDAPI godot_vector3_operator_multiply_scalar(const godot_vector3 *p_self, const godot_real p_b) {
	godot_vector3 raw_dest;
	Vector3 *dest = (Vector3 *)&raw_dest;
	Vector3 *self = (Vector3 *)p_self;
	*dest = *self * p_b;
	return raw_dest;
}

godot_vector3 GDAPI godot_vector3_operator_divide_vector(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	godot_vector3 raw_dest;
	Vector3 *dest = (Vector3 *)&raw_dest;
	Vector3 *self = (Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	*dest = *self / *b;
	return raw_dest;
}

godot_vector3 GDAPI godot_vector3_operator_divide_scalar(const godot_vector3 *p_self, const godot_real p_b) {
	godot_vector3 raw_dest;
	Vector3 *dest = (Vector3 *)&raw_dest;
	Vector3 *self = (Vector3 *)p_self;
	*dest = *self / p_b;
	return raw_dest;
}

godot_bool GDAPI godot_vector3_operator_equal(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	Vector3 *self = (Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	return *self == *b;
}

godot_bool GDAPI godot_vector3_operator_less(const godot_vector3 *p_self, const godot_vector3 *p_b) {
	Vector3 *self = (Vector3 *)p_self;
	const Vector3 *b = (const Vector3 *)p_b;
	return *self < *b;
}

godot_vector3 GDAPI godot_vector3_operator_neg(const godot_vector3 *p_self) {
	godot_vector3 raw_dest;
	Vector3 *dest = (Vector3 *)&raw_dest;
	const Vector3 *self = (const Vector3 *)p_self;
	*dest = -(*self);
	return raw_dest;
}

void GDAPI godot_vector3_set_axis(godot_vector3 *p_self, const godot_vector3_axis p_axis, const godot_real p_val) {
	Vector3 *self = (Vector3 *)p_self;
	self->set_axis(p_axis, p_val);
}

godot_real GDAPI godot_vector3_get_axis(const godot_vector3 *p_self, const godot_vector3_axis p_axis) {
	const Vector3 *self = (const Vector3 *)p_self;
	return self->get_axis(p_axis);
}

#ifdef __cplusplus
}
#endif
