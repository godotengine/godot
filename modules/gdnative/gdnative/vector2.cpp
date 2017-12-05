/*************************************************************************/
/*  vector2.cpp                                                          */
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
#include "gdnative/vector2.h"

#include "core/math/math_2d.h"
#include "core/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_vector2_new(godot_vector2 *r_dest, const godot_real p_x, const godot_real p_y) {

	Vector2 *dest = (Vector2 *)r_dest;
	*dest = Vector2(p_x, p_y);
}

godot_string GDAPI godot_vector2_as_string(const godot_vector2 *p_self) {
	godot_string ret;
	const Vector2 *self = (const Vector2 *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

godot_vector2 GDAPI godot_vector2_normalized(const godot_vector2 *p_self) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	*((Vector2 *)&dest) = self->normalized();
	return dest;
}

godot_real GDAPI godot_vector2_length(const godot_vector2 *p_self) {
	const Vector2 *self = (const Vector2 *)p_self;
	return self->length();
}

godot_real GDAPI godot_vector2_angle(const godot_vector2 *p_self) {
	const Vector2 *self = (const Vector2 *)p_self;
	return self->angle();
}

godot_real GDAPI godot_vector2_length_squared(const godot_vector2 *p_self) {
	const Vector2 *self = (const Vector2 *)p_self;
	return self->length_squared();
}

godot_bool GDAPI godot_vector2_is_normalized(const godot_vector2 *p_self) {
	const Vector2 *self = (const Vector2 *)p_self;
	return self->is_normalized();
}

godot_real GDAPI godot_vector2_distance_to(const godot_vector2 *p_self, const godot_vector2 *p_to) {
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *to = (const Vector2 *)p_to;
	return self->distance_to(*to);
}

godot_real GDAPI godot_vector2_distance_squared_to(const godot_vector2 *p_self, const godot_vector2 *p_to) {
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *to = (const Vector2 *)p_to;
	return self->distance_squared_to(*to);
}

godot_real GDAPI godot_vector2_angle_to(const godot_vector2 *p_self, const godot_vector2 *p_to) {
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *to = (const Vector2 *)p_to;
	return self->angle_to(*to);
}

godot_real GDAPI godot_vector2_angle_to_point(const godot_vector2 *p_self, const godot_vector2 *p_to) {
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *to = (const Vector2 *)p_to;
	return self->angle_to_point(*to);
}

godot_vector2 GDAPI godot_vector2_linear_interpolate(const godot_vector2 *p_self, const godot_vector2 *p_b, const godot_real p_t) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *b = (const Vector2 *)p_b;
	*((Vector2 *)&dest) = self->linear_interpolate(*b, p_t);
	return dest;
}

godot_vector2 GDAPI godot_vector2_cubic_interpolate(const godot_vector2 *p_self, const godot_vector2 *p_b, const godot_vector2 *p_pre_a, const godot_vector2 *p_post_b, const godot_real p_t) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *b = (const Vector2 *)p_b;
	const Vector2 *pre_a = (const Vector2 *)p_pre_a;
	const Vector2 *post_b = (const Vector2 *)p_post_b;
	*((Vector2 *)&dest) = self->cubic_interpolate(*b, *pre_a, *post_b, p_t);
	return dest;
}

godot_vector2 GDAPI godot_vector2_rotated(const godot_vector2 *p_self, const godot_real p_phi) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;

	*((Vector2 *)&dest) = self->rotated(p_phi);
	return dest;
}

godot_vector2 GDAPI godot_vector2_tangent(const godot_vector2 *p_self) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	*((Vector2 *)&dest) = self->tangent();
	return dest;
}

godot_vector2 GDAPI godot_vector2_floor(const godot_vector2 *p_self) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	*((Vector2 *)&dest) = self->floor();
	return dest;
}

godot_vector2 GDAPI godot_vector2_snapped(const godot_vector2 *p_self, const godot_vector2 *p_by) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *by = (const Vector2 *)p_by;
	*((Vector2 *)&dest) = self->snapped(*by);
	return dest;
}

godot_real GDAPI godot_vector2_aspect(const godot_vector2 *p_self) {
	const Vector2 *self = (const Vector2 *)p_self;
	return self->aspect();
}

godot_real GDAPI godot_vector2_dot(const godot_vector2 *p_self, const godot_vector2 *p_with) {
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *with = (const Vector2 *)p_with;
	return self->dot(*with);
}

godot_vector2 GDAPI godot_vector2_slide(const godot_vector2 *p_self, const godot_vector2 *p_n) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *n = (const Vector2 *)p_n;
	*((Vector2 *)&dest) = self->slide(*n);
	return dest;
}

godot_vector2 GDAPI godot_vector2_bounce(const godot_vector2 *p_self, const godot_vector2 *p_n) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *n = (const Vector2 *)p_n;
	*((Vector2 *)&dest) = self->bounce(*n);
	return dest;
}

godot_vector2 GDAPI godot_vector2_reflect(const godot_vector2 *p_self, const godot_vector2 *p_n) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *n = (const Vector2 *)p_n;
	*((Vector2 *)&dest) = self->reflect(*n);
	return dest;
}

godot_vector2 GDAPI godot_vector2_abs(const godot_vector2 *p_self) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;
	*((Vector2 *)&dest) = self->abs();
	return dest;
}

godot_vector2 GDAPI godot_vector2_clamped(const godot_vector2 *p_self, const godot_real p_length) {
	godot_vector2 dest;
	const Vector2 *self = (const Vector2 *)p_self;

	*((Vector2 *)&dest) = self->clamped(p_length);
	return dest;
}

godot_vector2 GDAPI godot_vector2_operator_add(const godot_vector2 *p_self, const godot_vector2 *p_b) {
	godot_vector2 raw_dest;
	Vector2 *dest = (Vector2 *)&raw_dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *b = (const Vector2 *)p_b;
	*dest = *self + *b;
	return raw_dest;
}

godot_vector2 GDAPI godot_vector2_operator_subtract(const godot_vector2 *p_self, const godot_vector2 *p_b) {
	godot_vector2 raw_dest;
	Vector2 *dest = (Vector2 *)&raw_dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *b = (const Vector2 *)p_b;
	*dest = *self - *b;
	return raw_dest;
}

godot_vector2 GDAPI godot_vector2_operator_multiply_vector(const godot_vector2 *p_self, const godot_vector2 *p_b) {
	godot_vector2 raw_dest;
	Vector2 *dest = (Vector2 *)&raw_dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *b = (const Vector2 *)p_b;
	*dest = *self * *b;
	return raw_dest;
}

godot_vector2 GDAPI godot_vector2_operator_multiply_scalar(const godot_vector2 *p_self, const godot_real p_b) {
	godot_vector2 raw_dest;
	Vector2 *dest = (Vector2 *)&raw_dest;
	const Vector2 *self = (const Vector2 *)p_self;
	*dest = *self * p_b;
	return raw_dest;
}

godot_vector2 GDAPI godot_vector2_operator_divide_vector(const godot_vector2 *p_self, const godot_vector2 *p_b) {
	godot_vector2 raw_dest;
	Vector2 *dest = (Vector2 *)&raw_dest;
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *b = (const Vector2 *)p_b;
	*dest = *self / *b;
	return raw_dest;
}

godot_vector2 GDAPI godot_vector2_operator_divide_scalar(const godot_vector2 *p_self, const godot_real p_b) {
	godot_vector2 raw_dest;
	Vector2 *dest = (Vector2 *)&raw_dest;
	const Vector2 *self = (const Vector2 *)p_self;
	*dest = *self / p_b;
	return raw_dest;
}

godot_bool GDAPI godot_vector2_operator_equal(const godot_vector2 *p_self, const godot_vector2 *p_b) {
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *b = (const Vector2 *)p_b;
	return *self == *b;
}

godot_bool GDAPI godot_vector2_operator_less(const godot_vector2 *p_self, const godot_vector2 *p_b) {
	const Vector2 *self = (const Vector2 *)p_self;
	const Vector2 *b = (const Vector2 *)p_b;
	return *self < *b;
}

godot_vector2 GDAPI godot_vector2_operator_neg(const godot_vector2 *p_self) {
	godot_vector2 raw_dest;
	Vector2 *dest = (Vector2 *)&raw_dest;
	const Vector2 *self = (const Vector2 *)p_self;
	*dest = -(*self);
	return raw_dest;
}

void GDAPI godot_vector2_set_x(godot_vector2 *p_self, const godot_real p_x) {
	Vector2 *self = (Vector2 *)p_self;
	self->x = p_x;
}

void GDAPI godot_vector2_set_y(godot_vector2 *p_self, const godot_real p_y) {
	Vector2 *self = (Vector2 *)p_self;
	self->y = p_y;
}

godot_real GDAPI godot_vector2_get_x(const godot_vector2 *p_self) {
	const Vector2 *self = (const Vector2 *)p_self;
	return self->x;
}

godot_real GDAPI godot_vector2_get_y(const godot_vector2 *p_self) {
	const Vector2 *self = (const Vector2 *)p_self;
	return self->y;
}

#ifdef __cplusplus
}
#endif
