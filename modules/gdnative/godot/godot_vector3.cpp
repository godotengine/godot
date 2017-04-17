/*************************************************************************/
/*  godot_vector3.cpp                                                    */
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
#include "godot_vector3.h"

#include "math/vector3.h"

#ifdef __cplusplus
extern "C" {
#endif

void _vector3_api_anchor() {
}

godot_vector3 GDAPI godot_vector3_new(const godot_real p_x, const godot_real p_y, const godot_real p_z) {
	godot_vector3 value;
	Vector3 *v = (Vector3 *)&value;
	*v = Vector3(p_x, p_y, p_z);
	return value;
}

void GDAPI godot_vector3_set_axis(godot_vector3 *p_v, const godot_int p_axis, const godot_real p_val) {
	Vector3 *v = (Vector3 *)p_v;
	v->set_axis(p_axis, p_val);
}

godot_real GDAPI godot_vector3_get_axis(const godot_vector3 *p_v, const godot_int p_axis) {
	Vector3 *v = (Vector3 *)p_v;
	return v->get_axis(p_axis);
}

godot_int GDAPI godot_vector3_min_axis(const godot_vector3 *p_v) {
	Vector3 *v = (Vector3 *)p_v;
	return v->min_axis();
}

godot_int GDAPI godot_vector3_max_axis(const godot_vector3 *p_v) {
	Vector3 *v = (Vector3 *)p_v;
	return v->max_axis();
}

godot_real GDAPI godot_vector3_length(const godot_vector3 *p_v) {
	Vector3 *v = (Vector3 *)p_v;
	return v->length();
}

godot_real GDAPI godot_vector3_length_squared(const godot_vector3 *p_v) {
	Vector3 *v = (Vector3 *)p_v;
	return v->length_squared();
}

void GDAPI godot_vector3_normalize(godot_vector3 *p_v) {
	Vector3 *v = (Vector3 *)p_v;
	v->normalize();
}

godot_vector3 GDAPI godot_vector3_normalized(const godot_vector3 *p_v) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	Vector3 *v = (Vector3 *)p_v;
	*d = v->normalized();
	return dest;
}

godot_vector3 godot_vector3_inverse(const godot_vector3 *p_v) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	*d = v->inverse();
	return dest;
}

void godot_vector3_zero(godot_vector3 *p_v) {
	Vector3 *v = (Vector3 *)p_v;
	v->zero();
}

void godot_vector3_snap(godot_vector3 *p_v, const godot_real val) {
	Vector3 *v = (Vector3 *)p_v;
	v->snap(val);
}

godot_vector3 godot_vector3_snapped(const godot_vector3 *p_v, const godot_real val) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	*d = v->snapped(val);
	return dest;
}

void godot_vector3_rotate(godot_vector3 *p_v, const godot_vector3 p_axis, const godot_real phi) {
	Vector3 *v = (Vector3 *)p_v;
	const Vector3 *axis = (Vector3 *)&p_axis;
	v->rotate(*axis, phi);
}

godot_vector3 godot_vector3_rotated(const godot_vector3 *p_v, const godot_vector3 p_axis, const godot_real phi) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *axis = (Vector3 *)&p_axis;
	*d = v->rotated(*axis, phi);
	return dest;
}

godot_vector3 godot_vector3_linear_interpolate(const godot_vector3 *p_v, const godot_vector3 p_b, const godot_real t) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *b = (Vector3 *)&p_b;
	*d = v->linear_interpolate(*b, t);
	return dest;
}

godot_vector3 godot_vector3_cubic_interpolate(const godot_vector3 *p_v,
		const godot_vector3 p_b, const godot_vector3 p_pre_a,
		const godot_vector3 p_post_b, const godot_real t) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *b = (Vector3 *)&p_b;
	const Vector3 *pre_a = (Vector3 *)&p_pre_a;
	const Vector3 *post_b = (Vector3 *)&p_post_b;
	*d = v->cubic_interpolate(*b, *pre_a, *post_b, t);
	return dest;
}

godot_vector3 godot_vector3_cubic_interpolaten(const godot_vector3 *p_v,
		const godot_vector3 p_b, const godot_vector3 p_pre_a,
		const godot_vector3 p_post_b, const godot_real t) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *b = (Vector3 *)&p_b;
	const Vector3 *pre_a = (Vector3 *)&p_pre_a;
	const Vector3 *post_b = (Vector3 *)&p_post_b;
	*d = v->cubic_interpolaten(*b, *pre_a, *post_b, t);
	return dest;
}

godot_vector3 godot_vector3_cross(const godot_vector3 *p_v, const godot_vector3 p_b) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *b = (Vector3 *)&p_b;
	*d = v->cross(*b);
	return dest;
}

godot_real godot_vector3_dot(const godot_vector3 *p_v, const godot_vector3 p_b) {
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *b = (Vector3 *)&p_b;
	return v->dot(*b);
}

godot_basis godot_vector3_outer(const godot_vector3 *p_v, const godot_vector3 p_b) {
	godot_basis dest;
	Basis *d = (Basis *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *b = (Vector3 *)&p_b;
	*d = v->outer(*b);
	return dest;
}

godot_basis godot_vector3_to_diagonal_matrix(const godot_vector3 *p_v) {
	godot_basis dest;
	Basis *d = (Basis *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	*d = v->to_diagonal_matrix();
	return dest;
}

godot_vector3 godot_vector3_abs(const godot_vector3 *p_v) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	*d = v->abs();
	return dest;
}

godot_vector3 godot_vector3_floor(const godot_vector3 *p_v) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	*d = v->floor();
	return dest;
}

godot_vector3 godot_vector3_ceil(const godot_vector3 *p_v) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	*d = v->ceil();
	return dest;
}

godot_real GDAPI godot_vector3_distance_to(const godot_vector3 *p_v, const godot_vector3 p_b) {
	Vector3 *v = (Vector3 *)p_v;
	Vector3 *b = (Vector3 *)&p_b;
	return v->distance_to(*b);
}

godot_real GDAPI godot_vector3_distance_squared_to(const godot_vector3 *p_v, const godot_vector3 p_b) {
	Vector3 *v = (Vector3 *)p_v;
	Vector3 *b = (Vector3 *)&p_b;
	return v->distance_squared_to(*b);
}

godot_real GDAPI godot_vector3_angle_to(const godot_vector3 *p_v, const godot_vector3 p_b) {
	Vector3 *v = (Vector3 *)p_v;
	Vector3 *b = (Vector3 *)&p_b;
	return v->angle_to(*b);
}

godot_vector3 godot_vector3_slide(const godot_vector3 *p_v, const godot_vector3 p_vec) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *vec = (Vector3 *)&p_vec;
	*d = v->slide(*vec);
	return dest;
}

godot_vector3 godot_vector3_bounce(const godot_vector3 *p_v, const godot_vector3 p_vec) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *vec = (Vector3 *)&p_vec;
	*d = v->bounce(*vec);
	return dest;
}

godot_vector3 godot_vector3_reflect(const godot_vector3 *p_v, const godot_vector3 p_vec) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	const Vector3 *vec = (Vector3 *)&p_vec;
	*d = v->reflect(*vec);
	return dest;
}

godot_vector3 GDAPI godot_vector3_operator_add(const godot_vector3 *p_v, const godot_vector3 p_b) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	Vector3 *v = (Vector3 *)p_v;
	Vector3 *b = (Vector3 *)&p_b;
	*d = *v + *b;
	return dest;
}

godot_vector3 GDAPI godot_vector3_operator_subtract(const godot_vector3 *p_v, const godot_vector3 p_b) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	Vector3 *v = (Vector3 *)p_v;
	Vector3 *b = (Vector3 *)&p_b;
	*d = *v - *b;
	return dest;
}

godot_vector3 GDAPI godot_vector3_operator_multiply_vector(const godot_vector3 *p_v, const godot_vector3 p_b) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	Vector3 *v = (Vector3 *)p_v;
	Vector3 *b = (Vector3 *)&p_b;
	*d = *v * *b;
	return dest;
}

godot_vector3 GDAPI godot_vector3_operator_multiply_scalar(const godot_vector3 *p_v, const godot_real p_b) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	Vector3 *v = (Vector3 *)p_v;
	*d = *v * p_b;
	return dest;
}

godot_vector3 GDAPI godot_vector3_operator_divide_vector(const godot_vector3 *p_v, const godot_vector3 p_b) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	Vector3 *v = (Vector3 *)p_v;
	Vector3 *b = (Vector3 *)&p_b;
	*d = *v / *b;
	return dest;
}

godot_vector3 GDAPI godot_vector3_operator_divide_scalar(const godot_vector3 *p_v, const godot_real p_b) {
	godot_vector3 dest;
	Vector3 *d = (Vector3 *)&dest;
	Vector3 *v = (Vector3 *)p_v;
	*d = *v / p_b;
	return dest;
}

godot_bool GDAPI godot_vector3_operator_equal(const godot_vector3 *p_v, const godot_vector3 p_b) {
	Vector3 *v = (Vector3 *)p_v;
	Vector3 *b = (Vector3 *)&p_b;
	return *v == *b;
}

godot_bool GDAPI godot_vector3_operator_less(const godot_vector3 *p_v, const godot_vector3 p_b) {
	Vector3 *v = (Vector3 *)p_v;
	Vector3 *b = (Vector3 *)&p_b;
	return *v < *b;
}

godot_string GDAPI godot_vector3_to_string(const godot_vector3 *p_v) {
	godot_string dest;
	String *d = (String *)&dest;
	const Vector3 *v = (Vector3 *)p_v;
	*d = "(" + *v + ")";
	return dest;
}

#ifdef __cplusplus
}
#endif
