/*************************************************************************/
/*  godot_vector2.cpp                                                    */
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
#include "godot_vector2.h"

#include "math/math_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

void _vector2_api_anchor() {}

godot_vector2 GDAPI godot_vector2_new(const godot_real p_x, const godot_real p_y) {
	godot_vector2 value;
	Vector2 *v = (Vector2 *)&value;
	v->x = p_x;
	v->y = p_y;
	return value;
}

void GDAPI godot_vector2_set_x(godot_vector2 *p_v, const godot_real p_x) {
	Vector2 *v = (Vector2 *)p_v;
	v->x = p_x;
}

void GDAPI godot_vector2_set_y(godot_vector2 *p_v, const godot_real p_y) {
	Vector2 *v = (Vector2 *)p_v;
	v->y = p_y;
}

godot_real GDAPI godot_vector2_get_x(const godot_vector2 *p_v) {
	const Vector2 *v = (Vector2 *)p_v;
	return v->x;
}
godot_real GDAPI godot_vector2_get_y(const godot_vector2 *p_v) {
	const Vector2 *v = (Vector2 *)p_v;
	return v->y;
}

void GDAPI godot_vector2_normalize(godot_vector2 *p_v) {
	Vector2 *v = (Vector2 *)p_v;
	v->normalize();
}

godot_vector2 GDAPI godot_vector2_normalized(const godot_vector2 *p_v) {
	godot_vector2 dest;
	const Vector2 *v = (Vector2 *)p_v;
	Vector2 *d = (Vector2 *)&dest;
	*d = v->normalized();
	return dest;
}

godot_real GDAPI godot_vector2_length(const godot_vector2 *p_v) {
	const Vector2 *v = (Vector2 *)p_v;
	return v->length();
}

godot_real GDAPI godot_vector2_length_squared(const godot_vector2 *p_v) {
	const Vector2 *v = (Vector2 *)p_v;
	return v->length_squared();
}

godot_real GDAPI godot_vector2_distance_to(const godot_vector2 *p_v, const godot_vector2 p_b) {
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	return v->distance_to(*b);
}

godot_real GDAPI godot_vector2_distance_squared_to(const godot_vector2 *p_v, const godot_vector2 p_b) {
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	return v->distance_squared_to(*b);
}

godot_vector2 GDAPI godot_vector2_operator_add(const godot_vector2 *p_v, const godot_vector2 p_b) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	*d = *v + *b;
	return dest;
}

godot_vector2 GDAPI godot_vector2_operator_subtract(const godot_vector2 *p_v, const godot_vector2 p_b) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	*d = *v - *b;
	return dest;
}

godot_vector2 GDAPI godot_vector2_operator_multiply_vector(const godot_vector2 *p_v, const godot_vector2 p_b) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	*d = *v * *b;
	return dest;
}

godot_vector2 GDAPI godot_vector2_operator_multiply_scalar(const godot_vector2 *p_v, const godot_real p_b) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	*d = *v * p_b;
	return dest;
}

godot_vector2 GDAPI godot_vector2_operator_divide_vector(const godot_vector2 *p_v, const godot_vector2 p_b) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	*d = *v / *b;
	return dest;
}

godot_vector2 GDAPI godot_vector2_operator_divide_scalar(const godot_vector2 *p_v, const godot_real p_b) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	*d = *v / p_b;
	return dest;
}

godot_bool GDAPI godot_vector2_operator_equal(const godot_vector2 *p_v, const godot_vector2 p_b) {
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	return *v == *b;
}

godot_bool GDAPI godot_vector2_operator_less(const godot_vector2 *p_v, const godot_vector2 p_b) {
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	return *v < *b;
}

godot_vector2 GDAPI godot_vector2_abs(const godot_vector2 *p_v) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	*d = v->abs();
	return dest;
}

godot_real GDAPI godot_vector2_angle(const godot_vector2 *p_v) {
	const Vector2 *v = (Vector2 *)p_v;
	return v->angle();
}

godot_real GDAPI godot_vector2_angle_to(const godot_vector2 *p_v, const godot_vector2 p_to) {
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *to = (Vector2 *)&p_to;
	return v->angle_to(*to);
}

godot_real GDAPI godot_vector2_angle_to_point(const godot_vector2 *p_v, const godot_vector2 p_to) {
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *to = (Vector2 *)&p_to;
	return v->angle_to_point(*to);
}

godot_vector2 GDAPI godot_vector2_clamped(const godot_vector2 *p_v, const godot_real length) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	*d = v->clamped(length);
	return dest;
}

godot_vector2 GDAPI godot_vector2_cubic_interpolate(
		const godot_vector2 *p_v, const godot_vector2 p_b, const godot_vector2 p_pre_a,
		const godot_vector2 p_post_b, godot_real t) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	const Vector2 *pre_a = (Vector2 *)&p_pre_a;
	const Vector2 *post_b = (Vector2 *)&p_post_b;
	*d = v->cubic_interpolate(*b, *pre_a, *post_b, t);
	return dest;
}

godot_real GDAPI godot_vector2_dot(const godot_vector2 *p_v, const godot_vector2 p_with) {
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *with = (Vector2 *)&p_with;
	return v->dot(*with);
}

godot_vector2 GDAPI godot_vector2_floor(const godot_vector2 *p_v) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	*d = v->floor();
	return dest;
}

godot_real GDAPI godot_vector2_aspect(const godot_vector2 *p_v) {
	const Vector2 *v = (Vector2 *)p_v;
	return v->aspect();
}

godot_vector2 GDAPI godot_vector2_linear_interpolate(
		const godot_vector2 *p_v,
		const godot_vector2 p_b,
		godot_real t) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *b = (Vector2 *)&p_b;
	*d = v->linear_interpolate(*b, t);
	return dest;
}

godot_vector2 GDAPI godot_vector2_reflect(const godot_vector2 *p_v, const godot_vector2 p_vec) {
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *vec = (Vector2 *)&p_vec;
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	*d = v->reflect(*vec);
	return dest;
}

godot_vector2 GDAPI godot_vector2_rotated(const godot_vector2 *p_v, godot_real phi) {
	const Vector2 *v = (Vector2 *)p_v;
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	*d = v->rotated(phi);
	return dest;
}

godot_vector2 GDAPI godot_vector2_slide(const godot_vector2 *p_v, godot_vector2 p_vec) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *vec = (Vector2 *)&p_vec;
	*d = v->slide(*vec);
	return dest;
}

godot_vector2 GDAPI godot_vector2_snapped(const godot_vector2 *p_v, godot_vector2 p_by) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	const Vector2 *by = (Vector2 *)&p_by;
	*d = v->snapped(*by);
	return dest;
}

godot_vector2 GDAPI godot_vector2_tangent(const godot_vector2 *p_v) {
	godot_vector2 dest;
	Vector2 *d = (Vector2 *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	*d = v->tangent();
	return dest;
}

godot_string GDAPI godot_vector2_to_string(const godot_vector2 *p_v) {
	godot_string dest;
	String *d = (String *)&dest;
	const Vector2 *v = (Vector2 *)p_v;
	*d = "(" + *v + ")";
	return dest;
}

#ifdef __cplusplus
}
#endif
