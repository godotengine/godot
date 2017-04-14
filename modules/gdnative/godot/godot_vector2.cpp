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

void GDAPI godot_vector2_new(godot_vector2 *p_v, godot_real p_x,
		godot_real p_y) {
	Vector2 *v = (Vector2 *)p_v;
	v->x = p_x;
	v->y = p_y;
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
	Vector2 *v = (Vector2 *)p_v;
	return v->x;
}
godot_real GDAPI godot_vector2_get_y(const godot_vector2 *p_v) {
	Vector2 *v = (Vector2 *)p_v;
	return v->y;
}

void GDAPI godot_vector2_normalize(godot_vector2 *p_v) {
	Vector2 *v = (Vector2 *)p_v;
	v->normalize();
}
void GDAPI godot_vector2_normalized(godot_vector2 *p_dest,
		const godot_vector2 *p_src) {
	Vector2 *v = (Vector2 *)p_src;
	Vector2 *d = (Vector2 *)p_dest;

	*d = v->normalized();
}

godot_real GDAPI godot_vector2_length(const godot_vector2 *p_v) {
	Vector2 *v = (Vector2 *)p_v;
	return v->length();
}

godot_real GDAPI godot_vector2_length_squared(const godot_vector2 *p_v) {
	Vector2 *v = (Vector2 *)p_v;
	return v->length_squared();
}

godot_real GDAPI godot_vector2_distance_to(const godot_vector2 *p_a,
		const godot_vector2 *p_b) {
	Vector2 *a = (Vector2 *)p_a;
	Vector2 *b = (Vector2 *)p_b;
	return a->distance_to(*b);
}

godot_real GDAPI godot_vector2_distance_squared_to(const godot_vector2 *p_a,
		const godot_vector2 *p_b) {
	Vector2 *a = (Vector2 *)p_a;
	Vector2 *b = (Vector2 *)p_b;
	return a->distance_squared_to(*b);
}

void GDAPI godot_vector2_operator_add(godot_vector2 *p_dest,
		const godot_vector2 *p_a,
		const godot_vector2 *p_b) {
	Vector2 *dest = (Vector2 *)p_dest;
	const Vector2 *a = (Vector2 *)p_a;
	const Vector2 *b = (Vector2 *)p_b;
	*dest = *a + *b;
}

void GDAPI godot_vector2_operator_subtract(godot_vector2 *p_dest,
		const godot_vector2 *p_a,
		const godot_vector2 *p_b) {
	Vector2 *dest = (Vector2 *)p_dest;
	const Vector2 *a = (Vector2 *)p_a;
	const Vector2 *b = (Vector2 *)p_b;
	*dest = *a - *b;
}

void GDAPI godot_vector2_operator_multiply_vector(godot_vector2 *p_dest,
		const godot_vector2 *p_a,
		const godot_vector2 *p_b) {
	Vector2 *dest = (Vector2 *)p_dest;
	const Vector2 *a = (Vector2 *)p_a;
	const Vector2 *b = (Vector2 *)p_b;
	*dest = *a * *b;
}

void GDAPI godot_vector2_operator_multiply_scalar(godot_vector2 *p_dest,
		const godot_vector2 *p_a,
		const godot_real p_b) {
	Vector2 *dest = (Vector2 *)p_dest;
	const Vector2 *a = (Vector2 *)p_a;
	*dest = *a * p_b;
}

void GDAPI godot_vector2_operator_divide_vector(godot_vector2 *p_dest,
		const godot_vector2 *p_a,
		const godot_vector2 *p_b) {
	Vector2 *dest = (Vector2 *)p_dest;
	const Vector2 *a = (Vector2 *)p_a;
	const Vector2 *b = (Vector2 *)p_b;
	*dest = *a / *b;
}

void GDAPI godot_vector2_operator_divide_scalar(godot_vector2 *p_dest,
		const godot_vector2 *p_a,
		const godot_real p_b) {
	Vector2 *dest = (Vector2 *)p_dest;
	const Vector2 *a = (Vector2 *)p_a;
	*dest = *a / p_b;
}

godot_bool GDAPI godot_vector2_operator_equal(const godot_vector2 *p_a,
		const godot_vector2 *p_b) {
	const Vector2 *a = (Vector2 *)p_a;
	const Vector2 *b = (Vector2 *)p_b;
	return *a == *b;
}

godot_bool GDAPI godot_vector2_operator_less(const godot_vector2 *p_a,
		const godot_vector2 *p_b) {
	const Vector2 *a = (Vector2 *)p_a;
	const Vector2 *b = (Vector2 *)p_b;
	return *a < *b;
}

void GDAPI godot_vector2_abs(godot_vector2 *p_dest,
		const godot_vector2 *p_src) {
	const Vector2 *src = (Vector2 *)p_src;
	Vector2 *dest = (Vector2 *)p_dest;
	*dest = src->abs();
}

godot_real GDAPI godot_vector2_angle(const godot_vector2 *p_src) {
	const Vector2 *src = (Vector2 *)p_src;
	return src->angle();
}

godot_real GDAPI godot_vector2_angle_to(const godot_vector2 *p_src,
		const godot_vector2 *p_to) {
	const Vector2 *src = (Vector2 *)p_src;
	const Vector2 *to = (Vector2 *)p_to;
	return src->angle_to(*to);
}

godot_real GDAPI godot_vector2_angle_to_point(const godot_vector2 *p_src,
		const godot_vector2 *p_to) {
	const Vector2 *src = (Vector2 *)p_src;
	const Vector2 *to = (Vector2 *)p_to;
	return src->angle_to_point(*to);
}

void GDAPI godot_vector2_clamped(godot_vector2 *p_dest,
		const godot_vector2 *p_src,
		godot_real length) {
	const Vector2 *src = (Vector2 *)p_src;
	Vector2 *dest = (Vector2 *)p_dest;
	*dest = src->clamped(length);
}

void GDAPI godot_vector2_cubic_interpolate(
		godot_vector2 *p_dest, const godot_vector2 *p_src, const godot_vector2 *p_b,
		const godot_vector2 *p_pre_a, const godot_vector2 *p_post_b, godot_real t) {
	Vector2 *dest = (Vector2 *)p_dest;
	const Vector2 *src = (Vector2 *)p_src;
	const Vector2 *b = (Vector2 *)p_b;
	const Vector2 *pre_a = (Vector2 *)p_pre_a;
	const Vector2 *post_b = (Vector2 *)p_post_b;
	*dest = src->cubic_interpolate(*b, *pre_a, *post_b, t);
}

godot_real GDAPI godot_vector2_dot(const godot_vector2 *p_src,
		const godot_vector2 *p_with) {
	const Vector2 *src = (Vector2 *)p_src;
	const Vector2 *with = (Vector2 *)p_with;
	return src->dot(*with);
}

void GDAPI godot_vector2_floor(godot_vector2 *p_dest,
		const godot_vector2 *p_src) {
	const Vector2 *src = (Vector2 *)p_src;
	Vector2 *dest = (Vector2 *)p_dest;
	*dest = src->floor();
}

godot_real GDAPI godot_vector2_aspect(const godot_vector2 *p_src) {
	const Vector2 *src = (Vector2 *)p_src;
	return src->aspect();
}

void GDAPI godot_vector2_linear_interpolate(godot_vector2 *p_dest,
		const godot_vector2 *p_src,
		const godot_vector2 *p_b,
		godot_real t) {
	Vector2 *dest = (Vector2 *)p_dest;
	const Vector2 *src = (Vector2 *)p_src;
	const Vector2 *b = (Vector2 *)p_b;
	*dest = src->linear_interpolate(*b, t);
}

void GDAPI godot_vector2_reflect(godot_vector2 *p_dest,
		const godot_vector2 *p_src,
		const godot_vector2 *p_vec) {
	const Vector2 *src = (Vector2 *)p_src;
	const Vector2 *vec = (Vector2 *)p_vec;
	Vector2 *dest = (Vector2 *)p_dest;
	*dest = src->reflect(*vec);
}

void GDAPI godot_vector2_rotated(godot_vector2 *p_dest,
		const godot_vector2 *p_src, godot_real phi) {
	const Vector2 *src = (Vector2 *)p_src;
	Vector2 *dest = (Vector2 *)p_dest;
	*dest = src->rotated(phi);
}

void GDAPI godot_vector2_slide(godot_vector2 *p_dest,
		const godot_vector2 *p_src,
		godot_vector2 *p_vec) {
	const Vector2 *src = (Vector2 *)p_src;
	const Vector2 *vec = (Vector2 *)p_vec;
	Vector2 *dest = (Vector2 *)p_dest;
	*dest = src->slide(*vec);
}

void GDAPI godot_vector2_snapped(godot_vector2 *p_dest,
		const godot_vector2 *p_src,
		godot_vector2 *p_by) {
	const Vector2 *src = (Vector2 *)p_src;
	const Vector2 *by = (Vector2 *)p_by;
	Vector2 *dest = (Vector2 *)p_dest;
	*dest = src->snapped(*by);
}

void GDAPI godot_vector2_tangent(godot_vector2 *p_dest,
		const godot_vector2 *p_src) {
	const Vector2 *src = (Vector2 *)p_src;
	Vector2 *dest = (Vector2 *)p_dest;
	*dest = src->tangent();
}

void GDAPI godot_vector2_to_string(godot_string *p_dest,
		const godot_vector2 *p_src) {
	const Vector2 *src = (Vector2 *)p_src;
	String *dest = (String *)p_dest;
	*dest = "(" + *src + ")";
}

#ifdef __cplusplus
}
#endif
