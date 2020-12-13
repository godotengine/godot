/*************************************************************************/
/*  vector2.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GODOT_VECTOR2_H
#define GODOT_VECTOR2_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define GODOT_VECTOR2_SIZE 8

#ifndef GODOT_CORE_API_GODOT_VECTOR2_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_VECTOR2_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_VECTOR2_SIZE];
} godot_vector2;
#endif

#define GODOT_VECTOR2I_SIZE 8

#ifndef GODOT_CORE_API_GODOT_VECTOR2I_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_VECTOR2I_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_VECTOR2I_SIZE];
} godot_vector2i;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/gdnative.h>

#ifdef __cplusplus
extern "C" {
#endif

// Vector2

void GDAPI godot_vector2_new(godot_vector2 *r_dest, const godot_real p_x, const godot_real p_y);

godot_string GDAPI godot_vector2_as_string(const godot_vector2 *p_self);

godot_vector2i GDAPI godot_vector2_as_vector2i(const godot_vector2 *p_self);

godot_vector2 GDAPI godot_vector2_normalized(const godot_vector2 *p_self);

godot_real GDAPI godot_vector2_length(const godot_vector2 *p_self);

godot_real GDAPI godot_vector2_angle(const godot_vector2 *p_self);

godot_real GDAPI godot_vector2_length_squared(const godot_vector2 *p_self);

godot_bool GDAPI godot_vector2_is_normalized(const godot_vector2 *p_self);

godot_vector2 GDAPI godot_vector2_direction_to(const godot_vector2 *p_self, const godot_vector2 *p_b);

godot_real GDAPI godot_vector2_distance_to(const godot_vector2 *p_self, const godot_vector2 *p_to);

godot_real GDAPI godot_vector2_distance_squared_to(const godot_vector2 *p_self, const godot_vector2 *p_to);

godot_real GDAPI godot_vector2_angle_to(const godot_vector2 *p_self, const godot_vector2 *p_to);

godot_real GDAPI godot_vector2_angle_to_point(const godot_vector2 *p_self, const godot_vector2 *p_to);

godot_vector2 GDAPI godot_vector2_lerp(const godot_vector2 *p_self, const godot_vector2 *p_b, const godot_real p_t);

godot_vector2 GDAPI godot_vector2_cubic_interpolate(const godot_vector2 *p_self, const godot_vector2 *p_b, const godot_vector2 *p_pre_a, const godot_vector2 *p_post_b, const godot_real p_t);

godot_vector2 GDAPI godot_vector2_move_toward(const godot_vector2 *p_self, const godot_vector2 *p_to, const godot_real p_delta);

godot_vector2 GDAPI godot_vector2_rotated(const godot_vector2 *p_self, const godot_real p_phi);

godot_vector2 GDAPI godot_vector2_tangent(const godot_vector2 *p_self);

godot_vector2 GDAPI godot_vector2_floor(const godot_vector2 *p_self);

godot_vector2 GDAPI godot_vector2_sign(const godot_vector2 *p_self);

godot_vector2 GDAPI godot_vector2_snapped(const godot_vector2 *p_self, const godot_vector2 *p_by);

godot_real GDAPI godot_vector2_aspect(const godot_vector2 *p_self);

godot_real GDAPI godot_vector2_dot(const godot_vector2 *p_self, const godot_vector2 *p_with);

godot_vector2 GDAPI godot_vector2_slide(const godot_vector2 *p_self, const godot_vector2 *p_n);

godot_vector2 GDAPI godot_vector2_bounce(const godot_vector2 *p_self, const godot_vector2 *p_n);

godot_vector2 GDAPI godot_vector2_reflect(const godot_vector2 *p_self, const godot_vector2 *p_n);

godot_vector2 GDAPI godot_vector2_abs(const godot_vector2 *p_self);

godot_vector2 GDAPI godot_vector2_clamped(const godot_vector2 *p_self, const godot_real p_length);

godot_vector2 GDAPI godot_vector2_operator_add(const godot_vector2 *p_self, const godot_vector2 *p_b);

godot_vector2 GDAPI godot_vector2_operator_subtract(const godot_vector2 *p_self, const godot_vector2 *p_b);

godot_vector2 GDAPI godot_vector2_operator_multiply_vector(const godot_vector2 *p_self, const godot_vector2 *p_b);

godot_vector2 GDAPI godot_vector2_operator_multiply_scalar(const godot_vector2 *p_self, const godot_real p_b);

godot_vector2 GDAPI godot_vector2_operator_divide_vector(const godot_vector2 *p_self, const godot_vector2 *p_b);

godot_vector2 GDAPI godot_vector2_operator_divide_scalar(const godot_vector2 *p_self, const godot_real p_b);

godot_bool GDAPI godot_vector2_operator_equal(const godot_vector2 *p_self, const godot_vector2 *p_b);

godot_bool GDAPI godot_vector2_operator_less(const godot_vector2 *p_self, const godot_vector2 *p_b);

godot_vector2 GDAPI godot_vector2_operator_neg(const godot_vector2 *p_self);

void GDAPI godot_vector2_set_x(godot_vector2 *p_self, const godot_real p_x);

void GDAPI godot_vector2_set_y(godot_vector2 *p_self, const godot_real p_y);

godot_real GDAPI godot_vector2_get_x(const godot_vector2 *p_self);

godot_real GDAPI godot_vector2_get_y(const godot_vector2 *p_self);

// Vector2i

void GDAPI godot_vector2i_new(godot_vector2i *r_dest, const godot_int p_x, const godot_int p_y);

godot_string GDAPI godot_vector2i_as_string(const godot_vector2i *p_self);

godot_vector2 GDAPI godot_vector2i_as_vector2(const godot_vector2i *p_self);

godot_real GDAPI godot_vector2i_aspect(const godot_vector2i *p_self);

godot_vector2i GDAPI godot_vector2i_abs(const godot_vector2i *p_self);

godot_vector2i GDAPI godot_vector2i_sign(const godot_vector2i *p_self);

godot_vector2i GDAPI godot_vector2i_operator_add(const godot_vector2i *p_self, const godot_vector2i *p_b);

godot_vector2i GDAPI godot_vector2i_operator_subtract(const godot_vector2i *p_self, const godot_vector2i *p_b);

godot_vector2i GDAPI godot_vector2i_operator_multiply_vector(const godot_vector2i *p_self, const godot_vector2i *p_b);

godot_vector2i GDAPI godot_vector2i_operator_multiply_scalar(const godot_vector2i *p_self, const godot_int p_b);

godot_vector2i GDAPI godot_vector2i_operator_divide_vector(const godot_vector2i *p_self, const godot_vector2i *p_b);

godot_vector2i GDAPI godot_vector2i_operator_divide_scalar(const godot_vector2i *p_self, const godot_int p_b);

godot_bool GDAPI godot_vector2i_operator_equal(const godot_vector2i *p_self, const godot_vector2i *p_b);

godot_bool GDAPI godot_vector2i_operator_less(const godot_vector2i *p_self, const godot_vector2i *p_b);

godot_vector2i GDAPI godot_vector2i_operator_neg(const godot_vector2i *p_self);

void GDAPI godot_vector2i_set_x(godot_vector2i *p_self, const godot_int p_x);

void GDAPI godot_vector2i_set_y(godot_vector2i *p_self, const godot_int p_y);

godot_int GDAPI godot_vector2i_get_x(const godot_vector2i *p_self);

godot_int GDAPI godot_vector2i_get_y(const godot_vector2i *p_self);

#ifdef __cplusplus
}
#endif

#endif // GODOT_VECTOR2_H
