/*************************************************************************/
/*  godot_vector3.h                                                      */
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
#ifndef GODOT_VECTOR3_H
#define GODOT_VECTOR3_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_VECTOR3_TYPE_DEFINED
typedef struct godot_vector3 {
	uint8_t _dont_touch_that[12];
} godot_vector3;
#endif

#define GODOT_VECTOR3_AXIX_X 0
#define GODOT_VECTOR3_AXIX_Y 1
#define GODOT_VECTOR3_AXIX_Z 2

#include "../godot.h"
#include "godot_basis.h"

void GDAPI godot_vector3_new(godot_vector3 *p_v, const godot_real p_x, const godot_real p_y, const godot_real p_z);

void GDAPI godot_vector3_set_axis(godot_vector3 *p_v, const godot_int p_axis, const godot_real p_val);
godot_real GDAPI godot_vector3_get_axis(const godot_vector3 *p_v, const godot_int p_axis);

godot_int GDAPI godot_vector3_min_axis(const godot_vector3 *p_v);
godot_int GDAPI godot_vector3_max_axis(const godot_vector3 *p_v);

godot_real GDAPI godot_vector3_length(const godot_vector3 *p_v);
godot_real GDAPI godot_vector3_length_squared(const godot_vector3 *p_v);

void GDAPI godot_vector3_normalize(godot_vector3 *p_v);
void GDAPI godot_vector3_normalized(godot_vector3 *p_dest, const godot_vector3 *p_src);

void GDAPI godot_vector3_inverse(godot_vector3 *p_dest, const godot_vector3 *p_src);
void GDAPI godot_vector3_zero(godot_vector3 *p_src);
void GDAPI godot_vector3_snap(godot_vector3 *p_src, godot_real val);
void GDAPI godot_vector3_snapped(godot_vector3 *p_dest, const godot_vector3 *p_src, godot_real val);
void GDAPI godot_vector3_rotate(godot_vector3 *p_src, const godot_vector3 *p_axis, godot_real phi);
void GDAPI godot_vector3_rotated(godot_vector3 *p_dest, const godot_vector3 *p_src,
		const godot_vector3 *p_axis, godot_real phi);
void GDAPI godot_vector3_linear_interpolate(godot_vector3 *p_dest, const godot_vector3 *p_src,
		const godot_vector3 *p_b, godot_real t);
void GDAPI godot_vector3_cubic_interpolate(godot_vector3 *p_dest, const godot_vector3 *p_src,
		const godot_vector3 *p_b, const godot_vector3 *p_pre_a,
		const godot_vector3 *p_post_b, godot_real t);
void GDAPI godot_vector3_cubic_interpolaten(godot_vector3 *p_dest, const godot_vector3 *p_src,
		const godot_vector3 *p_b, const godot_vector3 *p_pre_a,
		const godot_vector3 *p_post_b, godot_real t);
void GDAPI godot_vector3_cross(godot_vector3 *p_dest, const godot_vector3 *p_src, const godot_vector3 *p_b);
godot_real GDAPI godot_vector3_dot(const godot_vector3 *p_src, const godot_vector3 *p_b);
void GDAPI godot_vector3_outer(godot_basis *dest, const godot_vector3 *p_src, const godot_vector3 *p_b);
void GDAPI godot_vector3_to_diagonal_matrix(godot_basis *dest, const godot_vector3 *p_src);
void GDAPI godot_vector3_abs(godot_vector3 *p_dest, const godot_vector3 *p_src);
void GDAPI godot_vector3_floor(godot_vector3 *p_dest, const godot_vector3 *p_src);
void GDAPI godot_vector3_ceil(godot_vector3 *p_dest, const godot_vector3 *p_src);

godot_real GDAPI godot_vector3_distance_to(const godot_vector3 *p_a, const godot_vector3 *p_b);
godot_real GDAPI godot_vector3_distance_squared_to(const godot_vector3 *p_a, const godot_vector3 *p_b);
godot_real GDAPI godot_vector3_angle_to(const godot_vector3 *p_a, const godot_vector3 *p_b);

void GDAPI godot_vector3_slide(godot_vector3 *p_dest, const godot_vector3 *p_src, const godot_vector3 *p_vec);
void GDAPI godot_vector3_bounce(godot_vector3 *p_dest, const godot_vector3 *p_src, const godot_vector3 *p_vec);
void GDAPI godot_vector3_reflect(godot_vector3 *p_dest, const godot_vector3 *p_src, const godot_vector3 *p_vec);

void GDAPI godot_vector3_operator_add(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_vector3 *p_b);
void GDAPI godot_vector3_operator_subtract(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_vector3 *p_b);
void GDAPI godot_vector3_operator_multiply_vector(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_vector3 *p_b);
void GDAPI godot_vector3_operator_multiply_scalar(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_real p_b);
void GDAPI godot_vector3_operator_divide_vector(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_vector3 *p_b);
void GDAPI godot_vector3_operator_divide_scalar(godot_vector3 *p_dest, const godot_vector3 *p_a, const godot_real p_b);

godot_bool GDAPI godot_vector3_operator_equal(const godot_vector3 *p_a, const godot_vector3 *p_b);
godot_bool GDAPI godot_vector3_operator_less(const godot_vector3 *p_a, const godot_vector3 *p_b);

void GDAPI godot_vector3_to_string(godot_string *p_dest, const godot_vector3 *p_src);

#ifdef __cplusplus
}
#endif

#endif // GODOT_VECTOR3_H
