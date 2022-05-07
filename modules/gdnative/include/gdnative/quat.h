/*************************************************************************/
/*  quat.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GODOT_QUAT_H
#define GODOT_QUAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define GODOT_QUAT_SIZE 16

#ifndef GODOT_CORE_API_GODOT_QUAT_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_QUAT_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_QUAT_SIZE];
} godot_quat;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/gdnative.h>
#include <gdnative/vector3.h>

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_quat_new(godot_quat *r_dest, const godot_real p_x, const godot_real p_y, const godot_real p_z, const godot_real p_w);
void GDAPI godot_quat_new_with_axis_angle(godot_quat *r_dest, const godot_vector3 *p_axis, const godot_real p_angle);
void GDAPI godot_quat_new_with_basis(godot_quat *r_dest, const godot_basis *p_basis);
void GDAPI godot_quat_new_with_euler(godot_quat *r_dest, const godot_vector3 *p_euler);

godot_real GDAPI godot_quat_get_x(const godot_quat *p_self);
void GDAPI godot_quat_set_x(godot_quat *p_self, const godot_real val);

godot_real GDAPI godot_quat_get_y(const godot_quat *p_self);
void GDAPI godot_quat_set_y(godot_quat *p_self, const godot_real val);

godot_real GDAPI godot_quat_get_z(const godot_quat *p_self);
void GDAPI godot_quat_set_z(godot_quat *p_self, const godot_real val);

godot_real GDAPI godot_quat_get_w(const godot_quat *p_self);
void GDAPI godot_quat_set_w(godot_quat *p_self, const godot_real val);

godot_string GDAPI godot_quat_as_string(const godot_quat *p_self);

godot_real GDAPI godot_quat_length(const godot_quat *p_self);

godot_real GDAPI godot_quat_length_squared(const godot_quat *p_self);

godot_quat GDAPI godot_quat_normalized(const godot_quat *p_self);

godot_bool GDAPI godot_quat_is_normalized(const godot_quat *p_self);

godot_quat GDAPI godot_quat_inverse(const godot_quat *p_self);

godot_real GDAPI godot_quat_dot(const godot_quat *p_self, const godot_quat *p_b);

godot_vector3 GDAPI godot_quat_xform(const godot_quat *p_self, const godot_vector3 *p_v);

godot_quat GDAPI godot_quat_slerp(const godot_quat *p_self, const godot_quat *p_b, const godot_real p_t);

godot_quat GDAPI godot_quat_slerpni(const godot_quat *p_self, const godot_quat *p_b, const godot_real p_t);

godot_quat GDAPI godot_quat_cubic_slerp(const godot_quat *p_self, const godot_quat *p_b, const godot_quat *p_pre_a, const godot_quat *p_post_b, const godot_real p_t);

godot_quat GDAPI godot_quat_operator_multiply(const godot_quat *p_self, const godot_real p_b);

godot_quat GDAPI godot_quat_operator_add(const godot_quat *p_self, const godot_quat *p_b);

godot_quat GDAPI godot_quat_operator_subtract(const godot_quat *p_self, const godot_quat *p_b);

godot_quat GDAPI godot_quat_operator_divide(const godot_quat *p_self, const godot_real p_b);

godot_bool GDAPI godot_quat_operator_equal(const godot_quat *p_self, const godot_quat *p_b);

godot_quat GDAPI godot_quat_operator_neg(const godot_quat *p_self);

void GDAPI godot_quat_set_axis_angle(godot_quat *p_self, const godot_vector3 *p_axis, const godot_real p_angle);

#ifdef __cplusplus
}
#endif

#endif // GODOT_QUAT_H
