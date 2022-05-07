/*************************************************************************/
/*  transform2d.h                                                        */
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

#ifndef GODOT_TRANSFORM2D_H
#define GODOT_TRANSFORM2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define GODOT_TRANSFORM2D_SIZE 24

#ifndef GODOT_CORE_API_GODOT_TRANSFORM2D_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_TRANSFORM2D_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_TRANSFORM2D_SIZE];
} godot_transform2d;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/gdnative.h>
#include <gdnative/variant.h>
#include <gdnative/vector2.h>

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_transform2d_new(godot_transform2d *r_dest, const godot_real p_rot, const godot_vector2 *p_pos);
void GDAPI godot_transform2d_new_axis_origin(godot_transform2d *r_dest, const godot_vector2 *p_x_axis, const godot_vector2 *p_y_axis, const godot_vector2 *p_origin);

godot_string GDAPI godot_transform2d_as_string(const godot_transform2d *p_self);

godot_transform2d GDAPI godot_transform2d_inverse(const godot_transform2d *p_self);

godot_transform2d GDAPI godot_transform2d_affine_inverse(const godot_transform2d *p_self);

godot_real GDAPI godot_transform2d_get_rotation(const godot_transform2d *p_self);

godot_vector2 GDAPI godot_transform2d_get_origin(const godot_transform2d *p_self);

godot_vector2 GDAPI godot_transform2d_get_scale(const godot_transform2d *p_self);

godot_transform2d GDAPI godot_transform2d_orthonormalized(const godot_transform2d *p_self);

godot_transform2d GDAPI godot_transform2d_rotated(const godot_transform2d *p_self, const godot_real p_phi);

godot_transform2d GDAPI godot_transform2d_scaled(const godot_transform2d *p_self, const godot_vector2 *p_scale);

godot_transform2d GDAPI godot_transform2d_translated(const godot_transform2d *p_self, const godot_vector2 *p_offset);

godot_vector2 GDAPI godot_transform2d_xform_vector2(const godot_transform2d *p_self, const godot_vector2 *p_v);

godot_vector2 GDAPI godot_transform2d_xform_inv_vector2(const godot_transform2d *p_self, const godot_vector2 *p_v);

godot_vector2 GDAPI godot_transform2d_basis_xform_vector2(const godot_transform2d *p_self, const godot_vector2 *p_v);

godot_vector2 GDAPI godot_transform2d_basis_xform_inv_vector2(const godot_transform2d *p_self, const godot_vector2 *p_v);

godot_transform2d GDAPI godot_transform2d_interpolate_with(const godot_transform2d *p_self, const godot_transform2d *p_m, const godot_real p_c);

godot_bool GDAPI godot_transform2d_operator_equal(const godot_transform2d *p_self, const godot_transform2d *p_b);

godot_transform2d GDAPI godot_transform2d_operator_multiply(const godot_transform2d *p_self, const godot_transform2d *p_b);

void GDAPI godot_transform2d_new_identity(godot_transform2d *r_dest);

godot_rect2 GDAPI godot_transform2d_xform_rect2(const godot_transform2d *p_self, const godot_rect2 *p_v);

godot_rect2 GDAPI godot_transform2d_xform_inv_rect2(const godot_transform2d *p_self, const godot_rect2 *p_v);

#ifdef __cplusplus
}
#endif

#endif // GODOT_TRANSFORM2D_H
