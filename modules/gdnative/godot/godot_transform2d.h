/*************************************************************************/
/*  godot_transform2d.h                                                  */
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
#ifndef GODOT_TRANSFORM2D_H
#define GODOT_TRANSFORM2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_TRANSFORM2D_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_TRANSFORM2D_TYPE_DEFINED
typedef struct godot_transform2d {
	uint8_t _dont_touch_that[24];
} godot_transform2d;
#endif

#include "../godot.h"

#include "godot_vector2.h"

void GDAPI godot_transform2d_new_identity(godot_transform2d *p_t);
void GDAPI godot_transform2d_new_elements(godot_transform2d *p_t, const godot_vector2 *p_a, const godot_vector2 *p_b, const godot_vector2 *p_c);
void GDAPI godot_transform2d_new(godot_transform2d *p_t, const godot_real p_rot, const godot_vector2 *p_pos);

/*
godot_real GDAPI godot_transform2d_tdotx(const godot_transform2d *p_t, const godot_vector2 *p_v);
godot_real GDAPI godot_transform2d_tdoty(const godot_transform2d *p_t, const godot_vector2 *p_v);
*/

godot_vector2 const GDAPI *godot_transform2d_const_index(const godot_transform2d *p_t, const godot_int p_idx);
godot_vector2 GDAPI *godot_transform2d_index(godot_transform2d *p_t, const godot_int p_idx);

godot_vector2 GDAPI godot_transform2d_get_axis(const godot_transform2d *p_t, const godot_int p_axis);
void GDAPI godot_transform2d_set_axis(godot_transform2d *p_t, const godot_int p_axis, const godot_vector2 *p_vec);

/*
void GDAPI godot_transform2d_invert(godot_transform2d *p_t);
godot_transform2d GDAPI godot_transform2d_inverse(const godot_transform2d *p_t);
*/

// @Incomplete
// I feel like it should be enough to expose get and set, the whole logic can be done in the bindings.

#ifdef __cplusplus
}
#endif

#endif // GODOT_TRANSFORM2D_H
