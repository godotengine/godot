/*************************************************************************/
/*  godot_quat.h                                                         */
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
#ifndef GODOT_QUAT_H
#define GODOT_QUAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_QUAT_TYPE_DEFINED
typedef struct godot_quat {
	uint8_t _dont_touch_that[16];
} godot_quat;
#endif

#include "../godot.h"

void GDAPI godot_quat_new(godot_quat *p_quat);
void GDAPI godot_quat_new_with_elements(godot_quat *p_quat, const godot_real x, const godot_real y, const godot_real z, const godot_real w);
void GDAPI godot_quat_new_with_rotation(godot_quat *p_quat, const godot_vector3 *p_axis, const godot_real p_angle);
void GDAPI godot_quat_new_with_shortest_arc(godot_quat *p_quat, const godot_vector3 *p_v0, const godot_vector3 *p_v1);

godot_vector3 GDAPI godot_quat_get_euler(const godot_quat *p_quat);
void GDAPI godot_quat_set_euler(godot_quat *p_quat, const godot_vector3 *p_euler);

godot_real GDAPI *godot_quat_index(godot_quat *p_quat, const godot_int p_idx);
godot_real GDAPI godot_quat_const_index(const godot_quat *p_quat, const godot_int p_idx);

#ifdef __cplusplus
}
#endif

#endif // GODOT_QUAT_H
