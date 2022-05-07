/*************************************************************************/
/*  aabb.h                                                               */
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

#ifndef GODOT_AABB_H
#define GODOT_AABB_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define GODOT_AABB_SIZE 24

#ifndef GODOT_CORE_API_GODOT_AABB_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_AABB_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_AABB_SIZE];
} godot_aabb;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/gdnative.h>
#include <gdnative/plane.h>
#include <gdnative/vector3.h>

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_aabb_new(godot_aabb *r_dest, const godot_vector3 *p_pos, const godot_vector3 *p_size);

godot_vector3 GDAPI godot_aabb_get_position(const godot_aabb *p_self);
void GDAPI godot_aabb_set_position(const godot_aabb *p_self, const godot_vector3 *p_v);

godot_vector3 GDAPI godot_aabb_get_size(const godot_aabb *p_self);
void GDAPI godot_aabb_set_size(const godot_aabb *p_self, const godot_vector3 *p_v);

godot_string GDAPI godot_aabb_as_string(const godot_aabb *p_self);

godot_real GDAPI godot_aabb_get_area(const godot_aabb *p_self);

godot_bool GDAPI godot_aabb_has_no_area(const godot_aabb *p_self);

godot_bool GDAPI godot_aabb_has_no_surface(const godot_aabb *p_self);

godot_bool GDAPI godot_aabb_intersects(const godot_aabb *p_self, const godot_aabb *p_with);

godot_bool GDAPI godot_aabb_encloses(const godot_aabb *p_self, const godot_aabb *p_with);

godot_aabb GDAPI godot_aabb_merge(const godot_aabb *p_self, const godot_aabb *p_with);

godot_aabb GDAPI godot_aabb_intersection(const godot_aabb *p_self, const godot_aabb *p_with);

godot_bool GDAPI godot_aabb_intersects_plane(const godot_aabb *p_self, const godot_plane *p_plane);

godot_bool GDAPI godot_aabb_intersects_segment(const godot_aabb *p_self, const godot_vector3 *p_from, const godot_vector3 *p_to);

godot_bool GDAPI godot_aabb_has_point(const godot_aabb *p_self, const godot_vector3 *p_point);

godot_vector3 GDAPI godot_aabb_get_support(const godot_aabb *p_self, const godot_vector3 *p_dir);

godot_vector3 GDAPI godot_aabb_get_longest_axis(const godot_aabb *p_self);

godot_int GDAPI godot_aabb_get_longest_axis_index(const godot_aabb *p_self);

godot_real GDAPI godot_aabb_get_longest_axis_size(const godot_aabb *p_self);

godot_vector3 GDAPI godot_aabb_get_shortest_axis(const godot_aabb *p_self);

godot_int GDAPI godot_aabb_get_shortest_axis_index(const godot_aabb *p_self);

godot_real GDAPI godot_aabb_get_shortest_axis_size(const godot_aabb *p_self);

godot_aabb GDAPI godot_aabb_expand(const godot_aabb *p_self, const godot_vector3 *p_to_point);

godot_aabb GDAPI godot_aabb_grow(const godot_aabb *p_self, const godot_real p_by);

godot_vector3 GDAPI godot_aabb_get_endpoint(const godot_aabb *p_self, const godot_int p_idx);

godot_bool GDAPI godot_aabb_operator_equal(const godot_aabb *p_self, const godot_aabb *p_b);

#ifdef __cplusplus
}
#endif

#endif // GODOT_AABB_H
