/*************************************************************************/
/*  collision_solver_sw.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef COLLISION_SOLVER_SW_H
#define COLLISION_SOLVER_SW_H

#include "shape_sw.h"

class CollisionSolverSW {
public:
	typedef void (*CallbackResult)(const Vector3 &p_point_A, const Vector3 &p_point_B, void *p_userdata);

private:
	static void concave_callback(void *p_userdata, ShapeSW *p_convex);
	static bool solve_static_plane(const ShapeSW *p_shape_A, const Transform &p_transform_A, const ShapeSW *p_shape_B, const Transform &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result);
	static bool solve_ray(const ShapeSW *p_shape_A, const Transform &p_transform_A, const ShapeSW *p_shape_B, const Transform &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result);
	static bool solve_concave(const ShapeSW *p_shape_A, const Transform &p_transform_A, const ShapeSW *p_shape_B, const Transform &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result, real_t p_margin_A = 0, real_t p_margin_B = 0);
	static void concave_distance_callback(void *p_userdata, ShapeSW *p_convex);
	static bool solve_distance_plane(const ShapeSW *p_shape_A, const Transform &p_transform_A, const ShapeSW *p_shape_B, const Transform &p_transform_B, Vector3 &r_point_A, Vector3 &r_point_B);

public:
	static bool solve_static(const ShapeSW *p_shape_A, const Transform &p_transform_A, const ShapeSW *p_shape_B, const Transform &p_transform_B, CallbackResult p_result_callback, void *p_userdata, Vector3 *r_sep_axis = NULL, real_t p_margin_A = 0, real_t p_margin_B = 0);
	static bool solve_distance(const ShapeSW *p_shape_A, const Transform &p_transform_A, const ShapeSW *p_shape_B, const Transform &p_transform_B, Vector3 &r_point_A, Vector3 &r_point_B, const AABB &p_concave_hint, Vector3 *r_sep_axis = NULL);
};

#endif // COLLISION_SOLVER__SW_H
