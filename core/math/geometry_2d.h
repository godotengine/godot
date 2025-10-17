/**************************************************************************/
/*  geometry_2d.h                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/math/math_defs.h"

template <typename T>
class Vector;

struct Rect2;
struct Vector2;
struct Vector2i;
struct Vector3i;

namespace Geometry2D {

enum PolyBooleanOperation {
	OPERATION_UNION,
	OPERATION_DIFFERENCE,
	OPERATION_INTERSECTION,
	OPERATION_XOR,
};
enum PolyJoinType {
	JOIN_SQUARE,
	JOIN_ROUND,
	JOIN_MITER,
};
enum PolyEndType {
	END_POLYGON,
	END_JOINED,
	END_BUTT,
	END_SQUARE,
	END_ROUND,
};

real_t get_closest_points_between_segments(const Vector2 &p_p1, const Vector2 &p_q1, const Vector2 &p_p2, const Vector2 &p_q2, Vector2 &r_c1, Vector2 &r_c2);

Vector2 get_closest_point_to_segment(const Vector2 &p_point, const Vector2 &p_segment_a, const Vector2 &p_segment_b);
real_t get_distance_to_segment(const Vector2 &p_point, const Vector2 &p_segment_a, const Vector2 &p_segment_b);
Vector2 get_closest_point_to_segment_uncapped(const Vector2 &p_point, const Vector2 &p_segment_a, const Vector2 &p_segment_b);

#ifndef DISABLE_DEPRECATED
Vector2 get_closest_point_to_segment(const Vector2 &p_point, const Vector2 *p_segment);
real_t get_distance_to_segment(const Vector2 &p_point, const Vector2 *p_segment);
Vector2 get_closest_point_to_segment_uncapped(const Vector2 &p_point, const Vector2 *p_segment);
#endif // DISABLE_DEPRECATED

bool is_point_in_triangle(const Vector2 &p_s, const Vector2 &p_a, const Vector2 &p_b, const Vector2 &p_c);
bool line_intersects_line(const Vector2 &p_from_a, const Vector2 &p_dir_a, const Vector2 &p_from_b, const Vector2 &p_dir_b, Vector2 &r_result);
bool segment_intersects_segment(const Vector2 &p_from_a, const Vector2 &p_to_a, const Vector2 &p_from_b, const Vector2 &p_to_b, Vector2 *r_result);
bool is_point_in_circle(const Vector2 &p_point, const Vector2 &p_circle_pos, real_t p_circle_radius);
real_t segment_intersects_circle(const Vector2 &p_from, const Vector2 &p_to, const Vector2 &p_circle_pos, real_t p_circle_radius);
bool segment_intersects_rect(const Vector2 &p_from, const Vector2 &p_to, const Rect2 &p_rect);

Vector<Vector<Vector2>> merge_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b);
Vector<Vector<Vector2>> clip_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b);
Vector<Vector<Vector2>> intersect_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b);
Vector<Vector<Vector2>> exclude_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b);
Vector<Vector<Vector2>> clip_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon);
Vector<Vector<Vector2>> intersect_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon);
Vector<Vector<Vector2>> offset_polygon(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type);
Vector<Vector<Vector2>> offset_polyline(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type, PolyEndType p_end_type);

Vector<int32_t> triangulate_delaunay(const Vector<Vector2> &p_points);
Vector<int32_t> triangulate_polygon(const Vector<Vector2> &p_polygon);

bool is_polygon_clockwise(const Vector<Vector2> &p_polygon);
bool is_point_in_polygon(const Vector2 &p_point, const Vector<Vector2> &p_polygon);
bool is_segment_intersecting_polygon(const Vector2 &p_from, const Vector2 &p_to, const Vector<Vector2> &p_polygon);

real_t vec2_cross(const Vector2 &p_o, const Vector2 &p_a, const Vector2 &p_b);
Vector<Vector2> convex_hull(Vector<Vector2> p_points);
Vector<Vector2i> bresenham_line(const Vector2i &p_from, const Vector2i &p_to);

void merge_many_polygons(const Vector<Vector<Vector2>> &p_polygons, Vector<Vector<Vector2>> &r_out_polygons, Vector<Vector<Vector2>> &r_out_holes);
Vector<Vector<Vector2>> decompose_many_polygons_in_convex(const Vector<Vector<Vector2>> &p_polygons, const Vector<Vector<Vector2>> &p_holes);
Vector<Vector<Vector2>> decompose_polygon_in_convex(const Vector<Vector2> &p_polygon);

void make_atlas(const Vector<Vector2i> &p_rects, Vector<Vector2i> &r_result, Vector2i &r_size);
Vector<Vector3i> partial_pack_rects(const Vector<Vector2i> &p_sizes, const Vector2i &p_atlas_size);

}; // namespace Geometry2D
