/**************************************************************************/
/*  geometry3d.hpp                                                        */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/plane.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Geometry3D : public Object {
	GDEXTENSION_CLASS(Geometry3D, Object)

	static Geometry3D *singleton;

public:
	static Geometry3D *get_singleton();

	PackedVector3Array compute_convex_mesh_points(const TypedArray<Plane> &p_planes);
	TypedArray<Plane> build_box_planes(const Vector3 &p_extents);
	TypedArray<Plane> build_cylinder_planes(float p_radius, float p_height, int32_t p_sides, Vector3::Axis p_axis = (Vector3::Axis)2);
	TypedArray<Plane> build_capsule_planes(float p_radius, float p_height, int32_t p_sides, int32_t p_lats, Vector3::Axis p_axis = (Vector3::Axis)2);
	PackedVector3Array get_closest_points_between_segments(const Vector3 &p_p1, const Vector3 &p_p2, const Vector3 &p_q1, const Vector3 &p_q2);
	Vector3 get_closest_point_to_segment(const Vector3 &p_point, const Vector3 &p_s1, const Vector3 &p_s2);
	Vector3 get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 &p_s1, const Vector3 &p_s2);
	Vector3 get_triangle_barycentric_coords(const Vector3 &p_point, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c);
	Variant ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c);
	Variant segment_intersects_triangle(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c);
	PackedVector3Array segment_intersects_sphere(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_sphere_position, float p_sphere_radius);
	PackedVector3Array segment_intersects_cylinder(const Vector3 &p_from, const Vector3 &p_to, float p_height, float p_radius);
	PackedVector3Array segment_intersects_convex(const Vector3 &p_from, const Vector3 &p_to, const TypedArray<Plane> &p_planes);
	PackedVector3Array clip_polygon(const PackedVector3Array &p_points, const Plane &p_plane);
	PackedInt32Array tetrahedralize_delaunay(const PackedVector3Array &p_points);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~Geometry3D();

public:
};

} // namespace godot

