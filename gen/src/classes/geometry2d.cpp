/**************************************************************************/
/*  geometry2d.cpp                                                        */
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

#include <godot_cpp/classes/geometry2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Geometry2D *Geometry2D::singleton = nullptr;

Geometry2D *Geometry2D::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(Geometry2D::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<Geometry2D *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &Geometry2D::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(Geometry2D::get_class_static(), singleton);
		}
	}
	return singleton;
}

Geometry2D::~Geometry2D() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(Geometry2D::get_class_static());
		singleton = nullptr;
	}
}

bool Geometry2D::is_point_in_circle(const Vector2 &p_point, const Vector2 &p_circle_position, float p_circle_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("is_point_in_circle")._native_ptr(), 2929491703);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	double p_circle_radius_encoded;
	PtrToArg<double>::encode(p_circle_radius, &p_circle_radius_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_point, &p_circle_position, &p_circle_radius_encoded);
}

float Geometry2D::segment_intersects_circle(const Vector2 &p_segment_from, const Vector2 &p_segment_to, const Vector2 &p_circle_position, float p_circle_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("segment_intersects_circle")._native_ptr(), 1356928167);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_circle_radius_encoded;
	PtrToArg<double>::encode(p_circle_radius, &p_circle_radius_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_segment_from, &p_segment_to, &p_circle_position, &p_circle_radius_encoded);
}

Variant Geometry2D::segment_intersects_segment(const Vector2 &p_from_a, const Vector2 &p_to_a, const Vector2 &p_from_b, const Vector2 &p_to_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("segment_intersects_segment")._native_ptr(), 2058025344);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_from_a, &p_to_a, &p_from_b, &p_to_b);
}

Variant Geometry2D::line_intersects_line(const Vector2 &p_from_a, const Vector2 &p_dir_a, const Vector2 &p_from_b, const Vector2 &p_dir_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("line_intersects_line")._native_ptr(), 2058025344);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_from_a, &p_dir_a, &p_from_b, &p_dir_b);
}

PackedVector2Array Geometry2D::get_closest_points_between_segments(const Vector2 &p_p1, const Vector2 &p_q1, const Vector2 &p_p2, const Vector2 &p_q2) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("get_closest_points_between_segments")._native_ptr(), 3344690961);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, &p_p1, &p_q1, &p_p2, &p_q2);
}

Vector2 Geometry2D::get_closest_point_to_segment(const Vector2 &p_point, const Vector2 &p_s1, const Vector2 &p_s2) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("get_closest_point_to_segment")._native_ptr(), 4172901909);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_point, &p_s1, &p_s2);
}

Vector2 Geometry2D::get_closest_point_to_segment_uncapped(const Vector2 &p_point, const Vector2 &p_s1, const Vector2 &p_s2) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("get_closest_point_to_segment_uncapped")._native_ptr(), 4172901909);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_point, &p_s1, &p_s2);
}

bool Geometry2D::point_is_inside_triangle(const Vector2 &p_point, const Vector2 &p_a, const Vector2 &p_b, const Vector2 &p_c) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("point_is_inside_triangle")._native_ptr(), 1025948137);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_point, &p_a, &p_b, &p_c);
}

bool Geometry2D::is_polygon_clockwise(const PackedVector2Array &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("is_polygon_clockwise")._native_ptr(), 1361156557);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_polygon);
}

bool Geometry2D::is_point_in_polygon(const Vector2 &p_point, const PackedVector2Array &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("is_point_in_polygon")._native_ptr(), 738277916);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_point, &p_polygon);
}

PackedInt32Array Geometry2D::triangulate_polygon(const PackedVector2Array &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("triangulate_polygon")._native_ptr(), 1389921771);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_polygon);
}

PackedInt32Array Geometry2D::triangulate_delaunay(const PackedVector2Array &p_points) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("triangulate_delaunay")._native_ptr(), 1389921771);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_points);
}

PackedVector2Array Geometry2D::convex_hull(const PackedVector2Array &p_points) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("convex_hull")._native_ptr(), 2004331998);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, &p_points);
}

TypedArray<PackedVector2Array> Geometry2D::decompose_polygon_in_convex(const PackedVector2Array &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("decompose_polygon_in_convex")._native_ptr(), 3982393695);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner, &p_polygon);
}

TypedArray<PackedVector2Array> Geometry2D::merge_polygons(const PackedVector2Array &p_polygon_a, const PackedVector2Array &p_polygon_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("merge_polygons")._native_ptr(), 3637387053);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner, &p_polygon_a, &p_polygon_b);
}

TypedArray<PackedVector2Array> Geometry2D::clip_polygons(const PackedVector2Array &p_polygon_a, const PackedVector2Array &p_polygon_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("clip_polygons")._native_ptr(), 3637387053);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner, &p_polygon_a, &p_polygon_b);
}

TypedArray<PackedVector2Array> Geometry2D::intersect_polygons(const PackedVector2Array &p_polygon_a, const PackedVector2Array &p_polygon_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("intersect_polygons")._native_ptr(), 3637387053);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner, &p_polygon_a, &p_polygon_b);
}

TypedArray<PackedVector2Array> Geometry2D::exclude_polygons(const PackedVector2Array &p_polygon_a, const PackedVector2Array &p_polygon_b) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("exclude_polygons")._native_ptr(), 3637387053);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner, &p_polygon_a, &p_polygon_b);
}

TypedArray<PackedVector2Array> Geometry2D::clip_polyline_with_polygon(const PackedVector2Array &p_polyline, const PackedVector2Array &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("clip_polyline_with_polygon")._native_ptr(), 3637387053);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner, &p_polyline, &p_polygon);
}

TypedArray<PackedVector2Array> Geometry2D::intersect_polyline_with_polygon(const PackedVector2Array &p_polyline, const PackedVector2Array &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("intersect_polyline_with_polygon")._native_ptr(), 3637387053);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner, &p_polyline, &p_polygon);
}

TypedArray<PackedVector2Array> Geometry2D::offset_polygon(const PackedVector2Array &p_polygon, float p_delta, Geometry2D::PolyJoinType p_join_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("offset_polygon")._native_ptr(), 1275354010);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	int64_t p_join_type_encoded;
	PtrToArg<int64_t>::encode(p_join_type, &p_join_type_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner, &p_polygon, &p_delta_encoded, &p_join_type_encoded);
}

TypedArray<PackedVector2Array> Geometry2D::offset_polyline(const PackedVector2Array &p_polyline, float p_delta, Geometry2D::PolyJoinType p_join_type, Geometry2D::PolyEndType p_end_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("offset_polyline")._native_ptr(), 2328231778);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	int64_t p_join_type_encoded;
	PtrToArg<int64_t>::encode(p_join_type, &p_join_type_encoded);
	int64_t p_end_type_encoded;
	PtrToArg<int64_t>::encode(p_end_type, &p_end_type_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner, &p_polyline, &p_delta_encoded, &p_join_type_encoded, &p_end_type_encoded);
}

Dictionary Geometry2D::make_atlas(const PackedVector2Array &p_sizes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("make_atlas")._native_ptr(), 1337682371);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_sizes);
}

TypedArray<Vector2i> Geometry2D::bresenham_line(const Vector2i &p_from, const Vector2i &p_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry2D::get_class_static()._native_ptr(), StringName("bresenham_line")._native_ptr(), 1989391000);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner, &p_from, &p_to);
}

} // namespace godot
