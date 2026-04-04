/**************************************************************************/
/*  geometry3d.cpp                                                        */
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

#include <godot_cpp/classes/geometry3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Geometry3D *Geometry3D::singleton = nullptr;

Geometry3D *Geometry3D::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(Geometry3D::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<Geometry3D *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &Geometry3D::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(Geometry3D::get_class_static(), singleton);
		}
	}
	return singleton;
}

Geometry3D::~Geometry3D() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(Geometry3D::get_class_static());
		singleton = nullptr;
	}
}

PackedVector3Array Geometry3D::compute_convex_mesh_points(const TypedArray<Plane> &p_planes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("compute_convex_mesh_points")._native_ptr(), 1936902142);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_planes);
}

TypedArray<Plane> Geometry3D::build_box_planes(const Vector3 &p_extents) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("build_box_planes")._native_ptr(), 3622277145);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Plane>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Plane>>(_gde_method_bind, _owner, &p_extents);
}

TypedArray<Plane> Geometry3D::build_cylinder_planes(float p_radius, float p_height, int32_t p_sides, Vector3::Axis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("build_cylinder_planes")._native_ptr(), 449920067);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Plane>()));
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	int64_t p_sides_encoded;
	PtrToArg<int64_t>::encode(p_sides, &p_sides_encoded);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Plane>>(_gde_method_bind, _owner, &p_radius_encoded, &p_height_encoded, &p_sides_encoded, &p_axis_encoded);
}

TypedArray<Plane> Geometry3D::build_capsule_planes(float p_radius, float p_height, int32_t p_sides, int32_t p_lats, Vector3::Axis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("build_capsule_planes")._native_ptr(), 2113592876);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Plane>()));
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	int64_t p_sides_encoded;
	PtrToArg<int64_t>::encode(p_sides, &p_sides_encoded);
	int64_t p_lats_encoded;
	PtrToArg<int64_t>::encode(p_lats, &p_lats_encoded);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Plane>>(_gde_method_bind, _owner, &p_radius_encoded, &p_height_encoded, &p_sides_encoded, &p_lats_encoded, &p_axis_encoded);
}

PackedVector3Array Geometry3D::get_closest_points_between_segments(const Vector3 &p_p1, const Vector3 &p_p2, const Vector3 &p_q1, const Vector3 &p_q2) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("get_closest_points_between_segments")._native_ptr(), 1056373962);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_p1, &p_p2, &p_q1, &p_q2);
}

Vector3 Geometry3D::get_closest_point_to_segment(const Vector3 &p_point, const Vector3 &p_s1, const Vector3 &p_s2) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("get_closest_point_to_segment")._native_ptr(), 2168193209);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_point, &p_s1, &p_s2);
}

Vector3 Geometry3D::get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 &p_s1, const Vector3 &p_s2) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("get_closest_point_to_segment_uncapped")._native_ptr(), 2168193209);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_point, &p_s1, &p_s2);
}

Vector3 Geometry3D::get_triangle_barycentric_coords(const Vector3 &p_point, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("get_triangle_barycentric_coords")._native_ptr(), 1362048029);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_point, &p_a, &p_b, &p_c);
}

Variant Geometry3D::ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("ray_intersects_triangle")._native_ptr(), 1718655448);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_from, &p_dir, &p_a, &p_b, &p_c);
}

Variant Geometry3D::segment_intersects_triangle(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("segment_intersects_triangle")._native_ptr(), 1718655448);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_from, &p_to, &p_a, &p_b, &p_c);
}

PackedVector3Array Geometry3D::segment_intersects_sphere(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_sphere_position, float p_sphere_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("segment_intersects_sphere")._native_ptr(), 4080141172);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	double p_sphere_radius_encoded;
	PtrToArg<double>::encode(p_sphere_radius, &p_sphere_radius_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_from, &p_to, &p_sphere_position, &p_sphere_radius_encoded);
}

PackedVector3Array Geometry3D::segment_intersects_cylinder(const Vector3 &p_from, const Vector3 &p_to, float p_height, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("segment_intersects_cylinder")._native_ptr(), 2361316491);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_from, &p_to, &p_height_encoded, &p_radius_encoded);
}

PackedVector3Array Geometry3D::segment_intersects_convex(const Vector3 &p_from, const Vector3 &p_to, const TypedArray<Plane> &p_planes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("segment_intersects_convex")._native_ptr(), 537425332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_from, &p_to, &p_planes);
}

PackedVector3Array Geometry3D::clip_polygon(const PackedVector3Array &p_points, const Plane &p_plane) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("clip_polygon")._native_ptr(), 2603188319);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_points, &p_plane);
}

PackedInt32Array Geometry3D::tetrahedralize_delaunay(const PackedVector3Array &p_points) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Geometry3D::get_class_static()._native_ptr(), StringName("tetrahedralize_delaunay")._native_ptr(), 1230191221);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_points);
}

} // namespace godot
