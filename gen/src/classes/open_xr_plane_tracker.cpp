/**************************************************************************/
/*  open_xr_plane_tracker.cpp                                             */
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

#include <godot_cpp/classes/open_xr_plane_tracker.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/shape3d.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>

namespace godot {

void OpenXRPlaneTracker::set_bounds_size(const Vector2 &p_bounds_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("set_bounds_size")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bounds_size);
}

Vector2 OpenXRPlaneTracker::get_bounds_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("get_bounds_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void OpenXRPlaneTracker::set_plane_alignment(OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment p_plane_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("set_plane_alignment")._native_ptr(), 1214382230);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_plane_alignment_encoded;
	PtrToArg<int64_t>::encode(p_plane_alignment, &p_plane_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_plane_alignment_encoded);
}

OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment OpenXRPlaneTracker::get_plane_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("get_plane_alignment")._native_ptr(), 845541441);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment(0)));
	return (OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRPlaneTracker::set_plane_label(const String &p_plane_label) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("set_plane_label")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_plane_label);
}

String OpenXRPlaneTracker::get_plane_label() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("get_plane_label")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void OpenXRPlaneTracker::set_mesh_data(const Transform3D &p_origin, const PackedVector2Array &p_vertices, const PackedInt32Array &p_indices) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("set_mesh_data")._native_ptr(), 1877193149);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_origin, &p_vertices, &p_indices);
}

void OpenXRPlaneTracker::clear_mesh_data() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("clear_mesh_data")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Transform3D OpenXRPlaneTracker::get_mesh_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("get_mesh_offset")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

Ref<Mesh> OpenXRPlaneTracker::get_mesh() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("get_mesh")._native_ptr(), 4081188045);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Mesh>()));
	return Ref<Mesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Mesh>(_gde_method_bind, _owner));
}

Ref<Shape3D> OpenXRPlaneTracker::get_shape(float p_thickness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRPlaneTracker::get_class_static()._native_ptr(), StringName("get_shape")._native_ptr(), 3358509884);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Shape3D>()));
	double p_thickness_encoded;
	PtrToArg<double>::encode(p_thickness, &p_thickness_encoded);
	return Ref<Shape3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Shape3D>(_gde_method_bind, _owner, &p_thickness_encoded));
}

} // namespace godot
