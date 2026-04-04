/**************************************************************************/
/*  navigation_mesh_source_geometry_data3d.cpp                            */
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

#include <godot_cpp/classes/navigation_mesh_source_geometry_data3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/transform3d.hpp>

namespace godot {

void NavigationMeshSourceGeometryData3D::set_vertices(const PackedFloat32Array &p_vertices) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("set_vertices")._native_ptr(), 2899603908);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertices);
}

PackedFloat32Array NavigationMeshSourceGeometryData3D::get_vertices() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("get_vertices")._native_ptr(), 675695659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedFloat32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedFloat32Array>(_gde_method_bind, _owner);
}

void NavigationMeshSourceGeometryData3D::set_indices(const PackedInt32Array &p_indices) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("set_indices")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_indices);
}

PackedInt32Array NavigationMeshSourceGeometryData3D::get_indices() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("get_indices")._native_ptr(), 1930428628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void NavigationMeshSourceGeometryData3D::append_arrays(const PackedFloat32Array &p_vertices, const PackedInt32Array &p_indices) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("append_arrays")._native_ptr(), 3117535015);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertices, &p_indices);
}

void NavigationMeshSourceGeometryData3D::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool NavigationMeshSourceGeometryData3D::has_data() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("has_data")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NavigationMeshSourceGeometryData3D::add_mesh(const Ref<Mesh> &p_mesh, const Transform3D &p_xform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("add_mesh")._native_ptr(), 975462459);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_mesh != nullptr ? &p_mesh->_owner : nullptr), &p_xform);
}

void NavigationMeshSourceGeometryData3D::add_mesh_array(const Array &p_mesh_array, const Transform3D &p_xform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("add_mesh_array")._native_ptr(), 4235710913);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh_array, &p_xform);
}

void NavigationMeshSourceGeometryData3D::add_faces(const PackedVector3Array &p_faces, const Transform3D &p_xform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("add_faces")._native_ptr(), 1440358797);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_faces, &p_xform);
}

void NavigationMeshSourceGeometryData3D::merge(const Ref<NavigationMeshSourceGeometryData3D> &p_other_geometry) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("merge")._native_ptr(), 655828145);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_other_geometry != nullptr ? &p_other_geometry->_owner : nullptr));
}

void NavigationMeshSourceGeometryData3D::add_projected_obstruction(const PackedVector3Array &p_vertices, float p_elevation, float p_height, bool p_carve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("add_projected_obstruction")._native_ptr(), 3351846707);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_elevation_encoded;
	PtrToArg<double>::encode(p_elevation, &p_elevation_encoded);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	int8_t p_carve_encoded;
	PtrToArg<bool>::encode(p_carve, &p_carve_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertices, &p_elevation_encoded, &p_height_encoded, &p_carve_encoded);
}

void NavigationMeshSourceGeometryData3D::clear_projected_obstructions() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("clear_projected_obstructions")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void NavigationMeshSourceGeometryData3D::set_projected_obstructions(const Array &p_projected_obstructions) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("set_projected_obstructions")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_projected_obstructions);
}

Array NavigationMeshSourceGeometryData3D::get_projected_obstructions() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("get_projected_obstructions")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

AABB NavigationMeshSourceGeometryData3D::get_bounds() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData3D::get_class_static()._native_ptr(), StringName("get_bounds")._native_ptr(), 1021181044);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

} // namespace godot
