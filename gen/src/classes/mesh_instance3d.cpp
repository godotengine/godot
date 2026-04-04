/**************************************************************************/
/*  mesh_instance3d.cpp                                                   */
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

#include <godot_cpp/classes/mesh_instance3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/skin.hpp>
#include <godot_cpp/classes/skin_reference.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void MeshInstance3D::set_mesh(const Ref<Mesh> &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("set_mesh")._native_ptr(), 194775623);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_mesh != nullptr ? &p_mesh->_owner : nullptr));
}

Ref<Mesh> MeshInstance3D::get_mesh() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("get_mesh")._native_ptr(), 1808005922);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Mesh>()));
	return Ref<Mesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Mesh>(_gde_method_bind, _owner));
}

void MeshInstance3D::set_skeleton_path(const NodePath &p_skeleton_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("set_skeleton_path")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton_path);
}

NodePath MeshInstance3D::get_skeleton_path() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("get_skeleton_path")._native_ptr(), 277076166);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void MeshInstance3D::set_skin(const Ref<Skin> &p_skin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("set_skin")._native_ptr(), 3971435618);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_skin != nullptr ? &p_skin->_owner : nullptr));
}

Ref<Skin> MeshInstance3D::get_skin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("get_skin")._native_ptr(), 2074563878);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Skin>()));
	return Ref<Skin>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Skin>(_gde_method_bind, _owner));
}

Ref<SkinReference> MeshInstance3D::get_skin_reference() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("get_skin_reference")._native_ptr(), 2060603409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SkinReference>()));
	return Ref<SkinReference>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SkinReference>(_gde_method_bind, _owner));
}

int32_t MeshInstance3D::get_surface_override_material_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("get_surface_override_material_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MeshInstance3D::set_surface_override_material(int32_t p_surface, const Ref<Material> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("set_surface_override_material")._native_ptr(), 3671737478);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_surface_encoded, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Ref<Material> MeshInstance3D::get_surface_override_material(int32_t p_surface) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("get_surface_override_material")._native_ptr(), 2897466400);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner, &p_surface_encoded));
}

Ref<Material> MeshInstance3D::get_active_material(int32_t p_surface) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("get_active_material")._native_ptr(), 2897466400);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Material>()));
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	return Ref<Material>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Material>(_gde_method_bind, _owner, &p_surface_encoded));
}

void MeshInstance3D::create_trimesh_collision() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("create_trimesh_collision")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void MeshInstance3D::create_convex_collision(bool p_clean, bool p_simplify) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("create_convex_collision")._native_ptr(), 2751962654);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_clean_encoded;
	PtrToArg<bool>::encode(p_clean, &p_clean_encoded);
	int8_t p_simplify_encoded;
	PtrToArg<bool>::encode(p_simplify, &p_simplify_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_clean_encoded, &p_simplify_encoded);
}

void MeshInstance3D::create_multiple_convex_collisions(const Ref<MeshConvexDecompositionSettings> &p_settings) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("create_multiple_convex_collisions")._native_ptr(), 628789669);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_settings != nullptr ? &p_settings->_owner : nullptr));
}

int32_t MeshInstance3D::get_blend_shape_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("get_blend_shape_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t MeshInstance3D::find_blend_shape_by_name(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("find_blend_shape_by_name")._native_ptr(), 4150868206);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name);
}

float MeshInstance3D::get_blend_shape_value(int32_t p_blend_shape_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("get_blend_shape_value")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_blend_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_blend_shape_idx, &p_blend_shape_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_blend_shape_idx_encoded);
}

void MeshInstance3D::set_blend_shape_value(int32_t p_blend_shape_idx, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("set_blend_shape_value")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_blend_shape_idx_encoded;
	PtrToArg<int64_t>::encode(p_blend_shape_idx, &p_blend_shape_idx_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_blend_shape_idx_encoded, &p_value_encoded);
}

void MeshInstance3D::create_debug_tangents() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("create_debug_tangents")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Ref<ArrayMesh> MeshInstance3D::bake_mesh_from_current_blend_shape_mix(const Ref<ArrayMesh> &p_existing) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("bake_mesh_from_current_blend_shape_mix")._native_ptr(), 1457573577);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ArrayMesh>()));
	return Ref<ArrayMesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ArrayMesh>(_gde_method_bind, _owner, (p_existing != nullptr ? &p_existing->_owner : nullptr)));
}

Ref<ArrayMesh> MeshInstance3D::bake_mesh_from_current_skeleton_pose(const Ref<ArrayMesh> &p_existing) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MeshInstance3D::get_class_static()._native_ptr(), StringName("bake_mesh_from_current_skeleton_pose")._native_ptr(), 1457573577);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ArrayMesh>()));
	return Ref<ArrayMesh>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ArrayMesh>(_gde_method_bind, _owner, (p_existing != nullptr ? &p_existing->_owner : nullptr)));
}

} // namespace godot
