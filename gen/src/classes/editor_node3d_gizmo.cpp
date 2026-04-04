/**************************************************************************/
/*  editor_node3d_gizmo.cpp                                               */
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

#include <godot_cpp/classes/editor_node3d_gizmo.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/editor_node3d_gizmo_plugin.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/triangle_mesh.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/plane.hpp>
#include <godot_cpp/variant/vector2.hpp>

namespace godot {

void EditorNode3DGizmo::add_lines(const PackedVector3Array &p_lines, const Ref<Material> &p_material, bool p_billboard, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("add_lines")._native_ptr(), 2910971437);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_billboard_encoded;
	PtrToArg<bool>::encode(p_billboard, &p_billboard_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lines, (p_material != nullptr ? &p_material->_owner : nullptr), &p_billboard_encoded, &p_modulate);
}

void EditorNode3DGizmo::add_mesh(const Ref<Mesh> &p_mesh, const Ref<Material> &p_material, const Transform3D &p_transform, const Ref<SkinReference> &p_skeleton) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("add_mesh")._native_ptr(), 1579955111);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_mesh != nullptr ? &p_mesh->_owner : nullptr), (p_material != nullptr ? &p_material->_owner : nullptr), &p_transform, (p_skeleton != nullptr ? &p_skeleton->_owner : nullptr));
}

void EditorNode3DGizmo::add_collision_segments(const PackedVector3Array &p_segments) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("add_collision_segments")._native_ptr(), 334873810);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_segments);
}

void EditorNode3DGizmo::add_collision_triangles(const Ref<TriangleMesh> &p_triangles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("add_collision_triangles")._native_ptr(), 54901064);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_triangles != nullptr ? &p_triangles->_owner : nullptr));
}

void EditorNode3DGizmo::add_unscaled_billboard(const Ref<Material> &p_material, float p_default_scale, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("add_unscaled_billboard")._native_ptr(), 520007164);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_default_scale_encoded;
	PtrToArg<double>::encode(p_default_scale, &p_default_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_material != nullptr ? &p_material->_owner : nullptr), &p_default_scale_encoded, &p_modulate);
}

void EditorNode3DGizmo::add_handles(const PackedVector3Array &p_handles, const Ref<Material> &p_material, const PackedInt32Array &p_ids, bool p_billboard, bool p_secondary) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("add_handles")._native_ptr(), 2254560097);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_billboard_encoded;
	PtrToArg<bool>::encode(p_billboard, &p_billboard_encoded);
	int8_t p_secondary_encoded;
	PtrToArg<bool>::encode(p_secondary, &p_secondary_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_handles, (p_material != nullptr ? &p_material->_owner : nullptr), &p_ids, &p_billboard_encoded, &p_secondary_encoded);
}

void EditorNode3DGizmo::set_node_3d(Node *p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("set_node_3d")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

Node3D *EditorNode3DGizmo::get_node_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("get_node_3d")._native_ptr(), 151077316);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node3D>(_gde_method_bind, _owner);
}

Ref<EditorNode3DGizmoPlugin> EditorNode3DGizmo::get_plugin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("get_plugin")._native_ptr(), 4250544552);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<EditorNode3DGizmoPlugin>()));
	return Ref<EditorNode3DGizmoPlugin>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<EditorNode3DGizmoPlugin>(_gde_method_bind, _owner));
}

void EditorNode3DGizmo::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorNode3DGizmo::set_hidden(bool p_hidden) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("set_hidden")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_hidden_encoded;
	PtrToArg<bool>::encode(p_hidden, &p_hidden_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hidden_encoded);
}

bool EditorNode3DGizmo::is_subgizmo_selected(int32_t p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("is_subgizmo_selected")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_id_encoded);
}

PackedInt32Array EditorNode3DGizmo::get_subgizmo_selection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmo::get_class_static()._native_ptr(), StringName("get_subgizmo_selection")._native_ptr(), 1930428628);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void EditorNode3DGizmo::_redraw() {}

String EditorNode3DGizmo::_get_handle_name(int32_t p_id, bool p_secondary) const {
	return String();
}

bool EditorNode3DGizmo::_is_handle_highlighted(int32_t p_id, bool p_secondary) const {
	return false;
}

Variant EditorNode3DGizmo::_get_handle_value(int32_t p_id, bool p_secondary) const {
	return Variant();
}

void EditorNode3DGizmo::_begin_handle_action(int32_t p_id, bool p_secondary) {}

void EditorNode3DGizmo::_set_handle(int32_t p_id, bool p_secondary, Camera3D *p_camera, const Vector2 &p_point) {}

void EditorNode3DGizmo::_commit_handle(int32_t p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {}

int32_t EditorNode3DGizmo::_subgizmos_intersect_ray(Camera3D *p_camera, const Vector2 &p_point) const {
	return 0;
}

PackedInt32Array EditorNode3DGizmo::_subgizmos_intersect_frustum(Camera3D *p_camera, const TypedArray<Plane> &p_frustum) const {
	return PackedInt32Array();
}

void EditorNode3DGizmo::_set_subgizmo_transform(int32_t p_id, const Transform3D &p_transform) {}

Transform3D EditorNode3DGizmo::_get_subgizmo_transform(int32_t p_id) const {
	return Transform3D();
}

void EditorNode3DGizmo::_commit_subgizmos(const PackedInt32Array &p_ids, const TypedArray<Transform3D> &p_restores, bool p_cancel) {}

} // namespace godot
