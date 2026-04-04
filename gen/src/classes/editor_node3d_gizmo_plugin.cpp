/**************************************************************************/
/*  editor_node3d_gizmo_plugin.cpp                                        */
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

#include <godot_cpp/classes/editor_node3d_gizmo_plugin.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/standard_material3d.hpp>
#include <godot_cpp/variant/plane.hpp>
#include <godot_cpp/variant/vector2.hpp>

namespace godot {

void EditorNode3DGizmoPlugin::create_material(const String &p_name, const Color &p_color, bool p_billboard, bool p_on_top, bool p_use_vertex_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmoPlugin::get_class_static()._native_ptr(), StringName("create_material")._native_ptr(), 3486012546);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_billboard_encoded;
	PtrToArg<bool>::encode(p_billboard, &p_billboard_encoded);
	int8_t p_on_top_encoded;
	PtrToArg<bool>::encode(p_on_top, &p_on_top_encoded);
	int8_t p_use_vertex_color_encoded;
	PtrToArg<bool>::encode(p_use_vertex_color, &p_use_vertex_color_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_color, &p_billboard_encoded, &p_on_top_encoded, &p_use_vertex_color_encoded);
}

void EditorNode3DGizmoPlugin::create_icon_material(const String &p_name, const Ref<Texture2D> &p_texture, bool p_on_top, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmoPlugin::get_class_static()._native_ptr(), StringName("create_icon_material")._native_ptr(), 3804976916);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_on_top_encoded;
	PtrToArg<bool>::encode(p_on_top, &p_on_top_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_on_top_encoded, &p_color);
}

void EditorNode3DGizmoPlugin::create_handle_material(const String &p_name, bool p_billboard, const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmoPlugin::get_class_static()._native_ptr(), StringName("create_handle_material")._native_ptr(), 2486475223);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_billboard_encoded;
	PtrToArg<bool>::encode(p_billboard, &p_billboard_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_billboard_encoded, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

void EditorNode3DGizmoPlugin::add_material(const String &p_name, const Ref<StandardMaterial3D> &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmoPlugin::get_class_static()._native_ptr(), StringName("add_material")._native_ptr(), 1374068695);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_material != nullptr ? &p_material->_owner : nullptr));
}

Ref<StandardMaterial3D> EditorNode3DGizmoPlugin::get_material(const String &p_name, const Ref<EditorNode3DGizmo> &p_gizmo) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorNode3DGizmoPlugin::get_class_static()._native_ptr(), StringName("get_material")._native_ptr(), 974464017);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<StandardMaterial3D>()));
	return Ref<StandardMaterial3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<StandardMaterial3D>(_gde_method_bind, _owner, &p_name, (p_gizmo != nullptr ? &p_gizmo->_owner : nullptr)));
}

bool EditorNode3DGizmoPlugin::_has_gizmo(Node3D *p_for_node_3d) const {
	return false;
}

Ref<EditorNode3DGizmo> EditorNode3DGizmoPlugin::_create_gizmo(Node3D *p_for_node_3d) const {
	return Ref<EditorNode3DGizmo>();
}

String EditorNode3DGizmoPlugin::_get_gizmo_name() const {
	return String();
}

int32_t EditorNode3DGizmoPlugin::_get_priority() const {
	return 0;
}

bool EditorNode3DGizmoPlugin::_can_be_hidden() const {
	return false;
}

bool EditorNode3DGizmoPlugin::_is_selectable_when_hidden() const {
	return false;
}

void EditorNode3DGizmoPlugin::_redraw(const Ref<EditorNode3DGizmo> &p_gizmo) {}

String EditorNode3DGizmoPlugin::_get_handle_name(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_handle_id, bool p_secondary) const {
	return String();
}

bool EditorNode3DGizmoPlugin::_is_handle_highlighted(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_handle_id, bool p_secondary) const {
	return false;
}

Variant EditorNode3DGizmoPlugin::_get_handle_value(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_handle_id, bool p_secondary) const {
	return Variant();
}

void EditorNode3DGizmoPlugin::_begin_handle_action(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_handle_id, bool p_secondary) {}

void EditorNode3DGizmoPlugin::_set_handle(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_handle_id, bool p_secondary, Camera3D *p_camera, const Vector2 &p_screen_pos) {}

void EditorNode3DGizmoPlugin::_commit_handle(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_handle_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {}

int32_t EditorNode3DGizmoPlugin::_subgizmos_intersect_ray(const Ref<EditorNode3DGizmo> &p_gizmo, Camera3D *p_camera, const Vector2 &p_screen_pos) const {
	return 0;
}

PackedInt32Array EditorNode3DGizmoPlugin::_subgizmos_intersect_frustum(const Ref<EditorNode3DGizmo> &p_gizmo, Camera3D *p_camera, const TypedArray<Plane> &p_frustum_planes) const {
	return PackedInt32Array();
}

Transform3D EditorNode3DGizmoPlugin::_get_subgizmo_transform(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_subgizmo_id) const {
	return Transform3D();
}

void EditorNode3DGizmoPlugin::_set_subgizmo_transform(const Ref<EditorNode3DGizmo> &p_gizmo, int32_t p_subgizmo_id, const Transform3D &p_transform) {}

void EditorNode3DGizmoPlugin::_commit_subgizmos(const Ref<EditorNode3DGizmo> &p_gizmo, const PackedInt32Array &p_ids, const TypedArray<Transform3D> &p_restores, bool p_cancel) {}

} // namespace godot
