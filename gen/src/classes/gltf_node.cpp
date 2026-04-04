/**************************************************************************/
/*  gltf_node.cpp                                                         */
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

#include <godot_cpp/classes/gltf_node.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/gltf_state.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

String GLTFNode::get_original_name() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_original_name")._native_ptr(), 2841200299);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void GLTFNode::set_original_name(const String &p_original_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_original_name")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_original_name);
}

int32_t GLTFNode::get_parent() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_parent")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFNode::set_parent(int32_t p_parent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_parent")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_parent_encoded;
	PtrToArg<int64_t>::encode(p_parent, &p_parent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_parent_encoded);
}

int32_t GLTFNode::get_height() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_height")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFNode::set_height(int32_t p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_height")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_encoded);
}

Transform3D GLTFNode::get_xform() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_xform")._native_ptr(), 4183770049);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

void GLTFNode::set_xform(const Transform3D &p_xform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_xform")._native_ptr(), 2952846383);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_xform);
}

int32_t GLTFNode::get_mesh() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_mesh")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFNode::set_mesh(int32_t p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_mesh")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mesh_encoded;
	PtrToArg<int64_t>::encode(p_mesh, &p_mesh_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh_encoded);
}

int32_t GLTFNode::get_camera() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_camera")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFNode::set_camera(int32_t p_camera) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_camera")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_camera_encoded;
	PtrToArg<int64_t>::encode(p_camera, &p_camera_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera_encoded);
}

int32_t GLTFNode::get_skin() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_skin")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFNode::set_skin(int32_t p_skin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_skin")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_skin_encoded;
	PtrToArg<int64_t>::encode(p_skin, &p_skin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skin_encoded);
}

int32_t GLTFNode::get_skeleton() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_skeleton")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFNode::set_skeleton(int32_t p_skeleton) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_skeleton")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_skeleton_encoded;
	PtrToArg<int64_t>::encode(p_skeleton, &p_skeleton_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton_encoded);
}

Vector3 GLTFNode::get_position() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_position")._native_ptr(), 3783033775);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void GLTFNode::set_position(const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_position")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

Quaternion GLTFNode::get_rotation() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_rotation")._native_ptr(), 2916281908);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Quaternion()));
	return ::godot::internal::_call_native_mb_ret<Quaternion>(_gde_method_bind, _owner);
}

void GLTFNode::set_rotation(const Quaternion &p_rotation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_rotation")._native_ptr(), 1727505552);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rotation);
}

Vector3 GLTFNode::get_scale() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_scale")._native_ptr(), 3783033775);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void GLTFNode::set_scale(const Vector3 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_scale")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale);
}

PackedInt32Array GLTFNode::get_children() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_children")._native_ptr(), 969006518);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner);
}

void GLTFNode::set_children(const PackedInt32Array &p_children) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_children")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_children);
}

void GLTFNode::append_child_index(int32_t p_child_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("append_child_index")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_child_index_encoded;
	PtrToArg<int64_t>::encode(p_child_index, &p_child_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_child_index_encoded);
}

int32_t GLTFNode::get_light() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_light")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFNode::set_light(int32_t p_light) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_light")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_light_encoded;
	PtrToArg<int64_t>::encode(p_light, &p_light_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light_encoded);
}

bool GLTFNode::get_visible() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_visible")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GLTFNode::set_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

Variant GLTFNode::get_additional_data(const StringName &p_extension_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_additional_data")._native_ptr(), 2138907829);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_extension_name);
}

void GLTFNode::set_additional_data(const StringName &p_extension_name, const Variant &p_additional_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("set_additional_data")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_extension_name, &p_additional_data);
}

NodePath GLTFNode::get_scene_node_path(const Ref<GLTFState> &p_gltf_state, bool p_handle_skeletons) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFNode::get_class_static()._native_ptr(), StringName("get_scene_node_path")._native_ptr(), 573359477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int8_t p_handle_skeletons_encoded;
	PtrToArg<bool>::encode(p_handle_skeletons, &p_handle_skeletons_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, (p_gltf_state != nullptr ? &p_gltf_state->_owner : nullptr), &p_handle_skeletons_encoded);
}

} // namespace godot
