/**************************************************************************/
/*  node3d.cpp                                                            */
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

#include <godot_cpp/classes/node3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node3d_gizmo.hpp>
#include <godot_cpp/classes/world3d.hpp>

namespace godot {

void Node3D::set_transform(const Transform3D &p_local) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_transform")._native_ptr(), 2952846383);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_local);
}

Transform3D Node3D::get_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_transform")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

void Node3D::set_position(const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_position")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

Vector3 Node3D::get_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_position")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void Node3D::set_rotation(const Vector3 &p_euler_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_rotation")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_euler_radians);
}

Vector3 Node3D::get_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_rotation")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void Node3D::set_rotation_degrees(const Vector3 &p_euler_degrees) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_rotation_degrees")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_euler_degrees);
}

Vector3 Node3D::get_rotation_degrees() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_rotation_degrees")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void Node3D::set_rotation_order(EulerOrder p_order) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_rotation_order")._native_ptr(), 1820889989);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_order_encoded;
	PtrToArg<int64_t>::encode(p_order, &p_order_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_order_encoded);
}

EulerOrder Node3D::get_rotation_order() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_rotation_order")._native_ptr(), 916939469);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (EulerOrder(0)));
	return (EulerOrder)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Node3D::set_rotation_edit_mode(Node3D::RotationEditMode p_edit_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_rotation_edit_mode")._native_ptr(), 141483330);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_edit_mode_encoded;
	PtrToArg<int64_t>::encode(p_edit_mode, &p_edit_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_edit_mode_encoded);
}

Node3D::RotationEditMode Node3D::get_rotation_edit_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_rotation_edit_mode")._native_ptr(), 1572188370);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Node3D::RotationEditMode(0)));
	return (Node3D::RotationEditMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Node3D::set_scale(const Vector3 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_scale")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale);
}

Vector3 Node3D::get_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_scale")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void Node3D::set_quaternion(const Quaternion &p_quaternion) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_quaternion")._native_ptr(), 1727505552);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quaternion);
}

Quaternion Node3D::get_quaternion() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_quaternion")._native_ptr(), 1222331677);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Quaternion()));
	return ::godot::internal::_call_native_mb_ret<Quaternion>(_gde_method_bind, _owner);
}

void Node3D::set_basis(const Basis &p_basis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_basis")._native_ptr(), 1055510324);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_basis);
}

Basis Node3D::get_basis() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_basis")._native_ptr(), 2716978435);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Basis()));
	return ::godot::internal::_call_native_mb_ret<Basis>(_gde_method_bind, _owner);
}

void Node3D::set_global_transform(const Transform3D &p_global) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_global_transform")._native_ptr(), 2952846383);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_global);
}

Transform3D Node3D::get_global_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_global_transform")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

Transform3D Node3D::get_global_transform_interpolated() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_global_transform_interpolated")._native_ptr(), 4183770049);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

void Node3D::set_global_position(const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_global_position")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

Vector3 Node3D::get_global_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_global_position")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void Node3D::set_global_basis(const Basis &p_basis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_global_basis")._native_ptr(), 1055510324);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_basis);
}

Basis Node3D::get_global_basis() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_global_basis")._native_ptr(), 2716978435);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Basis()));
	return ::godot::internal::_call_native_mb_ret<Basis>(_gde_method_bind, _owner);
}

void Node3D::set_global_rotation(const Vector3 &p_euler_radians) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_global_rotation")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_euler_radians);
}

Vector3 Node3D::get_global_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_global_rotation")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void Node3D::set_global_rotation_degrees(const Vector3 &p_euler_degrees) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_global_rotation_degrees")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_euler_degrees);
}

Vector3 Node3D::get_global_rotation_degrees() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_global_rotation_degrees")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Node3D *Node3D::get_parent_node_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_parent_node_3d")._native_ptr(), 151077316);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node3D>(_gde_method_bind, _owner);
}

void Node3D::set_ignore_transform_notification(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_ignore_transform_notification")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

void Node3D::set_as_top_level(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_as_top_level")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node3D::is_set_as_top_level() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("is_set_as_top_level")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node3D::set_disable_scale(bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_disable_scale")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_disable_encoded);
}

bool Node3D::is_scale_disabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("is_scale_disabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<World3D> Node3D::get_world_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_world_3d")._native_ptr(), 317588385);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<World3D>()));
	return Ref<World3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<World3D>(_gde_method_bind, _owner));
}

void Node3D::force_update_transform() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("force_update_transform")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node3D::set_visibility_parent(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_visibility_parent")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

NodePath Node3D::get_visibility_parent() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_visibility_parent")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void Node3D::update_gizmos() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("update_gizmos")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node3D::add_gizmo(const Ref<Node3DGizmo> &p_gizmo) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("add_gizmo")._native_ptr(), 1544533845);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_gizmo != nullptr ? &p_gizmo->_owner : nullptr));
}

TypedArray<Ref<Node3DGizmo>> Node3D::get_gizmos() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("get_gizmos")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Node3DGizmo>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Node3DGizmo>>>(_gde_method_bind, _owner);
}

void Node3D::clear_gizmos() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("clear_gizmos")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node3D::set_subgizmo_selection(const Ref<Node3DGizmo> &p_gizmo, int32_t p_id, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_subgizmo_selection")._native_ptr(), 3317607635);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_gizmo != nullptr ? &p_gizmo->_owner : nullptr), &p_id_encoded, &p_transform);
}

void Node3D::clear_subgizmo_selection() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("clear_subgizmo_selection")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node3D::set_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool Node3D::is_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("is_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Node3D::is_visible_in_tree() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("is_visible_in_tree")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node3D::show() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("show")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node3D::hide() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("hide")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node3D::set_notify_local_transform(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_notify_local_transform")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node3D::is_local_transform_notification_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("is_local_transform_notification_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node3D::set_notify_transform(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_notify_transform")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Node3D::is_transform_notification_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("is_transform_notification_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Node3D::rotate(const Vector3 &p_axis, float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("rotate")._native_ptr(), 3436291937);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis, &p_angle_encoded);
}

void Node3D::global_rotate(const Vector3 &p_axis, float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("global_rotate")._native_ptr(), 3436291937);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis, &p_angle_encoded);
}

void Node3D::global_scale(const Vector3 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("global_scale")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale);
}

void Node3D::global_translate(const Vector3 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("global_translate")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

void Node3D::rotate_object_local(const Vector3 &p_axis, float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("rotate_object_local")._native_ptr(), 3436291937);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis, &p_angle_encoded);
}

void Node3D::scale_object_local(const Vector3 &p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("scale_object_local")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale);
}

void Node3D::translate_object_local(const Vector3 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("translate_object_local")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

void Node3D::rotate_x(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("rotate_x")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

void Node3D::rotate_y(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("rotate_y")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

void Node3D::rotate_z(float p_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("rotate_z")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angle_encoded;
	PtrToArg<double>::encode(p_angle, &p_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angle_encoded);
}

void Node3D::translate(const Vector3 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("translate")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

void Node3D::orthonormalize() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("orthonormalize")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node3D::set_identity() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("set_identity")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Node3D::look_at(const Vector3 &p_target, const Vector3 &p_up, bool p_use_model_front) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("look_at")._native_ptr(), 2882425029);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_model_front_encoded;
	PtrToArg<bool>::encode(p_use_model_front, &p_use_model_front_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_target, &p_up, &p_use_model_front_encoded);
}

void Node3D::look_at_from_position(const Vector3 &p_position, const Vector3 &p_target, const Vector3 &p_up, bool p_use_model_front) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("look_at_from_position")._native_ptr(), 2086826090);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_model_front_encoded;
	PtrToArg<bool>::encode(p_use_model_front, &p_use_model_front_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position, &p_target, &p_up, &p_use_model_front_encoded);
}

Vector3 Node3D::to_local(const Vector3 &p_global_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("to_local")._native_ptr(), 192990374);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_global_point);
}

Vector3 Node3D::to_global(const Vector3 &p_local_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Node3D::get_class_static()._native_ptr(), StringName("to_global")._native_ptr(), 192990374);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_local_point);
}

} // namespace godot
