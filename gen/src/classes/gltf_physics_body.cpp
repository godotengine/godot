/**************************************************************************/
/*  gltf_physics_body.cpp                                                 */
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

#include <godot_cpp/classes/gltf_physics_body.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/collision_object3d.hpp>

namespace godot {

Ref<GLTFPhysicsBody> GLTFPhysicsBody::from_node(CollisionObject3D *p_body_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("from_node")._native_ptr(), 420544174);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFPhysicsBody>()));
	return Ref<GLTFPhysicsBody>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFPhysicsBody>(_gde_method_bind, nullptr, (p_body_node != nullptr ? &p_body_node->_owner : nullptr)));
}

CollisionObject3D *GLTFPhysicsBody::to_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("to_node")._native_ptr(), 3224013656);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<CollisionObject3D>(_gde_method_bind, _owner);
}

Ref<GLTFPhysicsBody> GLTFPhysicsBody::from_dictionary(const Dictionary &p_dictionary) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("from_dictionary")._native_ptr(), 1177544336);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFPhysicsBody>()));
	return Ref<GLTFPhysicsBody>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFPhysicsBody>(_gde_method_bind, nullptr, &p_dictionary));
}

Dictionary GLTFPhysicsBody::to_dictionary() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("to_dictionary")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

String GLTFPhysicsBody::get_body_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("get_body_type")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void GLTFPhysicsBody::set_body_type(const String &p_body_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("set_body_type")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_body_type);
}

float GLTFPhysicsBody::get_mass() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("get_mass")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFPhysicsBody::set_mass(float p_mass) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("set_mass")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_mass_encoded;
	PtrToArg<double>::encode(p_mass, &p_mass_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mass_encoded);
}

Vector3 GLTFPhysicsBody::get_linear_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("get_linear_velocity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void GLTFPhysicsBody::set_linear_velocity(const Vector3 &p_linear_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("set_linear_velocity")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_linear_velocity);
}

Vector3 GLTFPhysicsBody::get_angular_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("get_angular_velocity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void GLTFPhysicsBody::set_angular_velocity(const Vector3 &p_angular_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("set_angular_velocity")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angular_velocity);
}

Vector3 GLTFPhysicsBody::get_center_of_mass() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("get_center_of_mass")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void GLTFPhysicsBody::set_center_of_mass(const Vector3 &p_center_of_mass) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("set_center_of_mass")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_center_of_mass);
}

Vector3 GLTFPhysicsBody::get_inertia_diagonal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("get_inertia_diagonal")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void GLTFPhysicsBody::set_inertia_diagonal(const Vector3 &p_inertia_diagonal) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("set_inertia_diagonal")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_inertia_diagonal);
}

Quaternion GLTFPhysicsBody::get_inertia_orientation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("get_inertia_orientation")._native_ptr(), 1222331677);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Quaternion()));
	return ::godot::internal::_call_native_mb_ret<Quaternion>(_gde_method_bind, _owner);
}

void GLTFPhysicsBody::set_inertia_orientation(const Quaternion &p_inertia_orientation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("set_inertia_orientation")._native_ptr(), 1727505552);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_inertia_orientation);
}

Basis GLTFPhysicsBody::get_inertia_tensor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("get_inertia_tensor")._native_ptr(), 2716978435);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Basis()));
	return ::godot::internal::_call_native_mb_ret<Basis>(_gde_method_bind, _owner);
}

void GLTFPhysicsBody::set_inertia_tensor(const Basis &p_inertia_tensor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFPhysicsBody::get_class_static()._native_ptr(), StringName("set_inertia_tensor")._native_ptr(), 1055510324);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_inertia_tensor);
}

} // namespace godot
