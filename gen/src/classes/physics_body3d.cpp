/**************************************************************************/
/*  physics_body3d.cpp                                                    */
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

#include <godot_cpp/classes/physics_body3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/transform3d.hpp>

namespace godot {

Ref<KinematicCollision3D> PhysicsBody3D::move_and_collide(const Vector3 &p_motion, bool p_test_only, float p_safe_margin, bool p_recovery_as_collision, int32_t p_max_collisions) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody3D::get_class_static()._native_ptr(), StringName("move_and_collide")._native_ptr(), 3208792678);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<KinematicCollision3D>()));
	int8_t p_test_only_encoded;
	PtrToArg<bool>::encode(p_test_only, &p_test_only_encoded);
	double p_safe_margin_encoded;
	PtrToArg<double>::encode(p_safe_margin, &p_safe_margin_encoded);
	int8_t p_recovery_as_collision_encoded;
	PtrToArg<bool>::encode(p_recovery_as_collision, &p_recovery_as_collision_encoded);
	int64_t p_max_collisions_encoded;
	PtrToArg<int64_t>::encode(p_max_collisions, &p_max_collisions_encoded);
	return Ref<KinematicCollision3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<KinematicCollision3D>(_gde_method_bind, _owner, &p_motion, &p_test_only_encoded, &p_safe_margin_encoded, &p_recovery_as_collision_encoded, &p_max_collisions_encoded));
}

bool PhysicsBody3D::test_move(const Transform3D &p_from, const Vector3 &p_motion, const Ref<KinematicCollision3D> &p_collision, float p_safe_margin, bool p_recovery_as_collision, int32_t p_max_collisions) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody3D::get_class_static()._native_ptr(), StringName("test_move")._native_ptr(), 2481691619);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	double p_safe_margin_encoded;
	PtrToArg<double>::encode(p_safe_margin, &p_safe_margin_encoded);
	int8_t p_recovery_as_collision_encoded;
	PtrToArg<bool>::encode(p_recovery_as_collision, &p_recovery_as_collision_encoded);
	int64_t p_max_collisions_encoded;
	PtrToArg<int64_t>::encode(p_max_collisions, &p_max_collisions_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_from, &p_motion, (p_collision != nullptr ? &p_collision->_owner : nullptr), &p_safe_margin_encoded, &p_recovery_as_collision_encoded, &p_max_collisions_encoded);
}

Vector3 PhysicsBody3D::get_gravity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody3D::get_class_static()._native_ptr(), StringName("get_gravity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void PhysicsBody3D::set_axis_lock(PhysicsServer3D::BodyAxis p_axis, bool p_lock) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody3D::get_class_static()._native_ptr(), StringName("set_axis_lock")._native_ptr(), 1787895195);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	int8_t p_lock_encoded;
	PtrToArg<bool>::encode(p_lock, &p_lock_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis_encoded, &p_lock_encoded);
}

bool PhysicsBody3D::get_axis_lock(PhysicsServer3D::BodyAxis p_axis) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody3D::get_class_static()._native_ptr(), StringName("get_axis_lock")._native_ptr(), 2264617709);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_axis_encoded);
}

TypedArray<PhysicsBody3D> PhysicsBody3D::get_collision_exceptions() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody3D::get_class_static()._native_ptr(), StringName("get_collision_exceptions")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PhysicsBody3D>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PhysicsBody3D>>(_gde_method_bind, _owner);
}

void PhysicsBody3D::add_collision_exception_with(Node *p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody3D::get_class_static()._native_ptr(), StringName("add_collision_exception_with")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_body != nullptr ? &p_body->_owner : nullptr));
}

void PhysicsBody3D::remove_collision_exception_with(Node *p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody3D::get_class_static()._native_ptr(), StringName("remove_collision_exception_with")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_body != nullptr ? &p_body->_owner : nullptr));
}

} // namespace godot
