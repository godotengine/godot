/**************************************************************************/
/*  physics_body2d.cpp                                                    */
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

#include <godot_cpp/classes/physics_body2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/transform2d.hpp>

namespace godot {

Ref<KinematicCollision2D> PhysicsBody2D::move_and_collide(const Vector2 &p_motion, bool p_test_only, float p_safe_margin, bool p_recovery_as_collision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody2D::get_class_static()._native_ptr(), StringName("move_and_collide")._native_ptr(), 3681923724);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<KinematicCollision2D>()));
	int8_t p_test_only_encoded;
	PtrToArg<bool>::encode(p_test_only, &p_test_only_encoded);
	double p_safe_margin_encoded;
	PtrToArg<double>::encode(p_safe_margin, &p_safe_margin_encoded);
	int8_t p_recovery_as_collision_encoded;
	PtrToArg<bool>::encode(p_recovery_as_collision, &p_recovery_as_collision_encoded);
	return Ref<KinematicCollision2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<KinematicCollision2D>(_gde_method_bind, _owner, &p_motion, &p_test_only_encoded, &p_safe_margin_encoded, &p_recovery_as_collision_encoded));
}

bool PhysicsBody2D::test_move(const Transform2D &p_from, const Vector2 &p_motion, const Ref<KinematicCollision2D> &p_collision, float p_safe_margin, bool p_recovery_as_collision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody2D::get_class_static()._native_ptr(), StringName("test_move")._native_ptr(), 3324464701);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	double p_safe_margin_encoded;
	PtrToArg<double>::encode(p_safe_margin, &p_safe_margin_encoded);
	int8_t p_recovery_as_collision_encoded;
	PtrToArg<bool>::encode(p_recovery_as_collision, &p_recovery_as_collision_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_from, &p_motion, (p_collision != nullptr ? &p_collision->_owner : nullptr), &p_safe_margin_encoded, &p_recovery_as_collision_encoded);
}

Vector2 PhysicsBody2D::get_gravity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody2D::get_class_static()._native_ptr(), StringName("get_gravity")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

TypedArray<PhysicsBody2D> PhysicsBody2D::get_collision_exceptions() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody2D::get_class_static()._native_ptr(), StringName("get_collision_exceptions")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PhysicsBody2D>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PhysicsBody2D>>(_gde_method_bind, _owner);
}

void PhysicsBody2D::add_collision_exception_with(Node *p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody2D::get_class_static()._native_ptr(), StringName("add_collision_exception_with")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_body != nullptr ? &p_body->_owner : nullptr));
}

void PhysicsBody2D::remove_collision_exception_with(Node *p_body) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsBody2D::get_class_static()._native_ptr(), StringName("remove_collision_exception_with")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_body != nullptr ? &p_body->_owner : nullptr));
}

} // namespace godot
