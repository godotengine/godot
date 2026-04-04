/**************************************************************************/
/*  physics_test_motion_parameters3d.cpp                                  */
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

#include <godot_cpp/classes/physics_test_motion_parameters3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Transform3D PhysicsTestMotionParameters3D::get_from() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("get_from")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

void PhysicsTestMotionParameters3D::set_from(const Transform3D &p_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("set_from")._native_ptr(), 2952846383);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from);
}

Vector3 PhysicsTestMotionParameters3D::get_motion() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("get_motion")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void PhysicsTestMotionParameters3D::set_motion(const Vector3 &p_motion) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("set_motion")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_motion);
}

float PhysicsTestMotionParameters3D::get_margin() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("get_margin")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicsTestMotionParameters3D::set_margin(float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("set_margin")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_margin_encoded);
}

int32_t PhysicsTestMotionParameters3D::get_max_collisions() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("get_max_collisions")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PhysicsTestMotionParameters3D::set_max_collisions(int32_t p_max_collisions) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("set_max_collisions")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_collisions_encoded;
	PtrToArg<int64_t>::encode(p_max_collisions, &p_max_collisions_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_collisions_encoded);
}

bool PhysicsTestMotionParameters3D::is_collide_separation_ray_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("is_collide_separation_ray_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PhysicsTestMotionParameters3D::set_collide_separation_ray_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("set_collide_separation_ray_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

TypedArray<RID> PhysicsTestMotionParameters3D::get_exclude_bodies() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("get_exclude_bodies")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner);
}

void PhysicsTestMotionParameters3D::set_exclude_bodies(const TypedArray<RID> &p_exclude_list) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("set_exclude_bodies")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exclude_list);
}

TypedArray<int> PhysicsTestMotionParameters3D::get_exclude_objects() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("get_exclude_objects")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<int>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<int>>(_gde_method_bind, _owner);
}

void PhysicsTestMotionParameters3D::set_exclude_objects(const TypedArray<int> &p_exclude_list) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("set_exclude_objects")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_exclude_list);
}

bool PhysicsTestMotionParameters3D::is_recovery_as_collision_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("is_recovery_as_collision_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PhysicsTestMotionParameters3D::set_recovery_as_collision_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsTestMotionParameters3D::get_class_static()._native_ptr(), StringName("set_recovery_as_collision_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

} // namespace godot
