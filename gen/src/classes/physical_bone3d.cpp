/**************************************************************************/
/*  physical_bone3d.cpp                                                   */
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

#include <godot_cpp/classes/physical_bone3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/physics_direct_body_state3d.hpp>

namespace godot {

void PhysicalBone3D::apply_central_impulse(const Vector3 &p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("apply_central_impulse")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_impulse);
}

void PhysicalBone3D::apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("apply_impulse")._native_ptr(), 2754756483);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_impulse, &p_position);
}

void PhysicalBone3D::set_joint_type(PhysicalBone3D::JointType p_joint_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_joint_type")._native_ptr(), 2289552604);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_joint_type_encoded;
	PtrToArg<int64_t>::encode(p_joint_type, &p_joint_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_joint_type_encoded);
}

PhysicalBone3D::JointType PhysicalBone3D::get_joint_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_joint_type")._native_ptr(), 931347320);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicalBone3D::JointType(0)));
	return (PhysicalBone3D::JointType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_joint_offset(const Transform3D &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_joint_offset")._native_ptr(), 2952846383);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Transform3D PhysicalBone3D::get_joint_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_joint_offset")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_joint_rotation(const Vector3 &p_euler) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_joint_rotation")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_euler);
}

Vector3 PhysicalBone3D::get_joint_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_joint_rotation")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_body_offset(const Transform3D &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_body_offset")._native_ptr(), 2952846383);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Transform3D PhysicalBone3D::get_body_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_body_offset")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

bool PhysicalBone3D::get_simulate_physics() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_simulate_physics")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool PhysicalBone3D::is_simulating_physics() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("is_simulating_physics")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t PhysicalBone3D::get_bone_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_bone_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_mass(float p_mass) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_mass")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_mass_encoded;
	PtrToArg<double>::encode(p_mass, &p_mass_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mass_encoded);
}

float PhysicalBone3D::get_mass() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_mass")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_friction(float p_friction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_friction")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_friction_encoded;
	PtrToArg<double>::encode(p_friction, &p_friction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_friction_encoded);
}

float PhysicalBone3D::get_friction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_friction")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_bounce(float p_bounce) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_bounce")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bounce_encoded;
	PtrToArg<double>::encode(p_bounce, &p_bounce_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bounce_encoded);
}

float PhysicalBone3D::get_bounce() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_bounce")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_gravity_scale(float p_gravity_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_gravity_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_gravity_scale_encoded;
	PtrToArg<double>::encode(p_gravity_scale, &p_gravity_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gravity_scale_encoded);
}

float PhysicalBone3D::get_gravity_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_gravity_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_linear_damp_mode(PhysicalBone3D::DampMode p_linear_damp_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_linear_damp_mode")._native_ptr(), 1244972221);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_linear_damp_mode_encoded;
	PtrToArg<int64_t>::encode(p_linear_damp_mode, &p_linear_damp_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_linear_damp_mode_encoded);
}

PhysicalBone3D::DampMode PhysicalBone3D::get_linear_damp_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_linear_damp_mode")._native_ptr(), 205884699);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicalBone3D::DampMode(0)));
	return (PhysicalBone3D::DampMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_angular_damp_mode(PhysicalBone3D::DampMode p_angular_damp_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_angular_damp_mode")._native_ptr(), 1244972221);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_angular_damp_mode_encoded;
	PtrToArg<int64_t>::encode(p_angular_damp_mode, &p_angular_damp_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angular_damp_mode_encoded);
}

PhysicalBone3D::DampMode PhysicalBone3D::get_angular_damp_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_angular_damp_mode")._native_ptr(), 205884699);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PhysicalBone3D::DampMode(0)));
	return (PhysicalBone3D::DampMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_linear_damp(float p_linear_damp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_linear_damp")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_linear_damp_encoded;
	PtrToArg<double>::encode(p_linear_damp, &p_linear_damp_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_linear_damp_encoded);
}

float PhysicalBone3D::get_linear_damp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_linear_damp")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_angular_damp(float p_angular_damp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_angular_damp")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angular_damp_encoded;
	PtrToArg<double>::encode(p_angular_damp, &p_angular_damp_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angular_damp_encoded);
}

float PhysicalBone3D::get_angular_damp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_angular_damp")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_linear_velocity(const Vector3 &p_linear_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_linear_velocity")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_linear_velocity);
}

Vector3 PhysicalBone3D::get_linear_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_linear_velocity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_angular_velocity(const Vector3 &p_angular_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_angular_velocity")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angular_velocity);
}

Vector3 PhysicalBone3D::get_angular_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("get_angular_velocity")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_use_custom_integrator(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_use_custom_integrator")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool PhysicalBone3D::is_using_custom_integrator() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("is_using_custom_integrator")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PhysicalBone3D::set_can_sleep(bool p_able_to_sleep) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("set_can_sleep")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_able_to_sleep_encoded;
	PtrToArg<bool>::encode(p_able_to_sleep, &p_able_to_sleep_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_able_to_sleep_encoded);
}

bool PhysicalBone3D::is_able_to_sleep() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicalBone3D::get_class_static()._native_ptr(), StringName("is_able_to_sleep")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PhysicalBone3D::_integrate_forces(PhysicsDirectBodyState3D *p_state) {}

} // namespace godot
