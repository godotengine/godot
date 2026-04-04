/**************************************************************************/
/*  rigid_body2d.cpp                                                      */
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

#include <godot_cpp/classes/rigid_body2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/classes/physics_direct_body_state2d.hpp>
#include <godot_cpp/classes/physics_material.hpp>

namespace godot {

void RigidBody2D::set_mass(float p_mass) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_mass")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_mass_encoded;
	PtrToArg<double>::encode(p_mass, &p_mass_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mass_encoded);
}

float RigidBody2D::get_mass() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_mass")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float RigidBody2D::get_inertia() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_inertia")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RigidBody2D::set_inertia(float p_inertia) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_inertia")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_inertia_encoded;
	PtrToArg<double>::encode(p_inertia, &p_inertia_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_inertia_encoded);
}

void RigidBody2D::set_center_of_mass_mode(RigidBody2D::CenterOfMassMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_center_of_mass_mode")._native_ptr(), 1757235706);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

RigidBody2D::CenterOfMassMode RigidBody2D::get_center_of_mass_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_center_of_mass_mode")._native_ptr(), 3277132817);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RigidBody2D::CenterOfMassMode(0)));
	return (RigidBody2D::CenterOfMassMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_center_of_mass(const Vector2 &p_center_of_mass) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_center_of_mass")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_center_of_mass);
}

Vector2 RigidBody2D::get_center_of_mass() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_center_of_mass")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void RigidBody2D::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_physics_material_override")._native_ptr(), 1784508650);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_physics_material_override != nullptr ? &p_physics_material_override->_owner : nullptr));
}

Ref<PhysicsMaterial> RigidBody2D::get_physics_material_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_physics_material_override")._native_ptr(), 2521850424);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PhysicsMaterial>()));
	return Ref<PhysicsMaterial>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PhysicsMaterial>(_gde_method_bind, _owner));
}

void RigidBody2D::set_gravity_scale(float p_gravity_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_gravity_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_gravity_scale_encoded;
	PtrToArg<double>::encode(p_gravity_scale, &p_gravity_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gravity_scale_encoded);
}

float RigidBody2D::get_gravity_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_gravity_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RigidBody2D::set_linear_damp_mode(RigidBody2D::DampMode p_linear_damp_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_linear_damp_mode")._native_ptr(), 3406533708);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_linear_damp_mode_encoded;
	PtrToArg<int64_t>::encode(p_linear_damp_mode, &p_linear_damp_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_linear_damp_mode_encoded);
}

RigidBody2D::DampMode RigidBody2D::get_linear_damp_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_linear_damp_mode")._native_ptr(), 2970511462);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RigidBody2D::DampMode(0)));
	return (RigidBody2D::DampMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_angular_damp_mode(RigidBody2D::DampMode p_angular_damp_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_angular_damp_mode")._native_ptr(), 3406533708);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_angular_damp_mode_encoded;
	PtrToArg<int64_t>::encode(p_angular_damp_mode, &p_angular_damp_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angular_damp_mode_encoded);
}

RigidBody2D::DampMode RigidBody2D::get_angular_damp_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_angular_damp_mode")._native_ptr(), 2970511462);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RigidBody2D::DampMode(0)));
	return (RigidBody2D::DampMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_linear_damp(float p_linear_damp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_linear_damp")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_linear_damp_encoded;
	PtrToArg<double>::encode(p_linear_damp, &p_linear_damp_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_linear_damp_encoded);
}

float RigidBody2D::get_linear_damp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_linear_damp")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RigidBody2D::set_angular_damp(float p_angular_damp) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_angular_damp")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angular_damp_encoded;
	PtrToArg<double>::encode(p_angular_damp, &p_angular_damp_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angular_damp_encoded);
}

float RigidBody2D::get_angular_damp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_angular_damp")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RigidBody2D::set_linear_velocity(const Vector2 &p_linear_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_linear_velocity")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_linear_velocity);
}

Vector2 RigidBody2D::get_linear_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_linear_velocity")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void RigidBody2D::set_angular_velocity(float p_angular_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_angular_velocity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_angular_velocity_encoded;
	PtrToArg<double>::encode(p_angular_velocity, &p_angular_velocity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_angular_velocity_encoded);
}

float RigidBody2D::get_angular_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_angular_velocity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RigidBody2D::set_max_contacts_reported(int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_max_contacts_reported")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_amount_encoded);
}

int32_t RigidBody2D::get_max_contacts_reported() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_max_contacts_reported")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t RigidBody2D::get_contact_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_contact_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_use_custom_integrator(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_use_custom_integrator")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool RigidBody2D::is_using_custom_integrator() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("is_using_custom_integrator")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_contact_monitor(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_contact_monitor")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool RigidBody2D::is_contact_monitor_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("is_contact_monitor_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_continuous_collision_detection_mode(RigidBody2D::CCDMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_continuous_collision_detection_mode")._native_ptr(), 1000241384);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

RigidBody2D::CCDMode RigidBody2D::get_continuous_collision_detection_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_continuous_collision_detection_mode")._native_ptr(), 815214376);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RigidBody2D::CCDMode(0)));
	return (RigidBody2D::CCDMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_axis_velocity(const Vector2 &p_axis_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_axis_velocity")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis_velocity);
}

void RigidBody2D::apply_central_impulse(const Vector2 &p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("apply_central_impulse")._native_ptr(), 3862383994);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_impulse);
}

void RigidBody2D::apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("apply_impulse")._native_ptr(), 4288681949);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_impulse, &p_position);
}

void RigidBody2D::apply_torque_impulse(float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("apply_torque_impulse")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_torque_encoded);
}

void RigidBody2D::apply_central_force(const Vector2 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("apply_central_force")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force);
}

void RigidBody2D::apply_force(const Vector2 &p_force, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("apply_force")._native_ptr(), 4288681949);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force, &p_position);
}

void RigidBody2D::apply_torque(float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("apply_torque")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_torque_encoded);
}

void RigidBody2D::add_constant_central_force(const Vector2 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("add_constant_central_force")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force);
}

void RigidBody2D::add_constant_force(const Vector2 &p_force, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("add_constant_force")._native_ptr(), 4288681949);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force, &p_position);
}

void RigidBody2D::add_constant_torque(float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("add_constant_torque")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_torque_encoded);
}

void RigidBody2D::set_constant_force(const Vector2 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_constant_force")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force);
}

Vector2 RigidBody2D::get_constant_force() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_constant_force")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void RigidBody2D::set_constant_torque(float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_constant_torque")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_torque_encoded);
}

float RigidBody2D::get_constant_torque() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_constant_torque")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RigidBody2D::set_sleeping(bool p_sleeping) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_sleeping")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_sleeping_encoded;
	PtrToArg<bool>::encode(p_sleeping, &p_sleeping_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sleeping_encoded);
}

bool RigidBody2D::is_sleeping() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("is_sleeping")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_can_sleep(bool p_able_to_sleep) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_can_sleep")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_able_to_sleep_encoded;
	PtrToArg<bool>::encode(p_able_to_sleep, &p_able_to_sleep_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_able_to_sleep_encoded);
}

bool RigidBody2D::is_able_to_sleep() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("is_able_to_sleep")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_lock_rotation_enabled(bool p_lock_rotation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_lock_rotation_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_lock_rotation_encoded;
	PtrToArg<bool>::encode(p_lock_rotation, &p_lock_rotation_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lock_rotation_encoded);
}

bool RigidBody2D::is_lock_rotation_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("is_lock_rotation_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_freeze_enabled(bool p_freeze_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_freeze_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_freeze_mode_encoded;
	PtrToArg<bool>::encode(p_freeze_mode, &p_freeze_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_freeze_mode_encoded);
}

bool RigidBody2D::is_freeze_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("is_freeze_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RigidBody2D::set_freeze_mode(RigidBody2D::FreezeMode p_freeze_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("set_freeze_mode")._native_ptr(), 1705112154);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_freeze_mode_encoded;
	PtrToArg<int64_t>::encode(p_freeze_mode, &p_freeze_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_freeze_mode_encoded);
}

RigidBody2D::FreezeMode RigidBody2D::get_freeze_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_freeze_mode")._native_ptr(), 2016872314);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RigidBody2D::FreezeMode(0)));
	return (RigidBody2D::FreezeMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

TypedArray<Node2D> RigidBody2D::get_colliding_bodies() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RigidBody2D::get_class_static()._native_ptr(), StringName("get_colliding_bodies")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Node2D>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Node2D>>(_gde_method_bind, _owner);
}

void RigidBody2D::_integrate_forces(PhysicsDirectBodyState2D *p_state) {}

} // namespace godot
