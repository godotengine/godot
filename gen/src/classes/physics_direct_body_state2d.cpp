/**************************************************************************/
/*  physics_direct_body_state2d.cpp                                       */
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

#include <godot_cpp/classes/physics_direct_body_state2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/physics_direct_space_state2d.hpp>

namespace godot {

Vector2 PhysicsDirectBodyState2D::get_total_gravity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_total_gravity")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

float PhysicsDirectBodyState2D::get_total_linear_damp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_total_linear_damp")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float PhysicsDirectBodyState2D::get_total_angular_damp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_total_angular_damp")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

Vector2 PhysicsDirectBodyState2D::get_center_of_mass() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_center_of_mass")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

Vector2 PhysicsDirectBodyState2D::get_center_of_mass_local() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_center_of_mass_local")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

float PhysicsDirectBodyState2D::get_inverse_mass() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_inverse_mass")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float PhysicsDirectBodyState2D::get_inverse_inertia() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_inverse_inertia")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicsDirectBodyState2D::set_linear_velocity(const Vector2 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("set_linear_velocity")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_velocity);
}

Vector2 PhysicsDirectBodyState2D::get_linear_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_linear_velocity")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void PhysicsDirectBodyState2D::set_angular_velocity(float p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("set_angular_velocity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_velocity_encoded;
	PtrToArg<double>::encode(p_velocity, &p_velocity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_velocity_encoded);
}

float PhysicsDirectBodyState2D::get_angular_velocity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_angular_velocity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicsDirectBodyState2D::set_transform(const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("set_transform")._native_ptr(), 2761652528);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_transform);
}

Transform2D PhysicsDirectBodyState2D::get_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Vector2 PhysicsDirectBodyState2D::get_velocity_at_local_position(const Vector2 &p_local_position) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_velocity_at_local_position")._native_ptr(), 2656412154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_local_position);
}

void PhysicsDirectBodyState2D::apply_central_impulse(const Vector2 &p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("apply_central_impulse")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_impulse);
}

void PhysicsDirectBodyState2D::apply_torque_impulse(float p_impulse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("apply_torque_impulse")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_impulse_encoded;
	PtrToArg<double>::encode(p_impulse, &p_impulse_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_impulse_encoded);
}

void PhysicsDirectBodyState2D::apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("apply_impulse")._native_ptr(), 4288681949);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_impulse, &p_position);
}

void PhysicsDirectBodyState2D::apply_central_force(const Vector2 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("apply_central_force")._native_ptr(), 3862383994);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force);
}

void PhysicsDirectBodyState2D::apply_force(const Vector2 &p_force, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("apply_force")._native_ptr(), 4288681949);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force, &p_position);
}

void PhysicsDirectBodyState2D::apply_torque(float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("apply_torque")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_torque_encoded);
}

void PhysicsDirectBodyState2D::add_constant_central_force(const Vector2 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("add_constant_central_force")._native_ptr(), 3862383994);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force);
}

void PhysicsDirectBodyState2D::add_constant_force(const Vector2 &p_force, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("add_constant_force")._native_ptr(), 4288681949);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force, &p_position);
}

void PhysicsDirectBodyState2D::add_constant_torque(float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("add_constant_torque")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_torque_encoded);
}

void PhysicsDirectBodyState2D::set_constant_force(const Vector2 &p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("set_constant_force")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_force);
}

Vector2 PhysicsDirectBodyState2D::get_constant_force() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_constant_force")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void PhysicsDirectBodyState2D::set_constant_torque(float p_torque) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("set_constant_torque")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_torque_encoded;
	PtrToArg<double>::encode(p_torque, &p_torque_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_torque_encoded);
}

float PhysicsDirectBodyState2D::get_constant_torque() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_constant_torque")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicsDirectBodyState2D::set_sleep_state(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("set_sleep_state")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool PhysicsDirectBodyState2D::is_sleeping() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("is_sleeping")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PhysicsDirectBodyState2D::set_collision_layer(uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("set_collision_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded);
}

uint32_t PhysicsDirectBodyState2D::get_collision_layer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_collision_layer")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PhysicsDirectBodyState2D::set_collision_mask(uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("set_collision_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mask_encoded);
}

uint32_t PhysicsDirectBodyState2D::get_collision_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_collision_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t PhysicsDirectBodyState2D::get_contact_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Vector2 PhysicsDirectBodyState2D::get_contact_local_position(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_local_position")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

Vector2 PhysicsDirectBodyState2D::get_contact_local_normal(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_local_normal")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

int32_t PhysicsDirectBodyState2D::get_contact_local_shape(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_local_shape")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

Vector2 PhysicsDirectBodyState2D::get_contact_local_velocity_at_position(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_local_velocity_at_position")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

RID PhysicsDirectBodyState2D::get_contact_collider(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_collider")._native_ptr(), 495598643);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

Vector2 PhysicsDirectBodyState2D::get_contact_collider_position(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_collider_position")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

uint64_t PhysicsDirectBodyState2D::get_contact_collider_id(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_collider_id")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

Object *PhysicsDirectBodyState2D::get_contact_collider_object(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_collider_object")._native_ptr(), 3332903315);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Object>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

int32_t PhysicsDirectBodyState2D::get_contact_collider_shape(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_collider_shape")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

Vector2 PhysicsDirectBodyState2D::get_contact_collider_velocity_at_position(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_collider_velocity_at_position")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

Vector2 PhysicsDirectBodyState2D::get_contact_impulse(int32_t p_contact_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_contact_impulse")._native_ptr(), 2299179447);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_contact_idx_encoded;
	PtrToArg<int64_t>::encode(p_contact_idx, &p_contact_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_contact_idx_encoded);
}

float PhysicsDirectBodyState2D::get_step() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_step")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PhysicsDirectBodyState2D::integrate_forces() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("integrate_forces")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

PhysicsDirectSpaceState2D *PhysicsDirectBodyState2D::get_space_state() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PhysicsDirectBodyState2D::get_class_static()._native_ptr(), StringName("get_space_state")._native_ptr(), 2506717822);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<PhysicsDirectSpaceState2D>(_gde_method_bind, _owner);
}

} // namespace godot
