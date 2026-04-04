/**************************************************************************/
/*  vehicle_wheel3d.cpp                                                   */
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

#include <godot_cpp/classes/vehicle_wheel3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void VehicleWheel3D::set_radius(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float VehicleWheel3D::get_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_suspension_rest_length(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_suspension_rest_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float VehicleWheel3D::get_suspension_rest_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_suspension_rest_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_suspension_travel(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_suspension_travel")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float VehicleWheel3D::get_suspension_travel() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_suspension_travel")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_suspension_stiffness(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_suspension_stiffness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float VehicleWheel3D::get_suspension_stiffness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_suspension_stiffness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_suspension_max_force(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_suspension_max_force")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float VehicleWheel3D::get_suspension_max_force() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_suspension_max_force")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_damping_compression(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_damping_compression")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float VehicleWheel3D::get_damping_compression() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_damping_compression")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_damping_relaxation(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_damping_relaxation")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float VehicleWheel3D::get_damping_relaxation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_damping_relaxation")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_use_as_traction(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_use_as_traction")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool VehicleWheel3D::is_used_as_traction() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("is_used_as_traction")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_use_as_steering(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_use_as_steering")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool VehicleWheel3D::is_used_as_steering() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("is_used_as_steering")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_friction_slip(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_friction_slip")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float VehicleWheel3D::get_friction_slip() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_friction_slip")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

bool VehicleWheel3D::is_in_contact() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("is_in_contact")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Node3D *VehicleWheel3D::get_contact_body() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_contact_body")._native_ptr(), 151077316);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node3D>(_gde_method_bind, _owner);
}

Vector3 VehicleWheel3D::get_contact_point() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_contact_point")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 VehicleWheel3D::get_contact_normal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_contact_normal")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_roll_influence(float p_roll_influence) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_roll_influence")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_roll_influence_encoded;
	PtrToArg<double>::encode(p_roll_influence, &p_roll_influence_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_roll_influence_encoded);
}

float VehicleWheel3D::get_roll_influence() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_roll_influence")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float VehicleWheel3D::get_skidinfo() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_skidinfo")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float VehicleWheel3D::get_rpm() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_rpm")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_engine_force(float p_engine_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_engine_force")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_engine_force_encoded;
	PtrToArg<double>::encode(p_engine_force, &p_engine_force_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_engine_force_encoded);
}

float VehicleWheel3D::get_engine_force() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_engine_force")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_brake(float p_brake) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_brake")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_brake_encoded;
	PtrToArg<double>::encode(p_brake, &p_brake_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_brake_encoded);
}

float VehicleWheel3D::get_brake() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_brake")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleWheel3D::set_steering(float p_steering) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("set_steering")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_steering_encoded;
	PtrToArg<double>::encode(p_steering, &p_steering_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_steering_encoded);
}

float VehicleWheel3D::get_steering() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleWheel3D::get_class_static()._native_ptr(), StringName("get_steering")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
