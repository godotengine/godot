/**************************************************************************/
/*  vehicle_body3d.cpp                                                    */
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

#include <godot_cpp/classes/vehicle_body3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void VehicleBody3D::set_engine_force(float p_engine_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleBody3D::get_class_static()._native_ptr(), StringName("set_engine_force")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_engine_force_encoded;
	PtrToArg<double>::encode(p_engine_force, &p_engine_force_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_engine_force_encoded);
}

float VehicleBody3D::get_engine_force() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleBody3D::get_class_static()._native_ptr(), StringName("get_engine_force")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleBody3D::set_brake(float p_brake) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleBody3D::get_class_static()._native_ptr(), StringName("set_brake")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_brake_encoded;
	PtrToArg<double>::encode(p_brake, &p_brake_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_brake_encoded);
}

float VehicleBody3D::get_brake() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleBody3D::get_class_static()._native_ptr(), StringName("get_brake")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void VehicleBody3D::set_steering(float p_steering) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleBody3D::get_class_static()._native_ptr(), StringName("set_steering")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_steering_encoded;
	PtrToArg<double>::encode(p_steering, &p_steering_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_steering_encoded);
}

float VehicleBody3D::get_steering() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VehicleBody3D::get_class_static()._native_ptr(), StringName("get_steering")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
