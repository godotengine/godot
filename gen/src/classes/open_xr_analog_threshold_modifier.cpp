/**************************************************************************/
/*  open_xr_analog_threshold_modifier.cpp                                 */
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

#include <godot_cpp/classes/open_xr_analog_threshold_modifier.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/open_xr_haptic_base.hpp>

namespace godot {

void OpenXRAnalogThresholdModifier::set_on_threshold(float p_on_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAnalogThresholdModifier::get_class_static()._native_ptr(), StringName("set_on_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_on_threshold_encoded;
	PtrToArg<double>::encode(p_on_threshold, &p_on_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_on_threshold_encoded);
}

float OpenXRAnalogThresholdModifier::get_on_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAnalogThresholdModifier::get_class_static()._native_ptr(), StringName("get_on_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRAnalogThresholdModifier::set_off_threshold(float p_off_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAnalogThresholdModifier::get_class_static()._native_ptr(), StringName("set_off_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_off_threshold_encoded;
	PtrToArg<double>::encode(p_off_threshold, &p_off_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_off_threshold_encoded);
}

float OpenXRAnalogThresholdModifier::get_off_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAnalogThresholdModifier::get_class_static()._native_ptr(), StringName("get_off_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRAnalogThresholdModifier::set_on_haptic(const Ref<OpenXRHapticBase> &p_haptic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAnalogThresholdModifier::get_class_static()._native_ptr(), StringName("set_on_haptic")._native_ptr(), 2998020150);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_haptic != nullptr ? &p_haptic->_owner : nullptr));
}

Ref<OpenXRHapticBase> OpenXRAnalogThresholdModifier::get_on_haptic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAnalogThresholdModifier::get_class_static()._native_ptr(), StringName("get_on_haptic")._native_ptr(), 922310751);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRHapticBase>()));
	return Ref<OpenXRHapticBase>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRHapticBase>(_gde_method_bind, _owner));
}

void OpenXRAnalogThresholdModifier::set_off_haptic(const Ref<OpenXRHapticBase> &p_haptic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAnalogThresholdModifier::get_class_static()._native_ptr(), StringName("set_off_haptic")._native_ptr(), 2998020150);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_haptic != nullptr ? &p_haptic->_owner : nullptr));
}

Ref<OpenXRHapticBase> OpenXRAnalogThresholdModifier::get_off_haptic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAnalogThresholdModifier::get_class_static()._native_ptr(), StringName("get_off_haptic")._native_ptr(), 922310751);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRHapticBase>()));
	return Ref<OpenXRHapticBase>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRHapticBase>(_gde_method_bind, _owner));
}

} // namespace godot
