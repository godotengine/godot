/**************************************************************************/
/*  open_xr_dpad_binding_modifier.cpp                                     */
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

#include <godot_cpp/classes/open_xr_dpad_binding_modifier.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/open_xr_action_set.hpp>
#include <godot_cpp/classes/open_xr_haptic_base.hpp>

namespace godot {

void OpenXRDpadBindingModifier::set_action_set(const Ref<OpenXRActionSet> &p_action_set) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("set_action_set")._native_ptr(), 2093310581);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_action_set != nullptr ? &p_action_set->_owner : nullptr));
}

Ref<OpenXRActionSet> OpenXRDpadBindingModifier::get_action_set() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("get_action_set")._native_ptr(), 619941079);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRActionSet>()));
	return Ref<OpenXRActionSet>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRActionSet>(_gde_method_bind, _owner));
}

void OpenXRDpadBindingModifier::set_input_path(const String &p_input_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("set_input_path")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_input_path);
}

String OpenXRDpadBindingModifier::get_input_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("get_input_path")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void OpenXRDpadBindingModifier::set_threshold(float p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("set_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_encoded;
	PtrToArg<double>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

float OpenXRDpadBindingModifier::get_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("get_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRDpadBindingModifier::set_threshold_released(float p_threshold_released) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("set_threshold_released")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_threshold_released_encoded;
	PtrToArg<double>::encode(p_threshold_released, &p_threshold_released_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_released_encoded);
}

float OpenXRDpadBindingModifier::get_threshold_released() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("get_threshold_released")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRDpadBindingModifier::set_center_region(float p_center_region) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("set_center_region")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_center_region_encoded;
	PtrToArg<double>::encode(p_center_region, &p_center_region_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_center_region_encoded);
}

float OpenXRDpadBindingModifier::get_center_region() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("get_center_region")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRDpadBindingModifier::set_wedge_angle(float p_wedge_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("set_wedge_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_wedge_angle_encoded;
	PtrToArg<double>::encode(p_wedge_angle, &p_wedge_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_wedge_angle_encoded);
}

float OpenXRDpadBindingModifier::get_wedge_angle() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("get_wedge_angle")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRDpadBindingModifier::set_is_sticky(bool p_is_sticky) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("set_is_sticky")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_is_sticky_encoded;
	PtrToArg<bool>::encode(p_is_sticky, &p_is_sticky_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_is_sticky_encoded);
}

bool OpenXRDpadBindingModifier::get_is_sticky() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("get_is_sticky")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void OpenXRDpadBindingModifier::set_on_haptic(const Ref<OpenXRHapticBase> &p_haptic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("set_on_haptic")._native_ptr(), 2998020150);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_haptic != nullptr ? &p_haptic->_owner : nullptr));
}

Ref<OpenXRHapticBase> OpenXRDpadBindingModifier::get_on_haptic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("get_on_haptic")._native_ptr(), 922310751);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRHapticBase>()));
	return Ref<OpenXRHapticBase>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRHapticBase>(_gde_method_bind, _owner));
}

void OpenXRDpadBindingModifier::set_off_haptic(const Ref<OpenXRHapticBase> &p_haptic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("set_off_haptic")._native_ptr(), 2998020150);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_haptic != nullptr ? &p_haptic->_owner : nullptr));
}

Ref<OpenXRHapticBase> OpenXRDpadBindingModifier::get_off_haptic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRDpadBindingModifier::get_class_static()._native_ptr(), StringName("get_off_haptic")._native_ptr(), 922310751);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRHapticBase>()));
	return Ref<OpenXRHapticBase>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRHapticBase>(_gde_method_bind, _owner));
}

} // namespace godot
