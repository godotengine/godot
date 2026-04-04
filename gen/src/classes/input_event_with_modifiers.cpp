/**************************************************************************/
/*  input_event_with_modifiers.cpp                                        */
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

#include <godot_cpp/classes/input_event_with_modifiers.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void InputEventWithModifiers::set_command_or_control_autoremap(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("set_command_or_control_autoremap")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool InputEventWithModifiers::is_command_or_control_autoremap() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("is_command_or_control_autoremap")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool InputEventWithModifiers::is_command_or_control_pressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("is_command_or_control_pressed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void InputEventWithModifiers::set_alt_pressed(bool p_pressed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("set_alt_pressed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_pressed_encoded;
	PtrToArg<bool>::encode(p_pressed, &p_pressed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pressed_encoded);
}

bool InputEventWithModifiers::is_alt_pressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("is_alt_pressed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void InputEventWithModifiers::set_shift_pressed(bool p_pressed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("set_shift_pressed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_pressed_encoded;
	PtrToArg<bool>::encode(p_pressed, &p_pressed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pressed_encoded);
}

bool InputEventWithModifiers::is_shift_pressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("is_shift_pressed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void InputEventWithModifiers::set_ctrl_pressed(bool p_pressed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("set_ctrl_pressed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_pressed_encoded;
	PtrToArg<bool>::encode(p_pressed, &p_pressed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pressed_encoded);
}

bool InputEventWithModifiers::is_ctrl_pressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("is_ctrl_pressed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void InputEventWithModifiers::set_meta_pressed(bool p_pressed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("set_meta_pressed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_pressed_encoded;
	PtrToArg<bool>::encode(p_pressed, &p_pressed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pressed_encoded);
}

bool InputEventWithModifiers::is_meta_pressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("is_meta_pressed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

BitField<KeyModifierMask> InputEventWithModifiers::get_modifiers_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventWithModifiers::get_class_static()._native_ptr(), StringName("get_modifiers_mask")._native_ptr(), 1258259499);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<KeyModifierMask>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
