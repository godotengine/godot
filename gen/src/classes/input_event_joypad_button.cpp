/**************************************************************************/
/*  input_event_joypad_button.cpp                                         */
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

#include <godot_cpp/classes/input_event_joypad_button.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void InputEventJoypadButton::set_button_index(JoyButton p_button_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventJoypadButton::get_class_static()._native_ptr(), StringName("set_button_index")._native_ptr(), 1466368136);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_button_index_encoded;
	PtrToArg<int64_t>::encode(p_button_index, &p_button_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_button_index_encoded);
}

JoyButton InputEventJoypadButton::get_button_index() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventJoypadButton::get_class_static()._native_ptr(), StringName("get_button_index")._native_ptr(), 595588182);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (JoyButton(0)));
	return (JoyButton)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void InputEventJoypadButton::set_pressure(float p_pressure) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventJoypadButton::get_class_static()._native_ptr(), StringName("set_pressure")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pressure_encoded;
	PtrToArg<double>::encode(p_pressure, &p_pressure_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pressure_encoded);
}

float InputEventJoypadButton::get_pressure() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventJoypadButton::get_class_static()._native_ptr(), StringName("get_pressure")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void InputEventJoypadButton::set_pressed(bool p_pressed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventJoypadButton::get_class_static()._native_ptr(), StringName("set_pressed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_pressed_encoded;
	PtrToArg<bool>::encode(p_pressed, &p_pressed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pressed_encoded);
}

} // namespace godot
