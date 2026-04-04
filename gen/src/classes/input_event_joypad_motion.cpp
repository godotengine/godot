/**************************************************************************/
/*  input_event_joypad_motion.cpp                                         */
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

#include <godot_cpp/classes/input_event_joypad_motion.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void InputEventJoypadMotion::set_axis(JoyAxis p_axis) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventJoypadMotion::get_class_static()._native_ptr(), StringName("set_axis")._native_ptr(), 1332685170);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_axis_encoded;
	PtrToArg<int64_t>::encode(p_axis, &p_axis_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis_encoded);
}

JoyAxis InputEventJoypadMotion::get_axis() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventJoypadMotion::get_class_static()._native_ptr(), StringName("get_axis")._native_ptr(), 4019121683);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (JoyAxis(0)));
	return (JoyAxis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void InputEventJoypadMotion::set_axis_value(float p_axis_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventJoypadMotion::get_class_static()._native_ptr(), StringName("set_axis_value")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_axis_value_encoded;
	PtrToArg<double>::encode(p_axis_value, &p_axis_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_axis_value_encoded);
}

float InputEventJoypadMotion::get_axis_value() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventJoypadMotion::get_class_static()._native_ptr(), StringName("get_axis_value")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
