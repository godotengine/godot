/**************************************************************************/
/*  input_event.cpp                                                       */
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

#include <godot_cpp/classes/input_event.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/transform2d.hpp>

namespace godot {

void InputEvent::set_device(int32_t p_device) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("set_device")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_device_encoded;
	PtrToArg<int64_t>::encode(p_device, &p_device_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_device_encoded);
}

int32_t InputEvent::get_device() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("get_device")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool InputEvent::is_action(const StringName &p_action, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("is_action")._native_ptr(), 1558498928);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action, &p_exact_match_encoded);
}

bool InputEvent::is_action_pressed(const StringName &p_action, bool p_allow_echo, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("is_action_pressed")._native_ptr(), 1631499404);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_allow_echo_encoded;
	PtrToArg<bool>::encode(p_allow_echo, &p_allow_echo_encoded);
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action, &p_allow_echo_encoded, &p_exact_match_encoded);
}

bool InputEvent::is_action_released(const StringName &p_action, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("is_action_released")._native_ptr(), 1558498928);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_action, &p_exact_match_encoded);
}

float InputEvent::get_action_strength(const StringName &p_action, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("get_action_strength")._native_ptr(), 801543509);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_action, &p_exact_match_encoded);
}

bool InputEvent::is_canceled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("is_canceled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool InputEvent::is_pressed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("is_pressed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool InputEvent::is_released() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("is_released")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool InputEvent::is_echo() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("is_echo")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

String InputEvent::as_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("as_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool InputEvent::is_match(const Ref<InputEvent> &p_event, bool p_exact_match) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("is_match")._native_ptr(), 1754951977);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_match_encoded;
	PtrToArg<bool>::encode(p_exact_match, &p_exact_match_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_event != nullptr ? &p_event->_owner : nullptr), &p_exact_match_encoded);
}

bool InputEvent::is_action_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("is_action_type")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool InputEvent::accumulate(const Ref<InputEvent> &p_with_event) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("accumulate")._native_ptr(), 1062211774);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_with_event != nullptr ? &p_with_event->_owner : nullptr));
}

Ref<InputEvent> InputEvent::xformed_by(const Transform2D &p_xform, const Vector2 &p_local_ofs) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEvent::get_class_static()._native_ptr(), StringName("xformed_by")._native_ptr(), 1282766827);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<InputEvent>()));
	return Ref<InputEvent>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<InputEvent>(_gde_method_bind, _owner, &p_xform, &p_local_ofs));
}

} // namespace godot
