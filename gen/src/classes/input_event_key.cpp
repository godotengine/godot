/**************************************************************************/
/*  input_event_key.cpp                                                   */
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

#include <godot_cpp/classes/input_event_key.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void InputEventKey::set_pressed(bool p_pressed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("set_pressed")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_pressed_encoded;
	PtrToArg<bool>::encode(p_pressed, &p_pressed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pressed_encoded);
}

void InputEventKey::set_keycode(Key p_keycode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("set_keycode")._native_ptr(), 888074362);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_keycode_encoded;
	PtrToArg<int64_t>::encode(p_keycode, &p_keycode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_keycode_encoded);
}

Key InputEventKey::get_keycode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("get_keycode")._native_ptr(), 1585896689);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void InputEventKey::set_physical_keycode(Key p_physical_keycode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("set_physical_keycode")._native_ptr(), 888074362);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_physical_keycode_encoded;
	PtrToArg<int64_t>::encode(p_physical_keycode, &p_physical_keycode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_physical_keycode_encoded);
}

Key InputEventKey::get_physical_keycode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("get_physical_keycode")._native_ptr(), 1585896689);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void InputEventKey::set_key_label(Key p_key_label) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("set_key_label")._native_ptr(), 888074362);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_key_label_encoded;
	PtrToArg<int64_t>::encode(p_key_label, &p_key_label_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_key_label_encoded);
}

Key InputEventKey::get_key_label() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("get_key_label")._native_ptr(), 1585896689);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void InputEventKey::set_unicode(char32_t p_unicode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("set_unicode")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_unicode_encoded;
	PtrToArg<int64_t>::encode(p_unicode, &p_unicode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_unicode_encoded);
}

char32_t InputEventKey::get_unicode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("get_unicode")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<char32_t>(_gde_method_bind, _owner);
}

void InputEventKey::set_location(KeyLocation p_location) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("set_location")._native_ptr(), 634453155);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_location_encoded;
	PtrToArg<int64_t>::encode(p_location, &p_location_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_location_encoded);
}

KeyLocation InputEventKey::get_location() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("get_location")._native_ptr(), 211810873);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (KeyLocation(0)));
	return (KeyLocation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void InputEventKey::set_echo(bool p_echo) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("set_echo")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_echo_encoded;
	PtrToArg<bool>::encode(p_echo, &p_echo_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_echo_encoded);
}

Key InputEventKey::get_keycode_with_modifiers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("get_keycode_with_modifiers")._native_ptr(), 1585896689);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Key InputEventKey::get_physical_keycode_with_modifiers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("get_physical_keycode_with_modifiers")._native_ptr(), 1585896689);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Key InputEventKey::get_key_label_with_modifiers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("get_key_label_with_modifiers")._native_ptr(), 1585896689);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String InputEventKey::as_text_keycode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("as_text_keycode")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String InputEventKey::as_text_physical_keycode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("as_text_physical_keycode")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String InputEventKey::as_text_key_label() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("as_text_key_label")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String InputEventKey::as_text_location() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(InputEventKey::get_class_static()._native_ptr(), StringName("as_text_location")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

} // namespace godot
