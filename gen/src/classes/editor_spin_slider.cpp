/**************************************************************************/
/*  editor_spin_slider.cpp                                                */
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

#include <godot_cpp/classes/editor_spin_slider.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void EditorSpinSlider::set_label(const String &p_label) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("set_label")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label);
}

String EditorSpinSlider::get_label() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("get_label")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void EditorSpinSlider::set_suffix(const String &p_suffix) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("set_suffix")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_suffix);
}

String EditorSpinSlider::get_suffix() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("get_suffix")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void EditorSpinSlider::set_read_only(bool p_read_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("set_read_only")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_read_only_encoded;
	PtrToArg<bool>::encode(p_read_only, &p_read_only_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_read_only_encoded);
}

bool EditorSpinSlider::is_read_only() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("is_read_only")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorSpinSlider::set_flat(bool p_flat) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("set_flat")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flat_encoded;
	PtrToArg<bool>::encode(p_flat, &p_flat_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flat_encoded);
}

bool EditorSpinSlider::is_flat() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("is_flat")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorSpinSlider::set_control_state(EditorSpinSlider::ControlState p_state) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("set_control_state")._native_ptr(), 1324557109);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_state_encoded);
}

EditorSpinSlider::ControlState EditorSpinSlider::get_control_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("get_control_state")._native_ptr(), 3406006200);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (EditorSpinSlider::ControlState(0)));
	return (EditorSpinSlider::ControlState)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void EditorSpinSlider::set_hide_slider(bool p_hide_slider) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("set_hide_slider")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_hide_slider_encoded;
	PtrToArg<bool>::encode(p_hide_slider, &p_hide_slider_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hide_slider_encoded);
}

bool EditorSpinSlider::is_hiding_slider() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("is_hiding_slider")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void EditorSpinSlider::set_editing_integer(bool p_editing_integer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("set_editing_integer")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_editing_integer_encoded;
	PtrToArg<bool>::encode(p_editing_integer, &p_editing_integer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_editing_integer_encoded);
}

bool EditorSpinSlider::is_editing_integer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSpinSlider::get_class_static()._native_ptr(), StringName("is_editing_integer")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
