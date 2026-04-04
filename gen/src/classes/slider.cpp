/**************************************************************************/
/*  slider.cpp                                                            */
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

#include <godot_cpp/classes/slider.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void Slider::set_ticks(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("set_ticks")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t Slider::get_ticks() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("get_ticks")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool Slider::get_ticks_on_borders() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("get_ticks_on_borders")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Slider::set_ticks_on_borders(bool p_ticks_on_border) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("set_ticks_on_borders")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_ticks_on_border_encoded;
	PtrToArg<bool>::encode(p_ticks_on_border, &p_ticks_on_border_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ticks_on_border_encoded);
}

Slider::TickPosition Slider::get_ticks_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("get_ticks_position")._native_ptr(), 3567635531);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Slider::TickPosition(0)));
	return (Slider::TickPosition)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Slider::set_ticks_position(Slider::TickPosition p_ticks_on_border) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("set_ticks_position")._native_ptr(), 2952822224);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_ticks_on_border_encoded;
	PtrToArg<int64_t>::encode(p_ticks_on_border, &p_ticks_on_border_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ticks_on_border_encoded);
}

void Slider::set_editable(bool p_editable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("set_editable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_editable_encoded;
	PtrToArg<bool>::encode(p_editable, &p_editable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_editable_encoded);
}

bool Slider::is_editable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("is_editable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Slider::set_scrollable(bool p_scrollable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("set_scrollable")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_scrollable_encoded;
	PtrToArg<bool>::encode(p_scrollable, &p_scrollable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scrollable_encoded);
}

bool Slider::is_scrollable() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Slider::get_class_static()._native_ptr(), StringName("is_scrollable")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
