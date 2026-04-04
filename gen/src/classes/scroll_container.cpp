/**************************************************************************/
/*  scroll_container.cpp                                                  */
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

#include <godot_cpp/classes/scroll_container.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/h_scroll_bar.hpp>
#include <godot_cpp/classes/v_scroll_bar.hpp>

namespace godot {

void ScrollContainer::set_h_scroll(int32_t p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_h_scroll")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

int32_t ScrollContainer::get_h_scroll() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_h_scroll")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ScrollContainer::set_v_scroll(int32_t p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_v_scroll")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

int32_t ScrollContainer::get_v_scroll() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_v_scroll")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ScrollContainer::set_horizontal_custom_step(float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_horizontal_custom_step")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

float ScrollContainer::get_horizontal_custom_step() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_horizontal_custom_step")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ScrollContainer::set_vertical_custom_step(float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_vertical_custom_step")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

float ScrollContainer::get_vertical_custom_step() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_vertical_custom_step")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void ScrollContainer::set_horizontal_scroll_mode(ScrollContainer::ScrollMode p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_horizontal_scroll_mode")._native_ptr(), 2750506364);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_enable_encoded;
	PtrToArg<int64_t>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

ScrollContainer::ScrollMode ScrollContainer::get_horizontal_scroll_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_horizontal_scroll_mode")._native_ptr(), 3987985145);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ScrollContainer::ScrollMode(0)));
	return (ScrollContainer::ScrollMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ScrollContainer::set_vertical_scroll_mode(ScrollContainer::ScrollMode p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_vertical_scroll_mode")._native_ptr(), 2750506364);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_enable_encoded;
	PtrToArg<int64_t>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

ScrollContainer::ScrollMode ScrollContainer::get_vertical_scroll_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_vertical_scroll_mode")._native_ptr(), 3987985145);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ScrollContainer::ScrollMode(0)));
	return (ScrollContainer::ScrollMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ScrollContainer::set_deadzone(int32_t p_deadzone) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_deadzone")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_deadzone_encoded;
	PtrToArg<int64_t>::encode(p_deadzone, &p_deadzone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_deadzone_encoded);
}

int32_t ScrollContainer::get_deadzone() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_deadzone")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ScrollContainer::set_scroll_hint_mode(ScrollContainer::ScrollHintMode p_scroll_hint_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_scroll_hint_mode")._native_ptr(), 578158943);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scroll_hint_mode_encoded;
	PtrToArg<int64_t>::encode(p_scroll_hint_mode, &p_scroll_hint_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scroll_hint_mode_encoded);
}

ScrollContainer::ScrollHintMode ScrollContainer::get_scroll_hint_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_scroll_hint_mode")._native_ptr(), 246835423);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ScrollContainer::ScrollHintMode(0)));
	return (ScrollContainer::ScrollHintMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ScrollContainer::set_tile_scroll_hint(bool p_tile_scroll_hint) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_tile_scroll_hint")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_tile_scroll_hint_encoded;
	PtrToArg<bool>::encode(p_tile_scroll_hint, &p_tile_scroll_hint_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tile_scroll_hint_encoded);
}

bool ScrollContainer::is_scroll_hint_tiled() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("is_scroll_hint_tiled")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ScrollContainer::set_follow_focus(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_follow_focus")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool ScrollContainer::is_following_focus() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("is_following_focus")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

HScrollBar *ScrollContainer::get_h_scroll_bar() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_h_scroll_bar")._native_ptr(), 4004517983);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<HScrollBar>(_gde_method_bind, _owner);
}

VScrollBar *ScrollContainer::get_v_scroll_bar() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_v_scroll_bar")._native_ptr(), 2630340773);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<VScrollBar>(_gde_method_bind, _owner);
}

void ScrollContainer::ensure_control_visible(Control *p_control) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("ensure_control_visible")._native_ptr(), 1496901182);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_control != nullptr ? &p_control->_owner : nullptr));
}

void ScrollContainer::set_draw_focus_border(bool p_draw) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("set_draw_focus_border")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_draw_encoded;
	PtrToArg<bool>::encode(p_draw, &p_draw_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_encoded);
}

bool ScrollContainer::get_draw_focus_border() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ScrollContainer::get_class_static()._native_ptr(), StringName("get_draw_focus_border")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
