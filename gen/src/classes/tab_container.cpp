/**************************************************************************/
/*  tab_container.cpp                                                     */
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

#include <godot_cpp/classes/tab_container.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/popup.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/vector2.hpp>

namespace godot {

int32_t TabContainer::get_tab_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TabContainer::set_current_tab(int32_t p_tab_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_current_tab")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_idx_encoded);
}

int32_t TabContainer::get_current_tab() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_current_tab")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TabContainer::get_previous_tab() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_previous_tab")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool TabContainer::select_previous_available() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("select_previous_available")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool TabContainer::select_next_available() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("select_next_available")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Control *TabContainer::get_current_tab_control() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_current_tab_control")._native_ptr(), 2783021301);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner);
}

TabBar *TabContainer::get_tab_bar() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_bar")._native_ptr(), 1865451809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<TabBar>(_gde_method_bind, _owner);
}

Control *TabContainer::get_tab_control(int32_t p_tab_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_control")._native_ptr(), 1065994134);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner, &p_tab_idx_encoded);
}

void TabContainer::set_tab_alignment(TabBar::AlignmentMode p_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_alignment")._native_ptr(), 2413632353);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_encoded);
}

TabBar::AlignmentMode TabContainer::get_tab_alignment() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_alignment")._native_ptr(), 2178122193);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TabBar::AlignmentMode(0)));
	return (TabBar::AlignmentMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TabContainer::set_tabs_position(TabContainer::TabPosition p_tabs_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tabs_position")._native_ptr(), 256673370);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tabs_position_encoded;
	PtrToArg<int64_t>::encode(p_tabs_position, &p_tabs_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tabs_position_encoded);
}

TabContainer::TabPosition TabContainer::get_tabs_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tabs_position")._native_ptr(), 919937023);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TabContainer::TabPosition(0)));
	return (TabContainer::TabPosition)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TabContainer::set_clip_tabs(bool p_clip_tabs) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_clip_tabs")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_clip_tabs_encoded;
	PtrToArg<bool>::encode(p_clip_tabs, &p_clip_tabs_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_clip_tabs_encoded);
}

bool TabContainer::get_clip_tabs() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_clip_tabs")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TabContainer::set_tabs_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tabs_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool TabContainer::are_tabs_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("are_tabs_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TabContainer::set_all_tabs_in_front(bool p_is_front) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_all_tabs_in_front")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_is_front_encoded;
	PtrToArg<bool>::encode(p_is_front, &p_is_front_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_is_front_encoded);
}

bool TabContainer::is_all_tabs_in_front() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("is_all_tabs_in_front")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TabContainer::set_tab_title(int32_t p_tab_idx, const String &p_title) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_title")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_idx_encoded, &p_title);
}

String TabContainer::get_tab_title(int32_t p_tab_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_title")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_tab_idx_encoded);
}

void TabContainer::set_tab_tooltip(int32_t p_tab_idx, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_tooltip")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_idx_encoded, &p_tooltip);
}

String TabContainer::get_tab_tooltip(int32_t p_tab_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_tooltip")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_tab_idx_encoded);
}

void TabContainer::set_tab_icon(int32_t p_tab_idx, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_icon")._native_ptr(), 666127730);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_idx_encoded, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

Ref<Texture2D> TabContainer::get_tab_icon(int32_t p_tab_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_icon")._native_ptr(), 3536238170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_tab_idx_encoded));
}

void TabContainer::set_tab_icon_max_width(int32_t p_tab_idx, int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_icon_max_width")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_idx_encoded, &p_width_encoded);
}

int32_t TabContainer::get_tab_icon_max_width(int32_t p_tab_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_icon_max_width")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_tab_idx_encoded);
}

void TabContainer::set_tab_disabled(int32_t p_tab_idx, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_disabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_idx_encoded, &p_disabled_encoded);
}

bool TabContainer::is_tab_disabled(int32_t p_tab_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("is_tab_disabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_tab_idx_encoded);
}

void TabContainer::set_tab_hidden(int32_t p_tab_idx, bool p_hidden) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_hidden")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	int8_t p_hidden_encoded;
	PtrToArg<bool>::encode(p_hidden, &p_hidden_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_idx_encoded, &p_hidden_encoded);
}

bool TabContainer::is_tab_hidden(int32_t p_tab_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("is_tab_hidden")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_tab_idx_encoded);
}

void TabContainer::set_tab_metadata(int32_t p_tab_idx, const Variant &p_metadata) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_metadata")._native_ptr(), 2152698145);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_idx_encoded, &p_metadata);
}

Variant TabContainer::get_tab_metadata(int32_t p_tab_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_metadata")._native_ptr(), 4227898402);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_tab_idx_encoded);
}

void TabContainer::set_tab_button_icon(int32_t p_tab_idx, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_button_icon")._native_ptr(), 666127730);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_tab_idx_encoded, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

Ref<Texture2D> TabContainer::get_tab_button_icon(int32_t p_tab_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_button_icon")._native_ptr(), 3536238170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_tab_idx_encoded;
	PtrToArg<int64_t>::encode(p_tab_idx, &p_tab_idx_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_tab_idx_encoded));
}

int32_t TabContainer::get_tab_idx_at_point(const Vector2 &p_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_idx_at_point")._native_ptr(), 3820158470);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_point);
}

int32_t TabContainer::get_tab_idx_from_control(Control *p_control) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_idx_from_control")._native_ptr(), 2787397975);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_control != nullptr ? &p_control->_owner : nullptr));
}

void TabContainer::set_popup(Node *p_popup) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_popup")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_popup != nullptr ? &p_popup->_owner : nullptr));
}

Popup *TabContainer::get_popup() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_popup")._native_ptr(), 111095082);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Popup>(_gde_method_bind, _owner);
}

void TabContainer::set_switch_on_drag_hover(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_switch_on_drag_hover")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TabContainer::get_switch_on_drag_hover() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_switch_on_drag_hover")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TabContainer::set_drag_to_rearrange_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_drag_to_rearrange_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TabContainer::get_drag_to_rearrange_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_drag_to_rearrange_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TabContainer::set_tabs_rearrange_group(int32_t p_group_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tabs_rearrange_group")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_group_id_encoded;
	PtrToArg<int64_t>::encode(p_group_id, &p_group_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_group_id_encoded);
}

int32_t TabContainer::get_tabs_rearrange_group() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tabs_rearrange_group")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TabContainer::set_use_hidden_tabs_for_min_size(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_use_hidden_tabs_for_min_size")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TabContainer::get_use_hidden_tabs_for_min_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_use_hidden_tabs_for_min_size")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TabContainer::set_tab_focus_mode(Control::FocusMode p_focus_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_tab_focus_mode")._native_ptr(), 3232914922);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_focus_mode_encoded;
	PtrToArg<int64_t>::encode(p_focus_mode, &p_focus_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_focus_mode_encoded);
}

Control::FocusMode TabContainer::get_tab_focus_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_tab_focus_mode")._native_ptr(), 2132829277);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::FocusMode(0)));
	return (Control::FocusMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TabContainer::set_deselect_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("set_deselect_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TabContainer::get_deselect_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TabContainer::get_class_static()._native_ptr(), StringName("get_deselect_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
