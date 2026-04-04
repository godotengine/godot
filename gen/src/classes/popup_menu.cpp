/**************************************************************************/
/*  popup_menu.cpp                                                        */
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

#include <godot_cpp/classes/popup_menu.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/shortcut.hpp>
#include <godot_cpp/classes/texture2d.hpp>

namespace godot {

bool PopupMenu::activate_item_by_event(const Ref<InputEvent> &p_event, bool p_for_global_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("activate_item_by_event")._native_ptr(), 3716412023);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_for_global_only_encoded;
	PtrToArg<bool>::encode(p_for_global_only, &p_for_global_only_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_event != nullptr ? &p_event->_owner : nullptr), &p_for_global_only_encoded);
}

void PopupMenu::set_prefer_native_menu(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_prefer_native_menu")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool PopupMenu::is_prefer_native_menu() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_prefer_native_menu")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool PopupMenu::is_native_menu() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_native_menu")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PopupMenu::add_item(const String &p_label, int32_t p_id, Key p_accel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_item")._native_ptr(), 3674230041);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int64_t p_accel_encoded;
	PtrToArg<int64_t>::encode(p_accel, &p_accel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label, &p_id_encoded, &p_accel_encoded);
}

void PopupMenu::add_icon_item(const Ref<Texture2D> &p_texture, const String &p_label, int32_t p_id, Key p_accel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_icon_item")._native_ptr(), 1086190128);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int64_t p_accel_encoded;
	PtrToArg<int64_t>::encode(p_accel, &p_accel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_label, &p_id_encoded, &p_accel_encoded);
}

void PopupMenu::add_check_item(const String &p_label, int32_t p_id, Key p_accel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_check_item")._native_ptr(), 3674230041);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int64_t p_accel_encoded;
	PtrToArg<int64_t>::encode(p_accel, &p_accel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label, &p_id_encoded, &p_accel_encoded);
}

void PopupMenu::add_icon_check_item(const Ref<Texture2D> &p_texture, const String &p_label, int32_t p_id, Key p_accel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_icon_check_item")._native_ptr(), 1086190128);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int64_t p_accel_encoded;
	PtrToArg<int64_t>::encode(p_accel, &p_accel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_label, &p_id_encoded, &p_accel_encoded);
}

void PopupMenu::add_radio_check_item(const String &p_label, int32_t p_id, Key p_accel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_radio_check_item")._native_ptr(), 3674230041);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int64_t p_accel_encoded;
	PtrToArg<int64_t>::encode(p_accel, &p_accel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label, &p_id_encoded, &p_accel_encoded);
}

void PopupMenu::add_icon_radio_check_item(const Ref<Texture2D> &p_texture, const String &p_label, int32_t p_id, Key p_accel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_icon_radio_check_item")._native_ptr(), 1086190128);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int64_t p_accel_encoded;
	PtrToArg<int64_t>::encode(p_accel, &p_accel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_label, &p_id_encoded, &p_accel_encoded);
}

void PopupMenu::add_multistate_item(const String &p_label, int32_t p_max_states, int32_t p_default_state, int32_t p_id, Key p_accel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_multistate_item")._native_ptr(), 150780458);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_states_encoded;
	PtrToArg<int64_t>::encode(p_max_states, &p_max_states_encoded);
	int64_t p_default_state_encoded;
	PtrToArg<int64_t>::encode(p_default_state, &p_default_state_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int64_t p_accel_encoded;
	PtrToArg<int64_t>::encode(p_accel, &p_accel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label, &p_max_states_encoded, &p_default_state_encoded, &p_id_encoded, &p_accel_encoded);
}

void PopupMenu::add_shortcut(const Ref<Shortcut> &p_shortcut, int32_t p_id, bool p_global, bool p_allow_echo) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_shortcut")._native_ptr(), 3451850107);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int8_t p_global_encoded;
	PtrToArg<bool>::encode(p_global, &p_global_encoded);
	int8_t p_allow_echo_encoded;
	PtrToArg<bool>::encode(p_allow_echo, &p_allow_echo_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr), &p_id_encoded, &p_global_encoded, &p_allow_echo_encoded);
}

void PopupMenu::add_icon_shortcut(const Ref<Texture2D> &p_texture, const Ref<Shortcut> &p_shortcut, int32_t p_id, bool p_global, bool p_allow_echo) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_icon_shortcut")._native_ptr(), 2997871092);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int8_t p_global_encoded;
	PtrToArg<bool>::encode(p_global, &p_global_encoded);
	int8_t p_allow_echo_encoded;
	PtrToArg<bool>::encode(p_allow_echo, &p_allow_echo_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr), &p_id_encoded, &p_global_encoded, &p_allow_echo_encoded);
}

void PopupMenu::add_check_shortcut(const Ref<Shortcut> &p_shortcut, int32_t p_id, bool p_global) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_check_shortcut")._native_ptr(), 1642193386);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int8_t p_global_encoded;
	PtrToArg<bool>::encode(p_global, &p_global_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr), &p_id_encoded, &p_global_encoded);
}

void PopupMenu::add_icon_check_shortcut(const Ref<Texture2D> &p_texture, const Ref<Shortcut> &p_shortcut, int32_t p_id, bool p_global) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_icon_check_shortcut")._native_ptr(), 3856247530);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int8_t p_global_encoded;
	PtrToArg<bool>::encode(p_global, &p_global_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr), &p_id_encoded, &p_global_encoded);
}

void PopupMenu::add_radio_check_shortcut(const Ref<Shortcut> &p_shortcut, int32_t p_id, bool p_global) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_radio_check_shortcut")._native_ptr(), 1642193386);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int8_t p_global_encoded;
	PtrToArg<bool>::encode(p_global, &p_global_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr), &p_id_encoded, &p_global_encoded);
}

void PopupMenu::add_icon_radio_check_shortcut(const Ref<Texture2D> &p_texture, const Ref<Shortcut> &p_shortcut, int32_t p_id, bool p_global) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_icon_radio_check_shortcut")._native_ptr(), 3856247530);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int8_t p_global_encoded;
	PtrToArg<bool>::encode(p_global, &p_global_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr), (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr), &p_id_encoded, &p_global_encoded);
}

void PopupMenu::add_submenu_item(const String &p_label, const String &p_submenu, int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_submenu_item")._native_ptr(), 2979222410);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label, &p_submenu, &p_id_encoded);
}

void PopupMenu::add_submenu_node_item(const String &p_label, PopupMenu *p_submenu, int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_submenu_node_item")._native_ptr(), 1325455216);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label, (p_submenu != nullptr ? &p_submenu->_owner : nullptr), &p_id_encoded);
}

void PopupMenu::set_item_text(int32_t p_index, const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_text")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_text);
}

void PopupMenu::set_item_text_direction(int32_t p_index, Control::TextDirection p_direction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_text_direction")._native_ptr(), 1707680378);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_direction_encoded;
	PtrToArg<int64_t>::encode(p_direction, &p_direction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_direction_encoded);
}

void PopupMenu::set_item_language(int32_t p_index, const String &p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_language")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_language);
}

void PopupMenu::set_item_auto_translate_mode(int32_t p_index, Node::AutoTranslateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_auto_translate_mode")._native_ptr(), 287402019);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_mode_encoded);
}

void PopupMenu::set_item_icon(int32_t p_index, const Ref<Texture2D> &p_icon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_icon")._native_ptr(), 666127730);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_icon != nullptr ? &p_icon->_owner : nullptr));
}

void PopupMenu::set_item_icon_max_width(int32_t p_index, int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_icon_max_width")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_width_encoded);
}

void PopupMenu::set_item_icon_modulate(int32_t p_index, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_icon_modulate")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_modulate);
}

void PopupMenu::set_item_checked(int32_t p_index, bool p_checked) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_checked")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_checked_encoded;
	PtrToArg<bool>::encode(p_checked, &p_checked_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_checked_encoded);
}

void PopupMenu::set_item_id(int32_t p_index, int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_id")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_id_encoded);
}

void PopupMenu::set_item_accelerator(int32_t p_index, Key p_accel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_accelerator")._native_ptr(), 2992817551);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_accel_encoded;
	PtrToArg<int64_t>::encode(p_accel, &p_accel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_accel_encoded);
}

void PopupMenu::set_item_metadata(int32_t p_index, const Variant &p_metadata) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_metadata")._native_ptr(), 2152698145);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_metadata);
}

void PopupMenu::set_item_disabled(int32_t p_index, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_disabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_disabled_encoded);
}

void PopupMenu::set_item_submenu(int32_t p_index, const String &p_submenu) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_submenu")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_submenu);
}

void PopupMenu::set_item_submenu_node(int32_t p_index, PopupMenu *p_submenu) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_submenu_node")._native_ptr(), 1068370740);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_submenu != nullptr ? &p_submenu->_owner : nullptr));
}

void PopupMenu::set_item_as_separator(int32_t p_index, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_as_separator")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enable_encoded);
}

void PopupMenu::set_item_as_checkable(int32_t p_index, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_as_checkable")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enable_encoded);
}

void PopupMenu::set_item_as_radio_checkable(int32_t p_index, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_as_radio_checkable")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_enable_encoded);
}

void PopupMenu::set_item_tooltip(int32_t p_index, const String &p_tooltip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_tooltip")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_tooltip);
}

void PopupMenu::set_item_shortcut(int32_t p_index, const Ref<Shortcut> &p_shortcut, bool p_global) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_shortcut")._native_ptr(), 825127832);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_global_encoded;
	PtrToArg<bool>::encode(p_global, &p_global_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, (p_shortcut != nullptr ? &p_shortcut->_owner : nullptr), &p_global_encoded);
}

void PopupMenu::set_item_indent(int32_t p_index, int32_t p_indent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_indent")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_indent_encoded;
	PtrToArg<int64_t>::encode(p_indent, &p_indent_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_indent_encoded);
}

void PopupMenu::set_item_multistate(int32_t p_index, int32_t p_state) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_multistate")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_state_encoded;
	PtrToArg<int64_t>::encode(p_state, &p_state_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_state_encoded);
}

void PopupMenu::set_item_multistate_max(int32_t p_index, int32_t p_max_states) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_multistate_max")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_max_states_encoded;
	PtrToArg<int64_t>::encode(p_max_states, &p_max_states_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_max_states_encoded);
}

void PopupMenu::set_item_shortcut_disabled(int32_t p_index, bool p_disabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_shortcut_disabled")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int8_t p_disabled_encoded;
	PtrToArg<bool>::encode(p_disabled, &p_disabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded, &p_disabled_encoded);
}

void PopupMenu::toggle_item_checked(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("toggle_item_checked")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void PopupMenu::toggle_item_multistate(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("toggle_item_multistate")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

String PopupMenu::get_item_text(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_text")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

Control::TextDirection PopupMenu::get_item_text_direction(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_text_direction")._native_ptr(), 4235602388);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Control::TextDirection(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (Control::TextDirection)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

String PopupMenu::get_item_language(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_language")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

Node::AutoTranslateMode PopupMenu::get_item_auto_translate_mode(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_auto_translate_mode")._native_ptr(), 906302372);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Node::AutoTranslateMode(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (Node::AutoTranslateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

Ref<Texture2D> PopupMenu::get_item_icon(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_icon")._native_ptr(), 3536238170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_index_encoded));
}

int32_t PopupMenu::get_item_icon_max_width(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_icon_max_width")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

Color PopupMenu::get_item_icon_modulate(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_icon_modulate")._native_ptr(), 3457211756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_index_encoded);
}

bool PopupMenu::is_item_checked(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_item_checked")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t PopupMenu::get_item_id(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_id")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t PopupMenu::get_item_index(int32_t p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_index")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_id_encoded);
}

Key PopupMenu::get_item_accelerator(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_accelerator")._native_ptr(), 253789942);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Key(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (Key)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

Variant PopupMenu::get_item_metadata(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_metadata")._native_ptr(), 4227898402);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_index_encoded);
}

bool PopupMenu::is_item_disabled(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_item_disabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

String PopupMenu::get_item_submenu(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_submenu")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

PopupMenu *PopupMenu::get_item_submenu_node(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_submenu_node")._native_ptr(), 2100501353);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<PopupMenu>(_gde_method_bind, _owner, &p_index_encoded);
}

bool PopupMenu::is_item_separator(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_item_separator")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

bool PopupMenu::is_item_checkable(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_item_checkable")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

bool PopupMenu::is_item_radio_checkable(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_item_radio_checkable")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

bool PopupMenu::is_item_shortcut_disabled(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_item_shortcut_disabled")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_index_encoded);
}

String PopupMenu::get_item_tooltip(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_tooltip")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

Ref<Shortcut> PopupMenu::get_item_shortcut(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_shortcut")._native_ptr(), 1449483325);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Shortcut>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<Shortcut>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Shortcut>(_gde_method_bind, _owner, &p_index_encoded));
}

int32_t PopupMenu::get_item_indent(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_indent")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t PopupMenu::get_item_multistate_max(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_multistate_max")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t PopupMenu::get_item_multistate(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_multistate")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

void PopupMenu::set_focused_item(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_focused_item")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t PopupMenu::get_focused_item() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_focused_item")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PopupMenu::set_item_count(int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_item_count")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_count_encoded);
}

int32_t PopupMenu::get_item_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_item_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PopupMenu::scroll_to_item(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("scroll_to_item")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void PopupMenu::remove_item(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("remove_item")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

void PopupMenu::add_separator(const String &p_label, int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("add_separator")._native_ptr(), 2266703459);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label, &p_id_encoded);
}

void PopupMenu::clear(bool p_free_submenus) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 107499316);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_free_submenus_encoded;
	PtrToArg<bool>::encode(p_free_submenus, &p_free_submenus_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_free_submenus_encoded);
}

void PopupMenu::set_hide_on_item_selection(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_hide_on_item_selection")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool PopupMenu::is_hide_on_item_selection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_hide_on_item_selection")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PopupMenu::set_hide_on_checkable_item_selection(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_hide_on_checkable_item_selection")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool PopupMenu::is_hide_on_checkable_item_selection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_hide_on_checkable_item_selection")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PopupMenu::set_hide_on_state_item_selection(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_hide_on_state_item_selection")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool PopupMenu::is_hide_on_state_item_selection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_hide_on_state_item_selection")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PopupMenu::set_submenu_popup_delay(float p_seconds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_submenu_popup_delay")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_seconds_encoded;
	PtrToArg<double>::encode(p_seconds, &p_seconds_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seconds_encoded);
}

float PopupMenu::get_submenu_popup_delay() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_submenu_popup_delay")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void PopupMenu::set_allow_search(bool p_allow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_allow_search")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_encoded;
	PtrToArg<bool>::encode(p_allow, &p_allow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_encoded);
}

bool PopupMenu::get_allow_search() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_allow_search")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool PopupMenu::is_system_menu() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("is_system_menu")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PopupMenu::set_system_menu(NativeMenu::SystemMenus p_system_menu_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_system_menu")._native_ptr(), 600639674);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_system_menu_id_encoded;
	PtrToArg<int64_t>::encode(p_system_menu_id, &p_system_menu_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_system_menu_id_encoded);
}

NativeMenu::SystemMenus PopupMenu::get_system_menu() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_system_menu")._native_ptr(), 1222557358);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NativeMenu::SystemMenus(0)));
	return (NativeMenu::SystemMenus)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void PopupMenu::set_shrink_height(bool p_shrink) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_shrink_height")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_shrink_encoded;
	PtrToArg<bool>::encode(p_shrink, &p_shrink_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shrink_encoded);
}

bool PopupMenu::get_shrink_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_shrink_height")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void PopupMenu::set_shrink_width(bool p_shrink) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("set_shrink_width")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_shrink_encoded;
	PtrToArg<bool>::encode(p_shrink, &p_shrink_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shrink_encoded);
}

bool PopupMenu::get_shrink_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PopupMenu::get_class_static()._native_ptr(), StringName("get_shrink_width")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
