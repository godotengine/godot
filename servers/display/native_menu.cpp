/**************************************************************************/
/*  native_menu.cpp                                                       */
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

#include "native_menu.h"

#include "scene/resources/image_texture.h"

NativeMenu *NativeMenu::singleton = nullptr;

void NativeMenu::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_feature", "feature"), &NativeMenu::has_feature);

	ClassDB::bind_method(D_METHOD("has_system_menu", "menu_id"), &NativeMenu::has_system_menu);
	ClassDB::bind_method(D_METHOD("get_system_menu", "menu_id"), &NativeMenu::get_system_menu);
	ClassDB::bind_method(D_METHOD("get_system_menu_name", "menu_id"), &NativeMenu::get_system_menu_name);

	ClassDB::bind_method(D_METHOD("get_system_menu_text", "menu_id"), &NativeMenu::get_system_menu_text);
	ClassDB::bind_method(D_METHOD("set_system_menu_text", "menu_id", "name"), &NativeMenu::set_system_menu_text);

	ClassDB::bind_method(D_METHOD("create_menu"), &NativeMenu::create_menu);
	ClassDB::bind_method(D_METHOD("has_menu", "rid"), &NativeMenu::has_menu);
	ClassDB::bind_method(D_METHOD("free_menu", "rid"), &NativeMenu::free_menu);

	ClassDB::bind_method(D_METHOD("get_size", "rid"), &NativeMenu::get_size);
	ClassDB::bind_method(D_METHOD("popup", "rid", "position"), &NativeMenu::popup);

	ClassDB::bind_method(D_METHOD("set_interface_direction", "rid", "is_rtl"), &NativeMenu::set_interface_direction);
	ClassDB::bind_method(D_METHOD("set_popup_open_callback", "rid", "callback"), &NativeMenu::set_popup_open_callback);
	ClassDB::bind_method(D_METHOD("get_popup_open_callback", "rid"), &NativeMenu::get_popup_open_callback);
	ClassDB::bind_method(D_METHOD("set_popup_close_callback", "rid", "callback"), &NativeMenu::set_popup_close_callback);
	ClassDB::bind_method(D_METHOD("get_popup_close_callback", "rid"), &NativeMenu::get_popup_close_callback);
	ClassDB::bind_method(D_METHOD("set_minimum_width", "rid", "width"), &NativeMenu::set_minimum_width);
	ClassDB::bind_method(D_METHOD("get_minimum_width", "rid"), &NativeMenu::get_minimum_width);

	ClassDB::bind_method(D_METHOD("is_opened", "rid"), &NativeMenu::is_opened);

	ClassDB::bind_method(D_METHOD("add_submenu_item", "rid", "label", "submenu_rid", "tag", "index"), &NativeMenu::add_submenu_item, DEFVAL(Variant()), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("add_item", "rid", "label", "callback", "key_callback", "tag", "accelerator", "index"), &NativeMenu::add_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("add_check_item", "rid", "label", "callback", "key_callback", "tag", "accelerator", "index"), &NativeMenu::add_check_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("add_icon_item", "rid", "icon", "label", "callback", "key_callback", "tag", "accelerator", "index"), &NativeMenu::add_icon_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("add_icon_check_item", "rid", "icon", "label", "callback", "key_callback", "tag", "accelerator", "index"), &NativeMenu::add_icon_check_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("add_radio_check_item", "rid", "label", "callback", "key_callback", "tag", "accelerator", "index"), &NativeMenu::add_radio_check_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("add_icon_radio_check_item", "rid", "icon", "label", "callback", "key_callback", "tag", "accelerator", "index"), &NativeMenu::add_icon_radio_check_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("add_multistate_item", "rid", "label", "max_states", "default_state", "callback", "key_callback", "tag", "accelerator", "index"), &NativeMenu::add_multistate_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("add_separator", "rid", "index"), &NativeMenu::add_separator, DEFVAL(-1));

	ClassDB::bind_method(D_METHOD("find_item_index_with_text", "rid", "text"), &NativeMenu::find_item_index_with_text);
	ClassDB::bind_method(D_METHOD("find_item_index_with_tag", "rid", "tag"), &NativeMenu::find_item_index_with_tag);
	ClassDB::bind_method(D_METHOD("find_item_index_with_submenu", "rid", "submenu_rid"), &NativeMenu::find_item_index_with_submenu);

	ClassDB::bind_method(D_METHOD("is_item_checked", "rid", "idx"), &NativeMenu::is_item_checked);
	ClassDB::bind_method(D_METHOD("is_item_checkable", "rid", "idx"), &NativeMenu::is_item_checkable);
	ClassDB::bind_method(D_METHOD("is_item_radio_checkable", "rid", "idx"), &NativeMenu::is_item_radio_checkable);
	ClassDB::bind_method(D_METHOD("get_item_callback", "rid", "idx"), &NativeMenu::get_item_callback);
	ClassDB::bind_method(D_METHOD("get_item_key_callback", "rid", "idx"), &NativeMenu::get_item_key_callback);
	ClassDB::bind_method(D_METHOD("get_item_tag", "rid", "idx"), &NativeMenu::get_item_tag);
	ClassDB::bind_method(D_METHOD("get_item_text", "rid", "idx"), &NativeMenu::get_item_text);
	ClassDB::bind_method(D_METHOD("get_item_submenu", "rid", "idx"), &NativeMenu::get_item_submenu);
	ClassDB::bind_method(D_METHOD("get_item_accelerator", "rid", "idx"), &NativeMenu::get_item_accelerator);
	ClassDB::bind_method(D_METHOD("is_item_disabled", "rid", "idx"), &NativeMenu::is_item_disabled);
	ClassDB::bind_method(D_METHOD("is_item_hidden", "rid", "idx"), &NativeMenu::is_item_hidden);
	ClassDB::bind_method(D_METHOD("get_item_tooltip", "rid", "idx"), &NativeMenu::get_item_tooltip);
	ClassDB::bind_method(D_METHOD("get_item_state", "rid", "idx"), &NativeMenu::get_item_state);
	ClassDB::bind_method(D_METHOD("get_item_max_states", "rid", "idx"), &NativeMenu::get_item_max_states);
	ClassDB::bind_method(D_METHOD("get_item_icon", "rid", "idx"), &NativeMenu::get_item_icon);
	ClassDB::bind_method(D_METHOD("get_item_indentation_level", "rid", "idx"), &NativeMenu::get_item_indentation_level);

	ClassDB::bind_method(D_METHOD("set_item_checked", "rid", "idx", "checked"), &NativeMenu::set_item_checked);
	ClassDB::bind_method(D_METHOD("set_item_checkable", "rid", "idx", "checkable"), &NativeMenu::set_item_checkable);
	ClassDB::bind_method(D_METHOD("set_item_radio_checkable", "rid", "idx", "checkable"), &NativeMenu::set_item_radio_checkable);
	ClassDB::bind_method(D_METHOD("set_item_callback", "rid", "idx", "callback"), &NativeMenu::set_item_callback);
	ClassDB::bind_method(D_METHOD("set_item_hover_callbacks", "rid", "idx", "callback"), &NativeMenu::set_item_hover_callbacks);
	ClassDB::bind_method(D_METHOD("set_item_key_callback", "rid", "idx", "key_callback"), &NativeMenu::set_item_key_callback);
	ClassDB::bind_method(D_METHOD("set_item_tag", "rid", "idx", "tag"), &NativeMenu::set_item_tag);
	ClassDB::bind_method(D_METHOD("set_item_text", "rid", "idx", "text"), &NativeMenu::set_item_text);
	ClassDB::bind_method(D_METHOD("set_item_submenu", "rid", "idx", "submenu_rid"), &NativeMenu::set_item_submenu);
	ClassDB::bind_method(D_METHOD("set_item_accelerator", "rid", "idx", "keycode"), &NativeMenu::set_item_accelerator);
	ClassDB::bind_method(D_METHOD("set_item_disabled", "rid", "idx", "disabled"), &NativeMenu::set_item_disabled);
	ClassDB::bind_method(D_METHOD("set_item_hidden", "rid", "idx", "hidden"), &NativeMenu::set_item_hidden);
	ClassDB::bind_method(D_METHOD("set_item_tooltip", "rid", "idx", "tooltip"), &NativeMenu::set_item_tooltip);
	ClassDB::bind_method(D_METHOD("set_item_state", "rid", "idx", "state"), &NativeMenu::set_item_state);
	ClassDB::bind_method(D_METHOD("set_item_max_states", "rid", "idx", "max_states"), &NativeMenu::set_item_max_states);
	ClassDB::bind_method(D_METHOD("set_item_icon", "rid", "idx", "icon"), &NativeMenu::set_item_icon);
	ClassDB::bind_method(D_METHOD("set_item_indentation_level", "rid", "idx", "level"), &NativeMenu::set_item_indentation_level);

	ClassDB::bind_method(D_METHOD("get_item_count", "rid"), &NativeMenu::get_item_count);
	ClassDB::bind_method(D_METHOD("is_system_menu", "rid"), &NativeMenu::is_system_menu);

	ClassDB::bind_method(D_METHOD("remove_item", "rid", "idx"), &NativeMenu::remove_item);
	ClassDB::bind_method(D_METHOD("clear", "rid"), &NativeMenu::clear);

	BIND_ENUM_CONSTANT(FEATURE_GLOBAL_MENU);
	BIND_ENUM_CONSTANT(FEATURE_POPUP_MENU);
	BIND_ENUM_CONSTANT(FEATURE_OPEN_CLOSE_CALLBACK);
	BIND_ENUM_CONSTANT(FEATURE_HOVER_CALLBACK);
	BIND_ENUM_CONSTANT(FEATURE_KEY_CALLBACK);

	BIND_ENUM_CONSTANT(INVALID_MENU_ID);
	BIND_ENUM_CONSTANT(MAIN_MENU_ID);
	BIND_ENUM_CONSTANT(APPLICATION_MENU_ID);
	BIND_ENUM_CONSTANT(WINDOW_MENU_ID);
	BIND_ENUM_CONSTANT(HELP_MENU_ID);
	BIND_ENUM_CONSTANT(DOCK_MENU_ID);
}

bool NativeMenu::has_feature(Feature p_feature) const {
	return false;
}

bool NativeMenu::has_system_menu(SystemMenus p_menu_id) const {
	return false;
}

RID NativeMenu::get_system_menu(SystemMenus p_menu_id) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return RID();
}

String NativeMenu::get_system_menu_name(SystemMenus p_menu_id) const {
	switch (p_menu_id) {
		case MAIN_MENU_ID:
			return "Main menu";
		case APPLICATION_MENU_ID:
			return "Application menu";
		case WINDOW_MENU_ID:
			return "Window menu";
		case HELP_MENU_ID:
			return "Help menu";
		case DOCK_MENU_ID:
			return "Dock menu";
		default:
			return "Invalid";
	}
}

String NativeMenu::get_system_menu_text(SystemMenus p_menu_id) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return String();
}

void NativeMenu::set_system_menu_text(SystemMenus p_menu_id, const String &p_name) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

RID NativeMenu::create_menu() {
	WARN_PRINT("Global menus are not supported on this platform.");
	return RID();
}

bool NativeMenu::has_menu(const RID &p_rid) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return false;
}

void NativeMenu::free_menu(const RID &p_rid) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

Size2 NativeMenu::get_size(const RID &p_rid) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return Size2();
}

void NativeMenu::popup(const RID &p_rid, const Vector2i &p_position) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_interface_direction(const RID &p_rid, bool p_is_rtl) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_popup_open_callback(const RID &p_rid, const Callable &p_callback) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

Callable NativeMenu::get_popup_open_callback(const RID &p_rid) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return Callable();
}

void NativeMenu::set_popup_close_callback(const RID &p_rid, const Callable &p_callback) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

Callable NativeMenu::get_popup_close_callback(const RID &p_rid) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return Callable();
}

bool NativeMenu::is_opened(const RID &p_rid) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return false;
}

void NativeMenu::set_minimum_width(const RID &p_rid, float p_width) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

float NativeMenu::get_minimum_width(const RID &p_rid) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return 0.f;
}

int NativeMenu::add_submenu_item(const RID &p_rid, const String &p_label, const RID &p_submenu_rid, const Variant &p_tag, int p_index) {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::add_item(const RID &p_rid, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::add_check_item(const RID &p_rid, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::add_icon_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::add_icon_check_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::add_radio_check_item(const RID &p_rid, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::add_icon_radio_check_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::add_multistate_item(const RID &p_rid, const String &p_label, int p_max_states, int p_default_state, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::add_separator(const RID &p_rid, int p_index) {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::find_item_index_with_text(const RID &p_rid, const String &p_text) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::find_item_index_with_tag(const RID &p_rid, const Variant &p_tag) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::find_item_index_with_submenu(const RID &p_rid, const RID &p_submenu_rid) const {
	if (!has_menu(p_rid) || !has_menu(p_submenu_rid)) {
		return -1;
	}
	int count = get_item_count(p_rid);
	for (int i = 0; i < count; i++) {
		if (p_submenu_rid == get_item_submenu(p_rid, i)) {
			return i;
		}
	}
	return -1;
}

bool NativeMenu::is_item_checked(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return false;
}

bool NativeMenu::is_item_checkable(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return false;
}

bool NativeMenu::is_item_radio_checkable(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return false;
}

Callable NativeMenu::get_item_callback(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return Callable();
}

Callable NativeMenu::get_item_key_callback(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return Callable();
}

Variant NativeMenu::get_item_tag(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return Variant();
}

String NativeMenu::get_item_text(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return String();
}

RID NativeMenu::get_item_submenu(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return RID();
}

Key NativeMenu::get_item_accelerator(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return Key::NONE;
}

bool NativeMenu::is_item_disabled(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return false;
}

bool NativeMenu::is_item_hidden(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return false;
}

String NativeMenu::get_item_tooltip(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return String();
}

int NativeMenu::get_item_state(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

int NativeMenu::get_item_max_states(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return -1;
}

Ref<Texture2D> NativeMenu::get_item_icon(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return Ref<Texture2D>();
}

int NativeMenu::get_item_indentation_level(const RID &p_rid, int p_idx) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return 0;
}

void NativeMenu::set_item_checked(const RID &p_rid, int p_idx, bool p_checked) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_checkable(const RID &p_rid, int p_idx, bool p_checkable) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_radio_checkable(const RID &p_rid, int p_idx, bool p_checkable) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_callback(const RID &p_rid, int p_idx, const Callable &p_callback) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_key_callback(const RID &p_rid, int p_idx, const Callable &p_key_callback) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_hover_callbacks(const RID &p_rid, int p_idx, const Callable &p_callback) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_tag(const RID &p_rid, int p_idx, const Variant &p_tag) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_text(const RID &p_rid, int p_idx, const String &p_text) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_submenu(const RID &p_rid, int p_idx, const RID &p_submenu_rid) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_accelerator(const RID &p_rid, int p_idx, Key p_keycode) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_disabled(const RID &p_rid, int p_idx, bool p_disabled) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_hidden(const RID &p_rid, int p_idx, bool p_hidden) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_tooltip(const RID &p_rid, int p_idx, const String &p_tooltip) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_state(const RID &p_rid, int p_idx, int p_state) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_max_states(const RID &p_rid, int p_idx, int p_max_states) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_icon(const RID &p_rid, int p_idx, const Ref<Texture2D> &p_icon) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::set_item_indentation_level(const RID &p_rid, int p_idx, int p_level) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

int NativeMenu::get_item_count(const RID &p_rid) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return 0;
}

bool NativeMenu::is_system_menu(const RID &p_rid) const {
	WARN_PRINT("Global menus are not supported on this platform.");
	return false;
}

void NativeMenu::remove_item(const RID &p_rid, int p_idx) {
	WARN_PRINT("Global menus are not supported on this platform.");
}

void NativeMenu::clear(const RID &p_rid) {
	WARN_PRINT("Global menus are not supported on this platform.");
}
