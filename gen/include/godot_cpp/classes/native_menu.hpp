/**************************************************************************/
/*  native_menu.hpp                                                       */
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

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;
struct Vector2i;

class NativeMenu : public Object {
	GDEXTENSION_CLASS(NativeMenu, Object)

	static NativeMenu *singleton;

public:
	enum Feature {
		FEATURE_GLOBAL_MENU = 0,
		FEATURE_POPUP_MENU = 1,
		FEATURE_OPEN_CLOSE_CALLBACK = 2,
		FEATURE_HOVER_CALLBACK = 3,
		FEATURE_KEY_CALLBACK = 4,
	};

	enum SystemMenus {
		INVALID_MENU_ID = 0,
		MAIN_MENU_ID = 1,
		APPLICATION_MENU_ID = 2,
		WINDOW_MENU_ID = 3,
		HELP_MENU_ID = 4,
		DOCK_MENU_ID = 5,
	};

	static NativeMenu *get_singleton();

	bool has_feature(NativeMenu::Feature p_feature) const;
	bool has_system_menu(NativeMenu::SystemMenus p_menu_id) const;
	RID get_system_menu(NativeMenu::SystemMenus p_menu_id) const;
	String get_system_menu_name(NativeMenu::SystemMenus p_menu_id) const;
	String get_system_menu_text(NativeMenu::SystemMenus p_menu_id) const;
	void set_system_menu_text(NativeMenu::SystemMenus p_menu_id, const String &p_name);
	RID create_menu();
	bool has_menu(const RID &p_rid) const;
	void free_menu(const RID &p_rid);
	Vector2 get_size(const RID &p_rid) const;
	void popup(const RID &p_rid, const Vector2i &p_position);
	void set_interface_direction(const RID &p_rid, bool p_is_rtl);
	void set_popup_open_callback(const RID &p_rid, const Callable &p_callback);
	Callable get_popup_open_callback(const RID &p_rid) const;
	void set_popup_close_callback(const RID &p_rid, const Callable &p_callback);
	Callable get_popup_close_callback(const RID &p_rid) const;
	void set_minimum_width(const RID &p_rid, float p_width);
	float get_minimum_width(const RID &p_rid) const;
	bool is_opened(const RID &p_rid) const;
	int32_t add_submenu_item(const RID &p_rid, const String &p_label, const RID &p_submenu_rid, const Variant &p_tag = nullptr, int32_t p_index = -1);
	int32_t add_item(const RID &p_rid, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = nullptr, Key p_accelerator = (Key)0, int32_t p_index = -1);
	int32_t add_check_item(const RID &p_rid, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = nullptr, Key p_accelerator = (Key)0, int32_t p_index = -1);
	int32_t add_icon_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = nullptr, Key p_accelerator = (Key)0, int32_t p_index = -1);
	int32_t add_icon_check_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = nullptr, Key p_accelerator = (Key)0, int32_t p_index = -1);
	int32_t add_radio_check_item(const RID &p_rid, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = nullptr, Key p_accelerator = (Key)0, int32_t p_index = -1);
	int32_t add_icon_radio_check_item(const RID &p_rid, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = nullptr, Key p_accelerator = (Key)0, int32_t p_index = -1);
	int32_t add_multistate_item(const RID &p_rid, const String &p_label, int32_t p_max_states, int32_t p_default_state, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = nullptr, Key p_accelerator = (Key)0, int32_t p_index = -1);
	int32_t add_separator(const RID &p_rid, int32_t p_index = -1);
	int32_t find_item_index_with_text(const RID &p_rid, const String &p_text) const;
	int32_t find_item_index_with_tag(const RID &p_rid, const Variant &p_tag) const;
	int32_t find_item_index_with_submenu(const RID &p_rid, const RID &p_submenu_rid) const;
	bool is_item_checked(const RID &p_rid, int32_t p_idx) const;
	bool is_item_checkable(const RID &p_rid, int32_t p_idx) const;
	bool is_item_radio_checkable(const RID &p_rid, int32_t p_idx) const;
	Callable get_item_callback(const RID &p_rid, int32_t p_idx) const;
	Callable get_item_key_callback(const RID &p_rid, int32_t p_idx) const;
	Variant get_item_tag(const RID &p_rid, int32_t p_idx) const;
	String get_item_text(const RID &p_rid, int32_t p_idx) const;
	RID get_item_submenu(const RID &p_rid, int32_t p_idx) const;
	Key get_item_accelerator(const RID &p_rid, int32_t p_idx) const;
	bool is_item_disabled(const RID &p_rid, int32_t p_idx) const;
	bool is_item_hidden(const RID &p_rid, int32_t p_idx) const;
	String get_item_tooltip(const RID &p_rid, int32_t p_idx) const;
	int32_t get_item_state(const RID &p_rid, int32_t p_idx) const;
	int32_t get_item_max_states(const RID &p_rid, int32_t p_idx) const;
	Ref<Texture2D> get_item_icon(const RID &p_rid, int32_t p_idx) const;
	int32_t get_item_indentation_level(const RID &p_rid, int32_t p_idx) const;
	void set_item_checked(const RID &p_rid, int32_t p_idx, bool p_checked);
	void set_item_checkable(const RID &p_rid, int32_t p_idx, bool p_checkable);
	void set_item_radio_checkable(const RID &p_rid, int32_t p_idx, bool p_checkable);
	void set_item_callback(const RID &p_rid, int32_t p_idx, const Callable &p_callback);
	void set_item_hover_callbacks(const RID &p_rid, int32_t p_idx, const Callable &p_callback);
	void set_item_key_callback(const RID &p_rid, int32_t p_idx, const Callable &p_key_callback);
	void set_item_tag(const RID &p_rid, int32_t p_idx, const Variant &p_tag);
	void set_item_text(const RID &p_rid, int32_t p_idx, const String &p_text);
	void set_item_submenu(const RID &p_rid, int32_t p_idx, const RID &p_submenu_rid);
	void set_item_accelerator(const RID &p_rid, int32_t p_idx, Key p_keycode);
	void set_item_disabled(const RID &p_rid, int32_t p_idx, bool p_disabled);
	void set_item_hidden(const RID &p_rid, int32_t p_idx, bool p_hidden);
	void set_item_tooltip(const RID &p_rid, int32_t p_idx, const String &p_tooltip);
	void set_item_state(const RID &p_rid, int32_t p_idx, int32_t p_state);
	void set_item_max_states(const RID &p_rid, int32_t p_idx, int32_t p_max_states);
	void set_item_icon(const RID &p_rid, int32_t p_idx, const Ref<Texture2D> &p_icon);
	void set_item_indentation_level(const RID &p_rid, int32_t p_idx, int32_t p_level);
	int32_t get_item_count(const RID &p_rid) const;
	bool is_system_menu(const RID &p_rid) const;
	void remove_item(const RID &p_rid, int32_t p_idx);
	void clear(const RID &p_rid);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~NativeMenu();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(NativeMenu::Feature);
VARIANT_ENUM_CAST(NativeMenu::SystemMenus);

