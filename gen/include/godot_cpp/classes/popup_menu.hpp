/**************************************************************************/
/*  popup_menu.hpp                                                        */
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

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/native_menu.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/popup.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class InputEvent;
class Shortcut;
class Texture2D;

class PopupMenu : public Popup {
	GDEXTENSION_CLASS(PopupMenu, Popup)

public:
	bool activate_item_by_event(const Ref<InputEvent> &p_event, bool p_for_global_only = false);
	void set_prefer_native_menu(bool p_enabled);
	bool is_prefer_native_menu() const;
	bool is_native_menu() const;
	void add_item(const String &p_label, int32_t p_id = -1, Key p_accel = (Key)0);
	void add_icon_item(const Ref<Texture2D> &p_texture, const String &p_label, int32_t p_id = -1, Key p_accel = (Key)0);
	void add_check_item(const String &p_label, int32_t p_id = -1, Key p_accel = (Key)0);
	void add_icon_check_item(const Ref<Texture2D> &p_texture, const String &p_label, int32_t p_id = -1, Key p_accel = (Key)0);
	void add_radio_check_item(const String &p_label, int32_t p_id = -1, Key p_accel = (Key)0);
	void add_icon_radio_check_item(const Ref<Texture2D> &p_texture, const String &p_label, int32_t p_id = -1, Key p_accel = (Key)0);
	void add_multistate_item(const String &p_label, int32_t p_max_states, int32_t p_default_state = 0, int32_t p_id = -1, Key p_accel = (Key)0);
	void add_shortcut(const Ref<Shortcut> &p_shortcut, int32_t p_id = -1, bool p_global = false, bool p_allow_echo = false);
	void add_icon_shortcut(const Ref<Texture2D> &p_texture, const Ref<Shortcut> &p_shortcut, int32_t p_id = -1, bool p_global = false, bool p_allow_echo = false);
	void add_check_shortcut(const Ref<Shortcut> &p_shortcut, int32_t p_id = -1, bool p_global = false);
	void add_icon_check_shortcut(const Ref<Texture2D> &p_texture, const Ref<Shortcut> &p_shortcut, int32_t p_id = -1, bool p_global = false);
	void add_radio_check_shortcut(const Ref<Shortcut> &p_shortcut, int32_t p_id = -1, bool p_global = false);
	void add_icon_radio_check_shortcut(const Ref<Texture2D> &p_texture, const Ref<Shortcut> &p_shortcut, int32_t p_id = -1, bool p_global = false);
	void add_submenu_item(const String &p_label, const String &p_submenu, int32_t p_id = -1);
	void add_submenu_node_item(const String &p_label, PopupMenu *p_submenu, int32_t p_id = -1);
	void set_item_text(int32_t p_index, const String &p_text);
	void set_item_text_direction(int32_t p_index, Control::TextDirection p_direction);
	void set_item_language(int32_t p_index, const String &p_language);
	void set_item_auto_translate_mode(int32_t p_index, Node::AutoTranslateMode p_mode);
	void set_item_icon(int32_t p_index, const Ref<Texture2D> &p_icon);
	void set_item_icon_max_width(int32_t p_index, int32_t p_width);
	void set_item_icon_modulate(int32_t p_index, const Color &p_modulate);
	void set_item_checked(int32_t p_index, bool p_checked);
	void set_item_id(int32_t p_index, int32_t p_id);
	void set_item_accelerator(int32_t p_index, Key p_accel);
	void set_item_metadata(int32_t p_index, const Variant &p_metadata);
	void set_item_disabled(int32_t p_index, bool p_disabled);
	void set_item_submenu(int32_t p_index, const String &p_submenu);
	void set_item_submenu_node(int32_t p_index, PopupMenu *p_submenu);
	void set_item_as_separator(int32_t p_index, bool p_enable);
	void set_item_as_checkable(int32_t p_index, bool p_enable);
	void set_item_as_radio_checkable(int32_t p_index, bool p_enable);
	void set_item_tooltip(int32_t p_index, const String &p_tooltip);
	void set_item_shortcut(int32_t p_index, const Ref<Shortcut> &p_shortcut, bool p_global = false);
	void set_item_indent(int32_t p_index, int32_t p_indent);
	void set_item_multistate(int32_t p_index, int32_t p_state);
	void set_item_multistate_max(int32_t p_index, int32_t p_max_states);
	void set_item_shortcut_disabled(int32_t p_index, bool p_disabled);
	void toggle_item_checked(int32_t p_index);
	void toggle_item_multistate(int32_t p_index);
	String get_item_text(int32_t p_index) const;
	Control::TextDirection get_item_text_direction(int32_t p_index) const;
	String get_item_language(int32_t p_index) const;
	Node::AutoTranslateMode get_item_auto_translate_mode(int32_t p_index) const;
	Ref<Texture2D> get_item_icon(int32_t p_index) const;
	int32_t get_item_icon_max_width(int32_t p_index) const;
	Color get_item_icon_modulate(int32_t p_index) const;
	bool is_item_checked(int32_t p_index) const;
	int32_t get_item_id(int32_t p_index) const;
	int32_t get_item_index(int32_t p_id) const;
	Key get_item_accelerator(int32_t p_index) const;
	Variant get_item_metadata(int32_t p_index) const;
	bool is_item_disabled(int32_t p_index) const;
	String get_item_submenu(int32_t p_index) const;
	PopupMenu *get_item_submenu_node(int32_t p_index) const;
	bool is_item_separator(int32_t p_index) const;
	bool is_item_checkable(int32_t p_index) const;
	bool is_item_radio_checkable(int32_t p_index) const;
	bool is_item_shortcut_disabled(int32_t p_index) const;
	String get_item_tooltip(int32_t p_index) const;
	Ref<Shortcut> get_item_shortcut(int32_t p_index) const;
	int32_t get_item_indent(int32_t p_index) const;
	int32_t get_item_multistate_max(int32_t p_index) const;
	int32_t get_item_multistate(int32_t p_index) const;
	void set_focused_item(int32_t p_index);
	int32_t get_focused_item() const;
	void set_item_count(int32_t p_count);
	int32_t get_item_count() const;
	void scroll_to_item(int32_t p_index);
	void remove_item(int32_t p_index);
	void add_separator(const String &p_label = String(), int32_t p_id = -1);
	void clear(bool p_free_submenus = false);
	void set_hide_on_item_selection(bool p_enable);
	bool is_hide_on_item_selection() const;
	void set_hide_on_checkable_item_selection(bool p_enable);
	bool is_hide_on_checkable_item_selection() const;
	void set_hide_on_state_item_selection(bool p_enable);
	bool is_hide_on_state_item_selection() const;
	void set_submenu_popup_delay(float p_seconds);
	float get_submenu_popup_delay() const;
	void set_allow_search(bool p_allow);
	bool get_allow_search() const;
	bool is_system_menu() const;
	void set_system_menu(NativeMenu::SystemMenus p_system_menu_id);
	NativeMenu::SystemMenus get_system_menu() const;
	void set_shrink_height(bool p_shrink);
	bool get_shrink_height() const;
	void set_shrink_width(bool p_shrink);
	bool get_shrink_width() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Popup::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

