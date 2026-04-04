/**************************************************************************/
/*  line_edit.hpp                                                         */
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
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PopupMenu;
class Texture2D;

class LineEdit : public Control {
	GDEXTENSION_CLASS(LineEdit, Control)

public:
	enum MenuItems {
		MENU_CUT = 0,
		MENU_COPY = 1,
		MENU_PASTE = 2,
		MENU_CLEAR = 3,
		MENU_SELECT_ALL = 4,
		MENU_UNDO = 5,
		MENU_REDO = 6,
		MENU_SUBMENU_TEXT_DIR = 7,
		MENU_DIR_INHERITED = 8,
		MENU_DIR_AUTO = 9,
		MENU_DIR_LTR = 10,
		MENU_DIR_RTL = 11,
		MENU_DISPLAY_UCC = 12,
		MENU_SUBMENU_INSERT_UCC = 13,
		MENU_INSERT_LRM = 14,
		MENU_INSERT_RLM = 15,
		MENU_INSERT_LRE = 16,
		MENU_INSERT_RLE = 17,
		MENU_INSERT_LRO = 18,
		MENU_INSERT_RLO = 19,
		MENU_INSERT_PDF = 20,
		MENU_INSERT_ALM = 21,
		MENU_INSERT_LRI = 22,
		MENU_INSERT_RLI = 23,
		MENU_INSERT_FSI = 24,
		MENU_INSERT_PDI = 25,
		MENU_INSERT_ZWJ = 26,
		MENU_INSERT_ZWNJ = 27,
		MENU_INSERT_WJ = 28,
		MENU_INSERT_SHY = 29,
		MENU_EMOJI_AND_SYMBOL = 30,
		MENU_MAX = 31,
	};

	enum VirtualKeyboardType {
		KEYBOARD_TYPE_DEFAULT = 0,
		KEYBOARD_TYPE_MULTILINE = 1,
		KEYBOARD_TYPE_NUMBER = 2,
		KEYBOARD_TYPE_NUMBER_DECIMAL = 3,
		KEYBOARD_TYPE_PHONE = 4,
		KEYBOARD_TYPE_EMAIL_ADDRESS = 5,
		KEYBOARD_TYPE_PASSWORD = 6,
		KEYBOARD_TYPE_URL = 7,
	};

	enum ExpandMode {
		EXPAND_MODE_ORIGINAL_SIZE = 0,
		EXPAND_MODE_FIT_TO_TEXT = 1,
		EXPAND_MODE_FIT_TO_LINE_EDIT = 2,
	};

	bool has_ime_text() const;
	void cancel_ime();
	void apply_ime();
	void set_horizontal_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_horizontal_alignment() const;
	void edit(bool p_hide_focus = false);
	void unedit();
	bool is_editing() const;
	void set_keep_editing_on_text_submit(bool p_enable);
	bool is_editing_kept_on_text_submit() const;
	void clear();
	void select(int32_t p_from = 0, int32_t p_to = -1);
	void select_all();
	void deselect();
	bool has_undo() const;
	bool has_redo() const;
	bool has_selection() const;
	String get_selected_text();
	int32_t get_selection_from_column() const;
	int32_t get_selection_to_column() const;
	void set_text(const String &p_text);
	String get_text() const;
	bool get_draw_control_chars() const;
	void set_draw_control_chars(bool p_enable);
	void set_text_direction(Control::TextDirection p_direction);
	Control::TextDirection get_text_direction() const;
	void set_language(const String &p_language);
	String get_language() const;
	void set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser);
	TextServer::StructuredTextParser get_structured_text_bidi_override() const;
	void set_structured_text_bidi_override_options(const Array &p_args);
	Array get_structured_text_bidi_override_options() const;
	void set_placeholder(const String &p_text);
	String get_placeholder() const;
	void set_caret_column(int32_t p_position);
	int32_t get_caret_column() const;
	int32_t get_next_composite_character_column(int32_t p_column) const;
	int32_t get_previous_composite_character_column(int32_t p_column) const;
	float get_scroll_offset() const;
	void set_expand_to_text_length_enabled(bool p_enabled);
	bool is_expand_to_text_length_enabled() const;
	void set_caret_blink_enabled(bool p_enabled);
	bool is_caret_blink_enabled() const;
	void set_caret_mid_grapheme_enabled(bool p_enabled);
	bool is_caret_mid_grapheme_enabled() const;
	void set_caret_force_displayed(bool p_enabled);
	bool is_caret_force_displayed() const;
	void set_caret_blink_interval(float p_interval);
	float get_caret_blink_interval() const;
	void set_max_length(int32_t p_chars);
	int32_t get_max_length() const;
	void insert_text_at_caret(const String &p_text);
	void delete_char_at_caret();
	void delete_text(int32_t p_from_column, int32_t p_to_column);
	void set_editable(bool p_enabled);
	bool is_editable() const;
	void set_secret(bool p_enabled);
	bool is_secret() const;
	void set_secret_character(const String &p_character);
	String get_secret_character() const;
	void menu_option(int32_t p_option);
	PopupMenu *get_menu() const;
	bool is_menu_visible() const;
	void set_context_menu_enabled(bool p_enable);
	bool is_context_menu_enabled();
	void set_emoji_menu_enabled(bool p_enable);
	bool is_emoji_menu_enabled() const;
	void set_backspace_deletes_composite_character_enabled(bool p_enable);
	bool is_backspace_deletes_composite_character_enabled() const;
	void set_virtual_keyboard_enabled(bool p_enable);
	bool is_virtual_keyboard_enabled() const;
	void set_virtual_keyboard_show_on_focus(bool p_show_on_focus);
	bool get_virtual_keyboard_show_on_focus() const;
	void set_virtual_keyboard_type(LineEdit::VirtualKeyboardType p_type);
	LineEdit::VirtualKeyboardType get_virtual_keyboard_type() const;
	void set_clear_button_enabled(bool p_enable);
	bool is_clear_button_enabled() const;
	void set_shortcut_keys_enabled(bool p_enable);
	bool is_shortcut_keys_enabled() const;
	void set_middle_mouse_paste_enabled(bool p_enable);
	bool is_middle_mouse_paste_enabled() const;
	void set_selecting_enabled(bool p_enable);
	bool is_selecting_enabled() const;
	void set_deselect_on_focus_loss_enabled(bool p_enable);
	bool is_deselect_on_focus_loss_enabled() const;
	void set_drag_and_drop_selection_enabled(bool p_enable);
	bool is_drag_and_drop_selection_enabled() const;
	void set_right_icon(const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_right_icon();
	void set_icon_expand_mode(LineEdit::ExpandMode p_mode);
	LineEdit::ExpandMode get_icon_expand_mode() const;
	void set_right_icon_scale(float p_scale);
	float get_right_icon_scale() const;
	void set_flat(bool p_enabled);
	bool is_flat() const;
	void set_select_all_on_focus(bool p_enabled);
	bool is_select_all_on_focus() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Control::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(LineEdit::MenuItems);
VARIANT_ENUM_CAST(LineEdit::VirtualKeyboardType);
VARIANT_ENUM_CAST(LineEdit::ExpandMode);

