/**************************************************************************/
/*  text_edit.hpp                                                         */
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
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/text_server.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/rect2i.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class HScrollBar;
class PopupMenu;
class SyntaxHighlighter;
class Texture2D;
class VScrollBar;

class TextEdit : public Control {
	GDEXTENSION_CLASS(TextEdit, Control)

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

	enum EditAction {
		ACTION_NONE = 0,
		ACTION_TYPING = 1,
		ACTION_BACKSPACE = 2,
		ACTION_DELETE = 3,
	};

	enum SearchFlags {
		SEARCH_MATCH_CASE = 1,
		SEARCH_WHOLE_WORDS = 2,
		SEARCH_BACKWARDS = 4,
	};

	enum CaretType {
		CARET_TYPE_LINE = 0,
		CARET_TYPE_BLOCK = 1,
	};

	enum SelectionMode {
		SELECTION_MODE_NONE = 0,
		SELECTION_MODE_SHIFT = 1,
		SELECTION_MODE_POINTER = 2,
		SELECTION_MODE_WORD = 3,
		SELECTION_MODE_LINE = 4,
	};

	enum LineWrappingMode {
		LINE_WRAPPING_NONE = 0,
		LINE_WRAPPING_BOUNDARY = 1,
	};

	enum GutterType {
		GUTTER_TYPE_STRING = 0,
		GUTTER_TYPE_ICON = 1,
		GUTTER_TYPE_CUSTOM = 2,
	};

	bool has_ime_text() const;
	void cancel_ime();
	void apply_ime();
	void set_editable(bool p_enabled);
	bool is_editable() const;
	void set_text_direction(Control::TextDirection p_direction);
	Control::TextDirection get_text_direction() const;
	void set_language(const String &p_language);
	String get_language() const;
	void set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser);
	TextServer::StructuredTextParser get_structured_text_bidi_override() const;
	void set_structured_text_bidi_override_options(const Array &p_args);
	Array get_structured_text_bidi_override_options() const;
	void set_tab_size(int32_t p_size);
	int32_t get_tab_size() const;
	void set_indent_wrapped_lines(bool p_enabled);
	bool is_indent_wrapped_lines() const;
	void set_tab_input_mode(bool p_enabled);
	bool get_tab_input_mode() const;
	void set_overtype_mode_enabled(bool p_enabled);
	bool is_overtype_mode_enabled() const;
	void set_context_menu_enabled(bool p_enabled);
	bool is_context_menu_enabled() const;
	void set_emoji_menu_enabled(bool p_enable);
	bool is_emoji_menu_enabled() const;
	void set_backspace_deletes_composite_character_enabled(bool p_enable);
	bool is_backspace_deletes_composite_character_enabled() const;
	void set_shortcut_keys_enabled(bool p_enabled);
	bool is_shortcut_keys_enabled() const;
	void set_virtual_keyboard_enabled(bool p_enabled);
	bool is_virtual_keyboard_enabled() const;
	void set_virtual_keyboard_show_on_focus(bool p_show_on_focus);
	bool get_virtual_keyboard_show_on_focus() const;
	void set_middle_mouse_paste_enabled(bool p_enabled);
	bool is_middle_mouse_paste_enabled() const;
	void set_empty_selection_clipboard_enabled(bool p_enabled);
	bool is_empty_selection_clipboard_enabled() const;
	void clear();
	void set_text(const String &p_text);
	String get_text() const;
	int32_t get_line_count() const;
	void set_placeholder(const String &p_text);
	String get_placeholder() const;
	void set_line(int32_t p_line, const String &p_new_text);
	String get_line(int32_t p_line) const;
	String get_line_with_ime(int32_t p_line) const;
	int32_t get_line_width(int32_t p_line, int32_t p_wrap_index = -1) const;
	int32_t get_line_height() const;
	int32_t get_indent_level(int32_t p_line) const;
	int32_t get_first_non_whitespace_column(int32_t p_line) const;
	void swap_lines(int32_t p_from_line, int32_t p_to_line);
	void insert_line_at(int32_t p_line, const String &p_text);
	void remove_line_at(int32_t p_line, bool p_move_carets_down = true);
	void insert_text_at_caret(const String &p_text, int32_t p_caret_index = -1);
	void insert_text(const String &p_text, int32_t p_line, int32_t p_column, bool p_before_selection_begin = true, bool p_before_selection_end = false);
	void remove_text(int32_t p_from_line, int32_t p_from_column, int32_t p_to_line, int32_t p_to_column);
	int32_t get_last_unhidden_line() const;
	int32_t get_next_visible_line_offset_from(int32_t p_line, int32_t p_visible_amount) const;
	Vector2i get_next_visible_line_index_offset_from(int32_t p_line, int32_t p_wrap_index, int32_t p_visible_amount) const;
	void backspace(int32_t p_caret_index = -1);
	void cut(int32_t p_caret_index = -1);
	void copy(int32_t p_caret_index = -1);
	void paste(int32_t p_caret_index = -1);
	void paste_primary_clipboard(int32_t p_caret_index = -1);
	void start_action(TextEdit::EditAction p_action);
	void end_action();
	void begin_complex_operation();
	void end_complex_operation();
	bool has_undo() const;
	bool has_redo() const;
	void undo();
	void redo();
	void clear_undo_history();
	void tag_saved_version();
	uint32_t get_version() const;
	uint32_t get_saved_version() const;
	void set_search_text(const String &p_search_text);
	void set_search_flags(uint32_t p_flags);
	Vector2i search(const String &p_text, uint32_t p_flags, int32_t p_from_line, int32_t p_from_column) const;
	void set_tooltip_request_func(const Callable &p_callback);
	Vector2 get_local_mouse_pos() const;
	String get_word_at_pos(const Vector2 &p_position) const;
	Vector2i get_line_column_at_pos(const Vector2i &p_position, bool p_clamp_line = true, bool p_clamp_column = true) const;
	Vector2i get_pos_at_line_column(int32_t p_line, int32_t p_column) const;
	Rect2i get_rect_at_line_column(int32_t p_line, int32_t p_column) const;
	int32_t get_minimap_line_at_pos(const Vector2i &p_position) const;
	bool is_dragging_cursor() const;
	bool is_mouse_over_selection(bool p_edges, int32_t p_caret_index = -1) const;
	void set_caret_type(TextEdit::CaretType p_type);
	TextEdit::CaretType get_caret_type() const;
	void set_caret_blink_enabled(bool p_enable);
	bool is_caret_blink_enabled() const;
	void set_caret_blink_interval(float p_interval);
	float get_caret_blink_interval() const;
	void set_draw_caret_when_editable_disabled(bool p_enable);
	bool is_drawing_caret_when_editable_disabled() const;
	void set_move_caret_on_right_click_enabled(bool p_enable);
	bool is_move_caret_on_right_click_enabled() const;
	void set_caret_mid_grapheme_enabled(bool p_enabled);
	bool is_caret_mid_grapheme_enabled() const;
	void set_multiple_carets_enabled(bool p_enabled);
	bool is_multiple_carets_enabled() const;
	int32_t add_caret(int32_t p_line, int32_t p_column);
	void remove_caret(int32_t p_caret);
	void remove_secondary_carets();
	int32_t get_caret_count() const;
	void add_caret_at_carets(bool p_below);
	PackedInt32Array get_sorted_carets(bool p_include_ignored_carets = false) const;
	void collapse_carets(int32_t p_from_line, int32_t p_from_column, int32_t p_to_line, int32_t p_to_column, bool p_inclusive = false);
	void merge_overlapping_carets();
	void begin_multicaret_edit();
	void end_multicaret_edit();
	bool is_in_mulitcaret_edit() const;
	bool multicaret_edit_ignore_caret(int32_t p_caret_index) const;
	bool is_caret_visible(int32_t p_caret_index = 0) const;
	Vector2 get_caret_draw_pos(int32_t p_caret_index = 0) const;
	void set_caret_line(int32_t p_line, bool p_adjust_viewport = true, bool p_can_be_hidden = true, int32_t p_wrap_index = 0, int32_t p_caret_index = 0);
	int32_t get_caret_line(int32_t p_caret_index = 0) const;
	void set_caret_column(int32_t p_column, bool p_adjust_viewport = true, int32_t p_caret_index = 0);
	int32_t get_caret_column(int32_t p_caret_index = 0) const;
	int32_t get_next_composite_character_column(int32_t p_line, int32_t p_column) const;
	int32_t get_previous_composite_character_column(int32_t p_line, int32_t p_column) const;
	int32_t get_caret_wrap_index(int32_t p_caret_index = 0) const;
	String get_word_under_caret(int32_t p_caret_index = -1) const;
	void set_use_default_word_separators(bool p_enabled);
	bool is_default_word_separators_enabled() const;
	void set_use_custom_word_separators(bool p_enabled);
	bool is_custom_word_separators_enabled() const;
	void set_custom_word_separators(const String &p_custom_word_separators);
	String get_custom_word_separators() const;
	void set_selecting_enabled(bool p_enable);
	bool is_selecting_enabled() const;
	void set_deselect_on_focus_loss_enabled(bool p_enable);
	bool is_deselect_on_focus_loss_enabled() const;
	void set_drag_and_drop_selection_enabled(bool p_enable);
	bool is_drag_and_drop_selection_enabled() const;
	void set_selection_mode(TextEdit::SelectionMode p_mode);
	TextEdit::SelectionMode get_selection_mode() const;
	void select_all();
	void select_word_under_caret(int32_t p_caret_index = -1);
	void add_selection_for_next_occurrence();
	void skip_selection_for_next_occurrence();
	void select(int32_t p_origin_line, int32_t p_origin_column, int32_t p_caret_line, int32_t p_caret_column, int32_t p_caret_index = 0);
	bool has_selection(int32_t p_caret_index = -1) const;
	String get_selected_text(int32_t p_caret_index = -1);
	int32_t get_selection_at_line_column(int32_t p_line, int32_t p_column, bool p_include_edges = true, bool p_only_selections = true) const;
	TypedArray<Vector2i> get_line_ranges_from_carets(bool p_only_selections = false, bool p_merge_adjacent = true) const;
	int32_t get_selection_origin_line(int32_t p_caret_index = 0) const;
	int32_t get_selection_origin_column(int32_t p_caret_index = 0) const;
	void set_selection_origin_line(int32_t p_line, bool p_can_be_hidden = true, int32_t p_wrap_index = -1, int32_t p_caret_index = 0);
	void set_selection_origin_column(int32_t p_column, int32_t p_caret_index = 0);
	int32_t get_selection_from_line(int32_t p_caret_index = 0) const;
	int32_t get_selection_from_column(int32_t p_caret_index = 0) const;
	int32_t get_selection_to_line(int32_t p_caret_index = 0) const;
	int32_t get_selection_to_column(int32_t p_caret_index = 0) const;
	bool is_caret_after_selection_origin(int32_t p_caret_index = 0) const;
	void deselect(int32_t p_caret_index = -1);
	void delete_selection(int32_t p_caret_index = -1);
	void set_line_wrapping_mode(TextEdit::LineWrappingMode p_mode);
	TextEdit::LineWrappingMode get_line_wrapping_mode() const;
	void set_autowrap_mode(TextServer::AutowrapMode p_autowrap_mode);
	TextServer::AutowrapMode get_autowrap_mode() const;
	bool is_line_wrapped(int32_t p_line) const;
	int32_t get_line_wrap_count(int32_t p_line) const;
	int32_t get_line_wrap_index_at_column(int32_t p_line, int32_t p_column) const;
	PackedStringArray get_line_wrapped_text(int32_t p_line) const;
	void set_smooth_scroll_enabled(bool p_enable);
	bool is_smooth_scroll_enabled() const;
	VScrollBar *get_v_scroll_bar() const;
	HScrollBar *get_h_scroll_bar() const;
	void set_v_scroll(double p_value);
	double get_v_scroll() const;
	void set_h_scroll(int32_t p_value);
	int32_t get_h_scroll() const;
	void set_scroll_past_end_of_file_enabled(bool p_enable);
	bool is_scroll_past_end_of_file_enabled() const;
	void set_v_scroll_speed(float p_speed);
	float get_v_scroll_speed() const;
	void set_fit_content_height_enabled(bool p_enabled);
	bool is_fit_content_height_enabled() const;
	void set_fit_content_width_enabled(bool p_enabled);
	bool is_fit_content_width_enabled() const;
	double get_scroll_pos_for_line(int32_t p_line, int32_t p_wrap_index = 0) const;
	void set_line_as_first_visible(int32_t p_line, int32_t p_wrap_index = 0);
	int32_t get_first_visible_line() const;
	void set_line_as_center_visible(int32_t p_line, int32_t p_wrap_index = 0);
	void set_line_as_last_visible(int32_t p_line, int32_t p_wrap_index = 0);
	int32_t get_last_full_visible_line() const;
	int32_t get_last_full_visible_line_wrap_index() const;
	int32_t get_visible_line_count() const;
	int32_t get_visible_line_count_in_range(int32_t p_from_line, int32_t p_to_line) const;
	int32_t get_total_visible_line_count() const;
	void adjust_viewport_to_caret(int32_t p_caret_index = 0);
	void center_viewport_to_caret(int32_t p_caret_index = 0);
	void set_draw_minimap(bool p_enabled);
	bool is_drawing_minimap() const;
	void set_minimap_width(int32_t p_width);
	int32_t get_minimap_width() const;
	int32_t get_minimap_visible_lines() const;
	void add_gutter(int32_t p_at = -1);
	void remove_gutter(int32_t p_gutter);
	int32_t get_gutter_count() const;
	void set_gutter_name(int32_t p_gutter, const String &p_name);
	String get_gutter_name(int32_t p_gutter) const;
	void set_gutter_type(int32_t p_gutter, TextEdit::GutterType p_type);
	TextEdit::GutterType get_gutter_type(int32_t p_gutter) const;
	void set_gutter_width(int32_t p_gutter, int32_t p_width);
	int32_t get_gutter_width(int32_t p_gutter) const;
	void set_gutter_draw(int32_t p_gutter, bool p_draw);
	bool is_gutter_drawn(int32_t p_gutter) const;
	void set_gutter_clickable(int32_t p_gutter, bool p_clickable);
	bool is_gutter_clickable(int32_t p_gutter) const;
	void set_gutter_overwritable(int32_t p_gutter, bool p_overwritable);
	bool is_gutter_overwritable(int32_t p_gutter) const;
	void merge_gutters(int32_t p_from_line, int32_t p_to_line);
	void set_gutter_custom_draw(int32_t p_column, const Callable &p_draw_callback);
	int32_t get_total_gutter_width() const;
	void set_line_gutter_metadata(int32_t p_line, int32_t p_gutter, const Variant &p_metadata);
	Variant get_line_gutter_metadata(int32_t p_line, int32_t p_gutter) const;
	void set_line_gutter_text(int32_t p_line, int32_t p_gutter, const String &p_text);
	String get_line_gutter_text(int32_t p_line, int32_t p_gutter) const;
	void set_line_gutter_icon(int32_t p_line, int32_t p_gutter, const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_line_gutter_icon(int32_t p_line, int32_t p_gutter) const;
	void set_line_gutter_item_color(int32_t p_line, int32_t p_gutter, const Color &p_color);
	Color get_line_gutter_item_color(int32_t p_line, int32_t p_gutter) const;
	void set_line_gutter_clickable(int32_t p_line, int32_t p_gutter, bool p_clickable);
	bool is_line_gutter_clickable(int32_t p_line, int32_t p_gutter) const;
	void set_line_background_color(int32_t p_line, const Color &p_color);
	Color get_line_background_color(int32_t p_line) const;
	void set_syntax_highlighter(const Ref<SyntaxHighlighter> &p_syntax_highlighter);
	Ref<SyntaxHighlighter> get_syntax_highlighter() const;
	void set_highlight_current_line(bool p_enabled);
	bool is_highlight_current_line_enabled() const;
	void set_highlight_all_occurrences(bool p_enabled);
	bool is_highlight_all_occurrences_enabled() const;
	bool get_draw_control_chars() const;
	void set_draw_control_chars(bool p_enabled);
	void set_draw_tabs(bool p_enabled);
	bool is_drawing_tabs() const;
	void set_draw_spaces(bool p_enabled);
	bool is_drawing_spaces() const;
	PopupMenu *get_menu() const;
	bool is_menu_visible() const;
	void menu_option(int32_t p_option);
	void adjust_carets_after_edit(int32_t p_caret, int32_t p_from_line, int32_t p_from_col, int32_t p_to_line, int32_t p_to_col);
	PackedInt32Array get_caret_index_edit_order();
	int32_t get_selection_line(int32_t p_caret_index = 0) const;
	int32_t get_selection_column(int32_t p_caret_index = 0) const;
	virtual void _handle_unicode_input(int32_t p_unicode_char, int32_t p_caret_index);
	virtual void _backspace(int32_t p_caret_index);
	virtual void _cut(int32_t p_caret_index);
	virtual void _copy(int32_t p_caret_index);
	virtual void _paste(int32_t p_caret_index);
	virtual void _paste_primary_clipboard(int32_t p_caret_index);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Control::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_handle_unicode_input), decltype(&T::_handle_unicode_input)>) {
			BIND_VIRTUAL_METHOD(T, _handle_unicode_input, 3937882851);
		}
		if constexpr (!std::is_same_v<decltype(&B::_backspace), decltype(&T::_backspace)>) {
			BIND_VIRTUAL_METHOD(T, _backspace, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_cut), decltype(&T::_cut)>) {
			BIND_VIRTUAL_METHOD(T, _cut, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_copy), decltype(&T::_copy)>) {
			BIND_VIRTUAL_METHOD(T, _copy, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_paste), decltype(&T::_paste)>) {
			BIND_VIRTUAL_METHOD(T, _paste, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_paste_primary_clipboard), decltype(&T::_paste_primary_clipboard)>) {
			BIND_VIRTUAL_METHOD(T, _paste_primary_clipboard, 1286410249);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TextEdit::MenuItems);
VARIANT_ENUM_CAST(TextEdit::EditAction);
VARIANT_ENUM_CAST(TextEdit::SearchFlags);
VARIANT_ENUM_CAST(TextEdit::CaretType);
VARIANT_ENUM_CAST(TextEdit::SelectionMode);
VARIANT_ENUM_CAST(TextEdit::LineWrappingMode);
VARIANT_ENUM_CAST(TextEdit::GutterType);

