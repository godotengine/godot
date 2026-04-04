/**************************************************************************/
/*  code_edit.hpp                                                         */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/classes/text_edit.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CodeEdit : public TextEdit {
	GDEXTENSION_CLASS(CodeEdit, TextEdit)

public:
	enum CodeCompletionKind {
		KIND_CLASS = 0,
		KIND_FUNCTION = 1,
		KIND_SIGNAL = 2,
		KIND_VARIABLE = 3,
		KIND_MEMBER = 4,
		KIND_ENUM = 5,
		KIND_CONSTANT = 6,
		KIND_NODE_PATH = 7,
		KIND_FILE_PATH = 8,
		KIND_PLAIN_TEXT = 9,
	};

	enum CodeCompletionLocation {
		LOCATION_LOCAL = 0,
		LOCATION_PARENT_MASK = 256,
		LOCATION_OTHER_USER_CODE = 512,
		LOCATION_OTHER = 1024,
	};

	void set_indent_size(int32_t p_size);
	int32_t get_indent_size() const;
	void set_indent_using_spaces(bool p_use_spaces);
	bool is_indent_using_spaces() const;
	void set_auto_indent_enabled(bool p_enable);
	bool is_auto_indent_enabled() const;
	void set_auto_indent_prefixes(const TypedArray<String> &p_prefixes);
	TypedArray<String> get_auto_indent_prefixes() const;
	void do_indent();
	void indent_lines();
	void unindent_lines();
	void convert_indent(int32_t p_from_line = -1, int32_t p_to_line = -1);
	void set_auto_brace_completion_enabled(bool p_enable);
	bool is_auto_brace_completion_enabled() const;
	void set_highlight_matching_braces_enabled(bool p_enable);
	bool is_highlight_matching_braces_enabled() const;
	void add_auto_brace_completion_pair(const String &p_start_key, const String &p_end_key);
	void set_auto_brace_completion_pairs(const Dictionary &p_pairs);
	Dictionary get_auto_brace_completion_pairs() const;
	bool has_auto_brace_completion_open_key(const String &p_open_key) const;
	bool has_auto_brace_completion_close_key(const String &p_close_key) const;
	String get_auto_brace_completion_close_key(const String &p_open_key) const;
	void set_draw_breakpoints_gutter(bool p_enable);
	bool is_drawing_breakpoints_gutter() const;
	void set_draw_bookmarks_gutter(bool p_enable);
	bool is_drawing_bookmarks_gutter() const;
	void set_draw_executing_lines_gutter(bool p_enable);
	bool is_drawing_executing_lines_gutter() const;
	void set_line_as_breakpoint(int32_t p_line, bool p_breakpointed);
	bool is_line_breakpointed(int32_t p_line) const;
	void clear_breakpointed_lines();
	PackedInt32Array get_breakpointed_lines() const;
	void set_line_as_bookmarked(int32_t p_line, bool p_bookmarked);
	bool is_line_bookmarked(int32_t p_line) const;
	void clear_bookmarked_lines();
	PackedInt32Array get_bookmarked_lines() const;
	void set_line_as_executing(int32_t p_line, bool p_executing);
	bool is_line_executing(int32_t p_line) const;
	void clear_executing_lines();
	PackedInt32Array get_executing_lines() const;
	void set_draw_line_numbers(bool p_enable);
	bool is_draw_line_numbers_enabled() const;
	void set_line_numbers_zero_padded(bool p_enable);
	bool is_line_numbers_zero_padded() const;
	void set_line_numbers_min_digits(int32_t p_count);
	int32_t get_line_numbers_min_digits() const;
	void set_draw_fold_gutter(bool p_enable);
	bool is_drawing_fold_gutter() const;
	void set_line_folding_enabled(bool p_enabled);
	bool is_line_folding_enabled() const;
	bool can_fold_line(int32_t p_line) const;
	void fold_line(int32_t p_line);
	void unfold_line(int32_t p_line);
	void fold_all_lines();
	void unfold_all_lines();
	void toggle_foldable_line(int32_t p_line);
	void toggle_foldable_lines_at_carets();
	bool is_line_folded(int32_t p_line) const;
	TypedArray<int> get_folded_lines() const;
	void create_code_region();
	String get_code_region_start_tag() const;
	String get_code_region_end_tag() const;
	void set_code_region_tags(const String &p_start = "region", const String &p_end = "endregion");
	bool is_line_code_region_start(int32_t p_line) const;
	bool is_line_code_region_end(int32_t p_line) const;
	void add_string_delimiter(const String &p_start_key, const String &p_end_key, bool p_line_only = false);
	void remove_string_delimiter(const String &p_start_key);
	bool has_string_delimiter(const String &p_start_key) const;
	void set_string_delimiters(const TypedArray<String> &p_string_delimiters);
	void clear_string_delimiters();
	TypedArray<String> get_string_delimiters() const;
	int32_t is_in_string(int32_t p_line, int32_t p_column = -1) const;
	void add_comment_delimiter(const String &p_start_key, const String &p_end_key, bool p_line_only = false);
	void remove_comment_delimiter(const String &p_start_key);
	bool has_comment_delimiter(const String &p_start_key) const;
	void set_comment_delimiters(const TypedArray<String> &p_comment_delimiters);
	void clear_comment_delimiters();
	TypedArray<String> get_comment_delimiters() const;
	int32_t is_in_comment(int32_t p_line, int32_t p_column = -1) const;
	String get_delimiter_start_key(int32_t p_delimiter_index) const;
	String get_delimiter_end_key(int32_t p_delimiter_index) const;
	Vector2 get_delimiter_start_position(int32_t p_line, int32_t p_column) const;
	Vector2 get_delimiter_end_position(int32_t p_line, int32_t p_column) const;
	void set_code_hint(const String &p_code_hint);
	void set_code_hint_draw_below(bool p_draw_below);
	String get_text_for_code_completion() const;
	void request_code_completion(bool p_force = false);
	void add_code_completion_option(CodeEdit::CodeCompletionKind p_type, const String &p_display_text, const String &p_insert_text, const Color &p_text_color = Color(1, 1, 1, 1), const Ref<Resource> &p_icon = nullptr, const Variant &p_value = nullptr, int32_t p_location = 1024);
	void update_code_completion_options(bool p_force);
	TypedArray<Dictionary> get_code_completion_options() const;
	Dictionary get_code_completion_option(int32_t p_index) const;
	int32_t get_code_completion_selected_index() const;
	void set_code_completion_selected_index(int32_t p_index);
	void confirm_code_completion(bool p_replace = false);
	void cancel_code_completion();
	void set_code_completion_enabled(bool p_enable);
	bool is_code_completion_enabled() const;
	void set_code_completion_prefixes(const TypedArray<String> &p_prefixes);
	TypedArray<String> get_code_completion_prefixes() const;
	void set_line_length_guidelines(const TypedArray<int> &p_guideline_columns);
	TypedArray<int> get_line_length_guidelines() const;
	void set_symbol_lookup_on_click_enabled(bool p_enable);
	bool is_symbol_lookup_on_click_enabled() const;
	String get_text_for_symbol_lookup() const;
	String get_text_with_cursor_char(int32_t p_line, int32_t p_column) const;
	void set_symbol_lookup_word_as_valid(bool p_valid);
	void set_symbol_tooltip_on_hover_enabled(bool p_enable);
	bool is_symbol_tooltip_on_hover_enabled() const;
	void move_lines_up();
	void move_lines_down();
	void delete_lines();
	void duplicate_selection();
	void duplicate_lines();
	virtual void _confirm_code_completion(bool p_replace);
	virtual void _request_code_completion(bool p_force);
	virtual TypedArray<Dictionary> _filter_code_completion_candidates(const TypedArray<Dictionary> &p_candidates) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		TextEdit::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_confirm_code_completion), decltype(&T::_confirm_code_completion)>) {
			BIND_VIRTUAL_METHOD(T, _confirm_code_completion, 2586408642);
		}
		if constexpr (!std::is_same_v<decltype(&B::_request_code_completion), decltype(&T::_request_code_completion)>) {
			BIND_VIRTUAL_METHOD(T, _request_code_completion, 2586408642);
		}
		if constexpr (!std::is_same_v<decltype(&B::_filter_code_completion_candidates), decltype(&T::_filter_code_completion_candidates)>) {
			BIND_VIRTUAL_METHOD(T, _filter_code_completion_candidates, 2560709669);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(CodeEdit::CodeCompletionKind);
VARIANT_ENUM_CAST(CodeEdit::CodeCompletionLocation);

