/*************************************************************************/
/*  code_edit.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef CODEEDIT_H
#define CODEEDIT_H

#include "scene/gui/text_edit.h"

class CodeEdit : public TextEdit {
	GDCLASS(CodeEdit, TextEdit)

public:
	/* Keep enum in sync with:                                           */
	/* /core/object/script_language.h - ScriptCodeCompletionOption::Kind */
	enum CodeCompletionKind {
		KIND_CLASS,
		KIND_FUNCTION,
		KIND_SIGNAL,
		KIND_VARIABLE,
		KIND_MEMBER,
		KIND_ENUM,
		KIND_CONSTANT,
		KIND_NODE_PATH,
		KIND_FILE_PATH,
		KIND_PLAIN_TEXT,
	};

private:
	/* Indent management */
	int indent_size = 4;
	String indent_text = "\t";

	bool auto_indent = false;
	Set<char32_t> auto_indent_prefixes;

	bool indent_using_spaces = false;
	int _calculate_spaces_till_next_left_indent(int p_column) const;
	int _calculate_spaces_till_next_right_indent(int p_column) const;

	void _new_line(bool p_split_current_line = true, bool p_above = false);

	/* Main Gutter */
	enum MainGutterType {
		MAIN_GUTTER_BREAKPOINT = 0x01,
		MAIN_GUTTER_BOOKMARK = 0x02,
		MAIN_GUTTER_EXECUTING = 0x04
	};

	int main_gutter = -1;
	void _update_draw_main_gutter();
	void _main_gutter_draw_callback(int p_line, int p_gutter, const Rect2 &p_region);

	// breakpoints
	HashMap<int, bool> breakpointed_lines;
	bool draw_breakpoints = false;
	Color breakpoint_color = Color(1, 1, 1);
	Ref<Texture2D> breakpoint_icon = Ref<Texture2D>();

	// bookmarks
	bool draw_bookmarks = false;
	Color bookmark_color = Color(1, 1, 1);
	Ref<Texture2D> bookmark_icon = Ref<Texture2D>();

	// executing lines
	bool draw_executing_lines = false;
	Color executing_line_color = Color(1, 1, 1);
	Ref<Texture2D> executing_line_icon = Ref<Texture2D>();

	/* Line numbers */
	int line_number_gutter = -1;
	int line_number_digits = 0;
	String line_number_padding = " ";
	Color line_number_color = Color(1, 1, 1);
	void _line_number_draw_callback(int p_line, int p_gutter, const Rect2 &p_region);

	/* Fold Gutter */
	int fold_gutter = -1;
	bool draw_fold_gutter = false;
	Color folding_color = Color(1, 1, 1);
	Ref<Texture2D> can_fold_icon = Ref<Texture2D>();
	Ref<Texture2D> folded_icon = Ref<Texture2D>();
	void _fold_gutter_draw_callback(int p_line, int p_gutter, Rect2 p_region);

	void _gutter_clicked(int p_line, int p_gutter);
	void _update_gutter_indexes();

	/* Line Folding */
	bool line_folding_enabled = true;

	/* Delimiters */
	enum DelimiterType {
		TYPE_STRING,
		TYPE_COMMENT,
	};

	struct Delimiter {
		DelimiterType type;
		String start_key = "";
		String end_key = "";
		bool line_only = true;
	};
	bool setting_delimiters = false;
	Vector<Delimiter> delimiters;
	/*
	 *  Vector entry per line, contains a Map of column numbers to delimiter index, -1 marks the end of a region.
	 *  e.g the following text will be stored as so:
	 *
	 *  0: nothing here
	 *  1:
	 *  2: # test
	 *  3: "test" text "multiline
	 *  4:
	 *  5: test
	 *  6: string"
	 *
	 *  Vector [
	 *      0 = []
	 *      1 = []
	 *      2 = [
	 *          1 = 1
	 *          6 = -1
	 *      ]
	 *      3 = [
	 *	        1 = 0
	 *          6 = -1
	 *          13 = 0
	 *      ]
	 *      4 = [
	 *          0 = 0
	 *      ]
	 *      5 = [
	 *          5 = 0
	 *      ]
	 *      6 = [
	 *          7 = -1
	 *      ]
	 *  ]
	 */
	Vector<Map<int, int>> delimiter_cache;

	void _update_delimiter_cache(int p_from_line = 0, int p_to_line = -1);
	int _is_in_delimiter(int p_line, int p_column, DelimiterType p_type) const;

	void _add_delimiter(const String &p_start_key, const String &p_end_key, bool p_line_only, DelimiterType p_type);
	void _remove_delimiter(const String &p_start_key, DelimiterType p_type);
	bool _has_delimiter(const String &p_start_key, DelimiterType p_type) const;

	void _set_delimiters(const TypedArray<String> &p_delimiters, DelimiterType p_type);
	void _clear_delimiters(DelimiterType p_type);
	TypedArray<String> _get_delimiters(DelimiterType p_type) const;

	/* Code Hint */
	String code_hint = "";

	bool code_hint_draw_below = true;
	int code_hint_xpos = -0xFFFF;

	/* Code Completion */
	bool code_completion_enabled = false;
	bool code_completion_forced = false;

	int code_completion_max_width = 0;
	int code_completion_max_lines = 7;
	int code_completion_scroll_width = 0;
	Color code_completion_scroll_color = Color(0, 0, 0, 0);
	Color code_completion_background_color = Color(0, 0, 0, 0);
	Color code_completion_selected_color = Color(0, 0, 0, 0);
	Color code_completion_existing_color = Color(0, 0, 0, 0);

	bool code_completion_active = false;
	Vector<ScriptCodeCompletionOption> code_completion_options;
	int code_completion_line_ofs = 0;
	int code_completion_current_selected = 0;
	int code_completion_longest_line = 0;
	Rect2i code_completion_rect;

	Set<String> code_completion_prefixes;
	List<ScriptCodeCompletionOption> code_completion_option_submitted;
	List<ScriptCodeCompletionOption> code_completion_option_sources;
	String code_completion_base;

	void _filter_code_completion_candidates();

	void _lines_edited_from(int p_from_line, int p_to_line);

protected:
	void _gui_input(const Ref<InputEvent> &p_gui_input) override;
	void _notification(int p_what);

	static void _bind_methods();

public:
	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;

	/* Indent management */
	void set_indent_size(const int p_size);
	int get_indent_size() const;

	void set_indent_using_spaces(const bool p_use_spaces);
	bool is_indent_using_spaces() const;

	void set_auto_indent_enabled(bool p_enabled);
	bool is_auto_indent_enabled() const;

	void set_auto_indent_prefixes(const TypedArray<String> &p_prefixes);
	TypedArray<String> get_auto_indent_prefixes() const;

	void do_indent();
	void do_unindent();

	void indent_lines();
	void unindent_lines();

	virtual void backspace() override;

	/* Main Gutter */
	void set_draw_breakpoints_gutter(bool p_draw);
	bool is_drawing_breakpoints_gutter() const;

	void set_draw_bookmarks_gutter(bool p_draw);
	bool is_drawing_bookmarks_gutter() const;

	void set_draw_executing_lines_gutter(bool p_draw);
	bool is_drawing_executing_lines_gutter() const;

	// breakpoints
	void set_line_as_breakpoint(int p_line, bool p_breakpointed);
	bool is_line_breakpointed(int p_line) const;
	void clear_breakpointed_lines();
	Array get_breakpointed_lines() const;

	// bookmarks
	void set_line_as_bookmarked(int p_line, bool p_bookmarked);
	bool is_line_bookmarked(int p_line) const;
	void clear_bookmarked_lines();
	Array get_bookmarked_lines() const;

	// executing lines
	void set_line_as_executing(int p_line, bool p_executing);
	bool is_line_executing(int p_line) const;
	void clear_executing_lines();
	Array get_executing_lines() const;

	/* Line numbers */
	void set_draw_line_numbers(bool p_draw);
	bool is_draw_line_numbers_enabled() const;
	void set_line_numbers_zero_padded(bool p_zero_padded);
	bool is_line_numbers_zero_padded() const;

	/* Fold gutter */
	void set_draw_fold_gutter(bool p_draw);
	bool is_drawing_fold_gutter() const;

	/* Line Folding */
	void set_line_folding_enabled(bool p_enabled);
	bool is_line_folding_enabled() const;

	bool can_fold_line(int p_line) const;

	void fold_line(int p_line);
	void unfold_line(int p_line);
	void fold_all_lines();
	void unfold_all_lines();
	void toggle_foldable_line(int p_line);

	bool is_line_folded(int p_line) const;
	TypedArray<int> get_folded_lines() const;

	/* Delimiters */
	void add_string_delimiter(const String &p_start_key, const String &p_end_key, bool p_line_only = false);
	void remove_string_delimiter(const String &p_start_key);
	bool has_string_delimiter(const String &p_start_key) const;

	void set_string_delimiters(const TypedArray<String> &p_string_delimiters);
	void clear_string_delimiters();
	TypedArray<String> get_string_delimiters() const;

	int is_in_string(int p_line, int p_column = -1) const;

	void add_comment_delimiter(const String &p_start_key, const String &p_end_key, bool p_line_only = false);
	void remove_comment_delimiter(const String &p_start_key);
	bool has_comment_delimiter(const String &p_start_key) const;

	void set_comment_delimiters(const TypedArray<String> &p_comment_delimiters);
	void clear_comment_delimiters();
	TypedArray<String> get_comment_delimiters() const;

	int is_in_comment(int p_line, int p_column = -1) const;

	String get_delimiter_start_key(int p_delimiter_idx) const;
	String get_delimiter_end_key(int p_delimiter_idx) const;

	Point2 get_delimiter_start_position(int p_line, int p_column) const;
	Point2 get_delimiter_end_position(int p_line, int p_column) const;

	/* Code hint */
	void set_code_hint(const String &p_hint);
	void set_code_hint_draw_below(bool p_below);

	/* Code Completion */
	void set_code_completion_enabled(bool p_enable);
	bool is_code_completion_enabled() const;

	void set_code_completion_prefixes(const TypedArray<String> &p_prefixes);
	TypedArray<String> get_code_completion_prefixes() const;

	String get_text_for_code_completion() const;

	void request_code_completion(bool p_force = false);

	void add_code_completion_option(CodeCompletionKind p_type, const String &p_display_text, const String &p_insert_text, const Color &p_text_color = Color(1, 1, 1), const RES &p_icon = RES(), const Variant &p_value = Variant::NIL);
	void update_code_completion_options(bool p_forced = false);

	TypedArray<Dictionary> get_code_completion_options() const;
	Dictionary get_code_completion_option(int p_index) const;

	int get_code_completion_selected_index() const;
	void set_code_completion_selected_index(int p_index);

	void confirm_code_completion(bool p_replace = false);
	void cancel_code_completion();

	CodeEdit();
	~CodeEdit();
};

VARIANT_ENUM_CAST(CodeEdit::CodeCompletionKind);

#endif // CODEEDIT_H
