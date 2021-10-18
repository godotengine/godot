/*************************************************************************/
/*  code_editor.h                                                        */
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

#ifndef CODE_EDITOR_H
#define CODE_EDITOR_H

#include "editor/editor_plugin.h"
#include "scene/gui/check_box.h"
#include "scene/gui/check_button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/tool_button.h"
#include "scene/main/timer.h"

class GotoLineDialog : public ConfirmationDialog {
	GDCLASS(GotoLineDialog, ConfirmationDialog);

	Label *line_label;
	LineEdit *line;

	TextEdit *text_editor;

	virtual void ok_pressed();

public:
	void popup_find_line(TextEdit *p_edit);
	int get_line() const;

	GotoLineDialog();
};

class FindReplaceBar : public HBoxContainer {
	GDCLASS(FindReplaceBar, HBoxContainer);

	LineEdit *search_text;
	Label *matches_label;
	ToolButton *find_prev;
	ToolButton *find_next;
	CheckBox *case_sensitive;
	CheckBox *whole_words;
	TextureButton *hide_button;

	LineEdit *replace_text;
	Button *replace;
	Button *replace_all;
	CheckBox *selection_only;

	VBoxContainer *vbc_lineedit;
	HBoxContainer *hbc_button_replace;
	HBoxContainer *hbc_option_replace;

	TextEdit *text_edit;

	int result_line;
	int result_col;
	int results_count;

	bool replace_all_mode;
	bool preserve_cursor;

	void _get_search_from(int &r_line, int &r_col);
	void _update_results_count();
	void _update_matches_label();

	void _show_search(bool p_focus_replace = false, bool p_show_only = false);
	void _hide_bar();

	void _editor_text_changed();
	void _search_options_changed(bool p_pressed);
	void _search_text_changed(const String &p_text);
	void _search_text_entered(const String &p_text);
	void _replace_text_entered(const String &p_text);

protected:
	void _notification(int p_what);
	void _unhandled_input(const Ref<InputEvent> &p_event);

	bool _search(uint32_t p_flags, int p_from_line, int p_from_col);

	void _replace();
	void _replace_all();

	static void _bind_methods();

public:
	String get_search_text() const;
	String get_replace_text() const;

	bool is_case_sensitive() const;
	bool is_whole_words() const;
	bool is_selection_only() const;
	void set_error(const String &p_label);

	void set_text_edit(TextEdit *p_text_edit);

	void popup_search(bool p_show_only = false);
	void popup_replace();

	bool search_current();
	bool search_prev();
	bool search_next();

	FindReplaceBar();
};

typedef void (*CodeTextEditorCodeCompleteFunc)(void *p_ud, const String &p_code, List<ScriptCodeCompletionOption> *r_options, bool &r_forced);

class CodeTextEditor : public VBoxContainer {
	GDCLASS(CodeTextEditor, VBoxContainer);

	TextEdit *text_editor;
	FindReplaceBar *find_replace_bar;
	HBoxContainer *status_bar;

	ToolButton *toggle_scripts_button;
	ToolButton *warning_button;
	Label *warning_count_label;

	Label *line_and_col_txt;

	Label *info;
	Timer *idle;
	Timer *code_complete_timer;

	Timer *font_resize_timer;
	int font_resize_val;
	real_t font_size;

	Label *error;
	int error_line;
	int error_column;

	void _on_settings_change();

	void _update_font();
	void _complete_request();
	Ref<Texture> _get_completion_icon(const ScriptCodeCompletionOption &p_option);
	void _font_resize_timeout();
	bool _add_font_size(int p_delta);

	void _input(const Ref<InputEvent> &event);
	void _text_editor_gui_input(const Ref<InputEvent> &p_event);
	void _zoom_in();
	void _zoom_out();
	void _zoom_changed();
	void _reset_zoom();

	CodeTextEditorCodeCompleteFunc code_complete_func;
	void *code_complete_ud;

	void _warning_label_gui_input(const Ref<InputEvent> &p_event);
	void _warning_button_pressed();
	void _set_show_warnings_panel(bool p_show);
	void _error_pressed(const Ref<InputEvent> &p_event);

	void _delete_line(int p_line);
	void _toggle_scripts_pressed();

protected:
	virtual void _load_theme_settings() {}
	virtual void _validate_script() {}
	virtual void _code_complete_script(const String &p_code, List<ScriptCodeCompletionOption> *r_options) {}

	void _text_changed_idle_timeout();
	void _code_complete_timer_timeout();
	void _text_changed();
	void _line_col_changed();
	void _notification(int);
	static void _bind_methods();

	bool is_warnings_panel_opened;

public:
	void trim_trailing_whitespace();
	void insert_final_newline();

	void convert_indent_to_spaces();
	void convert_indent_to_tabs();

	enum CaseStyle {
		UPPER,
		LOWER,
		CAPITALIZE,
	};
	void convert_case(CaseStyle p_case);

	void move_lines_up();
	void move_lines_down();
	void delete_lines();
	void duplicate_selection();

	/// Toggle inline comment on currently selected lines, or on current line if nothing is selected,
	/// by adding or removing comment delimiter
	void toggle_inline_comment(const String &delimiter);

	void goto_line(int p_line);
	void goto_line_selection(int p_line, int p_begin, int p_end);
	void goto_line_centered(int p_line);
	void set_executing_line(int p_line);
	void clear_executing_line();

	Variant get_edit_state();
	void set_edit_state(const Variant &p_state);

	void set_warning_nb(int p_warning_nb);

	void update_editor_settings();
	void set_error(const String &p_error);
	void set_error_pos(int p_line, int p_column);
	void update_line_and_column() { _line_col_changed(); }
	TextEdit *get_text_edit() { return text_editor; }
	FindReplaceBar *get_find_replace_bar() { return find_replace_bar; }
	virtual void apply_code() {}
	void goto_error();

	void toggle_bookmark();
	void goto_next_bookmark();
	void goto_prev_bookmark();
	void remove_all_bookmarks();

	void set_code_complete_func(CodeTextEditorCodeCompleteFunc p_code_complete_func, void *p_ud);

	void validate_script();

	void show_toggle_scripts_button();
	void update_toggle_scripts_button();

	CodeTextEditor();
};

#endif // CODE_EDITOR_H
