/*************************************************************************/
/*  code_editor.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

	void set_text_editor(TextEdit *p_text_editor);
	GotoLineDialog();
};

class FindReplaceBar : public HBoxContainer {

	GDCLASS(FindReplaceBar, HBoxContainer);

	LineEdit *search_text;
	ToolButton *find_prev;
	ToolButton *find_next;
	CheckBox *case_sensitive;
	CheckBox *whole_words;
	Label *error_label;
	TextureButton *hide_button;

	LineEdit *replace_text;
	Button *replace;
	Button *replace_all;
	CheckBox *selection_only;

	VBoxContainer *text_vbc;
	HBoxContainer *replace_hbc;
	HBoxContainer *replace_options_hbc;

	TextEdit *text_edit;

	int result_line;
	int result_col;

	bool replace_all_mode;
	bool preserve_cursor;

	void _get_search_from(int &r_line, int &r_col);

	void _show_search();
	void _hide_bar();

	void _editor_text_changed();
	void _search_options_changed(bool p_pressed);
	void _search_text_changed(const String &p_text);
	void _search_text_entered(const String &p_text);
	void _replace_text_entered(const String &p_text);

protected:
	void _notification(int p_what);
	void _unhandled_input(const InputEvent &p_event);

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

	void popup_search();
	void popup_replace();

	bool search_current();
	bool search_prev();
	bool search_next();

	FindReplaceBar();
};

class FindReplaceDialog : public ConfirmationDialog {

	GDCLASS(FindReplaceDialog, ConfirmationDialog);

	LineEdit *search_text;
	LineEdit *replace_text;
	CheckButton *whole_words;
	CheckButton *case_sensitive;
	CheckButton *backwards;
	CheckButton *prompt;
	CheckButton *selection_only;
	Button *skip;
	Label *error_label;
	MarginContainer *replace_mc;
	Label *replace_label;
	VBoxContainer *replace_vb;

	void _search_text_entered(const String &p_text);
	void _replace_text_entered(const String &p_text);
	void _prompt_changed();
	void _skip_pressed();

	TextEdit *text_edit;

protected:
	void _search_callback();
	void _replace_skip_callback();

	bool _search();
	void _replace();

	virtual void ok_pressed();
	static void _bind_methods();

public:
	String get_search_text() const;
	String get_replace_text() const;
	bool is_whole_words() const;
	bool is_case_sensitive() const;
	bool is_backwards() const;
	bool is_replace_mode() const;
	bool is_replace_all_mode() const;
	bool is_replace_selection_only() const;
	void set_replace_selection_only(bool p_enable);

	void set_error(const String &p_error);

	void popup_search();
	void popup_replace();

	void set_text_edit(TextEdit *p_text_edit);

	void search_next();
	FindReplaceDialog();
};

typedef void (*CodeTextEditorCodeCompleteFunc)(void *p_ud, const String &p_code, List<String> *r_options);

class CodeTextEditor : public VBoxContainer {

	GDCLASS(CodeTextEditor, VBoxContainer);

	TextEdit *text_editor;
	FindReplaceBar *find_replace_bar;

	Label *line_nb;
	Label *col_nb;
	Label *info;
	Timer *idle;
	Timer *code_complete_timer;
	bool enable_complete_timer;

	Timer *font_resize_timer;
	int font_resize_val;

	Label *error;

	void _on_settings_change();

	void _update_font();
	void _complete_request();
	void _font_resize_timeout();

	void _text_editor_gui_input(const InputEvent &p_event);
	void _zoom_in();
	void _zoom_out();
	void _reset_zoom();

	CodeTextEditorCodeCompleteFunc code_complete_func;
	void *code_complete_ud;

protected:
	virtual void _load_theme_settings() {}
	virtual void _validate_script() {}
	virtual void _code_complete_script(const String &p_code, List<String> *r_options) {}

	void _text_changed_idle_timeout();
	void _code_complete_timer_timeout();
	void _text_changed();
	void _line_col_changed();
	void _notification(int);
	static void _bind_methods();

public:
	void update_editor_settings();
	void set_error(const String &p_error);
	void update_line_and_column() { _line_col_changed(); }
	TextEdit *get_text_edit() { return text_editor; }
	FindReplaceBar *get_find_replace_bar() { return find_replace_bar; }
	virtual void apply_code() {}

	void set_code_complete_func(CodeTextEditorCodeCompleteFunc p_code_complete_func, void *p_ud);

	CodeTextEditor();
};

#endif // CODE_EDITOR_H
