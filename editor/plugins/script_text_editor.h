/*************************************************************************/
/*  script_text_editor.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef SCRIPT_TEXT_EDITOR_H
#define SCRIPT_TEXT_EDITOR_H

#include "scene/gui/color_picker.h"
#include "script_editor_plugin.h"

class ScriptTextEditor : public ScriptEditorBase {

	GDCLASS(ScriptTextEditor, ScriptEditorBase);

	CodeTextEditor *code_editor;

	Ref<Script> script;

	Vector<String> functions;

	HBoxContainer *edit_hb;

	MenuButton *edit_menu;
	MenuButton *search_menu;
	PopupMenu *context_menu;

	GotoLineDialog *goto_line_dialog;
	ScriptEditorQuickOpen *quick_open;

	PopupPanel *color_panel;
	ColorPicker *color_picker;
	int color_line;
	String color_args;

	struct ColorsCache {
		Color symbol_color;
		Color keyword_color;
		Color basetype_color;
		Color type_color;
		Color comment_color;
		Color string_color;
	} colors_cache;

	bool theme_loaded;

	enum {
		EDIT_UNDO,
		EDIT_REDO,
		EDIT_CUT,
		EDIT_COPY,
		EDIT_PASTE,
		EDIT_SELECT_ALL,
		EDIT_COMPLETE,
		EDIT_AUTO_INDENT,
		EDIT_TRIM_TRAILING_WHITESAPCE,
		EDIT_CONVERT_INDENT_TO_SPACES,
		EDIT_CONVERT_INDENT_TO_TABS,
		EDIT_TOGGLE_COMMENT,
		EDIT_MOVE_LINE_UP,
		EDIT_MOVE_LINE_DOWN,
		EDIT_INDENT_RIGHT,
		EDIT_INDENT_LEFT,
		EDIT_DELETE_LINE,
		EDIT_CLONE_DOWN,
		EDIT_PICK_COLOR,
		EDIT_TO_UPPERCASE,
		EDIT_TO_LOWERCASE,
		EDIT_CAPITALIZE,
		EDIT_TOGGLE_FOLD_LINE,
		EDIT_FOLD_ALL_LINES,
		EDIT_UNFOLD_ALL_LINES,
		SEARCH_FIND,
		SEARCH_FIND_NEXT,
		SEARCH_FIND_PREV,
		SEARCH_REPLACE,
		SEARCH_LOCATE_FUNCTION,
		SEARCH_GOTO_LINE,
		DEBUG_TOGGLE_BREAKPOINT,
		DEBUG_REMOVE_ALL_BREAKPOINTS,
		DEBUG_GOTO_NEXT_BREAKPOINT,
		DEBUG_GOTO_PREV_BREAKPOINT,
		HELP_CONTEXTUAL,
	};

protected:
	static void _code_complete_scripts(void *p_ud, const String &p_code, List<String> *r_options, bool &r_force);
	void _breakpoint_toggled(int p_row);

	//no longer virtual
	void _validate_script();
	void _code_complete_script(const String &p_code, List<String> *r_options, bool &r_force);
	void _load_theme_settings();
	void _set_theme_for_script();

	void _notification(int p_what);
	static void _bind_methods();

	void _edit_option(int p_op);
	void _make_context_menu(bool p_selection, bool p_color, bool p_can_fold, bool p_is_folded);
	void _text_edit_gui_input(const Ref<InputEvent> &ev);
	void _color_changed(const Color &p_color);

	void _goto_line(int p_line) { goto_line(p_line); }
	void _lookup_symbol(const String &p_symbol, int p_row, int p_column);

	enum CaseStyle {
		UPPER,
		LOWER,
		CAPITALIZE,
	};
	void _convert_case(CaseStyle p_case);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

public:
	virtual void apply_code();
	virtual Ref<Script> get_edited_script() const;
	virtual Vector<String> get_functions();
	virtual void set_edited_script(const Ref<Script> &p_script);
	virtual void reload_text();
	virtual String get_name();
	virtual Ref<Texture> get_icon();
	virtual bool is_unsaved();

	virtual Variant get_edit_state();
	virtual void set_edit_state(const Variant &p_state);
	virtual void ensure_focus();
	virtual void trim_trailing_whitespace();
	virtual void convert_indent_to_spaces();
	virtual void convert_indent_to_tabs();
	virtual void tag_saved_version();

	virtual void goto_line(int p_line, bool p_with_error = false);

	virtual void reload(bool p_soft);
	virtual void get_breakpoints(List<int> *p_breakpoints);

	virtual void add_callback(const String &p_function, PoolStringArray p_args);
	virtual void update_settings();

	virtual bool show_members_overview();

	virtual void set_tooltip_request_func(String p_method, Object *p_obj);

	virtual void set_debugger_active(bool p_active);

	Control *get_edit_menu();
	virtual void clear_edit_menu();
	static void register_editor();

	ScriptTextEditor();
};

#endif // SCRIPT_TEXT_EDITOR_H
