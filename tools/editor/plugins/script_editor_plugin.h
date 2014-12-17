/*************************************************************************/
/*  script_editor_plugin.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef SCRIPT_EDITOR_PLUGIN_H
#define SCRIPT_EDITOR_PLUGIN_H

#include "tools/editor/editor_plugin.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/tree.h"
#include "scene/main/timer.h"
#include "script_language.h"
#include "tools/editor/code_editor.h"
#include "scene/gui/split_container.h"


class ScriptEditorQuickOpen : public ConfirmationDialog {

	OBJ_TYPE(ScriptEditorQuickOpen,ConfirmationDialog )

	LineEdit *search_box;
	Tree *search_options;
	String function;

	void _update_search();

	void _sbox_input(const InputEvent& p_ie);
	Vector<String> functions;


	void _confirmed();
	void _text_changed(const String& p_newtext);

protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	void popup(const Vector<String>& p_base,bool p_dontclear=false);
	ScriptEditorQuickOpen();
};


class ScriptEditorDebugger;

class ScriptTextEditor : public CodeTextEditor {

	OBJ_TYPE( ScriptTextEditor, CodeTextEditor );

	Ref<Script> script;


	Vector<String> functions;


protected:



	virtual void _validate_script();
	virtual void _code_complete_script(const String& p_code, List<String>* r_options);
	virtual void _load_theme_settings();
	void _notification(int p_what);


public:

	virtual void apply_code();
	Ref<Script> get_edited_script() const;
	Vector<String> get_functions() ;
	void set_edited_script(const Ref<Script>& p_script);
	void reload_text();
	void _update_name();

	ScriptTextEditor();

};

class ScriptEditor : public VBoxContainer {

	OBJ_TYPE(ScriptEditor, VBoxContainer );


	EditorNode *editor;
	enum {

		FILE_OPEN,
		FILE_SAVE,
		FILE_SAVE_AS,
		FILE_SAVE_ALL,
		EDIT_UNDO,
		EDIT_REDO,
		EDIT_CUT,
		EDIT_COPY,
		EDIT_PASTE,
		EDIT_SELECT_ALL,
		EDIT_COMPLETE,
		EDIT_AUTO_INDENT,
        EDIT_TOGGLE_COMMENT,
        EDIT_MOVE_LINE_UP,
        EDIT_MOVE_LINE_DOWN,
        EDIT_INDENT_RIGHT,
        EDIT_INDENT_LEFT,
        EDIT_CLONE_DOWN,
		SEARCH_FIND,
		SEARCH_FIND_NEXT,
		SEARCH_REPLACE,
		SEARCH_LOCATE_FUNCTION,
		SEARCH_GOTO_LINE,
		DEBUG_TOGGLE_BREAKPOINT,
		DEBUG_NEXT,
		DEBUG_STEP,
		DEBUG_BREAK,
		DEBUG_CONTINUE,
		DEBUG_SHOW,
		HELP_CONTEXTUAL,
		WINDOW_CLOSE,
		WINDOW_MOVE_LEFT,
		WINDOW_MOVE_RIGHT,
		WINDOW_SELECT_BASE=100
	};

	HBoxContainer *menu_hb;
	MenuButton *file_menu;
	MenuButton *edit_menu;
	MenuButton *search_menu;
	MenuButton *window_menu;
	MenuButton *debug_menu;
	MenuButton *help_menu;
	uint64_t idle;

	TabContainer *tab_container;
	FindReplaceDialog *find_replace_dialog;
	GotoLineDialog *goto_line_dialog;
	ConfirmationDialog *erase_tab_confirm;
	ScriptEditorDebugger* debugger;

	void _tab_changed(int p_which);
	void _menu_option(int p_optin);

	Tree *disk_changed_list;
	ConfirmationDialog *disk_changed;

	VSplitContainer *v_split;

	String _get_debug_tooltip(const String&p_text,Node *_ste);

	void _resave_scripts(const String& p_str);
	void _reload_scripts();

	bool _test_script_times_on_disk();

	void _close_current_tab();

	ScriptEditorQuickOpen *quick_open;


	void _editor_play();
	void _editor_pause();
	void _editor_stop();

	void _add_callback(Object *p_obj, const String& p_function, const StringArray& p_args);
	void _res_saved_callback(const Ref<Resource>& p_res);

	void _goto_script_line2(int p_line);
	void _goto_script_line(REF p_script,int p_line);
	void _breaked(bool p_breaked,bool p_can_debug);
	void _show_debugger(bool p_show);
	void _update_window_menu();

	static ScriptEditor *script_editor;
protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	static ScriptEditor *get_singleton() { return script_editor; }
	void _save_files_state();
	void _load_files_state();


	void ensure_focus_current();
	void apply_scripts() const;

	void ensure_select_current();
	void edit(const Ref<Script>& p_script);

	Dictionary get_state() const;
	void set_state(const Dictionary& p_state);
	void clear();

	void get_breakpoints(List<String> *p_breakpoints);

    void swap_lines(TextEdit *tx, int line1, int line2);

	void save_external_data();

	ScriptEditor(EditorNode *p_editor);
};

class ScriptEditorPlugin : public EditorPlugin {

	OBJ_TYPE( ScriptEditorPlugin, EditorPlugin );

	ScriptEditor *script_editor;
	EditorNode *editor;
public:

	virtual String get_name() const { return "Script"; }
	bool has_main_screen() const { return true; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);
	virtual void selected_notify();

	Dictionary get_state() const;
	virtual void set_state(const Dictionary& p_state);
	virtual void clear();

	virtual void save_external_data();
	virtual void apply_changes();

	virtual void restore_global_state();
	virtual void save_global_state();

	virtual void get_breakpoints(List<String> *p_breakpoints);


	ScriptEditorPlugin(EditorNode *p_node);
	~ScriptEditorPlugin();

};

#endif // SCRIPT_EDITOR_PLUGIN_H
