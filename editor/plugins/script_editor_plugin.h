/*************************************************************************/
/*  script_editor_plugin.h                                               */
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
#ifndef SCRIPT_EDITOR_PLUGIN_H
#define SCRIPT_EDITOR_PLUGIN_H

#include "editor/code_editor.h"
#include "editor/editor_help.h"
#include "editor/editor_plugin.h"
#include "editor/script_create_dialog.h"
#include "scene/gui/item_list.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/tree.h"
#include "scene/main/timer.h"
#include "script_language.h"

class ScriptEditorQuickOpen : public ConfirmationDialog {

	GDCLASS(ScriptEditorQuickOpen, ConfirmationDialog)

	LineEdit *search_box;
	Tree *search_options;
	String function;

	void _update_search();

	void _sbox_input(const InputEvent &p_ie);
	Vector<String> functions;

	void _confirmed();
	void _text_changed(const String &p_newtext);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void popup(const Vector<String> &p_base, bool p_dontclear = false);
	ScriptEditorQuickOpen();
};

class ScriptEditorDebugger;

class ScriptEditorBase : public Control {

	GDCLASS(ScriptEditorBase, Control);

protected:
	static void _bind_methods();

public:
	virtual void apply_code() = 0;
	virtual Ref<Script> get_edited_script() const = 0;
	virtual Vector<String> get_functions() = 0;
	virtual void set_edited_script(const Ref<Script> &p_script) = 0;
	virtual void reload_text() = 0;
	virtual String get_name() = 0;
	virtual Ref<Texture> get_icon() = 0;
	virtual bool is_unsaved() = 0;
	virtual Variant get_edit_state() = 0;
	virtual void set_edit_state(const Variant &p_state) = 0;
	virtual void goto_line(int p_line, bool p_with_error = false) = 0;
	virtual void trim_trailing_whitespace() = 0;
	virtual void ensure_focus() = 0;
	virtual void tag_saved_version() = 0;
	virtual void reload(bool p_soft) = 0;
	virtual void get_breakpoints(List<int> *p_breakpoints) = 0;
	virtual bool goto_method(const String &p_method) = 0;
	virtual void add_callback(const String &p_function, PoolStringArray p_args) = 0;
	virtual void update_settings() = 0;
	virtual void set_debugger_active(bool p_active) = 0;
	virtual bool can_lose_focus_on_node_selection() { return true; }

	virtual void set_tooltip_request_func(String p_method, Object *p_obj) = 0;
	virtual Control *get_edit_menu() = 0;

	ScriptEditorBase() {}
};

typedef ScriptEditorBase *(*CreateScriptEditorFunc)(const Ref<Script> &p_script);

class EditorScriptCodeCompletionCache;

class ScriptEditor : public VBoxContainer {

	GDCLASS(ScriptEditor, VBoxContainer);

	EditorNode *editor;
	enum {
		FILE_NEW,
		FILE_OPEN,
		FILE_SAVE,
		FILE_SAVE_AS,
		FILE_SAVE_ALL,
		FILE_IMPORT_THEME,
		FILE_RELOAD_THEME,
		FILE_SAVE_THEME,
		FILE_SAVE_THEME_AS,
		FILE_CLOSE,
		CLOSE_DOCS,
		CLOSE_ALL,
		FILE_TOOL_RELOAD,
		FILE_TOOL_RELOAD_SOFT,
		DEBUG_NEXT,
		DEBUG_STEP,
		DEBUG_BREAK,
		DEBUG_CONTINUE,
		DEBUG_SHOW,
		DEBUG_SHOW_KEEP_OPEN,
		SEARCH_HELP,
		SEARCH_CLASSES,
		SEARCH_WEBSITE,
		HELP_SEARCH_FIND,
		HELP_SEARCH_FIND_NEXT,
		WINDOW_MOVE_LEFT,
		WINDOW_MOVE_RIGHT,
		WINDOW_NEXT,
		WINDOW_PREV,
		WINDOW_SELECT_BASE = 100
	};

	enum ScriptSortBy {
		SORT_BY_NAME,
		SORT_BY_PATH,
	};

	enum ScriptListName {
		DISPLAY_NAME,
		DISPLAY_DIR_AND_NAME,
		DISPLAY_FULL_PATH,
	};

	HBoxContainer *menu_hb;
	MenuButton *file_menu;
	MenuButton *edit_menu;
	MenuButton *script_search_menu;
	MenuButton *debug_menu;
	Timer *autosave_timer;
	uint64_t idle;

	Button *help_search;
	Button *site_search;
	Button *class_search;
	EditorHelpSearch *help_search_dialog;

	ItemList *script_list;
	HSplitContainer *script_split;
	TabContainer *tab_container;
	EditorFileDialog *file_dialog;
	ConfirmationDialog *erase_tab_confirm;
	ScriptCreateDialog *script_create_dialog;
	ScriptEditorDebugger *debugger;
	ToolButton *scripts_visible;

	String current_theme;

	TextureRect *script_icon;
	Label *script_name_label;

	ToolButton *script_back;
	ToolButton *script_forward;

	enum {
		SCRIPT_EDITOR_FUNC_MAX = 32
	};

	static int script_editor_func_count;
	static CreateScriptEditorFunc script_editor_funcs[SCRIPT_EDITOR_FUNC_MAX];

	struct ScriptHistory {

		Control *control;
		Variant state;
	};

	Vector<ScriptHistory> history;
	int history_pos;

	EditorHelpIndex *help_index;

	void _tab_changed(int p_which);
	void _menu_option(int p_optin);

	Tree *disk_changed_list;
	ConfirmationDialog *disk_changed;

	bool restoring_layout;

	String _get_debug_tooltip(const String &p_text, Node *_ste);

	void _resave_scripts(const String &p_str);
	void _reload_scripts();

	bool _test_script_times_on_disk(Ref<Script> p_for_script = Ref<Script>());

	void _close_tab(int p_idx, bool p_save = true);

	void _close_current_tab();
	void _close_discard_current_tab(const String &p_str);
	void _close_docs_tab();
	void _close_all_tabs();

	void _ask_close_current_unsaved_tab(ScriptEditorBase *current);

	bool grab_focus_block;

	bool pending_auto_reload;
	bool auto_reload_running_scripts;
	void _live_auto_reload_running_scripts();

	void _update_selected_editor_menu();

	EditorScriptCodeCompletionCache *completion_cache;

	void _editor_play();
	void _editor_pause();
	void _editor_stop();

	int edit_pass;

	void _add_callback(Object *p_obj, const String &p_function, const PoolStringArray &p_args);
	void _res_saved_callback(const Ref<Resource> &p_res);

	bool trim_trailing_whitespace_on_save;

	void _trim_trailing_whitespace(TextEdit *tx);

	void _goto_script_line2(int p_line);
	void _goto_script_line(REF p_script, int p_line);
	void _breaked(bool p_breaked, bool p_can_debug);
	void _show_debugger(bool p_show);
	void _update_window_menu();
	void _script_created(Ref<Script> p_script);

	void _save_layout();
	void _editor_settings_changed();
	void _autosave_scripts();

	void _update_script_names();

	void _script_selected(int p_idx);

	void _find_scripts(Node *p_base, Node *p_current, Set<Ref<Script> > &used);

	void _tree_changed();

	void _script_split_dragged(float);

	void _unhandled_input(const InputEvent &p_event);

	void _help_search(String p_text);

	void _history_forward();
	void _history_back();

	bool waiting_update_names;

	void _help_class_open(const String &p_class);
	void _help_class_goto(const String &p_desc);
	void _update_history_arrows();
	void _save_history();
	void _go_to_tab(int p_idx);
	void _update_history_pos(int p_new_pos);
	void _update_script_colors();
	void _update_modified_scripts_for_external_editor(Ref<Script> p_for_script = Ref<Script>());

	int file_dialog_option;
	void _file_dialog_action(String p_file);

	static void _open_script_request(const String &p_path);

	static ScriptEditor *script_editor;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static ScriptEditor *get_singleton() { return script_editor; }

	void ensure_focus_current();
	void apply_scripts() const;

	void ensure_select_current();
	void edit(const Ref<Script> &p_script, bool p_grab_focus = true);

	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);
	void clear();

	void get_breakpoints(List<String> *p_breakpoints);

	//void swap_lines(TextEdit *tx, int line1, int line2);

	void save_all_scripts();

	void set_window_layout(Ref<ConfigFile> p_layout);
	void get_window_layout(Ref<ConfigFile> p_layout);

	void set_scene_root_script(Ref<Script> p_script);

	bool script_go_to_method(Ref<Script> p_script, const String &p_method);

	virtual void edited_scene_changed();

	void close_builtin_scripts_from_scene(const String &p_scene);

	void goto_help(const String &p_desc) { _help_class_goto(p_desc); }

	bool can_take_away_focus() const;

	ScriptEditorDebugger *get_debugger() { return debugger; }
	void set_live_auto_reload_running_scripts(bool p_enabled);

	static void register_create_script_editor_function(CreateScriptEditorFunc p_func);
	ScriptEditor(EditorNode *p_editor);
	~ScriptEditor();
};

class ScriptEditorPlugin : public EditorPlugin {

	GDCLASS(ScriptEditorPlugin, EditorPlugin);

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
	virtual void set_state(const Dictionary &p_state);
	virtual void clear();

	virtual void save_external_data();
	virtual void apply_changes();

	virtual void restore_global_state();
	virtual void save_global_state();

	virtual void set_window_layout(Ref<ConfigFile> p_layout);
	virtual void get_window_layout(Ref<ConfigFile> p_layout);

	virtual void get_breakpoints(List<String> *p_breakpoints);

	virtual void edited_scene_changed();

	ScriptEditorPlugin(EditorNode *p_node);
	~ScriptEditorPlugin();
};

#endif // SCRIPT_EDITOR_PLUGIN_H
