/*************************************************************************/
/*  script_editor_plugin.h                                               */
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

#ifndef SCRIPT_EDITOR_PLUGIN_H
#define SCRIPT_EDITOR_PLUGIN_H

#include "core/object/script_language.h"
#include "editor/code_editor.h"
#include "editor/editor_help.h"
#include "editor/editor_help_search.h"
#include "editor/editor_plugin.h"
#include "editor/script_create_dialog.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/tree.h"
#include "scene/main/timer.h"
#include "scene/resources/text_file.h"

class EditorSyntaxHighlighter : public SyntaxHighlighter {
	GDCLASS(EditorSyntaxHighlighter, SyntaxHighlighter)

private:
	REF edited_resourse;

protected:
	static void _bind_methods();

public:
	virtual String _get_name() const;
	virtual Array _get_supported_languages() const;

	void _set_edited_resource(const RES &p_res) { edited_resourse = p_res; }
	REF _get_edited_resource() { return edited_resourse; }

	virtual Ref<EditorSyntaxHighlighter> _create() const;
};

class EditorStandardSyntaxHighlighter : public EditorSyntaxHighlighter {
	GDCLASS(EditorStandardSyntaxHighlighter, EditorSyntaxHighlighter)

private:
	Ref<CodeHighlighter> highlighter;

public:
	virtual void _update_cache() override;
	virtual Dictionary _get_line_syntax_highlighting(int p_line) override { return highlighter->get_line_syntax_highlighting(p_line); }

	virtual String _get_name() const override { return TTR("Standard"); }

	virtual Ref<EditorSyntaxHighlighter> _create() const override;

	EditorStandardSyntaxHighlighter() { highlighter.instance(); }
};

class EditorPlainTextSyntaxHighlighter : public EditorSyntaxHighlighter {
	GDCLASS(EditorPlainTextSyntaxHighlighter, EditorSyntaxHighlighter)

public:
	virtual String _get_name() const override { return TTR("Plain Text"); }

	virtual Ref<EditorSyntaxHighlighter> _create() const override;
};

///////////////////////////////////////////////////////////////////////////////

class ScriptEditorQuickOpen : public ConfirmationDialog {
	GDCLASS(ScriptEditorQuickOpen, ConfirmationDialog);

	LineEdit *search_box;
	Tree *search_options;
	String function;

	void _update_search();

	void _sbox_input(const Ref<InputEvent> &p_ie);
	Vector<String> functions;

	void _confirmed();
	void _text_changed(const String &p_newtext);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void popup_dialog(const Vector<String> &p_functions, bool p_dontclear = false);
	ScriptEditorQuickOpen();
};

class EditorDebuggerNode;

class ScriptEditorBase : public VBoxContainer {
	GDCLASS(ScriptEditorBase, VBoxContainer);

protected:
	static void _bind_methods();

public:
	virtual void add_syntax_highlighter(Ref<EditorSyntaxHighlighter> p_highlighter) = 0;
	virtual void set_syntax_highlighter(Ref<EditorSyntaxHighlighter> p_highlighter) = 0;

	virtual void apply_code() = 0;
	virtual RES get_edited_resource() const = 0;
	virtual Vector<String> get_functions() = 0;
	virtual void set_edited_resource(const RES &p_res) = 0;
	virtual void enable_editor() = 0;
	virtual void reload_text() = 0;
	virtual String get_name() = 0;
	virtual Ref<Texture2D> get_theme_icon() = 0;
	virtual bool is_unsaved() = 0;
	virtual Variant get_edit_state() = 0;
	virtual void set_edit_state(const Variant &p_state) = 0;
	virtual void goto_line(int p_line, bool p_with_error = false) = 0;
	virtual void set_executing_line(int p_line) = 0;
	virtual void clear_executing_line() = 0;
	virtual void trim_trailing_whitespace() = 0;
	virtual void insert_final_newline() = 0;
	virtual void convert_indent_to_spaces() = 0;
	virtual void convert_indent_to_tabs() = 0;
	virtual void ensure_focus() = 0;
	virtual void tag_saved_version() = 0;
	virtual void reload(bool p_soft) {}
	virtual Array get_breakpoints() = 0;
	virtual void add_callback(const String &p_function, PackedStringArray p_args) = 0;
	virtual void update_settings() = 0;
	virtual void set_debugger_active(bool p_active) = 0;
	virtual bool can_lose_focus_on_node_selection() { return true; }

	virtual bool show_members_overview() = 0;

	virtual void set_tooltip_request_func(String p_method, Object *p_obj) = 0;
	virtual Control *get_edit_menu() = 0;
	virtual void clear_edit_menu() = 0;

	virtual Control *get_base_editor() const = 0;

	virtual void validate() = 0;

	ScriptEditorBase() {}
};

typedef ScriptEditorBase *(*CreateScriptEditorFunc)(const RES &p_resource);

class EditorScriptCodeCompletionCache;
class FindInFilesDialog;
class FindInFilesPanel;

class ScriptEditor : public PanelContainer {
	GDCLASS(ScriptEditor, PanelContainer);

	EditorNode *editor;
	enum {
		FILE_NEW,
		FILE_NEW_TEXTFILE,
		FILE_OPEN,
		FILE_REOPEN_CLOSED,
		FILE_OPEN_RECENT,
		FILE_SAVE,
		FILE_SAVE_AS,
		FILE_SAVE_ALL,
		FILE_THEME,
		FILE_RUN,
		FILE_CLOSE,
		CLOSE_DOCS,
		CLOSE_ALL,
		CLOSE_OTHER_TABS,
		TOGGLE_SCRIPTS_PANEL,
		SHOW_IN_FILE_SYSTEM,
		FILE_COPY_PATH,
		FILE_TOOL_RELOAD,
		FILE_TOOL_RELOAD_SOFT,
		SEARCH_IN_FILES,
		REPLACE_IN_FILES,
		SEARCH_HELP,
		SEARCH_WEBSITE,
		HELP_SEARCH_FIND,
		HELP_SEARCH_FIND_NEXT,
		HELP_SEARCH_FIND_PREVIOUS,
		WINDOW_MOVE_UP,
		WINDOW_MOVE_DOWN,
		WINDOW_NEXT,
		WINDOW_PREV,
		WINDOW_SORT,
		WINDOW_SELECT_BASE = 100
	};

	enum {
		THEME_IMPORT,
		THEME_RELOAD,
		THEME_SAVE,
		THEME_SAVE_AS
	};

	enum ScriptSortBy {
		SORT_BY_NAME,
		SORT_BY_PATH,
		SORT_BY_NONE
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
	PopupMenu *context_menu;
	Timer *autosave_timer;
	uint64_t idle;

	PopupMenu *recent_scripts;
	PopupMenu *theme_submenu;

	Button *help_search;
	Button *site_search;
	EditorHelpSearch *help_search_dialog;

	ItemList *script_list;
	HSplitContainer *script_split;
	ItemList *members_overview;
	LineEdit *filter_scripts;
	LineEdit *filter_methods;
	VBoxContainer *scripts_vbox;
	VBoxContainer *overview_vbox;
	HBoxContainer *buttons_hbox;
	Label *filename;
	Button *members_overview_alphabeta_sort_button;
	bool members_overview_enabled;
	ItemList *help_overview;
	bool help_overview_enabled;
	VSplitContainer *list_split;
	TabContainer *tab_container;
	EditorFileDialog *file_dialog;
	AcceptDialog *error_dialog;
	ConfirmationDialog *erase_tab_confirm;
	ScriptCreateDialog *script_create_dialog;
	Button *scripts_visible;

	String current_theme;

	TextureRect *script_icon;
	Label *script_name_label;

	Button *script_back;
	Button *script_forward;

	FindInFilesDialog *find_in_files_dialog;
	FindInFilesPanel *find_in_files;
	Button *find_in_files_button;

	enum {
		SCRIPT_EDITOR_FUNC_MAX = 32,
	};

	static int script_editor_func_count;
	static CreateScriptEditorFunc script_editor_funcs[SCRIPT_EDITOR_FUNC_MAX];

	Vector<Ref<EditorSyntaxHighlighter>> syntax_highlighters;

	struct ScriptHistory {
		Control *control = nullptr;
		Variant state;
	};

	Vector<ScriptHistory> history;
	int history_pos;

	List<String> previous_scripts;

	void _tab_changed(int p_which);
	void _menu_option(int p_option);
	void _theme_option(int p_option);
	void _show_save_theme_as_dialog();

	Tree *disk_changed_list;
	ConfirmationDialog *disk_changed;

	bool restoring_layout;

	String _get_debug_tooltip(const String &p_text, Node *_se);

	void _resave_scripts(const String &p_str);
	void _reload_scripts();

	bool _test_script_times_on_disk(RES p_for_script = Ref<Resource>());

	void _add_recent_script(String p_path);
	void _update_recent_scripts();
	void _open_recent_script(int p_idx);

	void _show_error_dialog(String p_path);

	void _close_tab(int p_idx, bool p_save = true, bool p_history_back = true);

	void _close_current_tab(bool p_save = true);
	void _close_discard_current_tab(const String &p_str);
	void _close_docs_tab();
	void _close_other_tabs();
	void _close_all_tabs();

	void _copy_script_path();

	void _ask_close_current_unsaved_tab(ScriptEditorBase *current);

	bool grab_focus_block;

	bool pending_auto_reload;
	bool auto_reload_running_scripts;
	void _live_auto_reload_running_scripts();

	void _update_selected_editor_menu();

	EditorScriptCodeCompletionCache *completion_cache;

	void _editor_stop();

	int edit_pass;

	void _add_callback(Object *p_obj, const String &p_function, const PackedStringArray &p_args);
	void _res_saved_callback(const Ref<Resource> &p_res);

	bool trim_trailing_whitespace_on_save;
	bool use_space_indentation;
	bool convert_indent_on_save;

	void _trim_trailing_whitespace(TextEdit *tx);

	void _goto_script_line2(int p_line);
	void _goto_script_line(REF p_script, int p_line);
	void _set_execution(REF p_script, int p_line);
	void _clear_execution(REF p_script);
	void _breaked(bool p_breaked, bool p_can_debug);
	void _update_window_menu();
	void _script_created(Ref<Script> p_script);

	ScriptEditorBase *_get_current_editor() const;
	Array _get_open_script_editors() const;

	void _save_layout();
	void _editor_settings_changed();
	void _filesystem_changed();
	void _file_removed(const String &p_file);
	void _autosave_scripts();
	void _update_autosave_timer();

	void _update_members_overview_visibility();
	void _update_members_overview();
	void _toggle_members_overview_alpha_sort(bool p_alphabetic_sort);
	void _filter_scripts_text_changed(const String &p_newtext);
	void _filter_methods_text_changed(const String &p_newtext);
	void _update_script_names();
	void _update_script_connections();
	bool _sort_list_on_update;

	void _members_overview_selected(int p_idx);
	void _script_selected(int p_idx);

	void _update_help_overview_visibility();
	void _update_help_overview();
	void _help_overview_selected(int p_idx);

	void _find_scripts(Node *p_base, Node *p_current, Set<Ref<Script>> &used);

	void _tree_changed();

	void _script_split_dragged(float);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _unhandled_key_input(const Ref<InputEvent> &p_event);

	void _script_list_gui_input(const Ref<InputEvent> &ev);
	void _make_script_list_context_menu();

	void _help_search(String p_text);
	void _help_index(String p_text);

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

	void _script_changed();
	int file_dialog_option;
	void _file_dialog_action(String p_file);

	Ref<Script> _get_current_script();
	Array _get_open_scripts() const;

	Ref<TextFile> _load_text_file(const String &p_path, Error *r_error);
	Error _save_text_file(Ref<TextFile> p_text_file, const String &p_path);

	void _on_find_in_files_requested(String text);
	void _on_replace_in_files_requested(String text);
	void _on_find_in_files_result_selected(String fpath, int line_number, int begin, int end);
	void _start_find_in_files(bool with_replace);
	void _on_find_in_files_modified_files(PackedStringArray paths);

	static void _open_script_request(const String &p_path);

	static ScriptEditor *script_editor;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static ScriptEditor *get_singleton() { return script_editor; }

	bool toggle_scripts_panel();
	bool is_scripts_panel_toggled();
	void apply_scripts() const;
	void open_script_create_dialog(const String &p_base_name, const String &p_base_path);

	void ensure_select_current();

	_FORCE_INLINE_ bool edit(const RES &p_resource, bool p_grab_focus = true) { return edit(p_resource, -1, 0, p_grab_focus); }
	bool edit(const RES &p_resource, int p_line, int p_col, bool p_grab_focus = true);

	void get_breakpoints(List<String> *p_breakpoints);

	void save_current_script();
	void save_all_scripts();

	void set_window_layout(Ref<ConfigFile> p_layout);
	void get_window_layout(Ref<ConfigFile> p_layout);

	void set_scene_root_script(Ref<Script> p_script);
	Vector<Ref<Script>> get_open_scripts() const;

	bool script_goto_method(Ref<Script> p_script, const String &p_method);

	virtual void edited_scene_changed();

	void notify_script_close(const Ref<Script> &p_script);
	void notify_script_changed(const Ref<Script> &p_script);

	void close_builtin_scripts_from_scene(const String &p_scene);

	void goto_help(const String &p_desc) { _help_class_goto(p_desc); }
	void update_doc(const String &p_name);

	bool can_take_away_focus() const;

	VSplitContainer *get_left_list_split() { return list_split; }

	void set_live_auto_reload_running_scripts(bool p_enabled);

	void register_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter);
	void unregister_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter);

	static void register_create_script_editor_function(CreateScriptEditorFunc p_func);

	ScriptEditor(EditorNode *p_editor);
	~ScriptEditor();
};

class ScriptEditorPlugin : public EditorPlugin {
	GDCLASS(ScriptEditorPlugin, EditorPlugin);

	ScriptEditor *script_editor;
	EditorNode *editor;

public:
	virtual String get_name() const override { return "Script"; }
	bool has_main_screen() const override { return true; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;
	virtual void selected_notify() override;

	virtual void save_external_data() override;
	virtual void apply_changes() override;

	virtual void restore_global_state() override;
	virtual void save_global_state() override;

	virtual void set_window_layout(Ref<ConfigFile> p_layout) override;
	virtual void get_window_layout(Ref<ConfigFile> p_layout) override;

	virtual void get_breakpoints(List<String> *p_breakpoints) override;

	virtual void edited_scene_changed() override;

	ScriptEditorPlugin(EditorNode *p_node);
	~ScriptEditorPlugin();
};

#endif // SCRIPT_EDITOR_PLUGIN_H
