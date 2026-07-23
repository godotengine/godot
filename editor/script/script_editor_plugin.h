/**************************************************************************/
/*  script_editor_plugin.h                                                */
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

#pragma once

#include "core/error/error_list.h"
#include "core/object/script_language.h"
#include "editor/plugins/editor_plugin.h"
#include "editor/script/script_editor_base.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/text_file.h"

class CodeTextEditor;
class EditorFileDialog;
class EditorHelpSearch;
class FilterLineEdit;
class HSplitContainer;
class ItemList;
class TabContainer;
class Tree;
class WindowWrapper;

class ScriptEditorQuickOpen : public ConfirmationDialog {
	GDCLASS(ScriptEditorQuickOpen, ConfirmationDialog);

	FilterLineEdit *search_box = nullptr;
	Tree *search_options = nullptr;
	String function;

	void _update_search();

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

class DocumentList : public VBoxContainer {
	GDCLASS(DocumentList, VBoxContainer);

	struct ItemData {
		String display_name;
		String tooltip;
		Ref<Texture2D> icon;
		int index = 0;
		bool hidden = false;
		bool tool = false;
		bool used_in_scene = false;
		bool has_error = false;
	};

	LocalVector<ItemData> items;

	ItemList *item_list = nullptr;
	LineEdit *filter = nullptr;
	PopupMenu *context_menu = nullptr;

	DocumentEditorContainer *document_editor_container;

	bool script_temperature_enabled = false;
	int temperature_history_size = 0;

	void _document_selected(int p_index);
	void _document_clicked(int p_index, Vector2 p_local_mouse_pos, MouseButton p_mouse_button_index);
	void _show_context_menu();

	void _update_item_list();

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

protected:
	void _notification(int p_what);

public:
	void update_list();
	void ensure_current_is_visible();
	void goto_next_document(bool p_previous = false);
	void update_editor_settings();

	DocumentList(DocumentEditorContainer *p_document_editor_container);
};

class DocumentOutline : public VBoxContainer {
	GDCLASS(DocumentOutline, VBoxContainer);

	DocumentEditorContainer *document_editor_container = nullptr;
	ItemList *item_list = nullptr;
	HBoxContainer *buttons_hbox = nullptr;
	FilterLineEdit *filter = nullptr;
	Button *sort_button = nullptr;

	bool members_overview_enabled = false;
	bool help_overview_enabled = false;

	void _toggle_sort(bool p_alphabetic_sort);
	void _item_list_selected(int p_idx);

protected:
	void _notification(int p_what);

public:
	void update_editor_settings();
	void update_outline();
	void update_visibility();

	DocumentOutline(DocumentEditorContainer *p_document_editor_container);
};

class EditorScriptCodeCompletionCache;
class FindInFiles;
class TextFile;
class ShaderCreateDialog;

class DocumentEditorContainer : public MarginContainer {
	GDCLASS(DocumentEditorContainer, MarginContainer);

	enum MenuOptions {
		// File.
		FILE_MENU_NEW_SCRIPT,
		FILE_MENU_NEW_TEXTFILE,
		FILE_MENU_NEW_SHADER,
		FILE_MENU_NEW_INCLUDE,

		FILE_MENU_OPEN,
		FILE_MENU_OPEN_SHADER,
		FILE_MENU_OPEN_INCLUDE,
		FILE_MENU_REOPEN_CLOSED,
		FILE_MENU_OPEN_RECENT,

		FILE_MENU_SAVE,
		FILE_MENU_SAVE_AS,
		FILE_MENU_SAVE_ALL,

		FILE_MENU_INSPECT,
		FILE_MENU_INSPECT_NATIVE_SHADER_CODE,

		FILE_MENU_SOFT_RELOAD_TOOL,
		FILE_MENU_COPY_PATH,
		FILE_MENU_COPY_UID,
		FILE_MENU_SHOW_IN_FILE_SYSTEM,

		FILE_MENU_HISTORY_PREV,
		FILE_MENU_HISTORY_NEXT,

		FILE_MENU_THEME_SUBMENU,

		FILE_MENU_CLOSE,
		FILE_MENU_CLOSE_ALL,
		FILE_MENU_CLOSE_OTHER_TABS,
		FILE_MENU_CLOSE_TABS_BELOW,
		FILE_MENU_CLOSE_DOCS,

		FILE_MENU_RUN,

		FILE_MENU_TOGGLE_FILES_PANEL,

		FILE_MENU_MOVE_UP,
		FILE_MENU_MOVE_DOWN,
		FILE_MENU_SORT,

		// Search.
		HELP_SEARCH_FIND,
		HELP_SEARCH_FIND_NEXT,
		HELP_SEARCH_FIND_PREVIOUS,

		SEARCH_IN_FILES,
		REPLACE_IN_FILES,

		SEARCH_HELP,
		SEARCH_WEBSITE,

		// Theme.
		THEME_IMPORT,
		THEME_RELOAD,
		THEME_SAVE_AS,
	};

	enum DocumentSortMode {
		SORT_BY_NAME,
		SORT_BY_PATH,
		SORT_BY_NONE,
	};

	enum DocumentNameDisplayMode {
		DISPLAY_NAME,
		DISPLAY_DIR_AND_NAME,
		DISPLAY_FULL_PATH,
	};

	HBoxContainer *menu_hb = nullptr;
	MenuButton *file_menu = nullptr;
	MenuButton *script_search_menu = nullptr;
	MenuButton *debug_menu = nullptr;
	Timer *autosave_timer = nullptr;
	LocalVector<ScriptEditorBase::EditMenusBase *> editor_menus;

	PopupMenu *recent_scripts = nullptr;
	PopupMenu *theme_submenu = nullptr;

	Button *help_search = nullptr;
	Button *site_search = nullptr;
	Button *make_floating = nullptr;
	bool is_floating = false;

	friend class DocumentList;

	DocumentList *document_list = nullptr;
	HSplitContainer *script_split = nullptr;
	DocumentOutline *document_outline = nullptr;
	VSplitContainer *list_split = nullptr;
	TabContainer *tab_container = nullptr;
	AcceptDialog *error_dialog = nullptr;
	ConfirmationDialog *erase_tab_confirm = nullptr;
	FindReplaceBar *find_replace_bar = nullptr;

	EditorFileDialog *file_dialog = nullptr;
	ScriptCreateDialog *script_create_dialog = nullptr;
	ShaderCreateDialog *shader_create_dialog = nullptr;

	float zoom_factor = 1.0f;

	HBoxContainer *script_name_button_hbox = nullptr;
	Control *script_name_button_left_spacer = nullptr;
	Control *script_name_button_right_spacer = nullptr;
	Button *script_name_button = nullptr;
	int script_name_width = 0;

	bool list_update_queued = false;
	bool document_sort_group_help_pages = false;
	DocumentSortMode document_sort_by_mode = SORT_BY_NAME;
	DocumentNameDisplayMode document_display_name_mode = DISPLAY_NAME;
	bool highlight_scene_scripts = false;

	Button *script_back = nullptr;
	Button *script_forward = nullptr;

	WindowWrapper *window_wrapper = nullptr;

#ifdef ANDROID_ENABLED
	Control *virtual_keyboard_spacer = nullptr;
	int last_kb_height = -1;
#endif

	enum {
		SCRIPT_EDITOR_FUNC_MAX = 32,
	};

	static int script_editor_func_count;
	static CreateScriptEditorFunc script_editor_funcs[SCRIPT_EDITOR_FUNC_MAX];

	Vector<Ref<EditorSyntaxHighlighter>> syntax_highlighters;

	struct ScriptHistory {
		Control *control = nullptr;
		Dictionary state;
	};

	Vector<ScriptHistory> history;
	int history_pos;

	List<String> previous_scripts;
	List<int> script_close_queue;

	bool restoring_layout = false;
	bool grab_focus_block = false;

	int edit_pass = 0;

	bool is_main_editor = false;

	void _menu_option(int p_option);
	void _theme_option(int p_option);
	void _show_save_theme_as_dialog();
	bool _has_docs_tab() const;
	bool _has_script_tab() const;

	Tree *disk_changed_list = nullptr;
	ConfirmationDialog *disk_changed = nullptr;

	void _resave_scripts(const String &p_str);

	bool _test_script_times_on_disk(Ref<Resource> p_for_script = Ref<Resource>());
	bool _script_exists(const String &p_path) const;

	void _add_recent_script(const String &p_path);
	void _update_recent_scripts();
	void _open_recent_script(int p_idx);

	void _show_error_dialog(const String &p_path);

	void _close_tab(int p_idx, bool p_save = true);
	void _update_find_replace_bar();

	void _close_current_tab(bool p_save = true);
	void _close_discard_current_tab(const String &p_str);
	void _close_docs_tab();
	void _close_other_tabs();
	void _close_tabs_below();
	void _close_all_tabs();
	void _queue_close_tabs();

	void _copy_script_path();
	void _copy_script_uid();

	void _ask_close_current_unsaved_tab(ScriptEditorBase *p_seb);

	bool pending_auto_reload;
	bool auto_reload_running_scripts;
	Vector<String> script_paths_to_reload;
	void _live_auto_reload_running_scripts();

	void _update_selected_editor_menu();

	void _add_callback(Object *p_obj, const String &p_function, const PackedStringArray &p_args);
	void _res_saved_callback(const Ref<Resource> &p_res);
	void _scene_saved_callback(const String &p_path);
	void _mark_built_in_scripts_as_saved(const String &p_parent_path);

	bool trim_trailing_whitespace_on_save;
	bool trim_final_newlines_on_save;
	bool convert_indent_on_save;
	bool external_editor_active;

	void _goto_script_line(Ref<RefCounted> p_script, int p_line);
	void _change_execution(Ref<RefCounted> p_script, int p_line = -1, bool p_set = false);
	void _set_execution(Ref<RefCounted> p_script, int p_line) { _change_execution(p_script, p_line, true); }
	void _clear_execution(Ref<RefCounted> p_script) { _change_execution(p_script); }
	String _get_debug_tooltip(const String &p_text, Node *p_se);
	void _resource_created(Ref<Resource> p_res);
	void _set_breakpoint(Ref<RefCounted> p_script, int p_line, bool p_enabled);
	void _clear_breakpoints();
	Array _get_cached_breakpoints_for_script(const String &p_path) const;

	ScriptEditorBase *_get_current_editor() const;

	Ref<ConfigFile> script_editor_cache;
	String cache_path;

	void _save_editor_state(ScriptEditorBase *p_editor);
	void _save_layout();
	void _apply_editor_settings();
	void _update_filenames();
	void _files_moved(const String &p_old_file, const String &p_new_file);
	void _file_removed(const String &p_file);
	void _autosave_scripts();
	void _update_autosave_timer();
	void _reload_scripts(bool p_refresh_only = false);
	void _auto_format_text(ScriptEditorBase *p_seb);

	void _connect_to_scene();
	void _connect_to_scene_recursive(Node *p_current, Node *p_base);
	void _update_document_list();
	void _queue_update_list();

	void _update_online_doc();

	void _split_dragged(float);

	virtual void input(const Ref<InputEvent> &p_event) override;
	virtual void shortcut_input(const Ref<InputEvent> &p_event) override;

	void _setup_popup_menu(PopupMenu *p_menu, bool p_is_context_menu = true);
	void _prepare_popup_menu(PopupMenu *p_menu, bool p_is_context_menu = true);
	void _prepare_file_menu();
	void _file_menu_closed();

	void _calculate_script_name_button_size();
	void _calculate_script_name_button_ratio();
	void _update_document_name_button();

	void _help_search(const String &p_text = String());

	void _history_forward();
	void _history_back();
	void _roll_back_to_pre_tab();

	void _help_class_open(const String &p_class);
	void _help_class_goto(const String &p_desc);
	void _update_history_arrows();
	void _save_new_history(const Dictionary &p_state, Control *p_control);
	void _save_previous_state(const Dictionary &p_state, Control *p_control);
	void _compress_history_patterns(bool p_once);
	void _go_to_tab(int p_idx, bool p_save_history = false);
	void _update_history_pos(int p_new_pos);
	void _update_modified_scripts_for_external_editor(Ref<Script> p_for_script = Ref<Script>());

	void _script_changed();
	int file_dialog_option;
	void _file_dialog_action(const String &p_file);

	String config_section;

	HashSet<String> handled_resource_types;
	List<String> _get_recognized_extensions();
	HashSet<String> textfile_extensions;
	Ref<TextFile> _load_text_file(const String &p_path, Error *r_error) const;
	Error _save_text_file(Ref<TextFile> p_text_file, const String &p_path);

	void _set_script_zoom_factor(float p_zoom_factor);
	void _update_code_editor_zoom_factor(CodeTextEditor *p_code_text_editor);

	void _floating_state_changed(bool p_floating);
	void _make_floating(int p_screen);

	void _close_builtin_scripts_from_scene(const String &p_scene);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	bool toggle_files_panel();
	bool is_files_panel_toggled();
	void apply_scripts() const;
	void reload_scripts(bool p_refresh_only = false);
	void open_script_create_dialog(const String &p_base_name, const String &p_base_path);
	void open_text_file_create_dialog(const String &p_base_path, const String &p_base_name = "");
	Ref<Resource> open_file(const String &p_file);
	bool can_open_file(const String &p_file) const;
	Error close_file(const String &p_file);

	void ensure_select_current();

	bool is_editor_floating();

	void set_handled_resource_types(const HashSet<String> &p_file_types) { handled_resource_types = p_file_types; }

	_FORCE_INLINE_ bool edit(const Ref<Resource> &p_resource, bool p_grab_focus = true) { return edit(p_resource, -1, 0, p_grab_focus); }
	bool edit(const Ref<Resource> &p_resource, int p_line, int p_col, bool p_grab_focus = true);

	Control *get_active_editor() const;
	Vector<Control *> get_all_editors() const;

	Vector<String> _get_breakpoints();
	void get_breakpoints(List<String> *p_breakpoints);

	void reload_open_files();
	PackedStringArray get_unsaved_files() const;
	PackedStringArray get_unsaved_scripts() const;
	void save_current_script();
	void save_all_scripts();
	void update_script_times();
	void sort_document_editors();

	void set_window_layout(Ref<ConfigFile> p_layout);
	void get_window_layout(Ref<ConfigFile> p_layout);

	void set_scene_root_script(Ref<Script> p_script);
	Vector<Ref<Script>> get_open_scripts() const;
	ScriptEditorBase *get_resource_editor(const Ref<Resource> &p_res) const;

	ScriptEditorBase *get_current_editor() const { return _get_current_editor(); }

	bool script_goto_method(Ref<Script> p_script, const String &p_method);

	virtual void edited_scene_changed();

	void goto_help(const String &p_desc) { _help_class_goto(p_desc); }
	void update_doc(const String &p_name);
	void clear_docs_from_script(const Ref<Script> &p_script);
	void update_docs_from_script(const Ref<Script> &p_script);

	void trigger_live_script_reload(const String &p_script_path);

	void set_live_auto_reload_running_scripts(bool p_enabled);

	void register_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter);
	void unregister_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter);

	static void register_create_script_editor_function(CreateScriptEditorFunc p_func);

	DocumentEditorContainer(bool p_is_main_editor, const String &p_config_section, const String &p_cache_path);
};

class ScriptEditor : public PanelContainer {
	GDCLASS(ScriptEditor, PanelContainer);

	inline static ScriptEditor *script_editor = nullptr;

	DocumentEditorContainer *script_container = nullptr;

	EditorHelpSearch *help_search_dialog = nullptr;
	FindInFiles *find_in_files = nullptr;

	void _on_find_in_files_result_selected(const String &p_path, int p_line_number, int p_begin, int p_end);
	void _on_find_in_files_modified_files();

	TypedArray<ScriptEditorBase> _get_open_script_editors() const;
	void _goto_line(int p_line);
	Ref<Script> _get_current_script();
	TypedArray<Script> _get_open_scripts() const;

protected:
	static void _bind_methods();

public:
	static ScriptEditor *get_singleton() { return script_editor; }

	DocumentEditorContainer *get_script_container() { return script_container; }

	bool toggle_files_panel() { return script_container->toggle_files_panel(); }
	bool is_files_panel_toggled() { return script_container->is_files_panel_toggled(); }
	void apply_scripts() const { script_container->apply_scripts(); }
	void reload_scripts(bool p_refresh_only = false) { script_container->reload_scripts(p_refresh_only); }

	void open_find_in_files_dialog(const String &p_initial_text = String(), bool p_replace = false);
	void open_help_search_dialog(const String &p_search_text = String());

	void open_script_create_dialog(const String &p_base_name, const String &p_base_path) { script_container->open_script_create_dialog(p_base_name, p_base_path); }
	void open_text_file_create_dialog(const String &p_base_path, const String &p_base_name = "") { script_container->open_text_file_create_dialog(p_base_path, p_base_name); }
	Ref<Resource> open_file(const String &p_file) { return script_container->open_file(p_file); }
	Error close_file(const String &p_file) { return script_container->close_file(p_file); }

	void ensure_select_current() { script_container->ensure_select_current(); }

	bool is_editor_floating() { return script_container->is_editor_floating(); }

	_FORCE_INLINE_ bool edit(const Ref<Resource> &p_resource, bool p_grab_focus = true) { return edit(p_resource, -1, 0, p_grab_focus); }
	bool edit(const Ref<Resource> &p_resource, int p_line, int p_col, bool p_grab_focus = true) { return script_container->edit(p_resource, p_line, p_col, p_grab_focus); }

	Control *get_active_editor() const { return script_container->get_active_editor(); }

	Vector<String> _get_breakpoints() { return script_container->_get_breakpoints(); }
	void get_breakpoints(List<String> *p_breakpoints) { script_container->get_breakpoints(p_breakpoints); }

	void reload_open_files() { script_container->reload_open_files(); }
	PackedStringArray get_unsaved_files() const { return script_container->get_unsaved_files(); }
	PackedStringArray get_unsaved_scripts() const { return script_container->get_unsaved_scripts(); }
	void save_current_script() { script_container->save_current_script(); }
	void save_all_scripts() { script_container->save_all_scripts(); }
	void update_script_times() { script_container->update_script_times(); }

	void set_window_layout(Ref<ConfigFile> p_layout) { script_container->set_window_layout(p_layout); }
	void get_window_layout(Ref<ConfigFile> p_layout) { script_container->get_window_layout(p_layout); }

	void set_scene_root_script(Ref<Script> p_script) { script_container->set_scene_root_script(p_script); }
	Vector<Ref<Script>> get_open_scripts() const { return script_container->get_open_scripts(); }
	ScriptEditorBase *get_resource_editor(const Ref<Resource> &p_res) const { return script_container->get_resource_editor(p_res); }

	ScriptEditorBase *get_current_editor() const { return script_container->get_current_editor(); }

	bool script_goto_method(Ref<Script> p_script, const String &p_method) { return script_container->script_goto_method(p_script, p_method); }

	virtual void edited_scene_changed() { script_container->edited_scene_changed(); }

	void notify_script_close(const Ref<Script> &p_script);
	void notify_script_changed(const Ref<Script> &p_script);

	void goto_help(const String &p_desc) { script_container->goto_help(p_desc); }
	void update_doc(const String &p_name) { script_container->update_doc(p_name); }
	void clear_docs_from_script(const Ref<Script> &p_script) { script_container->clear_docs_from_script(p_script); }
	void update_docs_from_script(const Ref<Script> &p_script) { script_container->update_docs_from_script(p_script); }

	void trigger_live_script_reload(const String &p_script_path) { script_container->trigger_live_script_reload(p_script_path); }

	void set_live_auto_reload_running_scripts(bool p_enabled) { script_container->set_live_auto_reload_running_scripts(p_enabled); }

	void register_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter) { script_container->register_syntax_highlighter(p_syntax_highlighter); }
	void unregister_syntax_highlighter(const Ref<EditorSyntaxHighlighter> &p_syntax_highlighter) { script_container->unregister_syntax_highlighter(p_syntax_highlighter); }

	static void register_create_script_editor_function(CreateScriptEditorFunc p_func) { DocumentEditorContainer::register_create_script_editor_function(p_func); }

	ScriptEditor(WindowWrapper *p_wrapper);
	~ScriptEditor();
};

class ScriptEditorPlugin : public EditorPlugin {
	GDCLASS(ScriptEditorPlugin, EditorPlugin);

	ScriptEditor *script_editor = nullptr;
	WindowWrapper *window_wrapper = nullptr;

	String last_editor;

	void _focus_another_editor();

	void _save_last_editor(const String &p_editor);
	void _window_visibility_changed(bool p_visible);
	void _update_theme();

	static inline ScriptEditorPlugin *script_editor_plugin = nullptr;

protected:
	void _notification(int p_what);

public:
	static ScriptEditorPlugin *get_singleton() { return script_editor_plugin; }

	static bool open_in_external_editor(const String &p_path, int p_line, int p_col, bool p_ignore_project = false);

	virtual String get_plugin_name() const override { return TTRC("Script"); }
	virtual bool has_main_screen() const override { return true; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;
	virtual void selected_notify() override;

	virtual String get_unsaved_status(const String &p_for_scene) const override;
	virtual void save_external_data() override;
	virtual void apply_changes() override { script_editor->apply_scripts(); }

	virtual void set_window_layout(Ref<ConfigFile> p_layout) override;
	virtual void get_window_layout(Ref<ConfigFile> p_layout) override;

	virtual void get_breakpoints(List<String> *p_breakpoints) override { script_editor->get_breakpoints(p_breakpoints); }

	virtual void edited_scene_changed() override { script_editor->edited_scene_changed(); }

	ScriptEditorPlugin();
};
