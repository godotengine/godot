/**************************************************************************/
/*  find_in_files.h                                                       */
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

#include "core/templates/hash_map.h"
#include "editor/docks/editor_dock.h"
#include "scene/gui/dialogs.h"

// Performs the actual search.
class FindInFilesSearch : public Node {
	GDCLASS(FindInFilesSearch, Node);

	// Config.
	String pattern;
	HashSet<String> extension_filter;
	HashSet<String> include_wildcards;
	HashSet<String> exclude_wildcards;
	String root_dir;
	bool whole_words = true;
	bool match_case = true;

	// State.
	bool searching = false;
	String current_dir;
	Vector<PackedStringArray> folders_stack;
	Vector<String> files_to_scan;
	int initial_files_count = 0;

	void _process();
	void _iterate();
	void _scan_dir(const String &p_path, PackedStringArray &r_out_folders, PackedStringArray &r_out_files_to_scan);
	void _scan_file(const String &p_fpath);

	bool _is_file_matched(const HashSet<String> &p_wildcards, const String &p_file_path, bool p_case_sensitive) const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_search_text(const String &p_pattern);
	void set_whole_words(bool p_whole_word);
	void set_match_case(bool p_match_case);
	void set_folder(const String &p_folder);
	void set_filter(const HashSet<String> &p_exts);
	void set_includes(const HashSet<String> &p_include_wildcards);
	void set_excludes(const HashSet<String> &p_exclude_wildcards);

	String get_search_text() const { return pattern; }

	bool is_whole_words() const { return whole_words; }
	bool is_match_case() const { return match_case; }

	void start();
	void stop();

	bool is_searching() const { return searching; }
	float get_progress() const;
};

class LineEdit;
class CheckBox;
class FileDialog;
class HBoxContainer;

// Prompts search parameters.
class FindInFilesDialog : public AcceptDialog {
	GDCLASS(FindInFilesDialog, AcceptDialog);

	void _on_folder_button_pressed();
	void _on_folder_selected(String p_path);
	void _on_search_text_modified(const String &p_text);
	void _on_search_text_submitted(const String &p_text);
	void _on_replace_text_submitted(const String &p_text);

	String _validate_filter_wildcard(const String &p_expression) const;

	bool replace_mode = false;
	LineEdit *search_text_line_edit = nullptr;

	Label *replace_label = nullptr;
	LineEdit *replace_text_line_edit = nullptr;

	LineEdit *folder_line_edit = nullptr;
	CheckBox *match_case_checkbox = nullptr;
	CheckBox *whole_words_checkbox = nullptr;
	Button *find_button = nullptr;
	Button *replace_button = nullptr;
	FileDialog *folder_dialog = nullptr;
	HBoxContainer *filters_container = nullptr;
	LineEdit *includes_line_edit = nullptr;
	LineEdit *excludes_line_edit = nullptr;

	HashMap<String, bool> filters_preferences;

protected:
	void _notification(int p_what);

	void custom_action(const String &p_action) override;
	static void _bind_methods();

public:
	void set_search_text(const String &p_text);
	void set_replace_text(const String &p_text);

	void set_replace_mode(bool p_replace);

	String get_search_text() const;
	String get_replace_text() const;
	bool is_match_case() const;
	bool is_whole_words() const;
	String get_folder() const;
	HashSet<String> get_filter() const;
	HashSet<String> get_includes() const;
	HashSet<String> get_excludes() const;

	FindInFilesDialog();
};

class Button;
class CheckButton;
class Tree;
class TreeItem;
class ProgressBar;

// Display search results.
class FindInFilesPanel : public MarginContainer {
	GDCLASS(FindInFilesPanel, MarginContainer);

	enum FindButtons {
		FIND_BUTTON_REPLACE,
		FIND_BUTTON_REMOVE,
	};

	struct Result {
		int line_number = 0;
		int begin = 0;
		int end = 0;
		int begin_trimmed = 0;
	};

	FindInFilesSearch *finder = nullptr;
	Label *find_label = nullptr;
	Label *search_text_label = nullptr;
	Tree *results_display = nullptr;
	Label *status_label = nullptr;
	CheckButton *keep_results_button = nullptr;
	Button *refresh_button = nullptr;
	Button *cancel_button = nullptr;
	Button *close_button = nullptr;
	ProgressBar *progress_bar = nullptr;
	HashMap<String, TreeItem *> file_items;
	HashMap<TreeItem *, int> file_items_results_count;
	HashMap<TreeItem *, Result> result_items;
	bool with_replace = false;

	HBoxContainer *replace_container = nullptr;
	LineEdit *replace_line_edit = nullptr;
	Button *replace_all_button = nullptr;

	void _on_button_clicked(TreeItem *p_item, int p_column, int p_id, int p_mouse_button_index);
	void _on_result_found(const String &p_fpath, int p_line_number, int p_begin, int p_end, const String &p_text);
	void _on_theme_changed();
	void _on_finished();
	void _on_refresh_button_clicked();
	void _on_cancel_button_clicked();
	void _on_close_button_clicked();
	void _on_result_selected();
	void _on_item_edited();
	void _on_replace_text_changed(const String &p_text);
	void _on_replace_all_clicked();

	void _apply_replaces_in_file(const String &p_fpath, const Vector<Result> &p_locations, const String &p_new_text);
	void _update_replace_buttons();
	void _update_matches_text();
	String _get_replace_text();

	void _draw_result_text(Object *p_item_obj, const Rect2 &p_rect);

	void _clear();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	FindInFilesSearch *get_finder() const { return finder; }

	void set_with_replace(bool p_with_replace);
	void set_replace_text(const String &p_text);
	bool is_keep_results() const;
	void set_search_labels_visibility(bool p_visible);

	void start_search();
	void stop_search();

	FindInFilesPanel();
};

class PopupMenu;
class TabContainer;

// Contains several FindInFilesPanels. A FindInFilesPanel contains the results of a
// `Find in Files` search or a `Replace in Files` search, while a
// FindInFilesContainer can contain several FindInFilesPanels so that multiple search
// results can remain at the same time.
class FindInFilesContainer : public EditorDock {
	GDCLASS(FindInFilesContainer, EditorDock);

	enum PanelMenuOptions {
		PANEL_CLOSE,
		PANEL_CLOSE_OTHERS,
		PANEL_CLOSE_RIGHT,
		PANEL_CLOSE_ALL,
	};

	TabContainer *tabs = nullptr;
	bool update_bar = true;
	PopupMenu *tabs_context_menu = nullptr;

	void _on_tab_close_pressed(int p_tab);
	void _update_bar_visibility();
	void _bar_menu_option(int p_option);
	void _bar_input(const Ref<InputEvent> &p_input);
	void _on_theme_changed();

	FindInFilesPanel *_create_new_panel();
	FindInFilesPanel *_get_current_panel();

	void _result_selected(const String &p_fpath, int p_line_number, int p_begin, int p_end);
	void _files_modified();
	void _close_panel(FindInFilesPanel *p_panel);
	void _on_dock_closed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	FindInFilesPanel *get_panel_for_results(const String &p_label);

	FindInFilesContainer();
};

class FindInFiles : public Object {
	GDCLASS(FindInFiles, Object);

	FindInFilesDialog *dialog = nullptr;
	FindInFilesContainer *container = nullptr;

	void _start_search(bool p_with_replace);
	void _result_selected(const String &p_fpath, int p_line_number, int p_begin, int p_end);
	void _files_modified();

protected:
	static void _bind_methods();

public:
	void open_dialog(const String &p_initial_text, bool p_replace = false);
	FindInFiles();
};
