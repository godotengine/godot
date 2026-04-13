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
#include "scene/gui/scroll_container.h"

// Performs the actual search.
class FindInFilesSearch : public Node {
	GDCLASS(FindInFilesSearch, Node);

	// Config.
	String pattern;
	String include_string;
	String exclude_string;
	HashSet<String> extension_filter;
	String root_dir;
	bool whole_words = false;
	bool match_case = false;

	// State.
	HashSet<String> include_wildcards;
	HashSet<String> exclude_wildcards;
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

	void _calculate_wildcard(String &p_text, HashSet<String> *r_hash_set);
	String _validate_filter_wildcard(const String &p_expression) const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void copy_from(const FindInFilesSearch *p_other);

	void set_search_text(const String &p_pattern);
	void set_whole_words(bool p_whole_word);
	void set_match_case(bool p_match_case);
	void set_filter(const HashSet<String> &p_exts);
	void set_includes(const String &p_include_wildcards);
	void set_excludes(const String &p_exclude_wildcards);
	void set_folder(const String &p_folder);

	String get_search_text() const { return pattern; }
	bool get_whole_words() { return whole_words; }
	bool get_match_case() { return match_case; }
	HashSet<String> *get_filter() { return &extension_filter; }
	String get_includes() { return include_string; }
	String get_excludes() { return exclude_string; }

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
class SpinBox;

// Prompts search parameters.
class FindInFilesSearchPanel : public ScrollContainer {
	GDCLASS(FindInFilesSearchPanel, ScrollContainer);

	void _on_folder_button_pressed();
	void _on_folder_selected(String p_path);

	void _on_search_modified();
	void _on_search_submitted();
	void _emit_find_requested();
	void _toggle_replace_pressed(bool with_replace);
	void _on_replace_all_clicked();
	void _on_additional_options_toggled(bool p_toggled);
	void _on_add_new_tab_clicked();
	void _update_finder();

	FindInFilesSearch *finder = nullptr;
	Timer *debounce_timer = nullptr;
	Button *toggle_replace_button = nullptr;
	LineEdit *search_text_line_edit = nullptr;

	HBoxContainer *folder_hbc = nullptr;
	Button *additional_options_button = nullptr;
	VBoxContainer *additional_options_vbc = nullptr;
	Button *match_case_checkbox = nullptr;
	Button *whole_words_checkbox = nullptr;
	Button *find_button = nullptr;

	HBoxContainer *replace_hbox = nullptr;
	LineEdit *replace_line_edit = nullptr;
	Button *replace_button = nullptr;
	Button *replace_all_button = nullptr;

	FileDialog *folder_dialog = nullptr;
	HBoxContainer *filters_container = nullptr;
	LineEdit *folder_line_edit = nullptr;
	LineEdit *includes_line_edit = nullptr;
	LineEdit *excludes_line_edit = nullptr;

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	void set_finder(FindInFilesSearch *p_finder, bool p_init);

	void set_search_text(const String &p_text);
	void set_replace_text(const String &p_text);

	String get_search_text() const;
	bool is_replace_pressed() const;
	HashSet<String> get_filter() const;

	LineEdit *get_replace_line_edit() const;

	FindInFilesSearchPanel();
};

class Button;
class Tree;
class TreeItem;
class ProgressBar;

// Display search results.
class FindInFilesResultsPanel : public MarginContainer {
	GDCLASS(FindInFilesResultsPanel, MarginContainer);

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
	Button *new_tab_button = nullptr;
	Button *refresh_button = nullptr;
	Button *cancel_button = nullptr;
	Control *progress_bar_placeholder = nullptr;
	ProgressBar *progress_bar = nullptr;
	HashMap<String, TreeItem *> file_items;
	HashMap<TreeItem *, int> file_items_results_count;
	HashMap<TreeItem *, Result> result_items;
	bool with_replace = false;

	String replace_text;

	bool floating = false;
	MarginContainer *results_mc = nullptr;

	void _on_button_clicked(TreeItem *p_item, int p_column, int p_id, int p_mouse_button_index);
	void _on_result_found(const String &p_fpath, int p_line_number, int p_begin, int p_end, const String &p_text);
	void _on_theme_changed();
	void _on_finished();
	void _on_refresh_button_clicked();
	void _on_cancel_button_clicked();
	void _on_result_selected();
	void _on_item_edited();

	void _apply_replaces_in_file(const String &p_fpath, const Vector<Result> &p_locations, const String &p_new_text);
	void _update_matches_text();

	void _draw_result_text(Object *p_item_obj, const Rect2 &p_rect);

	void _clear();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	FindInFilesSearch *get_finder() const { return finder; }
	Button *get_new_tab_button() { return new_tab_button; }
	String get_replace_text() { return replace_text; }

	void set_with_replace(bool p_with_replace);
	void set_replace_text(const String &p_text);
	void set_search_labels_visibility(bool p_visible);

	void start_search();
	void stop_search();
	void replace_all();

	void update_layout(EditorDock::DockLayout p_layout);

	FindInFilesResultsPanel();
};

class PopupMenu;
class TabContainer;
class HSplitContainer;

// Contains several FindInFilesResultsPanels. A FindInFilesResultsPanel contains the results of a
// FindInFilesSearchPanel search, while a FindInFilesContainer can contain several FindInFilesResultsPanels
// so that multiple search results can remain at the same time.
class FindInFilesContainer : public EditorDock {
	GDCLASS(FindInFilesContainer, EditorDock);

	enum PanelMenuOptions {
		PANEL_CLOSE,
		PANEL_CLOSE_OTHERS,
		PANEL_CLOSE_RIGHT,
		PANEL_CLOSE_ALL,
	};

	HSplitContainer *hsplit = nullptr;
	FindInFilesSearchPanel *search_control = nullptr;
	TabContainer *tabs = nullptr;
	bool update_bar = true;
	PopupMenu *tabs_context_menu = nullptr;

	void _on_tab_close_pressed(int p_tab);
	void _on_tab_changed();
	void _update_bar_visibility();
	void _bar_menu_option(int p_option);
	void _bar_input(const Ref<InputEvent> &p_input);
	void _on_theme_changed();

	FindInFilesResultsPanel *_create_new_panel();
	void _update_current_title();

	void _result_selected(const String &p_fpath, int p_line_number, int p_begin, int p_end);
	void _files_modified();
	void _close_panel(FindInFilesResultsPanel *p_panel);
	void _set_replace_text(String p_text);
	void _replace_all();
	void _on_dock_closed();
	void _start_find_in_files();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void update_layout(EditorDock::DockLayout p_layout) override;

	FindInFilesSearchPanel *get_search_control() { return search_control; }

	FindInFilesContainer();
};

class FindInFiles : public Object {
	GDCLASS(FindInFiles, Object);

	FindInFilesContainer *container = nullptr;

	static FindInFiles *find_in_files;

public:
	static FindInFiles *get_singleton() { return find_in_files; }

	FindInFilesContainer *get_container() { return container; }

	void open_dock(const String &p_initial_text = "");

	FindInFiles();
};
