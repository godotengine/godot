/**************************************************************************/
/*  find_in_files.cpp                                                     */
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

#include "find_in_files.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/check_button.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tree.h"

// TODO: Would be nice in Vector and Vectors.
template <typename T>
inline void pop_back(T &r_container) {
	r_container.resize(r_container.size() - 1);
}

static bool find_next(const String &p_line, const String &p_pattern, int p_from, bool p_match_case, bool p_whole_words, int &r_out_begin, int &r_out_end) {
	int end = p_from;

	while (true) {
		int begin = p_match_case ? p_line.find(p_pattern, end) : p_line.findn(p_pattern, end);

		if (begin == -1) {
			return false;
		}

		end = begin + p_pattern.length();
		r_out_begin = begin;
		r_out_end = end;

		if (p_whole_words) {
			if (begin > 0 && is_ascii_identifier_char(p_line[begin - 1])) {
				continue;
			}
			if (end < p_line.size() && is_ascii_identifier_char(p_line[end])) {
				continue;
			}
		}

		return true;
	}
}

//--------------------------------------------------------------------------------

void FindInFilesSearch::set_search_text(const String &p_pattern) {
	pattern = p_pattern;
}

void FindInFilesSearch::set_whole_words(bool p_whole_word) {
	whole_words = p_whole_word;
}

void FindInFilesSearch::set_match_case(bool p_match_case) {
	match_case = p_match_case;
}

void FindInFilesSearch::set_folder(const String &p_folder) {
	root_dir = p_folder;
}

void FindInFilesSearch::set_filter(const HashSet<String> &p_exts) {
	extension_filter = p_exts;
}

void FindInFilesSearch::set_includes(const HashSet<String> &p_include_wildcards) {
	include_wildcards = p_include_wildcards;
}

void FindInFilesSearch::set_excludes(const HashSet<String> &p_exclude_wildcards) {
	exclude_wildcards = p_exclude_wildcards;
}

void FindInFilesSearch::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			_process();
		} break;
	}
}

void FindInFilesSearch::start() {
	if (pattern.is_empty()) {
		print_verbose("Nothing to search, pattern is empty");
		emit_signal(SceneStringName(finished));
		return;
	}
	if (extension_filter.is_empty()) {
		print_verbose("Nothing to search, filter matches no files");
		emit_signal(SceneStringName(finished));
		return;
	}

	// Init search.
	current_dir = "";
	PackedStringArray init_folder;
	init_folder.push_back(root_dir);
	folders_stack.clear();
	folders_stack.push_back(init_folder);

	initial_files_count = 0;

	searching = true;
	set_process(true);
}

void FindInFilesSearch::stop() {
	searching = false;
	current_dir = "";
	set_process(false);
}

void FindInFilesSearch::_process() {
	// This part can be moved to a thread if needed.

	OS &os = *OS::get_singleton();
	uint64_t time_before = os.get_ticks_msec();
	while (is_processing()) {
		_iterate();
		uint64_t elapsed = (os.get_ticks_msec() - time_before);
		if (elapsed > 8) { // Process again after waiting 8 ticks.
			break;
		}
	}
}

void FindInFilesSearch::_iterate() {
	if (folders_stack.size() != 0) {
		// Scan folders first so we can build a list of files and have progress info later.

		PackedStringArray &folders_to_scan = folders_stack.write[folders_stack.size() - 1];

		if (folders_to_scan.size() != 0) {
			// Scan one folder below.

			String folder_name = folders_to_scan[folders_to_scan.size() - 1];
			pop_back(folders_to_scan);

			current_dir = current_dir.path_join(folder_name);

			PackedStringArray sub_dirs;
			PackedStringArray more_files_to_scan;
			_scan_dir("res://" + current_dir, sub_dirs, more_files_to_scan);

			folders_stack.push_back(sub_dirs);
			files_to_scan.append_array(more_files_to_scan);

		} else {
			// Go back one level.

			pop_back(folders_stack);
			current_dir = current_dir.get_base_dir();

			if (folders_stack.is_empty()) {
				// All folders scanned.
				initial_files_count = files_to_scan.size();
			}
		}

	} else if (files_to_scan.size() != 0) {
		// Then scan files.

		String fpath = files_to_scan[files_to_scan.size() - 1];
		pop_back(files_to_scan);
		_scan_file(fpath);

	} else {
		print_verbose("Search complete");
		set_process(false);
		current_dir = "";
		searching = false;
		emit_signal(SceneStringName(finished));
	}
}

float FindInFilesSearch::get_progress() const {
	if (initial_files_count != 0) {
		return static_cast<float>(initial_files_count - files_to_scan.size()) / static_cast<float>(initial_files_count);
	}
	return 0;
}

void FindInFilesSearch::_scan_dir(const String &p_path, PackedStringArray &r_out_folders, PackedStringArray &r_out_files_to_scan) {
	Ref<DirAccess> dir = DirAccess::open(p_path);
	if (dir.is_null()) {
		print_verbose("Cannot open directory! " + p_path);
		return;
	}

	dir->list_dir_begin();

	// Limit to 100,000 iterations to avoid an infinite loop just in case
	// (this technically limits results to 100,000 files per folder).
	for (int i = 0; i < 100'000; ++i) {
		String file = dir->get_next();

		if (file.is_empty()) {
			break;
		}

		// If there is a .gdignore file in the directory, clear all the files/folders
		// to be searched on this path and skip searching the directory.
		if (file == ".gdignore") {
			r_out_folders.clear();
			r_out_files_to_scan.clear();
			break;
		}

		// Ignore special directories (such as those beginning with . and the project data directory).
		String project_data_dir_name = ProjectSettings::get_singleton()->get_project_data_dir_name();
		if (file.begins_with(".") || file == project_data_dir_name) {
			continue;
		}
		if (dir->current_is_hidden()) {
			continue;
		}

		if (dir->current_is_dir()) {
			r_out_folders.push_back(file);

		} else {
			String file_ext = file.get_extension();
			if (extension_filter.has(file_ext)) {
				String file_path = p_path.path_join(file);
				bool case_sensitive = dir->is_case_sensitive(p_path);

				if (!exclude_wildcards.is_empty() && _is_file_matched(exclude_wildcards, file_path, case_sensitive)) {
					continue;
				}

				if (include_wildcards.is_empty() || _is_file_matched(include_wildcards, file_path, case_sensitive)) {
					r_out_files_to_scan.push_back(file_path);
				}
			}
		}
	}
}

void FindInFilesSearch::_scan_file(const String &p_fpath) {
	Ref<FileAccess> f = FileAccess::open(p_fpath, FileAccess::READ);
	if (f.is_null()) {
		print_verbose("Cannot open file " + p_fpath);
		return;
	}

	int line_number = 0;

	while (!f->eof_reached()) {
		// Line number starts at 1.
		++line_number;

		int begin = 0;
		int end = 0;

		String line = f->get_line();

		while (find_next(line, pattern, end, match_case, whole_words, begin, end)) {
			emit_signal(SNAME("result_found"), p_fpath, line_number, begin, end, line);
		}
	}
}

bool FindInFilesSearch::_is_file_matched(const HashSet<String> &p_wildcards, const String &p_file_path, bool p_case_sensitive) const {
	const String file_path = "/" + p_file_path.replace_char('\\', '/') + "/";

	for (const String &wildcard : p_wildcards) {
		if (p_case_sensitive && file_path.match(wildcard)) {
			return true;
		} else if (!p_case_sensitive && file_path.matchn(wildcard)) {
			return true;
		}
	}
	return false;
}

void FindInFilesSearch::_bind_methods() {
	ADD_SIGNAL(MethodInfo("result_found",
			PropertyInfo(Variant::STRING, "path"),
			PropertyInfo(Variant::INT, "line_number"),
			PropertyInfo(Variant::INT, "begin"),
			PropertyInfo(Variant::INT, "end"),
			PropertyInfo(Variant::STRING, "text")));

	ADD_SIGNAL(MethodInfo("finished"));
}

//-----------------------------------------------------------------------------

void FindInFilesDialog::set_search_text(const String &p_text) {
	if (!p_text.is_empty()) {
		search_text_line_edit->set_text(p_text);
		_on_search_text_modified(p_text);
	}
	if (replace_mode && !p_text.is_empty()) {
		callable_mp((Control *)replace_text_line_edit, &Control::grab_focus).call_deferred(false);
		replace_text_line_edit->select_all();
	} else {
		callable_mp((Control *)search_text_line_edit, &Control::grab_focus).call_deferred(false);
		search_text_line_edit->select_all();
	}
}

void FindInFilesDialog::set_replace_text(const String &p_text) {
	replace_text_line_edit->set_text(p_text);
}

void FindInFilesDialog::set_replace_mode(bool p_replace) {
	if (replace_mode == p_replace) {
		return;
	}

	replace_mode = p_replace;

	if (replace_mode) {
		set_title(TTRC("Replace in Files"));
		replace_label->show();
		replace_text_line_edit->show();
	} else {
		set_title(TTRC("Find in Files"));
		replace_label->hide();
		replace_text_line_edit->hide();
	}

	// Recalculate the dialog size after hiding child controls.
	set_size(Size2(get_size().x, 0));
}

String FindInFilesDialog::get_search_text() const {
	return search_text_line_edit->get_text();
}

String FindInFilesDialog::get_replace_text() const {
	return replace_text_line_edit->get_text();
}

bool FindInFilesDialog::is_match_case() const {
	return match_case_checkbox->is_pressed();
}

bool FindInFilesDialog::is_whole_words() const {
	return whole_words_checkbox->is_pressed();
}

String FindInFilesDialog::get_folder() const {
	String p_text = folder_line_edit->get_text();
	return p_text.strip_edges();
}

HashSet<String> FindInFilesDialog::get_filter() const {
	// Could check the filters_preferences but it might not have been generated yet.
	HashSet<String> filters;
	for (int i = 0; i < filters_container->get_child_count(); ++i) {
		CheckBox *cb = static_cast<CheckBox *>(filters_container->get_child(i));
		if (cb->is_pressed()) {
			filters.insert(cb->get_text());
		}
	}
	return filters;
}

HashSet<String> FindInFilesDialog::get_includes() const {
	HashSet<String> includes;
	String p_text = includes_line_edit->get_text();

	if (p_text.is_empty()) {
		return includes;
	}

	PackedStringArray wildcards = p_text.split(",", false);
	for (const String &wildcard : wildcards) {
		includes.insert(_validate_filter_wildcard(wildcard));
	}
	return includes;
}

HashSet<String> FindInFilesDialog::get_excludes() const {
	HashSet<String> excludes;
	String p_text = excludes_line_edit->get_text();

	if (p_text.is_empty()) {
		return excludes;
	}

	PackedStringArray wildcards = p_text.split(",", false);
	for (const String &wildcard : wildcards) {
		excludes.insert(_validate_filter_wildcard(wildcard));
	}
	return excludes;
}

void FindInFilesDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				// Extensions might have changed in the meantime, we clean them and instance them again.
				for (int i = 0; i < filters_container->get_child_count(); i++) {
					filters_container->get_child(i)->queue_free();
				}
				Array exts = GLOBAL_GET("editor/script/search_in_file_extensions");
				for (int i = 0; i < exts.size(); ++i) {
					CheckBox *cb = memnew(CheckBox);
					cb->set_text(exts[i]);
					if (!filters_preferences.has(exts[i])) {
						filters_preferences[exts[i]] = true;
					}
					cb->set_pressed(filters_preferences[exts[i]]);
					filters_container->add_child(cb);
				}
			}
		} break;
	}
}

void FindInFilesDialog::_on_folder_button_pressed() {
	folder_dialog->popup_file_dialog();
}

void FindInFilesDialog::custom_action(const String &p_action) {
	for (int i = 0; i < filters_container->get_child_count(); ++i) {
		CheckBox *cb = static_cast<CheckBox *>(filters_container->get_child(i));
		filters_preferences[cb->get_text()] = cb->is_pressed();
	}

	if (p_action == "find") {
		emit_signal(SNAME("find_requested"));
	} else if (p_action == "replace") {
		emit_signal(SNAME("replace_requested"));
	}
	hide();
}

void FindInFilesDialog::_on_search_text_modified(const String &p_text) {
	ERR_FAIL_NULL(find_button);
	ERR_FAIL_NULL(replace_button);

	find_button->set_disabled(get_search_text().is_empty());
	replace_button->set_disabled(get_search_text().is_empty());
}

void FindInFilesDialog::_on_search_text_submitted(const String &p_text) {
	// This allows to trigger a global search without leaving the keyboard.
	if (!replace_mode && !find_button->is_disabled()) {
		custom_action("find");
	}

	if (replace_mode && !replace_button->is_disabled()) {
		custom_action("replace");
	}
}

void FindInFilesDialog::_on_replace_text_submitted(const String &p_text) {
	// This allows to trigger a global search without leaving the keyboard.
	if (replace_mode && !replace_button->is_disabled()) {
		custom_action("replace");
	}
}

void FindInFilesDialog::_on_folder_selected(String p_path) {
	int i = p_path.find("://");
	if (i != -1) {
		p_path = p_path.substr(i + 3);
	}
	folder_line_edit->set_text(p_path);
}

String FindInFilesDialog::_validate_filter_wildcard(const String &p_expression) const {
	String ret = p_expression.replace_char('\\', '/');
	if (ret.begins_with("./")) {
		// Relative to the project root.
		ret = "res://" + ret.trim_prefix("./");
	}

	if (ret.begins_with(".")) {
		// To match extension.
		ret = "*" + ret;
	}

	if (!ret.begins_with("*")) {
		ret = "*/" + ret.trim_prefix("/");
	}

	if (!ret.ends_with("*")) {
		ret = ret.trim_suffix("/") + "/*";
	}

	return ret;
}

void FindInFilesDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("find_requested"));
	ADD_SIGNAL(MethodInfo("replace_requested"));
}

FindInFilesDialog::FindInFilesDialog() {
	set_min_size(Size2(500 * EDSCALE, 0));
	set_title(TTRC("Find in Files"));

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_anchor_and_offset(SIDE_LEFT, Control::ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_offset(SIDE_TOP, Control::ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_offset(SIDE_RIGHT, Control::ANCHOR_END, -8 * EDSCALE);
	vbc->set_anchor_and_offset(SIDE_BOTTOM, Control::ANCHOR_END, -8 * EDSCALE);
	add_child(vbc);

	GridContainer *gc = memnew(GridContainer);
	gc->set_columns(2);
	vbc->add_child(gc);

	Label *find_label = memnew(Label);
	find_label->set_text(TTRC("Find:"));
	gc->add_child(find_label);

	search_text_line_edit = memnew(LineEdit);
	search_text_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_text_line_edit->set_accessibility_name(TTRC("Find:"));
	search_text_line_edit->connect(SceneStringName(text_changed), callable_mp(this, &FindInFilesDialog::_on_search_text_modified));
	search_text_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FindInFilesDialog::_on_search_text_submitted));
	gc->add_child(search_text_line_edit);

	replace_label = memnew(Label);
	replace_label->set_text(TTRC("Replace:"));
	replace_label->hide();
	gc->add_child(replace_label);

	replace_text_line_edit = memnew(LineEdit);
	replace_text_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	replace_text_line_edit->set_accessibility_name(TTRC("Replace:"));
	replace_text_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FindInFilesDialog::_on_replace_text_submitted));
	replace_text_line_edit->hide();
	gc->add_child(replace_text_line_edit);

	gc->add_child(memnew(Control)); // Space to maintain the grid alignment.

	{
		HBoxContainer *hbc = memnew(HBoxContainer);

		whole_words_checkbox = memnew(CheckBox);
		whole_words_checkbox->set_text(TTRC("Whole Words"));
		hbc->add_child(whole_words_checkbox);

		match_case_checkbox = memnew(CheckBox);
		match_case_checkbox->set_text(TTRC("Match Case"));
		hbc->add_child(match_case_checkbox);

		gc->add_child(hbc);
	}

	Label *folder_label = memnew(Label);
	folder_label->set_text(TTRC("Folder:"));
	gc->add_child(folder_label);

	{
		HBoxContainer *hbc = memnew(HBoxContainer);

		Label *prefix_label = memnew(Label);
		prefix_label->set_text("res://");
		prefix_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
		hbc->add_child(prefix_label);

		folder_line_edit = memnew(LineEdit);
		folder_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		folder_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FindInFilesDialog::_on_search_text_submitted));
		folder_line_edit->set_accessibility_name(TTRC("Folder:"));
		hbc->add_child(folder_line_edit);

		Button *folder_button = memnew(Button);
		folder_button->set_accessibility_name(TTRC("Select Folder"));
		folder_button->set_text("...");
		folder_button->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesDialog::_on_folder_button_pressed));
		hbc->add_child(folder_button);

		folder_dialog = memnew(FileDialog);
		folder_dialog->set_file_mode(FileDialog::FILE_MODE_OPEN_DIR);
		folder_dialog->connect("dir_selected", callable_mp(this, &FindInFilesDialog::_on_folder_selected));
		add_child(folder_dialog);

		gc->add_child(hbc);
	}

	Label *includes_label = memnew(Label);
	includes_label->set_text(TTRC("Includes:"));
	includes_label->set_tooltip_text(TTRC("Include the files with the following expressions. Use \",\" to separate."));
	includes_label->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	gc->add_child(includes_label);

	includes_line_edit = memnew(LineEdit);
	includes_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	includes_line_edit->set_placeholder(TTRC("example: scripts,scenes/*/test.gd"));
	includes_line_edit->set_accessibility_name(TTRC("Includes:"));
	includes_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FindInFilesDialog::_on_search_text_submitted));
	gc->add_child(includes_line_edit);

	Label *excludes_label = memnew(Label);
	excludes_label->set_text(TTRC("Excludes:"));
	excludes_label->set_tooltip_text(TTRC("Exclude the files with the following expressions. Use \",\" to separate."));
	excludes_label->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	gc->add_child(excludes_label);

	excludes_line_edit = memnew(LineEdit);
	excludes_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	excludes_line_edit->set_placeholder(TTRC("example: res://addons,scenes/test/*.gd"));
	excludes_line_edit->set_accessibility_name(TTRC("Excludes:"));
	excludes_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FindInFilesDialog::_on_search_text_submitted));
	gc->add_child(excludes_line_edit);

	Label *filter_label = memnew(Label);
	filter_label->set_text(TTRC("Filters:"));
	filter_label->set_tooltip_text(TTRC("Include the files with the following extensions. Add or remove them in ProjectSettings."));
	filter_label->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	gc->add_child(filter_label);

	filters_container = memnew(HBoxContainer);
	gc->add_child(filters_container);

	find_button = add_button(TTRC("Find..."), false, "find");
	find_button->set_disabled(true);

	replace_button = add_button(TTRC("Replace..."), false, "replace");
	replace_button->set_disabled(true);

	Button *cancel_button = get_ok_button();
	cancel_button->set_text(TTRC("Cancel"));
}

//-----------------------------------------------------------------------------

void FindInFilesPanel::set_with_replace(bool p_with_replace) {
	with_replace = p_with_replace;
	replace_container->set_visible(p_with_replace);

	if (with_replace) {
		// Results show checkboxes on their left so they can be opted out.
		results_display->set_columns(2);
		results_display->set_column_expand(0, false);
		results_display->set_column_custom_minimum_width(0, 48 * EDSCALE);
	} else {
		// Results are single-cell items.
		results_display->set_column_expand(0, true);
		results_display->set_columns(1);
	}
}

void FindInFilesPanel::set_replace_text(const String &p_text) {
	replace_line_edit->set_text(p_text);
}

bool FindInFilesPanel::is_keep_results() const {
	return keep_results_button->is_pressed();
}

void FindInFilesPanel::set_search_labels_visibility(bool p_visible) {
	find_label->set_visible(p_visible);
	search_text_label->set_visible(p_visible);
	close_button->set_visible(p_visible);
}

void FindInFilesPanel::_clear() {
	file_items.clear();
	file_items_results_count.clear();
	result_items.clear();
	results_display->clear();
	results_display->create_item(); // Root
}

void FindInFilesPanel::start_search() {
	_clear();

	status_label->set_text(TTRC("Searching..."));
	search_text_label->set_text(finder->get_search_text());
	search_text_label->set_tooltip_text(finder->get_search_text());

	int label_min_width = search_text_label->get_minimum_size().x + search_text_label->get_character_bounds(0).size.x;
	search_text_label->set_custom_minimum_size(Size2(label_min_width, 0));

	set_process(true);
	progress_bar->set_visible(true);

	finder->start();

	_update_replace_buttons();
	refresh_button->hide();
	cancel_button->show();
}

void FindInFilesPanel::stop_search() {
	finder->stop();

	status_label->set_text("");
	_update_replace_buttons();
	progress_bar->set_visible(false);
	refresh_button->show();
	cancel_button->hide();
}

void FindInFilesPanel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			_on_theme_changed();
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED: {
			_update_matches_text();

			TreeItem *file_item = results_display->get_root()->get_first_child();
			while (file_item) {
				if (with_replace) {
					file_item->set_button_tooltip_text(0, file_item->get_button_by_id(0, FIND_BUTTON_REPLACE), TTR("Replace all matches in file"));
				}
				file_item->set_button_tooltip_text(0, file_item->get_button_by_id(0, FIND_BUTTON_REMOVE), TTR("Remove result"));

				TreeItem *result_item = file_item->get_first_child();
				while (result_item) {
					if (with_replace) {
						result_item->set_button_tooltip_text(1, file_item->get_button_by_id(0, FIND_BUTTON_REPLACE), TTR("Replace"));
						result_item->set_button_tooltip_text(1, file_item->get_button_by_id(0, FIND_BUTTON_REMOVE), TTR("Remove result"));
					} else {
						result_item->set_button_tooltip_text(0, file_item->get_button_by_id(0, FIND_BUTTON_REMOVE), TTR("Remove result"));
					}
					result_item = result_item->get_next();
				}

				file_item = file_item->get_next();
			}
		} break;
		case NOTIFICATION_PROCESS: {
			progress_bar->set_as_ratio(finder->get_progress());
		} break;
	}
}

void FindInFilesPanel::_on_result_found(const String &p_fpath, int p_line_number, int p_begin, int p_end, const String &p_text) {
	TreeItem *file_item;
	Ref<Texture2D> remove_texture = get_editor_theme_icon(SNAME("Close"));
	Ref<Texture2D> replace_texture = get_editor_theme_icon(SNAME("ReplaceText"));

	HashMap<String, TreeItem *>::Iterator E = file_items.find(p_fpath);
	if (!E) {
		file_item = results_display->create_item();
		file_item->set_text(0, p_fpath);
		file_item->set_metadata(0, p_fpath);

		if (with_replace) {
			file_item->add_button(0, replace_texture, FIND_BUTTON_REPLACE, false, TTR("Replace all matches in file"));
		}
		file_item->add_button(0, remove_texture, FIND_BUTTON_REMOVE, false, TTR("Remove result"));

		// The width of this column is restrained to checkboxes,
		// but that doesn't make sense for the parent items,
		// so we override their width so they can expand to full width.
		file_item->set_expand_right(0, true);

		file_items[p_fpath] = file_item;
		file_items_results_count[file_item] = 1;
	} else {
		file_item = E->value;
		file_items_results_count[file_item]++;
	}

	Color file_item_color = results_display->get_theme_color(SceneStringName(font_color)) * Color(1, 1, 1, 0.67);
	file_item->set_custom_color(0, file_item_color);
	file_item->set_selectable(0, false);

	int text_index = with_replace ? 1 : 0;

	TreeItem *item = results_display->create_item(file_item);

	// Do this first because it resets properties of the cell...
	item->set_cell_mode(text_index, TreeItem::CELL_MODE_CUSTOM);

	// Trim result item line.
	String trimmed_text = p_text.strip_edges(true, false);
	int chars_removed = p_text.size() - trimmed_text.size();
	String start = vformat("%3s: ", p_line_number);

	item->set_text(text_index, start + trimmed_text);
	item->set_custom_draw_callback(text_index, callable_mp(this, &FindInFilesPanel::_draw_result_text));

	Result r;
	r.line_number = p_line_number;
	r.begin = p_begin;
	r.end = p_end;
	r.begin_trimmed = p_begin - chars_removed + start.size() - 1;
	result_items[item] = r;

	if (with_replace) {
		item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		item->set_checked(0, true);
		item->set_editable(0, true);
		item->add_button(1, replace_texture, FIND_BUTTON_REPLACE, false, TTR("Replace"));
		item->add_button(1, remove_texture, FIND_BUTTON_REMOVE, false, TTR("Remove result"));
	} else {
		item->add_button(0, remove_texture, FIND_BUTTON_REMOVE, false, TTR("Remove result"));
	}
}

void FindInFilesPanel::_on_theme_changed() {
	results_display->add_theme_font_override(SceneStringName(font), get_theme_font(SNAME("source"), EditorStringName(EditorFonts)));
	results_display->add_theme_font_size_override(SceneStringName(font_size), get_theme_font_size(SNAME("source_size"), EditorStringName(EditorFonts)));

	Color file_item_color = results_display->get_theme_color(SceneStringName(font_color)) * Color(1, 1, 1, 0.67);
	Ref<Texture2D> remove_texture = get_editor_theme_icon(SNAME("Close"));
	Ref<Texture2D> replace_texture = get_editor_theme_icon(SNAME("ReplaceText"));

	TreeItem *file_item = results_display->get_root()->get_first_child();
	while (file_item) {
		file_item->set_custom_color(0, file_item_color);
		if (with_replace) {
			file_item->set_button(0, file_item->get_button_by_id(0, FIND_BUTTON_REPLACE), replace_texture);
		}
		file_item->set_button(0, file_item->get_button_by_id(0, FIND_BUTTON_REMOVE), remove_texture);

		TreeItem *result_item = file_item->get_first_child();
		while (result_item) {
			if (with_replace) {
				result_item->set_button(1, result_item->get_button_by_id(1, FIND_BUTTON_REPLACE), replace_texture);
				result_item->set_button(1, result_item->get_button_by_id(1, FIND_BUTTON_REMOVE), remove_texture);
			} else {
				result_item->set_button(0, result_item->get_button_by_id(0, FIND_BUTTON_REMOVE), remove_texture);
			}

			result_item = result_item->get_next();
		}

		file_item = file_item->get_next();
	}
}

void FindInFilesPanel::_draw_result_text(Object *p_item_obj, const Rect2 &p_rect) {
	TreeItem *item = Object::cast_to<TreeItem>(p_item_obj);
	if (!item) {
		return;
	}

	HashMap<TreeItem *, Result>::Iterator E = result_items.find(item);
	if (!E) {
		return;
	}
	Result r = E->value;
	String item_text = item->get_text(with_replace ? 1 : 0);
	Ref<Font> font = results_display->get_theme_font(SceneStringName(font));
	int font_size = results_display->get_theme_font_size(SceneStringName(font_size));

	Rect2 match_rect = p_rect;
	match_rect.position.x += font->get_string_size(item_text.left(r.begin_trimmed), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x - 1;
	match_rect.size.x = font->get_string_size(search_text_label->get_text(), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x + 1;
	match_rect.position.y += 1 * EDSCALE;
	match_rect.size.y -= 2 * EDSCALE;

	Color accent_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
	results_display->draw_rect(match_rect, accent_color * Color(1, 1, 1, 0.33), false, 2.0);
	results_display->draw_rect(match_rect, accent_color * Color(1, 1, 1, 0.17), true);

	// Text is drawn by Tree already.
}

void FindInFilesPanel::_on_item_edited() {
	TreeItem *item = results_display->get_selected();

	// Change opacity to half if checkbox is checked, otherwise full.
	Color use_color = results_display->get_theme_color(SceneStringName(font_color));
	if (!item->is_checked(0)) {
		use_color.a *= 0.5;
	}
	item->set_custom_color(1, use_color);
}

void FindInFilesPanel::_on_finished() {
	_update_matches_text();
	_update_replace_buttons();
	progress_bar->set_visible(false);
	refresh_button->show();
	cancel_button->hide();
}

void FindInFilesPanel::_on_refresh_button_clicked() {
	start_search();
}

void FindInFilesPanel::_on_cancel_button_clicked() {
	stop_search();
}

void FindInFilesPanel::_on_close_button_clicked() {
	emit_signal(SNAME("close_button_clicked"));
}

void FindInFilesPanel::_on_result_selected() {
	TreeItem *item = results_display->get_selected();
	HashMap<TreeItem *, Result>::Iterator E = result_items.find(item);

	if (!E) {
		return;
	}
	Result r = E->value;

	TreeItem *file_item = item->get_parent();
	String fpath = file_item->get_metadata(0);

	emit_signal(SNAME("result_selected"), fpath, r.line_number, r.begin, r.end);
}

void FindInFilesPanel::_on_replace_text_changed(const String &p_text) {
	_update_replace_buttons();
}

void FindInFilesPanel::_on_replace_all_clicked() {
	String replace_text = _get_replace_text();

	for (KeyValue<String, TreeItem *> &E : file_items) {
		TreeItem *file_item = E.value;
		String fpath = file_item->get_metadata(0);

		Vector<Result> locations;
		for (TreeItem *item = file_item->get_first_child(); item; item = item->get_next()) {
			if (!item->is_checked(0)) {
				continue;
			}

			HashMap<TreeItem *, Result>::Iterator F = result_items.find(item);
			ERR_FAIL_COND(!F);
			locations.push_back(F->value);
		}

		if (locations.size() != 0) {
			// Results are sorted by file, so we can batch replaces.
			_apply_replaces_in_file(fpath, locations, replace_text);
		}
	}

	// Hide replace bar so we can't trigger the action twice without doing a new search.
	replace_container->hide();

	emit_signal(SNAME("files_modified"));
}

void FindInFilesPanel::_on_button_clicked(TreeItem *p_item, int p_column, int p_id, int p_mouse_button_index) {
	const String file_path = p_item->get_metadata(0);

	if (p_id == FIND_BUTTON_REPLACE) {
		const String replace_text = _get_replace_text();
		Vector<Result> locations;
		if (file_items.has(file_path)) {
			for (TreeItem *item = p_item->get_first_child(); item; item = item->get_next()) {
				HashMap<TreeItem *, Result>::Iterator F = result_items.find(item);
				ERR_FAIL_COND(!F);
				locations.push_back(F->value);
			}
			_apply_replaces_in_file(file_path, locations, replace_text);
		} else {
			locations.push_back(result_items.find(p_item)->value);
			const String path = p_item->get_parent()->get_metadata(0);
			_apply_replaces_in_file(path, locations, replace_text);
		}
		emit_signal(SNAME("files_modified"));
	}

	result_items.erase(p_item);
	if (file_items_results_count.has(p_item)) {
		int match_count = p_item->get_child_count();

		for (int i = 0; i < match_count; i++) {
			TreeItem *child_item = p_item->get_child(i);
			result_items.erase(child_item);
		}

		p_item->clear_children();
		file_items.erase(file_path);
		file_items_results_count.erase(p_item);
	}

	TreeItem *item_parent = p_item->get_parent();
	if (item_parent) {
		if (file_items_results_count.has(item_parent)) {
			file_items_results_count[item_parent]--;
		}
		if (item_parent->get_child_count() < 2 && item_parent != results_display->get_root()) {
			file_items.erase(item_parent->get_metadata(0));
			get_tree()->queue_delete(item_parent);
		}
	}
	get_tree()->queue_delete(p_item);
	_update_matches_text();
}

// Same as get_line, but preserves line ending characters.
class ConservativeGetLine {
public:
	String get_line(Ref<FileAccess> p_file) {
		_line_buffer.clear();

		char32_t c = p_file->get_8();

		while (!p_file->eof_reached()) {
			if (c == '\n') {
				_line_buffer.push_back(c);
				_line_buffer.push_back(0);
				return String::utf8(_line_buffer.ptr());

			} else if (c == '\0') {
				_line_buffer.push_back(c);
				return String::utf8(_line_buffer.ptr());

			} else if (c != '\r') {
				_line_buffer.push_back(c);
			}

			c = p_file->get_8();
		}

		_line_buffer.push_back(0);
		return String::utf8(_line_buffer.ptr());
	}

private:
	Vector<char> _line_buffer;
};

void FindInFilesPanel::_apply_replaces_in_file(const String &p_fpath, const Vector<Result> &p_locations, const String &p_new_text) {
	// If the file is already open, I assume the editor will reload it.
	// If there are unsaved changes, the user will be asked on focus,
	// however that means either losing changes or losing replaces.

	Ref<FileAccess> f = FileAccess::open(p_fpath, FileAccess::READ);
	ERR_FAIL_COND_MSG(f.is_null(), "Cannot open file from path '" + p_fpath + "'.");

	String buffer;
	int current_line = 1;

	ConservativeGetLine conservative;

	String line = conservative.get_line(f);
	String search_text = finder->get_search_text();

	int offset = 0;

	for (int i = 0; i < p_locations.size(); ++i) {
		int repl_line_number = p_locations[i].line_number;

		while (current_line < repl_line_number) {
			buffer += line;
			line = conservative.get_line(f);
			++current_line;
			offset = 0;
		}

		int repl_begin = p_locations[i].begin + offset;
		int repl_end = p_locations[i].end + offset;

		int _;
		if (!find_next(line, search_text, repl_begin, finder->is_match_case(), finder->is_whole_words(), _, _)) {
			// Make sure the replace is still valid in case the file was tampered with.
			print_verbose(vformat(R"(Occurrence no longer matches, replace will be ignored in "%s": line %d, col %d.)", p_fpath, repl_line_number, repl_begin));
			continue;
		}

		line = line.left(repl_begin) + p_new_text + line.substr(repl_end);
		// Keep an offset in case there are successive replaces in the same line.
		offset += p_new_text.length() - (repl_end - repl_begin);
	}

	buffer += line;

	while (!f->eof_reached()) {
		buffer += conservative.get_line(f);
	}

	// Now the modified contents are in the buffer, rewrite the file with our changes.

	Error err = f->reopen(p_fpath, FileAccess::WRITE);
	ERR_FAIL_COND_MSG(err != OK, "Cannot create file in path '" + p_fpath + "'.");

	f->store_string(buffer);
}

String FindInFilesPanel::_get_replace_text() {
	return replace_line_edit->get_text();
}

void FindInFilesPanel::_update_replace_buttons() {
	bool disabled = finder->is_searching();

	replace_all_button->set_disabled(disabled);
}

void FindInFilesPanel::_update_matches_text() {
	String results_text;
	int result_count = result_items.size();
	int file_count = file_items.size();

	if (result_count == 1 && file_count == 1) {
		results_text = vformat(TTR("%d match in %d file"), result_count, file_count);
	} else if (result_count != 1 && file_count == 1) {
		results_text = vformat(TTR("%d matches in %d file"), result_count, file_count);
	} else {
		results_text = vformat(TTR("%d matches in %d files"), result_count, file_count);
	}

	status_label->set_text(results_text);

	TreeItem *file_item = results_display->get_root()->get_first_child();
	while (file_item) {
		int file_matches_count = file_items_results_count[file_item];
		file_item->set_text(0, (String)file_item->get_metadata(0) + " (" + vformat(TTRN("%d match", "%d matches", file_matches_count), file_matches_count) + ")");
		file_item = file_item->get_next();
	}
}

void FindInFilesPanel::_bind_methods() {
	ADD_SIGNAL(MethodInfo("result_selected",
			PropertyInfo(Variant::STRING, "path"),
			PropertyInfo(Variant::INT, "line_number"),
			PropertyInfo(Variant::INT, "begin"),
			PropertyInfo(Variant::INT, "end")));

	ADD_SIGNAL(MethodInfo("files_modified"));

	ADD_SIGNAL(MethodInfo("close_button_clicked"));
}

FindInFilesPanel::FindInFilesPanel() {
	finder = memnew(FindInFilesSearch);
	finder->connect("result_found", callable_mp(this, &FindInFilesPanel::_on_result_found));
	finder->connect(SceneStringName(finished), callable_mp(this, &FindInFilesPanel::_on_finished));
	add_child(finder);

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_anchor_and_offset(SIDE_LEFT, ANCHOR_BEGIN, 0);
	vbc->set_anchor_and_offset(SIDE_TOP, ANCHOR_BEGIN, 0);
	vbc->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0);
	vbc->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0);
	add_child(vbc);

	{
		HBoxContainer *hbc = memnew(HBoxContainer);
		hbc->set_alignment(BoxContainer::ALIGNMENT_END);

		find_label = memnew(Label);
		find_label->set_text(TTRC("Find:"));
		hbc->add_child(find_label);

		search_text_label = memnew(Label);
		search_text_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
		search_text_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		search_text_label->set_focus_mode(FOCUS_ACCESSIBILITY);
		search_text_label->set_mouse_filter(Control::MOUSE_FILTER_PASS);
		search_text_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
		hbc->add_child(search_text_label);

		progress_bar = memnew(ProgressBar);
		progress_bar->set_h_size_flags(SIZE_EXPAND_FILL);
		progress_bar->set_v_size_flags(SIZE_SHRINK_CENTER);
		progress_bar->set_stretch_ratio(2.0);
		progress_bar->set_visible(false);
		hbc->add_child(progress_bar);

		status_label = memnew(Label);
		status_label->set_focus_mode(FOCUS_ACCESSIBILITY);
		hbc->add_child(status_label);

		keep_results_button = memnew(CheckButton);
		keep_results_button->set_text(TTRC("Keep Results"));
		keep_results_button->set_tooltip_text(TTRC("Keep these results and show subsequent results in a new window"));
		keep_results_button->set_pressed(false);
		hbc->add_child(keep_results_button);

		refresh_button = memnew(Button);
		refresh_button->set_text(TTRC("Refresh"));
		refresh_button->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesPanel::_on_refresh_button_clicked));
		refresh_button->hide();
		hbc->add_child(refresh_button);

		cancel_button = memnew(Button);
		cancel_button->set_text(TTRC("Cancel"));
		cancel_button->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesPanel::_on_cancel_button_clicked));
		cancel_button->hide();
		hbc->add_child(cancel_button);

		close_button = memnew(Button);
		close_button->set_text(TTRC("Close"));
		close_button->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesPanel::_on_close_button_clicked));
		hbc->add_child(close_button);

		vbc->add_child(hbc);
	}

	results_display = memnew(Tree);
	results_display->set_accessibility_name(TTRC("Search Results"));
	results_display->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	results_display->set_v_size_flags(SIZE_EXPAND_FILL);
	results_display->connect(SceneStringName(item_selected), callable_mp(this, &FindInFilesPanel::_on_result_selected));
	results_display->connect("item_edited", callable_mp(this, &FindInFilesPanel::_on_item_edited));
	results_display->connect("button_clicked", callable_mp(this, &FindInFilesPanel::_on_button_clicked));
	results_display->set_hide_root(true);
	results_display->set_select_mode(Tree::SELECT_ROW);
	results_display->set_allow_rmb_select(true);
	results_display->set_allow_reselect(true);
	results_display->add_theme_constant_override("inner_item_margin_left", 0);
	results_display->add_theme_constant_override("inner_item_margin_right", 0);
	results_display->create_item(); // Root
	vbc->add_child(results_display);

	{
		replace_container = memnew(HBoxContainer);

		Label *replace_label = memnew(Label);
		replace_label->set_text(TTRC("Replace:"));
		replace_container->add_child(replace_label);

		replace_line_edit = memnew(LineEdit);
		replace_line_edit->set_accessibility_name(TTRC("Replace:"));
		replace_line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
		replace_line_edit->connect(SceneStringName(text_changed), callable_mp(this, &FindInFilesPanel::_on_replace_text_changed));
		replace_container->add_child(replace_line_edit);

		replace_all_button = memnew(Button);
		replace_all_button->set_text(TTRC("Replace all (no undo)"));
		replace_all_button->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesPanel::_on_replace_all_clicked));
		replace_container->add_child(replace_all_button);

		replace_container->hide();

		vbc->add_child(replace_container);
	}
}

//-----------------------------------------------------------------------------

FindInFilesPanel *FindInFilesContainer::_create_new_panel() {
	int index = tabs->get_current_tab();
	FindInFilesPanel *panel = memnew(FindInFilesPanel);
	tabs->add_child(panel);
	tabs->move_child(panel, index + 1); // New panel is added after the current activated panel.
	tabs->set_current_tab(index + 1);
	_update_bar_visibility();

	panel->connect("result_selected", callable_mp(this, &FindInFilesContainer::_result_selected));
	panel->connect("files_modified", callable_mp(this, &FindInFilesContainer::_files_modified));
	panel->connect("close_button_clicked", callable_mp(this, &FindInFilesContainer::_close_panel).bind(panel));
	return panel;
}

FindInFilesPanel *FindInFilesContainer::_get_current_panel() {
	return Object::cast_to<FindInFilesPanel>(tabs->get_current_tab_control());
}

FindInFilesPanel *FindInFilesContainer::get_panel_for_results(const String &p_label) {
	FindInFilesPanel *panel = nullptr;
	// Prefer the current panel.
	if (_get_current_panel() && !_get_current_panel()->is_keep_results()) {
		panel = _get_current_panel();
	} else {
		// Find the first panel which does not keep results.
		for (int i = 0; i < tabs->get_tab_count(); i++) {
			FindInFilesPanel *p = Object::cast_to<FindInFilesPanel>(tabs->get_tab_control(i));
			if (p && !p->is_keep_results()) {
				panel = p;
				tabs->set_current_tab(i);
				break;
			}
		}

		if (!panel) {
			panel = _create_new_panel();
		}
	}
	tabs->set_tab_title(tabs->get_current_tab(), p_label);
	return panel;
}

void FindInFilesContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			connect("closed", callable_mp(this, &FindInFilesContainer::_on_dock_closed));
		} break;
	}
}

void FindInFilesContainer::_on_theme_changed() {
	const Ref<StyleBox> bottom_panel_style = EditorNode::get_singleton()->get_editor_theme()->get_stylebox(SNAME("BottomPanel"), EditorStringName(EditorStyles));
	if (bottom_panel_style.is_valid()) {
		begin_bulk_theme_override();
		add_theme_constant_override("margin_top", -bottom_panel_style->get_margin(SIDE_TOP));
		add_theme_constant_override("margin_left", -bottom_panel_style->get_margin(SIDE_LEFT));
		add_theme_constant_override("margin_right", -bottom_panel_style->get_margin(SIDE_RIGHT));
		add_theme_constant_override("margin_bottom", -bottom_panel_style->get_margin(SIDE_BOTTOM));
		end_bulk_theme_override();
	}
}

void FindInFilesContainer::_result_selected(const String &p_fpath, int p_line_number, int p_begin, int p_end) {
	emit_signal(SNAME("result_selected"), p_fpath, p_line_number, p_begin, p_end);
}

void FindInFilesContainer::_files_modified() {
	emit_signal(SNAME("files_modified"));
}

void FindInFilesContainer::_close_panel(FindInFilesPanel *p_panel) {
	ERR_FAIL_COND_MSG(p_panel->get_parent() != tabs, "This panel is not a child!");
	tabs->remove_child(p_panel);
	p_panel->queue_free();
	_update_bar_visibility();
	if (tabs->get_tab_count() == 0) {
		close();
	}
}

void FindInFilesContainer::_on_dock_closed() {
	while (tabs->get_tab_count() > 0) {
		Control *tab = tabs->get_tab_control(0);
		tabs->remove_child(tab);
		tab->queue_free();
	}
	_update_bar_visibility();
}

void FindInFilesContainer::_on_tab_close_pressed(int p_tab) {
	FindInFilesPanel *panel = Object::cast_to<FindInFilesPanel>(tabs->get_tab_control(p_tab));
	if (panel) {
		_close_panel(panel);
	}
}

void FindInFilesContainer::_update_bar_visibility() {
	if (!update_bar) {
		return;
	}

	// If tab count <= 1, behaves like this is not a TabContainer and the bar is hidden.
	bool bar_visible = tabs->get_tab_count() > 1;
	tabs->set_tabs_visible(bar_visible);

	// Hide or show the search labels based on the visibility of the bar, as the search terms are displayed in the title of each tab.
	for (int i = 0; i < tabs->get_tab_count(); i++) {
		FindInFilesPanel *panel = Object::cast_to<FindInFilesPanel>(tabs->get_tab_control(i));
		if (panel) {
			panel->set_search_labels_visibility(!bar_visible);
		}
	}
}

void FindInFilesContainer::_bar_menu_option(int p_option) {
	int tab_index = tabs->get_current_tab();
	switch (p_option) {
		case PANEL_CLOSE: {
			_on_tab_close_pressed(tab_index);
		} break;
		case PANEL_CLOSE_OTHERS: {
			update_bar = false;
			FindInFilesPanel *panel = Object::cast_to<FindInFilesPanel>(tabs->get_tab_control(tab_index));
			for (int i = tabs->get_tab_count() - 1; i >= 0; i--) {
				FindInFilesPanel *p = Object::cast_to<FindInFilesPanel>(tabs->get_tab_control(i));
				if (p != panel) {
					_close_panel(p);
				}
			}
			update_bar = true;
			_update_bar_visibility();
		} break;
		case PANEL_CLOSE_RIGHT: {
			update_bar = false;
			for (int i = tabs->get_tab_count() - 1; i > tab_index; i--) {
				_on_tab_close_pressed(i);
			}
			update_bar = true;
			_update_bar_visibility();
		} break;
		case PANEL_CLOSE_ALL: {
			update_bar = false;
			for (int i = tabs->get_tab_count() - 1; i >= 0; i--) {
				_on_tab_close_pressed(i);
			}
			update_bar = true;
		} break;
	}
}

void FindInFilesContainer::_bar_input(const Ref<InputEvent> &p_input) {
	int tab_id = tabs->get_tab_bar()->get_hovered_tab();
	Ref<InputEventMouseButton> mb = p_input;

	if (tab_id >= 0 && mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::RIGHT) {
		tabs_context_menu->set_item_disabled(tabs_context_menu->get_item_index(PANEL_CLOSE_RIGHT), tab_id == tabs->get_tab_count() - 1);
		tabs_context_menu->set_position(tabs->get_tab_bar()->get_screen_position() + mb->get_position());
		tabs_context_menu->reset_size();
		tabs_context_menu->popup();
	}
}

void FindInFilesContainer::_bind_methods() {
	ADD_SIGNAL(MethodInfo("result_selected",
			PropertyInfo(Variant::STRING, "path"),
			PropertyInfo(Variant::INT, "line_number"),
			PropertyInfo(Variant::INT, "begin"),
			PropertyInfo(Variant::INT, "end")));

	ADD_SIGNAL(MethodInfo("files_modified"));
}

FindInFilesContainer::FindInFilesContainer() {
	set_name(TTRC("Search Results"));
	set_icon_name("Search");
	set_dock_shortcut(ED_SHORTCUT_AND_COMMAND("bottom_panels/toggle_search_results_bottom_panel", TTRC("Toggle Search Results Bottom Panel")));
	set_default_slot(DockConstants::DOCK_SLOT_BOTTOM);
	set_available_layouts(EditorDock::DOCK_LAYOUT_HORIZONTAL | EditorDock::DOCK_LAYOUT_FLOATING);
	set_global(false);
	set_transient(true);
	set_closable(true);
	set_custom_minimum_size(Size2(0, 200 * EDSCALE));

	tabs = memnew(TabContainer);
	tabs->set_tabs_visible(false);
	add_child(tabs);

	tabs->set_drag_to_rearrange_enabled(true);
	tabs->get_tab_bar()->set_select_with_rmb(true);
	tabs->get_tab_bar()->set_tab_close_display_policy(TabBar::CLOSE_BUTTON_SHOW_ACTIVE_ONLY);
	tabs->get_tab_bar()->connect("tab_close_pressed", callable_mp(this, &FindInFilesContainer::_on_tab_close_pressed));
	tabs->get_tab_bar()->connect(SceneStringName(gui_input), callable_mp(this, &FindInFilesContainer::_bar_input));

	tabs_context_menu = memnew(PopupMenu);
	add_child(tabs_context_menu);
	tabs_context_menu->add_item(TTRC("Close Tab"), PANEL_CLOSE);
	tabs_context_menu->add_item(TTRC("Close Other Tabs"), PANEL_CLOSE_OTHERS);
	tabs_context_menu->add_item(TTRC("Close Tabs to the Right"), PANEL_CLOSE_RIGHT);
	tabs_context_menu->add_item(TTRC("Close All Tabs"), PANEL_CLOSE_ALL);
	tabs_context_menu->connect(SceneStringName(id_pressed), callable_mp(this, &FindInFilesContainer::_bar_menu_option));

	EditorNode::get_singleton()->get_gui_base()->connect(SceneStringName(theme_changed), callable_mp(this, &FindInFilesContainer::_on_theme_changed));
}

void FindInFiles::_start_search(bool p_with_replace) {
	FindInFilesPanel *panel = container->get_panel_for_results((p_with_replace ? TTR("Replace:") : TTR("Find:")) + " " + dialog->get_search_text());
	FindInFilesSearch *search = panel->get_finder();

	search->set_search_text(dialog->get_search_text());
	search->set_match_case(dialog->is_match_case());
	search->set_whole_words(dialog->is_whole_words());
	search->set_folder(dialog->get_folder());
	search->set_filter(dialog->get_filter());
	search->set_includes(dialog->get_includes());
	search->set_excludes(dialog->get_excludes());

	panel->set_with_replace(p_with_replace);
	panel->set_replace_text(dialog->get_replace_text());
	panel->start_search();

	container->make_visible();
}

void FindInFiles::open_dialog(const String &p_initial_text, bool p_replace) {
	dialog->set_replace_mode(p_replace);
	dialog->set_search_text(p_initial_text);
	if (p_replace) {
		dialog->set_replace_text(String());
	}
	dialog->popup_centered();
}

void FindInFiles::_result_selected(const String &p_fpath, int p_line_number, int p_begin, int p_end) {
	emit_signal(SNAME("result_selected"), p_fpath, p_line_number, p_begin, p_end);
}

void FindInFiles::_files_modified() {
	emit_signal(SNAME("files_modified"));
}

void FindInFiles::_bind_methods() {
	ADD_SIGNAL(MethodInfo("result_selected",
			PropertyInfo(Variant::STRING, "path"),
			PropertyInfo(Variant::INT, "line_number"),
			PropertyInfo(Variant::INT, "begin"),
			PropertyInfo(Variant::INT, "end")));
	ADD_SIGNAL(MethodInfo("files_modified"));
}

FindInFiles::FindInFiles() {
	dialog = memnew(FindInFilesDialog);
	dialog->connect("find_requested", callable_mp(this, &FindInFiles::_start_search).bind(false));
	dialog->connect("replace_requested", callable_mp(this, &FindInFiles::_start_search).bind(true));
	EditorNode::get_singleton()->get_gui_base()->add_child(dialog);

	container = memnew(FindInFilesContainer);
	EditorDockManager::get_singleton()->add_dock(container);
	container->close();
	container->connect("result_selected", callable_mp(this, &FindInFiles::_result_selected));
	container->connect("files_modified", callable_mp(this, &FindInFiles::_files_modified));
}
