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
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tree.h"
#include "scene/main/scene_tree.h"
#include "scene/main/timer.h"

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

void FindInFilesSearch::copy_from(const FindInFilesSearch *p_other) {
	if (!p_other) {
		return;
	}

	pattern = p_other->pattern;
	whole_words = p_other->whole_words;
	match_case = p_other->match_case;
	root_dir = p_other->root_dir;

	include_string = p_other->include_string;
	exclude_string = p_other->exclude_string;
	extension_filter = p_other->extension_filter;
}

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

void FindInFilesSearch::set_includes(const String &p_include_wildcards) {
	include_string = p_include_wildcards;
}

void FindInFilesSearch::set_excludes(const String &p_exclude_wildcards) {
	exclude_string = p_exclude_wildcards;
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
	_calculate_wildcard(include_string, &include_wildcards);
	_calculate_wildcard(exclude_string, &exclude_wildcards);
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

	files_to_scan.clear();
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
			PackedStringArray new_files_to_scan;
			_scan_dir("res://" + current_dir, sub_dirs, new_files_to_scan);

			folders_stack.push_back(sub_dirs);
			files_to_scan.append_array(new_files_to_scan);

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

void FindInFilesSearch::_calculate_wildcard(String &p_text, HashSet<String> *r_hash_set) {
	r_hash_set->clear();
	if (p_text.is_empty()) {
		return;
	}

	PackedStringArray wildcards = p_text.split(",", false);
	for (const String &wildcard : wildcards) {
		r_hash_set->insert(_validate_filter_wildcard(wildcard));
	}
}

String FindInFilesSearch::_validate_filter_wildcard(const String &p_expression) const {
	String ret = p_expression.replace_char('\\', '/');
	if (ret.begins_with("./")) {
		// Relative to the project root.
		ret = "res://" + ret.substr(2);
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

void FindInFilesSearchPanel::set_finder(FindInFilesSearch *p_finder, bool p_init) {
	finder = p_finder;
	search_text_line_edit->set_text(finder->get_search_text());
	match_case_checkbox->set_pressed(finder->get_match_case());
	whole_words_checkbox->set_pressed(finder->get_whole_words());
	includes_line_edit->set_text(finder->get_includes());
	excludes_line_edit->set_text(finder->get_excludes());
	for (int i = 0; i < filters_container->get_child_count(); ++i) {
		CheckBox *cb = static_cast<CheckBox *>(filters_container->get_child(i));
		cb->set_pressed(p_init);
		for (const String &extension : *finder->get_filter()) {
			if (cb->get_text() == extension) {
				cb->set_pressed(true);
			}
		}
	}
	// Update the finder in case we don't start a search before switching tabs to save initialization settings.
	_update_finder();
	// Don't search when setting finder to not overwrite old results in case they are desired.
	debounce_timer->stop();
}

void FindInFilesSearchPanel::set_search_text(const String &text) {
	if (text != "") {
		search_text_line_edit->set_text(text);
		_on_search_submitted();
	}
	callable_mp((Control *)search_text_line_edit, &Control::grab_focus).call_deferred(false);
	callable_mp((LineEdit *)search_text_line_edit, &LineEdit::select_all).call_deferred();
	callable_mp((LineEdit *)search_text_line_edit, &LineEdit::set_caret_column).bind(text.length()).call_deferred();
}

void FindInFilesSearchPanel::set_replace_text(const String &p_text) {
	replace_line_edit->set_text(p_text);
}

String FindInFilesSearchPanel::get_search_text() const {
	return search_text_line_edit->get_text();
}

bool FindInFilesSearchPanel::is_replace_pressed() const {
	return toggle_replace_button->is_pressed();
}

HashSet<String> FindInFilesSearchPanel::get_filter() const {
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

LineEdit *FindInFilesSearchPanel::get_replace_line_edit() const {
	return replace_line_edit;
}

void FindInFilesSearchPanel::_notification(int p_what) {
	switch (p_what) {
		// TODO: also update on `EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED` when setting is moved to editor settings
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
					cb->set_pressed(true);
					cb->connect(SceneStringName(toggled), callable_mp(this, &FindInFilesSearchPanel::_on_search_modified).unbind(1));
					cb->set_tooltip_text(TTRC("Include the files with this extension. Add or remove them in ProjectSettings (`search_in_file_extensions`)."));
					filters_container->add_child(cb);
				}
			}
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			match_case_checkbox->set_button_icon(get_editor_theme_icon(SNAME("MatchCase")));
			whole_words_checkbox->set_button_icon(get_editor_theme_icon(SNAME("SearchWholeWord")));
			replace_all_button->set_button_icon(get_editor_theme_icon(SNAME("SearchReplaceAll")));
			additional_options_button->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));
			_toggle_replace_pressed(toggle_replace_button->is_pressed());
		} break;
	}
}

void FindInFilesSearchPanel::_on_folder_button_pressed() {
	folder_dialog->popup_file_dialog();
}

void FindInFilesSearchPanel::_on_search_submitted() {
	debounce_timer->stop();
	_emit_find_requested();
}

void FindInFilesSearchPanel::_on_search_modified() {
	debounce_timer->start(0.5);
}

void FindInFilesSearchPanel::_update_finder() {
	finder->set_search_text(search_text_line_edit->get_text());
	finder->set_match_case(match_case_checkbox->is_pressed());
	finder->set_whole_words(whole_words_checkbox->is_pressed());
	finder->set_folder(folder_line_edit->get_text().strip_edges());
	finder->set_filter(get_filter());
	finder->set_includes(includes_line_edit->get_text());
	finder->set_excludes(excludes_line_edit->get_text());
}

void FindInFilesSearchPanel::_emit_find_requested() {
	_update_finder();
	emit_signal("find_requested");
}

void FindInFilesSearchPanel::_toggle_replace_pressed(bool toggle) {
	toggle_replace_button->set_tooltip_text(toggle ? TTRC("Hide Replace") : TTRC("Show Replace"));
	StringName rtl_compliant_arrow = is_layout_rtl() ? SNAME("GuiTreeArrowLeft") : SNAME("GuiTreeArrowRight");
	toggle_replace_button->set_button_icon(get_editor_theme_icon(toggle ? SNAME("GuiTreeArrowDown") : rtl_compliant_arrow));
	replace_hbox->set_visible(toggle);
}

void FindInFilesSearchPanel::_on_replace_all_clicked() {
	emit_signal("replace_all_requested");
}

void FindInFilesSearchPanel::_on_additional_options_toggled(bool p_toggled) {
	folder_hbc->set_visible(p_toggled);
	additional_options_vbc->set_visible(p_toggled);
}

void FindInFilesSearchPanel::_on_folder_selected(String p_path) {
	int i = p_path.find("://");
	if (i != -1) {
		p_path = p_path.substr(i + 3);
	}
	folder_line_edit->set_text(p_path);
}

void FindInFilesSearchPanel::_bind_methods() {
	ADD_SIGNAL(MethodInfo("find_requested"));
	ADD_SIGNAL(MethodInfo("replace_requested"));
	ADD_SIGNAL(MethodInfo("replace_all_requested"));
}

FindInFilesSearchPanel::FindInFilesSearchPanel() {
	set_custom_minimum_size(Size2(300 * EDSCALE, 50 * EDSCALE));
	set_name(TTRC("Find in Files"));
	set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);
	set_horizontal_scroll_mode(SCROLL_MODE_DISABLED);

	MarginContainer *mc = memnew(MarginContainer);
	int margin = 6 * EDSCALE;
	mc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	mc->add_theme_constant_override("margin_left", margin);
	mc->add_theme_constant_override("margin_right", margin);
	mc->add_theme_constant_override("margin_top", margin);
	mc->add_theme_constant_override("margin_bottom", margin);
	add_child(mc);

	VBoxContainer *vbc = memnew(VBoxContainer);
	mc->add_child(vbc);

	{
		HBoxContainer *hbc = memnew(HBoxContainer);
		hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		vbc->add_child(hbc);

		toggle_replace_button = memnew(Button);
		toggle_replace_button->set_theme_type_variation(SceneStringName(FlatButton));
		toggle_replace_button->set_toggle_mode(true);
		toggle_replace_button->set_pressed(false);
		toggle_replace_button->set_tooltip_text(TTRC("Replace Mode"));
		toggle_replace_button->connect(SceneStringName(toggled), callable_mp(this, &FindInFilesSearchPanel::_toggle_replace_pressed));
		hbc->add_child(toggle_replace_button);

		VBoxContainer *search_replace_vbox = memnew(VBoxContainer);
		search_replace_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hbc->add_child(search_replace_vbox);

		HBoxContainer *search_hbc = memnew(HBoxContainer);
		search_hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		search_replace_vbox->add_child(search_hbc);

		search_text_line_edit = memnew(LineEdit);
		search_text_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		search_text_line_edit->set_accessibility_name(TTRC("Search"));
		search_text_line_edit->set_placeholder(TTRC("Search"));
		search_text_line_edit->set_keep_editing_on_text_submit(true);
		search_text_line_edit->connect(SceneStringName(text_changed), callable_mp(this, &FindInFilesSearchPanel::_on_search_modified).unbind(1));
		search_text_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FindInFilesSearchPanel::_on_search_submitted).unbind(1));
		search_hbc->add_child(search_text_line_edit);

		whole_words_checkbox = memnew(Button);
		whole_words_checkbox->set_theme_type_variation(SceneStringName(FlatButton));
		whole_words_checkbox->set_toggle_mode(true);
		whole_words_checkbox->set_tooltip_text(TTRC("Whole Words"));
		whole_words_checkbox->connect(SceneStringName(toggled), callable_mp(this, &FindInFilesSearchPanel::_on_search_modified).unbind(1));
		search_hbc->add_child(whole_words_checkbox);

		match_case_checkbox = memnew(Button);
		match_case_checkbox->set_theme_type_variation(SceneStringName(FlatButton));
		match_case_checkbox->set_toggle_mode(true);
		match_case_checkbox->set_tooltip_text(TTRC("Match Case"));
		match_case_checkbox->connect(SceneStringName(toggled), callable_mp(this, &FindInFilesSearchPanel::_on_search_modified).unbind(1));
		search_hbc->add_child(match_case_checkbox);

		replace_hbox = memnew(HBoxContainer);
		replace_hbox->set_visible(toggle_replace_button->is_pressed());
		search_replace_vbox->add_child(replace_hbox);

		replace_line_edit = memnew(LineEdit);
		replace_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		replace_line_edit->set_accessibility_name(TTRC("Replace"));
		replace_line_edit->set_placeholder(TTRC("Replace"));
		replace_line_edit->set_keep_editing_on_text_submit(true);
		replace_hbox->add_child(replace_line_edit);

		replace_all_button = memnew(Button);
		replace_all_button->set_theme_type_variation(SceneStringName(FlatButton));
		replace_all_button->set_tooltip_text(TTRC("Replace all (no undo)"));
		replace_all_button->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesSearchPanel::_on_replace_all_clicked));
		replace_hbox->add_child(replace_all_button);
	}

	{
		HBoxContainer *hbc = memnew(HBoxContainer);
		vbc->add_child(hbc);

		additional_options_button = memnew(Button);
		additional_options_button->set_theme_type_variation(SceneStringName(FlatButton));
		additional_options_button->set_toggle_mode(true);
		additional_options_button->set_tooltip_text(TTRC("Show additional search options"));
		additional_options_button->connect(SceneStringName(toggled), callable_mp(this, &FindInFilesSearchPanel::_on_additional_options_toggled));
		hbc->add_child(additional_options_button);

		folder_hbc = memnew(HBoxContainer);
		folder_hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hbc->add_child(folder_hbc);

		Label *prefix_label = memnew(Label);
		prefix_label->set_text("res://");
		prefix_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
		folder_hbc->add_child(prefix_label);

		folder_line_edit = memnew(LineEdit);
		folder_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		folder_line_edit->connect(SceneStringName(text_changed), callable_mp(this, &FindInFilesSearchPanel::_on_search_modified).unbind(1));
		folder_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FindInFilesSearchPanel::_on_search_submitted).unbind(1));
		folder_line_edit->set_accessibility_name(TTRC("Folder"));
		folder_line_edit->set_placeholder(TTRC("Folder"));
		folder_line_edit->set_keep_editing_on_text_submit(true);
		folder_hbc->add_child(folder_line_edit);

		Button *folder_button = memnew(Button);
		folder_button->set_accessibility_name(TTRC("Select Folder"));
		folder_button->set_text("...");
		folder_button->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesSearchPanel::_on_folder_button_pressed));
		folder_hbc->add_child(folder_button);

		folder_dialog = memnew(EditorFileDialog);
		folder_dialog->set_file_mode(FileDialog::FILE_MODE_OPEN_DIR);
		folder_dialog->connect("dir_selected", callable_mp(this, &FindInFilesSearchPanel::_on_folder_selected));
		add_child(folder_dialog);
	}

	additional_options_vbc = memnew(VBoxContainer);
	additional_options_vbc->add_theme_constant_override("separation", 0);
	vbc->add_child(additional_options_vbc);

	Label *includes_label = memnew(Label);
	includes_label->set_text(TTRC("Includes"));
	includes_label->set_tooltip_text(TTRC("Include the files with the following expressions. Use \",\" to separate."));
	includes_label->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	additional_options_vbc->add_child(includes_label);

	includes_line_edit = memnew(LineEdit);
	includes_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	includes_line_edit->set_placeholder(TTRC("example: scripts,scenes/*/test.gd"));
	includes_line_edit->set_accessibility_name(TTRC("Includes"));
	includes_line_edit->set_keep_editing_on_text_submit(true);
	includes_line_edit->connect(SceneStringName(text_changed), callable_mp(this, &FindInFilesSearchPanel::_on_search_modified).unbind(1));
	includes_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FindInFilesSearchPanel::_on_search_submitted).unbind(1));
	additional_options_vbc->add_child(includes_line_edit);

	Label *excludes_label = memnew(Label);
	excludes_label->set_text(TTRC("Excludes"));
	excludes_label->set_tooltip_text(TTRC("Exclude the files with the following expressions. Use \",\" to separate."));
	excludes_label->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	additional_options_vbc->add_child(excludes_label);

	excludes_line_edit = memnew(LineEdit);
	excludes_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	excludes_line_edit->set_placeholder(TTRC("example: res://addons,scenes/test/*.gd"));
	excludes_line_edit->set_accessibility_name(TTRC("Excludes"));
	excludes_line_edit->set_keep_editing_on_text_submit(true);
	excludes_line_edit->connect(SceneStringName(text_changed), callable_mp(this, &FindInFilesSearchPanel::_on_search_modified).unbind(1));
	excludes_line_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FindInFilesSearchPanel::_on_search_submitted).unbind(1));
	additional_options_vbc->add_child(excludes_line_edit);

	filters_container = memnew(HBoxContainer);
	additional_options_vbc->add_child(filters_container);

	debounce_timer = memnew(Timer);
	debounce_timer->set_one_shot(true);
	debounce_timer->connect("timeout", callable_mp(this, &FindInFilesSearchPanel::_emit_find_requested));
	add_child(debounce_timer);

	_on_additional_options_toggled(additional_options_button->is_pressed());
}

//-----------------------------------------------------------------------------

void FindInFilesResultsPanel::set_with_replace(bool p_with_replace) {
	with_replace = p_with_replace;

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

void FindInFilesResultsPanel::set_replace_text(const String &p_text) {
	replace_text = p_text;
}

void FindInFilesResultsPanel::set_search_labels_visibility(bool p_visible) {
	find_label->set_visible(p_visible);
	search_text_label->set_visible(p_visible);
	progress_bar_placeholder->set_visible(!p_visible);
}

void FindInFilesResultsPanel::_clear() {
	file_items.clear();
	file_items_results_count.clear();
	result_items.clear();
	results_display->clear();
	results_display->create_item(); // Root
}

void FindInFilesResultsPanel::start_search() {
	_clear();

	status_label->set_text(TTRC("Searching..."));
	search_text_label->set_text(finder->get_search_text());
	search_text_label->set_tooltip_text(finder->get_search_text());

	int label_min_width = search_text_label->get_minimum_size().x + search_text_label->get_character_bounds(0).size.x;
	search_text_label->set_custom_minimum_size(Size2(label_min_width, 0));

	set_process(true);
	progress_bar->set_visible(true);

	finder->start();

	refresh_button->hide();
	cancel_button->show();
}

void FindInFilesResultsPanel::stop_search() {
	finder->stop();

	status_label->set_text("");
	progress_bar->set_visible(false);
	refresh_button->show();
	cancel_button->hide();
}

void FindInFilesResultsPanel::update_layout(EditorDock::DockLayout p_layout) {
	bool new_floating = (p_layout == EditorDock::DOCK_LAYOUT_FLOATING);
	if (floating == new_floating) {
		return;
	}
	floating = new_floating;

	if (floating) {
		results_mc->set_theme_type_variation("NoBorderHorizontalBottom");
		results_display->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_TOP);
	} else {
		results_mc->set_theme_type_variation("NoBorderHorizontal");
		results_display->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_BOTH);
	}
}

void FindInFilesResultsPanel::_notification(int p_what) {
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

void FindInFilesResultsPanel::_on_result_found(const String &p_fpath, int p_line_number, int p_begin, int p_end, const String &p_text) {
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
	item->set_custom_draw_callback(text_index, callable_mp(this, &FindInFilesResultsPanel::_draw_result_text));

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

void FindInFilesResultsPanel::_on_theme_changed() {
	new_tab_button->set_button_icon(get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
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

void FindInFilesResultsPanel::_draw_result_text(Object *p_item_obj, const Rect2 &p_rect) {
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
	match_rect.size.x = font->get_string_size(finder->get_search_text(), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x + 1;
	match_rect.position.y += 1 * EDSCALE;
	match_rect.size.y -= 2 * EDSCALE;

	Color accent_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
	results_display->draw_rect(match_rect, Color(accent_color, 0.33), false, 2.0);
	results_display->draw_rect(match_rect, Color(accent_color, 0.17), true);

	// Text is drawn by Tree already.
}

void FindInFilesResultsPanel::_on_item_edited() {
	TreeItem *item = results_display->get_selected();

	// Change opacity to half if checkbox is checked, otherwise full.
	Color use_color = results_display->get_theme_color(SceneStringName(font_color));
	if (!item->is_checked(0)) {
		use_color.a *= 0.5;
	}
	item->set_custom_color(1, use_color);
}

void FindInFilesResultsPanel::_on_finished() {
	_update_matches_text();
	progress_bar->set_visible(false);
	refresh_button->show();
	cancel_button->hide();
}

void FindInFilesResultsPanel::_on_refresh_button_clicked() {
	start_search();
}

void FindInFilesResultsPanel::_on_cancel_button_clicked() {
	stop_search();
}

void FindInFilesResultsPanel::_on_result_selected() {
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

void FindInFilesResultsPanel::replace_all() {
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

	emit_signal(SNAME("files_modified"));
}

void FindInFilesResultsPanel::_on_button_clicked(TreeItem *p_item, int p_column, int p_id, int p_mouse_button_index) {
	const String file_path = p_item->get_metadata(0);

	if (p_id == FIND_BUTTON_REPLACE) {
		const String rt = replace_text;
		Vector<Result> locations;
		if (file_items.has(file_path)) {
			for (TreeItem *item = p_item->get_first_child(); item; item = item->get_next()) {
				HashMap<TreeItem *, Result>::Iterator F = result_items.find(item);
				ERR_FAIL_COND(!F);
				locations.push_back(F->value);
			}
			_apply_replaces_in_file(file_path, locations, rt);
		} else {
			locations.push_back(result_items.find(p_item)->value);
			const String path = p_item->get_parent()->get_metadata(0);
			_apply_replaces_in_file(path, locations, rt);
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

void FindInFilesResultsPanel::_apply_replaces_in_file(const String &p_fpath, const Vector<Result> &p_locations, const String &p_new_text) {
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

void FindInFilesResultsPanel::_update_matches_text() {
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

void FindInFilesResultsPanel::_bind_methods() {
	ADD_SIGNAL(MethodInfo("result_selected",
			PropertyInfo(Variant::STRING, "path"),
			PropertyInfo(Variant::INT, "line_number"),
			PropertyInfo(Variant::INT, "begin"),
			PropertyInfo(Variant::INT, "end")));

	ADD_SIGNAL(MethodInfo("files_modified"));
}

FindInFilesResultsPanel::FindInFilesResultsPanel() {
	finder = memnew(FindInFilesSearch);
	finder->connect("result_found", callable_mp(this, &FindInFilesResultsPanel::_on_result_found));
	finder->connect(SceneStringName(finished), callable_mp(this, &FindInFilesResultsPanel::_on_finished));
	add_child(finder);

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	{
		HBoxContainer *hbc = memnew(HBoxContainer);
		hbc->set_alignment(BoxContainer::ALIGNMENT_END);

		new_tab_button = memnew(Button);
		new_tab_button->set_tooltip_text(TTRC("Create a new Search Tab"));
		hbc->add_child(new_tab_button);

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

		progress_bar_placeholder = memnew(Control);
		progress_bar_placeholder->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hbc->add_child(progress_bar_placeholder);

		status_label = memnew(Label);
		status_label->set_focus_mode(FOCUS_ACCESSIBILITY);
		hbc->add_child(status_label);

		refresh_button = memnew(Button);
		refresh_button->set_text(TTRC("Refresh"));
		refresh_button->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesResultsPanel::_on_refresh_button_clicked));
		refresh_button->hide();
		hbc->add_child(refresh_button);

		cancel_button = memnew(Button);
		cancel_button->set_text(TTRC("Cancel"));
		cancel_button->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesResultsPanel::_on_cancel_button_clicked));
		cancel_button->hide();
		hbc->add_child(cancel_button);

		vbc->add_child(hbc);
	}

	results_mc = memnew(MarginContainer);
	results_mc->set_theme_type_variation("NoBorderHorizontal");
	results_mc->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->add_child(results_mc);

	results_display = memnew(Tree);
	results_display->set_accessibility_name(TTRC("Find in Files"));
	results_display->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	results_display->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_BOTH);
	results_display->connect(SceneStringName(item_selected), callable_mp(this, &FindInFilesResultsPanel::_on_result_selected));
	results_display->connect("item_edited", callable_mp(this, &FindInFilesResultsPanel::_on_item_edited));
	results_display->connect("button_clicked", callable_mp(this, &FindInFilesResultsPanel::_on_button_clicked));
	results_display->set_hide_root(true);
	results_display->set_select_mode(Tree::SELECT_ROW);
	results_display->set_allow_rmb_select(true);
	results_display->set_allow_reselect(true);
	results_display->add_theme_constant_override("inner_item_margin_left", 0);
	results_display->add_theme_constant_override("inner_item_margin_right", 0);
	results_display->create_item(); // Root
	results_mc->add_child(results_display);
}

//-----------------------------------------------------------------------------

FindInFilesResultsPanel *FindInFilesContainer::_create_new_panel() {
	int index = tabs->get_current_tab();
	FindInFilesResultsPanel *old_panel = Object::cast_to<FindInFilesResultsPanel>(tabs->get_current_tab_control());

	FindInFilesResultsPanel *new_panel = memnew(FindInFilesResultsPanel);
	tabs->add_child(new_panel);
	tabs->move_child(new_panel, index + 1); // New panel is added after the current activated panel.
	tabs->set_current_tab(index + 1);
	_update_bar_visibility();
	_update_current_title();

	new_panel->connect("result_selected", callable_mp(this, &FindInFilesContainer::_result_selected));
	new_panel->connect("files_modified", callable_mp(this, &FindInFilesContainer::_files_modified));
	new_panel->get_new_tab_button()->connect(SceneStringName(pressed), callable_mp(this, &FindInFilesContainer::_create_new_panel));
	search_control->set_finder(new_panel->get_finder(), true);

	if (old_panel) {
		FindInFilesSearch *old_finder = old_panel->get_finder();
		FindInFilesSearch *new_finder = new_panel->get_finder();
		if (old_finder) {
			new_finder->copy_from(old_finder);
			search_control->set_search_text(old_finder->get_search_text());
		}
	}
	return new_panel;
}

void FindInFilesContainer::_update_current_title() {
	tabs->set_tab_title(tabs->get_current_tab(), vformat(TTR("Find: %s"), search_control->get_search_text()));
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
	hsplit->add_theme_style_override("split_bar_background", get_theme_stylebox(SceneStringName(panel), "ItemListSecondary"));
}

void FindInFilesContainer::_on_tab_changed() {
	FindInFilesResultsPanel *panel = Object::cast_to<FindInFilesResultsPanel>(tabs->get_current_tab_control());
	search_control->set_replace_text(panel->get_replace_text());
	search_control->set_finder(panel->get_finder(), false);
}

void FindInFilesContainer::_result_selected(const String &p_fpath, int p_line_number, int p_begin, int p_end) {
	emit_signal(SNAME("result_selected"), p_fpath, p_line_number, p_begin, p_end);
}

void FindInFilesContainer::_files_modified() {
	emit_signal(SNAME("files_modified"));
}

void FindInFilesContainer::_close_panel(FindInFilesResultsPanel *p_panel) {
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
	FindInFilesResultsPanel *panel = Object::cast_to<FindInFilesResultsPanel>(tabs->get_tab_control(p_tab));
	if (panel) {
		_close_panel(panel);
	}
}

void FindInFilesContainer::_set_replace_text(String p_text) {
	FindInFilesResultsPanel *panel = Object::cast_to<FindInFilesResultsPanel>(tabs->get_current_tab_control());
	panel->set_replace_text(p_text);
}

void FindInFilesContainer::_replace_all() {
	FindInFilesResultsPanel *panel = Object::cast_to<FindInFilesResultsPanel>(tabs->get_current_tab_control());
	panel->replace_all();
}

void FindInFilesContainer::_start_find_in_files() {
	if (search_control->get_search_text().is_empty()) {
		return;
	}
	FindInFilesResultsPanel *panel = Object::cast_to<FindInFilesResultsPanel>(tabs->get_current_tab_control());
	panel->set_with_replace(search_control->is_replace_pressed());
	_update_current_title();
	panel->start_search();
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
		FindInFilesResultsPanel *panel = Object::cast_to<FindInFilesResultsPanel>(tabs->get_tab_control(i));
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
			FindInFilesResultsPanel *panel = Object::cast_to<FindInFilesResultsPanel>(tabs->get_tab_control(tab_index));
			for (int i = tabs->get_tab_count() - 1; i >= 0; i--) {
				FindInFilesResultsPanel *p = Object::cast_to<FindInFilesResultsPanel>(tabs->get_tab_control(i));
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

void FindInFilesContainer::update_layout(EditorDock::DockLayout p_layout) {
	for (Node *node : tabs->iterate_children()) {
		FindInFilesResultsPanel *panel = Object::cast_to<FindInFilesResultsPanel>(node);
		if (panel) {
			panel->update_layout(p_layout);
		}
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
	set_name(TTRC("Find in Files"));
	set_icon_name("Search");
	set_dock_shortcut(ED_SHORTCUT_AND_COMMAND("bottom_panels/toggle_search_results_bottom_panel", TTRC("Toggle Find in Files Bottom Panel")));
	set_default_slot(EditorDock::DOCK_SLOT_BOTTOM);
	set_available_layouts(EditorDock::DOCK_LAYOUT_HORIZONTAL | EditorDock::DOCK_LAYOUT_FLOATING);
	set_global(false);
	set_transient(true);
	set_closable(true);
	set_custom_minimum_size(Size2(0, 200 * EDSCALE));

	hsplit = memnew(HSplitContainer);
	add_child(hsplit);

	search_control = memnew(FindInFilesSearchPanel);
	search_control->connect("find_requested", callable_mp(this, &FindInFilesContainer::_start_find_in_files));
	search_control->connect("replace_all_requested", callable_mp(this, &FindInFilesContainer::_replace_all));
	search_control->get_replace_line_edit()->connect(SceneStringName(text_changed), callable_mp(this, &FindInFilesContainer::_set_replace_text));
	hsplit->add_child(search_control);

	tabs = memnew(TabContainer);
	tabs->set_tabs_visible(false);
	hsplit->add_child(tabs);

	tabs->set_drag_to_rearrange_enabled(true);
	tabs->connect("tab_changed", callable_mp(this, &FindInFilesContainer::_on_tab_changed).unbind(1));
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
	_create_new_panel();
}

FindInFiles *FindInFiles::find_in_files = nullptr;

void FindInFiles::open_dock(const String &p_initial_text) {
	FindInFilesSearchPanel *search_control = container->get_search_control();
	search_control->set_search_text(p_initial_text);
	container->make_visible();
}

FindInFiles::FindInFiles() {
	find_in_files = this;
	container = memnew(FindInFilesContainer);
	EditorDockManager::get_singleton()->add_dock(container);
	container->close();
}
