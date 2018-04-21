/*************************************************************************/
/*  find_in_files.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "find_in_files.h"
#include "core/os/dir_access.h"
#include "core/os/os.h"
#include "editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/progress_bar.h"

#define ROOT_PREFIX "res://"

const char *FindInFiles::SIGNAL_RESULT_FOUND = "result_found";
const char *FindInFiles::SIGNAL_FINISHED = "finished";

// TODO Would be nice in Vector and PoolVectors
template <typename T>
inline void pop_back(T &container) {
	container.resize(container.size() - 1);
}

// TODO Copied from TextEdit private, would be nice to extract it in a single place
static bool is_text_char(CharType c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

FindInFiles::FindInFiles() {
	_root_prefix = ROOT_PREFIX;
	_extension_filter.insert("gd");
	_extension_filter.insert("cs");
	_searching = false;
	_whole_words = true;
	_match_case = true;
}

void FindInFiles::set_search_text(String p_pattern) {
	_pattern = p_pattern;
}

void FindInFiles::set_whole_words(bool p_whole_word) {
	_whole_words = p_whole_word;
}

void FindInFiles::set_match_case(bool p_match_case) {
	_match_case = p_match_case;
}

void FindInFiles::set_folder(String folder) {
	_root_dir = folder;
}

void FindInFiles::set_filter(const Set<String> &exts) {
	_extension_filter = exts;
}

void FindInFiles::_notification(int p_notification) {
	if (p_notification == NOTIFICATION_PROCESS) {
		_process();
	}
}

void FindInFiles::start() {
	if (_pattern == "") {
		print_line("Nothing to search, pattern is empty");
		emit_signal(SIGNAL_FINISHED);
		return;
	}
	if (_extension_filter.size() == 0) {
		print_line("Nothing to search, filter matches no files");
		emit_signal(SIGNAL_FINISHED);
		return;
	}

	// Init search
	_current_dir = "";
	PoolStringArray init_folder;
	init_folder.append(_root_dir);
	_folders_stack.push_back(init_folder);

	_initial_files_count = 0;

	_searching = true;
	set_process(true);
}

void FindInFiles::stop() {
	_searching = false;
	_current_dir = "";
	set_process(false);
}

void FindInFiles::_process() {
	// This part can be moved to a thread if needed

	OS &os = *OS::get_singleton();
	float duration = 0.0;
	while (duration < 1.0 / 120.0) {
		float time_before = os.get_ticks_msec();
		_iterate();
		duration += (os.get_ticks_msec() - time_before);
	}
}

void FindInFiles::_iterate() {

	if (_folders_stack.size() != 0) {

		// Scan folders first so we can build a list of files and have progress info later

		PoolStringArray &folders_to_scan = _folders_stack[_folders_stack.size() - 1];

		if (folders_to_scan.size() != 0) {
			// Scan one folder below

			String folder_name = folders_to_scan[folders_to_scan.size() - 1];
			pop_back(folders_to_scan);

			_current_dir = _current_dir.plus_file(folder_name);

			PoolStringArray sub_dirs;
			_scan_dir(_root_prefix + _current_dir, sub_dirs);

			_folders_stack.push_back(sub_dirs);

		} else {
			// Go back one level

			pop_back(_folders_stack);
			_current_dir = _current_dir.get_base_dir();

			if (_folders_stack.size() == 0) {
				// All folders scanned
				_initial_files_count = _files_to_scan.size();
			}
		}

	} else if (_files_to_scan.size() != 0) {

		// Then scan files

		String fpath = _files_to_scan[_files_to_scan.size() - 1];
		pop_back(_files_to_scan);
		_scan_file(fpath);

	} else {
		print_line("Search complete");
		set_process(false);
		_current_dir = "";
		_searching = false;
		emit_signal(SIGNAL_FINISHED);
	}
}

float FindInFiles::get_progress() const {
	if (_initial_files_count != 0) {
		return static_cast<float>(_initial_files_count - _files_to_scan.size()) / static_cast<float>(_initial_files_count);
	}
	return 0;
}

void FindInFiles::_scan_dir(String path, PoolStringArray &out_folders) {

	DirAccess *dir = DirAccess::open(path);
	if (dir == NULL) {
		print_line("Cannot open directory! " + path);
		return;
	}

	dir->list_dir_begin();

	for (int i = 0; i < 1000; ++i) {
		String file = dir->get_next();

		if (file == "")
			break;

		// Ignore special dirs and hidden dirs (such as .git and .import)
		if (file == "." || file == ".." || file.begins_with("."))
			continue;

		if (dir->current_is_dir())
			out_folders.append(file);

		else {
			String file_ext = file.get_extension();
			if (_extension_filter.has(file_ext)) {
				_files_to_scan.push_back(path.plus_file(file));
			}
		}
	}
}

void FindInFiles::_scan_file(String fpath) {

	FileAccess *f = FileAccess::open(fpath, FileAccess::READ);
	if (f == NULL) {
		print_line(String("Cannot open file ") + fpath);
		return;
	}

	int line_number = 0;

	while (!f->eof_reached()) {

		// line number starts at 1
		++line_number;

		int begin = 0;
		int end = 0;

		String line = f->get_line();

		// Find all occurrences in the current line
		while (true) {
			begin = _match_case ? line.find(_pattern, end) : line.findn(_pattern, end);

			if (begin == -1)
				break;

			end = begin + _pattern.length();

			if (_whole_words) {
				if (begin > 0 && is_text_char(line[begin - 1])) {
					continue;
				}
				if (end < line.size() && is_text_char(line[end])) {
					continue;
				}
			}

			emit_signal(SIGNAL_RESULT_FOUND, fpath, line_number, begin, end, line);
		}
	}

	f->close();
}

void FindInFiles::_bind_methods() {

	ADD_SIGNAL(MethodInfo(SIGNAL_RESULT_FOUND,
			PropertyInfo(Variant::STRING, "path"),
			PropertyInfo(Variant::INT, "line_number"),
			PropertyInfo(Variant::INT, "begin"),
			PropertyInfo(Variant::INT, "end"),
			PropertyInfo(Variant::STRING, "text")));

	ADD_SIGNAL(MethodInfo(SIGNAL_FINISHED));
}

//-----------------------------------------------------------------------------
const char *FindInFilesDialog::SIGNAL_FIND_REQUESTED = "find_requested";
const char *FindInFilesDialog::SIGNAL_REPLACE_REQUESTED = "replace_requested";

FindInFilesDialog::FindInFilesDialog() {

	set_custom_minimum_size(Size2(400, 190));
	set_resizable(true);
	set_title(TTR("Find in files"));

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, -8 * EDSCALE);
	vbc->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, -8 * EDSCALE);
	add_child(vbc);

	GridContainer *gc = memnew(GridContainer);
	gc->set_columns(2);
	vbc->add_child(gc);

	Label *find_label = memnew(Label);
	find_label->set_text(TTR("Find: "));
	gc->add_child(find_label);

	_search_text_line_edit = memnew(LineEdit);
	_search_text_line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	_search_text_line_edit->connect("text_changed", this, "_on_search_text_modified");
	_search_text_line_edit->connect("text_entered", this, "_on_search_text_entered");
	gc->add_child(_search_text_line_edit);

	{
		Control *placeholder = memnew(Control);
		gc->add_child(placeholder);
	}

	{
		HBoxContainer *hbc = memnew(HBoxContainer);

		_whole_words_checkbox = memnew(CheckBox);
		_whole_words_checkbox->set_text(TTR("Whole words"));
		_whole_words_checkbox->set_pressed(true);
		hbc->add_child(_whole_words_checkbox);

		_match_case_checkbox = memnew(CheckBox);
		_match_case_checkbox->set_text(TTR("Match case"));
		_match_case_checkbox->set_pressed(true);
		hbc->add_child(_match_case_checkbox);

		gc->add_child(hbc);
	}

	Label *folder_label = memnew(Label);
	folder_label->set_text(TTR("Folder: "));
	gc->add_child(folder_label);

	{
		HBoxContainer *hbc = memnew(HBoxContainer);

		Label *prefix_label = memnew(Label);
		prefix_label->set_text(ROOT_PREFIX);
		hbc->add_child(prefix_label);

		_folder_line_edit = memnew(LineEdit);
		_folder_line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(_folder_line_edit);

		Button *folder_button = memnew(Button);
		folder_button->set_text("...");
		folder_button->connect("pressed", this, "_on_folder_button_pressed");
		hbc->add_child(folder_button);

		_folder_dialog = memnew(FileDialog);
		_folder_dialog->set_mode(FileDialog::MODE_OPEN_DIR);
		_folder_dialog->connect("dir_selected", this, "_on_folder_selected");
		add_child(_folder_dialog);

		gc->add_child(hbc);
	}

	Label *filter_label = memnew(Label);
	filter_label->set_text(TTR("Filter: "));
	gc->add_child(filter_label);

	{
		HBoxContainer *hbc = memnew(HBoxContainer);

		Vector<String> exts;
		exts.push_back("gd");
		exts.push_back("cs");

		for (int i = 0; i < exts.size(); ++i) {
			CheckBox *cb = memnew(CheckBox);
			cb->set_text(exts[i]);
			cb->set_pressed(true);
			hbc->add_child(cb);
			_filters.push_back(cb);
		}

		gc->add_child(hbc);
	}

	{
		Control *placeholder = memnew(Control);
		placeholder->set_custom_minimum_size(Size2(0, EDSCALE * 16));
		vbc->add_child(placeholder);
	}

	{
		HBoxContainer *hbc = memnew(HBoxContainer);
		hbc->set_alignment(HBoxContainer::ALIGN_CENTER);

		_find_button = memnew(Button);
		_find_button->set_text(TTR("Find..."));
		_find_button->connect("pressed", this, "_on_find_button_pressed");
		_find_button->set_disabled(true);
		hbc->add_child(_find_button);

		{
			Control *placeholder = memnew(Control);
			placeholder->set_custom_minimum_size(Size2(EDSCALE * 16, 0));
			hbc->add_child(placeholder);
		}

		_replace_button = memnew(Button);
		_replace_button->set_text(TTR("Replace..."));
		_replace_button->connect("pressed", this, "_on_replace_button_pressed");
		_replace_button->set_disabled(true);
		hbc->add_child(_replace_button);

		{
			Control *placeholder = memnew(Control);
			placeholder->set_custom_minimum_size(Size2(EDSCALE * 16, 0));
			hbc->add_child(placeholder);
		}

		Button *cancel_button = memnew(Button);
		cancel_button->set_text(TTR("Cancel"));
		cancel_button->connect("pressed", this, "hide");
		hbc->add_child(cancel_button);

		vbc->add_child(hbc);
	}
}

void FindInFilesDialog::set_search_text(String text) {
	_search_text_line_edit->set_text(text);
}

String FindInFilesDialog::get_search_text() const {
	String text = _search_text_line_edit->get_text();
	return text.strip_edges();
}

bool FindInFilesDialog::is_match_case() const {
	return _match_case_checkbox->is_pressed();
}

bool FindInFilesDialog::is_whole_words() const {
	return _whole_words_checkbox->is_pressed();
}

String FindInFilesDialog::get_folder() const {
	String text = _folder_line_edit->get_text();
	return text.strip_edges();
}

Set<String> FindInFilesDialog::get_filter() const {
	Set<String> filters;
	for (int i = 0; i < _filters.size(); ++i) {
		CheckBox *cb = _filters[i];
		if (cb->is_pressed()) {
			filters.insert(_filters[i]->get_text());
		}
	}
	return filters;
}

void FindInFilesDialog::_notification(int p_what) {
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (is_visible()) {
			// Doesn't work more than once if not deferred...
			_search_text_line_edit->call_deferred("grab_focus");
			_search_text_line_edit->select_all();
		}
	}
}

void FindInFilesDialog::_on_folder_button_pressed() {
	_folder_dialog->popup_centered_ratio();
}

void FindInFilesDialog::_on_find_button_pressed() {
	emit_signal(SIGNAL_FIND_REQUESTED);
	hide();
}

void FindInFilesDialog::_on_replace_button_pressed() {
	emit_signal(SIGNAL_REPLACE_REQUESTED);
	hide();
}

void FindInFilesDialog::_on_search_text_modified(String text) {

	ERR_FAIL_COND(!_find_button);
	ERR_FAIL_COND(!_replace_button);

	_find_button->set_disabled(get_search_text().empty());
	_replace_button->set_disabled(get_search_text().empty());
}

void FindInFilesDialog::_on_search_text_entered(String text) {
	// This allows to trigger a global search without leaving the keyboard
	if (!_find_button->is_disabled())
		_on_find_button_pressed();
}

void FindInFilesDialog::_on_folder_selected(String path) {
	int i = path.find("://");
	if (i != -1)
		path = path.right(i + 3);
	_folder_line_edit->set_text(path);
}

void FindInFilesDialog::_bind_methods() {

	ClassDB::bind_method("_on_folder_button_pressed", &FindInFilesDialog::_on_folder_button_pressed);
	ClassDB::bind_method("_on_find_button_pressed", &FindInFilesDialog::_on_find_button_pressed);
	ClassDB::bind_method("_on_replace_button_pressed", &FindInFilesDialog::_on_replace_button_pressed);
	ClassDB::bind_method("_on_folder_selected", &FindInFilesDialog::_on_folder_selected);
	ClassDB::bind_method("_on_search_text_modified", &FindInFilesDialog::_on_search_text_modified);
	ClassDB::bind_method("_on_search_text_entered", &FindInFilesDialog::_on_search_text_entered);

	ADD_SIGNAL(MethodInfo(SIGNAL_FIND_REQUESTED));
	ADD_SIGNAL(MethodInfo(SIGNAL_REPLACE_REQUESTED));
}

//-----------------------------------------------------------------------------
const char *FindInFilesPanel::SIGNAL_RESULT_SELECTED = "result_selected";
const char *FindInFilesPanel::SIGNAL_FILES_MODIFIED = "files_modified";

FindInFilesPanel::FindInFilesPanel() {

	_finder = memnew(FindInFiles);
	_finder->connect(FindInFiles::SIGNAL_RESULT_FOUND, this, "_on_result_found");
	_finder->connect(FindInFiles::SIGNAL_FINISHED, this, "_on_finished");
	add_child(_finder);

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_BEGIN, 0);
	vbc->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 0);
	vbc->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
	vbc->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);
	add_child(vbc);

	{
		HBoxContainer *hbc = memnew(HBoxContainer);

		Label *find_label = memnew(Label);
		find_label->set_text(TTR("Find: "));
		hbc->add_child(find_label);

		_search_text_label = memnew(Label);
		_search_text_label->add_font_override("font", get_font("source", "EditorFonts"));
		hbc->add_child(_search_text_label);

		_progress_bar = memnew(ProgressBar);
		_progress_bar->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(_progress_bar);
		set_progress_visible(false);

		_status_label = memnew(Label);
		hbc->add_child(_status_label);

		_cancel_button = memnew(Button);
		_cancel_button->set_text(TTR("Cancel"));
		_cancel_button->connect("pressed", this, "_on_cancel_button_clicked");
		_cancel_button->set_disabled(true);
		hbc->add_child(_cancel_button);

		vbc->add_child(hbc);
	}

	// In the future, this should be replaced by a more specific list container,
	// which can highlight text regions and change opacity for enabled/disabled states
	_results_display = memnew(ItemList);
	_results_display->add_font_override("font", get_font("source", "EditorFonts"));
	_results_display->set_v_size_flags(SIZE_EXPAND_FILL);
	_results_display->connect("item_selected", this, "_on_result_selected");
	vbc->add_child(_results_display);

	{
		_replace_container = memnew(HBoxContainer);

		Label *replace_label = memnew(Label);
		replace_label->set_text(TTR("Replace: "));
		_replace_container->add_child(replace_label);

		_replace_line_edit = memnew(LineEdit);
		_replace_line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
		_replace_line_edit->connect("text_changed", this, "_on_replace_text_changed");
		_replace_container->add_child(_replace_line_edit);

		_replace_all_button = memnew(Button);
		_replace_all_button->set_text(TTR("Replace all (no undo)"));
		_replace_all_button->connect("pressed", this, "_on_replace_all_clicked");
		_replace_container->add_child(_replace_all_button);

		_replace_container->hide();

		vbc->add_child(_replace_container);
	}
}

void FindInFilesPanel::set_with_replace(bool with_replace) {

	_replace_container->set_visible(with_replace);
}

void FindInFilesPanel::start_search() {

	_results_display->clear();
	_status_label->set_text(TTR("Searching..."));
	_search_text_label->set_text(_finder->get_search_text());

	set_process(true);
	set_progress_visible(true);

	_finder->start();

	update_replace_buttons();
	_cancel_button->set_disabled(false);
}

void FindInFilesPanel::stop_search() {

	_finder->stop();

	_status_label->set_text("");
	update_replace_buttons();
	set_progress_visible(false);
	_cancel_button->set_disabled(true);
}

void FindInFilesPanel::_notification(int p_what) {
	if (p_what == NOTIFICATION_PROCESS) {
		_progress_bar->set_as_ratio(_finder->get_progress());
	}
}

void FindInFilesPanel::_on_result_found(String fpath, int line_number, int begin, int end, String text) {

	int i = _results_display->get_item_count();
	_results_display->add_item(fpath + ": " + String::num(line_number) + ":        " + text.replace("\t", "    "));
	_results_display->set_item_metadata(i, varray(fpath, line_number, begin, end));
}

void FindInFilesPanel::_on_finished() {

	_status_label->set_text(TTR("Search complete"));
	update_replace_buttons();
	set_progress_visible(false);
	_cancel_button->set_disabled(true);
}

void FindInFilesPanel::_on_cancel_button_clicked() {
	stop_search();
}

void FindInFilesPanel::_on_result_selected(int i) {

	Array meta = _results_display->get_item_metadata(i);
	emit_signal(SIGNAL_RESULT_SELECTED, meta[0], meta[1], meta[2], meta[3]);
}

void FindInFilesPanel::_on_replace_text_changed(String text) {
	update_replace_buttons();
}

void FindInFilesPanel::_on_replace_all_clicked() {

	String replace_text = get_replace_text();
	ERR_FAIL_COND(replace_text.empty());

	String last_fpath;
	PoolIntArray locations;
	PoolStringArray modified_files;

	for (int i = 0; i < _results_display->get_item_count(); ++i) {

		Array meta = _results_display->get_item_metadata(i);

		String fpath = meta[0];

		// Results are sorted by file, so we can batch replaces
		if (fpath != last_fpath) {
			if (locations.size() != 0) {
				apply_replaces_in_file(last_fpath, locations, replace_text);
				modified_files.append(last_fpath);
				locations.resize(0);
			}
		}

		locations.append(meta[1]); // line_number
		locations.append(meta[2]); // begin
		locations.append(meta[3]); // end

		last_fpath = fpath;
	}

	if (locations.size() != 0) {
		apply_replaces_in_file(last_fpath, locations, replace_text);
		modified_files.append(last_fpath);
	}

	// Hide replace bar so we can't trigger the action twice without doing a new search
	set_with_replace(false);

	emit_signal(SIGNAL_FILES_MODIFIED, modified_files);
}

// Same as get_line, but preserves line ending characters
class ConservativeGetLine {
public:
	String get_line(FileAccess *f) {

		_line_buffer.clear();

		CharType c = f->get_8();

		while (!f->eof_reached()) {

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

			c = f->get_8();
		}

		_line_buffer.push_back(0);
		return String::utf8(_line_buffer.ptr());
	}

private:
	Vector<char> _line_buffer;
};

void FindInFilesPanel::apply_replaces_in_file(String fpath, PoolIntArray locations, String text) {

	ERR_FAIL_COND(locations.size() % 3 != 0);

	//print_line(String("Replacing {0} occurrences in {1}").format(varray(fpath, locations.size() / 3)));

	// If the file is already open, I assume the editor will reload it.
	// If there are unsaved changes, the user will be asked on focus,
	// however that means either loosing changes or loosing replaces.

	FileAccess *f = FileAccess::open(fpath, FileAccess::READ);
	ERR_FAIL_COND(f == NULL);

	String buffer;
	int current_line = 1;

	ConservativeGetLine conservative;

	String line = conservative.get_line(f);

	PoolIntArray::Read locations_read = locations.read();
	for (int i = 0; i < locations.size(); i += 3) {

		int repl_line_number = locations_read[i];
		int repl_begin = locations_read[i + 1];
		int repl_end = locations_read[i + 2];

		while (current_line < repl_line_number) {
			buffer += line;
			line = conservative.get_line(f);
			++current_line;
		}

		line = line.left(repl_begin) + text + line.right(repl_end);
	}

	buffer += line;

	while (!f->eof_reached()) {
		buffer += conservative.get_line(f);
	}

	// Now the modified contents are in the buffer, rewrite the file with our changes

	Error err = f->reopen(fpath, FileAccess::WRITE);
	ERR_FAIL_COND(err != OK);

	f->store_string(buffer);

	f->close();
}

String FindInFilesPanel::get_replace_text() {
	return _replace_line_edit->get_text().strip_edges();
}

void FindInFilesPanel::update_replace_buttons() {

	String text = get_replace_text();
	bool disabled = text.empty() || _finder->is_searching();

	_replace_all_button->set_disabled(disabled);
}

void FindInFilesPanel::set_progress_visible(bool visible) {
	_progress_bar->set_self_modulate(Color(1, 1, 1, visible ? 1 : 0));
}

void FindInFilesPanel::_bind_methods() {

	ClassDB::bind_method("_on_result_found", &FindInFilesPanel::_on_result_found);
	ClassDB::bind_method("_on_finished", &FindInFilesPanel::_on_finished);
	ClassDB::bind_method("_on_cancel_button_clicked", &FindInFilesPanel::_on_cancel_button_clicked);
	ClassDB::bind_method("_on_result_selected", &FindInFilesPanel::_on_result_selected);
	ClassDB::bind_method("_on_replace_text_changed", &FindInFilesPanel::_on_replace_text_changed);
	ClassDB::bind_method("_on_replace_all_clicked", &FindInFilesPanel::_on_replace_all_clicked);

	ADD_SIGNAL(MethodInfo(SIGNAL_RESULT_SELECTED,
			PropertyInfo(Variant::STRING, "path"),
			PropertyInfo(Variant::INT, "line_number"),
			PropertyInfo(Variant::INT, "begin"),
			PropertyInfo(Variant::INT, "end")));

	ADD_SIGNAL(MethodInfo(SIGNAL_FILES_MODIFIED, PropertyInfo(Variant::STRING, "paths")));
}
