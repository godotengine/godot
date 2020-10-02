/*************************************************************************/
/*  file_dialog.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "file_dialog.h"

#include "core/os/keyboard.h"
#include "core/print_string.h"
#include "scene/gui/label.h"

FileDialog::GetIconFunc FileDialog::get_icon_func = nullptr;
FileDialog::GetIconFunc FileDialog::get_large_icon_func = nullptr;

FileDialog::RegisterFunc FileDialog::register_func = nullptr;
FileDialog::RegisterFunc FileDialog::unregister_func = nullptr;

void FileDialog::popup_file_dialog() {
	popup_centered_clamped(Size2i(700, 500), 0.8f);
}

VBoxContainer *FileDialog::get_vbox() {
	return vbox;
}

void FileDialog::_theme_changed() {
	Color font_color = vbox->get_theme_color("font_color", "Button");
	Color font_color_hover = vbox->get_theme_color("font_color_hover", "Button");
	Color font_color_pressed = vbox->get_theme_color("font_color_pressed", "Button");

	dir_up->add_theme_color_override("icon_color_normal", font_color);
	dir_up->add_theme_color_override("icon_color_hover", font_color_hover);
	dir_up->add_theme_color_override("icon_color_pressed", font_color_pressed);

	refresh->add_theme_color_override("icon_color_normal", font_color);
	refresh->add_theme_color_override("icon_color_hover", font_color_hover);
	refresh->add_theme_color_override("icon_color_pressed", font_color_pressed);

	show_hidden->add_theme_color_override("icon_color_normal", font_color);
	show_hidden->add_theme_color_override("icon_color_hover", font_color_hover);
	show_hidden->add_theme_color_override("icon_color_pressed", font_color_pressed);
}

void FileDialog::_notification(int p_what) {
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (!is_visible()) {
			set_process_unhandled_input(false);
		}
	}
	if (p_what == NOTIFICATION_ENTER_TREE) {
		dir_up->set_icon(vbox->get_theme_icon("parent_folder", "FileDialog"));
		refresh->set_icon(vbox->get_theme_icon("reload", "FileDialog"));
		show_hidden->set_icon(vbox->get_theme_icon("toggle_hidden", "FileDialog"));
		_theme_changed();
	}
}

void FileDialog::_unhandled_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && has_focus()) {
		if (k->is_pressed()) {
			bool handled = true;

			switch (k->get_keycode()) {
				case KEY_H: {
					if (k->get_command()) {
						set_show_hidden_files(!show_hidden_files);
					} else {
						handled = false;
					}

				} break;
				case KEY_F5: {
					invalidate();
				} break;
				case KEY_BACKSPACE: {
					_dir_entered("..");
				} break;
				default: {
					handled = false;
				}
			}

			if (handled) {
				set_input_as_handled();
			}
		}
	}
}

void FileDialog::set_enable_multiple_selection(bool p_enable) {
	tree->set_select_mode(p_enable ? Tree::SELECT_MULTI : Tree::SELECT_SINGLE);
};

Vector<String> FileDialog::get_selected_files() const {
	Vector<String> list;

	TreeItem *item = tree->get_root();
	while ((item = tree->get_next_selected(item))) {
		list.push_back(dir_access->get_current_dir().plus_file(item->get_text(0)));
	};

	return list;
};

void FileDialog::update_dir() {
	dir->set_text(dir_access->get_current_dir(false));

	if (drives->is_visible()) {
		drives->select(dir_access->get_current_drive());
	}

	// Deselect any item, to make "Select Current Folder" button text by default.
	deselect_items();
}

void FileDialog::_dir_entered(String p_dir) {
	dir_access->change_dir(p_dir);
	file->set_text("");
	invalidate();
	update_dir();
}

void FileDialog::_file_entered(const String &p_file) {
	_action_pressed();
}

void FileDialog::_save_confirm_pressed() {
	String f = dir_access->get_current_dir().plus_file(file->get_text());
	emit_signal("file_selected", f);
	hide();
}

void FileDialog::_post_popup() {
	ConfirmationDialog::_post_popup();
	if (invalidated) {
		update_file_list();
		invalidated = false;
	}
	if (mode == FILE_MODE_SAVE_FILE) {
		file->grab_focus();
	} else {
		tree->grab_focus();
	}

	set_process_unhandled_input(true);

	// For open dir mode, deselect all items on file dialog open.
	if (mode == FILE_MODE_OPEN_DIR) {
		deselect_items();
		file_box->set_visible(false);
	} else {
		file_box->set_visible(true);
	}
}

void FileDialog::_action_pressed() {
	if (mode == FILE_MODE_OPEN_FILES) {
		TreeItem *ti = tree->get_next_selected(nullptr);
		String fbase = dir_access->get_current_dir();

		Vector<String> files;
		while (ti) {
			files.push_back(fbase.plus_file(ti->get_text(0)));
			ti = tree->get_next_selected(ti);
		}

		if (files.size()) {
			emit_signal("files_selected", files);
			hide();
		}

		return;
	}

	String f = dir_access->get_current_dir().plus_file(file->get_text());

	if ((mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_OPEN_FILE) && dir_access->file_exists(f)) {
		emit_signal("file_selected", f);
		hide();
	} else if (mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_OPEN_DIR) {
		String path = dir_access->get_current_dir();

		path = path.replace("\\", "/");
		TreeItem *item = tree->get_selected();
		if (item) {
			Dictionary d = item->get_metadata(0);
			if (d["dir"] && d["name"] != "..") {
				path = path.plus_file(d["name"]);
			}
		}

		emit_signal("dir_selected", path);
		hide();
	}

	if (mode == FILE_MODE_SAVE_FILE) {
		bool valid = false;

		if (filter->get_selected() == filter->get_item_count() - 1) {
			valid = true; // match none
		} else if (filters.size() > 1 && filter->get_selected() == 0) {
			// match all filters
			for (int i = 0; i < filters.size(); i++) {
				String flt = filters[i].get_slice(";", 0);
				for (int j = 0; j < flt.get_slice_count(","); j++) {
					String str = flt.get_slice(",", j).strip_edges();
					if (f.match(str)) {
						valid = true;
						break;
					}
				}
				if (valid) {
					break;
				}
			}
		} else {
			int idx = filter->get_selected();
			if (filters.size() > 1) {
				idx--;
			}
			if (idx >= 0 && idx < filters.size()) {
				String flt = filters[idx].get_slice(";", 0);
				int filterSliceCount = flt.get_slice_count(",");
				for (int j = 0; j < filterSliceCount; j++) {
					String str = (flt.get_slice(",", j).strip_edges());
					if (f.match(str)) {
						valid = true;
						break;
					}
				}

				if (!valid && filterSliceCount > 0) {
					String str = (flt.get_slice(",", 0).strip_edges());
					f += str.substr(1, str.length() - 1);
					file->set_text(f.get_file());
					valid = true;
				}
			} else {
				valid = true;
			}
		}

		if (!valid) {
			exterr->popup_centered(Size2(250, 80));
			return;
		}

		if (dir_access->file_exists(f)) {
			confirm_save->set_text(RTR("File exists, overwrite?"));
			confirm_save->popup_centered(Size2(200, 80));
		} else {
			emit_signal("file_selected", f);
			hide();
		}
	}
}

void FileDialog::_cancel_pressed() {
	file->set_text("");
	invalidate();
	hide();
}

bool FileDialog::_is_open_should_be_disabled() {
	if (mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_SAVE_FILE) {
		return false;
	}

	TreeItem *ti = tree->get_next_selected(tree->get_root());
	while (ti) {
		TreeItem *prev_ti = ti;
		ti = tree->get_next_selected(tree->get_root());
		if (ti == prev_ti) {
			break;
		}
	}
	// We have something that we can't select?
	if (!ti) {
		return mode != FILE_MODE_OPEN_DIR; // In "Open folder" mode, having nothing selected picks the current folder.
	}

	Dictionary d = ti->get_metadata(0);

	// Opening a file, but selected a folder? Forbidden.
	return ((mode == FILE_MODE_OPEN_FILE || mode == FILE_MODE_OPEN_FILES) && d["dir"]) || // Flipped case, also forbidden.
		   (mode == FILE_MODE_OPEN_DIR && !d["dir"]);
}

void FileDialog::_go_up() {
	dir_access->change_dir("..");
	update_file_list();
	update_dir();
}

void FileDialog::deselect_items() {
	// Clear currently selected items in file manager.
	tree->deselect_all();

	// And change get_ok title.
	if (!tree->is_anything_selected()) {
		get_ok()->set_disabled(_is_open_should_be_disabled());

		switch (mode) {
			case FILE_MODE_OPEN_FILE:
			case FILE_MODE_OPEN_FILES:
				get_ok()->set_text(RTR("Open"));
				break;
			case FILE_MODE_OPEN_DIR:
				get_ok()->set_text(RTR("Select Current Folder"));
				break;
			case FILE_MODE_OPEN_ANY:
			case FILE_MODE_SAVE_FILE:
				// FIXME: Implement, or refactor to avoid duplication with set_mode
				break;
		}
	}
}

void FileDialog::_tree_multi_selected(Object *p_object, int p_cell, bool p_selected) {
	_tree_selected();
}

void FileDialog::_tree_selected() {
	TreeItem *ti = tree->get_selected();
	if (!ti) {
		return;
	}
	Dictionary d = ti->get_metadata(0);

	if (!d["dir"]) {
		file->set_text(d["name"]);
	} else if (mode == FILE_MODE_OPEN_DIR) {
		get_ok()->set_text(RTR("Select This Folder"));
	}

	get_ok()->set_disabled(_is_open_should_be_disabled());
}

void FileDialog::_tree_item_activated() {
	TreeItem *ti = tree->get_selected();
	if (!ti) {
		return;
	}

	Dictionary d = ti->get_metadata(0);

	if (d["dir"]) {
		dir_access->change_dir(d["name"]);
		if (mode == FILE_MODE_OPEN_FILE || mode == FILE_MODE_OPEN_FILES || mode == FILE_MODE_OPEN_DIR || mode == FILE_MODE_OPEN_ANY) {
			file->set_text("");
		}
		call_deferred("_update_file_list");
		call_deferred("_update_dir");
	} else {
		_action_pressed();
	}
}

void FileDialog::update_file_name() {
	int idx = filter->get_selected() - 1;
	if ((idx == -1 && filter->get_item_count() == 2) || (filter->get_item_count() > 2 && idx >= 0 && idx < filter->get_item_count() - 2)) {
		if (idx == -1) {
			idx += 1;
		}
		String filter_str = filters[idx];
		String file_str = file->get_text();
		String base_name = file_str.get_basename();
		file_str = base_name + "." + filter_str.strip_edges().to_lower();
		file->set_text(file_str);
	}
}

void FileDialog::update_file_list() {
	tree->clear();

	// Scroll back to the top after opening a directory
	tree->get_vscroll_bar()->set_value(0);

	dir_access->list_dir_begin();

	TreeItem *root = tree->create_item();
	Ref<Texture2D> folder = vbox->get_theme_icon("folder", "FileDialog");
	Ref<Texture2D> file_icon = vbox->get_theme_icon("file", "FileDialog");
	const Color folder_color = vbox->get_theme_color("folder_icon_modulate", "FileDialog");
	const Color file_color = vbox->get_theme_color("file_icon_modulate", "FileDialog");
	List<String> files;
	List<String> dirs;

	bool is_hidden;
	String item;

	while ((item = dir_access->get_next()) != "") {
		if (item == "." || item == "..") {
			continue;
		}

		is_hidden = dir_access->current_is_hidden();

		if (show_hidden_files || !is_hidden) {
			if (!dir_access->current_is_dir()) {
				files.push_back(item);
			} else {
				dirs.push_back(item);
			}
		}
	}

	dirs.sort_custom<NaturalNoCaseComparator>();
	files.sort_custom<NaturalNoCaseComparator>();

	while (!dirs.empty()) {
		String &dir_name = dirs.front()->get();
		TreeItem *ti = tree->create_item(root);
		ti->set_text(0, dir_name);
		ti->set_icon(0, folder);
		ti->set_icon_modulate(0, folder_color);

		Dictionary d;
		d["name"] = dir_name;
		d["dir"] = true;

		ti->set_metadata(0, d);

		dirs.pop_front();
	}

	List<String> patterns;
	// build filter
	if (filter->get_selected() == filter->get_item_count() - 1) {
		// match all
	} else if (filters.size() > 1 && filter->get_selected() == 0) {
		// match all filters
		for (int i = 0; i < filters.size(); i++) {
			String f = filters[i].get_slice(";", 0);
			for (int j = 0; j < f.get_slice_count(","); j++) {
				patterns.push_back(f.get_slice(",", j).strip_edges());
			}
		}
	} else {
		int idx = filter->get_selected();
		if (filters.size() > 1) {
			idx--;
		}

		if (idx >= 0 && idx < filters.size()) {
			String f = filters[idx].get_slice(";", 0);
			for (int j = 0; j < f.get_slice_count(","); j++) {
				patterns.push_back(f.get_slice(",", j).strip_edges());
			}
		}
	}

	String base_dir = dir_access->get_current_dir();

	while (!files.empty()) {
		bool match = patterns.empty();
		String match_str;

		for (List<String>::Element *E = patterns.front(); E; E = E->next()) {
			if (files.front()->get().matchn(E->get())) {
				match_str = E->get();
				match = true;
				break;
			}
		}

		if (match) {
			TreeItem *ti = tree->create_item(root);
			ti->set_text(0, files.front()->get());

			if (get_icon_func) {
				Ref<Texture2D> icon = get_icon_func(base_dir.plus_file(files.front()->get()));
				ti->set_icon(0, icon);
			} else {
				ti->set_icon(0, file_icon);
			}
			ti->set_icon_modulate(0, file_color);

			if (mode == FILE_MODE_OPEN_DIR) {
				ti->set_custom_color(0, vbox->get_theme_color("files_disabled", "FileDialog"));
				ti->set_selectable(0, false);
			}
			Dictionary d;
			d["name"] = files.front()->get();
			d["dir"] = false;
			ti->set_metadata(0, d);

			if (file->get_text() == files.front()->get() || match_str == files.front()->get()) {
				ti->select(0);
			}
		}

		files.pop_front();
	}

	if (tree->get_root() && tree->get_root()->get_children() && tree->get_selected() == nullptr) {
		tree->get_root()->get_children()->select(0);
	}
}

void FileDialog::_filter_selected(int) {
	update_file_name();
	update_file_list();
}

void FileDialog::update_filters() {
	filter->clear();

	if (filters.size() > 1) {
		String all_filters;

		const int max_filters = 5;

		for (int i = 0; i < MIN(max_filters, filters.size()); i++) {
			String flt = filters[i].get_slice(";", 0).strip_edges();
			if (i > 0) {
				all_filters += ", ";
			}
			all_filters += flt;
		}

		if (max_filters < filters.size()) {
			all_filters += ", ...";
		}

		filter->add_item(RTR("All Recognized") + " (" + all_filters + ")");
	}
	for (int i = 0; i < filters.size(); i++) {
		String flt = filters[i].get_slice(";", 0).strip_edges();
		String desc = filters[i].get_slice(";", 1).strip_edges();
		if (desc.length()) {
			filter->add_item(String(tr(desc)) + " (" + flt + ")");
		} else {
			filter->add_item("(" + flt + ")");
		}
	}

	filter->add_item(RTR("All Files (*)"));
}

void FileDialog::clear_filters() {
	filters.clear();
	update_filters();
	invalidate();
}

void FileDialog::add_filter(const String &p_filter) {
	filters.push_back(p_filter);
	update_filters();
	invalidate();
}

void FileDialog::set_filters(const Vector<String> &p_filters) {
	filters = p_filters;
	update_filters();
	invalidate();
}

Vector<String> FileDialog::get_filters() const {
	return filters;
}

String FileDialog::get_current_dir() const {
	return dir->get_text();
}

String FileDialog::get_current_file() const {
	return file->get_text();
}

String FileDialog::get_current_path() const {
	return dir->get_text().plus_file(file->get_text());
}

void FileDialog::set_current_dir(const String &p_dir) {
	dir_access->change_dir(p_dir);
	update_dir();
	invalidate();
}

void FileDialog::set_current_file(const String &p_file) {
	file->set_text(p_file);
	update_dir();
	invalidate();
	int lp = p_file.rfind(".");
	if (lp != -1) {
		file->select(0, lp);
		if (file->is_inside_tree() && !get_tree()->is_node_being_edited(file)) {
			file->grab_focus();
		}
	}
}

void FileDialog::set_current_path(const String &p_path) {
	if (!p_path.size()) {
		return;
	}
	int pos = MAX(p_path.rfind("/"), p_path.rfind("\\"));
	if (pos == -1) {
		set_current_file(p_path);
	} else {
		String dir = p_path.substr(0, pos);
		String file = p_path.substr(pos + 1, p_path.length());
		set_current_dir(dir);
		set_current_file(file);
	}
}

void FileDialog::set_mode_overrides_title(bool p_override) {
	mode_overrides_title = p_override;
}

bool FileDialog::is_mode_overriding_title() const {
	return mode_overrides_title;
}

void FileDialog::set_file_mode(FileMode p_mode) {
	ERR_FAIL_INDEX((int)p_mode, 5);

	mode = p_mode;
	switch (mode) {
		case FILE_MODE_OPEN_FILE:
			get_ok()->set_text(RTR("Open"));
			if (mode_overrides_title) {
				set_title(RTR("Open a File"));
			}
			makedir->hide();
			break;
		case FILE_MODE_OPEN_FILES:
			get_ok()->set_text(RTR("Open"));
			if (mode_overrides_title) {
				set_title(RTR("Open File(s)"));
			}
			makedir->hide();
			break;
		case FILE_MODE_OPEN_DIR:
			get_ok()->set_text(RTR("Select Current Folder"));
			if (mode_overrides_title) {
				set_title(RTR("Open a Directory"));
			}
			makedir->show();
			break;
		case FILE_MODE_OPEN_ANY:
			get_ok()->set_text(RTR("Open"));
			if (mode_overrides_title) {
				set_title(RTR("Open a File or Directory"));
			}
			makedir->show();
			break;
		case FILE_MODE_SAVE_FILE:
			get_ok()->set_text(RTR("Save"));
			if (mode_overrides_title) {
				set_title(RTR("Save a File"));
			}
			makedir->show();
			break;
	}

	if (mode == FILE_MODE_OPEN_FILES) {
		tree->set_select_mode(Tree::SELECT_MULTI);
	} else {
		tree->set_select_mode(Tree::SELECT_SINGLE);
	}
}

FileDialog::FileMode FileDialog::get_file_mode() const {
	return mode;
}

void FileDialog::set_access(Access p_access) {
	ERR_FAIL_INDEX(p_access, 3);
	if (access == p_access) {
		return;
	}
	memdelete(dir_access);
	switch (p_access) {
		case ACCESS_FILESYSTEM: {
			dir_access = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		} break;
		case ACCESS_RESOURCES: {
			dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		} break;
		case ACCESS_USERDATA: {
			dir_access = DirAccess::create(DirAccess::ACCESS_USERDATA);
		} break;
	}
	access = p_access;
	_update_drives();
	invalidate();
	update_filters();
	update_dir();
}

void FileDialog::invalidate() {
	if (is_visible()) {
		update_file_list();
		invalidated = false;
	} else {
		invalidated = true;
	}
}

FileDialog::Access FileDialog::get_access() const {
	return access;
}

void FileDialog::_make_dir_confirm() {
	Error err = dir_access->make_dir(makedirname->get_text());
	if (err == OK) {
		dir_access->change_dir(makedirname->get_text());
		invalidate();
		update_filters();
		update_dir();
	} else {
		mkdirerr->popup_centered(Size2(250, 50));
	}
	makedirname->set_text(""); // reset label
}

void FileDialog::_make_dir() {
	makedialog->popup_centered(Size2(250, 80));
	makedirname->grab_focus();
}

void FileDialog::_select_drive(int p_idx) {
	String d = drives->get_item_text(p_idx);
	dir_access->change_dir(d);
	file->set_text("");
	invalidate();
	update_dir();
}

void FileDialog::_update_drives() {
	int dc = dir_access->get_drive_count();
	if (dc == 0 || access != ACCESS_FILESYSTEM) {
		drives->hide();
	} else {
		drives->clear();
		Node *dp = drives->get_parent();
		if (dp) {
			dp->remove_child(drives);
		}
		dp = dir_access->drives_are_shortcuts() ? shortcuts_container : drives_container;
		dp->add_child(drives);
		drives->show();

		for (int i = 0; i < dir_access->get_drive_count(); i++) {
			drives->add_item(dir_access->get_drive(i));
		}

		drives->select(dir_access->get_current_drive());
	}
}

bool FileDialog::default_show_hidden_files = false;

void FileDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_unhandled_input"), &FileDialog::_unhandled_input);

	ClassDB::bind_method(D_METHOD("_cancel_pressed"), &FileDialog::_cancel_pressed);

	ClassDB::bind_method(D_METHOD("clear_filters"), &FileDialog::clear_filters);
	ClassDB::bind_method(D_METHOD("add_filter", "filter"), &FileDialog::add_filter);
	ClassDB::bind_method(D_METHOD("set_filters", "filters"), &FileDialog::set_filters);
	ClassDB::bind_method(D_METHOD("get_filters"), &FileDialog::get_filters);
	ClassDB::bind_method(D_METHOD("get_current_dir"), &FileDialog::get_current_dir);
	ClassDB::bind_method(D_METHOD("get_current_file"), &FileDialog::get_current_file);
	ClassDB::bind_method(D_METHOD("get_current_path"), &FileDialog::get_current_path);
	ClassDB::bind_method(D_METHOD("set_current_dir", "dir"), &FileDialog::set_current_dir);
	ClassDB::bind_method(D_METHOD("set_current_file", "file"), &FileDialog::set_current_file);
	ClassDB::bind_method(D_METHOD("set_current_path", "path"), &FileDialog::set_current_path);
	ClassDB::bind_method(D_METHOD("set_mode_overrides_title", "override"), &FileDialog::set_mode_overrides_title);
	ClassDB::bind_method(D_METHOD("is_mode_overriding_title"), &FileDialog::is_mode_overriding_title);
	ClassDB::bind_method(D_METHOD("set_file_mode", "mode"), &FileDialog::set_file_mode);
	ClassDB::bind_method(D_METHOD("get_file_mode"), &FileDialog::get_file_mode);
	ClassDB::bind_method(D_METHOD("get_vbox"), &FileDialog::get_vbox);
	ClassDB::bind_method(D_METHOD("get_line_edit"), &FileDialog::get_line_edit);
	ClassDB::bind_method(D_METHOD("set_access", "access"), &FileDialog::set_access);
	ClassDB::bind_method(D_METHOD("get_access"), &FileDialog::get_access);
	ClassDB::bind_method(D_METHOD("set_show_hidden_files", "show"), &FileDialog::set_show_hidden_files);
	ClassDB::bind_method(D_METHOD("is_showing_hidden_files"), &FileDialog::is_showing_hidden_files);
	ClassDB::bind_method(D_METHOD("_update_file_name"), &FileDialog::update_file_name);
	ClassDB::bind_method(D_METHOD("_update_dir"), &FileDialog::update_dir);
	ClassDB::bind_method(D_METHOD("_update_file_list"), &FileDialog::update_file_list);
	ClassDB::bind_method(D_METHOD("deselect_items"), &FileDialog::deselect_items);

	ClassDB::bind_method(D_METHOD("invalidate"), &FileDialog::invalidate);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mode_overrides_title"), "set_mode_overrides_title", "is_mode_overriding_title");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "file_mode", PROPERTY_HINT_ENUM, "Open File,Open Files,Open Folder,Open Any,Save"), "set_file_mode", "get_file_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "access", PROPERTY_HINT_ENUM, "Resources,User data,File system"), "set_access", "get_access");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "filters"), "set_filters", "get_filters");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_hidden_files"), "set_show_hidden_files", "is_showing_hidden_files");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_dir"), "set_current_dir", "get_current_dir");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_file"), "set_current_file", "get_current_file");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_path"), "set_current_path", "get_current_path");

	ADD_SIGNAL(MethodInfo("file_selected", PropertyInfo(Variant::STRING, "path")));
	ADD_SIGNAL(MethodInfo("files_selected", PropertyInfo(Variant::PACKED_STRING_ARRAY, "paths")));
	ADD_SIGNAL(MethodInfo("dir_selected", PropertyInfo(Variant::STRING, "dir")));

	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_FILE);
	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_FILES);
	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_DIR);
	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_ANY);
	BIND_ENUM_CONSTANT(FILE_MODE_SAVE_FILE);

	BIND_ENUM_CONSTANT(ACCESS_RESOURCES);
	BIND_ENUM_CONSTANT(ACCESS_USERDATA);
	BIND_ENUM_CONSTANT(ACCESS_FILESYSTEM);
}

void FileDialog::set_show_hidden_files(bool p_show) {
	show_hidden_files = p_show;
	invalidate();
}

bool FileDialog::is_showing_hidden_files() const {
	return show_hidden_files;
}

void FileDialog::set_default_show_hidden_files(bool p_show) {
	default_show_hidden_files = p_show;
}

FileDialog::FileDialog() {
	show_hidden_files = default_show_hidden_files;

	mode_overrides_title = true;

	vbox = memnew(VBoxContainer);
	add_child(vbox);
	vbox->connect("theme_changed", callable_mp(this, &FileDialog::_theme_changed));

	mode = FILE_MODE_SAVE_FILE;
	set_title(RTR("Save a File"));

	HBoxContainer *hbc = memnew(HBoxContainer);

	dir_up = memnew(Button);
	dir_up->set_flat(true);
	dir_up->set_tooltip(RTR("Go to parent folder."));
	hbc->add_child(dir_up);
	dir_up->connect("pressed", callable_mp(this, &FileDialog::_go_up));

	hbc->add_child(memnew(Label(RTR("Path:"))));

	drives_container = memnew(HBoxContainer);
	hbc->add_child(drives_container);

	drives = memnew(OptionButton);
	drives->connect("item_selected", callable_mp(this, &FileDialog::_select_drive));
	hbc->add_child(drives);

	dir = memnew(LineEdit);
	hbc->add_child(dir);
	dir->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	refresh = memnew(Button);
	refresh->set_flat(true);
	refresh->set_tooltip(RTR("Refresh files."));
	refresh->connect("pressed", callable_mp(this, &FileDialog::update_file_list));
	hbc->add_child(refresh);

	show_hidden = memnew(Button);
	show_hidden->set_flat(true);
	show_hidden->set_toggle_mode(true);
	show_hidden->set_pressed(is_showing_hidden_files());
	show_hidden->set_tooltip(RTR("Toggle the visibility of hidden files."));
	show_hidden->connect("toggled", callable_mp(this, &FileDialog::set_show_hidden_files));
	hbc->add_child(show_hidden);

	shortcuts_container = memnew(HBoxContainer);
	hbc->add_child(shortcuts_container);

	makedir = memnew(Button);
	makedir->set_text(RTR("Create Folder"));
	makedir->connect("pressed", callable_mp(this, &FileDialog::_make_dir));
	hbc->add_child(makedir);
	vbox->add_child(hbc);

	tree = memnew(Tree);
	tree->set_hide_root(true);
	vbox->add_margin_child(RTR("Directories & Files:"), tree, true);

	file_box = memnew(HBoxContainer);
	file_box->add_child(memnew(Label(RTR("File:"))));
	file = memnew(LineEdit);
	file->set_stretch_ratio(4);
	file->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	file_box->add_child(file);
	filter = memnew(OptionButton);
	filter->set_stretch_ratio(3);
	filter->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	filter->set_clip_text(true); // too many extensions overflows it
	file_box->add_child(filter);
	vbox->add_child(file_box);

	dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	access = ACCESS_RESOURCES;
	_update_drives();

	connect("confirmed", callable_mp(this, &FileDialog::_action_pressed));
	tree->connect("multi_selected", callable_mp(this, &FileDialog::_tree_multi_selected), varray(), CONNECT_DEFERRED);
	tree->connect("cell_selected", callable_mp(this, &FileDialog::_tree_selected), varray(), CONNECT_DEFERRED);
	tree->connect("item_activated", callable_mp(this, &FileDialog::_tree_item_activated), varray());
	tree->connect("nothing_selected", callable_mp(this, &FileDialog::deselect_items));
	dir->connect("text_entered", callable_mp(this, &FileDialog::_dir_entered));
	file->connect("text_entered", callable_mp(this, &FileDialog::_file_entered));
	filter->connect("item_selected", callable_mp(this, &FileDialog::_filter_selected));

	confirm_save = memnew(ConfirmationDialog);
	//	confirm_save->set_as_top_level(true);
	add_child(confirm_save);

	confirm_save->connect("confirmed", callable_mp(this, &FileDialog::_save_confirm_pressed));

	makedialog = memnew(ConfirmationDialog);
	makedialog->set_title(RTR("Create Folder"));
	VBoxContainer *makevb = memnew(VBoxContainer);
	makedialog->add_child(makevb);

	makedirname = memnew(LineEdit);
	makevb->add_margin_child(RTR("Name:"), makedirname);
	add_child(makedialog);
	makedialog->register_text_enter(makedirname);
	makedialog->connect("confirmed", callable_mp(this, &FileDialog::_make_dir_confirm));
	mkdirerr = memnew(AcceptDialog);
	mkdirerr->set_text(RTR("Could not create folder."));
	add_child(mkdirerr);

	exterr = memnew(AcceptDialog);
	exterr->set_text(RTR("Must use a valid extension."));
	add_child(exterr);

	update_filters();
	update_dir();

	set_hide_on_ok(false);

	invalidated = true;
	if (register_func) {
		register_func(this);
	}
}

FileDialog::~FileDialog() {
	if (unregister_func) {
		unregister_func(this);
	}
	memdelete(dir_access);
}
