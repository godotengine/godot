/*************************************************************************/
/*  editor_file_dialog.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "editor_file_dialog.h"

#include "editor_resource_preview.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "os/file_access.h"
#include "os/keyboard.h"
#include "print_string.h"
#include "scene/gui/center_container.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"

EditorFileDialog::GetIconFunc EditorFileDialog::get_icon_func = NULL;
EditorFileDialog::GetIconFunc EditorFileDialog::get_large_icon_func = NULL;

EditorFileDialog::RegisterFunc EditorFileDialog::register_func = NULL;
EditorFileDialog::RegisterFunc EditorFileDialog::unregister_func = NULL;

VBoxContainer *EditorFileDialog::get_vbox() {
	return vbox;
}

void EditorFileDialog::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		//_update_icons
		mode_thumbnails->set_icon(get_icon("FileThumbnail", "EditorIcons"));
		mode_list->set_icon(get_icon("FileList", "EditorIcons"));
		dir_prev->set_icon(get_icon("ArrowLeft", "EditorIcons"));
		dir_next->set_icon(get_icon("ArrowRight", "EditorIcons"));
		dir_up->set_icon(get_icon("ArrowUp", "EditorIcons"));
		refresh->set_icon(get_icon("Reload", "EditorIcons"));
		favorite->set_icon(get_icon("Favorites", "EditorIcons"));

		fav_up->set_icon(get_icon("MoveUp", "EditorIcons"));
		fav_down->set_icon(get_icon("MoveDown", "EditorIcons"));
		fav_rm->set_icon(get_icon("RemoveSmall", "EditorIcons"));

	} else if (p_what == NOTIFICATION_PROCESS) {

		if (preview_waiting) {
			preview_wheel_timeout -= get_process_delta_time();
			if (preview_wheel_timeout <= 0) {
				preview_wheel_index++;
				if (preview_wheel_index >= 8)
					preview_wheel_index = 0;
				Ref<Texture> frame = get_icon("WaitPreview" + itos(preview_wheel_index + 1), "EditorIcons");
				preview->set_texture(frame);
				preview_wheel_timeout = 0.1;
			}
		}
	} else if (p_what == NOTIFICATION_DRAW) {

		//RID ci = get_canvas_item();
		//get_stylebox("panel","PopupMenu")->draw(ci,Rect2(Point2(),get_size()));
	} else if (p_what == NOTIFICATION_POPUP_HIDE) {

		set_process_unhandled_input(false);

	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {

		bool show_hidden = EditorSettings::get_singleton()->get("filesystem/file_dialog/show_hidden_files");
		if (show_hidden_files != show_hidden)
			set_show_hidden_files(show_hidden);
		set_display_mode((DisplayMode)EditorSettings::get_singleton()->get("filesystem/file_dialog/display_mode").operator int());

		//_update_icons
		mode_thumbnails->set_icon(get_icon("FileThumbnail", "EditorIcons"));
		mode_list->set_icon(get_icon("FileList", "EditorIcons"));
		dir_prev->set_icon(get_icon("ArrowLeft", "EditorIcons"));
		dir_next->set_icon(get_icon("ArrowRight", "EditorIcons"));
		dir_up->set_icon(get_icon("ArrowUp", "EditorIcons"));
		refresh->set_icon(get_icon("Reload", "EditorIcons"));
		favorite->set_icon(get_icon("Favorites", "EditorIcons"));

		fav_up->set_icon(get_icon("MoveUp", "EditorIcons"));
		fav_down->set_icon(get_icon("MoveDown", "EditorIcons"));
		fav_rm->set_icon(get_icon("RemoveSmall", "EditorIcons"));

		update_file_list();
	}
}

void EditorFileDialog::_unhandled_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && is_window_modal_on_top()) {

		if (k->is_pressed()) {

			bool handled = false;

			if (ED_IS_SHORTCUT("file_dialog/go_back", p_event)) {
				_go_back();
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/go_forward", p_event)) {
				_go_forward();
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/go_up", p_event)) {
				_go_up();
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/refresh", p_event)) {
				invalidate();
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/toggle_hidden_files", p_event)) {
				bool show = !show_hidden_files;
				set_show_hidden_files(show);
				EditorSettings::get_singleton()->set("filesystem/file_dialog/show_hidden_files", show);
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/toggle_favorite", p_event)) {
				_favorite_toggled(favorite->is_pressed());
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/toggle_mode", p_event)) {
				if (mode_thumbnails->is_pressed()) {
					set_display_mode(DISPLAY_LIST);
				} else {
					set_display_mode(DISPLAY_THUMBNAILS);
				}
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/create_folder", p_event)) {
				_make_dir();
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/focus_path", p_event)) {
				dir->grab_focus();
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/move_favorite_up", p_event)) {
				_favorite_move_up();
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/move_favorite_down", p_event)) {
				_favorite_move_down();
				handled = true;
			}

			if (handled) {
				accept_event();
			}
		}
	}
}

void EditorFileDialog::set_enable_multiple_selection(bool p_enable) {

	item_list->set_select_mode(p_enable ? ItemList::SELECT_MULTI : ItemList::SELECT_SINGLE);
};

Vector<String> EditorFileDialog::get_selected_files() const {

	Vector<String> list;
	for (int i = 0; i < item_list->get_item_count(); i++) {
		if (item_list->is_selected(i))
			list.push_back(item_list->get_item_text(i));
	}
	return list;
};

void EditorFileDialog::update_dir() {

	dir->set_text(dir_access->get_current_dir());
}

void EditorFileDialog::_dir_entered(String p_dir) {

	dir_access->change_dir(p_dir);
	file->set_text("");
	invalidate();
	update_dir();
	_push_history();
}

void EditorFileDialog::_file_entered(const String &p_file) {

	_action_pressed();
}

void EditorFileDialog::_save_confirm_pressed() {
	String f = dir_access->get_current_dir().plus_file(file->get_text());
	_save_to_recent();
	emit_signal("file_selected", f);
	hide();
}

void EditorFileDialog::_post_popup() {

	ConfirmationDialog::_post_popup();
	if (invalidated) {
		update_file_list();
		invalidated = false;
	}
	if (mode == MODE_SAVE_FILE)
		file->grab_focus();
	else
		item_list->grab_focus();

	if (is_visible_in_tree() && get_current_file() != "")
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));

	if (is_visible_in_tree()) {
		Ref<Texture> folder = get_icon("folder", "FileDialog");
		recent->clear();

		bool res = access == ACCESS_RESOURCES;
		Vector<String> recentd = EditorSettings::get_singleton()->get_recent_dirs();
		for (int i = 0; i < recentd.size(); i++) {
			bool cres = recentd[i].begins_with("res://");
			if (cres != res)
				continue;
			String name = recentd[i];
			if (res && name == "res://") {
				name = "/";
			} else {
				name = name.get_file() + "/";
			}

			//print_line("file: "+name);
			recent->add_item(name, folder);
			recent->set_item_metadata(recent->get_item_count() - 1, recentd[i]);
		}

		local_history.clear();
		local_history_pos = -1;
		_push_history();

		_update_favorites();
	}

	set_process_unhandled_input(true);
}

void EditorFileDialog::_thumbnail_result(const String &p_path, const Ref<Texture> &p_preview, const Variant &p_udata) {

	if (display_mode == DISPLAY_LIST || p_preview.is_null())
		return;

	for (int i = 0; i < item_list->get_item_count(); i++) {
		Dictionary d = item_list->get_item_metadata(i);
		String pname = d["path"];
		if (pname == p_path) {
			item_list->set_item_icon(i, p_preview);
			item_list->set_item_tag_icon(i, Ref<Texture>());
		}
	}
}

void EditorFileDialog::_thumbnail_done(const String &p_path, const Ref<Texture> &p_preview, const Variant &p_udata) {

	set_process(false);
	preview_waiting = false;

	if (p_preview.is_valid() && get_current_path() == p_path) {

		preview->set_texture(p_preview);
		if (display_mode == DISPLAY_THUMBNAILS) {
			preview_vb->hide();
		} else {
			preview_vb->show();
		}

	} else {
		preview_vb->hide();
		preview->set_texture(Ref<Texture>());
	}
}

void EditorFileDialog::_request_single_thumbnail(const String &p_path) {

	if (!FileAccess::exists(p_path))
		return;

	EditorResourcePreview::get_singleton()->queue_resource_preview(p_path, this, "_thumbnail_done", p_path);
	//print_line("want file "+p_path);
	set_process(true);
	preview_waiting = true;
	preview_wheel_timeout = 0;
}

void EditorFileDialog::_action_pressed() {

	if (mode == MODE_OPEN_FILES) {

		String fbase = dir_access->get_current_dir();

		PoolVector<String> files;
		for (int i = 0; i < item_list->get_item_count(); i++) {
			if (item_list->is_selected(i))
				files.push_back(fbase.plus_file(item_list->get_item_text(i)));
		}

		if (files.size()) {
			_save_to_recent();
			emit_signal("files_selected", files);
			hide();
		}

		return;
	}

	String f = dir_access->get_current_dir().plus_file(file->get_text());

	if ((mode == MODE_OPEN_ANY || mode == MODE_OPEN_FILE) && dir_access->file_exists(f)) {
		_save_to_recent();
		emit_signal("file_selected", f);
		hide();
	} else if (mode == MODE_OPEN_ANY || mode == MODE_OPEN_DIR) {

		String path = dir_access->get_current_dir();

		path = path.replace("\\", "/");

		for (int i = 0; i < item_list->get_item_count(); i++) {
			if (item_list->is_selected(i)) {
				Dictionary d = item_list->get_item_metadata(i);
				if (d["dir"]) {
					path = path.plus_file(d["name"]);

					break;
				}
			}
		}

		_save_to_recent();
		emit_signal("dir_selected", path);
		hide();
	}

	if (mode == MODE_SAVE_FILE) {

		bool valid = false;

		if (filter->get_selected() == filter->get_item_count() - 1) {
			valid = true; //match none
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
				if (valid)
					break;
			}
		} else {
			int idx = filter->get_selected();
			if (filters.size() > 1)
				idx--;
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
					_request_single_thumbnail(get_current_dir().plus_file(f.get_file()));
					file->set_text(f.get_file());
					valid = true;
				}
			} else {
				valid = true;
			}
		}

		if (!valid) {

			exterr->popup_centered_minsize(Size2(250, 80) * EDSCALE);
			return;
		}

		if (dir_access->file_exists(f) && !disable_overwrite_warning) {
			confirm_save->set_text(TTR("File Exists, Overwrite?"));
			confirm_save->popup_centered(Size2(200, 80));
		} else {

			_save_to_recent();
			emit_signal("file_selected", f);
			hide();
		}
	}
}

void EditorFileDialog::_cancel_pressed() {

	file->set_text("");
	invalidate();
	hide();
}

void EditorFileDialog::_item_selected(int p_item) {

	int current = p_item;
	if (current < 0 || current >= item_list->get_item_count())
		return;

	Dictionary d = item_list->get_item_metadata(current);

	if (!d["dir"]) {

		file->set_text(d["name"]);
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));
	}
}

void EditorFileDialog::_push_history() {

	local_history.resize(local_history_pos + 1);
	String new_path = dir_access->get_current_dir();
	if (local_history.size() == 0 || new_path != local_history[local_history_pos]) {
		local_history.push_back(new_path);
		local_history_pos++;
		dir_prev->set_disabled(local_history_pos == 0);
		dir_next->set_disabled(true);
	}
}
void EditorFileDialog::_item_dc_selected(int p_item) {

	int current = p_item;
	if (current < 0 || current >= item_list->get_item_count())
		return;

	Dictionary d = item_list->get_item_metadata(current);

	if (d["dir"]) {

		dir_access->change_dir(d["name"]);
		call_deferred("_update_file_list");
		call_deferred("_update_dir");

		_push_history();

	} else {

		_action_pressed();
	}
}

void EditorFileDialog::update_file_list() {

	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;
	Ref<Texture> folder_thumbnail;
	Ref<Texture> file_thumbnail;

	item_list->clear();

	if (display_mode == DISPLAY_THUMBNAILS) {

		item_list->set_max_columns(0);
		item_list->set_icon_mode(ItemList::ICON_MODE_TOP);
		item_list->set_fixed_column_width(thumbnail_size * 3 / 2);
		item_list->set_max_text_lines(2);
		item_list->set_fixed_icon_size(Size2(thumbnail_size, thumbnail_size));

		if (thumbnail_size < 64) {
			folder_thumbnail = get_icon("FolderMediumThumb", "EditorIcons");
			file_thumbnail = get_icon("FileMediumThumb", "EditorIcons");
		} else {
			folder_thumbnail = get_icon("FolderBigThumb", "EditorIcons");
			file_thumbnail = get_icon("FileBigThumb", "EditorIcons");
		}

		preview_vb->hide();

	} else {

		item_list->set_icon_mode(ItemList::ICON_MODE_LEFT);
		item_list->set_max_columns(1);
		item_list->set_max_text_lines(1);
		item_list->set_fixed_column_width(0);
		item_list->set_fixed_icon_size(Size2());
		if (preview->get_texture().is_valid())
			preview_vb->show();
	}

	String cdir = dir_access->get_current_dir();
	bool skip_pp = access == ACCESS_RESOURCES && cdir == "res://";

	dir_access->list_dir_begin();

	Ref<Texture> folder = get_icon("folder", "FileDialog");
	List<String> files;
	List<String> dirs;

	bool isdir;
	bool ishidden;
	bool show_hidden = show_hidden_files;
	String item;

	while ((item = dir_access->get_next(&isdir)) != "") {

		ishidden = dir_access->current_is_hidden();

		if (show_hidden || !ishidden) {
			if (!isdir)
				files.push_back(item);
			else if (item != ".." || !skip_pp)
				dirs.push_back(item);
		}
	}

	if (dirs.find("..") == NULL) {
		//may happen if lacking permissions
		dirs.push_back("..");
	}

	dirs.sort_custom<NaturalNoCaseComparator>();
	files.sort_custom<NaturalNoCaseComparator>();

	while (!dirs.empty()) {
		const String &dir_name = dirs.front()->get();

		item_list->add_item(dir_name + "/");

		if (display_mode == DISPLAY_THUMBNAILS) {

			item_list->set_item_icon(item_list->get_item_count() - 1, folder_thumbnail);
		} else {

			item_list->set_item_icon(item_list->get_item_count() - 1, folder);
		}

		Dictionary d;
		d["name"] = dir_name;
		d["path"] = String();
		d["dir"] = true;

		item_list->set_item_metadata(item_list->get_item_count() - 1, d);

		dirs.pop_front();
	}

	dirs.clear();

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
		if (filters.size() > 1)
			idx--;

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

		for (List<String>::Element *E = patterns.front(); E; E = E->next()) {

			if (files.front()->get().matchn(E->get())) {

				match = true;
				break;
			}
		}

		if (match) {
			//TreeItem *ti=tree->create_item(root);
			//ti->set_text(0,files.front()->get());
			item_list->add_item(files.front()->get());

			if (get_icon_func) {

				Ref<Texture> icon = get_icon_func(base_dir.plus_file(files.front()->get()));
				//ti->set_icon(0,icon);
				if (display_mode == DISPLAY_THUMBNAILS) {

					item_list->set_item_icon(item_list->get_item_count() - 1, file_thumbnail);
					item_list->set_item_tag_icon(item_list->get_item_count() - 1, icon);
				} else {
					item_list->set_item_icon(item_list->get_item_count() - 1, icon);
				}
			}

			if (mode == MODE_OPEN_DIR) {
				//disabled mode?
				//ti->set_custom_color(0,get_color("files_disabled"));
				//ti->set_selectable(0,false);
			}
			Dictionary d;
			d["name"] = files.front()->get();
			d["dir"] = false;
			String fullpath = base_dir.plus_file(files.front()->get());

			if (display_mode == DISPLAY_THUMBNAILS) {
				EditorResourcePreview::get_singleton()->queue_resource_preview(fullpath, this, "_thumbnail_result", fullpath);
			}
			d["path"] = base_dir.plus_file(files.front()->get());
			//ti->set_metadata(0,d);
			item_list->set_item_metadata(item_list->get_item_count() - 1, d);

			if (file->get_text() == files.front()->get())
				item_list->set_current(item_list->get_item_count() - 1);
		}

		files.pop_front();
	}

	if (favorites->get_current() >= 0) {
		favorites->unselect(favorites->get_current());
	}

	favorite->set_pressed(false);
	fav_up->set_disabled(true);
	fav_down->set_disabled(true);
	for (int i = 0; i < favorites->get_item_count(); i++) {
		if (favorites->get_item_metadata(i) == base_dir) {
			favorites->select(i);
			favorite->set_pressed(true);
			if (i > 0) {
				fav_up->set_disabled(false);
			}
			if (i < favorites->get_item_count() - 1) {
				fav_down->set_disabled(false);
			}
			break;
		}
	}
	// ??
	/*
	if (tree->get_root() && tree->get_root()->get_children())
		tree->get_root()->get_children()->select(0);
	*/

	files.clear();
}

void EditorFileDialog::_filter_selected(int) {

	update_file_list();
}

void EditorFileDialog::update_filters() {

	filter->clear();

	if (filters.size() > 1) {
		String all_filters;

		const int max_filters = 5;

		for (int i = 0; i < MIN(max_filters, filters.size()); i++) {
			String flt = filters[i].get_slice(";", 0);
			if (i > 0)
				all_filters += ",";
			all_filters += flt;
		}

		if (max_filters < filters.size())
			all_filters += ", ...";

		filter->add_item(TTR("All Recognized") + " ( " + all_filters + " )");
	}
	for (int i = 0; i < filters.size(); i++) {

		String flt = filters[i].get_slice(";", 0).strip_edges();
		String desc = filters[i].get_slice(";", 1).strip_edges();
		if (desc.length())
			filter->add_item(desc + " ( " + flt + " )");
		else
			filter->add_item("( " + flt + " )");
	}

	filter->add_item(TTR("All Files (*)"));
}

void EditorFileDialog::clear_filters() {

	filters.clear();
	update_filters();
	invalidate();
}
void EditorFileDialog::add_filter(const String &p_filter) {

	filters.push_back(p_filter);
	update_filters();
	invalidate();
}

String EditorFileDialog::get_current_dir() const {

	return dir->get_text();
}
String EditorFileDialog::get_current_file() const {

	return file->get_text();
}
String EditorFileDialog::get_current_path() const {

	return dir->get_text().plus_file(file->get_text());
}
void EditorFileDialog::set_current_dir(const String &p_dir) {

	dir_access->change_dir(p_dir);
	update_dir();
	invalidate();
	//_push_history();
}
void EditorFileDialog::set_current_file(const String &p_file) {

	file->set_text(p_file);
	update_dir();
	invalidate();
	int lp = p_file.find_last(".");
	if (lp != -1) {
		file->select(0, lp);
		file->grab_focus();
	}

	if (is_visible_in_tree())
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));
}
void EditorFileDialog::set_current_path(const String &p_path) {

	if (!p_path.size())
		return;
	int pos = MAX(p_path.find_last("/"), p_path.find_last("\\"));
	if (pos == -1) {

		set_current_file(p_path);
	} else {

		String dir = p_path.substr(0, pos);
		String file = p_path.substr(pos + 1, p_path.length());
		set_current_dir(dir);
		set_current_file(file);
	}
}

void EditorFileDialog::set_mode(Mode p_mode) {

	mode = p_mode;
	switch (mode) {

		case MODE_OPEN_FILE:
			get_ok()->set_text(TTR("Open"));
			set_title(TTR("Open a File"));
			makedir->hide();
			break;
		case MODE_OPEN_FILES:
			get_ok()->set_text(TTR("Open"));
			set_title(TTR("Open File(s)"));
			makedir->hide();
			break;
		case MODE_OPEN_DIR:
			get_ok()->set_text(TTR("Open"));
			set_title(TTR("Open a Directory"));
			makedir->show();
			break;
		case MODE_OPEN_ANY:
			get_ok()->set_text(TTR("Open"));
			set_title(TTR("Open a File or Directory"));
			makedir->show();
			break;
		case MODE_SAVE_FILE:
			get_ok()->set_text(TTR("Save"));
			set_title(TTR("Save a File"));
			makedir->show();
			break;
	}

	if (mode == MODE_OPEN_FILES) {
		item_list->set_select_mode(ItemList::SELECT_MULTI);
	} else {
		item_list->set_select_mode(ItemList::SELECT_SINGLE);
	}
}

EditorFileDialog::Mode EditorFileDialog::get_mode() const {

	return mode;
}

void EditorFileDialog::set_access(Access p_access) {

	ERR_FAIL_INDEX(p_access, 3);
	if (access == p_access)
		return;
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

void EditorFileDialog::invalidate() {

	if (is_visible_in_tree()) {
		update_file_list();
		invalidated = false;
	} else {
		invalidated = true;
	}
}

EditorFileDialog::Access EditorFileDialog::get_access() const {

	return access;
}

void EditorFileDialog::_make_dir_confirm() {

	Error err = dir_access->make_dir(makedirname->get_text());
	if (err == OK) {
		dir_access->change_dir(makedirname->get_text());
		invalidate();
		update_filters();
		update_dir();
		_push_history();

	} else {
		mkdirerr->popup_centered_minsize(Size2(250, 50) * EDSCALE);
	}
	makedirname->set_text(""); // reset label
}

void EditorFileDialog::_make_dir() {

	makedialog->popup_centered_minsize(Size2(250, 80) * EDSCALE);
	makedirname->grab_focus();
}

void EditorFileDialog::_select_drive(int p_idx) {

	String d = drives->get_item_text(p_idx);
	dir_access->change_dir(d);
	file->set_text("");
	invalidate();
	update_dir();
	_push_history();
}

void EditorFileDialog::_update_drives() {

	int dc = dir_access->get_drive_count();
	if (dc == 0 || access != ACCESS_FILESYSTEM) {
		drives->hide();
	} else {
		drives->clear();
		drives->show();

		for (int i = 0; i < dir_access->get_drive_count(); i++) {
			String d = dir_access->get_drive(i);
			drives->add_item(dir_access->get_drive(i));
		}

		drives->select(dir_access->get_current_drive());
	}
}

void EditorFileDialog::_favorite_selected(int p_idx) {

	Vector<String> favorited = EditorSettings::get_singleton()->get_favorite_dirs();
	ERR_FAIL_INDEX(p_idx, favorited.size());

	dir_access->change_dir(favorited[p_idx]);
	file->set_text("");
	invalidate();
	update_dir();
	_push_history();
}

void EditorFileDialog::_favorite_move_up() {

	int current = favorites->get_current();

	if (current > 0 && current < favorites->get_item_count()) {
		Vector<String> favorited = EditorSettings::get_singleton()->get_favorite_dirs();

		int a_idx = favorited.find(String(favorites->get_item_metadata(current - 1)));
		int b_idx = favorited.find(String(favorites->get_item_metadata(current)));

		if (a_idx == -1 || b_idx == -1)
			return;
		SWAP(favorited[a_idx], favorited[b_idx]);

		EditorSettings::get_singleton()->set_favorite_dirs(favorited);

		_update_favorites();
		update_file_list();
	}
}
void EditorFileDialog::_favorite_move_down() {

	int current = favorites->get_current();

	if (current >= 0 && current < favorites->get_item_count() - 1) {
		Vector<String> favorited = EditorSettings::get_singleton()->get_favorite_dirs();

		int a_idx = favorited.find(String(favorites->get_item_metadata(current + 1)));
		int b_idx = favorited.find(String(favorites->get_item_metadata(current)));

		if (a_idx == -1 || b_idx == -1)
			return;
		SWAP(favorited[a_idx], favorited[b_idx]);

		EditorSettings::get_singleton()->set_favorite_dirs(favorited);

		_update_favorites();
		update_file_list();
	}
}

void EditorFileDialog::_update_favorites() {

	bool res = access == ACCESS_RESOURCES;

	String current = get_current_dir();
	Ref<Texture> star = get_icon("Favorites", "EditorIcons");
	favorites->clear();

	favorite->set_pressed(false);

	Vector<String> favorited = EditorSettings::get_singleton()->get_favorite_dirs();
	for (int i = 0; i < favorited.size(); i++) {
		bool cres = favorited[i].begins_with("res://");
		if (cres != res)
			continue;
		String name = favorited[i];

		bool setthis = name == current;

		if (res && name == "res://") {
			name = "/";
		} else {
			name = name.get_file() + "/";
		}

		//print_line("file: "+name);
		favorites->add_item(name, star);
		favorites->set_item_metadata(favorites->get_item_count() - 1, favorited[i]);

		if (setthis) {
			favorite->set_pressed(true);
			favorites->set_current(favorites->get_item_count() - 1);
		}
	}
}

void EditorFileDialog::_favorite_toggled(bool p_toggle) {
	bool res = access == ACCESS_RESOURCES;

	String cd = get_current_dir();

	Vector<String> favorited = EditorSettings::get_singleton()->get_favorite_dirs();

	bool found = false;
	for (int i = 0; i < favorited.size(); i++) {
		bool cres = favorited[i].begins_with("res://");
		if (cres != res)
			continue;

		if (favorited[i] == cd) {
			found = true;
			break;
		}
	}

	if (found) {
		favorited.erase(cd);
		favorite->set_pressed(false);
	} else {
		favorited.push_back(cd);
		favorite->set_pressed(true);
	}

	EditorSettings::get_singleton()->set_favorite_dirs(favorited);

	_update_favorites();
}

void EditorFileDialog::_recent_selected(int p_idx) {

	Vector<String> recentd = EditorSettings::get_singleton()->get_recent_dirs();
	ERR_FAIL_INDEX(p_idx, recentd.size());

	dir_access->change_dir(recent->get_item_metadata(p_idx));
	update_file_list();
	update_dir();
	_push_history();
}

void EditorFileDialog::_go_up() {

	dir_access->change_dir("..");
	update_file_list();
	update_dir();
	_push_history();
}

void EditorFileDialog::_go_back() {

	if (local_history_pos <= 0) {
		return;
	}

	local_history_pos--;
	dir_access->change_dir(local_history[local_history_pos]);
	update_file_list();
	update_dir();

	dir_prev->set_disabled(local_history_pos == 0);
	dir_next->set_disabled(local_history_pos == local_history.size() - 1);
}
void EditorFileDialog::_go_forward() {

	if (local_history_pos == local_history.size() - 1) {
		return;
	}

	local_history_pos++;
	dir_access->change_dir(local_history[local_history_pos]);
	update_file_list();
	update_dir();

	dir_prev->set_disabled(local_history_pos == 0);
	dir_next->set_disabled(local_history_pos == local_history.size() - 1);
}

bool EditorFileDialog::default_show_hidden_files = false;

EditorFileDialog::DisplayMode EditorFileDialog::default_display_mode = DISPLAY_THUMBNAILS;

void EditorFileDialog::set_display_mode(DisplayMode p_mode) {

	if (display_mode == p_mode)
		return;
	if (p_mode == DISPLAY_THUMBNAILS) {
		mode_list->set_pressed(false);
		mode_thumbnails->set_pressed(true);
	} else {
		mode_thumbnails->set_pressed(false);
		mode_list->set_pressed(true);
	}
	display_mode = p_mode;
	invalidate();
}

EditorFileDialog::DisplayMode EditorFileDialog::get_display_mode() const {

	return display_mode;
}

void EditorFileDialog::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_unhandled_input"), &EditorFileDialog::_unhandled_input);

	ClassDB::bind_method(D_METHOD("_item_selected"), &EditorFileDialog::_item_selected);
	ClassDB::bind_method(D_METHOD("_item_db_selected"), &EditorFileDialog::_item_dc_selected);
	ClassDB::bind_method(D_METHOD("_dir_entered"), &EditorFileDialog::_dir_entered);
	ClassDB::bind_method(D_METHOD("_file_entered"), &EditorFileDialog::_file_entered);
	ClassDB::bind_method(D_METHOD("_action_pressed"), &EditorFileDialog::_action_pressed);
	ClassDB::bind_method(D_METHOD("_cancel_pressed"), &EditorFileDialog::_cancel_pressed);
	ClassDB::bind_method(D_METHOD("_filter_selected"), &EditorFileDialog::_filter_selected);
	ClassDB::bind_method(D_METHOD("_save_confirm_pressed"), &EditorFileDialog::_save_confirm_pressed);

	ClassDB::bind_method(D_METHOD("clear_filters"), &EditorFileDialog::clear_filters);
	ClassDB::bind_method(D_METHOD("add_filter", "filter"), &EditorFileDialog::add_filter);
	ClassDB::bind_method(D_METHOD("get_current_dir"), &EditorFileDialog::get_current_dir);
	ClassDB::bind_method(D_METHOD("get_current_file"), &EditorFileDialog::get_current_file);
	ClassDB::bind_method(D_METHOD("get_current_path"), &EditorFileDialog::get_current_path);
	ClassDB::bind_method(D_METHOD("set_current_dir", "dir"), &EditorFileDialog::set_current_dir);
	ClassDB::bind_method(D_METHOD("set_current_file", "file"), &EditorFileDialog::set_current_file);
	ClassDB::bind_method(D_METHOD("set_current_path", "path"), &EditorFileDialog::set_current_path);
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &EditorFileDialog::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &EditorFileDialog::get_mode);
	ClassDB::bind_method(D_METHOD("get_vbox"), &EditorFileDialog::get_vbox);
	ClassDB::bind_method(D_METHOD("set_access", "access"), &EditorFileDialog::set_access);
	ClassDB::bind_method(D_METHOD("get_access"), &EditorFileDialog::get_access);
	ClassDB::bind_method(D_METHOD("set_show_hidden_files", "show"), &EditorFileDialog::set_show_hidden_files);
	ClassDB::bind_method(D_METHOD("is_showing_hidden_files"), &EditorFileDialog::is_showing_hidden_files);
	ClassDB::bind_method(D_METHOD("_select_drive"), &EditorFileDialog::_select_drive);
	ClassDB::bind_method(D_METHOD("_make_dir"), &EditorFileDialog::_make_dir);
	ClassDB::bind_method(D_METHOD("_make_dir_confirm"), &EditorFileDialog::_make_dir_confirm);
	ClassDB::bind_method(D_METHOD("_update_file_list"), &EditorFileDialog::update_file_list);
	ClassDB::bind_method(D_METHOD("_update_dir"), &EditorFileDialog::update_dir);
	ClassDB::bind_method(D_METHOD("_thumbnail_done"), &EditorFileDialog::_thumbnail_done);
	ClassDB::bind_method(D_METHOD("set_display_mode", "mode"), &EditorFileDialog::set_display_mode);
	ClassDB::bind_method(D_METHOD("get_display_mode"), &EditorFileDialog::get_display_mode);
	ClassDB::bind_method(D_METHOD("_thumbnail_result"), &EditorFileDialog::_thumbnail_result);
	ClassDB::bind_method(D_METHOD("set_disable_overwrite_warning", "disable"), &EditorFileDialog::set_disable_overwrite_warning);
	ClassDB::bind_method(D_METHOD("is_overwrite_warning_disabled"), &EditorFileDialog::is_overwrite_warning_disabled);

	ClassDB::bind_method(D_METHOD("_recent_selected"), &EditorFileDialog::_recent_selected);
	ClassDB::bind_method(D_METHOD("_go_back"), &EditorFileDialog::_go_back);
	ClassDB::bind_method(D_METHOD("_go_forward"), &EditorFileDialog::_go_forward);
	ClassDB::bind_method(D_METHOD("_go_up"), &EditorFileDialog::_go_up);

	ClassDB::bind_method(D_METHOD("_favorite_toggled"), &EditorFileDialog::_favorite_toggled);
	ClassDB::bind_method(D_METHOD("_favorite_selected"), &EditorFileDialog::_favorite_selected);
	ClassDB::bind_method(D_METHOD("_favorite_move_up"), &EditorFileDialog::_favorite_move_up);
	ClassDB::bind_method(D_METHOD("_favorite_move_down"), &EditorFileDialog::_favorite_move_down);

	ClassDB::bind_method(D_METHOD("invalidate"), &EditorFileDialog::invalidate);

	ADD_SIGNAL(MethodInfo("file_selected", PropertyInfo(Variant::STRING, "path")));
	ADD_SIGNAL(MethodInfo("files_selected", PropertyInfo(Variant::POOL_STRING_ARRAY, "paths")));
	ADD_SIGNAL(MethodInfo("dir_selected", PropertyInfo(Variant::STRING, "dir")));

	BIND_ENUM_CONSTANT(MODE_OPEN_FILE);
	BIND_ENUM_CONSTANT(MODE_OPEN_FILES);
	BIND_ENUM_CONSTANT(MODE_OPEN_DIR);
	BIND_ENUM_CONSTANT(MODE_OPEN_ANY);
	BIND_ENUM_CONSTANT(MODE_SAVE_FILE);

	BIND_ENUM_CONSTANT(ACCESS_RESOURCES);
	BIND_ENUM_CONSTANT(ACCESS_USERDATA);
	BIND_ENUM_CONSTANT(ACCESS_FILESYSTEM);

	BIND_ENUM_CONSTANT(DISPLAY_THUMBNAILS);
	BIND_ENUM_CONSTANT(DISPLAY_LIST);
}

void EditorFileDialog::set_show_hidden_files(bool p_show) {
	show_hidden_files = p_show;
	invalidate();
}

bool EditorFileDialog::is_showing_hidden_files() const {
	return show_hidden_files;
}

void EditorFileDialog::set_default_show_hidden_files(bool p_show) {
	default_show_hidden_files = p_show;
}

void EditorFileDialog::set_default_display_mode(DisplayMode p_mode) {
	default_display_mode = p_mode;
}

void EditorFileDialog::_save_to_recent() {

	String dir = get_current_dir();
	Vector<String> recent = EditorSettings::get_singleton()->get_recent_dirs();

	const int max = 20;
	int count = 0;
	bool res = dir.begins_with("res://");

	for (int i = 0; i < recent.size(); i++) {
		bool cres = recent[i].begins_with("res://");
		if (recent[i] == dir || (res == cres && count > max)) {
			recent.remove(i);
			i--;
		} else {
			count++;
		}
	}

	recent.insert(0, dir);

	EditorSettings::get_singleton()->set_recent_dirs(recent);
}

void EditorFileDialog::set_disable_overwrite_warning(bool p_disable) {

	disable_overwrite_warning = p_disable;
}

bool EditorFileDialog::is_overwrite_warning_disabled() const {

	return disable_overwrite_warning;
}

EditorFileDialog::EditorFileDialog() {

	show_hidden_files = default_show_hidden_files;
	display_mode = default_display_mode;
	local_history_pos = 0;
	disable_overwrite_warning = false;
	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	mode = MODE_SAVE_FILE;
	set_title(TTR("Save a File"));

	ED_SHORTCUT("file_dialog/go_back", TTR("Go Back"), KEY_MASK_ALT | KEY_LEFT);
	ED_SHORTCUT("file_dialog/go_forward", TTR("Go Forward"), KEY_MASK_ALT | KEY_RIGHT);
	ED_SHORTCUT("file_dialog/go_up", TTR("Go Up"), KEY_MASK_ALT | KEY_UP);
	ED_SHORTCUT("file_dialog/refresh", TTR("Refresh"), KEY_F5);
	ED_SHORTCUT("file_dialog/toggle_hidden_files", TTR("Toggle Hidden Files"), KEY_MASK_CMD | KEY_H);
	ED_SHORTCUT("file_dialog/toggle_favorite", TTR("Toggle Favorite"), KEY_MASK_ALT | KEY_F);
	ED_SHORTCUT("file_dialog/toggle_mode", TTR("Toggle Mode"), KEY_MASK_ALT | KEY_V);
	ED_SHORTCUT("file_dialog/create_folder", TTR("Create Folder"), KEY_MASK_CMD | KEY_N);
	ED_SHORTCUT("file_dialog/focus_path", TTR("Focus Path"), KEY_MASK_CMD | KEY_D);
	ED_SHORTCUT("file_dialog/move_favorite_up", TTR("Move Favorite Up"), KEY_MASK_CMD | KEY_UP);
	ED_SHORTCUT("file_dialog/move_favorite_down", TTR("Move Favorite Down"), KEY_MASK_CMD | KEY_DOWN);

	HBoxContainer *pathhb = memnew(HBoxContainer);

	dir_prev = memnew(ToolButton);
	dir_next = memnew(ToolButton);
	dir_up = memnew(ToolButton);

	pathhb->add_child(dir_prev);
	pathhb->add_child(dir_next);
	pathhb->add_child(dir_up);

	dir_prev->connect("pressed", this, "_go_back");
	dir_next->connect("pressed", this, "_go_forward");
	dir_up->connect("pressed", this, "_go_up");

	dir = memnew(LineEdit);
	pathhb->add_child(dir);
	dir->set_h_size_flags(SIZE_EXPAND_FILL);

	refresh = memnew(ToolButton);
	refresh->connect("pressed", this, "_update_file_list");
	pathhb->add_child(refresh);

	favorite = memnew(ToolButton);
	favorite->set_toggle_mode(true);
	favorite->connect("toggled", this, "_favorite_toggled");
	pathhb->add_child(favorite);

	Ref<ButtonGroup> view_mode_group;
	view_mode_group.instance();

	mode_thumbnails = memnew(ToolButton);
	mode_thumbnails->connect("pressed", this, "set_display_mode", varray(DISPLAY_THUMBNAILS));
	mode_thumbnails->set_toggle_mode(true);
	mode_thumbnails->set_pressed(display_mode == DISPLAY_THUMBNAILS);
	mode_thumbnails->set_button_group(view_mode_group);
	pathhb->add_child(mode_thumbnails);

	mode_list = memnew(ToolButton);
	mode_list->connect("pressed", this, "set_display_mode", varray(DISPLAY_LIST));
	mode_list->set_toggle_mode(true);
	mode_list->set_pressed(display_mode == DISPLAY_LIST);
	mode_list->set_button_group(view_mode_group);
	pathhb->add_child(mode_list);

	drives = memnew(OptionButton);
	pathhb->add_child(drives);
	drives->connect("item_selected", this, "_select_drive");

	makedir = memnew(Button);
	makedir->set_text(TTR("Create Folder"));
	makedir->connect("pressed", this, "_make_dir");
	pathhb->add_child(makedir);

	list_hb = memnew(HBoxContainer);

	vbc->add_margin_child(TTR("Path:"), pathhb);
	vbc->add_child(list_hb);
	list_hb->set_v_size_flags(SIZE_EXPAND_FILL);

	VBoxContainer *fav_vb = memnew(VBoxContainer);
	list_hb->add_child(fav_vb);
	HBoxContainer *fav_hb = memnew(HBoxContainer);
	fav_vb->add_child(fav_hb);
	fav_hb->add_child(memnew(Label(TTR("Favorites:"))));
	fav_hb->add_spacer();
	fav_up = memnew(ToolButton);
	fav_hb->add_child(fav_up);
	fav_up->connect("pressed", this, "_favorite_move_up");
	fav_down = memnew(ToolButton);
	fav_hb->add_child(fav_down);
	fav_down->connect("pressed", this, "_favorite_move_down");
	fav_rm = memnew(ToolButton);
	fav_hb->add_child(fav_rm);
	fav_rm->hide(); // redundant

	MarginContainer *fav_mv = memnew(MarginContainer);
	fav_vb->add_child(fav_mv);
	fav_mv->set_v_size_flags(SIZE_EXPAND_FILL);
	favorites = memnew(ItemList);
	fav_mv->add_child(favorites);
	favorites->connect("item_selected", this, "_favorite_selected");

	recent = memnew(ItemList);
	fav_vb->add_margin_child(TTR("Recent:"), recent, true);
	recent->connect("item_selected", this, "_recent_selected");

	VBoxContainer *item_vb = memnew(VBoxContainer);
	list_hb->add_child(item_vb);
	item_vb->set_h_size_flags(SIZE_EXPAND_FILL);

	item_list = memnew(ItemList);
	item_list->set_v_size_flags(SIZE_EXPAND_FILL);
	item_vb->add_margin_child(TTR("Directories & Files:"), item_list, true);

	HBoxContainer *filter_hb = memnew(HBoxContainer);
	item_vb->add_child(filter_hb);

	VBoxContainer *filter_vb = memnew(VBoxContainer);
	filter_hb->add_child(filter_vb);
	filter_vb->set_h_size_flags(SIZE_EXPAND_FILL);

	preview_vb = memnew(VBoxContainer);
	filter_hb->add_child(preview_vb);
	CenterContainer *prev_cc = memnew(CenterContainer);
	preview_vb->add_margin_child(TTR("Preview:"), prev_cc);
	preview = memnew(TextureRect);
	prev_cc->add_child(preview);
	preview_vb->hide();

	file = memnew(LineEdit);
	//add_child(file);
	filter_vb->add_margin_child(TTR("File:"), file);

	filter = memnew(OptionButton);
	//add_child(filter);
	filter_vb->add_margin_child(TTR("Filter:"), filter);
	filter->set_clip_text(true); //too many extensions overflow it

	dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	access = ACCESS_RESOURCES;
	_update_drives();

	connect("confirmed", this, "_action_pressed");
	//cancel->connect("pressed", this,"_cancel_pressed");
	item_list->connect("item_selected", this, "_item_selected", varray(), CONNECT_DEFERRED);
	item_list->connect("item_activated", this, "_item_db_selected", varray());
	dir->connect("text_entered", this, "_dir_entered");
	file->connect("text_entered", this, "_file_entered");
	filter->connect("item_selected", this, "_filter_selected");

	confirm_save = memnew(ConfirmationDialog);
	confirm_save->set_as_toplevel(true);
	add_child(confirm_save);

	confirm_save->connect("confirmed", this, "_save_confirm_pressed");

	makedialog = memnew(ConfirmationDialog);
	makedialog->set_title(TTR("Create Folder"));
	VBoxContainer *makevb = memnew(VBoxContainer);
	makedialog->add_child(makevb);

	makedirname = memnew(LineEdit);
	makevb->add_margin_child(TTR("Name:"), makedirname);
	add_child(makedialog);
	makedialog->register_text_enter(makedirname);
	makedialog->connect("confirmed", this, "_make_dir_confirm");
	mkdirerr = memnew(AcceptDialog);
	mkdirerr->set_text(TTR("Could not create folder."));
	add_child(mkdirerr);

	exterr = memnew(AcceptDialog);
	exterr->set_text(TTR("Must use a valid extension."));
	add_child(exterr);

	//update_file_list();
	update_filters();
	update_dir();

	set_hide_on_ok(false);
	vbox = vbc;

	invalidated = true;
	if (register_func)
		register_func(this);

	preview_wheel_timeout = 0;
	preview_wheel_index = 0;
	preview_waiting = false;
}

EditorFileDialog::~EditorFileDialog() {

	if (unregister_func)
		unregister_func(this);
	memdelete(dir_access);
}

void EditorLineEditFileChooser::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_browse"), &EditorLineEditFileChooser::_browse);
	ClassDB::bind_method(D_METHOD("_chosen"), &EditorLineEditFileChooser::_chosen);
	ClassDB::bind_method(D_METHOD("get_button"), &EditorLineEditFileChooser::get_button);
	ClassDB::bind_method(D_METHOD("get_line_edit"), &EditorLineEditFileChooser::get_line_edit);
	ClassDB::bind_method(D_METHOD("get_file_dialog"), &EditorLineEditFileChooser::get_file_dialog);
}

void EditorLineEditFileChooser::_chosen(const String &p_text) {

	line_edit->set_text(p_text);
	line_edit->emit_signal("text_entered", p_text);
}

void EditorLineEditFileChooser::_browse() {

	dialog->popup_centered_ratio();
}

EditorLineEditFileChooser::EditorLineEditFileChooser() {

	line_edit = memnew(LineEdit);
	add_child(line_edit);
	line_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	button = memnew(Button);
	button->set_text(" .. ");
	add_child(button);
	button->connect("pressed", this, "_browse");
	dialog = memnew(EditorFileDialog);
	add_child(dialog);
	dialog->connect("file_selected", this, "_chosen");
	dialog->connect("dir_selected", this, "_chosen");
	dialog->connect("files_selected", this, "_chosen");
}
