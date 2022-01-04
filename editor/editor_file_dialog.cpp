/*************************************************************************/
/*  editor_file_dialog.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/io/file_access.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "dependency_editor.h"
#include "editor_file_system.h"
#include "editor_resource_preview.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "scene/gui/center_container.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "servers/display_server.h"

EditorFileDialog::GetIconFunc EditorFileDialog::get_icon_func = nullptr;
EditorFileDialog::GetIconFunc EditorFileDialog::get_large_icon_func = nullptr;

EditorFileDialog::RegisterFunc EditorFileDialog::register_func = nullptr;
EditorFileDialog::RegisterFunc EditorFileDialog::unregister_func = nullptr;

void EditorFileDialog::popup_file_dialog() {
	popup_centered_clamped(Size2(1050, 700) * EDSCALE, 0.8);
}

VBoxContainer *EditorFileDialog::get_vbox() {
	return vbox;
}

void EditorFileDialog::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY || p_what == NOTIFICATION_THEME_CHANGED || p_what == Control::NOTIFICATION_LAYOUT_DIRECTION_CHANGED || p_what == NOTIFICATION_TRANSLATION_CHANGED) {
		// Update icons.
		mode_thumbnails->set_icon(item_list->get_theme_icon(SNAME("FileThumbnail"), SNAME("EditorIcons")));
		mode_list->set_icon(item_list->get_theme_icon(SNAME("FileList"), SNAME("EditorIcons")));
		if (is_layout_rtl()) {
			dir_prev->set_icon(item_list->get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
			dir_next->set_icon(item_list->get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
		} else {
			dir_prev->set_icon(item_list->get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
			dir_next->set_icon(item_list->get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
		}
		dir_up->set_icon(item_list->get_theme_icon(SNAME("ArrowUp"), SNAME("EditorIcons")));
		refresh->set_icon(item_list->get_theme_icon(SNAME("Reload"), SNAME("EditorIcons")));
		favorite->set_icon(item_list->get_theme_icon(SNAME("Favorites"), SNAME("EditorIcons")));
		show_hidden->set_icon(item_list->get_theme_icon(SNAME("GuiVisibilityVisible"), SNAME("EditorIcons")));

		fav_up->set_icon(item_list->get_theme_icon(SNAME("MoveUp"), SNAME("EditorIcons")));
		fav_down->set_icon(item_list->get_theme_icon(SNAME("MoveDown"), SNAME("EditorIcons")));

	} else if (p_what == NOTIFICATION_PROCESS) {
		if (preview_waiting) {
			preview_wheel_timeout -= get_process_delta_time();
			if (preview_wheel_timeout <= 0) {
				preview_wheel_index++;
				if (preview_wheel_index >= 8) {
					preview_wheel_index = 0;
				}
				Ref<Texture2D> frame = item_list->get_theme_icon("Progress" + itos(preview_wheel_index + 1), SNAME("EditorIcons"));
				preview->set_texture(frame);
				preview_wheel_timeout = 0.1;
			}
		}

	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		bool is_showing_hidden = EditorSettings::get_singleton()->get("filesystem/file_dialog/show_hidden_files");
		if (show_hidden_files != is_showing_hidden) {
			set_show_hidden_files(is_showing_hidden);
		}
		set_display_mode((DisplayMode)EditorSettings::get_singleton()->get("filesystem/file_dialog/display_mode").operator int());

		// Update icons.
		mode_thumbnails->set_icon(item_list->get_theme_icon(SNAME("FileThumbnail"), SNAME("EditorIcons")));
		mode_list->set_icon(item_list->get_theme_icon(SNAME("FileList"), SNAME("EditorIcons")));
		if (is_layout_rtl()) {
			dir_prev->set_icon(item_list->get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
			dir_next->set_icon(item_list->get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
		} else {
			dir_prev->set_icon(item_list->get_theme_icon(SNAME("Back"), SNAME("EditorIcons")));
			dir_next->set_icon(item_list->get_theme_icon(SNAME("Forward"), SNAME("EditorIcons")));
		}
		dir_up->set_icon(item_list->get_theme_icon(SNAME("ArrowUp"), SNAME("EditorIcons")));
		refresh->set_icon(item_list->get_theme_icon(SNAME("Reload"), SNAME("EditorIcons")));
		favorite->set_icon(item_list->get_theme_icon(SNAME("Favorites"), SNAME("EditorIcons")));

		fav_up->set_icon(item_list->get_theme_icon(SNAME("MoveUp"), SNAME("EditorIcons")));
		fav_down->set_icon(item_list->get_theme_icon(SNAME("MoveDown"), SNAME("EditorIcons")));
		// DO NOT CALL UPDATE FILE LIST HERE, ALL HUNDREDS OF HIDDEN DIALOGS WILL RESPOND, CALL INVALIDATE INSTEAD
		invalidate();
	} else if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (!is_visible()) {
			set_process_unhandled_input(false);
		}
	}
}

void EditorFileDialog::unhandled_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
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
				set_show_hidden_files(!show_hidden_files);
				handled = true;
			}
			if (ED_IS_SHORTCUT("file_dialog/toggle_favorite", p_event)) {
				_favorite_pressed();
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
			if (ED_IS_SHORTCUT("file_dialog/delete", p_event)) {
				_delete_items();
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
				set_input_as_handled();
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
		if (item_list->is_selected(i)) {
			list.push_back(item_list->get_item_text(i));
		}
	}
	return list;
};

void EditorFileDialog::update_dir() {
	if (drives->is_visible()) {
		drives->select(dir_access->get_current_drive());
	}
	dir->set_text(dir_access->get_current_dir(false));

	// Disable "Open" button only when selecting file(s) mode.
	get_ok_button()->set_disabled(_is_open_should_be_disabled());
	switch (mode) {
		case FILE_MODE_OPEN_FILE:
		case FILE_MODE_OPEN_FILES:
			get_ok_button()->set_text(TTR("Open"));
			break;
		case FILE_MODE_OPEN_DIR:
			get_ok_button()->set_text(TTR("Select Current Folder"));
			break;
		case FILE_MODE_OPEN_ANY:
		case FILE_MODE_SAVE_FILE:
			// FIXME: Implement, or refactor to avoid duplication with set_mode
			break;
	}
}

void EditorFileDialog::_dir_submitted(String p_dir) {
	dir_access->change_dir(p_dir);
	invalidate();
	update_dir();
	_push_history();
}

void EditorFileDialog::_file_submitted(const String &p_file) {
	_action_pressed();
}

void EditorFileDialog::_save_confirm_pressed() {
	String f = dir_access->get_current_dir().plus_file(file->get_text());
	_save_to_recent();
	hide();
	emit_signal(SNAME("file_selected"), f);
}

void EditorFileDialog::_post_popup() {
	ConfirmationDialog::_post_popup();

	// Check if the current path doesn't exist and correct it.
	String current = dir_access->get_current_dir();
	while (!dir_access->dir_exists(current)) {
		current = current.get_base_dir();
	}
	set_current_dir(current);

	if (invalidated) {
		update_file_list();
		invalidated = false;
	}
	if (mode == FILE_MODE_SAVE_FILE) {
		file->grab_focus();
	} else {
		item_list->grab_focus();
	}

	if (mode == FILE_MODE_OPEN_DIR) {
		file_box->set_visible(false);
	} else {
		file_box->set_visible(true);
	}

	if (is_visible() && !get_current_file().is_empty()) {
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));
	}

	if (is_visible()) {
		Ref<Texture2D> folder = item_list->get_theme_icon(SNAME("folder"), SNAME("FileDialog"));
		const Color folder_color = item_list->get_theme_color(SNAME("folder_icon_modulate"), SNAME("FileDialog"));
		recent->clear();

		bool res = access == ACCESS_RESOURCES;
		Vector<String> recentd = EditorSettings::get_singleton()->get_recent_dirs();
		for (int i = 0; i < recentd.size(); i++) {
			bool cres = recentd[i].begins_with("res://");
			if (cres != res) {
				continue;
			}
			String name = recentd[i];
			if (res && name == "res://") {
				name = "/";
			} else {
				if (name.ends_with("/")) {
					name = name.substr(0, name.length() - 1);
				}
				name = name.get_file() + "/";
			}
			bool exists = dir_access->dir_exists(recentd[i]);
			if (!exists) {
				// Remove invalid directory from the list of Recent directories.
				recentd.remove_at(i--);
			} else {
				recent->add_item(name, folder);
				recent->set_item_metadata(recent->get_item_count() - 1, recentd[i]);
				recent->set_item_icon_modulate(recent->get_item_count() - 1, folder_color);
			}
		}
		EditorSettings::get_singleton()->set_recent_dirs(recentd);

		local_history.clear();
		local_history_pos = -1;
		_push_history();

		_update_favorites();
	}

	set_process_unhandled_input(true);
}

void EditorFileDialog::_thumbnail_result(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, const Variant &p_udata) {
	if (display_mode == DISPLAY_LIST || p_preview.is_null()) {
		return;
	}

	for (int i = 0; i < item_list->get_item_count(); i++) {
		Dictionary d = item_list->get_item_metadata(i);
		String pname = d["path"];
		if (pname == p_path) {
			item_list->set_item_icon(i, p_preview);
			item_list->set_item_tag_icon(i, Ref<Texture2D>());
		}
	}
}

void EditorFileDialog::_thumbnail_done(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, const Variant &p_udata) {
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
		preview->set_texture(Ref<Texture2D>());
	}
}

void EditorFileDialog::_request_single_thumbnail(const String &p_path) {
	if (!FileAccess::exists(p_path)) {
		return;
	}

	set_process(true);
	preview_waiting = true;
	preview_wheel_timeout = 0;
	EditorResourcePreview::get_singleton()->queue_resource_preview(p_path, this, "_thumbnail_done", p_path);
}

void EditorFileDialog::_action_pressed() {
	if (mode == FILE_MODE_OPEN_FILES) {
		String fbase = dir_access->get_current_dir();

		Vector<String> files;
		for (int i = 0; i < item_list->get_item_count(); i++) {
			if (item_list->is_selected(i)) {
				files.push_back(fbase.plus_file(item_list->get_item_text(i)));
			}
		}

		if (files.size()) {
			_save_to_recent();
			hide();
			emit_signal(SNAME("files_selected"), files);
		}

		return;
	}

	String f = dir_access->get_current_dir().plus_file(file->get_text());

	if ((mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_OPEN_FILE) && dir_access->file_exists(f)) {
		_save_to_recent();
		hide();
		emit_signal(SNAME("file_selected"), f);
	} else if (mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_OPEN_DIR) {
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
		hide();
		emit_signal(SNAME("dir_selected"), path);
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
					_request_single_thumbnail(get_current_dir().plus_file(f.get_file()));
					file->set_text(f.get_file());
					valid = true;
				}
			} else {
				valid = true;
			}
		}

		// Add first extension of filter if no valid extension is found.
		if (!valid) {
			int idx = filter->get_selected();
			String flt = filters[idx].get_slice(";", 0);
			String ext = flt.get_slice(",", 0).strip_edges().get_extension();
			f += "." + ext;
		}

		if (dir_access->file_exists(f) && !disable_overwrite_warning) {
			confirm_save->set_text(TTR("File exists, overwrite?"));
			confirm_save->popup_centered(Size2(200, 80));
		} else {
			_save_to_recent();
			hide();
			emit_signal(SNAME("file_selected"), f);
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
	if (current < 0 || current >= item_list->get_item_count()) {
		return;
	}

	Dictionary d = item_list->get_item_metadata(current);

	if (!d["dir"]) {
		file->set_text(d["name"]);
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));
	} else if (mode == FILE_MODE_OPEN_DIR) {
		get_ok_button()->set_text(TTR("Select This Folder"));
	}

	get_ok_button()->set_disabled(_is_open_should_be_disabled());
}

void EditorFileDialog::_multi_selected(int p_item, bool p_selected) {
	int current = p_item;
	if (current < 0 || current >= item_list->get_item_count()) {
		return;
	}

	Dictionary d = item_list->get_item_metadata(current);

	if (!d["dir"] && p_selected) {
		file->set_text(d["name"]);
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));
	}

	get_ok_button()->set_disabled(_is_open_should_be_disabled());
}

void EditorFileDialog::_items_clear_selection() {
	item_list->deselect_all();

	// If nothing is selected, then block Open button.
	switch (mode) {
		case FILE_MODE_OPEN_FILE:
		case FILE_MODE_OPEN_FILES:
			get_ok_button()->set_text(TTR("Open"));
			get_ok_button()->set_disabled(!item_list->is_anything_selected());
			break;

		case FILE_MODE_OPEN_DIR:
			get_ok_button()->set_disabled(false);
			get_ok_button()->set_text(TTR("Select Current Folder"));
			break;

		case FILE_MODE_OPEN_ANY:
		case FILE_MODE_SAVE_FILE:
			// FIXME: Implement, or refactor to avoid duplication with set_mode
			break;
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
	if (current < 0 || current >= item_list->get_item_count()) {
		return;
	}

	Dictionary d = item_list->get_item_metadata(current);

	if (d["dir"]) {
		dir_access->change_dir(d["name"]);
		call_deferred(SNAME("_update_file_list"));
		call_deferred(SNAME("_update_dir"));

		_push_history();

	} else {
		_action_pressed();
	}
}

void EditorFileDialog::_item_list_item_rmb_selected(int p_item, const Vector2 &p_pos) {
	// Right click on specific file(s) or folder(s).
	item_menu->clear();
	item_menu->reset_size();

	// Allow specific actions only on one item.
	bool single_item_selected = item_list->get_selected_items().size() == 1;

	// Disallow deleting the .import folder, Godot kills a cat if you do and it is possibly a senseless novice action.
	bool allow_delete = true;
	for (int i = 0; i < item_list->get_item_count(); i++) {
		if (!item_list->is_selected(i)) {
			continue;
		}
		Dictionary item_meta = item_list->get_item_metadata(i);
		if (String(item_meta["path"]).begins_with(ProjectSettings::get_singleton()->get_project_data_path())) {
			allow_delete = false;
			break;
		}
	}

	if (single_item_selected) {
		item_menu->add_icon_item(item_list->get_theme_icon(SNAME("ActionCopy"), SNAME("EditorIcons")), TTR("Copy Path"), ITEM_MENU_COPY_PATH);
	}
	if (allow_delete) {
		item_menu->add_icon_item(item_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), TTR("Delete"), ITEM_MENU_DELETE, Key::KEY_DELETE);
	}
	if (single_item_selected) {
		item_menu->add_separator();
		Dictionary item_meta = item_list->get_item_metadata(p_item);
		String item_text = item_meta["dir"] ? TTR("Open in File Manager") : TTR("Show in File Manager");
		item_menu->add_icon_item(item_list->get_theme_icon(SNAME("Filesystem"), SNAME("EditorIcons")), item_text, ITEM_MENU_SHOW_IN_EXPLORER);
	}

	if (item_menu->get_item_count() > 0) {
		item_menu->set_position(item_list->get_screen_position() + p_pos);
		item_menu->reset_size();
		item_menu->popup();
	}
}

void EditorFileDialog::_item_list_rmb_clicked(const Vector2 &p_pos) {
	// Right click on folder background. Deselect all files so that actions are applied on the current folder.
	for (int i = 0; i < item_list->get_item_count(); i++) {
		item_list->deselect(i);
	}

	item_menu->clear();
	item_menu->reset_size();

	if (can_create_dir) {
		item_menu->add_icon_item(item_list->get_theme_icon(SNAME("folder"), SNAME("FileDialog")), TTR("New Folder..."), ITEM_MENU_NEW_FOLDER, KeyModifierMask::CMD | Key::N);
	}
	item_menu->add_icon_item(item_list->get_theme_icon(SNAME("Reload"), SNAME("EditorIcons")), TTR("Refresh"), ITEM_MENU_REFRESH, Key::F5);
	item_menu->add_separator();
	item_menu->add_icon_item(item_list->get_theme_icon(SNAME("Filesystem"), SNAME("EditorIcons")), TTR("Open in File Manager"), ITEM_MENU_SHOW_IN_EXPLORER);

	item_menu->set_position(item_list->get_screen_position() + p_pos);
	item_menu->reset_size();
	item_menu->popup();
}

void EditorFileDialog::_item_menu_id_pressed(int p_option) {
	switch (p_option) {
		case ITEM_MENU_COPY_PATH: {
			Dictionary item_meta = item_list->get_item_metadata(item_list->get_current());
			DisplayServer::get_singleton()->clipboard_set(item_meta["path"]);
		} break;

		case ITEM_MENU_DELETE: {
			_delete_items();
		} break;

		case ITEM_MENU_REFRESH: {
			invalidate();
		} break;

		case ITEM_MENU_NEW_FOLDER: {
			_make_dir();
		} break;

		case ITEM_MENU_SHOW_IN_EXPLORER: {
			String path;
			int idx = item_list->get_current();
			if (idx == -1 || item_list->get_selected_items().size() == 0) {
				// Folder background was clicked. Open this folder.
				path = ProjectSettings::get_singleton()->globalize_path(dir_access->get_current_dir());
			} else {
				// Specific item was clicked. Open folders directly, or the folder containing a selected file.
				Dictionary item_meta = item_list->get_item_metadata(idx);
				path = ProjectSettings::get_singleton()->globalize_path(item_meta["path"]);
				if (!item_meta["dir"]) {
					path = path.get_base_dir();
				}
			}
			OS::get_singleton()->shell_open(String("file://") + path);
		} break;
	}
}

bool EditorFileDialog::_is_open_should_be_disabled() {
	if (mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_SAVE_FILE) {
		return false;
	}

	Vector<int> items = item_list->get_selected_items();
	if (items.size() == 0) {
		return mode != FILE_MODE_OPEN_DIR; // In "Open folder" mode, having nothing selected picks the current folder.
	}

	for (int i = 0; i < items.size(); i++) {
		Dictionary d = item_list->get_item_metadata(items.get(i));

		if (((mode == FILE_MODE_OPEN_FILE || mode == FILE_MODE_OPEN_FILES) && d["dir"]) || (mode == FILE_MODE_OPEN_DIR && !d["dir"])) {
			return true;
		}
	}

	return false;
}

void EditorFileDialog::update_file_name() {
	int idx = filter->get_selected() - 1;
	if ((idx == -1 && filter->get_item_count() == 2) || (filter->get_item_count() > 2 && idx >= 0 && idx < filter->get_item_count() - 2)) {
		if (idx == -1) {
			idx += 1;
		}
		String filter_str = filters[idx];
		String file_str = file->get_text();
		String base_name = file_str.get_basename();
		Vector<String> filter_substr = filter_str.split(";");
		if (filter_substr.size() >= 2) {
			file_str = base_name + "." + filter_substr[0].strip_edges().lstrip("*.").to_lower();
		} else {
			file_str = base_name + "." + filter_str.get_extension().strip_edges().to_lower();
		}
		file->set_text(file_str);
	}
}

// DO NOT USE THIS FUNCTION UNLESS NEEDED, CALL INVALIDATE() INSTEAD.
void EditorFileDialog::update_file_list() {
	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;
	Ref<Texture2D> folder_thumbnail;
	Ref<Texture2D> file_thumbnail;

	item_list->clear();

	// Scroll back to the top after opening a directory
	item_list->get_v_scroll_bar()->set_value(0);

	if (display_mode == DISPLAY_THUMBNAILS) {
		item_list->set_max_columns(0);
		item_list->set_icon_mode(ItemList::ICON_MODE_TOP);
		item_list->set_fixed_column_width(thumbnail_size * 3 / 2);
		item_list->set_max_text_lines(2);
		item_list->set_text_overrun_behavior(TextParagraph::OVERRUN_TRIM_ELLIPSIS);
		item_list->set_fixed_icon_size(Size2(thumbnail_size, thumbnail_size));

		if (thumbnail_size < 64) {
			folder_thumbnail = item_list->get_theme_icon(SNAME("FolderMediumThumb"), SNAME("EditorIcons"));
			file_thumbnail = item_list->get_theme_icon(SNAME("FileMediumThumb"), SNAME("EditorIcons"));
		} else {
			folder_thumbnail = item_list->get_theme_icon(SNAME("FolderBigThumb"), SNAME("EditorIcons"));
			file_thumbnail = item_list->get_theme_icon(SNAME("FileBigThumb"), SNAME("EditorIcons"));
		}

		preview_vb->hide();

	} else {
		item_list->set_icon_mode(ItemList::ICON_MODE_LEFT);
		item_list->set_max_columns(1);
		item_list->set_max_text_lines(1);
		item_list->set_fixed_column_width(0);
		item_list->set_fixed_icon_size(Size2());
		if (preview->get_texture().is_valid()) {
			preview_vb->show();
		}
	}

	String cdir = dir_access->get_current_dir();

	dir_access->list_dir_begin();

	Ref<Texture2D> folder = item_list->get_theme_icon(SNAME("folder"), SNAME("FileDialog"));
	const Color folder_color = item_list->get_theme_color(SNAME("folder_icon_modulate"), SNAME("FileDialog"));
	List<String> files;
	List<String> dirs;

	String item = dir_access->get_next();

	while (!item.is_empty()) {
		if (item == "." || item == "..") {
			item = dir_access->get_next();
			continue;
		}

		if (show_hidden_files || !dir_access->current_is_hidden()) {
			if (!dir_access->current_is_dir()) {
				files.push_back(item);
			} else {
				dirs.push_back(item);
			}
		}
		item = dir_access->get_next();
	}

	dirs.sort_custom<NaturalNoCaseComparator>();
	files.sort_custom<NaturalNoCaseComparator>();

	while (!dirs.is_empty()) {
		const String &dir_name = dirs.front()->get();

		item_list->add_item(dir_name);

		if (display_mode == DISPLAY_THUMBNAILS) {
			item_list->set_item_icon(item_list->get_item_count() - 1, folder_thumbnail);
		} else {
			item_list->set_item_icon(item_list->get_item_count() - 1, folder);
		}

		Dictionary d;
		d["name"] = dir_name;
		d["path"] = cdir.plus_file(dir_name);
		d["dir"] = true;

		item_list->set_item_metadata(item_list->get_item_count() - 1, d);
		item_list->set_item_icon_modulate(item_list->get_item_count() - 1, folder_color);

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

	while (!files.is_empty()) {
		bool match = patterns.is_empty();

		for (const String &E : patterns) {
			if (files.front()->get().matchn(E)) {
				match = true;
				break;
			}
		}

		if (match) {
			item_list->add_item(files.front()->get());

			if (get_icon_func) {
				Ref<Texture2D> icon = get_icon_func(cdir.plus_file(files.front()->get()));
				if (display_mode == DISPLAY_THUMBNAILS) {
					item_list->set_item_icon(item_list->get_item_count() - 1, file_thumbnail);
					item_list->set_item_tag_icon(item_list->get_item_count() - 1, icon);
				} else {
					item_list->set_item_icon(item_list->get_item_count() - 1, icon);
				}
			}

			Dictionary d;
			d["name"] = files.front()->get();
			d["dir"] = false;
			String fullpath = cdir.plus_file(files.front()->get());
			d["path"] = fullpath;
			item_list->set_item_metadata(item_list->get_item_count() - 1, d);

			if (display_mode == DISPLAY_THUMBNAILS) {
				EditorResourcePreview::get_singleton()->queue_resource_preview(fullpath, this, "_thumbnail_result", fullpath);
			}

			if (file->get_text() == files.front()->get()) {
				item_list->set_current(item_list->get_item_count() - 1);
			}
		}

		files.pop_front();
	}

	if (favorites->get_current() >= 0) {
		favorites->deselect(favorites->get_current());
	}

	favorite->set_pressed(false);
	fav_up->set_disabled(true);
	fav_down->set_disabled(true);
	get_ok_button()->set_disabled(_is_open_should_be_disabled());
	for (int i = 0; i < favorites->get_item_count(); i++) {
		if (favorites->get_item_metadata(i) == cdir || favorites->get_item_metadata(i) == cdir + "/") {
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
}

void EditorFileDialog::_filter_selected(int) {
	update_file_name();
	update_file_list();
}

void EditorFileDialog::update_filters() {
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

		filter->add_item(TTR("All Recognized") + " (" + all_filters + ")");
	}
	for (int i = 0; i < filters.size(); i++) {
		String flt = filters[i].get_slice(";", 0).strip_edges();
		String desc = filters[i].get_slice(";", 1).strip_edges();
		if (desc.length()) {
			filter->add_item(desc + " (" + flt + ")");
		} else {
			filter->add_item("(" + flt + ")");
		}
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
	return dir_access->get_current_dir();
}

String EditorFileDialog::get_current_file() const {
	return file->get_text();
}

String EditorFileDialog::get_current_path() const {
	return dir_access->get_current_dir().plus_file(file->get_text());
}

void EditorFileDialog::set_current_dir(const String &p_dir) {
	if (p_dir.is_relative_path()) {
		dir_access->change_dir(OS::get_singleton()->get_resource_dir());
	}
	dir_access->change_dir(p_dir);
	update_dir();
	invalidate();
}

void EditorFileDialog::set_current_file(const String &p_file) {
	file->set_text(p_file);
	update_dir();
	invalidate();
	int lp = p_file.rfind(".");
	if (lp != -1) {
		file->select(0, lp);
		file->grab_focus();
	}

	if (is_visible()) {
		_request_single_thumbnail(get_current_dir().plus_file(get_current_file()));
	}
}

void EditorFileDialog::set_current_path(const String &p_path) {
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

void EditorFileDialog::set_file_mode(FileMode p_mode) {
	mode = p_mode;
	switch (mode) {
		case FILE_MODE_OPEN_FILE:
			get_ok_button()->set_text(TTR("Open"));
			set_title(TTR("Open a File"));
			can_create_dir = false;
			break;
		case FILE_MODE_OPEN_FILES:
			get_ok_button()->set_text(TTR("Open"));
			set_title(TTR("Open File(s)"));
			can_create_dir = false;
			break;
		case FILE_MODE_OPEN_DIR:
			get_ok_button()->set_text(TTR("Open"));
			set_title(TTR("Open a Directory"));
			can_create_dir = true;
			break;
		case FILE_MODE_OPEN_ANY:
			get_ok_button()->set_text(TTR("Open"));
			set_title(TTR("Open a File or Directory"));
			can_create_dir = true;
			break;
		case FILE_MODE_SAVE_FILE:
			get_ok_button()->set_text(TTR("Save"));
			set_title(TTR("Save a File"));
			can_create_dir = true;
			break;
	}

	if (mode == FILE_MODE_OPEN_FILES) {
		item_list->set_select_mode(ItemList::SELECT_MULTI);
	} else {
		item_list->set_select_mode(ItemList::SELECT_SINGLE);
	}

	if (can_create_dir) {
		makedir->show();
	} else {
		makedir->hide();
	}
}

EditorFileDialog::FileMode EditorFileDialog::get_file_mode() const {
	return mode;
}

void EditorFileDialog::set_access(Access p_access) {
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

void EditorFileDialog::invalidate() {
	if (is_visible()) {
		update_file_list();
		_update_favorites();
		invalidated = false;
	} else {
		invalidated = true;
	}
}

EditorFileDialog::Access EditorFileDialog::get_access() const {
	return access;
}

void EditorFileDialog::_make_dir_confirm() {
	Error err = dir_access->make_dir(makedirname->get_text().strip_edges());
	if (err == OK) {
		dir_access->change_dir(makedirname->get_text().strip_edges());
		invalidate();
		update_filters();
		update_dir();
		_push_history();
		EditorFileSystem::get_singleton()->scan_changes(); //we created a dir, so rescan changes
	} else {
		mkdirerr->popup_centered(Size2(250, 50) * EDSCALE);
	}
	makedirname->set_text(""); // reset label
}

void EditorFileDialog::_make_dir() {
	makedialog->popup_centered(Size2(250, 80) * EDSCALE);
	makedirname->grab_focus();
}

void EditorFileDialog::_delete_items() {
	// Collect the selected folders and files to delete and check them in the deletion dependency dialog.
	Vector<String> folders;
	Vector<String> files;
	for (int i = 0; i < item_list->get_item_count(); i++) {
		if (!item_list->is_selected(i)) {
			continue;
		}
		Dictionary item_meta = item_list->get_item_metadata(i);
		if (item_meta["dir"]) {
			folders.push_back(item_meta["path"]);
		} else {
			files.push_back(item_meta["path"]);
		}
	}
	if (folders.size() + files.size() > 0) {
		remove_dialog->reset_size();
		remove_dialog->show(folders, files);
	}
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
		Node *dp = drives->get_parent();
		if (dp) {
			dp->remove_child(drives);
		}
		dp = dir_access->drives_are_shortcuts() ? shortcuts_container : drives_container;
		dp->add_child(drives);
		drives->show();

		for (int i = 0; i < dir_access->get_drive_count(); i++) {
			String d = dir_access->get_drive(i);
			drives->add_item(dir_access->get_drive(i));
		}

		drives->select(dir_access->get_current_drive());
	}
}

void EditorFileDialog::_favorite_selected(int p_idx) {
	dir_access->change_dir(favorites->get_item_metadata(p_idx));
	update_dir();
	invalidate();
	_push_history();
}

void EditorFileDialog::_favorite_move_up() {
	int current = favorites->get_current();

	if (current > 0 && current < favorites->get_item_count()) {
		Vector<String> favorited = EditorSettings::get_singleton()->get_favorites();

		int a_idx = favorited.find(String(favorites->get_item_metadata(current - 1)));
		int b_idx = favorited.find(String(favorites->get_item_metadata(current)));

		if (a_idx == -1 || b_idx == -1) {
			return;
		}
		SWAP(favorited.write[a_idx], favorited.write[b_idx]);

		EditorSettings::get_singleton()->set_favorites(favorited);

		_update_favorites();
		update_file_list();
	}
}

void EditorFileDialog::_favorite_move_down() {
	int current = favorites->get_current();

	if (current >= 0 && current < favorites->get_item_count() - 1) {
		Vector<String> favorited = EditorSettings::get_singleton()->get_favorites();

		int a_idx = favorited.find(String(favorites->get_item_metadata(current + 1)));
		int b_idx = favorited.find(String(favorites->get_item_metadata(current)));

		if (a_idx == -1 || b_idx == -1) {
			return;
		}
		SWAP(favorited.write[a_idx], favorited.write[b_idx]);

		EditorSettings::get_singleton()->set_favorites(favorited);

		_update_favorites();
		update_file_list();
	}
}

void EditorFileDialog::_update_favorites() {
	bool res = access == ACCESS_RESOURCES;

	String current = get_current_dir();
	Ref<Texture2D> folder_icon = item_list->get_theme_icon(SNAME("Folder"), SNAME("EditorIcons"));
	const Color folder_color = item_list->get_theme_color(SNAME("folder_icon_modulate"), SNAME("FileDialog"));
	favorites->clear();

	favorite->set_pressed(false);

	Vector<String> favorited = EditorSettings::get_singleton()->get_favorites();
	for (int i = 0; i < favorited.size(); i++) {
		bool cres = favorited[i].begins_with("res://");
		if (cres != res) {
			continue;
		}
		String name = favorited[i];
		bool setthis = false;

		if (res && name == "res://") {
			if (name == current) {
				setthis = true;
			}
			name = "/";

			favorites->add_item(name, folder_icon);
		} else if (name.ends_with("/")) {
			if (name == current || name == current + "/") {
				setthis = true;
			}
			name = name.substr(0, name.length() - 1);
			name = name.get_file();

			favorites->add_item(name, folder_icon);
		} else {
			continue; // We don't handle favorite files here.
		}

		favorites->set_item_metadata(favorites->get_item_count() - 1, favorited[i]);
		favorites->set_item_icon_modulate(favorites->get_item_count() - 1, folder_color);

		if (setthis) {
			favorite->set_pressed(true);
			favorites->set_current(favorites->get_item_count() - 1);
			recent->deselect_all();
		}
	}
}

void EditorFileDialog::_favorite_pressed() {
	bool res = access == ACCESS_RESOURCES;

	String cd = get_current_dir();
	if (!cd.ends_with("/")) {
		cd += "/";
	}

	Vector<String> favorited = EditorSettings::get_singleton()->get_favorites();

	bool found = false;
	for (int i = 0; i < favorited.size(); i++) {
		bool cres = favorited[i].begins_with("res://");
		if (cres != res) {
			continue;
		}

		if (favorited[i] == cd) {
			found = true;
			break;
		}
	}

	if (found) {
		favorited.erase(cd);
	} else {
		favorited.push_back(cd);
	}

	EditorSettings::get_singleton()->set_favorites(favorited);

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
	if (display_mode == p_mode) {
		return;
	}
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
	ClassDB::bind_method(D_METHOD("_cancel_pressed"), &EditorFileDialog::_cancel_pressed);

	ClassDB::bind_method(D_METHOD("clear_filters"), &EditorFileDialog::clear_filters);
	ClassDB::bind_method(D_METHOD("add_filter", "filter"), &EditorFileDialog::add_filter);
	ClassDB::bind_method(D_METHOD("get_current_dir"), &EditorFileDialog::get_current_dir);
	ClassDB::bind_method(D_METHOD("get_current_file"), &EditorFileDialog::get_current_file);
	ClassDB::bind_method(D_METHOD("get_current_path"), &EditorFileDialog::get_current_path);
	ClassDB::bind_method(D_METHOD("set_current_dir", "dir"), &EditorFileDialog::set_current_dir);
	ClassDB::bind_method(D_METHOD("set_current_file", "file"), &EditorFileDialog::set_current_file);
	ClassDB::bind_method(D_METHOD("set_current_path", "path"), &EditorFileDialog::set_current_path);
	ClassDB::bind_method(D_METHOD("set_file_mode", "mode"), &EditorFileDialog::set_file_mode);
	ClassDB::bind_method(D_METHOD("get_file_mode"), &EditorFileDialog::get_file_mode);
	ClassDB::bind_method(D_METHOD("get_vbox"), &EditorFileDialog::get_vbox);
	ClassDB::bind_method(D_METHOD("set_access", "access"), &EditorFileDialog::set_access);
	ClassDB::bind_method(D_METHOD("get_access"), &EditorFileDialog::get_access);
	ClassDB::bind_method(D_METHOD("set_show_hidden_files", "show"), &EditorFileDialog::set_show_hidden_files);
	ClassDB::bind_method(D_METHOD("is_showing_hidden_files"), &EditorFileDialog::is_showing_hidden_files);
	ClassDB::bind_method(D_METHOD("_update_file_name"), &EditorFileDialog::update_file_name);
	ClassDB::bind_method(D_METHOD("_update_dir"), &EditorFileDialog::update_dir);
	ClassDB::bind_method(D_METHOD("_update_file_list"), &EditorFileDialog::update_file_list);
	ClassDB::bind_method(D_METHOD("_thumbnail_done"), &EditorFileDialog::_thumbnail_done);
	ClassDB::bind_method(D_METHOD("set_display_mode", "mode"), &EditorFileDialog::set_display_mode);
	ClassDB::bind_method(D_METHOD("get_display_mode"), &EditorFileDialog::get_display_mode);
	ClassDB::bind_method(D_METHOD("_thumbnail_result"), &EditorFileDialog::_thumbnail_result);
	ClassDB::bind_method(D_METHOD("set_disable_overwrite_warning", "disable"), &EditorFileDialog::set_disable_overwrite_warning);
	ClassDB::bind_method(D_METHOD("is_overwrite_warning_disabled"), &EditorFileDialog::is_overwrite_warning_disabled);

	ClassDB::bind_method(D_METHOD("invalidate"), &EditorFileDialog::invalidate);

	ADD_SIGNAL(MethodInfo("file_selected", PropertyInfo(Variant::STRING, "path")));
	ADD_SIGNAL(MethodInfo("files_selected", PropertyInfo(Variant::PACKED_STRING_ARRAY, "paths")));
	ADD_SIGNAL(MethodInfo("dir_selected", PropertyInfo(Variant::STRING, "dir")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "access", PROPERTY_HINT_ENUM, "Resources,User data,File system"), "set_access", "get_access");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "display_mode", PROPERTY_HINT_ENUM, "Thumbnails,List"), "set_display_mode", "get_display_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "file_mode", PROPERTY_HINT_ENUM, "Open one,Open many,Open folder,Open any,Save"), "set_file_mode", "get_file_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_dir", PROPERTY_HINT_DIR), "set_current_dir", "get_current_dir");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_file", PROPERTY_HINT_FILE, "*"), "set_current_file", "get_current_file");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_path"), "set_current_path", "get_current_path");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_hidden_files"), "set_show_hidden_files", "is_showing_hidden_files");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disable_overwrite_warning"), "set_disable_overwrite_warning", "is_overwrite_warning_disabled");

	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_FILE);
	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_FILES);
	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_DIR);
	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_ANY);
	BIND_ENUM_CONSTANT(FILE_MODE_SAVE_FILE);

	BIND_ENUM_CONSTANT(ACCESS_RESOURCES);
	BIND_ENUM_CONSTANT(ACCESS_USERDATA);
	BIND_ENUM_CONSTANT(ACCESS_FILESYSTEM);

	BIND_ENUM_CONSTANT(DISPLAY_THUMBNAILS);
	BIND_ENUM_CONSTANT(DISPLAY_LIST);
}

void EditorFileDialog::set_show_hidden_files(bool p_show) {
	if (p_show == show_hidden_files) {
		return;
	}

	EditorSettings::get_singleton()->set("filesystem/file_dialog/show_hidden_files", p_show);
	show_hidden_files = p_show;
	show_hidden->set_pressed(p_show);
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
			recent.remove_at(i);
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

	mode = FILE_MODE_SAVE_FILE;
	set_title(TTR("Save a File"));

	ED_SHORTCUT("file_dialog/go_back", TTR("Go Back"), KeyModifierMask::ALT | Key::LEFT);
	ED_SHORTCUT("file_dialog/go_forward", TTR("Go Forward"), KeyModifierMask::ALT | Key::RIGHT);
	ED_SHORTCUT("file_dialog/go_up", TTR("Go Up"), KeyModifierMask::ALT | Key::UP);
	ED_SHORTCUT("file_dialog/refresh", TTR("Refresh"), Key::F5);
	ED_SHORTCUT("file_dialog/toggle_hidden_files", TTR("Toggle Hidden Files"), KeyModifierMask::CMD | Key::H);
	ED_SHORTCUT("file_dialog/toggle_favorite", TTR("Toggle Favorite"), KeyModifierMask::ALT | Key::F);
	ED_SHORTCUT("file_dialog/toggle_mode", TTR("Toggle Mode"), KeyModifierMask::ALT | Key::V);
	ED_SHORTCUT("file_dialog/create_folder", TTR("Create Folder"), KeyModifierMask::CMD | Key::N);
	ED_SHORTCUT("file_dialog/delete", TTR("Delete"), Key::KEY_DELETE);
	ED_SHORTCUT("file_dialog/focus_path", TTR("Focus Path"), KeyModifierMask::CMD | Key::D);
	ED_SHORTCUT("file_dialog/move_favorite_up", TTR("Move Favorite Up"), KeyModifierMask::CMD | Key::UP);
	ED_SHORTCUT("file_dialog/move_favorite_down", TTR("Move Favorite Down"), KeyModifierMask::CMD | Key::DOWN);

	HBoxContainer *pathhb = memnew(HBoxContainer);

	dir_prev = memnew(Button);
	dir_prev->set_flat(true);
	dir_prev->set_tooltip(TTR("Go to previous folder."));
	dir_next = memnew(Button);
	dir_next->set_flat(true);
	dir_next->set_tooltip(TTR("Go to next folder."));
	dir_up = memnew(Button);
	dir_up->set_flat(true);
	dir_up->set_tooltip(TTR("Go to parent folder."));

	pathhb->add_child(dir_prev);
	pathhb->add_child(dir_next);
	pathhb->add_child(dir_up);

	dir_prev->connect("pressed", callable_mp(this, &EditorFileDialog::_go_back));
	dir_next->connect("pressed", callable_mp(this, &EditorFileDialog::_go_forward));
	dir_up->connect("pressed", callable_mp(this, &EditorFileDialog::_go_up));

	Label *l = memnew(Label(TTR("Path:")));
	l->set_theme_type_variation("HeaderSmall");
	pathhb->add_child(l);

	drives_container = memnew(HBoxContainer);
	pathhb->add_child(drives_container);

	dir = memnew(LineEdit);
	dir->set_structured_text_bidi_override(Control::STRUCTURED_TEXT_FILE);
	pathhb->add_child(dir);
	dir->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	refresh = memnew(Button);
	refresh->set_flat(true);
	refresh->set_tooltip(TTR("Refresh files."));
	refresh->connect("pressed", callable_mp(this, &EditorFileDialog::update_file_list));
	pathhb->add_child(refresh);

	favorite = memnew(Button);
	favorite->set_flat(true);
	favorite->set_toggle_mode(true);
	favorite->set_tooltip(TTR("(Un)favorite current folder."));
	favorite->connect("pressed", callable_mp(this, &EditorFileDialog::_favorite_pressed));
	pathhb->add_child(favorite);

	show_hidden = memnew(Button);
	show_hidden->set_flat(true);
	show_hidden->set_toggle_mode(true);
	show_hidden->set_pressed(is_showing_hidden_files());
	show_hidden->set_tooltip(TTR("Toggle the visibility of hidden files."));
	show_hidden->connect("toggled", callable_mp(this, &EditorFileDialog::set_show_hidden_files));
	pathhb->add_child(show_hidden);

	pathhb->add_child(memnew(VSeparator));

	Ref<ButtonGroup> view_mode_group;
	view_mode_group.instantiate();

	mode_thumbnails = memnew(Button);
	mode_thumbnails->set_flat(true);
	mode_thumbnails->connect("pressed", callable_mp(this, &EditorFileDialog::set_display_mode), varray(DISPLAY_THUMBNAILS));
	mode_thumbnails->set_toggle_mode(true);
	mode_thumbnails->set_pressed(display_mode == DISPLAY_THUMBNAILS);
	mode_thumbnails->set_button_group(view_mode_group);
	mode_thumbnails->set_tooltip(TTR("View items as a grid of thumbnails."));
	pathhb->add_child(mode_thumbnails);

	mode_list = memnew(Button);
	mode_list->set_flat(true);
	mode_list->connect("pressed", callable_mp(this, &EditorFileDialog::set_display_mode), varray(DISPLAY_LIST));
	mode_list->set_toggle_mode(true);
	mode_list->set_pressed(display_mode == DISPLAY_LIST);
	mode_list->set_button_group(view_mode_group);
	mode_list->set_tooltip(TTR("View items as a list."));
	pathhb->add_child(mode_list);

	shortcuts_container = memnew(HBoxContainer);
	pathhb->add_child(shortcuts_container);

	drives = memnew(OptionButton);
	drives->connect("item_selected", callable_mp(this, &EditorFileDialog::_select_drive));
	pathhb->add_child(drives);

	makedir = memnew(Button);
	makedir->set_text(TTR("Create Folder"));
	makedir->connect("pressed", callable_mp(this, &EditorFileDialog::_make_dir));
	pathhb->add_child(makedir);

	list_hb = memnew(HSplitContainer);

	vbc->add_child(pathhb);
	vbc->add_child(list_hb);
	list_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	VSplitContainer *vsc = memnew(VSplitContainer);
	list_hb->add_child(vsc);

	VBoxContainer *fav_vb = memnew(VBoxContainer);
	vsc->add_child(fav_vb);
	fav_vb->set_custom_minimum_size(Size2(150, 100) * EDSCALE);
	fav_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	HBoxContainer *fav_hb = memnew(HBoxContainer);
	fav_vb->add_child(fav_hb);

	l = memnew(Label(TTR("Favorites:")));
	l->set_theme_type_variation("HeaderSmall");
	fav_hb->add_child(l);

	fav_hb->add_spacer();
	fav_up = memnew(Button);
	fav_up->set_flat(true);
	fav_hb->add_child(fav_up);
	fav_up->connect("pressed", callable_mp(this, &EditorFileDialog::_favorite_move_up));
	fav_down = memnew(Button);
	fav_down->set_flat(true);
	fav_hb->add_child(fav_down);
	fav_down->connect("pressed", callable_mp(this, &EditorFileDialog::_favorite_move_down));

	favorites = memnew(ItemList);
	fav_vb->add_child(favorites);
	favorites->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	favorites->connect("item_selected", callable_mp(this, &EditorFileDialog::_favorite_selected));

	VBoxContainer *rec_vb = memnew(VBoxContainer);
	vsc->add_child(rec_vb);
	rec_vb->set_custom_minimum_size(Size2(150, 100) * EDSCALE);
	rec_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	recent = memnew(ItemList);
	recent->set_allow_reselect(true);
	rec_vb->add_margin_child(TTR("Recent:"), recent, true);
	recent->connect("item_selected", callable_mp(this, &EditorFileDialog::_recent_selected));

	VBoxContainer *item_vb = memnew(VBoxContainer);
	list_hb->add_child(item_vb);
	item_vb->set_custom_minimum_size(Size2(320, 0) * EDSCALE);

	HBoxContainer *preview_hb = memnew(HBoxContainer);
	preview_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	item_vb->add_child(preview_hb);

	VBoxContainer *list_vb = memnew(VBoxContainer);
	list_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	l = memnew(Label(TTR("Directories & Files:")));
	l->set_theme_type_variation("HeaderSmall");
	list_vb->add_child(l);
	preview_hb->add_child(list_vb);

	// Item (files and folders) list with context menu.

	item_list = memnew(ItemList);
	item_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	item_list->connect("item_rmb_selected", callable_mp(this, &EditorFileDialog::_item_list_item_rmb_selected));
	item_list->connect("rmb_clicked", callable_mp(this, &EditorFileDialog::_item_list_rmb_clicked));
	item_list->set_allow_rmb_select(true);

	list_vb->add_child(item_list);

	item_menu = memnew(PopupMenu);
	item_menu->connect("id_pressed", callable_mp(this, &EditorFileDialog::_item_menu_id_pressed));
	add_child(item_menu);

	// Other stuff.

	preview_vb = memnew(VBoxContainer);
	preview_hb->add_child(preview_vb);
	CenterContainer *prev_cc = memnew(CenterContainer);
	preview_vb->add_margin_child(TTR("Preview:"), prev_cc);
	preview = memnew(TextureRect);
	prev_cc->add_child(preview);
	preview_vb->hide();

	file_box = memnew(HBoxContainer);

	l = memnew(Label(TTR("File:")));
	l->set_theme_type_variation("HeaderSmall");
	file_box->add_child(l);

	file = memnew(LineEdit);
	file->set_structured_text_bidi_override(Control::STRUCTURED_TEXT_FILE);
	file->set_stretch_ratio(4);
	file->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	file_box->add_child(file);
	filter = memnew(OptionButton);
	filter->set_stretch_ratio(3);
	filter->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	filter->set_clip_text(true); // Too many extensions overflow it.
	file_box->add_child(filter);
	file_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	item_vb->add_child(file_box);

	dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	access = ACCESS_RESOURCES;
	_update_drives();

	connect("confirmed", callable_mp(this, &EditorFileDialog::_action_pressed));
	item_list->connect("item_selected", callable_mp(this, &EditorFileDialog::_item_selected), varray(), CONNECT_DEFERRED);
	item_list->connect("multi_selected", callable_mp(this, &EditorFileDialog::_multi_selected), varray(), CONNECT_DEFERRED);
	item_list->connect("item_activated", callable_mp(this, &EditorFileDialog::_item_dc_selected), varray());
	item_list->connect("nothing_selected", callable_mp(this, &EditorFileDialog::_items_clear_selection));
	dir->connect("text_submitted", callable_mp(this, &EditorFileDialog::_dir_submitted));
	file->connect("text_submitted", callable_mp(this, &EditorFileDialog::_file_submitted));
	filter->connect("item_selected", callable_mp(this, &EditorFileDialog::_filter_selected));

	confirm_save = memnew(ConfirmationDialog);
	//confirm_save->set_as_top_level(true);
	add_child(confirm_save);
	confirm_save->connect("confirmed", callable_mp(this, &EditorFileDialog::_save_confirm_pressed));

	remove_dialog = memnew(DependencyRemoveDialog);
	add_child(remove_dialog);

	makedialog = memnew(ConfirmationDialog);
	makedialog->set_title(TTR("Create Folder"));
	VBoxContainer *makevb = memnew(VBoxContainer);
	makedialog->add_child(makevb);

	makedirname = memnew(LineEdit);
	makedirname->set_structured_text_bidi_override(Control::STRUCTURED_TEXT_FILE);
	makevb->add_margin_child(TTR("Name:"), makedirname);
	add_child(makedialog);
	makedialog->register_text_enter(makedirname);
	makedialog->connect("confirmed", callable_mp(this, &EditorFileDialog::_make_dir_confirm));
	mkdirerr = memnew(AcceptDialog);
	mkdirerr->set_text(TTR("Could not create folder."));
	add_child(mkdirerr);

	update_filters();
	update_dir();

	set_hide_on_ok(false);
	vbox = vbc;

	invalidated = true;
	if (register_func) {
		register_func(this);
	}

	preview_wheel_timeout = 0;
	preview_wheel_index = 0;
	preview_waiting = false;
}

EditorFileDialog::~EditorFileDialog() {
	if (unregister_func) {
		unregister_func(this);
	}
	memdelete(dir_access);
}
