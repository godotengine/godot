/**************************************************************************/
/*  file_dialog.cpp                                                       */
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

#include "file_dialog.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/os/keyboard.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/theme/theme_db.h"

void FileDialog::popup_file_dialog() {
	popup_centered_clamped(Size2i(700, 500), 0.8f);
	_focus_file_text();
}

void FileDialog::_focus_file_text() {
	int lp = filename_edit->get_text().rfind_char('.');
	if (lp != -1) {
		filename_edit->select(0, lp);
		if (filename_edit->is_inside_tree() && !is_part_of_edited_scene()) {
			filename_edit->grab_focus();
		}
	}
}

void FileDialog::_native_popup() {
	// Show native dialog directly.
	String root;
	if (!root_prefix.is_empty()) {
		root = ProjectSettings::get_singleton()->globalize_path(root_prefix);
	} else if (access == ACCESS_RESOURCES) {
		root = ProjectSettings::get_singleton()->get_resource_path();
	} else if (access == ACCESS_USERDATA) {
		root = OS::get_singleton()->get_user_data_dir();
	}

	// Attach native file dialog to first persistent parent window.
	Window *w = (is_transient() || is_transient_to_focused()) ? get_parent_visible_window() : nullptr;
	while (w && w->get_flag(FLAG_POPUP) && w->get_parent_visible_window()) {
		w = w->get_parent_visible_window();
	}
	DisplayServer::WindowID wid = w ? w->get_window_id() : DisplayServer::INVALID_WINDOW_ID;

	if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_DIALOG_FILE_EXTRA)) {
		DisplayServer::get_singleton()->file_dialog_with_options_show(get_translated_title(), ProjectSettings::get_singleton()->globalize_path(full_dir), root, filename_edit->get_text().get_file(), show_hidden_files, DisplayServer::FileDialogMode(mode), processed_filters, _get_options(), callable_mp(this, &FileDialog::_native_dialog_cb_with_options), wid);
	} else {
		DisplayServer::get_singleton()->file_dialog_show(get_translated_title(), ProjectSettings::get_singleton()->globalize_path(full_dir), filename_edit->get_text().get_file(), show_hidden_files, DisplayServer::FileDialogMode(mode), processed_filters, callable_mp(this, &FileDialog::_native_dialog_cb), wid);
	}
}

bool FileDialog::_can_use_native_popup() {
	if (access == ACCESS_RESOURCES || access == ACCESS_USERDATA || options.size() > 0) {
		return DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_DIALOG_FILE_EXTRA);
	}
	return DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_DIALOG_FILE);
}

void FileDialog::popup(const Rect2i &p_rect) {
	_update_option_controls();

#ifdef TOOLS_ENABLED
	if (is_part_of_edited_scene()) {
		ConfirmationDialog::popup(p_rect);
		return;
	}
#endif

	if (_can_use_native_popup() && (use_native_dialog || OS::get_singleton()->is_sandboxed())) {
		_native_popup();
	} else {
		ConfirmationDialog::popup(p_rect);
	}
}

void FileDialog::set_visible(bool p_visible) {
	if (p_visible) {
		_update_option_controls();
	}

#ifdef TOOLS_ENABLED
	if (is_part_of_edited_scene()) {
		ConfirmationDialog::set_visible(p_visible);
		return;
	}
#endif

	if (_can_use_native_popup() && (use_native_dialog || OS::get_singleton()->is_sandboxed())) {
		if (p_visible) {
			_native_popup();
		}
	} else {
		ConfirmationDialog::set_visible(p_visible);
	}
}

void FileDialog::_native_dialog_cb(bool p_ok, const Vector<String> &p_files, int p_filter) {
	_native_dialog_cb_with_options(p_ok, p_files, p_filter, Dictionary());
}

void FileDialog::_native_dialog_cb_with_options(bool p_ok, const Vector<String> &p_files, int p_filter, const Dictionary &p_selected_options) {
	if (!p_ok) {
		filename_edit->set_text("");
		emit_signal(SNAME("canceled"));
		return;
	}

	if (p_files.is_empty()) {
		return;
	}

	Vector<String> files = p_files;
	if (access != ACCESS_FILESYSTEM) {
		for (String &file_name : files) {
			file_name = ProjectSettings::get_singleton()->localize_path(file_name);
		}
	}
	selected_options = p_selected_options;

	String f = files[0];
	filter->select(p_filter);
	directory_edit->set_text(f.get_base_dir());
	filename_edit->set_text(f.get_file());
	_change_dir(f.get_base_dir());

	if (mode == FILE_MODE_OPEN_FILES) {
		emit_signal(SNAME("files_selected"), files);
	} else {
		if (mode == FILE_MODE_SAVE_FILE) {
			bool valid = false;

			if (p_filter == filter->get_item_count() - 1) {
				valid = true; // Match none.
			} else if (filters.size() > 1 && p_filter == 0) {
				// Match all filters.
				for (int i = 0; i < filters.size(); i++) {
					String flt = filters[i].get_slicec(';', 0);
					for (int j = 0; j < flt.get_slice_count(","); j++) {
						String str = flt.get_slicec(',', j).strip_edges();
						if (f.matchn(str)) {
							valid = true;
							break;
						}
					}
					if (valid) {
						break;
					}
				}
			} else {
				int idx = p_filter;
				if (filters.size() > 1) {
					idx--;
				}
				if (idx >= 0 && idx < filters.size()) {
					String flt = filters[idx].get_slicec(';', 0);
					int filter_slice_count = flt.get_slice_count(",");
					for (int j = 0; j < filter_slice_count; j++) {
						String str = flt.get_slicec(',', j).strip_edges();
						if (f.matchn(str)) {
							valid = true;
							break;
						}
					}

					if (!valid && filter_slice_count > 0) {
						String str = flt.get_slicec(',', 0).strip_edges();
						f += str.substr(1);
						filename_edit->set_text(f.get_file());
						valid = true;
					}
				} else {
					valid = true;
				}
			}

			// Add first extension of filter if no valid extension is found.
			if (!valid) {
				int idx = p_filter;
				String flt = filters[idx].get_slicec(';', 0);
				String ext = flt.get_slicec(',', 0).strip_edges().get_extension();
				f += "." + ext;
			}
			emit_signal(SNAME("file_selected"), f);
		} else if ((mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_OPEN_FILE) && dir_access->file_exists(f)) {
			emit_signal(SNAME("file_selected"), f);
		} else if (mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_OPEN_DIR) {
			emit_signal(SNAME("dir_selected"), f);
		}
	}
}

void FileDialog::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "dialog_text") {
		// File dialogs have a custom layout, and dialog nodes can't have both a text and a layout.
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void FileDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
#ifdef TOOLS_ENABLED
			if (is_part_of_edited_scene()) {
				return;
			}
#endif

			// Replace the built-in dialog with the native one if it started visible.
			if (is_visible() && _can_use_native_popup() && (use_native_dialog || OS::get_singleton()->is_sandboxed())) {
				ConfirmationDialog::set_visible(false);
				_native_popup();
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				set_process_shortcut_input(false);
			}

			invalidate(); // Put it here to preview in the editor.
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			if (main_vbox->is_layout_rtl()) {
				_setup_button(dir_prev, theme_cache.forward_folder);
				_setup_button(dir_next, theme_cache.back_folder);
			} else {
				_setup_button(dir_prev, theme_cache.back_folder);
				_setup_button(dir_next, theme_cache.forward_folder);
			}
			_setup_button(dir_up, theme_cache.parent_folder);
			_setup_button(refresh_button, theme_cache.reload);
			_setup_button(show_hidden, theme_cache.toggle_hidden);
			_setup_button(make_dir_button, theme_cache.create_folder);
			_setup_button(show_filename_filter_button, theme_cache.toggle_filename_filter);
			invalidate();
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			update_filters();
		} break;
	}
}

void FileDialog::shortcut_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && has_focus()) {
		if (k->is_pressed()) {
			bool handled = true;

			switch (k->get_keycode()) {
				case Key::H: {
					if (k->is_command_or_control_pressed()) {
						set_show_hidden_files(!show_hidden_files);
					} else {
						handled = false;
					}

				} break;
				case Key::F: {
					if (k->is_command_or_control_pressed()) {
						show_filename_filter_button->set_pressed(!show_filename_filter_button->is_pressed());
					} else {
						handled = false;
					}

				} break;
				case Key::F5: {
					invalidate();
				} break;
				case Key::BACKSPACE: {
					_dir_submitted("..");
				} break;
#ifdef MACOS_ENABLED
				// Cmd + Shift + G (matches Finder's "Go To" shortcut).
				case Key::G: {
					if (k->is_command_or_control_pressed() && k->is_shift_pressed()) {
						directory_edit->grab_focus();
						directory_edit->select_all();
					} else {
						handled = false;
					}
				} break;
#endif
				// Ctrl + L (matches most Windows/Linux file managers' "focus on path bar" shortcut,
				// plus macOS Safari's "focus on address bar" shortcut).
				case Key::L: {
					if (k->is_command_or_control_pressed()) {
						directory_edit->grab_focus();
						directory_edit->select_all();
					} else {
						handled = false;
					}
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
	file_list->set_select_mode(p_enable ? ItemList::SELECT_MULTI : ItemList::SELECT_SINGLE);
}

Vector<String> FileDialog::get_selected_files() const {
	const String current_dir = dir_access->get_current_dir();
	Vector<String> list;
	for (int idx : file_list->get_selected_items()) {
		list.push_back(current_dir.path_join(file_list->get_item_text(idx)));
	}
	return list;
}

void FileDialog::update_dir() {
	full_dir = dir_access->get_current_dir();
	if (root_prefix.is_empty()) {
		directory_edit->set_text(dir_access->get_current_dir(false));
	} else {
		directory_edit->set_text(dir_access->get_current_dir(false).trim_prefix(root_prefix).trim_prefix("/"));
	}

	if (drives->is_visible()) {
		if (dir_access->get_current_dir().is_network_share_path()) {
			_update_drives(false);
			drives->add_item(ETR("Network"));
			drives->set_item_disabled(-1, true);
			drives->select(drives->get_item_count() - 1);
		} else {
			drives->select(dir_access->get_current_drive());
		}
	}

	// Deselect any item, to make "Select Current Folder" button text by default.
	deselect_all();
}

void FileDialog::_dir_submitted(String p_dir) {
	String new_dir = p_dir;
#ifdef WINDOWS_ENABLED
	if (root_prefix.is_empty() && drives->is_visible() && !new_dir.is_network_share_path() && new_dir.is_absolute_path() && new_dir.find(":/") == -1 && new_dir.find(":\\") == -1) {
		// Non network path without X:/ prefix on Windows, add drive letter.
		new_dir = drives->get_item_text(drives->get_selected()).path_join(new_dir);
	}
#endif
	if (!root_prefix.is_empty()) {
		new_dir = root_prefix.path_join(new_dir);
	}
	_change_dir(new_dir);
	filename_edit->set_text("");
	_push_history();
}

void FileDialog::_file_submitted(const String &p_file) {
	_action_pressed();
}

void FileDialog::_save_confirm_pressed() {
	String f = dir_access->get_current_dir().path_join(filename_edit->get_text());
	emit_signal(SNAME("file_selected"), f);
	hide();
}

void FileDialog::_post_popup() {
	ConfirmationDialog::_post_popup();
	if (mode == FILE_MODE_SAVE_FILE) {
		filename_edit->grab_focus();
	} else {
		file_list->grab_focus();
	}

	set_process_shortcut_input(true);

	// For open dir mode, deselect all items on file dialog open.
	if (mode == FILE_MODE_OPEN_DIR) {
		deselect_all();
		file_box->set_visible(false);
	} else {
		file_box->set_visible(true);
	}

	local_history.clear();
	local_history_pos = -1;
	_push_history();
}

void FileDialog::_push_history() {
	local_history.resize(local_history_pos + 1);
	String new_path = dir_access->get_current_dir();
	if (local_history.is_empty() || new_path != local_history[local_history_pos]) {
		local_history.push_back(new_path);
		local_history_pos++;
		dir_prev->set_disabled(local_history_pos == 0);
		dir_next->set_disabled(true);
	}
}

void FileDialog::_action_pressed() {
	if (mode == FILE_MODE_OPEN_FILES) {
		const Vector<String> files = get_selected_files();
		if (!files.is_empty()) {
			emit_signal(SNAME("files_selected"), files);
			hide();
		}
		return;
	}

	String file_text = filename_edit->get_text();
	String f = file_text.is_absolute_path() ? file_text : dir_access->get_current_dir().path_join(file_text);

	if ((mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_OPEN_FILE) && (dir_access->file_exists(f) || dir_access->is_bundle(f))) {
		emit_signal(SNAME("file_selected"), f);
		hide();
	} else if (mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_OPEN_DIR) {
		String path = dir_access->get_current_dir();

		path = path.replace_char('\\', '/');
		int selected = _get_selected_file_idx();
		if (selected > -1) {
			Dictionary d = file_list->get_item_metadata(selected);
			if (d["dir"] && d["name"] != "..") {
				path = path.path_join(d["name"]);
			}
		}

		emit_signal(SNAME("dir_selected"), path);
		hide();
	}

	if (mode == FILE_MODE_SAVE_FILE) {
		bool valid = false;

		if (filter->get_selected() == filter->get_item_count() - 1) {
			valid = true; // Match none.
		} else if (filters.size() > 1 && filter->get_selected() == 0) {
			// Match all filters.
			for (int i = 0; i < filters.size(); i++) {
				String flt = filters[i].get_slicec(';', 0);
				for (int j = 0; j < flt.get_slice_count(","); j++) {
					String str = flt.get_slicec(',', j).strip_edges();
					if (f.matchn(str)) {
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
				String flt = filters[idx].get_slicec(';', 0);
				int filter_slice_count = flt.get_slice_count(",");
				for (int j = 0; j < filter_slice_count; j++) {
					String str = (flt.get_slicec(',', j).strip_edges());
					if (f.matchn(str)) {
						valid = true;
						break;
					}
				}

				if (!valid && filter_slice_count > 0) {
					String str = flt.get_slicec(',', 0).strip_edges();
					f += str.substr(1);
					filename_edit->set_text(f.get_file());
					valid = true;
				}
			} else {
				valid = true;
			}
		}

		String file_name = file_text.strip_edges().get_file();
		if (!valid || file_name.is_empty()) {
			exterr->popup_centered(Size2(250, 80));
			return;
		}

		if (dir_access->file_exists(f) || dir_access->is_bundle(f)) {
			confirm_save->set_text(vformat(atr(ETR("File \"%s\" already exists.\nDo you want to overwrite it?")), f));
			confirm_save->popup_centered(Size2(250, 80));
		} else {
			emit_signal(SNAME("file_selected"), f);
			hide();
		}
	}
}

void FileDialog::_cancel_pressed() {
	filename_edit->set_text("");
	invalidate();
	hide();
}

bool FileDialog::_is_open_should_be_disabled() {
	if (mode == FILE_MODE_OPEN_ANY || mode == FILE_MODE_SAVE_FILE) {
		return false;
	}

	Vector<int> items = file_list->get_selected_items();
	if (items.is_empty()) {
		return mode != FILE_MODE_OPEN_DIR; // In "Open folder" mode, having nothing selected picks the current folder.
	}

	for (const int idx : items) {
		Dictionary d = file_list->get_item_metadata(idx);

		if (((mode == FILE_MODE_OPEN_FILE || mode == FILE_MODE_OPEN_FILES) && d["dir"]) || (mode == FILE_MODE_OPEN_DIR && !d["dir"])) {
			return true;
		}
	}
	return false;
}

void FileDialog::_go_up() {
	_change_dir("..");
	_push_history();
}

void FileDialog::_go_back() {
	if (local_history_pos <= 0) {
		return;
	}

	local_history_pos--;
	_change_dir(local_history[local_history_pos]);

	dir_prev->set_disabled(local_history_pos == 0);
	dir_next->set_disabled(local_history_pos == local_history.size() - 1);
}

void FileDialog::_go_forward() {
	if (local_history_pos >= local_history.size() - 1) {
		return;
	}

	local_history_pos++;
	_change_dir(local_history[local_history_pos]);

	dir_prev->set_disabled(local_history_pos == 0);
	dir_next->set_disabled(local_history_pos == local_history.size() - 1);
}

void FileDialog::deselect_all() {
	// Clear currently selected items in file manager.
	file_list->deselect_all();

	// And change get_ok title.
	get_ok_button()->set_disabled(_is_open_should_be_disabled());

	switch (mode) {
		case FILE_MODE_OPEN_FILE:
		case FILE_MODE_OPEN_FILES:
			set_internal_ok_text(ETR("Open"));
			break;
		case FILE_MODE_OPEN_DIR:
			set_internal_ok_text(ETR("Select Current Folder"));
			break;
		case FILE_MODE_OPEN_ANY:
			set_ok_button_text(ETR("Open"));
			break;
		case FILE_MODE_SAVE_FILE:
			set_ok_button_text(ETR("Save"));
			break;
	}
}

int FileDialog::_get_selected_file_idx() {
	const PackedInt32Array selected = file_list->get_selected_items();
	return selected.is_empty() ? -1 : selected[0];
}

void FileDialog::_file_list_multi_selected(int p_item, bool p_selected) {
	if (p_selected) {
		_file_list_selected(p_item);
	} else {
		get_ok_button()->set_disabled(_is_open_should_be_disabled());
	}
}

void FileDialog::_file_list_selected(int p_item) {
	Dictionary d = file_list->get_item_metadata(p_item);

	if (!d["dir"]) {
		filename_edit->set_text(d["name"]);
		if (mode == FILE_MODE_SAVE_FILE) {
			set_internal_ok_text(ETR("Save"));
		} else {
			set_internal_ok_text(ETR("Open"));
		}
	} else if (mode == FILE_MODE_OPEN_DIR || mode == FILE_MODE_OPEN_ANY || !dir_access->file_exists(filename_edit->get_text())) {
		filename_edit->set_text("");
		if (mode == FILE_MODE_OPEN_DIR || mode == FILE_MODE_OPEN_ANY) {
			set_internal_ok_text(ETR("Select This Folder"));
		}
	}

	get_ok_button()->set_disabled(_is_open_should_be_disabled());
}

void FileDialog::_file_list_item_activated(int p_item) {
	Dictionary d = file_list->get_item_metadata(p_item);

	if (d["dir"]) {
		_change_dir(d["name"]);
		if (mode == FILE_MODE_OPEN_FILE || mode == FILE_MODE_OPEN_FILES || mode == FILE_MODE_OPEN_DIR || mode == FILE_MODE_OPEN_ANY) {
			filename_edit->set_text("");
		}
		_push_history();
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
		String file_str = filename_edit->get_text();
		String base_name = file_str.get_basename();
		Vector<String> filter_substr = filter_str.split(";");
		if (filter_substr.size() >= 2) {
			file_str = base_name + "." + filter_substr[0].strip_edges().get_extension().to_lower();
		} else {
			file_str = base_name + "." + filter_str.strip_edges().get_extension().to_lower();
		}
		filename_edit->set_text(file_str);
	}
}

void FileDialog::_item_menu_id_pressed(int p_option) {
	switch (p_option) {
		case ITEM_MENU_SHOW_IN_EXPLORER: {
			String path;
			int selected = _get_selected_file_idx();
			if (selected > -1) {
				Dictionary d = file_list->get_item_metadata(selected);
				path = ProjectSettings::get_singleton()->globalize_path(dir_access->get_current_dir().path_join(d["name"]));
			} else {
				path = ProjectSettings::get_singleton()->globalize_path(dir_access->get_current_dir());
			}

			OS::get_singleton()->shell_show_in_file_manager(path, true);
		} break;

		case ITEM_MENU_SHOW_BUNDLE_CONTENT: {
			int selected = _get_selected_file_idx();
			if (selected == -1) {
				return;
			}
			Dictionary d = file_list->get_item_metadata(selected);
			_change_dir(d["name"]);
			if (mode == FILE_MODE_OPEN_FILE || mode == FILE_MODE_OPEN_FILES || mode == FILE_MODE_OPEN_DIR || mode == FILE_MODE_OPEN_ANY) {
				filename_edit->set_text("");
			}
			_push_history();
		} break;
	}
}

void FileDialog::_empty_clicked(const Vector2 &p_pos, MouseButton p_button) {
	if (p_button == MouseButton::RIGHT) {
		item_menu->clear();
#if !defined(ANDROID_ENABLED) && !defined(WEB_ENABLED)
		// Opening the system file manager is not supported on the Android and web editors.
		item_menu->add_item(ETR("Open in File Manager"), ITEM_MENU_SHOW_IN_EXPLORER);

		item_menu->set_position(file_list->get_screen_position() + p_pos);
		item_menu->reset_size();
		item_menu->popup();
#endif
	} else if (p_button == MouseButton::LEFT) {
		deselect_all();
	}
}

void FileDialog::_item_clicked(int p_item, const Vector2 &p_pos, MouseButton p_button) {
	if (p_button == MouseButton::RIGHT) {
		item_menu->clear();
#if !defined(ANDROID_ENABLED) && !defined(WEB_ENABLED)
		// Opening the system file manager is not supported on the Android and web editors.
		Dictionary d = file_list->get_item_metadata(p_item);
		if (d["bundle"]) {
			item_menu->add_item(ETR("Show Package Contents"), ITEM_MENU_SHOW_BUNDLE_CONTENT);
		}
		item_menu->add_item(ETR("Open in File Manager"), ITEM_MENU_SHOW_IN_EXPLORER);

		item_menu->set_position(file_list->get_screen_position() + p_pos);
		item_menu->reset_size();
		item_menu->popup();
#endif
	}
}

void FileDialog::update_file_list() {
	file_list->clear();

	// Scroll back to the top after opening a directory
	file_list->get_v_scroll_bar()->set_value(0);

	dir_access->list_dir_begin();

	if (dir_access->is_readable(dir_access->get_current_dir().utf8().get_data())) {
		message->hide();
	} else {
		message->set_text(ETR("You don't have permission to access contents of this folder."));
		message->show();
	}

	LocalVector<String> files;
	LocalVector<String> dirs;

	bool is_hidden;
	String item = dir_access->get_next();

	while (!item.is_empty()) {
		if (item == "." || item == "..") {
			item = dir_access->get_next();
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
		item = dir_access->get_next();
	}

	dirs.sort_custom<FileNoCaseComparator>();
	files.sort_custom<FileNoCaseComparator>();

	String filename_filter_lower = file_name_filter.to_lower();

	List<String> patterns;
	// build filter
	if (filter->get_selected() == filter->get_item_count() - 1) {
		// match all
	} else if (filters.size() > 1 && filter->get_selected() == 0) {
		// match all filters
		for (int i = 0; i < filters.size(); i++) {
			String f = filters[i].get_slicec(';', 0);
			for (int j = 0; j < f.get_slice_count(","); j++) {
				patterns.push_back(f.get_slicec(',', j).strip_edges());
			}
		}
	} else {
		int idx = filter->get_selected();
		if (filters.size() > 1) {
			idx--;
		}

		if (idx >= 0 && idx < filters.size()) {
			String f = filters[idx].get_slicec(';', 0);
			for (int j = 0; j < f.get_slice_count(","); j++) {
				patterns.push_back(f.get_slicec(',', j).strip_edges());
			}
		}
	}

	for (const String &dir_name : dirs) {
		bool bundle = dir_access->is_bundle(dir_name);
		bool found = true;
		if (bundle) {
			bool match = patterns.is_empty();
			for (const String &E : patterns) {
				if (dir_name.matchn(E)) {
					match = true;
					break;
				}
			}
			found = match;
		}

		if (found && (filename_filter_lower.is_empty() || dir_name.to_lower().contains(filename_filter_lower))) {
			file_list->add_item(dir_name, theme_cache.folder);
			file_list->set_item_icon_modulate(-1, theme_cache.folder_icon_color);

			Dictionary d;
			d["name"] = dir_name;
			d["dir"] = !bundle;
			d["bundle"] = bundle;
			file_list->set_item_metadata(-1, d);
		}
	}

	String base_dir = dir_access->get_current_dir();

	for (const String &filename : files) {
		bool match = patterns.is_empty();
		String match_str;

		for (const String &E : patterns) {
			if (filename.matchn(E)) {
				match_str = E;
				match = true;
				break;
			}
		}

		if (match && (filename_filter_lower.is_empty() || filename.to_lower().contains(filename_filter_lower))) {
			const Ref<Texture2D> icon = get_icon_func ? get_icon_func(base_dir.path_join(filename)) : theme_cache.file;
			file_list->add_item(filename, icon);
			file_list->set_item_icon_modulate(-1, theme_cache.file_icon_color);

			if (mode == FILE_MODE_OPEN_DIR) {
				file_list->set_item_disabled(-1, true);
			}
			Dictionary d;
			d["name"] = filename;
			d["dir"] = false;
			d["bundle"] = false;
			file_list->set_item_metadata(-1, d);

			if (filename_edit->get_text() == filename || match_str == filename) {
				file_list->select(file_list->get_item_count() - 1);
			}
		}
	}

	if (mode != FILE_MODE_SAVE_FILE && mode != FILE_MODE_OPEN_DIR) {
		// Select the first file from list if nothing is selected.
		int selected = _get_selected_file_idx();
		if (selected == -1) {
			_file_list_select_first();
		}
	}
}

void FileDialog::_filter_selected(int) {
	update_file_name();
	update_file_list();
}

void FileDialog::_filename_filter_changed() {
	update_filename_filter();
	update_file_list();
	callable_mp(this, &FileDialog::_file_list_select_first).call_deferred();
}

void FileDialog::_file_list_select_first() {
	if (file_list->get_item_count() > 0) {
		file_list->select(0);
		_file_list_selected(0);
	}
}

void FileDialog::_filename_filter_selected() {
	int selected = _get_selected_file_idx();
	if (selected > -1) {
		filename_edit->set_text(file_list->get_item_text(selected));
		filename_edit->emit_signal(SceneStringName(text_submitted), filename_edit->get_text());
	}
}

void FileDialog::update_filters() {
	filter->clear();
	processed_filters.clear();

	if (filters.size() > 1) {
		String all_filters;
		String all_mime;
		String all_filters_full;
		String all_mime_full;

		const int max_filters = 5;

		// "All Recognized" display name.
		for (int i = 0; i < MIN(max_filters, filters.size()); i++) {
			String flt = filters[i].get_slicec(';', 0).strip_edges();
			if (!all_filters.is_empty() && !flt.is_empty()) {
				all_filters += ", ";
			}
			all_filters += flt;

			String mime = filters[i].get_slicec(';', 2).strip_edges();
			if (!all_mime.is_empty() && !mime.is_empty()) {
				all_mime += ", ";
			}
			all_mime += mime;
		}

		// "All Recognized" filter.
		for (int i = 0; i < filters.size(); i++) {
			String flt = filters[i].get_slicec(';', 0).strip_edges();
			if (!all_filters_full.is_empty() && !flt.is_empty()) {
				all_filters_full += ",";
			}
			all_filters_full += flt;

			String mime = filters[i].get_slicec(';', 2).strip_edges();
			if (!all_mime_full.is_empty() && !mime.is_empty()) {
				all_mime_full += ",";
			}
			all_mime_full += mime;
		}

		String native_all_name;
		if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_DIALOG_FILE_MIME)) {
			native_all_name += all_filters;
		}
		if (!native_all_name.is_empty()) {
			native_all_name += ", ";
		}
		native_all_name += all_mime;

		if (max_filters < filters.size()) {
			all_filters += ", ...";
			native_all_name += ", ...";
		}

		filter->add_item(atr(ETR("All Recognized")) + " (" + all_filters + ")");
		processed_filters.push_back(all_filters_full + ";" + atr(ETR("All Recognized")) + " (" + native_all_name + ")" + ";" + all_mime_full);
	}
	for (int i = 0; i < filters.size(); i++) {
		String flt = filters[i].get_slicec(';', 0).strip_edges();
		String desc = filters[i].get_slicec(';', 1).strip_edges();
		String mime = filters[i].get_slicec(';', 2).strip_edges();
		String native_name;
		if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_DIALOG_FILE_MIME)) {
			native_name += flt;
		}
		if (!native_name.is_empty() && !mime.is_empty()) {
			native_name += ", ";
		}
		native_name += mime;
		if (!desc.is_empty()) {
			filter->add_item(atr(desc) + " (" + flt + ")");
			processed_filters.push_back(flt + ";" + atr(desc) + " (" + native_name + ");" + mime);
		} else {
			filter->add_item("(" + flt + ")");
			processed_filters.push_back(flt + ";(" + native_name + ");" + mime);
		}
	}

	String f = atr(ETR("All Files")) + " (*.*)";
	filter->add_item(f);
	processed_filters.push_back("*.*;" + f + ";application/octet-stream");
}

void FileDialog::clear_filename_filter() {
	set_filename_filter("");
	update_filename_filter_gui();
	invalidate();
}

void FileDialog::update_filename_filter_gui() {
	filename_filter_box->set_visible(show_filename_filter);
	if (!show_filename_filter) {
		file_name_filter.clear();
	}
	if (filename_filter->get_text() == file_name_filter) {
		return;
	}
	filename_filter->set_text(file_name_filter);
}

void FileDialog::update_filename_filter() {
	if (filename_filter->get_text() == file_name_filter) {
		return;
	}
	set_filename_filter(filename_filter->get_text());
}

void FileDialog::clear_filters() {
	filters.clear();
	update_filters();
	invalidate();
}

void FileDialog::add_filter(const String &p_filter, const String &p_description) {
	ERR_FAIL_COND_MSG(p_filter.begins_with("."), "Filter must be \"filename.extension\", can't start with dot.");
	if (p_description.is_empty()) {
		filters.push_back(p_filter);
	} else {
		filters.push_back(vformat("%s ; %s", p_filter, p_description));
	}
	update_filters();
	invalidate();
}

void FileDialog::set_filters(const Vector<String> &p_filters) {
	if (filters == p_filters) {
		return;
	}
	filters = p_filters;
	update_filters();
	invalidate();
}

void FileDialog::set_filename_filter(const String &p_filename_filter) {
	if (file_name_filter == p_filename_filter) {
		return;
	}
	file_name_filter = p_filename_filter;
	update_filename_filter_gui();
	emit_signal(SNAME("filename_filter_changed"), filter);
	invalidate();
}

Vector<String> FileDialog::get_filters() const {
	return filters;
}

String FileDialog::get_filename_filter() const {
	return file_name_filter;
}

String FileDialog::get_current_dir() const {
	return full_dir;
}

String FileDialog::get_current_file() const {
	return filename_edit->get_text();
}

String FileDialog::get_current_path() const {
	return full_dir.path_join(filename_edit->get_text());
}

void FileDialog::set_current_dir(const String &p_dir) {
	_change_dir(p_dir);

	_push_history();
}

void FileDialog::set_current_file(const String &p_file) {
	if (filename_edit->get_text() == p_file) {
		return;
	}
	filename_edit->set_text(p_file);
	update_dir();
	invalidate();
	_focus_file_text();
}

void FileDialog::set_current_path(const String &p_path) {
	if (!p_path.size()) {
		return;
	}
	int pos = MAX(p_path.rfind_char('/'), p_path.rfind_char('\\'));
	if (pos == -1) {
		set_current_file(p_path);
	} else {
		String path_dir = p_path.substr(0, pos);
		String path_file = p_path.substr(pos + 1);
		set_current_dir(path_dir);
		set_current_file(path_file);
	}
}

void FileDialog::set_root_subfolder(const String &p_root) {
	root_subfolder = p_root;
	ERR_FAIL_COND_MSG(!dir_access->dir_exists(p_root), "root_subfolder must be an existing sub-directory.");

	local_history.clear();
	local_history_pos = -1;

	dir_access->change_dir(root_subfolder);
	if (root_subfolder.is_empty()) {
		root_prefix = "";
	} else {
		root_prefix = dir_access->get_current_dir();
	}
	invalidate();
	update_dir();
}

String FileDialog::get_root_subfolder() const {
	return root_subfolder;
}

void FileDialog::set_mode_overrides_title(bool p_override) {
	mode_overrides_title = p_override;
}

bool FileDialog::is_mode_overriding_title() const {
	return mode_overrides_title;
}

void FileDialog::set_file_mode(FileMode p_mode) {
	ERR_FAIL_INDEX((int)p_mode, 5);
	if (mode == p_mode) {
		return;
	}
	mode = p_mode;
	switch (mode) {
		case FILE_MODE_OPEN_FILE:
			set_internal_ok_text(ETR("Open"));
			if (mode_overrides_title) {
				set_title(ETR("Open a File"));
			}
			make_dir_button->hide();
			break;
		case FILE_MODE_OPEN_FILES:
			set_internal_ok_text(ETR("Open"));
			if (mode_overrides_title) {
				set_title(ETR("Open File(s)"));
			}
			make_dir_button->hide();
			break;
		case FILE_MODE_OPEN_DIR:
			set_internal_ok_text(ETR("Select Current Folder"));
			if (mode_overrides_title) {
				set_title(ETR("Open a Directory"));
			}
			make_dir_button->show();
			break;
		case FILE_MODE_OPEN_ANY:
			set_internal_ok_text(ETR("Open"));
			if (mode_overrides_title) {
				set_title(ETR("Open a File or Directory"));
			}
			make_dir_button->show();
			break;
		case FILE_MODE_SAVE_FILE:
			set_internal_ok_text(ETR("Save"));
			if (mode_overrides_title) {
				set_title(ETR("Save a File"));
			}
			make_dir_button->show();
			break;
	}

	if (mode == FILE_MODE_OPEN_FILES) {
		file_list->set_select_mode(ItemList::SELECT_MULTI);
	} else {
		file_list->set_select_mode(ItemList::SELECT_SINGLE);
	}

	get_ok_button()->set_disabled(_is_open_should_be_disabled());
}

FileDialog::FileMode FileDialog::get_file_mode() const {
	return mode;
}

void FileDialog::set_access(Access p_access) {
	ERR_FAIL_INDEX(p_access, 3);
	if (access == p_access) {
		return;
	}
	access = p_access;
	root_prefix = "";
	root_subfolder = "";

	switch (p_access) {
		case ACCESS_FILESYSTEM: {
			dir_access = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
#ifdef ANDROID_ENABLED
			set_root_subfolder(OS::get_singleton()->get_system_dir(OS::SYSTEM_DIR_DESKTOP));
#endif
		} break;
		case ACCESS_RESOURCES: {
			dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		} break;
		case ACCESS_USERDATA: {
			dir_access = DirAccess::create(DirAccess::ACCESS_USERDATA);
		} break;
	}
	_update_drives();
	invalidate();
	update_filters();
	update_dir();
}

void FileDialog::invalidate() {
	if (!is_visible() || is_invalidating) {
		return;
	}

	is_invalidating = true;
	callable_mp(this, &FileDialog::_invalidate).call_deferred();
}

void FileDialog::_invalidate() {
	if (!is_invalidating) {
		return;
	}

	update_file_list();

	is_invalidating = false;
}

void FileDialog::_setup_button(Button *p_button, const Ref<Texture2D> &p_icon) {
	p_button->set_button_icon(p_icon);

	p_button->begin_bulk_theme_override();
	p_button->add_theme_color_override(SNAME("icon_normal_color"), theme_cache.icon_normal_color);
	p_button->add_theme_color_override(SNAME("icon_hover_color"), theme_cache.icon_hover_color);
	p_button->add_theme_color_override(SNAME("icon_focus_color"), theme_cache.icon_focus_color);
	p_button->add_theme_color_override(SNAME("icon_pressed_color"), theme_cache.icon_pressed_color);
	p_button->end_bulk_theme_override();
}

FileDialog::Access FileDialog::get_access() const {
	return access;
}

void FileDialog::_make_dir_confirm() {
	Error err = dir_access->make_dir(new_dir_name->get_text().strip_edges());
	if (err == OK) {
		_change_dir(new_dir_name->get_text().strip_edges());
		update_filters();
		_push_history();
	} else {
		mkdirerr->popup_centered(Size2(250, 50));
	}
	new_dir_name->set_text(""); // reset label
}

void FileDialog::_make_dir() {
	make_dir_dialog->popup_centered(Size2(250, 80));
	new_dir_name->grab_focus();
}

void FileDialog::_select_drive(int p_idx) {
	String d = drives->get_item_text(p_idx);
	_change_dir(d);
	filename_edit->set_text("");
	_push_history();
}

void FileDialog::_change_dir(const String &p_new_dir) {
	if (root_prefix.is_empty()) {
		dir_access->change_dir(p_new_dir);
	} else {
		String old_dir = dir_access->get_current_dir();
		dir_access->change_dir(p_new_dir);
		if (!dir_access->get_current_dir(false).begins_with(root_prefix)) {
			dir_access->change_dir(old_dir);
			return;
		}
	}

	invalidate();
	update_dir();
}

void FileDialog::_update_drives(bool p_select) {
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

		if (p_select) {
			drives->select(dir_access->get_current_drive());
		}
	}
}

TypedArray<Dictionary> FileDialog::_get_options() const {
	TypedArray<Dictionary> out;
	for (const FileDialog::Option &opt : options) {
		Dictionary dict;
		dict["name"] = opt.name;
		dict["values"] = opt.values;
		dict["default"] = (int)selected_options.get(opt.name, opt.default_idx);
		out.push_back(dict);
	}
	return out;
}

void FileDialog::_option_changed_checkbox_toggled(bool p_pressed, const String &p_name) {
	if (selected_options.has(p_name)) {
		selected_options[p_name] = p_pressed;
	}
}

void FileDialog::_option_changed_item_selected(int p_idx, const String &p_name) {
	if (selected_options.has(p_name)) {
		selected_options[p_name] = p_idx;
	}
}

void FileDialog::_update_option_controls() {
	if (!options_dirty) {
		return;
	}
	options_dirty = false;

	while (flow_checkbox_options->get_child_count() > 0) {
		Node *child = flow_checkbox_options->get_child(0);
		flow_checkbox_options->remove_child(child);
		child->queue_free();
	}
	while (grid_select_options->get_child_count() > 0) {
		Node *child = grid_select_options->get_child(0);
		grid_select_options->remove_child(child);
		child->queue_free();
	}
	selected_options.clear();

	for (const FileDialog::Option &opt : options) {
		if (opt.values.is_empty()) {
			CheckBox *cb = memnew(CheckBox);
			cb->set_text(opt.name);
			cb->set_accessibility_name(opt.name);
			cb->set_pressed(opt.default_idx);
			flow_checkbox_options->add_child(cb);
			cb->connect(SceneStringName(toggled), callable_mp(this, &FileDialog::_option_changed_checkbox_toggled).bind(opt.name));
			selected_options[opt.name] = (bool)opt.default_idx;
		} else {
			Label *lbl = memnew(Label);
			lbl->set_text(opt.name);
			grid_select_options->add_child(lbl);

			OptionButton *ob = memnew(OptionButton);
			for (const String &val : opt.values) {
				ob->add_item(val);
			}
			ob->set_accessibility_name(opt.name);
			ob->select(opt.default_idx);
			grid_select_options->add_child(ob);
			ob->connect(SceneStringName(item_selected), callable_mp(this, &FileDialog::_option_changed_item_selected).bind(opt.name));
			selected_options[opt.name] = opt.default_idx;
		}
	}
}

Dictionary FileDialog::get_selected_options() const {
	return selected_options;
}

String FileDialog::get_option_name(int p_option) const {
	ERR_FAIL_INDEX_V(p_option, options.size(), String());
	return options[p_option].name;
}

Vector<String> FileDialog::get_option_values(int p_option) const {
	ERR_FAIL_INDEX_V(p_option, options.size(), Vector<String>());
	return options[p_option].values;
}

int FileDialog::get_option_default(int p_option) const {
	ERR_FAIL_INDEX_V(p_option, options.size(), -1);
	return options[p_option].default_idx;
}

void FileDialog::set_option_name(int p_option, const String &p_name) {
	if (p_option < 0) {
		p_option += get_option_count();
	}
	ERR_FAIL_INDEX(p_option, options.size());
	options.write[p_option].name = p_name;
	options_dirty = true;
	if (is_visible()) {
		_update_option_controls();
	}
}

void FileDialog::set_option_values(int p_option, const Vector<String> &p_values) {
	if (p_option < 0) {
		p_option += get_option_count();
	}
	ERR_FAIL_INDEX(p_option, options.size());
	options.write[p_option].values = p_values;
	if (p_values.is_empty()) {
		options.write[p_option].default_idx = CLAMP(options[p_option].default_idx, 0, 1);
	} else {
		options.write[p_option].default_idx = CLAMP(options[p_option].default_idx, 0, options[p_option].values.size() - 1);
	}
	options_dirty = true;
	if (is_visible()) {
		_update_option_controls();
	}
}

void FileDialog::set_option_default(int p_option, int p_index) {
	if (p_option < 0) {
		p_option += get_option_count();
	}
	ERR_FAIL_INDEX(p_option, options.size());
	if (options[p_option].values.is_empty()) {
		options.write[p_option].default_idx = CLAMP(p_index, 0, 1);
	} else {
		options.write[p_option].default_idx = CLAMP(p_index, 0, options[p_option].values.size() - 1);
	}
	options_dirty = true;
	if (is_visible()) {
		_update_option_controls();
	}
}

void FileDialog::add_option(const String &p_name, const Vector<String> &p_values, int p_index) {
	Option opt;
	opt.name = p_name;
	opt.values = p_values;
	if (opt.values.is_empty()) {
		opt.default_idx = CLAMP(p_index, 0, 1);
	} else {
		opt.default_idx = CLAMP(p_index, 0, opt.values.size() - 1);
	}
	options.push_back(opt);
	options_dirty = true;
	if (is_visible()) {
		_update_option_controls();
	}
}

void FileDialog::set_option_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);

	if (options.size() == p_count) {
		return;
	}
	options.resize(p_count);

	options_dirty = true;
	notify_property_list_changed();
	if (is_visible()) {
		_update_option_controls();
	}
}

int FileDialog::get_option_count() const {
	return options.size();
}

void FileDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_cancel_pressed"), &FileDialog::_cancel_pressed);

	ClassDB::bind_method(D_METHOD("clear_filters"), &FileDialog::clear_filters);
	ClassDB::bind_method(D_METHOD("add_filter", "filter", "description"), &FileDialog::add_filter, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_filters", "filters"), &FileDialog::set_filters);
	ClassDB::bind_method(D_METHOD("get_filters"), &FileDialog::get_filters);
	ClassDB::bind_method(D_METHOD("clear_filename_filter"), &FileDialog::clear_filename_filter);
	ClassDB::bind_method(D_METHOD("set_filename_filter", "filter"), &FileDialog::set_filename_filter);
	ClassDB::bind_method(D_METHOD("get_filename_filter"), &FileDialog::get_filename_filter);
	ClassDB::bind_method(D_METHOD("get_option_name", "option"), &FileDialog::get_option_name);
	ClassDB::bind_method(D_METHOD("get_option_values", "option"), &FileDialog::get_option_values);
	ClassDB::bind_method(D_METHOD("get_option_default", "option"), &FileDialog::get_option_default);
	ClassDB::bind_method(D_METHOD("set_option_name", "option", "name"), &FileDialog::set_option_name);
	ClassDB::bind_method(D_METHOD("set_option_values", "option", "values"), &FileDialog::set_option_values);
	ClassDB::bind_method(D_METHOD("set_option_default", "option", "default_value_index"), &FileDialog::set_option_default);
	ClassDB::bind_method(D_METHOD("set_option_count", "count"), &FileDialog::set_option_count);
	ClassDB::bind_method(D_METHOD("get_option_count"), &FileDialog::get_option_count);
	ClassDB::bind_method(D_METHOD("add_option", "name", "values", "default_value_index"), &FileDialog::add_option);
	ClassDB::bind_method(D_METHOD("get_selected_options"), &FileDialog::get_selected_options);
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
	ClassDB::bind_method(D_METHOD("set_root_subfolder", "dir"), &FileDialog::set_root_subfolder);
	ClassDB::bind_method(D_METHOD("get_root_subfolder"), &FileDialog::get_root_subfolder);
	ClassDB::bind_method(D_METHOD("set_show_hidden_files", "show"), &FileDialog::set_show_hidden_files);
	ClassDB::bind_method(D_METHOD("is_showing_hidden_files"), &FileDialog::is_showing_hidden_files);
	ClassDB::bind_method(D_METHOD("set_use_native_dialog", "native"), &FileDialog::set_use_native_dialog);
	ClassDB::bind_method(D_METHOD("get_use_native_dialog"), &FileDialog::get_use_native_dialog);
	ClassDB::bind_method(D_METHOD("deselect_all"), &FileDialog::deselect_all);

	ClassDB::bind_method(D_METHOD("invalidate"), &FileDialog::invalidate);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mode_overrides_title"), "set_mode_overrides_title", "is_mode_overriding_title");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "file_mode", PROPERTY_HINT_ENUM, "Open File,Open Files,Open Folder,Open Any,Save"), "set_file_mode", "get_file_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "access", PROPERTY_HINT_ENUM, "Resources,User Data,File System"), "set_access", "get_access");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "root_subfolder"), "set_root_subfolder", "get_root_subfolder");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "filters"), "set_filters", "get_filters");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "filename_filter"), "set_filename_filter", "get_filename_filter");
	ADD_ARRAY_COUNT("Options", "option_count", "set_option_count", "get_option_count", "option_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_hidden_files"), "set_show_hidden_files", "is_showing_hidden_files");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_native_dialog"), "set_use_native_dialog", "get_use_native_dialog");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_dir", PROPERTY_HINT_DIR, "", PROPERTY_USAGE_NONE), "set_current_dir", "get_current_dir");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_file", PROPERTY_HINT_FILE, "*", PROPERTY_USAGE_NONE), "set_current_file", "get_current_file");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_path", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_current_path", "get_current_path");

	ADD_SIGNAL(MethodInfo("file_selected", PropertyInfo(Variant::STRING, "path")));
	ADD_SIGNAL(MethodInfo("files_selected", PropertyInfo(Variant::PACKED_STRING_ARRAY, "paths")));
	ADD_SIGNAL(MethodInfo("dir_selected", PropertyInfo(Variant::STRING, "dir")));
	ADD_SIGNAL(MethodInfo("filename_filter_changed", PropertyInfo(Variant::STRING, "filter")));

	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_FILE);
	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_FILES);
	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_DIR);
	BIND_ENUM_CONSTANT(FILE_MODE_OPEN_ANY);
	BIND_ENUM_CONSTANT(FILE_MODE_SAVE_FILE);

	BIND_ENUM_CONSTANT(ACCESS_RESOURCES);
	BIND_ENUM_CONSTANT(ACCESS_USERDATA);
	BIND_ENUM_CONSTANT(ACCESS_FILESYSTEM);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FileDialog, parent_folder);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FileDialog, forward_folder);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FileDialog, back_folder);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FileDialog, reload);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FileDialog, toggle_hidden);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FileDialog, folder);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FileDialog, toggle_filename_filter);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FileDialog, file);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FileDialog, create_folder);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, FileDialog, folder_icon_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, FileDialog, file_icon_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, FileDialog, file_disabled_color);

	// TODO: Define own colors?
	BIND_THEME_ITEM_EXT(Theme::DATA_TYPE_COLOR, FileDialog, icon_normal_color, "font_color", "Button");
	BIND_THEME_ITEM_EXT(Theme::DATA_TYPE_COLOR, FileDialog, icon_hover_color, "font_hover_color", "Button");
	BIND_THEME_ITEM_EXT(Theme::DATA_TYPE_COLOR, FileDialog, icon_focus_color, "font_focus_color", "Button");
	BIND_THEME_ITEM_EXT(Theme::DATA_TYPE_COLOR, FileDialog, icon_pressed_color, "font_pressed_color", "Button");

	Option defaults;

	base_property_helper.set_prefix("option_");
	base_property_helper.set_array_length_getter(&FileDialog::get_option_count);
	base_property_helper.register_property(PropertyInfo(Variant::STRING, "name"), defaults.name, &FileDialog::set_option_name, &FileDialog::get_option_name);
	base_property_helper.register_property(PropertyInfo(Variant::PACKED_STRING_ARRAY, "values"), defaults.values, &FileDialog::set_option_values, &FileDialog::get_option_values);
	base_property_helper.register_property(PropertyInfo(Variant::INT, "default"), defaults.default_idx, &FileDialog::set_option_default, &FileDialog::get_option_default);
	PropertyListHelper::register_base_helper(&base_property_helper);
}

void FileDialog::set_show_hidden_files(bool p_show) {
	if (show_hidden_files == p_show) {
		return;
	}
	show_hidden_files = p_show;
	invalidate();
}

void FileDialog::set_show_filename_filter(bool p_show) {
	if (p_show == show_filename_filter) {
		return;
	}
	if (p_show) {
		filename_filter->grab_focus();
	} else {
		if (filename_filter->has_focus()) {
			callable_mp((Control *)file_list, &Control::grab_focus).call_deferred();
		}
	}
	show_filename_filter = p_show;
	update_filename_filter_gui();
	invalidate();
}

bool FileDialog::get_show_filename_filter() const {
	return show_filename_filter;
}

bool FileDialog::is_showing_hidden_files() const {
	return show_hidden_files;
}

void FileDialog::set_default_show_hidden_files(bool p_show) {
	default_show_hidden_files = p_show;
}

void FileDialog::set_use_native_dialog(bool p_native) {
	use_native_dialog = p_native;

#ifdef TOOLS_ENABLED
	if (is_part_of_edited_scene()) {
		return;
	}
#endif

	// Replace the built-in dialog with the native one if it's currently visible.
	if (is_inside_tree() && is_visible() && _can_use_native_popup() && (use_native_dialog || OS::get_singleton()->is_sandboxed())) {
		ConfirmationDialog::set_visible(false);
		_native_popup();
	}
}

bool FileDialog::get_use_native_dialog() const {
	return use_native_dialog;
}

FileDialog::FileDialog() {
	set_title(ETR("Save a File"));
	set_hide_on_ok(false);
	set_size(Size2(640, 360));
	set_internal_ok_text(ETR("Save")); // Default mode text.

	show_hidden_files = default_show_hidden_files;
	dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);

	main_vbox = memnew(VBoxContainer);
	add_child(main_vbox, false, INTERNAL_MODE_FRONT);

	HBoxContainer *top_toolbar = memnew(HBoxContainer);
	main_vbox->add_child(top_toolbar);

	dir_prev = memnew(Button);
	dir_prev->set_theme_type_variation(SceneStringName(FlatButton));
	dir_prev->set_accessibility_name(ETR("Previous"));
	dir_prev->set_tooltip_text(ETR("Go to previous folder."));
	top_toolbar->add_child(dir_prev);
	dir_prev->connect(SceneStringName(pressed), callable_mp(this, &FileDialog::_go_back));

	dir_next = memnew(Button);
	dir_next->set_accessibility_name(ETR("Next"));
	dir_next->set_theme_type_variation(SceneStringName(FlatButton));
	dir_next->set_tooltip_text(ETR("Go to next folder."));
	top_toolbar->add_child(dir_next);
	dir_next->connect(SceneStringName(pressed), callable_mp(this, &FileDialog::_go_forward));

	dir_up = memnew(Button);
	dir_up->set_accessibility_name(ETR("Parent Folder"));
	dir_up->set_theme_type_variation(SceneStringName(FlatButton));
	dir_up->set_tooltip_text(ETR("Go to parent folder."));
	top_toolbar->add_child(dir_up);
	dir_up->connect(SceneStringName(pressed), callable_mp(this, &FileDialog::_go_up));

	{
		Label *label = memnew(Label(ETR("Path:")));
		top_toolbar->add_child(label);
	}

	drives_container = memnew(HBoxContainer);
	top_toolbar->add_child(drives_container);

	drives = memnew(OptionButton);
	drives->connect(SceneStringName(item_selected), callable_mp(this, &FileDialog::_select_drive));
	drives->set_accessibility_name(ETR("Drive"));
	top_toolbar->add_child(drives);

	directory_edit = memnew(LineEdit);
	directory_edit->set_accessibility_name(ETR("Directory Path"));
	directory_edit->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	directory_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	top_toolbar->add_child(directory_edit);
	directory_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FileDialog::_dir_submitted));

	shortcuts_container = memnew(HBoxContainer);
	top_toolbar->add_child(shortcuts_container);

	refresh_button = memnew(Button);
	refresh_button->set_theme_type_variation(SceneStringName(FlatButton));
	refresh_button->set_accessibility_name(ETR("Refresh"));
	refresh_button->set_tooltip_text(ETR("Refresh files."));
	top_toolbar->add_child(refresh_button);
	refresh_button->connect(SceneStringName(pressed), callable_mp(this, &FileDialog::update_file_list));

	show_hidden = memnew(Button);
	show_hidden->set_theme_type_variation(SceneStringName(FlatButton));
	show_hidden->set_toggle_mode(true);
	show_hidden->set_pressed(is_showing_hidden_files());
	show_hidden->set_accessibility_name(ETR("Show Hidden Files"));
	show_hidden->set_tooltip_text(ETR("Toggle the visibility of hidden files."));
	top_toolbar->add_child(show_hidden);
	show_hidden->connect(SceneStringName(toggled), callable_mp(this, &FileDialog::set_show_hidden_files));

	show_filename_filter_button = memnew(Button);
	show_filename_filter_button->set_theme_type_variation(SceneStringName(FlatButton));
	show_filename_filter_button->set_toggle_mode(true);
	show_filename_filter_button->set_pressed(false);
	show_filename_filter_button->set_accessibility_name(ETR("Filter File Names"));
	show_filename_filter_button->set_tooltip_text(ETR("Toggle the visibility of the filter for file names."));
	top_toolbar->add_child(show_filename_filter_button);
	show_filename_filter_button->connect(SceneStringName(toggled), callable_mp(this, &FileDialog::set_show_filename_filter));

	make_dir_button = memnew(Button);
	make_dir_button->set_theme_type_variation(SceneStringName(FlatButton));
	make_dir_button->set_accessibility_name(ETR("Create New Folder"));
	make_dir_button->set_tooltip_text(ETR("Create a new folder."));
	top_toolbar->add_child(make_dir_button);
	make_dir_button->connect(SceneStringName(pressed), callable_mp(this, &FileDialog::_make_dir));

	{
		Label *label = memnew(Label(ETR("Directories & Files:")));
		label->set_theme_type_variation("HeaderSmall");
		main_vbox->add_child(label);
	}

	file_list = memnew(ItemList);
	file_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	file_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	file_list->set_accessibility_name(ETR("Directories and Files"));
	file_list->set_allow_rmb_select(true);
	main_vbox->add_child(file_list);
	file_list->connect("multi_selected", callable_mp(this, &FileDialog::_file_list_multi_selected));
	file_list->connect("item_selected", callable_mp(this, &FileDialog::_file_list_selected));
	file_list->connect("item_activated", callable_mp(this, &FileDialog::_file_list_item_activated));
	file_list->connect("item_clicked", callable_mp(this, &FileDialog::_item_clicked));
	file_list->connect("empty_clicked", callable_mp(this, &FileDialog::_empty_clicked));

	message = memnew(Label);
	message->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	message->hide();
	message->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	message->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	message->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	file_list->add_child(message);

	filename_filter_box = memnew(HBoxContainer);
	filename_filter_box->set_visible(false);
	main_vbox->add_child(filename_filter_box);

	{
		Label *label = memnew(Label(ETR("Filter:")));
		filename_filter_box->add_child(label);
	}

	filename_filter = memnew(LineEdit);
	filename_filter->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	filename_filter->set_stretch_ratio(4);
	filename_filter->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	filename_filter->set_clear_button_enabled(true);
	filename_filter->set_accessibility_name(ETR("Filename Filter"));
	filename_filter_box->add_child(filename_filter);
	filename_filter->connect(SceneStringName(text_changed), callable_mp(this, &FileDialog::_filename_filter_changed).unbind(1));
	filename_filter->connect(SceneStringName(text_submitted), callable_mp(this, &FileDialog::_filename_filter_selected).unbind(1));

	file_box = memnew(HBoxContainer);
	main_vbox->add_child(file_box);

	{
		Label *label = memnew(Label(ETR("File:")));
		file_box->add_child(label);
	}

	filename_edit = memnew(LineEdit);
	filename_edit->set_accessibility_name(ETR("File Name"));
	filename_edit->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	filename_edit->set_stretch_ratio(4);
	filename_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	file_box->add_child(filename_edit);
	filename_edit->connect(SceneStringName(text_submitted), callable_mp(this, &FileDialog::_file_submitted));

	filter = memnew(OptionButton);
	filter->set_stretch_ratio(3);
	filter->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	filter->set_clip_text(true); // Too many extensions overflows it.
	file_box->add_child(filter);
	filter->connect(SceneStringName(item_selected), callable_mp(this, &FileDialog::_filter_selected));

	flow_checkbox_options = memnew(HFlowContainer);
	flow_checkbox_options->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	flow_checkbox_options->set_alignment(FlowContainer::ALIGNMENT_CENTER);
	main_vbox->add_child(flow_checkbox_options);

	grid_select_options = memnew(GridContainer);
	grid_select_options->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	grid_select_options->set_columns(2);
	main_vbox->add_child(grid_select_options);

	confirm_save = memnew(ConfirmationDialog);
	add_child(confirm_save, false, INTERNAL_MODE_FRONT);
	confirm_save->connect(SceneStringName(confirmed), callable_mp(this, &FileDialog::_save_confirm_pressed));

	make_dir_dialog = memnew(ConfirmationDialog);
	make_dir_dialog->set_title(ETR("Create Folder"));
	add_child(make_dir_dialog, false, INTERNAL_MODE_FRONT);
	make_dir_dialog->connect(SceneStringName(confirmed), callable_mp(this, &FileDialog::_make_dir_confirm));

	VBoxContainer *makevb = memnew(VBoxContainer);
	make_dir_dialog->add_child(makevb);

	new_dir_name = memnew(LineEdit);
	new_dir_name->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_FILE);
	makevb->add_margin_child(ETR("Name:"), new_dir_name);
	make_dir_dialog->register_text_enter(new_dir_name);

	mkdirerr = memnew(AcceptDialog);
	mkdirerr->set_text(ETR("Could not create folder."));
	add_child(mkdirerr, false, INTERNAL_MODE_FRONT);

	exterr = memnew(AcceptDialog);
	exterr->set_text(ETR("Invalid extension, or empty filename."));
	add_child(exterr, false, INTERNAL_MODE_FRONT);

	item_menu = memnew(PopupMenu);
	item_menu->connect(SceneStringName(id_pressed), callable_mp(this, &FileDialog::_item_menu_id_pressed));
	add_child(item_menu, false, INTERNAL_MODE_FRONT);

	connect(SceneStringName(confirmed), callable_mp(this, &FileDialog::_action_pressed));

	_update_drives();
	update_filters();
	update_filename_filter_gui();
	update_dir();

	if (register_func) {
		register_func(this);
	}

	property_helper.setup_for_instance(base_property_helper, this);
}

FileDialog::~FileDialog() {
	if (unregister_func) {
		unregister_func(this);
	}
}
