/**************************************************************************/
/*  editor_file_dialog.cpp                                                */
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

#include "editor_file_dialog.h"

#include "core/config/project_settings.h"
#include "editor/docks/filesystem_dock.h"
#include "editor/file_system/dependency_editor.h"
#include "editor/settings/editor_settings.h"

void EditorFileDialog::_item_menu_id_pressed(int p_option) {
	// Use dependency dialog to delete the entry in the editor, but only for project files.
	if (p_option == ITEM_MENU_DELETE && get_access() == ACCESS_RESOURCES) {
		const PackedInt32Array selected = get_file_item_list()->get_selected_items();
		if (selected.is_empty()) {
			return;
		}

		if (!dependency_remove_dialog) {
			dependency_remove_dialog = memnew(DependencyRemoveDialog);
			add_child(dependency_remove_dialog);
		}

		const Dictionary meta = get_file_item_list()->get_item_metadata(selected[0]);
		const String delete_path = dir_access->get_current_dir().path_join(meta["name"]);

		if (meta["dir"]) {
			dependency_remove_dialog->show(Vector<String>{ delete_path }, Vector<String>());
		} else {
			dependency_remove_dialog->show(Vector<String>(), Vector<String>{ delete_path });
		}
		return;
	}
	FileDialog::_item_menu_id_pressed(p_option);
}

bool EditorFileDialog::_should_use_native_popup() const {
#ifdef ANDROID_ENABLED
	// Native file dialog on Android, returns a file URI instead of a path and does not support res://, user://, or options. This requires editor-side changes to handle properly, so disabling it for now.
	return false;
#else
	return _can_use_native_popup() && (OS::get_singleton()->is_sandboxed() || EDITOR_GET("interface/editor/use_native_file_dialogs").operator bool());
#endif
}

bool EditorFileDialog::_should_hide_file(const String &p_file) const {
	if (Engine::get_singleton()->is_project_manager_hint()) {
		return false;
	}
	const String full_path = dir_access->get_current_dir().path_join(p_file);
	return EditorFileSystem::_should_skip_directory(full_path);
}

Color EditorFileDialog::_get_folder_color(const String &p_path) const {
	return FileSystemDock::get_dir_icon_color(p_path, FileDialog::_get_folder_color(p_path));
}

void EditorFileDialog::_bind_methods() {
#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("add_side_menu", "menu", "title"), &EditorFileDialog::add_side_menu, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_disable_overwrite_warning", "disable"), &EditorFileDialog::set_disable_overwrite_warning);
	ClassDB::bind_method(D_METHOD("is_overwrite_warning_disabled"), &EditorFileDialog::is_overwrite_warning_disabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disable_overwrite_warning", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_disable_overwrite_warning", "is_overwrite_warning_disabled");
#endif
}

void EditorFileDialog::_validate_property(PropertyInfo &p_property) const {
	// Hide properties controlled by editor settings.
	if (p_property.name == "use_native_dialog" || p_property.name == "show_hidden_files" || p_property.name == "display_mode") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void EditorFileDialog::_dir_contents_changed() {
	if (!EditorFileSystem::get_singleton()) {
		return;
	}

	bool scan_required = false;
	switch (get_access()) {
		case FileDialog::ACCESS_RESOURCES: {
			scan_required = true;
		} break;
		case FileDialog::ACCESS_USERDATA: {
			// Directories within the project dir are unlikely to be accessed.
		} break;
		case FileDialog::ACCESS_FILESYSTEM: {
			// Directories within the project dir may still be accessed.
			const String localized_path = ProjectSettings::get_singleton()->localize_path(get_current_dir());
			scan_required = localized_path.is_resource_file();
		} break;
	}
	if (scan_required) {
		EditorFileSystem::get_singleton()->scan_changes();
	}
}

void EditorFileDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				// Synchronize back favorites and recent directories, in case they have changed.
				EditorSettings::get_singleton()->set_favorites(get_favorite_list(), false);
				EditorSettings::get_singleton()->set_recent_dirs(get_recent_list(), false);
			}
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (!EditorSettings::get_singleton()->check_changed_settings_in_group("filesystem/file_dialog")) {
				break;
			}
			set_show_hidden_files(EDITOR_GET("filesystem/file_dialog/show_hidden_files"));
			set_display_mode((DisplayMode)EDITOR_GET("filesystem/file_dialog/display_mode").operator int());
		} break;
	}
}
