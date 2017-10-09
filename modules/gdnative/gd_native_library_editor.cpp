/*************************************************************************/
/*  gd_native_library_editor.cpp                                         */
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
#ifdef TOOLS_ENABLED
#include "gd_native_library_editor.h"

#include "gdnative.h"

void GDNativeLibraryEditor::_find_gdnative_singletons(EditorFileSystemDirectory *p_dir, const Set<String> &enabled_list) {

	// check children

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		String file_type = p_dir->get_file_type(i);

		if (file_type != "GDNativeLibrary") {
			continue;
		}

		Ref<GDNativeLibrary> lib = ResourceLoader::load(p_dir->get_file_path(i));
		if (lib.is_valid() && lib->is_singleton_gdnative()) {
			String path = p_dir->get_file_path(i);
			TreeItem *ti = libraries->create_item(libraries->get_root());
			ti->set_text(0, path.get_file());
			ti->set_tooltip(0, path);
			ti->set_metadata(0, path);
			ti->set_cell_mode(1, TreeItem::CELL_MODE_RANGE);
			ti->set_text(1, "Disabled,Enabled");
			bool enabled = enabled_list.has(path) ? true : false;

			ti->set_range(1, enabled ? 1 : 0);
			ti->set_custom_color(1, enabled ? Color(0, 1, 0) : Color(1, 0, 0));
		}
	}

	// check subdirectories
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_find_gdnative_singletons(p_dir->get_subdir(i), enabled_list);
	}
}

void GDNativeLibraryEditor::_update_libraries() {

	updating = true;
	libraries->clear();
	libraries->create_item(); //rppt

	Vector<String> enabled_paths;
	if (ProjectSettings::get_singleton()->has_setting("gdnative/singletons")) {
		enabled_paths = ProjectSettings::get_singleton()->get("gdnative/singletons");
	}
	Set<String> enabled_list;
	for (int i = 0; i < enabled_paths.size(); i++) {
		enabled_list.insert(enabled_paths[i]);
	}

	EditorFileSystemDirectory *fs = EditorFileSystem::get_singleton()->get_filesystem();
	if (fs) {
		_find_gdnative_singletons(fs, enabled_list);
	}

	updating = false;
}

void GDNativeLibraryEditor::_item_edited() {
	if (updating)
		return;

	TreeItem *item = libraries->get_edited();
	if (!item)
		return;

	bool enabled = item->get_range(1);
	String path = item->get_metadata(0);

	Vector<String> enabled_paths;
	if (ProjectSettings::get_singleton()->has_setting("gdnative/singletons")) {
		enabled_paths = ProjectSettings::get_singleton()->get("gdnative/singletons");
	}

	if (enabled) {
		if (enabled_paths.find(path) == -1) {
			enabled_paths.push_back(path);
		}
	} else {
		enabled_paths.erase(path);
	}

	if (enabled_paths.size()) {
		ProjectSettings::get_singleton()->set("gdnative/singletons", enabled_paths);
	} else {
		ProjectSettings::get_singleton()->set("gdnative/singletons", Variant());
	}
}

void GDNativeLibraryEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (is_visible_in_tree()) {
			_update_libraries();
		}
	}
}

void GDNativeLibraryEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_item_edited"), &GDNativeLibraryEditor::_item_edited);
}

GDNativeLibraryEditor::GDNativeLibraryEditor() {
	libraries = memnew(Tree);
	libraries->set_columns(2);
	libraries->set_column_titles_visible(true);
	libraries->set_column_title(0, TTR("Library"));
	libraries->set_column_title(1, TTR("Status"));
	libraries->set_hide_root(true);
	add_margin_child(TTR("Libraries: "), libraries, true);
	updating = false;
	libraries->connect("item_edited", this, "_item_edited");
}

#endif // TOOLS_ENABLED
