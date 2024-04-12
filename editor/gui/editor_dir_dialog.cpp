/**************************************************************************/
/*  editor_dir_dialog.cpp                                                 */
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

#include "editor_dir_dialog.h"

#include "editor/directory_create_dialog.h"
#include "editor/editor_file_system.h"
#include "scene/gui/box_container.h"
#include "scene/gui/tree.h"
#include "servers/display_server.h"

void EditorDirDialog::_update_dir(TreeItem *p_item, EditorFileSystemDirectory *p_dir, const String &p_select_path) {
	updating = true;

	const String path = p_dir->get_path();

	p_item->set_metadata(0, path);
	p_item->set_icon(0, tree->get_editor_theme_icon(SNAME("Folder")));
	p_item->set_icon_modulate(0, tree->get_theme_color(SNAME("folder_icon_color"), SNAME("FileDialog")));

	if (!p_item->get_parent()) {
		p_item->set_text(0, "res://");
	} else {
		if (!opened_paths.has(path) && (p_select_path.is_empty() || !p_select_path.begins_with(path))) {
			p_item->set_collapsed(true);
		}

		p_item->set_text(0, p_dir->get_name());
	}

	if (path == new_dir_path || !p_item->get_parent()) {
		p_item->select(0);
	}

	updating = false;
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		TreeItem *ti = tree->create_item(p_item);
		_update_dir(ti, p_dir->get_subdir(i));
	}
}

void EditorDirDialog::config(const Vector<String> &p_paths) {
	ERR_FAIL_COND(p_paths.is_empty());

	if (p_paths.size() == 1) {
		String path = p_paths[0];
		if (path.ends_with("/")) {
			path = path.substr(0, path.length() - 1);
		}
		// TRANSLATORS: %s is the file name that will be moved or duplicated.
		set_title(vformat(TTR("Move/Duplicate: %s"), path.get_file()));
	} else {
		// TRANSLATORS: %d is the number of files that will be moved or duplicated.
		set_title(vformat(TTRN("Move/Duplicate %d Item", "Move/Duplicate %d Items", p_paths.size()), p_paths.size()));
	}
}

void EditorDirDialog::reload(const String &p_path) {
	if (!is_visible()) {
		must_reload = true;
		return;
	}

	tree->clear();
	TreeItem *root = tree->create_item();
	_update_dir(root, EditorFileSystem::get_singleton()->get_filesystem(), p_path);
	_item_collapsed(root);
	new_dir_path.clear();
	must_reload = false;
}

void EditorDirDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &EditorDirDialog::reload).bind(""));
			reload();

			if (!tree->is_connected("item_collapsed", callable_mp(this, &EditorDirDialog::_item_collapsed))) {
				tree->connect("item_collapsed", callable_mp(this, &EditorDirDialog::_item_collapsed), CONNECT_DEFERRED);
			}

			if (!EditorFileSystem::get_singleton()->is_connected("filesystem_changed", callable_mp(this, &EditorDirDialog::reload))) {
				EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &EditorDirDialog::reload).bind(""));
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (EditorFileSystem::get_singleton()->is_connected("filesystem_changed", callable_mp(this, &EditorDirDialog::reload))) {
				EditorFileSystem::get_singleton()->disconnect("filesystem_changed", callable_mp(this, &EditorDirDialog::reload));
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (must_reload && is_visible()) {
				reload();
			}
		} break;
	}
}

void EditorDirDialog::_item_collapsed(Object *p_item) {
	TreeItem *item = Object::cast_to<TreeItem>(p_item);

	if (updating) {
		return;
	}

	if (item->is_collapsed()) {
		opened_paths.erase(item->get_metadata(0));
	} else {
		opened_paths.insert(item->get_metadata(0));
	}
}

void EditorDirDialog::_item_activated() {
	TreeItem *ti = tree->get_selected();
	ERR_FAIL_NULL(ti);
	if (ti->get_child_count() > 0) {
		ti->set_collapsed(!ti->is_collapsed());
	}
}

void EditorDirDialog::_copy_pressed() {
	TreeItem *ti = tree->get_selected();
	ERR_FAIL_NULL(ti);

	hide();
	emit_signal(SNAME("copy_pressed"), ti->get_metadata(0));
}

void EditorDirDialog::ok_pressed() {
	TreeItem *ti = tree->get_selected();
	ERR_FAIL_NULL(ti);

	hide();
	emit_signal(SNAME("move_pressed"), ti->get_metadata(0));
}

void EditorDirDialog::_make_dir() {
	TreeItem *ti = tree->get_selected();
	ERR_FAIL_NULL(ti);
	makedialog->config(ti->get_metadata(0));
	makedialog->popup_centered();
}

void EditorDirDialog::_make_dir_confirm(const String &p_path) {
	// Multiple level of directories can be created at once.
	String base_dir = p_path.get_base_dir();
	while (true) {
		opened_paths.insert(base_dir + "/");
		if (base_dir == "res://") {
			break;
		}
		base_dir = base_dir.get_base_dir();
	}

	new_dir_path = p_path + "/";
	EditorFileSystem::get_singleton()->scan_changes(); // We created a dir, so rescan changes.
}

void EditorDirDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("copy_pressed", PropertyInfo(Variant::STRING, "dir")));
	ADD_SIGNAL(MethodInfo("move_pressed", PropertyInfo(Variant::STRING, "dir")));
}

EditorDirDialog::EditorDirDialog() {
	set_hide_on_ok(false);

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	HBoxContainer *hb = memnew(HBoxContainer);
	vb->add_child(hb);

	hb->add_child(memnew(Label(TTR("Choose target directory:"))));
	hb->add_spacer();

	makedir = memnew(Button(TTR("Create Folder")));
	hb->add_child(makedir);
	makedir->connect("pressed", callable_mp(this, &EditorDirDialog::_make_dir));

	tree = memnew(Tree);
	vb->add_child(tree);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->connect("item_activated", callable_mp(this, &EditorDirDialog::_item_activated));

	set_ok_button_text(TTR("Move"));

	copy = add_button(TTR("Copy"), !DisplayServer::get_singleton()->get_swap_cancel_ok());
	copy->connect("pressed", callable_mp(this, &EditorDirDialog::_copy_pressed));

	makedialog = memnew(DirectoryCreateDialog);
	add_child(makedialog);
	makedialog->connect("dir_created", callable_mp(this, &EditorDirDialog::_make_dir_confirm));
}
