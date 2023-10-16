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

#include "core/io/dir_access.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "editor/editor_file_system.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "scene/gui/check_box.h"
#include "scene/gui/tree.h"
#include "servers/display_server.h"

void EditorDirDialog::_update_dir(TreeItem *p_item, EditorFileSystemDirectory *p_dir, const String &p_select_path) {
	updating = true;

	String path = p_dir->get_path();

	p_item->set_metadata(0, p_dir->get_path());
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

	updating = false;
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		TreeItem *ti = tree->create_item(p_item);
		_update_dir(ti, p_dir->get_subdir(i));
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
	must_reload = false;
}

bool EditorDirDialog::is_copy_pressed() const {
	return copy->is_pressed();
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

void EditorDirDialog::_copy_toggled(bool p_pressed) {
	if (p_pressed) {
		set_ok_button_text(TTR("Copy"));
	} else {
		set_ok_button_text(TTR("Move"));
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
	_ok_pressed(); // From AcceptDialog.
}

void EditorDirDialog::ok_pressed() {
	TreeItem *ti = tree->get_selected();
	if (!ti) {
		return;
	}

	String dir = ti->get_metadata(0);
	emit_signal(SNAME("dir_selected"), dir);
	hide();
}

void EditorDirDialog::_make_dir() {
	TreeItem *ti = tree->get_selected();
	if (!ti) {
		mkdirerr->set_text(TTR("Please select a base directory first."));
		mkdirerr->popup_centered();
		return;
	}

	makedialog->popup_centered(Size2(250, 80));
	makedirname->grab_focus();
}

void EditorDirDialog::_make_dir_confirm() {
	TreeItem *ti = tree->get_selected();
	if (!ti) {
		return;
	}

	String dir = ti->get_metadata(0);

	Ref<DirAccess> d = DirAccess::open(dir);
	ERR_FAIL_COND_MSG(d.is_null(), "Cannot open directory '" + dir + "'.");

	const String stripped_dirname = makedirname->get_text().strip_edges();

	if (d->dir_exists(stripped_dirname)) {
		mkdirerr->set_text(TTR("Could not create folder. File with that name already exists."));
		mkdirerr->popup_centered();
		return;
	}

	Error err = d->make_dir(stripped_dirname);
	if (err != OK) {
		mkdirerr->popup_centered(Size2(250, 80) * EDSCALE);
	} else {
		opened_paths.insert(dir);
		EditorFileSystem::get_singleton()->scan_changes(); // We created a dir, so rescan changes.
	}
	makedirname->set_text(""); // reset label
}

void EditorDirDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("dir_selected", PropertyInfo(Variant::STRING, "dir")));
}

EditorDirDialog::EditorDirDialog() {
	set_title(TTR("Choose a Directory"));
	set_hide_on_ok(false);

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	tree = memnew(Tree);
	vb->add_child(tree);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->connect("item_activated", callable_mp(this, &EditorDirDialog::_item_activated));

	copy = memnew(CheckBox);
	vb->add_child(copy);
	copy->set_text(TTR("Copy File(s)"));
	copy->connect("toggled", callable_mp(this, &EditorDirDialog::_copy_toggled));

	makedir = add_button(TTR("Create Folder"), DisplayServer::get_singleton()->get_swap_cancel_ok(), "makedir");
	makedir->connect("pressed", callable_mp(this, &EditorDirDialog::_make_dir));

	makedialog = memnew(ConfirmationDialog);
	makedialog->set_title(TTR("Create Folder"));
	add_child(makedialog);

	VBoxContainer *makevb = memnew(VBoxContainer);
	makedialog->add_child(makevb);

	makedirname = memnew(LineEdit);
	makevb->add_margin_child(TTR("Name:"), makedirname);
	makedialog->register_text_enter(makedirname);
	makedialog->connect("confirmed", callable_mp(this, &EditorDirDialog::_make_dir_confirm));

	mkdirerr = memnew(AcceptDialog);
	mkdirerr->set_text(TTR("Could not create folder."));
	add_child(mkdirerr);

	set_ok_button_text(TTR("Move"));
}
