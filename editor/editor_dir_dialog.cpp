/*************************************************************************/
/*  editor_dir_dialog.cpp                                                */
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

#include "editor_dir_dialog.h"

#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "editor/editor_file_system.h"
#include "editor/editor_settings.h"
#include "editor_scale.h"
#include "servers/display_server.h"

void EditorDirDialog::_update_dir(TreeItem *p_item, EditorFileSystemDirectory *p_dir, const String &p_select_path) {
	updating = true;

	String path = p_dir->get_path();

	p_item->set_metadata(0, p_dir->get_path());
	p_item->set_icon(0, tree->get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")));
	p_item->set_icon_modulate(0, tree->get_theme_color(SNAME("folder_icon_modulate"), SNAME("FileDialog")));

	if (!p_item->get_parent()) {
		p_item->set_text(0, "res://");
	} else {
		if (!opened_paths.has(path) && (p_select_path.is_empty() || !p_select_path.begins_with(path))) {
			p_item->set_collapsed(true);
		}

		p_item->set_text(0, p_dir->get_name());
	}

	//this should be handled by EditorFileSystem already
	//bool show_hidden = EditorSettings::get_singleton()->get("filesystem/file_dialog/show_hidden_files");
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

void EditorDirDialog::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &EditorDirDialog::reload), make_binds(""));
		reload();

		if (!tree->is_connected("item_collapsed", callable_mp(this, &EditorDirDialog::_item_collapsed))) {
			tree->connect("item_collapsed", callable_mp(this, &EditorDirDialog::_item_collapsed), varray(), CONNECT_DEFERRED);
		}

		if (!EditorFileSystem::get_singleton()->is_connected("filesystem_changed", callable_mp(this, &EditorDirDialog::reload))) {
			EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &EditorDirDialog::reload), make_binds(""));
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		if (EditorFileSystem::get_singleton()->is_connected("filesystem_changed", callable_mp(this, &EditorDirDialog::reload))) {
			EditorFileSystem::get_singleton()->disconnect("filesystem_changed", callable_mp(this, &EditorDirDialog::reload));
		}
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (must_reload && is_visible()) {
			reload();
		}
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

	DirAccessRef d = DirAccess::open(dir);
	ERR_FAIL_COND_MSG(!d, "Cannot open directory '" + dir + "'.");
	Error err = d->make_dir(makedirname->get_text());

	if (err != OK) {
		mkdirerr->popup_centered(Size2(250, 80) * EDSCALE);
	} else {
		opened_paths.insert(dir);
		//reload(dir.plus_file(makedirname->get_text()));
		EditorFileSystem::get_singleton()->scan_changes(); //we created a dir, so rescan changes
	}
	makedirname->set_text(""); // reset label
}

void EditorDirDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("dir_selected", PropertyInfo(Variant::STRING, "dir")));
}

EditorDirDialog::EditorDirDialog() {
	updating = false;

	set_title(TTR("Choose a Directory"));
	set_hide_on_ok(false);

	tree = memnew(Tree);
	add_child(tree);

	tree->connect("item_activated", callable_mp(this, &EditorDirDialog::_item_activated));

	makedir = add_button(TTR("Create Folder"), DisplayServer::get_singleton()->get_swap_cancel_ok(), "makedir");
	makedir->connect("pressed", callable_mp(this, &EditorDirDialog::_make_dir));

	makedialog = memnew(ConfirmationDialog);
	makedialog->set_title(TTR("Create Folder"));
	add_child(makedialog);

	VBoxContainer *makevb = memnew(VBoxContainer);
	makedialog->add_child(makevb);
	//makedialog->set_child_rect(makevb);

	makedirname = memnew(LineEdit);
	makevb->add_margin_child(TTR("Name:"), makedirname);
	makedialog->register_text_enter(makedirname);
	makedialog->connect("confirmed", callable_mp(this, &EditorDirDialog::_make_dir_confirm));

	mkdirerr = memnew(AcceptDialog);
	mkdirerr->set_text(TTR("Could not create folder."));
	add_child(mkdirerr);

	get_ok_button()->set_text(TTR("Choose"));

	must_reload = false;
}
