/*************************************************************************/
/*  editor_dir_dialog.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "editor_dir_dialog.h"

#include "editor/editor_file_system.h"
#include "editor/editor_settings.h"
#include "os/keyboard.h"
#include "os/os.h"

void EditorDirDialog::_update_dir(TreeItem *p_item) {

	updating = true;
	p_item->clear_children();
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	String cdir = p_item->get_metadata(0);

	da->change_dir(cdir);
	da->list_dir_begin();
	String p = da->get_next();

	List<String> dirs;
	bool ishidden;
	bool show_hidden = EditorSettings::get_singleton()->get("filesystem/file_dialog/show_hidden_files");

	while (p != "") {

		ishidden = da->current_is_hidden();

		if (show_hidden || !ishidden) {
			if (da->current_is_dir() && !p.begins_with(".")) {
				dirs.push_back(p);
			}
		}
		p = da->get_next();
	}

	dirs.sort();

	for (List<String>::Element *E = dirs.front(); E; E = E->next()) {
		TreeItem *ti = tree->create_item(p_item);
		ti->set_text(0, E->get());
		ti->set_icon(0, get_icon("Folder", "EditorIcons"));
		ti->set_collapsed(true);
	}

	memdelete(da);
	updating = false;
}

void EditorDirDialog::reload() {

	if (!is_visible_in_tree()) {
		must_reload = true;
		return;
	}

	tree->clear();
	TreeItem *root = tree->create_item();
	root->set_metadata(0, "res://");
	root->set_icon(0, get_icon("Folder", "EditorIcons"));
	root->set_text(0, "/");
	_update_dir(root);
	_item_collapsed(root);
	must_reload = false;
}

void EditorDirDialog::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
		reload();

		if (!tree->is_connected("item_collapsed", this, "_item_collapsed")) {
			tree->connect("item_collapsed", this, "_item_collapsed", varray(), CONNECT_DEFERRED);
		}

		if (!EditorFileSystem::get_singleton()->is_connected("filesystem_changed", this, "reload")) {
			EditorFileSystem::get_singleton()->connect("filesystem_changed", this, "reload");
		}
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (must_reload && is_visible_in_tree()) {
			reload();
		}
	}
}

void EditorDirDialog::_item_collapsed(Object *_p_item) {

	TreeItem *p_item = _p_item->cast_to<TreeItem>();

	if (updating || p_item->is_collapsed())
		return;

	TreeItem *ci = p_item->get_children();
	while (ci) {

		String p = ci->get_metadata(0);
		if (p == "") {
			String pp = p_item->get_metadata(0);
			ci->set_metadata(0, pp.plus_file(ci->get_text(0)));
			_update_dir(ci);
		}
		ci = ci->get_next();
	}
}

void EditorDirDialog::set_current_path(const String &p_path) {

	reload();
	String p = p_path;
	if (p.begins_with("res://"))
		p = p.replace_first("res://", "");

	Vector<String> dirs = p.split("/", false);

	TreeItem *r = tree->get_root();
	for (int i = 0; i < dirs.size(); i++) {

		String d = dirs[i];
		TreeItem *p = r->get_children();
		while (p) {

			if (p->get_text(0) == d)
				break;
			p = p->get_next();
		}

		ERR_FAIL_COND(!p);
		String pp = p->get_metadata(0);
		if (pp == "") {
			p->set_metadata(0, String(r->get_metadata(0)).plus_file(d));
			_update_dir(p);
		}
		updating = true;
		p->set_collapsed(false);
		updating = false;
		_item_collapsed(p);
		r = p;
	}

	r->select(0);
}

void EditorDirDialog::ok_pressed() {

	TreeItem *ti = tree->get_selected();
	if (!ti)
		return;

	String dir = ti->get_metadata(0);
	emit_signal("dir_selected", dir);
	hide();
}

void EditorDirDialog::_make_dir() {

	TreeItem *ti = tree->get_selected();
	if (!ti) {
		mkdirerr->set_text("Please select a base directory first");
		mkdirerr->popup_centered_minsize();
		return;
	}

	makedialog->popup_centered_minsize(Size2(250, 80));
	makedirname->grab_focus();
}

void EditorDirDialog::_make_dir_confirm() {

	TreeItem *ti = tree->get_selected();
	if (!ti)
		return;

	String dir = ti->get_metadata(0);

	DirAccess *d = DirAccess::open(dir);
	ERR_FAIL_COND(!d);
	Error err = d->make_dir(makedirname->get_text());

	if (err != OK) {
		mkdirerr->popup_centered_minsize(Size2(250, 80));
	} else {
		set_current_path(dir.plus_file(makedirname->get_text()));
	}
	makedirname->set_text(""); // reset label
}

void EditorDirDialog::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_item_collapsed"), &EditorDirDialog::_item_collapsed);
	ClassDB::bind_method(D_METHOD("_make_dir"), &EditorDirDialog::_make_dir);
	ClassDB::bind_method(D_METHOD("_make_dir_confirm"), &EditorDirDialog::_make_dir_confirm);
	ClassDB::bind_method(D_METHOD("reload"), &EditorDirDialog::reload);

	ADD_SIGNAL(MethodInfo("dir_selected", PropertyInfo(Variant::STRING, "dir")));
}

EditorDirDialog::EditorDirDialog() {

	updating = false;

	set_title(TTR("Choose a Directory"));
	set_hide_on_ok(false);

	tree = memnew(Tree);
	add_child(tree);

	tree->connect("item_activated", this, "_ok");

	makedir = add_button(TTR("Create Folder"), OS::get_singleton()->get_swap_ok_cancel() ? true : false, "makedir");
	makedir->connect("pressed", this, "_make_dir");

	makedialog = memnew(ConfirmationDialog);
	makedialog->set_title(TTR("Create Folder"));
	add_child(makedialog);

	VBoxContainer *makevb = memnew(VBoxContainer);
	makedialog->add_child(makevb);
	//makedialog->set_child_rect(makevb);

	makedirname = memnew(LineEdit);
	makevb->add_margin_child(TTR("Name:"), makedirname);
	makedialog->register_text_enter(makedirname);
	makedialog->connect("confirmed", this, "_make_dir_confirm");

	mkdirerr = memnew(AcceptDialog);
	mkdirerr->set_text(TTR("Could not create folder."));
	add_child(mkdirerr);

	get_ok()->set_text(TTR("Choose"));

	must_reload = false;
}
