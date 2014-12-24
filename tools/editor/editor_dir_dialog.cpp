/*************************************************************************/
/*  editor_dir_dialog.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "os/os.h"

void EditorDirDialog::_update_dir(TreeItem* p_item) {

	updating=true;
	p_item->clear_children();
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	String cdir = p_item->get_metadata(0);

	da->change_dir(cdir);
	da->list_dir_begin();
	String p=da->get_next();
	while(p!="") {
		if (da->current_is_dir() && !p.begins_with(".")) {
			TreeItem *ti = tree->create_item(p_item);
			ti->set_text(0,p);
			ti->set_icon(0,get_icon("Folder","EditorIcons"));
			ti->set_collapsed(true);
		}

		p=da->get_next();
	}

	memdelete(da);
	updating=false;

}


void EditorDirDialog::reload() {

	tree->clear();
	TreeItem *root = tree->create_item();
	root->set_metadata(0,"res://");
	root->set_icon(0,get_icon("Folder","EditorIcons"));
	root->set_text(0,"/");
	_update_dir(root);
	_item_collapsed(root);
}

void EditorDirDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {
		reload();
		tree->connect("item_collapsed",this,"_item_collapsed",varray(),CONNECT_DEFERRED);
	}
}

void EditorDirDialog::_item_collapsed(Object* _p_item){

	TreeItem *p_item=_p_item->cast_to<TreeItem>();

	if (updating || p_item->is_collapsed())
		return;

	TreeItem *ci = p_item->get_children();
	while(ci) {

		String p =ci->get_metadata(0);
		if (p=="") {
			String pp = p_item->get_metadata(0);
			ci->set_metadata(0,pp.plus_file(ci->get_text(0)));
			_update_dir(ci);
		}
		ci=ci->get_next();
	}

}

void EditorDirDialog::set_current_path(const String& p_path) {

	reload();
	String p = p_path;
	if (p.begins_with("res://"))
		p.replace_first("res://","");

	Vector<String> dirs = p.split("/");

	TreeItem *r=tree->get_root();
	for(int i=0;i<dirs.size();i++) {

		String d = dirs[i];
		TreeItem *p = r->get_children();
		while(p) {

			if (p->get_text(0)==d)
				break;
			p=p->get_next();
		}

		ERR_FAIL_COND(!p);
		String pp = p->get_metadata(0);
		if (pp=="") {
			_update_dir(p);
			updating=true;
			p->set_collapsed(false);
			updating=false;
			_item_collapsed(p);

		}
		r=p;
	}

	r->select(0);

}

void EditorDirDialog::ok_pressed() {

	TreeItem *ti=tree->get_selected();
	if (!ti)
		return;

	String dir = ti->get_metadata(0);
	emit_signal("dir_selected",dir);
	hide();
}


void EditorDirDialog::_make_dir() {

	TreeItem *ti=tree->get_selected();
	if (!ti)
		return;

	makedialog->popup_centered_minsize(Size2(250,80));
}

void EditorDirDialog::_make_dir_confirm() {

	TreeItem *ti=tree->get_selected();
	if (!ti)
		return;

	String dir = ti->get_metadata(0);
	DirAccess *d = DirAccess::open(dir);
	ERR_FAIL_COND(!d);
	Error err = d->make_dir(makedirname->get_text());
	if (err!=OK) {
		mkdirerr->popup_centered_minsize(Size2(250,80));
	} else {
		reload();
	}
}


void EditorDirDialog::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_item_collapsed"),&EditorDirDialog::_item_collapsed);
	ObjectTypeDB::bind_method(_MD("_make_dir"),&EditorDirDialog::_make_dir);
	ObjectTypeDB::bind_method(_MD("_make_dir_confirm"),&EditorDirDialog::_make_dir_confirm);

	ADD_SIGNAL(MethodInfo("dir_selected",PropertyInfo(Variant::STRING,"dir")));
}



EditorDirDialog::EditorDirDialog() {

	set_title("Choose a Directory");
	tree = memnew( Tree );
	add_child(tree);
	set_child_rect(tree);
	updating=false;
	get_ok()->set_text("Choose");
	set_hide_on_ok(false);



	makedir = add_button("Create Folder",OS::get_singleton()->get_swap_ok_cancel()?true:false,"makedir");
	makedir->connect("pressed",this,"_make_dir");

	makedialog = memnew( ConfirmationDialog );
	makedialog->set_title("Create Folder");
	VBoxContainer *makevb= memnew( VBoxContainer );
	makedialog->add_child(makevb);
	makedialog->set_child_rect(makevb);
	makedirname = memnew( LineEdit );
	makevb->add_margin_child("Name:",makedirname);
	add_child(makedialog);
	makedialog->register_text_enter(makedirname);
	makedialog->connect("confirmed",this,"_make_dir_confirm");
	mkdirerr = memnew( AcceptDialog );
	mkdirerr->set_text("Could not create folder.");
	add_child(mkdirerr);

}
