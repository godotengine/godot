/*************************************************************************/
/*  create_dialog.cpp                                                    */
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
#include "create_dialog.h"

#include "object_type_db.h"
#include "print_string.h"
#include "scene/gui/box_container.h"
#include "editor_node.h"

#if 1

#include "os/keyboard.h"


void CreateDialog::popup(bool p_dontclear) {

	popup_centered_ratio(0.6);
	if (p_dontclear)
		search_box->select_all();
	else
		search_box->clear();
	search_box->grab_focus();
	_update_search();


}


void CreateDialog::_text_changed(const String& p_newtext) {

	_update_search();
}

void CreateDialog::_sbox_input(const InputEvent& p_ie) {

	if (p_ie.type==InputEvent::KEY && (
		p_ie.key.scancode == KEY_UP ||
		p_ie.key.scancode == KEY_DOWN ||
		p_ie.key.scancode == KEY_PAGEUP ||
		p_ie.key.scancode == KEY_PAGEDOWN ) ) {

		search_options->call("_input_event",p_ie);
		search_box->accept_event();
	}

}

void CreateDialog::add_type(const String& p_type,HashMap<String,TreeItem*>& p_types,TreeItem *p_root,TreeItem **to_select) {

	if (p_types.has(p_type))
		return;
	if (!ObjectTypeDB::is_type(p_type,base_type) || p_type==base_type)
		return;

	String inherits=ObjectTypeDB::type_inherits_from(p_type);

	TreeItem *parent=p_root;


	if (inherits.length()) {

		if (!p_types.has(inherits)) {

			add_type(inherits,p_types,p_root,to_select);
		}

		if (p_types.has(inherits) )
			parent=p_types[inherits];
	}

	TreeItem *item = search_options->create_item(parent);
	item->set_text(0,p_type);
	if (!ObjectTypeDB::can_instance(p_type)) {
		item->set_custom_color(0, Color(0.5,0.5,0.5) );
		item->set_selectable(0,false);
	} else {

		if (!*to_select && (search_box->get_text()=="" || p_type.findn(search_box->get_text())!=-1)) {
			*to_select=item;
		}

	}


	if (has_icon(p_type,"EditorIcons")) {

		item->set_icon(0, get_icon(p_type,"EditorIcons"));
	}



	p_types[p_type]=item;
}

void CreateDialog::_update_search() {


	search_options->clear();

	/*
	TreeItem *root = search_options->create_item();
	_parse_fs(EditorFileSystem::get_singleton()->get_filesystem());
*/

	List<String> type_list;
	ObjectTypeDB::get_type_list(&type_list);

	HashMap<String,TreeItem*> types;

	TreeItem *root = search_options->create_item();

	root->set_text(0,base_type);

	List<String>::Element *I=type_list.front();
	TreeItem *to_select=NULL;

	for(;I;I=I->next()) {


		String type=I->get();


		if (!ObjectTypeDB::can_instance(type))
			continue; // cant create what can't be instanced
		if (search_box->get_text()=="")
			add_type(type,types,root,&to_select);
		else {

			bool found=false;
			String type=I->get();
			while(type!="" && ObjectTypeDB::is_type(type,base_type) && type!=base_type) {
				if (type.findn(search_box->get_text())!=-1) {

					found=true;
					break;
				}

				type=ObjectTypeDB::type_inherits_from(type);
			}


			if (found)
				add_type(I->get(),types,root,&to_select);
		}

		if (EditorNode::get_editor_data().get_custom_types().has(type)) {
			//there are custom types based on this... cool.


			const Vector<EditorData::CustomType> &ct = EditorNode::get_editor_data().get_custom_types()[type];
			for(int i=0;i<ct.size();i++) {

				bool show = search_box->get_text()=="" || ct[i].name.findn(search_box->get_text())!=-1;

				if (!show)
					continue;
				if (!types.has(type))
					add_type(type,types,root,&to_select);

				TreeItem *ti;
				if (types.has(type) )
					ti=types[type];
				else
					ti=search_options->get_root();


				TreeItem *item = search_options->create_item(ti);
				item->set_metadata(0,type);
				item->set_text(0,ct[i].name);
				if (ct[i].icon.is_valid()) {
					item->set_icon(0,ct[i].icon);

				}

				if (!to_select && (search_box->get_text()=="" || ct[i].name.findn(search_box->get_text())!=-1)) {
					to_select=item;
				}

			}

		}
	}

	if (to_select)
		to_select->select(0);

	get_ok()->set_disabled(root->get_children()==NULL);

}

void CreateDialog::_confirmed() {

	TreeItem *ti = search_options->get_selected();
	if (!ti)
		return;
	emit_signal("create");
	hide();
}

void CreateDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		connect("confirmed",this,"_confirmed");
		_update_search();
	}

	if (p_what==NOTIFICATION_VISIBILITY_CHANGED) {

		if (is_visible()) {

			search_box->call_deferred("grab_focus"); // still not visible
			search_box->select_all();
		}
	}

}

void CreateDialog::set_base_type(const String& p_base) {

	base_type=p_base;
	set_title("Create New "+p_base);
	_update_search();
}

Object *CreateDialog::instance_selected() {

	TreeItem *selected = search_options->get_selected();
	if (selected) {

		return ObjectTypeDB::instance(selected->get_text(0));
	}

	return NULL;
}


String CreateDialog::get_base_type() const {

	return base_type;
}

void CreateDialog::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_text_changed"),&CreateDialog::_text_changed);
	ObjectTypeDB::bind_method(_MD("_confirmed"),&CreateDialog::_confirmed);
	ObjectTypeDB::bind_method(_MD("_sbox_input"),&CreateDialog::_sbox_input);

	ADD_SIGNAL(MethodInfo("create"));

}


CreateDialog::CreateDialog() {


	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);
	set_child_rect(vbc);
	search_box = memnew( LineEdit );
	vbc->add_margin_child("Search:",search_box);
	search_box->connect("text_changed",this,"_text_changed");
	search_box->connect("input_event",this,"_sbox_input");
	search_options = memnew( Tree );
	vbc->add_margin_child("Matches:",search_options,true);
	get_ok()->set_text("Create");
	get_ok()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated",this,"_confirmed");
//	search_options->set_hide_root(true);
	base_type="Object";

}


#else

//old create dialog, disabled

void CreateDialog::_notification(int p_what) {
	
	if (p_what==NOTIFICATION_READY) {
		connect("confirmed",this,"_create");
		update_tree();
	}
	if (p_what==NOTIFICATION_DRAW) {
		
		//RID ci = get_canvas_item();
		//get_stylebox("panel","PopupMenu")->draw(ci,Rect2(Point2(),get_size()));
	}	
}


void CreateDialog::_create() {
	
	if (tree->get_selected())
		emit_signal("create");
	hide();
}

void CreateDialog::_cancel() {
	
	hide();
}

void CreateDialog::_text_changed(String p_text) {
	
	update_tree();
}

void CreateDialog::add_type(const String& p_type,HashMap<String,TreeItem*>& p_types,TreeItem *p_root) {
	
	if (p_types.has(p_type))
		return;
	if (!ObjectTypeDB::is_type(p_type,base) || p_type==base)
		return;
	
	String inherits=ObjectTypeDB::type_inherits_from(p_type);
	
	TreeItem *parent=p_root;

	
	if (inherits.length()) {
	
		if (!p_types.has(inherits)) {
			
			add_type(inherits,p_types,p_root);
		}
			
		if (p_types.has(inherits) )
			parent=p_types[inherits];		
	} 
	
	TreeItem *item = tree->create_item(parent);
	item->set_text(0,p_type);
	if (!ObjectTypeDB::can_instance(p_type)) {
		item->set_custom_color(0, Color(0.5,0.5,0.5) );
		item->set_selectable(0,false);
	}
	
	
	if (has_icon(p_type,"EditorIcons")) {
		
		item->set_icon(0, get_icon(p_type,"EditorIcons"));
	}


	
	p_types[p_type]=item;
}

void CreateDialog::update_tree() {

	tree->clear();
	
	List<String> type_list;
	ObjectTypeDB::get_type_list(&type_list);	
	
	HashMap<String,TreeItem*> types;
	
	TreeItem *root = tree->create_item();
		
	root->set_text(0,base);

	List<String>::Element *I=type_list.front();
	
	for(;I;I=I->next()) {
				
		
		String type=I->get();
		

		if (!ObjectTypeDB::can_instance(type))
			continue; // cant create what can't be instanced
		if (filter->get_text()=="")
			add_type(type,types,root);
		else {
			
			bool found=false;
			String type=I->get();
			while(type!="" && ObjectTypeDB::is_type(type,base) && type!=base) { 	
				if (type.findn(filter->get_text())!=-1) {
					
					found=true;
					break;
				}
				
				type=ObjectTypeDB::type_inherits_from(type);
			}
			

			if (found)
				add_type(I->get(),types,root);
		}

		if (EditorNode::get_editor_data().get_custom_types().has(type)) {
			//there are custom types based on this... cool.


			const Vector<EditorData::CustomType> &ct = EditorNode::get_editor_data().get_custom_types()[type];
			for(int i=0;i<ct.size();i++) {

				bool show = filter->get_text()=="" || ct[i].name.findn(filter->get_text())!=-1;

				if (!show)
					continue;
				if (!types.has(type))
					add_type(type,types,root);

				TreeItem *ti;
				if (types.has(type) )
					ti=types[type];
				else
					ti=tree->get_root();


				TreeItem *item = tree->create_item(ti);
				item->set_metadata(0,type);
				item->set_text(0,ct[i].name);
				if (ct[i].icon.is_valid()) {
					item->set_icon(0,ct[i].icon);

				}
			}

		}
	}


}


Object *CreateDialog::instance_selected() {

	if (!tree->get_selected())
		return NULL;

	String base = String(tree->get_selected()->get_metadata(0));
	if (base!="") {


		String name = tree->get_selected()->get_text(0);
		if (EditorNode::get_editor_data().get_custom_types().has(base)) {

			const Vector<EditorData::CustomType> &ct = EditorNode::get_editor_data().get_custom_types()[base];
			for(int i=0;i<ct.size();i++) {

				if (ct[i].name==name) {

					Object* obj = ObjectTypeDB::instance(base);
					ERR_FAIL_COND_V(!obj,NULL);
					obj->set_script(ct[i].script.get_ref_ptr());
					if (ct[i].icon.is_valid())
						obj->set_meta("_editor_icon",ct[i].icon);
					return obj;


				}
			}
		}

		ERR_FAIL_V(NULL);

	}

	return ObjectTypeDB::instance(tree->get_selected()->get_text(0));

}


void CreateDialog::_bind_methods() {
	
	ObjectTypeDB::bind_method("_create",&CreateDialog::_create);
	ObjectTypeDB::bind_method("_cancel",&CreateDialog::_cancel);
	ObjectTypeDB::bind_method("_text_changed", &CreateDialog::_text_changed);
	ADD_SIGNAL( MethodInfo("create"));
	
}




void CreateDialog::set_base_type(const String& p_base) {

	set_title("Create "+p_base+" Type");

	if (base==p_base)
		return;
	base=p_base;
	if (is_inside_scene())
		update_tree();	
}

String CreateDialog::get_base_type() const {

	return base;
}

CreateDialog::CreateDialog() {


	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);
	set_child_rect(vbc);

	get_ok()->set_text("Create");
	
	tree = memnew( Tree );
	vbc->add_margin_child("Type:",tree,true);
	//tree->set_hide_root(true);
	
	filter = memnew( LineEdit );	
	vbc->add_margin_child("Filter:",filter);
		
	base="Node";
	set_as_toplevel(true);


	tree->connect("item_activated", this, "_create");
	filter->connect("text_changed", this,"_text_changed");

}


CreateDialog::~CreateDialog()
{
}


#endif
