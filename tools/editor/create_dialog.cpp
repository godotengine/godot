/*************************************************************************/
/*  create_dialog.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "editor_settings.h"
#include "editor_help.h"


void CreateDialog::popup(bool p_dontclear) {

	recent->clear();

	FileAccess *f = FileAccess::open( EditorSettings::get_singleton()->get_project_settings_path().plus_file("create_recent."+base_type), FileAccess::READ );

	if (f) {

		TreeItem *root = recent->create_item();

		while(!f->eof_reached()) {
			String l = f->get_line().strip_edges();

			if (l!=String()) {

				TreeItem *ti = recent->create_item(root);
				ti->set_text(0,l);
				if (has_icon(l,"EditorIcons")) {

					ti->set_icon(0,get_icon(l,"EditorIcons"));
				} else {
					ti->set_icon(0,get_icon("Object","EditorIcons"));
				}
			}
		}

		memdelete(f);
	}

	favorites->clear();

	f = FileAccess::open( EditorSettings::get_singleton()->get_project_settings_path().plus_file("favorites."+base_type), FileAccess::READ );

	favorite_list.clear();

	if (f) {


		while(!f->eof_reached()) {
			String l = f->get_line().strip_edges();

			if (l!=String()) {
				favorite_list.push_back(l);
			}
		}

		memdelete(f);
	} else {
#if 0
// I think this was way too confusing
		if (base_type=="Node") {
			//harcode some favorites :D
			favorite_list.push_back("Panel");
			favorite_list.push_back("Button");
			favorite_list.push_back("Label");
			favorite_list.push_back("LineEdit");
			favorite_list.push_back("Node2D");
			favorite_list.push_back("Sprite");
			favorite_list.push_back("Camera2D");
			favorite_list.push_back("Area2D");
			favorite_list.push_back("CollisionShape2D");
			favorite_list.push_back("Spatial");
			favorite_list.push_back("Camera");
			favorite_list.push_back("Area");
			favorite_list.push_back("CollisionShape");
			favorite_list.push_back("TestCube");
			favorite_list.push_back("AnimationPlayer");

		}
#endif
	}

	_update_favorite_list();

	popup_centered_ratio();
	if (p_dontclear)
		search_box->select_all();
	else {
		search_box->clear();
	}
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
	if (!ClassDB::is_parent_class(p_type,base_type) || p_type==base_type)
		return;

	String inherits=ClassDB::get_parent_class(p_type);

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
	if (!ClassDB::can_instance(p_type)) {
		item->set_custom_color(0, Color(0.5,0.5,0.5) );
		item->set_selectable(0,false);
	} else {

		if ((!*to_select && (search_box->get_text().is_subsequence_ofi(p_type))) || search_box->get_text()==p_type) {
			*to_select=item;
		}

	}

	if (bool(EditorSettings::get_singleton()->get("docks/scene_tree/start_create_dialog_fully_expanded"))) {
		item->set_collapsed(false);
	} else {
		// don't collapse search results
		bool collapse = (search_box->get_text() == "");
		// don't collapse the root node
		collapse &= (item != p_root);
		// don't collapse abstract nodes on the first tree level
		collapse &= ((parent != p_root) || (ClassDB::can_instance(p_type)));
		item->set_collapsed(collapse);
	}

	const String& description = EditorHelp::get_doc_data()->class_list[p_type].brief_description;
	item->set_tooltip(0,description);


	if (has_icon(p_type,"EditorIcons")) {

		item->set_icon(0, get_icon(p_type,"EditorIcons"));
	}



	p_types[p_type]=item;
}

void CreateDialog::_update_search() {


	search_options->clear();
	favorite->set_disabled(true);

	help_bit->set_text("");
	/*
	TreeItem *root = search_options->create_item();
	_parse_fs(EditorFileSystem::get_singleton()->get_filesystem());
*/

	List<StringName> type_list;
	ClassDB::get_class_list(&type_list);

	HashMap<String,TreeItem*> types;

	TreeItem *root = search_options->create_item();

	root->set_text(0,base_type);
	if (has_icon(base_type,"EditorIcons")) {
		root->set_icon(0,get_icon(base_type,"EditorIcons"));
	}

	List<StringName>::Element *I=type_list.front();
	TreeItem *to_select=NULL;

	for(;I;I=I->next()) {



		String type=I->get();

		if (base_type=="Node" && type.begins_with("Editor"))
			continue; // do not show editor nodes

		if (!ClassDB::can_instance(type))
			continue; // cant create what can't be instanced

		if (search_box->get_text()=="") {
			add_type(type,types,root,&to_select);
		} else {

			bool found=false;
			String type=I->get();
			while(type!="" && ClassDB::is_parent_class(type,base_type) && type!=base_type) {
				if (search_box->get_text().is_subsequence_ofi(type)) {

					found=true;
					break;
				}

				type=ClassDB::get_parent_class(type);
			}


			if (found)
				add_type(I->get(),types,root,&to_select);
		}

		if (EditorNode::get_editor_data().get_custom_types().has(type) && ClassDB::is_parent_class(type, base_type)) {
			//there are custom types based on this... cool.
			//print_line("there are custom types");


			const Vector<EditorData::CustomType> &ct = EditorNode::get_editor_data().get_custom_types()[type];
			for(int i=0;i<ct.size();i++) {

				bool show = search_box->get_text().is_subsequence_ofi(ct[i].name);

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

				if (!to_select || ct[i].name==search_box->get_text()) {
					to_select=item;
				}

			}

		}
	}

	if (to_select) {
		to_select->select(0);
		favorite->set_disabled(false);
		favorite->set_pressed(favorite_list.find(to_select->get_text(0))!=-1);
	}

	get_ok()->set_disabled(root->get_children()==NULL);

}

void CreateDialog::_confirmed() {

	TreeItem *ti = search_options->get_selected();
	if (!ti)
		return;

	FileAccess *f = FileAccess::open( EditorSettings::get_singleton()->get_project_settings_path().plus_file("create_recent."+base_type), FileAccess::WRITE );

	if (f) {
		f->store_line(get_selected_type());
		TreeItem *t = recent->get_root();
		if (t)
			t=t->get_children();
		int count=0;
		while(t) {
			if (t->get_text(0)!=get_selected_type()) {

				f->store_line(t->get_text(0));
			}

			if (count>32) {
				//limit it to 32 entries..
				break;
			}
			t=t->get_next();
			count++;
		}

		memdelete(f);
	}

	emit_signal("create");
	hide();
}

void CreateDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		connect("confirmed",this,"_confirmed");
		favorite->set_icon(get_icon("Favorites","EditorIcons"));
	}
	if (p_what==NOTIFICATION_EXIT_TREE) {

		disconnect("confirmed",this,"_confirmed");

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
	set_title(TTR("Create New")+" "+p_base);
	_update_search();
}

String CreateDialog::get_selected_type() {

	TreeItem *selected = search_options->get_selected();
	if (selected)
		return selected->get_text(0);
	else
		return String();
}

Object *CreateDialog::instance_selected() {

	TreeItem *selected = search_options->get_selected();

	if (selected) {

		String custom = selected->get_metadata(0);


		if (custom!=String()) {
			if (EditorNode::get_editor_data().get_custom_types().has(custom)) {

				for(int i=0;i<EditorNode::get_editor_data().get_custom_types()[custom].size();i++) {
					if (EditorNode::get_editor_data().get_custom_types()[custom][i].name==selected->get_text(0)) {
						Ref<Texture> icon = EditorNode::get_editor_data().get_custom_types()[custom][i].icon;
						Ref<Script> script = EditorNode::get_editor_data().get_custom_types()[custom][i].script;
						String name = selected->get_text(0);

						Object *ob = ClassDB::instance(custom);
						ERR_FAIL_COND_V(!ob,NULL);
						if (ob->is_class("Node")) {
							ob->call("set_name",name);
						}
						ob->set_script(script.get_ref_ptr());
						if (icon.is_valid())
							ob->set_meta("_editor_icon",icon);
						return ob;

					}
				}

			}
		} else {
			return ClassDB::instance(selected->get_text(0));
		}
	}


	return NULL;
}


String CreateDialog::get_base_type() const {

	return base_type;
}

void CreateDialog::_item_selected() {

	TreeItem *item = search_options->get_selected();
	if (!item)
		return;

	String name = item->get_text(0);

	favorite->set_disabled(false);
	favorite->set_pressed(favorite_list.find(name)!=-1);

	if (!EditorHelp::get_doc_data()->class_list.has(name))
		return;

	help_bit->set_text(EditorHelp::get_doc_data()->class_list[name].brief_description);

}


void CreateDialog::_favorite_toggled() {

	TreeItem *item = search_options->get_selected();
	if (!item)
		return;

	String name = item->get_text(0);

	if (favorite_list.find(name)==-1) {
		favorite_list.push_back(name);
		favorite->set_pressed(true);
	} else {
		favorite_list.erase(name);
		favorite->set_pressed(false);
	}

	_save_favorite_list();
	_update_favorite_list();
}

void CreateDialog::_save_favorite_list() {

	FileAccess *f = FileAccess::open( EditorSettings::get_singleton()->get_project_settings_path().plus_file("favorites."+base_type), FileAccess::WRITE );

	if (f) {

		for(int i=0;i<favorite_list.size();i++) {

			f->store_line(favorite_list[i]);
		}
		memdelete(f);
	}
}

void CreateDialog::_update_favorite_list() {

	favorites->clear();
	TreeItem *root = favorites->create_item();
	for(int i=0;i<favorite_list.size();i++) {
		TreeItem *ti = favorites->create_item(root);
		String l = favorite_list[i];
		ti->set_text(0,l);

		if (has_icon(l,"EditorIcons")) {

			ti->set_icon(0,get_icon(l,"EditorIcons"));
		} else {
			ti->set_icon(0,get_icon("Object","EditorIcons"));
		}
	}
}


void CreateDialog::_history_selected() {

	TreeItem *item = recent->get_selected();
	if (!item)
		return;

	search_box->set_text(item->get_text(0));
	_update_search();

}

void CreateDialog::_favorite_selected(){

	TreeItem *item = favorites->get_selected();
	if (!item)
		return;

	search_box->set_text(item->get_text(0));
	_update_search();

}

void CreateDialog::_history_activated() {

	_confirmed();
}

void CreateDialog::_favorite_activated(){

	_confirmed();
}

Variant CreateDialog::get_drag_data_fw(const Point2& p_point,Control* p_from) {

	TreeItem *ti = favorites->get_item_at_pos(p_point);
	if (ti) {
		Dictionary d;
		d["type"]="create_favorite_drag";
		d["class"]=ti->get_text(0);

		ToolButton *tb = memnew( ToolButton );
		tb->set_icon(ti->get_icon(0));
		tb->set_text(ti->get_text(0));
		set_drag_preview(tb);

		return d;
	}

	return Variant();
}

bool CreateDialog::can_drop_data_fw(const Point2& p_point,const Variant& p_data,Control* p_from) const{

	Dictionary d = p_data;
	if (d.has("type") && String(d["type"])=="create_favorite_drag") {
		favorites->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);
		return true;
	}

	return false;
}
void CreateDialog::drop_data_fw(const Point2& p_point,const Variant& p_data,Control* p_from){

	Dictionary d = p_data;

	TreeItem *ti = favorites->get_item_at_pos(p_point);
	if (!ti)
		return;

	String drop_at = ti->get_text(0);
	int ds = favorites->get_drop_section_at_pos(p_point);

	int drop_idx = favorite_list.find(drop_at);
	if (drop_idx<0)
		return;

	String type = d["class"];

	int from_idx = favorite_list.find(type);
	if (from_idx<0)
		return;

	if (drop_idx==from_idx) {
		ds=-1; //cause it will be gone
	} else if (drop_idx>from_idx) {
		drop_idx--;
	}

	favorite_list.remove(from_idx);

	if (ds<0) {
		favorite_list.insert(drop_idx,type);
	} else {
		if (drop_idx>=favorite_list.size()-1) {
			favorite_list.push_back(type);
		} else {
			favorite_list.insert(drop_idx+1,type);
		}
	}

	_save_favorite_list();
	_update_favorite_list();


}

void CreateDialog::_bind_methods() {

	ClassDB::bind_method(_MD("_text_changed"),&CreateDialog::_text_changed);
	ClassDB::bind_method(_MD("_confirmed"),&CreateDialog::_confirmed);
	ClassDB::bind_method(_MD("_sbox_input"),&CreateDialog::_sbox_input);
	ClassDB::bind_method(_MD("_item_selected"),&CreateDialog::_item_selected);
	ClassDB::bind_method(_MD("_favorite_toggled"),&CreateDialog::_favorite_toggled);
	ClassDB::bind_method(_MD("_history_selected"),&CreateDialog::_history_selected);
	ClassDB::bind_method(_MD("_favorite_selected"),&CreateDialog::_favorite_selected);
	ClassDB::bind_method(_MD("_history_activated"),&CreateDialog::_history_activated);
	ClassDB::bind_method(_MD("_favorite_activated"),&CreateDialog::_favorite_activated);


	ClassDB::bind_method("get_drag_data_fw",&CreateDialog::get_drag_data_fw);
	ClassDB::bind_method("can_drop_data_fw",&CreateDialog::can_drop_data_fw);
	ClassDB::bind_method("drop_data_fw",&CreateDialog::drop_data_fw);

	ADD_SIGNAL(MethodInfo("create"));

}


CreateDialog::CreateDialog() {

	HSplitContainer *hbc = memnew( HSplitContainer );

	add_child(hbc);
	set_child_rect(hbc);

	VBoxContainer *lvbc = memnew( VBoxContainer);
	hbc->add_child(lvbc);
	lvbc->set_custom_minimum_size(Size2(150,0)*EDSCALE);

	favorites = memnew (Tree );
	lvbc->add_margin_child(TTR("Favorites:"),favorites,true);
	favorites->set_hide_root(true);
	favorites->set_hide_folding(true);
	favorites->connect("cell_selected",this,"_favorite_selected");
	favorites->connect("item_activated",this,"_favorite_activated");
	favorites->set_drag_forwarding(this);


	recent = memnew (Tree );
	lvbc->add_margin_child(TTR("Recent:"),recent,true);
	recent->set_hide_root(true);
	recent->set_hide_folding(true);
	recent->connect("cell_selected",this,"_history_selected");
	recent->connect("item_activated",this,"_history_activated");


	VBoxContainer *vbc = memnew( VBoxContainer );
	hbc->add_child(vbc);
	vbc->set_h_size_flags(SIZE_EXPAND_FILL);
	HBoxContainer *search_hb = memnew( HBoxContainer );
	search_box = memnew( LineEdit );
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	search_hb->add_child(search_box);
	favorite = memnew( Button );
	favorite->set_toggle_mode(true);
	search_hb->add_child(favorite);
	favorite->connect("pressed",this,"_favorite_toggled");
	vbc->add_margin_child(TTR("Search:"),search_hb);
	search_box->connect("text_changed",this,"_text_changed");
	search_box->connect("input_event",this,"_sbox_input");
	search_options = memnew( Tree );
	vbc->add_margin_child(TTR("Matches:"),search_options,true);
	get_ok()->set_text(TTR("Create"));
	get_ok()->set_disabled(true);
	register_text_enter(search_box);
	set_hide_on_ok(false);
	search_options->connect("item_activated",this,"_confirmed");
	search_options->connect("cell_selected",this,"_item_selected");
//	search_options->set_hide_root(true);
	base_type="Object";

	help_bit = memnew( EditorHelpBit );
	vbc->add_margin_child(TTR("Description:"),help_bit);
	help_bit->connect("request_hide",this,"_closed");

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
	if (!ClassDB::is_type(p_type,base) || p_type==base)
		return;

	String inherits=ClassDB::type_inherits_from(p_type);

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
	if (!ClassDB::can_instance(p_type)) {
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
	ClassDB::get_type_list(&type_list);

	HashMap<String,TreeItem*> types;

	TreeItem *root = tree->create_item();

	root->set_text(0,base);

	List<String>::Element *I=type_list.front();

	for(;I;I=I->next()) {


		String type=I->get();


		if (!ClassDB::can_instance(type))
			continue; // cant create what can't be instanced
		if (filter->get_text()=="")
			add_type(type,types,root);
		else {

			bool found=false;
			String type=I->get();
			while(type!="" && ClassDB::is_type(type,base) && type!=base) {
				if (type.findn(filter->get_text())!=-1) {

					found=true;
					break;
				}

				type=ClassDB::type_inherits_from(type);
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

					Object* obj = ClassDB::instance(base);
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

	return ClassDB::instance(tree->get_selected()->get_text(0));

}


void CreateDialog::_bind_methods() {

	ClassDB::bind_method("_create",&CreateDialog::_create);
	ClassDB::bind_method("_cancel",&CreateDialog::_cancel);
	ClassDB::bind_method("_text_changed", &CreateDialog::_text_changed);
	ADD_SIGNAL( MethodInfo("create"));

}




void CreateDialog::set_base_type(const String& p_base) {

	set_title(vformat("Create %s Type",p_base));

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
