/*************************************************************************/
/*  resource_preloader_editor_plugin.cpp                                 */
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
#include "resource_preloader_editor_plugin.h"
#include "io/resource_loader.h"
#include "globals.h"
#include "tools/editor/editor_settings.h"
#include "scene/resources/scene_preloader.h"



void ResourcePreloaderEditor::_input_event(InputEvent p_event) {


}

void ResourcePreloaderEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_FIXED_PROCESS) {

	}

	if (p_what==NOTIFICATION_ENTER_TREE) {
		load->set_icon( get_icon("Folder","EditorIcons") );		
		_delete->set_icon( get_icon("Del","EditorIcons") );
	}

	if (p_what==NOTIFICATION_READY) {

//		NodePath("/root")->connect("node_removed", this,"_node_removed",Vector<Variant>(),true);
	}

	if (p_what==NOTIFICATION_DRAW) {

	}
}

void ResourcePreloaderEditor::_file_load_request(const String& p_path) {


	RES resource;

	resource = ResourceLoader::load(p_path);

	if (resource.is_null()) {
		dialog->set_text("ERROR: Couldn't load resource!");
		dialog->set_title("Error!");
		//dialog->get_cancel()->set_text("Close");
		dialog->get_ok()->set_text("Close");
		dialog->popup_centered(Size2(300,60));
		return; ///beh should show an error i guess
	}

	String basename = p_path.get_file().basename();
	String name=basename;
	int counter=1;
	while(preloader->has_resource(name)) {
		counter++;
		name=basename+" "+itos(counter);
	}

	undo_redo->create_action("Add Resource");
	undo_redo->add_do_method(preloader,"add_resource",name,resource);
	undo_redo->add_undo_method(preloader,"remove_resource",name);
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();
}

void ResourcePreloaderEditor::_load_pressed() {

	loading_scene=false;

	file->clear_filters();
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("",&extensions);
	for(int i=0;i<extensions.size();i++)
		file->add_filter("*."+extensions[i]);

	file->set_mode(FileDialog::MODE_OPEN_FILE);

	file->popup_centered_ratio();

}


void ResourcePreloaderEditor::_item_edited() {

	if (!tree->get_selected())
		return;

	TreeItem *s = tree->get_selected();

	if (tree->get_selected_column()==0) {
		// renamed
		String old_name=s->get_metadata(0);
		String new_name=s->get_text(0);
		if (old_name==new_name)
			return;

		if (new_name=="" || new_name.find("\\")!=-1 || new_name.find("/")!=-1 || preloader->has_resource(new_name)) {

			s->set_text(0,old_name);
			return;
		}

		RES samp = preloader->get_resource(old_name);
		undo_redo->create_action("Rename Resource");
		undo_redo->add_do_method(preloader,"remove_resource",old_name);
		undo_redo->add_do_method(preloader,"add_resource",new_name,samp);
		undo_redo->add_undo_method(preloader,"remove_resource",new_name);
		undo_redo->add_undo_method(preloader,"add_resource",old_name,samp);
		undo_redo->add_do_method(this,"_update_library");
		undo_redo->add_undo_method(this,"_update_library");
		undo_redo->commit_action();

	}


}

void ResourcePreloaderEditor::_delete_confirm_pressed() {

	if (!tree->get_selected())
		return;

	String to_remove = tree->get_selected()->get_text(0);
	undo_redo->create_action("Delete Resource");
	undo_redo->add_do_method(preloader,"remove_resource",to_remove);
	undo_redo->add_undo_method(preloader,"add_resource",to_remove,preloader->get_resource(to_remove));
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();
}


void ResourcePreloaderEditor::_paste_pressed() {

	RES r=EditorSettings::get_singleton()->get_resource_clipboard();
	if (!r.is_valid()) {
		dialog->set_text("Resource clipboard is empty!");
		dialog->set_title("Error!");
		//dialog->get_cancel()->set_text("Close");
		dialog->get_ok()->set_text("Close");
		dialog->popup_centered(Size2(300,60));
		return; ///beh should show an error i guess
	}

	String name = r->get_name();
	if (name=="")
		name=r->get_path().get_file();
	if (name=="")
		name=r->get_type();

	String basename = name;
	int counter=1;
	while(preloader->has_resource(name)) {
		counter++;
		name=basename+" "+itos(counter);
	}

	undo_redo->create_action("Paste Resource");
	undo_redo->add_do_method(preloader,"add_resource",name,r);
	undo_redo->add_undo_method(preloader,"remove_resource",name);
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();

}


void ResourcePreloaderEditor::_delete_pressed() {


	if (!tree->get_selected())
		return;

	_delete_confirm_pressed(); //it has undo.. why bother with a dialog..
	/*
	dialog->set_title("Confirm...");
	dialog->set_text("Remove Resource '"+tree->get_selected()->get_text(0)+"' ?");
	//dialog->get_cancel()->set_text("Cancel");
	//dialog->get_ok()->show();
	dialog->get_ok()->set_text("Remove");
	dialog->popup_centered(Size2(300,60));*/

}


void ResourcePreloaderEditor::_update_library() {

	tree->clear();
	tree->set_hide_root(true);
	TreeItem *root = tree->create_item(NULL);

	List<StringName> rnames;
	preloader->get_resource_list(&rnames);

	List<String> names;
	for(List<StringName>::Element *E=rnames.front();E;E=E->next()) {
		names.push_back(E->get());
	}

	names.sort();

	for(List<String>::Element *E=names.front();E;E=E->next()) {

		TreeItem *ti = tree->create_item(root);
		ti->set_cell_mode(0,TreeItem::CELL_MODE_STRING);
		ti->set_editable(0,true);
		ti->set_selectable(0,true);
		ti->set_text(0,E->get());
		ti->set_metadata(0,E->get());



		RES r = preloader->get_resource(E->get());

		ERR_CONTINUE(r.is_null());

		ti->set_tooltip(0,r->get_path());
		String type = r->get_type();
		ti->set_text(1,type);
		ti->set_selectable(1,false);

		if (has_icon(type,"EditorIcons"))
			ti->set_icon( 1, get_icon(type,"EditorIcons") );

	}

	//player->add_resource("default",resource);
}



void ResourcePreloaderEditor::edit(ResourcePreloader* p_preloader) {

	preloader=p_preloader;


	if (p_preloader) {
		_update_library();
	} else {

		hide();
		set_fixed_process(false);
	}

}



void ResourcePreloaderEditor::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_input_event"),&ResourcePreloaderEditor::_input_event);
	ObjectTypeDB::bind_method(_MD("_load_pressed"),&ResourcePreloaderEditor::_load_pressed);
	ObjectTypeDB::bind_method(_MD("_item_edited"),&ResourcePreloaderEditor::_item_edited);
	ObjectTypeDB::bind_method(_MD("_delete_pressed"),&ResourcePreloaderEditor::_delete_pressed);
	ObjectTypeDB::bind_method(_MD("_paste_pressed"),&ResourcePreloaderEditor::_paste_pressed);
	ObjectTypeDB::bind_method(_MD("_delete_confirm_pressed"),&ResourcePreloaderEditor::_delete_confirm_pressed);
	ObjectTypeDB::bind_method(_MD("_file_load_request"),&ResourcePreloaderEditor::_file_load_request);
	ObjectTypeDB::bind_method(_MD("_update_library"),&ResourcePreloaderEditor::_update_library);
}

ResourcePreloaderEditor::ResourcePreloaderEditor() {

	//add_style_override("panel", get_stylebox("panel","Panel"));

	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);

	HBoxContainer *hbc = memnew( HBoxContainer );
	vbc->add_child(hbc);

	load = memnew( Button );
	load->set_tooltip("Load Resource");
	hbc->add_child(load);



	_delete = memnew( Button );
	hbc->add_child(_delete);

	paste = memnew( Button );
	paste->set_text("Paste");
	hbc->add_child(paste);

	file = memnew( FileDialog );
	add_child(file);


	tree = memnew( Tree );
	tree->set_columns(2);
	tree->set_column_min_width(0,3);
	tree->set_column_min_width(1,1);
	tree->set_column_expand(0,true);
	tree->set_column_expand(1,true);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);

	vbc->add_child(tree);

	dialog = memnew( AcceptDialog );
	add_child( dialog );

	load->connect("pressed", this,"_load_pressed");
	_delete->connect("pressed", this,"_delete_pressed");
	paste->connect("pressed", this,"_paste_pressed");
	file->connect("file_selected", this,"_file_load_request");
	//dialog->connect("confirmed", this,"_delete_confirm_pressed");
	tree->connect("item_edited", this,"_item_edited");
	loading_scene=false;

}


void ResourcePreloaderEditorPlugin::edit(Object *p_object) {

	preloader_editor->set_undo_redo(&get_undo_redo());
	ResourcePreloader * s = p_object->cast_to<ResourcePreloader>();
	if (!s)
		return;

	preloader_editor->edit(s);
}

bool ResourcePreloaderEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("ResourcePreloader");
}

void ResourcePreloaderEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		preloader_editor->show();
//		preloader_editor->set_process(true);
	} else {

		preloader_editor->hide();
//		preloader_editor->set_process(false);
	}

}

ResourcePreloaderEditorPlugin::ResourcePreloaderEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	preloader_editor = memnew( ResourcePreloaderEditor );
	editor->get_viewport()->add_child(preloader_editor);
	preloader_editor->set_area_as_parent_rect();
//	preloader_editor->set_anchor( MARGIN_TOP, Control::ANCHOR_END);
//	preloader_editor->set_margin( MARGIN_TOP, 120 );
	preloader_editor->hide();



}


ResourcePreloaderEditorPlugin::~ResourcePreloaderEditorPlugin()
{
}


