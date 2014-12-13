/*************************************************************************/
/*  sprite_frames_editor_plugin.cpp                                      */
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
#include "sprite_frames_editor_plugin.h"

#include "io/resource_loader.h"
#include "globals.h"
#include "tools/editor/editor_settings.h"




void SpriteFramesEditor::_input_event(InputEvent p_event) {


}

void SpriteFramesEditor::_notification(int p_what) {

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

void SpriteFramesEditor::_file_load_request(const DVector<String>& p_path) {


	List< Ref<Texture> > resources;

	for(int i=0;i<p_path.size();i++) {

		Ref<Texture>  resource;
		resource = ResourceLoader::load(p_path[i]);

		if (resource.is_null()) {
			dialog->set_text("ERROR: Couldn't load frame resource!");
			dialog->set_title("Error!");
			//dialog->get_cancel()->set_text("Close");
			dialog->get_ok()->set_text("Close");
			dialog->popup_centered(Size2(300,60));
			return; ///beh should show an error i guess
		}

		resources.push_back(resource);
	}


	if (resources.empty()) {
		print_line("added frames!");
		return;
	}

	undo_redo->create_action("Add Frame");
	int fc=frames->get_frame_count();

	for(List< Ref<Texture> >::Element *E=resources.front();E;E=E->next() ) {

		undo_redo->add_do_method(frames,"add_frame",E->get());
		undo_redo->add_undo_method(frames,"remove_frame",fc++);

	}
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");

	undo_redo->commit_action();
	print_line("added frames!");
}

void SpriteFramesEditor::_load_pressed() {

	loading_scene=false;

	file->clear_filters();
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Texture",&extensions);
	for(int i=0;i<extensions.size();i++)
		file->add_filter("*."+extensions[i]);

	file->set_mode(FileDialog::MODE_OPEN_FILES);

	file->popup_centered_ratio();

}


void SpriteFramesEditor::_item_edited() {

#if 0
	if (!tree->get_selected())
		return;

	TreeItem *s = tree->get_selected();

	if (tree->get_selected_column()==0) {
		// renamed
		String old_name=s->get_metadata(0);
		String new_name=s->get_text(0);
		if (old_name==new_name)
			return;

		if (new_name=="" || new_name.find("\\")!=-1 || new_name.find("/")!=-1 || frames->has_resource(new_name)) {

			s->set_text(0,old_name);
			return;
		}

		RES samp = frames->get_resource(old_name);
		undo_redo->create_action("Rename Resource");
		undo_redo->add_do_method(frames,"remove_resource",old_name);
		undo_redo->add_do_method(frames,"add_resource",new_name,samp);
		undo_redo->add_undo_method(frames,"remove_resource",new_name);
		undo_redo->add_undo_method(frames,"add_resource",old_name,samp);
		undo_redo->add_do_method(this,"_update_library");
		undo_redo->add_undo_method(this,"_update_library");
		undo_redo->commit_action();

	}
#endif

}

void SpriteFramesEditor::_delete_confirm_pressed() {

	if (!tree->get_selected())
		return;

	sel-=1;
	if (sel<0 && frames->get_frame_count())
		sel=0;

	int to_remove = tree->get_selected()->get_metadata(0);
	sel=to_remove;
	Ref<Texture> r = frames->get_frame(to_remove);
	undo_redo->create_action("Delete Resource");
	undo_redo->add_do_method(frames,"remove_frame",to_remove);
	undo_redo->add_undo_method(frames,"add_frame",r,to_remove);
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();

}


void SpriteFramesEditor::_paste_pressed() {

	Ref<Texture> r=EditorSettings::get_singleton()->get_resource_clipboard();
	if (!r.is_valid()) {
		dialog->set_text("Resource clipboard is empty or not a texture!");
		dialog->set_title("Error!");
		//dialog->get_cancel()->set_text("Close");
		dialog->get_ok()->set_text("Close");
		dialog->popup_centered(Size2(300,60));
		return; ///beh should show an error i guess
	}


	undo_redo->create_action("Paste Frame");
	undo_redo->add_do_method(frames,"add_frame",r);
	undo_redo->add_undo_method(frames,"remove_frame",frames->get_frame_count());
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();

}

void SpriteFramesEditor::_empty_pressed() {


	int from=-1;

	if (tree->get_selected()) {

		from = tree->get_selected()->get_metadata(0);
		sel=from;

	} else {
		from=frames->get_frame_count();
	}



	Ref<Texture> r;

	undo_redo->create_action("Add Empty");
	undo_redo->add_do_method(frames,"add_frame",r,from);
	undo_redo->add_undo_method(frames,"remove_frame",from);
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();

}

void SpriteFramesEditor::_up_pressed() {

	if (!tree->get_selected())
		return;
	int to_move = tree->get_selected()->get_metadata(0);
	if (to_move<1)
		return;

	sel=to_move;
	sel-=1;

	Ref<Texture> r = frames->get_frame(to_move);
	undo_redo->create_action("Delete Resource");
	undo_redo->add_do_method(frames,"set_frame",to_move,frames->get_frame(to_move-1));
	undo_redo->add_do_method(frames,"set_frame",to_move-1,frames->get_frame(to_move));
	undo_redo->add_undo_method(frames,"set_frame",to_move,frames->get_frame(to_move));
	undo_redo->add_undo_method(frames,"set_frame",to_move-1,frames->get_frame(to_move-1));
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();

}

void SpriteFramesEditor::_down_pressed() {

	if (!tree->get_selected())
		return;
	int to_move = tree->get_selected()->get_metadata(0);
	if (to_move<0 || to_move>=frames->get_frame_count()-1)
		return;

	sel=to_move;
	sel+=1;

	Ref<Texture> r = frames->get_frame(to_move);
	undo_redo->create_action("Delete Resource");
	undo_redo->add_do_method(frames,"set_frame",to_move,frames->get_frame(to_move+1));
	undo_redo->add_do_method(frames,"set_frame",to_move+1,frames->get_frame(to_move));
	undo_redo->add_undo_method(frames,"set_frame",to_move,frames->get_frame(to_move));
	undo_redo->add_undo_method(frames,"set_frame",to_move+1,frames->get_frame(to_move+1));
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();



}


void SpriteFramesEditor::_delete_pressed() {


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


void SpriteFramesEditor::_update_library() {

	tree->clear();
	tree->set_hide_root(true);
	TreeItem *root = tree->create_item(NULL);

	if (sel>=frames->get_frame_count())
		sel=frames->get_frame_count()-1;
	else if (sel<0 && frames->get_frame_count())
		sel=0;

	for(int i=0;i<frames->get_frame_count();i++) {

		TreeItem *ti = tree->create_item(root);
		ti->set_cell_mode(0,TreeItem::CELL_MODE_STRING);
		ti->set_editable(0,true);
		ti->set_selectable(0,true);

		if (frames->get_frame(i).is_null()) {

			ti->set_text(0,"Frame "+itos(i)+" (empty)");

		} else {
			ti->set_text(0,"Frame "+itos(i));
			ti->set_icon(0,frames->get_frame(i));
		}
		ti->set_metadata(0,i);
		ti->set_icon_max_width(0,96);
		if (sel==i)
			ti->select(0);
	}

	//player->add_resource("default",resource);
}



void SpriteFramesEditor::edit(SpriteFrames* p_frames) {

	frames=p_frames;


	if (p_frames) {
		_update_library();
	} else {

		hide();
		//set_fixed_process(false);
	}

}



void SpriteFramesEditor::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_input_event"),&SpriteFramesEditor::_input_event);
	ObjectTypeDB::bind_method(_MD("_load_pressed"),&SpriteFramesEditor::_load_pressed);
	ObjectTypeDB::bind_method(_MD("_empty_pressed"),&SpriteFramesEditor::_empty_pressed);
	ObjectTypeDB::bind_method(_MD("_item_edited"),&SpriteFramesEditor::_item_edited);
	ObjectTypeDB::bind_method(_MD("_delete_pressed"),&SpriteFramesEditor::_delete_pressed);
	ObjectTypeDB::bind_method(_MD("_paste_pressed"),&SpriteFramesEditor::_paste_pressed);
	ObjectTypeDB::bind_method(_MD("_delete_confirm_pressed"),&SpriteFramesEditor::_delete_confirm_pressed);
	ObjectTypeDB::bind_method(_MD("_file_load_request"),&SpriteFramesEditor::_file_load_request);
	ObjectTypeDB::bind_method(_MD("_update_library"),&SpriteFramesEditor::_update_library);
	ObjectTypeDB::bind_method(_MD("_up_pressed"),&SpriteFramesEditor::_up_pressed);
	ObjectTypeDB::bind_method(_MD("_down_pressed"),&SpriteFramesEditor::_down_pressed);
}

SpriteFramesEditor::SpriteFramesEditor() {

	//add_style_override("panel", get_stylebox("panel","Panel"));

	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);

	HBoxContainer *hbc = memnew( HBoxContainer );
	vbc->add_child(hbc);

	load = memnew( Button );
	load->set_tooltip("Load Resource");
	hbc->add_child(load);




	paste = memnew( Button );
	paste->set_text("Paste");
	hbc->add_child(paste);

	empty = memnew( Button );
	empty->set_text("Insert Empty");
	hbc->add_child(empty);

	move_up = memnew( Button );
	move_up->set_text("Up");
	hbc->add_child(move_up);

	move_down = memnew( Button );
	move_down->set_text("Down");
	hbc->add_child(move_down);

	_delete = memnew( Button );
	hbc->add_child(_delete);

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
	empty->connect("pressed", this,"_empty_pressed");
	move_up->connect("pressed", this,"_up_pressed");
	move_down->connect("pressed", this,"_down_pressed");
	file->connect("files_selected", this,"_file_load_request");
	//dialog->connect("confirmed", this,"_delete_confirm_pressed");
	tree->connect("item_edited", this,"_item_edited");
	loading_scene=false;
	sel=-1;

}


void SpriteFramesEditorPlugin::edit(Object *p_object) {

	frames_editor->set_undo_redo(&get_undo_redo());
	SpriteFrames * s = p_object->cast_to<SpriteFrames>();
	if (!s)
		return;

	frames_editor->edit(s);
}

bool SpriteFramesEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("SpriteFrames");
}

void SpriteFramesEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		frames_editor->show();
//		frames_editor->set_process(true);
	} else {

		frames_editor->hide();
//		frames_editor->set_process(false);
	}

}

SpriteFramesEditorPlugin::SpriteFramesEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	frames_editor = memnew( SpriteFramesEditor );
	editor->get_viewport()->add_child(frames_editor);
	frames_editor->set_area_as_parent_rect();
//	frames_editor->set_anchor( MARGIN_TOP, Control::ANCHOR_END);
//	frames_editor->set_margin( MARGIN_TOP, 120 );
	frames_editor->hide();



}


SpriteFramesEditorPlugin::~SpriteFramesEditorPlugin()
{
}


